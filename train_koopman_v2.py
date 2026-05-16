"""train_koopman_v2.py — Deep-Koopman v2 训练脚本（强化速度跟踪）

Section 1–8 of PROMPT_deep_koopman_rewrite.md 的实现。

主要改动相对于 v1 (`train_multistep_voyage.py`)：
1. **直接对物理速度做 Huber 损失**（L_vel）作为主损失，γ-加权 + per-channel
   缩放，使 (u, v, r) 三个通道贡献相当；解决「形似但漂移」的核心症结。
2. **Multi-step rollout 物理量监督**：rollout 过程在归一化空间走，
   reconstruct + 反归一化后与 GT 对齐，长程误差直接产生梯度。
3. **L_lin 仍保留**但权重大幅降低 + 前 5 epoch 线性 ramp-up，避免压垮 encoder。
4. **Encoder dropout 默认 0**（避免 encode(target) ≠ rollout 的分布污染）。
5. **Curriculum 学习**：pred_len 从 4 起，每 N 个 epoch +2 直到 max。
6. **归一化统计一次性算完**；Dataset 用 numpy 拼接 + 整数索引切片，
   `__getitem__` 不重组。
7. **EMA + cosine LR + AMP**（CPU 模式自动关闭 AMP）。
8. **Best ckpt 由 vel_rmse_mean 决定**（不再用 acc loss）；同时把
   `eval_koopman.compute_*` 的发散指标写进 TensorBoard。
9. 训练结束自动调用 `eval_koopman.evaluate_one` 在 test 集上落盘 + 打印
   QUANTITATIVE VERDICT 块。

CLI 见 ``--help``；冒烟自测 ``python3 train_koopman_v2.py --smoketest``。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml

from koopman import HorizontalKoopmanModel
from koopman_v3 import HorizontalKoopmanModelV3, FEATURE_DICT_ATOMS
import eval_koopman as ek


# =============================================================================
# 0a. 模型工厂
# =============================================================================

def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """根据 ``args.model`` 构造 v1 / v2 / v3 模型。

    v1 与 v2 共用同一个 ``HorizontalKoopmanModel`` 类（5 阶物理字典），
    v3 用 ``HorizontalKoopmanModelV3``（5+11 阶字典）。
    """
    model_tag = getattr(args, "model", "v3")
    if model_tag in ("v1", "v2"):
        model = HorizontalKoopmanModel(state_dim=3, control_dim=4, hidden_dim=24)
    elif model_tag == "v3":
        model = HorizontalKoopmanModelV3(
            state_dim=3, control_dim=4,
            hidden_dim=int(getattr(args, "hidden_dim", 24)),
            n_cubic=int(getattr(args, "n_cubic", 11)),
            clamp_pif=float(getattr(args, "clamp_pif", 5.0)),
        )
    else:
        raise ValueError(f"未知 --model {model_tag!r}; 支持 {{v1, v2, v3}}")
    return model.to(device)


# =============================================================================
# 0. 工具
# =============================================================================

def setup_logger(log_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"KoopmanTrainerV2_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, f"train_v2_{timestamp}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger, timestamp


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def get_gpu_power_bar(bar_length: int = 10) -> str:
    if not torch.cuda.is_available():
        return "[N/A]"
    try:
        cmd = "nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader,nounits"
        draw, limit = map(float, subprocess.check_output(cmd, shell=True).decode().strip().split("\n")[0].split(","))
        percent = (draw / limit) * 100 if limit > 0 else 0
        return f"[{'|' * int((percent / 100) * bar_length):<10}] {percent:4.1f}%"
    except Exception:
        return "[Pwr Error]"


# =============================================================================
# 1. Dataset：高吞吐版
# =============================================================================

class KoopmanVoyageDataset(Dataset):
    """一次性把所有段拼成大数组，``__getitem__`` 仅做整数切片。

    返回:
        x_t_norm        : (6,)
        x_target_seq_n  : (pred_len, 6)
        u_seq_norm      : (pred_len, 4)
    """

    def __init__(
        self,
        npz_path: str,
        pred_len: int,
        stride: int = 1,
        stats: Optional[Dict[str, np.ndarray]] = None,
        logger: Optional[logging.Logger] = None,
        tag: str = "DATA",
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        (
            self.states_full,
            self.ctrls_full,
            self.seg_starts,
            self.seg_lens,
            self.t0_global,
            self.seg_idx,
            self.t0_local,
        ) = ek._flatten_segments(npz_path, pred_len=self.pred_len, stride=self.stride)
        self.stats = stats if stats is not None else self._compute_stats()
        # 预先转好 norm 参数
        self._sm = self.stats["state_mean"].astype(np.float32)
        self._ss = self.stats["state_std"].astype(np.float32)
        self._cm = self.stats["ctrl_mean"].astype(np.float32)
        self._cs = self.stats["ctrl_std"].astype(np.float32)
        if logger is not None:
            logger.info(
                f"Dataset[{tag}] {npz_path} | segments={len(self.seg_lens)} "
                f"| samples={len(self.t0_global)} | pred_len={self.pred_len} | stride={self.stride}"
            )

    def _compute_stats(self) -> Dict[str, np.ndarray]:
        return {
            "state_mean": self.states_full.mean(axis=0).astype(np.float32),
            "state_std": (self.states_full.std(axis=0) + 1e-6).astype(np.float32),
            "ctrl_mean": self.ctrls_full.mean(axis=0).astype(np.float32),
            "ctrl_std": (self.ctrls_full.std(axis=0) + 1e-6).astype(np.float32),
        }

    def __len__(self) -> int:
        return int(self.t0_global.shape[0])

    def __getitem__(self, index: int):
        t0 = int(self.t0_global[index])
        K = self.pred_len
        x_t = self.states_full[t0]                           # (6,)
        x_seq = self.states_full[t0 + 1 : t0 + 1 + K]        # (K, 6)
        u_seq = self.ctrls_full[t0 : t0 + K]                 # (K, 4)
        x_t_n = (x_t - self._sm) / self._ss
        x_seq_n = (x_seq - self._sm) / self._ss
        u_seq_n = (u_seq - self._cm) / self._cs
        return (
            torch.from_numpy(x_t_n.astype(np.float32, copy=False)),
            torch.from_numpy(x_seq_n.astype(np.float32, copy=False)),
            torch.from_numpy(u_seq_n.astype(np.float32, copy=False)),
        )


# =============================================================================
# 2. EMA
# =============================================================================

class ModelEMA:
    """简化的指数滑动平均权重副本。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        # buffers 直接拷贝
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> Dict:
        return self.module.state_dict()

    def load_state_dict(self, sd: Dict) -> None:
        self.module.load_state_dict(sd)


# =============================================================================
# 3. YAML 导出
# =============================================================================

def export_params_to_yaml(model: nn.Module, stats: Dict[str, np.ndarray], save_path: str) -> None:
    A_w = (model.A.weight.detach().cpu() + torch.eye(model.A.weight.shape[0])).numpy().tolist()
    A_b = model.A.bias.detach().cpu().numpy().tolist() if model.A.bias is not None else []
    B_w = model.B.weight.detach().cpu().numpy().tolist()
    yaml_data: Dict = {
        "normalization": {
            "dyn_mean": np.asarray(stats["state_mean"][3:6], dtype=float).tolist(),
            "dyn_std": np.asarray(stats["state_std"][3:6], dtype=float).tolist(),
            "ctrl_mean": np.asarray(stats["ctrl_mean"], dtype=float).tolist(),
            "ctrl_std": np.asarray(stats["ctrl_std"], dtype=float).tolist(),
        },
        "system_matrices": {"A_weight": A_w, "A_bias": A_b, "B": B_w},
    }
    if isinstance(model, HorizontalKoopmanModelV3):
        n_cubic = int(getattr(model, "n_cubic", 11))
        all_quad = ["u_abs_u", "v_abs_v", "r_abs_r", "v_times_r", "u_times_r"]
        all_cubic = [
            "uvr", "u2r", "v2r", "ur2", "vr2",
            "u_vabs_v", "v_uabs_u", "r_uabs_u", "r_vabs_v",
            "uuu", "vvv",
        ]
        yaml_data["dictionary"] = {
            "state_atoms": ["u", "v", "r"],
            "quadratic_atoms": all_quad,
            "cubic_atoms": all_cubic[:n_cubic],
            "hidden_dim": int(getattr(model, "hidden_dim", 24)),
            "latent_dim": int(getattr(model, "latent_dim")),
            "note": (
                "encoder = concat(state(3), quadratic(5), cubic(%d), hidden_mlp(%d));"
                " all on NORMALIZED inputs"
            ) % (n_cubic, int(getattr(model, "hidden_dim", 24))),
        }
        yaml_data["info"] = (
            "Latent z = [u, v, r, "
            + ", ".join(all_quad + all_cubic[:n_cubic])
            + ", h_1..h_%d]" % int(getattr(model, "hidden_dim", 24))
        )
    else:
        yaml_data["info"] = "Latent z = [u, v, r, u|u|, v|v|, r|r|, vr, ur, h_1..h_24]"
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, indent=4)


# =============================================================================
# 4. 损失 & rollout
# =============================================================================

def huber(x: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """逐元素 Huber，返回相同 shape，便于后续加权。"""
    abs_x = x.abs()
    quad = 0.5 * (x ** 2) / beta
    lin = abs_x - 0.5 * beta
    return torch.where(abs_x < beta, quad, lin)


def rollout_train(
    model: nn.Module,
    x_t_dyn_n: torch.Tensor,           # (B, 3) normalized dyn at t0
    u_seq_n: torch.Tensor,             # (B, K, 4)
    noise_std: float = 0.0,
    ctrl_noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """归一化空间下的 rollout，可选择注入噪声以学到「鲁棒」的算子。

    注入策略：
    * x_t_dyn_n 加入 N(0, noise_std) 高斯扰动（输入鲁棒性）。
    * 控制信号每步加入 N(0, ctrl_noise_std)（鲁棒于控制噪声）。

    通过让模型在训练时见到「带扰动的输入」，强制鲁棒性 → 推断时
    每步注入的噪声不会同向累加 → 误差曲线更趋向 sqrt(k) 而非线性。
    """
    K = u_seq_n.size(1)
    if noise_std > 0:
        x_t_dyn_n = x_t_dyn_n + torch.randn_like(x_t_dyn_n) * noise_std
    z = model.encode(x_t_dyn_n)
    pred_norm: List[torch.Tensor] = []
    pred_lat: List[torch.Tensor] = []
    for k in range(K):
        u_in = u_seq_n[:, k, :]
        if ctrl_noise_std > 0:
            u_in = u_in + torch.randn_like(u_in) * ctrl_noise_std
        z = model.latent_step(z, u_in)
        pred_lat.append(z)
        pred_norm.append(model.reconstruct_state(z))
    return torch.stack(pred_norm, dim=1), torch.stack(pred_lat, dim=1)


def make_step_weights(K: int, gamma: float, device: torch.device) -> torch.Tensor:
    """w_k = γ^k；返回 shape (K,)。"""
    w = gamma ** torch.arange(K, device=device, dtype=torch.float32)
    # 归一化以保持总权重 ~ 1
    return w / w.mean()


def compute_losses(
    model: nn.Module,
    x_t_full_n: torch.Tensor,       # (B, 6)
    x_target_seq_n: torch.Tensor,   # (B, K, 6)
    u_seq_n: torch.Tensor,          # (B, K, 4)
    dyn_mean: torch.Tensor,         # (3,)
    dyn_std: torch.Tensor,          # (3,)
    args: argparse.Namespace,
    epoch: int,
    detach_target_lat: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, K, _ = x_target_seq_n.shape
    device = x_t_full_n.device

    dyn_t_n = x_t_full_n[:, 3:6]
    dyn_target_n = x_target_seq_n[:, :, 3:6]            # (B, K, 3)

    pred_norm_seq, pred_lat_seq = rollout_train(
        model, dyn_t_n, u_seq_n,
        noise_std=args.noise_std if model.training else 0.0,
        ctrl_noise_std=args.ctrl_noise_std if model.training else 0.0,
    )  # (B,K,3) (B,K,Dz)

    # 物理空间速度
    pred_phys = pred_norm_seq * dyn_std + dyn_mean      # (B,K,3)
    target_phys = dyn_target_n * dyn_std + dyn_mean

    step_w = make_step_weights(K, args.gamma_step, device).view(1, K, 1)  # (1,K,1)
    # per-channel 缩放：1/std_k 让三通道贡献接近
    chan_scale = 1.0 / dyn_std.view(1, 1, 3)             # (1,1,3)

    err_vel = pred_phys - target_phys                    # (B,K,3) 物理量
    huber_vel = huber(err_vel * chan_scale, beta=args.huber_beta)  # (B,K,3)
    L_vel = (huber_vel * step_w).mean()

    # 系统性偏置惩罚：每个 (step, channel) 上 batch 平均误差应接近 0。
    # 这是 v2_run01 失败的根因——u/v 上 bias 随 step 单调累积导致 slope_loglog≈1。
    # batch-mean(err) 的平方再按 step 加权（越后期越重，flatten 误差曲线）。
    bias_per_step = (err_vel * chan_scale).mean(dim=0)   # (K, 3)
    flatten_w = make_step_weights(K, args.gamma_bias, device).view(K, 1)
    # per-channel 加权（v3 引入）：默认 [u, v, r] 等权 1.0；CLI 通过
    # --w_bias_u/v/r 各自调整权重。
    chan_bias_w = torch.tensor(
        [getattr(args, "w_bias_u_eff", 1.0),
         getattr(args, "w_bias_v_eff", 1.0),
         getattr(args, "w_bias_r_eff", 1.0)],
        device=device, dtype=torch.float32,
    ).view(1, 3)
    L_bias = ((bias_per_step ** 2) * flatten_w * chan_bias_w).mean()

    # 加速度 Huber（带 dt）
    pred_acc = (pred_phys[:, 1:] - pred_phys[:, :-1]) / args.dt
    gt_acc = (target_phys[:, 1:] - target_phys[:, :-1]) / args.dt
    err_acc = pred_acc - gt_acc
    huber_acc = huber(err_acc * chan_scale, beta=args.huber_beta)
    step_w_acc = step_w[:, 1:, :]                        # (1,K-1,1)
    L_acc = (huber_acc * step_w_acc).mean() if K > 1 else torch.zeros((), device=device)

    # latent 一致性：把 GT 序列 encode 后比 latent
    target_flat = dyn_target_n.reshape(B * K, 3)
    target_lat = model.encode(target_flat).view(B, K, -1)
    if detach_target_lat:
        target_lat = target_lat.detach()
    L_lin = ((pred_lat_seq - target_lat) ** 2).mean()

    # 自编码恒等（理论上为 0；放着兼容 v2 模型替换）
    L_recon = ((model.reconstruct_state(model.encode(dyn_t_n)) - dyn_t_n) ** 2).mean()

    # 谱半径
    spec = model.spectral_radius()
    L_stab = torch.relu(spec - args.rho_max) ** 2

    # L2 on A, B
    A_w = model.A.weight
    B_w = model.B.weight
    L_l2 = (A_w * A_w).sum() + (B_w * B_w).sum()

    # 前 N epoch 对 L_lin / L_stab 做 ramp-up（线性 0→1）
    ramp = min(1.0, (epoch + 1) / max(args.ramp_epochs, 1))

    total = (
        args.w_vel * L_vel
        + args.w_acc * L_acc
        + args.w_lin * ramp * L_lin
        + args.w_recon * L_recon
        + args.w_stab * ramp * L_stab
        + args.w_l2 * L_l2
        + args.w_bias * L_bias
    )

    info = {
        "L_total": float(total.detach().item()),
        "L_vel": float(L_vel.detach().item()),
        "L_acc": float(L_acc.detach().item()) if isinstance(L_acc, torch.Tensor) else float(L_acc),
        "L_lin": float(L_lin.detach().item()),
        "L_stab": float(L_stab.detach().item()),
        "L_bias": float(L_bias.detach().item()),
        "L_l2": float(L_l2.detach().item()),
        "spec_radius": float(spec.detach().item()),
        "ramp": ramp,
    }
    return total, info


# =============================================================================
# 5. Validation：物理量 vel_rmse + 发散指标（用 eval_koopman 算）
# =============================================================================

@torch.no_grad()
def quick_validation(
    model: nn.Module,
    val_dataset: KoopmanVoyageDataset,
    pred_len: int,
    device: torch.device,
    dt: float,
    batch_size: int,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """完整 rollout 在 val 上的物理量评估。复用 eval_koopman 的核心函数。"""
    states_full = val_dataset.states_full
    ctrls_full = val_dataset.ctrls_full
    t0g = val_dataset.t0_global
    if max_samples is not None and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        model, states_full, ctrls_full, t0g, pred_len, val_dataset.stats, device, dt, batch_size,
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)
    K = pred_len
    diff = pred_dyn - gt_dyn
    vel = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
    return {
        "val/vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
        "val/vel_rmse_step1": float(per_step["vel_rmse"][0]),
        f"val/vel_rmse_step{K}": float(per_step["vel_rmse"][-1]),
        "val/acc_rmse_mean": float(np.nanmean(per_step["acc_rmse"])),
        f"val/traj_xy_rmse_step{K}": float(per_step["traj_xy_err"][-1]),
        "val/slope_loglog": float(div["slope_loglog"]),
        f"val/ratio_step{K}_over_step1": float(div[f"ratio_step{K}_over_step1"]),
        "val/instability_score": float(div["instability_score"]),
    }


# =============================================================================
# 6. CLI / Config 合并
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 数据
    p.add_argument("--train_data", type=str, default="koopman_train_merged.npz")
    p.add_argument("--val_data", type=str, default="koopman_val.npz")
    p.add_argument("--test_data", type=str, default="koopman_test.npz")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--out_dir", type=str, default="test_analysis/v2")
    p.add_argument("--run_tag", type=str, default="v2")
    # 训练超参
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", choices=["cosine", "onecycle", "step"], default="cosine")
    p.add_argument("--cos_T0", type=int, default=20)
    p.add_argument("--cos_Tmult", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true", default=True)
    p.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    # Curriculum
    p.add_argument("--pred_len_start", type=int, default=4)
    p.add_argument("--pred_len_max", type=int, default=20)
    p.add_argument("--pred_len_grow_every", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    # 损失权重
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_acc", type=float, default=0.2)
    p.add_argument("--w_lin", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.0)
    p.add_argument("--w_stab", type=float, default=0.1)
    p.add_argument("--w_l2", type=float, default=1e-4)
    p.add_argument("--w_bias", type=float, default=0.0,
                   help="batch-mean per-step bias 平方惩罚（强制误差零均值，flatten 曲线）。"
                        "若 --w_bias_u/v/r 任一显式给出则 --w_bias 被覆盖。")
    p.add_argument("--w_bias_u", type=float, default=None,
                   help="u 通道 bias 惩罚权重；显式给出则覆盖 --w_bias 标量行为。")
    p.add_argument("--w_bias_v", type=float, default=None,
                   help="v 通道 bias 惩罚权重；显式给出则覆盖 --w_bias 标量行为。")
    p.add_argument("--w_bias_r", type=float, default=None,
                   help="r 通道 bias 惩罚权重；显式给出则覆盖 --w_bias 标量行为。")
    p.add_argument("--model", type=str, choices=["v1", "v2", "v3"], default="v3",
                   help="选择模型：v1/v2 复用 HorizontalKoopmanModel，v3 用 V3。")
    p.add_argument("--n_cubic", type=int, default=11,
                   help="v3 的 cubic atom 数（最多 11，超出会报错）")
    p.add_argument("--clamp_pif", type=float, default=5.0,
                   help="v3 的物理字典 atom clamp 上限（绝对值），防止极端样本爆炸")
    p.add_argument("--hidden_dim", type=int, default=24,
                   help="v3 encoder 黑盒隐藏维度（与 v2 等同默认 24）")
    p.add_argument("--gamma_step", type=float, default=0.97, help="γ^k step weighting")
    p.add_argument("--gamma_bias", type=float, default=1.05,
                   help="L_bias 内的 γ^k step weighting（>1 强调晚期步）")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="输入 dyn 状态归一化噪声 σ（迫使 encoder 鲁棒，破坏误差线性累积）")
    p.add_argument("--ctrl_noise_std", type=float, default=0.0,
                   help="控制信号归一化噪声 σ（每步注入）")
    p.add_argument("--huber_beta", type=float, default=0.1)
    p.add_argument("--rho_max", type=float, default=1.005)
    p.add_argument("--ramp_epochs", type=int, default=5)
    p.add_argument("--detach_target_lat", action="store_true", default=False,
                   help="L_lin 用 encode(target).detach() 做消融对照")
    # EMA
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--no_ema", action="store_true", default=False)
    p.add_argument("--encoder_dropout", type=float, default=0.0)
    p.add_argument("--best_metric", choices=["vel_rmse_mean", "instability_score", "composite"],
                   default="composite",
                   help="select best ckpt by: vel_rmse_mean / instability_score / composite="
                        "vel_rmse_mean * max(1, instability_score)")
    # 工程
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--config", type=str, default=None,
                   help="YAML 配置；YAML > argparse 默认值；CLI 显式 > YAML")
    p.add_argument("--smoketest", action="store_true",
                   help="冒烟模式：2 段 / pred_len=4 / 2 epoch / batch=16，无 GPU 也能 1 分钟跑完")
    p.add_argument("--val_max_samples", type=int, default=2048,
                   help="每 epoch val 时使用的最大样本数（CPU 训练加速）")
    return p


def merge_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: List[str]) -> argparse.Namespace:
    """YAML 默认值 < argparse 默认值，但 CLI 显式参数始终最高优先。"""
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # CLI 显式给出的开关优先
    explicit = {a.dest for a in parser._actions if any(o in argv for o in (a.option_strings or []))}
    for k, v in cfg.items():
        if k in explicit:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    return args


# =============================================================================
# 7. 训练主循环
# =============================================================================

@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = float("inf")  # vel_rmse_mean


def make_dataloaders(
    args: argparse.Namespace,
    pred_len: int,
    train_ds_stats: Optional[Dict[str, np.ndarray]],
    logger: logging.Logger,
) -> Tuple[KoopmanVoyageDataset, KoopmanVoyageDataset, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    train_ds = KoopmanVoyageDataset(
        args.train_data, pred_len=pred_len, stride=args.stride,
        stats=train_ds_stats, logger=logger, tag="TRAIN",
    )
    stats = train_ds.stats
    val_ds = KoopmanVoyageDataset(
        args.val_data, pred_len=pred_len, stride=args.stride,
        stats=stats, logger=logger, tag="VAL",
    )
    nw = args.num_workers
    pw = args.persistent_workers and nw > 0
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=pw,
        prefetch_factor=args.prefetch_factor if nw > 0 else None,
        worker_init_fn=worker_init_fn if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(nw, 4),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=pw and min(nw, 4) > 0,
        prefetch_factor=args.prefetch_factor if min(nw, 4) > 0 else None,
    )
    return train_ds, val_ds, train_loader, val_loader, stats


def train(args: argparse.Namespace) -> None:
    logger, timestamp = setup_logger(args.log_dir)
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        logger.info("AMP 在 CPU 模式下自动关闭。")

    tb = SummaryWriter(log_dir=os.path.join(args.log_dir, f"tensorboard_v2_{timestamp}"))
    metrics_jsonl = os.path.join(args.log_dir, f"metrics_{timestamp}.jsonl")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- 解析 per-channel bias 权重（v3 引入；显式给出任一即覆盖标量 --w_bias） ----
    has_per_chan = any(
        getattr(args, f"w_bias_{c}", None) is not None for c in ("u", "v", "r")
    )
    if has_per_chan:
        wbu = float(args.w_bias_u) if args.w_bias_u is not None else float(args.w_bias)
        wbv = float(args.w_bias_v) if args.w_bias_v is not None else float(args.w_bias)
        wbr = float(args.w_bias_r) if args.w_bias_r is not None else float(args.w_bias)
        # 把全局 multiplier 设为 1，channel weight 内含全部权重
        args.w_bias = 1.0
        args.w_bias_u_eff, args.w_bias_v_eff, args.w_bias_r_eff = wbu, wbv, wbr
    else:
        # 兼容旧标量 --w_bias：channel weight 全 1（等价于历史 v2 行为）
        args.w_bias_u_eff = args.w_bias_v_eff = args.w_bias_r_eff = 1.0

    # ---- Curriculum 第一阶段 ----
    pred_len = max(args.pred_len_start, 1)
    pred_len = min(pred_len, args.pred_len_max)
    train_ds, val_ds, train_loader, val_loader, stats = make_dataloaders(args, pred_len, None, logger)

    # ---- Model ----
    model = build_model(args, device)
    # v3 字典自检（无论 smoketest 与否；smoketest 模式下必触发）
    if isinstance(model, HorizontalKoopmanModelV3):
        model._self_check_dict()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(args.encoder_dropout)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cos_T0, T_mult=args.cos_Tmult,
        )
    elif args.scheduler == "onecycle":
        steps_per_epoch = max(1, len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)

    state = TrainState()

    # ---- Resume ----
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        if ema is not None and ck.get("ema_state_dict"):
            ema.load_state_dict(ck["ema_state_dict"])
        state.epoch = int(ck.get("epoch", 0))
        state.best_metric = float(ck.get("best_metric", float("inf")))
        logger.info(f"Resumed from {args.resume} @ epoch={state.epoch} best={state.best_metric:.6g}")

    # ---- 启动横幅 ----
    spec_init = float(model.spectral_radius().detach().item())
    logger.info("=" * 80)
    logger.info(f"Device: {device} | AMP: {use_amp}")
    logger.info(
        f"Model: {args.model} | class={model.__class__.__name__} "
        f"| n_cubic={getattr(args, 'n_cubic', '-')} "
        f"| latent_dim={getattr(model, 'latent_dim', '-')} "
        f"| hidden_dim={getattr(args, 'hidden_dim', 24)} "
        f"| clamp_pif={getattr(args, 'clamp_pif', '-')}"
    )
    logger.info(f"Train data: {args.train_data} | Val: {args.val_data} | Test: {args.test_data}")
    logger.info(f"Batch: {args.batch_size} | Workers: {args.num_workers} | Seed: {args.seed}")
    logger.info(
        f"Curriculum pred_len: start={args.pred_len_start} max={args.pred_len_max} "
        f"grow_every={args.pred_len_grow_every} stride={args.stride}"
    )
    logger.info(
        f"Loss weights: w_vel={args.w_vel} w_acc={args.w_acc} w_lin={args.w_lin} "
        f"w_recon={args.w_recon} w_stab={args.w_stab} w_l2={args.w_l2} "
        f"w_bias={args.w_bias} (u={args.w_bias_u_eff}, v={args.w_bias_v_eff}, r={args.w_bias_r_eff}) "
        f"gamma_step={args.gamma_step} gamma_bias={args.gamma_bias} "
        f"huber_beta={args.huber_beta} rho_max={args.rho_max}"
    )
    logger.info(
        f"Noise: input_std={args.noise_std} ctrl_std={args.ctrl_noise_std} | "
        f"EMA decay={args.ema_decay} no_ema={args.no_ema} | best_metric={args.best_metric}"
    )
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | Params: {n_params}")
    logger.info(f"Initial spectral radius (I+A): {spec_init:.6g}")
    logger.info("=" * 80)

    dyn_mean_t = torch.tensor(stats["state_mean"][3:6], device=device, dtype=torch.float32)
    dyn_std_t = torch.tensor(stats["state_std"][3:6], device=device, dtype=torch.float32)

    metrics_summary_lines: List[str] = []

    for epoch in range(state.epoch, args.epochs):
        # Curriculum：是否需要扩窗 + 重建 dataloader
        target_pl = min(args.pred_len_max,
                        args.pred_len_start + 2 * (epoch // max(args.pred_len_grow_every, 1)))
        if target_pl != pred_len:
            pred_len = target_pl
            logger.info(f"[Curriculum] pred_len -> {pred_len}; rebuilding dataloaders...")
            train_ds, val_ds, train_loader, val_loader, _ = make_dataloaders(args, pred_len, stats, logger)

        model.train()
        epoch_acc: Dict[str, float] = {"L_total": 0.0, "L_vel": 0.0, "L_acc": 0.0,
                                       "L_lin": 0.0, "L_stab": 0.0, "L_bias": 0.0,
                                       "L_l2": 0.0, "spec_radius": 0.0}
        t0 = time.time()
        n_batches = 0
        for x_t_full, x_target_seq, u_seq in train_loader:
            x_t_full = x_t_full.to(device, non_blocking=True)
            x_target_seq = x_target_seq.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss, info = compute_losses(
                        model, x_t_full, x_target_seq, u_seq,
                        dyn_mean_t, dyn_std_t, args, epoch, args.detach_target_lat,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, info = compute_losses(
                    model, x_t_full, x_target_seq, u_seq,
                    dyn_mean_t, dyn_std_t, args, epoch, args.detach_target_lat,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            if ema is not None:
                ema.update(model)
            for k in epoch_acc:
                epoch_acc[k] += info.get(k, 0.0)
            n_batches += 1

        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        for k in epoch_acc:
            epoch_acc[k] /= max(n_batches, 1)

        # ---- 验证：用 EMA 权重（如果有） ----
        eval_model = ema.module if ema is not None else model
        val_metrics = quick_validation(
            eval_model, val_ds, pred_len, device, args.dt,
            batch_size=args.batch_size, max_samples=args.val_max_samples,
        )

        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # 写 TB
        for k, v in epoch_acc.items():
            tb.add_scalar(f"Train/{k}", v, epoch)
        tb.add_scalar("Train/lr", cur_lr, epoch)
        tb.add_scalar("Train/pred_len", pred_len, epoch)
        for k, v in val_metrics.items():
            tb.add_scalar(k, v, epoch)
        tb.add_scalar("Val/Divergence/slope_loglog", val_metrics["val/slope_loglog"], epoch)
        tb.add_scalar(
            "Val/Divergence/ratio_stepK_over_step1",
            val_metrics[f"val/ratio_step{pred_len}_over_step1"], epoch,
        )
        tb.add_scalar("Val/Divergence/instability_score", val_metrics["val/instability_score"], epoch)

        # JSONL
        rec = {"epoch": epoch + 1, "lr": cur_lr, "pred_len": pred_len,
               "elapsed_sec": elapsed, **epoch_acc, **val_metrics}
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # 日志行
        log_msg = (
            f"Epoch [{epoch + 1:03d}/{args.epochs}] pl={pred_len} "
            f"LR={cur_lr:.2e} "
            f"Ltot={epoch_acc['L_total']:.4f} (vel={epoch_acc['L_vel']:.4f}, "
            f"acc={epoch_acc['L_acc']:.4f}, lin={epoch_acc['L_lin']:.4f}, "
            f"stab={epoch_acc['L_stab']:.4f}) "
            f"| val_vel_rmse_mean={val_metrics['val/vel_rmse_mean']:.5f} "
            f"@K={val_metrics[f'val/vel_rmse_step{pred_len}']:.5f} "
            f"slope={val_metrics['val/slope_loglog']:.3f} "
            f"inst={val_metrics['val/instability_score']:.3f} "
            f"spec={epoch_acc['spec_radius']:.3f} "
            f"| {elapsed:.1f}s"
        )
        logger.info(log_msg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics_summary_lines.append(
                f"{epoch + 1:03d} | {cur_lr:.2e} | {epoch_acc['L_total']:.4f} | "
                f"{epoch_acc['L_vel']:.4f} | {epoch_acc['L_acc']:.4f} | {epoch_acc['L_lin']:.4f} | "
                f"{val_metrics['val/vel_rmse_mean']:.5f} | "
                f"{val_metrics[f'val/vel_rmse_step{pred_len}']:.5f} | "
                f"{val_metrics[f'val/traj_xy_rmse_step{pred_len}']:.4f} | "
                f"{val_metrics['val/slope_loglog']:.3f} | "
                f"{val_metrics['val/instability_score']:.3f} | "
                f"{epoch_acc['spec_radius']:.3f}"
            )

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "best_metric": state.best_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "args": vars(args),
            "model_class": model.__class__.__name__,
            "feature_dict_atoms": list(FEATURE_DICT_ATOMS),
            "latent_dim": int(getattr(model, "latent_dim", 32)),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_latest.pth"))

        # best 由 args.best_metric 决定（composite 默认平衡精度与发散）
        if args.best_metric == "vel_rmse_mean":
            cur_metric = val_metrics["val/vel_rmse_mean"]
        elif args.best_metric == "instability_score":
            cur_metric = val_metrics["val/instability_score"]
        else:  # composite
            cur_metric = val_metrics["val/vel_rmse_mean"] * max(1.0, val_metrics["val/instability_score"])
        if cur_metric < state.best_metric:
            state.best_metric = cur_metric
            ckpt["best_metric"] = state.best_metric
            torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_best.pth"))
            logger.info(
                f"  ↳ new best [{args.best_metric}]={cur_metric:.6g} "
                f"(vel_rmse_mean={val_metrics['val/vel_rmse_mean']:.6g}, "
                f"inst={val_metrics['val/instability_score']:.6g}, epoch {epoch + 1})"
            )

    # ---- 训练结束：导出 best 的 yaml ----
    best_path = os.path.join(args.ckpt_dir, "koopman_best.pth")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device, weights_only=False)
        sd = ck.get("ema_state_dict") or ck["model_state_dict"]
        # 用同一种 build_model 实例化以匹配 state_dict
        export_model = build_model(args, torch.device("cpu"))
        export_model.load_state_dict(sd)
        export_yaml = os.path.join(args.ckpt_dir, "koopman_best.yaml")
        export_params_to_yaml(export_model, ck["stats"], export_yaml)
        logger.info(f"Exported YAML -> {export_yaml}")

    # ---- 训练结束：在 test 上跑一次 best ckpt ----
    if os.path.exists(best_path) and not args.smoketest:
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            res = ek.evaluate_one(
                ckpt_path=best_path, data_path=args.test_data,
                pred_len=args.pred_len_max, dt=args.dt,
                tag=args.run_tag, out_dir=args.out_dir, device=device,
                batch_size=args.batch_size, max_samples=None, write_files=True,
            )
            verdict = ek.quantitative_verdict_block(res.summary)
            logger.info("\n" + verdict)
            # 若是 v3 模型，则额外打印 12 阈值 verdict（PROMPT §9）。
            if args.model == "v3":
                v1_res = None
                v2_res = None
                v1_path = os.path.join(args.ckpt_dir, "koopman_v1_best.pth")
                v2_path = os.path.join(args.ckpt_dir, "koopman_v2_best.pth")
                try:
                    if os.path.exists(v1_path):
                        v1_res = ek.evaluate_one(
                            ckpt_path=v1_path, data_path=args.test_data,
                            pred_len=args.pred_len_max, dt=args.dt,
                            tag="v1", out_dir=os.path.join(args.out_dir, "_v1_aux"),
                            device=device, batch_size=args.batch_size,
                            max_samples=None, write_files=False,
                        )
                    if os.path.exists(v2_path):
                        v2_res = ek.evaluate_one(
                            ckpt_path=v2_path, data_path=args.test_data,
                            pred_len=args.pred_len_max, dt=args.dt,
                            tag="v2", out_dir=os.path.join(args.out_dir, "_v2_aux"),
                            device=device, batch_size=args.batch_size,
                            max_samples=None, write_files=False,
                        )
                except Exception as ee:
                    logger.warning(f"v1/v2 aux eval failed: {ee}")
                v3_md, all_pass = ek.v3_threshold_verdict(res, v1_res=v1_res, v2_res=v2_res)
                logger.info("\n" + v3_md)
                logger.info(f"v3 auto-PASS = {all_pass}")
        except Exception as e:
            logger.warning(f"Auto post-train test eval failed: {e}")

    # 训练摘要
    logger.info("Per-5-epoch summary:")
    logger.info("ep | lr | L_total | L_vel | L_acc | L_lin | val_vel_mean | val_vel@K | val_traj_xy@K | slope_ll | inst | spec")
    for ln in metrics_summary_lines:
        logger.info("  " + ln)


# =============================================================================
# 8. Smoketest
# =============================================================================

def run_smoketest(args: argparse.Namespace) -> int:
    """前 2 段 / pred_len=4 / 2 epoch / batch=16，无 GPU 1 分钟内完成。"""
    logger, _ = setup_logger(args.log_dir)
    logger.info("[smoketest] starting...")
    # 临时构造 mini npz
    out_dir = "logs/smoketest_v2"
    os.makedirs(out_dir, exist_ok=True)
    raw = np.load(args.train_data, allow_pickle=True)["datas"]
    raw_v = np.load(args.val_data, allow_pickle=True)["datas"]
    mini_train = os.path.join(out_dir, "mini_train.npz")
    mini_val = os.path.join(out_dir, "mini_val.npz")
    np.savez(mini_train, datas=np.array([raw[0], raw[1]], dtype=object))
    np.savez(mini_val, datas=np.array([raw_v[0]], dtype=object))

    args.train_data = mini_train
    args.val_data = mini_val
    args.epochs = 2
    args.batch_size = 16
    args.pred_len_start = 4
    args.pred_len_max = 4
    args.pred_len_grow_every = 999
    args.num_workers = 0
    args.persistent_workers = False
    args.amp = False
    args.val_max_samples = 64
    args.ckpt_dir = os.path.join(out_dir, "ckpt")
    args.log_dir = os.path.join(out_dir, "log")
    args.out_dir = os.path.join(out_dir, "test_out")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    train(args)
    # 检查关键产物
    for fn in ["koopman_latest.pth", "koopman_best.pth", "koopman_best.yaml"]:
        p = os.path.join(args.ckpt_dir, fn)
        assert os.path.exists(p), f"missing {p}"
    logger.info("[smoketest] OK")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args = merge_yaml(args, parser, sys.argv[1:])
    if args.smoketest:
        return run_smoketest(args)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
