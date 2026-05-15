"""train_koopman_v2.py
==========================================================
重写版 Deep-Koopman 多步预测训练脚本 (v2)。

主要目标：显著提升速度 (u, v, r) 的多步跟踪精度，并保持与
`koopman.py::HorizontalKoopmanModel`、`test_and_plot.py`、YAML 部署管线
的接口完全兼容（checkpoint 字典字段、stats 结构、YAML 键名均不变）。

核心改动 vs `train_multistep_voyage.py`：
  1) 主损失改为「物理速度 Huber 损失 + γ^k 逐步加权 + per-channel 1/std」，
     直接监督绝对速度，对偏置/漂移敏感。
  2) 加速度损失保留为辅助平滑项 (Huber)。
  3) `loss_linear` 保留，对前 5 epoch 做 ramp-up，避免拉崩 encoder。
  4) Encoder dropout 默认关闭（保证 encode(GT) 与 rollout 分布一致）。
  5) curriculum：`pred_len` 由 4 起步、每 10 epoch +2 直至 20。
  6) AdamW + CosineAnnealingWarmRestarts，AMP，可选 EMA。
  7) best-ckpt 由验证集「物理速度 RMSE 均值」决定（不再用 acc loss）。
  8) Dataset 一次性拍平所有段为 numpy 矩阵，`__getitem__` 全切片，提速。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - 仅在缺少 PyYAML 的环境下走这条路
    yaml = None  # type: ignore

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore

from koopman import HorizontalKoopmanModel


# ==========================================================
# 0. 日志与杂项工具
# ==========================================================
def setup_logger(log_dir: str, timestamp: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("KoopmanTrainerV2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(
        os.path.join(log_dir, f"train_v2_{timestamp}.log"), encoding="utf-8"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:  # 保证 DataLoader worker 可复现
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ==========================================================
# 1. 数据集：所有段先拍平成一张大矩阵，再用整数索引切片
# ==========================================================
@dataclass
class KoopmanStats:
    """统计量 (numpy)；保留与旧版完全兼容的字段名。"""

    state_mean: np.ndarray  # (6,)
    state_std: np.ndarray   # (6,)
    ctrl_mean: np.ndarray   # (4,)
    ctrl_std: np.ndarray    # (4,)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "state_mean": self.state_mean.astype(np.float32),
            "state_std": self.state_std.astype(np.float32),
            "ctrl_mean": self.ctrl_mean.astype(np.float32),
            "ctrl_std": self.ctrl_std.astype(np.float32),
        }


def _flatten_segments(
    npz_path: str, max_segments: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 npz，把所有段拼成连续的 states_full(N,6)、ctrls_full(N,4)、seg_offsets(K+1,)"""
    segs = np.load(npz_path, allow_pickle=True)["datas"]
    if max_segments is not None:
        segs = segs[:max_segments]

    state_list: List[np.ndarray] = []
    ctrl_list: List[np.ndarray] = []
    offsets: List[int] = [0]
    for seg in segs:
        T = int(seg["len"])
        # 状态向量 [x, y, yaw, u, v, r] -> (T, 6)
        pos = np.asarray(seg["Pos"], dtype=np.float32)          # (2, T)
        euler = np.asarray(seg["Euler"], dtype=np.float32)       # (3, T)
        vel = np.asarray(seg["Vel"], dtype=np.float32)           # (2, T)
        pqr = np.asarray(seg["pqr"], dtype=np.float32)           # (1, T)
        cmd = np.asarray(seg["Thrusters_CMD"], dtype=np.float32) # (4, T)
        state = np.stack(
            [pos[0], pos[1], euler[2], vel[0], vel[1], pqr[0]], axis=1
        )  # (T, 6)
        state_list.append(state[:T])
        ctrl_list.append(cmd.T[:T])  # (T, 4)
        offsets.append(offsets[-1] + T)

    states_full = np.concatenate(state_list, axis=0).astype(np.float32)
    ctrls_full = np.concatenate(ctrl_list, axis=0).astype(np.float32)
    seg_offsets = np.asarray(offsets, dtype=np.int64)  # (K+1,)
    return states_full, ctrls_full, seg_offsets


class KoopmanVoyageDataset(Dataset):
    """一次性把所有段拼成大矩阵，`__getitem__` 只做整数索引切片。

    返回：
        x_t_norm            : (6,)
        x_target_seq_norm   : (pred_len, 6)
        u_seq_norm          : (pred_len, 4)
    """

    def __init__(
        self,
        npz_path: str,
        pred_len: int,
        stats: Optional[KoopmanStats] = None,
        stride: int = 1,
        max_segments: Optional[int] = None,
    ) -> None:
        super().__init__()
        states_full, ctrls_full, seg_offsets = _flatten_segments(
            npz_path, max_segments=max_segments
        )
        self.states_full = states_full
        self.ctrls_full = ctrls_full
        self.seg_offsets = seg_offsets
        self.pred_len = int(pred_len)
        self.stride = int(max(1, stride))

        # 统计量：用所有 (states_full, ctrls_full) 一次性算
        if stats is None:
            stats = KoopmanStats(
                state_mean=np.mean(states_full, axis=0),
                state_std=np.std(states_full, axis=0) + 1e-6,
                ctrl_mean=np.mean(ctrls_full, axis=0),
                ctrl_std=np.std(ctrls_full, axis=0) + 1e-6,
            )
        self.stats = stats

        # 预先归一化（一次性，避免 __getitem__ 重复除法）
        self.states_norm = (
            (states_full - stats.state_mean[None, :]) / stats.state_std[None, :]
        ).astype(np.float32)
        self.ctrls_norm = (
            (ctrls_full - stats.ctrl_mean[None, :]) / stats.ctrl_std[None, :]
        ).astype(np.float32)

        # 生成 (start_idx,) 索引：对每一段，t ∈ [0, seg_len - pred_len - 1]
        idx_list: List[np.ndarray] = []
        for k in range(len(seg_offsets) - 1):
            s, e = seg_offsets[k], seg_offsets[k + 1]
            seg_len = e - s
            if seg_len <= self.pred_len:
                continue
            local_t = np.arange(0, seg_len - self.pred_len, self.stride, dtype=np.int64)
            idx_list.append(s + local_t)
        if idx_list:
            self.start_indices = np.concatenate(idx_list).astype(np.int64)
        else:
            self.start_indices = np.empty(0, dtype=np.int64)

    def rebuild_indices(self, pred_len: int, stride: Optional[int] = None) -> None:
        """变更 pred_len（curriculum 时调用），仅重新生成索引，不重建数据矩阵。"""
        self.pred_len = int(pred_len)
        if stride is not None:
            self.stride = int(max(1, stride))
        idx_list: List[np.ndarray] = []
        for k in range(len(self.seg_offsets) - 1):
            s, e = self.seg_offsets[k], self.seg_offsets[k + 1]
            seg_len = e - s
            if seg_len <= self.pred_len:
                continue
            local_t = np.arange(0, seg_len - self.pred_len, self.stride, dtype=np.int64)
            idx_list.append(s + local_t)
        if idx_list:
            self.start_indices = np.concatenate(idx_list).astype(np.int64)
        else:
            self.start_indices = np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.start_indices.shape[0])

    def __getitem__(self, index: int):
        t = int(self.start_indices[index])
        pl = self.pred_len
        x_t = self.states_norm[t]                       # (6,)
        x_target = self.states_norm[t + 1 : t + 1 + pl] # (pl, 6)
        u_seq = self.ctrls_norm[t : t + pl]             # (pl, 4)
        return (
            torch.from_numpy(x_t),
            torch.from_numpy(x_target),
            torch.from_numpy(u_seq),
        )


# ==========================================================
# 2. EMA (Exponential Moving Average)
# ==========================================================
class ModelEMA:
    """轻量 EMA：内部维护一份 fp32 参数副本，验证 / 部署用 EMA 权重。"""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """把 EMA 权重灌进 model；返回原始权重以便后续恢复。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # 类型对齐
        new_sd: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            ema_v = self.shadow[k]
            new_sd[k] = ema_v.to(v.dtype).to(v.device)
        model.load_state_dict(new_sd, strict=True)
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        model.load_state_dict(backup, strict=True)


# ==========================================================
# 3. YAML 导出（沿用旧字段名）
# ==========================================================
def export_params_to_yaml(
    model: HorizontalKoopmanModel,
    stats: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """与现行 `train_multistep_voyage.py` 完全一致的 YAML 结构。"""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML 未安装，无法导出 YAML")

    A_w = model.A.weight.detach().cpu()
    I = torch.eye(A_w.size(0))
    A_w_plus_I = (A_w + I).numpy().tolist()
    A_bias = (
        model.A.bias.detach().cpu().numpy().tolist()
        if getattr(model.A, "bias", None) is not None
        else []
    )
    B = model.B.weight.detach().cpu().numpy().tolist()

    yaml_data = {
        "normalization": {
            "dyn_mean": np.asarray(stats["state_mean"])[3:6].astype(float).tolist(),
            "dyn_std": np.asarray(stats["state_std"])[3:6].astype(float).tolist(),
            "ctrl_mean": np.asarray(stats["ctrl_mean"]).astype(float).tolist(),
            "ctrl_std": np.asarray(stats["ctrl_std"]).astype(float).tolist(),
        },
        "system_matrices": {
            "A_weight": A_w_plus_I,
            "A_bias": A_bias,
            "B": B,
        },
        "info": "Latent z = [u, v, r, u|u|, v|v|, r|r|, vr, ur, h_1..h_24]",
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, indent=2, sort_keys=False)


# ==========================================================
# 4. 关键计算：Rollout + 各项损失
# ==========================================================
def rollout_pred(
    model: HorizontalKoopmanModel,
    dyn_init_norm: torch.Tensor,        # (B, 3)
    u_seq_norm: torch.Tensor,           # (B, T, 4)
    pred_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """非 teacher-force 的隐空间多步预测。

    Returns:
        pred_dyn_norm: (B, T, 3) 归一化后的 (u,v,r) 预测序列
        pred_latents : (B, T, latent_dim)
    """
    z = model.encode(dyn_init_norm)
    preds_dyn: List[torch.Tensor] = []
    preds_z: List[torch.Tensor] = []
    for k in range(pred_len):
        z = model.latent_step(z, u_seq_norm[:, k, :])
        preds_z.append(z)
        preds_dyn.append(model.reconstruct_state(z))
    return torch.stack(preds_dyn, dim=1), torch.stack(preds_z, dim=1)


def step_weights(pred_len: int, gamma: float, device: torch.device) -> torch.Tensor:
    """w_k = γ^k （k=0..pred_len-1），并归一化使 mean(w_k) = 1，避免与权重直接耦合。"""
    k = torch.arange(pred_len, device=device, dtype=torch.float32)
    w = gamma ** k
    w = w * (pred_len / w.sum().clamp_min(1e-8))
    return w  # (pred_len,)


def _huber(x: torch.Tensor, beta: float) -> torch.Tensor:
    """与 nn.SmoothL1Loss(beta) 等价、reduction='none' 的实现，便于自带逐元素加权。"""
    ax = x.abs()
    quad = 0.5 * x.pow(2) / beta
    lin = ax - 0.5 * beta
    return torch.where(ax < beta, quad, lin)


# ==========================================================
# 5. 验证：完整 rollout 物理空间指标 + 外部欧拉积分位置误差
# ==========================================================
@torch.no_grad()
def evaluate(
    model: HorizontalKoopmanModel,
    val_loader: DataLoader,
    stats_dyn_mean: torch.Tensor,   # (3,)
    stats_dyn_std: torch.Tensor,    # (3,)
    stats_state_mean: torch.Tensor, # (6,)
    stats_state_std: torch.Tensor,  # (6,)
    pred_len: int,
    dt: float,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sq_err_steps = torch.zeros(pred_len, device=device)  # 累加 sum (u-pred)^2+(v-pred)^2+(r-pred)^2
    n_samples = 0
    sum_acc_sq = 0.0
    n_acc = 0
    sum_traj_xy_sq_20 = 0.0
    n_traj = 0

    for x_t_full, x_target_seq_full, u_seq in val_loader:
        x_t_full = x_t_full.to(device, non_blocking=True)
        x_target_seq_full = x_target_seq_full.to(device, non_blocking=True)
        u_seq = u_seq.to(device, non_blocking=True)

        dyn_init = x_t_full[:, 3:6]
        pred_dyn_norm, _ = rollout_pred(model, dyn_init, u_seq, pred_len)

        gt_dyn_norm = x_target_seq_full[:, :, 3:6]
        pred_phys = pred_dyn_norm * stats_dyn_std + stats_dyn_mean         # (B, T, 3)
        gt_phys = gt_dyn_norm * stats_dyn_std + stats_dyn_mean             # (B, T, 3)

        diff = pred_phys - gt_phys                                          # (B, T, 3)
        sq_err_steps += diff.pow(2).sum(dim=(0, 2))
        n_samples += diff.shape[0]

        # 加速度（相邻差分）
        pred_acc = (pred_phys[:, 1:] - pred_phys[:, :-1]) / dt
        gt_acc = (gt_phys[:, 1:] - gt_phys[:, :-1]) / dt
        sum_acc_sq += (pred_acc - gt_acc).pow(2).sum().item()
        n_acc += pred_acc.numel()

        # 外部欧拉积分得到位置/航向（与 test_and_plot.py 完全一致的公式）
        if pred_len >= 20:
            x_t_phys = x_t_full * stats_state_std + stats_state_mean  # (B, 6)
            cur_x = x_t_phys[:, 0]
            cur_y = x_t_phys[:, 1]
            cur_yaw = x_t_phys[:, 2]
            gt_full_phys = x_target_seq_full * stats_state_std + stats_state_mean  # (B, T, 6)
            for step in range(20):
                u_s = pred_phys[:, step, 0]
                v_s = pred_phys[:, step, 1]
                r_s = pred_phys[:, step, 2]
                nx = cur_x + (u_s * torch.cos(cur_yaw) - v_s * torch.sin(cur_yaw)) * dt
                ny = cur_y + (u_s * torch.sin(cur_yaw) + v_s * torch.cos(cur_yaw)) * dt
                nyaw = cur_yaw + r_s * dt
                cur_x, cur_y, cur_yaw = nx, ny, nyaw
            gt_x20 = gt_full_phys[:, 19, 0]
            gt_y20 = gt_full_phys[:, 19, 1]
            sum_traj_xy_sq_20 += ((cur_x - gt_x20).pow(2) + (cur_y - gt_y20).pow(2)).sum().item()
            n_traj += cur_x.numel()

    vel_rmse_step = torch.sqrt(sq_err_steps / max(1, n_samples) / 3.0).detach().cpu().numpy()  # 每步 (u,v,r) 综合 RMSE
    acc_rmse_mean = math.sqrt(sum_acc_sq / max(1, n_acc))
    metrics = {
        "vel_rmse_mean": float(np.mean(vel_rmse_step)),
        "vel_rmse_step_1": float(vel_rmse_step[0]) if pred_len >= 1 else float("nan"),
        "vel_rmse_step_5": float(vel_rmse_step[4]) if pred_len >= 5 else float("nan"),
        "vel_rmse_step_10": float(vel_rmse_step[9]) if pred_len >= 10 else float("nan"),
        "vel_rmse_step_20": float(vel_rmse_step[19]) if pred_len >= 20 else float("nan"),
        "acc_rmse_mean": acc_rmse_mean,
        "traj_xy_rmse_20": math.sqrt(sum_traj_xy_sq_20 / n_traj) if n_traj > 0 else float("nan"),
    }
    return metrics


# ==========================================================
# 6. 主训练
# ==========================================================
def build_dataloader(
    dataset: KoopmanVoyageDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        generator=g if shuffle else None,
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def cur_pred_len(epoch: int, args: argparse.Namespace) -> int:
    """curriculum：每 N epoch 把窗口 +2，直至 max。"""
    grow = (epoch // max(1, args.pred_len_grow_every)) * 2
    return int(min(args.pred_len_max, args.pred_len_start + grow))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(args.log_dir, timestamp)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"

    tb_writer = None
    if SummaryWriter is not None:
        tb_writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"tb_v2_{timestamp}"))

    # ----------------- 数据 -----------------
    train_pred_len = cur_pred_len(0, args)
    if args.smoketest:
        train_ds = KoopmanVoyageDataset(
            args.train_data, pred_len=train_pred_len, max_segments=2
        )
        val_ds = KoopmanVoyageDataset(
            args.val_data,
            pred_len=args.pred_len_max,
            stats=train_ds.stats,
            max_segments=1,
        )
    else:
        train_ds = KoopmanVoyageDataset(args.train_data, pred_len=train_pred_len)
        val_ds = KoopmanVoyageDataset(
            args.val_data, pred_len=args.pred_len_max, stats=train_ds.stats
        )

    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=max(64, args.batch_size),
        shuffle=False,
        num_workers=min(4, args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor,
        seed=args.seed,
    )

    # ----------------- 模型 -----------------
    model = HorizontalKoopmanModel(state_dim=3, control_dim=4, hidden_dim=24).to(device)
    # 关闭/调整 encoder dropout
    for m in model.encoder_mlp.modules():
        if isinstance(m, nn.Dropout):
            m.p = float(args.encoder_dropout)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, args.cosine_T0), T_mult=max(1, args.cosine_Tmult)
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=max(1, args.epochs * len(train_loader)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    )

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # 统计量张量
    stats_d = train_ds.stats.as_dict()
    state_mean_t = torch.tensor(stats_d["state_mean"], device=device)
    state_std_t = torch.tensor(stats_d["state_std"], device=device)
    dyn_mean_t = state_mean_t[3:6]
    dyn_std_t = state_std_t[3:6]

    # ----------------- resume -----------------
    start_epoch = 0
    best_metric = float("inf")
    if args.resume and os.path.exists(args.resume):
        logger.info(f"[Resume] 从 {args.resume} 续训")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck and ck["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"scheduler resume 失败，忽略：{e}")
        if scaler is not None and ck.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ck["scaler_state_dict"])
        if ema is not None and ck.get("ema_state_dict") is not None:
            ema.load_state_dict(ck["ema_state_dict"])
        start_epoch = int(ck.get("epoch", 0))
        best_metric = float(ck.get("best_metric", best_metric))

    # ----------------- 打印启动信息 -----------------
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Device           = {device} | AMP = {use_amp}")
    logger.info(f"Batch / NumWork  = {args.batch_size} / {args.num_workers}")
    logger.info(
        f"pred_len curr.   = start {args.pred_len_start} -> max {args.pred_len_max}"
        f" (grow every {args.pred_len_grow_every} ep)"
    )
    logger.info(
        f"Loss weights     = vel {args.w_vel} | acc {args.w_acc} | lin {args.w_lin}"
        f" | recon {args.w_recon} | stab {args.w_stab} | l2 {args.w_l2}"
    )
    logger.info(
        f"Dataset          = train {len(train_ds)} samples | val {len(val_ds)} samples"
    )
    logger.info(f"Model params     = {n_params:,}")
    logger.info(f"Spec radius init = {model.spectral_radius().item():.4f}")
    logger.info(
        f"Optim/Sched      = AdamW(lr={args.lr}, wd={args.weight_decay}) / {args.scheduler}"
    )

    # ----------------- 训练 loop -----------------
    metrics_path = os.path.join(args.log_dir, f"metrics_v2_{timestamp}.jsonl")
    metrics_fh = open(metrics_path, "a", encoding="utf-8")

    current_pred_len = train_pred_len
    for epoch in range(start_epoch, args.epochs):
        # ---- curriculum：若 pred_len 变化则重建 indices ----
        desired_pred_len = cur_pred_len(epoch, args)
        if desired_pred_len != current_pred_len:
            current_pred_len = desired_pred_len
            train_ds.rebuild_indices(current_pred_len)
            train_loader = build_dataloader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=args.num_workers > 0,
                prefetch_factor=args.prefetch_factor,
                seed=args.seed + epoch,
            )
            logger.info(
                f"[Curriculum] epoch {epoch+1}: pred_len -> {current_pred_len}, "
                f"#train samples -> {len(train_ds)}"
            )

        # ---- 前 5 epoch 对 L_lin、L_stab 做线性 ramp-up ----
        ramp = min(1.0, (epoch + 1) / 5.0)
        w_lin_ep = args.w_lin * ramp
        w_stab_ep = args.w_stab * ramp

        model.train()
        ep_losses = {"total": 0.0, "vel": 0.0, "acc": 0.0, "lin": 0.0, "recon": 0.0, "stab": 0.0, "l2": 0.0}
        n_batches = 0
        t0 = time.time()
        for x_t_full, x_target_seq_full, u_seq in train_loader:
            x_t_full = x_t_full.to(device, non_blocking=True)
            x_target_seq_full = x_target_seq_full.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                dyn_init = x_t_full[:, 3:6]
                gt_dyn_norm = x_target_seq_full[:, :, 3:6]
                B, T, _ = gt_dyn_norm.shape

                pred_dyn_norm, pred_z = rollout_pred(model, dyn_init, u_seq, T)  # (B,T,3), (B,T,latent)

                # ------ L_vel: 物理速度 Huber + per-channel 1/std + γ^k 加权 ------
                pred_phys = pred_dyn_norm * dyn_std_t + dyn_mean_t
                gt_phys = gt_dyn_norm * dyn_std_t + dyn_mean_t
                vel_diff_unit = (pred_phys - gt_phys) / dyn_std_t  # 等价于归一化空间差，但语义清晰
                w_k = step_weights(T, args.gamma_w, device).view(1, T, 1)
                vel_huber = _huber(vel_diff_unit, beta=0.1) * w_k
                loss_vel = vel_huber.mean()

                # ------ L_acc: 物理加速度 Huber + γ^(k) 加权（取 k=1..T 的相邻差分） ------
                pred_acc = (pred_phys[:, 1:] - pred_phys[:, :-1]) / args.dt
                gt_acc = (gt_phys[:, 1:] - gt_phys[:, :-1]) / args.dt
                acc_diff_unit = (pred_acc - gt_acc) / dyn_std_t  # 也做 per-channel 缩放
                if T > 1:
                    w_acc_k = step_weights(T - 1, args.gamma_w, device).view(1, T - 1, 1)
                    loss_acc = (_huber(acc_diff_unit, beta=0.1) * w_acc_k).mean()
                else:
                    loss_acc = torch.tensor(0.0, device=device)

                # ------ L_lin: 隐空间一致性 ------
                z_gt = model.encode(gt_dyn_norm.reshape(B * T, 3)).view(B, T, -1)
                if args.detach_lin_target:
                    z_gt = z_gt.detach()
                loss_lin = (pred_z - z_gt).pow(2).mean()

                # ------ L_recon: 自编码恒等性（v1 模型恒为 0，留作 v2 复用） ------
                x_recon = model.reconstruct_state(model.encode(dyn_init))
                loss_recon = (x_recon - dyn_init).pow(2).mean()

                # ------ L_stab: 谱半径软约束 ------
                spec = model.spectral_radius()
                loss_stab = torch.relu(spec - args.rho_max).pow(2)

                # ------ L_l2_A: A_weight + B 的 L2 ------
                loss_l2 = model.A.weight.pow(2).sum() + model.B.weight.pow(2).sum()

                loss = (
                    args.w_vel * loss_vel
                    + args.w_acc * loss_acc
                    + w_lin_ep * loss_lin
                    + args.w_recon * loss_recon
                    + w_stab_ep * loss_stab
                    + args.w_l2 * loss_l2
                )

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if args.scheduler == "onecycle":
                scheduler.step()

            if ema is not None:
                ema.update(model)

            ep_losses["total"] += float(loss.item())
            ep_losses["vel"] += float(loss_vel.item())
            ep_losses["acc"] += float(loss_acc.item())
            ep_losses["lin"] += float(loss_lin.item())
            ep_losses["recon"] += float(loss_recon.item())
            ep_losses["stab"] += float(loss_stab.item())
            ep_losses["l2"] += float(loss_l2.item())
            n_batches += 1

        if args.scheduler != "onecycle":
            scheduler.step()

        for k in ep_losses:
            ep_losses[k] /= max(1, n_batches)
        train_time = time.time() - t0

        # ---- 验证（始终用 pred_len_max；若启用 EMA，验证用 EMA 权重）----
        backup = None
        if ema is not None:
            backup = ema.copy_to(model)
        metrics = evaluate(
            model,
            val_loader,
            stats_dyn_mean=dyn_mean_t,
            stats_dyn_std=dyn_std_t,
            stats_state_mean=state_mean_t,
            stats_state_std=state_std_t,
            pred_len=args.pred_len_max,
            dt=args.dt,
            device=device,
        )
        if backup is not None:
            ModelEMA.restore(model, backup)

        cur_lr = optimizer.param_groups[0]["lr"]
        log_msg = (
            f"Epoch [{epoch+1:03d}/{args.epochs}] LR={cur_lr:.6f} | "
            f"L_tot={ep_losses['total']:.4f} (vel={ep_losses['vel']:.4f}, "
            f"acc={ep_losses['acc']:.4f}, lin={ep_losses['lin']:.4f}, "
            f"stab={ep_losses['stab']:.4f}) | "
            f"val_vel_rmse_mean={metrics['vel_rmse_mean']:.4f} | "
            f"val_vel_rmse@20={metrics['vel_rmse_step_20']:.4f} | "
            f"val_traj_xy@20={metrics['traj_xy_rmse_20']:.4f} | "
            f"spec={model.spectral_radius().item():.4f} | t={train_time:.1f}s"
        )
        logger.info(log_msg)

        # TensorBoard
        if tb_writer is not None:
            for k, v in ep_losses.items():
                tb_writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in metrics.items():
                tb_writer.add_scalar(f"Val/{k}", v, epoch)
            tb_writer.add_scalar("LR", cur_lr, epoch)
            tb_writer.add_scalar("Train/pred_len", current_pred_len, epoch)
            tb_writer.add_scalar("Model/spec_radius", model.spectral_radius().item(), epoch)

        # metrics jsonl dump
        rec = {
            "epoch": epoch + 1,
            "lr": cur_lr,
            "pred_len": current_pred_len,
            "train": ep_losses,
            "val": metrics,
            "spec_radius": float(model.spectral_radius().item()),
            "train_time_s": train_time,
        }
        metrics_fh.write(json.dumps(rec) + "\n")
        metrics_fh.flush()

        # checkpoint
        os.makedirs(args.ckpt_dir, exist_ok=True)
        ckpt: Dict[str, Any] = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "stats": stats_d,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "best_metric": best_metric,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_latest.pth"))

        if metrics["vel_rmse_mean"] < best_metric:
            best_metric = float(metrics["vel_rmse_mean"])
            # 部署用 EMA 权重
            if ema is not None:
                ema_backup = ema.copy_to(model)
                ckpt_best = dict(ckpt)
                ckpt_best["model_state_dict"] = model.state_dict()
                ckpt_best["best_metric"] = best_metric
                torch.save(ckpt_best, os.path.join(args.ckpt_dir, "koopman_best.pth"))
                ModelEMA.restore(model, ema_backup)
            else:
                ckpt["best_metric"] = best_metric
                torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_best.pth"))
            logger.info(
                f"  ✅ 新 best: vel_rmse_mean={best_metric:.4f} (epoch {epoch+1})"
            )

    metrics_fh.close()

    # 训练结束：用 best ckpt 导 YAML
    best_path = os.path.join(args.ckpt_dir, "koopman_best.pth")
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        export_model = HorizontalKoopmanModel(state_dim=3, control_dim=4, hidden_dim=24)
        export_model.load_state_dict(ck["model_state_dict"])
        yaml_path = os.path.join(args.ckpt_dir, "koopman_best.yaml")
        try:
            export_params_to_yaml(export_model, ck["stats"], yaml_path)
            logger.info(f"[YAML] 已导出 best 模型 -> {yaml_path}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[YAML] 导出失败：{e}")

    if tb_writer is not None:
        tb_writer.close()


# ==========================================================
# 7. CLI 与 smoketest
# ==========================================================
def _maybe_load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML 未安装，无法读取 --config")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deep-Koopman trainer v2")
    # 数据
    p.add_argument("--train_data", type=str, default="koopman_train_merged.npz")
    p.add_argument("--val_data", type=str, default="koopman_val.npz")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--config", type=str, default=None, help="可选 YAML 配置；CLI 显式优先")
    p.add_argument("--model", type=str, default="v1", choices=["v1", "v2"],
                   help="v1 = koopman.py::HorizontalKoopmanModel；v2 仅在用户单独提供 koopman_v2.py 时启用")
    # 训练超参
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "onecycle", "step"])
    p.add_argument("--cosine_T0", type=int, default=20)
    p.add_argument("--cosine_Tmult", type=int, default=2)
    p.add_argument("--step_size", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--encoder_dropout", type=float, default=0.0)
    p.add_argument("--cpu", action="store_true")
    # curriculum
    p.add_argument("--pred_len_start", type=int, default=4)
    p.add_argument("--pred_len_max", type=int, default=20)
    p.add_argument("--pred_len_grow_every", type=int, default=10)
    p.add_argument("--dt", type=float, default=0.1)
    # 损失权重
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_acc", type=float, default=0.2)
    p.add_argument("--w_lin", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.0)
    p.add_argument("--w_stab", type=float, default=0.1)
    p.add_argument("--w_l2", type=float, default=1e-4)
    p.add_argument("--gamma_w", type=float, default=0.97, help="γ for step-wise weighting w_k=γ^k")
    p.add_argument("--rho_max", type=float, default=1.005)
    p.add_argument("--detach_lin_target", action="store_true", default=False,
                   help="对 latent 一致性目标做 detach（消融用）")
    # 数据 loader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    # 其它
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--smoketest", action="store_true",
                   help="只取训练集前 2 段、pred_len=4、epochs=2、batch=16，跑通全流程")
    return p


def apply_config_overrides(args: argparse.Namespace, cfg: Dict[str, Any]) -> argparse.Namespace:
    """YAML > CLI 默认；CLI 显式 > YAML。"""
    if not cfg:
        return args
    # 哪些参数是用户在命令行显式指定的
    user_set = set()
    seen = set()
    for tok in sys.argv[1:]:
        if tok.startswith("--"):
            name = tok.lstrip("-").split("=", 1)[0]
            user_set.add(name)
            seen.add(name)
    for k, v in cfg.items():
        if k in user_set:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def maybe_apply_smoketest(args: argparse.Namespace) -> argparse.Namespace:
    if not args.smoketest:
        return args
    args.epochs = 2
    args.batch_size = 16
    args.pred_len_start = 4
    args.pred_len_max = 4
    args.pred_len_grow_every = 1
    args.num_workers = 0
    args.prefetch_factor = 2
    args.amp = False
    args.ema_decay = 0.0  # 关 EMA 加速
    args.scheduler = "step"
    return args


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    cfg = _maybe_load_yaml_config(args.config)
    args = apply_config_overrides(args, cfg)
    args = maybe_apply_smoketest(args)

    if args.model == "v2":
        raise NotImplementedError(
            "本仓库默认只提供 v1 (koopman.py)；如需 v2 请单独添加 koopman_v2.py "
            "并扩展本脚本的模型构造分支。"
        )

    train(args)


if __name__ == "__main__":
    main()
