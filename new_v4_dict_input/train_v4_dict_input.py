"""Train script for v4 dict-input Koopman model.

特点：
- 参考 scripts/train_v2.py 的训练流程（dataset/curriculum/EMA/多项损失）。
- 使用 new_v4_dict_input.model_v4_dict_input.HorizontalKoopmanModelV4DictInput。
- 保留原有脚本不变；本脚本完全独立。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import paths as P  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402
from koopman import evalkit as ek  # noqa: E402
from new_v4_dict_input.model_v4_dict_input import (  # noqa: E402
    FEATURE_DICT_ATOMS_16,
    HorizontalKoopmanModelV4DictInput,
)
from new_v4_dict_input.eval_v4_dict_input import (  # noqa: E402
    evaluate,
    plot_channel_rmse_vs_step,
    plot_channel_sample_curves,
    plot_channel_scatter,
    write_per_step_csv,
)

setup_repo()


@dataclass
class DistInfo:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_distributed(backend: str = "nccl") -> DistInfo:
    """由 torchrun / torch.distributed.launch 注入的环境变量初始化 DDP。

    GPU 多卡用 nccl；无 GPU 环境可用 ``--dist_backend gloo`` 做 CPU 多进程
    DDP（主要用于流程调试 / CI 冒烟，吞吐不如单进程大 batch）。
    """
    if "LOCAL_RANK" not in os.environ:
        return DistInfo()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return DistInfo()
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("nccl 后端需要 CUDA；CPU 环境请改用 --dist_backend gloo。")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size, backend=backend)


def cleanup_distributed(dinfo: DistInfo) -> None:
    if dinfo.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(dinfo: DistInfo) -> bool:
    return dinfo.rank == 0


def steps_from_seconds(seconds: float, dt: float) -> int:
    return max(1, int(round(float(seconds) / float(dt))))


def curriculum_pred_len(epoch: int, args: argparse.Namespace) -> int:
    grow = max(int(args.pred_len_grow_every), 1)
    step = max(int(args.pred_len_step), 1)
    start = max(int(args.pred_len_start), 1)
    target = start + step * (epoch // grow)
    return min(int(args.pred_len_max), target)


def resolve_prediction_horizon(args: argparse.Namespace) -> argparse.Namespace:
    args.pred_len_max = max(int(args.pred_len_max), 1)
    args.pred_len_start = min(max(int(args.pred_len_start), 1), args.pred_len_max)
    return args


def setup_logger(log_dir: str, dinfo: DistInfo, ts: Optional[str] = None) -> Tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"KoopmanV4_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if is_main_process(dinfo):
        fh = logging.FileHandler(os.path.join(log_dir, f"train_v4_{ts}.log"), encoding="utf-8")
        ch = logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    else:
        logger.addHandler(logging.NullHandler())
    return logger, ts


def broadcast_run_id(dinfo: DistInfo, ts: str) -> str:
    if not dinfo.enabled:
        return ts
    payload = [ts]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KoopmanVoyageDataset(Dataset):
    """与 train_v2 同口径的数据集：返回规范化后的 x_t/x_seq/u_seq。

    原始数据集采样间隔为 data_dt（默认 0.1s）；模型步长 dt（默认 1.0s）通过
    model_stride=dt/data_dt 对序列做下采样，pred_len 步仍覆盖 pred_len*dt 秒。
    """

    def __init__(
        self,
        npz_path: str,
        pred_len: int,
        stride: int = 1,
        stats: Optional[Dict[str, np.ndarray]] = None,
        model_stride: int = 1,
        data_dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.model_stride = max(int(model_stride), 1)
        self.data_dt = float(data_dt)
        (
            self.states_full,
            self.ctrls_full,
            self.seg_starts,
            self.seg_lens,
            self.t0_global,
            self.seg_idx,
            self.t0_local,
        ) = ek._flatten_segments(
            npz_path,
            pred_len=self.pred_len,
            stride=self.stride,
            model_stride=self.model_stride,
        )
        self.stats = stats if stats is not None else self._compute_stats()
        self._sm = self.stats["state_mean"].astype(np.float32)
        self._ss = self.stats["state_std"].astype(np.float32)
        self._cm = self.stats["ctrl_mean"].astype(np.float32)
        self._cs = self.stats["ctrl_std"].astype(np.float32)

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
        k = self.pred_len
        ms = self.model_stride
        x_t = self.states_full[t0]
        # 模型步 k 对应原始数据 t0+(k+1)*ms 的状态；控制取每 1s 块起点
        x_seq = self.states_full[t0 + ms : t0 + 1 + k * ms : ms]
        u_seq = self.ctrls_full[t0 : t0 + k * ms : ms]
        x_t_n = (x_t - self._sm) / self._ss
        x_seq_n = (x_seq - self._sm) / self._ss
        u_seq_n = (u_seq - self._cm) / self._cs
        return (
            torch.from_numpy(x_t_n.astype(np.float32, copy=False)),
            torch.from_numpy(x_seq_n.astype(np.float32, copy=False)),
            torch.from_numpy(u_seq_n.astype(np.float32, copy=False)),
        )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> Dict:
        return self.module.state_dict()


def huber(x: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    abs_x = x.abs()
    quad = 0.5 * (x ** 2) / beta
    lin = abs_x - 0.5 * beta
    return torch.where(abs_x < beta, quad, lin)


def make_step_weights(k: int, gamma: float, device: torch.device) -> torch.Tensor:
    w = gamma ** torch.arange(k, device=device, dtype=torch.float32)
    return w / w.mean()


def wrap_yaw_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """航向误差 wrap 到 [-pi, pi]。"""
    d = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


def integrate_pose_from_vel(
    pose0_phys: torch.Tensor,
    vel_phys: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """由初始位姿与逐步预测速度 (u,v,r) 做船体坐标系欧拉积分。

    Args:
        pose0_phys: (B, 3) 初始 [x, y, yaw]
        vel_phys: (B, K, 3) 每步预测 [u, v, r]
        dt: 模型离散步长 [s]

    Returns:
        (B, K, 3) 各预测步末位姿，与 x_target_seq 逐步对齐
    """
    x = pose0_phys[:, 0]
    y = pose0_phys[:, 1]
    yaw = pose0_phys[:, 2]
    dt_t = vel_phys.new_tensor(float(dt))
    poses: List[torch.Tensor] = []
    for i in range(vel_phys.size(1)):
        u = vel_phys[:, i, 0]
        v = vel_phys[:, i, 1]
        r = vel_phys[:, i, 2]
        x = x + (u * torch.cos(yaw) - v * torch.sin(yaw)) * dt_t
        y = y + (u * torch.sin(yaw) + v * torch.cos(yaw)) * dt_t
        yaw = yaw + r * dt_t
        poses.append(torch.stack([x, y, yaw], dim=-1))
    return torch.stack(poses, dim=1)


def denorm_pose(pose_n: torch.Tensor, pose_mean: torch.Tensor, pose_std: torch.Tensor) -> torch.Tensor:
    """反归一化位姿 [x,y,yaw]；pose_n 可为 (B,3) 或 (B,K,3)。"""
    if pose_n.dim() == 2:
        return pose_n * pose_std + pose_mean
    return pose_n * pose_std.view(1, 1, 3) + pose_mean.view(1, 1, 3)


def rollout_train(
    model: nn.Module,
    x_t_dyn_n: torch.Tensor,
    u_seq_n: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = u_seq_n.size(1)
    z = model.encode(x_t_dyn_n)
    pred_norm: List[torch.Tensor] = []
    pred_lat: List[torch.Tensor] = []
    for i in range(k):
        z = model.latent_step(z, u_seq_n[:, i, :])
        pred_lat.append(z)
        pred_norm.append(model.reconstruct_state(z))
    return torch.stack(pred_norm, dim=1), torch.stack(pred_lat, dim=1)


def compute_losses(
    model: nn.Module,
    x_t_full_n: torch.Tensor,
    x_target_seq_n: torch.Tensor,
    u_seq_n: torch.Tensor,
    dyn_mean: torch.Tensor,
    dyn_std: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz, k, _ = x_target_seq_n.shape
    dyn_t_n = x_t_full_n[:, 3:6]
    dyn_target_n = x_target_seq_n[:, :, 3:6]

    pred_norm_seq, pred_lat_seq = rollout_train(model, dyn_t_n, u_seq_n)
    pred_phys = pred_norm_seq * dyn_std + dyn_mean
    target_phys = dyn_target_n * dyn_std + dyn_mean

    step_w = make_step_weights(k, args.gamma_step, x_t_full_n.device).view(1, k, 1)
    step_w_pose = step_w.squeeze(-1)
    chan_scale = 1.0 / dyn_std.view(1, 1, 3)
    err_vel = pred_phys - target_phys
    l_vel = (huber(err_vel * chan_scale, beta=args.huber_beta) * step_w).mean()

    pred_acc = (pred_phys[:, 1:] - pred_phys[:, :-1]) / args.dt
    gt_acc = (target_phys[:, 1:] - target_phys[:, :-1]) / args.dt
    l_acc = huber((pred_acc - gt_acc) * chan_scale, beta=args.huber_beta).mean() if k > 1 else torch.zeros((), device=x_t_full_n.device)

    target_flat = dyn_target_n.reshape(bsz * k, 3)
    target_lat = model.encode(target_flat).view(bsz, k, -1)
    l_lin = ((pred_lat_seq - target_lat) ** 2).mean()

    x_recon = model.reconstruct_state(model.encode(dyn_t_n))
    l_recon = ((x_recon - dyn_t_n) ** 2).mean()

    # 位姿损失：速度预测经欧拉积分后与 GT 位姿对齐（与 MPC / ONNX rollout 一致）
    l_xy = torch.zeros((), device=x_t_full_n.device)
    l_yaw = torch.zeros((), device=x_t_full_n.device)
    if args.w_xy > 0.0 or args.w_yaw > 0.0:
        pose0 = denorm_pose(x_t_full_n[:, :3], pose_mean, pose_std)
        target_pose = denorm_pose(x_target_seq_n[:, :, :3], pose_mean, pose_std)
        pred_pose = integrate_pose_from_vel(pose0, pred_phys, args.dt)
        if args.w_xy > 0.0:
            err_x = pred_pose[..., 0] - target_pose[..., 0]
            err_y = pred_pose[..., 1] - target_pose[..., 1]
            l_xy = ((err_x * err_x + err_y * err_y) * step_w_pose).mean()
        if args.w_yaw > 0.0:
            err_yaw = wrap_yaw_diff(pred_pose[..., 2], target_pose[..., 2])
            l_yaw = (huber(err_yaw, beta=args.huber_beta) * step_w_pose).mean()

    spec = model.spectral_radius()
    l_stab = torch.relu(spec - args.rho_max) ** 2
    l_l2 = (model.A.weight ** 2).sum() + (model.B.weight ** 2).sum()
    ramp = min(1.0, (epoch + 1) / max(args.ramp_epochs, 1))
    pose_ramp = min(1.0, (epoch + 1) / max(args.pose_ramp_epochs, 1))

    total = (
        args.w_vel * l_vel
        + args.w_acc * l_acc
        + args.w_lin * ramp * l_lin
        + args.w_recon * l_recon
        + args.w_stab * ramp * l_stab
        + args.w_l2 * l_l2
        + args.w_xy * pose_ramp * l_xy
        + args.w_yaw * pose_ramp * l_yaw
    )
    return total, {
        "L_total": float(total.detach().item()),
        "L_vel": float(l_vel.detach().item()),
        "L_acc": float(l_acc.detach().item()),
        "L_lin": float(l_lin.detach().item()),
        "L_recon": float(l_recon.detach().item()),
        "L_xy": float(l_xy.detach().item()),
        "L_yaw": float(l_yaw.detach().item()),
        "L_stab": float(l_stab.detach().item()),
        "spec_radius": float(spec.detach().item()),
    }


class V4LossModule(nn.Module):
    """把整套训练损失包进 ``forward``，供 DDP 正确包装。

    背景：DDP 的梯度 all-reduce 依赖 ``DDP.forward()`` 触发 reducer 注册；
    历史实现直接对 DDP 包装对象调用 ``model.encode(...)`` 等子方法，
    一方面 `DistributedDataParallel` 根本不转发自定义方法（直接
    AttributeError 崩溃），另一方面即使转发也会绕过 reducer 导致梯度
    不同步。因此训练侧统一通过本模块的 ``forward`` 计算损失。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x_t_full_n: torch.Tensor,
        x_target_seq_n: torch.Tensor,
        u_seq_n: torch.Tensor,
        dyn_mean: torch.Tensor,
        dyn_std: torch.Tensor,
        pose_mean: torch.Tensor,
        pose_std: torch.Tensor,
        args: argparse.Namespace,
        epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        return compute_losses(
            self.model, x_t_full_n, x_target_seq_n, u_seq_n,
            dyn_mean, dyn_std, pose_mean, pose_std, args, epoch,
        )


@torch.no_grad()
def quick_validation(
    model: nn.Module,
    eval_dataset: KoopmanVoyageDataset,
    pred_len: int,
    device: torch.device,
    dt: float,
    batch_size: int,
    max_samples: Optional[int],
    args: argparse.Namespace,
    epoch: int,
    dyn_mean: torch.Tensor,
    dyn_std: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    loss_max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """在 hold-out 测试集上评估 rollout 指标与训练同款 loss。

    ``loss_max_samples``: 限制「训练同款 loss」部分的评估样本数（等距子采样，
    与 rollout 子采样口径一致）；None 表示全量（历史行为）。best ckpt 选择只
    依赖 rollout 指标（vel_rmse_mean × instability），不受该参数影响。
    """
    model.eval()

    # 1. Rollout-based metrics (RMSE, instability, etc.)
    states_full = eval_dataset.states_full
    ctrls_full = eval_dataset.ctrls_full
    t0g = eval_dataset.t0_global
    if max_samples is not None and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        model,
        states_full,
        ctrls_full,
        t0g,
        pred_len,
        eval_dataset.stats,
        device,
        dt,
        batch_size,
        model_stride=getattr(eval_dataset, "model_stride", 1),
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)

    results = {
        "test/vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
        "test/slope_loglog": float(div["slope_loglog"]),
        "test/instability_score": float(div["instability_score"]),
        f"test/vel_rmse_step{pred_len}": float(per_step["vel_rmse"][-1]),
        f"test/u_rmse_step{pred_len}": float(per_step["u_rmse"][-1]),
        f"test/v_rmse_step{pred_len}": float(per_step["v_rmse"][-1]),
        f"test/r_rmse_step{pred_len}": float(per_step["r_rmse"][-1]),
        f"test/traj_xy_rmse_step{pred_len}": float(per_step["traj_xy_err"][-1]),
    }

    # 2. Loss-based metrics (same as training losses)
    # 旧实现固定遍历整个 test 集（与 rollout 部分的 max_samples 不一致，
    # 每个 epoch 代价高）；现支持等距子采样。
    if loss_max_samples is not None and 0 <= loss_max_samples < len(eval_dataset):
        sel = np.linspace(0, len(eval_dataset) - 1, loss_max_samples).astype(int)
        loss_dataset: Dataset = Subset(eval_dataset, sel.tolist())
    else:
        loss_dataset = eval_dataset
    eval_loader = DataLoader(
        loss_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    loss_keys = ["L_total", "L_vel", "L_acc", "L_lin", "L_recon", "L_xy", "L_yaw", "L_stab", "spec_radius"]
    eval_losses = {k: 0.0 for k in loss_keys}
    n_batches = 0
    for x_t_full, x_target_seq, u_seq in eval_loader:
        x_t_full = x_t_full.to(device, non_blocking=True)
        x_target_seq = x_target_seq.to(device, non_blocking=True)
        u_seq = u_seq.to(device, non_blocking=True)
        _, info = compute_losses(
            model, x_t_full, x_target_seq, u_seq,
            dyn_mean, dyn_std, pose_mean, pose_std, args, epoch,
        )
        for k in eval_losses:
            eval_losses[k] += info[k]
        n_batches += 1

    for k, v in eval_losses.items():
        results[f"test/{k}"] = v / max(n_batches, 1)

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train_data", type=str, default=str(P.TRAIN_MERGED))
    p.add_argument(
        "--test_data",
        type=str,
        default=str(P.TEST),
        help="每 epoch 验证、best 选择与训练结束评估使用的测试集",
    )
    p.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="已弃用；若指定则覆盖 --test_data（兼容旧配置）",
    )
    p.add_argument("--ckpt_dir", type=str, default=str(P.CKPT_DIR))
    p.add_argument("--log_dir", type=str, default=str(P.LOG_DIR))
    p.add_argument("--run_tag", type=str, default="v4_dict_input")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="每张 GPU 的 micro-batch；DDP 全局 batch = batch_size × world_size × grad_accum_steps",
    )
    p.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数，等效 batch = batch_size × 本值")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--pred_time_sec",
        type=float,
        default=20.0,
        help="目标预测时长 [s]；与 dt 共同决定 pred_len_max（默认 20s @ dt=1.0 -> 20 步）",
    )
    p.add_argument(
        "--pred_time_start_sec",
        type=float,
        default=2.0,
        help="curriculum 起始预测时长 [s]（默认 2s @ dt=1.0 -> 2 步）",
    )
    p.add_argument("--pred_len_start", type=int, default=None, help="覆盖 pred_time_start_sec 的步数")
    p.add_argument("--pred_len_max", type=int, default=None, help="覆盖 pred_time_sec 的步数")
    p.add_argument("--pred_len_step", type=int, default=2, help="curriculum 每阶段增加的预测步数（模型步，1 步=dt 秒）")
    p.add_argument("--pred_len_grow_every", type=int, default=5, help="每 N 个 epoch 增加一次 pred_len")
    p.add_argument("--stride", type=int, default=1, help="窗口起点在原始数据上的步进（非模型下采样）")
    p.add_argument("--dt", type=float, default=1.0, help="模型离散步长 [s]（默认 1.0，预测 20 步=20s）")
    p.add_argument("--data_dt", type=float, default=0.1, help="原始数据集采样间隔 [s]，保持不变")
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_acc", type=float, default=0.2)
    p.add_argument("--w_lin", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.5)
    p.add_argument("--w_xy", type=float, default=2.0, help="位姿平面跟踪损失权重（欧拉积分后 x,y MSE）")
    p.add_argument("--w_yaw", type=float, default=1.0, help="位姿航向损失权重（wrap 后 Huber）")
    p.add_argument("--w_stab", type=float, default=0.1)
    p.add_argument("--w_l2", type=float, default=1e-4)
    p.add_argument("--gamma_step", type=float, default=0.97)
    p.add_argument("--huber_beta", type=float, default=0.1)
    p.add_argument("--rho_max", type=float, default=1.005)
    p.add_argument("--ramp_epochs", type=int, default=5)
    p.add_argument("--pose_ramp_epochs", type=int, default=10, help="位姿损失权重线性 ramp 的 epoch 数")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--no_ema", action="store_true", default=False)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--clamp_pif", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true", default=False,
                   help="启用 CUDA 混合精度训练（与梯度累积兼容；CPU 自动关闭）")
    p.add_argument("--resume", type=str, default=None,
                   help="断点续训：从 ckpt 恢复 model/optimizer/scheduler/EMA/epoch/best")
    p.add_argument(
        "--scheduler",
        type=str,
        default="warmrestart",
        choices=("warmrestart", "cosine", "constant"),
        help="学习率调度：warmrestart=历史默认 CosineAnnealingWarmRestarts(T_0=cos_T0，"
             "T_0=1000 时常规 epoch 数内 LR 几乎恒定)；cosine=CosineAnnealingLR(T_max=epochs)"
             "（推荐）；constant=恒定 LR",
    )
    p.add_argument("--cos_T0", type=int, default=1000,
                   help="warmrestart 调度的 T_0（保持历史默认 1000）")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="早停：best composite 连续 N 个 epoch 无改善则提前结束；0=禁用。"
                        "仅在 curriculum 到达 pred_len_max 后才开始计数。")
    p.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=("nccl", "gloo"),
        help="DDP 通信后端；多 GPU 训练推荐 nccl",
    )
    p.add_argument("--val_max_samples", type=int, default=512)
    p.add_argument("--val_loss_max_samples", type=int, default=None,
                   help="每 epoch 验证「训练同款 loss」部分的最大样本数（等距子采样）；"
                        "默认 None=跟随 --val_max_samples；-1=全量（旧行为，遍历整个 test 集）。"
                        "仅影响日志中 test/L_* 数值口径，不影响 best ckpt 选择。")
    p.add_argument("--val_batch_size", type=int, default=128, help="验证 rollout 的 batch（无梯度，可小于训练 batch）")
    p.add_argument("--smoketest", action="store_true")
    p.add_argument("--config", type=str, default=None)
    return p


def merge_yaml(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: List[str]) -> argparse.Namespace:
    if not args.config:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    explicit = {a.dest for a in parser._actions if any(o in argv for o in (a.option_strings or []))}
    for k, v in cfg.items():
        if k in explicit:
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    return args


def export_params_to_yaml(model: nn.Module, stats: Dict[str, np.ndarray], path: str) -> None:
    data = {
        "normalization": {
            "dyn_mean": np.asarray(stats["state_mean"][3:6], dtype=float).tolist(),
            "dyn_std": np.asarray(stats["state_std"][3:6], dtype=float).tolist(),
            "ctrl_mean": np.asarray(stats["ctrl_mean"], dtype=float).tolist(),
            "ctrl_std": np.asarray(stats["ctrl_std"], dtype=float).tolist(),
        },
        "system_matrices": {
            "A_weight": (model.A.weight.detach().cpu() + torch.eye(model.A.weight.shape[0])).numpy().tolist(),
            "A_bias": model.A.bias.detach().cpu().numpy().tolist(),
            "B": model.B.weight.detach().cpu().numpy().tolist(),
        },
        "dictionary": {
            "atoms_16": list(FEATURE_DICT_ATOMS_16),
            "latent_dim": int(getattr(model, "latent_dim")),
            "hidden_dim": int(getattr(model, "hidden_dim")),
            "input_mode": "dict16_only",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, indent=4, allow_unicode=False)


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = float("inf")


def resolve_model_timing(args: argparse.Namespace) -> argparse.Namespace:
    args.model_stride = ek.model_stride_from_dt(args.dt, args.data_dt)
    return args


def make_dataloaders(
    args: argparse.Namespace,
    pred_len: int,
    train_stats: Optional[Dict[str, np.ndarray]],
    dinfo: DistInfo,
) -> Tuple[KoopmanVoyageDataset, KoopmanVoyageDataset, DataLoader, Optional[DistributedSampler], Dict[str, np.ndarray]]:
    train_ds = KoopmanVoyageDataset(
        args.train_data,
        pred_len=pred_len,
        stride=args.stride,
        stats=train_stats,
        model_stride=args.model_stride,
        data_dt=args.data_dt,
    )
    stats = train_ds.stats
    test_ds = KoopmanVoyageDataset(
        args.test_data,
        pred_len=pred_len,
        stride=args.stride,
        stats=stats,
        model_stride=args.model_stride,
        data_dt=args.data_dt,
    )
    train_sampler: Optional[DistributedSampler] = None
    if dinfo.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dinfo.world_size,
            rank=dinfo.rank,
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=dinfo.enabled,
    )
    return train_ds, test_ds, train_loader, train_sampler, stats


def resolve_device(args: argparse.Namespace, dinfo: DistInfo) -> torch.device:
    if dinfo.enabled:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{dinfo.local_rank}")
        return torch.device("cpu")  # gloo CPU DDP
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def train(args: argparse.Namespace, dinfo: DistInfo) -> None:
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main_process(dinfo) else ""
    run_ts = broadcast_run_id(dinfo, run_ts)
    logger, ts = setup_logger(args.log_dir, dinfo, ts=run_ts)
    seed_everything(args.seed + dinfo.rank)
    device = resolve_device(args, dinfo)
    tb = None
    if is_main_process(dinfo):
        tb = SummaryWriter(log_dir=os.path.join(args.log_dir, f"tb_v4_{ts}"))

    args.ckpt_dir = os.path.join(args.ckpt_dir, f"run_v4_{ts}")
    if is_main_process(dinfo):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    if dinfo.enabled:
        dist.barrier()
    metrics_jsonl = os.path.join(args.log_dir, f"metrics_v4_{ts}.jsonl")

    pred_len = curriculum_pred_len(0, args)
    train_ds, test_ds, train_loader, train_sampler, stats = make_dataloaders(args, pred_len, None, dinfo)
    min_seg = int(train_ds.seg_lens.min()) if train_ds.seg_lens.size else 0
    if min_seg <= ek.data_span_for_pred_len(args.pred_len_max, args.model_stride):
        logger.warning(
            "最短航段长度=%d <= data_span=%d（pred_len=%d, model_stride=%d, %.1fs）；"
            "部分段将被跳过，请检查数据或降低 pred_time_sec",
            min_seg,
            ek.data_span_for_pred_len(args.pred_len_max, args.model_stride),
            args.pred_len_max,
            args.model_stride,
            args.pred_len_max * args.dt,
        )
    base_model = HorizontalKoopmanModelV4DictInput(hidden_dim=args.hidden_dim, clamp_pif=args.clamp_pif).to(device)
    base_model._self_check_dict()
    # 训练侧统一走 V4LossModule.forward（DDP 梯度同步的正确姿势，见类注释）
    train_module: nn.Module = V4LossModule(base_model)
    if dinfo.enabled:
        if device.type == "cuda":
            train_module = DDP(
                train_module,
                device_ids=[dinfo.local_rank],
                output_device=dinfo.local_rank,
                find_unused_parameters=False,
            )
        else:  # gloo CPU DDP：不能传 device_ids
            train_module = DDP(train_module, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched_kind = str(getattr(args, "scheduler", "warmrestart")).lower()
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))
    elif sched_kind == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    else:  # warmrestart：历史默认（T_0=1000 时常规 epoch 数内 LR 几乎恒定）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(int(getattr(args, "cos_T0", 1000)), 1), T_mult=1
        )
    ema = None if args.no_ema else ModelEMA(base_model, decay=args.ema_decay)

    # AMP（默认关闭；仅 CUDA 生效）
    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    if getattr(args, "amp", False) and not use_amp:
        logger.info("AMP 仅在 CUDA 设备上生效，当前 device=%s，已自动关闭。", device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    state = TrainState()

    # ---- 断点续训 ----
    if getattr(args, "resume", None):
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume ckpt 不存在: {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        base_model.load_state_dict(ck["model_state_dict"])
        if ck.get("optimizer_state_dict"):
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(ck["scheduler_state_dict"])
            except Exception as e:  # 调度器类型改变时跳过（例如换了 --scheduler）
                logger.warning("scheduler state 加载失败（%s），将从当前 epoch 重新初始化调度。", e)
        if ema is not None and ck.get("ema_state_dict"):
            ema.module.load_state_dict(ck["ema_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        state.epoch = int(ck.get("epoch", 0))
        state.best_metric = float(ck.get("best_metric", float("inf")))
        logger.info(
            "Resumed from %s @ epoch=%d best=%.6g", args.resume, state.epoch, state.best_metric
        )

    # 验证 loss 部分的子采样上限（None=跟随 val_max_samples；-1=全量）
    val_loss_max = getattr(args, "val_loss_max_samples", None)
    if val_loss_max is None:
        val_loss_max = args.val_max_samples
    elif int(val_loss_max) < 0:
        val_loss_max = None

    dyn_mean_t = torch.tensor(stats["state_mean"][3:6], device=device, dtype=torch.float32)
    dyn_std_t = torch.tensor(stats["state_std"][3:6], device=device, dtype=torch.float32)
    pose_mean_t = torch.tensor(stats["state_mean"][:3], device=device, dtype=torch.float32)
    pose_std_t = torch.tensor(stats["state_std"][:3], device=device, dtype=torch.float32)
    eff_batch = args.batch_size * max(int(args.grad_accum_steps), 1)
    global_batch = eff_batch * dinfo.world_size
    logger.info(
        "Start v4 dict-input training | device=%s | ddp=%s rank=%d/%d | latent=%d | hidden=%d | atoms=%d | "
        "pred_time %.1fs->%.1fs (%d->%d steps, dt=%.3f, data_dt=%.3f, model_stride=%d) curriculum step=%d every=%d epoch | "
        "batch=%d accum=%d (per_gpu=%d global=%d) eval_batch=%d | w_xy=%.2f w_yaw=%.2f pose_ramp=%d | "
        "train=%s test=%s",
        device,
        dinfo.enabled,
        dinfo.rank,
        dinfo.world_size,
        base_model.latent_dim,
        base_model.hidden_dim,
        len(FEATURE_DICT_ATOMS_16),
        args.pred_len_start * args.dt,
        args.pred_len_max * args.dt,
        args.pred_len_start,
        args.pred_len_max,
        args.dt,
        args.data_dt,
        args.model_stride,
        args.pred_len_step,
        args.pred_len_grow_every,
        args.batch_size,
        args.grad_accum_steps,
        eff_batch,
        global_batch,
        args.val_batch_size,
        args.w_xy,
        args.w_yaw,
        args.pose_ramp_epochs,
        args.train_data,
        args.test_data,
    )

    es_no_improve = 0  # 早停计数（仅 pred_len 到达 max 后计数）
    for epoch in range(state.epoch, args.epochs):
        target_pl = curriculum_pred_len(epoch, args)
        if target_pl != pred_len:
            pred_len = target_pl
            train_ds, test_ds, train_loader, train_sampler, _ = make_dataloaders(args, pred_len, stats, dinfo)
            logger.info(
                "[Curriculum] pred_len -> %d (%.1fs)",
                pred_len,
                pred_len * args.dt,
            )

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_module.train()
        t0 = time.time()
        ep = {"L_total": 0.0, "L_vel": 0.0, "L_acc": 0.0, "L_lin": 0.0, "L_recon": 0.0, "L_xy": 0.0, "L_yaw": 0.0, "L_stab": 0.0, "spec_radius": 0.0}
        n_batches = 0
        accum = max(int(args.grad_accum_steps), 1)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_t_full, x_target_seq, u_seq) in enumerate(train_loader):
            x_t_full = x_t_full.to(device, non_blocking=True)
            x_target_seq = x_target_seq.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            step_now = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == len(train_loader))
            # DDP + 梯度累积：非步进 micro-batch 用 no_sync() 跳过 all-reduce，
            # 仅在真正 optimizer.step() 前同步一次（梯度数值不变，省通信）。
            sync_ctx = (
                train_module.no_sync() if (dinfo.enabled and not step_now) else contextlib.nullcontext()
            )
            with sync_ctx:
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss, info = train_module(
                            x_t_full, x_target_seq, u_seq,
                            dyn_mean_t, dyn_std_t, pose_mean_t, pose_std_t, args, epoch,
                        )
                    scaler.scale(loss / accum).backward()
                else:
                    loss, info = train_module(
                        x_t_full, x_target_seq, u_seq,
                        dyn_mean_t, dyn_std_t, pose_mean_t, pose_std_t, args, epoch,
                    )
                    (loss / accum).backward()
            if step_now:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if ema is not None:
                    ema.update(base_model)
                optimizer.zero_grad(set_to_none=True)
            for k in ep:
                ep[k] += info[k]
            n_batches += 1

        scheduler.step()
        train_time = time.time() - t0  # 纯训练耗时（不含验证），用于吞吐统计
        for k in ep:
            ep[k] /= max(n_batches, 1)

        eval_model = ema.module if ema is not None else base_model
        vm: Dict[str, float] = {}
        if is_main_process(dinfo):
            vm = quick_validation(
                eval_model,
                test_ds,
                pred_len,
                device,
                args.dt,
                batch_size=args.val_batch_size,
                max_samples=args.val_max_samples,
                args=args,
                epoch=epoch,
                dyn_mean=dyn_mean_t,
                dyn_std=dyn_std_t,
                pose_mean=pose_mean_t,
                pose_std=pose_std_t,
                loss_max_samples=val_loss_max,
            )
        if dinfo.enabled:
            dist.barrier()

        cur_metric = vm.get("test/vel_rmse_mean", float("inf")) * max(
            1.0, vm.get("test/instability_score", 1.0)
        )
        elapsed = time.time() - t0

        if is_main_process(dinfo):
            for k, v in ep.items():
                tb.add_scalar(f"Train/{k}", v, epoch)
            for k, v in vm.items():
                tb.add_scalar(k, v, epoch)
            tb.add_scalar("Train/pred_len", pred_len, epoch)
            tb.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

            rec = {"epoch": epoch + 1, "pred_len": pred_len, "elapsed_sec": elapsed, **ep, **vm}
            with open(metrics_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            logger.info(
                "Epoch [%03d/%d] pl=%d Ltot=%.4f Lvel=%.4f Lxy=%.4f Lyaw=%.4f Lrecon=%.4f | "
                "test_vel_mean=%.5f test_xy@K=%.5f test_vel@K=%.5f slope=%.3f inst=%.3f | "
                "%.1fs (%.0f samp/s)",
                epoch + 1,
                args.epochs,
                pred_len,
                ep["L_total"],
                ep["L_vel"],
                ep["L_xy"],
                ep["L_yaw"],
                ep["L_recon"],
                vm.get("test/vel_rmse_mean", float("nan")),
                vm.get(f"test/traj_xy_rmse_step{pred_len}", float("nan")),
                vm.get(f"test/vel_rmse_step{pred_len}", float("nan")),
                vm.get("test/slope_loglog", float("nan")),
                vm.get("test/instability_score", float("nan")),
                elapsed,
                n_batches * args.batch_size * max(dinfo.world_size, 1) / max(train_time, 1e-9),
            )

            ckpt = {
                "epoch": epoch + 1,
                "model_state_dict": base_model.state_dict(),
                "stats": stats,
                "best_metric": state.best_metric,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "ema_state_dict": ema.state_dict() if ema is not None else None,
                "args": vars(args),
                "model_class": base_model.__class__.__name__,
                "feature_dict_atoms": list(FEATURE_DICT_ATOMS_16),
                "latent_dim": int(base_model.latent_dim),
            }
            torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_v4_latest.pth"))

            # 最后30步迭代，每5步保存一次模型
            if epoch >= args.epochs - 30 and (epoch + 1) % 5 == 0:
                epoch_ckpt_path = os.path.join(args.ckpt_dir, f"koopman_v4_epoch{epoch+1}.pth")
                torch.save(ckpt, epoch_ckpt_path)
                logger.info("Saved periodic checkpoint: %s", epoch_ckpt_path)

            improved = bool(vm) and cur_metric < state.best_metric
            if improved:
                state.best_metric = cur_metric
                ckpt["best_metric"] = state.best_metric
                torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_v4_best.pth"))
                logger.info("  ↳ new best composite=%.6g @ epoch %d", cur_metric, epoch + 1)

        # ---- 早停（--early_stop_patience > 0 时启用；rank0 判定后广播）----
        stop_flag = False
        if is_main_process(dinfo) and int(getattr(args, "early_stop_patience", 0)) > 0:
            if pred_len >= args.pred_len_max:
                es_no_improve = 0 if improved else es_no_improve + 1
                if es_no_improve >= int(args.early_stop_patience):
                    logger.info(
                        "[EarlyStop] best composite 连续 %d 个 epoch 无改善"
                        "（patience=%d），在 epoch %d 提前停止训练。",
                        es_no_improve, int(args.early_stop_patience), epoch + 1,
                    )
                    stop_flag = True
            else:
                es_no_improve = 0  # curriculum 未到 max，不计数
        if dinfo.enabled:
            payload = [stop_flag]
            dist.broadcast_object_list(payload, src=0)
            stop_flag = bool(payload[0])
        if stop_flag:
            break

    if dinfo.enabled:
        dist.barrier()

    if not is_main_process(dinfo):
        return

    best_path = os.path.join(args.ckpt_dir, "koopman_v4_best.pth")
    if os.path.exists(best_path):
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        sd = best.get("ema_state_dict") or best["model_state_dict"]
        export_model = HorizontalKoopmanModelV4DictInput(hidden_dim=args.hidden_dim, clamp_pif=args.clamp_pif)
        export_model.load_state_dict(sd)
        export_yaml = os.path.join(args.ckpt_dir, "koopman_v4_best.yaml")
        export_params_to_yaml(export_model, best["stats"], export_yaml)
        logger.info("Exported YAML -> %s", export_yaml)

        logger.info("Running evaluation on test set using best checkpoint...")
        try:
            eval_out_dir = os.path.join(args.log_dir, f"eval_v4_{ts}")
            os.makedirs(eval_out_dir, exist_ok=True)
            
            per_step, div, summary, gt_dyn, pred_dyn = evaluate(
                ckpt_path=best_path,
                data_path=str(args.test_data),
                pred_len=args.pred_len_max,
                dt=args.dt,
                batch_size=args.batch_size,
                device=device,
                max_samples=None,
                model_stride=args.model_stride,
                data_dt=args.data_dt,
            )
            
            tag = args.run_tag
            per_step_csv = os.path.join(eval_out_dir, f"{tag}_per_step_metrics.csv")
            summary_json = os.path.join(eval_out_dir, f"{tag}_summary.json")
            u_rmse_png = os.path.join(eval_out_dir, f"{tag}_u_rmse_vs_step.png")
            v_rmse_png = os.path.join(eval_out_dir, f"{tag}_v_rmse_vs_step.png")
            u_scatter_png = os.path.join(eval_out_dir, f"{tag}_u_scatter_compare.png")
            v_scatter_png = os.path.join(eval_out_dir, f"{tag}_v_scatter_compare.png")
            u_curve_png = os.path.join(eval_out_dir, f"{tag}_u_curve_compare.png")
            v_curve_png = os.path.join(eval_out_dir, f"{tag}_v_curve_compare.png")
            
            write_per_step_csv(per_step, per_step_csv)
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            plot_channel_rmse_vs_step(per_step, "u", u_rmse_png, tag)
            plot_channel_rmse_vs_step(per_step, "v", v_rmse_png, tag)
            plot_channel_scatter(gt_dyn, pred_dyn, "u", u_scatter_png, tag)
            plot_channel_scatter(gt_dyn, pred_dyn, "v", v_scatter_png, tag)
            plot_channel_sample_curves(gt_dyn, pred_dyn, "u", u_curve_png, tag)
            plot_channel_sample_curves(gt_dyn, pred_dyn, "v", v_curve_png, tag)
            
            logger.info("Evaluation completed. Plots and metrics saved to %s", eval_out_dir)

            # 将本次训练保存的所有模型作图对比 (调用 compare_mpc_tracking.py)
            compare_models = []
            for e in range(max(0, args.epochs - 30), args.epochs):
                if (e + 1) % 5 == 0:
                    ep_path = os.path.join(args.ckpt_dir, f"koopman_v4_epoch{e+1}.pth")
                    if os.path.exists(ep_path):
                        compare_models.append(f"{ep_path}:ep{e+1}")
            if os.path.exists(best_path):
                compare_models.append(f"{best_path}:best")

            if len(compare_models) > 1:
                logger.info("Running MPC tracking comparison on all saved models...")
                import subprocess
                cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "compare_mpc_tracking.py"),
                    "--models",
                ] + compare_models + [
                    "--data", str(args.test_data),
                    "--out_dir", os.path.join(eval_out_dir, "mpc_compare"),
                    "--horizon", str(args.pred_len_max),
                    "--dt", str(args.dt),
                    "--data_dt", str(args.data_dt),
                    "--opt_iters", "8",
                    "--opt_control_steps", "2",
                    "--steps", str(min(args.pred_len_max * 4, 80)),
                ]
                subprocess.run(cmd, check=True)
                logger.info("MPC tracking comparison plot saved to %s", os.path.join(eval_out_dir, "mpc_compare"))
        except Exception as e:
            logger.error("Test evaluation or MPC tracking failed: %s", e)

    if tb is not None:
        tb.close()


def run_smoketest(args: argparse.Namespace, dinfo: DistInfo) -> int:
    out_dir = Path("logs/smoketest_v4_dict_input")
    mini_train = out_dir / "mini_train.npz"
    mini_test = out_dir / "mini_test.npz"
    # DDP 下仅 rank0 写迷你数据集，避免多进程同写同一文件
    if is_main_process(dinfo):
        raw = np.load(args.train_data, allow_pickle=True)["datas"]
        raw_t = np.load(args.test_data, allow_pickle=True)["datas"]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(mini_train, datas=np.array([raw[0], raw[1]], dtype=object))
        np.savez(mini_test, datas=np.array([raw_t[0]], dtype=object))
    if dinfo.enabled:
        dist.barrier()

    args.train_data = str(mini_train)
    args.test_data = str(mini_test)
    args.epochs = 2
    args.batch_size = 16
    args.pred_len_start = 4
    args.pred_len_max = 4
    args.pred_len_grow_every = 999
    args.num_workers = 0
    args.ckpt_dir = str(out_dir / "ckpt")
    args.log_dir = str(out_dir / "log")
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    train(args, dinfo)
    if not is_main_process(dinfo):
        return 0
    for fn in ("koopman_v4_latest.pth", "koopman_v4_best.pth", "koopman_v4_best.yaml"):
        p = Path(args.ckpt_dir) / fn
        if not p.exists():
            raise RuntimeError(f"smoketest missing artifact: {p}")
    print("[smoketest] v4 dict-input OK")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args = merge_yaml(args, parser, sys.argv[1:])
    if args.pred_len_max is None:
        args.pred_len_max = steps_from_seconds(args.pred_time_sec, args.dt)
    if args.pred_len_start is None:
        args.pred_len_start = steps_from_seconds(args.pred_time_start_sec, args.dt)
    args = resolve_prediction_horizon(args)
    args = resolve_model_timing(args)
    if args.val_data is not None:
        args.test_data = args.val_data

    dinfo = init_distributed(backend=args.dist_backend)
    try:
        if args.smoketest:
            return run_smoketest(args, dinfo)
        train(args, dinfo)
        return 0
    finally:
        cleanup_distributed(dinfo)


if __name__ == "__main__":
    raise SystemExit(main())
