"""Train script for v4 dict-input Koopman model.

特点：
- 参考 scripts/train_v2.py 的训练流程（dataset/curriculum/EMA/多项损失）。
- 使用 new_v4_dict_input.model_v4_dict_input.HorizontalKoopmanModelV4DictInput。
- 保留原有脚本不变；本脚本完全独立。
"""
from __future__ import annotations

import argparse
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
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
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


def setup_logger(log_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(f"KoopmanV4_{ts}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(log_dir, f"train_v4_{ts}.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, ts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KoopmanVoyageDataset(Dataset):
    """与 train_v2 同口径的数据集：返回规范化后的 x_t/x_seq/u_seq。"""

    def __init__(
        self,
        npz_path: str,
        pred_len: int,
        stride: int = 1,
        stats: Optional[Dict[str, np.ndarray]] = None,
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
        x_t = self.states_full[t0]
        x_seq = self.states_full[t0 + 1 : t0 + 1 + k]
        u_seq = self.ctrls_full[t0 : t0 + k]
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

    spec = model.spectral_radius()
    l_stab = torch.relu(spec - args.rho_max) ** 2
    l_l2 = (model.A.weight ** 2).sum() + (model.B.weight ** 2).sum()
    ramp = min(1.0, (epoch + 1) / max(args.ramp_epochs, 1))

    total = (
        args.w_vel * l_vel
        + args.w_acc * l_acc
        + args.w_lin * ramp * l_lin
        + args.w_recon * l_recon
        + args.w_stab * ramp * l_stab
        + args.w_l2 * l_l2
    )
    return total, {
        "L_total": float(total.detach().item()),     # 总加权损失
        "L_vel": float(l_vel.detach().item()),       # 速度预测损失 (物理空间上速度的Huber误差)
        "L_acc": float(l_acc.detach().item()),       # 加速度预测损失 (相邻两步速度差分的误差)
        "L_lin": float(l_lin.detach().item()),       # 隐空间线性一致性损失 (递推隐变量与GT真实序列编码的误差)
        "L_recon": float(l_recon.detach().item()),   # 状态重构损失 (单步自编码器对输入的重构误差)
        "L_stab": float(l_stab.detach().item()),     # 稳定性惩罚损失 (惩罚Koopman矩阵A谱半径越限的部分)
        "spec_radius": float(spec.detach().item()),  # 系统矩阵A当前的谱半径 (监控指标)
    }


@torch.no_grad()
def quick_validation(
    model: nn.Module,
    val_dataset: KoopmanVoyageDataset,
    pred_len: int,
    device: torch.device,
    dt: float,
    batch_size: int,
    max_samples: Optional[int],
    args: argparse.Namespace,
    epoch: int,
    dyn_mean: torch.Tensor,
    dyn_std: torch.Tensor,
) -> Dict[str, float]:
    model.eval()

    # 1. Rollout-based metrics (RMSE, instability, etc.)
    states_full = val_dataset.states_full
    ctrls_full = val_dataset.ctrls_full
    t0g = val_dataset.t0_global
    if max_samples is not None and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        model, states_full, ctrls_full, t0g, pred_len, val_dataset.stats, device, dt, batch_size
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)

    results = {
        "val/vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
        "val/slope_loglog": float(div["slope_loglog"]),
        "val/instability_score": float(div["instability_score"]),
        f"val/vel_rmse_step{pred_len}": float(per_step["vel_rmse"][-1]),
        f"val/u_rmse_step{pred_len}": float(per_step["u_rmse"][-1]),
        f"val/v_rmse_step{pred_len}": float(per_step["v_rmse"][-1]),
        f"val/r_rmse_step{pred_len}": float(per_step["r_rmse"][-1]),
    }

    # 2. Loss-based metrics (same as training losses)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
    )
    loss_keys = ["L_total", "L_vel", "L_acc", "L_lin", "L_recon", "L_stab", "spec_radius"]
    val_losses = {k: 0.0 for k in loss_keys}
    n_batches = 0
    for x_t_full, x_target_seq, u_seq in val_loader:
        x_t_full = x_t_full.to(device, non_blocking=True)
        x_target_seq = x_target_seq.to(device, non_blocking=True)
        u_seq = u_seq.to(device, non_blocking=True)
        _, info = compute_losses(model, x_t_full, x_target_seq, u_seq, dyn_mean, dyn_std, args, epoch)
        for k in val_losses:
            val_losses[k] += info[k]
        n_batches += 1

    for k, v in val_losses.items():
        results[f"val/{k}"] = v / max(n_batches, 1)

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train_data", type=str, default=str(P.TRAIN_MERGED))
    p.add_argument("--val_data", type=str, default=str(P.VAL))
    p.add_argument("--ckpt_dir", type=str, default=str(P.CKPT_DIR))
    p.add_argument("--log_dir", type=str, default=str(P.LOG_DIR))
    p.add_argument("--run_tag", type=str, default="v4_dict_input")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=512, help="单步 micro-batch；8GB 显存 + 20s 预测建议 384~512")
    p.add_argument("--grad_accum_steps", type=int, default=2, help="梯度累积步数，等效 batch = batch_size × 本值")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--pred_time_sec",
        type=float,
        default=20.0,
        help="目标预测时长 [s]；与 dt 共同决定 pred_len_max（默认 20s -> 200 步）",
    )
    p.add_argument(
        "--pred_time_start_sec",
        type=float,
        default=2.0,
        help="curriculum 起始预测时长 [s]（默认 2s -> 20 步）",
    )
    p.add_argument("--pred_len_start", type=int, default=None, help="覆盖 pred_time_start_sec 的步数")
    p.add_argument("--pred_len_max", type=int, default=None, help="覆盖 pred_time_sec 的步数")
    p.add_argument("--pred_len_step", type=int, default=10, help="curriculum 每阶段增加的预测步数")
    p.add_argument("--pred_len_grow_every", type=int, default=5, help="每 N 个 epoch 增加一次 pred_len")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--w_vel", type=float, default=1.0)
    p.add_argument("--w_acc", type=float, default=0.2)
    p.add_argument("--w_lin", type=float, default=1.0)
    p.add_argument("--w_recon", type=float, default=0.5)
    p.add_argument("--w_stab", type=float, default=0.1)
    p.add_argument("--w_l2", type=float, default=1e-4)
    p.add_argument("--gamma_step", type=float, default=0.97)
    p.add_argument("--huber_beta", type=float, default=0.1)
    p.add_argument("--rho_max", type=float, default=1.005)
    p.add_argument("--ramp_epochs", type=int, default=5)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--no_ema", action="store_true", default=False)
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--clamp_pif", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--val_max_samples", type=int, default=512)
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


def make_dataloaders(
    args: argparse.Namespace,
    pred_len: int,
    train_stats: Optional[Dict[str, np.ndarray]],
) -> Tuple[KoopmanVoyageDataset, KoopmanVoyageDataset, DataLoader, Dict[str, np.ndarray]]:
    train_ds = KoopmanVoyageDataset(args.train_data, pred_len=pred_len, stride=args.stride, stats=train_stats)
    stats = train_ds.stats
    val_ds = KoopmanVoyageDataset(args.val_data, pred_len=pred_len, stride=args.stride, stats=stats)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    return train_ds, val_ds, train_loader, stats


def train(args: argparse.Namespace) -> None:
    logger, ts = setup_logger(args.log_dir)
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    tb = SummaryWriter(log_dir=os.path.join(args.log_dir, f"tb_v4_{ts}"))
    
    # 将本次训练的所有模型单独保存在一个以时间戳命名的子文件夹中
    args.ckpt_dir = os.path.join(args.ckpt_dir, f"run_v4_{ts}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_jsonl = os.path.join(args.log_dir, f"metrics_v4_{ts}.jsonl")

    pred_len = curriculum_pred_len(0, args)
    train_ds, val_ds, train_loader, stats = make_dataloaders(args, pred_len, None)
    min_seg = int(train_ds.seg_lens.min()) if train_ds.seg_lens.size else 0
    if min_seg <= args.pred_len_max:
        logger.warning(
            "最短航段长度=%d <= pred_len_max=%d（%.1fs）；部分段将被跳过，请检查数据或降低 pred_time_sec",
            min_seg,
            args.pred_len_max,
            args.pred_len_max * args.dt,
        )
    model = HorizontalKoopmanModelV4DictInput(hidden_dim=args.hidden_dim, clamp_pif=args.clamp_pif).to(device)
    model._self_check_dict()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1)
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    state = TrainState()

    dyn_mean_t = torch.tensor(stats["state_mean"][3:6], device=device, dtype=torch.float32)
    dyn_std_t = torch.tensor(stats["state_std"][3:6], device=device, dtype=torch.float32)
    logger.info(
        "Start v4 dict-input training | device=%s | latent=%d | hidden=%d | atoms=%d | "
        "pred_time %.1fs->%.1fs (%d->%d steps, dt=%.3f) curriculum step=%d every=%d epoch | "
        "batch=%d accum=%d (effective=%d) val_batch=%d",
        device,
        model.latent_dim,
        model.hidden_dim,
        len(FEATURE_DICT_ATOMS_16),
        args.pred_len_start * args.dt,
        args.pred_len_max * args.dt,
        args.pred_len_start,
        args.pred_len_max,
        args.dt,
        args.pred_len_step,
        args.pred_len_grow_every,
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * max(int(args.grad_accum_steps), 1),
        args.val_batch_size,
    )

    for epoch in range(state.epoch, args.epochs):
        target_pl = curriculum_pred_len(epoch, args)
        if target_pl != pred_len:
            pred_len = target_pl
            train_ds, val_ds, train_loader, _ = make_dataloaders(args, pred_len, stats)
            logger.info(
                "[Curriculum] pred_len -> %d (%.1fs)",
                pred_len,
                pred_len * args.dt,
            )

        model.train()
        t0 = time.time()
        ep = {"L_total": 0.0, "L_vel": 0.0, "L_acc": 0.0, "L_lin": 0.0, "L_recon": 0.0, "L_stab": 0.0, "spec_radius": 0.0}
        n_batches = 0
        accum = max(int(args.grad_accum_steps), 1)
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_t_full, x_target_seq, u_seq) in enumerate(train_loader):
            x_t_full = x_t_full.to(device, non_blocking=True)
            x_target_seq = x_target_seq.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            loss, info = compute_losses(model, x_t_full, x_target_seq, u_seq, dyn_mean_t, dyn_std_t, args, epoch)
            (loss / accum).backward()
            step_now = ((batch_idx + 1) % accum == 0) or (batch_idx + 1 == len(train_loader))
            if step_now:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
            for k in ep:
                ep[k] += info[k]
            n_batches += 1

        scheduler.step()
        for k in ep:
            ep[k] /= max(n_batches, 1)

        eval_model = ema.module if ema is not None else model
        vm = quick_validation(
            eval_model,
            val_ds,
            pred_len,
            device,
            args.dt,
            batch_size=args.val_batch_size,
            max_samples=args.val_max_samples,
            args=args,
            epoch=epoch,
            dyn_mean=dyn_mean_t,
            dyn_std=dyn_std_t,
        )
        cur_metric = vm["val/vel_rmse_mean"] * max(1.0, vm["val/instability_score"])
        elapsed = time.time() - t0

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
            "Epoch [%03d/%d] pl=%d Ltot=%.4f Lvel=%.4f Lrecon=%.4f | val_vel_mean=%.5f val_vel@K=%.5f slope=%.3f inst=%.3f | %.1fs",
            epoch + 1,                           # 当前训练轮次
            args.epochs,                         # 总训练轮次
            pred_len,                            # 当前预测步长 (Prediction length)
            ep["L_total"],                       # 总损失 (Total loss)
            ep["L_vel"],                         # 速度预测损失 (Velocity prediction loss)
            ep["L_recon"],                       # 状态重构损失 (State reconstruction loss)
            vm["val/vel_rmse_mean"],             # 验证集平均速度均方根误差
            vm[f"val/vel_rmse_step{pred_len}"],  # 验证集预测最后一步的速度均方根误差
            vm["val/slope_loglog"],              # 验证集误差发散斜率 (衡量随步长误差增长速度)
            vm["val/instability_score"],         # 验证集不稳定性得分 (衡量发散和轨迹抖动情况)
            elapsed,                             # 当前轮次耗时（秒）
        )

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "stats": stats,
            "best_metric": state.best_metric,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "args": vars(args),
            "model_class": model.__class__.__name__,
            "feature_dict_atoms": list(FEATURE_DICT_ATOMS_16),
            "latent_dim": int(model.latent_dim),
        }
        torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_v4_latest.pth"))

        # 最后30步迭代，每5步保存一次模型
        if epoch >= args.epochs - 30 and (epoch + 1) % 5 == 0:
            epoch_ckpt_path = os.path.join(args.ckpt_dir, f"koopman_v4_epoch{epoch+1}.pth")
            torch.save(ckpt, epoch_ckpt_path)
            logger.info("Saved periodic checkpoint: %s", epoch_ckpt_path)

        if cur_metric < state.best_metric:
            state.best_metric = cur_metric
            ckpt["best_metric"] = state.best_metric
            torch.save(ckpt, os.path.join(args.ckpt_dir, "koopman_v4_best.pth"))
            logger.info("  ↳ new best composite=%.6g @ epoch %d", cur_metric, epoch + 1)

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
                data_path=str(P.TEST),
                pred_len=args.pred_len_max,
                dt=args.dt,
                batch_size=args.batch_size,
                device=device,
                max_samples=None,
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
                    "--models"
                ] + compare_models + [
                    "--data", str(P.TEST),
                    "--out_dir", os.path.join(eval_out_dir, "mpc_compare")
                ]
                subprocess.run(cmd, check=True)
                logger.info("MPC tracking comparison plot saved to %s", os.path.join(eval_out_dir, "mpc_compare"))
        except Exception as e:
            logger.error("Test evaluation or MPC tracking failed: %s", e)


def run_smoketest(args: argparse.Namespace) -> int:
    raw = np.load(args.train_data, allow_pickle=True)["datas"]
    raw_v = np.load(args.val_data, allow_pickle=True)["datas"]
    out_dir = Path("logs/smoketest_v4_dict_input")
    out_dir.mkdir(parents=True, exist_ok=True)
    mini_train = out_dir / "mini_train.npz"
    mini_val = out_dir / "mini_val.npz"
    np.savez(mini_train, datas=np.array([raw[0], raw[1]], dtype=object))
    np.savez(mini_val, datas=np.array([raw_v[0]], dtype=object))

    args.train_data = str(mini_train)
    args.val_data = str(mini_val)
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

    train(args)
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
    if args.smoketest:
        return run_smoketest(args)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
