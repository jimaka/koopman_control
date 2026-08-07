#!/usr/bin/env python3
"""阶段 2（M2）：MMG + 残差 MLP 训练脚本。

对应 docs/MMG残差MLP建模技术方案.md §4。训练骨架与 v4 同风格：
课程式 pred_len、EMA、Huber（std 归一化误差）、按验证集 vel_rmse_mean 存 best。

模型:  dyn_{t+1} = ∫ [ f_MMG(dyn,cmd; θ冻结) + g_MLP(norm(dyn,cmd)) ] dt
MMG 参数来自 checkpoints/mmg_baseline.npz（scripts/fit_mmg_baseline.py 产物）；
--mmg_params auto 时先现场辨识。

用法:
    python3 scripts/train_mmg_residual.py                       # 正常训练
    python3 scripts/train_mmg_residual.py --smoketest           # 冒烟（秒级）
    python3 scripts/train_mmg_residual.py --finetune_mmg        # 联合微调 MMG 参数（小学习率）
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

import argparse
import copy
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from koopman import evalkit as ek
from koopman import paths as P
from koopman.mmg_model import (
    MmgModel,
    PhysStepAdapter,
    compute_train_stats,
    least_squares_fit,
    load_mmg_npz,
    save_mmg_npz,
)
from koopman.mmg_residual import MmgResidualModel


# ---------------------------------------------------------------------------
# 数据集（物理量窗口；归一化在模型内部完成）
# ---------------------------------------------------------------------------


class MmgWindowDataset(Dataset):
    """返回物理量 x_t (3,) / x_seq (K,3) / u_seq (K,4)，口径与 KoopmanVoyageDataset 一致。"""

    def __init__(self, npz_path: str, pred_len: int, model_stride: int, stride: int = 1,
                 max_samples: Optional[int] = None) -> None:
        super().__init__()
        states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
            npz_path, pred_len=pred_len, stride=stride, model_stride=model_stride
        )
        if max_samples is not None and t0g.shape[0] > max_samples:
            sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
            t0g = t0g[sel]
        self.dyn = np.ascontiguousarray(states_full[:, 3:6])
        self.ctrl = ctrls_full
        self.t0 = t0g
        self.K = int(pred_len)
        self.ms = int(model_stride)

    def __len__(self) -> int:
        return int(self.t0.shape[0])

    def __getitem__(self, index: int):
        t0 = int(self.t0[index])
        k, ms = self.K, self.ms
        x_t = self.dyn[t0].copy()
        x_seq = self.dyn[t0 + ms: t0 + 1 + k * ms: ms].copy()
        u_seq = self.ctrl[t0: t0 + k * ms: ms].copy()
        return (
            torch.from_numpy(x_t),
            torch.from_numpy(x_seq),
            torch.from_numpy(u_seq),
        )


# ---------------------------------------------------------------------------
# 小组件（与 train_v4 同风格）
# ---------------------------------------------------------------------------


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


def rollout_model(model: nn.Module, x_t: torch.Tensor, u_seq: torch.Tensor, dt: float) -> torch.Tensor:
    """x_t (B,3)、u_seq (B,K,4) → pred (B,K,3)（无 teacher forcing）。"""
    x = x_t
    outs: List[torch.Tensor] = []
    for k in range(u_seq.size(1)):
        x = model.step_phys(x, u_seq[:, k, :], dt)
        outs.append(x)
    return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# 验证 / 测试（复用 evalkit，与 v4 同口径）
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_rollout(model: nn.Module, stats: Dict[str, np.ndarray], npz_path: str,
                 K: int, dt: float, data_dt: float, device: torch.device,
                 max_samples: int, batch_size: int = 1024) -> Tuple[Dict[str, np.ndarray], int]:
    ms = ek.model_stride_from_dt(dt, data_dt)
    eval_model = model.eval()
    adapter = PhysStepAdapter(lambda d, c: eval_model.step_phys(d, c, dt), stats).to(device)
    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
        npz_path, pred_len=K, stride=1, model_stride=ms
    )
    if max_samples and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        adapter, states_full, ctrls_full, t0g, K, stats, device, dt,
        batch_size=batch_size, model_stride=ms,
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    return per_step, int(gt_dyn.shape[0])


def fmt_metrics(per_step: Dict[str, np.ndarray]) -> str:
    K = len(per_step["step"]) - 1
    return (f"vel_rmse mean={float(np.mean(per_step['vel_rmse'])):.4f} "
            f"step1={per_step['vel_rmse'][0]:.4f} step{K + 1}={per_step['vel_rmse'][-1]:.4f} "
            f"| u/v/r@{K + 1} = {per_step['u_rmse'][-1]:.4f}/{per_step['v_rmse'][-1]:.4f}/{per_step['r_rmse'][-1]:.4f}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_mmg(args, segments_for_fit, stats) -> Tuple[MmgModel, np.ndarray]:
    if args.mmg_params != "auto" and os.path.exists(args.mmg_params):
        theta, _, _ = load_mmg_npz(args.mmg_params)
        print(f"MMG 参数加载自 {args.mmg_params}")
    else:
        print("MMG 参数现场最小二乘辨识（--mmg_params auto 或文件不存在）")
        theta, report = least_squares_fit(segments_for_fit)
        for ch in ("surge", "sway", "yaw"):
            print(f"  [{ch}] R²={report[ch]['r2']:.4f} rmse={report[ch]['rmse']:.4e}")
        if args.mmg_params != "auto":
            os.makedirs(os.path.dirname(os.path.abspath(args.mmg_params)), exist_ok=True)
            save_mmg_npz(args.mmg_params, theta, stats, report)
            print(f"  已保存: {args.mmg_params}")
    mmg = MmgModel(theta=theta, trainable=args.finetune_mmg)
    return mmg, theta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train_data", type=str, default=str(P.TRAIN_MERGED))
    ap.add_argument("--val_data", type=str, default=str(P.VAL))
    ap.add_argument("--test_data", type=str, default=str(P.TEST))
    ap.add_argument("--mmg_params", type=str, default=str(P.CKPT_DIR / "mmg_baseline.npz"),
                    help="MMG 参数 npz；'auto' = 现场辨识")
    ap.add_argument("--out", type=str, default=str(P.CKPT_DIR / "mmg_residual_best.pth"))
    ap.add_argument("--dt", type=float, default=0.5, help="模型步长 [s]（0.5 对齐实船控制周期）")
    ap.add_argument("--data_dt", type=float, default=0.1)
    ap.add_argument("--pred_time_sec", type=float, default=20.0, help="最终 rollout 时长 [s]")
    ap.add_argument("--pred_time_start_sec", type=float, default=2.0, help="课程起始时长 [s]")
    ap.add_argument("--grow_sec", type=float, default=2.0, help="每次课程增加的时长 [s]")
    ap.add_argument("--grow_every_epochs", type=int, default=3)
    ap.add_argument("--stride", type=int, default=5, help="训练窗口取样间隔（原始行）")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n_blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--mmg_lr_scale", type=float, default=0.01, help="--finetune_mmg 时 MMG 参数的学习率倍率")
    ap.add_argument("--finetune_mmg", action="store_true", help="解冻 MMG 参数联合微调")
    ap.add_argument("--huber_beta", type=float, default=0.1)
    ap.add_argument("--step_gamma", type=float, default=0.95, help="多步损失随步数的折扣")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--max_val_samples", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--smoketest", action="store_true")
    args = ap.parse_args()

    if args.smoketest:
        args.epochs = 2
        args.pred_time_sec = 4.0
        args.pred_time_start_sec = 4.0
        args.max_val_samples = 500

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms = ek.model_stride_from_dt(args.dt, args.data_dt)
    K_max = int(round(args.pred_time_sec / args.dt))
    K_start = max(1, int(round(args.pred_time_start_sec / args.dt)))
    K_grow = max(1, int(round(args.grow_sec / args.dt)))
    print(f"device={device}  dt={args.dt}s (stride={ms})  K: {K_start}→{K_max}")

    # ---- 数据与统计 ----
    states_full, ctrls_full, seg_starts, seg_lens = ek._load_segments_cached(args.train_data)
    segments = [(states_full[s:s + L], ctrls_full[s:s + L])
                for s, L in zip(seg_starts.tolist(), seg_lens.tolist())]
    stats = compute_train_stats(states_full, ctrls_full)
    dyn_std_t = torch.tensor(stats["state_std"][3:6], device=device)

    # ---- 模型 ----
    mmg, theta = build_mmg(args, segments, stats)
    model = MmgResidualModel(mmg, stats, hidden=args.hidden, n_blocks=args.n_blocks,
                             freeze_mmg=not args.finetune_mmg).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {n_train}（hidden={args.hidden}, blocks={args.n_blocks}, finetune_mmg={args.finetune_mmg}）")

    param_groups = [{"params": model.net.parameters(), "lr": args.lr}]
    if args.finetune_mmg:
        param_groups.append({"params": model.mmg.parameters(), "lr": args.lr * args.mmg_lr_scale})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    ema = ModelEMA(model, decay=args.ema_decay)

    def lr_at(epoch: int) -> float:
        t = epoch / max(args.epochs - 1, 1)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + np.cos(np.pi * t))

    # ---- 课程式训练 ----
    cur_K = -1
    loader = None
    best_val = float("inf")
    history: List[Dict] = []
    for epoch in range(args.epochs):
        K = min(K_max, K_start + (epoch // args.grow_every_epochs) * K_grow)
        if K != cur_K:
            cur_K = K
            ds = MmgWindowDataset(args.train_data, pred_len=K, model_stride=ms, stride=args.stride,
                                  max_samples=2000 if args.smoketest else None)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True)
        for g in optimizer.param_groups:
        # MMG 组保持缩放比例
            g["lr"] = lr_at(epoch) * (args.mmg_lr_scale if g is param_groups[-1] and args.finetune_mmg else 1.0)

        model.train()
        t0 = time.time()
        losses: List[float] = []
        step_w = make_step_weights(cur_K, args.step_gamma, device).view(1, cur_K, 1)
        for x_t, x_seq, u_seq in loader:
            x_t = x_t.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)
            u_seq = u_seq.to(device, non_blocking=True)
            pred = rollout_model(model, x_t, u_seq, args.dt)
            err = (pred - x_seq) / dyn_std_t
            loss = (huber(err, args.huber_beta) * step_w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)
            losses.append(float(loss.detach().cpu()))

        # ---- 验证（EMA 权重，K_max 全程）----
        per_step, n_val = eval_rollout(ema.module, stats, args.val_data, K_max, args.dt,
                                       args.data_dt, device, args.max_val_samples)
        val_metric = float(np.mean(per_step["vel_rmse"]))
        lr_now = lr_at(epoch)
        print(f"epoch {epoch + 1:>3d}/{args.epochs}  K={cur_K:>2d}  lr={lr_now:.2e}  "
              f"loss={np.mean(losses):.5f}  |  val(n={n_val}) {fmt_metrics(per_step)}  "
              f"({time.time() - t0:.1f}s)")
        history.append({"epoch": epoch + 1, "K": cur_K, "loss": float(np.mean(losses)),
                        "val_vel_rmse_mean": val_metric})

        if val_metric < best_val:
            best_val = val_metric
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            torch.save({
                "model_class": "MmgResidualModel",
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "stats": stats,
                "mmg_theta": np.asarray(theta, dtype=np.float64),
                "args": vars(args),
                "best_epoch": epoch + 1,
                "best_val_vel_rmse_mean": best_val,
                "history": history,
            }, args.out)
            print(f"    ✓ best 更新: {args.out} (val={best_val:.4f})")

    # ---- 最终测试 ----
    print("\n== 测试集最终评估（best ckpt 的 EMA 权重）==")
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    final_model = MmgResidualModel.load_from_ckpt(ckpt, device)
    per_step, n_test = eval_rollout(final_model, stats, args.test_data, K_max, args.dt,
                                    args.data_dt, device, args.max_val_samples * 2)
    print(f"test(n={n_test}) {fmt_metrics(per_step)}")
    ckpt["test_metrics"] = {
        "n_samples": n_test,
        "vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
        "vel_rmse_step_1": float(per_step["vel_rmse"][0]),
        "vel_rmse_step_K": float(per_step["vel_rmse"][-1]),
        "u_rmse_step_K": float(per_step["u_rmse"][-1]),
        "v_rmse_step_K": float(per_step["v_rmse"][-1]),
        "r_rmse_step_K": float(per_step["r_rmse"][-1]),
    }
    torch.save(ckpt, args.out)
    print(f"完成。best val={best_val:.4f}  ckpt: {args.out}")


if __name__ == "__main__":
    main()
