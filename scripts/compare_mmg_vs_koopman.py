#!/usr/bin/env python3
"""三方对比评估：MMG 基线 / MMG+残差MLP / v4 Koopman（可选）。

对应 docs/MMG残差MLP建模技术方案.md §5。同一测试集、同一步长 dt（默认取 v4
原生 dt，MMG 侧为连续动力学可任意步长 rollout）、同一 horizon、同一评估代码
（koopman/evalkit.py），输出 compare_summary.csv/md 与 per-step CSV。

用法:
    python3 scripts/compare_mmg_vs_koopman.py
    python3 scripts/compare_mmg_vs_koopman.py --no_v4 --dt 1.0
    python3 scripts/compare_mmg_vs_koopman.py --res_ckpt checkpoints/mmg_residual_best.pth --max_samples 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from koopman import evalkit as ek
from koopman import paths as P
from koopman.mmg_model import MmgModel, PhysStepAdapter, load_mmg_npz
from koopman.mmg_residual import MmgResidualModel


def eval_phys_model(
    tag: str,
    ckpt_path: str,
    model: nn.Module,
    stats: Dict[str, np.ndarray],
    states_full: np.ndarray,
    ctrls_full: np.ndarray,
    seg_starts: np.ndarray,
    seg_lens: np.ndarray,
    t0g: np.ndarray,
    seg_idx: np.ndarray,
    K: int,
    ms: int,
    dt: float,
    device: torch.device,
    batch_size: int,
) -> ek.EvalResult:
    """对物理步进模型（MMG / MMG+残差）走与 v4 完全相同的 rollout + 指标管线。"""
    adapter = PhysStepAdapter(lambda d, c: model.step_phys(d, c, dt), stats).to(device)
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        adapter, states_full, ctrls_full, t0g, K, stats, device, dt,
        batch_size=batch_size, model_stride=ms,
    )
    return _build_result(tag, ckpt_path, gt_dyn, pred_dyn, gt_xy, pred_xy,
                         states_full, ctrls_full, seg_starts, seg_lens, seg_idx, K, dt)


def eval_koopman_model(
    tag: str,
    ckpt_path: str,
    states_full: np.ndarray,
    ctrls_full: np.ndarray,
    seg_starts: np.ndarray,
    seg_lens: np.ndarray,
    t0g: np.ndarray,
    seg_idx: np.ndarray,
    K: int,
    ms: int,
    dt: float,
    device: torch.device,
    batch_size: int,
) -> Tuple[ek.EvalResult, Dict]:
    """v4 Koopman：evalkit 原生接口，stats 用其自身 ckpt 内的。"""
    model, stats = ek.load_model_from_ckpt(ckpt_path, device)
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        model, states_full, ctrls_full, t0g, K, stats, device, dt,
        batch_size=batch_size, model_stride=ms,
    )
    res = _build_result(tag, ckpt_path, gt_dyn, pred_dyn, gt_xy, pred_xy,
                        states_full, ctrls_full, seg_starts, seg_lens, seg_idx, K, dt)
    return res, stats


def _build_result(tag, ckpt_path, gt_dyn, pred_dyn, gt_xy, pred_xy,
                  states_full, ctrls_full, seg_starts, seg_lens, seg_idx, K, dt) -> ek.EvalResult:
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)
    per_sample = ek.compute_per_sample_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy)
    summary = ek.build_summary_dict(
        tag=tag, ckpt_path=ckpt_path, n_samples=int(gt_dyn.shape[0]),
        pred_len=K, dt=dt, per_step=per_step, div=div, per_sample=per_sample,
    )
    per_seg = ek.compute_per_segment_metrics(
        gt_dyn, pred_dyn, gt_xy, pred_xy, seg_idx,
        states_full, ctrls_full, seg_starts, seg_lens, K=K,
    )
    summary["per_segment"] = ek.build_per_segment_summary(per_seg)
    diff = pred_dyn - gt_dyn
    vel_horiz_err = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
    return ek.EvalResult(
        tag=tag, ckpt_path=ckpt_path, summary=summary, per_step=per_step,
        per_sample=per_sample, gt_dyn=gt_dyn, pred_dyn=pred_dyn,
        gt_xy=gt_xy, pred_xy=pred_xy, vel_horiz_err=vel_horiz_err,
        seg_idx=seg_idx, per_segment=per_seg,
    )


def print_table(results: List[ek.EvalResult], K: int) -> None:
    cols = [1, 5, 10, K]
    cols = sorted({c for c in cols if 1 <= c <= K})
    print("\n== 对比总表（RMSE，物理量；u/v 单位 m/s，r 单位 rad/s，traj 单位 m）==")
    header = f"{'model':<16s}" + "".join(f" | vel@{c:<2d} " for c in cols) + " |  u@K   |  v@K   |  r@K    | traj@K | 发散比"
    print(header)
    print("-" * len(header))
    for r in results:
        ps = r.per_step
        row = f"{r.tag:<16s}" + "".join(f" | {ps['vel_rmse'][c - 1]:.4f}" for c in cols)
        row += (f" | {ps['u_rmse'][-1]:.4f} | {ps['v_rmse'][-1]:.4f} | {ps['r_rmse'][-1]:.5f}"
                f" | {ps['traj_xy_err'][-1]:.3f} | {r.summary['divergence'].get('ratio_stepK_over_step1', float('nan')):.2f}")
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", type=str, default=str(P.TEST))
    ap.add_argument("--dt", type=float, default=None,
                    help="评估步长 [s]；缺省取 v4 ckpt 的原生 dt（无 v4 时为 1.0）")
    ap.add_argument("--horizon_sec", type=float, default=20.0)
    ap.add_argument("--data_dt", type=float, default=0.1)
    ap.add_argument("--mmg_params", type=str, default=str(P.CKPT_DIR / "mmg_baseline.npz"))
    ap.add_argument("--res_ckpt", type=str, default=str(P.CKPT_DIR / "mmg_residual_best.pth"))
    ap.add_argument("--v4_ckpt", type=str, default=str(P.CKPT_DIR / "koopman_v4_best.pth"))
    ap.add_argument("--no_v4", action="store_true")
    ap.add_argument("--max_samples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--out_dir", type=str, default=str(P.EVAL_OUT_DIR / "mmg_compare"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 确定 dt ----
    v4_dt = None
    if not args.no_v4 and os.path.exists(args.v4_ckpt):
        ck = torch.load(args.v4_ckpt, map_location="cpu", weights_only=False)
        v4_dt = float((ck.get("args", {}) or {}).get("dt", 1.0))
    dt = args.dt if args.dt is not None else (v4_dt or 1.0)
    ms = ek.model_stride_from_dt(dt, args.data_dt)
    K = int(round(args.horizon_sec / dt))
    print(f"评估配置: dt={dt}s (stride={ms})  horizon={K}步={K * dt:.0f}s  data={args.data}")

    # ---- 统一取样（所有模型同一批 t0）----
    states_full, ctrls_full, seg_starts, seg_lens, t0g, seg_idx, _ = ek._flatten_segments(
        args.data, pred_len=K, stride=1, model_stride=ms
    )
    if args.max_samples and t0g.shape[0] > args.max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, args.max_samples).astype(int)
        t0g, seg_idx = t0g[sel], seg_idx[sel]
    print(f"样本数: {t0g.shape[0]}")

    results: List[ek.EvalResult] = []
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- MMG 基线 ----
    theta, mmg_stats, _ = load_mmg_npz(args.mmg_params)
    mmg = MmgModel(theta=theta).to(device)
    results.append(eval_phys_model(
        "mmg_baseline", args.mmg_params, mmg, mmg_stats,
        states_full, ctrls_full, seg_starts, seg_lens, t0g, seg_idx, K, ms, dt, device, args.batch_size,
    ))
    print(f"[ok] mmg_baseline  vel_rmse@K={results[-1].per_step['vel_rmse'][-1]:.4f}")
    ek.write_per_step_csv(results[-1].per_step, os.path.join(args.out_dir, "mmg_baseline_per_step.csv"))

    # ---- MMG + 残差 ----
    if os.path.exists(args.res_ckpt):
        ckpt = torch.load(args.res_ckpt, map_location=device, weights_only=False)
        res_model = MmgResidualModel.load_from_ckpt(ckpt, device)
        res_stats = ckpt["stats"]
        results.append(eval_phys_model(
            "mmg_residual", args.res_ckpt, res_model, res_stats,
            states_full, ctrls_full, seg_starts, seg_lens, t0g, seg_idx, K, ms, dt, device, args.batch_size,
        ))
        print(f"[ok] mmg_residual  vel_rmse@K={results[-1].per_step['vel_rmse'][-1]:.4f}")
        ek.write_per_step_csv(results[-1].per_step, os.path.join(args.out_dir, "mmg_residual_per_step.csv"))
    else:
        print(f"[skip] 残差模型 ckpt 不存在: {args.res_ckpt}（先运行 train_mmg_residual.py）")

    # ---- v4 Koopman ----
    if not args.no_v4:
        if os.path.exists(args.v4_ckpt):
            results.append(eval_koopman_model(
                "koopman_v4", args.v4_ckpt,
                states_full, ctrls_full, seg_starts, seg_lens, t0g, seg_idx, K, ms, dt, device, args.batch_size,
            )[0])
            print(f"[ok] koopman_v4   vel_rmse@K={results[-1].per_step['vel_rmse'][-1]:.4f}")
            ek.write_per_step_csv(results[-1].per_step, os.path.join(args.out_dir, "koopman_v4_per_step.csv"))
        else:
            print(f"[skip] v4 ckpt 不存在: {args.v4_ckpt}")

    print_table(results, K)
    csv_path, md_path = ek.write_compare_csv_md(results, args.out_dir)
    print(f"\n产物: {csv_path}\n      {md_path}")

    # ---- 对比图（与 evalkit CLI 同套图件）----
    ek.plot_compare_error_vs_step(results, os.path.join(args.out_dir, "compare_error_vs_step.png"))
    ek.plot_compare_step_K_box(results, os.path.join(args.out_dir, f"compare_step{K}_box.png"))
    ek.plot_compare_trajectory_grid(results, os.path.join(args.out_dir, "compare_trajectory_grid.png"))
    ek.plot_compare_per_segment_bar(results, os.path.join(args.out_dir, "compare_per_segment_bar.png"))
    ek.plot_compare_u_bias_per_step(results, os.path.join(args.out_dir, "compare_u_bias_per_step.png"))
    print(f"图件: {args.out_dir}/compare_*.png")


if __name__ == "__main__":
    main()
