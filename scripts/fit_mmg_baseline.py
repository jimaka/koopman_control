#!/usr/bin/env python3
"""阶段 1（M1）：MMG 基线最小二乘辨识 + 基线精度报告。

对应 docs/MMG残差MLP建模技术方案.md §3.3 / §8-M1。辨识结果保存到
checkpoints/mmg_baseline.npz，供 train_mmg_residual.py / compare 脚本复用。

用法:
    python3 scripts/fit_mmg_baseline.py                  # 辨识 + 测试集基线评估
    python3 scripts/fit_mmg_baseline.py --no_eval        # 只辨识保存参数
    python3 scripts/fit_mmg_baseline.py --train_data data/sim_10HZ.npz --out checkpoints/mmg_baseline_sim.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch

from koopman import evalkit as ek  # noqa: E402
from koopman import paths as P  # noqa: E402
from koopman.mmg_model import (  # noqa: E402
    MMG_PARAM_NAMES,
    MmgModel,
    PhysStepAdapter,
    compute_train_stats,
    least_squares_fit,
    save_mmg_npz,
)


def print_report(report: Dict) -> None:
    print("\n== 最小二乘辨识结果（每单位质量/惯量组合参数）==")
    for ch, ch_name in [("surge", "纵向 u"), ("sway", "横向 v"), ("yaw", "回转 r")]:
        r = report[ch]
        print(f"[{ch_name}] n={r['n_samples']}  accel_rmse={r['rmse']:.4e}  R²={r['r2']:.4f}")
        for k, v in r["params"].items():
            print(f"    {k:>10s} = {v:+.6e}")


def eval_baseline(theta: np.ndarray, stats: Dict[str, np.ndarray], args) -> Dict:
    """在测试集上做开环 rollout，返回关键指标。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MmgModel(theta=theta).to(device)
    dt = float(args.dt)
    adapter = PhysStepAdapter(lambda d, c: model.step_phys(d, c, dt), stats).to(device)

    ms = ek.model_stride_from_dt(dt, args.data_dt)
    K = int(round(args.horizon_sec / dt))
    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(
        args.test_data, pred_len=K, stride=1, model_stride=ms
    )
    if args.max_samples and t0g.shape[0] > args.max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, args.max_samples).astype(int)
        t0g = t0g[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        adapter, states_full, ctrls_full, t0g, K, stats, device, dt,
        batch_size=args.batch_size, model_stride=ms,
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)

    print(f"\n== MMG 基线开环 rollout（{os.path.basename(args.test_data)}, dt={dt}s, "
          f"horizon={K}步={K * dt:.0f}s, n={gt_dyn.shape[0]}）==")
    header = f"{'step':>4s} {'u_rmse':>9s} {'v_rmse':>9s} {'r_rmse':>9s} {'vel_rmse':>9s} {'traj_xy':>9s}"
    print(header)
    for k in range(K):
        if k == 0 or (k + 1) % 5 == 0 or k == K - 1:
            print(f"{k + 1:>4d} {per_step['u_rmse'][k]:>9.4f} {per_step['v_rmse'][k]:>9.4f} "
                  f"{per_step['r_rmse'][k]:>9.4f} {per_step['vel_rmse'][k]:>9.4f} {per_step['traj_xy_err'][k]:>9.3f}")
    print(f"发散比 step{K}/step1 = {div.get('ratio_stepK_over_step1', float('nan')):.2f}")

    summary = {
        "tag": "mmg_baseline", "dt": dt, "horizon_sec": float(K * dt),
        "n_samples": int(gt_dyn.shape[0]),
        "vel_rmse_step_1": float(per_step["vel_rmse"][0]),
        "vel_rmse_step_K": float(per_step["vel_rmse"][-1]),
        "u_rmse_step_K": float(per_step["u_rmse"][-1]),
        "v_rmse_step_K": float(per_step["v_rmse"][-1]),
        "r_rmse_step_K": float(per_step["r_rmse"][-1]),
        "traj_xy_err_step_K": float(per_step["traj_xy_err"][-1]),
        "divergence_ratio": float(div.get("ratio_stepK_over_step1", float("nan"))),
    }
    if args.eval_out:
        os.makedirs(args.eval_out, exist_ok=True)
        ek.write_per_step_csv(per_step, os.path.join(args.eval_out, "mmg_baseline_per_step_metrics.csv"))
        with open(os.path.join(args.eval_out, "mmg_baseline_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"评估产物已写入 {args.eval_out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train_data", type=str, default=str(P.TRAIN_MERGED), help="辨识用训练集")
    ap.add_argument("--test_data", type=str, default=str(P.TEST), help="基线评估数据集")
    ap.add_argument("--out", type=str, default=str(P.CKPT_DIR / "mmg_baseline.npz"), help="参数保存路径")
    ap.add_argument("--data_dt", type=float, default=0.1)
    ap.add_argument("--smooth", type=int, default=5, help="差分前滑动平均窗长")
    ap.add_argument("--ridge", type=float, default=1e-6, help="岭正则系数")
    ap.add_argument("--dt", type=float, default=1.0, help="基线评估的模型步长 [s]")
    ap.add_argument("--horizon_sec", type=float, default=20.0, help="开环 rollout 时长 [s]")
    ap.add_argument("--max_samples", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--eval_out", type=str, default=str(P.EVAL_OUT_DIR / "mmg_baseline"))
    ap.add_argument("--no_eval", action="store_true", help="只辨识保存，不做 rollout 评估")
    args = ap.parse_args()

    states_full, ctrls_full, seg_starts, seg_lens = ek._load_segments_cached(args.train_data)
    segments = [
        (states_full[s:s + L], ctrls_full[s:s + L])
        for s, L in zip(seg_starts.tolist(), seg_lens.tolist())
    ]
    print(f"辨识数据: {args.train_data}  段数={len(segments)}  样本={states_full.shape[0]}")

    theta, report = least_squares_fit(segments, data_dt=args.data_dt, smooth=args.smooth, ridge=args.ridge)
    print_report(report)

    stats = compute_train_stats(states_full, ctrls_full)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_mmg_npz(args.out, theta, stats, report)
    print(f"\n参数已保存: {args.out}")

    if not args.no_eval:
        eval_baseline(theta, stats, args)


if __name__ == "__main__":
    main()
