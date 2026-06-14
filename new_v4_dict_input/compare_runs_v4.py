#!/usr/bin/env python3
"""多 v4 run 对比评估（正确处理 model_stride，dt=1.0 / data_dt=0.1 → stride=10）。

背景：``scripts/eval.py --compare`` 走 ``evalkit.evaluate_one``，未传 ``model_stride``，
对 v4（dt=1.0）会以原始 0.1s 数据按 1 步 rollout，结果不可用。本脚本统一用
``eval_v4_dict_input.evaluate``（与训练/部署一致的 model_stride）做多模型对比，
输出合并的逐步 RMSE 图（vel/u/v）与指标对比表（Markdown + CSV）。

用法::

    python3 new_v4_dict_input/compare_runs_v4.py \
        --models BASE.pth:baseline OPT.pth:optimized \
        --data data/koopman_test.npz --pred_len 20 --dt 1.0 \
        --out_dir eval_out/opt/compare_v4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402
from new_v4_dict_input.eval_v4_dict_input import evaluate  # noqa: E402

setup_repo()


def parse_models(values: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for v in values:
        if ":" not in v:
            raise argparse.ArgumentTypeError(f"--models 元素须形如 path:label，收到 {v}")
        path, label = v.rsplit(":", 1)
        out.append((path, label))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--models", type=str, nargs="+", required=True, help="path:label 列表")
    p.add_argument("--data", type=str, default="data/koopman_test.npz")
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--data_dt", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out_dir", type=str, default="eval_out/opt/compare_v4")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model_stride = ek.model_stride_from_dt(args.dt, args.data_dt)
    pairs = parse_models(args.models)

    per_steps: Dict[str, Dict[str, np.ndarray]] = {}
    summaries: Dict[str, Dict] = {}
    for path, label in pairs:
        per_step, _, summary, _, _ = evaluate(
            ckpt_path=path, data_path=args.data, pred_len=args.pred_len, dt=args.dt,
            batch_size=args.batch_size, device=device, max_samples=args.max_samples,
            model_stride=model_stride, data_dt=args.data_dt,
        )
        per_steps[label] = per_step
        summaries[label] = summary
        print(f"[{label}] {path}")

    K = args.pred_len
    steps = np.arange(1, K + 1)

    # --- 合并逐步 RMSE 图：vel / u / v ---
    for ch, ylabel, fname in [
        ("vel_rmse", "vel rmse [m/s]", "compare_vel_rmse_vs_step.png"),
        ("u_rmse", "u rmse [m/s]", "compare_u_rmse_vs_step.png"),
        ("v_rmse", "v rmse [m/s]", "compare_v_rmse_vs_step.png"),
    ]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        for label in per_steps:
            ax1.plot(steps, per_steps[label][ch], "o-", ms=3, label=label)
            ax2.plot(steps, per_steps[label][ch], "o-", ms=3, label=label)
        for ax in (ax1, ax2):
            ax.set_xlabel("step (1 step = dt = %.1fs)" % args.dt)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        ax1.set_title(f"{ch} vs step (linear)")
        ax2.set_title(f"{ch} vs step (log y)")
        ax2.set_yscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, fname), dpi=200)
        plt.close(fig)

    # --- 指标表 ---
    rows = []
    for label in summaries:
        agg = summaries[label]["aggregate"]
        div = summaries[label]["divergence"]
        rows.append({
            "label": label,
            "vel_rmse_mean": agg["vel_rmse_mean"],
            "vel_rmse@1": agg["vel_rmse_step_1"],
            f"vel_rmse@{K}": agg[f"vel_rmse_step_{K}"],
            f"u_rmse@{K}": agg[f"u_rmse_step_{K}"],
            f"v_rmse@{K}": agg[f"v_rmse_step_{K}"],
            f"r_rmse@{K}": agg[f"r_rmse_step_{K}"],
            f"traj_xy_rmse@{K}": agg[f"traj_xy_rmse_step_{K}"],
            "slope_loglog": div["slope_loglog"],
            "instability": div["instability_score"],
        })
    cols = list(rows[0].keys())
    # CSV
    with open(os.path.join(args.out_dir, "compare_metrics.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(
                (r[c] if c == "label" else f"{r[c]:.6g}") for c in cols
            ) + "\n")
    # Markdown
    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(
            (r[c] if c == "label" else f"{r[c]:.6g}") for c in cols
        ) + " |")
    md = "\n".join(md_lines) + "\n"
    with open(os.path.join(args.out_dir, "compare_metrics.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.out_dir, "compare_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print("\n" + md)
    print(f"outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
