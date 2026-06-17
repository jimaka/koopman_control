#!/usr/bin/env python3
"""对比 v4 dt=4s 与 dt=1s 原模型在相同预测时长（20s）上的评估指标。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dt4s_dir", type=str, default="eval_out/v4_dt4s")
    p.add_argument("--dt1s_dir", type=str, default="eval_out/v4_dt1s_original")
    p.add_argument("--out_dir", type=str, default="eval_out/v4_dt4s_vs_original")
    p.add_argument("--dt4s_tag", type=str, default="v4_dt4s")
    p.add_argument("--dt1s_tag", type=str, default="v4_dt1s_original")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.dt1s_dir, f"{args.dt1s_tag}_summary.json"), encoding="utf-8") as f:
        s1 = json.load(f)
    with open(os.path.join(args.dt4s_dir, f"{args.dt4s_tag}_summary.json"), encoding="utf-8") as f:
        s4 = json.load(f)

    k1 = s1["pred_len"]
    k4 = s4["pred_len"]

    rows = [
        ("vel_rmse@20s", s1["aggregate"][f"vel_rmse_step_{k1}"], s4["aggregate"][f"vel_rmse_step_{k4}"]),
        ("u_rmse@20s", s1["aggregate"][f"u_rmse_step_{k1}"], s4["aggregate"][f"u_rmse_step_{k4}"]),
        ("v_rmse@20s", s1["aggregate"][f"v_rmse_step_{k1}"], s4["aggregate"][f"v_rmse_step_{k4}"]),
        ("r_rmse@20s", s1["aggregate"][f"r_rmse_step_{k1}"], s4["aggregate"][f"r_rmse_step_{k4}"]),
        (
            "traj_xy@20s",
            s1["aggregate"][f"traj_xy_rmse_step_{k1}"],
            s4["aggregate"][f"traj_xy_rmse_step_{k4}"],
        ),
    ]

    md = "# v4 dt=1s vs dt=4s 对比\n\n"
    md += "| 指标 | dt=1s (原模型) | dt=4s (新训练) |\n|---|---:|---:|\n"
    for name, v1, v4 in rows:
        md += f"| {name} | {v1:.6f} | {v4:.6f} |\n"

    md_path = os.path.join(args.out_dir, "compare_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    c1 = load_csv(os.path.join(args.dt1s_dir, f"{args.dt1s_tag}_per_step_metrics.csv"))
    c4 = load_csv(os.path.join(args.dt4s_dir, f"{args.dt4s_tag}_per_step_metrics.csv"))
    t1 = np.array([float(r["step"]) for r in c1])
    t4 = np.array([float(r["step"]) * 4.0 for r in c4])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, ch, ylab in zip(axes, ["vel_rmse", "u_rmse", "v_rmse"], ["vel rmse", "u rmse", "v rmse"]):
        ax.plot(t1, [float(r[ch]) for r in c1], "o-", ms=3, label="dt=1s (original)")
        ax.plot(t4, [float(r[ch]) for r in c4], "s-", ms=5, label="dt=4s (new)")
        ax.set_xlabel("prediction time [s]")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("v4 model: dt=1s vs dt=4s RMSE vs prediction horizon")
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "compare_rmse_vs_time.png")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    print(f"[OK] {md_path}")
    print(f"[OK] {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
