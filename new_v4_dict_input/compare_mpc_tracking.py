#!/usr/bin/env python3
"""Compare MPC tracking accuracy across multiple Koopman checkpoints.

示例：
python3 new_v4_dict_input/compare_mpc_tracking.py \
  --models checkpoints/koopman_v2_best.pth:v2 checkpoints/koopman_v3a_best.pth:v3a checkpoints/koopman_v4_best.pth:v4 \
  --ref segment --data data/koopman_test.npz --segment 0 --steps 120 \
  --out_dir eval_out/mpc_compare_seg0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koopman.paths import setup_repo

setup_repo()

from koopman import paths as P
from koopman.mpc import (
    KoopmanMPC,
    MPCConfig,
    make_circle_reference,
    make_line_reference,
    segment_to_state_ctrl,
    tracking_metrics,
)


def parse_models(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"--models item must be 'ckpt_path:tag', got {item!r}")
        ckpt, tag = item.rsplit(":", 1)
        ckpt = ckpt.strip()
        tag = tag.strip()
        if not ckpt or not tag:
            raise argparse.ArgumentTypeError(f"invalid --models item: {item!r}")
        out.append((ckpt, tag))
    return out


def ensure_reference(
    ref_type: str,
    data_path: str,
    segment: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray | None, str]:
    if ref_type == "segment":
        raw = np.load(data_path, allow_pickle=True)["datas"]
        if segment < 0 or segment >= len(raw):
            raise ValueError(f"segment index out of range: {segment}, valid [0, {len(raw)-1}]")
        ref_state, ref_ctrl = segment_to_state_ctrl(raw[segment])
        title = f"segment {segment}"
        return ref_state, ref_ctrl, title
    if ref_type == "line":
        ref_state = make_line_reference(0.0, 0.0, 0.0, u_ref=2.0, length_m=100.0, dt=dt)
        return ref_state, None, "line"
    ref_state = make_circle_reference(0.0, 0.0, radius=25.0, speed=1.5, dt=dt)
    return ref_state, None, "circle"


def plot_compare(results: List[Dict], out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax_xy, ax_err = axes
    ref = results[0]["traj"]["ref_state"]
    n = ref.shape[0]
    ax_xy.plot(ref[:n, 0], ref[:n, 1], "k--", lw=1.6, label="reference")
    for r in results:
        sim = r["traj"]["state"]
        m = min(len(sim), len(ref))
        ax_xy.plot(sim[:m, 0], sim[:m, 1], lw=1.4, label=r["tag"])
    ax_xy.set_title(f"XY tracking ({title})")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=9)

    for r in results:
        t = r["traj"]["t"]
        sim = r["traj"]["state"]
        ref = r["traj"]["ref_state"]
        m = min(len(sim), len(ref))
        xy_err = np.linalg.norm(sim[:m, :2] - ref[:m, :2], axis=1)
        ax_err.plot(t[:m], xy_err, lw=1.4, label=f"{r['tag']} (rmse={r['metrics']['xy_rmse_m']:.3f})")
    ax_err.set_title("XY error vs time")
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mpc_tracking_compare.png"), dpi=180)
    plt.close(fig)


def write_table_files(rows: List[Dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fields = [
        "tag",
        "ckpt",
        "xy_rmse_m",
        "xy_max_m",
        "yaw_rmse_deg",
        "final_xy_err_m",
        "steps",
        "horizon",
        "opt_iters",
    ]
    csv_path = os.path.join(out_dir, "mpc_compare_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    md_path = os.path.join(out_dir, "mpc_compare_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# MPC tracking comparison\n\n")
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            vals = [str(row.get(k, "")) for k in fields]
            f.write("| " + " | ".join(vals) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="multi models in format ckpt_path:tag",
    )
    parser.add_argument("--data", type=str, default=str(P.TEST))
    parser.add_argument("--ref", choices=["segment", "line", "circle"], default="segment")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--opt_iters", type=int, default=40)
    parser.add_argument("--w_xy", type=float, default=10.0)
    parser.add_argument("--w_yaw", type=float, default=5.0)
    parser.add_argument("--w_vel", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default=str(P.EVAL_OUT_DIR / "mpc_compare"))
    args = parser.parse_args()

    model_pairs = parse_models(args.models)
    for ckpt, _ in model_pairs:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    ref_state, ref_ctrl, ref_title = ensure_reference(args.ref, args.data, args.segment, args.dt)
    cfg = MPCConfig(
        horizon=args.horizon,
        dt=args.dt,
        w_xy=args.w_xy,
        w_yaw=args.w_yaw,
        w_vel=args.w_vel,
        opt_iters=args.opt_iters,
        device=args.device,
    )

    all_results: List[Dict] = []
    table_rows: List[Dict] = []
    for ckpt, tag in model_pairs:
        mpc = KoopmanMPC.from_checkpoint(ckpt, cfg)
        traj = mpc.simulate(ref_state[0].copy(), ref_state, ref_ctrl=ref_ctrl, max_steps=args.steps)
        metrics = tracking_metrics(traj)
        row = {
            "tag": tag,
            "ckpt": ckpt,
            "xy_rmse_m": round(metrics["xy_rmse_m"], 6),
            "xy_max_m": round(metrics["xy_max_m"], 6),
            "yaw_rmse_deg": round(metrics["yaw_rmse_deg"], 6),
            "final_xy_err_m": round(metrics["final_xy_err_m"], 6),
            "steps": int(args.steps),
            "horizon": int(args.horizon),
            "opt_iters": int(args.opt_iters),
        }
        table_rows.append(row)
        all_results.append(
            {
                "tag": tag,
                "ckpt": ckpt,
                "metrics": metrics,
                "traj": {
                    "t": traj.t,
                    "state": traj.state,
                    "control": traj.control,
                    "ref_state": traj.ref_state,
                },
            }
        )

    table_rows.sort(key=lambda x: x["xy_rmse_m"])

    os.makedirs(args.out_dir, exist_ok=True)
    write_table_files(table_rows, args.out_dir)
    plot_compare(all_results, args.out_dir, ref_title)

    summary = {
        "ref": {"type": args.ref, "segment": args.segment, "title": ref_title},
        "config": {
            "steps": args.steps,
            "horizon": args.horizon,
            "dt": args.dt,
            "opt_iters": args.opt_iters,
            "w_xy": args.w_xy,
            "w_yaw": args.w_yaw,
            "w_vel": args.w_vel,
            "device": args.device,
        },
        "ranking_by_xy_rmse": table_rows,
    }
    with open(os.path.join(args.out_dir, "mpc_compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== MPC TRACKING COMPARE ===")
    for i, row in enumerate(table_rows, 1):
        print(
            f"{i:>2d}. {row['tag']}: xy_rmse={row['xy_rmse_m']:.4f} m | "
            f"xy_max={row['xy_max_m']:.4f} m | yaw_rmse={row['yaw_rmse_deg']:.3f} deg | "
            f"final_xy={row['final_xy_err_m']:.4f} m"
        )
    print(f"Outputs -> {args.out_dir}")
    print("============================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
