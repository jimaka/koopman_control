#!/usr/bin/env python3
"""Evaluate v4 dict-input Koopman model on test dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402
from koopman import paths as P  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402
from new_v4_dict_input.model_v4_dict_input import HorizontalKoopmanModelV4DictInput  # noqa: E402

setup_repo()


def load_v4_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    args_d = ckpt.get("args", {}) or {}
    model = HorizontalKoopmanModelV4DictInput(
        hidden_dim=int(args_d.get("hidden_dim", 32)),
        clamp_pif=float(args_d.get("clamp_pif", 5.0)),
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, stats 


def write_per_step_csv(per_step: Dict[str, np.ndarray], path: str) -> None:
    cols = [
        "step",
        "n_samples",
        "vel_rmse",
        "vel_mae",
        "u_rmse",
        "v_rmse",
        "r_rmse",
        "u_bias",
        "v_bias",
        "r_bias",
        "traj_xy_err",
        "acc_rmse",
    ]
    k = int(per_step["step"].shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for i in range(k):
            row = []
            for c in cols:
                v = per_step[c][i]
                if isinstance(v, (np.integer, int)):
                    row.append(str(int(v)))
                elif isinstance(v, (np.floating, float)):
                    if np.isnan(v):
                        row.append("nan")
                    else:
                        row.append(f"{float(v):.8g}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")


def build_summary(
    ckpt_path: str,
    pred_len: int,
    per_step: Dict[str, np.ndarray],
    div: Dict[str, float],
) -> Dict:
    return {
        "ckpt": ckpt_path,
        "pred_len": int(pred_len),
        "aggregate": {
            "vel_rmse_mean": float(np.mean(per_step["vel_rmse"])),
            "vel_rmse_step_1": float(per_step["vel_rmse"][0]),
            f"vel_rmse_step_{pred_len}": float(per_step["vel_rmse"][-1]),
            f"u_rmse_step_{pred_len}": float(per_step["u_rmse"][-1]),
            f"v_rmse_step_{pred_len}": float(per_step["v_rmse"][-1]),
            f"r_rmse_step_{pred_len}": float(per_step["r_rmse"][-1]),
            f"traj_xy_rmse_step_{pred_len}": float(per_step["traj_xy_err"][-1]),
            "acc_rmse_mean": float(np.nanmean(per_step["acc_rmse"])),
        },
        "divergence": {
            f"ratio_step{pred_len}_over_step1": float(div.get(f"ratio_step{pred_len}_over_step1", np.nan)),
            "slope_loglog": float(div["slope_loglog"]),
            "lyapunov_like": float(div["lyapunov_like"]),
            "instability_score": float(div["instability_score"]),
            "monotonic_increasing": bool(div["monotonic_increasing"]),
        },
        "channel_bias_mean": {
            "u_bias_mean": float(np.mean(per_step["u_bias"])),
            "v_bias_mean": float(np.mean(per_step["v_bias"])),
            "r_bias_mean": float(np.mean(per_step["r_bias"])),
        },
    }


def _pick_quantile_indices(values: np.ndarray, n: int) -> np.ndarray:
    m = int(values.shape[0])
    n = min(max(int(n), 1), m)
    sorted_idx = np.argsort(values)
    q = np.linspace(0, m - 1, n).astype(int)
    return sorted_idx[q]


def plot_channel_rmse_vs_step(
    per_step: Dict[str, np.ndarray],
    channel: str,
    path: str,
    tag: str,
) -> None:
    steps = per_step["step"]
    key = f"{channel}_rmse"
    if channel == "u":
        title_name = "u (surge)"
    elif channel == "v":
        title_name = "v (sway)"
    else:
        raise ValueError(f"unsupported channel: {channel}")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(steps, per_step[key], "o-", lw=1.6, label=f"{title_name} rmse")
    ax.set_title(f"{tag} {title_name} RMSE vs step")
    ax.set_xlabel("step")
    ax.set_ylabel("rmse [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_channel_scatter(
    gt_dyn: np.ndarray,
    pred_dyn: np.ndarray,
    channel: str,
    path: str,
    tag: str,
    max_points: int = 30000,
) -> None:
    if channel == "u":
        ch_idx = 0
        title_name = "u (surge)"
    elif channel == "v":
        ch_idx = 1
        title_name = "v (sway)"
    else:
        raise ValueError(f"unsupported channel: {channel}")

    gt_ch = gt_dyn[..., ch_idx].reshape(-1)
    pred_ch = pred_dyn[..., ch_idx].reshape(-1)
    n = gt_ch.shape[0]
    if n > max_points:
        sel = np.linspace(0, n - 1, max_points).astype(int)
        gt_ch, pred_ch = gt_ch[sel], pred_ch[sel]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(gt_ch, pred_ch, s=6, alpha=0.25)
    lo = float(min(gt_ch.min(), pred_ch.min()))
    hi = float(max(gt_ch.max(), pred_ch.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal y=x")
    ax.set_xlabel(f"GT {title_name} [m/s]")
    ax.set_ylabel(f"Pred {title_name} [m/s]")
    ax.set_title(f"{tag} {title_name}: GT vs Pred")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_channel_sample_curves(
    gt_dyn: np.ndarray,
    pred_dyn: np.ndarray,
    channel: str,
    path: str,
    tag: str,
) -> None:
    if channel == "u":
        ch_idx = 0
        title_name = "u (surge)"
    elif channel == "v":
        ch_idx = 1
        title_name = "v (sway)"
    else:
        raise ValueError(f"unsupported channel: {channel}")

    diff = pred_dyn - gt_dyn
    vel_err_step_k = np.sqrt(diff[:, -1, 0] ** 2 + diff[:, -1, 1] ** 2)
    pick = _pick_quantile_indices(vel_err_step_k, n=6)
    k = gt_dyn.shape[1]
    steps = np.arange(1, k + 1)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for i, sidx in enumerate(pick):
        ax = axes[i // 2, i % 2]
        ax.plot(steps, gt_dyn[sidx, :, ch_idx], "g-", lw=1.4, label=f"GT {channel}")
        ax.plot(steps, pred_dyn[sidx, :, ch_idx], "g--", lw=1.2, label=f"Pred {channel}")
        ax.set_title(f"sample #{sidx} | vel_err@K={vel_err_step_k[sidx]:.4f}")
        ax.set_xlabel("step")
        ax.set_ylabel("speed [m/s]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"{tag} {title_name} curve comparison (6 quantile samples)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate(
    ckpt_path: str,
    data_path: str,
    pred_len: int,
    dt: float,
    batch_size: int,
    device: torch.device,
    max_samples: Optional[int],
) -> tuple[Dict[str, np.ndarray], Dict[str, float], Dict, np.ndarray, np.ndarray]:
    model, stats = load_v4_model(ckpt_path, device)
    states_full, ctrls_full, _, _, t0g, _, _ = ek._flatten_segments(data_path, pred_len=pred_len, stride=1)
    if max_samples is not None and t0g.shape[0] > max_samples:
        sel = np.linspace(0, t0g.shape[0] - 1, max_samples).astype(int)
        t0g = t0g[sel]

    gt_dyn, pred_dyn, gt_xy, pred_xy = ek.rollout_dataset(
        model,
        states_full,
        ctrls_full,
        t0g,
        pred_len,
        stats,
        device,
        dt,
        batch_size=batch_size,
    )
    per_step = ek.compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = ek.compute_divergence_metrics(per_step)
    summary = build_summary(ckpt_path, pred_len, per_step, div)
    summary["n_samples"] = int(gt_dyn.shape[0])
    return per_step, div, summary, gt_dyn, pred_dyn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=str(P.CKPT_DIR / "koopman_v4_best.pth"))
    p.add_argument("--data", type=str, default=str(P.TEST))
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default=str(P.EVAL_OUT_DIR / "v4_test"))
    p.add_argument("--tag", type=str, default="v4")
    return p


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")
    if not os.path.isfile(args.data):
        raise FileNotFoundError(f"dataset not found: {args.data}")

    os.makedirs(args.out_dir, exist_ok=True)
    per_step, _, summary, gt_dyn, pred_dyn = evaluate(
        ckpt_path=args.ckpt,
        data_path=args.data,
        pred_len=args.pred_len,
        dt=args.dt,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
    )

    per_step_csv = os.path.join(args.out_dir, f"{args.tag}_per_step_metrics.csv")
    summary_json = os.path.join(args.out_dir, f"{args.tag}_summary.json")
    u_rmse_png = os.path.join(args.out_dir, f"{args.tag}_u_rmse_vs_step.png")
    v_rmse_png = os.path.join(args.out_dir, f"{args.tag}_v_rmse_vs_step.png")
    u_scatter_png = os.path.join(args.out_dir, f"{args.tag}_u_scatter_compare.png")
    v_scatter_png = os.path.join(args.out_dir, f"{args.tag}_v_scatter_compare.png")
    u_curve_png = os.path.join(args.out_dir, f"{args.tag}_u_curve_compare.png")
    v_curve_png = os.path.join(args.out_dir, f"{args.tag}_v_curve_compare.png")
    write_per_step_csv(per_step, per_step_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_channel_rmse_vs_step(per_step, "u", u_rmse_png, args.tag)
    plot_channel_rmse_vs_step(per_step, "v", v_rmse_png, args.tag)
    plot_channel_scatter(gt_dyn, pred_dyn, "u", u_scatter_png, args.tag)
    plot_channel_scatter(gt_dyn, pred_dyn, "v", v_scatter_png, args.tag)
    plot_channel_sample_curves(gt_dyn, pred_dyn, "u", u_curve_png, args.tag)
    plot_channel_sample_curves(gt_dyn, pred_dyn, "v", v_curve_png, args.tag)

    k = args.pred_len
    agg = summary["aggregate"]
    div = summary["divergence"]
    print("=== V4 TEST EVAL ===")
    print(f"ckpt: {args.ckpt}")
    print(f"data: {args.data}")
    print(f"device: {device}")
    print(f"n_samples: {summary['n_samples']}")
    print(
        f"vel_rmse@1={agg['vel_rmse_step_1']:.6g}, "
        f"vel_rmse@{k}={agg[f'vel_rmse_step_{k}']:.6g}, "
        f"u/v/r@{k}=({agg[f'u_rmse_step_{k}']:.6g}, {agg[f'v_rmse_step_{k}']:.6g}, {agg[f'r_rmse_step_{k}']:.6g})"
    )
    print(
        f"ratio={div[f'ratio_step{k}_over_step1']:.6g}, "
        f"slope_loglog={div['slope_loglog']:.6g}, "
        f"instability={div['instability_score']:.6g}"
    )
    print(f"outputs: {per_step_csv}, {summary_json}")
    print(
        "plots: "
        f"{u_rmse_png}, {v_rmse_png}, {u_scatter_png}, {v_scatter_png}, {u_curve_png}, {v_curve_png}"
    )
    print("====================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
