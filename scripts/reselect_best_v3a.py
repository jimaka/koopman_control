"""scripts/reselect_best_v3a.py — PROMPT_v3a §4.2 A2 离线重选 best ckpt。

不重训：遍历一组已有 ckpt（默认 ``checkpoints/koopman_v3_run*_best.pth``），
在 val 集上算 composite_v3a 公式得分（含 slope_loglog 与 degraded_pct_vs_v1
的两个软上界），取最低复制为 ``checkpoints/koopman_v3a_reselect_best.pth``。

公式（与 train_koopman_v2.py 中 _compute_composite_v3a 完全等价）：

    composite_v3a = vel_rmse_mean * max(1, instability_score)
                  * (1 + 2 * max(0, slope_loglog - 0.65))
                  * (1 + 5 * max(0, degraded_pct_vs_v1/100 - 0.18))

落盘：
    <out_dir>/reselect_table.md  逐 ckpt 一行的得分明细。
    <out_path>                   把最低分 ckpt 直接 cp 过去。

CLI:
    python3 scripts/reselect_best_v3a.py \\
        --ckpt_glob 'checkpoints/koopman_v3_run*_best.pth' \\
        --data koopman_val.npz \\
        --baseline_ckpt checkpoints/koopman_v1_best.pth \\
        --out checkpoints/koopman_v3a_reselect_best.pth
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from typing import Optional

import numpy as np
import torch

# 让本脚本可在 repo 根目录之外直接 python3 -m 调用。
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import eval_koopman as ek  # noqa: E402


SLOPE_TARGET = 0.65          # PROMPT_v3a 公式
DEG_TARGET_PCT = 18.0        # PROMPT_v3a 公式


def composite_v3a(
    vel_rmse_mean: float,
    instability: float,
    slope_loglog: float,
    degraded_pct_v1: float,
) -> float:
    base = float(vel_rmse_mean) * max(1.0, float(instability))
    slope_pen = 1.0 + 2.0 * max(0.0, float(slope_loglog) - SLOPE_TARGET)
    deg = float(degraded_pct_v1) / 100.0
    deg_pen = 1.0 + 5.0 * max(0.0, deg - DEG_TARGET_PCT / 100.0)
    return float(base * slope_pen * deg_pen)


def _baseline_per_sample_vel_err_K(
    baseline_ckpt: str, data_path: str, pred_len: int, device: torch.device, batch_size: int,
) -> Optional[np.ndarray]:
    if not os.path.exists(baseline_ckpt):
        print(f"[reselect] WARNING baseline {baseline_ckpt!r} 不存在；degraded_pct 强制 0.",
              file=sys.stderr)
        return None
    model, stats = ek.load_model_from_ckpt(baseline_ckpt, device)
    states_full, ctrls_full, _, _, t0_global, _, _ = ek._flatten_segments(
        data_path, pred_len=pred_len, stride=1,
    )
    gt, pred, _, _ = ek.rollout_dataset(
        model, states_full, ctrls_full, t0_global, pred_len, stats, device, 0.1, batch_size,
    )
    diff = pred - gt
    return np.sqrt(diff[:, -1, 0] ** 2 + diff[:, -1, 1] ** 2).astype(np.float64)


def _eval_ckpt_on_val(
    ckpt_path: str, data_path: str, pred_len: int, device: torch.device, batch_size: int,
    baseline_per_sample: Optional[np.ndarray],
) -> dict:
    model, stats = ek.load_model_from_ckpt(ckpt_path, device)
    states_full, ctrls_full, _, _, t0_global, _, _ = ek._flatten_segments(
        data_path, pred_len=pred_len, stride=1,
    )
    gt, pred, gt_xy, pred_xy = ek.rollout_dataset(
        model, states_full, ctrls_full, t0_global, pred_len, stats, device, 0.1, batch_size,
    )
    per_step = ek.compute_per_step_metrics(gt, pred, gt_xy, pred_xy, dt=0.1)
    div = ek.compute_divergence_metrics(per_step)
    vel_rmse_mean = float(np.mean(per_step["vel_rmse"]))
    inst = float(div["instability_score"])
    slope = float(div["slope_loglog"])
    diff = pred - gt
    vel_err_K = np.sqrt(diff[:, -1, 0] ** 2 + diff[:, -1, 1] ** 2)
    if baseline_per_sample is not None:
        n = min(vel_err_K.shape[0], baseline_per_sample.shape[0])
        deg_pct = float(np.mean(vel_err_K[:n] > baseline_per_sample[:n]) * 100.0)
    else:
        deg_pct = 0.0
    score = composite_v3a(vel_rmse_mean, inst, slope, deg_pct)
    return {
        "ckpt": ckpt_path,
        "vel_rmse_mean": vel_rmse_mean,
        "vel_rmse_K": float(per_step["vel_rmse"][-1]),
        "u_rmse_K": float(per_step["u_rmse"][-1]),
        "v_rmse_K": float(per_step["v_rmse"][-1]),
        "instability_score": inst,
        "slope_loglog": slope,
        "degraded_pct_v1": deg_pct,
        "composite_v3a": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt_glob", type=str,
        default="checkpoints/koopman_v3_run*_best.pth",
        help="ckpt glob pattern；也可加多个 --ckpt_glob 重复参数。",
        action="append",
    )
    parser.add_argument("--data", type=str, default="koopman_val.npz")
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--baseline_ckpt", type=str,
                        default="checkpoints/koopman_v1_best.pth")
    parser.add_argument("--out", type=str,
                        default="checkpoints/koopman_v3a_reselect_best.pth")
    parser.add_argument("--out_dir", type=str, default="test_analysis/v3a")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--smoketest", action="store_true",
        help="只评估匹配到的第一个 ckpt（PROMPT_v3a §8）",
    )
    args = parser.parse_args()

    # argparse 在使用 action='append' 时不会丢弃 default —— 手动处理。
    if isinstance(args.ckpt_glob, list) and len(args.ckpt_glob) > 1:
        args.ckpt_glob = [g for g in args.ckpt_glob if g]
    elif isinstance(args.ckpt_glob, list):
        # 只有一个：可能就是默认值
        pass

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    # 收集所有 ckpt
    paths: list = []
    globs = args.ckpt_glob if isinstance(args.ckpt_glob, list) else [args.ckpt_glob]
    for g in globs:
        paths.extend(sorted(glob.glob(g)))
    paths = sorted(set(paths))
    if not paths:
        print(f"[reselect] 没有 ckpt 匹配 {globs}", file=sys.stderr)
        return 2
    print(f"[reselect] {len(paths)} ckpts to evaluate on {args.data}, K={args.pred_len}")

    if args.smoketest:
        paths = paths[:1]

    base_per_sample = _baseline_per_sample_vel_err_K(
        args.baseline_ckpt, args.data, args.pred_len, device, args.batch_size,
    )

    rows = []
    for p in paths:
        try:
            row = _eval_ckpt_on_val(p, args.data, args.pred_len, device, args.batch_size, base_per_sample)
            rows.append(row)
            print(
                f"  {os.path.basename(p):40s}  "
                f"vel={row['vel_rmse_mean']:.5f}  velK={row['vel_rmse_K']:.5f}  "
                f"slope={row['slope_loglog']:.3f}  inst={row['instability_score']:.3f}  "
                f"deg%={row['degraded_pct_v1']:.2f}  composite_v3a={row['composite_v3a']:.5f}"
            )
        except Exception as e:
            print(f"  {os.path.basename(p):40s}  FAILED: {e}", file=sys.stderr)

    if not rows:
        print("[reselect] 全部失败", file=sys.stderr)
        return 3

    rows.sort(key=lambda r: r["composite_v3a"])
    best = rows[0]
    print(f"\n[reselect] BEST = {best['ckpt']}  composite_v3a={best['composite_v3a']:.5f}")

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "reselect_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# PROMPT_v3a §4.2 A2 — 离线 ckpt 重选 (composite_v3a)\n\n")
        f.write(
            f"- val data: `{args.data}`  K={args.pred_len}  "
            f"baseline=`{args.baseline_ckpt}`\n"
        )
        f.write(
            f"- 公式: composite_v3a = vel_rmse_mean × max(1, inst) × "
            f"(1+2·max(0, slope-{SLOPE_TARGET})) × "
            f"(1+5·max(0, deg%/100-{DEG_TARGET_PCT/100:.2f}))\n\n"
        )
        f.write(
            "| ckpt | vel_rmse_mean | vel_K | u_K | v_K | inst | slope | deg% | composite_v3a |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            mark = "  ⭐" if r is best else ""
            f.write(
                f"| `{os.path.basename(r['ckpt'])}`{mark} | "
                f"{r['vel_rmse_mean']:.5f} | {r['vel_rmse_K']:.5f} | "
                f"{r['u_rmse_K']:.5f} | {r['v_rmse_K']:.5f} | "
                f"{r['instability_score']:.4f} | {r['slope_loglog']:.4f} | "
                f"{r['degraded_pct_v1']:.2f} | {r['composite_v3a']:.6f} |\n"
            )
        f.write(f"\n**BEST**: `{best['ckpt']}`  →  `{args.out}`\n")
    print(f"[reselect] table -> {md_path}")

    if not args.smoketest:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        shutil.copyfile(best["ckpt"], args.out)
        print(f"[reselect] BEST copied -> {args.out}")
    else:
        print("[reselect] smoketest mode; no copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
