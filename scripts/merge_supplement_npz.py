#!/usr/bin/env python3
"""
合并多个 Koopman NPZ（datas 对象数组），可选过滤低信息段。

在 merge_npz.py 基础上扩展：
  - 支持多个 --append
  - --filter-zero-u：剔除 |u_mean|<0.5 且 u_std<0.01 的段（left_turn 类）
  - --min-u-std：剔除 u_std 低于阈值的直线段
  - --report：打印合并前后段统计

用法:
  python3 scripts/merge_supplement_npz.py \\
      --base koopman_train.npz \\
      --append koopman_train_supplement.npz \\
      --out koopman_train_merged_v2.npz \\
      --filter-zero-u --min-u-std 0.05 --report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_datas(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)["datas"]


def seg_dynamics(seg: dict) -> tuple[float, float, float, float]:
    u = np.asarray(seg["Vel"][0], dtype=np.float64)
    v = np.asarray(seg["Vel"][1], dtype=np.float64)
    r = np.asarray(seg["pqr"][0], dtype=np.float64)
    return float(u.mean()), float(u.std()), float(v.std()), float(r.std())


def should_drop(
    seg: dict,
    filter_zero_u: bool,
    min_u_std: float | None,
    min_len: int,
) -> bool:
    T = int(seg["len"])
    if T < min_len:
        return True
    u_mean, u_std, v_std, r_std = seg_dynamics(seg)
    if filter_zero_u and abs(u_mean) < 0.5 and u_std < 0.01:
        return True
    if min_u_std is not None and u_std < min_u_std and v_std < 1e-4 and r_std < 1e-4:
        return True
    return False


def filter_segments(
    datas: np.ndarray,
    filter_zero_u: bool,
    min_u_std: float | None,
    min_len: int,
    tag: str,
) -> np.ndarray:
    kept = []
    dropped = 0
    for seg in datas:
        if should_drop(seg, filter_zero_u, min_u_std, min_len):
            dropped += 1
        else:
            kept.append(seg)
    if dropped:
        print(f"  [{tag}] dropped {dropped} segments, kept {len(kept)}")
    return np.array(kept, dtype=object)


def report_datas(datas: np.ndarray, title: str) -> None:
    print(f"\n--- {title} ({len(datas)} segments) ---")
    u_stds = []
    for i, seg in enumerate(datas):
        um, us, vs, rs = seg_dynamics(seg)
        u_stds.append(us)
        if i < 5 or i >= len(datas) - 2:
            print(
                f"  seg{i}: T={seg['len']} u_mean={um:.3f} u_std={us:.3f} "
                f"v_std={vs:.4f} r_std={rs:.4f}"
            )
    if len(datas) > 7:
        print("  ...")
    if u_stds:
        print(
            f"  u_std: min={min(u_stds):.4f} max={max(u_stds):.4f} "
            f"mean={np.mean(u_stds):.4f}"
        )


def merge_npz_files(
    base_path: str,
    append_paths: list[str],
    output_path: str,
    filter_zero_u: bool = False,
    min_u_std: float | None = None,
    min_len: int = 21,
    do_report: bool = False,
) -> None:
    print(f"Base: {base_path}")
    merged = list(load_datas(base_path))
    if do_report:
        report_datas(np.array(merged, dtype=object), "base")

    for ap in append_paths:
        print(f"Append: {ap}")
        extra = list(load_datas(ap))
        if do_report:
            report_datas(np.array(extra, dtype=object), Path(ap).name)
        merged.extend(extra)

    merged_arr = np.array(merged, dtype=object)
    if filter_zero_u or min_u_std is not None:
        merged_arr = filter_segments(
            merged_arr,
            filter_zero_u=filter_zero_u,
            min_u_std=min_u_std,
            min_len=min_len,
            tag="merged",
        )

    np.savez_compressed(output_path, datas=merged_arr)
    print(f"\nSaved {output_path} | total segments = {len(merged_arr)}")
    if do_report:
        report_datas(merged_arr, "merged output")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge Koopman NPZ datasets with optional filters")
    p.add_argument("--base", type=str, required=True, help="基础 NPZ（如 koopman_train.npz）")
    p.add_argument(
        "--append",
        type=str,
        action="append",
        default=[],
        help="追加 NPZ，可多次指定",
    )
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--filter-zero-u",
        action="store_true",
        help="剔除 |u_mean|<0.5 且 u_std<0.01（近似 left_turn 原地转）",
    )
    p.add_argument(
        "--min-u-std",
        type=float,
        default=None,
        help="与近零 v_std/r_std 联用时剔除过低 u_std 直线段",
    )
    p.add_argument("--min-len", type=int, default=21, help="最短段长（> pred_len_max）")
    p.add_argument("--report", action="store_true")
    args = p.parse_args()

    if not args.append:
        print("Warning: no --append files; only base (+ filter) will be written.")

    merge_npz_files(
        args.base,
        args.append,
        args.out,
        filter_zero_u=args.filter_zero_u,
        min_u_std=args.min_u_std,
        min_len=args.min_len,
        do_report=args.report,
    )


if __name__ == "__main__":
    main()
