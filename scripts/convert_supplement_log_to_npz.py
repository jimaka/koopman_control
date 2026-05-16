#!/usr/bin/env python3
"""
将 koopman_supplement_mission.cpp 输出的 CSV 转为 Koopman NPZ 段格式。

段字典键与 split_high_density_bag.py / extract_left_turn.py 一致：
  len, Pos(2,T), Vel(2,T), pqr(1,T), Euler(3,T), Thrusters_CMD(4,T)

用法:
  cd scripts && make
  # 舵手头文件: koopman_supplement_voyage_pilot.hpp
  # 类名: KoopmanSupplementVoyagePilot::update(heading_rad)
  ./koopman_supplement_mission -o supplement_mission_log.csv

  python3 scripts/convert_supplement_log_to_npz.py \\
      --csv supplement_mission_log.csv \\
      --out koopman_train_supplement.npz \\
      --segment_length 200 --hz 10

  python3 scripts/merge_supplement_npz.py \\
      --base koopman_train.npz \\
      --append koopman_train_supplement.npz \\
      --out koopman_train_merged_v2.npz \\
      --filter-zero-u
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def load_csv_log(csv_path: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """读取 C++ 仿真 CSV（数值列 + 可选 phase_id）。"""
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1, usecols=range(12))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    phase_id = None
    if arr.shape[1] >= 12:
        phase_id = arr[:, 11].astype(np.int32)
    return {
        "time": arr[:, 0].astype(np.float64),
        "Pos": arr[:, 1:3].astype(np.float32),
        "yaw": arr[:, 3].astype(np.float32),
        "Vel": arr[:, 4:6].astype(np.float32),
        "r": arr[:, 6].astype(np.float32),
        "Thrusters_CMD": arr[:, 7:11].astype(np.float32),
    }, phase_id


def resample_uniform(
    t: np.ndarray,
    arrays: dict[str, np.ndarray],
    hz: float,
) -> dict[str, np.ndarray]:
    """按固定 hz 重采样（若 CSV 已是 10Hz 则近似恒等）。"""
    from scipy.interpolate import interp1d

    t0, t1 = float(t[0]), float(t[-1])
    t_common = np.arange(t0, t1, 1.0 / hz, dtype=np.float64)
    out: dict[str, np.ndarray] = {"time": t_common}
    for key, val in arrays.items():
        if key == "time":
            continue
        if val.ndim == 1:
            out[key] = interp1d(t, val, fill_value="extrapolate")(t_common).astype(
                np.float32
            )
        else:
            out[key] = interp1d(t, val, axis=0, fill_value="extrapolate")(
                t_common
            ).astype(np.float32)
    return out


def aligned_to_segments(
    aligned: dict[str, np.ndarray],
    segment_length: float,
    hz: float,
    min_fill_ratio: float = 0.9,
) -> list[dict]:
    """按固定时长切分，逻辑对齐 extract_left_turn.py。"""
    t = aligned["time"]
    segs: list[dict] = []
    total = float(t[-1] - t[0])
    n_seg = int(math.ceil(total / segment_length))

    yaw_unwrapped = np.unwrap(aligned["yaw"].astype(np.float64)).astype(np.float32)

    for i in range(n_seg):
        start_t = t[0] + i * segment_length
        end_t = start_t + segment_length
        mask = (t >= start_t) & (t < end_t)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        n_frames = int(idx[-1] - idx[0])
        min_frames = int(hz * segment_length * min_fill_ratio)
        if n_frames < min_frames:
            continue

        sl = slice(idx[0], idx[-1])
        seg_len = n_frames
        pos = aligned["Pos"][sl].T  # (2, T)
        vel = aligned["Vel"][sl].T
        pqr = aligned["r"][sl].reshape(1, -1)
        thr = aligned["Thrusters_CMD"][sl].T  # (4, T)
        euler = np.zeros((3, seg_len), dtype=np.float32)
        euler[2, :] = yaw_unwrapped[sl]

        segs.append(
            {
                "len": seg_len,
                "Pos": pos.astype(np.float32, copy=False),
                "Vel": vel.astype(np.float32, copy=False),
                "pqr": pqr.astype(np.float32, copy=False),
                "Thrusters_CMD": thr.astype(np.float32, copy=False),
                "Euler": euler,
            }
        )
    return segs


def resample_phase_ids(
    t_src: np.ndarray,
    phase_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    """phase_id 用最近邻重采样到统一时间轴。"""
    out = np.empty(len(t_dst), dtype=np.int32)
    for i, td in enumerate(t_dst):
        j = int(np.argmin(np.abs(t_src - td)))
        out[i] = int(phase_src[j])
    return out


def segments_by_phase(
    aligned: dict[str, np.ndarray],
    phase_ids: np.ndarray,
    min_frames: int,
) -> list[dict]:
    """按 C++ 任务相位切段（默认每相位 200s）。"""
    yaw_unwrapped = np.unwrap(aligned["yaw"].astype(np.float64)).astype(np.float32)
    segs: list[dict] = []
    for pid in np.unique(phase_ids):
        mask = phase_ids == pid
        idx = np.where(mask)[0]
        if len(idx) < min_frames:
            continue
        sl = slice(idx[0], idx[-1])
        seg_len = int(idx[-1] - idx[0])
        if seg_len < min_frames:
            continue
        segs.append(
            {
                "len": seg_len,
                "Pos": aligned["Pos"][sl].T.astype(np.float32),
                "Vel": aligned["Vel"][sl].T.astype(np.float32),
                "pqr": aligned["r"][sl].reshape(1, -1).astype(np.float32),
                "Thrusters_CMD": aligned["Thrusters_CMD"][sl].T.astype(np.float32),
                "Euler": _euler_slice(yaw_unwrapped, sl, seg_len),
            }
        )
    return segs


def _euler_slice(yaw_unwrapped: np.ndarray, sl: slice, seg_len: int) -> np.ndarray:
    euler = np.zeros((3, seg_len), dtype=np.float32)
    euler[2, :] = yaw_unwrapped[sl]
    return euler


def print_segment_stats(segs: list[dict], tag: str) -> None:
    print(f"\n=== {tag}: {len(segs)} segments ===")
    for i, s in enumerate(segs):
        u = s["Vel"][0]
        v = s["Vel"][1]
        r = s["pqr"][0]
        print(
            f"  seg{i}: T={s['len']} u_mean={u.mean():.3f} u_std={u.std():.3f} "
            f"v_std={v.std():.3f} r_std={r.std():.4f}"
        )


def convert(
    csv_path: str,
    out_npz: str,
    segment_length: float,
    hz: float,
    split_mode: str,
) -> None:
    raw, phase_src = load_csv_log(csv_path)
    t = raw["time"]
    arrays = {
        "Pos": raw["Pos"],
        "yaw": raw["yaw"],
        "Vel": raw["Vel"],
        "r": raw["r"],
        "Thrusters_CMD": raw["Thrusters_CMD"],
    }
    aligned = resample_uniform(t, arrays, hz)

    min_frames = int(hz * 20)
    segs: list[dict] = []
    if split_mode == "phase" and phase_src is not None:
        phase_aligned = resample_phase_ids(t, phase_src, aligned["time"])
        segs = segments_by_phase(aligned, phase_aligned, min_frames)
    if not segs:
        segs = aligned_to_segments(aligned, segment_length, hz)

    if not segs:
        raise RuntimeError("No segments produced; check CSV length and parameters.")

    np.savez_compressed(out_npz, datas=np.array(segs, dtype=object))
    print(f"Saved {out_npz} ({len(segs)} segments)")
    print_segment_stats(segs, Path(out_npz).name)


def main() -> None:
    p = argparse.ArgumentParser(description="Convert supplement mission CSV to Koopman NPZ")
    p.add_argument("--csv", type=str, default="supplement_mission_log.csv")
    p.add_argument("--out", type=str, default="koopman_train_supplement.npz")
    p.add_argument("--segment_length", type=float, default=200.0)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument(
        "--split",
        choices=("fixed", "phase"),
        default="fixed",
        help="fixed: 按 segment_length 切; phase: 按 phase_id（实验性）",
    )
    args = p.parse_args()
    convert(args.csv, args.out, args.segment_length, args.hz, args.split)


if __name__ == "__main__":
    main()
