#!/usr/bin/env python3
"""
将 1800s 补充航迹 ROS bag 转为 Koopman NPZ（段格式与 split_high_density_bag.py 一致）。

默认按 KoopmanSupplementVoyagePilot 的 9×200s 相位切分（1800s 录包）；
也可用 --split fixed 按固定 200s 窗口切（与 extract_left_turn.py 相同）。

用法:
  python3 extract_supplement_bag.py --bag /path/to/supplement_1800s.bag

  # 提取并合并进训练集（推荐：基于 koopman_train.npz，过滤零 u 段）
  python3 extract_supplement_bag.py --bag supplement_1800s.bag --merge --report

  # 仅提取
  python3 extract_supplement_bag.py --bag supplement_1800s.bag \\
      --out koopman_train_supplement.npz --split phase
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

TOPIC_POSE = "/localization/fusion_pose"
TOPIC_THRUSTER = "/system/chassis_feedback"

# 与 split_high_density_bag.py / extract_left_turn.py 相同
SEGMENT_LENGTH_DEFAULT = 200.0
TARGET_HZ_DEFAULT = 10.0
SUPPLEMENT_DURATION_DEFAULT = 1800.0


def generate_supplement_phases(
    duration_sec: float = SUPPLEMENT_DURATION_DEFAULT,
    segment_length: float = SEGMENT_LENGTH_DEFAULT,
) -> list[tuple[float, float, str]]:
    """
    与 koopman_supplement_voyage_pilot.hpp 前 9 段一致（1800s = 9×200s）。
    若录包为 2000s，将 duration_sec 设为 2000 可得 10 段。
    """
    stage_names = [
        "SUPP_U_CHIRP",
        "SUPP_ZIGZAG_HS",
        "SUPP_FIG8_SURGE",
        "SUPP_SPEED_RAMP",
        "SUPP_YAW_SURGE",
        "SUPP_U_CHIRP_2",
        "SUPP_ZIGZAG_HS_2",
        "SUPP_FIG8_SURGE_2",
        "SUPP_SPEED_RAMP_2",
        "SUPP_YAW_SURGE_2",
    ]
    n_seg = int(round(duration_sec / segment_length))
    n_seg = min(n_seg, len(stage_names))
    phases = []
    for i in range(n_seg):
        t0 = i * segment_length
        t1 = (i + 1) * segment_length
        phases.append((t0, t1, stage_names[i]))
    return phases


def _import_rosbag():
    try:
        import rosbag
    except ImportError as e:
        raise SystemExit(
            "需要 rosbag（ROS1）。请在有 rosbag 的环境中运行本脚本。"
        ) from e
    return rosbag


def read_bag_aligned(
    bag_path: str,
    target_hz: float = TARGET_HZ_DEFAULT,
) -> dict[str, np.ndarray]:
    """读取 bag 并对齐到统一 10Hz 时间轴（逻辑同 split_high_density_bag.py）。"""
    rosbag = _import_rosbag()
    raw = {
        "odom_ts": [],
        "Pos": [],
        "Vel": [],
        "Yaw": [],
        "pqr": [],
        "cmd_ts": [],
        "Thrusters_CMD": [],
    }
    print(f">>> 正在读取补充航迹 Bag: {bag_path}")

    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages(topics=[TOPIC_POSE, TOPIC_THRUSTER]):
        if topic == TOPIC_POSE:
            raw["odom_ts"].append(t.to_sec())
            raw["Pos"].append([msg.position.x, msg.position.y])
            raw["Vel"].append([msg.velocity.x, msg.velocity.y])
            raw["pqr"].append([msg.angular_velocity.z])
            raw["Yaw"].append(msg.rpy.yaw)
        elif topic == TOPIC_THRUSTER:
            raw["cmd_ts"].append(t.to_sec())
            raw["Thrusters_CMD"].append(
                [
                    msg.port_thruster_throttle,
                    msg.port_thruster_angle,
                    msg.starboard_thruster_throttle,
                    msg.starboard_thruster_angle,
                ]
            )
    bag.close()

    if not raw["odom_ts"] or not raw["cmd_ts"]:
        raise RuntimeError(f"Bag 中未读到 pose/thruster 话题: {bag_path}")

    ts_odom = np.asarray(raw["odom_ts"], dtype=np.float64)
    ts_cmd = np.asarray(raw["cmd_ts"], dtype=np.float64)
    t0 = min(ts_odom[0], ts_cmd[0])
    ts_odom -= t0
    ts_cmd -= t0

    t_start = max(ts_odom[0], ts_cmd[0])
    t_end = min(ts_odom[-1], ts_cmd[-1])
    t_common = np.arange(t_start, t_end, 1.0 / target_hz, dtype=np.float64)
    duration = float(t_common[-1] - t_common[0]) if len(t_common) > 1 else 0.0
    print(f"    对齐后时长: {duration:.1f}s | 采样点: {len(t_common)} @ {target_hz}Hz")

    aligned = {"time": t_common}
    aligned["Pos"] = (
        interp1d(ts_odom, raw["Pos"], axis=0, fill_value="extrapolate")(t_common)
        .astype(np.float32)
    )
    aligned["Vel"] = (
        interp1d(ts_odom, raw["Vel"], axis=0, fill_value="extrapolate")(t_common)
        .astype(np.float32)
    )
    aligned["pqr"] = (
        interp1d(ts_odom, raw["pqr"], axis=0, fill_value="extrapolate")(t_common)
        .astype(np.float32)
    )
    aligned["Euler_yaw"] = (
        interp1d(ts_odom, np.unwrap(raw["Yaw"]), fill_value="extrapolate")(t_common)
        .astype(np.float32)
    )
    aligned["Thrusters_CMD"] = (
        interp1d(ts_cmd, raw["Thrusters_CMD"], axis=0, fill_value="extrapolate")(
            t_common
        ).astype(np.float32)
    )
    return aligned


def _seg_dict_from_slice(aligned: dict[str, np.ndarray], i0: int, i1: int) -> dict:
    seg_len = i1 - i0
    euler = np.zeros((3, seg_len), dtype=np.float32)
    euler[2, :] = aligned["Euler_yaw"][i0:i1]
    return {
        "len": seg_len,
        "Pos": aligned["Pos"][i0:i1].T.copy(),
        "Vel": aligned["Vel"][i0:i1].T.copy(),
        "pqr": aligned["pqr"][i0:i1].T.copy(),
        "Thrusters_CMD": aligned["Thrusters_CMD"][i0:i1].T.copy(),
        "Euler": euler,
    }


def extract_segments_phase(
    aligned: dict[str, np.ndarray],
    phases: list[tuple[float, float, str]],
    target_hz: float,
    min_sec: float = 20.0,
) -> list[dict]:
    """按相位时间边界切分（同 split_high_density_bag.generate_exact_phases 用法）。"""
    t = aligned["time"]
    segs: list[dict] = []
    for start_t, end_t, name in phases:
        mask = (t >= start_t) & (t < end_t)
        if not np.any(mask):
            print(f"⚠️ 跳过空段 {name}: [{start_t}, {end_t})")
            continue
        idx = np.where(mask)[0]
        n_frames = int(idx[-1] - idx[0])
        if n_frames < target_hz * min_sec:
            print(f"⚠️ 跳过过短段 {name}: {n_frames / target_hz:.1f}s")
            continue
        seg = _seg_dict_from_slice(aligned, idx[0], idx[-1])
        seg["stage_name"] = name
        segs.append(seg)
    return segs


def extract_segments_fixed(
    aligned: dict[str, np.ndarray],
    segment_length: float,
    target_hz: float,
    min_fill_ratio: float = 0.9,
) -> list[dict]:
    """按固定时长切分（同 extract_left_turn.py）。"""
    t = aligned["time"]
    t0 = float(t[0])
    total = float(t[-1] - t0)
    n_seg = int(math.ceil(total / segment_length))
    segs: list[dict] = []

    for i in range(n_seg):
        start_t = t0 + i * segment_length
        end_t = start_t + segment_length
        mask = (t >= start_t) & (t < end_t)
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        n_frames = int(idx[-1] - idx[0])
        if n_frames < target_hz * segment_length * min_fill_ratio:
            print(
                f"⚠️ 忽略过短段 #{i + 1}: {n_frames / target_hz:.1f}s "
                f"(需要 ≥ {segment_length * min_fill_ratio:.0f}s)"
            )
            continue
        segs.append(_seg_dict_from_slice(aligned, idx[0], idx[-1]))
    return segs


def print_segment_stats(segs: list[dict], title: str) -> None:
    print(f"\n=== {title}: {len(segs)} 段 ===")
    for i, s in enumerate(segs):
        u = s["Vel"][0]
        v = s["Vel"][1]
        r = s["pqr"][0]
        stage = s.get("stage_name", "")
        extra = f" [{stage}]" if stage else ""
        print(
            f"  seg{i}: T={s['len']} u_mean={float(u.mean()):.3f} "
            f"u_std={float(u.std()):.3f} v_std={float(v.std()):.4f} "
            f"r_std={float(r.std()):.4f}{extra}"
        )


def process_supplement_bag(
    bag_path: str,
    output_npz: str,
    *,
    target_hz: float = TARGET_HZ_DEFAULT,
    segment_length: float = SEGMENT_LENGTH_DEFAULT,
    duration_sec: float = SUPPLEMENT_DURATION_DEFAULT,
    split: str = "phase",
    min_fill_ratio: float = 0.9,
    report: bool = True,
) -> list[dict]:
    aligned = read_bag_aligned(bag_path, target_hz=target_hz)
    actual_dur = float(aligned["time"][-1] - aligned["time"][0])

    if split == "phase":
        phases = generate_supplement_phases(duration_sec, segment_length)
        # 若实际录包短于标称，按实际时长截断相位表
        if actual_dur + 1.0 < duration_sec:
            print(
                f"    提示: 录包时长 {actual_dur:.1f}s < 标称 {duration_sec:.1f}s，"
                f"按实际时长生成相位边界"
            )
            phases = generate_supplement_phases(actual_dur, segment_length)
        segs = extract_segments_phase(aligned, phases, target_hz)
    elif split == "fixed":
        segs = extract_segments_fixed(
            aligned, segment_length, target_hz, min_fill_ratio=min_fill_ratio
        )
    else:
        raise ValueError(f"未知 split 模式: {split}")

    if not segs:
        raise RuntimeError("未提取到任何有效数据段，请检查 bag 时长与切分参数")

    for s in segs:
        s.pop("stage_name", None)

    np.savez_compressed(output_npz, datas=np.array(segs, dtype=object))
    print(f"✅ 已保存: {output_npz} ({len(segs)} 段)")
    if report:
        print_segment_stats(segs, Path(output_npz).name)
    return segs


def run_merge(
    base_npz: str,
    supplement_npz: str,
    out_npz: str,
    *,
    filter_zero_u: bool = True,
    report: bool = True,
) -> None:
    """调用 scripts/merge_supplement_npz.py 合并到训练集。"""
    repo_root = Path(__file__).resolve().parent
    merge_script = repo_root / "scripts" / "merge_supplement_npz.py"
    if not merge_script.is_file():
        raise FileNotFoundError(f"未找到合并脚本: {merge_script}")

    import subprocess

    cmd = [
        sys.executable,
        str(merge_script),
        "--base",
        base_npz,
        "--append",
        supplement_npz,
        "--out",
        out_npz,
    ]
    if filter_zero_u:
        cmd.append("--filter-zero-u")
    if report:
        cmd.append("--report")
    print(f"\n>>> 合并训练集: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(repo_root))


def main() -> None:
    p = argparse.ArgumentParser(
        description="补充航迹 ROS bag → koopman_train_supplement.npz"
    )
    p.add_argument("--bag", type=str, required=True, help="1800s 补充航迹 .bag 路径")
    p.add_argument(
        "--out",
        type=str,
        default="koopman_train_supplement.npz",
        help="输出 NPZ",
    )
    p.add_argument("--hz", type=float, default=TARGET_HZ_DEFAULT)
    p.add_argument("--segment_length", type=float, default=SEGMENT_LENGTH_DEFAULT)
    p.add_argument(
        "--duration",
        type=float,
        default=SUPPLEMENT_DURATION_DEFAULT,
        help="标称任务时长 (s)，phase 切分用；默认 1800",
    )
    p.add_argument(
        "--split",
        choices=("phase", "fixed"),
        default="phase",
        help="phase: 按 SupplementVoyage 相位; fixed: 固定 200s 窗",
    )
    p.add_argument("--min_fill_ratio", type=float, default=0.9)
    p.add_argument(
        "--report",
        dest="report",
        action="store_true",
        help="打印每段 u/v/r 统计（默认开启）",
    )
    p.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="不打印段统计",
    )
    p.set_defaults(report=True)
    p.add_argument(
        "--merge",
        action="store_true",
        help="提取后合并到训练集",
    )
    p.add_argument(
        "--base",
        type=str,
        default="koopman_train.npz",
        help="合并时的基础训练 NPZ（推荐 koopman_train，勿用含 left_turn 的 merged）",
    )
    p.add_argument(
        "--merged_out",
        type=str,
        default="koopman_train_merged_v2.npz",
        help="--merge 时输出的合并训练集",
    )
    p.add_argument(
        "--no-filter-zero-u",
        action="store_true",
        help="合并时不剔除 |u_mean|<0.5 且 u_std<0.01 的段",
    )
    args = p.parse_args()

    if not Path(args.bag).is_file():
        raise SystemExit(f"Bag 不存在: {args.bag}")

    process_supplement_bag(
        args.bag,
        args.out,
        target_hz=args.hz,
        segment_length=args.segment_length,
        duration_sec=args.duration,
        split=args.split,
        min_fill_ratio=args.min_fill_ratio,
        report=not args.no_report,
    )

    if args.merge:
        if not Path(args.base).is_file():
            raise SystemExit(f"基础 NPZ 不存在: {args.base}")
        run_merge(
            args.base,
            args.out,
            args.merged_out,
            filter_zero_u=not args.no_filter_zero_u,
            report=not args.no_report,
        )
        print(
            f"\n训练请使用: --train_data {args.merged_out}\n"
            f"（或替换 koopman_train_merged.npz）"
        )


if __name__ == "__main__":
    main()
