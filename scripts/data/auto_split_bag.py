#!/usr/bin/env python3
"""自动化 bag 切分与训练数据集生成。

替代 ``bag_test.py`` / ``split_high_density_bag.py`` 中手工维护的
``generate_exact_phases()``：采集日程由 ``scripts/sea_trial/schedule.py``
确定性生成（``run_collection.py --export-schedule`` 导出的 JSON，或直接按
profile 重建），切分时按日程边界切片，并用控制命令的首个非零 onset 自动
对齐 bag 时钟与日程起点。

用法::

    # 采集时导出日程，录完包后一条命令生成数据集
    python3 scripts/sea_trial/run_collection.py --profile dense \
        --export-schedule /tmp/schedule.json
    python3 scripts/data/auto_split_bag.py --bag sea_trial.bag \
        --schedule /tmp/schedule.json

    # 没有日程 JSON 时按 profile 重建（须与采集时一致）
    python3 scripts/data/auto_split_bag.py --bag sea_trial.bag --profile dense

    # 自动对时失败时手动指定日程起点在 bag 中的时刻（bag 相对秒）
    python3 scripts/data/auto_split_bag.py --bag sea_trial.bag --profile dense \
        --time-offset 123.5

输出（默认 ``data/auto_<时间戳>/``）::

    koopman_train.npz / koopman_val.npz / koopman_test.npz / koopman_supplement.npz
    manifest.json   # 每段的机动名/划分/时间边界/丢弃原因，可追溯

段 dict 在既有六键（len/Pos/Vel/pqr/Euler/Thrusters_CMD）之外附带
name/split/t0/t1 元数据；训练加载器只读既有六键，不受影响。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

TOPIC_POSE = "/localization/fusion_pose"
TOPIC_THRUSTER = "/system/chassis_feedback"
TOPIC_CMD = "/control/control_cmd"

SPLIT_TO_FILENAME = {
    "train": "koopman_train.npz",
    "val": "koopman_val.npz",
    "test": "koopman_test.npz",
    "supplement": "koopman_supplement.npz",
}


# ---------------------------------------------------------------------------
# bag 读取与 10Hz 对齐（唯一依赖 ROS 的部分，rosbag 延迟导入）
# ---------------------------------------------------------------------------
def read_bag_raw(bag_path: str) -> Dict[str, list]:
    """读取 fusion_pose / chassis_feedback（有 control_cmd 也一并读）。"""
    import rosbag  # noqa: PLC0415 - 仅后处理机器上有 ROS 环境

    raw: Dict[str, list] = {
        "odom_ts": [], "Pos": [], "Vel": [], "Yaw": [], "pqr": [],
        "cmd_ts": [], "Thrusters_CMD": [],
    }
    topics = [TOPIC_POSE, TOPIC_THRUSTER, TOPIC_CMD]
    print(f">>> 正在读取 Bag: {bag_path}")
    with rosbag.Bag(bag_path) as bag:
        available = {t for t in topics if bag.get_message_count(t) > 0}
        if TOPIC_POSE not in available or TOPIC_THRUSTER not in available:
            raise RuntimeError(
                f"bag 缺少必录话题: 需要 {TOPIC_POSE} 与 {TOPIC_THRUSTER}，"
                f"实际只有 {sorted(available)}"
            )
        if TOPIC_CMD not in available:
            print(f"  (未录 {TOPIC_CMD}，自动对时退化为 chassis_feedback 油门 onset)")
        for topic, msg, t in bag.read_messages(topics=sorted(available)):
            if topic == TOPIC_POSE:
                raw["odom_ts"].append(t.to_sec())
                raw["Pos"].append([msg.position.x, msg.position.y])
                raw["Vel"].append([msg.velocity.x, msg.velocity.y])
                raw["pqr"].append([msg.angular_velocity.z])
                raw["Yaw"].append(msg.rpy.yaw)
            else:  # chassis_feedback 与 control_cmd 字段同名同序
                raw["cmd_ts"].append(t.to_sec())
                raw["Thrusters_CMD"].append([
                    msg.port_thruster_throttle, msg.port_thruster_angle,
                    msg.starboard_thruster_throttle, msg.starboard_thruster_angle,
                ])
    return raw


def align_timeline(
    raw: Dict[str, list],
    target_hz: float,
    max_gap_s: float,
) -> Dict[str, np.ndarray]:
    """把两路话题插值到统一 10Hz 网格，并给出缺口门控掩码。

    返回 dict 含 time/Pos/Vel/pqr/Euler/Thrusters_CMD 与 valid 布尔掩码：
    valid[i] 为 False 表示该网格点邻近的原始采样缺口超过 max_gap_s
    （对应 sim2real 指南的"缺口门控 禁外推"纪律）。
    """
    from scipy.interpolate import interp1d

    ts_odom = np.asarray(raw["odom_ts"], dtype=np.float64)
    ts_cmd = np.asarray(raw["cmd_ts"], dtype=np.float64)
    t0 = min(ts_odom[0], ts_cmd[0])
    ts_odom -= t0
    ts_cmd -= t0

    t_common = np.arange(
        max(ts_odom[0], ts_cmd[0]), min(ts_odom[-1], ts_cmd[-1]), 1.0 / target_hz
    )

    def _gap_mask(ts: np.ndarray) -> np.ndarray:
        """网格点落在宽度 > max_gap_s 的原始采样间隔内则为无效。"""
        idx = np.clip(np.searchsorted(ts, t_common) - 1, 0, len(ts) - 2)
        gap = ts[idx + 1] - ts[idx]
        return gap <= max_gap_s

    valid = _gap_mask(ts_odom) & _gap_mask(ts_cmd)

    aligned = {
        "time": t_common,
        "valid": valid,
        "Pos": interp1d(ts_odom, raw["Pos"], axis=0, fill_value="extrapolate")(t_common).astype(np.float32),
        "Vel": interp1d(ts_odom, raw["Vel"], axis=0, fill_value="extrapolate")(t_common).astype(np.float32),
        "pqr": interp1d(ts_odom, raw["pqr"], axis=0, fill_value="extrapolate")(t_common).astype(np.float32),
        "Euler": interp1d(ts_odom, np.unwrap(raw["Yaw"]), fill_value="extrapolate")(t_common).reshape(-1, 1).astype(np.float32),
        "Thrusters_CMD": interp1d(ts_cmd, raw["Thrusters_CMD"], axis=0, fill_value="extrapolate")(t_common).astype(np.float32),
    }
    return aligned


# ---------------------------------------------------------------------------
# 控制命令 onset 自动对时（纯函数，可测）
# ---------------------------------------------------------------------------
def find_cmd_onset(
    time: np.ndarray,
    cmd: np.ndarray,
    throttle_eps: float = 1.0,
    angle_eps: float = 0.5,
    hold_s: float = 1.0,
    target_hz: float = 10.0,
) -> Optional[float]:
    """返回控制命令首次持续非零的时刻（bag 相对秒）；找不到返回 None。

    「持续」要求之后 hold_s 内至少 80% 采样保持非零，避免单个毛刺触发。
    四个 profile 的首段均为非 idle 机动（fwd/left_turn），onset ≈ 日程 t=0。
    """
    active = (np.abs(cmd[:, 0]) > throttle_eps) | (np.abs(cmd[:, 2]) > throttle_eps)
    active |= (np.abs(cmd[:, 1]) > angle_eps) | (np.abs(cmd[:, 3]) > angle_eps)
    hold_n = max(int(round(hold_s * target_hz)), 1)
    for i in np.flatnonzero(active):
        j = min(i + hold_n, len(active))
        if active[i:j].mean() >= 0.8:
            return float(time[i])
    return None


# ---------------------------------------------------------------------------
# 日程加载与切分（纯函数，可测）
# ---------------------------------------------------------------------------
def load_phases(
    schedule_json: Optional[str] = None,
    profile: Optional[str] = None,
) -> List[dict]:
    """从 run_collection 导出的 JSON 或按 profile 重建日程，统一成 dict 列表。"""
    if schedule_json:
        payload = json.loads(Path(schedule_json).read_text(encoding="utf-8"))
        phases = payload.get("phases")
        if not phases:
            raise ValueError(f"{schedule_json} 中没有 phases 字段")
        return phases
    if profile:
        from scripts.sea_trial.schedule import build_schedule

        return [p.to_dict() for p in build_schedule(profile)]  # type: ignore[arg-type]
    raise ValueError("必须给出 --schedule 或 --profile 之一")


def slice_phases(
    aligned: Dict[str, np.ndarray],
    phases: Sequence[dict],
    offset_s: float,
    min_seg_s: float = 20.0,
    target_hz: float = 10.0,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """按日程边界切片。返回 (按 split 分组的段列表, 被丢弃段及原因)。"""
    t_common = aligned["time"]
    min_frames = int(round(min_seg_s * target_hz))
    by_split: Dict[str, List[dict]] = {k: [] for k in SPLIT_TO_FILENAME}
    dropped: List[dict] = []

    for ph in phases:
        name = ph["name"]
        if ph.get("generator") == "idle" or name == "idle":
            continue  # 段间停泊间隙不入数据集
        split = ph.get("split", "train")
        if split not in by_split:
            split = "train"
        start_t = float(ph["t0"]) + offset_s
        end_t = float(ph["t1"]) + offset_s
        mask = (t_common >= start_t) & (t_common < end_t)
        if not np.any(mask):
            dropped.append({"name": name, "split": split, "t0": ph["t0"], "t1": ph["t1"],
                            "reason": "超出 bag 时间范围"})
            continue
        idx = np.where(mask)[0]
        n = idx[-1] - idx[0]
        if n < min_frames:
            dropped.append({"name": name, "split": split, "t0": ph["t0"], "t1": ph["t1"],
                            "reason": f"长度不足 {min_seg_s:.0f}s（{n / target_hz:.1f}s）"})
            continue
        if not aligned["valid"][idx[0]:idx[-1]].all():
            dropped.append({"name": name, "split": split, "t0": ph["t0"], "t1": ph["t1"],
                            "reason": "段内存在数据缺口（禁外推）"})
            continue

        seg = {
            "len": n,
            "Pos": aligned["Pos"][idx[0]:idx[-1]].T,
            "Vel": aligned["Vel"][idx[0]:idx[-1]].T,
            "pqr": aligned["pqr"][idx[0]:idx[-1]].T,
            "Thrusters_CMD": aligned["Thrusters_CMD"][idx[0]:idx[-1]].T,
            # 元数据：训练加载器只读上述既有键，以下字段仅用于追溯
            "name": name,
            "split": split,
            "t0": float(ph["t0"]),
            "t1": float(ph["t1"]),
        }
        euler_3d = np.zeros((3, n), dtype=np.float32)
        euler_3d[2, :] = aligned["Euler"][idx[0]:idx[-1]].flatten()
        seg["Euler"] = euler_3d
        by_split[split].append(seg)

    return by_split, dropped


def save_datasets(
    by_split: Dict[str, List[dict]],
    out_dir: Path,
) -> Dict[str, str]:
    """按 split 各存一个 npz；返回 split -> 文件路径。空 split 不写文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}
    for split, segs in by_split.items():
        if not segs:
            continue
        path = out_dir / SPLIT_TO_FILENAME[split]
        np.savez_compressed(str(path), datas=np.array(segs, dtype=object))
        written[split] = str(path)
        print(f"  - {split:<10} {len(segs):>3} 段 → {path}")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="日程驱动 + 控制命令对时的自动 bag 切分与数据集生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bag", required=True, help="rosbag 路径")
    p.add_argument("--schedule", default=None,
                   help="run_collection --export-schedule 导出的日程 JSON")
    p.add_argument("--profile", default=None,
                   choices=("dense", "standard", "smoke", "left_turn"),
                   help="无日程 JSON 时按 profile 重建（须与采集时一致）")
    p.add_argument("--out-dir", default=None,
                   help="输出目录（默认 data/auto_<时间戳>）")
    p.add_argument("--target-hz", type=float, default=10.0, help="重采样频率")
    p.add_argument("--min-seg-s", type=float, default=20.0, help="最短段长（秒）")
    p.add_argument("--max-gap-s", type=float, default=0.5,
                   help="原始采样允许的最大缺口，超过则整段丢弃（禁外推）")
    p.add_argument("--time-offset", type=float, default=None,
                   help="日程 t=0 在 bag 相对时间中的时刻；不给则按控制命令 onset 自动对齐")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.schedule and not args.profile:
        print("错误：必须给出 --schedule 或 --profile 之一", file=sys.stderr)
        return 2

    from koopman import paths as P

    phases = load_phases(args.schedule, args.profile)
    raw = read_bag_raw(args.bag)
    aligned = align_timeline(raw, args.target_hz, args.max_gap_s)

    if args.time_offset is not None:
        offset = float(args.time_offset)
        align_info = {"mode": "manual", "offset_s": offset}
    else:
        onset = find_cmd_onset(aligned["time"], aligned["Thrusters_CMD"],
                               target_hz=args.target_hz)
        if onset is None:
            print("错误：未检测到有效的控制命令 onset，自动对时失败。"
                  "请用 --time-offset 手动指定。", file=sys.stderr)
            return 1
        offset = onset - float(phases[0]["t0"])
        align_info = {"mode": "cmd_onset", "onset_s": onset, "offset_s": offset}
    print(f">>> 时间对齐: {align_info}")

    by_split, dropped = slice_phases(
        aligned, phases, offset_s=offset,
        min_seg_s=args.min_seg_s, target_hz=args.target_hz,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (
        P.DATA_DIR / f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f">>> 写出数据集 → {out_dir}")
    written = save_datasets(by_split, out_dir)
    if not written:
        print("错误：没有任何段通过质量门，未生成数据集。", file=sys.stderr)
        return 1

    manifest = {
        "bag": str(Path(args.bag).resolve()),
        "schedule": args.schedule,
        "profile": args.profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_hz": args.target_hz,
        "alignment": align_info,
        "counts": {k: len(v) for k, v in by_split.items()},
        "segments": [
            {"split": split, "name": s["name"], "t0": s["t0"], "t1": s["t1"], "len": s["len"]}
            for split, segs in by_split.items() for s in segs
        ],
        "dropped": dropped,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(manifest["counts"].values())
    print(f"\n✅ 自动切分完成：共 {total} 段（丢弃 {len(dropped)} 段，明细见 manifest.json）")
    for d in dropped:
        print(f"  [丢弃] {d['name']} t={d['t0']:.0f}-{d['t1']:.0f}s: {d['reason']}")
    print(f"合并进主训练集：python3 scripts/data/merge_npz.py（按需修改路径）")
    print(f"质量检查：python3 scripts/data/check_dataset.py --data {written.get('train', '')} --seg 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
