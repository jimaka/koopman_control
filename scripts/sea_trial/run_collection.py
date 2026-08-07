#!/usr/bin/env python3
"""船舶数据采集运行脚本（开环机动调度）。

覆盖训练 / 验证 / 测试所需全部辨识动作，时间表与 ``scripts/data/bag_test.py``、
``split_high_density_bag.py`` 对齐，便于事后切段。

用法示例::

    # 打印完整 dense 日程（不发指令）
    python3 scripts/sea_trial/run_collection.py --profile dense --dry-run

    # 短冒烟（约 8 分钟，每种机动各一段）
    python3 scripts/sea_trial/run_collection.py --profile smoke --dry-run

    # ROS 实船开环下发（需 elane_msgs + 接管权）
    python3 scripts/sea_trial/run_collection.py --profile smoke --ros --rate 2

    # 导出日程 JSON（给试验员 / 与后处理对齐）
    python3 scripts/sea_trial/run_collection.py --profile dense --export-schedule /tmp/schedule.json

必录话题（rosbag）::

    /localization/fusion_pose
    /system/chassis_feedback
    /control/control_cmd          # 建议：分析延迟
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sea_trial.maneuvers import (  # noqa: E402
    GENERATORS,
    apply_limits,
    rate_limit,
)
from scripts.sea_trial.schedule import (  # noqa: E402
    MANEUVER_CATALOG,
    Phase,
    ProfileName,
    build_schedule,
    schedule_summary,
)

# Global stop flag for Ctrl+C → zero thrust
_STOP = False


def _on_sigint(_sig, _frame) -> None:
    global _STOP
    _STOP = True
    print("\n[ABORT] Ctrl+C — will publish STOP and exit.", flush=True)


def bag_record_hint(out_bag: str) -> str:
    topics = [
        "/localization/fusion_pose",
        "/system/chassis_feedback",
        "/control/control_cmd",
        "/planning/trajectory",
        "/state/controller_state",
    ]
    return "rosbag record -O {bag} {topics}".format(
        bag=out_bag, topics=" ".join(topics)
    )


def print_catalog() -> None:
    print("机动动作目录（数据采集）\n")
    print(f"{'name':<14} {'中文':<12} {'建议时长s':>8} {'split':<10} generator")
    print("-" * 60)
    for m in MANEUVER_CATALOG:
        print(f"{m.name:<14} {m.label_zh:<12} {m.duration_s:8.0f} {m.split:<10} {m.generator}")


def print_schedule(phases: List[Phase]) -> None:
    summary = schedule_summary(phases)
    print(
        f"日程：{summary['n_phases']} 段，总时长 {summary['total_s']:.0f} s "
        f"({summary['total_h']:.2f} h)\n"
    )
    print(f"{'#':>3} {'t0':>8} {'t1':>8} {'dur':>6} {'split':<10} {'name':<12} 中文")
    print("-" * 72)
    for i, p in enumerate(phases):
        print(
            f"{i:3d} {p.t0:8.1f} {p.t1:8.1f} {p.duration_s:6.1f} "
            f"{p.split:<10} {p.name:<12} {p.label_zh}"
        )
    print("\n按机动合计 (s):")
    for k, v in sorted(summary["by_name_s"].items(), key=lambda x: -x[1]):
        print(f"  {k:<14} {v:8.1f}")
    print("\n按划分合计 (s):")
    for k, v in summary["by_split_s"].items():
        print(f"  {k:<14} {v:8.1f}")


def export_schedule(phases: List[Phase], path: Path, profile: str) -> None:
    payload = {
        "profile": profile,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": schedule_summary(phases),
        "topics_required": [
            "/localization/fusion_pose",
            "/system/chassis_feedback",
        ],
        "topics_recommended": [
            "/control/control_cmd",
            "/planning/trajectory",
            "/state/controller_state",
        ],
        "postprocess": {
            "dense": "scripts/data/split_high_density_bag.py",
            "standard": "scripts/data/bag_test.py",
            "left_turn": "scripts/data/extract_left_turn.py",
            "merge": "scripts/data/merge_npz.py",
            "check": "scripts/data/check_dataset.py",
        },
        "phases": [p.to_dict() for p in phases],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote schedule → {path}")


def _make_cmd_msg(port_th, port_ang, stbd_th, stbd_ang, rospy, ControlCmd):
    msg = ControlCmd()
    msg.header.stamp = rospy.Time.now()
    msg.port_thruster_throttle = float(port_th)
    msg.port_thruster_angle = float(port_ang)
    msg.starboard_thruster_throttle = float(stbd_th)
    msg.starboard_thruster_angle = float(stbd_ang)
    return msg


def publish_stop(pub, rospy, ControlCmd) -> None:
    msg = _make_cmd_msg(0, 0, 0, 0, rospy, ControlCmd)
    for _ in range(5):
        pub.publish(msg)
        rospy.sleep(0.1)


def run_ros(
    phases: List[Phase],
    rate_hz: float,
    throttle_max: float,
    rudder_max: float,
    start_index: int,
    confirm: bool,
) -> int:
    try:
        import rospy
        from elane_msgs.msg import ControlCmd
    except ImportError as e:
        print(
            "ROS / elane_msgs 不可用。请在 Elane ROS 环境中运行，或先 --dry-run。\n"
            f"ImportError: {e}",
            file=sys.stderr,
        )
        return 2

    if confirm:
        ans = input(
            "即将向 /control/control_cmd 开环下发采集指令。确认船舶处于安全水域且已接管？[y/N] "
        ).strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 1

    rospy.init_node("koopman_sea_trial_collection", anonymous=True)
    pub = rospy.Publisher("/control/control_cmd", ControlCmd, queue_size=1)
    # wait for connection
    time.sleep(0.5)
    rate = rospy.Rate(rate_hz)
    dt = 1.0 / rate_hz
    prev = (0.0, 0.0, 0.0, 0.0)

    print(bag_record_hint(f"sea_trial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bag"))
    print("建议另开终端先启动 rosbag record，再继续本脚本。\n")

    wall0 = time.time()
    try:
        for idx, phase in enumerate(phases):
            if idx < start_index:
                continue
            if _STOP or rospy.is_shutdown():
                break
            gen = GENERATORS[phase.generator]
            local_t0 = time.time()
            print(
                f"[{idx}/{len(phases)-1}] {phase.label_zh} ({phase.name}) "
                f"schedule {phase.t0:.0f}-{phase.t1:.0f}s  dur={phase.duration_s:.0f}s",
                flush=True,
            )
            while not _STOP and not rospy.is_shutdown():
                elapsed = time.time() - local_t0
                if elapsed >= phase.duration_s:
                    break
                raw = gen(elapsed, duration=phase.duration_s)
                limited = apply_limits(raw, throttle_max, rudder_max)
                prev = rate_limit(prev, limited, dt)
                pub.publish(_make_cmd_msg(*prev, rospy=rospy, ControlCmd=ControlCmd))
                rate.sleep()
    finally:
        print("[STOP] zero thrust", flush=True)
        publish_stop(pub, rospy, ControlCmd)

    print(f"Finished. wall_time={time.time()-wall0:.1f}s")
    return 0


def run_dry(
    phases: List[Phase],
    rate_hz: float,
    sample_every_phase: bool,
) -> int:
    print_schedule(phases)
    print("\n" + bag_record_hint("sea_trial_YYYYMMDD_HHMMSS.bag"))
    if not sample_every_phase:
        return 0
    print("\n各机动首秒指令样例 [port_th, port_ang, stbd_th, stbd_ang]:")
    for p in phases:
        if p.generator == "idle":
            continue
        cmd = apply_limits(GENERATORS[p.generator](1.0, duration=p.duration_s))
        print(f"  {p.name:<12} t=1s → ({cmd[0]:6.1f}, {cmd[1]:6.1f}, {cmd[2]:6.1f}, {cmd[3]:6.1f})")
    # optional: write a tiny CSV preview of smoke profile
    if sample_every_phase and rate_hz > 0:
        print(f"\n(生成器已加载 {len(GENERATORS)} 种；rate={rate_hz} Hz 仅用于实船发布)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Koopman 船舶数据采集运行脚本：调度全部辨识机动并可选开环下发",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--profile",
        choices=("dense", "standard", "smoke", "left_turn"),
        default="smoke",
        help="dense=7h高密度(对齐 split_high_density_bag)；"
             "standard=4.1h(对齐 bag_test)；smoke=短冒烟；left_turn=左转补充",
    )
    p.add_argument("--list", action="store_true", help="列出机动目录后退出")
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="只打印日程/样例，不连接 ROS（默认）")
    p.add_argument("--ros", action="store_true",
                   help="向 /control/control_cmd 发布开环指令（关闭 dry-run）")
    p.add_argument("--no-confirm", action="store_true", help="ROS 模式跳过交互确认")
    p.add_argument("--rate", type=float, default=2.0, help="发布频率 [Hz]（实船控制环约 2 Hz）")
    p.add_argument("--throttle-max", type=float, default=100.0)
    p.add_argument("--rudder-max", type=float, default=35.0)
    p.add_argument("--start-index", type=int, default=0, help="从第 N 个 phase 续跑")
    p.add_argument("--export-schedule", type=str, default=None,
                   help="导出日程 JSON 路径")
    p.add_argument("--left-turn-segments", type=int, default=10)
    p.add_argument("--sample-cmds", action="store_true", default=True,
                   help="dry-run 时打印各机动样例指令")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGINT, _on_sigint)

    if args.list:
        print_catalog()
        return 0

    phases = build_schedule(
        args.profile,  # type: ignore[arg-type]
        n_segments=args.left_turn_segments,
    )

    if args.export_schedule:
        export_schedule(phases, Path(args.export_schedule), args.profile)

    use_ros = bool(args.ros)
    if use_ros:
        return run_ros(
            phases,
            rate_hz=args.rate,
            throttle_max=args.throttle_max,
            rudder_max=args.rudder_max,
            start_index=args.start_index,
            confirm=not args.no_confirm,
        )

    return run_dry(phases, rate_hz=args.rate, sample_every_phase=args.sample_cmds)


if __name__ == "__main__":
    raise SystemExit(main())
