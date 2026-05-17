#!/usr/bin/env python3
"""run_mpc_tracking.py — 使用训练好的 Koopman 模型做 MPC 航迹跟踪闭环仿真。

示例::

    # 跟踪测试集中某段的 GT 航迹（推荐）
    python3 run_mpc_tracking.py \\
        --ckpt checkpoints/koopman_v3a_best.pth \\
        --data koopman_test.npz --segment 0 --steps 150 \\
        --out_dir eval_out/mpc_seg0

    # 跟踪合成圆周路径
    python3 run_mpc_tracking.py --ref circle --steps 200 --out_dir eval_out/mpc_circle

    # 快速冒烟（30 步）
    python3 run_mpc_tracking.py --smoketest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mpc_koopman import (
    KoopmanMPC,
    MPCConfig,
    make_circle_reference,
    make_line_reference,
    segment_to_state_ctrl,
    tracking_metrics,
)


def _plot_results(traj, metrics: dict, out_dir: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    t = traj.t
    sim, ref = traj.state, traj.ref_state
    n = min(len(sim), len(ref))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=13)

    ax = axes[0, 0]
    ax.plot(ref[:n, 0], ref[:n, 1], "k--", lw=1.5, label="reference")
    ax.plot(sim[:n, 0], sim[:n, 1], "C0", lw=1.5, label="MPC")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory")
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    xy_err = np.linalg.norm(sim[:n, :2] - ref[:n, :2], axis=1)
    ax.plot(t[:n], xy_err, "C1")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("horizontal error [m]")
    ax.set_title(f"XY error (RMSE={metrics['xy_rmse_m']:.3f} m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t[:n], np.degrees(ref[:n, 2]), "k--", label="ref ψ")
    ax.plot(t[:n], np.degrees(sim[:n, 2]), "C2", label="MPC ψ")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("yaw [deg]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    labels = ["port_thr", "port_ang", "stbd_thr", "stbd_ang"]
    for i in range(4):
        ax.plot(t[: len(traj.control)], traj.control[:, i], label=labels[i])
    ax.set_xlabel("t [s]")
    ax.set_title("control")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "mpc_tracking_overview.png"), dpi=160)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t[:n], sim[:n, 3], label="u")
    ax2.plot(t[:n], sim[:n, 4], label="v")
    ax2.plot(t[:n], sim[:n, 5], label="r")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("vel [m/s] or [rad/s]")
    ax2.set_title("MPC predicted velocities")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "mpc_velocities.png"), dpi=160)
    plt.close(fig2)


def run_smoketest() -> int:
    cfg = MPCConfig(horizon=8, opt_iters=15, device="cpu")
    ckpt = "checkpoints/koopman_v3a_best.pth"
    if not os.path.isfile(ckpt):
        ckpt = "checkpoints/koopman_v3_best.pth"
    if not os.path.isfile(ckpt):
        print("[smoketest] SKIP — no v3/v3a checkpoint", file=sys.stderr)
        return 0
    mpc = KoopmanMPC.from_checkpoint(ckpt, cfg)
    ref = make_line_reference(0.0, 0.0, 0.0, u_ref=1.5, length_m=20.0, dt=cfg.dt)
    traj = mpc.simulate(ref[0], ref, max_steps=30)
    m = tracking_metrics(traj)
    assert m["xy_rmse_m"] < 50.0, m
    print(f"[smoketest] OK xy_rmse={m['xy_rmse_m']:.3f} m")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ckpt", type=str, default="checkpoints/koopman_v3a_best.pth")
    parser.add_argument("--data", type=str, default="koopman_test.npz",
                        help="ref=segment 时使用的 npz")
    parser.add_argument("--ref", choices=["segment", "line", "circle"], default="segment",
                        help="参考航迹类型")
    parser.add_argument("--segment", type=int, default=0, help="ref=segment 时段索引")
    parser.add_argument("--steps", type=int, default=150, help="闭环仿真步数")
    parser.add_argument("--horizon", type=int, default=20, help="MPC 预测步长")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--opt_iters", type=int, default=40, help="每步 Adam 迭代次数")
    parser.add_argument("--w_xy", type=float, default=10.0)
    parser.add_argument("--w_yaw", type=float, default=5.0)
    parser.add_argument("--w_vel", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="eval_out/mpc")
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()

    if args.smoketest:
        return run_smoketest()

    if not os.path.isfile(args.ckpt):
        print(f"checkpoint 不存在: {args.ckpt}", file=sys.stderr)
        return 1

    cfg = MPCConfig(
        horizon=args.horizon,
        dt=args.dt,
        w_xy=args.w_xy,
        w_yaw=args.w_yaw,
        w_vel=args.w_vel,
        opt_iters=args.opt_iters,
        device=args.device,
    )
    mpc = KoopmanMPC.from_checkpoint(args.ckpt, cfg)

    ref_ctrl = None
    if args.ref == "segment":
        raw = np.load(args.data, allow_pickle=True)["datas"]
        if args.segment < 0 or args.segment >= len(raw):
            print(f"segment 索引越界: 0..{len(raw)-1}", file=sys.stderr)
            return 1
        ref_state, ref_ctrl = segment_to_state_ctrl(raw[args.segment])
        title = f"MPC track segment {args.segment} ({args.data})"
    elif args.ref == "line":
        ref_state = make_line_reference(0.0, 0.0, 0.0, u_ref=2.0, length_m=100.0, dt=args.dt)
        title = "MPC track line"
    else:
        ref_state = make_circle_reference(0.0, 0.0, radius=25.0, speed=1.5, dt=args.dt)
        title = "MPC track circle"

    state0 = ref_state[0].copy()
    traj = mpc.simulate(state0, ref_state, ref_ctrl=ref_ctrl, max_steps=args.steps)
    metrics = tracking_metrics(traj)
    metrics["horizon"] = args.horizon
    metrics["steps"] = args.steps
    metrics["ckpt"] = args.ckpt

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "mpc_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        os.path.join(args.out_dir, "mpc_trajectory.npz"),
        t=traj.t,
        state=traj.state,
        control=traj.control,
        ref_state=traj.ref_state,
    )

    _plot_results(traj, metrics, args.out_dir, title)

    print("=== MPC TRACKING RESULT ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"  plots -> {args.out_dir}/mpc_tracking_overview.png")
    print("===========================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
