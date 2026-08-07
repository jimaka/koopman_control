#!/usr/bin/env python3
"""2D 常速模型卡尔曼滤波演示。

状态 [px, vx, py, vy]，观测位置 [px, py]。

用法::

    python3 scripts/demo_kalman.py
    python3 scripts/demo_kalman.py --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koopman.estimation import LinearKalmanFilter


def constant_velocity_matrices(dt: float, q_std: float, r_std: float):
    """常速模型 F, H, Q, R。"""
    F = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    # 连续白噪声加速度离散化的简化 Q（按速度随机游走）
    q = q_std**2
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0.0, 0.0],
            [dt**2 / 2, dt, 0.0, 0.0],
            [0.0, 0.0, dt**3 / 3, dt**2 / 2],
            [0.0, 0.0, dt**2 / 2, dt],
        ]
    )
    R = (r_std**2) * np.eye(2)
    return F, H, Q, R


def simulate(
    T: int,
    dt: float,
    process_std: float,
    meas_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    F, H, Q, R = constant_velocity_matrices(dt, process_std, meas_std)
    x = np.array([0.0, 1.0, 0.0, 0.5])
    truth = np.zeros((T, 4))
    zs = np.zeros((T, 2))
    for k in range(T):
        w = rng.multivariate_normal(np.zeros(4), Q)
        x = F @ x + w
        truth[k] = x
        v = rng.multivariate_normal(np.zeros(2), R)
        zs[k] = H @ x + v
    return truth, zs


def main() -> None:
    parser = argparse.ArgumentParser(description="2D 常速卡尔曼滤波演示")
    parser.add_argument("--T", type=int, default=100, help="步数")
    parser.add_argument("--dt", type=float, default=0.1, help="采样周期 [s]")
    parser.add_argument("--q_std", type=float, default=0.5, help="过程噪声强度")
    parser.add_argument("--r_std", type=float, default=1.0, help="观测噪声标准差")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true", help="绘制轨迹")
    args = parser.parse_args()

    truth, zs = simulate(args.T, args.dt, args.q_std, args.r_std, args.seed)
    F, H, Q, R = constant_velocity_matrices(args.dt, args.q_std, args.r_std)
    x0 = np.array([zs[0, 0], 0.0, zs[0, 1], 0.0])
    P0 = np.diag([10.0, 10.0, 10.0, 10.0])
    kf = LinearKalmanFilter(F, H, Q, R, x0, P0, joseph=True)
    out = kf.filter(zs)

    pos_err = out["x"][:, [0, 2]] - truth[:, [0, 2]]
    meas_err = zs - truth[:, [0, 2]]
    rmse_filt = float(np.sqrt(np.mean(pos_err**2)))
    rmse_meas = float(np.sqrt(np.mean(meas_err**2)))
    print(f"steps={args.T}  dt={args.dt}")
    print(f"position RMSE (raw measurements): {rmse_meas:.4f}")
    print(f"position RMSE (Kalman filter):    {rmse_filt:.4f}")
    print(f"improvement: {(1.0 - rmse_filt / rmse_meas) * 100:.1f}%")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.plot(truth[:, 0], truth[:, 2], "k-", label="truth", linewidth=2)
        ax.plot(zs[:, 0], zs[:, 1], "C1.", label="measurements", alpha=0.5)
        ax.plot(out["x"][:, 0], out["x"][:, 2], "C0-", label="KF", linewidth=1.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("px")
        ax.set_ylabel("py")
        ax.legend()
        ax.set_title("2D constant-velocity Kalman filter")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_path = ROOT / "eval_out" / "kalman_demo.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        print(f"saved figure: {out_path}")
        plt.show()


if __name__ == "__main__":
    main()
