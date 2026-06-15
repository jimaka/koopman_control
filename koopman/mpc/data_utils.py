"""MPC 数据与参考航迹工具（不含 Python 求解器；MPC 仅 C++ OSQP）。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MPCTrajectory:
    t: np.ndarray
    state: np.ndarray
    control: np.ndarray
    ref_state: np.ndarray
    cost_history: List[float] = field(default_factory=list)


def segment_to_state_ctrl(seg: dict) -> Tuple[np.ndarray, np.ndarray]:
    T = int(seg["len"])
    state = np.empty((T, 6), dtype=np.float32)
    state[:, 0] = seg["Pos"][0, :T]
    state[:, 1] = seg["Pos"][1, :T]
    state[:, 2] = seg["Euler"][2, :T]
    state[:, 3] = seg["Vel"][0, :T]
    state[:, 4] = seg["Vel"][1, :T]
    state[:, 5] = seg["pqr"][0, :T]
    ctrl = seg["Thrusters_CMD"][:, :T].T.astype(np.float32, copy=False)
    return state, ctrl


def make_line_reference(
    x0: float, y0: float, yaw0: float,
    u_ref: float = 2.0,
    length_m: float = 80.0,
    dt: float = 0.1,
) -> np.ndarray:
    n = max(2, int(length_m / max(abs(u_ref) * dt, 1e-3)))
    t = np.arange(n, dtype=np.float32)
    state = np.zeros((n, 6), dtype=np.float32)
    state[:, 0] = x0 + u_ref * np.cos(yaw0) * t * dt
    state[:, 1] = y0 + u_ref * np.sin(yaw0) * t * dt
    state[:, 2] = yaw0
    state[:, 3] = u_ref
    return state


def make_circle_reference(
    cx: float, cy: float, radius: float,
    speed: float = 1.5,
    dt: float = 0.1,
    n_turns: float = 1.0,
) -> np.ndarray:
    omega = speed / max(radius, 0.5)
    period = 2 * math.pi / omega
    n = max(10, int(n_turns * period / dt))
    t = np.arange(n, dtype=np.float32) * dt
    yaw = omega * t + math.pi / 2
    state = np.zeros((n, 6), dtype=np.float32)
    state[:, 0] = cx + radius * np.cos(omega * t)
    state[:, 1] = cy + radius * np.sin(omega * t)
    state[:, 2] = yaw
    state[:, 3] = speed
    state[:, 5] = omega
    return state


def tracking_metrics(traj: MPCTrajectory) -> Dict[str, float]:
    sim = traj.state
    ref = traj.ref_state
    n = min(len(sim), len(ref))
    sim, ref = sim[:n], ref[:n]
    xy_err = np.linalg.norm(sim[:, :2] - ref[:, :2], axis=1)
    yaw_err = np.abs(np.arctan2(np.sin(sim[:, 2] - ref[:, 2]), np.cos(sim[:, 2] - ref[:, 2])))
    return {
        "xy_rmse_m": float(np.sqrt(np.mean(xy_err ** 2))),
        "xy_max_m": float(np.max(xy_err)),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_err ** 2))),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw_err ** 2)))),
        "final_xy_err_m": float(xy_err[-1]),
    }
