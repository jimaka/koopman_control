"""MPC 数据工具；求解器见 C++ `koopman_control`（OSQP）。"""
from koopman.mpc.data_utils import (
    MPCTrajectory,
    make_circle_reference,
    make_line_reference,
    segment_to_state_ctrl,
    tracking_metrics,
)

__all__ = [
    "MPCTrajectory",
    "make_circle_reference",
    "make_line_reference",
    "segment_to_state_ctrl",
    "tracking_metrics",
]
