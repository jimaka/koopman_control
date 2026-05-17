"""mpc_koopman.py — 基于 Deep-Koopman 模型的 MPC 航迹跟踪。

使用训练好的 ``HorizontalKoopmanModel(V3)`` 作为预测模型，在 receding-horizon
框架下优化推进器指令，跟踪参考平面轨迹 ``(x, y, ψ)`` 及可选速度 ``(u, v, r)``。

与 ``eval_koopman.py`` 一致：
* 动力学在归一化 ``(u,v,r)`` 空间 rollout；
* 位姿由预测速度经船体坐标系欧拉积分得到（dt=0.1 s）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from koopman import evalkit as ek


@dataclass
class MPCConfig:
    """MPC 权重与求解器参数。"""

    horizon: int = 20
    dt: float = 0.1
    # 代价权重
    w_xy: float = 10.0
    w_yaw: float = 5.0
    w_vel: float = 0.5
    w_u: float = 1e-4
    w_du: float = 0.05
    # 求解
    opt_iters: int = 40
    opt_lr: float = 0.08
    # 控制盒约束 [port_thr, port_ang, stbd_thr, stbd_ang]
    u_min: Tuple[float, float, float, float] = (-100.0, -35.0, -100.0, -35.0)
    u_max: Tuple[float, float, float, float] = (100.0, 35.0, 100.0, 35.0)
    device: str = "cpu"


@dataclass
class MPCTrajectory:
    """闭环仿真记录。"""

    t: np.ndarray
    state: np.ndarray          # (T, 6)  [x,y,yaw,u,v,r]
    control: np.ndarray        # (T, 4)
    ref_state: np.ndarray      # (T, 6)
    cost_history: List[float] = field(default_factory=list)


def segment_to_state_ctrl(seg: dict) -> Tuple[np.ndarray, np.ndarray]:
    """将单段 npz dict 转为 (T,6) 状态与 (T,4) 控制。"""
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
    """生成直线匀速参考航迹 (T,6)。"""
    n = max(2, int(length_m / max(abs(u_ref) * dt, 1e-3)))
    t = np.arange(n, dtype=np.float32)
    state = np.zeros((n, 6), dtype=np.float32)
    state[:, 0] = x0 + u_ref * np.cos(yaw0) * t * dt
    state[:, 1] = y0 + u_ref * np.sin(yaw0) * t * dt
    state[:, 2] = yaw0
    state[:, 3] = u_ref
    state[:, 4] = 0.0
    state[:, 5] = 0.0
    return state


def make_circle_reference(
    cx: float, cy: float, radius: float,
    speed: float = 1.5,
    dt: float = 0.1,
    n_turns: float = 1.0,
) -> np.ndarray:
    """生成匀速圆周参考航迹 (T,6)。"""
    omega = speed / max(radius, 0.5)
    period = 2 * math.pi / omega
    n = max(10, int(n_turns * period / dt))
    t = np.arange(n, dtype=np.float32) * dt
    yaw = omega * t + math.pi / 2  # 切向
    state = np.zeros((n, 6), dtype=np.float32)
    state[:, 0] = cx + radius * np.cos(omega * t)
    state[:, 1] = cy + radius * np.sin(omega * t)
    state[:, 2] = yaw
    state[:, 3] = speed
    state[:, 4] = 0.0
    state[:, 5] = omega
    return state


class KoopmanMPC:
    """Koopman 线性化算子 + 位姿积分的 MPC 控制器。"""

    def __init__(
        self,
        model: nn.Module,
        stats: Dict[str, np.ndarray],
        cfg: Optional[MPCConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or MPCConfig()
        self.device = torch.device(self.cfg.device)

        sm = stats["state_mean"].astype(np.float32)
        ss = stats["state_std"].astype(np.float32)
        cm = stats["ctrl_mean"].astype(np.float32)
        cs = stats["ctrl_std"].astype(np.float32)

        self.dyn_mean = torch.tensor(sm[3:6], device=self.device)
        self.dyn_std = torch.tensor(ss[3:6], device=self.device)
        self.ctrl_mean = torch.tensor(cm, device=self.device)
        self.ctrl_std = torch.tensor(cs, device=self.device)

        self.u_min = torch.tensor(self.cfg.u_min, device=self.device, dtype=torch.float32)
        self.u_max = torch.tensor(self.cfg.u_max, device=self.device, dtype=torch.float32)

        self._u_warm: Optional[torch.Tensor] = None  # (H,4) 上一步解，用于 warm-start

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, cfg: Optional[MPCConfig] = None) -> "KoopmanMPC":
        device = (cfg.device if cfg else "cpu")
        model, stats = ek.load_model_from_ckpt(ckpt_path, torch.device(device))
        return cls(model, stats, cfg)

    def _normalize_u(self, u_phys: torch.Tensor) -> torch.Tensor:
        return (u_phys - self.ctrl_mean) / self.ctrl_std

    def _clamp_u(self, u: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(u, self.u_max), self.u_min)

    def rollout(
        self,
        state0: torch.Tensor,
        u_seq: torch.Tensor,
    ) -> torch.Tensor:
        """开环 rollout。

        Args:
            state0: (6,) 物理状态 [x,y,yaw,u,v,r]
            u_seq:  (H,4) 物理控制

        Returns:
            states: (H+1, 6) 含初值
        """
        H = u_seq.shape[0]
        dt = self.cfg.dt
        x, y, yaw = state0[0], state0[1], state0[2]
        dyn = state0[3:6]
        dyn_n = (dyn - self.dyn_mean) / self.dyn_std
        z = self.model.encode(dyn_n.unsqueeze(0)).squeeze(0)

        out = [state0]
        for k in range(H):
            u_n = self._normalize_u(u_seq[k])
            z = self.model.latent_step(z.unsqueeze(0), u_n.unsqueeze(0)).squeeze(0)
            dyn = self.model.reconstruct_state(z.unsqueeze(0)).squeeze(0)
            dyn = dyn * self.dyn_std + self.dyn_mean
            u, v, r = dyn[0], dyn[1], dyn[2]
            x = x + (u * torch.cos(yaw) - v * torch.sin(yaw)) * dt
            y = y + (u * torch.sin(yaw) + v * torch.cos(yaw)) * dt
            yaw = yaw + r * dt
            out.append(torch.stack([x, y, yaw, u, v, r]))
        return torch.stack(out, dim=0)

    def _mpc_cost(
        self,
        u_flat: torch.Tensor,
        state0: torch.Tensor,
        ref: torch.Tensor,
        u_prev: torch.Tensor,
    ) -> torch.Tensor:
        """标量代价（可微）。"""
        H = self.cfg.horizon
        u_seq = u_flat.view(H, 4)
        traj = self.rollout(state0, u_seq)
        # ref: (H+1, 6)
        err_xy = traj[:, 0:2] - ref[:, 0:2]
        err_yaw = torch.atan2(
            torch.sin(traj[:, 2] - ref[:, 2]),
            torch.cos(traj[:, 2] - ref[:, 2]),
        )
        err_vel = traj[:, 3:6] - ref[:, 3:6]

        c = (
            self.cfg.w_xy * (err_xy ** 2).sum()
            + self.cfg.w_yaw * (err_yaw ** 2).sum()
            + self.cfg.w_vel * (err_vel ** 2).sum()
            + self.cfg.w_u * (u_seq ** 2).sum()
        )
        du = u_seq - torch.cat([u_prev.unsqueeze(0), u_seq[:-1]], dim=0)
        c = c + self.cfg.w_du * (du ** 2).sum()
        return c

    def solve(
        self,
        state0: np.ndarray,
        ref_window: np.ndarray,
        u_init: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """求解长度为 horizon 的控制序列，返回 (u_opt, cost)。

        Args:
            state0: (6,) 当前状态
            ref_window: (H+1, 6) 参考轨迹窗口
            u_init: 可选 (H,4) 初值
        """
        cfg = self.cfg
        H = cfg.horizon
        s0 = torch.tensor(state0, device=self.device, dtype=torch.float32)
        ref = torch.tensor(ref_window, device=self.device, dtype=torch.float32)

        if u_init is not None:
            u0 = torch.tensor(u_init, device=self.device, dtype=torch.float32)
        elif self._u_warm is not None:
            # 移位 warm-start
            w = self._u_warm.detach().clone()
            u0 = torch.cat([w[1:], w[-1:]], dim=0)
        else:
            u0 = torch.zeros(H, 4, device=self.device)

        u0 = self._clamp_u(u0)
        u_prev = u0[0].detach().clone()
        u_param = u0.clone().requires_grad_(True)
        opt = torch.optim.Adam([u_param], lr=cfg.opt_lr)

        best_u, best_cost = u0.clone(), float("inf")
        for _ in range(cfg.opt_iters):
            opt.zero_grad()
            u_clamped = self._clamp_u(u_param)
            cost = self._mpc_cost(u_clamped.reshape(-1), s0, ref, u_prev)
            cost.backward()
            opt.step()
            cval = float(cost.detach().cpu())
            if cval < best_cost:
                best_cost = cval
                best_u = self._clamp_u(u_param).detach().clone()

        self._u_warm = best_u.clone()
        return best_u.cpu().numpy(), best_cost

    def simulate(
        self,
        state0: np.ndarray,
        ref_traj: np.ndarray,
        ref_ctrl: Optional[np.ndarray] = None,
        max_steps: Optional[int] = None,
    ) -> MPCTrajectory:
        """闭环 MPC 跟踪仿真。

        Args:
            state0: 初始状态 (6,)
            ref_traj: 完整参考 (T,6)
            ref_ctrl: 可选 (T,4)，用于第一帧 warm-start
            max_steps: 仿真步数，默认 len(ref)-1
        """
        T = ref_traj.shape[0]
        H = self.cfg.horizon
        n_sim = min(max_steps or (T - 1), T - 1)

        states = np.zeros((n_sim + 1, 6), dtype=np.float32)
        controls = np.zeros((n_sim, 4), dtype=np.float32)
        refs = ref_traj[: n_sim + 1].copy()
        states[0] = state0.astype(np.float32)
        costs: List[float] = []

        if ref_ctrl is not None and len(ref_ctrl) >= H:
            self._u_warm = torch.tensor(ref_ctrl[:H], dtype=torch.float32)

        for t in range(n_sim):
            end = min(t + H + 1, T)
            ref_win = ref_traj[t:end]
            if ref_win.shape[0] < H + 1:
                pad = np.tile(ref_traj[-1], (H + 1 - ref_win.shape[0], 1))
                ref_win = np.concatenate([ref_win, pad], axis=0)

            u_init = ref_ctrl[t : t + H] if ref_ctrl is not None and t + H <= len(ref_ctrl) else None
            u_opt, c = self.solve(states[t], ref_win, u_init=u_init)
            controls[t] = u_opt[0]
            costs.append(c)

            # 用最优控制的第 0 步推进真实（模型）状态
            s_next = self.rollout(
                torch.tensor(states[t], device=self.device),
                torch.tensor(u_opt[:1], device=self.device),
            )
            states[t + 1] = s_next[1].detach().cpu().numpy()

        return MPCTrajectory(
            t=np.arange(n_sim + 1, dtype=np.float32) * self.cfg.dt,
            state=states,
            control=controls,
            ref_state=refs,
            cost_history=costs,
        )


def tracking_metrics(traj: MPCTrajectory) -> Dict[str, float]:
    """计算跟踪 RMSE 等指标。"""
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
