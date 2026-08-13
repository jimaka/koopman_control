#!/usr/bin/env python3
"""潜空间 Tier-2 位姿跟踪 NLP 的 SQP（Gauss-Newton）参考实现。

本模块是 `cpp/koopman_control` 中 Tier-1/Tier-2 求解链路的 **float64 参考实现**，
用途有三：

1. 把 C++ 侧隐含的优化问题写成显式 NLP（`LatentPoseNlp`），使「代价 / 梯度 /
   线性化 / 可行集」可被独立校验（见 `tests/test_sqp_latent_reference.py`）；
2. 提供带**步长控制**（Armijo 线搜索 + 信赖域）、**可行点投影**、**wrap 冻结**、
   **move-blocking 降维**的 SQP 求解器（`SqpSolver`），作为 C++ 改造的算法基线；
3. 复现当前 C++ 的定步长外迭代（`SqpScheme.baseline_cpp()`）以便逐项对照。

与 C++ 的对应关系：

| 本模块 | C++ |
|--------|-----|
| `LatentSystem.condensed` | `KoopmanLatentModel::precomputePredictionMatrices` |
| `LatentSystem.decode_physical / decode_jacobian_physical` | `KoopmanDecoder` |
| `LatentPoseNlp.pose_rollout` | `verify_pose_linearize.cpp::rolloutPose` |
| `LatentPoseNlp.linearize` | `buildPoseLinearization`（`pose_linearize.cpp`） |
| `LatentPoseNlp.gn_model` | `LatentMpcQpSolver::solve` 的 P/q 组装 |
| `SqpSolver.solve` | `KoopmanMpcController::solveStep` 的 SQP 外循环 |

数学推导见 `docs/v4训练与MPC任务拆解与SQP方案推导.md`。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LatentSystem",
    "MpcWeights",
    "MpcLimits",
    "Blocking",
    "LatentPoseNlp",
    "SqpScheme",
    "SqpResult",
    "SqpSolver",
    "solve_qp_box",
]


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_erf = np.vectorize(math.erf)


def gelu(x: np.ndarray) -> np.ndarray:
    """精确 GELU（erf 形式），与 C++ `detail::gelu` 一致。"""
    return 0.5 * x * (1.0 + _erf(x / _SQRT2))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """GELU 一阶导，与 C++ `detail::geluGrad` 一致。"""
    return 0.5 * (1.0 + _erf(x / _SQRT2)) + x * np.exp(-0.5 * x * x) * _INV_SQRT_2PI


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """把角度折叠到 (-pi, pi]，与 C++ `detail::wrapAngle` 一致。"""
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------------------
# 潜空间系统
# ---------------------------------------------------------------------------
@dataclass
class LatentSystem:
    """v4 潜空间仿射动力学 + decoder。

    z_{k+1} = A_bar z_k + B u_k + beta,   d_k = diag(dyn_std) MLP(z_k) + dyn_mean
    """

    A_bar: np.ndarray
    B: np.ndarray
    beta: np.ndarray
    dyn_mean: np.ndarray
    dyn_std: np.ndarray
    ctrl_mean: np.ndarray
    ctrl_std: np.ndarray
    decoder_layers: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]]

    @property
    def nz(self) -> int:
        return int(self.A_bar.shape[0])

    @property
    def nu(self) -> int:
        return int(self.B.shape[1])

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_torch_model(cls, model, stats: Dict[str, np.ndarray]) -> "LatentSystem":
        """从 v4 PyTorch 模型 + 训练统计量构造（等价于 export_v4_encode_weights.py）。"""
        import torch

        nz = int(model.latent_dim)
        A_bar = (model.A.weight.detach().cpu() + torch.eye(nz)).numpy().astype(np.float64)
        beta = model.A.bias.detach().cpu().numpy().astype(np.float64)
        B = model.B.weight.detach().cpu().numpy().astype(np.float64)

        layers: List[Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]] = []
        for layer in model.decoder_mlp:
            if isinstance(layer, torch.nn.Linear):
                layers.append(
                    (
                        "linear",
                        layer.weight.detach().cpu().numpy().astype(np.float64),
                        layer.bias.detach().cpu().numpy().astype(np.float64),
                    )
                )
            elif isinstance(layer, torch.nn.GELU):
                layers.append(("gelu", None, None))
            else:  # pragma: no cover - 结构变更时立刻暴露
                raise TypeError(f"unexpected decoder layer {type(layer)}")

        return cls(
            A_bar=A_bar,
            B=B,
            beta=beta,
            dyn_mean=np.asarray(stats["state_mean"][3:6], dtype=np.float64),
            dyn_std=np.asarray(stats["state_std"][3:6], dtype=np.float64),
            ctrl_mean=np.asarray(stats["ctrl_mean"], dtype=np.float64),
            ctrl_std=np.asarray(stats["ctrl_std"], dtype=np.float64),
            decoder_layers=layers,
        )

    # -- condensed 预测矩阵 -------------------------------------------------
    def condensed(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (Gamma, Theta, xi)，使 Z = Gamma z0 + Theta U + xi。"""
        nz, nu, n = self.nz, self.nu, int(horizon)
        a_pow = [np.eye(nz)]
        for _ in range(n):
            a_pow.append(a_pow[-1] @ self.A_bar)

        Gamma = np.zeros((nz * n, nz))
        Theta = np.zeros((nz * n, nu * n))
        xi = np.zeros(nz * n)
        for k in range(1, n + 1):
            row = (k - 1) * nz
            Gamma[row : row + nz, :] = a_pow[k]
            xi[row : row + nz] = sum(a_pow[i] @ self.beta for i in range(k))
            for j in range(k):
                Theta[row : row + nz, j * nu : (j + 1) * nu] = a_pow[k - j - 1] @ self.B
        return Gamma, Theta, xi

    # -- decoder ------------------------------------------------------------
    def decode_physical(self, z: np.ndarray) -> np.ndarray:
        h = np.asarray(z, dtype=np.float64)
        for kind, w, b in self.decoder_layers:
            h = w @ h + b if kind == "linear" else gelu(h)
        return h * self.dyn_std + self.dyn_mean

    def decode_jacobian_physical(self, z: np.ndarray) -> np.ndarray:
        """物理速度对 z 的 Jacobian（3 x nz），与 C++ `jacobianPhysical` 一致。"""
        h = np.asarray(z, dtype=np.float64)
        jac = np.eye(self.nz)
        for kind, w, b in self.decoder_layers:
            if kind == "linear":
                jac = w @ jac
                h = w @ h + b
            else:
                g = gelu_grad(h)
                jac = g[:, None] * jac
                h = gelu(h)
        return self.dyn_std[:, None] * jac

    # -- 归一化 -------------------------------------------------------------
    def normalize_control(self, u_phys: np.ndarray) -> np.ndarray:
        return (np.asarray(u_phys, dtype=np.float64) - self.ctrl_mean) / self.ctrl_std

    def denormalize_control(self, u_tilde: np.ndarray) -> np.ndarray:
        return np.asarray(u_tilde, dtype=np.float64) * self.ctrl_std + self.ctrl_mean


# ---------------------------------------------------------------------------
# 权重 / 约束 / move-blocking
# ---------------------------------------------------------------------------
@dataclass
class MpcWeights:
    """代价权重，与 `mpc_config.yaml` 同名字段一致。"""

    w_z: float = 1.0
    w_u: float = 1e-4
    w_du: float = 0.05
    w_xy: float = 0.0
    w_yaw: float = 0.0


@dataclass
class MpcLimits:
    """物理量约束，运行时映射到归一化空间。"""

    u_min: np.ndarray = field(default_factory=lambda: np.array([-100.0, -35.0, -100.0, -35.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([100.0, 35.0, 100.0, 35.0]))
    du_max: np.ndarray = field(default_factory=lambda: np.array([15.0, 3.5, 15.0, 3.5]))

    def tilde(self, sys: LatentSystem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lo = (np.asarray(self.u_min, dtype=np.float64) - sys.ctrl_mean) / sys.ctrl_std
        hi = (np.asarray(self.u_max, dtype=np.float64) - sys.ctrl_mean) / sys.ctrl_std
        dmax = np.where(
            np.asarray(self.du_max, dtype=np.float64) > 0.0,
            np.asarray(self.du_max, dtype=np.float64) / sys.ctrl_std,
            np.inf,
        )
        return lo, hi, dmax


@dataclass
class Blocking:
    """move-blocking：前 opt_control_steps 步自由（按 hold 分块），其后保持常值。"""

    horizon: int
    opt_control_steps: int
    hold: int = 1

    def __post_init__(self) -> None:
        if self.horizon % self.hold != 0:
            raise ValueError("horizon must be divisible by hold")

    @property
    def n_free_blocks(self) -> int:
        n_blk = self.horizon // self.hold
        opt_blk = (self.opt_control_steps + self.hold - 1) // self.hold
        return max(1, min(n_blk, opt_blk))

    def block_of_step(self, k: int) -> int:
        return min(k // self.hold, self.n_free_blocks - 1)

    def selector(self, nu: int) -> np.ndarray:
        """U = M V，M 形状 (horizon*nu, n_free_blocks*nu)。"""
        rows = self.horizon * nu
        cols = self.n_free_blocks * nu
        M = np.zeros((rows, cols))
        for k in range(self.horizon):
            b = self.block_of_step(k)
            for j in range(nu):
                M[k * nu + j, b * nu + j] = 1.0
        return M

    def expand_posthoc(self, u_full: np.ndarray, nu: int) -> np.ndarray:
        """复刻 C++ `expandToFull`：解出全 horizon 后再截断 / 保持（事后覆写）。"""
        out = np.array(u_full, dtype=np.float64, copy=True)
        n_opt = self.n_free_blocks * self.hold
        if n_opt < self.horizon:
            tail = out[(n_opt - 1) * nu : n_opt * nu].copy()
            for k in range(n_opt, self.horizon):
                out[k * nu : (k + 1) * nu] = tail
        if self.hold > 1:
            for k in range(self.horizon):
                leader = (k // self.hold) * self.hold
                out[k * nu : (k + 1) * nu] = out[leader * nu : (leader + 1) * nu]
        return out


# ---------------------------------------------------------------------------
# NLP
# ---------------------------------------------------------------------------
@dataclass
class LatentPoseNlp:
    """一个控制周期内 MPC 求解的精确 NLP。

    决策变量 U = [u~_0; ...; u~_{N-1}] ∈ R^{N*nu}（归一化控制），代价

        F(U) = w_z ||Theta U + e_free||^2
             + w_u ||U||^2
             + w_du ||D U||^2
             + || W^{1/2} g(U) ||^2

    其中 e_free = Gamma z0 + xi - Z_ref，g(U) 为位姿残差（非线性），
    W = diag(w_xy, w_xy, w_yaw) 逐步重复。可行集为盒约束 + 速率约束（多面体）。
    """

    sys: LatentSystem
    horizon: int
    dt: float
    weights: MpcWeights
    z0: np.ndarray
    z_ref_stack: np.ndarray
    pose0: np.ndarray
    pose_ref: np.ndarray
    u_prev_tilde: np.ndarray
    limits: MpcLimits = field(default_factory=MpcLimits)
    blocking: Optional[Blocking] = None

    def __post_init__(self) -> None:
        self.nu = self.sys.nu
        self.nz = self.sys.nz
        self.nvar = self.horizon * self.nu
        self.Gamma, self.Theta, self.xi = self.sys.condensed(self.horizon)
        self.z_free = self.Gamma @ np.asarray(self.z0, dtype=np.float64) + self.xi
        self.e_free = self.z_free - np.asarray(self.z_ref_stack, dtype=np.float64)
        self.D = self._diff_matrix()
        self.W = self._pose_weight_vector()
        if self.blocking is None:
            self.blocking = Blocking(self.horizon, self.horizon, 1)
        self.lo_t, self.hi_t, self.dmax_t = self.limits.tilde(self.sys)

    # -- 结构矩阵 ------------------------------------------------------------
    def _diff_matrix(self) -> np.ndarray:
        """D U 的第 k 行块为 u~_k - u~_{k-1}（k=1..N-1），与 C++ `D_du` 一致。"""
        n, nu = self.horizon, self.nu
        D = np.zeros((max(0, nu * (n - 1)), nu * n))
        for k in range(1, n):
            for j in range(nu):
                D[(k - 1) * nu + j, k * nu + j] = 1.0
                D[(k - 1) * nu + j, (k - 1) * nu + j] = -1.0
        return D

    def _pose_weight_vector(self) -> np.ndarray:
        w = np.zeros(3 * self.horizon)
        w[0::3] = self.weights.w_xy
        w[1::3] = self.weights.w_xy
        w[2::3] = self.weights.w_yaw
        return w

    @property
    def pose_active(self) -> bool:
        return self.weights.w_xy > 0.0 or self.weights.w_yaw > 0.0

    # -- rollout / 残差 ------------------------------------------------------
    def latent_stack(self, U: np.ndarray) -> np.ndarray:
        return self.z_free + self.Theta @ np.asarray(U, dtype=np.float64)

    def pose_rollout(self, U: np.ndarray) -> np.ndarray:
        """船体系欧拉积分（速度 d_m 配艏向 psi_{m-1}），返回 p_1..p_N。"""
        Z = self.latent_stack(U)
        x, y, yaw = (float(v) for v in self.pose0)
        poses = np.zeros((self.horizon, 3))
        for m in range(1, self.horizon + 1):
            z_m = Z[(m - 1) * self.nz : m * self.nz]
            um, vm, rm = self.sys.decode_physical(z_m)
            c, s = math.cos(yaw), math.sin(yaw)
            x += (um * c - vm * s) * self.dt
            y += (um * s + vm * c) * self.dt
            yaw += rm * self.dt
            poses[m - 1] = (x, y, yaw)
        return poses

    def wrap_offsets(self, U: np.ndarray) -> np.ndarray:
        """yaw 展开圈数 k_m，使 psi_m - psi_ref,m - 2*pi*k_m ∈ (-pi, pi]。"""
        poses = self.pose_rollout(U)
        raw = poses[:, 2] - np.asarray(self.pose_ref, dtype=np.float64)[2::3]
        return np.round((raw - wrap_angle(raw)) / (2.0 * math.pi))

    def pose_residual(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> np.ndarray:
        poses = self.pose_rollout(U)
        ref = np.asarray(self.pose_ref, dtype=np.float64).reshape(self.horizon, 3)
        g = np.zeros(3 * self.horizon)
        g[0::3] = poses[:, 0] - ref[:, 0]
        g[1::3] = poses[:, 1] - ref[:, 1]
        raw = poses[:, 2] - ref[:, 2]
        g[2::3] = raw - 2.0 * math.pi * wrap_k if wrap_k is not None else wrap_angle(raw)
        return g

    # -- 代价 / 梯度 ---------------------------------------------------------
    def cost_terms(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> Dict[str, float]:
        U = np.asarray(U, dtype=np.float64)
        e_z = self.Theta @ U + self.e_free
        terms = {
            "latent": float(self.weights.w_z * e_z @ e_z),
            "ctrl": float(self.weights.w_u * U @ U),
            "rate": float(self.weights.w_du * (self.D @ U) @ (self.D @ U)),
            "pose": 0.0,
        }
        if self.pose_active:
            g = self.pose_residual(U, wrap_k)
            terms["pose"] = float(g @ (self.W * g))
        terms["total"] = terms["latent"] + terms["ctrl"] + terms["rate"] + terms["pose"]
        return terms

    def objective(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> float:
        return self.cost_terms(U, wrap_k)["total"]

    def pose_metrics(self, U: np.ndarray) -> Dict[str, float]:
        """把位姿残差换成可读指标：xy RMSE [m]、yaw RMSE [deg]。"""
        g = self.pose_residual(U)
        xy = np.sqrt(g[0::3] ** 2 + g[1::3] ** 2)
        return {
            "xy_rmse_m": float(np.sqrt(np.mean(xy**2))),
            "xy_max_m": float(np.max(xy)),
            "yaw_rmse_deg": float(np.sqrt(np.mean(g[2::3] ** 2)) * 180.0 / math.pi),
        }

    def gradient(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> np.ndarray:
        """精确梯度 ∇F(U)（位姿项用该点的 Phi，理论上与 GN 模型梯度一致）。"""
        U = np.asarray(U, dtype=np.float64)
        grad = 2.0 * (
            self.weights.w_z * self.Theta.T @ (self.Theta @ U + self.e_free)
            + self.weights.w_u * U
            + self.weights.w_du * self.D.T @ (self.D @ U)
        )
        if self.pose_active:
            Phi = self.pose_jacobian(U)
            g = self.pose_residual(U, wrap_k)
            grad += 2.0 * Phi.T @ (self.W * g)
        return grad

    # -- Gauss-Newton 线性化 -------------------------------------------------
    def pose_jacobian(self, U: np.ndarray) -> np.ndarray:
        """位姿 Jacobian Phi = dg/dU（3N x nvar），镜像 C++ `buildPoseLinearization`。"""
        Z = self.latent_stack(U)
        d = np.zeros((self.horizon + 1, 3))
        V = [None] * (self.horizon + 1)
        for m in range(1, self.horizon + 1):
            z_m = Z[(m - 1) * self.nz : m * self.nz]
            d[m] = self.sys.decode_physical(z_m)
            Jp = self.sys.decode_jacobian_physical(z_m)  # 3 x nz
            V[m] = Jp @ self.Theta[(m - 1) * self.nz : m * self.nz, :]  # 3 x nvar

        yaw = np.zeros(self.horizon + 1)
        yaw[0] = float(self.pose0[2])
        for m in range(1, self.horizon + 1):
            yaw[m] = yaw[m - 1] + d[m, 2] * self.dt

        Phi = np.zeros((3 * self.horizon, self.nvar))
        Sx = np.zeros(self.nvar)
        Sy = np.zeros(self.nvar)
        Sp = np.zeros(self.nvar)
        for m in range(1, self.horizon + 1):
            c, s = math.cos(yaw[m - 1]), math.sin(yaw[m - 1])
            um, vm = d[m, 0], d[m, 1]
            dx_dpsi = (-um * s - vm * c) * self.dt
            dy_dpsi = (um * c - vm * s) * self.dt
            Vu, Vv, Vr = V[m][0], V[m][1], V[m][2]
            Sx = Sx + self.dt * (c * Vu - s * Vv) + dx_dpsi * Sp
            Sy = Sy + self.dt * (s * Vu + c * Vv) + dy_dpsi * Sp
            Sp = Sp + self.dt * Vr
            Phi[(m - 1) * 3 + 0] = Sx
            Phi[(m - 1) * 3 + 1] = Sy
            Phi[(m - 1) * 3 + 2] = Sp
        return Phi

    def linearize(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (Phi, b)，使 g(U) ≈ Phi U + b，b = g(U0) - Phi U0。"""
        U = np.asarray(U, dtype=np.float64)
        Phi = self.pose_jacobian(U)
        b = self.pose_residual(U, wrap_k) - Phi @ U
        return Phi, b

    def gn_model(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """Gauss-Newton 二次模型 m(U) = 1/2 U'P U + q'U + c（P/q 与 C++ 组装一致）。"""
        P = 2.0 * (
            self.weights.w_z * self.Theta.T @ self.Theta
            + self.weights.w_u * np.eye(self.nvar)
            + self.weights.w_du * self.D.T @ self.D
        )
        q = 2.0 * self.weights.w_z * self.Theta.T @ self.e_free
        const = float(self.weights.w_z * self.e_free @ self.e_free)
        if self.pose_active:
            Phi, b = self.linearize(U, wrap_k)
            P = P + 2.0 * Phi.T @ (self.W[:, None] * Phi)
            q = q + 2.0 * Phi.T @ (self.W * b)
            const += float(b @ (self.W * b))
        return P, q, const

    def model_value(self, P: np.ndarray, q: np.ndarray, const: float, U: np.ndarray) -> float:
        U = np.asarray(U, dtype=np.float64)
        return float(0.5 * U @ (P @ U) + q @ U + const)

    # -- 可行集 --------------------------------------------------------------
    def feasible_constraints(
        self, M: Optional[np.ndarray] = None, center: Optional[np.ndarray] = None, radius: float = math.inf
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """构造 l <= A v <= u（v 为降维变量时传入 M；radius 有限时附加信赖域）。"""
        nu = self.nu
        if M is None:
            n_free = self.horizon
            step_to_var = list(range(self.horizon))
        else:
            n_free = self.blocking.n_free_blocks
            step_to_var = [self.blocking.block_of_step(k) for k in range(self.horizon)]
        nvar_red = n_free * nu

        rows: List[np.ndarray] = []
        lo: List[float] = []
        hi: List[float] = []

        # 盒约束（每个自由变量一行）
        for b in range(n_free):
            for j in range(nu):
                row = np.zeros(nvar_red)
                row[b * nu + j] = 1.0
                rows.append(row)
                lo.append(self.lo_t[j])
                hi.append(self.hi_t[j])

        # 速率约束：逐物理步，块内自动为 0（跳过），块边界与 k=0 锚定 u_prev
        prev_var = -1
        for k in range(self.horizon):
            cur_var = step_to_var[k]
            if k > 0 and cur_var == prev_var:
                prev_var = cur_var
                continue
            for j in range(nu):
                row = np.zeros(nvar_red)
                row[cur_var * nu + j] = 1.0
                if k == 0:
                    anchor = float(self.u_prev_tilde[j])
                    lo.append(anchor - self.dmax_t[j])
                    hi.append(anchor + self.dmax_t[j])
                else:
                    row[prev_var * nu + j] = -1.0
                    lo.append(-self.dmax_t[j])
                    hi.append(self.dmax_t[j])
                rows.append(row)
            prev_var = cur_var

        # 信赖域（无穷范数盒）
        if math.isfinite(radius) and center is not None:
            for i in range(nvar_red):
                row = np.zeros(nvar_red)
                row[i] = 1.0
                rows.append(row)
                lo.append(float(center[i]) - radius)
                hi.append(float(center[i]) + radius)

        return np.asarray(rows), np.asarray(lo), np.asarray(hi)

    def retract_feasible(self, U: np.ndarray) -> np.ndarray:
        """顺序钳位得到可行点（盒约束 → 速率约束），用于修正不可行的 warm start。"""
        U = np.array(U, dtype=np.float64, copy=True).reshape(self.horizon, self.nu)
        prev = np.asarray(self.u_prev_tilde, dtype=np.float64)
        for k in range(self.horizon):
            U[k] = np.clip(U[k], self.lo_t, self.hi_t)
            U[k] = np.clip(U[k], prev - self.dmax_t, prev + self.dmax_t)
            prev = U[k]
        return U.reshape(-1)

    def is_feasible(self, U: np.ndarray, tol: float = 1e-8) -> bool:
        U = np.asarray(U, dtype=np.float64).reshape(self.horizon, self.nu)
        if np.any(U < self.lo_t - tol) or np.any(U > self.hi_t + tol):
            return False
        prev = np.asarray(self.u_prev_tilde, dtype=np.float64)
        for k in range(self.horizon):
            if np.any(np.abs(U[k] - prev) > self.dmax_t + tol):
                return False
            prev = U[k]
        return True

    # -- 一阶最优性度量 ------------------------------------------------------
    def kkt_residual(self, U: np.ndarray, wrap_k: Optional[np.ndarray] = None,
                     M: Optional[np.ndarray] = None) -> float:
        """投影梯度范数 ||U - Proj_F(U - grad F(U))||_inf（0 表示满足一阶最优）。"""
        U = np.asarray(U, dtype=np.float64)
        grad = self.gradient(U, wrap_k)
        if M is None:
            target, var, n_red = U - grad, U, self.nvar
            A, lo, hi = self.feasible_constraints(None)
        else:
            n_red = M.shape[1]
            var = np.linalg.lstsq(M, U, rcond=None)[0]
            target = var - M.T @ grad
            A, lo, hi = self.feasible_constraints(M)
        proj = solve_qp_box(2.0 * np.eye(n_red), -2.0 * target, A, lo, hi, x0=var)
        return float(np.max(np.abs(var - proj)))


# ---------------------------------------------------------------------------
# QP 子问题求解
# ---------------------------------------------------------------------------
def solve_qp_box(
    P: np.ndarray,
    q: np.ndarray,
    A: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """min 1/2 x'P x + q'x s.t. lo <= A x <= hi。

    优先使用 OSQP（与 C++ 部署一致）；不可用时退回 scipy trust-constr。
    """
    try:
        import osqp
        from scipy import sparse

        prob = osqp.OSQP()
        prob.setup(
            P=sparse.csc_matrix(sparse.triu(sparse.csc_matrix(P))),
            q=np.asarray(q, dtype=np.float64),
            A=sparse.csc_matrix(A),
            l=np.asarray(lo, dtype=np.float64),
            u=np.asarray(hi, dtype=np.float64),
            eps_abs=1e-10,
            eps_rel=1e-10,
            max_iter=200000,
            polish=False,  # polish 在 OSQP 1.x 会绕过 verbose 打印
            verbose=False,
        )
        if x0 is not None:
            prob.warm_start(x=np.asarray(x0, dtype=np.float64))
        res = prob.solve()
        if res.x is None or np.any(~np.isfinite(res.x)):
            raise RuntimeError(f"osqp failed: {res.info.status}")
        return np.asarray(res.x, dtype=np.float64)
    except ImportError:  # pragma: no cover - 依赖缺失时的兜底路径
        from scipy.optimize import LinearConstraint, minimize

        n = P.shape[0]
        start = np.zeros(n) if x0 is None else np.asarray(x0, dtype=np.float64)
        res = minimize(
            lambda x: 0.5 * x @ (P @ x) + q @ x,
            start,
            jac=lambda x: P @ x + q,
            hess=lambda x: P,
            constraints=[LinearConstraint(A, lo, hi)],
            method="trust-constr",
            options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 2000},
        )
        return np.asarray(res.x, dtype=np.float64)


# ---------------------------------------------------------------------------
# SQP 求解器
# ---------------------------------------------------------------------------
@dataclass
class SqpScheme:
    """SQP 外循环的算法开关（用于「现状 vs 目标方案」对照）。

    充分下降判据用**模型预测下降量**：接受 alpha 当

        F(U) - F(U + alpha d) >= c1 * alpha * dm,   dm = m(U) - m(U + d) > 0

    由 dm >= -0.5 * grad'd 可知该判据强于常数 c1/2 的 Armijo 条件；而当 F 精确二次
    （Tier-1，无位姿项）时 alpha=1 必被接受，退化为单次 QP。
    """

    iters: int = 4
    line_search: bool = True
    sufficient_decrease_c1: float = 1e-4
    step_shrink: float = 0.5
    max_backtrack: int = 12
    trust_region: bool = True
    trust_radius_init: float = math.inf  # 起始无界：良态时与定步长同速
    trust_radius_max: float = math.inf
    trust_radius_min: float = 1e-3
    blocking_mode: str = "reduced"  # "reduced" | "posthoc"
    freeze_wrap: bool = True
    project_init: bool = True
    step_tol: float = 1e-8
    rel_cost_tol: float = 1e-10

    @classmethod
    def baseline_cpp(cls, iters: int = 4) -> "SqpScheme":
        """当前 C++ 行为：定步长、事后 move-blocking、无投影 / 无 wrap 冻结。"""
        return cls(
            iters=iters,
            line_search=False,
            trust_region=False,
            blocking_mode="posthoc",
            freeze_wrap=False,
            project_init=False,
            step_tol=0.0,
            rel_cost_tol=0.0,
        )

    @classmethod
    def guarded(cls, iters: int = 4) -> "SqpScheme":
        """推导得到的方案：降维 blocking + 可行投影 + wrap 冻结 + 信赖域 + 充分下降。"""
        return cls(iters=iters)


@dataclass
class SqpResult:
    U: np.ndarray
    cost_history: List[float]
    step_norms: List[float] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    radii: List[float] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    kkt: float = float("nan")
    iters_run: int = 0
    feasible: bool = True
    qp_solves: int = 0

    @property
    def cost(self) -> float:
        return self.cost_history[-1]

    @property
    def monotone(self) -> bool:
        return all(b <= a + 1e-12 for a, b in zip(self.cost_history, self.cost_history[1:]))


class SqpSolver:
    """Gauss-Newton SQP 外循环（QP 子问题在归一化控制空间求解）。"""

    def __init__(self, nlp: LatentPoseNlp, scheme: SqpScheme = SqpScheme()) -> None:
        self.nlp = nlp
        self.scheme = scheme

    def solve(self, U_init: Optional[np.ndarray] = None) -> SqpResult:
        nlp, sch = self.nlp, self.scheme
        U = np.zeros(nlp.nvar) if U_init is None else np.array(U_init, dtype=np.float64, copy=True)

        reduced = sch.blocking_mode == "reduced"
        M = nlp.blocking.selector(nlp.nu) if reduced else None
        if reduced:
            # 把初值投到 blocking 子空间（块内取块首值，与实际下发一致）
            U = nlp.blocking.expand_posthoc(U, nlp.nu)
        if sch.project_init:
            U = nlp.retract_feasible(U)

        wrap_k = nlp.wrap_offsets(U) if (sch.freeze_wrap and nlp.pose_active) else None
        # F 精确二次（Tier-1，无位姿项）时步长控制无意义：单次 QP 即全局最优。
        guards_on = nlp.pose_active
        radius = float(sch.trust_radius_init) if (guards_on and sch.trust_region) else math.inf

        res = SqpResult(U=U.copy(), cost_history=[nlp.objective(U, wrap_k)])
        for _ in range(max(1, sch.iters)):
            P, q, const = nlp.gn_model(U, wrap_k)
            grad = P @ U + q  # = ∇F(U)（GN 模型在展开点梯度精确）

            if reduced:
                V = np.linalg.lstsq(M, U, rcond=None)[0]
                P_r, q_r = M.T @ P @ M, M.T @ (q + 0.0)
                A, lo, hi = nlp.feasible_constraints(M, center=V, radius=radius)
                V_new = solve_qp_box(P_r, q_r, A, lo, hi, x0=V)
                U_qp = M @ V_new
            else:
                A, lo, hi = nlp.feasible_constraints(None, center=U, radius=radius)
                U_qp = solve_qp_box(P, q, A, lo, hi, x0=U)
                U_qp = nlp.blocking.expand_posthoc(U_qp, nlp.nu)
            res.qp_solves += 1

            d = U_qp - U
            step = float(np.max(np.abs(d)))
            res.step_norms.append(step)
            if step <= sch.step_tol:
                res.iters_run += 1
                break

            gtd = float(grad @ d)
            model_red = -(nlp.model_value(P, q, const, U_qp) - nlp.model_value(P, q, const, U))
            cost_cur = res.cost_history[-1]

            alpha = 1.0
            cost_new = nlp.objective(U + d, wrap_k)
            if sch.line_search and guards_on:
                thresh = sch.sufficient_decrease_c1 * max(model_red, 0.0)
                for _ in range(sch.max_backtrack):
                    if cost_cur - cost_new >= alpha * thresh:
                        break
                    alpha *= sch.step_shrink
                    cost_new = nlp.objective(U + alpha * d, wrap_k)
                else:
                    alpha = 0.0
                    cost_new = cost_cur

            actual_red = cost_cur - cost_new
            ratio = actual_red / model_red if model_red > 1e-14 else 0.0
            res.alphas.append(alpha)
            res.ratios.append(ratio)
            res.radii.append(radius)

            if guards_on and sch.trust_region:
                if ratio < 0.25:
                    radius = max(sch.trust_radius_min, 0.5 * min(step, radius))
                elif ratio > 0.75 and step >= 0.9 * radius:
                    radius = min(sch.trust_radius_max, 2.0 * radius)

            if alpha > 0.0:
                U = U + alpha * d
            res.cost_history.append(nlp.objective(U, wrap_k))
            res.iters_run += 1

            if alpha == 0.0 and radius <= sch.trust_radius_min:
                break
            if sch.rel_cost_tol > 0.0 and actual_red <= sch.rel_cost_tol * max(1.0, abs(cost_cur)):
                break

        res.U = U
        res.feasible = nlp.is_feasible(U)
        res.kkt = nlp.kkt_residual(U, wrap_k, M)
        return res


def summarize(name: str, res: SqpResult, nlp: LatentPoseNlp) -> str:
    """单行摘要，便于对照打印。"""
    terms = nlp.cost_terms(res.U)
    pm = nlp.pose_metrics(res.U) if nlp.pose_active else {"xy_rmse_m": float("nan"), "yaw_rmse_deg": float("nan")}
    return (
        f"{name:<18s} J={terms['total']:<11.6g} xy_rmse={pm['xy_rmse_m']:8.3f}m "
        f"yaw_rmse={pm['yaw_rmse_deg']:7.2f}deg  it={res.iters_run} qp={res.qp_solves} "
        f"monotone={str(res.monotone):5s} feasible={str(res.feasible):5s} kkt={res.kkt:.2e}"
    )
