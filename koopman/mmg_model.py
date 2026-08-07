"""双推进器 MMG 基线模型（3-DOF: surge u / sway v / yaw rate r）与最小二乘参数辨识。

对应 docs/MMG残差MLP建模技术方案.md §3.2 / §3.3。

模型方程（每单位质量/惯量参数化，油门为百分制指令 ±100，舵角为度）::

    u̇ = v·r + X_u·u + X_uu·u|u| + k_tx·(cp·cosδp + cs·cosδs)
    v̇ = −u·r + Y_v·v + Y_r·r + Y_vv·v|v| + Y_rr·r|r| + Y_vr·v·r + k_ty·(cp·sinδp + cs·sinδs)
    ṙ = N_v·v + N_r·r + N_rr·r|r| + N_vr·v·r + k_tn_lat·(cp·sinδp + cs·sinδs)
        + k_tn_diff·(cs·cosδs − cp·cosδp)

经典 MMG 的「螺旋桨 RPM + 舵」在此改型为双推进器的「油门% + 推力矢量角」：
  * k_tx / k_ty     —— 油门→纵/横向推力增益（每单位质量）；
  * k_tn_lat        —— 横向推力经纵向力臂 x_t 产生的回转力矩；
  * k_tn_diff       —— 左右舷推力差经横向间距产生的力矩（原地差速回转主导项，
                       对应 data/koopman_train_left_turn.npz 工况，δ=0 纯差速）。

所有方程对未知参数均为线性，可用最小二乘一步辨识（least_squares_fit）。
质量/惯量与水动力导数存在尺度耦合，只能辨识组合量（每单位质量），对预测无影响。
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# 参数顺序（ least_squares_fit 的输出与 MmgModel.theta 一致）
MMG_PARAM_NAMES: List[str] = [
    # surge: u̇ = v·r + X_u·u + X_uu·u|u| + k_tx·Σc·cosδ
    "X_u", "X_uu", "k_tx",
    # sway: v̇ = −u·r + Y_v·v + Y_r·r + Y_vv·v|v| + Y_rr·r|r| + Y_vr·v·r + k_ty·Σc·sinδ
    "Y_v", "Y_r", "Y_vv", "Y_rr", "Y_vr", "k_ty",
    # yaw: ṙ = N_v·v + N_r·r + N_rr·r|r| + N_vr·v·r + k_tn_lat·Σc·sinδ + k_tn_diff·Δ(c·cosδ)
    "N_v", "N_r", "N_rr", "N_vr", "k_tn_lat", "k_tn_diff",
]
N_MMG_PARAMS = len(MMG_PARAM_NAMES)  # 15


# ---------------------------------------------------------------------------
# 最小二乘辨识
# ---------------------------------------------------------------------------


def _box_smooth(x: np.ndarray, win: int) -> np.ndarray:
    """等权滑动平均（mode='same'，边缘后续会被裁掉）。"""
    if win <= 1:
        return x.astype(np.float64)
    ker = np.ones(int(win), dtype=np.float64) / float(win)
    return np.convolve(x.astype(np.float64), ker, mode="same")


def least_squares_fit(
    segments: Sequence[Tuple[np.ndarray, np.ndarray]],
    data_dt: float = 0.1,
    smooth: int = 5,
    ridge: float = 1e-6,
) -> Tuple[np.ndarray, Dict]:
    """从分段时序辨识 MMG 名义参数（线性最小二乘 + 岭正则）。

    Args:
        segments: [(states (T,6) [x,y,yaw,u,v,r], ctrls (T,4) [cp,δp,cs,δs])]，
                  物理量、10Hz（data_dt）。
        data_dt:  原始采样间隔 [s]。
        smooth:   差分前对 (u,v,r) 做滑动平均的窗长（抑制差分噪声放大）。
        ridge:    岭正则系数 λ（解 (ΦᵀΦ+λI)θ = Φᵀy）。

    Returns:
        theta:  (N_MMG_PARAMS,) float64，顺序见 MMG_PARAM_NAMES。
        report: 每通道拟合 RMSE / R² 与样本数。
    """
    m = max(int(smooth) // 2, 1)

    feats_u: List[np.ndarray] = []  # [u, u|u|, Σc·cosδ]
    feats_v: List[np.ndarray] = []  # [v, r, v|v|, r|r|, v·r, Σc·sinδ]
    feats_r: List[np.ndarray] = []  # [v, r, r|r|, v·r, Σc·sinδ, Δc·cosδ]
    targ_u: List[np.ndarray] = []   # u̇ − v·r
    targ_v: List[np.ndarray] = []   # v̇ + u·r
    targ_r: List[np.ndarray] = []   # ṙ

    for states, ctrls in segments:
        T = states.shape[0]
        if T < 2 * m + 3:
            continue
        u_s = _box_smooth(states[:, 3], smooth)
        v_s = _box_smooth(states[:, 4], smooth)
        r_s = _box_smooth(states[:, 5], smooth)
        du = (u_s[1:] - u_s[:-1]) / data_dt
        dv = (v_s[1:] - v_s[:-1]) / data_dt
        dr = (r_s[1:] - r_s[:-1]) / data_dt

        sl = slice(m, T - 1 - m)  # 特征与差分都对齐到平滑区中部，k 对应 [k, k+1] 差分
        u, v, r = u_s[sl], v_s[sl], r_s[sl]
        cp = ctrls[sl, 0].astype(np.float64)
        dp = np.deg2rad(ctrls[sl, 1].astype(np.float64))
        cs = ctrls[sl, 2].astype(np.float64)
        ds = np.deg2rad(ctrls[sl, 3].astype(np.float64))
        lon = cp * np.cos(dp) + cs * np.cos(ds)
        lat = cp * np.sin(dp) + cs * np.sin(ds)
        dif = cs * np.cos(ds) - cp * np.cos(dp)

        feats_u.append(np.stack([u, u * np.abs(u), lon], axis=1))
        feats_v.append(np.stack([v, r, v * np.abs(v), r * np.abs(r), v * r, lat], axis=1))
        feats_r.append(np.stack([v, r, r * np.abs(r), v * r, lat, dif], axis=1))
        targ_u.append(du[sl] - v * r)
        targ_v.append(dv[sl] + u * r)
        targ_r.append(dr[sl])

    if not feats_u:
        raise ValueError("no valid segments for least_squares_fit")

    def _solve(Phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        Phi = np.concatenate(Phi, axis=0)
        y = np.concatenate(y, axis=0)
        n_feat = Phi.shape[1]
        n_data = y.shape[0]
        if ridge > 0:
            # 岭正则通过增广最小二乘实现（SVD 求解，避免法方程条件数平方）
            lam = np.sqrt(float(ridge))
            Phi = np.vstack([Phi, lam * np.eye(n_feat)])
            y = np.concatenate([y, np.zeros(n_feat)])
        th, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        y_data = y[:n_data]
        res = y_data - Phi[:n_data] @ th
        rmse = float(np.sqrt(np.mean(res ** 2)))
        ss_tot = float(np.sum((y_data - y_data.mean()) ** 2))
        r2 = float(1.0 - np.sum(res ** 2) / ss_tot) if ss_tot > 1e-12 else float("nan")
        return th, {"rmse": rmse, "r2": r2, "n_samples": int(n_data)}

    th_u, rep_u = _solve(feats_u, targ_u)
    th_v, rep_v = _solve(feats_v, targ_v)
    th_r, rep_r = _solve(feats_r, targ_r)

    theta = np.concatenate([th_u, th_v, th_r])
    report = {
        "surge": {**rep_u, "params": dict(zip(MMG_PARAM_NAMES[0:3], th_u.tolist()))},
        "sway": {**rep_v, "params": dict(zip(MMG_PARAM_NAMES[3:9], th_v.tolist()))},
        "yaw": {**rep_r, "params": dict(zip(MMG_PARAM_NAMES[9:15], th_r.tolist()))},
        "data_dt": float(data_dt),
        "smooth": int(smooth),
        "ridge": float(ridge),
    }
    return theta, report


# ---------------------------------------------------------------------------
# PyTorch 模型
# ---------------------------------------------------------------------------


class MmgModel(nn.Module):
    """MMG 基线（torch，可嵌入训练计算图）。theta 默认冻结（buffer 语义），
    trainable=True 时作为 nn.Parameter 参与联合微调。"""

    def __init__(self, theta: Optional[np.ndarray] = None, trainable: bool = False, sub_dt: float = 0.1) -> None:
        super().__init__()
        t = torch.zeros(N_MMG_PARAMS) if theta is None else torch.as_tensor(
            np.asarray(theta, dtype=np.float32))
        self.theta = nn.Parameter(t, requires_grad=bool(trainable))
        self.sub_dt = float(sub_dt)

    def accel(self, dyn: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """dyn (B,3) [u,v,r]、ctrl (B,4) [cp,δp(deg),cs,δs(deg)] → (B,3) [u̇,v̇,ṙ]。"""
        th = self.theta
        u, v, r = dyn.unbind(-1)
        cp, dp, cs, ds = ctrl.unbind(-1)
        dp = torch.deg2rad(dp)
        ds = torch.deg2rad(ds)
        lon = cp * torch.cos(dp) + cs * torch.cos(ds)
        lat = cp * torch.sin(dp) + cs * torch.sin(ds)
        dif = cs * torch.cos(ds) - cp * torch.cos(dp)
        du = v * r + th[0] * u + th[1] * u * u.abs() + th[2] * lon
        dv = -u * r + th[3] * v + th[4] * r + th[5] * v * v.abs() + th[6] * r * r.abs() + th[7] * v * r + th[8] * lat
        dr = th[9] * v + th[10] * r + th[11] * r * r.abs() + th[12] * v * r + th[13] * lat + th[14] * dif
        return torch.stack([du, dv, dr], dim=-1)

    def step_phys(self, dyn: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
        """单步积分 dt 秒（内部按 sub_dt 细分做 Euler，ZOH 控制）。"""
        n = max(1, int(round(float(dt) / self.sub_dt)))
        h = float(dt) / n
        x = dyn
        for _ in range(n):
            x = x + h * self.accel(x, ctrl)
        return x

    def param_dict(self) -> Dict[str, float]:
        t = self.theta.detach().cpu().numpy()
        return {name: float(t[i]) for i, name in enumerate(MMG_PARAM_NAMES)}


# ---------------------------------------------------------------------------
# evalkit 适配器
# ---------------------------------------------------------------------------


class PhysStepAdapter(nn.Module):
    """把物理步进模型包装成 evalkit.rollout_dataset 所需的
    encode / latent_step / reconstruct_state 接口（潜空间 = 归一化 (u,v,r)）。

    step_fn: (dyn_phys (B,3), ctrl_phys (B,4)) -> dyn_phys_next (B,3)，
             dt 由调用方闭包捕获。
    """

    def __init__(self, step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], stats: Dict[str, np.ndarray]) -> None:
        super().__init__()
        self.step_fn = step_fn
        self.register_buffer("dyn_mean", torch.as_tensor(np.asarray(stats["state_mean"][3:6]), dtype=torch.float32))
        self.register_buffer("dyn_std", torch.as_tensor(np.asarray(stats["state_std"][3:6]), dtype=torch.float32))
        self.register_buffer("ctrl_mean", torch.as_tensor(np.asarray(stats["ctrl_mean"]), dtype=torch.float32))
        self.register_buffer("ctrl_std", torch.as_tensor(np.asarray(stats["ctrl_std"]), dtype=torch.float32))

    def encode(self, dyn0_norm: torch.Tensor) -> torch.Tensor:
        return dyn0_norm

    def latent_step(self, z: torch.Tensor, u_norm: torch.Tensor) -> torch.Tensor:
        dyn = z * self.dyn_std + self.dyn_mean
        ctrl = u_norm * self.ctrl_std + self.ctrl_mean
        nxt = self.step_fn(dyn, ctrl)
        return (nxt - self.dyn_mean) / self.dyn_std

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        return z


# ---------------------------------------------------------------------------
# 统计量 / 持久化
# ---------------------------------------------------------------------------


def compute_train_stats(states_full: np.ndarray, ctrls_full: np.ndarray) -> Dict[str, np.ndarray]:
    """与 KoopmanVoyageDataset._compute_stats 同口径的 z-score 统计。"""
    return {
        "state_mean": states_full.mean(axis=0).astype(np.float32),
        "state_std": (states_full.std(axis=0) + 1e-6).astype(np.float32),
        "ctrl_mean": ctrls_full.mean(axis=0).astype(np.float32),
        "ctrl_std": (ctrls_full.std(axis=0) + 1e-6).astype(np.float32),
    }


def save_mmg_npz(path: str, theta: np.ndarray, stats: Dict[str, np.ndarray], report: Dict) -> None:
    np.savez(
        path,
        theta=np.asarray(theta, dtype=np.float64),
        param_names=np.asarray(MMG_PARAM_NAMES),
        state_mean=stats["state_mean"], state_std=stats["state_std"],
        ctrl_mean=stats["ctrl_mean"], ctrl_std=stats["ctrl_std"],
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )


def load_mmg_npz(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    z = np.load(path, allow_pickle=True)
    theta = z["theta"].astype(np.float64)
    stats = {
        "state_mean": z["state_mean"], "state_std": z["state_std"],
        "ctrl_mean": z["ctrl_mean"], "ctrl_std": z["ctrl_std"],
    }
    report = json.loads(str(z["report_json"])) if "report_json" in z else {}
    return theta, stats, report
