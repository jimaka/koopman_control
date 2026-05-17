"""eval_koopman.py — 量化评估 + 多 ckpt 对比

Section 9 of PROMPT_deep_koopman_rewrite.md 的实现。

核心设计原则：
* **物理空间评估**：所有 RMSE / MAE / R² / bias 都在反归一化的物理量上计算，
  单位明确（u, v: m/s；r: rad/s；traj_xy: m）。
* **机器可读优先**：CSV / JSON / MD 都先写到磁盘，再画图。
* **发散指标**：除均值外，必须能定量回答「误差是否随预测步数发散」——
  ratio_step_K_over_step_1 / slope_loglog / lyapunov_like / instability_score
  详见函数 :func:`compute_divergence_metrics`。

使用方式
--------

单 ckpt::

    python3 eval_koopman.py --ckpt checkpoints/koopman_best.pth \\
        --data koopman_test.npz --pred_len 20 --tag v2 --out_dir test_analysis/v2

多 ckpt 横向对比::

    python3 eval_koopman.py --compare \\
        checkpoints/koopman_v1_best.pth:v1 \\
        checkpoints/koopman_best.pth:v2 \\
        --data koopman_test.npz --out_dir test_analysis/compare_v1_v2

冒烟自测（无 GPU 1 分钟内）::

    python3 eval_koopman.py --smoketest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from koopman.model_v1_v2 import HorizontalKoopmanModel
try:
    from koopman.model_v3 import FEATURE_DICT_ATOMS, HorizontalKoopmanModelV3
except Exception:  # 容错：v3 模块缺失时仍能用 v1/v2
    HorizontalKoopmanModelV3 = None  # type: ignore
    FEATURE_DICT_ATOMS = []

# ---------------------------------------------------------------------------
# 1. 数据装载（与 train_koopman_v2.py 中的 dataset 等价，但不依赖它，
#    以便 eval_koopman.py 可独立使用）
# ---------------------------------------------------------------------------


def _flatten_segments(npz_path: str, pred_len: int, stride: int = 1):
    """把所有段拼成 (N_total, 6) 状态张量与 (N_total, 4) 控制张量。

    返回:
        states_full: (N, 6) float32 —— [x, y, yaw, u, v, r]
        ctrls_full:  (N, 4) float32
        seg_starts:  (S,) int  —— 每段在 states_full 中的起点
        seg_lens:    (S,) int  —— 每段长度
        sample_index: (M, 2) int —— 每个样本的 (global_t0, seg_idx)
                      满足 t0 + pred_len < seg_len（注意是严格小于，
                      因为我们要取 t0..t0+pred_len 共 pred_len+1 个状态）
    """
    raw = np.load(npz_path, allow_pickle=True)["datas"]
    state_chunks: List[np.ndarray] = []
    ctrl_chunks: List[np.ndarray] = []
    seg_starts: List[int] = []
    seg_lens: List[int] = []
    cursor = 0
    for seg in raw:
        T = int(seg["len"])
        # 防御性切片：有的字段长度可能与 len 不匹配，统一用 :T
        st = np.empty((T, 6), dtype=np.float32)
        st[:, 0] = seg["Pos"][0, :T]
        st[:, 1] = seg["Pos"][1, :T]
        st[:, 2] = seg["Euler"][2, :T]
        st[:, 3] = seg["Vel"][0, :T]
        st[:, 4] = seg["Vel"][1, :T]
        st[:, 5] = seg["pqr"][0, :T]
        ct = seg["Thrusters_CMD"][:, :T].T.astype(np.float32, copy=False)
        state_chunks.append(st)
        ctrl_chunks.append(ct)
        seg_starts.append(cursor)
        seg_lens.append(T)
        cursor += T
    states_full = np.concatenate(state_chunks, axis=0)
    ctrls_full = np.concatenate(ctrl_chunks, axis=0)
    seg_starts_arr = np.asarray(seg_starts, dtype=np.int64)
    seg_lens_arr = np.asarray(seg_lens, dtype=np.int64)

    # 生成 (sample_idx, seg_idx, t0_global) 索引
    sample_global_t0 = []
    sample_seg_idx = []
    sample_local_t0 = []
    for sidx, (start, T) in enumerate(zip(seg_starts, seg_lens)):
        if T <= pred_len:
            continue
        # t_local 取值 0..T-pred_len-1（保证 t_local + pred_len <= T-1）
        local = np.arange(0, T - pred_len, stride, dtype=np.int64)
        sample_global_t0.append(local + start)
        sample_seg_idx.append(np.full_like(local, sidx))
        sample_local_t0.append(local)
    if not sample_global_t0:
        raise ValueError(f"No valid samples in {npz_path} with pred_len={pred_len}")
    sample_global_t0 = np.concatenate(sample_global_t0)
    sample_seg_idx = np.concatenate(sample_seg_idx)
    sample_local_t0 = np.concatenate(sample_local_t0)
    return (
        states_full,
        ctrls_full,
        seg_starts_arr,
        seg_lens_arr,
        sample_global_t0,
        sample_seg_idx,
        sample_local_t0,
    )


# ---------------------------------------------------------------------------
# 2. Rollout（核心数值）
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_dataset(
    model: nn.Module,
    states_full: np.ndarray,
    ctrls_full: np.ndarray,
    sample_global_t0: np.ndarray,
    pred_len: int,
    stats: Dict[str, np.ndarray],
    device: torch.device,
    dt: float,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对所有样本做无 teacher-force 的多步 rollout，得到物理量预测/真值。

    返回（全部 float32，单位均为物理量）:
        gt_dyn:    (M, K, 3) [u, v, r]
        pred_dyn:  (M, K, 3)
        gt_xy:     (M, K, 2) 用 GT 速度 + GT 起始位姿欧拉积分得到的位置
        pred_xy:   (M, K, 2) 用 Pred 速度 + GT 起始位姿欧拉积分得到的位置
    """
    state_mean = stats["state_mean"].astype(np.float32)
    state_std = stats["state_std"].astype(np.float32)
    ctrl_mean = stats["ctrl_mean"].astype(np.float32)
    ctrl_std = stats["ctrl_std"].astype(np.float32)

    dyn_mean_t = torch.tensor(state_mean[3:6], device=device)
    dyn_std_t = torch.tensor(state_std[3:6], device=device)
    ctrl_mean_t = torch.tensor(ctrl_mean, device=device)
    ctrl_std_t = torch.tensor(ctrl_std, device=device)

    M = sample_global_t0.shape[0]
    K = pred_len
    pred_dyn = np.empty((M, K, 3), dtype=np.float32)
    gt_dyn = np.empty((M, K, 3), dtype=np.float32)
    pred_xy = np.empty((M, K, 2), dtype=np.float32)
    gt_xy = np.empty((M, K, 2), dtype=np.float32)

    model.eval()
    for start in range(0, M, batch_size):
        end = min(M, start + batch_size)
        idx = sample_global_t0[start:end]  # (b,)
        b = idx.shape[0]

        # 取 t0 处 6 维状态（用于位置初值与 dyn 起始点）
        x0 = states_full[idx]  # (b, 6) [x,y,yaw,u,v,r]
        # 取未来 K 步控制（CPU numpy 切片，再一次性转 GPU）
        u_future = np.empty((b, K, 4), dtype=np.float32)
        gt_dyn_future = np.empty((b, K, 3), dtype=np.float32)
        for j, t0 in enumerate(idx):
            u_future[j] = ctrls_full[t0 : t0 + K]
            gt_dyn_future[j] = states_full[t0 + 1 : t0 + 1 + K, 3:6]

        x0_t = torch.from_numpy(x0).to(device)
        u_future_t = torch.from_numpy(u_future).to(device)
        u_norm = (u_future_t - ctrl_mean_t) / ctrl_std_t

        dyn0_phys = x0_t[:, 3:6]
        dyn0_norm = (dyn0_phys - dyn_mean_t) / dyn_std_t

        z = model.encode(dyn0_norm)
        cur_x = x0_t[:, 0].clone()
        cur_y = x0_t[:, 1].clone()
        cur_yaw = x0_t[:, 2].clone()
        # GT 积分用同样的初值
        gt_x = x0_t[:, 0].clone()
        gt_y = x0_t[:, 1].clone()
        gt_yaw = x0_t[:, 2].clone()

        gt_dyn_future_t = torch.from_numpy(gt_dyn_future).to(device)

        for k in range(K):
            z = model.latent_step(z, u_norm[:, k, :])
            pred_norm = model.reconstruct_state(z)
            pred_phys = pred_norm * dyn_std_t + dyn_mean_t  # (b,3)
            up, vp, rp = pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2]
            cur_x = cur_x + (up * torch.cos(cur_yaw) - vp * torch.sin(cur_yaw)) * dt
            cur_y = cur_y + (up * torch.sin(cur_yaw) + vp * torch.cos(cur_yaw)) * dt
            cur_yaw = cur_yaw + rp * dt
            pred_dyn[start:end, k, 0] = up.cpu().numpy()
            pred_dyn[start:end, k, 1] = vp.cpu().numpy()
            pred_dyn[start:end, k, 2] = rp.cpu().numpy()
            pred_xy[start:end, k, 0] = cur_x.cpu().numpy()
            pred_xy[start:end, k, 1] = cur_y.cpu().numpy()

            ug = gt_dyn_future_t[:, k, 0]
            vg = gt_dyn_future_t[:, k, 1]
            rg = gt_dyn_future_t[:, k, 2]
            gt_x = gt_x + (ug * torch.cos(gt_yaw) - vg * torch.sin(gt_yaw)) * dt
            gt_y = gt_y + (ug * torch.sin(gt_yaw) + vg * torch.cos(gt_yaw)) * dt
            gt_yaw = gt_yaw + rg * dt
            gt_dyn[start:end, k, 0] = ug.cpu().numpy()
            gt_dyn[start:end, k, 1] = vg.cpu().numpy()
            gt_dyn[start:end, k, 2] = rg.cpu().numpy()
            gt_xy[start:end, k, 0] = gt_x.cpu().numpy()
            gt_xy[start:end, k, 1] = gt_y.cpu().numpy()
    return gt_dyn, pred_dyn, gt_xy, pred_xy


# ---------------------------------------------------------------------------
# 3. 指标
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相关；常数序列返回 NaN 不抛错。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    sa = a.std()
    sb = b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot；常数 GT 时返回 NaN。"""
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        if ss_tot < 1e-12:
            return float("nan")
        return 1.0 - ss_res / ss_tot


def compute_per_step_metrics(
    gt_dyn: np.ndarray,
    pred_dyn: np.ndarray,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    dt: float,
) -> Dict[str, np.ndarray]:
    """全数据集聚合的逐步指标，返回 dict[列名] = 长度 K 的数组。"""
    M, K, _ = gt_dyn.shape
    out: Dict[str, np.ndarray] = {}
    out["step"] = np.arange(1, K + 1, dtype=np.int64)
    out["n_samples"] = np.full(K, M, dtype=np.int64)

    diff = pred_dyn - gt_dyn  # (M, K, 3) 物理量误差
    vel_horiz_err = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # (M,K)

    out["vel_rmse"] = np.sqrt(np.mean(vel_horiz_err ** 2, axis=0))
    out["vel_mae"] = np.mean(vel_horiz_err, axis=0)
    out["vel_p50"] = np.percentile(vel_horiz_err, 50, axis=0)
    out["vel_p90"] = np.percentile(vel_horiz_err, 90, axis=0)
    out["vel_p99"] = np.percentile(vel_horiz_err, 99, axis=0)

    for ci, name in enumerate(["u", "v", "r"]):
        out[f"{name}_rmse"] = np.sqrt(np.mean(diff[..., ci] ** 2, axis=0))
        out[f"{name}_bias"] = np.mean(diff[..., ci], axis=0)
        gt_std = np.std(gt_dyn[..., ci], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{name}_nrmse"] = np.where(gt_std > 1e-9, out[f"{name}_rmse"] / gt_std, np.nan)
        # R² 与 Pearson 按 step 上的所有样本计算
        r2_arr = np.empty(K, dtype=np.float64)
        corr_arr = np.empty(K, dtype=np.float64)
        for k in range(K):
            r2_arr[k] = _r2(gt_dyn[:, k, ci], pred_dyn[:, k, ci])
            corr_arr[k] = _safe_pearson(gt_dyn[:, k, ci], pred_dyn[:, k, ci])
        out[f"{name}_r2"] = r2_arr
        out[f"{name}_corr"] = corr_arr

    # 加速度 RMSE（k>=2 才有）：用相邻 step 差分
    acc_rmse = np.full(K, np.nan, dtype=np.float64)
    pred_acc = (pred_dyn[:, 1:, :] - pred_dyn[:, :-1, :]) / dt
    gt_acc = (gt_dyn[:, 1:, :] - gt_dyn[:, :-1, :]) / dt
    acc_err = np.sqrt(np.sum((pred_acc - gt_acc) ** 2, axis=-1))  # (M, K-1)
    acc_rmse[1:] = np.sqrt(np.mean(acc_err ** 2, axis=0))
    out["acc_rmse"] = acc_rmse

    # traj xy 误差
    traj_err = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))  # (M, K)
    out["traj_xy_err"] = np.sqrt(np.mean(traj_err ** 2, axis=0))

    return out


def compute_divergence_metrics(per_step: Dict[str, np.ndarray]) -> Dict[str, float]:
    """发散指标——回答「误差是否随预测步数发散」。

    定义（与 PROMPT 9.1 对应）：
      * ratio_stepK_over_step1 = vel_rmse[K-1] / max(vel_rmse[0], 1e-6)
        理想 ≈ 1~3；> 5 强烈发散。
      * slope_loglog = lstsq slope of log(step) vs log(vel_rmse)
        ≈ 0 几乎不增长；≈ 1 线性累积；> 1 超线性发散，强警报。
      * slope_linear = lstsq slope of step vs vel_rmse (m/s per step)。
      * lyapunov_like = mean_k log(vel_rmse[k+1]/vel_rmse[k])  k>=0
        > 0 即指数级发散。
      * auc_error_curve = trapz / K。
      * monotonic_increasing = ∀ diff >= -1e-4。
      * instability_score = sigmoid(slope_loglog) * (1 + ratio/10)。
    """
    vel = per_step["vel_rmse"].astype(np.float64)
    K = vel.shape[0]
    steps = np.arange(1, K + 1, dtype=np.float64)

    eps = 1e-6
    ratio = float(vel[-1] / max(vel[0], eps))

    log_steps = np.log(steps)
    log_vel = np.log(np.maximum(vel, eps))
    A = np.vstack([log_steps, np.ones_like(log_steps)]).T
    slope_ll, _ = np.linalg.lstsq(A, log_vel, rcond=None)[0]

    A2 = np.vstack([steps, np.ones_like(steps)]).T
    slope_lin, _ = np.linalg.lstsq(A2, vel, rcond=None)[0]

    if K >= 2:
        lyap = float(np.mean(np.log(np.maximum(vel[1:], eps) / np.maximum(vel[:-1], eps))))
    else:
        lyap = float("nan")

    # numpy>=2 移除了 np.trapz，统一用 trapezoid
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc = float(trap(vel, steps) / K)
    monotonic = bool(np.all(np.diff(vel) >= -1e-4))

    sig = 1.0 / (1.0 + np.exp(-slope_ll))
    inst = float(sig * (1.0 + ratio / 10.0))

    return {
        f"ratio_step{K}_over_step1": ratio,
        "ratio_step20_over_step1": ratio,  # 别名，兼容 PROMPT 文档
        "slope_loglog": float(slope_ll),
        "slope_linear": float(slope_lin),
        "lyapunov_like": lyap,
        "auc_error_curve": auc,
        "monotonic_increasing": monotonic,
        "instability_score": inst,
    }


def compute_per_sample_step_metrics(
    gt_dyn: np.ndarray, pred_dyn: np.ndarray, gt_xy: np.ndarray, pred_xy: np.ndarray
) -> Dict[str, np.ndarray]:
    """每个样本在 step K（最后一步）的指标 + 自身的 divergence_ratio/slope。"""
    M, K, _ = gt_dyn.shape
    diff = pred_dyn - gt_dyn
    vel_err = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # (M, K)
    traj_err = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))  # (M, K)

    eps = 1e-6
    div_ratio = vel_err[:, -1] / np.maximum(vel_err[:, 0], eps)

    # 每条样本的 log-log 斜率
    steps = np.arange(1, K + 1, dtype=np.float64)
    log_steps = np.log(steps)
    A = np.vstack([log_steps, np.ones_like(log_steps)]).T
    log_vel = np.log(np.maximum(vel_err.astype(np.float64), eps))
    slopes = np.linalg.lstsq(A, log_vel.T, rcond=None)[0][0, :]

    return {
        "vel_err_stepK": vel_err[:, -1],
        "u_err_stepK": np.abs(diff[:, -1, 0]),
        "v_err_stepK": np.abs(diff[:, -1, 1]),
        "r_err_stepK": np.abs(diff[:, -1, 2]),
        "traj_xy_err_stepK": traj_err[:, -1],
        "divergence_ratio": div_ratio,
        "divergence_slope": slopes,
    }


# ---------------------------------------------------------------------------
# 4. 文件落盘 (CSV / JSON / MD)
# ---------------------------------------------------------------------------

CSV_HEADER_UNITS = {
    "vel_rmse": "[m/s]",
    "vel_mae": "[m/s]",
    "vel_p50": "[m/s]",
    "vel_p90": "[m/s]",
    "vel_p99": "[m/s]",
    "u_rmse": "[m/s]",
    "v_rmse": "[m/s]",
    "r_rmse": "[rad/s]",
    "u_bias": "[m/s]",
    "v_bias": "[m/s]",
    "r_bias": "[rad/s]",
    "u_nrmse": "[-]",
    "v_nrmse": "[-]",
    "r_nrmse": "[-]",
    "u_r2": "[-]",
    "v_r2": "[-]",
    "r_r2": "[-]",
    "u_corr": "[-]",
    "v_corr": "[-]",
    "r_corr": "[-]",
    "acc_rmse": "[m/s^2]",
    "traj_xy_err": "[m]",
}

PER_STEP_COLUMNS = [
    "step", "n_samples", "vel_rmse", "vel_mae", "vel_p50", "vel_p90", "vel_p99",
    "u_rmse", "v_rmse", "r_rmse", "u_bias", "v_bias", "r_bias",
    "u_r2", "v_r2", "r_r2", "u_corr", "v_corr", "r_corr",
    "u_nrmse", "v_nrmse", "r_nrmse", "acc_rmse", "traj_xy_err",
]


def _fmt(v: float, digits: int = 6) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return f"{v:.{digits}g}"


def write_per_step_csv(per_step: Dict[str, np.ndarray], path: str) -> None:
    K = per_step["step"].shape[0]
    with open(path, "w", encoding="utf-8") as f:
        header_cols = []
        for c in PER_STEP_COLUMNS:
            unit = CSV_HEADER_UNITS.get(c, "")
            header_cols.append(f"{c}{(' ' + unit) if unit else ''}")
        f.write(",".join(header_cols) + "\n")
        for k in range(K):
            row = [_fmt(per_step[c][k]) for c in PER_STEP_COLUMNS]
            f.write(",".join(row) + "\n")


def write_per_step_md(
    per_step: Dict[str, np.ndarray],
    div: Dict[str, float],
    path: str,
    tag: str,
) -> None:
    K = per_step["step"].shape[0]
    vel_at_K = float(per_step["vel_rmse"][-1])
    title = (
        f"# Per-step metrics ({tag}) — vel_rmse@{K} = {vel_at_K:.6g}, "
        f"divergence_slope_loglog = {div['slope_loglog']:.6g}, "
        f"instability_score = {div['instability_score']:.6g}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        cols_with_units = []
        for c in PER_STEP_COLUMNS:
            unit = CSV_HEADER_UNITS.get(c, "")
            cols_with_units.append(f"{c}{(' ' + unit) if unit else ''}")
        f.write("| " + " | ".join(cols_with_units) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols_with_units)) + "|\n")
        for k in range(K):
            row = [_fmt(per_step[c][k]) for c in PER_STEP_COLUMNS]
            f.write("| " + " | ".join(row) + " |\n")


def write_per_sample_csv(per_sample: Dict[str, np.ndarray], seg_idx: np.ndarray, t0: np.ndarray, K: int, path: str) -> None:
    M = per_sample["vel_err_stepK"].shape[0]
    header = (
        f"sample_idx,seg_idx,t_start,"
        f"vel_err_step{K} [m/s],u_err_step{K} [m/s],v_err_step{K} [m/s],r_err_step{K} [rad/s],"
        f"traj_xy_err_step{K} [m],divergence_ratio,divergence_slope\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i in range(M):
            f.write(
                f"{i},{int(seg_idx[i])},{int(t0[i])},"
                f"{_fmt(per_sample['vel_err_stepK'][i])},{_fmt(per_sample['u_err_stepK'][i])},"
                f"{_fmt(per_sample['v_err_stepK'][i])},{_fmt(per_sample['r_err_stepK'][i])},"
                f"{_fmt(per_sample['traj_xy_err_stepK'][i])},"
                f"{_fmt(per_sample['divergence_ratio'][i])},{_fmt(per_sample['divergence_slope'][i])}\n"
            )


def build_summary_dict(
    tag: str,
    ckpt_path: str,
    n_samples: int,
    pred_len: int,
    dt: float,
    per_step: Dict[str, np.ndarray],
    div: Dict[str, float],
    per_sample: Dict[str, np.ndarray],
) -> Dict:
    K = pred_len
    vel = per_step["vel_rmse"]

    # 选定关键步 (1, 5, 10, 20)，超过 K 的填 nan
    def at(k: int, arr: np.ndarray) -> float:
        return float(arr[k - 1]) if 1 <= k <= K else float("nan")

    aggregate = {
        "vel_rmse_mean": float(np.mean(vel)),
        "vel_rmse_step_1": at(1, vel),
        "vel_rmse_step_5": at(5, vel),
        "vel_rmse_step_10": at(10, vel),
        f"vel_rmse_step_{K}": at(K, vel),
        f"u_rmse_step_{K}": at(K, per_step["u_rmse"]),
        f"v_rmse_step_{K}": at(K, per_step["v_rmse"]),
        f"r_rmse_step_{K}": at(K, per_step["r_rmse"]),
        "acc_rmse_mean": float(np.nanmean(per_step["acc_rmse"])),
        f"traj_xy_rmse_step_{K}": at(K, per_step["traj_xy_err"]),
    }
    # 兼容 PROMPT 文档里的 step_20 命名（即使 K!=20 也保留主键）
    if K != 20:
        aggregate["vel_rmse_step_K"] = at(K, vel)

    div_ext = dict(div)
    div_ext["divergent_sample_pct"] = float(np.mean(per_sample["divergence_ratio"] > 5.0) * 100.0)

    summary = {
        "tag": tag,
        "ckpt": ckpt_path,
        "n_samples": int(n_samples),
        "pred_len": int(pred_len),
        "dt": float(dt),
        "aggregate": aggregate,
        "divergence": div_ext,
        "channel_bias": {
            "u_bias_mean": float(np.mean(per_step["u_bias"])),
            "v_bias_mean": float(np.mean(per_step["v_bias"])),
            "r_bias_mean": float(np.mean(per_step["r_bias"])),
        },
        "tail": {
            f"vel_err_step{K}_p50": float(np.percentile(per_sample["vel_err_stepK"], 50)),
            f"vel_err_step{K}_p90": float(np.percentile(per_sample["vel_err_stepK"], 90)),
            f"vel_err_step{K}_p99": float(np.percentile(per_sample["vel_err_stepK"], 99)),
            f"vel_err_step{K}_max": float(np.max(per_sample["vel_err_stepK"])),
        },
        "fit_quality": {
            "u_r2_mean": float(np.nanmean(per_step["u_r2"])),
            "v_r2_mean": float(np.nanmean(per_step["v_r2"])),
            "r_r2_mean": float(np.nanmean(per_step["r_r2"])),
            "u_corr_mean": float(np.nanmean(per_step["u_corr"])),
            "v_corr_mean": float(np.nanmean(per_step["v_corr"])),
            "r_corr_mean": float(np.nanmean(per_step["r_corr"])),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# 5. 绘图
# ---------------------------------------------------------------------------

def _pick_quantile_indices(values: np.ndarray, n: int) -> np.ndarray:
    """按 values 升序的 n 个均匀分位点选出 n 个样本下标。"""
    M = values.shape[0]
    n = min(n, M)
    sorted_idx = np.argsort(values)
    quant_pos = np.linspace(0, M - 1, n).astype(int)
    return sorted_idx[quant_pos]


def plot_velocity_curve_grid(
    gt_dyn: np.ndarray, pred_dyn: np.ndarray, per_sample: Dict[str, np.ndarray],
    path: str, tag: str,
) -> None:
    """6×3 网格：18 条按 step20 误差均匀分位选取的样本，u/v/r 时间曲线。"""
    M, K, _ = gt_dyn.shape
    # 6 行 × 3 列 = 18 条样本（每行展示 1 条样本的 u/v/r）
    pick = _pick_quantile_indices(per_sample["vel_err_stepK"], 18)
    n_rows = len(pick)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, max(n_rows * 1.6, 4.5)), squeeze=False)
    steps = np.arange(1, K + 1)
    titles = ["u [m/s]", "v [m/s]", "r [rad/s]"]
    for row, sample_idx in enumerate(pick):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(steps, gt_dyn[sample_idx, :, col], "g-", label="GT", lw=1.4)
            ax.plot(steps, pred_dyn[sample_idx, :, col], "r--", label="Pred", lw=1.2)
            verr = per_sample["vel_err_stepK"][sample_idx]
            dr = per_sample["divergence_ratio"][sample_idx]
            ax.set_title(f"#{sample_idx} {titles[col]}\nvel_err@{K}={verr:.3g} ratio={dr:.2g}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"{tag} velocity curves (18 samples by step{K} err quantile)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_vs_step(per_step: Dict[str, np.ndarray], per_sample_full: np.ndarray, path: str, tag: str) -> None:
    """折线 + [P10, P90] 阴影，线性 + log 双子图。
    per_sample_full: (M, K) 每样本每步 vel_horiz_err，用于分位带。"""
    K = per_step["vel_rmse"].shape[0]
    steps = np.arange(1, K + 1)
    p10 = np.percentile(per_sample_full, 10, axis=0)
    p90 = np.percentile(per_sample_full, 90, axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax in (ax1, ax2):
        ax.fill_between(steps, p10, p90, alpha=0.25, color="C0", label="[P10, P90]")
        ax.plot(steps, per_step["vel_rmse"], "o-", color="C0", label="vel_rmse")
        ax.plot(steps, per_step["vel_p50"], "s--", color="C2", label="vel_p50", alpha=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel("vel error [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_title(f"{tag} vel error vs step (linear)")
    ax2.set_title(f"{tag} vel error vs step (log y)")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_band_per_channel(
    gt_dyn: np.ndarray, pred_dyn: np.ndarray, path: str, tag: str
) -> None:
    M, K, _ = gt_dyn.shape
    diff = np.abs(pred_dyn - gt_dyn)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ["|u err| [m/s]", "|v err| [m/s]", "|r err| [rad/s]"]
    steps = np.arange(1, K + 1)
    for ci, ax in enumerate(axes):
        rmse = np.sqrt(np.mean(diff[..., ci] ** 2, axis=0))
        p50 = np.percentile(diff[..., ci], 50, axis=0)
        p90 = np.percentile(diff[..., ci], 90, axis=0)
        ax.fill_between(steps, p50, p90, alpha=0.25, color="C1", label="[P50, P90]")
        ax.plot(steps, rmse, "o-", color="C1", label="RMSE")
        ax.plot(steps, p50, "s--", color="C3", label="P50", alpha=0.6)
        ax.set_title(f"{tag} {titles[ci]} vs step", fontsize=10)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_step_K_error_hist(per_sample: Dict[str, np.ndarray], path: str, tag: str, K: int) -> None:
    err = per_sample["vel_err_stepK"]
    p50, p90, p99 = np.percentile(err, [50, 90, 99])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(err, bins=60, color="C0", alpha=0.6, density=True)
    # 简易 KDE：高斯核
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(err)
        xs = np.linspace(err.min(), err.max(), 256)
        ax.plot(xs, kde(xs), "k-", lw=1.5, label="KDE")
    except Exception:
        pass
    for q, name, c in [(p50, "P50", "g"), (p90, "P90", "orange"), (p99, "P99", "r")]:
        ax.axvline(q, color=c, linestyle="--", label=f"{name}={q:.3g}")
    ax.set_xlabel(f"vel err @ step {K} [m/s]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"{tag} step{K} velocity error distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bias_vs_step(per_step: Dict[str, np.ndarray], path: str, tag: str) -> None:
    K = per_step["step"].shape[0]
    steps = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ci, name in enumerate(["u", "v", "r"]):
        ax = axes[ci]
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.plot(steps, per_step[f"{name}_bias"], "o-", color=f"C{ci}", label=f"{name} bias")
        ax.set_xlabel("step")
        unit = "m/s" if name != "r" else "rad/s"
        ax.set_ylabel(f"{name} bias [{unit}]")
        ax.set_title(f"{tag} {name} bias vs step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_trajectory_grid(
    gt_xy: np.ndarray, pred_xy: np.ndarray, per_sample: Dict[str, np.ndarray],
    path: str, tag: str,
) -> None:
    pick = _pick_quantile_indices(per_sample["vel_err_stepK"], 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, sample_idx in enumerate(pick):
        ax = axes[i // 3, i % 3]
        ax.plot(gt_xy[sample_idx, :, 0], gt_xy[sample_idx, :, 1], "g-o", label="GT", alpha=0.7, ms=3)
        ax.plot(pred_xy[sample_idx, :, 0], pred_xy[sample_idx, :, 1], "r--x", label="Pred", ms=3)
        ax.set_title(
            f"#{sample_idx} traj@stepK err={per_sample['traj_xy_err_stepK'][sample_idx]:.3g} m",
            fontsize=9,
        )
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{tag} trajectory grid (6 samples by stepK err quantile)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 评估单个 ckpt
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    tag: str
    ckpt_path: str
    summary: Dict
    per_step: Dict[str, np.ndarray]
    per_sample: Dict[str, np.ndarray]
    gt_dyn: np.ndarray
    pred_dyn: np.ndarray
    gt_xy: np.ndarray
    pred_xy: np.ndarray
    vel_horiz_err: np.ndarray  # (M,K)
    seg_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    per_segment: Dict[int, Dict[str, float]] = field(default_factory=dict)
    seg_order_by_v2: Optional[np.ndarray] = None


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """根据 ckpt['model_class'] 自动 dispatch v1/v2/v3。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]
    sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model_class = ckpt.get("model_class", "HorizontalKoopmanModel")
    args_d = ckpt.get("args", {}) or {}
    if model_class == "HorizontalKoopmanModelV3":
        if HorizontalKoopmanModelV3 is None:
            raise RuntimeError(
                "ckpt 是 v3 模型但当前进程无 koopman_v3 模块，无法加载。"
            )
        model = HorizontalKoopmanModelV3(
            state_dim=3, control_dim=4,
            hidden_dim=int(args_d.get("hidden_dim", 24)),
            n_cubic=int(args_d.get("n_cubic", 11)),
            clamp_pif=float(args_d.get("clamp_pif", 5.0)),
        )
    else:
        # v1/v2 共用同一个 HorizontalKoopmanModel
        model = HorizontalKoopmanModel(state_dim=3, control_dim=4, hidden_dim=24)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, stats


PER_SEGMENT_COLUMNS = [
    "seg_idx", "n_windows",
    "vel_rmse_1", "vel_rmse_5", "vel_rmse_10", "vel_rmse_K",
    "u_rmse_K", "v_rmse_K", "r_rmse_K", "u_bias_K", "traj_xy_rmse_K",
    "vel_err_p50_K", "vel_err_p90_K", "vel_err_p99_K", "vel_err_max_K",
    "ratio_K_over_1", "slope_loglog", "instability_score",
    "u_mean", "u_std", "v_std", "r_std", "dthr_mean",
]


def _segment_raw_stats(states_full: np.ndarray, ctrls_full: np.ndarray,
                       seg_starts: np.ndarray, seg_lens: np.ndarray,
                       sidx: int) -> Dict[str, float]:
    """段内原始（未归一化）状态/控制统计量。

    控制量 dthr_mean = 该段内 4 个推进器之间标准差的时间平均（粗略表征异轮转
    强度，与 PROMPT v2 归因里的 dthr 含义一致）。
    """
    s0 = int(seg_starts[sidx])
    T = int(seg_lens[sidx])
    if T <= 0:
        return {"u_mean": float("nan"), "u_std": float("nan"),
                "v_std": float("nan"), "r_std": float("nan"),
                "dthr_mean": float("nan")}
    seg_state = states_full[s0:s0 + T]
    seg_ctrl = ctrls_full[s0:s0 + T]
    u_arr = seg_state[:, 3]
    v_arr = seg_state[:, 4]
    r_arr = seg_state[:, 5]
    # dthr：相邻推进器差值的逐时刻标准差，再做时间均值
    dthr_per_t = seg_ctrl.std(axis=1) if seg_ctrl.shape[1] > 1 else np.zeros(T)
    return {
        "u_mean": float(np.mean(u_arr)),
        "u_std": float(np.std(u_arr)),
        "v_std": float(np.std(v_arr)),
        "r_std": float(np.std(r_arr)),
        "dthr_mean": float(np.mean(dthr_per_t)),
    }


def compute_per_segment_metrics(
    gt_dyn: np.ndarray, pred_dyn: np.ndarray,
    gt_xy: np.ndarray, pred_xy: np.ndarray,
    seg_idx_arr: np.ndarray,
    states_full: np.ndarray, ctrls_full: np.ndarray,
    seg_starts: np.ndarray, seg_lens: np.ndarray,
    K: int,
) -> Dict[int, Dict[str, float]]:
    """每段聚合：返回 {seg_idx: dict[列名 -> 值]}。"""
    eps = 1e-6
    out: Dict[int, Dict[str, float]] = {}
    unique_segs = np.unique(seg_idx_arr)
    diff = pred_dyn - gt_dyn
    vel_err = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)  # (M, K)
    traj_err = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))  # (M, K)

    log_steps = np.log(np.arange(1, K + 1, dtype=np.float64))
    A = np.vstack([log_steps, np.ones_like(log_steps)]).T

    for sidx in unique_segs:
        mask = (seg_idx_arr == sidx)
        n_win = int(mask.sum())
        if n_win == 0:
            continue
        d_seg = diff[mask]                   # (n_win, K, 3)
        verr_seg = vel_err[mask]             # (n_win, K)
        terr_seg = traj_err[mask]            # (n_win, K)
        # NOTE: PROMPT v3 §2 中标的 v2 baseline (worst=0.01961, best=0.00566,
        # ratio=3.47) 使用的是「mean(vel_horiz_err)」而非 sqrt-mean-square。
        # 为了让 §7.2 的 S1/S2/S3 绝对阈值与基线对齐，per-segment 的
        # vel_rmse_{1,5,10,K} 字段在此按 PROMPT 定义采用 mean（注意命名
        # 沿用 PROMPT §6.2 spec 的 "vel_rmse_K"）。
        vel_rmse_step = np.mean(verr_seg, axis=0)  # (K,)
        u_rmse_K = float(np.sqrt(np.mean(d_seg[:, -1, 0] ** 2)))
        v_rmse_K = float(np.sqrt(np.mean(d_seg[:, -1, 1] ** 2)))
        r_rmse_K = float(np.sqrt(np.mean(d_seg[:, -1, 2] ** 2)))
        u_bias_K = float(np.mean(d_seg[:, -1, 0]))
        traj_xy_rmse_K = float(np.sqrt(np.mean(terr_seg[:, -1] ** 2)))

        verr_K = verr_seg[:, -1]
        ratio = float(verr_seg[:, -1].mean() / max(verr_seg[:, 0].mean(), eps))
        log_vel = np.log(np.maximum(vel_rmse_step.astype(np.float64), eps))
        slope_ll, _ = np.linalg.lstsq(A, log_vel, rcond=None)[0]
        sig = 1.0 / (1.0 + np.exp(-slope_ll))
        inst = float(sig * (1.0 + ratio / 10.0))

        raw = _segment_raw_stats(states_full, ctrls_full, seg_starts, seg_lens, int(sidx))

        row = {
            "seg_idx": int(sidx),
            "n_windows": n_win,
            "vel_rmse_1": float(vel_rmse_step[0]),
            "vel_rmse_5": float(vel_rmse_step[min(4, K - 1)]),
            "vel_rmse_10": float(vel_rmse_step[min(9, K - 1)]),
            "vel_rmse_K": float(vel_rmse_step[-1]),
            "u_rmse_K": u_rmse_K, "v_rmse_K": v_rmse_K, "r_rmse_K": r_rmse_K,
            "u_bias_K": u_bias_K, "traj_xy_rmse_K": traj_xy_rmse_K,
            "vel_err_p50_K": float(np.percentile(verr_K, 50)),
            "vel_err_p90_K": float(np.percentile(verr_K, 90)),
            "vel_err_p99_K": float(np.percentile(verr_K, 99)),
            "vel_err_max_K": float(np.max(verr_K)),
            "ratio_K_over_1": ratio,
            "slope_loglog": float(slope_ll),
            "instability_score": inst,
            **raw,
        }
        out[int(sidx)] = row
    return out


def write_per_segment_csv(per_seg: Dict[int, Dict[str, float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(PER_SEGMENT_COLUMNS) + "\n")
        for sidx in sorted(per_seg.keys()):
            row = per_seg[sidx]
            f.write(",".join(_fmt(row.get(c, float("nan"))) for c in PER_SEGMENT_COLUMNS) + "\n")


def write_per_segment_md(per_seg: Dict[int, Dict[str, float]], path: str, tag: str) -> None:
    K_keys = list(per_seg.keys())
    if not K_keys:
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# per-segment metrics ({tag}) — no segments\n")
        return
    velK = {k: v["vel_rmse_K"] for k, v in per_seg.items()}
    worst = max(velK, key=lambda k: velK[k])
    best = min(velK, key=lambda k: velK[k])
    ratio_wb = velK[worst] / max(velK[best], 1e-9)
    high_speed = [v["vel_rmse_K"] for v in per_seg.values() if v.get("u_mean", 0.0) > 3.0]
    low_speed = [v["vel_rmse_K"] for v in per_seg.values() if v.get("u_mean", 0.0) <= 3.0]
    hi_mean = float(np.mean(high_speed)) if high_speed else float("nan")
    lo_mean = float(np.mean(low_speed)) if low_speed else float("nan")
    title = (
        f"# per-segment metrics ({tag}) — worst_seg = {worst} (vel@K={velK[worst]:.6g}); "
        f"best_seg = {best} (vel@K={velK[best]:.6g}); "
        f"段间 mean 差距 = {ratio_wb:.3g}×; "
        f"high_speed_mean = {hi_mean:.6g}; low_speed_mean = {lo_mean:.6g}\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("| " + " | ".join(PER_SEGMENT_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(PER_SEGMENT_COLUMNS)) + "|\n")
        for sidx in sorted(per_seg.keys()):
            row = per_seg[sidx]
            f.write("| " + " | ".join(_fmt(row.get(c, float("nan"))) for c in PER_SEGMENT_COLUMNS) + " |\n")


def build_per_segment_summary(per_seg: Dict[int, Dict[str, float]]) -> Dict:
    if not per_seg:
        return {}
    velK = {k: v["vel_rmse_K"] for k, v in per_seg.items()}
    worst = max(velK, key=lambda k: velK[k])
    best = min(velK, key=lambda k: velK[k])
    ratio_wb = velK[worst] / max(velK[best], 1e-9)
    velK_arr = np.array(list(velK.values()), dtype=np.float64)
    high = [v["vel_rmse_K"] for v in per_seg.values() if v.get("u_mean", 0.0) > 3.0]
    low = [v["vel_rmse_K"] for v in per_seg.values() if v.get("u_mean", 0.0) <= 3.0]
    return {
        "worst_seg_idx": int(worst),
        "worst_vel_rmse_K": float(velK[worst]),
        "best_seg_idx": int(best),
        "best_vel_rmse_K": float(velK[best]),
        "ratio_worst_over_best": float(ratio_wb),
        "per_seg_vel_rmse_K_std": float(velK_arr.std()),
        "high_speed_seg_mean": float(np.mean(high)) if high else float("nan"),
        "low_speed_seg_mean": float(np.mean(low)) if low else float("nan"),
        "high_speed_seg_count": int(len(high)),
        "low_speed_seg_count": int(len(low)),
    }


def evaluate_one(
    ckpt_path: str,
    data_path: str,
    pred_len: int,
    dt: float,
    tag: str,
    out_dir: str,
    device: torch.device,
    batch_size: int = 1024,
    max_samples: Optional[int] = None,
    write_files: bool = True,
    baseline_ckpt: Optional[str] = None,
) -> EvalResult:
    """Evaluate one ckpt.

    ``baseline_ckpt`` (PROMPT_v3a §4.4): 若给出且存在，则额外计算
    逐样本 ``degraded_pct_vs_v1`` 并写入 ``summary["s4"]``。该字段为
    *额外* 字段，**不传 baseline_ckpt 时输出与历史 v3 产物逐位一致**
    （PROMPT_v3a §3 回归保护要求）。
    """
    os.makedirs(out_dir, exist_ok=True)
    model, stats = load_model_from_ckpt(ckpt_path, device)
    states_full, ctrls_full, seg_starts, seg_lens, t0_global, seg_idx, t0_local = _flatten_segments(
        data_path, pred_len=pred_len, stride=1
    )
    if max_samples is not None and t0_global.shape[0] > max_samples:
        # 等距取样以保留 segment 多样性
        sel = np.linspace(0, t0_global.shape[0] - 1, max_samples).astype(int)
        t0_global = t0_global[sel]
        seg_idx = seg_idx[sel]
        t0_local = t0_local[sel]
    gt_dyn, pred_dyn, gt_xy, pred_xy = rollout_dataset(
        model, states_full, ctrls_full, t0_global, pred_len, stats, device, dt, batch_size
    )

    per_step = compute_per_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy, dt)
    div = compute_divergence_metrics(per_step)
    per_sample = compute_per_sample_step_metrics(gt_dyn, pred_dyn, gt_xy, pred_xy)
    summary = build_summary_dict(
        tag=tag, ckpt_path=ckpt_path, n_samples=int(gt_dyn.shape[0]),
        pred_len=pred_len, dt=dt, per_step=per_step, div=div, per_sample=per_sample,
    )
    # per-segment 指标（PROMPT v3 §6.2/§6.4）
    per_seg = compute_per_segment_metrics(
        gt_dyn, pred_dyn, gt_xy, pred_xy, seg_idx,
        states_full, ctrls_full, seg_starts, seg_lens, K=pred_len,
    )
    summary["per_segment"] = build_per_segment_summary(per_seg)

    # PROMPT_v3a §4.4 - 可选 degraded_pct vs baseline ckpt（不传则不输出此字段）
    if baseline_ckpt is not None and os.path.exists(baseline_ckpt):
        try:
            b_model, _ = load_model_from_ckpt(baseline_ckpt, device)
            b_gt, b_pred, _, _ = rollout_dataset(
                b_model, states_full, ctrls_full, t0_global, pred_len, stats, device, dt, batch_size,
            )
            b_diff = b_pred - b_gt
            b_vel_err_K = np.sqrt(b_diff[:, -1, 0] ** 2 + b_diff[:, -1, 1] ** 2)
            cur_vel_err_K = per_sample["vel_err_stepK"]
            n = min(cur_vel_err_K.shape[0], b_vel_err_K.shape[0])
            deg = float(np.mean(cur_vel_err_K[:n] > b_vel_err_K[:n]) * 100.0)
            summary["s4"] = {
                "baseline_ckpt": baseline_ckpt,
                "degraded_pct_vs_baseline": deg,
                "n_samples_compared": int(n),
            }
        except Exception as _e:
            summary["s4"] = {
                "baseline_ckpt": baseline_ckpt,
                "error": str(_e),
            }

    diff = pred_dyn - gt_dyn
    vel_horiz_err = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)

    if write_files:
        write_per_step_csv(per_step, os.path.join(out_dir, f"{tag}_per_step_metrics.csv"))
        write_per_step_md(per_step, div, os.path.join(out_dir, f"{tag}_per_step_metrics.md"), tag)
        write_per_sample_csv(per_sample, seg_idx, t0_local, pred_len,
                             os.path.join(out_dir, f"{tag}_per_sample_step{pred_len}.csv"))
        write_per_segment_csv(per_seg, os.path.join(out_dir, f"{tag}_per_segment_metrics.csv"))
        write_per_segment_md(per_seg, os.path.join(out_dir, f"{tag}_per_segment_metrics.md"), tag)
        with open(os.path.join(out_dir, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        plot_velocity_curve_grid(gt_dyn, pred_dyn, per_sample,
                                 os.path.join(out_dir, f"{tag}_velocity_curve_grid.png"), tag)
        plot_error_vs_step(per_step, vel_horiz_err,
                           os.path.join(out_dir, f"{tag}_error_vs_step.png"), tag)
        plot_error_band_per_channel(gt_dyn, pred_dyn,
                                    os.path.join(out_dir, f"{tag}_error_band_per_channel.png"), tag)
        plot_step_K_error_hist(per_sample,
                               os.path.join(out_dir, f"{tag}_step{pred_len}_error_hist.png"), tag, pred_len)
        plot_bias_vs_step(per_step, os.path.join(out_dir, f"{tag}_bias_vs_step.png"), tag)
        plot_trajectory_grid(gt_xy, pred_xy, per_sample,
                             os.path.join(out_dir, f"{tag}_trajectory_grid.png"), tag)

    return EvalResult(
        tag=tag, ckpt_path=ckpt_path, summary=summary, per_step=per_step,
        per_sample=per_sample, gt_dyn=gt_dyn, pred_dyn=pred_dyn,
        gt_xy=gt_xy, pred_xy=pred_xy, vel_horiz_err=vel_horiz_err,
        seg_idx=seg_idx, per_segment=per_seg,
    )


# ---------------------------------------------------------------------------
# 7. 多 ckpt 对比
# ---------------------------------------------------------------------------


def write_compare_csv_md(results: List[EvalResult], out_dir: str) -> Tuple[str, str]:
    csv_path = os.path.join(out_dir, "compare_summary.csv")
    md_path = os.path.join(out_dir, "compare_summary.md")
    # 拍平 aggregate + divergence
    all_keys: List[str] = []
    flat_rows: List[Dict[str, str]] = []
    for r in results:
        flat: Dict[str, str] = {"tag": r.tag, "ckpt": r.ckpt_path,
                                "n_samples": str(r.summary["n_samples"]),
                                "pred_len": str(r.summary["pred_len"])}
        for k, v in r.summary["aggregate"].items():
            flat[f"agg.{k}"] = _fmt(v)
        for k, v in r.summary["divergence"].items():
            flat[f"div.{k}"] = _fmt(v) if not isinstance(v, bool) else str(v)
        for k, v in r.summary["channel_bias"].items():
            flat[f"bias.{k}"] = _fmt(v)
        for k, v in r.summary["tail"].items():
            flat[f"tail.{k}"] = _fmt(v)
        for k, v in r.summary["fit_quality"].items():
            flat[f"fit.{k}"] = _fmt(v)
        for k in flat:
            if k not in all_keys:
                all_keys.append(k)
        flat_rows.append(flat)
    # CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(all_keys) + "\n")
        for row in flat_rows:
            f.write(",".join(row.get(k, "") for k in all_keys) + "\n")
    # MD
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comparison summary\n\n")
        f.write("| " + " | ".join(all_keys) + " |\n")
        f.write("|" + "|".join(["---"] * len(all_keys)) + "|\n")
        for row in flat_rows:
            f.write("| " + " | ".join(row.get(k, "") for k in all_keys) + " |\n")
        f.write("\n## Verdict\n\n")
        f.write(_compare_verdict_text_with_v3(results))
    return csv_path, md_path


def _compare_verdict_text_with_v3(results: List[EvalResult]) -> str:
    """若有 v3 结果，则在末尾补 12 阈值 verdict + 3 阶字典归因；
    若有 v3a 结果，则额外补 14 阈值 verdict + plan-A 归因。"""
    base_text = _compare_verdict_text(results)
    v3a_res = next((r for r in results if r.tag == "v3a"), None)
    v3_res = next((r for r in results if r.tag == "v3"), None)
    v2_res = next((r for r in results if r.tag == "v2"), None)
    v1_res = next((r for r in results if r.tag == "v1"), None)
    extra = ""
    if v3_res is not None:
        v3_md, _ = v3_threshold_verdict(v3_res, v1_res=v1_res, v2_res=v2_res)
        extra += "\n" + v3_md
        attribution = cubic_dictionary_attribution(v3_res, v2_res)
        if attribution:
            extra += "\n" + attribution
    if v3a_res is not None:
        v3a_md, _ = v3a_threshold_verdict(v3a_res, v1_res=v1_res, v2_res=v2_res, v3_res=v3_res)
        extra += "\n" + v3a_md
        plan_a_md = plan_a_attribution(v3a_res, v3_res, v2_res)
        if plan_a_md:
            extra += "\n" + plan_a_md
    return base_text + extra


def plan_a_attribution(
    v3a_res: EvalResult,
    v3_res: Optional[EvalResult],
    v2_res: Optional[EvalResult],
) -> str:
    """PROMPT_v3a §9 - A1+A2+A3 三件事的数字归因（v3a vs v3 / v2）。"""
    if v3_res is None:
        return ""
    K = v3a_res.summary["pred_len"]
    a_v3a = v3a_res.summary["aggregate"]
    a_v3 = v3_res.summary["aggregate"]
    a_v2 = v2_res.summary["aggregate"] if v2_res is not None else {}
    b_v3a = v3a_res.summary["channel_bias"]
    b_v3 = v3_res.summary["channel_bias"]
    b_v2 = v2_res.summary["channel_bias"] if v2_res is not None else {}
    d_v3a = v3a_res.summary["divergence"]
    d_v3 = v3_res.summary["divergence"]
    ps_v3a = v3a_res.summary.get("per_segment", {}) or {}
    ps_v3 = v3_res.summary.get("per_segment", {}) or {}

    lines = []
    lines.append("### PROMPT_v3a A1+A2+A3 三件事归因（v3a vs v3，全部来自 summary.json）")
    lines.append("")
    lines.append("| 维度 | v2 | v3 | v3a | v3a/v3 | 说明 |")
    lines.append("|---|---|---|---|---|---|")
    # A1: v 通道压住了多少
    v2v = a_v2.get(f"v_rmse_step_{K}", float("nan"))
    v3v = a_v3.get(f"v_rmse_step_{K}", float("nan"))
    v3av = a_v3a.get(f"v_rmse_step_{K}", float("nan"))
    ratio_v = v3av / v3v if v3v > 0 else float("nan")
    lines.append(
        f"| A1: v_rmse@K | {_fmt(v2v)} | {_fmt(v3v)} | {_fmt(v3av)} | "
        f"{_fmt(ratio_v)} | A1 直接目标，越低越好 |"
    )
    v2vb = abs(b_v2.get("v_bias_mean", float("nan"))) if b_v2 else float("nan")
    v3vb = abs(b_v3.get("v_bias_mean", float("nan")))
    v3avb = abs(b_v3a.get("v_bias_mean", float("nan")))
    lines.append(
        f"| A1: |v_bias| | {_fmt(v2vb)} | {_fmt(v3vb)} | {_fmt(v3avb)} | "
        f"{_fmt(v3avb / v3vb if v3vb > 0 else float('nan'))} | A1 副指标 |"
    )
    # G4 slope
    s3 = d_v3.get("slope_loglog", float("nan"))
    s3a = d_v3a.get("slope_loglog", float("nan"))
    lines.append(
        f"| A2: slope_loglog (G4) | - | {_fmt(s3)} | {_fmt(s3a)} | "
        f"{_fmt(s3a / s3 if s3 > 0 else float('nan'))} | "
        "composite_v3a 显式加罚 |"
    )
    # G5 instability
    i3 = d_v3.get("instability_score", float("nan"))
    i3a = d_v3a.get("instability_score", float("nan"))
    lines.append(
        f"| A2: instability (G5) | - | {_fmt(i3)} | {_fmt(i3a)} | "
        f"{_fmt(i3a / i3 if i3 > 0 else float('nan'))} | composite 中 max(1, inst) |"
    )
    # A3 worst seg & high_speed
    ws3 = ps_v3.get("worst_vel_rmse_K", float("nan"))
    ws3a = ps_v3a.get("worst_vel_rmse_K", float("nan"))
    lines.append(
        f"| A3: worst_seg vel@K (S1) | - | {_fmt(ws3)} | {_fmt(ws3a)} | "
        f"{_fmt(ws3a / ws3 if ws3 > 0 else float('nan'))} | seg_resample 直接目标 |"
    )
    hs3 = ps_v3.get("high_speed_seg_mean", float("nan"))
    hs3a = ps_v3a.get("high_speed_seg_mean", float("nan"))
    lines.append(
        f"| A3: high_speed_seg_mean (S3) | - | {_fmt(hs3)} | {_fmt(hs3a)} | "
        f"{_fmt(hs3a / hs3 if hs3 > 0 else float('nan'))} | seg_resample 衍生效果 |"
    )
    # Aggregate vel@K
    vk3 = a_v3.get(f"vel_rmse_step_{K}", float("nan"))
    vk3a = a_v3a.get(f"vel_rmse_step_{K}", float("nan"))
    lines.append(
        f"| 总:vel_rmse@K (G1) | {_fmt(a_v2.get(f'vel_rmse_step_{K}', float('nan')))} | "
        f"{_fmt(vk3)} | {_fmt(vk3a)} | "
        f"{_fmt(vk3a / vk3 if vk3 > 0 else float('nan'))} | 总精度指标 |"
    )
    lines.append("")
    return "\n".join(lines)


def _compare_verdict_text(results: List[EvalResult]) -> str:
    """脚本规则化结论摘要。"""
    K = results[0].summary["pred_len"]
    by_step_K = {r.tag: r.summary["aggregate"].get(f"vel_rmse_step_{K}", float("nan")) for r in results}
    by_slope = {r.tag: r.summary["divergence"]["slope_loglog"] for r in results}
    by_inst = {r.tag: r.summary["divergence"]["instability_score"] for r in results}
    by_step1 = {r.tag: r.summary["aggregate"]["vel_rmse_step_1"] for r in results}

    best_K_tag = min(by_step_K, key=lambda k: by_step_K[k])
    best_slope_tag = min(by_slope, key=lambda k: by_slope[k])
    best_inst_tag = min(by_inst, key=lambda k: by_inst[k])

    lines: List[str] = []
    lines.append(f"- **vel_rmse_step_{K} 最低**: `{best_K_tag}` ({by_step_K[best_K_tag]:.6g} m/s)")
    lines.append(f"- **slope_loglog 最低（最不发散）**: `{best_slope_tag}` ({by_slope[best_slope_tag]:.6g})")
    lines.append(f"- **instability_score 最低**: `{best_inst_tag}` ({by_inst[best_inst_tag]:.6g})")

    # 是否随版本下降（按 results 列表顺序看 instability_score 是否单调递减）
    insts = [by_inst[r.tag] for r in results]
    if len(insts) >= 2 and all(insts[i] >= insts[i + 1] for i in range(len(insts) - 1)):
        lines.append("- ✅ `instability_score` 随版本顺序单调下降。")
    else:
        lines.append("- ⚠️ `instability_score` 未随版本顺序单调下降，请检查改进方向。")

    # 短程退化、长程改善 提示
    if len(results) >= 2:
        first, last = results[0], results[-1]
        s1_first = first.summary["aggregate"]["vel_rmse_step_1"]
        s1_last = last.summary["aggregate"]["vel_rmse_step_1"]
        sK_first = first.summary["aggregate"].get(f"vel_rmse_step_{K}", float("nan"))
        sK_last = last.summary["aggregate"].get(f"vel_rmse_step_{K}", float("nan"))
        if s1_last > s1_first * 1.02 and sK_last < sK_first * 0.98:
            lines.append(
                "- 提示：新 ckpt 的 `vel_rmse_step_1` 反而变差但 `step_K` 变好，"
                "可能是过强的 latent 一致性损失在压短程，可适当下调 `--w_lin`。"
            )
        # 9.4 验收（首尾对比）
        ratio_K = (sK_first / sK_last) if sK_last > 1e-9 else float("inf")
        slope_first = first.summary["divergence"]["slope_loglog"]
        slope_last = last.summary["divergence"]["slope_loglog"]
        # 处理可能为负的 slope：用 abs 比例不合适；改用绝对值差
        slope_drop_pct = (slope_first - slope_last) / max(abs(slope_first), 1e-6) * 100.0
        inst_drop_pct = (
            (first.summary["divergence"]["instability_score"] - last.summary["divergence"]["instability_score"])
            / max(abs(first.summary["divergence"]["instability_score"]), 1e-6) * 100.0
        )
        vel_drop_pct = (sK_first - sK_last) / max(abs(sK_first), 1e-6) * 100.0
        cond_a = (vel_drop_pct >= 30.0) and (slope_drop_pct >= 20.0)
        cond_b = (inst_drop_pct >= 25.0) and (sK_last <= sK_first)
        passed = cond_a or cond_b
        lines.append("")
        lines.append("### 9.4 硬性指标门槛（first → last，正号 = 改善 / 下降）")
        lines.append(f"- `vel_rmse_step_{K}`: {sK_first:.6g} → {sK_last:.6g}  (↓ {vel_drop_pct:+.1f}%)")
        lines.append(f"- `slope_loglog`: {slope_first:.6g} → {slope_last:.6g}  (↓ {slope_drop_pct:+.1f}%)")
        lines.append(f"- `instability_score`: {first.summary['divergence']['instability_score']:.6g} → "
                     f"{last.summary['divergence']['instability_score']:.6g}  (↓ {inst_drop_pct:+.1f}%)")
        lines.append(f"- 条件 A (vel↓≥30% **且** slope↓≥20%) = **{cond_a}**")
        lines.append(f"- 条件 B (inst↓≥25% **且** vel 不上升) = **{cond_b}**")
        lines.append(f"- **{'✅ PASS' if passed else '❌ FAIL'}** — 9.4 验收。")
    return "\n".join(lines) + "\n"


def plot_compare_error_vs_step(results: List[EvalResult], path: str) -> None:
    K = results[0].summary["pred_len"]
    steps = np.arange(1, K + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for r in results:
        ax1.plot(steps, r.per_step["vel_rmse"], "o-", label=r.tag)
        ax2.plot(steps, r.per_step["vel_rmse"], "o-", label=r.tag)
    for ax in (ax1, ax2):
        ax.set_xlabel("step")
        ax.set_ylabel("vel rmse [m/s]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    ax1.set_title("Compare vel_rmse vs step (linear)")
    ax2.set_title("Compare vel_rmse vs step (log y)")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_compare_step_K_box(results: List[EvalResult], path: str) -> None:
    K = results[0].summary["pred_len"]
    data = [r.per_sample["vel_err_stepK"] for r in results]
    labels = [r.tag for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    ax1.boxplot(data, labels=labels, showfliers=False)
    ax1.set_ylabel(f"vel err @ step{K} [m/s]")
    ax1.set_title("Boxplot")
    ax1.grid(True, alpha=0.3)
    try:
        ax2.violinplot(data, showmeans=True)
        ax2.set_xticks(np.arange(1, len(labels) + 1))
        ax2.set_xticklabels(labels)
    except Exception:
        ax2.boxplot(data, labels=labels)
    ax2.set_ylabel(f"vel err @ step{K} [m/s]")
    ax2.set_title("Violin")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_compare_per_segment_bar(results: List[EvalResult], path: str) -> None:
    """段间柱状对比（x 轴 = seg idx，按 v2 段误差降序固定顺序）。"""
    if not results:
        return
    K = results[0].summary["pred_len"]
    # 选一个 reference 用作排序：优先 tag=='v2'，否则用第一个
    ref = next((r for r in results if r.tag == "v2"), results[0])
    if not ref.per_segment:
        return
    seg_ids_sorted = sorted(ref.per_segment.keys(),
                            key=lambda s: -ref.per_segment[s]["vel_rmse_K"])
    n_groups = len(seg_ids_sorted)
    n_runs = len(results)
    width = 0.8 / max(n_runs, 1)
    fig, ax = plt.subplots(figsize=(max(8.0, n_groups * 0.7), 5.0))
    x = np.arange(n_groups)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for j, r in enumerate(results):
        vals = []
        for s in seg_ids_sorted:
            if s in r.per_segment:
                vals.append(r.per_segment[s]["vel_rmse_K"])
            else:
                vals.append(np.nan)
        ax.bar(x + j * width - 0.4 + width / 2, vals, width=width,
               label=r.tag, color=colors[j % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seg_ids_sorted], rotation=0, fontsize=8)
    ax.set_xlabel(f"segment idx (sorted by {ref.tag} vel_rmse@{K} desc)")
    ax.set_ylabel(f"vel_rmse@{K} [m/s]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Per-segment vel_rmse@K (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_compare_u_bias_per_step(results: List[EvalResult], path: str) -> None:
    """每个 ckpt 的 u_bias_per_step（K 个值）叠图，证明 v3 把 u 漂移斜率压平。"""
    if not results:
        return
    K = results[0].summary["pred_len"]
    steps = np.arange(1, K + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for j, r in enumerate(results):
        ub = r.per_step["u_bias"]
        c = colors[j % len(colors)]
        ax1.plot(steps, ub, "o-", label=r.tag, color=c, lw=1.5)
        ax2.plot(steps, np.cumsum(ub), "o-", label=r.tag, color=c, lw=1.5)
    for ax, title, ylab in [
        (ax1, "u_bias per step (target=0)", "u_bias [m/s]"),
        (ax2, "cumulative sum of u_bias per step (proxy for drift)", "cumsum u_bias [m/s]"),
    ]:
        ax.axhline(0, color="k", linestyle="--", alpha=0.4)
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("u-bias drift profile (lower |slope| is better)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12-threshold verdict (PROMPT v3 §7)
# ---------------------------------------------------------------------------

# 阈值常量；与 PROMPT §7 一一对应。
V3_THRESHOLDS = {
    # 全局 5
    "G1_vel_rmse_step_K": 0.01285,
    "G2_u_rmse_step_K":   0.01185,
    "G3_abs_u_bias_mean": 0.00153,
    "G4_slope_loglog":    0.6695,
    "G5_instability":     1.2363,
    # 段间 4
    "S1_worst_vel_rmse_K": 0.01569,
    "S2_ratio_w_over_b":   2.78,
    "S3_high_speed_mean":  0.01144,
    "S4_degraded_pct":     20.0,   # %
    # 不准退化 3
    "N1_vel_rmse_step_1":  0.00240,
    "N2_traj_xy_rmse_K":   0.02108,
    "N3_spec_radius":      1.005,
}

# PROMPT_deep_koopman_v3_planA §5.2 - 2 条新增 v 通道保护阈值。
# 派生自 v2 baseline：V1 = v2 的 v_rmse@K × 1.2 = 0.00342；
# V2 = max(|v2 v_bias|, plan-A floor 0.00040) ≈ 0.00040。
V3A_EXTRA_THRESHOLDS = {
    "V1_v_rmse_step_K":    0.00342,
    "V2_abs_v_bias_mean":  0.00040,
}

V2_BASELINE = {
    "vel_rmse_step_K": 0.016059,
    "u_rmse_step_K": 0.01580,
    "u_bias_mean": -0.00255,
    "slope_loglog": 0.6695,
    "instability": 1.2363,
    "worst_vel_rmse_K": 0.01961,
    "ratio_worst_over_best": 3.47,
    "high_speed_mean": 0.01525,
    "vel_rmse_step_1": 0.001847,
    "traj_xy_rmse_K": 0.02008,
}


def _spectral_radius_from_ckpt(ckpt_path: str) -> float:
    """从 ckpt 还原模型并算 spec_radius (I+A)。"""
    try:
        device = torch.device("cpu")
        m, _ = load_model_from_ckpt(ckpt_path, device)
        return float(m.spectral_radius().detach().item())
    except Exception:
        return float("nan")


def _compute_degraded_pct(new_res: EvalResult, base_res: EvalResult) -> float:
    """新 ckpt 相对 base 的逐样本退化占比（vel_err@K 大于 base 的样本占比）。

    PROMPT 中 v2 vs v1 的 26.1% 即用类似方式计算。
    """
    a = new_res.per_sample["vel_err_stepK"]
    b = base_res.per_sample["vel_err_stepK"]
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    return float(np.mean(a[:n] > b[:n]) * 100.0)


def _u_bias_drift_rate(per_step: Dict[str, np.ndarray]) -> float:
    """u_bias_per_step 的 |斜率| 用于「漂移率」对比。"""
    ub = per_step["u_bias"].astype(np.float64)
    K = ub.shape[0]
    if K < 2:
        return float("nan")
    steps = np.arange(1, K + 1, dtype=np.float64)
    A = np.vstack([steps, np.ones_like(steps)]).T
    slope, _ = np.linalg.lstsq(A, ub, rcond=None)[0]
    return float(abs(slope))


def v3_threshold_verdict(
    v3_res: EvalResult,
    v1_res: Optional[EvalResult] = None,
    v2_res: Optional[EvalResult] = None,
) -> Tuple[str, bool]:
    """对 v3 模型按 §7 的 12 条阈值逐条 ✅/❌ 判定，返回 (md_text, all_pass)。"""
    K = v3_res.summary["pred_len"]
    agg = v3_res.summary["aggregate"]
    div = v3_res.summary["divergence"]
    bias = v3_res.summary["channel_bias"]
    ps = v3_res.summary.get("per_segment", {}) or {}

    # 取 spec_radius
    spec = _spectral_radius_from_ckpt(v3_res.ckpt_path)

    # S4：v3 vs v1 退化样本占比
    if v1_res is not None and v3_res.per_sample["vel_err_stepK"].shape == v1_res.per_sample["vel_err_stepK"].shape:
        degraded_pct = _compute_degraded_pct(v3_res, v1_res)
    else:
        degraded_pct = float("nan")

    th = V3_THRESHOLDS

    checks = [
        ("G1", "vel_rmse_step_K",
            agg.get(f"vel_rmse_step_{K}", float("nan")), th["G1_vel_rmse_step_K"], "≤"),
        ("G2", "u_rmse_step_K",
            agg.get(f"u_rmse_step_{K}", float("nan")), th["G2_u_rmse_step_K"], "≤"),
        ("G3", "|u_bias_mean|",
            abs(bias.get("u_bias_mean", float("nan"))), th["G3_abs_u_bias_mean"], "≤"),
        ("G4", "slope_loglog",
            div.get("slope_loglog", float("nan")), th["G4_slope_loglog"], "≤"),
        ("G5", "instability_score",
            div.get("instability_score", float("nan")), th["G5_instability"], "≤"),
        ("S1", "per_segment.worst_vel_rmse_K",
            ps.get("worst_vel_rmse_K", float("nan")), th["S1_worst_vel_rmse_K"], "≤"),
        ("S2", "per_segment.ratio_worst_over_best",
            ps.get("ratio_worst_over_best", float("nan")), th["S2_ratio_w_over_b"], "≤"),
        ("S3", "per_segment.high_speed_seg_mean",
            ps.get("high_speed_seg_mean", float("nan")), th["S3_high_speed_mean"], "≤"),
        ("S4", "degraded_pct (vs v1)",
            degraded_pct, th["S4_degraded_pct"], "≤"),
        ("N1", "vel_rmse_step_1",
            agg.get("vel_rmse_step_1", float("nan")), th["N1_vel_rmse_step_1"], "≤"),
        ("N2", "traj_xy_rmse_step_K",
            agg.get(f"traj_xy_rmse_step_{K}", float("nan")), th["N2_traj_xy_rmse_K"], "≤"),
        ("N3", "spectral_radius (I+A)",
            spec, th["N3_spec_radius"], "≤"),
    ]
    all_pass = True
    lines: List[str] = []
    lines.append("### §7 12 条阈值逐条判定 (test set, K=20)")
    lines.append("")
    lines.append("| # | 指标 | v3 实测 | 阈值 | 判定 |")
    lines.append("|---|---|---|---|---|")
    for code, name, value, threshold, op in checks:
        if isinstance(value, float) and np.isnan(value):
            ok = False
        elif op == "≤":
            ok = value <= threshold
        else:
            ok = value >= threshold
        if not ok:
            all_pass = False
        mark = "✅" if ok else "❌"
        lines.append(f"| {code} | {name} | {_fmt(value)} | {op} {_fmt(threshold)} | {mark} |")
    lines.append("")
    lines.append(f"**{'PASS' if all_pass else 'FAIL'}** — §7 v3 验收。")
    lines.append("")
    return "\n".join(lines), all_pass


def v3a_threshold_verdict(
    v3a_res: EvalResult,
    v1_res: Optional[EvalResult] = None,
    v2_res: Optional[EvalResult] = None,
    v3_res: Optional[EvalResult] = None,
) -> Tuple[str, bool]:
    """PROMPT_v3a §5 - 14 条阈值逐条 ✅/❌ 判定（原 12 + V1/V2 v 通道保护 2 条）。

    返回 (md_text, all_pass)。展示列：v2 / v3 / v3a / 阈值 / 判定，
    其中 v2/v3 列允许为空（仅辅助参考）。
    """
    K = v3a_res.summary["pred_len"]
    agg = v3a_res.summary["aggregate"]
    div = v3a_res.summary["divergence"]
    bias = v3a_res.summary["channel_bias"]
    ps = v3a_res.summary.get("per_segment", {}) or {}

    spec = _spectral_radius_from_ckpt(v3a_res.ckpt_path)

    if v1_res is not None and v3a_res.per_sample["vel_err_stepK"].shape == v1_res.per_sample["vel_err_stepK"].shape:
        degraded_pct = _compute_degraded_pct(v3a_res, v1_res)
    else:
        degraded_pct = float("nan")

    th = V3_THRESHOLDS
    thx = V3A_EXTRA_THRESHOLDS

    def _agg_get(res: Optional[EvalResult], key: str) -> float:
        if res is None:
            return float("nan")
        K_ = res.summary["pred_len"]
        if key == "vel_rmse_step_K":
            return res.summary["aggregate"].get(f"vel_rmse_step_{K_}", float("nan"))
        if key == "u_rmse_step_K":
            return res.summary["aggregate"].get(f"u_rmse_step_{K_}", float("nan"))
        if key == "v_rmse_step_K":
            return res.summary["aggregate"].get(f"v_rmse_step_{K_}", float("nan"))
        if key == "traj_xy_rmse_step_K":
            return res.summary["aggregate"].get(f"traj_xy_rmse_step_{K_}", float("nan"))
        if key == "abs_u_bias_mean":
            return abs(res.summary["channel_bias"].get("u_bias_mean", float("nan")))
        if key == "abs_v_bias_mean":
            return abs(res.summary["channel_bias"].get("v_bias_mean", float("nan")))
        if key == "slope_loglog":
            return res.summary["divergence"].get("slope_loglog", float("nan"))
        if key == "instability_score":
            return res.summary["divergence"].get("instability_score", float("nan"))
        if key == "vel_rmse_step_1":
            return res.summary["aggregate"].get("vel_rmse_step_1", float("nan"))
        ps_ = res.summary.get("per_segment", {}) or {}
        return ps_.get(key, float("nan"))

    rows = [
        # (code, name_md, key_or_value(v3a), threshold)
        ("G1", "vel_rmse_step_K",             agg.get(f"vel_rmse_step_{K}", float("nan")),  th["G1_vel_rmse_step_K"], "vel_rmse_step_K"),
        ("G2", "u_rmse_step_K",               agg.get(f"u_rmse_step_{K}",   float("nan")),  th["G2_u_rmse_step_K"],   "u_rmse_step_K"),
        ("G3", "|u_bias_mean|",               abs(bias.get("u_bias_mean", float("nan"))),    th["G3_abs_u_bias_mean"], "abs_u_bias_mean"),
        ("G4", "slope_loglog",                div.get("slope_loglog",     float("nan")),    th["G4_slope_loglog"],     "slope_loglog"),
        ("G5", "instability_score",           div.get("instability_score",float("nan")),    th["G5_instability"],      "instability_score"),
        ("V1", "v_rmse_step_K (plan-A)",      agg.get(f"v_rmse_step_{K}",  float("nan")),   thx["V1_v_rmse_step_K"],   "v_rmse_step_K"),
        ("V2", "|v_bias_mean| (plan-A)",      abs(bias.get("v_bias_mean", float("nan"))),    thx["V2_abs_v_bias_mean"], "abs_v_bias_mean"),
        ("S1", "per_segment.worst_vel_rmse_K", ps.get("worst_vel_rmse_K", float("nan")),    th["S1_worst_vel_rmse_K"], "worst_vel_rmse_K"),
        ("S2", "per_segment.ratio_worst_over_best", ps.get("ratio_worst_over_best", float("nan")), th["S2_ratio_w_over_b"], "ratio_worst_over_best"),
        ("S3", "per_segment.high_speed_seg_mean",   ps.get("high_speed_seg_mean", float("nan")),   th["S3_high_speed_mean"], "high_speed_seg_mean"),
        ("S4", "degraded_pct (vs v1)",        degraded_pct,                                  th["S4_degraded_pct"],     None),
        ("N1", "vel_rmse_step_1",             agg.get("vel_rmse_step_1", float("nan")),      th["N1_vel_rmse_step_1"],  "vel_rmse_step_1"),
        ("N2", "traj_xy_rmse_step_K",         agg.get(f"traj_xy_rmse_step_{K}", float("nan")), th["N2_traj_xy_rmse_K"], "traj_xy_rmse_step_K"),
        ("N3", "spectral_radius (I+A)",       spec,                                          th["N3_spec_radius"],      None),
    ]
    all_pass = True
    lines: List[str] = []
    lines.append("### §5 PROMPT_v3a 14 条阈值逐条判定 (test set, K=" + str(K) + ")")
    lines.append("")
    lines.append("| # | 指标 | v2 | v3 | v3a 实测 | 阈值 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    for code, name, value, threshold, lookup_key in rows:
        v2_val = _agg_get(v2_res, lookup_key) if lookup_key else float("nan")
        v3_val = _agg_get(v3_res, lookup_key) if lookup_key else float("nan")
        if isinstance(value, float) and np.isnan(value):
            ok = False
        else:
            ok = value <= threshold
        if not ok:
            all_pass = False
        mark = "✅" if ok else "❌"
        lines.append(
            f"| {code} | {name} | {_fmt(v2_val)} | {_fmt(v3_val)} | "
            f"{_fmt(value)} | ≤ {_fmt(threshold)} | {mark} |"
        )
    lines.append("")
    lines.append(f"**{'PASS' if all_pass else 'FAIL'}** — §5 PROMPT_v3a 14 条阈值。")
    lines.append("")
    return "\n".join(lines), all_pass


def cubic_dictionary_attribution(
    v3_res: EvalResult,
    v2_res: Optional[EvalResult],
) -> str:
    """3 阶字典效果归因（PROMPT v3 §6.5/§11）。"""
    if v2_res is None:
        return ""
    K = v3_res.summary["pred_len"]
    ps_v3 = v3_res.summary.get("per_segment", {}) or {}
    ps_v2 = v2_res.summary.get("per_segment", {}) or {}
    drift_v3 = _u_bias_drift_rate(v3_res.per_step)
    drift_v2 = _u_bias_drift_rate(v2_res.per_step)
    ratio_drift = drift_v3 / max(drift_v2, 1e-9)
    worst_v3 = ps_v3.get("worst_vel_rmse_K", float("nan"))
    worst_v2 = ps_v2.get("worst_vel_rmse_K", float("nan"))
    ratio_worst = worst_v3 / max(worst_v2, 1e-9) if worst_v2 > 0 else float("nan")
    hi_v3 = ps_v3.get("high_speed_seg_mean", float("nan"))
    hi_v2 = ps_v2.get("high_speed_seg_mean", float("nan"))
    lo_v3 = ps_v3.get("low_speed_seg_mean", float("nan"))
    lo_v2 = ps_v2.get("low_speed_seg_mean", float("nan"))
    ratio_hi_lo = (hi_v3 / max(hi_v2, 1e-9)) / max(lo_v3 / max(lo_v2, 1e-9), 1e-9) if (
        hi_v2 > 0 and lo_v2 > 0
    ) else float("nan")

    lines = []
    lines.append("### 3 阶字典效果归因（v3 vs v2，全部数字来自 summary.json）")
    lines.append("")
    lines.append("| 问题 | v2 | v3 | v3/v2 | 判定 |")
    lines.append("|---|---|---|---|---|")
    a_ok = (ratio_drift < 0.6) if not np.isnan(ratio_drift) else False
    lines.append(
        f"| u_bias 漂移斜率 |slope| | {_fmt(drift_v2)} | {_fmt(drift_v3)} | "
        f"{_fmt(ratio_drift)} | {'✅' if a_ok else '❌'} (<0.6) |"
    )
    b_ok = (ratio_worst < 0.80) if not np.isnan(ratio_worst) else False
    lines.append(
        f"| worst-seg vel_rmse@K | {_fmt(worst_v2)} | {_fmt(worst_v3)} | "
        f"{_fmt(ratio_worst)} | {'✅' if b_ok else '❌'} (<0.80) |"
    )
    c_ok = (ratio_hi_lo < 0.95) if not np.isnan(ratio_hi_lo) else False
    lines.append(
        f"| (high/v2)/(low/v2) 高速相对低速改善 | {_fmt(hi_v2)} / {_fmt(lo_v2)} | "
        f"{_fmt(hi_v3)} / {_fmt(lo_v3)} | "
        f"{_fmt(ratio_hi_lo)} | {'✅' if c_ok else '❌'} (<0.95) |"
    )
    lines.append("")
    return "\n".join(lines)


def plot_compare_trajectory_grid(results: List[EvalResult], path: str) -> None:
    """以第一个 ckpt 的 stepK 误差均匀分位选 6 个固定样本，画多 ckpt 同样本对比。"""
    base = results[0]
    pick = _pick_quantile_indices(base.per_sample["vel_err_stepK"], 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    colors = ["g", "r", "b", "m", "c", "orange"]
    for i, sample_idx in enumerate(pick):
        ax = axes[i // 3, i % 3]
        ax.plot(base.gt_xy[sample_idx, :, 0], base.gt_xy[sample_idx, :, 1], "k-o", label="GT", ms=3, alpha=0.7)
        for j, r in enumerate(results):
            ax.plot(
                r.pred_xy[sample_idx, :, 0], r.pred_xy[sample_idx, :, 1],
                linestyle="--", marker="x", color=colors[(j + 1) % len(colors)],
                label=r.tag, ms=3, alpha=0.85,
            )
        ax.set_title(f"sample #{sample_idx}", fontsize=9)
        ax.axis("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Compare trajectories (6 samples by base-ckpt stepK err quantile)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def quantitative_verdict_block(summary: Dict) -> str:
    K = summary["pred_len"]
    agg = summary["aggregate"]
    div = summary["divergence"]
    bias = summary["channel_bias"]
    fit = summary["fit_quality"]
    return (
        "=== QUANTITATIVE VERDICT (test set) ===\n"
        f"vel_rmse@1={agg['vel_rmse_step_1']:.6g}  "
        f"vel_rmse@5={agg.get('vel_rmse_step_5', float('nan')):.6g}  "
        f"vel_rmse@10={agg.get('vel_rmse_step_10', float('nan')):.6g}  "
        f"vel_rmse@{K}={agg.get(f'vel_rmse_step_{K}', float('nan')):.6g}\n"
        f"divergence: ratio{K}/1={div.get(f'ratio_step{K}_over_step1', float('nan')):.6g}  "
        f"slope_loglog={div['slope_loglog']:.6g}  "
        f"lyapunov_like={div['lyapunov_like']:.6g}\n"
        f"instability_score={div['instability_score']:.6g}  "
        f"monotonic_increasing={div['monotonic_increasing']}  "
        f"divergent_sample_pct={div['divergent_sample_pct']:.6g}%\n"
        f"bias: u={bias['u_bias_mean']:+.6g}  v={bias['v_bias_mean']:+.6g}  r={bias['r_bias_mean']:+.6g}\n"
        f"fit_quality: R^2(u,v,r) = "
        f"{fit['u_r2_mean']:.4g} / {fit['v_r2_mean']:.4g} / {fit['r_r2_mean']:.4g}\n"
        "========================================\n"
    )


def parse_compare(values: List[str]) -> List[Tuple[str, str]]:
    out = []
    for v in values:
        if ":" not in v:
            raise argparse.ArgumentTypeError(f"--compare 元素必须形如 path:tag, 收到 {v}")
        path, tag = v.rsplit(":", 1)
        out.append((path, tag))
    return out


def main() -> int:
    from koopman import paths as P

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--ckpt", type=str, default=None, help="单 ckpt 评估")
    parser.add_argument("--data", type=str, default=str(P.TEST))
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tag", type=str, default="run")
    parser.add_argument("--out_dir", type=str, default=str(P.EVAL_OUT_DIR / "run"))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制评估样本数（默认全部，用于 CI/快速跑）")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                        help="多 ckpt 对比，元素形如 path:tag")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="PROMPT_v3a §4.4 - 若给出，额外计算 degraded_pct_vs_baseline 并"
                             " 写入 summary['s4']。--compare 模式下默认 v1 列做 baseline。")
    parser.add_argument("--smoketest", action="store_true",
                        help="冒烟模式：仅取 1 段 / pred_len=4 / 跑通全流程并退出 0")
    args = parser.parse_args()

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )

    if args.smoketest:
        # 仅做最小可行测试：1 段、pred_len=4、max_samples=128
        pred_len = 4
        out_dir = str(P.EVAL_OUT_DIR / "smoketest")
        os.makedirs(out_dir, exist_ok=True)
        ckpt = args.ckpt or str(P.CKPT_V1_BEST)
        if not os.path.exists(ckpt):
            print(f"[smoketest] 找不到 ckpt {ckpt}", file=sys.stderr)
            return 2
        # 用一个临时小 npz：取 koopman_test.npz 的第一个段
        tmp_path = os.path.join(out_dir, "_smoketest_data.npz")
        d = np.load(args.data, allow_pickle=True)["datas"]
        np.savez(tmp_path, datas=np.array([d[0]], dtype=object))
        res = evaluate_one(
            ckpt_path=ckpt, data_path=tmp_path, pred_len=pred_len,
            dt=args.dt, tag="smoke", out_dir=out_dir, device=device,
            batch_size=args.batch_size, max_samples=128, write_files=True,
        )
        print(quantitative_verdict_block(res.summary))
        # 简单 self-check：必须存在所有产物文件
        for fn in [
            "smoke_per_step_metrics.csv", "smoke_per_step_metrics.md",
            f"smoke_per_sample_step{pred_len}.csv", "smoke_summary.json",
            "smoke_velocity_curve_grid.png", "smoke_error_vs_step.png",
            "smoke_error_band_per_channel.png", f"smoke_step{pred_len}_error_hist.png",
            "smoke_bias_vs_step.png", "smoke_trajectory_grid.png",
        ]:
            p = os.path.join(out_dir, fn)
            assert os.path.exists(p), f"missing artifact {p}"
        print("[smoketest] OK — all artifacts present.")
        return 0

    if args.compare:
        pairs = parse_compare(args.compare)
        os.makedirs(args.out_dir, exist_ok=True)
        results: List[EvalResult] = []
        # 找 v1 baseline 路径（若 compare 列表中含 v1）做 s4 算用
        v1_path = None
        for path, tag in pairs:
            if tag == "v1":
                v1_path = path
                break
        for path, tag in pairs:
            sub_out = os.path.join(args.out_dir, tag)
            # 给 v3 / v3a 附带 baseline_ckpt（v1）以填 summary["s4"]
            this_baseline = args.baseline_ckpt
            if this_baseline is None and tag in ("v3", "v3a") and v1_path is not None and v1_path != path:
                this_baseline = v1_path
            r = evaluate_one(
                ckpt_path=path, data_path=args.data, pred_len=args.pred_len,
                dt=args.dt, tag=tag, out_dir=sub_out, device=device,
                batch_size=args.batch_size, max_samples=args.max_samples,
                write_files=True, baseline_ckpt=this_baseline,
            )
            results.append(r)
            print(f"[{tag}] {path}")
            print(quantitative_verdict_block(r.summary))
        write_compare_csv_md(results, args.out_dir)
        plot_compare_error_vs_step(results, os.path.join(args.out_dir, "compare_error_vs_step.png"))
        plot_compare_step_K_box(results, os.path.join(args.out_dir, f"compare_step{args.pred_len}_box.png"))
        plot_compare_trajectory_grid(results, os.path.join(args.out_dir, "compare_trajectory_grid.png"))
        plot_compare_per_segment_bar(results, os.path.join(args.out_dir, "compare_per_segment_bar.png"))
        plot_compare_u_bias_per_step(results, os.path.join(args.out_dir, "compare_u_bias_per_step.png"))
        with open(os.path.join(args.out_dir, "compare_summary.md"), "r", encoding="utf-8") as f:
            print("\n" + f.read())
        return 0

    if not args.ckpt:
        parser.error("必须指定 --ckpt 或 --compare 或 --smoketest")
    res = evaluate_one(
        ckpt_path=args.ckpt, data_path=args.data, pred_len=args.pred_len,
        dt=args.dt, tag=args.tag, out_dir=args.out_dir, device=device,
        batch_size=args.batch_size, max_samples=args.max_samples,
        write_files=True, baseline_ckpt=args.baseline_ckpt,
    )
    print(quantitative_verdict_block(res.summary))
    return 0


# CLI 入口见 scripts/eval.py
