#!/usr/bin/env python3
"""多步预测漂移的机制诊断工具。

配套文档：``docs/多步漂移可解性分析.md``（本脚本是该文档全部数字的复现入口）。

工程内既有的评估以「模型步数 K」为横轴，而各代模型 dt 不同（v3a 0.1s、
v4 1.0s / 4.0s），K=20 分别是 2s / 20s / 80s，跨代读数不可比；同时缺少
persistence 这类平凡基线，无法分辨"误差曲线平坦"是模型好还是模型退化。
本工具统一以**物理秒数**为横轴，并强制给出平凡基线。

五个子命令：

    spectral   线性算子 Ā 的谱半径与有限步传播增益 ||Ā^k||（二者可差一个数量级）
    decompose  把误差增长拆成 相干累积 / 随机游走 / 放大 三种机制
    horizon    同物理时长的横向对比（含 persistence 与 mean 基线）
    residual   单步残差的结构诊断：剩余误差里还有多少是可建模的
    offset     在环扰动补偿（offset-free）能挽回多少漂移
    all        依次执行以上全部

用法::

    python3 scripts/analysis/drift_analysis.py all
    python3 scripts/analysis/drift_analysis.py horizon --horizon_s 40
    python3 scripts/analysis/drift_analysis.py residual --npz data/koopman_train_merged.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from koopman import evalkit as ek
from koopman import paths as P

DATA_DT = 0.1  # 原始数据采样步长 [s]


# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    """待诊断模型。kind: koopman / mmg / mmg_residual。"""

    label: str
    kind: str
    dt: float
    ckpt: Optional[Path] = None

    @property
    def model_stride(self) -> int:
        return int(round(self.dt / DATA_DT))


def _first_existing(*candidates: Path) -> Optional[Path]:
    """文件直接返回；目录则在其中递归找 v4 best 权重（各 run 存在子目录里）。"""
    for c in candidates:
        if c.is_file():
            return c
        if c.is_dir():
            # run 目录按时间戳命名，取最新一次（续训后的部署权重）
            hit = sorted(c.glob("**/koopman_v4_best.pth"))
            if hit:
                return hit[-1]
    return None


def default_registry() -> List[ModelSpec]:
    """按仓库现有 checkpoint 自动组装；缺失的条目会被跳过。"""
    ck = P.CKPT_DIR
    specs = [
        ModelSpec("v3a  (dt=0.1s)", "koopman", 0.1, _first_existing(P.CKPT_V3A_BEST)),
        ModelSpec("v4   (dt=1.0s, optimized)", "koopman", 1.0,
                  _first_existing(ck / "opt_warm_mlp")),
        ModelSpec("v4   (dt=1.0s, cap64)", "koopman", 1.0,
                  _first_existing(ck / "opt_cap64")),
        ModelSpec("v4   (dt=4.0s, 部署)", "koopman", 4.0,
                  _first_existing(ck / "v4_dt4s")),
        ModelSpec("MMG  物理基线", "mmg", 1.0, _first_existing(ck / "mmg_baseline.npz")),
        ModelSpec("MMG+残差MLP", "mmg_residual", 1.0,
                  _first_existing(ck / "mmg_residual_best.pth")),
    ]
    return [s for s in specs if s.ckpt is not None]


def load_spec(spec: ModelSpec, device: torch.device):
    """返回 (可 rollout 的模块, stats)。物理模型经 PhysStepAdapter 走同一管线。"""
    if spec.kind == "koopman":
        return ek.load_model_from_ckpt(str(spec.ckpt), device)

    from koopman.mmg_model import MmgModel, PhysStepAdapter, load_mmg_npz

    theta, stats, _report = load_mmg_npz(str(P.CKPT_DIR / "mmg_baseline.npz"))
    base = MmgModel(theta)
    if spec.kind == "mmg":
        step = base
    else:
        from koopman.mmg_residual import MmgResidualModel

        step = MmgResidualModel(base, stats=stats)
        sd = torch.load(str(spec.ckpt), map_location="cpu", weights_only=False)
        step.load_state_dict(sd.get("model_state_dict", sd), strict=False)
        step.eval()
    dt = spec.dt
    adapter = PhysStepAdapter(lambda d, c: step.step_phys(d, c, dt), stats).to(device)
    return adapter, stats


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def loglog_slope(curve: np.ndarray) -> float:
    """误差-步数曲线在 log-log 上的最小二乘斜率（与 evalkit 同口径）。"""
    k = np.arange(1, len(curve) + 1, dtype=np.float64)
    m = curve > 0
    if m.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[m]), np.log(curve[m]), 1)[0])


def vel_rms(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """逐步水平速度误差 RMS：sqrt(mean(du²+dv²))，与 evalkit 的 vel_rmse 一致。"""
    e = pred - gt
    return np.sqrt(np.mean(e[:, :, 0] ** 2 + e[:, :, 1] ** 2, axis=0))


def sample_starts(npz: str, span_rows: int, stride: int, max_samples: int,
                  lookback_rows: int = 0) -> Tuple[np.ndarray, ...]:
    """在所有段内枚举可用起点。span_rows 为窗口需要的原始行数。"""
    states, ctrls, seg_starts, seg_lens = ek._load_segments_cached(npz)
    picks, seg_of = [], []
    for si, (start, T) in enumerate(zip(seg_starts.tolist(), seg_lens.tolist())):
        lo, hi = lookback_rows, T - span_rows - 1
        if hi <= lo:
            continue
        cur = np.arange(start + lo, start + hi, stride, dtype=np.int64)
        picks.append(cur)
        seg_of.append(np.full(len(cur), si, dtype=np.int64))
    if not picks:
        raise ValueError(f"{npz} 中没有长度足够的段（需要 {span_rows + lookback_rows} 行）")
    t0 = np.concatenate(picks)
    seg_of = np.concatenate(seg_of)
    if len(t0) > max_samples:
        sel = np.linspace(0, len(t0) - 1, max_samples).astype(np.int64)
        t0, seg_of = t0[sel], seg_of[sel]
    return states, ctrls, t0, seg_of


def _norm_tensors(stats: Dict[str, np.ndarray]):
    sm = stats["state_mean"].astype(np.float32)
    ss = stats["state_std"].astype(np.float32)
    return (torch.tensor(sm[3:6]), torch.tensor(ss[3:6]),
            torch.tensor(stats["ctrl_mean"].astype(np.float32)),
            torch.tensor(stats["ctrl_std"].astype(np.float32)))


@torch.no_grad()
def rollout(model, stats, states, ctrls, t0, K, ms,
            disturbance: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
    """无 teacher-forcing 的 K 步开环 rollout，返回 (pred, gt) 物理量 (M,K,3)。

    ``disturbance`` 非空时做在环扣除（offset-free：扰动进状态方程，随窗传播）。
    """
    dm, ds, cm, cs = _norm_tensors(stats)
    offs = np.arange(K, dtype=np.int64) * ms
    gidx = t0[:, None] + offs[None, :]
    u_n = (torch.from_numpy(ctrls[gidx]) - cm) / cs
    gt = np.ascontiguousarray(states[gidx + ms][:, :, 3:6])

    z = model.encode((torch.from_numpy(states[t0][:, 3:6]) - dm) / ds)
    steps = []
    for k in range(K):
        z = model.latent_step(z, u_n[:, k, :])
        x = model.reconstruct_state(z) * ds + dm
        if disturbance is not None:
            x = x - disturbance
            z = model.encode((x - dm) / ds)  # 带扰动的状态重新提升
        steps.append(x)
    return torch.stack(steps, dim=1).numpy(), gt


@torch.no_grad()
def one_step_error(model, stats, states, ctrls, t0, ms, back: int = 1) -> torch.Tensor:
    """回看第 ``back`` 拍、在真值状态处的单步预测误差（部署侧可直接观测到）。"""
    dm, ds, cm, cs = _norm_tensors(stats)
    xp = torch.from_numpy(states[t0 - back * ms][:, 3:6])
    up = (torch.from_numpy(ctrls[t0 - back * ms]) - cm) / cs
    z = model.latent_step(model.encode((xp - dm) / ds), up)
    pred = model.reconstruct_state(z) * ds + dm
    return pred - torch.from_numpy(states[t0 - (back - 1) * ms][:, 3:6])


# ---------------------------------------------------------------------------
# 子命令 1：谱与传播增益
# ---------------------------------------------------------------------------


def cmd_spectral(args) -> None:
    """谱半径 ρ(Ā) 只刻画渐近行为；有限时域误差放大由 ||Ā^k|| 决定。

    Ā 非正规时（特征向量矩阵条件数大）二者可以差一个数量级——这正是
    "把 ρ 压到 <1 却对长程误差曲线毫无作用"的原因。
    """
    dev = torch.device("cpu")
    print(f"{'模型':<26}{'nz':>5}{'rho(Ā)':>10}{'||Ā||₂':>10}{'||Ā^20||₂':>11}"
          f"{'||Ā^K||₂':>11}{'K':>5}{'cond(V)':>11}")
    print("-" * 90)
    detail = []
    for spec in default_registry():
        if spec.kind != "koopman":
            continue
        model, _ = load_spec(spec, dev)
        nz = model.A.weight.shape[0]
        A_bar = model.A.weight.detach().cpu().numpy().astype(np.float64) + np.eye(nz)
        K = int(round(args.horizon_s / spec.dt))
        norms, Ak = [], np.eye(nz)
        for _ in range(max(K, 20)):
            Ak = Ak @ A_bar
            norms.append(float(np.linalg.norm(Ak, 2)))
        _w, V = np.linalg.eig(A_bar)
        print(f"{spec.label:<26}{nz:>5}{float(np.max(np.abs(np.linalg.eigvals(A_bar)))):>10.4f}"
              f"{float(np.linalg.norm(A_bar, 2)):>10.4f}{norms[19]:>11.4f}"
              f"{norms[K-1]:>11.4f}{K:>5}{float(np.linalg.cond(V)):>11.3e}")
        detail.append((spec.label, K, norms))

    print(f"\n（K 为 {args.horizon_s:.0f}s 物理时长对应的模型步数；"
          f"||Ā^20|| 列固定 20 步，便于按步数横向比较）")
    print("逐步传播增益 ||Ā^k||₂：")
    for label, K, norms in detail:
        pts = sorted({p for p in (1, 2, 5, 10, 20, K) if p <= len(norms)})
        print(f"  {label:<26}" + "  ".join(f"k={p}:{norms[p-1]:.3f}" for p in pts))


# ---------------------------------------------------------------------------
# 子命令 2：误差增长机制分解
# ---------------------------------------------------------------------------


def cmd_decompose(args) -> None:
    """把 K 步误差与「单步误差的两种极端叠加方式」对比，判定主导机制。

      相干累积（误差同向叠加）  : |e_k| ≈ Σ|d_j| ≈ k·|d|    -> 斜率 ≈ 1
      随机游走（误差互不相关）  : rms(e_k) ≈ sqrt(k)·rms(d)  -> 斜率 ≈ 0.5
    """
    dev = torch.device("cpu")
    for spec in default_registry():
        K = int(round(args.horizon_s / spec.dt))
        ms = spec.model_stride
        try:
            states, ctrls, t0, _ = sample_starts(
                str(args.npz), K * ms, args.stride, args.max_samples)
            model, stats = load_spec(spec, dev)
            pred, gt = rollout(model, stats, states, ctrls, t0, K, ms)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {spec.label}: {type(exc).__name__}: {exc}")
            continue

        # teacher-forced 单步误差 d_k：每步都从真值状态出发
        dm, ds, cm, cs = _norm_tensors(stats)
        offs = np.arange(K, dtype=np.int64) * ms
        gidx = t0[:, None] + offs[None, :]
        u_n = (torch.from_numpy(ctrls[gidx]) - cm) / cs
        prev = torch.from_numpy(np.ascontiguousarray(states[gidx][:, :, 3:6]))
        M = len(t0)
        with torch.no_grad():
            z = model.encode(((prev - dm) / ds).reshape(M * K, 3))
            z = model.latent_step(z, u_n.reshape(M * K, -1))
            tf = (model.reconstruct_state(z) * ds + dm).reshape(M, K, 3).numpy()

        rms_free = vel_rms(pred, gt)
        rms_d = vel_rms(tf, gt)
        mean_e = (pred - gt).mean(axis=0)
        bias = np.sqrt(mean_e[:, 0] ** 2 + mean_e[:, 1] ** 2)
        coh = np.cumsum((tf - gt).mean(axis=0), axis=0)
        coh_v = np.sqrt(coh[:, 0] ** 2 + coh[:, 1] ** 2)
        rw = np.sqrt(np.cumsum(rms_d ** 2))

        print(f"\n{spec.label}   M={M}  K={K}×{spec.dt}s={K*spec.dt:.0f}s")
        print(f"  {'k':>4}{'时长s':>8}{'开环误差':>11}{'其中bias':>11}"
              f"{'单步误差d':>11}{'相干基准':>11}{'随机游走基准':>13}{'实际/随机游走':>14}")
        for k in sorted({1, 2, 5, 10, K}):
            if not 1 <= k <= K:
                continue
            i = k - 1
            print(f"  {k:>4}{k*spec.dt:>8.1f}{rms_free[i]:>11.5f}{bias[i]:>11.5f}"
                  f"{rms_d[i]:>11.5f}{coh_v[i]:>11.5f}{rw[i]:>13.5f}"
                  f"{rms_free[i]/max(rw[i],1e-12):>14.2f}")
        share = 100 * bias[-1] ** 2 / max(rms_free[-1] ** 2, 1e-18)
        print(f"  斜率 free={loglog_slope(rms_free):.3f}（随机游走=0.500，相干≈1.0）；"
              f"末步 bias² 只占 MSE {share:.1f}%")


# ---------------------------------------------------------------------------
# 子命令 3：同物理时长横向对比
# ---------------------------------------------------------------------------


def cmd_horizon(args) -> None:
    """所有模型在同一批样本起点、同一物理时长上比较，并强制给出平凡基线。

    persistence（假设速度不变）是动力学模型必须显著超越的下限；
    mean（输出训练集均值）是零信息上限——误差接近它说明模型已退化。
    """
    span = int(round(args.horizon_s / DATA_DT))
    states, ctrls, t0, _ = sample_starts(str(args.npz), span, args.stride, args.max_samples)
    dev = torch.device("cpu")
    grid = np.arange(args.grid_s, args.horizon_s + 1e-9, args.grid_s)
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for spec in default_registry():
        K = int(round(args.horizon_s / spec.dt))
        try:
            model, stats = load_spec(spec, dev)
            pred, gt = rollout(model, stats, states, ctrls, t0, K, spec.model_stride)
            curves[spec.label] = (np.arange(1, K + 1) * spec.dt, vel_rms(pred, gt))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {spec.label}: {type(exc).__name__}: {exc}")

    K = span
    gidx = t0[:, None] + (np.arange(K, dtype=np.int64) + 1)[None, :]
    gt_full = np.ascontiguousarray(states[gidx][:, :, 3:6])
    x0 = states[t0][:, 3:6]
    times = np.arange(1, K + 1) * DATA_DT
    curves["persistence 基线"] = (
        times, vel_rms(np.repeat(x0[:, None, :], K, axis=1), gt_full))
    gmean = states[:, 3:6].mean(axis=0)
    curves["mean 基线（零信息）"] = (
        times, vel_rms(np.broadcast_to(gmean[None, None, :], gt_full.shape), gt_full))

    print(f"\n同物理时长横向对比  数据集={Path(args.npz).name}  样本 M={len(t0)}  "
          f"水平速度误差 RMS [m/s]")
    header = f"{'模型':<26}" + "".join(f"{t:>9.0f}s" for t in grid)
    print(header)
    print("-" * len(header))
    order = ["persistence 基线", "mean 基线（零信息）"] + [
        s.label for s in default_registry() if s.label in curves]
    for label in order:
        if label not in curves:
            continue
        t, c = curves[label]
        cells = ""
        for g in grid:
            i = int(np.argmin(np.abs(t - g)))
            cells += f"{c[i]:>10.5f}" if abs(t[i] - g) < 1e-6 else f"{'-':>10}"
        print(f"{label:<26}{cells}")

    ranked = sorted(((c[-1], lb) for lb, (_t, c) in curves.items()
                     if "基线" not in lb), key=lambda r: r[0])
    if len(ranked) >= 2:
        print(f"\n@{args.horizon_s:.0f}s 最优 = {ranked[0][1]} ({ranked[0][0]:.5f})，"
              f"次优 = {ranked[1][1]} ({ranked[1][0]:.5f})，"
              f"相差 {ranked[1][0]/max(ranked[0][0],1e-12):.1f} 倍")


# ---------------------------------------------------------------------------
# 子命令 4：单步残差结构诊断
# ---------------------------------------------------------------------------


def _ridge_r2(X_tr, y_tr, X_te, y_te, lam=1e-6):
    Xtr = np.concatenate([X_tr, np.ones((len(X_tr), 1))], axis=1)
    Xte = np.concatenate([X_te, np.ones((len(X_te), 1))], axis=1)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    mu[-1], sd[-1] = 0.0, 1.0
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = np.linalg.solve(Xtr.T @ Xtr + lam * len(Xtr) * np.eye(Xtr.shape[1]), Xtr.T @ y_tr)
    res = ((y_te - Xte @ W) ** 2).sum(axis=0)
    tot = ((y_te - y_te.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - res / np.maximum(tot, 1e-18), 1.0 - res.sum() / max(tot.sum(), 1e-18)


def cmd_residual(args) -> None:
    """残差若是不可约噪声，对任何特征的 R² 都应接近 0；反之说明模型族缺结构。

    注意必须**按整段留出**：残差在秒级尺度上自相关极高（实测 lag=0.3s 达 0.97），
    随机划分会让留出点的近邻出现在训练集里，把 R² 抬虚。
    """
    dev = torch.device("cpu")
    spec = next((s for s in default_registry() if s.label == args.model),
                None) or next(s for s in default_registry() if s.kind == "koopman")
    ms = spec.model_stride
    model, stats = load_spec(spec, dev)
    states, ctrls, idx, seg_of = sample_starts(
        str(args.npz), ms + 1, args.stride, args.max_samples, lookback_rows=2 * ms)

    dm, ds, cm, cs = _norm_tensors(stats)
    x_t, x_next, u_t = states[idx][:, 3:6], states[idx + ms][:, 3:6], ctrls[idx]
    with torch.no_grad():
        z = model.latent_step(model.encode((torch.from_numpy(x_t) - dm) / ds),
                              (torch.from_numpy(u_t) - cm) / cs)
        pred = (model.reconstruct_state(z) * ds + dm).numpy()
    y = (pred - x_next).astype(np.float64)

    u, v, r = x_t[:, 0], x_t[:, 1], x_t[:, 2]
    cp, dp = u_t[:, 0], np.deg2rad(u_t[:, 1])
    csb, dsb = u_t[:, 2], np.deg2rad(u_t[:, 3])
    lon = cp * np.cos(dp) + csb * np.cos(dsb)
    lat = cp * np.sin(dp) + csb * np.sin(dsb)
    dif = csb * np.cos(dsb) - cp * np.cos(dp)

    F1 = np.stack([u * np.abs(u), v * np.abs(v), r * np.abs(r), v * r, u * r,
                   u * v * r, u * u * r, v * v * r, u * r * r, v * r * r,
                   u * np.abs(v) * v, v * np.abs(u) * u, r * np.abs(u) * u,
                   r * np.abs(v) * v, u * np.abs(u) * u, v * np.abs(v) * v], axis=1)
    F2 = u_t.astype(np.float64)
    F3 = np.stack([lon, lat, dif, np.abs(lon) * lon, np.abs(lat) * lat], axis=1)
    F4 = np.stack([u * lon, u * lat, u * dif, v * lat, v * dif, r * lon, r * lat,
                   r * dif, u * u * lon, u * u * lat, u * u * dif,
                   np.abs(u) * u * dif], axis=1)
    F5 = np.concatenate([ctrls[idx - ms], ctrls[idx - 2 * ms]], axis=1).astype(np.float64)

    segs = np.unique(seg_of)
    np.random.default_rng(0).shuffle(segs)
    held = set(segs[:max(1, int(round(0.3 * len(segs))))].tolist())
    te = np.where(np.isin(seg_of, list(held)))[0]
    tr = np.where(~np.isin(seg_of, list(held)))[0]

    print(f"\n单步残差结构诊断   {spec.label}   数据集={Path(args.npz).name}  "
          f"dt={spec.dt}s  M={len(idx)}（按段留出 {len(tr)}/{len(te)}）")
    print(f"  残差 RMS: u={np.sqrt((y[:,0]**2).mean()):.5f} m/s  "
          f"v={np.sqrt((y[:,1]**2).mean()):.5f} m/s  "
          f"r={np.sqrt((y[:,2]**2).mean()):.6f} rad/s")
    print(f"  {'特征族（累加）':<40}{'R²(u)':>10}{'R²(v)':>10}{'R²(r)':>10}{'R²(总)':>10}")
    for name, mats in [("F1 状态字典16阶〔模型已有〕", [F1]),
                       ("+F2 原始控制4维〔模型已有〕", [F1, F2]),
                       ("+F3 推进器分解 lon/lat/dif", [F1, F2, F3]),
                       ("+F4 双线性 状态×控制", [F1, F2, F3, F4]),
                       ("+F5 滞后控制（执行器延迟）", [F1, F2, F3, F4, F5])]:
        X = np.concatenate(mats, axis=1)
        per, tot = _ridge_r2(X[tr], y[tr], X[te], y[te])
        print(f"  {name:<40}{per[0]:>10.3f}{per[1]:>10.3f}{per[2]:>10.3f}{tot:>10.3f}")

    print("  残差时间自相关（决定误差能否相干累积）：", end="")
    for lag in (1, 2, 5, 10):
        c = np.corrcoef(y[:-lag, 0], y[lag:, 0])[0, 1]
        print(f"  Δt={lag*args.stride*DATA_DT:.1f}s:{c:+.3f}", end="")
    print()


# ---------------------------------------------------------------------------
# 子命令 5：在环扰动补偿
# ---------------------------------------------------------------------------


def cmd_offset(args) -> None:
    """offset-free：把上一拍实测单步误差当常值扰动加进状态方程，随预测窗传播。

    这是部署侧不需要重训练就能上的手段。``oracle`` 用窗内真实单步误差的均值，
    是任何常值扰动估计的性能上界——若 oracle 都无增益，说明该模型的残差里
    根本不存在可利用的持续偏置分量。
    """
    dev = torch.device("cpu")
    print(f"\n在环扰动补偿  预测时长={args.horizon_s:.0f}s  数据集={Path(args.npz).name}")
    print(f"{'模型':<26}{'步数':>6}{'无补偿':>11}{'last':>11}{'oracle':>11}"
          f"{'last增益':>11}{'oracle增益':>12}")
    print("-" * 88)
    for spec in default_registry():
        K = int(round(args.horizon_s / spec.dt))
        ms = spec.model_stride
        try:
            states, ctrls, t0, _ = sample_starts(
                str(args.npz), K * ms, args.stride, args.max_samples, lookback_rows=ms)
            model, stats = load_spec(spec, dev)
            raw_p, gt = rollout(model, stats, states, ctrls, t0, K, ms)
            raw = vel_rms(raw_p, gt)

            d_last = one_step_error(model, stats, states, ctrls, t0, ms, back=1)
            cor = vel_rms(*rollout(model, stats, states, ctrls, t0, K, ms,
                                   disturbance=d_last))

            dm, ds, cm, cs = _norm_tensors(stats)
            gidx = t0[:, None] + (np.arange(K, dtype=np.int64) * ms)[None, :]
            u_n = (torch.from_numpy(ctrls[gidx]) - cm) / cs
            prev = np.concatenate([states[t0][:, None, 3:6], gt[:, :-1, :]], axis=1)
            with torch.no_grad():
                zp = model.encode((torch.from_numpy(prev.reshape(-1, 3)) - dm) / ds)
                zp = model.latent_step(zp, u_n.reshape(-1, u_n.shape[-1]))
                pn = (model.reconstruct_state(zp) * ds + dm).reshape(gt.shape)
            d_or = (pn - torch.from_numpy(gt)).mean(dim=1)
            orc = vel_rms(*rollout(model, stats, states, ctrls, t0, K, ms,
                                   disturbance=d_or))

            print(f"{spec.label:<26}{K:>6}{raw[-1]:>11.5f}{cor[-1]:>11.5f}{orc[-1]:>11.5f}"
                  f"{100*(1-cor[-1]/raw[-1]):>10.1f}%{100*(1-orc[-1]/raw[-1]):>11.1f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {spec.label}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *, stride=10, samples=6000):
        p.add_argument("--npz", default=str(P.TEST), help="评估数据集")
        p.add_argument("--horizon_s", type=float, default=20.0, help="预测物理时长 [s]")
        p.add_argument("--stride", type=int, default=stride, help="样本起点间隔（原始行）")
        p.add_argument("--max_samples", type=int, default=samples)
        return p

    common(sub.add_parser("spectral", help="谱半径与有限步传播增益"))
    common(sub.add_parser("decompose", help="误差增长机制分解"))
    h = common(sub.add_parser("horizon", help="同物理时长横向对比"))
    h.add_argument("--grid_s", type=float, default=4.0, help="读数网格间隔 [s]")
    rp = common(sub.add_parser("residual", help="单步残差结构诊断"), stride=3, samples=20000)
    rp.add_argument("--model", default="v4   (dt=1.0s, cap64)", help="注册表中的模型标签")
    common(sub.add_parser("offset", help="在环扰动补偿"))
    common(sub.add_parser("all", help="依次执行全部诊断")).add_argument(
        "--grid_s", type=float, default=4.0)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    registry = default_registry()
    if not registry:
        print("未找到任何可用 checkpoint，请先训练或下载权重。")
        return 1
    print(f"可用模型 {len(registry)} 个（权重路径决定全部读数，务必核对）：")
    for s in registry:
        print(f"  {s.label:<26} dt={s.dt:<5} {s.ckpt.relative_to(_ROOT)}")

    if args.cmd == "all":
        for fn, title in ((cmd_spectral, "谱与传播增益"),
                          (cmd_decompose, "误差增长机制分解"),
                          (cmd_horizon, "同物理时长横向对比"),
                          (cmd_residual, "单步残差结构诊断"),
                          (cmd_offset, "在环扰动补偿")):
            print(f"\n{'='*100}\n【{title}】\n{'='*100}")
            ns = argparse.Namespace(**vars(args))
            if fn is cmd_residual:
                ns.stride, ns.max_samples = 3, 20000
                ns.model = "v4   (dt=1.0s, cap64)"
            fn(ns)
        return 0

    {"spectral": cmd_spectral, "decompose": cmd_decompose, "horizon": cmd_horizon,
     "residual": cmd_residual, "offset": cmd_offset}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
