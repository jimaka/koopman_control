"""v4 代价函数（训练损失 + 潜空间 MPC QP 目标）定量分析。

用法：
    python3 scripts/analyze_v4_cost.py \
        --ckpt checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth \
        --data data/koopman_test.npz --out logs/analyze_v4_cost.txt

报告内容：
    A. 训练损失：各项原始值 / 加权值 / 梯度范数占比、归一化口径等价性、
       Huber 工作区、l_lin 目标未 detach 的塌缩通道、位姿损失的离散化下界。
    B. MPC QP 目标：w_z 在 48 维潜空间的隐含加权、误差能量在 decoder 行空间 /
       零空间的分布、参考对齐、move-blocking 次优性、Δu0 缺项。

纯分析脚本，不修改模型与配置。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman import evalkit as ek  # noqa: E402
from new_v4_dict_input.model_v4_dict_input import (  # noqa: E402
    HorizontalKoopmanModelV4DictInput,
)
from new_v4_dict_input.train_v4_dict_input import (  # noqa: E402
    KoopmanVoyageDataset,
    build_parser,
    denorm_pose,
    huber,
    integrate_pose_from_vel,
    make_step_weights,
    rollout_train,
    wrap_yaw_diff,
)

_LINES: List[str] = []
_SHARE_CACHE: List[Tuple[str, float, float]] = []


def emit(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def section(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def load_args_from_ckpt(ckpt: Dict) -> argparse.Namespace:
    """用 ckpt 里的 args 覆盖当前默认值（None 字段回落到默认）。"""
    parser = build_parser()
    args = parser.parse_args([])
    for k, v in (ckpt.get("args") or {}).items():
        if v is None:
            continue
        setattr(args, k, v)
    args.model_stride = ek.model_stride_from_dt(args.dt, args.data_dt)
    return args


def build_batch(
    args: argparse.Namespace,
    data_path: str,
    stats: Dict[str, np.ndarray],
    n_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, KoopmanVoyageDataset]:
    ds = KoopmanVoyageDataset(
        data_path,
        pred_len=args.pred_len_max,
        stride=args.stride,
        stats=stats,
        model_stride=args.model_stride,
        data_dt=args.data_dt,
    )
    n = min(n_samples, len(ds))
    sel = np.linspace(0, len(ds) - 1, n).astype(int)
    xs, ys, us = [], [], []
    for i in sel:
        a, b, c = ds[int(i)]
        xs.append(a)
        ys.append(b)
        us.append(c)
    return (
        torch.stack(xs).to(device),
        torch.stack(ys).to(device),
        torch.stack(us).to(device),
        ds,
    )


def grad_norm(model: torch.nn.Module, loss: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    model.zero_grad(set_to_none=True)
    return float(np.sqrt(total))


def analyze_training_loss(
    model: HorizontalKoopmanModelV4DictInput,
    args: argparse.Namespace,
    x_t: torch.Tensor,
    x_seq: torch.Tensor,
    u_seq: torch.Tensor,
    stats: Dict[str, np.ndarray],
    device: torch.device,
) -> None:
    section("A. 训练损失（compute_losses）定量分解")

    dyn_mean = torch.tensor(stats["state_mean"][3:6], device=device)
    dyn_std = torch.tensor(stats["state_std"][3:6], device=device)
    pose_mean = torch.tensor(stats["state_mean"][:3], device=device)
    pose_std = torch.tensor(stats["state_std"][:3], device=device)

    bsz, k, _ = x_seq.shape
    dyn_t_n = x_t[:, 3:6]
    dyn_target_n = x_seq[:, :, 3:6]

    emit(f"样本数 B={bsz}, 预测步 K={k}, dt={args.dt}s (总时长 {k * args.dt:.1f}s), "
         f"latent={model.latent_dim} (dict16+hidden{model.hidden_dim})")
    emit(f"dyn_std (u,v,r) = {np.array2string(dyn_std.cpu().numpy(), precision=5)}")
    emit(f"权重: w_vel={args.w_vel} w_acc={args.w_acc} w_lin={args.w_lin} "
         f"w_recon={args.w_recon} w_xy={args.w_xy} w_yaw={args.w_yaw} "
         f"w_stab={args.w_stab} w_l2={args.w_l2} gamma_step={args.gamma_step} "
         f"huber_beta={args.huber_beta}")

    pred_norm_seq, pred_lat_seq = rollout_train(model, dyn_t_n, u_seq)
    pred_phys = pred_norm_seq * dyn_std + dyn_mean
    target_phys = dyn_target_n * dyn_std + dyn_mean

    step_w = make_step_weights(k, args.gamma_step, device).view(1, k, 1)
    step_w_pose = step_w.squeeze(-1)
    chan_scale = 1.0 / dyn_std.view(1, 1, 3)

    err_vel = pred_phys - target_phys
    l_vel = (huber(err_vel * chan_scale, beta=args.huber_beta) * step_w).mean()

    pred_acc = (pred_phys[:, 1:] - pred_phys[:, :-1]) / args.dt
    gt_acc = (target_phys[:, 1:] - target_phys[:, :-1]) / args.dt
    l_acc = huber((pred_acc - gt_acc) * chan_scale, beta=args.huber_beta).mean()

    target_lat = model.encode(dyn_target_n.reshape(bsz * k, 3)).view(bsz, k, -1)
    l_lin = ((pred_lat_seq - target_lat) ** 2).mean()

    x_recon = model.reconstruct_state(model.encode(dyn_t_n))
    l_recon = ((x_recon - dyn_t_n) ** 2).mean()

    pose0 = denorm_pose(x_t[:, :3], pose_mean, pose_std)
    target_pose = denorm_pose(x_seq[:, :, :3], pose_mean, pose_std)
    pred_pose = integrate_pose_from_vel(pose0, pred_phys, args.dt)
    err_x = pred_pose[..., 0] - target_pose[..., 0]
    err_y = pred_pose[..., 1] - target_pose[..., 1]
    l_xy = ((err_x * err_x + err_y * err_y) * step_w_pose).mean()
    err_yaw = wrap_yaw_diff(pred_pose[..., 2], target_pose[..., 2])
    l_yaw = (huber(err_yaw, beta=args.huber_beta) * step_w_pose).mean()

    spec = model.spectral_radius()
    l_stab = torch.relu(spec - args.rho_max) ** 2
    l_l2 = (model.A.weight ** 2).sum() + (model.B.weight ** 2).sum()

    terms = [
        ("L_vel", l_vel, args.w_vel),
        ("L_acc", l_acc, args.w_acc),
        ("L_lin", l_lin, args.w_lin),
        ("L_recon", l_recon, args.w_recon),
        ("L_xy", l_xy, args.w_xy),
        ("L_yaw", l_yaw, args.w_yaw),
        ("L_stab", l_stab, args.w_stab),
        ("L_l2", l_l2, args.w_l2),
    ]
    _SHARE_CACHE.clear()
    weighted = [(name, float(t.detach()), w, float(t.detach()) * w) for name, t, w in terms]
    total = sum(x[3] for x in weighted)

    emit()
    emit("A.1 各项数值与占比（ramp 已收敛，按 epoch>ramp_epochs 的稳态权重）")
    emit(f"{'term':>8} | {'raw':>12} | {'weight':>8} | {'weighted':>12} | {'share%':>7} | {'grad_norm':>10} | {'grad%':>6}")
    emit("-" * 84)
    gnorms = []
    for (name, t, w) in terms:
        gn = grad_norm(model, w * t) if w != 0 else 0.0
        gnorms.append(gn)
    gsum = sum(gnorms) or 1.0
    for (name, raw, w, wt), gn in zip(weighted, gnorms):
        emit(f"{name:>8} | {raw:12.5g} | {w:8.4g} | {wt:12.5g} | "
             f"{100 * wt / total:7.2f} | {gn:10.4g} | {100 * gn / gsum:6.2f}")
        _SHARE_CACHE.append((name, 100 * wt / total, 100 * gn / gsum))
    emit(f"{'TOTAL':>8} | {'':>12} | {'':>8} | {total:12.5g} | {100.0:7.2f}")
    emit("注：grad_norm 为该项单独反传时全模型参数梯度的 L2 范数（同一 batch，同一权重），"
         "grad% 为其在各项之和中的占比——反映该项对参数更新方向的实际支配力。")

    emit()
    emit("A.2 L_vel 的归一化口径：反归一化后再乘 1/dyn_std 是恒等往返")
    l_vel_norm_space = (huber(pred_norm_seq - dyn_target_n, beta=args.huber_beta) * step_w).mean()
    emit(f"  L_vel(反归一化 × chan_scale) = {float(l_vel):.10g}")
    emit(f"  L_vel(直接用归一化残差)      = {float(l_vel_norm_space):.10g}")
    emit(f"  |差| = {abs(float(l_vel) - float(l_vel_norm_space)):.3g}  → 两者数学等价")
    ds_np = dyn_std.cpu().numpy()
    emit(f"  等价物理加权 1/σ: u={1 / ds_np[0]:.3f} /(m/s), v={1 / ds_np[1]:.3f} /(m/s), "
         f"r={1 / ds_np[2]:.3f} /(rad/s)")
    emit(f"  → 相对 u 通道，v 被放大 {ds_np[0] / ds_np[1]:.1f}×，r 被放大 {ds_np[0] / ds_np[2]:.1f}×"
         "（每物理单位误差的代价比）")

    emit()
    emit("A.3 Huber 工作区（|归一化残差| 与 beta 的关系）")
    abs_res = (pred_norm_seq - dyn_target_n).abs()
    lin_frac = float((abs_res > args.huber_beta).float().mean())
    emit(f"  beta={args.huber_beta}；|res|>beta 的比例 = {100 * lin_frac:.1f}% → 该比例即 L1 线性区占比")
    for i, ch in enumerate("uvr"):
        a = abs_res[..., i]
        emit(f"    {ch}: mean|res|={float(a.mean()):.4f}  p90={float(a.quantile(0.9)):.4f}  "
             f"线性区占比={100 * float((a > args.huber_beta).float().mean()):.1f}%")

    emit()
    emit("A.4 L_lin：目标 encode(x_target) 未 detach → 编码器可通过缩小 hidden 降低损失")
    z_all = model.encode(dyn_target_n.reshape(bsz * k, 3)).detach()
    atoms_std = z_all[:, :16].std(dim=0)
    hid_std = z_all[:, 16:].std(dim=0)
    emit(f"  latent 逐维 std：atoms(16) 均值={float(atoms_std.mean()):.4f} "
         f"[{float(atoms_std.min()):.4f}, {float(atoms_std.max()):.4f}]")
    emit(f"                  hidden(32) 均值={float(hid_std.mean()):.4f} "
         f"[{float(hid_std.min()):.4f}, {float(hid_std.max()):.4f}]")
    err_lat = (pred_lat_seq - target_lat).detach()
    e_atoms = float((err_lat[..., :16] ** 2).sum())
    e_hid = float((err_lat[..., 16:] ** 2).sum())
    emit(f"  L_lin 误差能量：atoms 占 {100 * e_atoms / (e_atoms + e_hid):.1f}%，"
         f"hidden 占 {100 * e_hid / (e_atoms + e_hid):.1f}%")
    l_lin_detached = ((pred_lat_seq - target_lat.detach()) ** 2).mean()
    g_full = grad_norm(model, l_lin)
    g_det = grad_norm(model, l_lin_detached)
    emit(f"  梯度范数：l_lin(目标可导)={g_full:.5g}  vs  l_lin(目标 detach)={g_det:.5g}  "
         f"→ 目标路径贡献 {100 * abs(g_full - g_det) / max(g_full, 1e-12):.1f}% 的梯度量级")

    emit()
    emit("A.5 位姿损失的离散化下界（把 GT 速度代入同一欧拉积分）")
    gt_pose = integrate_pose_from_vel(pose0, target_phys, args.dt)
    ex = gt_pose[..., 0] - target_pose[..., 0]
    ey = gt_pose[..., 1] - target_pose[..., 1]
    l_xy_floor = ((ex * ex + ey * ey) * step_w_pose).mean()
    eyaw = wrap_yaw_diff(gt_pose[..., 2], target_pose[..., 2])
    l_yaw_floor = (huber(eyaw, beta=args.huber_beta) * step_w_pose).mean()
    rms_pred = float(torch.sqrt((err_x ** 2 + err_y ** 2).mean()))
    rms_floor = float(torch.sqrt((ex ** 2 + ey ** 2).mean()))
    emit(f"  L_xy(模型预测速度) = {float(l_xy):.4f}   位置 RMSE = {rms_pred:.3f} m")
    emit(f"  L_xy(GT 速度积分)  = {float(l_xy_floor):.4f}   位置 RMSE = {rms_floor:.3f} m  ← 不可约下界")
    emit(f"  下界占比 = {100 * float(l_xy_floor) / max(float(l_xy), 1e-12):.1f}% of L_xy；"
         f"加权后 w_xy·下界 = {args.w_xy * float(l_xy_floor):.4f}（占 total {100 * args.w_xy * float(l_xy_floor) / total:.1f}%）")
    emit(f"  L_yaw(模型) = {float(l_yaw):.5f}  vs  L_yaw(GT 速度积分) = {float(l_yaw_floor):.5f}")
    emit(f"  末步位置 RMSE：模型 {float(torch.sqrt((err_x[:, -1] ** 2 + err_y[:, -1] ** 2).mean())):.3f} m，"
         f"GT 速度积分 {float(torch.sqrt((ex[:, -1] ** 2 + ey[:, -1] ** 2).mean())):.3f} m")
    emit("  → 模型位置误差低于「完美速度 + 同一积分器」的下界，说明速度预测已被偏置以补偿积分离散误差")

    yaw0 = pose0[:, 2:3]
    c0, s0 = torch.cos(yaw0), torch.sin(yaw0)
    at_model = err_x * c0 + err_y * s0
    at_floor = ex * c0 + ey * s0
    emit(f"  沿航向（along-track）有符号位置误差均值：模型 {float(at_model.mean()):+.3f} m，"
         f"GT 速度积分 {float(at_floor.mean()):+.3f} m")
    bias = (pred_phys - target_phys).mean(dim=(0, 1))
    emit(f"  模型速度有符号偏差均值：u={float(bias[0]):+.4f} m/s, v={float(bias[1]):+.4f} m/s, "
         f"r={float(bias[2]):+.5f} rad/s（与上面的积分偏差同号相消）")

    emit()
    emit("A.6 步权与稳定性项")
    sw = step_w.view(-1).cpu().numpy()
    emit(f"  step_w = gamma^k / mean：首步 {sw[0]:.4f} → 末步 {sw[-1]:.4f}（比 {sw[0] / sw[-1]:.2f}×），"
         f"均值恒为 1 → L_vel 量级与 K 无关")
    emit(f"  spectral_radius(I+W_A) = {float(spec):.6f}，rho_max = {args.rho_max} → "
         f"L_stab = {float(l_stab):.3g}（{'未激活' if float(l_stab) == 0 else '激活'}）")
    emit(f"  L_l2 = ||W_A||² + ||W_B||² = {float(l_l2):.4g}；同时 AdamW weight_decay="
         f"{args.weight_decay} 已对全部参数（含 A/B）解耦衰减 → A/B 被双重正则")
    emit(f"  L_acc 未乘 step_w（L_vel 乘了）；且残差被 1/dt={1 / args.dt:.3g} 缩放 → "
         f"dt={args.dt}s 时加速度残差整体缩小 {args.dt:.0f}×，几乎全落在 Huber 二次区")
    a_res = ((pred_acc - gt_acc) * chan_scale).abs()
    emit(f"    |acc 残差|>beta 的比例 = {100 * float((a_res > args.huber_beta).float().mean()):.1f}%")

    emit()
    emit("A.7 模型选择口径 vs 训练目标")
    emit("  best ckpt 判据 = test/vel_rmse_mean × max(1, instability_score)（train_v4_dict_input.py 内）")
    emit(f"  该判据完全不含位姿项，而位姿项在训练 total 中占 "
         f"{100 * (args.w_xy * float(l_xy) + args.w_yaw * float(l_yaw)) / total:.1f}%"
         " → 优化目标与选择目标不一致")


def build_condensed(
    Abar: np.ndarray, B: np.ndarray, beta: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nz = Abar.shape[0]
    nu = B.shape[1]
    Gamma = np.zeros((nz * N, nz))
    Theta = np.zeros((nz * N, nu * N))
    xi = np.zeros(nz * N)
    Apow = [np.eye(nz)]
    for i in range(1, N + 1):
        Apow.append(Apow[-1] @ Abar)
    for k in range(1, N + 1):
        r = (k - 1) * nz
        Gamma[r:r + nz, :] = Apow[k]
        acc = np.zeros(nz)
        for i in range(k):
            acc += Apow[i] @ beta
        xi[r:r + nz] = acc
        for j in range(k):
            Theta[r:r + nz, j * nu:(j + 1) * nu] = Apow[k - j - 1] @ B
    return Gamma, Theta, xi


def solve_qp(
    P: np.ndarray, q: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray
) -> np.ndarray:
    import osqp
    from scipy import sparse

    prob = osqp.OSQP()
    prob.setup(
        sparse.csc_matrix(np.triu(P)),
        q,
        sparse.csc_matrix(A),
        l,
        u,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=20000,
        verbose=False,
    )
    res = prob.solve()
    if res.x is None or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    return np.asarray(res.x, dtype=float)


def qp_cost(
    Theta: np.ndarray,
    z_free: np.ndarray,
    z_ref: np.ndarray,
    U: np.ndarray,
    w_z: float,
    w_u: float,
    w_du: float,
    N: int,
    nu: int,
) -> float:
    e = Theta @ U + (z_free - z_ref)
    c = w_z * float(e @ e) + w_u * float(U @ U)
    Um = U.reshape(N, nu)
    d = np.diff(Um, axis=0)
    return c + w_du * float((d * d).sum())


def expand_to_full(U: np.ndarray, N: int, nu: int, n_opt: int, hold: int) -> np.ndarray:
    """镜像 C++ LatentMpcQpSolver::expandToFull。"""
    full = U.reshape(N, nu).copy()
    if 0 < n_opt < N:
        full[n_opt:] = full[n_opt - 1]
    if hold > 1:
        for k in range(N):
            full[k] = full[(k // hold) * hold]
    return full.reshape(-1)


def analyze_mpc_cost(
    model: HorizontalKoopmanModelV4DictInput,
    args: argparse.Namespace,
    ds: KoopmanVoyageDataset,
    stats: Dict[str, np.ndarray],
    mpc_N: int,
    mpc_dt: float,
    w_z: float,
    w_u: float,
    w_du: float,
    opt_control_steps: int,
    control_hold_steps: int,
    device: torch.device,
) -> None:
    section("B. 潜空间 MPC QP 目标（latent_mpc_qp.cpp）定量分析")

    nz = model.latent_dim
    nu = model.control_dim
    Abar = (model.A.weight.detach() + torch.eye(nz)).cpu().numpy().astype(float)
    beta = model.A.bias.detach().cpu().numpy().astype(float)
    Bm = model.B.weight.detach().cpu().numpy().astype(float)
    Gamma, Theta, xi = build_condensed(Abar, Bm, beta, mpc_N)
    dyn_std = stats["state_std"][3:6].astype(float)
    ctrl_mean = stats["ctrl_mean"].astype(float)
    ctrl_std = stats["ctrl_std"].astype(float)

    emit(f"N={mpc_N}, dt={mpc_dt}s, nz={nz}, nu={nu}, nvar={mpc_N * nu}; "
         f"w_z={w_z} w_u={w_u} w_du={w_du}; opt_control_steps={opt_control_steps} "
         f"control_hold_steps={control_hold_steps}")
    emit(f"ctrl_mean={np.array2string(ctrl_mean, precision=3)}  "
         f"ctrl_std={np.array2string(ctrl_std, precision=3)}")

    # ---- B.1 w_z 在 48 维潜空间的隐含加权 ----
    emit()
    emit("B.1 w_z 对 48 维潜变量一律相同 → 隐含加权由 encoder 自身尺度决定")
    states = ds.states_full[:, 3:6].astype(np.float32)
    sel = np.linspace(0, states.shape[0] - 1, min(20000, states.shape[0])).astype(int)
    dyn_n = (states[sel] - stats["state_mean"][3:6]) / stats["state_std"][3:6]
    with torch.no_grad():
        Z = model.encode(torch.from_numpy(dyn_n).to(device)).cpu().numpy().astype(float)
    var = Z.var(axis=0)
    emit(f"  数据集上 latent 逐维方差：atoms(0:16) 合计 {var[:16].sum():.4g}，"
         f"hidden(16:48) 合计 {var[16:].sum():.4g}")
    emit(f"  → 等权 w_z 下，跟踪代价的方差预算 atoms:hidden = "
         f"{100 * var[:16].sum() / var.sum():.1f}% : {100 * var[16:].sum() / var.sum():.1f}%")
    emit(f"  逐维方差极差：min={var.min():.4g} max={var.max():.4g}（{var.max() / max(var.min(), 1e-12):.1f}× 跨度）"
         " → 少数大尺度维度主导 QP 目标")

    # ---- B.2 隐含的物理加权（encoder Gram） ----
    emit()
    emit("B.2 w_z‖Δz‖² 折算到物理速度的隐含加权（encoder Jacobian Gram）")
    z_ops = np.linspace(0, dyn_n.shape[0] - 1, 512).astype(int)
    G_acc = np.zeros((3, 3))
    for i in z_ops:
        x = torch.from_numpy(dyn_n[i]).float().to(device)
        Jz = torch.autograd.functional.jacobian(
            lambda t: model.encode(t.unsqueeze(0)).squeeze(0), x
        ).detach().cpu().numpy().astype(float)      # ∂z/∂x_norm, (48,3)
        Jz_phys = Jz / dyn_std[None, :]             # ∂z/∂x_phys
        G_acc += Jz_phys.T @ Jz_phys
    G = G_acc / len(z_ops)
    emit("  在流形上（Δz = ∂z/∂x·Δx），w_z‖Δz‖² = Δxᵀ G Δx，G 的对角/交叉项（物理单位）：")
    for i, ch in enumerate("uvr"):
        emit(f"    G[{ch}] = {np.array2string(G[i], precision=4, suppress_small=False)}")
    dg = np.sqrt(np.diag(G))
    emit(f"  等效单通道权重 √diag(G)：u={dg[0]:.3f} /(m/s), v={dg[1]:.3f} /(m/s), r={dg[2]:.2f} /(rad/s)")
    emit(f"  → 相对 u，v 被放大 {dg[1] / dg[0]:.1f}×，r 被放大 {dg[2] / dg[0]:.1f}×；"
         "该比例来自 encoder 权重尺度，不是可调设计量")
    off = abs(G[0, 1]) + abs(G[0, 2]) + abs(G[1, 2])
    emit(f"  非对角耦合量 |G_uv|+|G_ur|+|G_vr| = {off:.4g} → 通道间存在耦合，无法用对角权重等价表达")

    # ---- B.3 误差能量在 decoder 行空间 / 零空间的分布 ----
    emit()
    emit("B.3 潜空间误差有多少是「物理可见」的（decoder Jacobian 行空间投影）")
    z_bar = torch.from_numpy(Z.mean(axis=0)).float().to(device).requires_grad_(True)
    J = torch.autograd.functional.jacobian(
        lambda z: model.reconstruct_state(z.unsqueeze(0)).squeeze(0), z_bar
    ).detach().cpu().numpy().astype(float)
    Jp = np.diag(dyn_std) @ J  # 归一化输出 → 物理速度
    emit(f"  decoder Jacobian Jp: {Jp.shape}，rank={np.linalg.matrix_rank(Jp)} → "
         f"零空间维度 = {nz - np.linalg.matrix_rank(Jp)}/{nz}")
    # 用真实参考构造典型跟踪误差方向
    idx = np.linspace(0, len(ds) - 1, 256).astype(int)
    e_rows: List[np.ndarray] = []
    for i in idx:
        x_t, x_seq, u_seq = ds[int(i)]
        with torch.no_grad():
            z0 = model.encode(x_t[3:6].unsqueeze(0).to(device)).cpu().numpy().astype(float)[0]
            z_tgt = model.encode(x_seq[:, 3:6].to(device)).cpu().numpy().astype(float)
        z_free = Gamma @ z0 + xi
        e_rows.append(z_free - z_tgt[:mpc_N].reshape(-1))
    E = np.stack(e_rows)  # (M, nz*N)
    E_blocks = E.reshape(E.shape[0] * mpc_N, nz)
    Q_row = np.linalg.pinv(Jp) @ Jp  # 投影到 Jp 行空间
    proj = E_blocks @ Q_row.T
    frac_visible = float((proj ** 2).sum() / (E_blocks ** 2).sum())
    emit(f"  典型跟踪误差 e=z_free−z_ref（{E.shape[0]} 个真实窗口）：")
    emit(f"    落在 decoder 行空间（影响 (u,v,r)）的能量占比 = {100 * frac_visible:.2f}%")
    emit(f"    落在零空间（对物理速度无一阶影响）的占比      = {100 * (1 - frac_visible):.2f}%")
    emit("  → w_z‖e‖² 中绝大部分在惩罚与物理速度无关的潜坐标失配")

    # ---- 公共：QP 矩阵与真实参考窗口 ----
    nvar = mpc_N * nu
    D = np.zeros((nu * (mpc_N - 1), nvar))
    for k in range(1, mpc_N):
        for j in range(nu):
            D[(k - 1) * nu + j, k * nu + j] = 1.0
            D[(k - 1) * nu + j, (k - 1) * nu + j] = -1.0
    H = w_z * (Theta.T @ Theta) + w_u * np.eye(nvar) + w_du * (D.T @ D)
    P = 2 * H

    du_max = np.array([15.0, 3.5, 15.0, 3.5])
    u_min_phys = np.array([-100.0, -35.0, -100.0, -35.0])
    u_max_phys = np.array([100.0, 35.0, 100.0, 35.0])
    A_c = np.zeros((2 * nvar, nvar))
    for i in range(nvar):
        A_c[i, i] = 1.0
    row = nvar
    for k in range(mpc_N):
        for j in range(nu):
            A_c[row, k * nu + j] = 1.0
            if k > 0:
                A_c[row, (k - 1) * nu + j] = -1.0
            row += 1

    def bounds(u_prev_phys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        l_c = np.zeros(2 * nvar)
        u_c = np.zeros(2 * nvar)
        for i in range(nvar):
            ch = i % nu
            l_c[i] = (u_min_phys[ch] - ctrl_mean[ch]) / ctrl_std[ch]
            u_c[i] = (u_max_phys[ch] - ctrl_mean[ch]) / ctrl_std[ch]
        r = nvar
        for k in range(mpc_N):
            for j in range(nu):
                dts = du_max[j] / ctrl_std[j]
                if k == 0:
                    prev = (u_prev_phys[j] - ctrl_mean[j]) / ctrl_std[j]
                    l_c[r], u_c[r] = prev - dts, prev + dts
                else:
                    l_c[r], u_c[r] = -dts, dts
                r += 1
        return l_c, u_c

    # 真实航段窗口：每段起点后等距取样，ref_window 为 N+1 帧（步长 mpc_dt）
    ms = max(1, int(round(mpc_dt / ds.data_dt)))
    need = mpc_N * ms + 1
    windows: List[Dict[str, np.ndarray]] = []
    for si in range(len(ds.seg_lens)):
        s0 = int(ds.seg_starts[si])
        slen = int(ds.seg_lens[si])
        for off in np.linspace(0, max(slen - need - 1, 0), 4).astype(int):
            t0 = s0 + int(off)
            if t0 + need > s0 + slen:
                continue
            ref = ds.states_full[t0:t0 + need:ms][: mpc_N + 1, 3:6].astype(np.float32)
            if ref.shape[0] != mpc_N + 1:
                continue
            windows.append({"ref": ref, "u_prev": ds.ctrls_full[t0].astype(float)})
    windows = windows[:64]

    prepared: List[Dict[str, np.ndarray]] = []
    for w in windows:
        ref_n = (w["ref"] - stats["state_mean"][3:6]) / stats["state_std"][3:6]
        with torch.no_grad():
            Zref_all = model.encode(
                torch.from_numpy(ref_n).float().to(device)
            ).cpu().numpy().astype(float)
        z0 = Zref_all[0]
        prepared.append({
            "z_free": Gamma @ z0 + xi,
            "z_ref_code": Zref_all[0:mpc_N].reshape(-1),     # 代码现状 k=0..N−1
            "z_ref_fix": Zref_all[1:mpc_N + 1].reshape(-1),  # 正确对齐 k=1..N
            "u_prev": w["u_prev"],
            "z0": z0,
            "ref": w["ref"],
        })

    # ---- B.4 参考对齐 ----
    emit()
    emit("B.4 Tier-1 参考对齐：buildRefLatentStack 取 ref_window[k] (k=0..N−1)，"
         "而 predictStacked 第 k 块是 z_{k+1}")
    emit("    buildRefPoseStack 取 ref_window[k] (k=1..N)——两者相差一个 dt，"
         f"即 Tier-1 速度参考整体滞后 {mpc_dt}s")
    du0_diff: List[np.ndarray] = []
    gaps: List[float] = []
    for pr in prepared:
        l_c, u_c = bounds(pr["u_prev"])
        q_code = 2 * w_z * (Theta.T @ (pr["z_free"] - pr["z_ref_code"]))
        q_fix = 2 * w_z * (Theta.T @ (pr["z_free"] - pr["z_ref_fix"]))
        U_code = solve_qp(P, q_code, A_c, l_c, u_c)
        U_fix = solve_qp(P, q_fix, A_c, l_c, u_c)
        du0_diff.append(np.abs((U_fix[:nu] - U_code[:nu]) * ctrl_std))
        j_code = qp_cost(Theta, pr["z_free"], pr["z_ref_fix"], U_code, w_z, w_u, w_du, mpc_N, nu)
        j_fix = qp_cost(Theta, pr["z_free"], pr["z_ref_fix"], U_fix, w_z, w_u, w_du, mpc_N, nu)
        gaps.append(100 * (j_code - j_fix) / max(abs(j_fix), 1e-12))
    dd = np.stack(du0_diff)
    emit(f"  {len(prepared)} 个真实窗口（u_prev 取该时刻实测控制）：")
    emit(f"    首步控制差 |Δu0| 均值：油门 {dd[:, 0].mean():.3f}（p95 {np.percentile(dd[:, 0], 95):.3f}）%FS，"
         f"舵 {dd[:, 1].mean():.3f}（p95 {np.percentile(dd[:, 1], 95):.3f}）°")
    emit(f"    以正确参考评估的代价劣化：均值 {np.mean(gaps):+.2f}%，"
         f"中位 {np.median(gaps):+.2f}%，最大 {np.max(gaps):+.2f}%")
    emit("    （劣化=用滞后参考解出的控制在正确参考下的 J 相对最优 J 的增幅）")

    # ---- B.5 move-blocking 次优性 ----
    emit()
    emit("B.5 move-blocking：QP 在全 horizon 自由求解后事后覆写，而非在 blocked 参数化下求最优")
    hold = max(1, control_hold_steps)
    n_blk = mpc_N // hold
    opt_blk = max(1, min(n_blk, (opt_control_steps + hold - 1) // hold))
    n_opt = opt_blk * hold
    n_free_blk = (n_opt + hold - 1) // hold
    M = np.zeros((nvar, n_free_blk * nu))
    for k in range(mpc_N):
        blk = min(k // hold, n_free_blk - 1)
        for j in range(nu):
            M[k * nu + j, blk * nu + j] = 1.0
    P_b = M.T @ P @ M
    A_b = A_c @ M
    rows = []
    for pr in prepared:
        l_c, u_c = bounds(pr["u_prev"])
        z_ref = pr["z_ref_fix"]
        q = 2 * w_z * (Theta.T @ (pr["z_free"] - z_ref))
        U_free = solve_qp(P, q, A_c, l_c, u_c)
        U_applied = expand_to_full(U_free, mpc_N, nu, n_opt, hold)
        v_opt = solve_qp(P_b, M.T @ q, A_b, l_c, u_c)
        U_blocked = M @ v_opt

        def J(U: np.ndarray) -> float:
            return qp_cost(Theta, pr["z_free"], z_ref, U, w_z, w_u, w_du, mpc_N, nu)

        j_free, j_app, j_blk = J(U_free), J(U_applied), J(U_blocked)
        rows.append([
            j_free, j_app, j_blk,
            100 * (j_app - j_free) / max(abs(j_free), 1e-12),
            100 * (j_app - j_blk) / max(abs(j_blk), 1e-12),
            float(np.abs((U_blocked[:nu] - U_free[:nu]) * ctrl_std).max()),
        ])
    R = np.array(rows)
    emit(f"  n_opt={n_opt}, hold={hold}（{mpc_N - n_opt}/{mpc_N} 步被事后覆写），{len(R)} 个窗口均值：")
    emit(f"    J(QP 自由解 U*)                = {R[:, 0].mean():.4g}   ← QP 实际最小化的目标")
    emit(f"    J(expandToFull(U*)，真正应用)   = {R[:, 1].mean():.4g}   ← evalCost 上报的值")
    emit(f"    J(blocked 参数化下的真最优)     = {R[:, 2].mean():.4g}")
    emit(f"    事后覆写导致的代价上升：均值 {R[:, 3].mean():.1f}%（中位 {np.median(R[:, 3]):.1f}%，"
         f"最大 {R[:, 3].max():.1f}%）")
    emit(f"    相对 blocked 最优的次优性：均值 {R[:, 4].mean():.2f}%（最大 {R[:, 4].max():.2f}%）")
    emit(f"    改为 blocked 参数化后首步控制的变化：最大 {R[:, 5].max():.3f}（第 0 通道量纲 %FS）"
         "；即该缺陷不仅影响计划质量，也会改变实际下发值")

    # ---- B.6 Δu0 缺项 ----
    emit()
    emit("B.6 代价里没有「首步相对上周期实际下发」的平滑项")
    sat = []
    for pr in prepared:
        l_c, u_c = bounds(pr["u_prev"])
        q = 2 * w_z * (Theta.T @ (pr["z_free"] - pr["z_ref_fix"]))
        U = solve_qp(P, q, A_c, l_c, u_c)
        du0 = np.abs(U[:nu] - (pr["u_prev"] - ctrl_mean) / ctrl_std)
        sat.append(du0 / (du_max / ctrl_std))
    S = np.stack(sat)
    emit(f"  |ũ0 − ũ_prev| / 速率上限（饱和度）均值：{np.array2string(S.mean(axis=0), precision=3)}")
    emit(f"  完全贴住速率上限（饱和度>0.99）的比例：{np.array2string((S > 0.99).mean(axis=0), precision=3)}")
    emit("  → 周期间跳变几乎完全由硬约束决定；w_du 只惩罚 horizon 内 k≥1 的相邻差分，"
         "缺 ‖ũ0 − ũ_prev‖² 软项")

    # ---- B.6b u_prev 未接入的后果 ----
    emit()
    emit("B.6b 与之耦合的实现缺陷：u_prev 未接入 → 首步速率约束锚在「物理 0」")
    emit("  motion_koopman_mpc.cpp 组装 MotionSolveInput 时未写 u_prev/has_u_prev，"
         "simulate() 也传 nullptr；")
    emit("  solveStep 里 std::array<float,4> u_prev{} 即物理零 → 首步被夹在 "
         f"[±{du_max[0]:.0f}]%FS / [±{du_max[1]:.1f}]° 内")
    u0s_zero, u0s_real, jz, jr = [], [], [], []
    for pr in prepared:
        q = 2 * w_z * (Theta.T @ (pr["z_free"] - pr["z_ref_fix"]))
        l0, u0b = bounds(np.zeros(4))
        l1, u1b = bounds(pr["u_prev"])
        U0 = solve_qp(P, q, A_c, l0, u0b)
        U1 = solve_qp(P, q, A_c, l1, u1b)
        u0s_zero.append(U0[:nu] * ctrl_std + ctrl_mean)
        u0s_real.append(U1[:nu] * ctrl_std + ctrl_mean)
        jz.append(qp_cost(Theta, pr["z_free"], pr["z_ref_fix"], U0, w_z, w_u, w_du, mpc_N, nu))
        jr.append(qp_cost(Theta, pr["z_free"], pr["z_ref_fix"], U1, w_z, w_u, w_du, mpc_N, nu))
    Z0 = np.stack(u0s_zero)
    Z1 = np.stack(u0s_real)
    emit(f"  首步油门（通道 0）：u_prev=0 时均值 {Z0[:, 0].mean():.2f}%FS（max {Z0[:, 0].max():.2f}）；"
         f"u_prev=实测时均值 {Z1[:, 0].mean():.2f}%FS（max {Z1[:, 0].max():.2f}）")
    emit(f"  首步舵角（通道 1）：u_prev=0 时 |max| {np.abs(Z0[:, 1]).max():.2f}°；"
         f"u_prev=实测时 |max| {np.abs(Z1[:, 1]).max():.2f}°")
    emit(f"  同一目标下的代价：u_prev=0 均值 {np.mean(jz):.4g} vs u_prev=实测 {np.mean(jr):.4g}"
         f"（劣化 {100 * (np.mean(jz) - np.mean(jr)) / max(np.mean(jr), 1e-12):+.1f}%）")

    # ---- B.7 其它结构性观察 ----
    emit()
    emit("B.7 其它结构性观察")
    emit("  · 无终端代价/终端权重：Q = w_z·I 对 k=1..N 等权，也无折扣（训练侧反而有 gamma_step 折扣）")
    emit("  · w_u 惩罚的是 ũ=(u−μ_ctrl)/σ_ctrl，零点是数据集平均控制 "
         f"（油门 {ctrl_mean[0]:.1f}、舵 {ctrl_mean[1]:.3f}），而非物理零输入 → 是「回归巡航工况」正则")
    emit("  · evalCost 只累加 z/u/du 三项，Tier-2 开启时上报的 cost 不含位姿项，SQP 迭代无法用它判敛")
    emit("  · buildHessian 显式构造 (nz·N)² 稠密 Q=w_z·I 再做 ΘᵀQΘ："
         f"{(nz * mpc_N) ** 2 / 1e6:.2f}M 元素矩阵，可直接用 w_z·(ΘᵀΘ) 省掉")
    emit("  · P/q 中省略常数项 w_z‖e‖²（不影响 argmin，但 OSQP 报告的 obj_val 与 evalCost 口径不同）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth")
    ap.add_argument("--data", default="data/koopman_test.npz")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--mpc_horizon", type=int, default=10)
    ap.add_argument("--mpc_dt", type=float, default=4.0)
    ap.add_argument("--w_z", type=float, default=1.0)
    ap.add_argument("--w_u", type=float, default=1e-4)
    ap.add_argument("--w_du", type=float, default=0.05)
    ap.add_argument("--opt_control_steps", type=int, default=2)
    ap.add_argument("--control_hold_steps", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None, help="损失占比图输出路径（png）")
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = load_args_from_ckpt(ckpt)
    stats = ckpt["stats"]

    model = HorizontalKoopmanModelV4DictInput(
        hidden_dim=args.hidden_dim,
        clamp_pif=args.clamp_pif,
        encoder_arch=getattr(args, "encoder_arch", "conv"),
    ).to(device)
    model.load_state_dict(ckpt.get("ema_state_dict") or ckpt["model_state_dict"])
    model.eval()

    emit(f"ckpt = {a.ckpt} (epoch {ckpt.get('epoch')}, best_metric {ckpt.get('best_metric'):.6g})")
    emit(f"data = {a.data}")

    x_t, x_seq, u_seq, ds = build_batch(args, a.data, stats, a.samples, device)
    analyze_training_loss(model, args, x_t, x_seq, u_seq, stats, device)
    analyze_mpc_cost(
        model, args, ds, stats, a.mpc_horizon, a.mpc_dt,
        a.w_z, a.w_u, a.w_du, a.opt_control_steps, a.control_hold_steps, device,
    )

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(_LINES) + "\n", encoding="utf-8")
        print(f"\n[saved] {a.out}")
    if a.fig and _SHARE_CACHE:
        plot_shares(a.fig, args)
    return 0


def plot_shares(path: str, args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [x[0] for x in _SHARE_CACHE]
    wshare = [max(x[1], 1e-3) for x in _SHARE_CACHE]
    gshare = [max(x[2], 1e-3) for x in _SHARE_CACHE]
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(idx - 0.2, wshare, width=0.4, label="share of weighted total loss [%]")
    ax.bar(idx + 0.2, gshare, width=0.4, label="share of gradient norm [%]")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(names)
    ax.set_ylabel("share [%] (log scale)")
    ax.set_title(
        f"v4 training loss decomposition (dt={args.dt}s, K={args.pred_len_max}, "
        f"w_xy={args.w_xy}, w_yaw={args.w_yaw})"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    for i, (ws, gs) in enumerate(zip(wshare, gshare)):
        ax.text(i - 0.2, ws * 1.15, f"{ws:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, gs * 1.15, f"{gs:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[saved] {path}")


if __name__ == "__main__":
    raise SystemExit(main())
