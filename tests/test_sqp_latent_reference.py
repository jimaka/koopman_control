#!/usr/bin/env python3
"""校验潜空间 Tier-2 SQP（Gauss-Newton）推导，并对照「现状 C++ 方案 vs 加固方案」。

配套文档：`docs/v4训练与MPC任务拆解与SQP方案推导.md`（§4 推导、§6 实测）

校验项（全部 float64，算法与 C++ `koopman_control` 一一对应）：

1. condensed 预测矩阵 (Gamma, Theta, xi) vs 逐步递推；
2. decoder 物理 Jacobian vs torch.autograd；
3. 位姿 Jacobian Phi 的二阶收敛，以及**实际 QP 步长下**的线性化相对误差；
4. 引理 1：GN 二次模型在展开点的梯度 == 精确梯度 ∇F（步长控制的合法性前提）；
5. 引理 2：∇F(U0)'d <= -0.5 d'P d（要求 U0 可行）+ warm start 不可行时的风险扫描；
6. move-blocking：事后覆写 vs 降维求解（最优性 + hold>1 时的速率可行性）；
7. 方案对照：定步长（现状）vs 信赖域 + 充分下降（推导方案），可达 / 不可达两种参考；
8. GN 丢弃的残差曲率项量级：精确 Hessian 的最小特征值 vs GN Hessian。

用法：
    python3 tests/test_sqp_latent_reference.py \
        --ckpt checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth --dt 4.0 --horizon 10
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from koopman.mpc.sqp_latent import (  # noqa: E402
    Blocking,
    LatentPoseNlp,
    LatentSystem,
    MpcLimits,
    MpcWeights,
    SqpScheme,
    SqpSolver,
    solve_qp_box,
    summarize,
)
from new_v4_dict_input.model_v4_dict_input import HorizontalKoopmanModelV4DictInput  # noqa: E402

FAILURES: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FAIL'}]{'   ' if cond else ' '}{msg}")
    if not cond:
        FAILURES.append(msg)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 场景搭建
# ---------------------------------------------------------------------------
def load_model(ckpt: str) -> Tuple[HorizontalKoopmanModelV4DictInput, Dict[str, np.ndarray], str]:
    path = Path(ckpt) if Path(ckpt).is_absolute() else REPO_ROOT / ckpt
    if path.is_file():
        from new_v4_dict_input.export_v4_encode_weights import load_v4_model

        model, stats = load_v4_model(str(path))
        return model, stats, f"checkpoint {path}"

    torch.manual_seed(0)
    model = HorizontalKoopmanModelV4DictInput(hidden_dim=32, clamp_pif=5.0, encoder_arch="conv")
    model.eval()
    stats = {
        "state_mean": np.array([0.0, 0.0, 0.0, 2.2514, 0.0002, -0.0001], dtype=np.float32),
        "state_std": np.array([1.0, 1.0, 1.0, 1.5311, 0.2245, 0.01497], dtype=np.float32),
        "ctrl_mean": np.array([47.047, 0.0454, 47.047, 0.0454], dtype=np.float32),
        "ctrl_std": np.array([44.922, 19.125, 44.915, 19.125], dtype=np.float32),
    }
    return model, stats, "随机初始化模型（未找到 checkpoint）"


def encode_np(model, sys: LatentSystem, dyn_phys: np.ndarray) -> np.ndarray:
    dyn_n = (np.asarray(dyn_phys, dtype=np.float64) - sys.dyn_mean) / sys.dyn_std
    with torch.no_grad():
        z = model.encode(torch.tensor(dyn_n[None], dtype=torch.float32)).squeeze(0)
    return z.numpy().astype(np.float64)


def make_nlp(
    model,
    sys: LatentSystem,
    horizon: int,
    dt: float,
    weights: MpcWeights,
    limits: MpcLimits,
    blocking: Blocking,
    dyn0: np.ndarray,
    dyn_ref: np.ndarray,
    pose_ref: np.ndarray,
    u_prev_phys: np.ndarray,
) -> LatentPoseNlp:
    return LatentPoseNlp(
        sys=sys,
        horizon=horizon,
        dt=dt,
        weights=weights,
        z0=encode_np(model, sys, dyn0),
        z_ref_stack=np.tile(encode_np(model, sys, dyn_ref), horizon),
        pose0=np.zeros(3),
        pose_ref=pose_ref,
        u_prev_tilde=sys.normalize_control(u_prev_phys),
        limits=limits,
        blocking=blocking,
    )


def turn_reference(horizon: int, dt: float, speed: float, omega: float) -> np.ndarray:
    """随体系定常回转参考位姿（3N）。omega 越大越接近 / 超出船的操纵能力上限。"""
    radius = speed / omega
    pose = np.zeros(3 * horizon)
    for m in range(1, horizon + 1):
        t = m * dt
        pose[(m - 1) * 3 + 0] = radius * math.sin(omega * t)
        pose[(m - 1) * 3 + 1] = radius * (1.0 - math.cos(omega * t))
        pose[(m - 1) * 3 + 2] = omega * t
    return pose


def reachable_reference(nlp_proto: LatentPoseNlp, u_target_phys: np.ndarray) -> np.ndarray:
    """由一条已知控制序列 rollout 出的参考位姿：残差可趋近 0（零残差 GN 的理想工况）。"""
    U = np.tile(nlp_proto.sys.normalize_control(u_target_phys), nlp_proto.horizon)
    return nlp_proto.pose_rollout(U).reshape(-1)


# ---------------------------------------------------------------------------
# 校验项
# ---------------------------------------------------------------------------
def test_condensed(sys: LatentSystem, horizon: int) -> None:
    section("1. condensed 预测矩阵 vs 逐步递推")
    rng = np.random.default_rng(0)
    z0 = rng.normal(size=sys.nz)
    U = rng.normal(size=sys.nu * horizon) * 0.3
    Gamma, Theta, xi = sys.condensed(horizon)
    Z_cond = Gamma @ z0 + Theta @ U + xi
    z, steps = z0.copy(), []
    for k in range(horizon):
        z = sys.A_bar @ z + sys.B @ U[k * sys.nu : (k + 1) * sys.nu] + sys.beta
        steps.append(z.copy())
    err = float(np.max(np.abs(Z_cond - np.concatenate(steps))))
    print(f"  max_abs_err = {err:.3e}")
    check(err < 1e-9, "condensed (Gamma,Theta,xi) 与逐步递推一致 (<1e-9)")


def test_decoder_jacobian(model, sys: LatentSystem) -> None:
    section("2. decoder 物理 Jacobian vs autograd")
    z = np.random.default_rng(1).normal(size=sys.nz)
    dyn_std_t = torch.tensor(sys.dyn_std, dtype=torch.float32)
    dyn_mean_t = torch.tensor(sys.dyn_mean, dtype=torch.float32)

    def dec(zt: torch.Tensor) -> torch.Tensor:
        return model.reconstruct_state(zt.unsqueeze(0)).squeeze(0) * dyn_std_t + dyn_mean_t

    J_auto = torch.autograd.functional.jacobian(dec, torch.tensor(z, dtype=torch.float32)).numpy()
    err = float(np.max(np.abs(J_auto - sys.decode_jacobian_physical(z))))
    print(f"  max_abs_err = {err:.3e}")
    check(err < 1e-4, "decoder Jacobian 与 autograd 一致 (<1e-4, float32 参照)")


def test_linearization(nlp: LatentPoseNlp) -> None:
    section("3. 位姿线性化精度：渐近阶 + 实际 QP 步长下的误差")
    rng = np.random.default_rng(2)
    U0 = nlp.retract_feasible(rng.normal(size=nlp.nvar) * 0.2)
    Phi = nlp.pose_jacobian(U0)
    g0 = nlp.pose_residual(U0)
    direction = rng.normal(size=nlp.nvar)
    direction /= np.max(np.abs(direction))

    prev, ratios = None, []
    for scale in (1e-2, 5e-3, 2.5e-3, 1.25e-3):
        err = float(np.max(np.abs(nlp.pose_residual(U0 + scale * direction) - g0 - Phi @ (scale * direction))))
        ratios.append(prev / err if prev else float("nan"))
        print(f"  |dU|inf={scale:.2e}  lin_err={err:.3e}  err(prev)/err={ratios[-1]:.2f}")
        prev = err
    check(all(3.5 <= r <= 4.5 for r in ratios[1:]), "线性化误差为 O(|dU|^2)（步长减半误差降 ~4 倍）")

    # 实际 QP 步长：从 U=0 解一次 QP，看线性化在该步长上的相对误差
    U_start = np.zeros(nlp.nvar)
    P, q, const = nlp.gn_model(U_start)
    A, lo, hi = nlp.feasible_constraints(None)
    d = solve_qp_box(P, q, A, lo, hi, x0=U_start) - U_start
    Phi0 = nlp.pose_jacobian(U_start)
    g_true = nlp.pose_residual(U_start + d)
    g_lin = nlp.pose_residual(U_start) + Phi0 @ d
    rel = float(np.max(np.abs(g_true - g_lin)) / (np.max(np.abs(g_true - nlp.pose_residual(U_start))) + 1e-12))
    print(f"  实际 QP 步长 |d|inf = {float(np.max(np.abs(d))):.3f} "
          f"→ 位姿增量线性化相对误差 = {rel * 100:.1f}%")
    check(np.max(np.abs(d)) > 0.05, "首次 QP 步长远大于线性化的高精度邻域（需步长控制）")


def test_gn_gradient_lemma(nlp: LatentPoseNlp) -> None:
    section("4. 引理 1：GN 模型在展开点的梯度 == 精确梯度 ∇F")
    rng = np.random.default_rng(3)
    for tag, U in (("U=0", np.zeros(nlp.nvar)),
                   ("U=随机可行点", nlp.retract_feasible(rng.normal(size=nlp.nvar) * 0.3))):
        wrap_k = nlp.wrap_offsets(U)
        P, q, _ = nlp.gn_model(U, wrap_k)
        grad_exact = nlp.gradient(U, wrap_k)
        rel = float(np.max(np.abs(P @ U + q - grad_exact)) / (np.max(np.abs(grad_exact)) + 1e-12))
        print(f"  {tag}: rel_err = {rel:.3e}")
        check(rel < 1e-10, f"∇m(U0) == ∇F(U0)（{tag}）")


def test_descent_lemma(nlp: LatentPoseNlp) -> None:
    section("5. 引理 2：QP 解给出下降方向（U0 可行）+ 不可行 warm start 风险")
    rng = np.random.default_rng(4)
    M = nlp.blocking.selector(nlp.nu)
    worst = -math.inf
    for i in range(4):
        U0 = nlp.retract_feasible(nlp.blocking.expand_posthoc(rng.normal(size=nlp.nvar) * 0.4, nlp.nu))
        wrap_k = nlp.wrap_offsets(U0)
        P, q, _ = nlp.gn_model(U0, wrap_k)
        grad = P @ U0 + q
        V0 = np.linalg.lstsq(M, U0, rcond=None)[0]
        A, lo, hi = nlp.feasible_constraints(M, center=V0, radius=math.inf)
        d = M @ solve_qp_box(M.T @ P @ M, M.T @ q, A, lo, hi, x0=V0) - U0
        gtd, curv = float(grad @ d), float(0.5 * d @ (P @ d))
        worst = max(worst, (gtd + curv) / (abs(gtd) + 1e-12))
        print(f"  #{i}: g'd={gtd:+.4e}  0.5 d'Pd={curv:.4e}  g'd+0.5d'Pd={gtd + curv:+.3e}")
    check(worst <= 1e-6, "∇F(U0)'d <= -0.5 d'P d 恒成立（QP 最优性 + 可行 U0）")

    # 不可行 warm start 扫描：随机构造上周期解 → 本周期速率锚变化后不可行
    n_infeasible = n_worse = 0
    for seed in range(40):
        r = np.random.default_rng(100 + seed)
        U_warm = np.clip(r.normal(size=nlp.nvar) * 1.2, -2.0, 2.0)
        if nlp.is_feasible(U_warm):
            continue
        n_infeasible += 1
        base = SqpSolver(nlp, SqpScheme.baseline_cpp(iters=1)).solve(U_warm)
        if base.cost_history[-1] > base.cost_history[0] + 1e-9:
            n_worse += 1
    print(f"  随机不可行 warm start {n_infeasible} 例中，定步长首轮令真实代价上升的有 {n_worse} 例")
    check(n_infeasible > 0, "warm start 在真实系统里确实可能落在新周期可行集之外")


def test_move_blocking(model, sys, horizon, dt, limits) -> None:
    section("6. move-blocking：事后覆写 vs 降维求解")
    weights = MpcWeights(w_z=1.0, w_u=1e-4, w_du=0.05, w_xy=0.0, w_yaw=0.0)
    kwargs = dict(
        dyn0=np.array([1.2, 0.0, 0.0]),
        dyn_ref=np.array([3.2, 0.0, 0.01]),
        pose_ref=turn_reference(horizon, dt, 3.2, 0.01),
        u_prev_phys=np.array([30.0, 0.0, 30.0, 0.0]),
    )

    for tag, blocking in (
        ("opt=2, hold=1（默认配置）", Blocking(horizon, opt_control_steps=2, hold=1)),
        ("opt=N, hold=5（控制周期 20s）", Blocking(horizon, opt_control_steps=horizon, hold=5)),
    ):
        nlp = make_nlp(model, sys, horizon, dt, weights, limits, blocking, **kwargs)
        post = SqpSolver(nlp, SqpScheme.baseline_cpp(iters=1)).solve(np.zeros(nlp.nvar))
        red = SqpSolver(nlp, SqpScheme.guarded(iters=1)).solve(np.zeros(nlp.nvar))
        j_post, j_red = nlp.objective(post.U), nlp.objective(red.U)
        print(f"  --- {tag} ---")
        print(f"    事后覆写 J={j_post:.6g}  可行={nlp.is_feasible(post.U)}  "
              f"u0={np.round(sys.denormalize_control(post.U[:4]), 2)}")
        print(f"    降维求解 J={j_red:.6g}  可行={nlp.is_feasible(red.U)}  "
              f"u0={np.round(sys.denormalize_control(red.U[:4]), 2)}   "
              f"次优量 {100.0 * (j_post - j_red) / abs(j_red):+.2f}%")
        check(j_red <= j_post + 1e-6, f"降维求解不劣于事后覆写（{tag}）")
        check(nlp.is_feasible(red.U), f"降维求解满足逐步速率约束（{tag}）")


def test_scheme_comparison(model, sys, horizon, dt, blocking, plot_dir: Optional[Path]) -> None:
    section("7. 方案对照：定步长（现状）vs 信赖域 + 充分下降（推导方案）")
    weights = MpcWeights(w_z=0.0, w_u=1e-4, w_du=0.05, w_xy=1.0, w_yaw=50.0)
    proto = make_nlp(
        model, sys, horizon, dt, weights, MpcLimits(), blocking,
        dyn0=np.array([3.0, 0.0, 0.0]), dyn_ref=np.array([3.0, 0.0, 0.0]),
        pose_ref=np.zeros(3 * horizon), u_prev_phys=np.array([40.0, 0.0, 40.0, 0.0]),
    )
    scenarios = [
        ("A 可达参考 + 速率约束放开", reachable_reference(proto, np.array([80.0, -12.0, 80.0, -12.0])),
         MpcLimits(du_max=np.zeros(4))),
        ("B 不可达激进回转 + 默认速率约束", turn_reference(horizon, dt, 3.0, 0.03), MpcLimits()),
        ("C 不可达激进回转 + 速率约束放开", turn_reference(horizon, dt, 3.0, 0.03),
         MpcLimits(du_max=np.zeros(4))),
    ]

    curves: Dict[str, Dict[str, List[float]]] = {}
    for tag, pose_ref, limits in scenarios:
        print(f"\n  --- 场景：{tag} ---")
        nlp = make_nlp(
            model, sys, horizon, dt, weights, limits, blocking,
            dyn0=np.array([3.0, 0.0, 0.0]), dyn_ref=np.array([3.0, 0.0, 0.0]),
            pose_ref=pose_ref, u_prev_phys=np.array([40.0, 0.0, 40.0, 0.0]),
        )
        base = SqpSolver(nlp, SqpScheme.baseline_cpp(iters=8)).solve(np.zeros(nlp.nvar))
        guard = SqpSolver(nlp, SqpScheme.guarded(iters=8)).solve(np.zeros(nlp.nvar))
        curves[tag] = {"定步长(现状)": base.cost_history, "信赖域+充分下降": guard.cost_history}
        for name, res in (("定步长(现状)", base), ("信赖域+充分下降", guard)):
            print("  " + summarize(name, res, nlp))
            print(f"      J 序列 : {[f'{c:.4g}' for c in res.cost_history]}")
            print(f"      最终/最好 J = {res.cost_history[-1]:.6g} / {min(res.cost_history):.6g}"
                  f"   |d|inf: {[f'{s:.3g}' for s in res.step_norms]}")
            if any(a != 1.0 for a in res.alphas) or any(math.isfinite(r) for r in res.radii):
                print(f"      alpha={[f'{a:.3g}' for a in res.alphas]}  "
                      f"radius={['inf' if not math.isfinite(r) else f'{r:.3g}' for r in res.radii]}")
        check(guard.monotone, f"加固方案代价单调不增（{tag}）")
        check(
            guard.cost_history[-1] <= base.cost_history[-1] * (1.0 + 1e-6) + 1e-9,
            f"加固方案最终代价不劣于定步长（{tag}）",
        )
        if not base.monotone:
            print(f"      * 定步长非单调：最终 J 比自身最好迭代差 "
                  f"{100 * (base.cost_history[-1] / min(base.cost_history) - 1):+.1f}%")

        # 迭代预算：当前默认 sqp_iters=2 够不够（用位姿 RMSE 表达，便于工程判断）
        j_best = min(min(base.cost_history), min(guard.cost_history))
        print("      迭代预算(位姿 xy RMSE / J)：")
        for iters in (1, 2, 4, 8):
            b = SqpSolver(nlp, SqpScheme.baseline_cpp(iters=iters)).solve(np.zeros(nlp.nvar))
            g = SqpSolver(nlp, SqpScheme.guarded(iters=iters)).solve(np.zeros(nlp.nvar))
            print(f"        iters={iters}: 定步长 {nlp.pose_metrics(b.U)['xy_rmse_m']:7.3f}m / "
                  f"{b.cost_history[-1]:<10.5g}  加固 {nlp.pose_metrics(g.U)['xy_rmse_m']:7.3f}m / "
                  f"{g.cost_history[-1]:<10.5g}  (J_best={j_best:.5g})")

        if "A 可达" in tag:  # 零残差工况：检验超线性收敛
            steps = [s for s in guard.step_norms if s > 1e-12]
            orders = [math.log(steps[i + 1]) / math.log(steps[i]) for i in range(len(steps) - 1)
                      if steps[i] < 1.0 and steps[i + 1] > 0]
            print(f"      零残差收敛：|d|inf 序列 {[f'{s:.2e}' for s in steps]}")
            if orders:
                print(f"      末段收敛阶估计 log|d_{{k+1}}|/log|d_k| = {[f'{o:.2f}' for o in orders]}")
            check(min(guard.cost_history) < 1e-2, "可达参考下 SQP 收敛到近零残差（J < 1e-2）")

    if plot_dir is not None:
        _plot_curves(curves, plot_dir)


def test_hessian_curvature(model, sys, horizon, dt, blocking) -> None:
    section("8. GN 丢弃的残差曲率项量级（精确 Hessian vs GN Hessian）")
    weights = MpcWeights(w_z=0.0, w_u=1e-4, w_du=0.05, w_xy=1.0, w_yaw=50.0)
    proto = make_nlp(
        model, sys, horizon, dt, weights, MpcLimits(du_max=np.zeros(4)), blocking,
        dyn0=np.array([3.0, 0.0, 0.0]), dyn_ref=np.array([3.0, 0.0, 0.0]),
        pose_ref=np.zeros(3 * horizon), u_prev_phys=np.array([40.0, 0.0, 40.0, 0.0]),
    )
    cases = [
        ("可达参考 / SQP 收敛点（残差→0）",
         reachable_reference(proto, np.array([80.0, -12.0, 80.0, -12.0])), True),
        ("不可达参考 / SQP 收敛点（残差大）", turn_reference(horizon, dt, 3.0, 0.03), True),
        ("不可达参考 / U=0（残差最大）", turn_reference(horizon, dt, 3.0, 0.03), False),
    ]
    for tag, pose_ref, converge in cases:
        nlp = make_nlp(
            model, sys, horizon, dt, weights, MpcLimits(du_max=np.zeros(4)), blocking,
            dyn0=np.array([3.0, 0.0, 0.0]), dyn_ref=np.array([3.0, 0.0, 0.0]),
            pose_ref=pose_ref, u_prev_phys=np.array([40.0, 0.0, 40.0, 0.0]),
        )
        U = (SqpSolver(nlp, SqpScheme.guarded(iters=12)).solve(np.zeros(nlp.nvar)).U
             if converge else np.zeros(nlp.nvar))
        wrap_k = nlp.wrap_offsets(U)
        P_gn, _, _ = nlp.gn_model(U, wrap_k)
        eps = 1e-6
        H = np.zeros((nlp.nvar, nlp.nvar))
        for i in range(nlp.nvar):
            e = np.zeros(nlp.nvar)
            e[i] = eps
            H[:, i] = (nlp.gradient(U + e, wrap_k) - nlp.gradient(U - e, wrap_k)) / (2 * eps)
        H = 0.5 * (H + H.T)
        g = nlp.pose_residual(U, wrap_k)
        print(f"  {tag}")
        print(f"    ||g||inf={float(np.max(np.abs(g))):8.3f}  "
              f"||H-H_GN||/||H_GN||={float(np.linalg.norm(H - P_gn) / np.linalg.norm(P_gn)):.3f}  "
              f"lam_min(H)={float(np.min(np.linalg.eigvalsh(H))):+.3e}  "
              f"lam_min(H_GN)={float(np.min(np.linalg.eigvalsh(P_gn))):+.3e}")
        check(float(np.min(np.linalg.eigvalsh(P_gn))) >= -1e-9, f"GN Hessian 半正定（{tag}）")


def _plot_curves(curves: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    """把各场景的代价下降曲线画在一张图上（横轴 = SQP 外迭代次数）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 4.0))
    axes = np.atleast_1d(axes)
    for ax, (tag, series) in zip(axes, curves.items()):
        for name, hist in series.items():
            style = "--o" if "定步长" in name else "-s"
            ax.semilogy(range(len(hist)), np.maximum(hist, 1e-12), style, label=name, markersize=4)
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("SQP outer iteration")
        ax.set_ylabel("objective J (log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, prop={"family": ["DejaVu Sans"]})
    fig.suptitle("Latent Tier-2 SQP: fixed unit step (current C++) vs trust region + sufficient decrease", fontsize=10)
    fig.tight_layout()
    path = out_dir / "sqp_cost_history.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\n  [plot] {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--dt", type=float, default=4.0)
    ap.add_argument("--opt_control_steps", type=int, default=10)
    ap.add_argument("--plot_dir", default="", help="非空则输出代价下降曲线 PNG")
    args = ap.parse_args()

    model, stats, src = load_model(args.ckpt)
    sys_lat = LatentSystem.from_torch_model(model, stats)
    rho = float(np.max(np.abs(np.linalg.eigvals(sys_lat.A_bar))))
    print("=" * 88)
    print("潜空间 Tier-2 SQP 参考实现校验")
    print(f"  模型来源: {src}")
    print(f"  nz={sys_lat.nz} nu={sys_lat.nu} horizon={args.horizon} dt={args.dt}s "
          f"rho(A_bar)={rho:.5f} rho^N={rho ** args.horizon:.4f}")
    print("=" * 88)

    blocking = Blocking(horizon=args.horizon, opt_control_steps=args.opt_control_steps, hold=1)
    nlp_ref = make_nlp(
        model, sys_lat, args.horizon, args.dt,
        MpcWeights(w_z=0.0, w_u=1e-4, w_du=0.05, w_xy=1.0, w_yaw=50.0),
        MpcLimits(), blocking,
        dyn0=np.array([3.0, 0.0, 0.0]), dyn_ref=np.array([3.0, 0.0, 0.0]),
        pose_ref=turn_reference(args.horizon, args.dt, 3.0, 0.03),
        u_prev_phys=np.array([40.0, 0.0, 40.0, 0.0]),
    )

    test_condensed(sys_lat, args.horizon)
    test_decoder_jacobian(model, sys_lat)
    test_linearization(nlp_ref)
    test_gn_gradient_lemma(nlp_ref)
    test_descent_lemma(nlp_ref)
    test_move_blocking(model, sys_lat, args.horizon, args.dt, MpcLimits())
    test_scheme_comparison(
        model, sys_lat, args.horizon, args.dt, blocking,
        Path(args.plot_dir) if args.plot_dir else None,
    )
    test_hessian_curvature(model, sys_lat, args.horizon, args.dt, blocking)

    print("\n" + "=" * 88)
    if FAILURES:
        print(f"[FAIL] {len(FAILURES)} 项未通过：")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("[OK] 全部校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
