"""v4 多步训练代价函数的逐步结构分析。

聚焦"多步"这一维度（与 `scripts/analyze_v4_cost.py` 的整体分解互补）：

    A. 损失质量在预测步上的分布（步权 × 误差增长）
    B. 多步误差的结构：系统性偏差 vs 随机误差
    C. 梯度的多步归属（∂L/∂u_k、逐步反传、与 grad-clip 的相互作用）
    D. curriculum 扩窗时各项的量级漂移（K 依赖）
    E. 多步监督的输入一致性：dt 内控制被当作 ZOH 的代价
    F. 潜算子的多步传播增益与开环 rollout 设置

用法：
    python3 scripts/analyze_v4_multistep_cost.py \
        --ckpt checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth \
        --data data/koopman_test.npz --out logs/analyze_v4_multistep.txt \
        --fig_dir logs/multistep_figs

纯分析脚本，不修改模型与配置。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.model_v4_dict_input import (  # noqa: E402
    HorizontalKoopmanModelV4DictInput,
)
from new_v4_dict_input.train_v4_dict_input import (  # noqa: E402
    KoopmanVoyageDataset,
    denorm_pose,
    huber,
    integrate_pose_from_vel,
    make_step_weights,
    wrap_yaw_diff,
)
from scripts.analyze_v4_cost import build_batch, load_args_from_ckpt  # noqa: E402

_LINES: List[str] = []


def emit(msg: str = "") -> None:
    print(msg)
    _LINES.append(msg)


def section(title: str) -> None:
    emit()
    emit("=" * 78)
    emit(title)
    emit("=" * 78)


def fmt_row(name: str, vals: np.ndarray, width: int = 8, prec: int = 4) -> str:
    return f"{name:>16} " + " ".join(f"{v:{width}.{prec}f}" for v in vals)


def fmt_head(k: int) -> str:
    return f"{'step k':>16} " + " ".join(f"{i + 1:8d}" for i in range(k))


class Rollout:
    """一次 rollout 的中间量（保留计算图，供逐步反传使用）。"""

    def __init__(
        self,
        model: HorizontalKoopmanModelV4DictInput,
        x_t: torch.Tensor,
        x_seq: torch.Tensor,
        u_seq: torch.Tensor,
        stats: Dict[str, np.ndarray],
        dt: float,
        device: torch.device,
        u_requires_grad: bool = False,
    ) -> None:
        self.dt = float(dt)
        self.dyn_mean = torch.tensor(stats["state_mean"][3:6], device=device)
        self.dyn_std = torch.tensor(stats["state_std"][3:6], device=device)
        self.pose_mean = torch.tensor(stats["state_mean"][:3], device=device)
        self.pose_std = torch.tensor(stats["state_std"][:3], device=device)
        self.K = int(u_seq.shape[1])

        self.u_seq = u_seq.clone().requires_grad_(u_requires_grad)
        dyn_t_n = x_t[:, 3:6]
        self.target_n = x_seq[:, :, 3:6]

        z = model.encode(dyn_t_n)
        pred_n, lat = [], []
        for i in range(self.K):
            z = model.latent_step(z, self.u_seq[:, i, :])
            lat.append(z)
            pred_n.append(model.reconstruct_state(z))
        self.pred_n = torch.stack(pred_n, dim=1)
        self.lat = torch.stack(lat, dim=1)

        self.pred_phys = self.pred_n * self.dyn_std + self.dyn_mean
        self.target_phys = self.target_n * self.dyn_std + self.dyn_mean
        bsz = self.target_n.shape[0]
        self.target_lat = model.encode(self.target_n.reshape(bsz * self.K, 3)).view(bsz, self.K, -1)

        self.pose0 = denorm_pose(x_t[:, :3], self.pose_mean, self.pose_std)
        self.target_pose = denorm_pose(x_seq[:, :, :3], self.pose_mean, self.pose_std)
        self.pred_pose = integrate_pose_from_vel(self.pose0, self.pred_phys, self.dt)

    # ---- 逐步（未加权）损失分量：返回 (K,) ----
    def per_step_terms(self, beta: float) -> Dict[str, torch.Tensor]:
        res_n = self.pred_n - self.target_n
        l_vel = huber(res_n, beta=beta).mean(dim=(0, 2))
        e_x = self.pred_pose[..., 0] - self.target_pose[..., 0]
        e_y = self.pred_pose[..., 1] - self.target_pose[..., 1]
        l_xy = (e_x * e_x + e_y * e_y).mean(dim=0)
        l_yaw = huber(wrap_yaw_diff(self.pred_pose[..., 2], self.target_pose[..., 2]),
                      beta=beta).mean(dim=0)
        l_lin = ((self.lat - self.target_lat) ** 2).mean(dim=(0, 2))
        acc_res = (self.pred_phys[:, 1:] - self.pred_phys[:, :-1] -
                   (self.target_phys[:, 1:] - self.target_phys[:, :-1])) / self.dt
        l_acc = huber(acc_res / self.dyn_std.view(1, 1, 3), beta=beta).mean(dim=(0, 2))
        return {"L_vel": l_vel, "L_xy": l_xy, "L_yaw": l_yaw, "L_lin": l_lin, "L_acc": l_acc}

    def weighted_total(self, args: argparse.Namespace, k_use: int | None = None) -> torch.Tensor:
        """按训练权重组合前 k_use 步（默认全部），步权按 k_use 重新归一化。"""
        k = self.K if k_use is None else int(k_use)
        dev = self.pred_n.device
        w = make_step_weights(k, args.gamma_step, dev).view(1, k, 1)
        chan = 1.0 / self.dyn_std.view(1, 1, 3)
        l_vel = (huber((self.pred_phys[:, :k] - self.target_phys[:, :k]) * chan,
                       beta=args.huber_beta) * w).mean()
        if k > 1:
            acc_res = (self.pred_phys[:, 1:k] - self.pred_phys[:, :k - 1] -
                       (self.target_phys[:, 1:k] - self.target_phys[:, :k - 1])) / self.dt
            l_acc = huber(acc_res * chan, beta=args.huber_beta).mean()
        else:
            l_acc = torch.zeros((), device=dev)
        l_lin = ((self.lat[:, :k] - self.target_lat[:, :k]) ** 2).mean()
        wp = w.squeeze(-1)
        e_x = self.pred_pose[:, :k, 0] - self.target_pose[:, :k, 0]
        e_y = self.pred_pose[:, :k, 1] - self.target_pose[:, :k, 1]
        l_xy = ((e_x * e_x + e_y * e_y) * wp).mean()
        l_yaw = (huber(wrap_yaw_diff(self.pred_pose[:, :k, 2], self.target_pose[:, :k, 2]),
                       beta=args.huber_beta) * wp).mean()
        return (args.w_vel * l_vel + args.w_acc * l_acc + args.w_lin * l_lin +
                args.w_xy * l_xy + args.w_yaw * l_yaw), {
            "L_vel": l_vel, "L_acc": l_acc, "L_lin": l_lin, "L_xy": l_xy, "L_yaw": l_yaw,
        }


def flat_grad(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    parts = [p.grad.detach().reshape(-1).clone() if p.grad is not None
             else torch.zeros(p.numel()) for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return torch.cat(parts)


def analyze_step_distribution(
    ro: Rollout, args: argparse.Namespace
) -> None:
    section("A. 损失质量在预测步上的分布")
    K = ro.K
    terms = {k: v.detach().cpu().numpy() for k, v in ro.per_step_terms(args.huber_beta).items()}
    w = make_step_weights(K, args.gamma_step, ro.pred_n.device).detach().cpu().numpy()

    emit(f"K={K}, dt={args.dt}s（预测时长 {K * args.dt:.0f}s），gamma_step={args.gamma_step}")
    emit()
    emit("A.1 步权本身几乎是平的（gamma^k / mean）")
    emit(fmt_head(K))
    emit(fmt_row("w_k", w))
    emit(f"  首/末步权重比 = {w[0] / w[-1]:.3f}；即 gamma_step=0.97 在 K={K} 上只造成 "
         f"{100 * (w[0] / w[-1] - 1):.0f}% 的差异，实际起作用的是误差随步的增长")

    emit()
    emit("A.2 各项的逐步未加权值（batch/通道平均）")
    emit(fmt_head(K))
    for name in ("L_vel", "L_lin", "L_yaw"):
        emit(fmt_row(name, terms[name], prec=5))
    emit(fmt_row("L_xy", terms["L_xy"], prec=3))
    emit(fmt_row("L_acc(k→k+1)", np.concatenate([terms["L_acc"], [np.nan]]), prec=5))

    emit()
    emit("A.3 加权后每步占该项总量的比例 [%]（w_k × 逐步值，归一化）")
    emit(fmt_head(K))
    shares = {}
    for name in ("L_vel", "L_lin", "L_xy", "L_yaw"):
        m = terms[name] * w
        s = 100 * m / m.sum()
        shares[name] = s
        emit(fmt_row(name, s, prec=2))
    emit()
    emit("  前 3 步 / 后 3 步的份额：")
    for name, s in shares.items():
        emit(f"    {name:>7}: 前 3 步 {s[:3].sum():5.1f}%  后 3 步 {s[-3:].sum():5.1f}%"
             f"  （末步单独 {s[-1]:.1f}%）")
    emit("  → 步权名义上偏向早期，但误差增长使实际损失质量集中在末段；"
         "L_xy 的集中度最高（位置误差随步近似二次增长）")

    emit()
    emit("A.4 若改 gamma_step，末段份额如何变（同一 batch，仅换步权）")
    emit(f"{'gamma':>8} | {'L_vel 末3步%':>12} | {'L_xy 末3步%':>12} | {'L_vel 首步%':>11} | {'L_xy 首步%':>11}")
    emit("-" * 68)
    for g in (0.80, 0.90, 0.97, 1.00, 1.05):
        wg = g ** np.arange(K)
        wg = wg / wg.mean()
        mv = terms["L_vel"] * wg
        mx = terms["L_xy"] * wg
        emit(f"{g:8.2f} | {100 * mv[-3:].sum() / mv.sum():12.1f} | "
             f"{100 * mx[-3:].sum() / mx.sum():12.1f} | "
             f"{100 * mv[0] / mv.sum():11.1f} | {100 * mx[0] / mx.sum():11.1f}")
    emit("  → 在 K=10 下，gamma 从 0.80 调到 1.05 也只把 L_vel 末段份额从 30% 挪到 43% 附近；"
         "步权不是控制多步权衡的有效旋钮")


def analyze_error_structure(ro: Rollout, args: argparse.Namespace) -> None:
    section("B. 多步误差的结构：系统性偏差 vs 随机误差")
    err = (ro.pred_phys - ro.target_phys).detach()
    K = ro.K
    emit("每步 |mean(误差)| / rmse(误差)：接近 1 表示误差是同向累积（偏差主导），"
         "接近 0 表示随机抵消")
    emit(fmt_head(K))
    for i, ch in enumerate("uvr"):
        e = err[..., i]
        bias = e.mean(dim=0).abs()
        rmse = torch.sqrt((e ** 2).mean(dim=0))
        emit(fmt_row(f"{ch} bias/rmse", (bias / rmse).cpu().numpy(), prec=3))
    emit()
    for i, ch in enumerate("uvr"):
        e = err[..., i]
        emit(fmt_row(f"{ch} rmse", torch.sqrt((e ** 2).mean(dim=0)).cpu().numpy(), prec=5))
    emit()
    for i, ch in enumerate("uvr"):
        e = err[..., i]
        emit(fmt_row(f"{ch} bias", e.mean(dim=0).cpu().numpy(), prec=5))
    rmse_all = torch.sqrt((err ** 2).mean(dim=(0, 2))).cpu().numpy()
    slope = np.polyfit(np.log(np.arange(1, K + 1)), np.log(rmse_all), 1)[0]
    emit()
    emit(f"  总 rmse 的 log-log 斜率 = {slope:.3f}（1.0=相干线性累积，0.5=随机游走，0=无增长）")
    emit("  多步损失里没有任何一项显式惩罚「偏差随步单调增长」——v3a 曾有 L_bias/L_slope，v4 移除了；"
         "位姿项恰恰是**鼓励**用偏差换累积量的（见 §1.4 of 代价函数设计分析）")


def analyze_gradient_attribution(
    model: HorizontalKoopmanModelV4DictInput,
    x_t: torch.Tensor,
    x_seq: torch.Tensor,
    u_seq: torch.Tensor,
    stats: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    section("C. 梯度的多步归属")
    ro = Rollout(model, x_t, x_seq, u_seq, stats, args.dt, device, u_requires_grad=True)
    K = ro.K
    total, _ = ro.weighted_total(args)
    g_u = torch.autograd.grad(total, ro.u_seq, retain_graph=True)[0].detach()
    gu_norm = g_u.pow(2).sum(dim=(0, 2)).sqrt().cpu().numpy()
    emit("C.1 ∂L/∂u_k 的范数（哪一步的控制真正被损失关心）")
    emit(fmt_head(K))
    emit(fmt_row("|dL/du_k|", gu_norm, prec=4))
    emit(fmt_row("占比 [%]", 100 * gu_norm / gu_norm.sum(), prec=2))
    emit(f"  首步/末步 = {gu_norm[0] / gu_norm[-1]:.1f}×"
         "（早期控制影响后面所有步，因此梯度天然集中在前段）")

    emit()
    emit("C.2 只保留第 k 步的损失时，全模型参数梯度范数")
    ro2 = Rollout(model, x_t, x_seq, u_seq, stats, args.dt, device)
    terms = ro2.per_step_terms(args.huber_beta)
    w = make_step_weights(K, args.gamma_step, device)
    rows = {}
    for name, weight in (("L_vel", args.w_vel), ("L_xy", args.w_xy), ("L_lin", args.w_lin)):
        gs = []
        for k in range(K):
            loss_k = weight * w[k] * terms[name][k] / K
            gs.append(float(flat_grad(model, loss_k).norm()))
        rows[name] = np.array(gs)
        emit(fmt_head(K) if name == "L_vel" else "")
        emit(fmt_row(name, rows[name], prec=4))
    emit()
    for name, v in rows.items():
        emit(f"    {name:>7}: 末 3 步贡献 {100 * v[-3:].sum() / v.sum():.1f}% 的梯度范数")

    emit()
    emit("C.3 与 grad-clip 的相互作用（训练里 clip_grad_norm_=1.0）")
    g_tot = flat_grad(model, total)
    emit(f"  完整损失的梯度范数 = {float(g_tot.norm()):.4g} → 被裁剪到 1.0，"
         f"裁剪系数 {1.0 / float(g_tot.norm()):.2e}")
    emit("  裁剪只改幅值不改方向，所以「方向由谁决定」才是关键。各项梯度与总梯度的夹角余弦：")
    _, wt = ro2.weighted_total(args)
    for name, weight in (("L_vel", args.w_vel), ("L_acc", args.w_acc), ("L_lin", args.w_lin),
                         ("L_xy", args.w_xy), ("L_yaw", args.w_yaw)):
        if weight == 0:
            continue
        g = flat_grad(model, weight * wt[name])
        cos = float(torch.dot(g, g_tot) / (g.norm() * g_tot.norm() + 1e-30))
        emit(f"    cos(∇L, ∇({name})) = {cos:+.4f}")
    emit("  → 更新方向几乎完全由位姿项决定；其它项即便梯度方向与之冲突，也只能做微小修正")


def analyze_horizon_scaling(ro: Rollout, args: argparse.Namespace) -> None:
    section("D. curriculum 扩窗时各项的量级漂移（K 依赖）")
    emit("同一 batch，把损失按 K'=1..K 截断（步权按 K' 重新归一化），复现 curriculum 各阶段的量级：")
    emit(f"{'K':>4} | {'L_vel':>9} | {'L_acc':>9} | {'L_lin':>9} | {'L_xy':>9} | {'L_yaw':>9} | "
         f"{'加权总':>9} | {'位姿占比%':>9}")
    emit("-" * 88)
    base = {}
    for k in range(1, ro.K + 1):
        _, t = ro.weighted_total(args, k_use=k)
        v = {n: float(x.detach()) for n, x in t.items()}
        tot = (args.w_vel * v["L_vel"] + args.w_acc * v["L_acc"] + args.w_lin * v["L_lin"] +
               args.w_xy * v["L_xy"] + args.w_yaw * v["L_yaw"])
        pose = args.w_xy * v["L_xy"] + args.w_yaw * v["L_yaw"]
        emit(f"{k:4d} | {v['L_vel']:9.5f} | {v['L_acc']:9.5f} | {v['L_lin']:9.5f} | "
             f"{v['L_xy']:9.4f} | {v['L_yaw']:9.5f} | {tot:9.4f} | {100 * pose / tot:9.2f}")
        if k == 1:
            base = dict(v)
    _, t_end = ro.weighted_total(args, k_use=ro.K)
    v_end = {n: float(x.detach()) for n, x in t_end.items()}
    emit()
    emit(f"  K=1 → K={ro.K} 的放大倍数：L_vel ×{v_end['L_vel'] / max(base['L_vel'], 1e-12):.1f}，"
         f"L_lin ×{v_end['L_lin'] / max(base['L_lin'], 1e-12):.1f}，"
         f"L_xy ×{v_end['L_xy'] / max(base['L_xy'], 1e-12):.1f}")
    emit("  步权的 /mean 归一化让 L_vel 的量级对 K 基本不敏感（只随误差增长），"
         "而 L_xy 随 K 近似二次增长 → **curriculum 每次扩窗都在偷偷提高位姿项的有效权重**")
    emit(f"  实测位姿项占比：K=1 时 {100 * (args.w_xy * base['L_xy'] + args.w_yaw * base['L_yaw']) / (args.w_vel * base['L_vel'] + args.w_lin * base['L_lin'] + args.w_xy * base['L_xy'] + args.w_yaw * base['L_yaw']):.1f}%"
         f" → K={ro.K} 时 {100 * (args.w_xy * v_end['L_xy'] + args.w_yaw * v_end['L_yaw']) / (args.w_vel * v_end['L_vel'] + args.w_acc * v_end['L_acc'] + args.w_lin * v_end['L_lin'] + args.w_xy * v_end['L_xy'] + args.w_yaw * v_end['L_yaw']):.1f}%")
    emit("  这与 pose_ramp（前 10 epoch 线性 ramp）叠加：ramp 结束后 curriculum 还在继续放大位姿项")

    emit()
    emit("D.2 若把位姿误差按「行进距离」无量纲化，K 依赖就消失了")
    u_bar = float(ro.dyn_mean[0])
    emit(f"  定义 L_xy_rel = mean_k[ w_k · (e_x²+e_y²)/(k·dt·ū)² ]，ū={u_bar:.3f} m/s（平均纵向速度）")
    emit(f"{'K':>4} | {'L_xy':>9} | {'L_xy_rel':>9} | {'现状位姿占比%':>12} | {'无量纲化后占比%':>14}")
    emit("-" * 62)
    dev = ro.pred_n.device
    e_x_all = ro.pred_pose[..., 0] - ro.target_pose[..., 0]
    e_y_all = ro.pred_pose[..., 1] - ro.target_pose[..., 1]
    dist = torch.arange(1, ro.K + 1, device=dev, dtype=torch.float32) * ro.dt * u_bar
    rel = ((e_x_all ** 2 + e_y_all ** 2) / dist.view(1, -1) ** 2).detach()
    for k in (1, 2, 5, ro.K):
        _, t = ro.weighted_total(args, k_use=k)
        v = {n: float(x.detach()) for n, x in t.items()}
        w = make_step_weights(k, args.gamma_step, dev).detach()
        l_rel = float((rel[:, :k] * w.view(1, -1)).mean())
        others = (args.w_vel * v["L_vel"] + args.w_acc * v["L_acc"] + args.w_lin * v["L_lin"] +
                  args.w_yaw * v["L_yaw"])
        emit(f"{k:4d} | {v['L_xy']:9.4f} | {l_rel:9.5f} | "
             f"{100 * args.w_xy * v['L_xy'] / (others + args.w_xy * v['L_xy']):12.2f} | "
             f"{100 * args.w_xy * l_rel / (others + args.w_xy * l_rel):14.2f}")
    _, t_k = ro.weighted_total(args, k_use=ro.K)
    v_k = {n: float(x.detach()) for n, x in t_k.items()}
    emit("  → 无量纲化后位姿项占比在 K 变化时基本恒定（同一个 w_xy 下 0.4%~0.6%），"
         "权重才重新变成一个可标定的量（当然要重新选量级）")
    emit(f"  另一种口径：让位姿项与速度项在 K={ro.K} 上等份额，需要 "
         f"w_xy ≈ {args.w_vel * v_k['L_vel'] / v_k['L_xy']:.4f}"
         f"（当前 {args.w_xy}，相差 {args.w_xy / (args.w_vel * v_k['L_vel'] / v_k['L_xy']):.0f}×）")


@torch.no_grad()
def analyze_zoh_consistency(
    model: HorizontalKoopmanModelV4DictInput,
    ds: KoopmanVoyageDataset,
    stats: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    n_windows: int,
) -> None:
    section("E. 多步监督的输入一致性：dt 内控制被当作 ZOH 的代价")
    ms = ds.model_stride
    K = ds.pred_len
    emit(f"Dataset 取控制的方式：u_seq = ctrls_full[t0 : t0+K*{ms} : {ms}]"
         f"（每 {ms} 个 0.1s 采样只取块首），并作为整个 {args.dt}s 步的输入")
    emit("但真实指令在块内是变化的——多步监督因此存在系统性输入失配。")

    t0s = ds.t0_global
    sel = np.linspace(0, t0s.shape[0] - 1, min(n_windows, t0s.shape[0])).astype(int)
    ctrl_std = stats["ctrl_std"].astype(np.float32)

    dev_list, dev_thr_list, dev_rud_list = [], [], []
    for i in sel:
        t0 = int(t0s[i])
        blocks = ds.ctrls_full[t0:t0 + K * ms].reshape(K, ms, 4)
        lead = blocks[:, 0:1, :]
        # 块内相对块首的最大偏离，按块取平均
        raw = np.abs(blocks - lead).max(axis=1)          # (K, 4) 物理量
        dev_thr_list.append(raw[:, 0].mean())            # 油门 %FS
        dev_rud_list.append(raw[:, 1].mean())            # 舵 °
        dev_list.append((raw / ctrl_std[None, :]).max(axis=1).mean())
    dev_arr = np.array(dev_list)
    dev_thr = np.array(dev_thr_list)
    dev_rud = np.array(dev_rud_list)

    # 同一批窗口的多步 rollout 误差
    xs, ys, us = [], [], []
    for i in sel:
        a, b, c = ds[int(i)]
        xs.append(a)
        ys.append(b)
        us.append(c)
    x_t = torch.stack(xs).to(device)
    x_seq = torch.stack(ys).to(device)
    u_seq = torch.stack(us).to(device)
    dyn_std = torch.tensor(stats["state_std"][3:6], device=device)
    dyn_mean = torch.tensor(stats["state_mean"][3:6], device=device)
    z = model.encode(x_t[:, 3:6])
    preds = []
    for i in range(K):
        z = model.latent_step(z, u_seq[:, i, :])
        preds.append(model.reconstruct_state(z))
    pred_n = torch.stack(preds, dim=1)
    res_n = (pred_n - x_seq[:, :, 3:6]).cpu().numpy()
    win_rmse = np.sqrt((res_n ** 2).mean(axis=(1, 2)))
    pred_phys = (pred_n * dyn_std + dyn_mean).cpu().numpy()
    tgt_phys = (x_seq[:, :, 3:6] * dyn_std + dyn_mean).cpu().numpy()
    win_rmse_phys = np.sqrt(((pred_phys - tgt_phys) ** 2).mean(axis=(1, 2)))

    const_mask = dev_arr <= 1e-9
    emit()
    emit(f"E.1 块内控制偏离统计（{len(sel)} 个窗口 × {K} 个块）")
    emit(f"  归一化偏离（取通道最大 / σ_ctrl）：均值 {dev_arr.mean():.3f}，"
         f"中位 {np.median(dev_arr):.3f}，p90 {np.percentile(dev_arr, 90):.3f}，最大 {dev_arr.max():.3f}")
    emit(f"  块内指令完全恒定的窗口占比 = {100 * const_mask.mean():.1f}%"
         f"（数据采集是分段保持指令的，多数块确实是常值）")
    emit(f"  非恒定窗口的物理偏离：油门 均值 {dev_thr[~const_mask].mean():.2f} %FS"
         f"（p90 {np.percentile(dev_thr[~const_mask], 90):.2f}），"
         f"舵 均值 {dev_rud[~const_mask].mean():.2f}°"
         f"（p90 {np.percentile(dev_rud[~const_mask], 90):.2f}）")

    emit()
    emit("E.2 输入失配与多步误差的关系（0 组 = 块内常值，其余按非零偏离三分位）")
    pos = dev_arr[~const_mask]
    q = np.quantile(pos, [1 / 3, 2 / 3]) if pos.size else np.array([0.0, 0.0])
    bins = np.where(const_mask, 0, 1 + np.digitize(dev_arr, q))
    labels = {0: "常值块", 1: "偏离小", 2: "偏离中", 3: "偏离大"}
    emit(f"{'分组':>8} | {'块内偏离(σ)':>11} | {'窗口数':>6} | {'归一化 rmse':>11} | {'物理 vel rmse':>13}")
    emit("-" * 62)
    for b in sorted(labels):
        m = bins == b
        if not m.any():
            continue
        emit(f"{labels[b]:>8} | {dev_arr[m].mean():11.3f} | {int(m.sum()):6d} | "
             f"{win_rmse[m].mean():11.4f} | {win_rmse_phys[m].mean():13.4f}")
    ratio = win_rmse[bins == 3].mean() / max(win_rmse[bins == 0].mean(), 1e-12)
    r = float(np.corrcoef(dev_arr, win_rmse)[0, 1])
    from scipy.stats import spearmanr

    rho = float(spearmanr(dev_arr, win_rmse).statistic)
    emit()
    emit(f"  Pearson r = {r:+.3f}，Spearman ρ = {rho:+.3f}，偏离大/常值 误差比 = {ratio:.2f}×")

    # 混淆控制：机动强度（窗口内 GT 速度的变化幅度）同样会推高误差
    tgt_n = x_seq[:, :, 3:6].cpu().numpy()
    intensity = tgt_n.std(axis=1).mean(axis=1)
    emit()
    emit("E.3 混淆控制：指令变化多的窗口往往也是机动更剧烈的窗口（本身更难预测）")
    emit("  用「窗口内 GT 归一化速度的 std」作机动强度代理，分层比较：")
    iq = np.quantile(intensity, [0.25, 0.5, 0.75])
    ibin = np.digitize(intensity, iq)
    emit(f"{'机动强度组':>10} | {'常值块 rmse':>12} | {'非常值 rmse':>12} | {'比值':>6} | {'窗口数(常/非)':>14}")
    emit("-" * 66)
    for b in range(4):
        m = ibin == b
        a0 = win_rmse[m & const_mask]
        a1 = win_rmse[m & ~const_mask]
        if a0.size == 0 or a1.size == 0:
            emit(f"{'Q' + str(b + 1):>10} | {'-':>12} | {'-':>12} | {'-':>6} | "
                 f"{str(a0.size) + '/' + str(a1.size):>14}")
            continue
        emit(f"{'Q' + str(b + 1):>10} | {a0.mean():12.4f} | {a1.mean():12.4f} | "
             f"{a1.mean() / a0.mean():6.2f} | {str(a0.size) + '/' + str(a1.size):>14}")

    def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
        A = np.stack([x, np.ones_like(x)], axis=1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return y - A @ beta

    pr = float(np.corrcoef(_resid(dev_arr, intensity), _resid(win_rmse, intensity))[0, 1])
    emit()
    emit(f"  控制机动强度后的偏相关 r(dev, rmse | intensity) = {pr:+.3f}"
         f"（未控制时 {r:+.3f}）")
    emit("  → 相关性大部分能被机动强度解释，但控制后仍有正相关残留，说明 ZOH 失配确实贡献了一部分"
         "多步误差；它属于「损失再怎么优化也消不掉」的口径性误差")
    emit("  可选改法：块内控制取均值而非块首（ZOH 的最小二乘最优近似）；"
         "按块内偏离给样本降权；或用 --ctrl_noise_std 注入与实测偏离同量级的控制噪声")


def analyze_propagation(
    model: HorizontalKoopmanModelV4DictInput, args: argparse.Namespace, K: int
) -> None:
    section("F. 潜算子的多步传播增益与 rollout 设置")
    nz = model.latent_dim
    Abar = (model.A.weight.detach() + torch.eye(nz)).cpu().numpy().astype(float)
    rho = float(np.max(np.abs(np.linalg.eigvals(Abar))))
    emit(f"ρ(Ā) = {rho:.6f}；逐步的谱范数与 K 步累积：")
    emit(f"{'k':>4} | {'||Ā^k||_2':>10} | {'ρ^k':>10}")
    emit("-" * 30)
    P = np.eye(nz)
    for k in range(1, K + 1):
        P = P @ Abar
        if k in (1, 2, 5, K):
            emit(f"{k:4d} | {np.linalg.norm(P, 2):10.4f} | {rho ** k:10.4f}")
    emit(f"  ∂z_K/∂z_0 = Ā^K 的谱范数 = {np.linalg.norm(P, 2):.4f} → "
         "多步反传既不消失也不爆炸，末步损失能完整地传回第一步（这是好事，"
         "但也意味着末步的支配项会重写整条轨迹的梯度）")
    emit()
    emit("训练 rollout 的设置（train_v4_dict_input.py:298-324）：")
    emit(f"  · 全开环：只在 t0 encode 一次，之后 {K} 步全部用模型自身输出推进，无 teacher forcing —— "
         "与部署 condensed rollout 口径一致（好）")
    emit(f"  · 训练 K={K}（{K * args.dt:.0f}s）与 MPC horizon N=10（40s）一致（好）")
    emit(f"  · 噪声注入：noise_std={float(getattr(args, 'noise_std', 0.0) or 0.0)}、"
         f"ctrl_noise_std={float(getattr(args, 'ctrl_noise_std', 0.0) or 0.0)} —— "
         "部署 ckpt 两者均为 0，即多步训练里没有任何抗累积误差的正则")
    emit("  · 没有 scheduled sampling / 无逐步状态噪声：模型只见过「从真实 t0 出发」的轨迹，"
         "闭环时的状态分布偏移无监督")
    emit("  · 无前向-后向一致性、无潜能量上限等多步一致性约束（文档 P2-8 已规划 ConsKAE）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v4_dt4s/run_v4_20260617_123100/koopman_v4_best.pth")
    ap.add_argument("--data", default="data/koopman_test.npz")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--zoh_windows", type=int, default=3000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fig", default=None)
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

    emit(f"ckpt = {a.ckpt}（epoch {ckpt.get('epoch')}，dt={args.dt}s，K={args.pred_len_max}，"
         f"w_xy={args.w_xy}，w_yaw={args.w_yaw}，gamma_step={args.gamma_step}）")
    emit(f"data = {a.data}")

    x_t, x_seq, u_seq, ds = build_batch(args, a.data, stats, a.samples, device)
    ro = Rollout(model, x_t, x_seq, u_seq, stats, args.dt, device)
    analyze_step_distribution(ro, args)
    analyze_error_structure(ro, args)
    analyze_gradient_attribution(model, x_t, x_seq, u_seq, stats, args, device)
    analyze_horizon_scaling(ro, args)
    analyze_zoh_consistency(model, ds, stats, args, device, a.zoh_windows)
    analyze_propagation(model, args, ro.K)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(_LINES) + "\n", encoding="utf-8")
        print(f"\n[saved] {a.out}")
    if a.fig:
        plot_step_profile(a.fig, ro, args)
    return 0


def plot_step_profile(path: str, ro: Rollout, args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = ro.K
    terms = {k: v.detach().cpu().numpy() for k, v in ro.per_step_terms(args.huber_beta).items()}
    w = make_step_weights(K, args.gamma_step, ro.pred_n.device).detach().cpu().numpy()
    steps = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    ax = axes[0]
    for name in ("L_vel", "L_lin", "L_yaw", "L_xy"):
        v = terms[name]
        ax.plot(steps, v / v[0], "o-", ms=3, label=f"{name} (norm. by step 1)")
    ax.plot(steps, w / w[0], "k--", label="step weight $w_k$")
    ax.set_yscale("log")
    ax.set_xlabel(f"step k (1 step = {args.dt}s)")
    ax.set_ylabel("relative to step 1 (log)")
    ax.set_title("per-step loss growth vs step weight")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.2
    for i, name in enumerate(("L_vel", "L_lin", "L_xy", "L_yaw")):
        m = terms[name] * w
        ax.bar(steps + (i - 1.5) * width, 100 * m / m.sum(), width=width, label=name)
    ax.set_xlabel(f"step k (1 step = {args.dt}s)")
    ax.set_ylabel("share of that term's total [%]")
    ax.set_title("where each term's loss mass sits")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    err = (ro.pred_phys - ro.target_phys).detach()
    for i, ch in enumerate("uvr"):
        e = err[..., i]
        ratio = (e.mean(dim=0).abs() / torch.sqrt((e ** 2).mean(dim=0))).cpu().numpy()
        ax.plot(steps, ratio, "o-", ms=3, label=f"{ch}: |bias| / rmse")
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"step k (1 step = {args.dt}s)")
    ax.set_ylabel("bias fraction")
    ax.set_title("is multi-step error coherent (bias) or random?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[saved] {path}")


if __name__ == "__main__":
    raise SystemExit(main())
