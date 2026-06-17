#!/usr/bin/env python3
"""验证 Tier-2 位姿线性化 (Phi) 与 decoder Jacobian 的正确性。

不依赖 checkpoint：用随机初始化的 v4 模型即可校验数学算法，
该算法与 C++ `pose_linearize.cpp` / `koopman_decoder.cpp` 一一对应。

校验项：
1. decoder 物理 Jacobian (含 dyn_std 缩放) vs torch.autograd。
2. 位姿灵敏度 Phi：pose(U+dU) - pose(U) ≈ Phi·dU（小扰动，误差应为 O(|dU|^2)）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.model_v4_dict_input import HorizontalKoopmanModelV4DictInput  # noqa: E402


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0))) + x * np.exp(-0.5 * x * x) * inv_sqrt2pi


def decoder_layers(model):
    """返回 [(W,b) | None for gelu] 序列，模仿 C++ decoder.layers。"""
    layers = []
    for layer in model.decoder_mlp:
        if isinstance(layer, torch.nn.Linear):
            layers.append(("linear", layer.weight.detach().numpy(), layer.bias.detach().numpy()))
        elif isinstance(layer, torch.nn.GELU):
            layers.append(("gelu", None, None))
    return layers


def decode_physical(layers, dyn_mean, dyn_std, z):
    h = z.copy()
    for t, w, b in layers:
        if t == "linear":
            h = w @ h + b
        else:
            h = gelu(h)
    return h * dyn_std + dyn_mean


def decode_jacobian_physical(layers, dyn_std, z):
    nz = z.shape[0]
    J = np.eye(nz)
    h = z.copy()
    for t, w, b in layers:
        if t == "linear":
            J = w @ J
            h = w @ h + b
        else:
            g = gelu_grad(h)
            J = g[:, None] * J
            h = gelu(h)
    return dyn_std[:, None] * J  # 3 x nz


def latent_matrices(model):
    nz = model.latent_dim
    Abar = (model.A.weight.detach() + torch.eye(nz)).numpy()
    bias = model.A.bias.detach().numpy()
    Bbar = model.B.weight.detach().numpy()
    return Abar, Bbar, bias


def rollout_nonlinear(model, layers, dyn_mean, dyn_std, z0, pose0, U, N, nu, dt):
    """与 C++ 一致的标称 rollout：返回 pose p_1..p_N (N x 3)。"""
    Abar, Bbar, bias = latent_matrices(model)
    z = z0.copy()
    x, y, yaw = pose0
    poses = []
    for m in range(1, N + 1):
        u = U[(m - 1) * nu:(m) * nu]
        z = Abar @ z + Bbar @ u + bias
        d = decode_physical(layers, dyn_mean, dyn_std, z)
        um, vm, rm = d[0], d[1], d[2]
        c, s = math.cos(yaw), math.sin(yaw)
        x = x + (um * c - vm * s) * dt
        y = y + (um * s + vm * c) * dt
        yaw = yaw + rm * dt
        poses.append([x, y, yaw])
    return np.array(poses)


def build_phi(model, layers, dyn_mean, dyn_std, z0, pose0, U, N, nu, dt):
    """镜像 C++ pose_linearize：返回 Phi (3N x nu*N)。"""
    Abar, Bbar, bias = latent_matrices(model)
    nz = model.latent_dim
    nvar = nu * N

    # 标称 Z (z_1..z_N) 与 Theta 行块。Theta 用递推构造。
    A_pow = [np.eye(nz)]
    for _ in range(N):
        A_pow.append(A_pow[-1] @ Abar)
    Theta = np.zeros((nz * N, nvar))
    for k in range(1, N + 1):
        for j in range(k):
            block = A_pow[k - j - 1] @ Bbar
            Theta[(k - 1) * nz:k * nz, j * nu:(j + 1) * nu] = block

    # 标称速度 d_m, Jacobian, V_m
    z = z0.copy()
    Z = []
    for m in range(1, N + 1):
        u = U[(m - 1) * nu:m * nu]
        z = Abar @ z + Bbar @ u + bias
        Z.append(z.copy())

    d = [None] * (N + 1)
    V = [None] * (N + 1)
    for m in range(1, N + 1):
        zk = Z[m - 1]
        d[m] = decode_physical(layers, dyn_mean, dyn_std, zk)
        Jp = decode_jacobian_physical(layers, dyn_std, zk)  # 3 x nz
        V[m] = Jp @ Theta[(m - 1) * nz:m * nz, :]  # 3 x nvar

    # 标称位姿
    p0 = np.zeros((N + 1, 3))
    p0[0] = pose0
    for m in range(1, N + 1):
        yaw_prev = p0[m - 1, 2]
        um, vm, rm = d[m]
        c, s = math.cos(yaw_prev), math.sin(yaw_prev)
        p0[m, 0] = p0[m - 1, 0] + (um * c - vm * s) * dt
        p0[m, 1] = p0[m - 1, 1] + (um * s + vm * c) * dt
        p0[m, 2] = yaw_prev + rm * dt

    Phi = np.zeros((3 * N, nvar))
    Sx = np.zeros(nvar)
    Sy = np.zeros(nvar)
    Sp = np.zeros(nvar)
    for m in range(1, N + 1):
        yaw_prev = p0[m - 1, 2]
        um, vm, _ = d[m]
        c, s = math.cos(yaw_prev), math.sin(yaw_prev)
        dxdpsi = (-um * s - vm * c) * dt
        dydpsi = (um * c - vm * s) * dt
        Vu, Vv, Vr = V[m][0], V[m][1], V[m][2]
        nSx = Sx + dt * (c * Vu - s * Vv) + dxdpsi * Sp
        nSy = Sy + dt * (s * Vu + c * Vv) + dydpsi * Sp
        nSp = Sp + dt * Vr
        Phi[(m - 1) * 3 + 0] = nSx
        Phi[(m - 1) * 3 + 1] = nSy
        Phi[(m - 1) * 3 + 2] = nSp
        Sx, Sy, Sp = nSx, nSy, nSp
    return Phi


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    model = HorizontalKoopmanModelV4DictInput(hidden_dim=32, clamp_pif=5.0, encoder_arch="conv")
    model.eval()

    dyn_mean = np.array([2.25, 0.0, 0.0], dtype=np.float64)
    dyn_std = np.array([1.53, 0.22, 0.015], dtype=np.float64)
    layers = decoder_layers(model)

    # --- 1. decoder Jacobian vs autograd ---
    nz = model.latent_dim
    z_t = torch.randn(nz, dtype=torch.float32, requires_grad=True)
    dyn_std_t = torch.tensor(dyn_std, dtype=torch.float32)
    dyn_mean_t = torch.tensor(dyn_mean, dtype=torch.float32)

    def dec_phys(z):
        return model.reconstruct_state(z.unsqueeze(0)).squeeze(0) * dyn_std_t + dyn_mean_t

    J_auto = torch.autograd.functional.jacobian(dec_phys, z_t).detach().numpy()
    J_ours = decode_jacobian_physical(layers, dyn_std, z_t.detach().numpy().astype(np.float64))
    jerr = float(np.max(np.abs(J_auto - J_ours)))
    print(f"[decoder Jacobian] max_abs_err vs autograd = {jerr:.3e}")

    # --- 2. Phi 线性化精度 ---
    N, nu, dt = 20, 4, 1.0
    dyn0_n = np.random.randn(3).astype(np.float64) * 0.3
    z0 = model.encode(torch.tensor(dyn0_n[None], dtype=torch.float32)).squeeze(0).detach().numpy().astype(np.float64)
    pose0 = np.array([0.0, 0.0, 0.0])
    U0 = (np.random.randn(nu * N) * 0.2).astype(np.float64)

    Phi = build_phi(model, layers, dyn_mean, dyn_std, z0, pose0, U0, N, nu, dt)
    p_base = rollout_nonlinear(model, layers, dyn_mean, dyn_std, z0, pose0, U0, N, nu, dt)

    # 收敛阶检查：误差应随 |dU| 二阶下降
    rel_errs = []
    for scale in [1e-2, 5e-3, 2.5e-3]:
        dU = (np.random.RandomState(7).randn(nu * N) * scale).astype(np.float64)
        p_true = rollout_nonlinear(model, layers, dyn_mean, dyn_std, z0, pose0, U0 + dU, N, nu, dt)
        true_delta = (p_true - p_base).reshape(-1)
        lin_delta = Phi @ dU
        err = np.max(np.abs(true_delta - lin_delta))
        denom = np.max(np.abs(true_delta)) + 1e-9
        rel_errs.append(err / denom)
        print(f"[Phi] |dU|~{scale:.1e}  abs_lin_err={err:.3e}  rel={err/denom:.3e}")

    # 二阶收敛：scale 减半，误差应约降到 1/4
    ratio = rel_errs[1] / (rel_errs[2] + 1e-12)
    print(f"[Phi] second-order ratio (expect ~2-4) = {ratio:.2f}")

    ok = jerr < 1e-4 and rel_errs[-1] < 1e-3
    if not ok:
        print("[FAIL] linearization checks did not pass")
        return 1
    print("[OK] decoder Jacobian + pose linearization verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
