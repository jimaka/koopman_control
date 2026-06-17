#!/usr/bin/env python3
"""Reference implementation for v4 latent condensed prediction matrices (Gamma, Theta, xi)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.export_v4_encode_weights import load_v4_model  # noqa: E402


def precompute_prediction(nz: int, nu: int, n: int, A_bar: np.ndarray, B: np.ndarray, bias: np.ndarray):
    """Build Z = Gamma z0 + Theta U + xi_stack."""
    Gamma = np.zeros((nz * n, nz), dtype=np.float64)
    Theta = np.zeros((nz * n, nu * n), dtype=np.float64)
    xi_stack = np.zeros(nz * n, dtype=np.float64)

    A_pow_cache = {0: np.eye(nz)}
    for i in range(1, n + 1):
        A_pow_cache[i] = A_pow_cache[i - 1] @ A_bar

    for k in range(1, n + 1):
        Gamma[k * nz - nz : k * nz, :] = A_pow_cache[k]

        xi_k = np.zeros(nz)
        A_i = np.eye(nz)
        for _ in range(k):
            xi_k += A_i @ bias
            A_i = A_bar @ A_i
        xi_stack[k * nz - nz : k * nz] = xi_k

        for j in range(k):
            power = k - j - 1
            A_kj = A_pow_cache[power]
            Theta[k * nz - nz : k * nz, j * nu : (j + 1) * nu] = A_kj @ B

    return Gamma, Theta, xi_stack


def rollout_latent_ref(model, z0: np.ndarray, u_norm: np.ndarray) -> np.ndarray:
    """Step-by-step latent rollout in PyTorch for cross-check."""
    z = torch.tensor(z0, dtype=torch.float32)
    traj = [z.numpy().copy()]
    for k in range(u_norm.shape[0]):
        u = torch.tensor(u_norm[k], dtype=torch.float32)
        with torch.no_grad():
            z = model.latent_step(z.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
        traj.append(z.numpy().copy())
    return np.stack(traj, axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/koopman_v4_best.pth")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--out", default="eval_out/latent_qp_ref.npz")
    args = p.parse_args()

    model, _ = load_v4_model(args.ckpt)
    nz = model.latent_dim
    nu = model.control_dim
    n = args.horizon

    A_bar = (model.A.weight.detach().cpu().numpy() + np.eye(nz))
    bias = model.A.bias.detach().cpu().numpy()
    B = model.B.weight.detach().cpu().numpy()

    Gamma, Theta, xi_stack = precompute_prediction(nz, nu, n, A_bar, B, bias)

    rng = np.random.default_rng(0)
    z0 = rng.normal(size=nz).astype(np.float64)
    U = rng.normal(size=(nu * n,)).astype(np.float64)
    Z_condensed = Gamma @ z0 + Theta @ U + xi_stack

    u_norm = U.reshape(n, nu)
    z_step = rollout_latent_ref(model, z0.astype(np.float32), u_norm.astype(np.float32))
    Z_step = z_step[1:].reshape(-1)

    err = float(np.max(np.abs(Z_condensed - Z_step)))
    print(f"[condensed vs step] max_abs_err={err:.6e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        nz=nz,
        nu=nu,
        horizon=n,
        A_bar=A_bar,
        bias=bias,
        B=B,
        Gamma=Gamma,
        Theta=Theta,
        xi_stack=xi_stack,
        z0=z0,
        U=U,
        Z_condensed=Z_condensed,
        Z_step=Z_step,
        max_err=err,
    )
    print(f"[OK] wrote {out}")
    if err > 1e-4:
        print("[FAIL] condensed prediction mismatch")
        return 1
    print("[OK] condensed prediction matches step rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
