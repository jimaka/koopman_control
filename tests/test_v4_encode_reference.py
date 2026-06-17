#!/usr/bin/env python3
"""Verify C++-equivalent encode forward matches PyTorch."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.export_v4_encode_weights import load_v4_model  # noqa: E402


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + torch.erf(torch.tensor(x, dtype=torch.float32) / math.sqrt(2.0)).numpy())


def compute_atoms_16(u: float, v: float, r: float, clamp_pif: float) -> np.ndarray:
    abs_u, abs_v, abs_r = abs(u), abs(v), abs(r)
    atoms = np.array(
        [
            u * abs_u,
            v * abs_v,
            r * abs_r,
            v * r,
            u * r,
            u * v * r,
            u * u * r,
            v * v * r,
            u * r * r,
            v * r * r,
            u * abs_v * v,
            v * abs_u * u,
            r * abs_u * u,
            r * abs_v * v,
            u * abs_u * u,
            v * abs_v * v,
        ],
        dtype=np.float32,
    )
    if clamp_pif > 0:
        atoms = np.clip(atoms, -clamp_pif, clamp_pif)
    return atoms


def linear(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return w @ x + b


def encode_cpp_equiv(model, dyn_norm: np.ndarray) -> np.ndarray:
    clamp_pif = float(model.clamp_pif)
    atoms = compute_atoms_16(*dyn_norm.tolist(), clamp_pif)
    h = atoms.copy()
    for layer in model.encoder_mlp:
        if hasattr(layer, "fc"):
            identity = h if isinstance(layer.shortcut, nn.Identity) else linear(
                layer.shortcut.weight.detach().cpu().numpy(),
                layer.shortcut.bias.detach().cpu().numpy(),
                h,
            )
            out = linear(layer.fc.weight.detach().cpu().numpy(), layer.fc.bias.detach().cpu().numpy(), h)
            conv_w = layer.conv.weight.detach().cpu().numpy()[:, 0, 1]
            conv_b = layer.conv.bias.detach().cpu().numpy()
            out = gelu(out * conv_w + conv_b)
            h = out + identity
        elif isinstance(layer, nn.Linear):
            hidden = linear(layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy(), h)
            return np.concatenate([atoms, hidden], axis=0)
    raise RuntimeError("encoder tail not found")


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/koopman_v4_best.pth")
    args = p.parse_args()

    model, stats = load_v4_model(args.ckpt)
    dyn_mean = np.asarray(stats["state_mean"][3:6], dtype=np.float32)
    dyn_std = np.asarray(stats["state_std"][3:6], dtype=np.float32)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(32):
        dyn = rng.normal(size=3).astype(np.float32) * dyn_std + dyn_mean
        dyn_n = (dyn - dyn_mean) / dyn_std
        with torch.no_grad():
            z_pt = model.encode(torch.tensor(dyn_n[None, :], dtype=torch.float32)).squeeze(0).numpy()
        z_cpp = encode_cpp_equiv(model, dyn_n)
        max_err = max(max_err, float(np.max(np.abs(z_pt - z_cpp))))

    print(f"[encode cpp-equiv] max_abs_err={max_err:.6e}")
    if max_err > 1e-4:
        return 1
    print("[OK] C++ encode algorithm matches PyTorch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
