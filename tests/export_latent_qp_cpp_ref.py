#!/usr/bin/env python3
"""Export reference vectors for C++ verify_latent_qp."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.export_v4_encode_weights import load_v4_model  # noqa: E402
from tests.test_latent_qp_matrices import precompute_prediction  # noqa: E402


def write_vec(path: Path, v: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{v.size}\n")
        for x in v.reshape(-1):
            f.write(f"{float(x):.9g}\n")


def main() -> int:
    import argparse
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="checkpoints/koopman_v4_best.pth")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--out_dir", default="eval_out/latent_qp_cpp_ref")
    args = p.parse_args()

    horizon = int(args.horizon)
    model, _ = load_v4_model(args.ckpt)
    nz = model.latent_dim
    nu = model.control_dim

    A_bar = (model.A.weight.detach().cpu().numpy() + np.eye(nz))
    bias = model.A.bias.detach().cpu().numpy()
    B = model.B.weight.detach().cpu().numpy()
    _, Theta, _ = precompute_prediction(nz, nu, horizon, A_bar, B, bias)

    rng = np.random.default_rng(42)
    z0 = rng.normal(size=nz).astype(np.float32)
    U = rng.normal(size=(horizon * nu,)).astype(np.float32)
    Z_ref = (Theta @ U).astype(np.float32)
    # include free response for full predict test
    Gamma, _, xi = precompute_prediction(nz, nu, horizon, A_bar, B, bias)
    Z_ref = (Gamma @ z0 + Theta @ U + xi).astype(np.float32)

    dyn = np.array([1.5, 0.02, 0.01], dtype=np.float32)

    out = Path(args.out_dir)
    write_vec(out / "z0.txt", z0)
    write_vec(out / "U.txt", U)
    write_vec(out / "Z_ref.txt", Z_ref)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    dyn_mean = np.asarray(stats["state_mean"][3:6], dtype=np.float32)
    dyn_std = np.asarray(stats["state_std"][3:6], dtype=np.float32)
    dyn_norm = (dyn - dyn_mean) / dyn_std
    write_vec(out / "dyn_norm.txt", dyn_norm)

    with torch.no_grad():
        z_enc = model.encode(torch.tensor(dyn_norm[None, :], dtype=torch.float32)).squeeze(0).numpy()
    write_vec(out / "z_encode_ref.txt", z_enc.astype(np.float32))

    print(f"[OK] wrote C++ ref vectors to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
