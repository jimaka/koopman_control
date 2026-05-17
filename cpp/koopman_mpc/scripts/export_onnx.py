#!/usr/bin/env python3
"""从 .pth checkpoint 导出 ONNX rollout，并与 PyTorch 前向对照。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from koopman import evalkit as ek
from koopman import paths as P
from koopman.export import KoopmanRollout, TRACED_HORIZON


def export_onnx(rollout: torch.nn.Module, out_path: str, opset: int = 18) -> None:
    rollout.eval()
    s0 = torch.zeros(6, dtype=torch.float32)
    u = torch.zeros(TRACED_HORIZON, 4, dtype=torch.float32)
    dt = torch.tensor(0.1, dtype=torch.float32)
    torch.onnx.export(
        rollout,
        (s0, u, dt),
        out_path,
        input_names=["state0", "u_seq", "dt"],
        output_names=["states"],
        opset_version=opset,
        dynamo=False,
    )


def verify_onnx_vs_pytorch(
    rollout: torch.nn.Module,
    onnx_path: str,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    n_random: int = 8,
) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rollout.eval()
    max_err = 0.0
    rng = np.random.default_rng(42)

    cases = [
        (torch.zeros(6), torch.zeros(TRACED_HORIZON, 4)),
    ]
    for _ in range(n_random):
        s0 = torch.tensor(rng.normal(0, 1, size=6).astype(np.float32))
        s0[2] = float(rng.uniform(-0.5, 0.5))
        u = torch.tensor(rng.uniform(-5, 5, size=(TRACED_HORIZON, 4)).astype(np.float32))
        cases.append((s0, u))

    with torch.no_grad():
        for s0, u in cases:
            dt = torch.tensor(0.1, dtype=torch.float32)
            pt_out = rollout(s0, u, dt).numpy()
            ort_out = sess.run(
                None,
                {
                    "state0": s0.numpy(),
                    "u_seq": u.numpy(),
                    "dt": np.array(0.1, dtype=np.float32),
                },
            )[0]
            err = float(np.max(np.abs(pt_out - ort_out)))
            max_err = max(max_err, err)

    if max_err > atol + rtol:
        raise RuntimeError(
            f"ONNX vs PyTorch max_abs_err={max_err:.6e} exceeds tol atol={atol} rtol={rtol}"
        )
    return max_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(P.CKPT_V3A_BEST))
    parser.add_argument("--out_dir", default=str(P.CPP_MPC_DIR / "weights"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO_ROOT, args.ckpt)

    device = torch.device("cpu")
    model, stats = ek.load_model_from_ckpt(ckpt_path, device)
    model.eval()
    rollout = KoopmanRollout(model, stats).cpu().eval()

    onnx_path = os.path.join(args.out_dir, "koopman_rollout.onnx")
    export_onnx(rollout, onnx_path, opset=args.opset)
    print(f"Saved ONNX -> {onnx_path}")

    max_err = verify_onnx_vs_pytorch(rollout, onnx_path, atol=args.atol)
    print(f"ONNX vs PyTorch max_abs_err={max_err:.6e} OK")

    meta_path = os.path.join(args.out_dir, "model_meta.json")
    meta = {
        "ckpt": args.ckpt,
        "onnx": "koopman_rollout.onnx",
        "format": "onnx",
        "dt": 0.1,
        "horizon_default": TRACED_HORIZON,
        "onnx_verify_max_abs_err": max_err,
        "u_min": [-100.0, -35.0, -100.0, -35.0],
        "u_max": [100.0, 35.0, 100.0, 35.0],
        "dyn_mean": stats["state_mean"][3:6].tolist(),
        "dyn_std": stats["state_std"][3:6].tolist(),
        "ctrl_mean": stats["ctrl_mean"].tolist(),
        "ctrl_std": stats["ctrl_std"].tolist(),
    }
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            prev = json.load(f)
        meta = {**prev, **meta}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated {meta_path}")


if __name__ == "__main__":
    main()
