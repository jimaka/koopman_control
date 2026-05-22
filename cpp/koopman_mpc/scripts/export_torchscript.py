#!/usr/bin/env python3
"""导出 Koopman TorchScript（可选，与 ONNX 共用 rollout）。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from koopman import evalkit as ek
from koopman import paths as P
from koopman.export import KoopmanRollout, TRACED_HORIZON


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(P.CKPT_V3A_BEST))
    parser.add_argument("--out_dir", default=str(P.CPP_MPC_DIR / "weights"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO_ROOT, args.ckpt)
    model, stats = ek.load_model_from_ckpt(ckpt_path, device)
    model.eval()

    rollout = KoopmanRollout(model, stats).cpu().eval()
    s0_ex = torch.zeros(6, dtype=torch.float32)
    u_ex = torch.zeros(TRACED_HORIZON, 4, dtype=torch.float32)
    with torch.no_grad():
        dt_ex = torch.tensor(0.1, dtype=torch.float32)
        scripted = torch.jit.trace(rollout, (s0_ex, u_ex, dt_ex), strict=False)
    ts_path = os.path.join(args.out_dir, "koopman_rollout.pt")
    scripted.save(ts_path)
    print(f"Saved TorchScript -> {ts_path}")

    meta_path = os.path.join(args.out_dir, "model_meta.json")
    meta = {
        "ckpt": args.ckpt,
        "torchscript": "koopman_rollout.pt",
        "dt": 0.1,
        "horizon_default": TRACED_HORIZON,
    }
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = {**json.load(f), **meta}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
