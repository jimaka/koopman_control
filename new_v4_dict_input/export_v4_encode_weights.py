#!/usr/bin/env python3
"""Export v4 encoder weights + Koopman matrices for C++ latent QP-MPC."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from new_v4_dict_input.model_v4_dict_input import (  # noqa: E402
    FEATURE_DICT_ATOMS_16,
    HorizontalKoopmanModelV4DictInput,
)


def _linear_to_dict(m: nn.Linear) -> dict:
    return {
        "weight": m.weight.detach().cpu().numpy().tolist(),
        "bias": m.bias.detach().cpu().numpy().tolist(),
    }


def _export_res_mlp(encoder_mlp: nn.Sequential) -> list:
    """Export res_mlp [16,64,64,32] block weights in forward order."""
    blocks = []
    for layer in encoder_mlp:
        if hasattr(layer, "fc"):
            blk = {
                "type": "residual_conv_block",
                "fc": _linear_to_dict(layer.fc),
                "conv_weight": layer.conv.weight.detach().cpu().numpy().tolist(),
                "conv_bias": layer.conv.bias.detach().cpu().numpy().tolist(),
                "shortcut": None
                if isinstance(layer.shortcut, nn.Identity)
                else _linear_to_dict(layer.shortcut),
            }
            blocks.append(blk)
        elif isinstance(layer, nn.Linear):
            blocks.append({"type": "linear", **_linear_to_dict(layer)})
        else:
            raise TypeError(f"unexpected layer {type(layer)}")
    return blocks


def load_v4_model(ckpt_path: str) -> tuple[HorizontalKoopmanModelV4DictInput, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    args_d = ckpt.get("args", {}) or {}
    model = HorizontalKoopmanModelV4DictInput(
        hidden_dim=int(args_d.get("hidden_dim", 32)),
        clamp_pif=float(args_d.get("clamp_pif", 5.0)),
        encoder_arch=str(args_d.get("encoder_arch", "conv")),
    )
    model.load_state_dict(sd)
    model.eval()
    return model, stats


def main() -> int:
    p = argparse.ArgumentParser(description="Export v4 encode weights for C++ QP-MPC")
    p.add_argument("--ckpt", default="checkpoints/koopman_v4_best.pth")
    p.add_argument(
        "--out",
        default="cpp/koopman_mpc/weights/koopman_v4_latent.json",
    )
    p.add_argument(
        "--out-yaml",
        default="cpp/koopman_mpc/weights/koopman_v4_latent.yaml",
    )
    p.add_argument("--horizon", type=int, default=20)
    args = p.parse_args()

    model, stats = load_v4_model(args.ckpt)
    nz = int(model.latent_dim)
    nu = int(model.control_dim)
    n = int(args.horizon)

    A_w = (model.A.weight.detach().cpu() + torch.eye(nz)).numpy()
    A_b = model.A.bias.detach().cpu().numpy()
    B_w = model.B.weight.detach().cpu().numpy()

    payload = {
        "model_class": "HorizontalKoopmanModelV4DictInput",
        "latent_dim": nz,
        "control_dim": nu,
        "dict_dim": int(model.dict_dim),
        "hidden_dim": int(model.hidden_dim),
        "clamp_pif": float(model.clamp_pif),
        "encoder_arch": str(getattr(model, "encoder_arch", "conv")),
        "feature_dict_atoms": list(FEATURE_DICT_ATOMS_16),
        "horizon_default": n,
        "normalization": {
            "dyn_mean": np.asarray(stats["state_mean"][3:6], dtype=float).tolist(),
            "dyn_std": np.asarray(stats["state_std"][3:6], dtype=float).tolist(),
            "ctrl_mean": np.asarray(stats["ctrl_mean"], dtype=float).tolist(),
            "ctrl_std": np.asarray(stats["ctrl_std"], dtype=float).tolist(),
        },
        "koopman": {
            "A_bar": A_w.tolist(),
            "bias": A_b.tolist(),
            "B": B_w.tolist(),
        },
        "encoder": {
            "arch": str(getattr(model, "encoder_arch", "conv")),
            "layers": _export_res_mlp(model.encoder_mlp),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] wrote {out_path} (nz={nz}, horizon={n})")

    yaml_path = Path(args.out_yaml)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    print(f"[OK] wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
