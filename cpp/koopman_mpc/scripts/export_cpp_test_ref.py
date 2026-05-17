#!/usr/bin/env python3
"""导出 C++ 测试用参考航迹（文本格式）与 rollout 对照张量。"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SCRIPTS_DIR = os.path.dirname(__file__)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from mpc_koopman import segment_to_state_ctrl
import eval_koopman as ek
from export_torchscript import KoopmanRollout


def main() -> None:
    out_dir = os.path.join(REPO_ROOT, "cpp/koopman_mpc/weights")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = os.path.join(REPO_ROOT, "checkpoints/koopman_v3a_best.pth")
    data = os.path.join(REPO_ROOT, "koopman_test.npz")
    segment = 0
    max_len = 120

    raw = np.load(data, allow_pickle=True)["datas"]
    ref_state, ref_ctrl = segment_to_state_ctrl(raw[segment])
    ref_state = ref_state[:max_len]
    ref_ctrl = ref_ctrl[:max_len]

    txt_path = os.path.join(out_dir, "cpp_test_ref.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        for row in ref_state:
            f.write("state " + " ".join(f"{x:.8g}" for x in row) + "\n")
        for row in ref_ctrl:
            f.write("ctrl " + " ".join(f"{x:.8g}" for x in row) + "\n")
    print(f"Wrote {txt_path} ({len(ref_state)} states)")

    device = torch.device("cpu")
    model, stats = ek.load_model_from_ckpt(ckpt, device)
    model.eval()
    rollout = KoopmanRollout(model, stats)
    s0 = torch.tensor(ref_state[0], dtype=torch.float32)
    H = 20  # 与 TorchScript trace 固定 horizon 一致
    u_seq = torch.tensor(ref_ctrl[:H], dtype=torch.float32)
    with torch.no_grad():
        states = rollout(s0, u_seq, 0.1)
    np.savez_compressed(
        os.path.join(out_dir, "rollout_check.npz"),
        state0=ref_state[0],
        u_seq=u_seq.numpy(),
        states=states.numpy(),
    )
    print("Wrote rollout_check.npz")


if __name__ == "__main__":
    main()
