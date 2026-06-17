#!/usr/bin/env python3
"""导出 v4 C++ MPC 测试用参考航迹与 rollout 对照张量（dt=1s, H=20）。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from koopman import evalkit as ek  # noqa: E402
from koopman import paths as P  # noqa: E402
from koopman.export import KoopmanRollout, TRACED_HORIZON_V4  # noqa: E402
from koopman.mpc import segment_to_state_ctrl  # noqa: E402
from koopman.paths import setup_repo  # noqa: E402
from new_v4_dict_input.export_v4_onnx import load_v4_model  # noqa: E402

setup_repo()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(P.CKPT_DIR / "koopman_v4_best.pth"))
    parser.add_argument(
        "--out_dir",
        default=os.environ.get("KOOPMAN_WEIGHTS_DIR", str(P.CPP_MPC_DIR / "weights")),
    )
    parser.add_argument("--horizon", type=int, default=TRACED_HORIZON_V4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--data_dt", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=450, help="参考航迹长度（需 >= horizon + steps）")
    args = parser.parse_args()

    model_stride = ek.model_stride_from_dt(args.dt, args.data_dt)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(REPO_ROOT, args.ckpt)
    data = str(P.TEST)
    segment = 0

    raw = np.load(data, allow_pickle=True)["datas"]
    ref_state, ref_ctrl = segment_to_state_ctrl(raw[segment])
    ref_state = ref_state[: args.max_len]
    ref_ctrl = ref_ctrl[: args.max_len]

    txt_path = os.path.join(args.out_dir, "cpp_test_ref.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        for row in ref_state:
            f.write("state " + " ".join(f"{x:.8g}" for x in row) + "\n")
        for row in ref_ctrl:
            f.write("ctrl " + " ".join(f"{x:.8g}" for x in row) + "\n")
    print(f"Wrote {txt_path} ({len(ref_state)} states)")

    device = torch.device("cpu")
    model, stats = load_v4_model(ckpt, device)
    model.eval()
    rollout = KoopmanRollout(model, stats)
    s0 = torch.tensor(ref_state[0], dtype=torch.float32)
    u_seq = torch.tensor(ref_ctrl[: args.horizon * model_stride : model_stride], dtype=torch.float32)
    with torch.no_grad():
        states = rollout(s0, u_seq, torch.tensor(float(args.dt), dtype=torch.float32))
    np.savez_compressed(
        os.path.join(args.out_dir, "rollout_check.npz"),
        state0=ref_state[0],
        u_seq=u_seq.numpy(),
        states=states.numpy(),
        dt=np.float32(args.dt),
    )
    print(f"Wrote rollout_check.npz (horizon={args.horizon})")


if __name__ == "__main__":
    main()
