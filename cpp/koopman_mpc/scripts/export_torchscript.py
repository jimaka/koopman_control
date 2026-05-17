#!/usr/bin/env python3
"""导出 Koopman v3 TorchScript 与参考轨迹，供 C++ MPC 加载与数值对照。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import eval_koopman as ek
from mpc_koopman import KoopmanMPC, MPCConfig, segment_to_state_ctrl


class KoopmanRollout(nn.Module):
    """可 TorchScript 的 rollout（归一化动力学 + 欧拉积分）。"""

    def __init__(self, model: nn.Module, stats: dict) -> None:
        super().__init__()
        self.model = model
        sm = torch.tensor(stats["state_mean"], dtype=torch.float32)
        ss = torch.tensor(stats["state_std"], dtype=torch.float32)
        cm = torch.tensor(stats["ctrl_mean"], dtype=torch.float32)
        cs = torch.tensor(stats["ctrl_std"], dtype=torch.float32)
        self.register_buffer("dyn_mean", sm[3:6].clone())
        self.register_buffer("dyn_std", ss[3:6].clone())
        self.register_buffer("ctrl_mean", cm.clone())
        self.register_buffer("ctrl_std", cs.clone())

    def forward(self, state0: torch.Tensor, u_seq: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """state0: (6,)  u_seq: (H,4)  dt: scalar tensor  -> states (H+1, 6)"""
        H = u_seq.size(0)
        dt_f = float(dt.item()) if isinstance(dt, torch.Tensor) else float(dt)
        x = state0[0]
        y = state0[1]
        yaw = state0[2]
        dyn = state0[3:6]
        dyn_n = (dyn - self.dyn_mean) / self.dyn_std
        z = self.model.encode(dyn_n.unsqueeze(0)).squeeze(0)

        out = [state0]
        for k in range(H):
            u_n = (u_seq[k] - self.ctrl_mean) / self.ctrl_std
            z = self.model.latent_step(z.unsqueeze(0), u_n.unsqueeze(0)).squeeze(0)
            dyn = self.model.reconstruct_state(z.unsqueeze(0)).squeeze(0)
            dyn = dyn * self.dyn_std + self.dyn_mean
            u, v, r = dyn[0], dyn[1], dyn[2]
            x = x + (u * torch.cos(yaw) - v * torch.sin(yaw)) * dt_f
            y = y + (u * torch.sin(yaw) + v * torch.cos(yaw)) * dt_f
            yaw = yaw + r * dt_f
            out.append(torch.stack([x, y, yaw, u, v, r]))
        return torch.stack(out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/koopman_v3a_best.pth")
    parser.add_argument("--out_dir", default="cpp/koopman_mpc/weights")
    parser.add_argument("--data", default="koopman_test.npz")
    parser.add_argument("--segment", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu")
    model, stats = ek.load_model_from_ckpt(
        os.path.join(REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt,
        device,
    )
    model.eval()

    rollout = KoopmanRollout(model, stats).cpu().eval()
    s0_ex = torch.zeros(6, dtype=torch.float32)
    u_ex = torch.zeros(20, 4, dtype=torch.float32)
    with torch.no_grad():
        dt_ex = torch.tensor(0.1, dtype=torch.float32)
        scripted = torch.jit.trace(rollout, (s0_ex, u_ex, dt_ex), strict=False)
    ts_path = os.path.join(args.out_dir, "koopman_rollout.pt")
    scripted.save(ts_path)
    print(f"Saved TorchScript -> {ts_path}")

    meta = {
        "ckpt": args.ckpt,
        "dt": 0.1,
        "horizon_default": 20,
        "u_min": [-100.0, -35.0, -100.0, -35.0],
        "u_max": [100.0, 35.0, 100.0, 35.0],
        "dyn_mean": stats["state_mean"][3:6].tolist(),
        "dyn_std": stats["state_std"][3:6].tolist(),
        "ctrl_mean": stats["ctrl_mean"].tolist(),
        "ctrl_std": stats["ctrl_std"].tolist(),
    }
    with open(os.path.join(args.out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Python MPC 参考结果
    cfg = MPCConfig(horizon=20, opt_iters=25, device="cpu")
    ckpt_path = os.path.join(REPO_ROOT, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    mpc = KoopmanMPC.from_checkpoint(ckpt_path, cfg)
    raw = np.load(os.path.join(REPO_ROOT, args.data), allow_pickle=True)["datas"]
    ref_state, ref_ctrl = segment_to_state_ctrl(raw[args.segment])
    traj = mpc.simulate(ref_state[0], ref_state, ref_ctrl=ref_ctrl, max_steps=args.steps)

    ref_npz = os.path.join(args.out_dir, "python_mpc_ref.npz")
    np.savez_compressed(
        ref_npz,
        state=traj.state,
        control=traj.control,
        ref_state=traj.ref_state,
        t=traj.t,
    )
    print(f"Saved Python MPC reference -> {ref_npz}")


if __name__ == "__main__":
    main()
