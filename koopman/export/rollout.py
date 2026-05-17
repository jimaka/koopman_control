"""可导出的 Koopman rollout（TorchScript / ONNX 共用）。"""
from __future__ import annotations

import torch
import torch.nn as nn

TRACED_HORIZON = 20


class KoopmanRollout(nn.Module):
    """归一化潜空间动力学 + 船体坐标系欧拉积分。"""

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

    def forward(
        self,
        state0: torch.Tensor,
        u_seq: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """state0: (6,)  u_seq: (H,4)  dt: scalar tensor  -> states (H+1, 6)"""
        H = u_seq.size(0)
        dt_s = dt.reshape(()) if isinstance(dt, torch.Tensor) else torch.tensor(dt, dtype=torch.float32)

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
            x = x + (u * torch.cos(yaw) - v * torch.sin(yaw)) * dt_s
            y = y + (u * torch.sin(yaw) + v * torch.cos(yaw)) * dt_s
            yaw = yaw + r * dt_s
            out.append(torch.stack([x, y, yaw, u, v, r]))
        return torch.stack(out, dim=0)
