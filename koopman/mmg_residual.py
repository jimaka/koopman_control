"""MMG 基线 + 残差 MLP 混合模型（docs/MMG残差MLP建模技术方案.md §3.4）。

    dyn_{t+1} = ∫ [ f_MMG(dyn, cmd; θ冻结) + g_MLP(归一化(dyn, cmd); φ) ] dt

残差 MLP 输出**物理加速度残差**，输出层零初始化——训练起点严格等于纯 MMG
基线，训练过程单调改善。输入做 z-score 归一化（统计量来自训练集，与 v4 同口径）。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn

from koopman.mmg_model import MmgModel


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class ResidualMLP(nn.Module):
    """in_dim → hidden → n_blocks × ResidualBlock → out_dim（末层零初始化）。"""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, n_blocks: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([ResidualBlock(hidden, hidden) for _ in range(int(n_blocks))])
        self.out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.inp(x))
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class MmgResidualModel(nn.Module):
    """MMG 基线 + 残差 MLP。状态 (u,v,r) 与控制均为物理量接口。"""

    def __init__(
        self,
        mmg: MmgModel,
        stats: Dict[str, np.ndarray],
        hidden: int = 64,
        n_blocks: int = 2,
        freeze_mmg: bool = True,
    ) -> None:
        super().__init__()
        self.mmg = mmg
        if freeze_mmg:
            for p in self.mmg.parameters():
                p.requires_grad_(False)
        self.net = ResidualMLP(3 + 4, 3, hidden=hidden, n_blocks=n_blocks)
        self.register_buffer("dyn_mean", torch.as_tensor(np.asarray(stats["state_mean"][3:6]), dtype=torch.float32))
        self.register_buffer("dyn_std", torch.as_tensor(np.asarray(stats["state_std"][3:6]), dtype=torch.float32))
        self.register_buffer("ctrl_mean", torch.as_tensor(np.asarray(stats["ctrl_mean"]), dtype=torch.float32))
        self.register_buffer("ctrl_std", torch.as_tensor(np.asarray(stats["ctrl_std"]), dtype=torch.float32))

    def accel(self, dyn: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([
            (dyn - self.dyn_mean) / self.dyn_std,
            (ctrl - self.ctrl_mean) / self.ctrl_std,
        ], dim=-1)
        return self.mmg.accel(dyn, ctrl) + self.net(feat)

    def step_phys(self, dyn: torch.Tensor, ctrl: torch.Tensor, dt: float) -> torch.Tensor:
        """与 MmgModel.step_phys 同策略：按 sub_dt 细分 Euler。"""
        n = max(1, int(round(float(dt) / self.mmg.sub_dt)))
        h = float(dt) / n
        x = dyn
        for _ in range(n):
            x = x + h * self.accel(x, ctrl)
        return x

    @staticmethod
    def load_from_ckpt(ckpt: Dict, device: Optional[torch.device] = None) -> "MmgResidualModel":
        """从 train_mmg_residual.py 保存的 ckpt dict 重建模型。"""
        args = ckpt.get("args", {}) or {}
        stats = ckpt["stats"]
        theta = ckpt["mmg_theta"]
        mmg = MmgModel(theta=theta, trainable=False)
        model = MmgResidualModel(
            mmg, stats,
            hidden=int(args.get("hidden", 64)),
            n_blocks=int(args.get("n_blocks", 2)),
        )
        sd = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
        model.load_state_dict(sd)
        if device is not None:
            model = model.to(device)
        return model.eval()
