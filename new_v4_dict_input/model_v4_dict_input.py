"""Deep-Koopman v4: use 16 dictionary atoms as encoder input.

设计目标：
1) 保持与既有训练/评估流程的接口兼容：
   - encode(x_dyn_norm)
   - latent_step(z, u_norm)
   - reconstruct_state(z) -> x_dyn_norm
   - spectral_radius() 由 BaseKoopmanModel 提供
2) 与 v3 的差别：encoder 不再直接吃 [u, v, r]，而是吃固定 16 阶物理字典。
3) 因为 latent 中不再包含 state 直通分量，新增 decoder 从 latent 回归 [u, v, r]。
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from koopman.model_v1_v2 import BaseKoopmanModel, res_mlp


FEATURE_DICT_ATOMS_16: List[str] = [
    "u_abs_u", "v_abs_v", "r_abs_r", "v_times_r", "u_times_r",
    "uvr", "u2r", "v2r", "ur2", "vr2",
    "u_vabs_v", "v_uabs_u", "r_uabs_u", "r_vabs_v",
    "uuu", "vvv",
]


def _compute_atoms_16(u: torch.Tensor, v: torch.Tensor, r: torch.Tensor) -> List[torch.Tensor]:
    abs_u = torch.abs(u)
    abs_v = torch.abs(v)
    abs_r = torch.abs(r)
    return [
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
    ]


class _ResidualMLPEncoder(nn.Module):
    """干净的残差 MLP 编码器：dict16 -> hidden。

    相对历史 ``res_mlp``（ResidualConvBlock 内 Conv1d 退化为逐通道缩放），
    这里用标准 Linear+GELU，并在等宽隐藏层之间加残差连接，表达力更直接。
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x))
        h = h + self.act(self.fc1(h))   # 等宽残差块 1
        h = h + self.act(self.fc2(h))   # 等宽残差块 2
        return self.out_proj(h)


class HorizontalKoopmanModelV4DictInput(BaseKoopmanModel):
    """v4 模型：z = [dict16, hidden]，dict16 作为模型主输入。"""

    def __init__(
        self,
        state_dim: int = 3,
        control_dim: int = 4,
        dict_dim: int = 16,
        hidden_dim: int = 32,
        clamp_pif: float = 5.0,
        encoder_arch: str = "conv",
    ) -> None:
        super().__init__()
        if dict_dim != 16:
            raise ValueError(f"当前实现固定 dict_dim=16，收到 {dict_dim}")
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dict_dim = dict_dim
        self.hidden_dim = hidden_dim
        self.clamp_pif = float(clamp_pif)
        self.latent_dim = self.dict_dim + self.hidden_dim
        if encoder_arch not in ("conv", "mlp"):
            raise ValueError(f"encoder_arch 仅支持 'conv'/'mlp'，收到 {encoder_arch}")
        self.encoder_arch = encoder_arch

        # encoder: dict16 -> hidden。
        # - "conv"（历史默认）：res_mlp（ResidualConvBlock，旧 ckpt 兼容）。
        #   注意该 Conv1d 作用在长度=1 序列上会退化为逐通道缩放，仅为兼容保留。
        # - "mlp"：干净的残差 MLP（Linear+GELU+skip），表达力更直接，无退化层。
        if encoder_arch == "conv":
            self.encoder_mlp = res_mlp([self.dict_dim, 64, 64, hidden_dim], dropout=0.0)
        else:
            self.encoder_mlp = _ResidualMLPEncoder(self.dict_dim, 64, hidden_dim)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, state_dim),
        )

        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.encoder_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(max(fan_in, 1))
                    nn.init.uniform_(m.bias, -bound, bound)
        for m in self.decoder_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(max(fan_in, 1))
                    nn.init.uniform_(m.bias, -bound, bound)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.A.bias)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    def compute_pif_atoms(self, x: torch.Tensor) -> torch.Tensor:
        u = x[..., 0:1]
        v = x[..., 1:2]
        r = x[..., 2:3]
        atoms = torch.cat(_compute_atoms_16(u, v, r), dim=-1)
        if self.clamp_pif > 0:
            atoms = torch.clamp(atoms, -self.clamp_pif, self.clamp_pif)
        return atoms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        atoms = self.compute_pif_atoms(x)
        h = self.encoder_mlp(atoms)
        return torch.cat([atoms, h], dim=-1)

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder_mlp(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.A(z) + self.B(u)

    def forward(
        self,
        x_t: torch.Tensor,
        u_t: torch.Tensor,
        x_tp1: Optional[torch.Tensor] = None,
    ):
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        if x_tp1 is not None:
            return z_t, self.encode(x_tp1), z_tp1_hat, x_t_recon, x_tp1_hat
        return z_t, z_tp1_hat, x_t_recon, x_tp1_hat

    def _self_check_dict(self, tol: float = 1e-6) -> None:
        original_clamp = self.clamp_pif
        self.clamp_pif = 0.0
        try:
            x = torch.tensor([[0.5, -0.3, 0.2]], dtype=torch.float32)
            atoms = self.compute_pif_atoms(x).squeeze(0).detach().cpu().numpy()
            u, v, r = 0.5, -0.3, 0.2
            expected = [
                u * abs(u), v * abs(v), r * abs(r), v * r, u * r,
                u * v * r, u * u * r, v * v * r, u * r * r, v * r * r,
                u * abs(v) * v, v * abs(u) * u, r * abs(u) * u, r * abs(v) * v,
                u * abs(u) * u, v * abs(v) * v,
            ]
            if atoms.shape[0] != 16:
                raise AssertionError(f"atom dim mismatch: got {atoms.shape[0]}, expected 16")
            for i, (got, exp) in enumerate(zip(atoms.tolist(), expected)):
                if abs(got - exp) > tol:
                    raise AssertionError(f"atom[{i}] mismatch: got={got}, expected={exp}")
        finally:
            self.clamp_pif = original_clamp


def smoketest_self_check() -> None:
    m = HorizontalKoopmanModelV4DictInput()
    m._self_check_dict()
    x = torch.randn(4, 3)
    u = torch.randn(4, 4)
    z = m.encode(x)
    z1 = m.latent_step(z, u)
    xr = m.reconstruct_state(z1)
    print("[v4_dict_input] OK", z.shape, xr.shape)


if __name__ == "__main__":
    smoketest_self_check()
