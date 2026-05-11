import math
from typing import List, Tuple, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.GELU, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)                  
        x = x.unsqueeze(-1)             
        x = self.conv(x)                
        x = x.squeeze(-1)               
        x = self.act(x)
        x = self.drop(x)
        return x

def mlp(sizes: List[int], activation: Type[nn.Module] = nn.GELU, out_activation: Optional[Type[nn.Module]] = None, dropout: float = 0.1, use_conv: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        in_dim, out_dim = sizes[i], sizes[i + 1]
        is_last = i == len(sizes) - 2
        if not is_last:
            if use_conv: layers.append(ConvMLPBlock(in_dim, out_dim, activation=activation, dropout=dropout))
            else: layers.extend([nn.Linear(in_dim, out_dim), activation(), nn.Dropout(dropout)])
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_activation is not None: layers.append(out_activation())
    return nn.Sequential(*layers)

class BaseKoopmanModel(nn.Module):
    def spectral_radius(self) -> torch.Tensor:
        A_diff = self.A.weight
        I = torch.eye(A_diff.size(0), device=A_diff.device)
        A_eff = I + A_diff
        try:
            return torch.max(torch.abs(torch.linalg.eigvals(A_eff)))
        except:
            return torch.linalg.svdvals(A_eff).max()

class HorizontalKoopmanModel(BaseKoopmanModel):
    """
    Direct Observable Koopman Model
    隐变量的前 8 维强制等于输入的 8 维物理特征，彻底消除自编码器重构误差！
    """
    def __init__(
        self,
        input_dim: int = 8,
        state_dim: int = 3,
        control_dim: int = 4,
        latent_dim: int = 32, 
        enc_hidden: List[int] = [128, 128]
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        
        # 【核心改动】：Encoder 只负责生成附加的隐藏维度 (32 - 8 = 24维)
        self.hidden_dim = latent_dim - input_dim
        self.encoder = mlp([input_dim] + enc_hidden + [self.hidden_dim], dropout=0.1, use_conv=True)
        
        # 抛弃 Decoder，物理状态直接从隐变量切片获取
        self.A = nn.Linear(latent_dim, latent_dim, bias=True)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.apply(lambda m: nn.init.kaiming_uniform_(m.weight) if isinstance(m, nn.Linear) else None)
        # 初始化 A 矩阵为较小的值，允许系统学习自然阻尼
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.A.bias)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x 是 8 维物理特征，h 是 24 维神经网络特征
        h = self.encoder(x)
        # 强制拼接：z 的前 8 维永远等于输入的物理特征
        return torch.cat([x, h], dim=-1)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.A(z) + self.B(u)

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        # 【核心改动】：无需解码网络！物理量 u, v, r 就是隐变量的前 3 维
        return z[..., :self.state_dim]

    def forward(self, x_t: torch.Tensor, u_t: torch.Tensor, x_tp1: Optional[torch.Tensor] = None):
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        
        if x_tp1 is not None:
            return z_t, self.encode(x_tp1), z_tp1_hat, x_t_recon, x_tp1_hat
        return z_t, z_tp1_hat, x_t_recon, x_tp1_hat