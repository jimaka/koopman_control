import math
from typing import List, Tuple, Optional, Type
import torch
import torch.nn as nn

class ResidualConvBlock(nn.Module):
    """带残差连接的 1D 特征增强块"""
    def __init__(self, in_dim, out_dim, activation=nn.GELU, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim)
        self.act = activation()
        self.drop = nn.Dropout(dropout)
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x):
        identity = self.shortcut(x)      
        out = self.fc(x).unsqueeze(-1)          
        out = self.conv(out).squeeze(-1)            
        out = self.act(out)
        out = self.drop(out)
        return out + identity

def res_mlp(sizes: List[int], activation: Type[nn.Module] = nn.GELU, dropout: float = 0.1) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        in_dim, out_dim = sizes[i], sizes[i + 1]
        is_last = (i == len(sizes) - 2)
        if not is_last:
            layers.append(ResidualConvBlock(in_dim, out_dim, activation, dropout))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
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
    【终极版：物理先验 + 严格可观测架构 (Physics-Informed Strict Koopman)】
    - z = [物理状态(3), 显式物理字典(5), 黑盒隐藏特征(24)]
    - 将二次阻尼、科里奥利力强行作为基函数，极大减轻神经网络的死记硬背负担。
    """
    def __init__(
        self,
        state_dim: int = 3, 
        control_dim: int = 4,
        hidden_dim: int = 24, # 黑盒网络只需补充 24 维未知特征
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # 定义显式物理字典的维度: u|u|, v|v|, r|r|, vr, ur 共 5 个
        self.pif_dim = 5 
        
        # 总隐空间维度: 3 + 5 + 24 = 32
        self.latent_dim = state_dim + self.pif_dim + hidden_dim 
        
        # Encoder 仅仅用来挖掘“除了已知物理公式外，还有什么未知干扰”
        self.encoder_mlp = res_mlp([state_dim, 64, 64, 64, hidden_dim], dropout=0.1)
        
        # Koopman 转移矩阵 (带 bias 以应对归一化空间的非零稳态)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=True) 
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.encoder_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.A.bias)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)

    def compute_physics_informed_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算显式物理字典 (Physics-Informed Features)
        x: [..., 3] 代表归一化后的 [u, v, r]
        这里构造流体力学中最典型的非线性耦合项。
        """
        u = x[..., 0:1]
        v = x[..., 1:2]
        r = x[..., 2:3]
        
        # 二次阻尼 (Quadratic Damping)
        uu = u * torch.abs(u)
        vv = v * torch.abs(v)
        rr = r * torch.abs(r)
        
        # 科里奥利/向心力耦合 (Coriolis / Centripetal)
        vr = v * r
        ur = u * r
        
        return torch.cat([uu, vv, rr, vr, ur], dim=-1) # shape: [..., 5]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        隐状态拼接：z = [基础状态(3), 显式物理特征(5), 黑盒隐藏特征(24)]
        """
        pif = self.compute_physics_informed_features(x)
        h = self.encoder_mlp(x)
        return torch.cat([x, pif, h], dim=-1)

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        """切片提取，依然是完美的零重构误差"""
        return z[..., :self.state_dim]

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.A(z) + self.B(u)

    def forward(self, x_t: torch.Tensor, u_t: torch.Tensor, x_tp1: Optional[torch.Tensor] = None):
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        
        if x_tp1 is not None: 
            return z_t, self.encode(x_tp1), z_tp1_hat, x_t_recon, x_tp1_hat
        return z_t, z_tp1_hat, x_t_recon, x_tp1_hat