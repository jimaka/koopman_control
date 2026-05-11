import math
from typing import List, Tuple, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvMLPBlock(nn.Module):
    """
    Linear + Depthwise Conv1d + Activation + Dropout
    用于1维向量特征增强
    """
    def __init__(self, in_dim, out_dim, activation=nn.GELU, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

        # depthwise 1D conv（几乎不增加参数）
        self.conv = nn.Conv1d(
            out_dim,
            out_dim,
            kernel_size=3,
            padding=1,
            groups=out_dim
        )

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


def mlp(
    sizes: List[int],
    activation: Type[nn.Module] = nn.GELU,
    out_activation: Optional[Type[nn.Module]] = None,
    dropout: float = 0.1,
    use_conv: bool = True,
) -> nn.Sequential:
    layers: List[nn.Module] = []

    for i in range(len(sizes) - 1):
        in_dim = sizes[i]
        out_dim = sizes[i + 1]

        is_last = i == len(sizes) - 2

        if not is_last:
            if use_conv:
                layers.append(
                    ConvMLPBlock(
                        in_dim,
                        out_dim,
                        activation=activation,
                        dropout=dropout
                    )
                )
            else:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation())
                layers.append(nn.Dropout(dropout))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if out_activation is not None:
                layers.append(out_activation())

    return nn.Sequential(*layers)


class BaseKoopmanModel(nn.Module):
    def spectral_radius(self) -> torch.Tensor:
        """Differentiable spectral radius estimate of A matrix."""
        A_diff = self.A.weight
        I = torch.eye(A_diff.size(0), device=A_diff.device)
        A_eff = I + A_diff
        
        try:
            eigvals = torch.linalg.eigvals(A_eff)
            rho = torch.max(torch.abs(eigvals))
            self._spec_backend = "eigvals"
            return rho
        except Exception:
            try:
                s = torch.linalg.svdvals(A_eff)
                self._spec_backend = "svdvals"
                return s.max()
            except Exception:
                _U, S, _V = torch.svd(A_eff)
                self._spec_backend = "svd"
                return S.max()


class HorizontalKoopmanModel(BaseKoopmanModel):
    """Koopman model for horizontal plane ship motion (Affine EDMD Mode)."""
    
    def __init__(
        self,
        input_dim: int = 8,
        state_dim: int = 3,
        control_dim: int = 4,
        latent_dim: int = 32, # 【优化】：降维打击，压制虚假震荡
        enc_hidden: List[int] = [128, 128],
        dec_hidden: List[int] = [128, 128],
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.use_skip = use_skip
        
        self.encoder = mlp([input_dim] + enc_hidden + [latent_dim], dropout=0.1, use_conv=True)
        self.decoder = mlp([latent_dim] + dec_hidden + [state_dim], dropout=0.1, use_conv=True)
        
        # 【优化】：开启 A 矩阵的偏置 (bias=True)，吸收稳态基础阻尼与水流扰动
        self.A = nn.Linear(latent_dim, latent_dim, bias=True)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        if use_skip:
            self.C = nn.Linear(latent_dim, state_dim, bias=False)
        
        self.reset_parameters()
        
        print(f"Initialized HorizontalKoopmanModel (Affine EDMD Mode):")
        print(f"  Input dim: {input_dim}")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim} (Bottleneck)")

    def reset_parameters(self) -> None:
        def init_mlp(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

        self.encoder.apply(init_mlp)
        self.decoder.apply(init_mlp)
        
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        if self.A.bias is not None:
            nn.init.zeros_(self.A.bias)
            
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)
        
        if self.use_skip and hasattr(self, 'C'):
            nn.init.xavier_uniform_(self.C.weight, gain=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        delta_z = self.A(z) + self.B(u)
        return z + delta_z

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        if self.use_skip:
            return self.decoder(z) + self.C(z)
        else:
            return self.decoder(z)

    def forward(
        self, 
        x_t: torch.Tensor, 
        u_t: torch.Tensor, 
        x_tp1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        
        if x_tp1 is not None:
            z_tp1 = self.encode(x_tp1)
            return z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat
        else:
            return z_t, z_tp1_hat, x_t_recon, x_tp1_hat