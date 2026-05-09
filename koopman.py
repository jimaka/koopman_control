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
        # x: (B,C)
        x = self.fc(x)                  # (B,out_dim)
        x = x.unsqueeze(-1)             # (B,out_dim,1)
        x = self.conv(x)                # (B,out_dim,1)
        x = x.squeeze(-1)               # (B,out_dim)
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
    """
    MLP with optional Conv1d enhancement

    sizes: [in, h1, h2, ..., out]
    """
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
    """Base class for Koopman models with common functionality."""
    
    def spectral_radius(self) -> torch.Tensor:
        """Differentiable spectral radius estimate of A matrix.
        
        Returns:
            Spectral radius (largest eigenvalue magnitude) of A matrix
        """
        A_diff = self.A.weight
        # 【残差学习修改】：实际的离散转移矩阵是 I + A_diff
        I = torch.eye(A_diff.size(0), device=A_diff.device)
        A_eff = I + A_diff
        
        try:
            # Preferred method: compute eigenvalues
            eigvals = torch.linalg.eigvals(A_eff)
            rho = torch.max(torch.abs(eigvals))
            self._spec_backend = "eigvals"
            return rho
        except Exception:
            # Fallback: use singular values (upper bound)
            try:
                s = torch.linalg.svdvals(A_eff)
                self._spec_backend = "svdvals"
                return s.max()
            except Exception:
                # Legacy fallback
                _U, S, _V = torch.svd(A_eff)
                self._spec_backend = "svd"
                return S.max()


class HorizontalKoopmanModel(BaseKoopmanModel):
    """Koopman model for horizontal plane ship motion (3-DOF).
    
    State dimension: 6 [x, y, yaw, u, v, r]
    Control dimension: 4 [port_throttle, port_angle, starboard_throttle, starboard_angle]
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 4,
        latent_dim: int = 16,
        enc_hidden: List[int] = [128, 128],
        dec_hidden: List[int] = [128, 128],
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.use_skip = use_skip
        
        # Encoder network: maps state to latent space
        self.encoder = mlp([state_dim] + enc_hidden + [latent_dim], dropout=0.1, use_conv=True)
        
        # Decoder network: maps latent space back to state
        self.decoder = mlp([latent_dim] + dec_hidden + [state_dim], dropout=0.1, use_conv=True)
        
        # Linear dynamics in latent space: delta_z = A z + B u
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        # Optional: Additional linear term for state reconstruction
        if use_skip:
            self.C = nn.Linear(latent_dim, state_dim, bias=False)
        
        self.reset_parameters()
        
        print(f"Initialized HorizontalKoopmanModel (Residual Mode):")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Use skip: {use_skip}")

    def reset_parameters(self) -> None:
        """Initialize model parameters."""
        def init_mlp(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

        self.encoder.apply(init_mlp)
        self.decoder.apply(init_mlp)
        
        # 【残差学习修改】：A 矩阵初始化为 0 附近的极小值，代表初始系统状态维持不变（增量为0）
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)  # Small gains
        
        if self.use_skip and hasattr(self, 'C'):
            nn.init.xavier_uniform_(self.C.weight, gain=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply Koopman dynamics in latent space (Residual Form)."""
        # 【残差学习修改】：计算增量并与原状态相加
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

    def predict_sequence(
        self, 
        x0: torch.Tensor, 
        controls: torch.Tensor,
        return_latent: bool = False
    ) -> torch.Tensor:
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if controls.dim() == 2:
            controls = controls.unsqueeze(1)
        
        batch_size = x0.size(0)
        seq_len = controls.size(0)
        
        states = torch.zeros(seq_len + 1, batch_size, self.state_dim, device=x0.device)
        states[0] = x0
        
        if return_latent:
            latents = torch.zeros(seq_len + 1, batch_size, self.latent_dim, device=x0.device)
            z = self.encode(x0)
            latents[0] = z
        
        for t in range(seq_len):
            z = self.encode(states[t])
            u_t = controls[t]
            z_next = self.latent_step(z, u_t)
            states[t + 1] = self.reconstruct_state(z_next)
            
            if return_latent:
                latents[t + 1] = z_next
        
        if return_latent:
            return states, latents
        return states


class DeepKoopmanModel(BaseKoopmanModel):
    """Deep Koopman model with additional nonlinear terms in latent dynamics."""
    
    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 4,
        latent_dim: int = 16,
        enc_hidden: List[int] = [64, 64],
        dyn_hidden: List[int] = [64, 64],
        dec_hidden: List[int] = [64, 64],
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        
        self.encoder = mlp([state_dim] + enc_hidden + [latent_dim], dropout=0.1, use_conv=True)
        self.decoder = mlp([latent_dim] + dec_hidden + [state_dim], dropout=0.1, use_conv=True)
        
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        self.dynamics_net = mlp([latent_dim + control_dim] + dyn_hidden + [latent_dim], dropout=0.1, use_conv=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.reset_parameters()
        
        print(f"Initialized DeepKoopmanModel (Residual Mode):")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim}")

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
        self.dynamics_net.apply(init_mlp)
        
        # 【残差学习修改】：初始化 A 为 0 附近极小值
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)
        
        nn.init.constant_(self.alpha, 0.5)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply deep Koopman dynamics in latent space (Residual Form)."""
        # 计算线性增量
        delta_linear = self.A(z) + self.B(u)
        
        # 计算非线性增量
        zu = torch.cat([z, u], dim=-1)
        delta_nonlinear = self.dynamics_net(zu)
        
        # 融合总增量
        alpha = torch.sigmoid(self.alpha)  
        delta_z = alpha * delta_linear + (1 - alpha) * delta_nonlinear
        
        # 【残差学习修改】：返回残差相加结果
        return z + delta_z

    def forward(
        self, 
        x_t: torch.Tensor, 
        u_t: torch.Tensor, 
        x_tp1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.decode(z_t)
        x_tp1_hat = self.decode(z_tp1_hat)
        
        if x_tp1 is not None:
            z_tp1 = self.encode(x_tp1)
            return z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat
        else:
            return z_t, z_tp1_hat, x_t_recon, x_tp1_hat


def create_koopman_model(
    model_type: str = "horizontal",
    state_dim: int = 6,
    control_dim: int = 4,
    latent_dim: int = 16,
    enc_hidden: List[int] = [64, 64],
    dec_hidden: List[int] = [64, 64],
    use_skip: bool = True,
) -> nn.Module:
    if model_type == "horizontal":
        return HorizontalKoopmanModel(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            use_skip=use_skip,
        )
    elif model_type == "deep":
        return DeepKoopmanModel(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'horizontal' or 'deep'.")


if __name__ == "__main__":
    # Test the models
    print("Testing Koopman models for horizontal ship motion...")
    
    # Test HorizontalKoopmanModel
    print("\n1. Testing HorizontalKoopmanModel:")
    model = HorizontalKoopmanModel(
        state_dim=6,
        control_dim=4,
        latent_dim=16,
        enc_hidden=[32, 32],
        dec_hidden=[32, 32],
        use_skip=True,
    )
    
    # Create dummy data
    batch_size = 8
    x_t = torch.randn(batch_size, 6)
    u_t = torch.randn(batch_size, 4)
    x_tp1 = torch.randn(batch_size, 6)
    
    # Forward pass
    z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat = model(x_t, u_t, x_tp1)
    
    print(f"  Input shapes: x_t={x_t.shape}, u_t={u_t.shape}")
    print(f"  Output shapes:")
    print(f"    z_t: {z_t.shape}")
    print(f"    z_tp1: {z_tp1.shape}")
    print(f"    z_tp1_hat: {z_tp1_hat.shape}")
    print(f"    x_t_recon: {x_t_recon.shape}")
    print(f"    x_tp1_hat: {x_tp1_hat.shape}")
    
    # Test spectral radius
    rho = model.spectral_radius()
    print(f"  Spectral radius: {rho.item():.4f} (computed via {model._spec_backend})")
    
    # Test sequence prediction
    print("\n2. Testing sequence prediction:")
    seq_len = 10
    controls_seq = torch.randn(seq_len, batch_size, 4)
    states_seq = model.predict_sequence(x_t[0], controls_seq[:, 0, :])
    print(f"  Sequence shape: {states_seq.shape}")
    
    # Test DeepKoopmanModel
    print("\n3. Testing DeepKoopmanModel:")
    deep_model = DeepKoopmanModel(
        state_dim=6,
        control_dim=4,
        latent_dim=16,
        enc_hidden=[32, 32],
        dec_hidden=[32, 32],
    )
    
    z_t_deep, z_tp1_deep, z_tp1_hat_deep, x_t_recon_deep, x_tp1_hat_deep = deep_model(x_t, u_t, x_tp1)
    print(f"  Output shapes match: {z_t.shape == z_t_deep.shape}")
    
    # Test spectral radius for deep model
    rho_deep = deep_model.spectral_radius()
    print(f"  Deep model spectral radius: {rho_deep.item():.4f} (computed via {deep_model._spec_backend})")
    
    print("\nAll tests passed!")