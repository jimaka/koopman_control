import math
from typing import List, Tuple, Optional

from typing import List, Optional, Type

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
        A = self.A.weight
        try:
            # Preferred method: compute eigenvalues
            eigvals = torch.linalg.eigvals(A)
            rho = torch.max(torch.abs(eigvals))
            self._spec_backend = "eigvals"
            return rho
        except Exception:
            # Fallback: use singular values (upper bound)
            try:
                s = torch.linalg.svdvals(A)
                self._spec_backend = "svdvals"
                return s.max()
            except Exception:
                # Legacy fallback
                _U, S, _V = torch.svd(A)
                self._spec_backend = "svd"
                return S.max()


class HorizontalKoopmanModel(BaseKoopmanModel):
    """Koopman model for horizontal plane ship motion (3-DOF).
    
    State dimension: 6 [x, y, yaw, u, v, r]
    Control dimension: 4 [port_throttle, port_angle, starboard_throttle, starboard_angle]
    
    Args:
        state_dim: State dimension (default: 6 for horizontal plane)
        control_dim: Control dimension (default: 4 for thruster commands)
        latent_dim: Latent dimension (default: 16)
        enc_hidden: Encoder hidden layer sizes
        dec_hidden: Decoder hidden layer sizes
        use_skip: Whether to use skip connections in decoder
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 4,
        latent_dim: int = 16,
        enc_hidden: List[int] = [64, 64],
        dec_hidden: List[int] = [64, 64],
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.use_skip = use_skip
        
        # Encoder network: maps state to latent space
        self.encoder = mlp([state_dim] + enc_hidden + [latent_dim], dropout=0.1,use_conv=True)
        
        # Decoder network: maps latent space back to state
        self.decoder = mlp([latent_dim] + dec_hidden + [state_dim],dropout=0.1,use_conv=True)
        
        # Linear dynamics in latent space: z' = A z + B u
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        # Optional: Additional linear term for state reconstruction
        if use_skip:
            self.C = nn.Linear(latent_dim, state_dim, bias=False)
        
        self.reset_parameters()
        
        print(f"Initialized HorizontalKoopmanModel:")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Use skip: {use_skip}")

    def reset_parameters(self) -> None:
        """Initialize model parameters."""
        # Kaiming uniform for MLPs
        def init_mlp(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

        self.encoder.apply(init_mlp)
        self.decoder.apply(init_mlp)
        
        # Near identity for A; small values for B
        nn.init.eye_(self.A.weight)
        with torch.no_grad():
            self.A.weight.data *= 0.95  # Slightly less than identity for stability
        
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)  # Small gains
        
        if self.use_skip and hasattr(self, 'C'):
            nn.init.xavier_uniform_(self.C.weight, gain=0.1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode state to latent space.
        
        Args:
            x: State tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            Latent representation of shape (batch_size, latent_dim) or (latent_dim,)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to state space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim) or (latent_dim,)
        
        Returns:
            Reconstructed state of shape (batch_size, state_dim) or (state_dim,)
        """
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply Koopman dynamics in latent space.
        
        Args:
            z: Current latent state of shape (batch_size, latent_dim) or (latent_dim,)
            u: Control input of shape (batch_size, control_dim) or (control_dim,)
        
        Returns:
            Next latent state of shape (batch_size, latent_dim) or (latent_dim,)
        """
        return self.A(z) + self.B(u)

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct state from latent vector with optional skip connection.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim) or (latent_dim,)
        
        Returns:
            Reconstructed state of shape (batch_size, state_dim) or (state_dim,)
        """
        if self.use_skip:
            # Combine nonlinear decoder with linear projection
            return self.decoder(z) + self.C(z)
        else:
            return self.decoder(z)

    def forward(
        self, 
        x_t: torch.Tensor, 
        u_t: torch.Tensor, 
        x_tp1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the Koopman model.
        
        Args:
            x_t: Current state of shape (batch_size, state_dim)
            u_t: Control input of shape (batch_size, control_dim)
            x_tp1: Optional next state for training
        
        Returns:
            Tuple containing:
                z_t: Encoded current state
                z_tp1_hat: Predicted next latent state
                x_t_recon: Reconstructed current state
                x_tp1_hat: Predicted next state (if x_tp1 provided, also returns)
        """
        # Encode current state
        z_t = self.encode(x_t)
        
        # Predict next latent state using Koopman dynamics
        z_tp1_hat = self.latent_step(z_t, u_t)
        
        # Reconstruct current state
        x_t_recon = self.reconstruct_state(z_t)
        
        # Predict next state
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)
        
        if x_tp1 is not None:
            # Training mode: also encode true next state
            z_tp1 = self.encode(x_tp1)
            return z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat
        else:
            # Inference mode
            return z_t, z_tp1_hat, x_t_recon, x_tp1_hat

    def predict_sequence(
        self, 
        x0: torch.Tensor, 
        controls: torch.Tensor,
        return_latent: bool = False
    ) -> torch.Tensor:
        """Predict a sequence of states given initial state and control sequence.
        
        Args:
            x0: Initial state of shape (batch_size, state_dim) or (state_dim,)
            controls: Control sequence of shape (seq_len, batch_size, control_dim) 
                     or (seq_len, control_dim)
            return_latent: Whether to return latent states as well
        
        Returns:
            Predicted state sequence of shape (seq_len+1, batch_size, state_dim)
            If return_latent is True, also returns latent sequence
        """
        # Ensure proper dimensions
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)  # (state_dim) -> (1, state_dim)
        if controls.dim() == 2:
            controls = controls.unsqueeze(1)  # (seq_len, control_dim) -> (seq_len, 1, control_dim)
        
        batch_size = x0.size(0)
        seq_len = controls.size(0)
        
        # Initialize sequences
        states = torch.zeros(seq_len + 1, batch_size, self.state_dim, device=x0.device)
        states[0] = x0
        
        if return_latent:
            latents = torch.zeros(seq_len + 1, batch_size, self.latent_dim, device=x0.device)
            z = self.encode(x0)
            latents[0] = z
        
        # Iterate through sequence
        for t in range(seq_len):
            z = self.encode(states[t])
            u_t = controls[t]
            
            # Apply Koopman dynamics
            z_next = self.latent_step(z, u_t)
            
            # Decode to state space
            states[t + 1] = self.reconstruct_state(z_next)
            
            if return_latent:
                latents[t + 1] = z_next
        
        if return_latent:
            return states, latents
        return states


class DeepKoopmanModel(BaseKoopmanModel):
    """Deep Koopman model with additional nonlinear terms in latent dynamics.
    
    Extends the basic Koopman model with additional nonlinear dynamics term.
    
    Args:
        state_dim: State dimension (default: 6)
        control_dim: Control dimension (default: 4)
        latent_dim: Latent dimension (default: 16)
        enc_hidden: Encoder hidden layer sizes
        dyn_hidden: Dynamics network hidden layer sizes
        dec_hidden: Decoder hidden layer sizes
    """
    
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
        
        # Encoder
        self.encoder = mlp([state_dim] + enc_hidden + [latent_dim],dropout=0.1,use_conv=True)
        
        # Decoder
        self.decoder = mlp([latent_dim] + dec_hidden + [state_dim],dropout=0.1,use_conv=True )
        
        # Linear dynamics terms
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)
        
        # Nonlinear dynamics term (optional)
        self.dynamics_net = mlp([latent_dim + control_dim] + dyn_hidden + [latent_dim],dropout=0.1,use_conv=True)
        
        # Weight for mixing linear and nonlinear dynamics
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        self.reset_parameters()
        
        print(f"Initialized DeepKoopmanModel:")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim}")

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
        self.dynamics_net.apply(init_mlp)
        
        # Initialize linear dynamics
        nn.init.eye_(self.A.weight)
        with torch.no_grad():
            self.A.weight.data *= 0.95
        
        nn.init.xavier_uniform_(self.B.weight, gain=0.1)
        
        # Initialize alpha
        nn.init.constant_(self.alpha, 0.5)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode state to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to state space."""
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply deep Koopman dynamics in latent space.
        
        Combines linear and nonlinear dynamics terms.
        """
        # Linear term
        linear_term = self.A(z) + self.B(u)
        
        # Nonlinear term
        zu = torch.cat([z, u], dim=-1)
        nonlinear_term = self.dynamics_net(zu)
        
        # Combine with learnable weight
        alpha = torch.sigmoid(self.alpha)  # Constrain to [0, 1]
        return alpha * linear_term + (1 - alpha) * nonlinear_term

    def forward(
        self, 
        x_t: torch.Tensor, 
        u_t: torch.Tensor, 
        x_tp1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the deep Koopman model."""
        # Encode current state
        z_t = self.encode(x_t)
        
        # Predict next latent state
        z_tp1_hat = self.latent_step(z_t, u_t)
        
        # Reconstruct states
        x_t_recon = self.decode(z_t)
        x_tp1_hat = self.decode(z_tp1_hat)
        
        if x_tp1 is not None:
            # Training mode
            z_tp1 = self.encode(x_tp1)
            return z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat
        else:
            # Inference mode
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
    """Factory function to create Koopman models.
    
    Args:
        model_type: Type of model to create ('horizontal' or 'deep')
        state_dim: State dimension
        control_dim: Control dimension
        latent_dim: Latent dimension
        enc_hidden: Encoder hidden layer sizes
        dec_hidden: Decoder hidden layer sizes
        use_skip: Whether to use skip connections (for horizontal model)
    
    Returns:
        Koopman model instance
    """
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