import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn


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

    def stability_regularization(
        self,
        rho_limit: float = 0.995,
        frobenius_weight: float = 1e-4,
    ) -> torch.Tensor:
        """Penalize unstable latent dynamics without changing the data model."""
        rho = self.spectral_radius()
        spectral_loss = torch.relu(rho - rho_limit) ** 2
        frob_loss = torch.mean(self.A.weight ** 2)
        return spectral_loss + frobenius_weight * frob_loss

    @torch.no_grad()
    def project_stable_dynamics_(self, rho_limit: float = 0.995) -> None:
        """Project A back inside the requested spectral radius after an update."""
        rho = self.spectral_radius()
        if torch.isfinite(rho) and rho > rho_limit:
            self.A.weight.mul_(rho_limit / (rho + 1e-8))

    def rollout_latent(self, z0: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """Roll out the latent state using only z_{k+1}=Az_k+Bu_k."""
        if controls.dim() == 2:
            controls = controls.unsqueeze(1)

        latents = [z0]
        z = z0
        for t in range(controls.size(0)):
            z = self.latent_step(z, controls[t])
            latents.append(z)
        return torch.stack(latents, dim=0)

    def predict_sequence(
        self,
        x0: torch.Tensor,
        controls: torch.Tensor,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """Predict a sequence with one encode followed by open-loop latent rollout."""
        if x0.dim() == 1:
            x0 = x0.unsqueeze(0)
        if controls.dim() == 2:
            controls = controls.unsqueeze(1)

        z0 = self.encode(x0)
        latents = self.rollout_latent(z0, controls)
        states = [x0]
        states.extend(self.reconstruct_state(latents[t]) for t in range(1, latents.size(0)))
        states = torch.stack(states, dim=0)

        if return_latent:
            return states, latents
        return states


class HorizontalKoopmanModel(BaseKoopmanModel):
    """State-lifted Deep Koopman model for horizontal ship motion.

    The ship state definition stays unchanged:
        [x, y, yaw, u, v, r]

    The encoder builds Koopman observables as g(x) = [x, phi(x)].  This
    makes the physical state directly observable in latent space, while the
    latent evolution itself remains the standard controlled linear form:
        z_{k+1} = A z_k + B u_k
    """
    
    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 4,
        latent_dim: int = 16,
        enc_hidden: List[int] = [64, 64],
        dec_hidden: List[int] = [64, 64],
        use_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim < state_dim:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be >= state_dim ({state_dim}) "
                "for state-lifted Koopman observables."
            )

        self.state_dim = state_dim
        self.control_dim = control_dim
        self.latent_dim = latent_dim
        self.use_skip = use_skip
        self.observable_dim = latent_dim - state_dim

        if self.observable_dim > 0:
            self.encoder = mlp(
                [state_dim] + enc_hidden + [self.observable_dim],
                dropout=dropout,
                use_conv=True,
            )
        else:
            self.encoder = nn.Identity()

        # The decoder learns a residual around the physically anchored state
        # slice.  This avoids mean-output collapse while still allowing a
        # nonlinear inverse map for the learned observables.
        self.decoder = mlp(
            [latent_dim] + dec_hidden + [state_dim],
            dropout=dropout,
            use_conv=True,
        )

        # Linear dynamics in latent space: z' = A z + B u
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

        # Fixed state selector used by legacy export code as C.
        self.C = nn.Linear(latent_dim, state_dim, bias=False)
        self.C.weight.requires_grad_(False)

        self.reset_parameters()

        print(f"Initialized HorizontalKoopmanModel:")
        print(f"  State dim: {state_dim}")
        print(f"  Control dim: {control_dim}")
        print(f"  Latent dim: {latent_dim}")
        print(f"  Observable dim: {self.observable_dim}")
        print(f"  State-lifted latent: True")

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

        # Make the residual decoder initially zero so decode(encode(x)) starts
        # as the state anchor rather than an arbitrary nonlinear projection.
        for module in reversed(self.decoder):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break

        with torch.no_grad():
            self.C.weight.zero_()
            self.C.weight[:, :self.state_dim] = torch.eye(
                self.state_dim,
                device=self.C.weight.device,
                dtype=self.C.weight.dtype,
            )

        nn.init.eye_(self.A.weight)
        with torch.no_grad():
            self.A.weight.mul_(0.98)
            if self.observable_dim > 0:
                # Learned observables should not immediately dominate the
                # anchored physical state during early rollout.
                self.A.weight[:self.state_dim, self.state_dim:] *= 0.05
                self.A.weight[self.state_dim:, :self.state_dim] *= 0.05

        nn.init.xavier_uniform_(self.B.weight, gain=0.05)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode state to Koopman observables g(x) = [x, phi(x)]."""
        phi = self.encoder(x)
        if self.observable_dim == 0:
            return x
        return torch.cat([x, phi], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to the current normalized state space."""
        residual = self.decoder(z)
        if self.use_skip:
            return self.C(z) + residual
        return residual

    def latent_step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply the controlled linear Koopman operator in latent space."""
        return self.A(z) + self.B(u)

    def reconstruct_state(self, z: torch.Tensor) -> torch.Tensor:
        """Compatibility wrapper used by training and rollout scripts."""
        return self.decode(z)

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
        z_t = self.encode(x_t)
        z_tp1_hat = self.latent_step(z_t, u_t)
        x_t_recon = self.reconstruct_state(z_t)
        x_tp1_hat = self.reconstruct_state(z_tp1_hat)

        if x_tp1 is not None:
            z_tp1 = self.encode(x_tp1)
            return z_t, z_tp1, z_tp1_hat, x_t_recon, x_tp1_hat
        return z_t, z_tp1_hat, x_t_recon, x_tp1_hat


class DeepKoopmanModel(HorizontalKoopmanModel):
    """Standard Deep Koopman model with linear latent evolution.

    Kept as a separate class name for checkpoint/config compatibility.  Unlike
    the previous implementation, this class does not add a nonlinear dynamics
    network; nonlinearity belongs in the encoder/decoder and the latent
    operator remains linear.
    """

    def __init__(
        self,
        state_dim: int = 6,
        control_dim: int = 4,
        latent_dim: int = 16,
        enc_hidden: List[int] = [64, 64],
        dyn_hidden: List[int] = [64, 64],
        dec_hidden: List[int] = [64, 64],
        use_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        _ = dyn_hidden  # accepted for backward-compatible constructor calls
        super().__init__(
            state_dim=state_dim,
            control_dim=control_dim,
            latent_dim=latent_dim,
            enc_hidden=enc_hidden,
            dec_hidden=dec_hidden,
            use_skip=use_skip,
            dropout=dropout,
        )


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