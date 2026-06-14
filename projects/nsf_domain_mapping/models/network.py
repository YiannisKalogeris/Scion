"""
Neural network architecture for learning diffeomorphisms (PyTorch version).

The network consists of:
    1. CNN Encoder: Extracts shape features from SDF images
    2. Velocity Field: FiLM-conditioned MLP defining the ODE dynamics
    3. Neural ODE: Integrates the velocity field to produce diffeomorphism

The diffeomorphism property is guaranteed by:
    - Neural ODE formulation (flows are diffeomorphisms)
    - Spectral normalization (ensures Lipschitz velocity field)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchdiffeq import odeint_adjoint, odeint
import math
from typing import Optional, Tuple


# =============================================================================
# Building Blocks
# =============================================================================

class SinusoidalEmbedding(nn.Module):
    """Fourier feature embedding for coordinates and time."""
    
    def __init__(self, n_frequencies: int, max_freq: float = 10.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0, math.log(max_freq), n_frequencies))
        self.register_buffer('frequencies', freqs)
        self.n_frequencies = n_frequencies
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., d) tensor
        Returns:
            (..., d * n_freq * 2) tensor
        """
        # Handle empty tensors
        if x.numel() == 0:
            out_dim = x.shape[-1] * self.n_frequencies * 2
            return x.new_empty(*x.shape[:-1], out_dim)
        
        x_freq = x.unsqueeze(-1) * self.frequencies  # (..., d, n_freq)
        x_sin = torch.sin(2 * math.pi * x_freq)
        x_cos = torch.cos(2 * math.pi * x_freq)
        out = torch.cat([x_sin, x_cos], dim=-1)  # (..., d, n_freq * 2)
        return out.reshape(*x.shape[:-1], -1)  # (..., d * n_freq * 2)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer."""
    
    def __init__(self, in_features: int, out_features: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.film_gamma = nn.Linear(cond_dim, out_features)
        self.film_beta = nn.Linear(cond_dim, out_features)
        
        # Initialize FiLM parameters to be very small
        # So initial gamma ≈ 1, beta ≈ 0
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        
        # Also use smaller init for main linear layer
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        # Clamp gamma to prevent extreme scaling
        gamma = 1.0 + 0.1 * torch.tanh(self.film_gamma(cond))  # gamma in [0.9, 1.1]
        beta = 0.1 * torch.tanh(self.film_beta(cond))          # beta in [-0.1, 0.1]
        return gamma * h + beta


class SpectralNormedLinear(nn.Module):
    """Linear layer with spectral normalization for Lipschitz constraint."""
    
    def __init__(self, in_features: int, out_features: int, max_sv: float = 1.0):
        super().__init__()
        self.max_sv = max_sv
        self.linear = spectral_norm(nn.Linear(in_features, out_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_sv * self.linear(x)


# =============================================================================
# Image Encoder
# =============================================================================

class CNNEncoder(nn.Module):
    """CNN encoder for shape images."""
    
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers: input (1, 64, 64)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# -> 4x4
            nn.LeakyReLU(0.2),
        )
        
        # Fully connected: 256 * 4 * 4 = 4096 -> latent_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim),
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (H, W, C) or (B, H, W, C) tensor
        Returns:
            (latent_dim,) or (B, latent_dim) tensor
        """
        # Handle single image vs batch
        if image.dim() == 3:
            # (H, W, C) -> (1, C, H, W)
            x = image.permute(2, 0, 1).unsqueeze(0)
            squeeze_output = True
        else:
            # (B, H, W, C) -> (B, C, H, W)
            x = image.permute(0, 3, 1, 2)
            squeeze_output = False
        
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        
        # CRITICAL: Normalize latent code to unit sphere
        # This prevents FiLM conditioning from exploding
        x = F.normalize(x, p=2, dim=-1)
        
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


# =============================================================================
# Velocity Field
# =============================================================================

class VelocityField(nn.Module):
    """
    Velocity field v(x, t, z) for the Neural ODE.
    
    The flow dx/dt = v(x, t, z) defines the diffeomorphism.
    """
    
    def __init__(
        self, 
        hidden_dim: int = 256, 
        n_layers: int = 4,
        cond_dim: int = 256,
        n_coord_freq: int = 8,
        n_time_freq: int = 4,
        lipschitz_const: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.coord_embed = SinusoidalEmbedding(n_coord_freq, max_freq=10.0)
        self.time_embed = SinusoidalEmbedding(n_time_freq, max_freq=1.0)
        
        coord_embed_dim = 2 * n_coord_freq * 2
        time_embed_dim = 1 * n_time_freq * 2
        input_dim = coord_embed_dim + time_embed_dim
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(n_layers):
            self.layers.append(FiLMLayer(in_dim, hidden_dim, cond_dim))
            in_dim = hidden_dim
        
        self.output_layer = SpectralNormedLinear(hidden_dim, 2, max_sv=lipschitz_const)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (2,) or (B, 2) spatial coordinates
            t: scalar or (1,) time
            z: (cond_dim,) or (B, cond_dim) conditioning vector
        Returns:
            (2,) or (B, 2) velocity
        """
        # Ensure t is at least 1D
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        x_emb = self.coord_embed(x)
        t_emb = self.time_embed(t)
        
        # Handle batched vs unbatched
        if x.dim() == 1:
            h = torch.cat([x_emb, t_emb])
        else:
            # Broadcast t_emb to batch size
            if t_emb.dim() == 1:
                t_emb = t_emb.unsqueeze(0).expand(x.size(0), -1)
            h = torch.cat([x_emb, t_emb], dim=-1)
        
        for layer in self.layers:
            h = layer(h, z)
            h = torch.sin(h)  # SIREN activation (bounded in [-1, 1])
        
        out = self.output_layer(h)
        
        # Clamp output velocity to prevent numerical explosion
        out = torch.clamp(out, -10.0, 10.0)
        
        return out


# =============================================================================
# ODE Function Wrapper
# =============================================================================

class ODEFunc(nn.Module):
    """Wrapper for velocity field to interface with torchdiffeq."""
    
    def __init__(self, velocity_field: VelocityField, z: torch.Tensor):
        super().__init__()
        self.velocity_field = velocity_field
        self.z = z
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.velocity_field(x, t, self.z)


# =============================================================================
# Main Network
# =============================================================================

class DiffeomorphismNet(nn.Module):
    """
    Neural network that learns diffeomorphisms from arbitrary domains to [-1,1]².
    
    Architecture:
        1. CNN encoder: image → latent code z
        2. Neural ODE: integrates velocity field v(x,t,z) from t=0 to t=1
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_coord_freq: int = 8,
        n_time_freq: int = 4,
        lipschitz_const: float = 5.0,
        solver_method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        super().__init__()
        
        self.encoder = CNNEncoder(latent_dim=latent_dim)
        self.velocity_field = VelocityField(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            cond_dim=latent_dim,
            n_coord_freq=n_coord_freq,
            n_time_freq=n_time_freq,
            lipschitz_const=lipschitz_const,
        )
        
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
    
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode shape image to latent vector."""
        return self.encoder(image)
    
    def forward_map(
        self, 
        x: torch.Tensor, 
        z: torch.Tensor,
        return_trajectory: bool = False,
        use_adjoint: bool = True,
    ) -> torch.Tensor:
        """
        Compute forward diffeomorphism: domain → square.
        
        Args:
            x: (2,) or (B, 2) points in domain
            z: (latent_dim,) or (B, latent_dim) latent code
            return_trajectory: if True, return full trajectory
            use_adjoint: if True, use adjoint method for memory-efficient backprop
        """
        ode_func = ODEFunc(self.velocity_field, z)
        
        if return_trajectory:
            t_span = torch.linspace(0, 1, 50, device=x.device, dtype=torch.float32)
        else:
            t_span = torch.tensor([0., 1.], device=x.device, dtype=torch.float32)
        
        # Choose solver
        solver = odeint_adjoint if use_adjoint else odeint
        
        # Use fixed-step method on MPS (avoids float64 rtol/atol issue)
        # Use adaptive method on CUDA/CPU
        if x.device.type == 'mps':
            trajectory = solver(
                ode_func, 
                x, 
                t_span,
                method='euler',  # Fixed-step method
                options={'step_size': 0.05},
            )
        else:
            trajectory = solver(
                ode_func, 
                x, 
                t_span,
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
            )
        
        if return_trajectory:
            return trajectory  # (T, ..., 2)
        return trajectory[-1]  # (..., 2)
    
    def inverse_map(
        self, 
        y: torch.Tensor, 
        z: torch.Tensor,
        use_adjoint: bool = True,
    ) -> torch.Tensor:
        """
        Compute inverse diffeomorphism: square → domain.
        
        Args:
            y: (2,) or (B, 2) points in square
            z: (latent_dim,) latent code
        """
        ode_func = ODEFunc(self.velocity_field, z)
        t_span = torch.tensor([1., 0.], device=y.device, dtype=torch.float32)
        
        solver = odeint_adjoint if use_adjoint else odeint
        
        # Use fixed-step method on MPS (avoids float64 rtol/atol issue)
        if y.device.type == 'mps':
            trajectory = solver(
                ode_func, 
                y, 
                t_span,
                method='euler',
                options={'step_size': 0.05},
            )
        else:
            trajectory = solver(
                ode_func, 
                y, 
                t_span,
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
            )
        
        return trajectory[-1]
    
    def forward(self, image: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode image, then map point."""
        z = self.encode(image)
        return self.forward_map(x, z)
    
    def map_points(self, image: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        Map multiple points.
        
        Args:
            image: (H, W, C) shape image
            points: (N, 2) points to map
        Returns:
            (N, 2) mapped points
        """
        z = self.encode(image)
        # Expand z to match points batch
        z_expanded = z.unsqueeze(0).expand(points.size(0), -1)
        return self.forward_map(points, z_expanded)


# =============================================================================
# Utilities for FEM Applications
# =============================================================================

class DiffeomorphismMapping:
    """
    Wrapper providing FEM-relevant quantities: mapping, Jacobian, determinant.
    """
    
    def __init__(self, model: DiffeomorphismNet, image: torch.Tensor):
        self.model = model
        self.z = model.encode(image)
        self.device = image.device
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map: domain → square."""
        return self.model.forward_map(x, self.z, use_adjoint=False)
    
    def phi_inv(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse map: square → domain."""
        return self.model.inverse_map(y, self.z, use_adjoint=False)
    
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian matrix J = ∂φ/∂x."""
        # Must enable gradients even during evaluation
        with torch.enable_grad():
            x = x.detach().clone().requires_grad_(True)
            y = self.phi(x)
            
            J = torch.zeros(2, 2, device=self.device, dtype=torch.float32)
            for i in range(2):
                grad_outputs = torch.zeros_like(y)
                grad_outputs[i] = 1.0
                grad = torch.autograd.grad(y, x, grad_outputs, create_graph=False, retain_graph=True)[0]
                J[i] = grad
        
        return J
    
    def jacobian_inv(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse Jacobian J⁻¹."""
        return torch.linalg.inv(self.jacobian(x))
    
    def det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian determinant |J|."""
        return torch.linalg.det(self.jacobian(x))
    
    # Batched versions
    def phi_batch(self, points: torch.Tensor) -> torch.Tensor:
        """Map batch of points."""
        z_expanded = self.z.unsqueeze(0).expand(points.size(0), -1)
        return self.model.forward_map(points, z_expanded, use_adjoint=False)
    
    def jacobian_batch(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian for batch of points."""
        return torch.stack([self.jacobian(p) for p in points])
    
    def det_jacobian_batch(self, points: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian determinant for batch of points."""
        return torch.stack([self.det_jacobian(p) for p in points])


def create_model(
    latent_dim: int = 256,
    hidden_dim: int = 256,
    n_layers: int = 4,
    n_coord_freq: int = 8,
    n_time_freq: int = 4,
    lipschitz_const: float = 5.0,
    device: str = 'cpu',
) -> DiffeomorphismNet:
    """Create model with given hyperparameters."""
    model = DiffeomorphismNet(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_coord_freq=n_coord_freq,
        n_time_freq=n_time_freq,
        lipschitz_const=lipschitz_const,
    )
    return model.to(device)


if __name__ == "__main__":
    # Test model creation and forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = DiffeomorphismNet(
        latent_dim=256,
        hidden_dim=256,
        n_layers=4,
        lipschitz_const=5.0,
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params:,}")
    
    # Dummy inputs
    dummy_image = torch.ones(64, 64, 1, device=device, dtype=torch.float32)
    dummy_point = torch.tensor([0.5, 0.3], device=device, dtype=torch.float32)
    
    # Forward pass
    with torch.no_grad():
        z = model.encode(dummy_image)
        print(f"Latent dim: {z.shape}")
        
        mapped = model.forward_map(dummy_point, z)
        print(f"Mapped point: {mapped}")
        
        recovered = model.inverse_map(mapped, z)
        print(f"Recovered point: {recovered}")
        print(f"Cycle error: {torch.norm(dummy_point - recovered):.6f}")
    
    # Test Jacobian
    mapping = DiffeomorphismMapping(model, dummy_image)
    J = mapping.jacobian(dummy_point)
    det_J = mapping.det_jacobian(dummy_point)
    print(f"Jacobian:\n{J}")
    print(f"Jacobian determinant: {det_J:.6f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    dummy_points = torch.randn(10, 2, device=device, dtype=torch.float32, requires_grad=True)
    dummy_targets = torch.randn(10, 2, device=device, dtype=torch.float32)
    
    z = model.encode(dummy_image)
    z_expanded = z.unsqueeze(0).expand(10, -1)
    mapped = model.forward_map(dummy_points, z_expanded)
    
    loss = F.mse_loss(mapped, dummy_targets)
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Gradient norm: {dummy_points.grad.norm().item():.6f}")
    print("✓ Gradient flow works!")