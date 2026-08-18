"""
Normalizing flows for variational Bayesian inference.

Learns an invertible map from a simple base distribution (Gaussian)
to the posterior. Useful when MCMC is too expensive (e.g. with
costly PDE forward models).

Planned implementations
-----------------------
- RealNVP (affine coupling layers)
- Neural Spline Flows (NSF)
- Masked Autoregressive Flows (MAF)
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import numpy as np


class AffineCouplingLayer(nn.Module):
    """Single affine coupling layer (RealNVP building block).

    Splits input into two halves; one half is transformed by a
    scale-and-shift predicted from the other half.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, mask_even: bool = True):
        super().__init__()
        self.dim = dim
        self.mask_even = mask_even
        self.net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2),  # outputs: [shift, log_scale] packed
        )
        self.log_scale_net = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - dim // 2),
            nn.Tanh(),  # bounded log-scale for stability
        )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dim // 2
        if self.mask_even:
            return x[:, :d], x[:, d:]
        return x[:, d:], x[:, :d]

    def _merge(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.mask_even:
            return torch.cat([x1, x2], dim=-1)
        return torch.cat([x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: x → z, returns (z, log_det_jacobian)."""
        x1, x2 = self._split(x)
        shift = self.net(x1)
        log_scale = self.log_scale_net(x1)
        z2 = x2 * torch.exp(log_scale) + shift
        log_det = log_scale.sum(dim=-1)
        return self._merge(x1, z2), log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z → x."""
        z1, z2 = self._split(z)
        shift = self.net(z1)
        log_scale = self.log_scale_net(z1)
        x2 = (z2 - shift) * torch.exp(-log_scale)
        return self._merge(z1, x2)


class RealNVP(nn.Module):
    """RealNVP normalizing flow.

    Parameters
    ----------
    dim : int
        Dimension of the parameter space.
    n_layers : int
        Number of coupling layers.
    hidden_dim : int
        Hidden dimension in each coupling layer.
    """

    def __init__(self, dim: int, n_layers: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, hidden_dim, mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map from parameter space to latent space."""
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det += log_det
        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Map from latent space to parameter space."""
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def sample(self, n_samples: int) -> torch.Tensor:
        """Draw samples by mapping standard normal through the inverse flow."""
        z = torch.randn(n_samples, self.dim)
        return self.inverse(z)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate log-probability under the flow model."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
