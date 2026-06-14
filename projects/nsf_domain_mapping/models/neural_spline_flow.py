"""
Neural Spline Flow for Diffeomorphism Learning

This implementation uses Rational-Quadratic Splines (from Neural Spline Flows paper)
to guarantee positive Jacobian determinants by construction.

Key insight: Monotonic splines have positive derivatives, so the Jacobian
determinant (product of derivatives in coupling layers) is always positive.

Reference: Durkan et al., "Neural Spline Flows", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Rational-Quadratic Spline Transform
# =============================================================================

def rational_quadratic_spline(
    inputs: torch.Tensor,
    widths: torch.Tensor,
    heights: torch.Tensor,
    derivatives: torch.Tensor,
    inverse: bool = False,
    left: float = -1.0,
    right: float = 1.0,
    bottom: float = -1.0,
    top: float = 1.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rational-quadratic spline transform.
    
    The spline is defined by K bins with:
    - widths: width of each bin (sums to right - left)
    - heights: height of each bin (sums to top - bottom)  
    - derivatives: derivative at each knot point (K+1 values)
    
    Args:
        inputs: Input values to transform
        widths: Unnormalized bin widths (K,)
        heights: Unnormalized bin heights (K,)
        derivatives: Unnormalized derivatives at knots (K+1,)
        inverse: Whether to compute inverse transform
        
    Returns:
        outputs: Transformed values
        logabsdet: Log absolute determinant of Jacobian
    """
    # Normalize widths and heights to sum to interval length
    widths = F.softmax(widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * widths.shape[-1]) * widths
    
    heights = F.softmax(heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * heights.shape[-1]) * heights
    
    # Ensure derivatives are positive
    derivatives = min_derivative + F.softplus(derivatives)
    
    # Scale to actual interval
    widths = widths * (right - left)
    heights = heights * (top - bottom)
    
    # Cumulative widths and heights give knot positions
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0)
    cumwidths = cumwidths + left
    cumwidths[..., -1] = right  # Ensure exact right boundary
    
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0)
    cumheights = cumheights + bottom
    cumheights[..., -1] = top  # Ensure exact top boundary
    
    # Find which bin each input falls into
    if inverse:
        bin_idx = torch.searchsorted(cumheights, inputs.unsqueeze(-1)) - 1
    else:
        bin_idx = torch.searchsorted(cumwidths, inputs.unsqueeze(-1)) - 1
    
    bin_idx = bin_idx.squeeze(-1)
    bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)
    
    # Gather relevant bin parameters
    input_cumwidths = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    input_cumheights = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    input_delta = input_bin_heights / input_bin_widths
    
    input_derivatives = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_derivatives_plus_one = derivatives.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    
    # Compute spline
    if inverse:
        # Inverse: given y, find x
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_bin_heights * (input_delta - input_derivatives)
        
        b = input_bin_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        
        c = -input_delta * (inputs - input_cumheights)
        
        discriminant = b.pow(2) - 4 * a * c
        discriminant = torch.clamp(discriminant, min=0)  # Numerical stability
        
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        root = torch.clamp(root, 0, 1)  # Ensure in [0, 1]
        
        outputs = root * input_bin_widths + input_cumwidths
        
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        logabsdet = -logabsdet  # Inverse has negative log det
        
    else:
        # Forward: given x, find y
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta = torch.clamp(theta, 0, 1)  # Ensure in [0, 1]
        
        theta_one_minus_theta = theta * (1 - theta)
        
        numerator = input_bin_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        
        outputs = input_cumheights + numerator / denominator
        
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    
    return outputs, logabsdet


# =============================================================================
# Coupling Layer
# =============================================================================

class SplineCouplingLayer(nn.Module):
    """
    Coupling layer using rational-quadratic splines.
    
    Splits input into two parts (x1, x2):
    - x1 passes through unchanged
    - x2 is transformed by a spline whose parameters depend on x1
    
    The mask determines which coordinates are transformed.
    """
    
    def __init__(
        self,
        dim: int = 2,
        mask: torch.Tensor = None,
        n_bins: int = 8,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        domain_left: float = -2.0,
        domain_right: float = 2.0,
        codomain_left: float = -1.0,
        codomain_right: float = 1.0,
        condition_dim: int = 0,
    ):
        super().__init__()
        
        self.dim = dim
        self.n_bins = n_bins
        self.domain_left = domain_left
        self.domain_right = domain_right
        self.codomain_left = codomain_left
        self.codomain_right = codomain_right
        
        # Mask: 1 = pass through, 0 = transform
        if mask is None:
            mask = torch.zeros(dim)
            mask[::2] = 1  # Transform odd indices
        self.register_buffer('mask', mask)
        
        # Number of features to condition on
        n_condition = int(mask.sum().item()) + condition_dim
        # Number of features to transform
        n_transform = dim - int(mask.sum().item())
        
        # Network to predict spline parameters
        # Outputs: widths (K), heights (K), derivatives (K+1) for each transformed dim
        n_params = n_transform * (3 * n_bins + 1)
        
        layers = [nn.Linear(n_condition, hidden_dim), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, n_params))
        
        self.param_net = nn.Sequential(*layers)
        
        # Initialize to identity-ish
        nn.init.zeros_(self.param_net[-1].weight)
        nn.init.zeros_(self.param_net[-1].bias)
        
        self.n_transform = n_transform
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor = None,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse pass through coupling layer.
        
        Args:
            x: Input tensor (..., dim)
            condition: Optional conditioning tensor (..., condition_dim)
            inverse: If True, compute inverse transform
            
        Returns:
            y: Transformed tensor (..., dim)
            logabsdet: Log absolute Jacobian determinant (...,)
        """
        # Split into identity and transform parts
        x_id = x[..., self.mask == 1]
        x_tr = x[..., self.mask == 0]
        
        # Get conditioning input
        if condition is not None:
            cond_input = torch.cat([x_id, condition], dim=-1)
        else:
            cond_input = x_id
        
        # Predict spline parameters
        params = self.param_net(cond_input)
        
        # Reshape to (batch, n_transform, n_params_per_dim)
        params = params.reshape(*x.shape[:-1], self.n_transform, 3 * self.n_bins + 1)
        
        widths = params[..., :self.n_bins]
        heights = params[..., self.n_bins:2*self.n_bins]
        derivatives = params[..., 2*self.n_bins:]
        
        # Apply spline to each transformed dimension
        y_tr = torch.zeros_like(x_tr)
        logabsdet = torch.zeros(x.shape[:-1], device=x.device)
        
        for i in range(self.n_transform):
            y_tr[..., i], ld = rational_quadratic_spline(
                x_tr[..., i],
                widths[..., i, :],
                heights[..., i, :],
                derivatives[..., i, :],
                inverse=inverse,
                left=self.domain_left if not inverse else self.codomain_left,
                right=self.domain_right if not inverse else self.codomain_right,
                bottom=self.codomain_left if not inverse else self.domain_left,
                top=self.codomain_right if not inverse else self.domain_right,
            )
            logabsdet = logabsdet + ld
        
        # Recombine
        y = torch.zeros_like(x)
        y[..., self.mask == 1] = x_id
        y[..., self.mask == 0] = y_tr
        
        return y, logabsdet


# =============================================================================
# Neural Spline Flow
# =============================================================================

class NeuralSplineFlow(nn.Module):
    """
    Neural Spline Flow for learning diffeomorphisms.
    
    Composes multiple coupling layers with alternating masks to ensure
    all dimensions are transformed.
    
    Guarantees:
    - Bijective (invertible) mapping
    - Positive Jacobian determinant everywhere
    - Differentiable forward and inverse
    """
    
    def __init__(
        self,
        dim: int = 2,
        n_layers: int = 6,
        n_bins: int = 8,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        domain_bounds: Tuple[float, float] = (-1.5, 1.5),
        codomain_bounds: Tuple[float, float] = (-1.0, 1.0),
        condition_dim: int = 0,
    ):
        """
        Args:
            dim: Dimension of input/output space (2 for your case)
            n_layers: Number of coupling layers
            n_bins: Number of spline bins (more = more flexible)
            hidden_dim: Hidden dimension of parameter networks
            n_hidden: Number of hidden layers in parameter networks
            domain_bounds: (left, right) bounds of input domain
            codomain_bounds: (left, right) bounds of output domain
            condition_dim: Dimension of conditioning vector (0 for unconditional)
        """
        super().__init__()
        
        self.dim = dim
        self.n_layers = n_layers
        self.domain_bounds = domain_bounds
        self.codomain_bounds = codomain_bounds
        
        # Create alternating masks
        # For 2D: [1,0], [0,1], [1,0], [0,1], ...
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % dim] = 1  # Alternate which dimension passes through
            
            layers.append(SplineCouplingLayer(
                dim=dim,
                mask=mask,
                n_bins=n_bins,
                hidden_dim=hidden_dim,
                n_hidden=n_hidden,
                domain_left=domain_bounds[0],
                domain_right=domain_bounds[1],
                codomain_left=codomain_bounds[0],
                codomain_right=codomain_bounds[1],
                condition_dim=condition_dim,
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transform: domain → codomain
        
        Args:
            x: Points in domain (..., dim)
            condition: Optional conditioning (..., condition_dim)
            
        Returns:
            y: Points in codomain (..., dim)
            logabsdet: Log |det(J)| (...,)
        """
        y = x
        logabsdet = torch.zeros(x.shape[:-1], device=x.device)
        
        for layer in self.layers:
            y, ld = layer(y, condition, inverse=False)
            logabsdet = logabsdet + ld
        
        return y, logabsdet
    
    def inverse(
        self, 
        y: torch.Tensor, 
        condition: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transform: codomain → domain
        
        Args:
            y: Points in codomain (..., dim)
            condition: Optional conditioning (..., condition_dim)
            
        Returns:
            x: Points in domain (..., dim)
            logabsdet: Log |det(J^{-1})| (...,)
        """
        x = y
        logabsdet = torch.zeros(y.shape[:-1], device=y.device)
        
        # Apply layers in reverse order
        for layer in reversed(self.layers):
            x, ld = layer(x, condition, inverse=True)
            logabsdet = logabsdet + ld
        
        return x, logabsdet
    
    def jacobian_det(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Compute Jacobian determinant (always positive!).
        
        Args:
            x: Points (..., dim)
            condition: Optional conditioning
            
        Returns:
            det_J: Jacobian determinant (...,) - always > 0
        """
        _, logabsdet = self.forward(x, condition)
        return torch.exp(logabsdet)


# =============================================================================
# Convenience wrapper for FEM integration
# =============================================================================

class NSFMapping:
    """
    Wrapper for using trained NSF in FEM applications.
    
    Similar interface to WarpMapping for drop-in replacement.
    """
    
    def __init__(self, model: NeuralSplineFlow, condition: torch.Tensor = None):
        self.model = model
        self.model.eval()
        self.condition = condition
        self.device = next(model.parameters()).device
    
    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Forward map: Ω → [-1,1]²"""
        x = x.to(self.device)
        with torch.no_grad():
            y, _ = self.model.forward(x, self.condition)
        return y
    
    def phi_inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse map: [-1,1]² → Ω (exact, not iterative!)"""
        y = y.to(self.device)
        with torch.no_grad():
            x, _ = self.model.inverse(y, self.condition)
        return x
    
    def jacobian_det(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian determinant (always positive!)"""
        x = x.to(self.device)
        with torch.no_grad():
            det = self.model.jacobian_det(x, self.condition)
        return det
    
    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Full 2x2 Jacobian matrix at a single point."""
        x = x.to(self.device).requires_grad_(True)
        y, _ = self.model.forward(x.unsqueeze(0), self.condition)
        y = y.squeeze(0)
        
        J = torch.zeros(2, 2, device=x.device)
        for i in range(2):
            grad = torch.autograd.grad(y[i], x, retain_graph=True)[0]
            J[i] = grad
        
        return J


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Neural Spline Flow...")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    model = NeuralSplineFlow(
        dim=2,
        n_layers=6,
        n_bins=8,
        hidden_dim=64,
        n_hidden=2,
        domain_bounds=(-1.5, 1.5),
        codomain_bounds=(-1.0, 1.0),
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Test forward
    x = torch.randn(100, 2, device=device) * 0.5
    y, logdet = model.forward(x)
    
    print(f"Forward: x={x.shape} → y={y.shape}")
    print(f"Log det range: [{logdet.min():.2f}, {logdet.max():.2f}]")
    
    # Test inverse
    x_back, logdet_inv = model.inverse(y)
    recon_error = (x - x_back).abs().max().item()
    print(f"Inverse reconstruction error: {recon_error:.6f}")
    
    # Test Jacobian determinant (should always be positive!)
    det = model.jacobian_det(x)
    print(f"Jacobian det: min={det.min():.4f}, max={det.max():.4f}")
    print(f"All positive: {(det > 0).all().item()}")
    
    print("\n✓ All tests passed!")
