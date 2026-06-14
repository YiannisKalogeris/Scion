"""
Loss functions for training diffeomorphism networks (PyTorch version).

Loss components:
    - supervised_loss: MSE between predicted and ground truth mappings
    - boundary_loss: Boundary points should map to square boundary
    - jacobian_loss: Jacobian determinant should be positive (bijectivity)
    - cycle_loss: Forward-backward consistency
    - coverage_loss: Mapped points should stay in [-1,1]²
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


def supervised_loss(
    model,
    image: torch.Tensor,
    points: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised loss: MSE between predicted and ground truth mapped points.
    
    Args:
        model: DiffeomorphismNet
        image: (H, W, C) SDF image
        points: (N, 2) interior points
        targets: (N, 2) ground truth mapped locations
    """
    # Guard against empty points
    if points.size(0) == 0:
        return torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    z = model.encode(image)
    z_expanded = z.unsqueeze(0).expand(points.size(0), -1)
    predicted = model.forward_map(points, z_expanded)
    return F.mse_loss(predicted, targets)


def boundary_loss(
    model,
    image: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_targets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Boundary should map to square boundary.
    
    Args:
        model: DiffeomorphismNet
        image: (H, W, C) SDF image
        boundary_points: (M, 2) points on domain boundary
        boundary_targets: (M, 2) optional explicit boundary targets
    """
    # Guard against empty points
    if boundary_points.size(0) == 0:
        return torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    z = model.encode(image)
    z_expanded = z.unsqueeze(0).expand(boundary_points.size(0), -1)
    mapped = model.forward_map(boundary_points, z_expanded)
    
    if boundary_targets is not None:
        # Direct supervision if targets available
        return F.mse_loss(mapped, boundary_targets)
    else:
        # Soft constraint: distance to boundary of [-1,1]²
        dist_to_boundary = torch.abs(torch.max(torch.abs(mapped), dim=-1).values - 1.0)
        return torch.mean(dist_to_boundary ** 2)


def jacobian_loss(
    model,
    image: torch.Tensor,
    points: torch.Tensor,
    min_det: float = 0.01,
) -> torch.Tensor:
    """
    Regularize Jacobian determinant to stay positive.
    Uses finite differences to avoid second-order autodiff through ODE.
    
    Args:
        model: DiffeomorphismNet
        image: (H, W, C) SDF image
        points: (N, 2) interior points
        min_det: minimum allowed determinant
    """
    # Guard against empty points
    if points.size(0) == 0:
        return torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    z = model.encode(image)
    eps = 1e-4
    
    def compute_det(x):
        # Finite difference Jacobian
        x = x.unsqueeze(0)
        z_exp = z.unsqueeze(0)
        
        fx = model.forward_map(x, z_exp, use_adjoint=False).squeeze(0)
        
        x_dx = x + torch.tensor([[eps, 0.0]], device=x.device, dtype=torch.float32)
        x_dy = x + torch.tensor([[0.0, eps]], device=x.device, dtype=torch.float32)
        
        f_dx = model.forward_map(x_dx, z_exp, use_adjoint=False).squeeze(0)
        f_dy = model.forward_map(x_dy, z_exp, use_adjoint=False).squeeze(0)
        
        # J = [[df1/dx, df1/dy], [df2/dx, df2/dy]]
        j00 = (f_dx[0] - fx[0]) / eps
        j01 = (f_dy[0] - fx[0]) / eps
        j10 = (f_dx[1] - fx[1]) / eps
        j11 = (f_dy[1] - fx[1]) / eps
        
        det = j00 * j11 - j01 * j10
        return det
    
    dets = torch.stack([compute_det(p) for p in points])
    
    # Penalize determinants below threshold
    violation = F.relu(min_det - dets)
    return torch.mean(violation ** 2)


def cycle_consistency_loss(
    model,
    image: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Forward-backward consistency loss.
    Uses stop_gradient on inverse to avoid double backprop through ODE.
    
    Args:
        model: DiffeomorphismNet
        image: (H, W, C) SDF image
        points: (N, 2) interior points
    """
    # Guard against empty points
    if points.size(0) == 0:
        return torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    z = model.encode(image)
    z_expanded = z.unsqueeze(0).expand(points.size(0), -1)
    
    # Forward map
    mapped = model.forward_map(points, z_expanded)
    
    # Inverse map (with no gradient)
    with torch.no_grad():
        recovered = model.inverse_map(mapped, z_expanded)
    
    return F.mse_loss(points, recovered)


def coverage_loss(
    model,
    image: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize mapped points escaping [-1, 1]².
    
    Args:
        model: DiffeomorphismNet
        image: (H, W, C) SDF image
        points: (N, 2) interior points
    """
    # Guard against empty points
    if points.size(0) == 0:
        return torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    z = model.encode(image)
    z_expanded = z.unsqueeze(0).expand(points.size(0), -1)
    mapped = model.forward_map(points, z_expanded)
    
    # Soft penalty for escaping
    excess = F.relu(torch.abs(mapped) - 1.0)
    return torch.mean(excess ** 2)


def total_loss(
    model,
    batch: dict,
    weights: Optional[Dict[str, float]] = None,
    min_jacobian_det: float = 0.01,
) -> torch.Tensor:
    """
    Compute total weighted loss.
    
    Args:
        model: DiffeomorphismNet
        batch: dict with keys: image, interior_points, interior_targets, 
               boundary_points, boundary_targets
        weights: dict of loss weights
        min_jacobian_det: threshold for Jacobian regularization
    """
    if weights is None:
        weights = {
            'supervised': 1.0,
            'boundary': 10.0,
            'jacobian': 0.1,
            'cycle': 1.0,
            'coverage': 0.5,
        }
    
    image = batch['image']
    interior_pts = batch['interior_points']
    interior_tgt = batch['interior_targets']
    boundary_pts = batch['boundary_points']
    boundary_tgt = batch['boundary_targets']
    
    total = torch.tensor(0.0, device=image.device, dtype=torch.float32)
    
    if weights.get('supervised', 0) > 0:
        loss_sup = supervised_loss(model, image, interior_pts, interior_tgt)
        total = total + weights['supervised'] * loss_sup
    
    if weights.get('boundary', 0) > 0:
        loss_bnd = boundary_loss(model, image, boundary_pts, boundary_tgt)
        total = total + weights['boundary'] * loss_bnd
    
    if weights.get('jacobian', 0) > 0:
        # Use subset of points for efficiency
        n_jac = min(32, interior_pts.size(0))
        loss_jac = jacobian_loss(model, image, interior_pts[:n_jac], min_jacobian_det)
        total = total + weights['jacobian'] * loss_jac
    
    if weights.get('cycle', 0) > 0:
        # Use subset of points for efficiency
        n_cyc = min(32, interior_pts.size(0))
        loss_cyc = cycle_consistency_loss(model, image, interior_pts[:n_cyc])
        total = total + weights['cycle'] * loss_cyc
    
    if weights.get('coverage', 0) > 0:
        loss_cov = coverage_loss(model, image, interior_pts)
        total = total + weights['coverage'] * loss_cov
    
    return total


def compute_loss_components(
    model,
    batch: dict,
    min_jacobian_det: float = 0.01,
) -> Dict[str, float]:
    """
    Compute individual loss components for logging.
    """
    image = batch['image']
    interior_pts = batch['interior_points']
    interior_tgt = batch['interior_targets']
    boundary_pts = batch['boundary_points']
    boundary_tgt = batch['boundary_targets']
    
    with torch.no_grad():
        components = {
            'supervised': supervised_loss(model, image, interior_pts, interior_tgt).item(),
            'boundary': boundary_loss(model, image, boundary_pts, boundary_tgt).item(),
            'coverage': coverage_loss(model, image, interior_pts).item(),
        }
        
        # These are expensive, use fewer points
        n_sub = min(16, interior_pts.size(0))
        
        model_eval = model  # Use same model but with no_grad
        components['jacobian'] = jacobian_loss(model_eval, image, interior_pts[:n_sub], min_jacobian_det).item()
        components['cycle'] = cycle_consistency_loss(model_eval, image, interior_pts[:n_sub]).item()
    
    return components


if __name__ == "__main__":
    from network import DiffeomorphismNet
    
    device = 'cpu'
    model = DiffeomorphismNet(latent_dim=64, hidden_dim=64, n_layers=2).to(device)
    
    # Create dummy batch
    batch = {
        'image': torch.randn(64, 64, 1, device=device, dtype=torch.float32),
        'interior_points': torch.randn(32, 2, device=device, dtype=torch.float32) * 0.5,
        'interior_targets': torch.randn(32, 2, device=device, dtype=torch.float32) * 0.5,
        'boundary_points': torch.randn(16, 2, device=device, dtype=torch.float32) * 0.5,
        'boundary_targets': torch.randn(16, 2, device=device, dtype=torch.float32) * 0.5,
    }
    
    loss = total_loss(model, batch)
    print(f"Total loss: {loss.item():.4f}")
    
    components = compute_loss_components(model, batch)
    print(f"Components: {components}")
    
    # Test backward
    loss.backward()
    print("✓ Backward pass works!")