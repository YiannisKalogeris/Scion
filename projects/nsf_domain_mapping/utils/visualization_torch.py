"""
Visualization utilities for diffeomorphism networks (PyTorch version).

Provides functions to visualize:
    - Learned mappings (domain → square)
    - Grid deformations
    - Point trajectories through the ODE flow
    - Jacobian determinant fields
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Optional, Tuple
import sys


def visualize_mapping(
    model,
    image: torch.Tensor,
    interior_points: torch.Tensor,
    boundary_points: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Diffeomorphism Mapping",
    device: str = 'cpu',
):
    """
    Visualize the learned diffeomorphism for a single shape.
    
    Creates a 2x2 figure:
        - Top left: SDF image with sampled points
        - Top right: Mapped points in target square
        - Bottom left: Grid in original domain
        - Bottom right: Deformed grid in target square
    
    Args:
        model: Trained DiffeomorphismNet
        image: (H, W, 1) SDF image tensor
        interior_points: (N, 2) points inside the shape
        boundary_points: (M, 2) points on boundary
        save_path: Path to save figure (optional)
        title: Figure title
        device: Device for computation
    """
    model.eval()
    model = model.to(device)
    image = image.to(device)
    interior_points = interior_points.to(device)
    boundary_points = boundary_points.to(device)
    
    with torch.no_grad():
        # Encode shape
        z = model.encode(image)
        
        # Map interior points
        z_int = z.unsqueeze(0).expand(interior_points.size(0), -1)
        mapped_interior = model.forward_map(interior_points, z_int).cpu().numpy()
        
        # Map boundary points
        z_bnd = z.unsqueeze(0).expand(boundary_points.size(0), -1)
        mapped_boundary = model.forward_map(boundary_points, z_bnd).cpu().numpy()
    
    # Convert to numpy for plotting
    interior_np = interior_points.cpu().numpy()
    boundary_np = boundary_points.cpu().numpy()
    image_np = image.cpu().numpy().squeeze()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=14)
    
    # Top left: Original domain with points
    ax = axes[0, 0]
    ax.imshow(image_np, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu', alpha=0.7)
    ax.scatter(interior_np[:, 0], interior_np[:, 1], c='blue', s=10, alpha=0.6, label='Interior')
    ax.scatter(boundary_np[:, 0], boundary_np[:, 1], c='red', s=15, alpha=0.8, label='Boundary')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Original Domain')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Mapped points in target square
    ax = axes[0, 1]
    ax.scatter(mapped_interior[:, 0], mapped_interior[:, 1], c='blue', s=10, alpha=0.6, label='Interior')
    ax.scatter(mapped_boundary[:, 0], mapped_boundary[:, 1], c='red', s=15, alpha=0.8, label='Boundary')
    # Draw target square
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(square)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Mapped to Target Square')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Grid in original domain
    ax = axes[1, 0]
    ax.imshow(image_np, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu', alpha=0.5)
    
    # Create grid lines inside the shape
    n_grid = 15
    grid_x = np.linspace(-1.5, 1.5, n_grid)
    grid_y = np.linspace(-1.5, 1.5, n_grid)
    
    # Draw grid lines
    for x in grid_x:
        ax.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
    for y in grid_y:
        ax.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Grid in Original Domain')
    
    # Bottom right: Deformed grid
    ax = axes[1, 1]
    
    # Create grid points and map them
    with torch.no_grad():
        # Horizontal lines
        for y_val in grid_y:
            line_points = torch.tensor([[x, y_val] for x in np.linspace(-1.5, 1.5, 50)], 
                                       dtype=torch.float32, device=device)
            z_line = z.unsqueeze(0).expand(line_points.size(0), -1)
            mapped_line = model.forward_map(line_points, z_line).cpu().numpy()
            ax.plot(mapped_line[:, 0], mapped_line[:, 1], 'b-', linewidth=0.5, alpha=0.7)
        
        # Vertical lines
        for x_val in grid_x:
            line_points = torch.tensor([[x_val, y] for y in np.linspace(-1.5, 1.5, 50)], 
                                       dtype=torch.float32, device=device)
            z_line = z.unsqueeze(0).expand(line_points.size(0), -1)
            mapped_line = model.forward_map(line_points, z_line).cpu().numpy()
            ax.plot(mapped_line[:, 0], mapped_line[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    
    # Draw target square
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(square)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Deformed Grid in Target')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    return fig


def visualize_trajectory(
    model,
    image: torch.Tensor,
    start_points: torch.Tensor,
    n_steps: int = 50,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Visualize point trajectories through the ODE flow.
    
    Args:
        model: Trained DiffeomorphismNet
        image: (H, W, 1) SDF image tensor
        start_points: (N, 2) starting points
        n_steps: Number of trajectory steps to show
        save_path: Path to save figure
        device: Device for computation
    """
    model.eval()
    model = model.to(device)
    image = image.to(device)
    start_points = start_points.to(device)
    
    with torch.no_grad():
        z = model.encode(image)
        
        # Get trajectories
        trajectories = []
        for point in start_points:
            z_exp = z.unsqueeze(0)
            traj = model.forward_map(point.unsqueeze(0), z_exp, return_trajectory=True)
            trajectories.append(traj.squeeze(1).cpu().numpy())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    image_np = image.cpu().numpy().squeeze()
    start_np = start_points.cpu().numpy()
    
    # Left: trajectories in original space (conceptual)
    ax = axes[0]
    ax.imshow(image_np, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu', alpha=0.5)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], linewidth=2, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], c='green', s=50, zorder=5, marker='o')
        ax.scatter(traj[-1, 0], traj[-1, 1], c='red', s=50, zorder=5, marker='x')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('Point Trajectories (t: 0→1)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Right: final positions
    ax = axes[1]
    for i, traj in enumerate(trajectories):
        ax.scatter(traj[-1, 0], traj[-1, 1], c=[colors[i]], s=100, marker='o')
    
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='green', linewidth=2, linestyle='--')
    ax.add_patch(square)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Final Mapped Positions')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    return fig


def visualize_jacobian_field(
    model,
    image: torch.Tensor,
    grid_size: int = 20,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Visualize Jacobian determinant field.
    
    Args:
        model: Trained DiffeomorphismNet
        image: (H, W, 1) SDF image tensor
        grid_size: Number of points per dimension
        save_path: Path to save figure
        device: Device for computation
    """
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
    from network import DiffeomorphismMapping
    
    model.eval()
    model = model.to(device)
    image = image.to(device)
    
    mapping = DiffeomorphismMapping(model, image)
    
    # Create grid
    x = np.linspace(-1.5, 1.5, grid_size)
    y = np.linspace(-1.5, 1.5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Compute Jacobian determinant at each point
    det_J = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32, device=device)
            det_J[i, j] = mapping.det_jacobian(point).cpu().item()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    image_np = image.cpu().numpy().squeeze()
    
    # Left: SDF image
    ax = axes[0]
    ax.imshow(image_np, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu')
    ax.set_title('Shape (SDF)')
    ax.set_aspect('equal')
    
    # Right: Jacobian determinant
    ax = axes[1]
    im = ax.imshow(det_J, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower', cmap='RdYlGn', 
                   vmin=-np.abs(det_J).max(), vmax=np.abs(det_J).max())
    ax.contour(X, Y, det_J, levels=[0], colors='black', linewidths=2)
    plt.colorbar(im, ax=ax, label='det(J)')
    ax.set_title(f'Jacobian Determinant\n(min={det_J.min():.3f}, max={det_J.max():.3f})')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    return fig


def visualize_inverse_mapping(
    model,
    image: torch.Tensor,
    n_points: int = 100,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Visualize forward and inverse mapping consistency.
    
    Args:
        model: Trained DiffeomorphismNet
        image: (H, W, 1) SDF image tensor
        n_points: Number of random test points
        save_path: Path to save figure
        device: Device for computation
    """
    model.eval()
    model = model.to(device)
    image = image.to(device)
    
    # Generate random points in domain
    points = (torch.rand(n_points, 2, device=device) - 0.5) * 2  # [-1, 1]
    
    with torch.no_grad():
        z = model.encode(image)
        z_exp = z.unsqueeze(0).expand(n_points, -1)
        
        # Forward map
        mapped = model.forward_map(points, z_exp)
        
        # Inverse map
        recovered = model.inverse_map(mapped, z_exp)
    
    points_np = points.cpu().numpy()
    mapped_np = mapped.cpu().numpy()
    recovered_np = recovered.cpu().numpy()
    
    # Compute errors
    errors = np.linalg.norm(points_np - recovered_np, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original points
    ax = axes[0]
    ax.scatter(points_np[:, 0], points_np[:, 1], c='blue', s=20, alpha=0.7)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Original Points')
    ax.grid(True, alpha=0.3)
    
    # Mapped points
    ax = axes[1]
    ax.scatter(mapped_np[:, 0], mapped_np[:, 1], c='green', s=20, alpha=0.7)
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(square)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('Forward Mapped φ(x)')
    ax.grid(True, alpha=0.3)
    
    # Recovered points with error coloring
    ax = axes[2]
    sc = ax.scatter(recovered_np[:, 0], recovered_np[:, 1], c=errors, s=20, 
                    cmap='hot_r', alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Recovery error')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(f'Recovered φ⁻¹(φ(x))\nMean error: {errors.mean():.4f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.show()
    return fig


def run_all_visualizations(
    model,
    dataset,
    output_dir: str,
    n_samples: int = 5,
    device: str = 'cpu',
):
    """
    Run all visualizations on multiple samples.
    
    Args:
        model: Trained DiffeomorphismNet
        dataset: DiffeomorphismDataset
        output_dir: Directory to save figures
        n_samples: Number of samples to visualize
        device: Device for computation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations for {n_samples} samples...")
    
    for i in range(min(n_samples, len(dataset))):
        sample = dataset[i]
        
        image = torch.tensor(sample.image, dtype=torch.float32)
        interior = torch.tensor(sample.interior_points, dtype=torch.float32)
        boundary = torch.tensor(sample.boundary_points, dtype=torch.float32)
        
        print(f"\n  Sample {i+1}/{n_samples}")
        
        # Main mapping visualization
        visualize_mapping(
            model, image, interior, boundary,
            save_path=str(output_dir / f"mapping_sample_{i}.png"),
            title=f"Sample {i+1}: Diffeomorphism Mapping",
            device=device,
        )
        
        # Trajectory visualization (subset of points)
        n_traj = min(10, len(interior))
        visualize_trajectory(
            model, image, interior[:n_traj],
            save_path=str(output_dir / f"trajectory_sample_{i}.png"),
            device=device,
        )
        
        # Inverse mapping check
        visualize_inverse_mapping(
            model, image, n_points=50,
            save_path=str(output_dir / f"inverse_sample_{i}.png"),
            device=device,
        )
    
    print(f"\n  All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    print("Visualization utilities loaded.")
    print("Use run_all_visualizations(model, dataset, output_dir) to generate figures.")
