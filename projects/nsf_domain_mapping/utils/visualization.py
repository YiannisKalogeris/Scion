"""
Visualization utilities for diffeomorphism networks.

Provides functions for:
    - Visualizing learned mappings as deformed grids
    - Plotting training history
    - Comparing ground truth vs predicted mappings
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional, Tuple, Any
from pathlib import Path


class _MappingHelper:
    """Simple helper class to wrap model for visualization (avoids import issues)."""
    
    def __init__(self, model, image):
        self.model = model
        self.z = model.encode(image)
    
    def phi(self, x):
        return self.model.forward_map(x, self.z)
    
    def phi_inv(self, y):
        return self.model.inverse_map(y, self.z)
    
    def jacobian(self, x):
        return jax.jacobian(self.phi)(x)
    
    def det_jacobian(self, x):
        return jnp.linalg.det(self.jacobian(x))
    
    def phi_batch(self, points):
        return jax.vmap(self.phi)(points)
    
    def det_jacobian_batch(self, points):
        return jax.vmap(self.det_jacobian)(points)


def visualize_mapping(
    model,  # DiffeomorphismNet
    image: jnp.ndarray,
    n_grid: int = 15,
    save_path: Optional[str] = None,
    title: str = "Learned Diffeomorphism",
) -> plt.Figure:
    """
    Visualize the learned diffeomorphism as a deformed grid.
    
    Shows:
        1. Input SDF image
        2. Reference square with grid
        3. Grid mapped to physical domain (inverse map)
    """
    z = model.encode(image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Input SDF image
    im = axes[0].imshow(
        image[:, :, 0], 
        extent=[-2, 2, -2, 2], 
        origin='lower',
        cmap='RdBu',
        vmin=-1, vmax=1
    )
    axes[0].contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2])
    axes[0].set_title("Input Domain (SDF)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0], label="SDF")
    
    # Middle: Reference square with grid
    x = np.linspace(-0.95, 0.95, n_grid)
    y = np.linspace(-0.95, 0.95, n_grid)
    
    for xi in x:
        axes[1].axvline(xi, color='blue', alpha=0.5, lw=0.8)
    for yi in y:
        axes[1].axhline(yi, color='blue', alpha=0.5, lw=0.8)
    
    # Draw square boundary
    rect = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', lw=2)
    axes[1].add_patch(rect)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_title("Reference Square")
    axes[1].set_xlabel("ξ")
    axes[1].set_ylabel("η")
    axes[1].set_aspect('equal')
    
    # Right: Inverse mapped grid (square → domain)
    for xi in x:
        pts = jnp.stack([jnp.full(n_grid, xi), jnp.array(y)], axis=-1)
        mapped = jax.vmap(lambda p: model.inverse_map(p, z))(pts)
        axes[2].plot(mapped[:, 0], mapped[:, 1], 'b-', alpha=0.6, lw=0.8)
    
    for yi in y:
        pts = jnp.stack([jnp.array(x), jnp.full(n_grid, yi)], axis=-1)
        mapped = jax.vmap(lambda p: model.inverse_map(p, z))(pts)
        axes[2].plot(mapped[:, 0], mapped[:, 1], 'b-', alpha=0.6, lw=0.8)
    
    # Draw domain boundary (zero contour of SDF)
    axes[2].contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2], linewidths=2)
    axes[2].set_xlim(-2.2, 2.2)
    axes[2].set_ylim(-2.2, 2.2)
    axes[2].set_title("Mapped Grid on Domain")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect('equal')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def visualize_forward_mapping(
    model,  # DiffeomorphismNet
    image: jnp.ndarray,
    interior_points: jnp.ndarray,
    ground_truth: Optional[jnp.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize forward mapping: domain → square.
    
    Shows points in original domain and their mapped locations.
    """
    z = model.encode(image)
    
    # Map points
    mapped = jax.vmap(lambda p: model.forward_map(p, z))(interior_points)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Points in domain
    axes[0].imshow(image[:, :, 0], extent=[-2, 2, -2, 2], origin='lower', cmap='gray', alpha=0.3)
    axes[0].contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2])
    axes[0].scatter(interior_points[:, 0], interior_points[:, 1], c='blue', s=10, alpha=0.6)
    axes[0].set_title("Points in Domain")
    axes[0].set_aspect('equal')
    
    # Middle: Mapped points (predicted)
    rect = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', lw=2)
    axes[1].add_patch(rect)
    axes[1].scatter(mapped[:, 0], mapped[:, 1], c='blue', s=10, alpha=0.6, label='Predicted')
    if ground_truth is not None:
        axes[1].scatter(ground_truth[:, 0], ground_truth[:, 1], c='red', s=10, alpha=0.3, label='Ground Truth')
        axes[1].legend()
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_title("Mapped Points")
    axes[1].set_aspect('equal')
    
    # Right: Error distribution (if ground truth available)
    if ground_truth is not None:
        errors = jnp.linalg.norm(mapped - ground_truth, axis=-1)
        im = axes[2].scatter(
            interior_points[:, 0], interior_points[:, 1], 
            c=errors, s=20, cmap='hot', alpha=0.8
        )
        axes[2].contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2])
        axes[2].set_title(f"Mapping Error (mean={float(jnp.mean(errors)):.4f})")
        plt.colorbar(im, ax=axes[2], label='Error')
    else:
        axes[2].text(0.5, 0.5, 'No ground truth\navailable', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title("Error Distribution")
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_jacobian(
    model,  # DiffeomorphismNet
    image: jnp.ndarray,
    n_grid: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize Jacobian determinant over the domain.
    
    Positive everywhere indicates valid diffeomorphism.
    """
    mapping = _MappingHelper(model, image)
    
    # Create grid of points inside domain
    x = np.linspace(-1.8, 1.8, n_grid)
    y = np.linspace(-1.8, 1.8, n_grid)
    X, Y = np.meshgrid(x, y)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    
    # Check which points are inside (SDF < 0)
    from scipy.interpolate import RegularGridInterpolator
    x_img = np.linspace(-2, 2, image.shape[1])
    y_img = np.linspace(-2, 2, image.shape[0])
    sdf_interp = RegularGridInterpolator((y_img, x_img), np.array(image[:, :, 0]))
    
    inside_mask = sdf_interp(points[:, ::-1]) < 0
    
    # Compute Jacobian determinants for inside points
    det_J = np.full(len(points), np.nan)
    inside_points = points[inside_mask]
    
    if len(inside_points) > 0:
        det_values = mapping.det_jacobian_batch(inside_points)
        det_J[inside_mask] = np.array(det_values)
    
    det_J = det_J.reshape(n_grid, n_grid)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Jacobian determinant heatmap
    im = axes[0].imshow(
        det_J, extent=[-1.8, 1.8, -1.8, 1.8], origin='lower',
        cmap='RdYlGn', vmin=0, vmax=2
    )
    axes[0].contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2], linewidths=2)
    axes[0].set_title("Jacobian Determinant")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0], label='det(J)')
    
    # Right: Histogram of determinants
    valid_dets = det_J[~np.isnan(det_J)]
    axes[1].hist(valid_dets.ravel(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', label='det(J) = 0')
    axes[1].axvline(1, color='green', linestyle='--', label='det(J) = 1 (area preserving)')
    axes[1].set_xlabel("Jacobian Determinant")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"det(J) Distribution\nmin={valid_dets.min():.3f}, max={valid_dets.max():.3f}")
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train', alpha=0.8)
    if history['val_loss']:
        val_epochs = np.linspace(0, len(history['train_loss']), len(history['val_loss']))
        axes[0].plot(val_epochs, history['val_loss'], label='Validation', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Progress')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Loss components
    if history['loss_components']:
        components = history['loss_components']
        component_names = list(components[0].keys())
        
        for name in component_names:
            values = [c[name] for c in components]
            epochs = np.linspace(0, len(history['train_loss']), len(values))
            axes[1].plot(epochs, values, label=name, alpha=0.8)
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss Component')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_animation_frames(
    model,  # DiffeomorphismNet
    image: jnp.ndarray,
    output_dir: str,
    n_frames: int = 50,
    n_grid: int = 15,
):
    """
    Create animation frames showing the flow from domain to square.
    
    Use with ffmpeg to create video:
        ffmpeg -framerate 10 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p flow.mp4
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    z = model.encode(image)
    
    # Create grid of points in the square
    x = np.linspace(-0.95, 0.95, n_grid)
    y = np.linspace(-0.95, 0.95, n_grid)
    
    # Get trajectory for each gridline
    def get_trajectory(pt):
        return model.forward_map(pt, z, return_trajectory=True)
    
    # Collect all trajectories
    trajectories_h = []  # Horizontal lines
    trajectories_v = []  # Vertical lines
    
    print("Computing trajectories...")
    for xi in x:
        pts = jnp.stack([jnp.full(n_grid, xi), jnp.array(y)], axis=-1)
        # Use inverse map trajectories starting from square
        traj = jax.vmap(get_trajectory)(pts)  # (n_grid, n_times, 2)
        trajectories_v.append(traj)
    
    for yi in y:
        pts = jnp.stack([jnp.array(x), jnp.full(n_grid, yi)], axis=-1)
        traj = jax.vmap(get_trajectory)(pts)
        trajectories_h.append(traj)
    
    n_times = trajectories_h[0].shape[1]
    
    print(f"Creating {n_frames} frames...")
    for frame_idx in range(n_frames):
        t_idx = int(frame_idx * (n_times - 1) / (n_frames - 1))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot domain boundary
        ax.contour(image[:, :, 0], levels=[0], colors='black', extent=[-2, 2, -2, 2], linewidths=2)
        
        # Plot grid at current time
        for traj in trajectories_v:
            pts = traj[:, t_idx, :]
            ax.plot(pts[:, 0], pts[:, 1], 'b-', alpha=0.6, lw=0.8)
        
        for traj in trajectories_h:
            pts = traj[:, t_idx, :]
            ax.plot(pts[:, 0], pts[:, 1], 'b-', alpha=0.6, lw=0.8)
        
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.set_title(f"t = {frame_idx / (n_frames - 1):.2f}")
        
        plt.savefig(output_dir / f"frame_{frame_idx:03d}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Frames saved to {output_dir}")
    print(f"Create video with: ffmpeg -framerate 10 -i {output_dir}/frame_%03d.png -c:v libx264 flow.mp4")


if __name__ == "__main__":
    print("Visualization module loaded. Import and use in main.py.")
