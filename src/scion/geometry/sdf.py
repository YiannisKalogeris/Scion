"""
Signed Distance Field (SDF) computation and rendering.

SDF provides richer information than binary masks:
    - Negative inside, positive outside, zero on boundary
    - Gradient points toward nearest boundary point
    - Useful for the neural network to understand shape geometry
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from scion.geometry.shapes import ClosedCurve



def compute_sdf_from_boundary(
    boundary_points: np.ndarray,
    grid_size: int = 64,
    bounds: Tuple[float, float] = (-2.0, 2.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute signed distance field on a regular grid.
    
    Args:
        boundary_points: (N, 2) array of points on the boundary
        grid_size: Number of grid points per dimension
        bounds: Domain bounds (min, max)
        
    Returns:
        sdf: (grid_size, grid_size) signed distance field
        x_grid: 1D array of x coordinates
        y_grid: 1D array of y coordinates
    """
    x_grid = np.linspace(bounds[0], bounds[1], grid_size)
    y_grid = np.linspace(bounds[0], bounds[1], grid_size)
    
    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (grid_size², 2)
    
    # Compute unsigned distance to boundary
    # For each grid point, find distance to nearest boundary point
    distances = np.linalg.norm(
        grid_points[:, None, :] - boundary_points[None, :, :], 
        axis=-1
    )  # (grid_size², N_boundary)
    unsigned_dist = np.min(distances, axis=-1)  # (grid_size²,)
    
    # Determine sign using winding number
    signs = _compute_winding_sign(grid_points, boundary_points)
    
    sdf = signs * unsigned_dist
    sdf = sdf.reshape(grid_size, grid_size)
    
    return sdf, x_grid, y_grid


def _compute_winding_sign(
    query_points: np.ndarray, 
    boundary_points: np.ndarray
) -> np.ndarray:
    """
    Compute sign of SDF using ray casting (crossing number algorithm).
    
    Returns:
        signs: -1 for inside, +1 for outside
    """
    n_query = len(query_points)
    n_boundary = len(boundary_points)
    
    signs = np.ones(n_query)
    
    for i in range(n_query):
        p = query_points[i]
        crossings = 0
        
        for j in range(n_boundary):
            p1 = boundary_points[j]
            p2 = boundary_points[(j + 1) % n_boundary]
            
            # Check if ray from p in +x direction crosses edge p1-p2
            if ((p1[1] > p[1]) != (p2[1] > p[1])):
                # Compute x-coordinate of intersection
                x_intersect = (p2[0] - p1[0]) * (p[1] - p1[1]) / (p2[1] - p1[1] + 1e-10) + p1[0]
                if p[0] < x_intersect:
                    crossings += 1
        
        if crossings % 2 == 1:
            signs[i] = -1.0  # Inside
    
    return signs


def render_shape_to_sdf(
    shape: ClosedCurve,
    grid_size: int = 64,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    n_boundary_samples: int = 500,
) -> np.ndarray:
    """
    Render a shape to a signed distance field image.
    
    Args:
        shape: ClosedCurve instance
        grid_size: Output image size (grid_size × grid_size)
        bounds: Domain bounds
        n_boundary_samples: Number of points to sample on boundary
        
    Returns:
        sdf: (grid_size, grid_size) SDF image, negative inside
    """
    boundary = shape.sample(n_boundary_samples)
    sdf, _, _ = compute_sdf_from_boundary(boundary, grid_size, bounds)
    return sdf


def render_shape_to_binary(
    shape: ClosedCurve,
    grid_size: int = 64,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    n_boundary_samples: int = 500,
) -> np.ndarray:
    """
    Render a shape to a binary mask (1 inside, 0 outside).
    """
    sdf = render_shape_to_sdf(shape, grid_size, bounds, n_boundary_samples)
    return (sdf < 0).astype(np.float32)


def normalize_sdf(sdf: np.ndarray, method: str = "tanh") -> np.ndarray:
    """
    Normalize SDF values to [-1, 1] range for neural network input.
    
    Args:
        sdf: Raw SDF values
        method: 
            "tanh": Apply tanh scaling
            "clip": Clip to [-1, 1]
            "linear": Linear scaling based on max distance
    """
    if method == "tanh":
        # Smooth normalization, preserves sign
        scale = np.percentile(np.abs(sdf), 95) + 1e-6
        return np.tanh(sdf / scale)
    
    elif method == "clip":
        max_dist = np.percentile(np.abs(sdf), 95) + 1e-6
        return np.clip(sdf / max_dist, -1, 1)
    
    elif method == "linear":
        max_dist = np.max(np.abs(sdf)) + 1e-6
        return sdf / max_dist
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def sample_points_in_shape(
    shape: ClosedCurve,
    n_points: int,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample random points inside a shape using rejection sampling.
    
    Args:
        shape: ClosedCurve instance
        n_points: Number of interior points to sample
        bounds: Domain bounds for initial sampling
        rng: Random number generator
        
    Returns:
        points: (n_points, 2) array of interior points
    """
    rng = rng or np.random.default_rng()
    
    # Get boundary for inside/outside testing
    boundary = shape.sample(500)
    
    points = []
    n_attempts = 0
    max_attempts = n_points * 100
    
    while len(points) < n_points and n_attempts < max_attempts:
        # Sample random points in bounding box
        batch_size = min(n_points * 2, 1000)
        candidates = rng.uniform(bounds[0], bounds[1], size=(batch_size, 2))
        
        # Check which are inside
        inside_mask = _is_inside_vectorized(candidates, boundary)
        inside_points = candidates[inside_mask]
        
        points.extend(inside_points[:n_points - len(points)])
        n_attempts += batch_size
    
    if len(points) < n_points:
        raise RuntimeError(f"Could not sample {n_points} points after {max_attempts} attempts")
    
    return np.array(points)


def _is_inside_vectorized(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Vectorized inside/outside test using ray casting."""
    n_points = len(points)
    n_boundary = len(boundary)
    
    inside = np.zeros(n_points, dtype=bool)
    
    for i in range(n_points):
        p = points[i]
        crossings = 0
        
        for j in range(n_boundary):
            p1 = boundary[j]
            p2 = boundary[(j + 1) % n_boundary]
            
            if ((p1[1] > p[1]) != (p2[1] > p[1])):
                x_intersect = (p2[0] - p1[0]) * (p[1] - p1[1]) / (p2[1] - p1[1] + 1e-10) + p1[0]
                if p[0] < x_intersect:
                    crossings += 1
        
        inside[i] = (crossings % 2 == 1)
    
    return inside


class SDFRenderer:
    """Batch renderer for converting shapes to SDF images."""
    
    def __init__(
        self,
        grid_size: int = 64,
        bounds: Tuple[float, float] = (-2.0, 2.0),
        n_boundary_samples: int = 500,
        normalize: bool = True,
        normalization_method: str = "tanh",
    ):
        self.grid_size = grid_size
        self.bounds = bounds
        self.n_boundary_samples = n_boundary_samples
        self.normalize = normalize
        self.normalization_method = normalization_method
    
    def render(self, shape: ClosedCurve) -> np.ndarray:
        """Render a single shape to SDF image."""
        sdf = render_shape_to_sdf(
            shape, 
            self.grid_size, 
            self.bounds, 
            self.n_boundary_samples
        )
        
        if self.normalize:
            sdf = normalize_sdf(sdf, self.normalization_method)
        
        # Add channel dimension (H, W, 1)
        return sdf[:, :, np.newaxis].astype(np.float32)
    
    def render_batch(self, shapes: list) -> np.ndarray:
        """Render multiple shapes to SDF images."""
        images = [self.render(shape) for shape in shapes]
        return np.stack(images, axis=0)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .shapes import ShapeGenerator
    
    # Generate test shapes
    gen = ShapeGenerator(seed=42)
    shapes = gen.generate_batch(4)
    
    # Render to SDF
    renderer = SDFRenderer(grid_size=64)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, shape in enumerate(shapes):
        # Plot shape boundary
        pts = shape.sample(500)
        axes[0, i].plot(pts[:, 0], pts[:, 1], 'b-', lw=2)
        axes[0, i].set_xlim(-2.2, 2.2)
        axes[0, i].set_ylim(-2.2, 2.2)
        axes[0, i].set_aspect('equal')
        axes[0, i].set_title(f"Shape {i+1}")
        
        # Plot SDF
        sdf = renderer.render(shape)
        im = axes[1, i].imshow(
            sdf[:, :, 0], 
            extent=[-2, 2, -2, 2],
            origin='lower',
            cmap='RdBu',
            vmin=-1, vmax=1
        )
        axes[1, i].contour(
            sdf[:, :, 0], 
            levels=[0], 
            colors='black',
            extent=[-2, 2, -2, 2]
        )
        axes[1, i].set_title("SDF (blue=inside)")
        plt.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.savefig("sdf_demo.png", dpi=150)
    print("Saved sdf_demo.png")
