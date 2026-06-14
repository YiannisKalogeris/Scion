"""
Harmonic map computation for ground truth diffeomorphisms.

A harmonic map φ: Ω → [-1,1]² satisfies:
    Δφ = 0  in Ω
    φ(∂Ω) = ∂([-1,1]²)  (boundary maps to boundary)

We solve two Laplace equations:
    Δu = 0, Δv = 0
with boundary conditions that map ∂Ω to the square boundary.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple, Optional
from pathlib import Path


from simulacrum.geometry.shapes import ClosedCurve


def compute_harmonic_map(
    shape: ClosedCurve,
    grid_size: int = 64,
    bounds: Tuple[float, float] = (-2.0, 2.0),
    tol: float = 1e-6,
    max_iter: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute harmonic map from shape interior to [-1,1]² using finite differences.
    
    Strategy:
        1. Create a regular grid covering the bounding box
        2. Identify interior and boundary nodes
        3. Set boundary conditions: map boundary to square boundary
        4. Solve Laplace equation Δu = 0, Δv = 0 for interior nodes
    
    Args:
        shape: ClosedCurve defining the domain
        grid_size: Number of grid points per dimension
        bounds: Domain bounds
        tol: Solver tolerance
        max_iter: Maximum solver iterations
        
    Returns:
        u: (grid_size, grid_size) x-coordinate of map (NaN outside)
        v: (grid_size, grid_size) y-coordinate of map (NaN outside)
        x_grid: 1D array of x coordinates
        y_grid: 1D array of y coordinates
    """
    # Create grid
    x_grid = np.linspace(bounds[0], bounds[1], grid_size)
    y_grid = np.linspace(bounds[0], bounds[1], grid_size)
    h = x_grid[1] - x_grid[0]  # Grid spacing
    
    X, Y = np.meshgrid(x_grid, y_grid, indexing='xy')
    
    # Sample boundary densely
    boundary_points = shape.sample(1000)
    
    # Classify grid points: interior, boundary, exterior
    interior_mask, boundary_mask, boundary_values = _classify_grid_points(
        X, Y, boundary_points, h
    )
    
    # Build Laplacian matrix for interior points
    n_interior = np.sum(interior_mask)
    n_boundary = np.sum(boundary_mask)
    
    if n_interior == 0:
        raise ValueError("No interior points found. Shape may be too small for grid.")
    
    # Create index mapping
    interior_indices = np.where(interior_mask.ravel())[0]
    boundary_indices = np.where(boundary_mask.ravel())[0]
    
    idx_to_interior = {idx: i for i, idx in enumerate(interior_indices)}
    idx_to_boundary = {idx: i for i, idx in enumerate(boundary_indices)}
    
    # Build sparse Laplacian matrix
    A, b_u, b_v = _build_laplacian_system(
        grid_size, interior_mask, boundary_mask, boundary_values,
        idx_to_interior, idx_to_boundary
    )
    
    # Solve for u (x-component) and v (y-component)
    u_interior = spsolve(A.tocsr(), b_u)
    v_interior = spsolve(A.tocsr(), b_v)
    
    # Assemble full solution
    u = np.full((grid_size, grid_size), np.nan)
    v = np.full((grid_size, grid_size), np.nan)
    
    # Fill interior values
    for i, idx in enumerate(interior_indices):
        row, col = np.unravel_index(idx, (grid_size, grid_size))
        u[row, col] = u_interior[i]
        v[row, col] = v_interior[i]
    
    # Fill boundary values
    for i, idx in enumerate(boundary_indices):
        row, col = np.unravel_index(idx, (grid_size, grid_size))
        u[row, col] = boundary_values[i, 0]
        v[row, col] = boundary_values[i, 1]
    
    return u, v, x_grid, y_grid


def _classify_grid_points(
    X: np.ndarray, 
    Y: np.ndarray, 
    boundary_points: np.ndarray,
    h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Classify grid points as interior, boundary, or exterior.
    
    A point is:
        - Boundary: close to the shape boundary
        - Interior: inside the shape and not boundary
        - Exterior: outside the shape
    """
    grid_size = X.shape[0]
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    n_points = len(grid_points)
    
    # Distance to nearest boundary point
    distances = np.linalg.norm(
        grid_points[:, None, :] - boundary_points[None, :, :],
        axis=-1
    )
    min_distances = np.min(distances, axis=-1)
    nearest_boundary_idx = np.argmin(distances, axis=-1)
    
    # Inside/outside using winding number
    inside = _is_inside_batch(grid_points, boundary_points)
    
    # Boundary points: inside and close to boundary
    boundary_threshold = 1.5 * h
    boundary_mask = inside & (min_distances < boundary_threshold)
    
    # Interior points: inside and not boundary
    interior_mask = inside & ~boundary_mask
    
    # Compute boundary values (where boundary points map to on the square)
    boundary_indices = np.where(boundary_mask)[0]
    boundary_values = np.zeros((len(boundary_indices), 2))
    
    for i, idx in enumerate(boundary_indices):
        # Find nearest point on shape boundary
        nearest_idx = nearest_boundary_idx[idx]
        
        # Map to square boundary based on arc length parameter
        t = nearest_idx / len(boundary_points)  # Normalized arc length [0, 1)
        boundary_values[i] = _param_to_square_boundary(t)
    
    return (
        interior_mask.reshape(grid_size, grid_size),
        boundary_mask.reshape(grid_size, grid_size),
        boundary_values
    )


def _param_to_square_boundary(t: float) -> np.ndarray:
    """
    Map parameter t ∈ [0, 1) to point on square boundary [-1,1]².
    
    FIXED: Traverses counterclockwise starting from (1, -1) to produce
    orientation-PRESERVING maps (positive Jacobian determinants).
    
    The direction must match how shapes typically traverse their boundaries
    (counterclockwise) to ensure the resulting harmonic map preserves orientation.
    """
    # Perimeter = 8, each side has length 2
    t = t % 1.0
    s = t * 8  # Arc length along perimeter
    
    # Counterclockwise from (1, -1):
    # Right edge UP, Top edge LEFT, Left edge DOWN, Bottom edge RIGHT
    if s < 2:      # Right edge: (1, -1) to (1, 1) going UP
        return np.array([1.0, -1.0 + s])
    elif s < 4:    # Top edge: (1, 1) to (-1, 1) going LEFT
        return np.array([1.0 - (s - 2), 1.0])
    elif s < 6:    # Left edge: (-1, 1) to (-1, -1) going DOWN
        return np.array([-1.0, 1.0 - (s - 4)])
    else:          # Bottom edge: (-1, -1) to (1, -1) going RIGHT
        return np.array([-1.0 + (s - 6), -1.0])


def _is_inside_batch(points: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Check if points are inside using ray casting."""
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
                denom = p2[1] - p1[1]
                if abs(denom) > 1e-10:
                    x_intersect = (p2[0] - p1[0]) * (p[1] - p1[1]) / denom + p1[0]
                    if p[0] < x_intersect:
                        crossings += 1
        
        inside[i] = (crossings % 2 == 1)
    
    return inside


def _build_laplacian_system(
    grid_size: int,
    interior_mask: np.ndarray,
    boundary_mask: np.ndarray,
    boundary_values: np.ndarray,
    idx_to_interior: dict,
    idx_to_boundary: dict,
) -> Tuple[sparse.lil_matrix, np.ndarray, np.ndarray]:
    """
    Build the sparse linear system for the Laplace equation.
    
    Δu = 0 becomes: -4*u[i,j] + u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] = 0
    """
    interior_mask_flat = interior_mask.ravel()
    boundary_mask_flat = boundary_mask.ravel()
    
    n_interior = np.sum(interior_mask_flat)
    interior_indices = np.where(interior_mask_flat)[0]
    boundary_indices = np.where(boundary_mask_flat)[0]
    
    # Sparse matrix in LIL format for efficient construction
    A = sparse.lil_matrix((n_interior, n_interior))
    b_u = np.zeros(n_interior)
    b_v = np.zeros(n_interior)
    
    # Neighbor offsets (up, down, left, right)
    offsets = [-grid_size, grid_size, -1, 1]
    
    for i, idx in enumerate(interior_indices):
        A[i, i] = -4.0
        
        for offset in offsets:
            neighbor_idx = idx + offset
            
            # Check bounds
            if neighbor_idx < 0 or neighbor_idx >= grid_size * grid_size:
                continue
            
            # Check row/column consistency (no wrap-around)
            row, col = np.unravel_index(idx, (grid_size, grid_size))
            n_row, n_col = np.unravel_index(neighbor_idx, (grid_size, grid_size))
            if abs(row - n_row) > 1 or abs(col - n_col) > 1:
                continue
            
            if neighbor_idx in idx_to_interior:
                # Neighbor is interior: add to matrix
                j = idx_to_interior[neighbor_idx]
                A[i, j] = 1.0
            elif neighbor_idx in idx_to_boundary:
                # Neighbor is boundary: move to RHS
                j = idx_to_boundary[neighbor_idx]
                b_u[i] -= boundary_values[j, 0]
                b_v[i] -= boundary_values[j, 1]
            # Exterior neighbors are ignored (Neumann-like at domain edge)
    
    return A, b_u, b_v


def interpolate_harmonic_map(
    u: np.ndarray,
    v: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """
    Interpolate harmonic map at arbitrary query points using bilinear interpolation.
    
    Args:
        u, v: (grid_size, grid_size) map components
        x_grid, y_grid: 1D grid coordinates
        query_points: (N, 2) points to interpolate at
        
    Returns:
        mapped_points: (N, 2) interpolated map values
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # Create interpolators (handling NaN with nearest extrapolation)
    u_interp = RegularGridInterpolator(
        (y_grid, x_grid), u, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    v_interp = RegularGridInterpolator(
        (y_grid, x_grid), v,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    # Interpolate (note: scipy uses (y, x) ordering)
    query_yx = query_points[:, ::-1]
    u_vals = u_interp(query_yx)
    v_vals = v_interp(query_yx)
    
    return np.stack([u_vals, v_vals], axis=-1)


class HarmonicMapSolver:
    """Convenience class for computing and interpolating harmonic maps."""
    
    def __init__(
        self,
        grid_size: int = 64,
        bounds: Tuple[float, float] = (-2.0, 2.0),
    ):
        self.grid_size = grid_size
        self.bounds = bounds
        
        # Cache for current shape
        self._u = None
        self._v = None
        self._x_grid = None
        self._y_grid = None
    
    def solve(self, shape: ClosedCurve):
        """Compute harmonic map for a shape."""
        self._u, self._v, self._x_grid, self._y_grid = compute_harmonic_map(
            shape, self.grid_size, self.bounds
        )
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate harmonic map at query points."""
        if self._u is None:
            raise RuntimeError("Must call solve() first")
        
        return interpolate_harmonic_map(
            self._u, self._v, self._x_grid, self._y_grid, points
        )
    
    def get_grid_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the full grid map (u, v arrays)."""
        return self._u, self._v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from .shapes import Ellipse, Superellipse, FourierBlob
    
    # Test with different shapes
    shapes = [
        ("Ellipse", Ellipse(a=1.2, b=0.7)),
        ("Superellipse", Superellipse(a=1.0, b=1.0, n=3.0)),
        ("Blob", FourierBlob(r0=1.0, coeffs_cos=[0.2, 0.1], coeffs_sin=[0.15, -0.1])),
    ]
    
    fig, axes = plt.subplots(len(shapes), 3, figsize=(15, 5*len(shapes)))
    
    for i, (name, shape) in enumerate(shapes):
        print(f"Computing harmonic map for {name}...")
        
        solver = HarmonicMapSolver(grid_size=64)
        solver.solve(shape)
        u, v = solver.get_grid_map()
        
        # Plot original shape
        boundary = shape.sample(200)
        axes[i, 0].plot(boundary[:, 0], boundary[:, 1], 'b-', lw=2)
        axes[i, 0].set_xlim(-2.2, 2.2)
        axes[i, 0].set_ylim(-2.2, 2.2)
        axes[i, 0].set_aspect('equal')
        axes[i, 0].set_title(f"{name} - Original")
        axes[i, 0].grid(True, alpha=0.3)
        
        # Plot u component
        im1 = axes[i, 1].imshow(u, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        axes[i, 1].contour(u, levels=15, extent=[-2, 2, -2, 2], colors='white', alpha=0.5)
        axes[i, 1].set_title(f"{name} - u component")
        plt.colorbar(im1, ax=axes[i, 1])
        
        # Plot v component  
        im2 = axes[i, 2].imshow(v, extent=[-2, 2, -2, 2], origin='lower', cmap='viridis')
        axes[i, 2].contour(v, levels=15, extent=[-2, 2, -2, 2], colors='white', alpha=0.5)
        axes[i, 2].set_title(f"{name} - v component")
        plt.colorbar(im2, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig("harmonic_maps_demo.png", dpi=150)
    print("Saved harmonic_maps_demo.png")