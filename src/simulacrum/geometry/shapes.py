"""
Parameterized closed curves for generating 2D closed shapes.

Each shape class provides:
    - __call__(t): Evaluate curve at parameter t ∈ [0, 2π]
    - sample(n_points): Get n_points uniformly sampled on curve
    - random(rng): Class method to generate random instance
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from numpy.random import Generator


class ClosedCurve(ABC):
    """Base class for parameterized closed curves γ: [0, 2π] → ℝ²"""
    
    @abstractmethod
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate curve at parameter values t ∈ [0, 2π]
        
        Args:
            t: Parameter values, shape (N,)
            
        Returns:
            Points on curve, shape (N, 2)
        """
        pass
    
    def sample(self, n_points: int = 200) -> np.ndarray:
        """Sample n_points uniformly in parameter space."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        return self(t)
    
    def contains(self, points: np.ndarray, n_rays: int = 4) -> np.ndarray:
        """Check if points are inside the curve using ray casting."""
        # Simple winding number test
        boundary = self.sample(500)
        inside = np.zeros(len(points), dtype=bool)
        
        for i, p in enumerate(points):
            # Count crossings
            crossings = 0
            for j in range(len(boundary)):
                p1 = boundary[j]
                p2 = boundary[(j + 1) % len(boundary)]
                
                if ((p1[1] > p[1]) != (p2[1] > p[1])):
                    x_intersect = (p2[0] - p1[0]) * (p[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]
                    if p[0] < x_intersect:
                        crossings += 1
            
            inside[i] = (crossings % 2 == 1)
        
        return inside


class Ellipse(ClosedCurve):
    """Axis-aligned or rotated ellipse."""
    
    def __init__(self, center=(0, 0), a=1.0, b=0.5, theta=0.0):
        self.center = np.array(center)
        self.a = a  # semi-major axis
        self.b = b  # semi-minor axis
        self.theta = theta  # rotation angle
        
    def __call__(self, t):
        x = self.a * np.cos(t)
        y = self.b * np.sin(t)
        # Rotate
        c, s = np.cos(self.theta), np.sin(self.theta)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        return np.column_stack([x_rot, y_rot]) + self.center
    
    @classmethod
    def random(cls, rng: Optional[Generator] = None):
        rng = rng or np.random.default_rng()
        center = rng.uniform(-0.5, 0.5, size=2)
        a = rng.uniform(0.8, 1.5)
        b = rng.uniform(0.4, a)  # b ≤ a
        theta = rng.uniform(0, np.pi)
        return cls(center, a, b, theta)


class Superellipse(ClosedCurve):
    """Lamé curve: |x/a|^n + |y/b|^n = 1
    
    n = 2: ellipse
    n > 2: rounded rectangle
    n < 2: astroid-like (pinched)
    """
    
    def __init__(self, center=(0, 0), a=1.0, b=1.0, n=2.5, theta=0.0):
        self.center = np.array(center)
        self.a = a
        self.b = b
        self.n = n
        self.theta = theta
        
    def __call__(self, t):
        def signed_pow(x, p):
            return np.sign(x) * np.abs(x) ** p
        
        x = self.a * signed_pow(np.cos(t), 2 / self.n)
        y = self.b * signed_pow(np.sin(t), 2 / self.n)
        # Rotate
        c, s = np.cos(self.theta), np.sin(self.theta)
        x_rot = c * x - s * y
        y_rot = s * x + c * y
        return np.column_stack([x_rot, y_rot]) + self.center
    
    @classmethod
    def random(cls, rng: Optional[Generator] = None):
        rng = rng or np.random.default_rng()
        center = rng.uniform(-0.3, 0.3, size=2)
        a = rng.uniform(0.8, 1.4)
        b = rng.uniform(0.6, 1.4)
        n = rng.uniform(1.5, 5.0)  # Avoid n < 1.5 (too pointy)
        theta = rng.uniform(0, np.pi)
        return cls(center, a, b, n, theta)


class FourierBlob(ClosedCurve):
    """Blob defined by Fourier radial perturbation.
    
    r(θ) = r₀ + Σ aₖcos(kθ) + bₖsin(kθ)
    """
    
    def __init__(self, center=(0, 0), r0=1.0, coeffs_cos=None, coeffs_sin=None):
        self.center = np.array(center)
        self.r0 = r0
        self.coeffs_cos = coeffs_cos if coeffs_cos is not None else []
        self.coeffs_sin = coeffs_sin if coeffs_sin is not None else []
        
    def __call__(self, t):
        r = self.r0 * np.ones_like(t)
        for k, ak in enumerate(self.coeffs_cos, start=1):
            r = r + ak * np.cos(k * t)
        for k, bk in enumerate(self.coeffs_sin, start=1):
            r = r + bk * np.sin(k * t)
        r = np.maximum(r, 0.1)  # Ensure positive radius
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y]) + self.center
    
    @classmethod
    def random(cls, n_modes: int = 5, rng: Optional[Generator] = None):
        rng = rng or np.random.default_rng()
        center = rng.uniform(-0.3, 0.3, size=2)
        r0 = rng.uniform(0.8, 1.2)
        # Coefficients decay with frequency to keep shape smooth
        coeffs_cos = [rng.uniform(-0.3, 0.3) / (k + 1) for k in range(n_modes)]
        coeffs_sin = [rng.uniform(-0.3, 0.3) / (k + 1) for k in range(n_modes)]
        return cls(center, r0, coeffs_cos, coeffs_sin)


class StarPolygon(ClosedCurve):
    """Star-like polygon with n vertices alternating between two radii."""
    
    def __init__(self, center=(0, 0), n_points=5, r_outer=1.2, r_inner=0.6, theta=0.0):
        self.center = np.array(center)
        self.n_points = n_points
        self.r_outer = r_outer
        self.r_inner = r_inner
        self.theta = theta
        
    def __call__(self, t):
        n = self.n_points
        angles = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) + self.theta
        radii = np.array([self.r_outer if i % 2 == 0 else self.r_inner for i in range(2 * n)])
        
        vertices_x = radii * np.cos(angles)
        vertices_y = radii * np.sin(angles)
        
        t_normalized = (t / (2 * np.pi)) * (2 * n)
        edge_idx = np.floor(t_normalized).astype(int) % (2 * n)
        local_t = t_normalized - np.floor(t_normalized)
        
        next_idx = (edge_idx + 1) % (2 * n)
        x = (1 - local_t) * vertices_x[edge_idx] + local_t * vertices_x[next_idx]
        y = (1 - local_t) * vertices_y[edge_idx] + local_t * vertices_y[next_idx]
        
        return np.column_stack([x, y]) + self.center
    
    @classmethod
    def random(cls, rng: Optional[Generator] = None):
        rng = rng or np.random.default_rng()
        center = rng.uniform(-0.3, 0.3, size=2)
        n_points = rng.integers(4, 8)
        r_outer = rng.uniform(1.0, 1.5)
        r_inner = rng.uniform(0.3, 0.7) * r_outer
        theta = rng.uniform(0, 2 * np.pi / n_points)
        return cls(center, n_points, r_outer, r_inner, theta)


class SmoothPolygon(ClosedCurve):
    """Polygon with rounded corners using Catmull-Rom spline interpolation."""
    
    def __init__(self, vertices, tension=0.5):
        self.vertices = np.array(vertices)
        self.tension = tension
        
    def _catmull_rom(self, p0, p1, p2, p3, t):
        """Catmull-Rom spline interpolation between p1 and p2."""
        tau = self.tension
        t2 = t * t
        t3 = t2 * t
        
        out = (
            (-tau * t + 2 * tau * t2 - tau * t3) * p0 +
            (1 + (tau - 3) * t2 + (2 - tau) * t3) * p1 +
            (tau * t + (3 - 2 * tau) * t2 + (tau - 2) * t3) * p2 +
            (-tau * t2 + tau * t3) * p3
        )
        return out
        
    def __call__(self, t):
        n = len(self.vertices)
        pts = []
        
        for ti in t:
            seg_float = (ti / (2 * np.pi)) * n
            seg_idx = int(np.floor(seg_float)) % n
            local_t = seg_float - np.floor(seg_float)
            
            # Four consecutive vertices with wrapping
            p0 = self.vertices[(seg_idx - 1) % n]
            p1 = self.vertices[seg_idx]
            p2 = self.vertices[(seg_idx + 1) % n]
            p3 = self.vertices[(seg_idx + 2) % n]
            
            pt = self._catmull_rom(p0, p1, p2, p3, local_t)
            pts.append(pt)
        
        return np.array(pts)
    
    @classmethod
    def random(cls, n_vertices: int = 6, rng: Optional[Generator] = None):
        rng = rng or np.random.default_rng()
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
        radii = rng.uniform(0.7, 1.4, n_vertices)
        vertices = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        center_offset = rng.uniform(-0.3, 0.3, size=2)
        vertices += center_offset
        tension = rng.uniform(0.3, 0.7)
        return cls(vertices, tension)


# Registry of all shape classes
SHAPE_CLASSES = [Ellipse, Superellipse, FourierBlob, StarPolygon, SmoothPolygon]

# Name to class mapping
SHAPE_REGISTRY = {
    'Ellipse': Ellipse,
    'Superellipse': Superellipse,
    'FourierBlob': FourierBlob,
    'StarPolygon': StarPolygon,
    'SmoothPolygon': SmoothPolygon,
}


class ShapeGenerator:
    """Factory for generating random shapes with configurable distribution."""
    
    def __init__(
        self, 
        seed: Optional[int] = None,
        shape_distribution: Optional[dict] = None,
    ):
        """
        Initialize shape generator.
        
        Args:
            seed: Random seed
            shape_distribution: Dict mapping shape names to weights.
                Example: {"Ellipse": 0.5, "Superellipse": 0.5}
                Example: {"Ellipse": 1.0}  # Only ellipses
                If None, uniform distribution over all shapes.
        """
        self.rng = np.random.default_rng(seed)
        
        if shape_distribution is None:
            # Uniform over all shapes
            self.shape_classes = SHAPE_CLASSES
            self.weights = None
        else:
            # Validate and parse distribution
            self.shape_classes = []
            weights = []
            
            for name, weight in shape_distribution.items():
                if name not in SHAPE_REGISTRY:
                    raise ValueError(
                        f"Unknown shape '{name}'. Available: {list(SHAPE_REGISTRY.keys())}"
                    )
                if weight < 0:
                    raise ValueError(f"Weight for '{name}' must be non-negative")
                
                self.shape_classes.append(SHAPE_REGISTRY[name])
                weights.append(weight)
            
            # Normalize weights to probabilities
            total = sum(weights)
            if total <= 0:
                raise ValueError("Total weight must be positive")
            self.weights = [w / total for w in weights]
    
    def generate(self, shape_type=None) -> ClosedCurve:
        """Generate a random shape, optionally of a specific type."""
        if shape_type is None:
            if self.weights is None:
                shape_class = self.rng.choice(self.shape_classes)
            else:
                idx = self.rng.choice(len(self.shape_classes), p=self.weights)
                shape_class = self.shape_classes[idx]
        else:
            shape_class = shape_type
        return shape_class.random(rng=self.rng)
    
    def generate_batch(self, n: int, shape_type=None) -> list:
        """Generate n random shapes."""
        return [self.generate(shape_type) for _ in range(n)]
    
    def get_distribution_summary(self) -> str:
        """Return a string describing the current distribution."""
        if self.weights is None:
            return f"Uniform over {len(self.shape_classes)} shapes: {[c.__name__ for c in self.shape_classes]}"
        else:
            parts = [f"{c.__name__}: {w:.1%}" for c, w in zip(self.shape_classes, self.weights)]
            return "Custom distribution: " + ", ".join(parts)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    rng = np.random.default_rng(42)
    
    shapes = [
        ("Ellipse", Ellipse.random(rng)),
        ("Superellipse", Superellipse.random(rng)),
        ("Fourier Blob", FourierBlob.random(rng=rng)),
        ("Star Polygon", StarPolygon.random(rng)),
        ("Smooth Polygon", SmoothPolygon.random(rng=rng)),
    ]
    
    for ax, (name, shape) in zip(axes, shapes):
        pts = shape.sample(500)
        ax.plot(pts[:, 0], pts[:, 1], lw=2)
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(name)
    
    # Random mix
    gen = ShapeGenerator(seed=123)
    for _ in range(6):
        shape = gen.generate()
        pts = shape.sample(500)
        axes[-1].plot(pts[:, 0], pts[:, 1], alpha=0.7, lw=1.5)
    axes[-1].set_xlim(-2.2, 2.2)
    axes[-1].set_ylim(-2.2, 2.2)
    axes[-1].set_aspect('equal')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].set_title("Random Mix")
    
    plt.tight_layout()
    plt.savefig("shapes_demo.png", dpi=150)
    print("Saved shapes_demo.png")
