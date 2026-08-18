"""
Particle (material point) state.

Struct-of-arrays, not a list of ``Particle`` objects. With 10^5-10^7 particles
the whole method is vectorised gather/scatter, and SoA is the only layout that
survives the move to Warp or a Rust backend -- each field becomes a ``wp.array``
with no restructuring.

State carried per particle:

    x      (n, dim)        current position
    v      (n, dim)        velocity
    F      (n, dim, dim)   deformation gradient
    C      (n, dim, dim)   APIC affine velocity matrix
    vol0   (n,)            reference volume
    mass   (n,)            mass (constant)
    lp0    (n, dim)        initial particle-domain half-width (GIMP/CPDI)
    lp     (n, dim)        current  particle-domain half-width (GIMP/CPDI)

History variables for plasticity would be added here as extra arrays.
"""

from __future__ import annotations

import numpy as np

from scion.core import small_linalg as sla


class ParticleSet:
    """Struct-of-arrays material point state."""

    def __init__(self, x, v=None, vol0=None, density=1.0, lp0=None,
                 dtype=np.float64):
        self.dtype = dtype
        self.x = np.ascontiguousarray(x, dtype=dtype)
        self.n, self.dim = self.x.shape

        self.v = (np.zeros_like(self.x) if v is None
                  else np.ascontiguousarray(v, dtype=dtype))
        self.F = sla.identity(self.n, self.dim, dtype)
        self.C = np.zeros((self.n, self.dim, self.dim), dtype=dtype)

        if vol0 is None:
            raise ValueError("vol0 is required: it sets the stress scale")
        self.vol0 = (np.full(self.n, float(vol0), dtype=dtype)
                     if np.isscalar(vol0)
                     else np.ascontiguousarray(vol0, dtype=dtype))
        self.mass = float(density) * self.vol0

        # Particle domain half-width. Point bases ignore it; GIMP and CPDI need
        # it. Default: half the particle spacing implied by vol0, i.e. the
        # domains tile the body exactly with no gaps or overlap.
        if lp0 is None:
            spacing = self.vol0 ** (1.0 / self.dim)
            lp0 = 0.5 * np.repeat(spacing[:, None], self.dim, axis=1)
        self.lp0 = np.ascontiguousarray(np.broadcast_to(
            np.asarray(lp0, dtype=dtype), (self.n, self.dim)))
        self.lp = self.lp0.copy()

    # ------------------------------------------------------------------
    @classmethod
    def from_sampler(cls, mask_fn, bounds, dx: float, ppc: int = 2,
                     velocity=None, density: float = 1.0, dtype=np.float64):
        """Seed particles on a regular sub-grid, keeping those inside a region.

        Parameters
        ----------
        mask_fn : callable (n, dim) -> (n,) bool
        bounds  : ((lo...), (hi...)) sampling box
        dx      : background cell size
        ppc     : particles per cell per axis
        """
        lo, hi = np.asarray(bounds[0], dtype=dtype), np.asarray(bounds[1], dtype=dtype)
        dim = len(lo)
        step = dx / ppc
        axes = [np.arange(lo[a] + step / 2, hi[a], step) for a in range(dim)]
        mg = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([m.ravel() for m in mg], axis=1)

        keep = mask_fn(pts)
        pts = pts[keep]

        vol0 = step ** dim
        v = None
        if velocity is not None:
            v = np.tile(np.asarray(velocity, dtype=dtype), (len(pts), 1))
        # domains exactly tile the sampling lattice
        lp0 = np.full((len(pts), dim), 0.5 * step, dtype=dtype)
        return cls(pts, v=v, vol0=vol0, density=density, lp0=lp0, dtype=dtype)

    # ------------------------------------------------------------------
    def concatenate(self, other: "ParticleSet") -> "ParticleSet":
        """Merge two particle sets (e.g. two bodies with different velocities)."""
        out = object.__new__(ParticleSet)
        out.dtype, out.dim = self.dtype, self.dim
        for f in ("x", "v", "F", "C", "vol0", "mass", "lp0", "lp"):
            setattr(out, f, np.concatenate([getattr(self, f), getattr(other, f)]))
        out.n = out.x.shape[0]
        return out

    # ------------------------------------------------------------------
    def update_domain(self, mode: str = "none") -> None:
        """Advect the particle domain with the deformation.

        ``"none"``    uGIMP: the domain keeps its initial size. Cheap, stable,
                      and the usual choice; the domain is only a quadrature
                      device, not a material region.
        ``"stretch"`` cpGIMP: each half-width scales with the diagonal stretch,
                      ``lp_i = lp0_i * F_ii``. Tracks large uniaxial stretching
                      better, but ignores rotation -- for a rotating body the
                      diagonal of F is not a stretch, so the domain shrinks
                      spuriously. Prefer ``"none"`` unless you know the motion
                      is close to axis-aligned.

        Note the uGIMP support requires ``lp <= dx/2``. Under ``"stretch"`` that
        can be violated once a particle has stretched past 2x, which raises from
        the basis rather than failing silently.
        """
        if mode == "none":
            return
        if mode != "stretch":
            raise ValueError(f"Unknown domain update mode '{mode}'")
        diag = np.einsum("pii->pi", self.F)
        self.lp = self.lp0 * np.abs(diag)

    # ------------------------------------------------------------------
    # Diagnostics -- cheap, and worth logging every N steps
    # ------------------------------------------------------------------
    def linear_momentum(self) -> np.ndarray:
        return (self.mass[:, None] * self.v).sum(axis=0)

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass * (self.v ** 2).sum(axis=1)).sum())

    def jacobian(self) -> np.ndarray:
        return sla.det(self.F)

    def volume(self) -> np.ndarray:
        """Current volume, V = J V0."""
        return self.jacobian() * self.vol0
