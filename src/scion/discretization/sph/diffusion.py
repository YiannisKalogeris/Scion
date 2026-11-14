r"""
Diffusion on a fixed SPH particle set.

The anisotropic coefficient is a directional projection, not a tensor product
-----------------------------------------------------------------------------
For a diffusivity tensor ``D`` the inter-particle coefficient is

    D_ij = 1 / (e_ij . D^-1 e_ij)

which is what SPHinXsys computes as ``1/|T e|^2`` with ``T`` the inverse
Cholesky factor of ``D`` -- the same number, since ``|T e|^2 = e . D^-1 e``.
Worth writing in the second form: the Cholesky factor is an implementation
detail, the quantity is the harmonic projection of ``D`` on the line joining
the two particles, and it reduces to ``d`` exactly when ``D = d I``, which is
the check that catches a transposed or un-inverted tensor.

The time step is the one SPHinXsys uses
---------------------------------------
``dt = 0.5 h^2 / (D_ref * dim)``, with ``D_ref`` the largest eigenvalue. Note
it scales with the *smoothing length*, not the particle spacing, so at the
usual ``h = 1.3 dp`` it is 1.7 times what a spacing-based rule would give;
with the two-stage scheme below that is stable here and is not a general
guarantee. :func:`stable_time_step` reports the spectral bound of the actual
assembled operator, which is the honest check and costs one eigenvalue
estimate.

Two stages, and the second one is a mean
----------------------------------------
``step`` is Heun's method written the way SPHinXsys writes it: a forward
Euler stage to an intermediate state, a second rate from there, and then the
average of the two states. That is RK2 and is second order; evaluating the
second rate and *not* averaging gives the unstable midpoint-free scheme, and
the symptom is growth rather than an obviously wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import get_kernel
from .neighbors import NeighborList, build_neighbors
from .operators import (kernel_pair_values, laplacian_rate,
                        linear_gradient_correction, pair_conductance,
                        shepard_interpolate)

__all__ = ["IsotropicDiffusivity", "DirectionalDiffusivity", "DiffusionSolver"]


@dataclass(frozen=True)
class DirectionalDiffusivity:
    """``D = d I + d_bias (b (x) b)``, biased along the unit vector ``b``."""

    d_coeff: float
    bias_coeff: float = 0.0
    bias_direction: tuple = (1.0, 0.0)

    def tensor(self, dim: int) -> np.ndarray:
        b = np.asarray(self.bias_direction, dtype=float)[:dim]
        norm = np.linalg.norm(b)
        if self.bias_coeff != 0.0 and norm == 0.0:
            raise ValueError("a biased diffusivity needs a nonzero direction")
        if norm > 0.0:
            b = b / norm
        return self.d_coeff * np.eye(dim) + self.bias_coeff * np.outer(b, b)

    def reference(self) -> float:
        return max(self.d_coeff, self.d_coeff + self.bias_coeff)

    def inter_particle(self, e: np.ndarray) -> np.ndarray:
        """``D_ij`` at unit separations ``e``, ``(n_pairs,)``."""
        dim = e.shape[1]
        inv = np.linalg.inv(self.tensor(dim))
        return 1.0 / np.einsum("na,ab,nb->n", e, inv, e)

    def time_step(self, h: float, dim: int) -> float:
        return 0.5 * h * h / self.reference() / dim


def IsotropicDiffusivity(d_coeff: float) -> DirectionalDiffusivity:
    """``D = d I``. A function rather than a class: the directional form with
    no bias *is* the isotropic one, and a second class would be a second
    place for the projection to be implemented differently."""
    return DirectionalDiffusivity(d_coeff=d_coeff, bias_coeff=0.0)


class DiffusionSolver:
    """Explicit diffusion of one species on a fixed particle set.

    The particles do not move, so the neighbour list, the kernel values, the
    correction matrices and the inter-particle coefficients are built once in
    the constructor and reused by every step -- which is the whole reason a
    pure-diffusion problem is cheap, and the reason this class exists instead
    of a function.
    """

    def __init__(self, positions, volumes, diffusivity, *, h, kernel="wendland_c2",
                 lower=None, upper=None, periodic=None, correction=True,
                 correction_alpha=0.0, correction_epsilon=None):
        self.positions = np.asarray(positions, dtype=float)
        self.n, self.dim = self.positions.shape
        self.volumes = np.broadcast_to(np.asarray(volumes, dtype=float),
                                       (self.n,)).copy()
        self.diffusivity = diffusivity
        self.kernel = get_kernel(kernel)
        self.h = float(h)
        self.lower, self.upper, self.periodic = lower, upper, periodic
        self.pairs: NeighborList = build_neighbors(
            self.positions, self.kernel.support(self.h),
            lower=lower, upper=upper, periodic=periodic)
        self.w_ij, self.dw_ij = kernel_pair_values(
            self.pairs, self.kernel, self.h, self.dim)
        self.B = (linear_gradient_correction(
            self.pairs, self.volumes, self.dw_ij, self.dim,
            alpha=correction_alpha, epsilon=correction_epsilon)
            if correction else None)
        self.d_ij = self.diffusivity.inter_particle(self.pairs.e)
        #: the edge weights, which depend on the positions alone and so are
        #: built once -- 5x on the step, since the projection they hoist
        #: materialises a (n_pairs, dim, dim) gather every time it is formed
        self.conductance = pair_conductance(self.pairs, self.dw_ij, self.d_ij,
                                            self.B)

    # -- operators ------------------------------------------------------
    def rate(self, phi) -> np.ndarray:
        """``d phi / dt``; sums to zero against the volumes, exactly."""
        return laplacian_rate(self.pairs, phi, self.volumes, self.conductance)

    def time_step(self) -> float:
        """SPHinXsys's rule, ``0.5 h^2 / (D_ref dim)``."""
        return self.diffusivity.time_step(self.h, self.dim)

    def stable_time_step(self, safety: float = 1.0, iterations: int = 60) -> float:
        """``2 / rho(L)`` for the assembled operator, by power iteration.

        The forward-Euler bound; Heun has the same real-axis limit, so this is
        the step the scheme actually tolerates, as opposed to the rule above
        which is a smoothing-length heuristic.
        """
        rng = np.random.default_rng(0)
        v = rng.standard_normal(self.n)
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(iterations):
            w = -self.rate(v)
            lam = float(np.linalg.norm(w))
            if lam == 0.0:
                return np.inf
            v = w / lam
        return safety * 2.0 / lam

    # -- time stepping --------------------------------------------------
    def step(self, phi, dt: float) -> np.ndarray:
        """One Heun (RK2) step, returning a new array."""
        phi = np.asarray(phi, dtype=float)
        first = phi + dt * self.rate(phi)
        second = first + dt * self.rate(first)
        return 0.5 * (phi + second)

    def run(self, phi, dt: float, end_time: float, callback=None):
        """March to ``end_time``, calling ``callback(t, phi)`` after each step.

        The last step is *not* clipped to land on ``end_time``: SPHinXsys's
        loop overshoots, and the comparison has to run the same clock.
        """
        phi = np.array(phi, dtype=float, copy=True)
        t = 0.0
        while t < end_time:
            phi = self.step(phi, dt)
            t += dt
            if callback is not None:
                callback(t, phi)
        return phi

    # -- observation ----------------------------------------------------
    def observe(self, points, phi) -> np.ndarray:
        return shepard_interpolate(points, self.positions, phi, self.volumes,
                                   self.kernel, self.h, self.dim,
                                   lower=self.lower, upper=self.upper,
                                   periodic=self.periodic)
