"""
Explicit MPM time stepper.

One step is the standard four-stage update:

    1. P2G      scatter mass, momentum and internal force to the grid
    2. solve    explicit momentum update on the grid, then boundary conditions
    3. G2P      gather velocity (and the affine matrix, for APIC) back
    4. reset    the grid is thrown away; particles carry all state forward

Note this does *not* subclass ``DiscretizationBase``: explicit MPM never
assembles a matrix, so there is no ``LinearSystem`` to return. An implicit
stepper would assemble one and could then reuse ``solvers/nonlinear.py``
directly -- which is the reason the material interface here is PK1-plus-tangent-
compatible rather than something MPM-specific.
"""

from __future__ import annotations

import numpy as np

from .background_grid import CartesianGrid
from .particles import ParticleSet
from .transfer import Transfer


class ExplicitMPMStepper:
    """Symplectic-Euler explicit MPM.

    Parameters
    ----------
    grid        : CartesianGrid
    material    : object with ``pk1(F) -> (P, J)``
    transfer    : Transfer (APIC by default)
    dt          : time step
    body_force  : (dim,) acceleration, e.g. gravity
    boundaries  : sequence of GridBC applied after the grid velocity update
    domain_update : how the particle domain evolves, for domain bases such as
                  uGIMP. "none" (uGIMP, default) or "stretch" (cpGIMP); see
                  ``ParticleSet.update_domain``. Ignored by point bases.
    """

    def __init__(self, grid: CartesianGrid, material, transfer: Transfer,
                 dt: float, body_force=None, boundaries=(),
                 domain_update: str = "none"):
        self.grid = grid
        self.material = material
        self.transfer = transfer
        self.dt = float(dt)
        self.body_force = (np.zeros(grid.dim) if body_force is None
                           else np.asarray(body_force, dtype=grid.dtype))
        self.boundaries = list(boundaries)
        self.domain_update = domain_update

    # ------------------------------------------------------------------
    def cfl_dt(self, particles: ParticleSet, density: float,
               safety: float = 0.3) -> float:
        """Suggested stable step from the elastic wave speed."""
        c = self.material.wave_speed_factor / np.sqrt(density)
        v_max = np.abs(particles.v).max() if particles.n else 0.0
        return safety * self.grid.dx / (c + v_max)

    # ------------------------------------------------------------------
    def internal_force(self, particles: ParticleSet, st) -> np.ndarray:
        """Nodal internal force, ``f_i = -sum_p V_p sigma_p grad_w_ip``.

        Evaluated as ``-sum_p V_p^0 P_p F_p^T grad_w_ip``, which is identical
        (V_p sigma_p = J V_p^0 * (1/J) P F^T) but avoids dividing by J.
        """
        P, J = self.material.pk1(particles.F)
        VsigmaT = particles.vol0[:, None, None] * np.einsum(
            "pij,pkj->pik", P, particles.F
        )
        return -np.einsum("pij,pkj->pki", VsigmaT, st.grad_w)   # (n, K, dim)

    # ------------------------------------------------------------------
    def step(self, particles: ParticleSet) -> None:
        """Advance the particle state by one ``dt``, in place."""
        grid, dt = self.grid, self.dt

        # 1. stencil + P2G
        #    lp is passed unconditionally; point bases ignore it
        st = grid.build_stencil(particles.x, lp=getattr(particles, "lp", None))
        grid_m, grid_mv = self.transfer.p2g(particles, grid, st)

        f_int = self.internal_force(particles, st)
        np.add.at(grid_mv, st.node_id, dt * f_int)

        # 2. grid velocity update, body force, boundary conditions
        active = grid_m > 1.0e-12
        grid_v = np.zeros_like(grid_mv)
        grid_v[active] = grid_mv[active] / grid_m[active, None]
        if np.any(self.body_force):
            grid_v[active] += dt * self.body_force

        for bc in self.boundaries:
            bc(grid, grid_v, grid_m)

        # 3. G2P (also updates F and x)
        self.transfer.g2p(particles, grid, st, grid_v, dt)

        #    domain follows the new F, so it must come after G2P
        if grid.basis.requires_domain and self.domain_update != "none":
            particles.update_domain(self.domain_update)

        # 4. grid reset is implicit: grid_m / grid_v go out of scope

    # ------------------------------------------------------------------
    def run(self, particles: ParticleSet, n_steps: int, callback=None,
            callback_every: int = 100) -> ParticleSet:
        for s in range(n_steps + 1):
            if callback is not None and s % callback_every == 0:
                callback(s, s * self.dt, particles)
            if s < n_steps:
                self.step(particles)
        return particles
