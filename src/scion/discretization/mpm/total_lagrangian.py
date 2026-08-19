"""
Total-Lagrangian MPM (de Vaucorbeil, Nguyen & Hutchinson, CMAME 2020).

The background grid is anchored in the **reference** configuration and never
moves. Particle reference positions ``X_p`` are fixed, so the basis values and
gradients at the particles are computed **once** and reused for the whole run.

What that changes, against ``stepper.py``
-----------------------------------------
============================  ==============================  ====================
                              updated-Lagrangian              total-Lagrangian
============================  ==============================  ====================
grid                          rebuilt at ``x`` every step      fixed at ``X``
stencil                       recomputed every step            computed once
``F``                         accumulated, ``F <- (I+dt L)F``  ``I + sum_a u_a (x) Grad w_a``
gradients                     spatial                          material
cell crossing                 every time a particle moves      never
quadrature drift              grows as particles cluster       frozen at t = 0
============================  ==============================  ====================

The three consequences worth stating plainly:

**``F`` does not drift.** It is reconstructed from the nodal displacement field
each step rather than accumulated through a product of increments, so there is
no integration error in it at all -- only the interpolation error of the grid.

**There is no cell-crossing noise**, because nothing crosses a cell. This is the
method's main attraction and also the reason it cannot replace the
updated-Lagrangian stepper: a body that folds onto itself, fragments, or makes
self-contact has a reference configuration that no longer describes the
mechanics, and the fixed grid has no way to notice.

**The reference grid must contain the whole body**, undeformed, for the whole
run. Nothing checks this at every step -- the constructor checks it once.

Relation to FEM
---------------
With a fixed grid, fixed quadrature points and ``F`` from nodal displacements,
this *is* a finite element method whose quadrature happens to be a scattered
particle cloud rather than Gauss points. That is not a criticism: it is why the
material manifold is unambiguous here, which is what makes Eshelby stress and
the discrete configurational force well defined.
"""

from __future__ import annotations

import numpy as np

from scion.core import small_linalg as sla

from .background_grid import CartesianGrid
from .particles import ParticleSet
from .stepper import _promote
from .transfer import Transfer

__all__ = ["TotalLagrangianStepper"]


class TotalLagrangianStepper:
    """Explicit total-Lagrangian MPM.

    Parameters
    ----------
    grid       : reference-configuration grid; must contain the undeformed body
    material   : ``pk1(F) -> (P, J)`` or ``update(F, L, dt, state)``
    transfer   : PIC / FLIP / APIC, used for the momentum transfer as usual
    dt         : time step
    body_force : (dim,) acceleration
    boundaries : grid boundary conditions, applied in the reference frame
    state      : initial history for a path-dependent material
    hessian    : also build the basis second gradients, once, so that
                 configurational forces can be evaluated during the run

    The grid carries nodal *displacement* between steps -- the one piece of
    state a total-Lagrangian scheme needs and an updated-Lagrangian one does
    not, since here ``F`` is reconstructed from it rather than accumulated.
    """

    def __init__(self, grid: CartesianGrid, material, transfer: Transfer,
                 dt: float, body_force=None, boundaries=(), state=None,
                 hessian: bool = False):
        self.grid = grid
        self.material = material
        self.transfer = transfer
        self.dt = float(dt)
        self.body_force = (np.zeros(grid.dim) if body_force is None
                           else np.asarray(body_force, dtype=grid.dtype))
        self.boundaries = list(boundaries)

        self.state = state
        if state is not None and not hasattr(material, "update"):
            raise TypeError(
                "a state was given but the material has no update(F, L, dt, "
                "state); a hyperelastic pk1 material carries no history")

        self.hessian = bool(hessian)
        #: nodal displacement field, the total-Lagrangian grid state
        self.u_nodes = np.zeros((grid.n_nodes, grid.dim), dtype=grid.dtype)
        #: Cauchy stress from the previous step (path-dependent path only)
        self.stress = None
        self._stencil = None

    # ------------------------------------------------------------------
    def prepare(self, particles: ParticleSet):
        """Build the fixed stencil at the reference positions. Idempotent."""
        if self._stencil is not None:
            return self._stencil

        X = particles.X
        lo = self.grid.origin
        hi = lo + np.asarray(self.grid.shape) * self.grid.dx
        if np.any(X < lo) or np.any(X > hi):
            raise ValueError(
                "the reference grid does not contain the undeformed body: X "
                f"spans [{X.min(axis=0)}, {X.max(axis=0)}] but the grid covers "
                f"[{lo}, {hi}]. A total-Lagrangian grid is never rebuilt, so "
                "this cannot be fixed later in the run."
            )

        from .cpdi import CPDIGrid
        extra = ({"domain": particles.domain}
                 if isinstance(self.grid, CPDIGrid) else {})
        self._stencil = self.grid.build_stencil(
            X, lp=getattr(particles, "lp", None), hessian=self.hessian, **extra)
        return self._stencil

    @property
    def stencil(self):
        """The fixed stencil, or None before the first step."""
        return self._stencil

    # ------------------------------------------------------------------
    def cfl_dt(self, particles: ParticleSet, density: float,
               safety: float = 0.3) -> float:
        c = self.material.wave_speed_factor / np.sqrt(density)
        v_max = np.abs(particles.v).max() if particles.n else 0.0
        return safety * self.grid.dx / (c + v_max)

    # ------------------------------------------------------------------
    def deformation_gradient(self, st) -> np.ndarray:
        """``F = I + sum_a u_a (x) Grad w_a``, reconstructed not accumulated."""
        H = np.einsum("pka,pkJ->paJ", self.u_nodes[st.node_id], st.grad_w)
        return sla.identity(H.shape[0], self.grid.dim, self.grid.dtype) + H

    def internal_force(self, particles: ParticleSet, st) -> np.ndarray:
        """``f_i = -sum_p V_p^0 P_p Grad w_ip``, in the reference frame.

        No ``J`` and no push-forward: the reference volume and the material
        gradient are both fixed, which is the whole economy of the method.
        """
        if self.state is None:
            P, _ = self.material.pk1(particles.F)
        else:
            if self.stress is None:
                self._constitutive_update(particles,
                                          np.zeros_like(particles.F), 0.0)
            # sigma -> P = J sigma F^-T
            J = sla.det(particles.F)
            P = J[:, None, None] * np.einsum(
                "pij,pkj->pik", self.stress[:, :self.grid.dim, :self.grid.dim],
                sla.inv(particles.F, J))
        VP = particles.vol0[:, None, None] * P
        return -np.einsum("pij,pkj->pki", VP,
                          self.transfer.force_kernel(self.grid, st))

    def _constitutive_update(self, particles: ParticleSet, L: np.ndarray,
                             dt: float) -> None:
        sigma, self.state = self.material.update(
            _promote(particles.F, fill_diagonal=True),
            _promote(L, fill_diagonal=False), dt, self.state)
        self.stress = sigma

    # ------------------------------------------------------------------
    def step(self, particles: ParticleSet) -> None:
        """Advance one ``dt`` in place."""
        grid, dt = self.grid, self.dt
        st = self.prepare(particles)

        grid_m, grid_mv = self.transfer.p2g(particles, grid, st)
        np.add.at(grid_mv, st.node_id, dt * self.internal_force(particles, st))

        active = grid_m > 1.0e-12
        grid_v = np.zeros_like(grid_mv)
        grid_v[active] = grid_mv[active] / grid_m[active, None]
        if np.any(self.body_force):
            grid_v[active] += dt * self.body_force

        for bc in self.boundaries:
            bc(grid, grid_v, grid_m)

        # nodal displacement is integrated on the grid; it is the field F is
        # rebuilt from, so it must advance before the kinematics are read back
        self.u_nodes += dt * grid_v

        # velocity gradient in the *reference* frame, for a history material
        L = None
        if self.state is not None:
            L = np.einsum("pka,pkb->pab", grid_v[st.node_id], st.grad_w)

        # G2P through the transfer itself, so PIC/FLIP/APIC velocity logic is
        # used verbatim rather than reimplemented here. It also updates F and x
        # incrementally, and total-Lagrangian overrides both below -- one wasted
        # einsum, against two copies of the blend rule drifting apart.
        self.transfer.g2p(particles, grid, st, grid_v, dt)

        particles.F = self.deformation_gradient(st)
        # total-Lagrangian: the stencil is fixed, so a convected domain would
        # change the weights it was built with. CPDI is an updated-Lagrangian
        # construction and is not combined with this stepper.
        # position from the same nodal field F is built from, so the two cannot
        # disagree; advecting x by v independently lets them
        particles.x = particles.X + np.einsum(
            "pk,pka->pa", st.w, self.u_nodes[st.node_id])

        if self.state is not None:
            self._constitutive_update(particles, L, dt)

    # ------------------------------------------------------------------
    def run(self, particles: ParticleSet, n_steps: int, callback=None,
            callback_every: int = 100) -> ParticleSet:
        for s in range(n_steps + 1):
            if callback is not None and s % callback_every == 0:
                callback(s, s * self.dt, particles)
            if s < n_steps:
                self.step(particles)
        return particles
