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

Two material interfaces
-----------------------
``pk1(F) -> (P, J)``
    Hyperelastic. Stress is a function of the current deformation gradient and
    nothing else, so it can be evaluated at any point in the step.

``update(F, L, dt, state) -> (sigma, new_state)``
    Path-dependent: plasticity, damage, anything carrying history. Selected by
    passing ``state=`` to the constructor.

The second changes *when* the stress is computed. A history-carrying material
needs the velocity gradient over the step, which only exists after the grid
solve, so the constitutive update runs at the **end** of the step and the
internal force at the start uses the stress left by the previous one. That is
Update Stress Last, and it is what the MUSL schemes in the impact literature do.
The hyperelastic path is unchanged and still evaluates stress in place.

Plane strain is embedded, not reduced
-------------------------------------
A 2-D run carries ``(n, 2, 2)`` kinematics, and constitutive code in this
library is always full 3-D -- ``sigma_33`` is nonzero in plane strain and enters
``J2``, so a model handed 2-D tensors computes the wrong yield surface. The
stepper therefore promotes ``F`` and ``L`` into the 3-D corner of an identity,
calls the material, and takes the in-plane block of the stress back. The
out-of-plane component is not discarded -- it lives in the material's state,
where the yield calculation needs it -- it simply exerts no in-plane force.
"""

from __future__ import annotations

import numpy as np

from scion.core import small_linalg as sla

from .background_grid import CartesianGrid
from .particles import ParticleSet
from .transfer import Transfer


def _domain_kwarg(grid, particles) -> dict:
    """Pass the convected domain only to a grid that wants it."""
    from .cpdi import CPDIGrid
    if isinstance(grid, CPDIGrid):
        return {"domain": particles.domain}
    return {}


def _promote(tensor: np.ndarray, fill_diagonal: bool) -> np.ndarray:
    """Embed ``(n, d, d)`` in ``(n, 3, 3)``. ``d = 3`` passes straight through.

    ``fill_diagonal`` distinguishes a deformation gradient, whose out-of-plane
    entry is 1 in plane strain, from a velocity gradient, whose is 0.
    """
    tensor = np.asarray(tensor, dtype=float)
    d = tensor.shape[-1]
    if d == 3:
        return tensor
    out = np.zeros((tensor.shape[0], 3, 3))
    out[:, :d, :d] = tensor
    if fill_diagonal:
        out[:, 2, 2] = 1.0
    return out


class ExplicitMPMStepper:
    """Symplectic-Euler explicit MPM.

    Parameters
    ----------
    grid        : CartesianGrid
    material    : either ``pk1(F) -> (P, J)`` (hyperelastic) or
                  ``update(F, L, dt, state) -> (sigma, state)`` (path-dependent,
                  selected by passing ``state``)
    transfer    : Transfer (APIC by default)
    dt          : time step
    state       : initial history for a path-dependent material. Its presence
                  is what switches the stepper to the Update-Stress-Last path.
    body_force  : (dim,) acceleration, e.g. gravity
    boundaries  : sequence of GridBC applied after the grid velocity update
    domain_update : how the particle domain evolves, for domain bases such as
                  uGIMP. "none" (uGIMP, default) or "stretch" (cpGIMP); see
                  ``ParticleSet.update_domain``. Ignored by point bases.
    """

    def __init__(self, grid: CartesianGrid, material, transfer: Transfer,
                 dt: float, body_force=None, boundaries=(),
                 domain_update: str = "none", state=None):
        self.grid = grid
        self.material = material
        self.transfer = transfer
        self.dt = float(dt)
        self.body_force = (np.zeros(grid.dim) if body_force is None
                           else np.asarray(body_force, dtype=grid.dtype))
        self.boundaries = list(boundaries)
        self.domain_update = domain_update

        self.state = state
        if state is not None and not hasattr(material, "update"):
            raise TypeError(
                "a state was given but the material has no update(F, L, dt, "
                "state); a hyperelastic pk1 material carries no history and "
                "needs no state")
        #: Cauchy stress left by the previous step, ``(n, 3, 3)``. None until
        #: the first evaluation.
        self.stress = None

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
        if self.state is None:
            P, J = self.material.pk1(particles.F)
            VsigmaT = particles.vol0[:, None, None] * np.einsum(
                "pij,pkj->pik", P, particles.F
            )
            return -np.einsum("pij,pkj->pki", VsigmaT,
                              self.transfer.force_kernel(self.grid, st))

        # path-dependent: use the stress the previous step left behind. On the
        # very first call there is none, so evaluate the material at zero rate
        # -- which reads out the initial state rather than assuming it is
        # unstressed.
        if self.stress is None:
            self._constitutive_update(particles, np.zeros_like(particles.F),
                                      0.0)
        dim = particles.dim
        J = sla.det(particles.F)
        Vsigma = ((particles.vol0 * J)[:, None, None]
                  * self.stress[:, :dim, :dim])
        return -np.einsum("pij,pkj->pki", Vsigma,
                          self.transfer.force_kernel(self.grid, st))

    # ------------------------------------------------------------------
    def _constitutive_update(self, particles: ParticleSet, L: np.ndarray,
                             dt: float) -> None:
        """Advance the path-dependent material and store the new stress."""
        sigma, self.state = self.material.update(
            _promote(particles.F, fill_diagonal=True),
            _promote(L, fill_diagonal=False), dt, self.state)
        self.stress = sigma

    # ------------------------------------------------------------------
    def step(self, particles: ParticleSet) -> None:
        """Advance the particle state by one ``dt``, in place."""
        grid, dt = self.grid, self.dt

        # 1. stencil + P2G
        #    lp is passed unconditionally; point bases ignore it
        st = grid.build_stencil(particles.x, lp=getattr(particles, "lp", None),
                                **_domain_kwarg(grid, particles))
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

        #    positions must be folded back before the next stencil build; the
        #    stencil itself wraps indices, so an unwrapped particle finds nodes
        #    happily and nothing complains while the body leaves the domain
        if grid.any_periodic:
            grid.wrap_positions(particles.x)

        #    3b. path-dependent constitutive update, on the new F and the
        #        velocity gradient the grid solve just produced. Recomputed
        #        here rather than returned by g2p so the transfer classes stay
        #        untouched; it is one einsum over the same stencil.
        if self.state is not None:
            L = np.einsum("pka,pkb->pab", grid_v[st.node_id], st.grad_w)
            self._constitutive_update(particles, L, dt)

        #    domain follows the new F, so it must come after G2P.
        #    A CPDIGrid always needs it convected, whatever its grid basis --
        #    the domain is the method, not a property of the basis, and
        #    `requires_domain` is False for the linear hat CPDI usually pairs
        #    with, so keying off it alone leaves the domain frozen at t = 0.
        from .cpdi import CPDIGrid
        if isinstance(grid, CPDIGrid):
            particles.update_domain("cpdi")
        elif grid.basis.requires_domain and self.domain_update != "none":
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
