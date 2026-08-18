"""
Particle <-> grid transfer operators.

PIC, FLIP and APIC differ *only* in these two kernels, which is why they belong
behind one interface rather than being baked into the stepper:

* PIC  -- dissipative; loses the angular momentum stored between particle and node
* FLIP -- transfers the velocity *increment*; noisy, needs blending with PIC
* APIC -- carries an affine velocity field per particle; conserves angular
          momentum and has neither PIC's damping nor FLIP's noise

APIC is the sensible default and is barely more code than PIC.

Reference: Jiang et al. (2015), "The affine particle-in-cell method".
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .background_grid import CartesianGrid, StencilData
from .particles import ParticleSet


class Transfer(ABC):
    """Strategy for moving momentum between particles and the grid."""

    @abstractmethod
    def p2g(self, particles: ParticleSet, grid: CartesianGrid,
            st: StencilData) -> tuple[np.ndarray, np.ndarray]:
        """Scatter mass and momentum. Returns ``(grid_mass, grid_momentum)``."""

    @abstractmethod
    def g2p(self, particles: ParticleSet, grid: CartesianGrid,
            st: StencilData, grid_v: np.ndarray, dt: float) -> None:
        """Gather velocity back to particles. Updates particle state in place."""


class APIC(Transfer):
    """Affine particle-in-cell."""

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()

        wm = st.w * particles.mass[:, None]                     # (n, K)
        np.add.at(grid_m, st.node_id, wm)

        # m_p (v_p + C_p (x_i - x_p))
        affine = particles.v[:, None, :] + np.einsum(
            "pab,pkb->pka", particles.C, st.dpos
        )
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * affine)
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]                                  # (n, K, dim)

        v_new = np.einsum("pk,pka->pa", st.w, vi)
        C_new = grid.basis.apic_inv_D(grid.dx) * np.einsum(
            "pk,pka,pkb->pab", st.w, vi, st.dpos
        )
        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)

        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.C = C_new
        particles.x = particles.x + dt * v_new


class PIC(Transfer):
    """Particle-in-cell. Kept as a reference: it is APIC with C = 0."""

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        wm = st.w * particles.mass[:, None]
        np.add.at(grid_m, st.node_id, wm)
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * particles.v[:, None, :])
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]
        v_new = np.einsum("pk,pka->pa", st.w, vi)
        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)
        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.x = particles.x + dt * v_new


class FLIP(Transfer):
    """FLIP/PIC blend. ``alpha = 1`` is pure FLIP, ``0`` is pure PIC."""

    def __init__(self, alpha: float = 0.95):
        self.alpha = float(alpha)
        self._v_old_grid = None

    def p2g(self, particles, grid, st):
        grid_m = grid.zeros_scalar()
        grid_mv = grid.zeros_vector()
        wm = st.w * particles.mass[:, None]
        np.add.at(grid_m, st.node_id, wm)
        np.add.at(grid_mv, st.node_id, wm[:, :, None] * particles.v[:, None, :])
        active = grid_m > 1.0e-12
        v_old = np.zeros_like(grid_mv)
        v_old[active] = grid_mv[active] / grid_m[active, None]
        self._v_old_grid = v_old
        return grid_m, grid_mv

    def g2p(self, particles, grid, st, grid_v, dt):
        vi = grid_v[st.node_id]
        dvi = (grid_v - self._v_old_grid)[st.node_id]
        v_pic = np.einsum("pk,pka->pa", st.w, vi)
        v_flip = particles.v + np.einsum("pk,pka->pa", st.w, dvi)
        v_new = self.alpha * v_flip + (1.0 - self.alpha) * v_pic

        grad_v = np.einsum("pka,pkb->pab", vi, st.grad_w)
        eye = np.eye(particles.dim, dtype=particles.dtype)
        particles.F = np.einsum("pij,pjk->pik", eye + dt * grad_v, particles.F)
        particles.v = v_new
        particles.x = particles.x + dt * v_pic


TRANSFER_REGISTRY: dict[str, type[Transfer]] = {
    "pic": PIC,
    "flip": FLIP,
    "apic": APIC,
}


def get_transfer(name: str, **kwargs) -> Transfer:
    if name not in TRANSFER_REGISTRY:
        raise NotImplementedError(f"Unknown transfer scheme '{name}'")
    return TRANSFER_REGISTRY[name](**kwargs)
