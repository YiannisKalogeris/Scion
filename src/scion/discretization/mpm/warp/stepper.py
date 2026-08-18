"""
Warp-backed explicit APIC-MPM stepper (2-D).

The device-side twin of ``ExplicitMPMStepper``. Same physics, same defaults,
same benchmarks; the difference is that all state lives in ``wp.array`` and the
per-step work is three kernel launches instead of a chain of NumPy operations.

Interop with the reference implementation is deliberate and cheap:

    wp_particles = WarpParticleSet.from_numpy(np_particles, device)
    ...
    np_particles = wp_particles.to_numpy(np_particles)

so any NumPy test case, sampler or verification script can be run on either
backend without rewriting the setup. That is what makes the cross-check test
possible.

Scope: quadratic B-spline basis, APIC transfer, compressible Neo-Hookean,
2-D, float32. Those are the choices baked into the kernels; anything else
needs a new kernel rather than a new argument. The NumPy backend remains the
place to prototype variants.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from . import kernels as K
from .device import get_device, sync

__all__ = ["WarpParticleSet", "WarpMPMStepper"]


class WarpParticleSet:
    """Particle state as Warp device arrays.

    Mirrors ``ParticleSet``: x, v, F, C, vol0, mass. No ``lp`` -- the kernels
    implement the quadratic B-spline basis, which needs no particle domain.
    """

    def __init__(self, x, v=None, vol0=None, density=1.0, device: str = "cpu"):
        self.device = device
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.n, self.dim = x.shape
        if self.dim != 2:
            raise NotImplementedError("the Warp backend is 2-D only")

        v = (np.zeros_like(x) if v is None
             else np.ascontiguousarray(v, dtype=np.float32))
        vol0 = (np.full(self.n, float(vol0), dtype=np.float32)
                if np.isscalar(vol0)
                else np.ascontiguousarray(vol0, dtype=np.float32))
        mass = (np.float32(density) * vol0).astype(np.float32)
        eye = np.tile(np.eye(2, dtype=np.float32), (self.n, 1, 1))

        self.x = wp.array(x, dtype=wp.vec2, device=device)
        self.v = wp.array(v, dtype=wp.vec2, device=device)
        self.F = wp.array(eye, dtype=wp.mat22, device=device)
        self.C = wp.array(np.zeros((self.n, 2, 2), np.float32),
                          dtype=wp.mat22, device=device)
        self.vol0 = wp.array(vol0, dtype=float, device=device)
        self.mass = wp.array(mass, dtype=float, device=device)

    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(cls, particles, device: str = "cpu") -> "WarpParticleSet":
        """Build from a reference :class:`ParticleSet`, carrying F and C over."""
        out = cls(particles.x, particles.v, particles.vol0, 1.0, device)
        out.mass = wp.array(particles.mass.astype(np.float32),
                            dtype=float, device=device)
        out.F = wp.array(particles.F.astype(np.float32),
                         dtype=wp.mat22, device=device)
        out.C = wp.array(particles.C.astype(np.float32),
                         dtype=wp.mat22, device=device)
        return out

    def to_numpy(self, particles=None):
        """Copy device state back. Fills ``particles`` in place if given."""
        x, v = self.x.numpy(), self.v.numpy()
        F, C = self.F.numpy(), self.C.numpy()
        if particles is not None:
            particles.x = x.astype(particles.dtype)
            particles.v = v.astype(particles.dtype)
            particles.F = F.astype(particles.dtype)
            particles.C = C.astype(particles.dtype)
            return particles
        return {"x": x, "v": v, "F": F, "C": C}

    # ------------------------------------------------------------------
    def linear_momentum(self) -> np.ndarray:
        return (self.mass.numpy()[:, None] * self.v.numpy()).sum(axis=0)

    def kinetic_energy(self) -> float:
        v = self.v.numpy()
        return float(0.5 * (self.mass.numpy() * (v ** 2).sum(axis=1)).sum())

    def jacobian(self) -> np.ndarray:
        F = self.F.numpy()
        return F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]


class WarpMPMStepper:
    """Explicit APIC-MPM on Warp.

    Parameters
    ----------
    shape       : (nx, ny) background grid cells
    dx          : cell size
    mu, lam     : compressible Neo-Hookean parameters
    dt          : time step
    body_force  : (2,) acceleration
    walls       : None, or ((lo_x, lo_y), (hi_x, hi_y))
    wall_mode   : "separating" (default) or "sticky"
    device      : "auto" | "cpu" | "gpu" | any Warp device string
    """

    PAD = 3          # matches CartesianGrid's padding for a 3-node stencil
    INV_D = 4.0      # APIC normalisation for quadratic B-splines: 4 / dx^2

    def __init__(self, shape, dx: float, mu: float, lam: float, dt: float,
                 body_force=(0.0, 0.0), walls=None,
                 wall_mode: str = "separating", device: str = "auto"):
        self.device = get_device(device)
        self.nx, self.ny = int(shape[0]), int(shape[1])
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.mu, self.lam = float(mu), float(lam)
        self.dt = float(dt)
        self.gravity = wp.vec2(float(body_force[0]), float(body_force[1]))

        self.node_nx = self.nx + 1 + 2 * self.PAD
        self.node_ny = self.ny + 1 + 2 * self.PAD
        self.n_nodes = self.node_nx * self.node_ny

        self.grid_m = wp.zeros(self.n_nodes, dtype=float, device=self.device)
        self.grid_mv = wp.zeros(self.n_nodes, dtype=wp.vec2, device=self.device)
        self.grid_v = wp.zeros(self.n_nodes, dtype=wp.vec2, device=self.device)

        if walls is None:
            self.wall_mode = 0
            self.lo = (0.0, 0.0)
            self.hi = (0.0, 0.0)
        else:
            self.wall_mode = 2 if wall_mode == "sticky" else 1
            self.lo = (float(walls[0][0]), float(walls[0][1]))
            self.hi = (float(walls[1][0]), float(walls[1][1]))

    # ------------------------------------------------------------------
    @property
    def inv_D(self) -> float:
        return self.INV_D * self.inv_dx * self.inv_dx

    def cfl_dt(self, density: float, safety: float = 0.3) -> float:
        c = np.sqrt((self.lam + 2.0 * self.mu) / density)
        return safety * self.dx / c

    # ------------------------------------------------------------------
    def step(self, p: WarpParticleSet) -> None:
        d = self.device
        wp.launch(K.zero_grid, dim=self.n_nodes,
                  inputs=[self.grid_m, self.grid_mv], device=d)
        wp.launch(K.p2g, dim=p.n,
                  inputs=[p.x, p.v, p.F, p.C, p.mass, p.vol0,
                          self.grid_m, self.grid_mv,
                          self.mu, self.lam,
                          self.dx, self.inv_dx, self.dt,
                          self.node_ny, self.PAD],
                  device=d)
        wp.launch(K.grid_update, dim=self.n_nodes,
                  inputs=[self.grid_m, self.grid_mv, self.grid_v,
                          self.gravity, self.dt,
                          self.dx, self.node_nx, self.node_ny, self.PAD,
                          self.lo[0], self.lo[1], self.hi[0], self.hi[1],
                          self.wall_mode],
                  device=d)
        wp.launch(K.g2p, dim=p.n,
                  inputs=[p.x, p.v, p.F, p.C, self.grid_v,
                          self.dx, self.inv_dx, self.dt,
                          self.inv_D, self.node_ny, self.PAD],
                  device=d)

    # ------------------------------------------------------------------
    def run(self, p: WarpParticleSet, n_steps: int, callback=None,
            callback_every: int = 100) -> WarpParticleSet:
        for s in range(n_steps + 1):
            if callback is not None and s % callback_every == 0:
                sync(self.device)          # results must be readable
                callback(s, s * self.dt, p)
            if s < n_steps:
                self.step(p)
        sync(self.device)
        return p
