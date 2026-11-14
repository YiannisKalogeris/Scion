r"""
Total-Lagrangian SPH for an elastic solid.

Why total Lagrangian, and what it buys
--------------------------------------
The neighbour list, the kernel values, the particle volumes and the
correction matrix are all built **once**, in the reference configuration,
and never touched again. Nothing is searched, nothing is rebuilt, and the
tensile instability that an updated-Lagrangian SPH solid suffers from --
particles clumping under tension because the kernel gradient is evaluated on
the deformed spacing -- cannot arise, because the kernel never sees the
deformed configuration at all.

The price is that the formulation is only valid while the reference
neighbourhood remains a good description of the material, so it does not do
fracture, self-contact or very large distortion. For a beam oscillating at a
few percent of its length that is exactly the right trade.

The deformation gradient is integrated, not measured
-----------------------------------------------------
``F`` is advanced by ``dF/dt = (-sum_j (v_i - v_j) (x) grad W_ij V_j) B_i``
rather than recomputed from the positions each step. The two agree in exact
arithmetic and the integrated one is what the Verlet splitting needs: it is
the quantity that sits at the half step. Recomputing ``F`` from positions
inside the loop is a different scheme and loses second-order accuracy.

The correction matrix is transposed here, and that is not a typo
-----------------------------------------------------------------
``stress_PK1_B = P(F) B^T`` where the gradient correction enters the
*divergence* rather than the gradient. SPHinXsys's comment says the same.
Dropping the transpose is invisible on a symmetric ``B`` -- which it is on a
uniform lattice away from a surface -- and wrong at every free surface,
which for a beam is most of it.

Numerical dissipation is a pair term and it is needed
------------------------------------------------------
``0.25 (W_ij/W(0)) * 0.5 (F_i + F_j) * rho0 c0 h * strain_rate`` with
``strain_rate = (dim/r)^2 (x_i - x_j).(v_i - v_j)``. It is a Monaghan-type
artificial viscosity written in the reference frame; without it the beam
develops a short-wavelength ripple that grows. The factor 0.25 is
SPHinXsys's default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import get_kernel
from .neighbors import build_neighbors
from .operators import kernel_pair_values, linear_gradient_correction

__all__ = ["SaintVenantKirchhoffSolid", "ElasticSolver"]


@dataclass(frozen=True)
class SaintVenantKirchhoffSolid:
    """``S = lambda tr(E) I + 2 G E`` on the Green strain ``E``.

    The simplest finite-strain extension of Hooke's law: linear in ``E``,
    which is quadratic in ``F``. It is not polyconvex and misbehaves under
    large compression, which is irrelevant here and worth knowing before
    reusing it.

    The sound speed is built from the **bulk** modulus, ``c0 = sqrt(K/rho0)``
    with ``K = E/(3(1-2nu))``, which is what sets the time step and what the
    beam's initial velocity is scaled by.
    """

    rho0: float
    youngs_modulus: float
    poisson: float

    @property
    def shear_modulus(self) -> float:
        return 0.5 * self.youngs_modulus / (1.0 + self.poisson)

    @property
    def lame_lambda(self) -> float:
        return (self.youngs_modulus * self.poisson
                / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)))

    @property
    def bulk_modulus(self) -> float:
        return self.youngs_modulus / 3.0 / (1.0 - 2.0 * self.poisson)

    @property
    def c0(self) -> float:
        return float(np.sqrt(self.bulk_modulus / self.rho0))

    def stress_pk2(self, F: np.ndarray) -> np.ndarray:
        """``(n, d, d) -> (n, d, d)``."""
        dim = F.shape[-1]
        eye = np.eye(dim)
        green = 0.5 * (np.swapaxes(F, 1, 2) @ F - eye)
        trace = np.trace(green, axis1=1, axis2=2)
        return (self.lame_lambda * trace[:, None, None] * eye
                + 2.0 * self.shear_modulus * green)

    def stress_pk1(self, F: np.ndarray) -> np.ndarray:
        return F @ self.stress_pk2(F)

    def pair_damping(self, strain_rate, smoothing_length: float):
        return 0.5 * self.rho0 * self.c0 * strain_rate * smoothing_length


class ElasticSolver:
    """Stress relaxation by Verlet time stepping, on a fixed reference list."""

    def __init__(self, positions, volumes, solid: SaintVenantKirchhoffSolid, *,
                 h_ratio: float = 1.3, kernel="wendland_c2", spacing=None,
                 dissipation: float = 0.25):
        self.reference = np.array(positions, dtype=float, copy=True)
        self.positions = self.reference.copy()
        self.n, self.dim = self.reference.shape
        self.volumes = np.broadcast_to(np.asarray(volumes, dtype=float),
                                       (self.n,)).copy()
        self.solid = solid
        self.kernel = get_kernel(kernel)
        self.spacing = (float(spacing) if spacing is not None
                        else float(np.mean(self.volumes) ** (1.0 / self.dim)))
        self.h = h_ratio * self.spacing
        self.dissipation = float(dissipation)

        self.mass = solid.rho0 * self.volumes
        self.velocity = np.zeros((self.n, self.dim))
        self.force = np.zeros((self.n, self.dim))
        self.force_prior = np.zeros((self.n, self.dim))
        eye = np.eye(self.dim)
        self.F = np.broadcast_to(eye, (self.n, self.dim, self.dim)).copy()
        self.dF_dt = np.zeros((self.n, self.dim, self.dim))
        self.time = 0.0

        # everything below is the REFERENCE configuration and never changes
        self.pairs = build_neighbors(self.reference, self.kernel.support(self.h))
        self.w_ij, self.dw_ij = kernel_pair_values(self.pairs, self.kernel,
                                                   self.h, self.dim)
        self.B = linear_gradient_correction(self.pairs, self.volumes,
                                            self.dw_ij, self.dim)
        self.weight = self.w_ij / self.kernel.W(0.0, self.h, self.dim)

    # -- the two halves -------------------------------------------------
    def first_half(self, dt: float) -> None:
        self.positions += 0.5 * dt * self.velocity
        self.F += 0.5 * dt * self.dF_dt
        stress = self.solid.stress_pk1(self.F) @ np.swapaxes(self.B, 1, 2)

        i, j = self.pairs.i, self.pairs.j
        jump_x = self.positions[i] - self.positions[j]
        jump_v = self.velocity[i] - self.velocity[j]
        # the strain rate is measured on the REFERENCE separation, which is
        # what keeps the damping from stiffening as the body deforms
        scale = self.dim / self.pairs.r
        strain_rate = scale * scale * (jump_x * jump_v).sum(axis=1)
        damping = self.solid.pair_damping(strain_rate, self.h)
        numerical = (0.5 * (self.F[i] + self.F[j])
                     * (self.dissipation * self.weight * damping)[:, None, None])

        total = stress[i] + stress[j] + numerical
        traction = np.einsum("nab,nb->na", total, self.pairs.e) * self.dw_ij[:, None]
        force = np.zeros_like(self.force)
        np.add.at(force, i, (self.mass[i] / self.solid.rho0
                             * self.volumes[j])[:, None] * traction)
        np.add.at(force, j, -(self.mass[j] / self.solid.rho0
                              * self.volumes[i])[:, None] * traction)
        self.force = force
        self.velocity += dt * (self.force_prior + self.force) / self.mass[:, None]

    def second_half(self, dt: float) -> None:
        self.positions += 0.5 * dt * self.velocity
        i, j = self.pairs.i, self.pairs.j
        jump_v = self.velocity[i] - self.velocity[j]
        grad_w = self.dw_ij[:, None] * self.pairs.e
        outer = -jump_v[:, :, None] * grad_w[:, None, :]
        rate = np.zeros_like(self.dF_dt)
        np.add.at(rate, i, self.volumes[j][:, None, None] * outer)
        np.add.at(rate, j, self.volumes[i][:, None, None] * outer)
        self.dF_dt = rate @ self.B
        self.F += 0.5 * dt * self.dF_dt

    def constrain(self, fixed) -> None:
        """Hold a region at its reference position and zero velocity.

        Applied *between* the two halves, as the reference case does: after
        the velocity update and before the deformation rate is formed, so the
        constrained particles contribute a zero velocity to their neighbours'
        ``dF/dt`` rather than a stale one.
        """
        self.positions[fixed] = self.reference[fixed]
        self.velocity[fixed] = 0.0

    # -- time step ------------------------------------------------------
    def time_step(self, cfl: float = 0.6) -> float:
        """``min(sqrt(h/|a|), h/(c0 + |v|))``, the smaller of the two limits.

        The acceleration branch is what controls the first step, when the
        velocity is zero everywhere and the sound-speed branch alone would
        allow a step that the forces do not.
        """
        acceleration = np.linalg.norm(
            (self.force + self.force_prior) / self.mass[:, None], axis=1)
        speed = np.linalg.norm(self.velocity, axis=1)
        by_force = np.sqrt(self.h / (acceleration + 1e-300))
        by_sound = self.h / (self.solid.c0 + speed)
        return float(cfl * np.minimum(by_force, by_sound).min())

    def step(self, dt: float, fixed=None) -> None:
        self.first_half(dt)
        if fixed is not None:
            self.constrain(fixed)
        self.second_half(dt)
        self.time += dt

    # -- diagnostics ----------------------------------------------------
    def displacement(self) -> np.ndarray:
        return self.positions - self.reference

    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass * (self.velocity ** 2).sum(axis=1)).sum())

    def strain_energy(self) -> float:
        """``sum V0 (S : E) / 2`` -- the Saint-Venant-Kirchhoff energy."""
        green = 0.5 * (np.swapaxes(self.F, 1, 2) @ self.F - np.eye(self.dim))
        stress = self.solid.stress_pk2(self.F)
        density = 0.5 * np.einsum("nab,nab->n", stress, green)
        return float((self.volumes * density).sum())

    def probe(self, point, field, h_ratio: float | None = None) -> np.ndarray:
        """Shepard interpolation of a particle field at one reference point.

        The observer follows the *material*: it is built on the reference
        configuration, so its weights never change. That is the whole point
        of a Lagrangian observer and is what SPHinXsys's observer body does.

        ``h_ratio`` overrides the body's smoothing length, because an
        observer body carries its own. The beam case sets 1.15 against the
        body's 1.3, and it matters: at 1.3 the tip probe reads 0.198772 at
        rest and at 1.15 it reads **0.198877**, which is SPHinXsys's recorded
        value to all six digits. Neither is wrong -- a Shepard average over a
        wider support reaches further back from the free end -- but comparing
        against their number requires their support.
        """
        from .operators import shepard_interpolate
        h = self.h if h_ratio is None else h_ratio * self.spacing
        return shepard_interpolate(point, self.reference, field, self.volumes,
                                   self.kernel, h, self.dim)
