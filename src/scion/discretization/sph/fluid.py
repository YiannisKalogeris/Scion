r"""
Weakly compressible SPH: the dual-criteria scheme SPHinXsys runs its fluid
examples with.

Why weakly compressible at all
------------------------------
The pressure comes from an equation of state rather than from a Poisson
solve, so there is no global system and no incompressibility constraint --
the price is an acoustic time step set by an artificial sound speed, which is
taken as ten times the reference velocity so the density varies by about 1%.
That ratio is the whole design: smaller and the flow is compressible, larger
and the time step collapses.

The dual-criteria loop, which is not an optimization
-----------------------------------------------------
An advection step ``Dt`` (CFL on the particle velocity) carries the neighbour
list, the density summation, the viscous force and the transport-velocity
shift. Inside it, several acoustic steps ``dt`` (CFL on the sound speed)
carry only the pressure and continuity updates, **on a frozen neighbour
list**. SPHinXsys says why in a comment: the configuration does not change
materially over one acoustic step. Rebuilding every acoustic step is a
different scheme, not a more accurate one, and costs an order of magnitude.

Each acoustic step is itself two halves
---------------------------------------
``pressure_step`` advances density by a half step, computes the pressure,
moves the particles by a half step, builds the pressure force and updates the
velocity. ``density_step`` moves the other half and advances the density by
the other half. The two halves straddle the position update, which is what
makes the scheme second order in the Verlet sense; collapsing them into one
update is first order and still looks stable.

Where the Riemann dissipation goes
----------------------------------
The acoustic Riemann solver contributes a density dissipation in the first
half (``(p_i - p_j)/(rho0 c0)``) and a pressure dissipation in the second
(``rho0 c0 u_jump``, limited). SPHinXsys's Taylor-Green case uses the Riemann
form in the first half and *not* in the second, deliberately -- the flow is
viscous and the transport-velocity formulation already adds dissipation. That
choice is the ``riemann_density`` / ``riemann_pressure`` pair here, so the
combination is stated rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kernels import get_kernel
from .levelset import LevelSet
from .neighbors import build_neighbors
from .operators import kernel_pair_values

__all__ = ["WeaklyCompressibleFluid", "lattice_number_density", "FluidSolver"]


@dataclass(frozen=True)
class WeaklyCompressibleFluid:
    """Linear equation of state, ``p = c0^2 (rho - rho0)``.

    SPHinXsys's default, not the Tait/Cole form with an exponent of seven.
    They agree to first order in the density variation and the linear one has
    no exponent to get wrong; at the usual 1% density swing the difference is
    a few percent of the pressure.
    """

    rho0: float
    c0: float
    viscosity: float = 0.0

    @property
    def p0(self) -> float:
        return self.rho0 * self.c0 * self.c0

    def pressure(self, rho):
        return self.p0 * (np.asarray(rho, dtype=float) / self.rho0 - 1.0)

    def density(self, p):
        return self.rho0 * (np.asarray(p, dtype=float) / self.p0 + 1.0)


def lattice_number_density(kernel, h: float, spacing: float, dim: int) -> float:
    """``sigma0``: the kernel sum over an infinite regular lattice.

    The density summation divides by this, so that a particle in an
    undisturbed lattice reads exactly ``rho0`` -- it is the normalisation
    that turns a kernel sum into a density, and it depends on ``h/dp`` and on
    the kernel together. Computing it from the actual lattice rather than
    from a continuum integral is what makes the reading exact instead of
    approximate: the integral would give ``1/spacing^dim`` and be wrong by
    the discretisation of the kernel.
    """
    kernel = get_kernel(kernel)
    depth = int(kernel.support(h) / spacing) + 1
    axis = np.arange(-depth, depth + 1) * spacing
    grid = np.meshgrid(*([axis] * dim), indexing="ij")
    r = np.sqrt(sum(g**2 for g in grid))
    return float(kernel.W(r[r < kernel.support(h)], h, dim).sum())


class FluidSolver:
    """Weakly compressible SPH on a periodic (or unbounded) box.

    State is the particle positions, velocities and densities; the mass is
    fixed at ``rho0 dp^dim`` and the volume follows as ``m/rho``, which is
    what couples the density summation to every other operator.

    Walls, free surfaces and fluid-structure coupling are **not** here: this
    is the inner relation alone, which is what the Taylor-Green case needs
    and what the wall and level-set cases will build on.
    """

    def __init__(self, positions, spacing, fluid: WeaklyCompressibleFluid, *,
                 h_ratio: float = 1.3, kernel="wendland_c2",
                 lower=None, upper=None, periodic=None,
                 riemann_density: bool = True, riemann_pressure: bool = False,
                 riemann_limiter: float = 3.0,
                 transport_coefficient: float = 0.2,
                 transport_limiter: float = 100.0,
                 confinement=None, free_surface: bool = False):
        self.positions = np.array(positions, dtype=float, copy=True)
        self.n, self.dim = self.positions.shape
        self.spacing = float(spacing)
        self.fluid = fluid
        self.kernel = get_kernel(kernel)
        self.h = h_ratio * self.spacing
        self.lower = np.asarray(lower, dtype=float) if lower is not None else None
        self.upper = np.asarray(upper, dtype=float) if upper is not None else None
        self.periodic = (False,) * self.dim if periodic is None else tuple(periodic)
        self.riemann_density = riemann_density
        self.riemann_pressure = riemann_pressure
        self.riemann_limiter = float(riemann_limiter)
        self.transport_coefficient = float(transport_coefficient)
        self.transport_limiter = float(transport_limiter)
        #: static confinement: one or more level sets whose interior is
        #: solid, replacing wall particles entirely -- see `levelset.py`
        if confinement is None:
            self.confinement = []
        elif isinstance(confinement, LevelSet):
            self.confinement = [confinement]
        else:
            self.confinement = list(confinement)
        #: a free surface cannot hold a density below the reference one, and
        #: a kernel truncated by it reads exactly that
        self.free_surface = bool(free_surface)

        self.sigma0 = lattice_number_density(self.kernel, self.h, self.spacing, self.dim)
        self.mass = np.full(self.n, fluid.rho0 * self.spacing**self.dim)
        self.density = np.full(self.n, fluid.rho0)
        self.volume = self.mass / self.density
        self.velocity = np.zeros((self.n, self.dim))
        self.pressure = np.zeros(self.n)
        self.drho_dt = np.zeros(self.n)
        #: forces that do not change over an advection step -- gravity and
        #: the viscous force -- kept apart from the pressure force because
        #: they are evaluated on a different clock
        self.force_prior = np.zeros((self.n, self.dim))
        self.force = np.zeros((self.n, self.dim))
        self.gravity = np.zeros(self.dim)
        self.time = 0.0
        self.pairs = None
        self.update_configuration()

    # -- configuration --------------------------------------------------
    def wrap(self) -> None:
        if self.lower is None:
            return
        length = self.upper - self.lower
        for d, flag in enumerate(self.periodic):
            if flag:
                self.positions[:, d] = (self.lower[d]
                                        + np.mod(self.positions[:, d] - self.lower[d],
                                                 length[d]))

    def update_configuration(self) -> None:
        """Rebuild the neighbour list and the kernel values on it."""
        self.wrap()
        self.pairs = build_neighbors(self.positions, self.kernel.support(self.h),
                                     lower=self.lower, upper=self.upper,
                                     periodic=self.periodic)
        self.w_ij, self.dw_ij = kernel_pair_values(self.pairs, self.kernel,
                                                   self.h, self.dim)

    # -- the operators of one advection step ----------------------------
    def density_summation(self) -> None:
        """``rho_i = (W(0) + sum_j W_ij) rho0 / sigma0``, then ``V = m/rho``."""
        sigma = np.full(self.n, self.kernel.W(0.0, self.h, self.dim))
        np.add.at(sigma, self.pairs.i, self.w_ij)
        np.add.at(sigma, self.pairs.j, self.w_ij)
        for level_set in self.confinement:
            reference_volume = self.mass / self.fluid.rho0
            sigma += level_set.kernel_integral_at(self.positions) / reference_volume
        self.density = sigma * self.fluid.rho0 / self.sigma0
        if self.free_surface:
            self.density = np.maximum(self.density, self.fluid.rho0)
        self.volume = self.mass / self.density

    def viscous_force(self) -> np.ndarray:
        """The Morris-type viscous force, ``2 mu (v_i - v_j)/r dW V_i V_j``.

        The ``0.01 h`` in the denominator is SPHinXsys's regularisation of
        the ``1/r``; it matters only for a pair that has collapsed, and
        removing it turns a close encounter into a division by zero rather
        than a large force.
        """
        if self.fluid.viscosity == 0.0:
            return np.zeros((self.n, self.dim))
        dv = self.velocity[self.pairs.i] - self.velocity[self.pairs.j]
        weight = (2.0 * self.fluid.viscosity * self.dw_ij
                  / (self.pairs.r + 0.01 * self.h))
        pair_force = (weight * self.volume[self.pairs.i]
                      * self.volume[self.pairs.j])[:, None] * dv
        force = np.zeros((self.n, self.dim))
        np.add.at(force, self.pairs.i, pair_force)
        np.add.at(force, self.pairs.j, -pair_force)
        return force

    def transport_velocity_shift(self) -> None:
        """Shift particles down the kernel-gradient inconsistency.

        ``sum_j 2 grad W_ij V_j`` is zero for a perfect distribution, so it
        measures how far the local arrangement is from one; stepping along it
        keeps particles from clumping under a tensile pressure. The limiter
        truncates the shift, which is what keeps it from fighting the flow.
        """
        gw = (2.0 * self.dw_ij)[:, None] * self.pairs.e
        inconsistency = np.zeros((self.n, self.dim))
        np.add.at(inconsistency, self.pairs.i,
                  -gw * self.volume[self.pairs.j][:, None])
        np.add.at(inconsistency, self.pairs.j,
                  gw * self.volume[self.pairs.i][:, None])
        squared = (inconsistency * inconsistency).sum(axis=1)
        limiter = np.minimum(self.transport_limiter * self.h**2 * squared, 1.0)
        self.positions += (self.transport_coefficient * self.h**2
                           * limiter[:, None] * inconsistency)

    # -- the two halves of an acoustic step -----------------------------
    def pressure_step(self, dt: float) -> None:
        self.density += 0.5 * dt * self.drho_dt
        self.pressure = self.fluid.pressure(self.density)
        self.positions += 0.5 * dt * self.velocity

        p_sum = self.pressure[self.pairs.i] + self.pressure[self.pairs.j]
        pair_force = -(p_sum * self.dw_ij * self.volume[self.pairs.i]
                       * self.volume[self.pairs.j])[:, None] * self.pairs.e
        force = np.zeros((self.n, self.dim))
        np.add.at(force, self.pairs.i, pair_force)
        np.add.at(force, self.pairs.j, -pair_force)
        self.force += force

        if self.riemann_density:
            jump = ((self.pressure[self.pairs.i] - self.pressure[self.pairs.j])
                    / (self.fluid.rho0 * self.fluid.c0))
            term = jump * self.dw_ij
            rate = np.zeros(self.n)
            np.add.at(rate, self.pairs.i, term * self.volume[self.pairs.j])
            np.add.at(rate, self.pairs.j, -term * self.volume[self.pairs.i])
            self.drho_dt = rate * self.density
        else:
            self.drho_dt = np.zeros(self.n)

        for level_set in self.confinement:
            # the solid as a mirror at equal pressure: the fluid term with
            # p_j = p_i and the neighbour sum replaced by the integral
            gradient = level_set.kernel_gradient_at(self.positions)
            self.force -= ((2.0 * self.mass * self.pressure / self.density)[:, None]
                           * gradient)

        self.velocity += dt * (self.force_prior + self.force) / self.mass[:, None]

    def density_step(self, dt: float) -> None:
        self.positions += 0.5 * dt * self.velocity

        u_jump = ((self.velocity[self.pairs.i] - self.velocity[self.pairs.j])
                  * self.pairs.e).sum(axis=1)
        term = u_jump * self.dw_ij
        rate = np.zeros(self.n)
        np.add.at(rate, self.pairs.i, term * self.volume[self.pairs.j])
        np.add.at(rate, self.pairs.j, term * self.volume[self.pairs.i])
        self.drho_dt += rate * self.density

        if self.riemann_pressure:
            rho0c0 = self.fluid.rho0 * self.fluid.c0
            limited = np.minimum(self.riemann_limiter
                                 * np.maximum(u_jump, 0.0) / self.fluid.c0, 1.0)
            pair_force = ((rho0c0 * u_jump * limited * self.dw_ij
                           * self.volume[self.pairs.i]
                           * self.volume[self.pairs.j])[:, None] * self.pairs.e)
            force = np.zeros((self.n, self.dim))
            np.add.at(force, self.pairs.i, pair_force)
            np.add.at(force, self.pairs.j, -pair_force)
            self.force = force
        else:
            self.force = np.zeros((self.n, self.dim))

        for level_set in self.confinement:
            # the same mirror with the velocity reversed, which is no slip
            gradient = level_set.kernel_gradient_at(self.positions)
            self.drho_dt += self.density * 2.0 * (self.velocity * gradient).sum(axis=1)

        self.density += 0.5 * dt * self.drho_dt
        for level_set in self.confinement:
            self.bound_to_surface(level_set)

    def bound_to_surface(self, level_set, margin: float | None = None) -> None:
        """Push back any particle that has entered the solid.

        The confinement force is strong but not infinite, so a particle can
        cross during a step; left there it reads a kernel integral above one
        half, is pushed harder, and the error feeds itself. The bound is half
        a particle spacing outside the surface, which is where a surface
        particle sits on an undisturbed lattice.
        """
        distance = 0.5 * self.spacing if margin is None else float(margin)
        phi = level_set.signed_distance(self.positions)
        crossed = phi > -distance
        if not crossed.any():
            return
        normal = level_set.normal(self.positions[crossed])
        self.positions[crossed] -= (phi[crossed] + distance)[:, None] * normal

    # -- time steps -----------------------------------------------------
    def acoustic_time_step(self, cfl: float = 0.6) -> float:
        speed = self.fluid.c0 + np.linalg.norm(self.velocity, axis=1).max()
        return cfl * self.h / speed

    def advection_time_step(self, reference_speed: float, cfl: float = 0.25) -> float:
        """CFL on the particle speed, with the viscous speed as a floor.

        The acceleration term is what keeps this finite at ``t = 0`` of a
        gravity-driven problem, where every velocity is zero and the naive
        rule divides by nothing.
        """
        total = self.force_prior + self.force
        scale = 4.0 * self.h * np.linalg.norm(total, axis=1) / self.mass
        speed = np.sqrt(np.maximum((self.velocity * self.velocity).sum(axis=1),
                                   scale).max())
        viscous = self.fluid.viscosity / self.fluid.rho0 / self.h
        return cfl * self.h / max(speed, reference_speed, viscous)

    # -- the loop -------------------------------------------------------
    def advance(self, end_time: float, reference_speed: float,
                callback=None) -> None:
        """March to ``end_time`` through the dual-criteria loop."""
        while self.time < end_time:
            self.force_prior = self.viscous_force() + self.gravity * self.mass[:, None]
            advection_dt = self.advection_time_step(reference_speed)
            self.density_summation()
            self.force_prior = self.viscous_force() + self.gravity * self.mass[:, None]
            self.transport_velocity_shift()

            elapsed = 0.0
            while elapsed < advection_dt:
                dt = min(self.acoustic_time_step(), advection_dt - elapsed)
                self.pressure_step(dt)
                self.density_step(dt)
                elapsed += dt
                self.time += dt
            self.update_configuration()
            if callback is not None:
                callback(self)

    # -- diagnostics ----------------------------------------------------
    def kinetic_energy(self) -> float:
        return float(0.5 * (self.mass * (self.velocity ** 2).sum(axis=1)).sum())

    def maximum_speed(self) -> float:
        return float(np.linalg.norm(self.velocity, axis=1).max())
