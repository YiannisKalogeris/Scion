"""
The level set and the static confinement it supplies.

The decisive check is not that the kernel integrals look right, it is that
they are *exactly* the sums the fluid would have got from mirror particles.
Two tests pin that: a lattice filling a closed tank reads the reference
density to 1e-15 at the wall and in the corner, and a constant pressure
produces a net force of 4e-15 there. Either would fail for any error in the
cut-cell fraction, the correlation sense, or the grid alignment.
"""

import numpy as np
import pytest

from scion.discretization.sph import WendlandC2, lattice
from scion.discretization.sph.fluid import FluidSolver, WeaklyCompressibleFluid
from scion.discretization.sph.levelset import (LevelSet, box_interior_distance,
                                               polygon_distance)

DP = 0.025
H = 1.3 * DP


def half_space(spacing=DP / 4):
    """Solid below ``y = 0``."""
    return LevelSet(lambda p: -p[:, 1], [-0.3, -0.3], [0.3, 0.3], spacing,
                    "wendland_c2", H)


def test_box_distance_is_positive_outside():
    points = np.array([[0.5, 0.5], [0.5, 0.1], [-0.1, 0.5], [0.0, 0.5]])
    got = box_interior_distance(points, [0.0, 0.0], [1.0, 1.0])
    assert got == pytest.approx([-0.5, -0.1, 0.1, 0.0], abs=1e-15)


def test_polygon_distance_is_positive_inside():
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert polygon_distance(np.array([[0.1, 0.1]]), triangle)[0] > 0.0
    assert polygon_distance(np.array([[1.0, 1.0]]), triangle)[0] < 0.0
    # on an edge the distance is zero whichever side it is called from
    assert abs(polygon_distance(np.array([[0.5, 0.0]]), triangle)[0]) < 1e-15


def test_half_space_kernel_integral_is_antisymmetric():
    """``K(d) + K(-d) = 1``: what falls inside the solid plus what falls
    outside is the whole kernel. The sharpest check on the cut-cell
    fraction, and it needs no reference value."""
    level_set = half_space()
    d = np.linspace(-0.06, 0.06, 41)
    points = np.stack([np.zeros_like(d), d], axis=1)
    K = level_set.kernel_integral_at(points)
    assert np.abs(K + K[::-1] - 1.0).max() < 1e-4
    assert level_set.kernel_integral_at(np.array([[0.0, 0.0]]))[0] == pytest.approx(0.5, abs=1e-3)
    assert level_set.kernel_integral_at(np.array([[0.0, -0.1]]))[0] == pytest.approx(1.0, abs=1e-3)
    assert level_set.kernel_integral_at(np.array([[0.0, 0.1]]))[0] == pytest.approx(0.0, abs=1e-9)


def test_kernel_gradient_is_the_derivative_of_the_kernel_integral():
    """``G = dK/dd`` for a half-space, by definition of the two integrals.
    An identity between the two fields, so it catches a sign or a factor in
    either without needing a reference value for either."""
    level_set = half_space()
    d = np.linspace(-0.06, 0.06, 61)
    points = np.stack([np.zeros_like(d), d], axis=1)
    K = level_set.kernel_integral_at(points)
    G = level_set.kernel_gradient_at(points)[:, 1]
    finite = np.gradient(K, d[1] - d[0])
    window = slice(5, -5)
    assert np.abs(finite[window] - G[window]).max() < 0.05 * np.abs(G).max()


def test_kernel_gradient_is_normal_to_a_flat_wall():
    """Exactly, by symmetry -- so this is asserted at round-off."""
    level_set = half_space()
    d = np.linspace(-0.05, 0.05, 21)
    points = np.stack([np.zeros_like(d), d], axis=1)
    G = level_set.kernel_gradient_at(points)
    assert np.abs(G[:, 0]).max() < 1e-14
    assert (G[:, 1] <= 0.0).all()        # points into the solid


def test_the_level_set_grid_must_match_the_particle_spacing():
    """The trap: a *finer* grid is worse, not better.

    The density summation normalises by the lattice number density, so the
    wall must contribute the mirror-*lattice* sum. At a grid spacing equal to
    the particle spacing, with the wall halfway between grid rows, every cell
    is wholly in or out and the integral reproduces that sum exactly. Refine
    the grid and it converges instead to the continuum integral, which is 9%
    larger here -- so the near-wall density reads high and the fluid is blown
    off the wall.
    """
    kernel = WendlandC2()
    d = DP / 2
    depth = int(np.ceil(kernel.support(H) / DP)) + 2
    xs = np.arange(-depth, depth + 1) * DP
    ys = -(np.arange(depth) + 0.5) * DP
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    mirror_sum = float((kernel.W(np.hypot(X, Y - d), H, 2) * DP * DP).sum())

    coarse = half_space(DP).kernel_integral_at(np.array([[0.0, d]]))[0]
    fine = half_space(DP / 8).kernel_integral_at(np.array([[0.0, d]]))[0]
    assert coarse == pytest.approx(mirror_sum, rel=1e-6)
    assert fine > 1.08 * mirror_sum


def _closed_tank(spacing=DP):
    """A lattice filling a closed tank, with the tank as the solid."""
    side, margin = 1.0, 4 * spacing
    fluid = WeaklyCompressibleFluid(rho0=1.0, c0=14.14, viscosity=0.0)
    x, _ = lattice([0.0, 0.0], [side, side], spacing)
    wall = LevelSet(lambda p: box_interior_distance(p, [0.0, 0.0], [side, side]),
                    [-margin, -margin], [side + margin, side + margin],
                    spacing, "wendland_c2", 1.3 * spacing)
    solver = FluidSolver(x, spacing, fluid, lower=[-margin, -margin],
                         upper=[side + margin, side + margin], confinement=wall)
    solver.density_summation()
    return solver, wall, x


def test_confinement_restores_the_density_at_a_wall_and_in_a_corner():
    """The wall contributes exactly the neighbours the kernel is missing, so
    a lattice filling a closed tank is indistinguishable from an infinite
    one. Without the term the wall row reads about 30% light."""
    solver, _, x = _closed_tank()
    assert np.abs(solver.density - 1.0).max() < 1e-12
    corner = (x[:, 0] < DP) & (x[:, 1] < DP)
    assert np.abs(solver.density[corner] - 1.0).max() < 1e-12


def test_a_constant_pressure_produces_no_force_at_a_wall():
    """The identity that makes the confinement a mirror rather than an
    approximation to one: the truncated fluid sum and the solid integral
    cancel to round-off, at the wall and in the corner."""
    solver, wall, x = _closed_tank()
    solver.pressure = np.ones(len(x))
    i, j = solver.pairs.i, solver.pairs.j
    pair = -((solver.pressure[i] + solver.pressure[j]) * solver.dw_ij
             * solver.volume[i] * solver.volume[j])[:, None] * solver.pairs.e
    force = np.zeros_like(x)
    np.add.at(force, i, pair)
    np.add.at(force, j, -pair)
    gradient = wall.kernel_gradient_at(solver.positions)
    force -= (2.0 * solver.mass * solver.pressure / solver.density)[:, None] * gradient
    assert np.abs(force).max() < 1e-13


def test_surface_bounding_pushes_a_particle_back_out():
    solver, wall, _ = _closed_tank()
    solver.positions[0] = np.array([0.5, -0.01])       # inside the floor
    solver.bound_to_surface(wall)
    assert solver.positions[0][1] == pytest.approx(0.5 * DP, abs=1e-12)
