"""
The weakly compressible fluid core.

Four of these are exact identities and the rest are external: a fluid at rest
must stay at rest, a uniform translation must stay rigid, an undisturbed
lattice must read the reference density, and a constant pressure must produce
no force. None of them can be passed by a scheme that is merely plausible,
and all four fail loudly for the usual mistakes -- a stray self-term, an
asymmetric pair sum, a missing `sigma0`, a sign on `dW/dr`.

The external checks are the Taylor-Green vortex: an exact solution of the
incompressible Navier-Stokes equations, and SPHinXsys's own recorded kinetic
energy for the same case.
"""

import numpy as np
import pytest

from examples.sph.taylor_green_2d import build, decay_rate, run
from scion.discretization.sph import lattice
from scion.discretization.sph.fluid import (FluidSolver, WeaklyCompressibleFluid,
                                            lattice_number_density)

SPACING = 0.025
BOX = ([0.0, 0.0], [1.0, 1.0])


def quiet_fluid(viscosity=0.0):
    fluid = WeaklyCompressibleFluid(rho0=1.0, c0=10.0, viscosity=viscosity)
    x, _ = lattice(*BOX, SPACING, periodic=(True, True))
    solver = FluidSolver(x, SPACING, fluid, lower=BOX[0], upper=BOX[1],
                         periodic=(True, True))
    solver.density_summation()
    return solver


def test_equation_of_state_round_trips():
    fluid = WeaklyCompressibleFluid(rho0=1000.0, c0=15.0)
    assert fluid.pressure(1000.0) == 0.0
    assert fluid.density(fluid.pressure(1013.0)) == pytest.approx(1013.0)


def test_undisturbed_lattice_reads_the_reference_density():
    """``sigma0`` is the lattice sum, so this is exact rather than close.
    Using the continuum ``1/dp^dim`` instead is wrong by about 1% at
    ``h = 1.3 dp``, which reads as a physical compression."""
    solver = quiet_fluid()
    assert np.abs(solver.density - 1.0).max() < 1e-12
    continuum = 1.0 / SPACING**2
    assert abs(solver.sigma0 / continuum - 1.0) > 5e-3   # the 1% that matters


def test_a_fluid_at_rest_stays_at_rest():
    solver = quiet_fluid()
    for _ in range(20):
        solver.pressure_step(1e-3)
        solver.density_step(1e-3)
    assert solver.maximum_speed() < 1e-13


def test_uniform_translation_is_rigid():
    solver = quiet_fluid()
    solver.velocity[:] = np.array([1.0, -0.5])
    for _ in range(20):
        solver.pressure_step(1e-3)
        solver.density_step(1e-3)
    assert np.abs(solver.velocity - np.array([1.0, -0.5])).max() < 1e-12
    assert np.ptp(solver.density) < 1e-12


def test_pressure_force_vanishes_for_a_constant_pressure_and_not_otherwise():
    """The SPH pressure operator is exact on a constant and first-order
    accurate on a gradient; both halves are asserted so a formulation that
    returned zero always would fail the second."""
    solver = quiet_fluid()
    pairs = solver.pairs

    def force_from(pressure):
        solver.pressure = pressure
        p_sum = solver.pressure[pairs.i] + solver.pressure[pairs.j]
        pair = -(p_sum * solver.dw_ij * solver.volume[pairs.i]
                 * solver.volume[pairs.j])[:, None] * pairs.e
        out = np.zeros_like(solver.positions)
        np.add.at(out, pairs.i, pair)
        np.add.at(out, pairs.j, -pair)
        return out / solver.mass[:, None]

    constant = force_from(np.full(len(solver.positions), 3.0))
    assert np.abs(constant).max() < 1e-11

    x = solver.positions[:, 0]
    acceleration = force_from(np.sin(2 * np.pi * x))
    exact = -2 * np.pi * np.cos(2 * np.pi * x) / solver.density
    assert np.abs(acceleration[:, 0] - exact).max() < 0.05 * np.abs(exact).max()
    assert np.abs(acceleration[:, 1]).max() < 0.05 * np.abs(exact).max()


def test_transport_shift_restores_a_disordered_lattice():
    """It must reduce the density scatter, not merely move particles. A sign
    slip there amplifies clumping, which reads as an instability much later
    and in a different part of the code."""
    fluid = WeaklyCompressibleFluid(rho0=1.0, c0=10.0)
    x, _ = lattice(*BOX, SPACING, periodic=(True, True))
    rng = np.random.default_rng(0)
    solver = FluidSolver(x + rng.uniform(-0.3, 0.3, x.shape) * SPACING, SPACING,
                         fluid, lower=BOX[0], upper=BOX[1], periodic=(True, True))
    solver.density_summation()
    before = float(np.ptp(solver.density))
    for _ in range(6):
        solver.transport_velocity_shift()
        solver.update_configuration()
        solver.density_summation()
    assert float(np.ptp(solver.density)) < 0.5 * before


def test_lattice_number_density_is_dimension_aware():
    from scion.discretization.sph import WendlandC2
    h, dp = 1.3 * 0.01, 0.01
    two = lattice_number_density(WendlandC2(), h, dp, 2)
    three = lattice_number_density(WendlandC2(), h, dp, 3)
    assert two == pytest.approx(1.0 / dp**2, rel=0.02)
    assert three == pytest.approx(1.0 / dp**3, rel=0.02)


# SPHinXsys master,
# tests/2d_examples/test_2d_taylor_green/regression_test_tool/
#   WaterBody_TotalKineticEnergy_Run_0_result.xml, snapshot_1 (t = 0.1).
# Their own run of this case at the same resolution.
SPHINXSYS_ENERGY_AT_0_1 = 0.20646315402108778


@pytest.fixture(scope="module")
def early_taylor_green():
    return run(n=100, end_time=0.1, interval=0.1, quiet=True)


def test_taylor_green_matches_the_closed_form_early(early_taylor_green):
    """Before the startup transient has dissipated anything the scheme has to
    track the exact solution. Later it does not, and that is a property of
    weakly compressible SPH rather than of this implementation -- see the
    comparison against SPHinXsys below, which loses the same energy."""
    _, history, rate, energy0 = early_taylor_green
    t, energy, _, _ = history[-1]
    exact = energy0 * np.exp(-2.0 * rate * t)
    assert energy / exact == pytest.approx(1.0, abs=0.05)


def test_taylor_green_matches_sphinxsys_at_the_same_instant(early_taylor_green):
    _, history, _, _ = early_taylor_green
    assert history[-1][1] == pytest.approx(SPHINXSYS_ENERGY_AT_0_1, rel=0.02)


def test_initial_energy_is_exactly_a_quarter(early_taylor_green):
    """``int 0.5 |u|^2`` over the unit square for this field is 1/4, and the
    particle sum reproduces it because the lattice integrates the squared
    cosine exactly. SPHinXsys records the same 0.25."""
    _, _, _, energy0 = early_taylor_green
    assert energy0 == pytest.approx(0.25, rel=1e-12)
