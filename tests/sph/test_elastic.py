"""
Total-Lagrangian SPH elasticity, and SPHinXsys example 13.

Four exact identities carry the solver -- an undeformed body is stress free,
a body at rest stays at rest, a rigid translation builds no strain, and a
rigid rotation builds no stress -- and the beam is checked against two
external references: the closed-form first-mode period of a cantilever, and
SPHinXsys's own recorded tip position.
"""

import numpy as np
import pytest

from examples.sph.oscillating_beam import (LENGTH, OBSERVER_H_RATIO, build,
                                           analytic_periods, mode_shape)
from scion.discretization.sph.elastic import (ElasticSolver,
                                              SaintVenantKirchhoffSolid)

SOLID = SaintVenantKirchhoffSolid(1.0e3, 2.0e6, 0.3975)

# SPHinXsys master, tests/2d_examples/test_2d_oscillating_beam/
#   regression_test_tool/BeamObserver_Position_ensemble_averaged_mean_variance.xml
TIP_AT_REST = (0.198877, 0.0)
TIP_AT_PEAK = (0.157345, 0.11298)


def block(nx=40, ny=10, dp=0.002):
    x = np.stack(np.meshgrid((np.arange(nx) + 0.5) * dp,
                             (np.arange(ny) + 0.5) * dp,
                             indexing="ij"), axis=-1).reshape(-1, 2)
    return ElasticSolver(x, np.full(len(x), dp * dp), SOLID, spacing=dp)


def test_small_strain_limit_is_hookes_law():
    """Catches a factor on either Lame constant, which no qualitative check
    would -- the beam would simply ring at the wrong frequency."""
    F = np.broadcast_to(np.eye(2), (1, 2, 2)).copy()
    F[0, 0, 0] = 1.0 + 1e-6
    P = SOLID.stress_pk1(F)[0]
    assert P[0, 0] == pytest.approx((SOLID.lame_lambda + 2 * SOLID.shear_modulus) * 1e-6,
                                    rel=1e-5)
    assert P[1, 1] == pytest.approx(SOLID.lame_lambda * 1e-6, rel=1e-5)


def test_an_undeformed_body_is_stress_free():
    F = np.broadcast_to(np.eye(2), (3, 2, 2)).copy()
    assert np.abs(SOLID.stress_pk1(F)).max() == 0.0


def test_sound_speed_comes_from_the_bulk_modulus():
    """It sets the time step and scales the beam's initial velocity, so an
    error here is a factor on both at once."""
    assert SOLID.c0 == pytest.approx(np.sqrt(SOLID.bulk_modulus / SOLID.rho0))
    assert SOLID.c0 == pytest.approx(57.0266, rel=1e-5)


def test_a_body_at_rest_does_not_move():
    solver = block()
    fixed = np.zeros(len(solver.positions), dtype=bool)
    for _ in range(20):
        solver.step(1e-6, fixed=fixed)
    assert np.abs(solver.velocity).max() < 1e-12
    assert np.abs(solver.positions - solver.reference).max() < 1e-15


def test_rigid_translation_builds_no_strain():
    solver = block()
    fixed = np.zeros(len(solver.positions), dtype=bool)
    solver.velocity[:] = np.array([1.0, -0.5])
    for _ in range(20):
        solver.step(1e-6, fixed=fixed)
    assert np.abs(solver.F - np.eye(2)).max() < 1e-12


def test_rigid_rotation_builds_no_stress():
    """Objectivity. A rotation rate leaves ``F`` a rotation, so the Green
    strain stays zero -- the check that the stress is built from ``F^T F``
    and not from the displacement gradient."""
    solver = block()
    fixed = np.zeros(len(solver.positions), dtype=bool)
    omega = 20.0
    centre = solver.reference.mean(axis=0)
    for _ in range(40):
        offset = solver.positions - centre
        solver.velocity[:] = omega * np.stack([-offset[:, 1], offset[:, 0]], axis=1)
        solver.step(2e-6, fixed=fixed)
    green = 0.5 * (np.swapaxes(solver.F, 1, 2) @ solver.F - np.eye(2))
    assert np.abs(green).max() < 1e-6
    assert solver.strain_energy() < 1e-6 * solver.kinetic_energy()


def test_the_clamp_holds_the_collar_and_not_the_beam():
    """Clamping the whole base would hold the first three centimetres of the
    beam rigid and shorten the cantilever, which shows up only as a wrong
    frequency much later."""
    solver, fixed = build()
    assert fixed.sum() == 312
    assert (solver.reference[fixed, 0] <= 0.0).all()
    # no particle on the beam's centreline inside the insert is clamped
    inside = (solver.reference[:, 0] > -0.06) & (np.abs(solver.reference[:, 1]) < 0.009)
    assert not fixed[inside].any()


def test_mode_shape_is_normalised_at_the_tip():
    assert mode_shape(np.array([1.0]))[0] == pytest.approx(1.0, rel=1e-10)
    assert mode_shape(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)


def test_tip_at_rest_matches_the_reference_observer():
    """No dynamics: this pins the lattice, the particle volumes and the
    observer's Shepard weights against their recorded value. It needs their
    observer smoothing length of 1.15 dp -- at the body's own 1.3 the probe
    reads 0.198772 instead."""
    solver, _ = build()
    tip = np.array([[LENGTH, 0.0]])
    x = solver.probe(tip, solver.positions[:, 0], h_ratio=OBSERVER_H_RATIO)[0]
    y = solver.probe(tip, solver.positions[:, 1], h_ratio=OBSERVER_H_RATIO)[0]
    assert x == pytest.approx(TIP_AT_REST[0], abs=1e-6)
    assert abs(y) < 1e-15


@pytest.fixture(scope="module")
def first_peak():
    """March to the first tip reversal, which is a quarter period."""
    solver, fixed = build()
    tip = np.array([[LENGTH, 0.0]])

    def probe():
        return np.array([solver.probe(tip, solver.positions[:, k],
                                      h_ratio=OBSERVER_H_RATIO)[0] for k in (0, 1)])

    best, when = probe(), 0.0
    energy = [solver.kinetic_energy() + solver.strain_energy()]
    while solver.time < 0.09:
        solver.step(solver.time_step(), fixed=fixed)
        here = probe()
        if here[1] > best[1]:
            best, when = here, solver.time
        energy.append(solver.kinetic_energy() + solver.strain_energy())
    return best, when, np.array(energy)


def test_peak_tip_position_matches_the_reference(first_peak):
    best, _, _ = first_peak
    assert best[1] == pytest.approx(TIP_AT_PEAK[1], rel=0.01)
    assert best[0] == pytest.approx(TIP_AT_PEAK[0], abs=1e-3)


def test_quarter_period_matches_the_cantilever_frequency(first_peak):
    """The first reversal is a quarter period, so the peak time is a
    frequency measurement -- a rate, which no amplitude tuning can reach.

    It lands on the **plane-stress** Euler-Bernoulli reading. The plane
    strain and Timoshenko readings are 10% away, and the test asserts that
    too: the beam is thick (``L/h = 10``) and its clamp is compliant, so
    which reading it should match is a measurement rather than a derivation.
    """
    _, when, _ = first_peak
    periods = analytic_periods(SOLID)
    assert 4.0 * when == pytest.approx(periods["plane stress"], rel=0.02)
    assert 4.0 * when / periods["plane strain"] > 1.05


def test_energy_is_conserved_through_the_first_swing(first_peak):
    """No damping is applied other than the pair dissipation, which is small
    here, so the total may drift slowly but must not grow.

    The 1e-5 allowance is not slack: a Verlet splitting conserves a shadow
    energy, not this one, so the measured total oscillates by about that much
    at the step level. What matters is that it does not *trend* upward --
    hence the second assertion, which compares the end against the start.
    """
    _, _, energy = first_peak
    assert energy.max() <= energy[0] * (1.0 + 1e-5)
    assert energy[-1] < energy[0]
    assert energy.min() > 0.99 * energy[0]
