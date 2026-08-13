"""
The thick-walled cylinder against its closed-form solution.

This is the only test in the plasticity suite that compares against a result
derived outside the library. Everything else verifies the code against finite
differences of its own residual, which catches inconsistency but not a shared
misconception -- if the yield criterion and the return map were wrong in the
same way, every other test would still pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from examples.plasticity.thick_cylinder import (
    A_IN,
    B_OUT,
    YIELD,
    elastic_limit,
    limit_pressure,
    pressure_for_front,
    radial_stress_fully_plastic,
    run,
    yield_plane_strain,
)


# ---------------------------------------------------------------------------
# The analytic expressions, checked against each other
# ---------------------------------------------------------------------------

def test_plane_strain_yield_relation():
    """sigma_theta - sigma_r = 2Y/sqrt(3) when sigma_z is intermediate."""
    assert yield_plane_strain(YIELD) == pytest.approx(2 * YIELD / np.sqrt(3))


def test_front_equation_reproduces_both_endpoints():
    assert pressure_for_front(A_IN) == pytest.approx(elastic_limit())
    assert pressure_for_front(B_OUT) == pytest.approx(limit_pressure())


def test_the_pressure_rises_monotonically_with_the_front():
    c = np.linspace(A_IN, B_OUT, 100)
    assert np.all(np.diff(pressure_for_front(c)) > 0)


def test_fully_plastic_radial_stress_satisfies_its_boundary_conditions():
    assert radial_stress_fully_plastic(B_OUT) == pytest.approx(0.0)
    assert radial_stress_fully_plastic(A_IN) == pytest.approx(
        -limit_pressure())


# ---------------------------------------------------------------------------
# The computed solution against the analytic one
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def solution():
    asm, ops, history, result = run(n_r=10, n_theta=10, n_steps=30,
                                    verbose=False)
    return asm, ops, history, result


def test_every_load_step_converges(solution):
    _, _, _, result = solution
    assert all(s.converged for s in result.step_info)
    assert max(s.nr_iterations for s in result.step_info) <= 12


def test_limit_pressure_matches_the_closed_form(solution):
    """The strongest single check: p_lim = 2Y ln(b/a) / sqrt(3) contains no
    elastic constant, so it tests the yield surface and the assembled
    equilibrium without depending on how well the mesh resolves elastic
    strains."""
    _, _, history, _ = solution
    computed = history[:, 1].max()
    assert computed == pytest.approx(limit_pressure(), rel=0.01)


def test_the_response_is_elastic_below_the_elastic_limit(solution):
    """Below p_e the pressure-displacement curve must be a straight line."""
    _, _, history, _ = solution
    below = history[history[:, 1] < 0.9 * elastic_limit()]
    assert len(below) >= 3
    u, p = below[:, 0], below[:, 1]
    stiffness = p / u
    assert stiffness.std() / stiffness.mean() < 0.02


def test_the_plastic_zone_grows_as_predicted(solution):
    """Equation (1) relates the pressure to the front position; checking the
    whole path tests far more than the two endpoints do."""
    _, _, history, _ = solution
    moved = history[(history[:, 2] > A_IN * 1.02)
                    & (history[:, 2] < B_OUT * 0.98)]
    assert len(moved) >= 3

    errors = [abs(p - pressure_for_front(c)) / pressure_for_front(c)
              for _, p, c in moved]
    assert max(errors) < 0.06        # one element width of front resolution


def test_the_front_only_advances(solution):
    """The elastic-plastic boundary cannot retreat under monotonic loading."""
    _, _, history, _ = solution
    assert np.all(np.diff(history[:, 2]) >= -1e-9)


def test_yielding_starts_at_the_bore(solution):
    """Not somewhere in the middle of the wall, which is what a sign error in
    the pressure or a mismatched symmetry condition would produce."""
    asm, ops, history, _ = solution
    first = history[history[:, 2] > A_IN * 1.001]
    assert len(first)
    assert first[0, 2] < A_IN * 1.35


def test_the_computed_limit_is_not_sensitive_to_the_mesh():
    """Two meshes differing by a factor of four in element count must agree far
    more closely than either differs from the analytic value."""
    _, _, h1, _ = run(n_r=6, n_theta=6, n_steps=25, verbose=False)
    _, _, h2, _ = run(n_r=12, n_theta=12, n_steps=25, verbose=False)
    p1, p2 = h1[:, 1].max(), h2[:, 1].max()
    assert abs(p1 - p2) / p2 < 0.01
    for p in (p1, p2):
        assert p == pytest.approx(limit_pressure(), rel=0.01)
