"""
Verification for the closed-form Bree regime map.

The point of this file is that three of the four boundaries are *derived*
rather than transcribed, so they can be checked against something other than
the paper they came from. The shakedown boundary is Melan's problem on the
uniaxial strip, which is small enough to solve as a linear program here, and
the elastic one is the peak elastic stress, which is arithmetic. Only the
alternating-to-ratchetting curve ``x y = 1`` is taken on authority, and what is
checked for it is that it meets the others where it must.

python -m pytest tests/test_shakedown/test_bree.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from scion.shakedown import (
    BREE_CORNER,
    elastic_boundary,
    plastic_strain_range_exact,
    regime,
    rsdm_strip,
    shakedown_boundary,
    strip_quadrature,
    transition_loads,
)

RAYS = [(0.1, 0.8), (0.1, 2.0), (0.2, 1.0), (0.25, 4.0), (0.4, 0.5),
        (0.5, 2.0), (0.6, 0.3), (0.8, 1.0), (0.9, 0.2), (1.0, 3.0),
        (0.05, 1.0), (0.7, 2.8)]


def melan_factor(x: float, y: float, n: int = 201) -> float:
    """Melan's lower bound for the strip, as a linear program.

    Unknowns ``(lambda, rho_1..rho_n)``. The admissibility constraints are
    linear in both, because the elastic stress is fixed data, so this is one LP
    and not a bisection -- and it is an independent route to the boundary, not
    a rearrangement of the formula under test.

    Only ``g = 0`` and ``g = 1`` are imposed: the constraint is linear in the
    thermal multiplier, so the ends of the cycle bind.
    """
    zeta = np.linspace(-1.0, 1.0, n)
    w = np.full(n, 2.0 / (n - 1))
    w[0] = w[-1] = 1.0 / (n - 1)                  # trapezoid, for INT rho dzeta

    c = np.zeros(n + 1)
    c[0] = -1.0                                    # maximise lambda

    rows, rhs = [], []
    for elastic in (np.full(n, x), x + y * zeta):
        for sign in (+1.0, -1.0):
            block = np.zeros((n, n + 1))
            block[:, 0] = sign * elastic
            block[:, 1:] = sign * np.eye(n)
            rows.append(block)
            rhs.append(np.ones(n))                 # units of the yield stress

    A_eq = np.zeros((1, n + 1))
    A_eq[0, 1:] = w
    res = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  A_eq=A_eq, b_eq=[0.0],
                  bounds=[(0.0, None)] + [(None, None)] * n, method="highs")
    assert res.success, res.message
    return float(res.x[0])


# ---------------------------------------------------------------------------
# The boundaries, against something other than the paper
# ---------------------------------------------------------------------------

def test_the_shakedown_boundary_is_melans_problem_on_the_strip():
    """Both branches, from a linear program that knows nothing about
    ``y = 2`` or ``y = 4(1 - x)``. They are the two constraints of that
    program, and which binds is the ``y >= 4x`` branch condition."""
    for x, y in RAYS:
        computed = melan_factor(x, y)
        stated = float(transition_loads(x, y).shakedown)
        assert computed == pytest.approx(stated, rel=1e-9), (x, y)


def test_the_elastic_limit_is_the_peak_elastic_stress():
    """``max |sigma^E| = x + y``, attained at the face at the top of the
    cycle. This is what fixes ``sigma_T`` as the surface value rather than the
    through-thickness range, and the two differ by a factor of two in every
    boundary."""
    zeta = np.linspace(-1.0, 1.0, 2001)
    g = np.linspace(0.0, 1.0, 51)[:, None]

    for x, y in RAYS:
        peak = np.abs(x + y * zeta[None, :] * g).max()
        assert peak == pytest.approx(x + y, rel=1e-12)
        assert float(transition_loads(x, y).elastic) == pytest.approx(
            1.0 / (x + y), rel=1e-12)


def test_the_shakedown_window_is_twice_first_yield_at_zero_primary_stress():
    """The classical result for a self-equilibrated cyclic stress, and the
    check that catches the range/surface convention being mixed: under
    ``lambda_el = 1/(x + y/2)`` this ratio goes to 1 instead, which would
    leave no shakedown region at all as ``x -> 0``."""
    ratios = []
    for x in (1e-2, 1e-4, 1e-8):
        loads = transition_loads(x, 1.0)
        ratios.append(float(loads.shakedown / loads.elastic))

    assert ratios[-1] == pytest.approx(2.0, rel=1e-6)
    assert ratios[0] > ratios[1] > ratios[2]              # monotone from above


def test_the_three_boundaries_meet_at_the_corner():
    """``(0.5, 2)``: ``y = 2``, ``y = 4(1 - x)`` and ``x y = 1`` all pass
    through it, which is the one place the transcribed curve can be checked
    against the two derived ones."""
    x, y = BREE_CORNER
    assert shakedown_boundary(x) == pytest.approx(y)
    assert 4.0 * (1.0 - x) == pytest.approx(y)
    assert x * y == pytest.approx(1.0)

    loads = transition_loads(x, y)
    assert float(loads.shakedown) == pytest.approx(1.0)
    assert float(loads.ratchet) == pytest.approx(1.0)


def test_alternating_plasticity_is_empty_exactly_when_y_is_below_4x():
    """``lambda_rr >= lambda_sd`` iff ``y >= 4x``, which is why the branch
    condition in the transition loads is the same inequality twice."""
    for x, y in RAYS + [(0.3, 1.19), (0.31, 1.2)]:
        if y == pytest.approx(4.0 * x):
            continue                      # on the branch itself, both agree
        loads = transition_loads(x, y)
        assert float(loads.ratchet) >= float(loads.shakedown) * (1 - 1e-12)
        assert bool(loads.alternating_is_empty) is (y < 4.0 * x)


def test_the_transition_loads_are_ordered():
    """``lambda_el <= lambda_sd <= lambda_rr`` for every ray, which is the
    ordering the regime table relies on."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1e-3, 2.0, 500)
    y = rng.uniform(1e-3, 6.0, 500)
    loads = transition_loads(x, y)

    assert np.all(loads.elastic <= loads.shakedown * (1.0 + 1e-12))
    assert np.all(loads.shakedown <= loads.ratchet * (1.0 + 1e-12))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_each_regime_is_entered_just_past_its_transition_load():
    """Walking up a ray that exercises all four regimes."""
    x_hat, y_hat = 0.1, 0.8                          # y/x = 8, ray A
    loads = transition_loads(x_hat, y_hat)
    eps = 1e-6

    assert loads.regime_at(loads.elastic * (1 - eps)) == "elastic"
    assert loads.regime_at(loads.elastic * (1 + eps)) == "shakedown"
    assert loads.regime_at(loads.shakedown * (1 + eps)) == "alternating"
    assert loads.regime_at(loads.ratchet * (1 + eps)) == "ratchetting"
    assert not loads.alternating_is_empty


def test_a_shallow_ray_skips_alternating_plasticity():
    """Ray B of the planned example, ``y/x = 1``: it leaves the shakedown
    region through ``y = 4(1 - x)`` and ratchets immediately."""
    loads = transition_loads(1.0, 1.0)

    assert loads.alternating_is_empty
    assert float(loads.shakedown) == pytest.approx(float(loads.ratchet))
    assert loads.regime_at(float(loads.shakedown) * 1.000001) == "ratchetting"


def test_the_wedge_above_the_shakedown_boundary_is_ratchetting():
    """``(0.6, 1.65)`` is above ``y = 4(1 - x)``, below ``y = 2`` and inside
    ``x y = 1``, so a classifier written as three independent inequalities
    assigns it to no regime at all. Defining the regime as the ray at
    ``lambda = 1`` is what closes the gap."""
    assert regime(0.6, 1.65) == "ratchetting"
    assert 1.65 > 4.0 * (1.0 - 0.6)                  # out of shakedown
    assert 1.65 < 2.0 and 0.6 * 1.65 < 1.0           # but inside both others


def test_the_regime_map_covers_the_plane():
    """No point anywhere in a dense sweep is unclassified."""
    x, y = np.meshgrid(np.linspace(0.0, 1.5, 150),
                       np.linspace(0.0, 5.0, 150))
    out = regime(x.ravel(), y.ravel())

    assert set(np.unique(out)) <= {"elastic", "shakedown", "alternating",
                                   "ratchetting"}
    assert len(set(np.unique(out))) == 4
    assert regime(0.0, 0.0) == "elastic"             # no load at all


# ---------------------------------------------------------------------------
# Degenerate rays
# ---------------------------------------------------------------------------

def test_a_purely_mechanical_ray_has_no_shakedown_margin():
    """With no thermal load the strip simply collapses at ``sigma_P =
    sigma_Y``, so all three factors coincide at ``1/x``."""
    loads = transition_loads(0.4, 0.0)

    assert float(loads.elastic) == pytest.approx(2.5)
    assert float(loads.shakedown) == pytest.approx(2.5)
    assert float(loads.ratchet) == pytest.approx(2.5)


def test_a_purely_thermal_ray_never_ratchets():
    """Nothing drives the accumulation, so alternating plasticity continues
    for ever and the ratchetting load is infinite."""
    loads = transition_loads(0.0, 1.0)

    assert float(loads.elastic) == pytest.approx(1.0)
    assert float(loads.shakedown) == pytest.approx(2.0)
    assert np.isinf(float(loads.ratchet))
    assert loads.regime_at(1e6) == "alternating"


def test_an_unloaded_ray_is_elastic_at_every_load_factor():
    loads = transition_loads(0.0, 0.0)
    assert np.isinf(float(loads.elastic))
    assert loads.regime_at(1e12) == "elastic"


def test_the_boundaries_broadcast_and_refuse_negative_input():
    loads = transition_loads(np.linspace(0.1, 1.0, 7), 0.5)
    assert loads.elastic.shape == (7,)

    with pytest.raises(ValueError, match="cannot be negative"):
        transition_loads(0.5, -1.0)


def test_the_boundary_curves_agree_with_the_transition_loads():
    """``elastic_boundary`` and ``shakedown_boundary`` are drawing helpers, so
    they have to be the same curves the load factors come from."""
    x = np.linspace(0.0, 0.99, 40)

    for curve in (elastic_boundary, shakedown_boundary):
        y = curve(x)
        keep = y > 0.0
        loads = transition_loads(x[keep], y[keep])
        which = (loads.elastic if curve is elastic_boundary
                 else loads.shakedown)
        assert np.allclose(which, 1.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# RSDM on the same problem, against the closed form
# ---------------------------------------------------------------------------

ETA = 1.0e-6


def _indicators(x, y, lam, waveform="hump"):
    result = rsdm_strip(x, y, lam, n_thickness=61, n_time=16,
                        max_iter=4000, tol=1e-11, waveform=waveform)
    return (result.cyclic_amplitude,
            result.ratchet_rate / result.plastic_strain_scale)


def _first_passage(x, y, which, lo, hi, waveform="hump", n_bisect=14):
    """Smallest load factor at which the indicator leaves the safe set."""
    def left(lam):
        amplitude, ratchet = _indicators(x, y, lam, waveform)
        return (amplitude > ETA or ratchet > ETA) if which == "sd" \
            else ratchet > ETA

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if left(mid) else (mid, hi)
    return 0.5 * (lo + hi)


def test_rsdm_reproduces_the_shakedown_boundary_on_both_branches():
    """One ray per branch. The closed form is the answer here, so this is a
    real external benchmark rather than a self-consistency check -- and it
    covers a two-parameter map, where the thick cylinder pins one number."""
    for x, y in [(0.10, 0.80), (0.40, 0.50)]:
        exact = float(transition_loads(x, y).shakedown)
        got = _first_passage(x, y, "sd", 0.3 * exact, 3.0 * exact)

        assert got == pytest.approx(exact, rel=2e-3), (x, y)


def test_rsdm_reproduces_the_ratchetting_boundary_taken_on_authority():
    """``x y = 1`` is the one boundary the Melan derivation cannot reach, so
    it is the one piece of the closed form that is transcribed. RSDM finding
    it independently is what validates it."""
    x, y = 0.10, 0.80
    loads = transition_loads(x, y)
    assert not loads.alternating_is_empty               # the branch that uses it

    exact = float(loads.ratchet)
    got = _first_passage(x, y, "rr", 0.9 * float(loads.shakedown), 3.0 * exact)

    assert got == pytest.approx(exact, rel=2e-3)


def test_the_regimes_agree_with_the_closed_form_across_the_plane():
    """Including both the alternating and the direct-to-ratchetting paths."""
    for x, y in [(0.10, 0.80), (0.05, 1.00), (0.40, 0.50), (0.60, 0.30)]:
        loads = transition_loads(x, y)
        for frac in (0.8, 1.3, 1.8):
            lam = frac * float(loads.shakedown)
            result = rsdm_strip(x, y, lam, n_thickness=61, n_time=16,
                                max_iter=4000, tol=1e-11)

            assert result.regime(ETA, ETA) == str(loads.regime_at(lam)), \
                (x, y, frac)


def test_the_shakedown_load_does_not_depend_on_the_cycle_SHAPE():
    """Melan's condition involves only the extremes of the cycle, so a smooth
    hump and a triangle between the same limits must give the same answer.
    Measured identical to five decimals, which is tighter than the bisection
    width -- so this is an exact property, not a tolerance."""
    x, y = 0.10, 0.80
    exact = float(transition_loads(x, y).shakedown)
    hump = _first_passage(x, y, "sd", 0.3 * exact, 3.0 * exact, "hump")
    triangle = _first_passage(x, y, "sd", 0.3 * exact, 3.0 * exact, "triangle")

    assert hump == pytest.approx(triangle, rel=1e-9)


def test_the_residual_field_carries_no_net_force_by_construction():
    """``rho_dot = mean(sigma_p) - sigma_p`` has zero thickness mean by
    inspection, so self-equilibrium is exact rather than solver-accurate --
    unlike the finite element path, where it holds to the linear solve's
    tolerance."""
    _, weight = strip_quadrature(61)
    result = rsdm_strip(0.10, 0.80, 1.3 * 2.5, n_thickness=61, n_time=16,
                        max_iter=2000, tol=1e-11)

    share = weight / weight.sum()
    net = np.einsum("p,tpi->ti", share, result.residual)
    assert np.abs(net).max() < 1e-14 * np.abs(result.residual).max()


def test_the_modulus_cancels_from_the_residual_stress():
    """A structure indeterminate to degree one has a residual stress history
    that does not depend on how stiff it is -- the modulus enters only the
    plastic strain, through ``eps^p = sigma_p/E'``."""
    kwargs = dict(n_thickness=61, n_time=16, max_iter=2000, tol=1e-11)
    soft = rsdm_strip(0.10, 0.80, 3.0, modulus=1.0, **kwargs)
    stiff = rsdm_strip(0.10, 0.80, 3.0, modulus=1000.0, **kwargs)

    assert np.abs(stiff.residual - soft.residual).max() == 0.0
    assert stiff.plastic_strain_range == pytest.approx(
        soft.plastic_strain_range / 1000.0, rel=1e-12)


def test_the_strain_range_is_structurally_right_and_quantitatively_not():
    """Zero in shakedown, positive and growing in alternating plasticity --
    so the indicator responds correctly -- while the *magnitude* is wrong by
    an order of magnitude against the closed form.

    Asserted as a known limitation rather than hidden: the 2012 iteration
    drives the residual stress with ``sigma_p`` itself, so its rate scale is
    a free penalty parameter and the physical state sits past the stability
    limit. See :func:`~scion.shakedown.rsdm_strip` for the measurements. This
    test fails the day the iteration is reformulated, which is the point.
    """
    kwargs = dict(n_thickness=61, n_time=16, max_iter=4000, tol=1e-11)
    safe = rsdm_strip(0.0, 1.5, 1.0, **kwargs)              # y = 1.5 < 2
    plastic = rsdm_strip(0.0, 3.0, 1.0, **kwargs)
    hotter = rsdm_strip(0.0, 4.0, 1.0, **kwargs)

    assert safe.plastic_strain_range < 1e-9
    assert plastic.plastic_strain_range > 0.0
    assert hotter.plastic_strain_range > plastic.plastic_strain_range

    exact = float(plastic_strain_range_exact(3.0, 1.0))
    assert exact == pytest.approx(1.0)
    assert plastic.plastic_strain_range < 0.2 * exact        # ~0.084, measured


def test_the_closed_form_strain_range_vanishes_at_the_boundary():
    """And does not depend on the primary stress, which is the claim worth
    pinning: the range at a point is set by the thermal part alone."""
    assert float(plastic_strain_range_exact(2.0, 1.0)) == 0.0
    assert float(plastic_strain_range_exact(1.0, 2.0)) == 0.0
    assert float(plastic_strain_range_exact(3.0, 1.0)) == pytest.approx(1.0)
    assert float(plastic_strain_range_exact(3.0, 1.0, modulus=200.0)) \
        == pytest.approx(1.0 / 200.0)


def test_the_thickness_quadrature_puts_nodes_on_the_faces():
    """Every boundary is attained at ``|zeta| = 1`` and nowhere else, so a
    midpoint rule would report a shakedown limit that is too high."""
    zeta, weight = strip_quadrature(5)

    assert zeta[0] == -1.0 and zeta[-1] == 1.0
    assert weight.sum() == pytest.approx(2.0)

    with pytest.raises(ValueError, match="three points"):
        strip_quadrature(2)
