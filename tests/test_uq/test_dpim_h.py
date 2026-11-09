"""
Verification for DPIM-H, the Heaviside form that gives probabilities directly.

DPIM-S smooths the Diracs of the discrete probability density integral
equation into a density and pays a bandwidth for it; DPIM-H integrates them
over an event instead, and integrating a Dirac needs no smoothing. So the only
error here is the point set's quadrature error, which makes the checks
sharper than anything in ``test_dpim.py``: against a closed-form CDF, and
against exact identities that hold at any point count.

python -m pytest tests/test_uq/test_dpim_h.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from scion.uq.density import (
    RepresentativePoints,
    band_probability,
    exceedance_probability,
    lattice,
    response_spacing,
    silverman_bandwidth,
    spacing_uniformity,
)
from scion.uq.forward import JointIndependent, Normal


def _standard_normal(n: int) -> RepresentativePoints:
    return lattice(JointIndependent([Normal(0.0, 1.0)]), [n])


# ---------------------------------------------------------------------------
# Against a closed form
# ---------------------------------------------------------------------------

def test_it_converges_to_the_exact_survival_function():
    """``Pr[Z > t] = 1 - Phi(t)`` for a standard normal response, and the
    error falls with the point count. No bandwidth appears anywhere, which is
    the whole point of the Heaviside form."""
    t = np.linspace(-2.5, 2.5, 51)
    errors = []
    for n in (64, 256, 1024):
        points = _standard_normal(n)
        got = exceedance_probability(points, points.z[:, 0], t)
        errors.append(float(np.abs(got - norm.sf(t)).max()))

    assert errors[0] < 0.02
    assert errors[-1] < 0.002
    assert errors[0] > errors[1] > errors[2]


def test_a_nonlinear_monotone_map_is_just_as_exact():
    """Nothing in the method sees the map, so a strongly nonlinear response
    costs no accuracy -- unlike the density, where it is exactly what defeats
    a global bandwidth."""
    points = _standard_normal(1024)
    z = points.z[:, 0]
    t = np.linspace(0.05, 6.0, 40)

    got = exceedance_probability(points, z ** 2, t)
    exact = 2.0 * norm.sf(np.sqrt(t))               # chi-square with 1 dof

    assert np.abs(got - exact).max() < 3e-3


# ---------------------------------------------------------------------------
# Exact identities, at any point count
# ---------------------------------------------------------------------------

def test_the_bands_of_a_stack_sum_to_exactly_one():
    """The identity the regime curves rest on: with ordered per-point
    transition loads, the four bands partition the probability at every
    threshold. Exact, not to a tolerance, because each band is closed on one
    side only."""
    rng = np.random.default_rng(0)
    n = 128
    points = _standard_normal(n)
    a = rng.uniform(0.5, 1.5, n)
    b = a + rng.uniform(0.0, 1.0, n)
    c = b + rng.uniform(0.0, 1.0, n)
    lam = np.linspace(0.0, 4.0, 80)

    elastic = exceedance_probability(points, a, lam, strict=False)
    shakedown = band_probability(points, a, b, lam)
    alternating = band_probability(points, b, c, lam)
    ratchetting = 1.0 - exceedance_probability(points, c, lam, strict=False)
    total = elastic + shakedown + alternating + ratchetting

    assert np.abs(total - 1.0).max() < 1e-15
    for band in (elastic, shakedown, alternating, ratchetting):
        assert band.min() >= -1e-15


def test_it_is_a_weighted_count_and_nothing_more():
    """Checked against the sum written out by hand, so the implementation
    cannot drift from Eq. (dpimh)."""
    points = RepresentativePoints(np.array([[0.0], [1.0], [2.0], [3.0]]),
                                  np.array([0.1, 0.2, 0.3, 0.4]))
    values = np.array([5.0, 1.0, 7.0, 3.0])

    assert exceedance_probability(points, values, 4.0) == pytest.approx(0.4)
    assert exceedance_probability(points, values, 0.0) == pytest.approx(1.0)
    assert exceedance_probability(points, values, 8.0) == pytest.approx(0.0)


def test_strict_and_closed_differ_exactly_at_a_tie():
    """And the tie is deliberate in the intended use, where the thresholds are
    load factors and the values are transition loads, so a sweep lands on a
    point's own transition load. The gap is that point's own probability,
    0.4 here."""
    points = RepresentativePoints(np.array([[0.0], [1.0], [2.0], [3.0]]),
                                  np.array([0.1, 0.2, 0.3, 0.4]))
    values = np.array([5.0, 1.0, 7.0, 3.0])

    assert exceedance_probability(points, values, 3.0, strict=True) \
        == pytest.approx(0.4)
    assert exceedance_probability(points, values, 3.0, strict=False) \
        == pytest.approx(0.8)


def test_an_infinite_value_means_never_and_is_not_an_error():
    """A realisation that never leaves a regime has an infinite transition
    load, so it must stay counted at every finite threshold."""
    points = RepresentativePoints(np.array([[0.0], [1.0]]),
                                  np.array([0.25, 0.75]))

    got = exceedance_probability(points, np.array([np.inf, 2.0]), 1e9)
    assert got == pytest.approx(0.25)


def test_the_result_takes_the_shape_of_the_thresholds():
    points = _standard_normal(32)
    z = points.z[:, 0]

    assert np.isscalar(exceedance_probability(points, z, 0.0)) or \
        np.ndim(exceedance_probability(points, z, 0.0)) == 0
    assert exceedance_probability(points, z, np.zeros(7)).shape == (7,)


def test_mismatched_lengths_and_unordered_bands_are_refused():
    points = _standard_normal(16)

    with pytest.raises(ValueError, match="against 16 representative"):
        exceedance_probability(points, np.zeros(5), 0.0)

    with pytest.raises(ValueError, match="upper >= lower"):
        band_probability(points, np.ones(16), np.zeros(16), 0.5)


# ---------------------------------------------------------------------------
# The sample-family bandwidth
# ---------------------------------------------------------------------------

def test_silverman_matches_the_textbook_rule_on_a_standard_normal():
    """``0.9 min(s, IQR/1.34) N^{-1/5}`` with ``s = 1``. The quantile lattice
    puts the interquartile range marginally below the normal's, so the rule
    picks that branch and lands just under the nominal value."""
    points = _standard_normal(256)

    assert silverman_bandwidth(points, points.z[:, 0]) \
        == pytest.approx(0.9 * 256 ** -0.2, rel=0.01)


def test_it_scales_as_the_fifth_root_and_not_as_the_spacing():
    """The whole point of having both rules. Over a 16x range in the point
    count the sample rule falls by ``16^{-1/5} = 1.74`` while the spacing
    between adjacent responses falls by 16 -- so on a scattered response set
    the spacing rule undersmooths by a factor that grows with ``N``."""
    widths, gaps = [], []
    for n in (64, 1024):
        points = _standard_normal(n)
        widths.append(silverman_bandwidth(points, points.z[:, 0]))
        gaps.append(response_spacing(points.z[:, 0]))

    assert widths[0] / widths[1] == pytest.approx(16 ** 0.2, rel=0.02)
    assert gaps[0] / gaps[1] > 8.0


def test_the_factor_scales_it_for_a_sensitivity_study():
    points = _standard_normal(128)
    base = silverman_bandwidth(points, points.z[:, 0])

    for factor in (0.5, 2.0):
        assert silverman_bandwidth(points, points.z[:, 0], factor=factor) \
            == pytest.approx(factor * base, rel=1e-12)


def test_it_is_robust_to_a_heavy_tail_where_the_standard_deviation_is_not():
    """``min(s, IQR/1.34)`` is the robust half of the rule: one far outlier
    inflates the standard deviation without moving the interquartile range,
    and a bandwidth taken from ``s`` alone would oversmooth the whole
    density. Measured here: the robust rule gives 0.38 of the naive one."""
    n = 256
    values = np.concatenate([np.linspace(-2.0, 2.0, n - 1), [60.0]])
    points = RepresentativePoints(values[:, None], np.full(n, 1.0 / n))

    robust = silverman_bandwidth(points, values)
    naive = 0.9 * float(np.sqrt(np.cov(values, bias=True))) * n ** -0.2

    assert robust < 0.45 * naive


def test_a_degenerate_response_is_refused_rather_than_floored():
    """Every realization coinciding is the ``t = 0`` instant of a dynamic run,
    and a bandwidth of zero there would destroy the grid through the padding
    rather than fail."""
    points = _standard_normal(32)

    with pytest.raises(ValueError, match="no spread"):
        silverman_bandwidth(points, np.full(32, 2.0))


def test_the_uniformity_diagnostic_is_what_selects_between_the_rules():
    """A tensor lattice's responses are regular and the spacing rule suits
    them; a nonlinear map of several inputs is not, and that is the case this
    bandwidth exists for."""
    lattice_points = _standard_normal(256)
    regular = lattice_points.z[:, 0]
    scattered = regular ** 2                      # strongly nonlinear map

    assert spacing_uniformity(regular) < spacing_uniformity(scattered)
    assert silverman_bandwidth(lattice_points, scattered) > 0.0
