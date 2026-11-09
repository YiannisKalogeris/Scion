"""Bounded supports and the atom a first-passage indicator carries.

The sharp test here is the one against a *known* density at the bound: an
unreflected kernel reads exactly half of it, which is not a tolerance but the
kernel losing the half of its mass that falls outside the support.
"""

import numpy as np
import pytest

from scion.uq.density import (MixedDensity, RepresentativePoints,
                              mixed_density, reflected_density)


def _points(n, rng, dim=2):
    return RepresentativePoints(rng.normal(size=(n, dim)), np.full(n, 1.0 / n))


def _integral(grid, y):
    return float(0.5 * np.sum((y[1:] + y[:-1]) * np.diff(grid)))


# ---------------------------------------------------------------------------
# the reflection
# ---------------------------------------------------------------------------

def test_an_unreflected_kernel_reads_exactly_half_at_the_bound():
    """Not a tolerance: a Gaussian centred *on* the bound puts half its mass
    outside the support, so the estimate there is half the truth as the
    bandwidth narrows. This is the whole reason the reflection exists."""
    from scipy.stats import norm

    # quantiles of the half-normal rather than a draw from it: at the bound
    # only samples within a few bandwidths contribute, so a random sample
    # carries several percent of Monte Carlo noise there and the test would
    # be measuring that instead of the kernel
    p = (np.arange(4000) + 0.5) / 4000
    sample = norm.ppf(0.5 * (1.0 + p))
    weights = np.full(sample.size, 1.0 / sample.size)
    grid = np.linspace(0.0, 4.0, 400)
    h = 0.05                                   # narrow, to isolate the edge

    reflected = reflected_density(grid, sample, weights, h)
    z = (grid[:, None] - sample[None, :]) / h
    plain = (np.exp(-0.5 * z * z) @ weights) / (h * np.sqrt(2.0 * np.pi))

    truth = np.sqrt(2.0 / np.pi)               # half-normal density at zero
    assert plain[0] / truth == pytest.approx(0.5, abs=0.01)
    assert reflected[0] / truth == pytest.approx(1.0, abs=0.02)


def test_reflection_returns_the_mass_rather_than_discarding_it():
    """``int p`` must equal the sum of the weights, not less: the reflected
    kernel folds the leaked tail back inside instead of losing it."""
    rng = np.random.default_rng(1)
    sample = np.abs(rng.normal(0.0, 1.0, 2000))
    weights = np.full(sample.size, 1.0 / sample.size)
    grid = np.linspace(0.0, 8.0, 4000)
    assert _integral(grid, reflected_density(grid, sample, weights, 0.2)) \
        == pytest.approx(1.0, abs=1.0e-3)


def test_the_weights_are_used_as_given_and_not_normalised():
    """What lets a caller scale a continuous part by ``1 - atom`` without a
    second normalisation."""
    rng = np.random.default_rng(2)
    sample = np.abs(rng.normal(0.0, 1.0, 500))
    grid = np.linspace(0.0, 8.0, 3000)
    half = np.full(sample.size, 0.5 / sample.size)
    assert _integral(grid, reflected_density(grid, sample, half, 0.25)) \
        == pytest.approx(0.5, abs=1.0e-3)


def test_reflection_works_about_an_upper_bound():
    """The mirror ``2 b - v`` is the same expression on either side, so a
    response bounded above needs no separate branch."""
    rng = np.random.default_rng(3)
    sample = -np.abs(rng.normal(0.0, 1.0, 3000))
    weights = np.full(sample.size, 1.0 / sample.size)
    grid = np.linspace(-4.0, 0.0, 400)
    reflected = reflected_density(grid, sample, weights, 0.05, bound=0.0)
    assert reflected[-1] / np.sqrt(2.0 / np.pi) == pytest.approx(1.0, abs=0.08)


def test_an_empty_sample_gives_an_empty_density():
    grid = np.linspace(0.0, 1.0, 10)
    out = reflected_density(grid, np.array([]), np.array([]), 0.1)
    assert out.shape == grid.shape and out.max() == 0.0


# ---------------------------------------------------------------------------
# the atom
# ---------------------------------------------------------------------------

def test_the_atom_is_the_probability_of_the_points_on_the_bound():
    rng = np.random.default_rng(4)
    values = np.abs(rng.normal(size=300))
    values[:75] = 0.0
    split = mixed_density(_points(300, rng), values)
    assert split.atom == pytest.approx(0.25, abs=1.0e-12)
    assert split.n_atom == 75
    assert split.n_continuous == 225


def test_the_atom_and_the_continuous_part_sum_to_one():
    """The convention that makes the pair a distribution rather than two
    separately normalised objects."""
    rng = np.random.default_rng(5)
    values = np.abs(rng.normal(size=400))
    values[:120] = 0.0
    split = mixed_density(_points(400, rng), values, n_cells=4000, span=4.0)
    assert split.mass == pytest.approx(1.0, abs=5.0e-3)
    assert _integral(split.grid, split.conditional) == pytest.approx(
        1.0, abs=5.0e-3)


def test_the_bandwidth_comes_from_the_continuous_part_alone():
    """Taking it over the whole set would let the atom's zeros widen the
    kernel that draws the part above the bound. Asserted as an exact equality
    because the continuous responses and their *relative* weights are
    untouched by how many points sit on the atom."""
    rng = np.random.default_rng(6)
    tail = np.abs(rng.normal(size=200)) + 0.1

    few = np.concatenate([np.zeros(50), tail])
    many = np.concatenate([np.zeros(400), tail])
    h_few = mixed_density(_points(few.size, rng), few).bandwidth
    h_many = mixed_density(_points(many.size, rng), many).bandwidth
    # not bit-identical: the subset weights are renormalised by a different
    # total, so the last ulp moves. Mathematically it is an equality, which
    # is why the tolerance is round-off and not a fraction of a percent
    assert h_few == pytest.approx(h_many, rel=1.0e-12)


def test_smoothing_through_the_atom_overstates_the_density_at_the_bound():
    """The measurement the split exists for: running one kernel over every
    point, atom included, piles the point mass onto the continuous part
    exactly where it is read."""
    rng = np.random.default_rng(7)
    values = np.concatenate([np.zeros(250), np.abs(rng.normal(size=250)) + 0.2])
    points = _points(values.size, rng)
    split = mixed_density(points, values)
    naive = reflected_density(split.grid, values, points.probability,
                              split.bandwidth)
    assert naive[0] > 2.0 * split.density[0]


def test_a_response_with_no_atom_is_an_ordinary_reflected_density():
    rng = np.random.default_rng(8)
    values = np.abs(rng.normal(size=200)) + 0.5
    split = mixed_density(_points(200, rng), values)
    assert split.atom == 0.0
    assert split.n_atom == 0
    assert np.abs(split.conditional - split.density).max() == 0.0


def test_an_upper_bounded_response_puts_its_grid_below_the_bound():
    rng = np.random.default_rng(9)
    values = -np.abs(rng.normal(size=200))
    values[:40] = 0.0
    split = mixed_density(_points(200, rng), values)
    assert split.grid[-1] == 0.0 and split.grid[0] < 0.0
    assert split.atom == pytest.approx(0.2, abs=1.0e-12)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_responses_straddling_the_bound_are_refused():
    """Then it is not a bound, and a mixed distribution is the wrong object."""
    rng = np.random.default_rng(10)
    values = rng.normal(size=100)
    with pytest.raises(ValueError, match="not a bound"):
        mixed_density(_points(100, rng), values)


def test_too_few_points_off_the_bound_is_refused_rather_than_guessed():
    rng = np.random.default_rng(11)
    values = np.zeros(100)
    values[0] = 1.0
    with pytest.raises(ValueError, match="not enough to choose a bandwidth"):
        mixed_density(_points(100, rng), values)
    # ...unless the caller supplies one
    split = mixed_density(_points(100, rng), values, bandwidth=0.2)
    assert split.atom == pytest.approx(0.99)


def test_an_all_atom_response_has_no_conditional_density():
    rng = np.random.default_rng(12)
    split = mixed_density(_points(50, rng), np.zeros(50), bandwidth=0.1)
    assert split.atom == pytest.approx(1.0)
    with pytest.raises(ValueError, match="every realization sits on the atom"):
        split.conditional


def test_a_mismatched_response_count_is_refused():
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError, match="against"):
        mixed_density(_points(20, rng), np.zeros(19))


def test_a_nonfinite_response_is_refused():
    rng = np.random.default_rng(14)
    values = np.abs(rng.normal(size=30))
    values[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mixed_density(_points(30, rng), values)


def test_a_nonpositive_bandwidth_is_refused():
    grid = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="positive and finite"):
        reflected_density(grid, np.ones(3), np.full(3, 1 / 3), 0.0)


def test_mismatched_values_and_weights_are_refused():
    grid = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="values against"):
        reflected_density(grid, np.ones(3), np.ones(2), 0.1)
