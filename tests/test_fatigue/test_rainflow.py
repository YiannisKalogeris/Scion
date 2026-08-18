"""
Verification for rainflow counting.

The external benchmark is the worked example of ASTM E1049-85, which is the
only thing here that can catch the algorithm and the test sharing a
misconception. Everything else checks conservation properties, which are
necessary but not sufficient.

python -m pytest tests/test_fatigue/test_rainflow.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.fatigue.rainflow import (
    CycleCounts,
    rainflow,
    rainflow_matrix,
    turning_points,
)
from scion.fatigue.rainflow import _collapse, _extract

#: The sequence of ASTM E1049-85, rainflow counting example.
ASTM = [-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0]

#: Its published result, as (range, count) after aggregating equal ranges.
ASTM_EXPECTED = {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def _by_range(c: CycleCounts) -> dict[float, float]:
    out: dict[float, float] = {}
    for r, n in zip(c.ranges, c.counts):
        out[round(float(r), 9)] = out.get(round(float(r), 9), 0.0) + float(n)
    return out


# ---------------------------------------------------------------------------
# External benchmark
# ---------------------------------------------------------------------------

def test_astm_e1049_example():
    """The published count for the standard's own example sequence.

    This is the one test here that is not the code checking itself. ASTM
    E1049-85 counts residual reversals as half cycles, so residue="half" is the
    mode that reproduces it -- the default "repeat" answers a different
    question and gives different ranges.
    """
    assert _by_range(rainflow(ASTM, residue="half")) == \
        pytest.approx(ASTM_EXPECTED)


def test_astm_example_conserves_reversals_in_every_residue_mode():
    """Nine turning points are eight reversals, so four cycles' worth however
    the residue is treated. Only the distribution over ranges changes."""
    for mode in ("half", "repeat"):
        assert rainflow(ASTM, residue=mode).total == pytest.approx(4.0)


def test_repeat_mode_closes_the_residue_across_the_block_boundary():
    """The residue of the ASTM sequence holds the largest excursions. Chaining
    the block with itself pairs 5 with -4 and 4 with -3, which half-cycle
    counting reports as six unpaired halves instead."""
    got = _by_range(rainflow(ASTM, residue="repeat"))
    assert got == pytest.approx({3.0: 1.0, 4.0: 1.0, 7.0: 1.0, 9.0: 1.0})


def test_discarding_the_residue_throws_away_the_largest_cycles():
    """Which is why it is not the default: the range-9 cycle, worth more than
    all the others together on any power-law curve, is in the residue."""
    kept = rainflow(ASTM, residue="discard")
    assert kept.total == pytest.approx(1.0)
    assert kept.ranges.max() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_the_residue_is_a_fixed_point_of_the_extraction(seed):
    """Chaining the residue with itself closes cycles and leaves the *same*
    residue behind. That is what makes "repeat" well defined: the second block
    is in the same state as the first, so the count per block is stationary
    rather than drifting with the number of repetitions."""
    rng = np.random.default_rng(seed)
    v, ix = turning_points(rng.normal(size=300).cumsum())
    *_, rv, ri = _extract(v, ix)

    chained, keep = _collapse(np.concatenate([rv, rv]))
    *_, rv2, _ = _extract(chained, np.concatenate([ri, ri])[keep])
    assert np.allclose(rv2, rv)


@pytest.mark.parametrize("seed", range(8))
def test_half_mode_conserves_reversals(seed):
    """Every turning point belongs to exactly one counted reversal, so the
    total is half the number of reversals. A stack bug that drops or duplicates
    a point shows up here and nowhere else.

    Only "half" satisfies this exactly. Chaining a block with itself merges the
    turning points at the junction, so a repeated block genuinely has slightly
    fewer reversals per block than a single one -- see
    :func:`test_repeat_mode_matches_a_long_chained_history`.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=400).cumsum()
    v, _ = turning_points(x)
    assert rainflow(x, residue="half").total == \
        pytest.approx(0.5 * (len(v) - 1))


@pytest.mark.parametrize("seed", range(3))
def test_repeat_mode_matches_a_long_chained_history(seed):
    """The defining property of "repeat", and the reason it is the default.

    Counting one block in repeat mode must give what you would get by chaining
    the block many times, counting that, and dividing -- both in cycles and in
    the damage a power-law curve would compute from them. Anything else means
    the residual cycles at the block joins are being mishandled, which is where
    the largest ranges live.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=200).cumsum()

    def damage(c):
        return float((c.counts * c.ranges ** 3).sum())

    one = rainflow(x, residue="repeat")
    many = rainflow(np.concatenate([x] * 64), residue="half")

    assert many.total / 64.0 == pytest.approx(one.total, rel=2e-3)
    assert damage(many) / 64.0 == pytest.approx(damage(one), rel=0.03)


@pytest.mark.parametrize("seed", range(4))
def test_the_largest_range_is_the_span_of_the_history(seed):
    """No counted cycle can be wider than peak-to-trough, and one always
    reaches it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=300).cumsum()
    c = rainflow(x, residue="half")
    assert c.ranges.max() == pytest.approx(x.max() - x.min())


def test_constant_amplitude_gives_exactly_the_applied_cycles():
    """Ten sine cycles must count as ten cycles of the full range, not as some
    number of partial ones -- the property every downstream damage sum leans
    on."""
    t = np.linspace(0.0, 10.0 * 2 * np.pi, 4001)
    x = 150.0 * np.sin(t)
    c = rainflow(x, residue="repeat")
    assert c.total == pytest.approx(10.0, abs=0.5)
    big = c.counts[c.ranges > 299.0].sum()
    assert big == pytest.approx(10.0, abs=0.5)
    assert c.means.max() == pytest.approx(0.0, abs=1e-6)


def test_a_flat_history_has_no_cycles():
    assert len(rainflow(np.zeros(100))) == 0
    assert len(rainflow([1.0])) == 0
    assert len(rainflow([])) == 0


# ---------------------------------------------------------------------------
# Turning points and the gate
# ---------------------------------------------------------------------------

def test_turning_points_keeps_the_endpoints_and_drops_the_rest():
    x = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0]
    v, ix = turning_points(x)
    assert list(v) == [0.0, 3.0, 1.0, 2.0]
    assert list(ix) == [0, 3, 4, 5]


def test_turning_points_removes_repeated_values():
    v, _ = turning_points([0.0, 1.0, 1.0, 1.0, 0.0])
    assert list(v) == [0.0, 1.0, 0.0]


def test_the_gate_removes_small_interior_pairs_only():
    """The pair (0.1, 0.2) spans less than the gate and goes; the record's own
    endpoints stay, because dropping either would change the residue."""
    v, ix = turning_points([0.0, 1.0, 0.1, 0.2, 0.05, 3.0, 0.0], gate=0.5)
    assert list(v) == [0.0, 1.0, 0.05, 3.0, 0.0]
    assert list(ix) == [0, 1, 4, 5, 6]


def test_the_gate_clips_the_large_cycles_as_well_as_removing_small_ones():
    """The failure mode that makes a gate dangerous rather than merely lossy.

    A gate is usually justified as discarding cycles too small to matter. It
    also removes the *peak* of every large noisy cycle, because that peak is
    itself one end of a small pair -- so the large cycles get clipped, and they
    are the ones carrying the damage. Here a gate of 5 on a 200-wide signal
    throws away 95% of the cycles and 17% of the slope-3 damage, almost all of
    it from the clipping.

    Asserted as a property, not a target: the point is that the damage error
    is an order of magnitude larger than the discarded cycles could account
    for.
    """
    rng = np.random.default_rng(0)
    x = 100.0 * np.sin(np.linspace(0, 20 * np.pi, 2000)) + rng.normal(size=2000)

    def damage(c):
        return float((c.counts * c.ranges ** 3).sum())

    plain, gated = rainflow(x), rainflow(x, gate=5.0)

    assert len(gated) < 0.1 * len(plain)             # most cycles gone
    assert gated.ranges.max() < plain.ranges.max()   # and the peak is clipped

    removed = plain.ranges[plain.ranges < 5.0]
    small = float((removed ** 3).sum())
    lost = damage(plain) - damage(gated)
    assert lost > 100.0 * small


# ---------------------------------------------------------------------------
# CycleCounts
# ---------------------------------------------------------------------------

def test_amplitude_is_half_the_range_and_maxima_follow_the_mean():
    """The factor of two has exactly one home, and this is it."""
    c = rainflow(ASTM, residue="half")
    assert np.allclose(c.amplitudes, 0.5 * c.ranges)
    assert np.allclose(c.maxima, c.means + c.amplitudes)
    assert np.allclose(c.minima, c.means - c.amplitudes)


def test_indices_point_at_the_turning_points_that_formed_each_cycle():
    """Strain-life reads the mean stress at these positions, so they must index
    the original history and not the filtered turning-point list."""
    x = np.asarray(ASTM)
    c = rainflow(ASTM, residue="half")
    assert c.i_start.max() < x.size and c.i_end.max() < x.size
    assert np.allclose(np.abs(x[c.i_start] - x[c.i_end]), c.ranges)
    assert np.allclose(0.5 * (x[c.i_start] + x[c.i_end]), c.means)


def test_scaling_a_block_scales_the_counts_only():
    c = rainflow(ASTM, residue="half")
    s = c.scaled(3.0)
    assert s.total == pytest.approx(3.0 * c.total)
    assert np.allclose(s.ranges, c.ranges)


def test_rainflow_matrix_preserves_the_total_count():
    c = rainflow(np.random.default_rng(1).normal(size=500).cumsum())
    H, _, _ = rainflow_matrix(c, n_bins=16)
    assert H.sum() == pytest.approx(c.total)


def test_an_unknown_residue_mode_is_refused():
    with pytest.raises(ValueError, match="residue"):
        rainflow(ASTM, residue="whatever")
