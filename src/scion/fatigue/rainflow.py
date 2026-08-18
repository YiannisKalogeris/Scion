r"""
Rainflow cycle counting.

A variable-amplitude history is not a sequence of cycles. Rainflow counting is
what turns one into the other: it identifies the closed stress-strain hysteresis
loops the material actually traverses, which is the only decomposition for which
a constant-amplitude S-N or eps-N curve is the right thing to apply.

The algorithm here is the four-point method, which is equivalent to ASTM
E1049-85 rainflow and is verified against the worked example in that standard
(``tests/test_fatigue/test_rainflow.py``). Four consecutive turning points
``s1 s2 s3 s4`` close the inner pair when

    |s2 - s3| <= |s1 - s2|   and   |s2 - s3| <= |s3 - s4|,

at which point ``s2`` and ``s3`` are removed and the scan continues. What
survives to the end is the **residue**, an alternately divergent-convergent
sequence whose loops never closed inside the record.

The residue is where the damage is
----------------------------------
The residue holds the *largest* excursions in the history -- the ones that never
found a partner. How they are treated changes the answer by a factor of two on
exactly the cycles that dominate the damage sum, so ``residue`` is a required
decision, not a detail:

``"repeat"``   the history is one block of a repeating load. The residue is
               chained with a copy of itself and re-counted, which closes each
               residual loop exactly once per block. This is the right choice
               for a design load block and the default.
``"half"``     ASTM E1049. Each residual reversal is counted as half a cycle.
               Correct for a one-off transient (a single earthquake, a proof
               test), and it under-counts a repeating block, because the largest
               range in the record is then worth half of a cycle rather than one.
``"discard"``  residue thrown away. Fast, and unconservative in the worst
               possible place. Use only to isolate the residue's contribution.

Range, not amplitude
--------------------
:class:`CycleCounts` stores **ranges**, because that is what the counting
produces and what code-based S-N curves are written in. Every life curve in this
package takes an **amplitude**. The conversion is
:attr:`CycleCounts.amplitudes`, and it happens in exactly one place so that the
factor of two has one home. Half of all published fatigue disagreements are this
factor.

Indices survive the count
-------------------------
Each counted cycle records the position in the *original* history of the two
turning points that formed it. Strain-life needs this: the cycle is counted on
the strain channel, but Morrow and Smith-Watson-Topper need the mean and maximum
*stress* of that same loop, which is read from the stress history at those
indices. Counting the two channels separately would pair the wrong values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CycleCounts", "turning_points", "rainflow", "rainflow_matrix"]

_RESIDUE_MODES = ("repeat", "half", "discard")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CycleCounts:
    """Counted cycles of one history.

    All arrays are ``(m,)`` and aligned. ``counts`` is 1.0 for a closed cycle
    and 0.5 for a residual half cycle; it is a weight rather than an integer so
    that a block repeated ``k`` times is expressed by scaling it.
    """

    ranges: np.ndarray
    means: np.ndarray
    counts: np.ndarray
    i_start: np.ndarray          # index into the original history
    i_end: np.ndarray

    # ------------------------------------------------------------------
    @property
    def amplitudes(self) -> np.ndarray:
        """``range / 2``. What every life curve in this package expects."""
        return 0.5 * self.ranges

    @property
    def maxima(self) -> np.ndarray:
        """Peak value of each loop, ``mean + amplitude``. Needed by SWT."""
        return self.means + self.amplitudes

    @property
    def minima(self) -> np.ndarray:
        return self.means - self.amplitudes

    @property
    def total(self) -> float:
        """Sum of the counts: the number of whole cycles the block is worth."""
        return float(self.counts.sum())

    # ------------------------------------------------------------------
    def scaled(self, factor: float) -> "CycleCounts":
        """The same cycles with the counts multiplied -- a block repeated."""
        return CycleCounts(self.ranges, self.means, self.counts * float(factor),
                           self.i_start, self.i_end)

    def largest(self) -> int:
        """Index of the widest cycle. Usually the one that sets the life."""
        return int(np.argmax(self.ranges)) if len(self) else -1

    def sort_by_range(self) -> "CycleCounts":
        o = np.argsort(self.ranges)[::-1]
        return CycleCounts(self.ranges[o], self.means[o], self.counts[o],
                           self.i_start[o], self.i_end[o])

    def __len__(self) -> int:
        return int(self.ranges.size)

    def __repr__(self) -> str:
        if not len(self):
            return "CycleCounts(empty)"
        return (f"CycleCounts({len(self)} distinct, {self.total:g} cycles, "
                f"range {self.ranges.min():.4g}..{self.ranges.max():.4g}, "
                f"half={int((self.counts == 0.5).sum())})")


def _empty() -> CycleCounts:
    z = np.zeros(0)
    return CycleCounts(z, z, z, np.zeros(0, dtype=int), np.zeros(0, dtype=int))


# ---------------------------------------------------------------------------
# Turning points
# ---------------------------------------------------------------------------

def turning_points(x, gate: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Peaks and valleys of ``x``, with an optional small-range filter.

    Returns ``(values, indices)``, the indices being positions in ``x``. The
    first and last samples are always kept: they bound the record, and dropping
    either changes the residue.

    Rainflow counting depends only on the turning points, so everything between
    them is discarded first -- a 10^6-sample signal usually has a few thousand.

    The gate
    --------
    ``gate > 0`` removes any interior pair of turning points whose range is
    below it, merging the segments on either side. This is the standard
    racetrack/hysteresis filter, and it is not cosmetic: it is what stops a
    noisy signal from generating millions of microscopic cycles whose damage is
    numerically zero but whose *count* dominates the run time.

    It is not free, and not for the reason usually given. Discarding cycles
    below the endurance limit costs nothing under Miner, true. But the gate does
    not only discard small cycles: the peak of a *large* noisy cycle is itself
    usually one end of a small pair, so gating removes it and the cycle is
    counted to the next turning point in. The large cycles get clipped, and they
    are the ones carrying the damage. On a 200-wide signal with unit noise, a
    gate of 5 removes 95% of the counted cycles -- and 17% of the
    ``sum n r^3`` that a slope-3 curve would see, nearly all of it from the
    clipping rather than from the cycles it meant to remove.

    So: gate to make a long record tractable, and check what it cost by
    comparing the damage with and without, not the cycle count. Zero is the only
    value that changes nothing.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int)

    gate = float(gate)
    v: list[float] = []
    ix: list[int] = []

    for i in range(x.size):
        v.append(float(x[i]))
        ix.append(i)
        # Collapse until the tail is a strictly alternating sequence with no
        # gated pair. Both rules can expose new violations further back, hence
        # the loop rather than two passes.
        while True:
            if len(v) >= 3 and (v[-2] - v[-3]) * (v[-1] - v[-2]) >= 0.0:
                del v[-2], ix[-2]          # middle point is not a turning point
                continue
            # len >= 4 keeps the first sample out of the pair: the start of the
            # record is not a turning point that may be filtered away.
            if gate > 0.0 and len(v) >= 4 and abs(v[-2] - v[-3]) < gate:
                del v[-3:-1], ix[-3:-1]
                continue
            break

    return np.asarray(v), np.asarray(ix, dtype=int)


# ---------------------------------------------------------------------------
# The four-point extraction
# ---------------------------------------------------------------------------

def _extract(v: np.ndarray, ix: np.ndarray):
    """Closed cycles and the residue, by the four-point rule.

    ``v`` must already be turning points. Returns
    ``(ranges, means, i_start, i_end, residue_values, residue_indices)``.
    """
    sv: list[float] = []
    si: list[int] = []
    rng: list[float] = []
    mean: list[float] = []
    a: list[int] = []
    b: list[int] = []

    for val, i in zip(v, ix):
        sv.append(float(val))
        si.append(int(i))
        while len(sv) >= 4:
            s1, s2, s3, s4 = sv[-4:]
            r2 = abs(s2 - s3)
            if r2 <= abs(s1 - s2) and r2 <= abs(s3 - s4):
                rng.append(r2)
                mean.append(0.5 * (s2 + s3))
                a.append(si[-3])
                b.append(si[-2])
                del sv[-3:-1], si[-3:-1]
            else:
                break

    return (np.asarray(rng), np.asarray(mean),
            np.asarray(a, dtype=int), np.asarray(b, dtype=int),
            np.asarray(sv), np.asarray(si, dtype=int))


def _halves(rv: np.ndarray, ri: np.ndarray):
    """Each consecutive pair of residual turning points as a half cycle."""
    if rv.size < 2:
        return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=int),
                np.zeros(0, dtype=int))
    return (np.abs(np.diff(rv)),
            0.5 * (rv[:-1] + rv[1:]),
            ri[:-1], ri[1:])


def rainflow(x, gate: float = 0.0, residue: str = "repeat") -> CycleCounts:
    """Rainflow-count a scalar history.

    Parameters
    ----------
    x : ``(n,)`` the history. Stress, strain, force, temperature -- the counting
        knows nothing about units.
    gate : small-range filter applied to the turning points first; see
        :func:`turning_points`.
    residue : ``"repeat"``, ``"half"`` or ``"discard"``. See the module
        docstring; this choice is worth a factor of two on the largest cycles.

    Returns
    -------
    :class:`CycleCounts` with ranges, means, weights and the indices in ``x`` of
    the turning points that formed each cycle.
    """
    if residue not in _RESIDUE_MODES:
        raise ValueError(f"residue must be one of {_RESIDUE_MODES}, "
                         f"got '{residue}'")

    v, ix = turning_points(x, gate=gate)
    if v.size < 3:
        return _empty()

    rng, mean, a, b, rv, ri = _extract(v, ix)
    counts = np.ones(rng.size)

    if residue == "half":
        hr, hm, ha, hb = _halves(rv, ri)
        rng = np.concatenate([rng, hr])
        mean = np.concatenate([mean, hm])
        a = np.concatenate([a, ha])
        b = np.concatenate([b, hb])
        counts = np.concatenate([counts, np.full(hr.size, 0.5)])

    elif residue == "repeat":
        # Chaining the residue with a copy of itself closes every residual loop
        # exactly once, which is the count for one repetition of the block. The
        # leftover of that second pass is the residue again -- it is a fixed
        # point of the extraction, which is the property the test asserts.
        if rv.size >= 2:
            v2 = np.concatenate([rv, rv])
            i2 = np.concatenate([ri, ri])
            v2, keep = _collapse(v2)
            i2 = i2[keep]
            r2, m2, a2, b2, _, _ = _extract(v2, i2)
            rng = np.concatenate([rng, r2])
            mean = np.concatenate([mean, m2])
            a = np.concatenate([a, a2])
            b = np.concatenate([b, b2])
            counts = np.concatenate([counts, np.ones(r2.size)])

    return CycleCounts(rng, mean, counts, a, b)


def _collapse(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-turning points from ``v``; returns the values and a keep mask.

    Used on the chained residue, where the junction repeats a value and can
    leave three collinear points that the four-point rule would mis-handle.
    """
    out_v: list[float] = []
    out_k: list[int] = []
    for k, val in enumerate(v):
        out_v.append(float(val))
        out_k.append(k)
        while len(out_v) >= 3 and \
                (out_v[-2] - out_v[-3]) * (out_v[-1] - out_v[-2]) >= 0.0:
            del out_v[-2], out_k[-2]
    return np.asarray(out_v), np.asarray(out_k, dtype=int)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def rainflow_matrix(counts: CycleCounts, n_bins: int = 32,
                    range_limits: tuple[float, float] | None = None,
                    mean_limits: tuple[float, float] | None = None):
    """Range-mean histogram of a count, weighted by the cycle counts.

    Returns ``(matrix, range_edges, mean_edges)`` with ``matrix[i, j]`` the
    number of cycles in range-bin ``i`` and mean-bin ``j``.

    Binning is lossy in the direction that matters: an S-N curve is a power law,
    so the damage of a bin is not the damage of its centre times the count. Bin
    for plotting; sum the damage on the unbinned cycles.
    """
    if not len(counts):
        e = np.linspace(0.0, 1.0, n_bins + 1)
        return np.zeros((n_bins, n_bins)), e, e

    rl = range_limits or (0.0, float(counts.ranges.max()) * (1.0 + 1e-12))
    ml = mean_limits or (float(counts.means.min()), float(counts.means.max()))
    if ml[0] == ml[1]:
        ml = (ml[0] - 0.5, ml[1] + 0.5)

    H, re, me = np.histogram2d(counts.ranges, counts.means,
                               bins=[n_bins, n_bins], range=[rl, ml],
                               weights=counts.counts)
    return H, re, me
