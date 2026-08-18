r"""
Fatigue assessment of a field of points.

The four pieces -- reduce, count, evaluate the curve, accumulate -- each work on
their own and are each testable on their own. This is the object that runs them
in order over a stress or strain history at every quadrature point, and its only
real content is the wiring: which channel gets counted, and where the mean
stress of each counted cycle comes from.

    from simulacrum.fatigue import Basquin, FatigueAssessment

    curve = Basquin(sigma_f=1200.0, b=-0.095, mean_stress="goodman",
                    mean_stress_kwargs={"sigma_u": 600.0})
    job = FatigueAssessment(curve)
    result = job.evaluate(stress=sigma_history, dim=2, nu=0.3)
    result.damage            # per point, per block of the given history
    result.blocks_to_failure

``sigma_history`` is ``(n_points, n_steps, ...)``: a scalar per step, a Voigt
vector, or a full ``(3, 3)`` tensor. The middle axis is the load history, and it
is the only axis whose order matters.

Which channel gets counted
--------------------------
The curve decides, through
:attr:`~simulacrum.fatigue.curves.LifeCurve.variable`. Stress-life counts the
stress history; strain-life counts the strain history. This is not
interchangeable and the failure is silent: once a point is yielding, its stress
history flattens onto the yield surface while its strain history keeps growing,
so counting stress in the low-cycle regime reports a handful of small cycles
where the material saw large ones.

Where the mean stress comes from
--------------------------------
For stress-life, from the counted cycle itself.

For strain-life, from the **stress** history at the two turning points that
formed the counted **strain** cycle -- which is why
:class:`~simulacrum.fatigue.rainflow.CycleCounts` carries indices. Counting the
two channels separately and pairing the results by position would pair the wrong
loops: the two counts need not even have the same length.

This assumes the stress and strain histories turn at the same steps. Under
monotone loading branches they do; the assumption breaks only where the material
unloads elastically while the strain still rises, which small-strain plasticity
does not do.

Cost
----
Rainflow counting is a stack walk over one point's history and cannot be
vectorised across points, so this loops in Python: roughly a millisecond per
thousand steps per point. For a full mesh that is minutes, and for a mesh where
only the hot spots matter it is wasted. Pass ``points`` to restrict the
assessment to an index set; :meth:`hotspots` picks a reasonable one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .accumulation import (
    AccumulationRule,
    LinearAccumulation,
    cycle_damage,
    get_accumulation_rule,
)
from .curves import LifeCurve
from .multiaxial import (
    get_equivalent,
    voigt_strain_to_tensor,
    voigt_to_tensor,
)
from .rainflow import CycleCounts, rainflow

__all__ = ["FatigueResult", "FatigueAssessment"]


# ---------------------------------------------------------------------------

@dataclass
class FatigueResult:
    """Damage and life at every assessed point.

    ``damage`` is per repetition of the history that was supplied. If that
    history is one block of a spectrum that repeats, ``blocks_to_failure`` is
    the number of repetitions; if it is the whole design life, a value below one
    is the failure.
    """

    damage: np.ndarray                 # (n_assessed,)
    cycles_per_block: np.ndarray       # (n_assessed,)
    blocks_to_failure: np.ndarray      # (n_assessed,)
    points: np.ndarray                 # (n_assessed,) index into the mesh
    n_points: int                      # total in the input, assessed or not
    counts: list[CycleCounts] | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    @property
    def life_cycles(self) -> np.ndarray:
        """Counted cycles survived, ``blocks_to_failure * cycles_per_block``.

        Comparable with a constant-amplitude life, and only that: the cycles are
        of mixed amplitude, so this is a bookkeeping total rather than a
        physical count of anything.
        """
        return self.blocks_to_failure * self.cycles_per_block

    @property
    def critical(self) -> int:
        """Mesh index of the most damaged point."""
        return int(self.points[np.argmax(self.damage)])

    @property
    def max_damage(self) -> float:
        return float(np.max(self.damage)) if self.damage.size else 0.0

    def scatter(self, values: np.ndarray | None = None,
                fill: float = 0.0) -> np.ndarray:
        """Spread an assessed-point array back over the full mesh, ``(n_points,)``.

        Points that were not assessed get ``fill``. Needed whenever ``points``
        was used, because the result arrays are then indexed by assessment order
        rather than by mesh point, and writing them straight to a VTK field
        would silently mis-place every value.
        """
        v = self.damage if values is None else np.asarray(values)
        out = np.full(self.n_points, fill, dtype=float)
        out[self.points] = v
        return out

    def __repr__(self) -> str:
        if not self.damage.size:
            return "FatigueResult(no points assessed)"
        d = self.max_damage
        return (f"FatigueResult({len(self.damage)}/{self.n_points} points, "
                f"max D={d:.4g} at point {self.critical}, "
                f"blocks to failure={np.min(self.blocks_to_failure):.4g}, "
                f"cycles/block={np.median(self.cycles_per_block):.4g})")


# ---------------------------------------------------------------------------

class FatigueAssessment:
    """Runs a life curve over a field of load histories.

    Parameters
    ----------
    curve : any :class:`~simulacrum.fatigue.curves.LifeCurve`. Its
        ``variable`` decides which history is counted and its ``needs_stress``
        decides whether the stress history is also required.
    accumulation : rule for combining the per-cycle damage; Miner by default.
    equivalent : name of the stress reduction, or a callable
        ``(..., 3, 3) -> (...)``. Ignored for histories that are already scalar.
    strain_equivalent : the same for strain histories, and **not** the same
        function. The von Mises invariant of a strain tensor carries a
        ``1/(1 + nu)`` normalisation that the stress one does not; reducing a
        strain history with the stress formula overstates the equivalent strain
        by ``1 + nu``, which is 50% at ``nu = 1/2`` and a factor of thousands in
        life once a strain-life curve has finished with it. Two parameters
        rather than one, because a single default cannot be right for both.
    gate, residue : passed to :func:`~simulacrum.fatigue.rainflow.rainflow`.
        ``residue`` defaults to ``"repeat"``, which treats the history as one
        block of a repeating load; use ``"half"`` for a one-off transient.
    """

    def __init__(self, curve: LifeCurve,
                 accumulation: str | AccumulationRule | None = None,
                 equivalent: str | callable = "signed-von-mises",
                 equivalent_kwargs: dict | None = None,
                 strain_equivalent: str | callable = "equivalent-strain",
                 strain_equivalent_kwargs: dict | None = None,
                 gate: float = 0.0, residue: str = "repeat"):
        self.curve = curve
        self.accumulation = (
            get_accumulation_rule(accumulation) if isinstance(accumulation, str)
            else (accumulation or LinearAccumulation()))
        self.equivalent = (get_equivalent(equivalent, **(equivalent_kwargs or {}))
                           if isinstance(equivalent, str) else equivalent)
        self.strain_equivalent = (
            get_equivalent(strain_equivalent, **(strain_equivalent_kwargs or {}))
            if isinstance(strain_equivalent, str) else strain_equivalent)
        self.gate = float(gate)
        self.residue = residue

    # ------------------------------------------------------------------
    def _reduce(self, history, kind: str, dim: int, sigma_33, nu
                ) -> np.ndarray:
        """Any accepted history shape to ``(n_points, n_steps)``."""
        h = np.asarray(history, dtype=float)
        reduce = (self.equivalent if kind == "stress"
                  else self.strain_equivalent)
        if h.ndim == 2:
            return h                                    # already scalar
        if h.ndim == 4 and h.shape[-2:] == (3, 3):
            return reduce(h)
        if h.ndim == 3:
            t = (voigt_to_tensor(h, dim=dim, sigma_33=sigma_33, nu=nu)
                 if kind == "stress" else voigt_strain_to_tensor(h, dim=dim))
            return reduce(t)
        raise ValueError(
            f"a {kind} history must be (n_points, n_steps), "
            f"(n_points, n_steps, n_voigt) or (n_points, n_steps, 3, 3); "
            f"got shape {h.shape}")

    # ------------------------------------------------------------------
    def evaluate(self, stress=None, strain=None, plastic_strain=None, *,
                 dim: int = 3, sigma_33=None, nu: float | None = None,
                 repeats: float = 1.0, points=None,
                 keep_counts: bool = False) -> FatigueResult:
        """Assess every point (or the subset in ``points``).

        Parameters
        ----------
        stress, strain, plastic_strain : histories, ``(n_points, n_steps, ...)``.
            Supply the channel the curve counts on; supply ``stress`` as well if
            the curve's mean-stress treatment needs it.
        dim, sigma_33, nu : only for Voigt input. In 2-D the stress conversion
            needs ``sigma_33`` or ``nu``; see
            :func:`~simulacrum.fatigue.multiaxial.voigt_to_tensor`.
        repeats : how many times the supplied history occurs per block. Scales
            the counts, not the history -- chaining the history with itself and
            counting that is a different and more accurate calculation, because
            it closes the residual cycles at the joins.
        points : indices to assess. ``None`` assesses all of them.
        keep_counts : retain the per-point :class:`CycleCounts`. Off by default;
            it is one object per point and the mesh is usually large.
        """
        channel = {"stress": stress, "strain": strain,
                   "plastic-strain": plastic_strain}[self.curve.variable]
        if channel is None:
            raise ValueError(
                f"{type(self.curve).__name__} counts cycles on the "
                f"'{self.curve.variable}' history; pass "
                f"{self.curve.variable.replace('-', '_')}=... . Substituting "
                f"another channel changes the answer without any symptom.")
        if self.curve.needs_stress and stress is None:
            raise ValueError(
                f"the mean-stress treatment of this curve reads the stress "
                f"history as well as the counted "
                f"'{self.curve.variable}' one; pass stress=... too.")

        counted = self._reduce(channel, self.curve.variable, dim, sigma_33, nu)
        n_points, _ = counted.shape

        stress_scalar = None
        if self.curve.needs_stress:
            stress_scalar = self._reduce(stress, "stress", dim, sigma_33, nu)
            if stress_scalar.shape != counted.shape:
                raise ValueError(
                    f"the stress history {stress_scalar.shape} and the counted "
                    f"history {counted.shape} must have the same points and "
                    f"steps; the mean stress of a loop is read from the stress "
                    f"channel at that loop's own turning points.")

        idx = (np.arange(n_points) if points is None
               else np.atleast_1d(np.asarray(points, dtype=int)))

        damage = np.zeros(idx.size)
        per_block = np.zeros(idx.size)
        kept: list[CycleCounts] | None = [] if keep_counts else None

        for k, p in enumerate(idx):
            c = rainflow(counted[p], gate=self.gate, residue=self.residue)
            if repeats != 1.0:
                c = c.scaled(repeats)
            if not len(c):
                if kept is not None:
                    kept.append(c)
                continue

            if stress_scalar is not None:
                s = stress_scalar[p]
                s0, s1 = s[c.i_start], s[c.i_end]
                mean = 0.5 * (s0 + s1)
                s_max = np.maximum(s0, s1)
            else:
                mean, s_max = c.means, c.maxima

            N = self.curve.cycles(c.amplitudes, mean_stress=mean,
                                  max_stress=s_max)
            damage[k] = self.accumulation.accumulate(
                cycle_damage(c.counts, N))
            per_block[k] = c.total
            if kept is not None:
                kept.append(c)

        return FatigueResult(
            damage=damage, cycles_per_block=per_block,
            blocks_to_failure=self.accumulation.blocks_to_failure(damage),
            points=idx, n_points=n_points, counts=kept)

    # ------------------------------------------------------------------
    def hotspots(self, history, n: int = 32, *, dim: int = 3,
                 sigma_33=None, nu: float | None = None,
                 kind: str = "stress") -> np.ndarray:
        """Indices of the ``n`` points with the widest reduced history.

        Ranked by peak-to-trough span of the equivalent scalar, which is the
        cheapest quantity that correlates with damage: on a power-law curve the
        largest cycle usually dominates the sum, and it cannot be larger than
        the span.

        A screening step, not a substitute for assessing everything. It ranks by
        the largest cycle and can therefore miss a point whose life is set by
        many medium cycles, and it knows nothing about mean stress. Widen ``n``
        rather than trusting the top few.
        """
        r = self._reduce(history, kind, dim, sigma_33, nu)
        span = r.max(axis=1) - r.min(axis=1)
        n = min(int(n), span.size)
        return np.argsort(span)[::-1][:n]

    def __repr__(self) -> str:
        channel = self.curve.variable
        reducer = (self.equivalent if channel == "stress"
                   else self.strain_equivalent)
        return (f"FatigueAssessment(curve={self.curve!r}, "
                f"counting {channel} through "
                f"{getattr(reducer, 'func', reducer).__name__}, "
                f"accumulation={self.accumulation.name}, "
                f"residue='{self.residue}', gate={self.gate:g})")
