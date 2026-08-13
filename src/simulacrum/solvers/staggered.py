r"""
Staggered solution of coupled two-field problems.

Written against callbacks rather than a specific physics, so the same driver
serves phase-field fracture, thermo-mechanics and poromechanics: it alternates
"solve field A at fixed B" and "solve field B at fixed A" until an energy
functional stops decreasing.

Monolithic schemes need fewer outer iterations but are markedly less robust for
fracture -- the coupled tangent loses positive definiteness as soon as damage
localises. Staggered is the standard choice for that reason.

The stopping criterion
----------------------
This is the part worth getting right, and the reason it is a named function
rather than an inline tolerance. Ambati, Gerasimov & De Lorenzis (2015) show
that the two obvious criteria both fail on the same problem:

* relative energy change, ``(E_k - E_k+1) / E_k+1 < 1%``
* raw slope, ``arctan(E_k - E_k+1) < 1 degree``

Both stop after **two** iterations on a case that needs about sixty, because
the scales of ``E`` and ``k`` are wildly different -- ``E`` spans 3.14 to 3.25
while ``k`` runs 1 to 61 -- so a "small" slope means nothing. The crack field at
iteration 10 and at iteration 60 are visibly different, and the early one is
simply wrong.

Their fix is to normalise both axes over the sequence so far before measuring
the slope:

    E_norm_k = (E_k - E_N) / (E_1 - E_N),   s_k = k / N,
    gamma_k  = arctan( (E_norm_k - E_norm_k+1) / (s_k+1 - s_k) )

``gamma`` is then a dimensionless angle in degrees for which a tolerance of
about 10 degrees is meaningful across problems. :func:`normalized_slope`
implements it.

Two failure modes the paper does not discuss
--------------------------------------------
The normalisation buys scale-independence, and pays for it in two ways. Both
were found by running the single-edge notched tension benchmark, and both look
identical from outside -- the cycle simply runs to its cap.

1. **Converged state.** In an elastic step the energy stops changing entirely.
   The span ``E_1 - E_N`` is then round-off (1e-13 on an energy of 1e-3, sign
   flipping between iterations), ``gamma`` is computed from noise and sits at
   80-88 degrees forever. Guarded by ``atol_rel`` in :func:`normalized_slope`.

2. **Linear drift.** If the energy creeps down at a steady rate -- which it
   does near a crack tip, where raising ``u`` raises ``H`` raises ``d`` softens
   the material raises ``u`` -- then ``gamma`` pins near 45 degrees *regardless
   of the size of the creep*::

       slope 1e-03, N=50 -> gamma 45.58
       slope 1e-08, N=50 -> gamma 45.58
       slope 1e-14, N=50 -> gamma 45.87

   No number of iterations helps, because the shape never changes. This one
   cannot be fixed inside the energy criterion: a measure of shape cannot see
   magnitude. The answer is a second, independent test on the *field increment*
   -- see ``change`` / ``change_tol`` on :class:`StaggeredSolver`.

The two tests are complementary and cheap, so the solver applies both and
stops on either.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "normalized_slope",
    "relative_energy_change",
    "StaggeredResult",
    "StaggeredSolver",
]


# ---------------------------------------------------------------------------
# Convergence measures
# ---------------------------------------------------------------------------

def normalized_slope(energies, atol_rel: float = 1.0e-10) -> float:
    """Normalised slope of the energy history, in degrees.

    Both axes are rescaled over the sequence seen so far, so the value depends
    on the *shape* of the decay and not on the magnitude of the energy or the
    number of iterations taken. Returns ``inf`` for sequences shorter than
    three entries, so a cycle always takes at least two iterations.

    ``atol_rel`` guards failure mode 1 above: when the total change over the
    cycle is at round-off level the state has stopped moving, and the
    normalisation would divide noise by noise.

    It cannot guard failure mode 2. Pair it with a field-increment test.
    """
    E = np.asarray(energies, dtype=float)
    N = E.size
    if N < 3:
        return float("inf")
    span = E[0] - E[-1]
    # A span at round-off level means the state has stopped changing, not that
    # it is converging slowly -- normalising would divide noise by noise.
    if abs(span) <= atol_rel * max(abs(E[-1]), 1.0):
        return 0.0
    dE_norm = (E[-2] - E[-1]) / span
    return float(np.degrees(np.arctan2(dE_norm, 1.0 / N)))


def relative_energy_change(energies) -> float:
    """``(E_k - E_k+1) / E_k+1`` in percent.

    Kept for comparison only: the paper shows this criterion stops far too
    early. Do not use it as the primary test.
    """
    E = np.asarray(energies, dtype=float)
    if E.size < 2:
        return float("inf")
    denom = E[-1] if abs(E[-1]) > 1.0e-300 else 1.0
    return float(100.0 * (E[-2] - E[-1]) / denom)


# ---------------------------------------------------------------------------

@dataclass
class StaggeredResult:
    """Outcome of one staggered cycle (one load step)."""

    converged: bool
    iterations: int
    energies: list = field(default_factory=list)
    criterion: float = float("inf")
    inner_iterations: list = field(default_factory=list)
    field_change: float = float("inf")
    stopped_on: str = "cap"        # "energy", "field" or "cap"

    def __repr__(self) -> str:
        return (f"StaggeredResult(converged={self.converged}, "
                f"iterations={self.iterations}, on={self.stopped_on}, "
                f"gamma={self.criterion:.3g}, "
                f"dfield={self.field_change:.3g}, "
                f"E_final={self.energies[-1]:.6g})"
                if self.energies else
                f"StaggeredResult(converged={self.converged}, iterations=0)")


class StaggeredSolver:
    """Alternating solver for a two-field problem.

    Parameters
    ----------
    solve_first, solve_second
        Callables taking no arguments and updating their field in place. For
        phase-field fracture these are the displacement and phase-field solves.
        Each may return an integer iteration count, which is recorded.
    energy
        Callable returning the total energy of the current state. The sequence
        it produces is what the stopping criterion measures, so it must include
        *both* the elastic and the fracture contributions -- monitoring only one
        of them can decrease while the total does not.
    tol
        Tolerance on ``gamma`` in degrees. About 10 is reasonable; smaller is
        safer and costs iterations.
    max_iterations
        Cap per load step. Hitting it is a signal the load increment is too
        large, not that the tolerance is too tight.
    criterion
        Override the energy measure; defaults to :func:`normalized_slope`.
    change
        Optional callable returning the largest change in the second field
        over the last iteration, e.g. ``max|d_k - d_k-1|``. This is the second,
        independent test, and it is the one that catches the linear-drift mode
        the energy criterion is blind to. Dimensionless and bounded for a
        damage field, so ``change_tol`` transfers between problems in a way
        that an energy tolerance does not.
    change_tol
        Tolerance on ``change``. 1e-4 is a reasonable default for a phase
        field; the cycle stops when *either* test passes.
    """

    def __init__(self, solve_first, solve_second, energy,
                 tol: float = 10.0, max_iterations: int = 100,
                 min_iterations: int = 2, criterion=None,
                 change=None, change_tol: float = 1.0e-4,
                 verbose: bool = False):
        self.solve_first = solve_first
        self.solve_second = solve_second
        self.energy = energy
        self.tol = float(tol)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.criterion = criterion or normalized_slope
        self.change = change
        self.change_tol = float(change_tol)
        self.verbose = verbose

    # ------------------------------------------------------------------
    def run(self) -> StaggeredResult:
        energies: list[float] = []
        inner: list[tuple] = []

        for k in range(1, self.max_iterations + 1):
            n1 = self.solve_first()
            n2 = self.solve_second()
            inner.append((n1, n2))
            energies.append(float(self.energy()))

            value = self.criterion(energies)
            delta = (float(self.change()) if self.change is not None
                     else float("inf"))

            if self.verbose:
                print(f"    staggered {k:3d}: E = {energies[-1]:.8g}   "
                      f"gamma = {value:8.3f} deg   dfield = {delta:.3e}")

            if k >= self.min_iterations:
                # either test is sufficient: they fail in different regimes
                if delta <= self.change_tol:
                    return StaggeredResult(True, k, energies, value, inner,
                                           delta, "field")
                if value <= self.tol:
                    return StaggeredResult(True, k, energies, value, inner,
                                           delta, "energy")

        delta = (float(self.change()) if self.change is not None
                 else float("inf"))
        return StaggeredResult(False, self.max_iterations, energies,
                               self.criterion(energies), inner, delta, "cap")
