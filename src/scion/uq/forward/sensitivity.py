r"""
Variance-based sensitivity: which inputs the output actually depends on.

Sobol' indices decompose the output variance into contributions from each input
and each combination of inputs. For independent inputs the decomposition is
exact and unique, and two summaries come out of it:

    S_i   = Var(E[Y | X_i]) / Var(Y)          first order
    S_Ti  = E[Var(Y | X_~i)] / Var(Y)         total

``S_i`` is the variance removed by learning ``X_i`` alone -- what that input
contributes *by itself*. ``S_Ti`` is the variance that would remain if
everything *except* ``X_i`` were fixed -- its own effect plus every interaction
it takes part in. The gap ``S_Ti - S_i`` is exactly that interaction share.

Reading them
------------
* ``sum S_i = 1`` **only if the model is additive.** A shortfall is the
  interaction content, and it is a measurement, not an error.
* ``sum S_Ti >= 1`` always, because interactions are counted once per
  participant.
* ``S_Ti = 0`` is the one that licenses a decision: an input with zero total
  index does not affect the output at all and can be fixed at any value. A
  small ``S_i`` licenses nothing, because the input may still matter through an
  interaction -- the Ishigami function's third variable has ``S_3 = 0`` exactly
  and ``S_T3 = 0.244``, which is the standard demonstration of the difference.

These are *variance-based*, so they inherit the assumption that variance is the
right summary of the output. For a strongly skewed or multimodal response they
answer a question that may not be the one being asked.

The estimator
-------------
Saltelli's design: two independent sample matrices ``A`` and ``B``, plus ``d``
matrices ``AB_i`` where column ``i`` of ``A`` is replaced by column ``i`` of
``B``. Cost is ``N(d + 2)`` model evaluations.

The pairing is the mechanism and cannot be shortcut. ``AB_i`` differs from ``A``
in *only* that one column, so the correlation between ``f(A)`` and ``f(AB_i)``
isolates the effect of ``X_i``. Building ``AB_i`` from fresh random points
instead destroys the pairing and returns noise.

Two estimator forms are provided, and the default is not the original:

``jansen``    the default. ``S_Ti`` from ``mean((f(A) - f(AB_i))^2) / 2V`` and
              ``S_i`` from ``mean((f(B) - f(AB_i))^2) / 2V``. Difference-based,
              so it is well behaved when an index is near zero.
``saltelli``  the classical products. Simple and unbiased, and noisy for small
              indices -- it routinely returns small negative values for an index
              that is genuinely zero, because it is a difference of two large
              nearly equal averages.

Neither is guaranteed non-negative in finite samples. Small negatives are
reported as computed rather than clipped, since clipping hides the one signal
that says the sample was too small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scion.uq.forward.distributions import JointIndependent
from scion.uq.forward.estimators import unit_design

__all__ = ["SobolResult", "sobol_indices"]


@dataclass
class SobolResult:
    """First-order and total indices, with bootstrap intervals."""

    first_order: np.ndarray
    total: np.ndarray
    variance: float
    mean: float
    n_evaluations: int
    estimator: str = "jansen"
    names: list = field(default_factory=list)
    first_order_ci: np.ndarray | None = None
    total_ci: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.first_order.size)

    @property
    def interaction(self) -> np.ndarray:
        """``S_Ti - S_i`` per input: the share carried by interactions."""
        return self.total - self.first_order

    @property
    def additive_fraction(self) -> float:
        """``sum S_i``. One for a purely additive model, less otherwise."""
        return float(np.sum(self.first_order))

    def summary(self) -> str:
        names = self.names or [f"x{i + 1}" for i in range(self.dim)]
        lines = [f"Sobol indices ({self.estimator}), "
                 f"{self.n_evaluations} model evaluations",
                 f"  output mean {self.mean:.6g}, variance {self.variance:.6g}",
                 f"  {'input':<10}{'S_i':>10}{'S_Ti':>10}"
                 f"{'interaction':>14}"]
        for i, name in enumerate(names):
            lines.append(f"  {name:<10}{self.first_order[i]:10.4f}"
                         f"{self.total[i]:10.4f}{self.interaction[i]:14.4f}")
        lines.append(f"  {'sum':<10}{self.additive_fraction:10.4f}"
                     f"{np.sum(self.total):10.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"SobolResult(S={np.round(self.first_order, 4)}, "
                f"ST={np.round(self.total, 4)})")


def sobol_indices(model, distribution: JointIndependent, n_base: int = 1024,
                  estimator: str = "jansen", seed: int | None = None,
                  vectorised: bool = False, design: str = "sobol",
                  n_bootstrap: int = 0) -> SobolResult:
    """Sobol' first-order and total indices by the Saltelli design.

    Parameters
    ----------
    model : ``x -> scalar``, or ``(n, dim) -> (n,)`` if ``vectorised``.
    n_base : rows in each of ``A`` and ``B``. Total cost is
        ``n_base * (dim + 2)``.
    estimator : ``"jansen"`` (default) or ``"saltelli"``.
    design : the unit-hypercube design for ``A`` and ``B``. Quasi-random is the
        usual choice and is why ``n_base`` should be a power of two.
    n_bootstrap : resamples for confidence intervals. Worth having: without
        them there is nothing to say whether a small index is small or is noise.

    Notes
    -----
    ``A`` and ``B`` come from one design of **2*dim columns**, split down the
    middle: ``A`` is columns ``0:dim`` and ``B`` is columns ``dim:2*dim``. This
    is not interchangeable with generating ``2 * n_base`` rows and splitting
    them in half, which is the natural-looking thing to do and is wrong for a
    quasi-random sequence. Rows ``j`` and ``j + n_base`` of a Sobol sequence are
    related by its digital construction, so the two halves are not independent,
    the estimator's pairing argument collapses, and the indices come out
    meaningless -- measured on Ishigami at 0.588/0.628/0.958 against a true
    0.314/0.442/0. The column split is the standard construction for exactly
    this reason.
    """
    if estimator not in ("jansen", "saltelli"):
        raise NotImplementedError(
            f"Unknown estimator '{estimator}'; use 'jansen' or 'saltelli'")
    if n_base < 8:
        raise ValueError("n_base is far too small for a variance decomposition")

    rng = np.random.default_rng(seed)
    dim = distribution.dim

    both = unit_design(design, n_base, 2 * dim, rng)
    A = distribution.ppf(both[:, :dim])
    B = distribution.ppf(both[:, dim:])

    def run(points):
        if vectorised:
            return np.asarray(model(points), dtype=float).ravel()
        return np.array([float(model(row)) for row in points])

    fA, fB = run(A), run(B)
    fAB = np.empty((dim, n_base))
    for i in range(dim):
        AB = A.copy()
        AB[:, i] = B[:, i]          # one column swapped, everything else held
        fAB[i] = run(AB)

    variance = float(np.var(np.concatenate([fA, fB]), ddof=1))
    if variance <= 0.0:
        raise ValueError("the model output is constant, so its variance cannot "
                         "be decomposed")

    def indices(a, b, ab):
        v = float(np.var(np.concatenate([a, b]), ddof=1))
        if estimator == "jansen":
            # both difference-based: V_i = V - E[(f(B) - f(AB_i))^2]/2 and
            # E[Var(Y|X_~i)] = E[(f(A) - f(AB_i))^2]/2
            first = np.array([1.0 - np.mean((b - ab[i]) ** 2) / (2.0 * v)
                              for i in range(dim)])
            total = np.array([np.mean((a - ab[i]) ** 2) / (2.0 * v)
                              for i in range(dim)])
        else:
            first = np.array([(np.mean(b * (ab[i] - a))) / v
                              for i in range(dim)])
            total = np.array([1.0 - np.mean(a * (ab[i] - b)) / v
                              for i in range(dim)])
        return first, total

    first_order, total = indices(fA, fB, fAB)

    first_ci = total_ci = None
    if n_bootstrap > 0:
        boot_first = np.empty((n_bootstrap, dim))
        boot_total = np.empty((n_bootstrap, dim))
        for k in range(n_bootstrap):
            pick = rng.integers(0, n_base, n_base)
            bf, bt = indices(fA[pick], fB[pick], fAB[:, pick])
            boot_first[k], boot_total[k] = bf, bt
        first_ci = np.percentile(boot_first, [2.5, 97.5], axis=0)
        total_ci = np.percentile(boot_total, [2.5, 97.5], axis=0)

    return SobolResult(
        first_order=first_order, total=total, variance=variance,
        mean=float(np.mean(np.concatenate([fA, fB]))),
        n_evaluations=n_base * (dim + 2), estimator=estimator,
        names=list(distribution.names),
        first_order_ci=first_ci, total_ci=total_ci)
