r"""
Propagating a distribution through a model, and the designs that do it.

The whole of crude Monte Carlo is: draw inputs from their distribution,
evaluate the model, average. What varies between the methods here is only
*where the points go*, and the differences matter more than the simplicity
suggests.

The rate, and why it cannot be improved
---------------------------------------
The Monte Carlo error falls as ``sigma / sqrt(N)``. Two consequences that decide
how these methods get used:

* **It does not depend on the dimension.** That is the property no quadrature
  rule has, and the entire reason Monte Carlo exists. A tensor-product Gauss
  rule needs ``m^d`` points; at ``d = 20`` that is over.
* **It cannot be beaten by a cleverer average.** Halving the error costs four
  times the samples, always. The only lever is reducing ``sigma`` itself, which
  is what :mod:`~.variance_reduction` is for, or abandoning the iid assumption,
  which is what quasi-Monte Carlo does.

The designs
-----------
``crude``       independent draws. The baseline, and the only one whose error
                estimate needs no qualification.
``lhs``         Latin hypercube: each 1-D margin is stratified into ``N`` equal
                bins with one point in each. Exact stratification per
                *dimension*, and nothing at all about interactions -- so it
                helps in proportion to how additive the model is, and does
                nothing for a pure interaction term.
``sobol``       a low-discrepancy sequence. Better than ``N^-1/2`` for a smooth,
                low effective dimension integrand, and subject to the two traps
                below.
``halton``      the same idea, older, and weaker in high dimensions unless the
                sequence is scrambled.
``antithetic``  pairs ``u`` with ``1-u``. Free, exact for a linear model, and
                capable of *increasing* the variance -- see below.

Two quasi-Monte Carlo traps
---------------------------
**Sobol wants a power of two.** Its equidistribution guarantees hold for
``N = 2^m``; at other counts the balance property is lost and the sequence can
be worse than random. :func:`propagate` warns rather than silently rounding,
because rounding would change the number of model evaluations behind your back.

**A QMC sample has no honest standard error.** The points are deterministic and
correlated by construction, so the usual ``s / sqrt(N)`` is meaningless -- it
typically *understates* the error, which is the dangerous direction. The fix is
randomised QMC: run several independently scrambled copies and take the spread
of their means. Pass ``n_replications`` to get that; without it, ``std_error``
on a QMC result is ``None`` rather than a number that would be believed.

Antithetic variates can make things worse
-----------------------------------------
The estimator is exact for a linear model -- the two halves of each pair have
correlation ``-1`` and the variance collapses. But the reduction is guaranteed
only when the model is *monotone* in every input. For a function symmetric about
the distribution's centre, ``f(u)`` and ``f(1-u)`` are positively correlated and
the pairing makes things worse. Both cases measured on a standard normal input,
400 repetitions of 2000 samples:

    f(x) = exp(0.7 x)   monotone     0.41x the variance of crude
    f(x) = x^2          symmetric    1.84x

So it is free, and it is not safe. Stratification is the variance-reduction
method that cannot backfire.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.stats import qmc

from simulacrum.uq.forward.distributions import JointIndependent

__all__ = ["MonteCarloResult", "propagate", "unit_design", "DESIGNS"]

DESIGNS = ("crude", "lhs", "sobol", "halton", "antithetic")


def unit_design(design: str, n: int, dim: int,
                rng: np.random.Generator) -> np.ndarray:
    """An ``(n, dim)`` design on the unit hypercube.

    Everything distribution-specific happens afterwards, in
    :meth:`~.distributions.JointIndependent.ppf`.
    """
    design = design.lower()
    if design == "crude":
        return rng.random((n, dim))

    if design == "antithetic":
        half = (n + 1) // 2
        base = rng.random((half, dim))
        # the pair (u, 1-u) has the same margin and perfectly negative rank
        # correlation, which is what cancels the linear part of the model
        return np.vstack([base, 1.0 - base])[:n]

    if design == "lhs":
        return qmc.LatinHypercube(d=dim, seed=rng).random(n)

    if design in ("sobol", "halton"):
        if design == "sobol" and n > 0 and (n & (n - 1)) != 0:
            warnings.warn(
                f"Sobol's balance properties hold for powers of two; {n} is "
                f"not one. Use {1 << int(np.ceil(np.log2(n)))} or accept a "
                f"sequence that may be no better than random.",
                RuntimeWarning, stacklevel=3)
        engine = (qmc.Sobol(d=dim, scramble=True, seed=rng) if design == "sobol"
                  else qmc.Halton(d=dim, scramble=True, seed=rng))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # scipy's own power-of-2 notice
            return engine.random(n)

    raise NotImplementedError(
        f"Unknown design '{design}'. Available: {list(DESIGNS)}")


@dataclass
class MonteCarloResult:
    """Outputs of a propagation, and an honest error estimate or none."""

    samples: np.ndarray                      #: ``(n, dim)`` inputs used
    values: np.ndarray                       #: ``(n,)`` model outputs
    design: str = "crude"
    n_evaluations: int = 0
    #: spread of replicate means, for randomised QMC; ``None`` otherwise
    replicate_means: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
        self.values = np.asarray(self.values, dtype=float).ravel()
        if not self.n_evaluations:
            self.n_evaluations = self.values.size

    @property
    def n_samples(self) -> int:
        return int(self.values.size)

    def mean(self) -> float:
        return float(self.values.mean())

    def variance(self) -> float:
        """Variance of the *output*, not of the estimator."""
        return float(self.values.var(ddof=1))

    def std(self) -> float:
        return float(np.sqrt(self.variance()))

    @property
    def std_error(self) -> float | None:
        """Standard error of the mean, or ``None`` when it would be a lie.

        For an iid design it is ``s / sqrt(N)``. For an unreplicated
        quasi-random design the points are deterministic and correlated, so that
        formula understates the true error -- and understating is the dangerous
        direction. Replicate the design (``n_replications``) to get a real one.
        """
        if self.replicate_means is not None:
            k = self.replicate_means.size
            return float(self.replicate_means.std(ddof=1) / np.sqrt(k))
        if self.design in ("sobol", "halton"):
            return None
        if self.design == "antithetic":
            # the pairs are dependent, so treat each pair as one observation
            pair_means = 0.5 * (self.values[:self.n_samples // 2]
                                + self.values[(self.n_samples + 1) // 2:][
                                    :self.n_samples // 2])
            k = pair_means.size
            return float(pair_means.std(ddof=1) / np.sqrt(k)) if k > 1 else None
        return float(np.sqrt(self.variance() / self.n_samples))

    def confidence_interval(self, level: float = 0.95):
        """Normal-theory interval on the mean.

        Leans on the central limit theorem, which is an assumption and not
        always a safe one: for a heavy-tailed output, or an indicator function
        with a small failure probability, the sample mean is nowhere near
        normal at practical ``N`` and this interval is optimistic. That is the
        reason :mod:`~.reliability` reports a coefficient of variation instead.
        """
        error = self.std_error
        if error is None:
            raise ValueError(
                f"a '{self.design}' design has no valid standard error without "
                f"replication; pass n_replications to propagate()")
        z = float(stats.norm.ppf(0.5 * (1.0 + level)))
        return self.mean() - z * error, self.mean() + z * error

    def quantile(self, q) -> np.ndarray:
        return np.quantile(self.values, q)

    def running_mean(self) -> np.ndarray:
        """Cumulative mean, for a convergence plot. Only meaningful for a
        design whose points arrive in an exchangeable order."""
        return np.cumsum(self.values) / np.arange(1, self.n_samples + 1)

    def summary(self) -> str:
        error = self.std_error
        tail = ("standard error unavailable (unreplicated QMC)"
                if error is None else f"standard error {error:.6g}")
        return (f"{self.design}: {self.n_samples} samples, "
                f"{self.n_evaluations} evaluations\n"
                f"  mean {self.mean():.6g}   std {self.std():.6g}   {tail}")

    def __repr__(self) -> str:
        return (f"MonteCarloResult({self.design}, n={self.n_samples}, "
                f"mean={self.mean():.6g})")


def propagate(model, distribution: JointIndependent, n_samples: int,
              design: str = "crude", n_replications: int | None = None,
              seed: int | None = None, vectorised: bool = False
              ) -> MonteCarloResult:
    """Push ``distribution`` through ``model`` and describe the output.

    Parameters
    ----------
    model : ``x -> scalar`` for a single input row, or ``(n, dim) -> (n,)`` if
        ``vectorised``. The only contact with the physics; an FE solve goes
        here.
    distribution : a :class:`~.distributions.JointIndependent`.
    n_samples : points per replication.
    design : one of :data:`DESIGNS`.
    n_replications : independent scrambles. Required to get a standard error
        out of a quasi-random design, and harmless for the others. The total
        cost is ``n_samples * n_replications``.
    """
    if design.lower() not in DESIGNS:
        raise NotImplementedError(
            f"Unknown design '{design}'. Available: {list(DESIGNS)}")
    if n_samples < 2:
        raise ValueError("at least two samples are needed")

    rng = np.random.default_rng(seed)
    reps = max(1, int(n_replications or 1))

    all_x, all_y, rep_means = [], [], []
    for _ in range(reps):
        unit = unit_design(design, n_samples, distribution.dim, rng)
        x = distribution.ppf(unit)
        y = (np.asarray(model(x), dtype=float).ravel() if vectorised
             else np.array([float(model(row)) for row in x]))
        if y.size != n_samples:
            raise ValueError(f"the model returned {y.size} values for "
                             f"{n_samples} inputs")
        all_x.append(x)
        all_y.append(y)
        rep_means.append(y.mean())

    return MonteCarloResult(
        samples=np.vstack(all_x), values=np.concatenate(all_y),
        design=design.lower(),
        n_evaluations=n_samples * reps,
        replicate_means=(np.array(rep_means) if reps > 1 else None),
        info={"n_replications": reps, "n_per_replication": n_samples})
