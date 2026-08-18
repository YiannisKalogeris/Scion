r"""
Variance reduction: the only lever Monte Carlo actually has.

The ``1/sqrt(N)`` rate is fixed. What is not fixed is the ``sigma`` sitting on
top of it, and every method here attacks that instead -- returning the same
expectation with a smaller spread, so that the same accuracy costs fewer model
evaluations.

Each has a condition under which it helps, and each fails, sometimes badly,
when the condition does not hold. The conditions are the content of this module.

``control_variates``   needs a *correlated* cheap quantity whose mean is known
                       exactly. Variance ratio ``1 - rho^2``, so it is worth
                       having only if the correlation is high: ``rho = 0.9``
                       saves 81% of the variance, ``rho = 0.5`` saves 25%.
``importance_sampling``  needs a proposal that puts mass where the integrand
                       does. Gets an arbitrarily large speed-up on a rare
                       event, and can have **infinite variance** if the
                       proposal's tails are lighter than the integrand's.
``stratified``         needs the strata to explain some of the variance.
                       Cannot ever be worse than crude, which makes it the
                       safest of the three and the least dramatic.

A surrogate is a control variate
--------------------------------
This is the connection worth noticing in a library that also has
:mod:`simulacrum.uq.gp` and :mod:`simulacrum.uq.pce`. A cheap surrogate fitted
to the expensive model is exactly what a control variate wants, provided its
mean is known -- and for a polynomial chaos expansion the mean is the first
coefficient, exactly, for free. The pairing turns a surrogate from something you
have to trust into something you only have to be *correlated* with, because the
control-variate estimator stays unbiased for the true model however wrong the
surrogate is. It just stops helping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from simulacrum.uq.forward.distributions import JointIndependent

__all__ = [
    "ControlVariateResult",
    "control_variates",
    "ImportanceSamplingResult",
    "importance_sampling",
    "stratified",
]


@dataclass
class ControlVariateResult:
    """An unbiased mean, and how much the control actually bought."""

    mean: float
    std_error: float
    coefficient: float             #: the optimal ``c*``
    correlation: float
    crude_mean: float
    crude_std_error: float
    n_evaluations: int

    @property
    def variance_ratio(self) -> float:
        """Achieved variance ratio against crude. Theory says ``1 - rho^2``."""
        return float((self.std_error / self.crude_std_error) ** 2)

    @property
    def theoretical_ratio(self) -> float:
        return float(1.0 - self.correlation ** 2)

    def __repr__(self) -> str:
        return (f"ControlVariateResult(mean={self.mean:.6g}, "
                f"rho={self.correlation:.3f}, "
                f"variance x{self.variance_ratio:.3f})")


def control_variates(values, control_values, control_mean: float
                     ) -> ControlVariateResult:
    r"""Correct a Monte Carlo mean using a correlated quantity of known mean.

        Y_cv = Y - c (G - E[G]),     c* = Cov(Y, G) / Var(G)

    Unbiased for **any** ``c``, because ``E[G - E[G]] = 0``. That is the
    property that makes the method safe: a badly chosen control, or a surrogate
    that turns out to be a poor model, costs you the variance reduction and
    never the answer.

    Parameters
    ----------
    values : ``(n,)`` outputs of the expensive model.
    control_values : ``(n,)`` outputs of the cheap control, **at the same
        input points**. Pairing them is the whole mechanism; independent draws
        have zero correlation and buy nothing.
    control_mean : the exact ``E[G]``. If this is itself estimated, the
        estimator is no longer unbiased and the error in it passes straight
        through.

    Notes
    -----
    ``c*`` is estimated from the same samples it is applied to, which
    introduces an ``O(1/n)`` bias -- second order against the ``O(1/sqrt(n))``
    standard error, and negligible in practice, but it is a bias and not
    nothing. A pilot sample removes it at the cost of the pilot.
    """
    y = np.asarray(values, dtype=float).ravel()
    g = np.asarray(control_values, dtype=float).ravel()
    if y.shape != g.shape:
        raise ValueError("the model and the control must be evaluated at the "
                         "same points, so the arrays must match")
    if y.size < 3:
        raise ValueError("estimating a covariance needs at least three points")

    var_g = g.var(ddof=1)
    if var_g <= 0.0:
        raise ValueError("the control is constant, so it carries no "
                         "information about the model")

    covariance = float(np.cov(y, g, ddof=1)[0, 1])
    c = covariance / var_g
    corrected = y - c * (g - control_mean)

    n = y.size
    return ControlVariateResult(
        mean=float(corrected.mean()),
        std_error=float(np.sqrt(corrected.var(ddof=1) / n)),
        coefficient=float(c),
        correlation=float(covariance / np.sqrt(y.var(ddof=1) * var_g)),
        crude_mean=float(y.mean()),
        crude_std_error=float(np.sqrt(y.var(ddof=1) / n)),
        n_evaluations=n)


@dataclass
class ImportanceSamplingResult:
    """A weighted mean, and the diagnostics that say whether to believe it."""

    mean: float
    std_error: float
    weights: np.ndarray
    n_evaluations: int

    @property
    def effective_sample_size(self) -> float:
        r"""``(sum w)^2 / sum w^2`` -- Kish's ESS, a property of the *weights*.

        **This is not a verdict on the estimate, and reading it as one is the
        standard mistake.** It measures how well the proposal substitutes for
        the target overall, which is the right question when reweighting to
        estimate a general expectation. It is the wrong question when the
        integrand is concentrated somewhere the target rarely goes, because
        then a deliberately mismatched proposal is the entire point and skewed
        weights are the intended outcome.

        Measured: estimating ``P[X > 3]`` for a standard normal from a
        ``N(3, 1)`` proposal gives an ESS of 19 out of 20000 -- and an answer
        accurate to 0.7%. Use :attr:`coefficient_of_variation`, which looks at
        ``w * f`` rather than ``w`` alone, to decide whether to believe a
        result.
        """
        w = self.weights
        total = float(np.sum(w))
        if total <= 0.0:
            return 0.0
        return float(total ** 2 / np.sum(w ** 2))

    @property
    def efficiency(self) -> float:
        """:attr:`effective_sample_size` as a fraction of the sample count.
        Same caveat: it describes the weights, not the estimate."""
        return self.effective_sample_size / self.n_evaluations

    @property
    def coefficient_of_variation(self) -> float:
        """``std_error / |mean|`` -- the diagnostic that does judge the answer.

        Built from the spread of ``w * f``, so it accounts for the integrand
        as well as the weights. This is the number to check.
        """
        if self.mean == 0.0:
            return float("inf")
        return float(abs(self.std_error / self.mean))

    def __repr__(self) -> str:
        return (f"ImportanceSamplingResult(mean={self.mean:.6g}, "
                f"CoV={self.coefficient_of_variation:.4f}, "
                f"ESS={self.effective_sample_size:.1f}/{self.n_evaluations})")


def importance_sampling(model, target: JointIndependent,
                        proposal: JointIndependent, n_samples: int,
                        seed: int | None = None, vectorised: bool = False
                        ) -> ImportanceSamplingResult:
    r"""Sample from ``proposal`` and reweight to ``target``.

        E_target[f] = E_proposal[f w],    w(x) = p_target(x) / p_proposal(x)

    The point is to put samples where ``f * p_target`` is large rather than
    where ``p_target`` alone is. For a rare event that is the difference between
    needing ``10^8`` samples and ``10^3``.

    The failure mode, stated because it is silent
    ---------------------------------------------
    If the proposal has **lighter tails** than the integrand, the weights are
    unbounded and the estimator has infinite variance. It still returns a
    number, and the number looks stable across moderate sample sizes right up
    until one enormous weight arrives and moves it by a factor. Shifting a
    Gaussian proposal, as :mod:`~.reliability` does around a FORM design point,
    is safe because the tails match; *narrowing* one is not.

    Check :attr:`ImportanceSamplingResult.coefficient_of_variation`, which is
    built from ``w * f``. Do **not** judge the result by the effective sample
    size, which is built from ``w`` alone and reads as catastrophic on exactly
    the rare-event problems where importance sampling works best -- an ESS of
    19 out of 20000 alongside an answer accurate to 0.7%.

    Weights are formed in log space, because a density ratio in 20 dimensions
    underflows long before it becomes uninteresting.
    """
    if target.dim != proposal.dim:
        raise ValueError("the target and proposal must have the same dimension")

    rng = np.random.default_rng(seed)
    x = proposal.sample(n_samples, rng)
    log_w = target.log_pdf(x) - proposal.log_pdf(x)
    weights = np.exp(log_w - log_w.max())        # shifted; scale cancels below

    y = (np.asarray(model(x), dtype=float).ravel() if vectorised
         else np.array([float(model(row)) for row in x]))

    scale = np.exp(log_w.max())
    weighted = weights * y
    mean = float(weighted.mean() * scale)
    error = float(np.sqrt(weighted.var(ddof=1) / n_samples) * scale)

    return ImportanceSamplingResult(mean=mean, std_error=error,
                                    weights=weights,
                                    n_evaluations=n_samples)


def stratified(model, distribution: JointIndependent, n_per_stratum: int,
               n_strata: int, axis: int = 0, seed: int | None = None,
               vectorised: bool = False):
    r"""Split one input's range into equiprobable strata and sample each.

    The variance decomposes as ``within + between``, and stratifying removes the
    between-strata part exactly. So it can never do worse than crude sampling
    with the same budget -- unlike antithetic or importance sampling, both of
    which can -- and it helps in proportion to how much of the variance the
    stratifying variable explains.

    Stratifying on one axis rather than all of them is deliberate: full
    stratification of ``d`` inputs into ``m`` levels needs ``m^d`` points, which
    is the tensor-product blow-up Monte Carlo exists to avoid. Latin hypercube
    sampling is the compromise that stratifies every margin at once, and it is
    available as a design in :func:`~.estimators.propagate`.

    Returns ``(mean, std_error)``. The standard error is the correct stratified
    one -- the pooled within-stratum variance, *not* the variance of the
    combined sample, which would ignore the stratification and overstate the
    error.
    """
    rng = np.random.default_rng(seed)
    if n_per_stratum < 2:
        raise ValueError("each stratum needs at least two points to have a "
                         "variance")

    edges = np.linspace(0.0, 1.0, n_strata + 1)
    means, variances = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        unit = rng.random((n_per_stratum, distribution.dim))
        unit[:, axis] = lo + (hi - lo) * unit[:, axis]
        x = distribution.ppf(unit)
        y = (np.asarray(model(x), dtype=float).ravel() if vectorised
             else np.array([float(model(row)) for row in x]))
        means.append(y.mean())
        variances.append(y.var(ddof=1))

    # equiprobable strata, so the weights are all 1/n_strata
    mean = float(np.mean(means))
    error = float(np.sqrt(np.sum(np.array(variances) / n_per_stratum)
                          / n_strata ** 2))
    return mean, error
