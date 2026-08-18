r"""
Failure probability: the Monte Carlo problem where crude Monte Carlo fails.

A limit state ``g(x)`` divides the input space into safe (``g > 0``) and failed
(``g <= 0``), and the quantity wanted is ``P_f = P[g(X) <= 0]``. Crude Monte
Carlo estimates it by counting failures, and for the probabilities structural
engineering actually cares about that does not work.

The arithmetic is the whole motivation
--------------------------------------
The estimator is a binomial proportion, so its coefficient of variation is

    CoV = sqrt((1 - P_f) / (N P_f))  ~  1 / sqrt(N P_f).

Ten percent accuracy therefore needs ``N ~ 100 / P_f``:

    P_f = 1e-2  ->  1e4 samples
    P_f = 1e-4  ->  1e6
    P_f = 1e-6  ->  1e8       a target reliability index of 4.75

At one FE solve per sample, the last line is not an analysis anyone runs. Worse,
the failure is not loud: with ``N`` far too small the estimator usually returns
*zero failures*, and zero is a perfectly plausible-looking answer.
:func:`monte_carlo_probability` reports the CoV and says so when no failures
were seen, rather than returning ``0.0`` unqualified.

The three ways out, in the order they are usually applied
---------------------------------------------------------
**FORM** transforms to standard-normal space, finds the failure point closest to
the origin, and reports ``P_f = Phi(-beta)`` where ``beta`` is that distance.
Not a Monte Carlo method and not exact -- it replaces the limit state by its
tangent plane at that point -- but exact when the limit state really is linear
in ``u``-space, cheap, and it produces the design point the next method needs.

**Importance sampling about the design point** shifts the sampling density onto
the failure region, which is where the integrand lives. Unbiased, and the
variance drops by roughly the factor the failure probability was small by. Safe
here specifically because the proposal is the same Gaussian merely translated,
so its tails match the target's -- a *narrower* proposal would give an
infinite-variance estimator.

**Subset simulation** needs no design point and no assumption about the shape of
the failure region. It writes the rare event as a chain of conditional events
that are each common,

    P_f = P(F_1) * P(F_2|F_1) * ... * P(F_m|F_{m-1}),

with the intermediate thresholds chosen adaptively so each conditional
probability is ``p0`` (0.1 by convention). The conditional samples come from a
Markov chain, so the cost is ``m`` levels of ``N`` samples rather than
``1/P_f``, and ``m`` grows like ``log(P_f)``. That turns 1e8 into a few
thousand.

The price of subset simulation is that its samples within a level are
correlated, so its coefficient of variation is larger than the independent
formula suggests, and the standard estimate below assumes independence between
levels, which is optimistic. It is reported as an estimate and labelled as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from simulacrum.uq.forward.distributions import JointIndependent

__all__ = [
    "ReliabilityResult",
    "monte_carlo_probability",
    "form",
    "importance_sampling_probability",
    "subset_simulation",
]


@dataclass
class ReliabilityResult:
    """A failure probability, its reliability index, and its own credibility."""

    probability: float
    method: str
    n_evaluations: int
    cov: float | None = None            #: coefficient of variation, if known
    design_point: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def beta(self) -> float:
        """Generalised reliability index ``-Phi^-1(P_f)``.

        The number codes are written in. It is a monotone relabelling of the
        probability and carries no extra information, but it compresses the
        interesting range into roughly 2 to 5, which is why it is used.
        """
        if self.probability <= 0.0:
            return float("inf")
        if self.probability >= 1.0:
            return float("-inf")
        return float(-stats.norm.ppf(self.probability))

    def confidence_interval(self, level: float = 0.95):
        if self.cov is None:
            raise ValueError(f"'{self.method}' does not provide a coefficient "
                             f"of variation")
        z = float(stats.norm.ppf(0.5 * (1.0 + level)))
        spread = z * self.cov * self.probability
        return max(0.0, self.probability - spread), self.probability + spread

    def __repr__(self) -> str:
        cov = "?" if self.cov is None else f"{self.cov:.3f}"
        return (f"ReliabilityResult({self.method}, Pf={self.probability:.4g}, "
                f"beta={self.beta:.3f}, CoV={cov}, "
                f"{self.n_evaluations} evaluations)")


def _evaluate(limit_state, x, vectorised):
    if vectorised:
        return np.asarray(limit_state(x), dtype=float).ravel()
    return np.array([float(limit_state(row)) for row in np.atleast_2d(x)])


def required_samples(probability: float, cov: float = 0.1) -> int:
    """Crude Monte Carlo samples needed for a given coefficient of variation.

    ``N = (1 - P_f) / (P_f * CoV^2)``. Worth calling before starting a run, to
    find out whether the run is possible.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError("the probability must lie strictly in (0, 1)")
    return int(np.ceil((1.0 - probability) / (probability * cov ** 2)))


def monte_carlo_probability(limit_state, distribution: JointIndependent,
                            n_samples: int, seed: int | None = None,
                            vectorised: bool = False) -> ReliabilityResult:
    """Count failures. The baseline, and the reference the others are checked
    against whenever the probability is large enough to afford it."""
    rng = np.random.default_rng(seed)
    x = distribution.sample(n_samples, rng)
    g = _evaluate(limit_state, x, vectorised)

    n_fail = int(np.count_nonzero(g <= 0.0))
    probability = n_fail / n_samples

    if n_fail == 0:
        # returning 0.0 unqualified is the failure mode this guards against;
        # the honest statement is an upper bound
        upper = 3.0 / n_samples                      # rule of three, 95%
        return ReliabilityResult(
            probability=0.0, method="monte-carlo", n_evaluations=n_samples,
            cov=None,
            info={"n_failures": 0,
                  "note": f"no failures in {n_samples} samples; this is an "
                          f"upper bound of about {upper:.2g} at 95% "
                          f"confidence, not a probability of zero",
                  "upper_bound_95": upper})

    cov = float(np.sqrt((1.0 - probability) / (n_samples * probability)))
    return ReliabilityResult(
        probability=probability, method="monte-carlo",
        n_evaluations=n_samples, cov=cov,
        info={"n_failures": n_fail,
              "samples_for_10_percent_cov": required_samples(probability)})


def form(limit_state, distribution: JointIndependent, x0=None,
         max_iterations: int = 100, tolerance: float = 1e-8,
         step: float = 1e-6, vectorised: bool = False) -> ReliabilityResult:
    r"""First-order reliability method by the HL-RF iteration.

    Works in standard-normal space, where the joint density is spherically
    symmetric and decreasing in ``|u|``, so the most probable failure point is
    simply the closest point of the failure surface to the origin. HL-RF is the
    fixed-point iteration for that constrained minimisation,

        u <- [ (grad g . u - g) / |grad g|^2 ] grad g.

    Then ``P_f ~ Phi(-beta)`` with ``beta = |u*|``.

    **Exact only for a linear limit state**, where the failure surface is a
    plane and the tangent-plane approximation is not an approximation at all.
    For a curved surface it errs in a direction set by the curvature -- towards
    safety for a convex failure region, against it for a concave one -- and
    nothing in the output reveals which. For a strongly nonlinear or
    multiply-connected limit state it can also converge to a *local* closest
    point and miss the dominant failure mode entirely. Use it to get a design
    point and then verify with sampling, not on its own.

    The gradient is taken by central differences in ``u``-space, at
    ``2 * dim`` limit-state evaluations per iteration.
    """
    dim = distribution.dim
    u = (np.zeros(dim) if x0 is None
         else distribution.to_standard_normal(np.atleast_2d(x0))[0])
    n_eval = 0

    def g_of_u(u_point):
        nonlocal n_eval
        x = distribution.from_standard_normal(np.atleast_2d(u_point))
        n_eval += 1
        return float(_evaluate(limit_state, x, vectorised)[0])

    history = []
    for iteration in range(max_iterations):
        g = g_of_u(u)
        grad = np.zeros(dim)
        for i in range(dim):
            plus, minus = u.copy(), u.copy()
            plus[i] += step
            minus[i] -= step
            grad[i] = (g_of_u(plus) - g_of_u(minus)) / (2.0 * step)

        norm_sq = float(np.dot(grad, grad))
        if norm_sq <= 0.0:
            raise RuntimeError(
                "the limit-state gradient vanished; HL-RF has no direction to "
                "move in, which usually means g is flat or the finite-"
                "difference step is too small for its scale")

        u_new = ((np.dot(grad, u) - g) / norm_sq) * grad
        history.append(float(np.linalg.norm(u_new)))
        if np.linalg.norm(u_new - u) < tolerance:
            u = u_new
            break
        u = u_new
    else:
        raise RuntimeError(
            f"HL-RF did not converge in {max_iterations} iterations; the last "
            f"reliability indices were {np.round(history[-5:], 6)}")

    beta = float(np.linalg.norm(u))
    # the sign of g at the origin says which side the mean point is on
    if g_of_u(np.zeros(dim)) < 0.0:
        beta = -beta

    return ReliabilityResult(
        probability=float(stats.norm.cdf(-beta)), method="form",
        n_evaluations=n_eval,
        design_point=distribution.from_standard_normal(np.atleast_2d(u))[0],
        info={"beta_hl": beta, "u_star": u, "iterations": iteration + 1,
              "alpha": u / max(np.linalg.norm(u), 1e-300),
              "history": np.array(history)})


def importance_sampling_probability(
        limit_state, distribution: JointIndependent, n_samples: int,
        design_point=None, seed: int | None = None, vectorised: bool = False
        ) -> ReliabilityResult:
    r"""Failure probability by importance sampling about the design point.

    Samples from a unit Gaussian in ``u``-space *centred on the design point*
    instead of the origin, and reweights by ``phi(u) / phi(u - u*)``. Roughly
    half the samples then land in the failure region regardless of how small the
    probability is, so the cost stops scaling with ``1/P_f``.

    Unbiased for any centre. The shift is safe because the proposal is the same
    Gaussian translated, so the weights are bounded on any bounded region and
    the tails match; narrowing the proposal instead would give an
    infinite-variance estimator that still returns confident-looking numbers.

    ``design_point`` is in input space; if omitted, :func:`form` is run first
    and its cost is included in the reported evaluation count.
    """
    n_eval = 0
    if design_point is None:
        first = form(limit_state, distribution, vectorised=vectorised)
        u_star = first.info["u_star"]
        n_eval += first.n_evaluations
    else:
        u_star = distribution.to_standard_normal(
            np.atleast_2d(design_point))[0]

    rng = np.random.default_rng(seed)
    dim = distribution.dim
    u = rng.standard_normal((n_samples, dim)) + u_star

    # log phi(u) - log phi(u - u*), the Gaussian ratio written to avoid
    # forming either density
    log_w = -0.5 * np.sum(u ** 2, axis=1) + 0.5 * np.sum((u - u_star) ** 2,
                                                         axis=1)
    weights = np.exp(log_w)

    x = distribution.from_standard_normal(u)
    g = _evaluate(limit_state, x, vectorised)
    n_eval += n_samples

    contributions = weights * (g <= 0.0)
    probability = float(contributions.mean())
    if probability <= 0.0:
        return ReliabilityResult(
            probability=0.0, method="importance-sampling",
            n_evaluations=n_eval, cov=None,
            info={"note": "no failures even under the shifted density, which "
                          "means the design point is wrong rather than that "
                          "the probability is zero"})

    cov = float(np.sqrt(contributions.var(ddof=1) / n_samples) / probability)
    effective = float(weights.sum() ** 2 / np.sum(weights ** 2))
    return ReliabilityResult(
        probability=probability, method="importance-sampling",
        n_evaluations=n_eval, cov=cov,
        design_point=distribution.from_standard_normal(
            np.atleast_2d(u_star))[0],
        info={"effective_sample_size": effective,
              "fraction_failed": float(np.mean(g <= 0.0))})


def subset_simulation(limit_state, distribution: JointIndependent,
                      n_per_level: int = 2000, p0: float = 0.1,
                      max_levels: int = 20, proposal_std: float = 1.0,
                      seed: int | None = None, vectorised: bool = False
                      ) -> ReliabilityResult:
    r"""Failure probability as a product of conditional probabilities.

    Au and Beck (2001). Sample the whole space, keep the worst ``p0`` fraction,
    use them as seeds for Markov chains that populate that conditional region,
    keep the worst ``p0`` of *those*, and so on until the threshold reaches
    zero. Then

        P_f = p0^(m-1) * (fraction failing at the last level).

    Needs no design point and makes no assumption about the shape or
    connectedness of the failure region, which is what distinguishes it from
    FORM and from importance sampling built on FORM. The cost grows like
    ``log(P_f)`` rather than ``1/P_f``.

    The Markov step is the modified Metropolis of the original paper: each
    coordinate is perturbed and accepted *independently* against its own 1-D
    standard normal, and the whole candidate is then rejected if it leaves the
    current level's region. Perturbing all ``dim`` coordinates jointly and
    accepting once is the obvious alternative and its acceptance rate decays
    exponentially with dimension, which is exactly the regime this method is
    for.

    The reported coefficient of variation assumes the levels are independent.
    They are not -- each is seeded from the last -- so it is optimistic, and it
    is labelled an estimate rather than a bound.
    """
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must lie strictly in (0, 1)")

    rng = np.random.default_rng(seed)
    dim = distribution.dim
    n_keep = max(1, int(round(p0 * n_per_level)))
    n_eval = 0

    u = rng.standard_normal((n_per_level, dim))
    g = _evaluate(limit_state, distribution.from_standard_normal(u), vectorised)
    n_eval += n_per_level

    thresholds, level_probabilities, acceptances = [], [], []

    for level in range(max_levels):
        order = np.argsort(g)
        u, g = u[order], g[order]
        threshold = float(g[n_keep - 1])

        if threshold <= 0.0:
            # the last level reaches the real limit state; its conditional
            # probability is measured rather than fixed at p0
            final = float(np.mean(g <= 0.0))
            thresholds.append(0.0)
            level_probabilities.append(final)
            probability = float(np.prod(level_probabilities))

            # Au and Beck's independent-levels estimate
            terms = [(1.0 - p) / (p * n_per_level)
                     for p in level_probabilities if p > 0.0]
            cov = float(np.sqrt(np.sum(terms))) if terms else None
            return ReliabilityResult(
                probability=probability, method="subset-simulation",
                n_evaluations=n_eval, cov=cov,
                info={"n_levels": len(level_probabilities),
                      "thresholds": np.array(thresholds),
                      "level_probabilities": np.array(level_probabilities),
                      "acceptance_rates": np.array(acceptances),
                      "cov_note": "assumes independent levels; optimistic"})

        thresholds.append(threshold)
        level_probabilities.append(p0)

        # repopulate the conditional region from the survivors
        seeds_u, seeds_g = u[:n_keep].copy(), g[:n_keep].copy()
        chain_u = [seeds_u]
        chain_g = [seeds_g]
        accepted = 0
        n_chains = n_keep
        steps = int(np.ceil(n_per_level / n_chains)) - 1

        current_u, current_g = seeds_u.copy(), seeds_g.copy()
        for _ in range(max(1, steps)):
            candidate = current_u.copy()
            # modified Metropolis: componentwise accept against the 1-D margin,
            # which keeps the acceptance rate from collapsing with dimension
            perturbed = current_u + proposal_std * rng.standard_normal(
                (n_chains, dim))
            ratio = (-0.5 * perturbed ** 2) - (-0.5 * current_u ** 2)
            take = np.log(rng.random((n_chains, dim))) < ratio
            candidate[take] = perturbed[take]

            candidate_g = _evaluate(
                limit_state, distribution.from_standard_normal(candidate),
                vectorised)
            n_eval += n_chains

            inside = candidate_g <= threshold
            accepted += int(np.count_nonzero(inside))
            current_u = np.where(inside[:, None], candidate, current_u)
            current_g = np.where(inside, candidate_g, current_g)

            chain_u.append(current_u.copy())
            chain_g.append(current_g.copy())

        u = np.vstack(chain_u)[:n_per_level]
        g = np.concatenate(chain_g)[:n_per_level]
        acceptances.append(accepted / max(1, n_chains * max(1, steps)))

    raise RuntimeError(
        f"subset simulation did not reach the limit state in {max_levels} "
        f"levels; the last threshold was {thresholds[-1]:.6g}, implying a "
        f"probability below {p0 ** max_levels:.2g}")
