r"""
What a sampler returns.

One container for all three methods, which is only defensible because it knows
that they do not all return the same *kind* of thing:

**A chain** (Metropolis-Hastings, HMC) is an ordered, correlated sequence. Its
health is measured by autocorrelation -- effective sample size, R-hat across
several chains -- and burn-in and thinning are meaningful operations on it.

**A particle set** (TMCMC) is an unordered population carried through a
sequence of tempered distributions. Lag-based autocorrelation is undefined for
it: sample 7 has no temporal relationship to sample 8. Asking for an ESS is a
category error, and the corresponding diagnostic is
:attr:`PosteriorSamples.unique_fraction`, because resampling duplicates
particles and a collapsed population can hold ten thousand entries with forty
distinct values in them.

:attr:`PosteriorSamples.ordered` records which one it is, and the methods that
do not apply raise rather than returning a number that means nothing. Burn-in on
a particle set is the same category error and is refused for the same reason.

Evidence
--------
:attr:`log_evidence` is the log marginal likelihood ``log p(data)``, and it is
``None`` for the samplers that cannot produce it. Metropolis-Hastings and HMC
work with an unnormalised posterior and never see the constant they are missing;
TMCMC gets it for free from the tempering weights. That asymmetry is worth
knowing before choosing a sampler for model comparison, since the evidence is
the entire basis of a Bayes factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulacrum.uq.bayesian.diagnostics import (
    effective_sample_size,
    geweke,
    integrated_autocorrelation_time,
)

__all__ = ["PosteriorSamples"]


@dataclass
class PosteriorSamples:
    """Posterior draws plus everything needed to judge them."""

    samples: np.ndarray                      #: ``(n, dim)``
    method: str = "unknown"
    log_posterior: np.ndarray | None = None  #: ``(n,)`` where the sampler kept it
    acceptance_rate: float | None = None
    log_evidence: float | None = None
    n_evaluations: int = 0                   #: forward-model calls, the real cost
    #: False for a particle set, where order carries no information
    ordered: bool = True
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.samples = np.atleast_2d(np.asarray(self.samples, dtype=float))
        if self.log_posterior is not None:
            self.log_posterior = np.asarray(self.log_posterior, dtype=float)

    # -- shape ----------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def dim(self) -> int:
        return int(self.samples.shape[1])

    # -- summaries ------------------------------------------------------
    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=0, ddof=1)

    def cov(self) -> np.ndarray:
        return np.atleast_2d(np.cov(self.samples, rowvar=False))

    def quantile(self, q) -> np.ndarray:
        return np.quantile(self.samples, q, axis=0)

    def credible_interval(self, level: float = 0.95) -> np.ndarray:
        """Equal-tailed interval, ``(2, dim)``.

        Equal-tailed, not highest-density: they differ for a skewed or
        multimodal posterior, and the equal-tailed one is the one that is a
        pure function of the marginal quantiles.
        """
        tail = 0.5 * (1.0 - level)
        return np.quantile(self.samples, [tail, 1.0 - tail], axis=0)

    # -- chain operations -----------------------------------------------
    def _require_ordered(self, what: str) -> None:
        if not self.ordered:
            raise ValueError(
                f"{what} assumes an ordered chain, and '{self.method}' returns "
                f"an unordered particle set -- sample i and sample i+1 have no "
                f"temporal relationship. Use unique_fraction instead.")

    def burn(self, n_burn: int) -> PosteriorSamples:
        """Drop the first ``n_burn`` samples."""
        self._require_ordered("burn-in")
        if n_burn >= self.n_samples:
            raise ValueError(f"cannot burn {n_burn} of {self.n_samples}")
        return self._slice(slice(n_burn, None))

    def thin(self, factor: int) -> PosteriorSamples:
        """Keep every ``factor``-th sample.

        Thinning throws information away: the mean of the thinned chain is a
        strictly worse estimator than the mean of the full one. It is worth
        doing to bound memory, or to make a scatter plot legible, and it is not
        worth doing to "remove correlation" -- that is what the ESS is for.
        """
        self._require_ordered("thinning")
        if factor < 1:
            raise ValueError("the thinning factor must be at least one")
        return self._slice(slice(None, None, factor))

    def _slice(self, index) -> PosteriorSamples:
        return PosteriorSamples(
            samples=self.samples[index], method=self.method,
            log_posterior=(None if self.log_posterior is None
                           else self.log_posterior[index]),
            acceptance_rate=self.acceptance_rate,
            log_evidence=self.log_evidence,
            n_evaluations=self.n_evaluations, ordered=self.ordered,
            info=dict(self.info))

    # -- diagnostics -----------------------------------------------------
    def ess(self) -> np.ndarray:
        """Effective sample size per component. Chains only."""
        self._require_ordered("the effective sample size")
        return effective_sample_size(self.samples)

    def autocorrelation_time(self) -> np.ndarray:
        self._require_ordered("the autocorrelation time")
        return integrated_autocorrelation_time(self.samples)

    def geweke(self, **kwargs) -> np.ndarray:
        self._require_ordered("the Geweke score")
        return geweke(self.samples, **kwargs)

    @property
    def unique_fraction(self) -> float:
        """Fraction of distinct rows.

        The diagnostic for a particle set. TMCMC resamples, so duplicates are
        expected and normal; a fraction that has collapsed towards zero means
        the tempering steps were too aggressive and the population degenerated
        into copies of a few survivors.

        Also informative for a chain, where it is one minus the rejection rate,
        so a very low value means the proposal is too wide.
        """
        return len(np.unique(self.samples, axis=0)) / self.n_samples

    # -- reporting -------------------------------------------------------
    def summary(self, names=None, level: float = 0.95) -> str:
        names = names or [f"theta[{i}]" for i in range(self.dim)]
        mean, std = self.mean(), self.std()
        lo, hi = self.credible_interval(level)

        head = (f"{self.method}: {self.n_samples} samples, "
                f"{self.n_evaluations} model evaluations")
        if self.acceptance_rate is not None:
            head += f", acceptance {self.acceptance_rate:.3f}"
        if self.log_evidence is not None:
            head += f", log Z = {self.log_evidence:.4f}"

        lines = [head, f"{'parameter':<14}{'mean':>12}{'std':>12}"
                       f"{f'{level:.0%} interval':>26}"]
        for i, name in enumerate(names):
            lines.append(f"{name:<14}{mean[i]:12.5f}{std[i]:12.5f}"
                         f"{lo[i]:13.5f}{hi[i]:13.5f}")

        if self.ordered:
            ess = self.ess()
            lines.append(f"{'ESS':<14}" + "".join(f"{e:12.1f}" for e in ess))
        else:
            lines.append(f"{'unique':<14}{self.unique_fraction:12.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        kind = "chain" if self.ordered else "particles"
        return (f"PosteriorSamples({self.method}, {kind}, "
                f"{self.n_samples} x {self.dim})")
