"""
Markov Chain Monte Carlo samplers for Bayesian inference.

Suitable for parameter estimation in PDE-constrained inverse problems:
given observed data, infer material properties, boundary conditions, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MCMCResult:
    """Container for MCMC output."""
    samples: np.ndarray          # (n_samples, dim)
    log_probs: np.ndarray        # (n_samples,)
    acceptance_rate: float
    n_samples: int

    @property
    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.samples.std(axis=0)

    def thin(self, factor: int) -> MCMCResult:
        """Thin the chain by keeping every `factor`-th sample."""
        return MCMCResult(
            samples=self.samples[::factor],
            log_probs=self.log_probs[::factor],
            acceptance_rate=self.acceptance_rate,
            n_samples=len(self.samples[::factor]),
        )

    def burn(self, n_burn: int) -> MCMCResult:
        """Discard the first `n_burn` samples."""
        return MCMCResult(
            samples=self.samples[n_burn:],
            log_probs=self.log_probs[n_burn:],
            acceptance_rate=self.acceptance_rate,
            n_samples=len(self.samples[n_burn:]),
        )


class MetropolisHastings:
    """Random-walk Metropolis–Hastings sampler.

    Parameters
    ----------
    log_posterior : callable
        ``(theta) -> float`` log-posterior (up to a constant).
    proposal_std : float or array
        Standard deviation of the Gaussian proposal.
    seed : int or None
    """

    def __init__(
        self,
        log_posterior: Callable[[np.ndarray], float],
        proposal_std: float | np.ndarray = 0.1,
        seed: int | None = None,
    ):
        self.log_posterior = log_posterior
        self.proposal_std = np.asarray(proposal_std)
        self.rng = np.random.default_rng(seed)

    def sample(self, x0: np.ndarray, n_samples: int, n_burn: int = 0) -> MCMCResult:
        """Run the chain.

        Parameters
        ----------
        x0 : (dim,) initial state
        n_samples : total samples to generate (including burn-in)
        n_burn : samples to discard from the beginning
        """
        dim = len(x0)
        samples = np.empty((n_samples, dim))
        log_probs = np.empty(n_samples)

        x = x0.copy()
        lp = self.log_posterior(x)
        n_accept = 0

        for i in range(n_samples):
            # Propose
            x_prop = x + self.rng.normal(0, self.proposal_std, size=dim)
            lp_prop = self.log_posterior(x_prop)

            # Accept/reject
            log_alpha = lp_prop - lp
            if np.log(self.rng.random()) < log_alpha:
                x = x_prop
                lp = lp_prop
                n_accept += 1

            samples[i] = x
            log_probs[i] = lp

        rate = n_accept / n_samples
        logger.info(f"MH: {n_samples} samples, acceptance rate = {rate:.3f}")

        result = MCMCResult(
            samples=samples, log_probs=log_probs,
            acceptance_rate=rate, n_samples=n_samples,
        )
        if n_burn > 0:
            result = result.burn(n_burn)
        return result


class AdaptiveMCMC(MetropolisHastings):
    """Adaptive Metropolis with Haario-style covariance adaptation.

    Adapts the proposal covariance during burn-in using the empirical
    covariance of accepted samples.

    Parameters
    ----------
    log_posterior : callable
    initial_std : float
        Initial isotropic proposal standard deviation.
    adapt_interval : int
        Adapt the proposal every N steps.
    adapt_after : int
        Start adapting after this many samples.
    target_rate : float
        Target acceptance rate (default: 0.234 for high dimensions).
    seed : int or None
    """

    def __init__(
        self,
        log_posterior: Callable[[np.ndarray], float],
        initial_std: float = 0.1,
        adapt_interval: int = 100,
        adapt_after: int = 500,
        target_rate: float = 0.234,
        seed: int | None = None,
    ):
        super().__init__(log_posterior, initial_std, seed)
        self.adapt_interval = adapt_interval
        self.adapt_after = adapt_after
        self.target_rate = target_rate

    def sample(self, x0: np.ndarray, n_samples: int, n_burn: int = 0) -> MCMCResult:
        dim = len(x0)
        samples = np.empty((n_samples, dim))
        log_probs = np.empty(n_samples)

        x = x0.copy()
        lp = self.log_posterior(x)
        n_accept = 0
        cov = np.eye(dim) * float(self.proposal_std) ** 2

        for i in range(n_samples):
            # Propose from adapted covariance
            x_prop = self.rng.multivariate_normal(x, (2.38**2 / dim) * cov)
            lp_prop = self.log_posterior(x_prop)

            log_alpha = lp_prop - lp
            if np.log(self.rng.random()) < log_alpha:
                x = x_prop
                lp = lp_prop
                n_accept += 1

            samples[i] = x
            log_probs[i] = lp

            # Adapt covariance
            if i > self.adapt_after and i % self.adapt_interval == 0:
                cov = np.cov(samples[:i].T) + 1e-6 * np.eye(dim)

        rate = n_accept / n_samples
        logger.info(f"Adaptive MCMC: {n_samples} samples, acceptance rate = {rate:.3f}")

        result = MCMCResult(
            samples=samples, log_probs=log_probs,
            acceptance_rate=rate, n_samples=n_samples,
        )
        if n_burn > 0:
            result = result.burn(n_burn)
        return result
