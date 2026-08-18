"""
Derivative-free optimisation methods.

Suitable when gradients are unavailable or the objective is noisy
(e.g. stochastic simulations, experimental data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from scion.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvolutionResult:
    """Result of an evolutionary optimisation."""
    x: np.ndarray
    fun: float
    n_generations: int
    n_fev: int
    converged: bool
    history: list[float] = field(default_factory=list)


class CMAES:
    """Covariance Matrix Adaptation Evolution Strategy.

    A robust derivative-free optimizer for continuous optimisation.
    Well-suited for ill-conditioned, non-separable problems with
    moderate dimensionality (up to ~100 parameters).

    Parameters
    ----------
    sigma0 : float
        Initial step size.
    population_size : int or None
        Population size (default: 4 + floor(3 * ln(dim))).
    maxiter : int
        Maximum generations.
    tol : float
        Convergence tolerance on function value spread.
    bounds : (dim, 2) array or None
    seed : int or None
    """

    def __init__(
        self,
        sigma0: float = 0.5,
        population_size: int | None = None,
        maxiter: int = 1000,
        tol: float = 1e-10,
        bounds: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.sigma0 = sigma0
        self._pop_size = population_size
        self.maxiter = maxiter
        self.tol = tol
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
    ) -> EvolutionResult:
        """Run CMA-ES.

        Parameters
        ----------
        objective : ``f(x) -> scalar``
        x0 : (dim,) initial mean
        """
        dim = len(x0)
        lam = self._pop_size or 4 + int(3 * np.log(dim))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        mean = x0.copy()
        sigma = self.sigma0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        history = []
        n_fev = 0

        for gen in range(self.maxiter):
            # Sample population
            eig_vals, eig_vecs = np.linalg.eigh(C)
            eig_vals = np.maximum(eig_vals, 1e-20)
            sqrt_C = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T

            zs = self.rng.standard_normal((lam, dim))
            xs = mean + sigma * (zs @ sqrt_C.T)

            # Clip to bounds
            if self.bounds is not None:
                xs = np.clip(xs, self.bounds[:, 0], self.bounds[:, 1])

            # Evaluate
            fitnesses = np.array([objective(x) for x in xs])
            n_fev += lam

            # Sort by fitness
            order = np.argsort(fitnesses)
            best_fit = fitnesses[order[0]]
            history.append(best_fit)

            # Recombination
            selected = xs[order[:mu]]
            old_mean = mean.copy()
            mean = weights @ selected

            # Update evolution paths
            inv_sqrt_C = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
            p_sigma = (1 - c_sigma) * p_sigma + \
                np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (mean - old_mean) / sigma

            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + \
                h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            # Covariance update
            artmp = (selected - old_mean) / sigma
            C = (1 - c1 - c_mu) * C + \
                c1 * np.outer(p_c, p_c) + \
                c_mu * (artmp.T * weights) @ artmp

            # Step-size update
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Convergence check
            if len(history) > 10 and np.std(history[-10:]) < self.tol:
                logger.info(f"CMA-ES converged at generation {gen}")
                return EvolutionResult(
                    x=xs[order[0]], fun=best_fit,
                    n_generations=gen, n_fev=n_fev,
                    converged=True, history=history,
                )

        logger.warning(f"CMA-ES did not converge after {self.maxiter} generations")
        return EvolutionResult(
            x=xs[order[0]], fun=best_fit,
            n_generations=self.maxiter, n_fev=n_fev,
            converged=False, history=history,
        )
