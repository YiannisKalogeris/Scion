"""
Bayesian Optimisation using Gaussian Process surrogates.

Efficiently optimises expensive black-box functions (e.g. PDE solves)
by building a GP model and using acquisition functions to decide
where to evaluate next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from simulacrum.uq.gp import GaussianProcessEmulator
from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcessEmulator,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected Improvement acquisition function."""
    pred = gp.predict(X)
    improvement = f_best - pred.mean - xi
    Z = improvement / (pred.std + 1e-12)
    ei = improvement * norm.cdf(Z) + pred.std * norm.pdf(Z)
    ei[pred.std < 1e-12] = 0.0
    return ei


def upper_confidence_bound(
    X: np.ndarray,
    gp: GaussianProcessEmulator,
    beta: float = 2.0,
) -> np.ndarray:
    """Upper Confidence Bound (negated for minimisation)."""
    pred = gp.predict(X)
    return -(pred.mean - beta * pred.std)


@dataclass
class BOResult:
    """Result of Bayesian optimisation."""
    x_best: np.ndarray
    f_best: float
    X_evaluated: np.ndarray
    y_evaluated: np.ndarray
    n_iterations: int


class BayesianOptimizer:
    """Sequential Bayesian optimisation.

    Parameters
    ----------
    bounds : (dim, 2) array
        Parameter bounds [[low, high], ...].
    n_initial : int
        Number of initial random evaluations.
    acquisition : str
        Acquisition function: ``"ei"`` (Expected Improvement) or
        ``"ucb"`` (Upper Confidence Bound).
    gp_kwargs : dict
        Keyword arguments for the GP emulator.
    seed : int or None
    """

    def __init__(
        self,
        bounds: np.ndarray,
        n_initial: int = 5,
        acquisition: str = "ei",
        gp_kwargs: dict | None = None,
        seed: int | None = None,
    ):
        self.bounds = np.asarray(bounds)
        self.dim = self.bounds.shape[0]
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.gp_kwargs = gp_kwargs or {}
        self.rng = np.random.default_rng(seed)

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        n_iterations: int = 50,
    ) -> BOResult:
        """Run Bayesian optimisation.

        Parameters
        ----------
        objective : expensive function to minimise
        n_iterations : total budget (including initial samples)
        """
        # Initial sampling (Latin Hypercube)
        from simulacrum.uq.sampling import latin_hypercube
        X = latin_hypercube(self.n_initial, self.dim, self.bounds, seed=42)
        y = np.array([objective(x) for x in X])

        for it in range(self.n_initial, n_iterations):
            # Fit GP
            gp = GaussianProcessEmulator(**self.gp_kwargs)
            gp.fit(X, y)

            f_best = y.min()

            # Optimise acquisition function
            x_next = self._optimize_acquisition(gp, f_best)

            # Evaluate
            y_next = objective(x_next)
            logger.info(
                f"BO iter {it}: f = {y_next:.6e}, best = {min(f_best, y_next):.6e}"
            )

            X = np.vstack([X, x_next.reshape(1, -1)])
            y = np.append(y, y_next)

        best_idx = np.argmin(y)
        return BOResult(
            x_best=X[best_idx],
            f_best=y[best_idx],
            X_evaluated=X,
            y_evaluated=y,
            n_iterations=n_iterations,
        )

    def _optimize_acquisition(
        self, gp: GaussianProcessEmulator, f_best: float
    ) -> np.ndarray:
        """Find the point that maximises the acquisition function."""

        def neg_acq(x: np.ndarray) -> float:
            x = x.reshape(1, -1)
            if self.acquisition == "ei":
                return -float(expected_improvement(x, gp, f_best))
            elif self.acquisition == "ucb":
                return float(upper_confidence_bound(x, gp))
            raise ValueError(f"Unknown acquisition: {self.acquisition}")

        # Multi-start optimisation
        best_x, best_val = None, np.inf
        n_restarts = 10 * self.dim
        candidates = self.rng.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, self.dim)
        )

        for x0 in candidates:
            result = minimize(
                neg_acq, x0, bounds=self.bounds.tolist(), method="L-BFGS-B"
            )
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x

        return best_x
