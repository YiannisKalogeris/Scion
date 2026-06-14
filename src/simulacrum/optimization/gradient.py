"""
Gradient-based optimization wrappers around SciPy.

Provides a clean interface for PDE-constrained optimisation problems:
minimise J(θ) subject to PDE constraints, where θ are design parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize, OptimizeResult


@dataclass
class OptimizationResult:
    """Result of an optimisation run."""
    x: np.ndarray                    # optimal parameters
    fun: float                       # objective at optimum
    n_iter: int                      # number of iterations
    n_fev: int                       # number of function evaluations
    converged: bool
    history: list[float] = field(default_factory=list)


class ScipyOptimizer:
    """Wrapper around ``scipy.optimize.minimize``.

    Parameters
    ----------
    method : str
        SciPy method name: ``"L-BFGS-B"``, ``"SLSQP"``, ``"trust-constr"``, etc.
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    bounds : list of (min, max) tuples or None
        Parameter bounds.
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        tol: float = 1e-8,
        maxiter: int = 500,
        bounds: list[tuple[float, float]] | None = None,
    ):
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.bounds = bounds

    def minimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Callable[[np.ndarray], np.ndarray] | str | None = None,
        constraints: list | None = None,
    ) -> OptimizationResult:
        """Run the optimisation.

        Parameters
        ----------
        objective : ``f(x) -> scalar``
        x0 : initial guess
        jac : gradient function, ``"2-point"`` for finite differences, or None
        constraints : SciPy constraint dicts (for SLSQP, trust-constr)
        """
        history = []

        def callback(xk):
            history.append(objective(xk))

        result: OptimizeResult = minimize(
            objective,
            x0,
            method=self.method,
            jac=jac,
            bounds=self.bounds,
            constraints=constraints,
            tol=self.tol,
            options={"maxiter": self.maxiter},
            callback=callback,
        )

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            n_iter=result.nit,
            n_fev=result.nfev,
            converged=result.success,
            history=history,
        )
