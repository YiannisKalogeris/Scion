"""
Newton–Raphson nonlinear solver.

Solves R(u) = 0 by repeated linearisation:
    K(u_k) * δu = -R(u_k)
    u_{k+1} = u_k + δu
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse

from simulacrum.solvers.linear import LinearSolverBase, DirectSolver
from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConvergenceInfo:
    converged: bool
    iterations: int
    residual_norms: list[float] = field(default_factory=list)


class NewtonRaphson:
    """Full Newton–Raphson iteration.

    Parameters
    ----------
    residual_fn : callable
        ``R(u) -> np.ndarray`` — evaluates the residual vector.
    tangent_fn : callable
        ``K(u) -> sparse matrix`` — evaluates the tangent (Jacobian) matrix.
    linear_solver : LinearSolverBase
        Solver for the linearised system at each iteration.
    tol : float
        Residual norm tolerance.
    max_iter : int
        Maximum Newton iterations.
    """

    def __init__(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        tangent_fn: Callable[[np.ndarray], sparse.spmatrix],
        linear_solver: LinearSolverBase | None = None,
        tol: float = 1e-8,
        max_iter: int = 50,
    ):
        self.residual = residual_fn
        self.tangent = tangent_fn
        self.linear_solver = linear_solver or DirectSolver()
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, u0: np.ndarray) -> tuple[np.ndarray, ConvergenceInfo]:
        u = u0.copy()
        norms: list[float] = []

        for it in range(self.max_iter):
            R = self.residual(u)
            norm_R = np.linalg.norm(R)
            norms.append(norm_R)
            logger.info("Newton iter %d: ||R|| = %.6e", it, norm_R)

            if norm_R < self.tol:
                return u, ConvergenceInfo(converged=True, iterations=it, residual_norms=norms)

            K = self.tangent(u)
            du = self.linear_solver.solve(K, -R)
            u += du

        logger.warning("Newton–Raphson did not converge after %d iterations", self.max_iter)
        return u, ConvergenceInfo(converged=False, iterations=self.max_iter, residual_norms=norms)
