"""
Linear solver wrappers.

Provides a uniform interface over SciPy direct/iterative solvers and
optional external backends (PETSc, MUMPS, Rust).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from scion.core.logging import get_logger

logger = get_logger(__name__)


class LinearSolverBase(ABC):
    """Interface for linear solvers: solve A x = b."""

    @abstractmethod
    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        ...


class DirectSolver(LinearSolverBase):
    """Sparse direct solver (SciPy SuperLU / UMFPACK)."""

    def __init__(self, use_umfpack: bool = True):
        self.use_umfpack = use_umfpack

    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        logger.info("Direct solve: n = %d, nnz = %d", A.shape[0], A.nnz)
        return spla.spsolve(A.tocsc(), b, use_umfpack=self.use_umfpack)


class IterativeSolver(LinearSolverBase):
    """Preconditioned Krylov solver.

    Parameters
    ----------
    method : str
        Krylov method name (``"cg"``, ``"gmres"``, ``"bicgstab"``, ``"minres"``).
    tol : float
        Convergence tolerance.
    maxiter : int
        Maximum iterations.
    preconditioner : str | None
        Built-in preconditioner: ``"jacobi"``, ``"ilu"``, or ``None``.
    """

    _METHODS = {
        "cg": spla.cg,
        "gmres": spla.gmres,
        "bicgstab": spla.bicgstab,
        "minres": spla.minres,
    }

    def __init__(
        self,
        method: str = "cg",
        tol: float = 1e-10,
        maxiter: int = 1000,
        preconditioner: str | None = None,
    ):
        if method not in self._METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {list(self._METHODS)}")
        self.method = method
        self.tol = tol
        self.maxiter = maxiter
        self.preconditioner = preconditioner

    def _build_preconditioner(self, A: sparse.spmatrix) -> spla.LinearOperator | None:
        if self.preconditioner is None:
            return None
        elif self.preconditioner == "jacobi":
            diag_inv = 1.0 / A.diagonal()
            return spla.LinearOperator(A.shape, matvec=lambda x: diag_inv * x)
        elif self.preconditioner == "ilu":
            ilu = spla.spilu(A.tocsc())
            return spla.LinearOperator(A.shape, matvec=ilu.solve)
        else:
            raise ValueError(f"Unknown preconditioner '{self.preconditioner}'")

    def solve(self, A: sparse.spmatrix, b: np.ndarray) -> np.ndarray:
        M = self._build_preconditioner(A)
        logger.info(
            "Iterative solve (%s): n = %d, nnz = %d, precond = %s",
            self.method, A.shape[0], A.nnz, self.preconditioner,
        )
        solver_fn = self._METHODS[self.method]
        x, info = solver_fn(A, b, tol=self.tol, maxiter=self.maxiter, M=M)
        if info > 0:
            logger.warning("%s did not converge (info=%d)", self.method, info)
        elif info < 0:
            logger.error("%s encountered an error (info=%d)", self.method, info)
        return x
