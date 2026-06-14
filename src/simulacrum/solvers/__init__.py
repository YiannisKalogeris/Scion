"""
Linear and nonlinear algebraic solvers.

Wraps SciPy sparse solvers and provides hooks for external libraries
(PETSc, MUMPS) and future Rust/C++ backends.
"""

from simulacrum.solvers.linear import DirectSolver, IterativeSolver
from simulacrum.solvers.nonlinear import NewtonRaphson
from .eigen import solve_eigenvalue_problem, compute_rayleigh_damping
from .dynamic import NewmarkBeta, CentralDifference
from .nonlinear import NewtonRaphson, DisplacementControl, LoadControl, AnalysisResult

__all__ = [
    "DirectSolver", "IterativeSolver", "NewtonRaphson",
    "DisplacementControl", "LoadControl", "AnalysisResult",
    "solve_eigenvalue_problem", "compute_rayleigh_damping",
    "NewmarkBeta", "CentralDifference",
]
