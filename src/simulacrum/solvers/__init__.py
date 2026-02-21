"""
Linear and nonlinear algebraic solvers.

Wraps SciPy sparse solvers and provides hooks for external libraries
(PETSc, MUMPS) and future Rust/C++ backends.
"""

from simulacrum.solvers.linear import DirectSolver, IterativeSolver
from simulacrum.solvers.nonlinear import NewtonRaphson

__all__ = ["DirectSolver", "IterativeSolver", "NewtonRaphson"]
