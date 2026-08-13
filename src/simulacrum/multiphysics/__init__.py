"""
Coupled multi-field problems.

The orchestration layer: owns two or more fields on one mesh, their DOF maps and
boundary conditions, assembles both residuals, and drives the load or time
stepping.

Deliberately thin. Constitutive behaviour stays in ``materials`` so it can be
reused by a non-FEM discretization, and solution algorithms stay in ``solvers``
so ``StaggeredSolver`` serves every physics here. What lives in this package is
only the wiring the other two cannot see.
"""

from .phase_field_fracture import PhaseFieldFractureProblem
from .two_field import Field, StepResult, TwoFieldProblem

__all__ = [
    "Field",
    "StepResult",
    "TwoFieldProblem",
    "PhaseFieldFractureProblem",
]
