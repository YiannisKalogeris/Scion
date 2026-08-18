"""
Linear and nonlinear algebraic solvers.

Algorithms that know no physics. ``StaggeredSolver`` takes three callables,
``check_tangent`` takes two, ``pcg`` takes an operator and a preconditioner --
anything physics-aware belongs elsewhere.

Krylov and preconditioning
--------------------------
``krylov``          CG and PCG for SPD systems
``nonsymmetric``    GMRES(m), FGMRES, BiCG, BiCGSTAB for everything else
``preconditioners`` Jacobi, IC(0), ILU, AMG, behind one ``apply`` interface
``amg``             the smoothed-aggregation hierarchy and its V-cycle
``schwarz``         algebraic overlapping domain decomposition (ASM, RAS)

The distinguishing property, measured on 2-D Poisson as the mesh is refined:
unpreconditioned CG grows like ``1/h`` (26, 52, 102, 148 iterations), IC(0)
grows more slowly (12, 20, 35, 50), and AMG does not grow (10, 11, 14, 14).
Nothing else here has that. The two-level diffusion-filtered coarse space that
these serve as baselines for lives in ``projects/diffusion_preconditioner``.
"""

from scion.solvers.amg import (
    AMGHierarchy,
    AMGLevel,
    aggregate,
    nodal_strength,
    rigid_body_translations,
    smoothed_aggregation,
    strength_of_connection,
)
from scion.solvers.bound_constrained import (
    projected_gauss_seidel,
    solve_bounded,
)
from scion.solvers.dynamic import CentralDifference, NewmarkBeta
from scion.solvers.eigen import (
    compute_rayleigh_damping,
    solve_eigenvalue_problem,
)
from scion.solvers.krylov import (
    KrylovBreakdown,
    KrylovResult,
    NotPositiveDefinite,
    conjugate_gradient,
    pcg,
)
from scion.solvers.linear import DirectSolver, IterativeSolver
from scion.solvers.nonsymmetric import bicg, bicgstab, fgmres, gmres
from scion.solvers.nonlinear import (
    AnalysisResult,
    DisplacementControl,
    LoadControl,
    NewtonRaphson,
)
from scion.solvers.preconditioners import (
    PRECONDITIONER_REGISTRY,
    AdditiveSchwarz,
    PYAMG_AVAILABLE,
    AlgebraicMultigrid,
    Identity,
    IncompleteCholesky,
    IncompleteLU,
    Jacobi,
    Preconditioner,
    get_preconditioner,
)
from scion.solvers.schwarz import (
    PYMETIS_AVAILABLE,
    SchwarzOperator,
    build_schwarz,
    grow_overlap,
    partition_graph,
)
from scion.solvers.staggered import StaggeredSolver, normalized_slope
from scion.solvers.tangent_check import check_tangent, numerical_tangent

#: Name -> Krylov routine. Lives here rather than in either module because it
#: spans both, and importing across them would be circular. Every one takes
#: ``(A, b, M=None, x0=None, tol, max_iter)`` and returns a ``KrylovResult``.
KRYLOV_SOLVERS = {
    "cg": conjugate_gradient,
    "pcg": pcg,
    "gmres": gmres,
    "fgmres": fgmres,
    "bicg": bicg,
    "bicgstab": bicgstab,
}


def get_krylov_solver(name: str):
    """Look up a Krylov routine by name.

    ``cg``/``pcg`` require an SPD operator *and* an SPD preconditioner; the rest
    do not. See :mod:`scion.solvers.nonsymmetric` for which to reach for.
    """
    key = name.lower().replace("_", "-")
    if key not in KRYLOV_SOLVERS:
        raise NotImplementedError(
            f"Unknown Krylov solver '{name}'. Available: "
            f"{sorted(KRYLOV_SOLVERS)}. MINRES and QMR are not implemented.")
    return KRYLOV_SOLVERS[key]


__all__ = [
    # linear
    "DirectSolver",
    "IterativeSolver",
    # krylov
    "conjugate_gradient",
    "pcg",
    "gmres",
    "fgmres",
    "bicg",
    "bicgstab",
    "KrylovResult",
    "KrylovBreakdown",
    "NotPositiveDefinite",
    "KRYLOV_SOLVERS",
    "get_krylov_solver",
    # preconditioners
    "Preconditioner",
    "Identity",
    "Jacobi",
    "IncompleteCholesky",
    "IncompleteLU",
    "AlgebraicMultigrid",
    "AdditiveSchwarz",
    "PRECONDITIONER_REGISTRY",
    "get_preconditioner",
    "PYAMG_AVAILABLE",
    # amg internals, for building a hierarchy directly
    "smoothed_aggregation",
    "AMGHierarchy",
    "AMGLevel",
    "strength_of_connection",
    "nodal_strength",
    "rigid_body_translations",
    "aggregate",
    # schwarz
    "build_schwarz",
    "SchwarzOperator",
    "partition_graph",
    "grow_overlap",
    "PYMETIS_AVAILABLE",
    # nonlinear
    "NewtonRaphson",
    "DisplacementControl",
    "LoadControl",
    "AnalysisResult",
    # eigen and dynamics
    "solve_eigenvalue_problem",
    "compute_rayleigh_damping",
    "NewmarkBeta",
    "CentralDifference",
    # misc
    "projected_gauss_seidel",
    "solve_bounded",
    "StaggeredSolver",
    "normalized_slope",
    "check_tangent",
    "numerical_tangent",
]
