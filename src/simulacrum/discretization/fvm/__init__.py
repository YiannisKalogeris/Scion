"""
Finite volume: conservation as the organising principle.

The method integrates a conservation law over a cell rather than differentiating
it at a point, so what it discretises is a *balance* -- what accumulates in a
cell equals what crosses its faces. Because each face is shared and enters its
two cells with opposite signs, the total is conserved to round-off on any mesh,
whatever the scheme. Nothing else here has that property; FDM and FEM conserve
only to truncation error.

Modules
-------
``flux``       scalar convective face flux: central, upwind, rusanov
``limiters``   slope limiters for MUSCL reconstruction, and the Sweby test
``assembler``  ``FVMAssembler``: scalar advection-diffusion, steady or explicit
``riemann``    Euler Riemann solvers -- rusanov, HLL, HLLC -- plus the exact one
``euler``      ``EulerSolver1D`` and the named shock tubes

The two halves share the limiters and nothing else. A scalar face flux takes
two states and a velocity; a Riemann solver takes two states and works out the
wave speeds itself, because for a system there is no single direction that
"upwind" could mean. They are deliberately separate registries.

Gotcha shared by everything here: ``grid_shape`` counts **cells**, not points as
it does in :mod:`simulacrum.discretization.fdm`, so ``h = L/n`` rather than
``L/(n+1)``.
"""

from simulacrum.discretization.fvm.assembler import FVMAssembler
from simulacrum.discretization.fvm.euler import (
    SHOCK_TUBES,
    SOD,
    EulerEquations,
    EulerResult,
    EulerSolver1D,
    ShockTube,
    get_shock_tube,
)
from simulacrum.discretization.fvm.flux import (
    FLUX_REGISTRY,
    CentralFlux,
    FluxScheme,
    RusanovFlux,
    UpwindFlux,
    get_flux,
    peclet_limit,
)
from simulacrum.discretization.fvm.limiters import (
    LIMITER_REGISTRY,
    FirstOrder,
    Limiter,
    MinMod,
    MonotonizedCentral,
    NoLimiter,
    Superbee,
    VanAlbada,
    VanLeer,
    get_limiter,
    slope_ratio,
)
from simulacrum.discretization.fvm.riemann import (
    RIEMANN_REGISTRY,
    HLLCSolver,
    HLLSolver,
    RiemannSolver,
    RusanovSolver,
    VacuumError,
    exact_riemann,
    get_riemann_solver,
    star_pressure,
)

__all__ = [
    "FVMAssembler",
    "FluxScheme",
    "CentralFlux",
    "UpwindFlux",
    "RusanovFlux",
    "FLUX_REGISTRY",
    "get_flux",
    "peclet_limit",
    "Limiter",
    "NoLimiter",
    "FirstOrder",
    "MinMod",
    "VanLeer",
    "VanAlbada",
    "MonotonizedCentral",
    "Superbee",
    "LIMITER_REGISTRY",
    "get_limiter",
    "slope_ratio",
    "RiemannSolver",
    "RusanovSolver",
    "HLLSolver",
    "HLLCSolver",
    "RIEMANN_REGISTRY",
    "get_riemann_solver",
    "exact_riemann",
    "star_pressure",
    "VacuumError",
    "EulerEquations",
    "EulerSolver1D",
    "EulerResult",
    "ShockTube",
    "SOD",
    "SHOCK_TUBES",
    "get_shock_tube",
]
