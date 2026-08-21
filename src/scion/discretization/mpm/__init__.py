"""Material Point Method discretisation.

Curated public API for the subpackage: callers write

    from scion.discretization.mpm import ParticleSet, CartesianGrid

rather than reaching into the individual modules.

Deliberately eager, because every name here is NumPy-only and cheap to import.
When a Warp backend is added, do NOT re-export it from this module -- importing
it would pull in `warp` on machines that do not have it. Expose it lazily via a
module-level __getattr__ instead.
"""

from .background_grid import CartesianGrid, StencilData
from .basis import (
    Basis,
    CubicBSpline,
    LinearHat,
    QuadraticBSpline,
    get_basis,
    uGIMP,
)
from .cpdi import CPDIGrid, corner_offsets
from .ghost import ghost_faces, ghost_penalty, populated_cells
from .implicit import (
    GridDirichlet,
    ImplicitMPMStepper,
    InvertedParticle,
    LoadStepResult,
)
from .boundary import Box, GridBC, HalfSpace, PrescribedVelocity
from .micropolar import (
    MicropolarMPMStepper,
    attach_micropolar_state,
)
from .particles import ParticleSet
from .stepper import ExplicitMPMStepper
from .total_lagrangian import TotalLagrangianStepper
from .transfer import APIC, FLIP, MLSMPM, PIC, PolyPIC, Transfer, get_transfer

__all__ = [
    # grid
    "CartesianGrid",
    "StencilData",
    # basis functions
    "Basis",
    "LinearHat",
    "QuadraticBSpline",
    "CubicBSpline",
    "uGIMP",
    "get_basis",
    # boundary conditions
    "GridBC",
    "HalfSpace",
    "Box",
    "PrescribedVelocity",
    # particle state
    "ParticleSet",
    # time integration
    "ExplicitMPMStepper",
    "TotalLagrangianStepper",
    "ImplicitMPMStepper",
    "MicropolarMPMStepper",
    "attach_micropolar_state",
    "GridDirichlet",
    "LoadStepResult",
    "InvertedParticle",
    "ghost_penalty",
    "ghost_faces",
    "populated_cells",
    "PolyPIC",
    "MLSMPM",
    "CPDIGrid",
    # particle <-> grid transfer
    "Transfer",
    "PIC",
    "FLIP",
    "APIC",
    "get_transfer",
]
