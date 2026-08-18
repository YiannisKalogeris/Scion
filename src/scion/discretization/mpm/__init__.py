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
from .boundary import Box, GridBC, HalfSpace, PrescribedVelocity
from .particles import ParticleSet
from .stepper import ExplicitMPMStepper
from .transfer import APIC, FLIP, PIC, Transfer, get_transfer

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
    # particle <-> grid transfer
    "Transfer",
    "PIC",
    "FLIP",
    "APIC",
    "get_transfer",
]
