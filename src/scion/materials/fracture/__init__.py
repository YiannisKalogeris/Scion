"""
Phase-field fracture.

A model is a composition of four orthogonal choices, each with its own registry:

    from scion.materials.fracture import PhaseFieldModel

    model = PhaseFieldModel(E=210e3, nu=0.3, Gc=2.7e-3, ell=0.02,
                            split="spectral",        # none | amor | spectral
                            degradation="quadratic", # quadratic | cubic | rational
                            dissipation="AT2",       # AT1 | AT2
                            coupling="hybrid")       # isotropic | anisotropic | hybrid

Adding a variant means adding a class to one module and one registry entry;
nothing else changes.

The split axis lives one level up, in :mod:`scion.materials.splits`: it is
a decomposition of linear elastic energy with no fracture content, and continuum
damage needs the same one. It is re-exported here, so the names below are
unchanged.
"""

from .coupling import (
    Anisotropic,
    Coupling,
    Hybrid,
    Isotropic,
    get_coupling,
)
from .degradation import Cubic, Degradation, Quadratic, Rational, get_degradation
from .dissipation import AT1, AT2, Dissipation, get_dissipation
from .model import PhaseFieldModel
from ..splits import (
    NoSplit,
    Spectral,
    Split,
    VolumetricDeviatoric,
    get_split,
    voigt_stress,
    voigt_tangent,
)

__all__ = [
    "PhaseFieldModel",
    # splits
    "Split", "NoSplit", "VolumetricDeviatoric", "Spectral", "get_split",
    "voigt_stress", "voigt_tangent",
    # degradation
    "Degradation", "Quadratic", "Cubic", "Rational", "get_degradation",
    # dissipation
    "Dissipation", "AT1", "AT2", "get_dissipation",
    # coupling
    "Coupling", "Isotropic", "Anisotropic", "Hybrid", "get_coupling",
]
