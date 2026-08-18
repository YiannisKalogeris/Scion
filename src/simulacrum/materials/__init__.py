"""
Constitutive / material models.

All models inherit from ``Material`` and implement the tangent-modulus
interface so they plug into any discretization scheme.
"""

from simulacrum.materials.base import Material
from simulacrum.materials.linear_elastic import LinearElastic3D, PlaneStress, PlaneStrain
from simulacrum.materials.thermal import IsotropicThermal
from .hyperelastic import NeoHookean
from .geometric_nonlinear import StVenantKirchhoff

__all__ = [
    "Material",
    "LinearElastic3D",
    "PlaneStress",
    "PlaneStrain",
    "IsotropicThermal",
    "NeoHookean",
    "StVenantKirchhoff",
]

from simulacrum.materials.eos import (
    EOS_REGISTRY, EquationOfState, LinearEOS, MieGruneisen, get_eos,
)
from simulacrum.materials.high_rate import HighRateState, JohnsonCookModel

__all__ = list(globals().get("__all__", [])) + [
    "EquationOfState", "LinearEOS", "MieGruneisen", "EOS_REGISTRY", "get_eos",
    "HighRateState", "JohnsonCookModel",
]
