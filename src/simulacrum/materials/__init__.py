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
