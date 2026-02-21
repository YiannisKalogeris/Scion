"""
Constitutive / material models.

All models inherit from ``Material`` and implement the tangent-modulus
interface so they plug into any discretization scheme.
"""

from simulacrum.materials.base import Material
from simulacrum.materials.linear_elastic import LinearElastic, PlaneStress, PlaneStrain
from simulacrum.materials.thermal import IsotropicThermal

__all__ = [
    "Material",
    "LinearElastic",
    "PlaneStress",
    "PlaneStrain",
    "IsotropicThermal",
]
