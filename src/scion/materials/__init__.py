"""
Constitutive / material models.

All models inherit from ``Material`` and implement the tangent-modulus
interface so they plug into any discretization scheme.
"""

from scion.materials.base import Material
from scion.materials.linear_elastic import LinearElastic3D, PlaneStress, PlaneStrain
from scion.materials.thermal import IsotropicThermal
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

from scion.materials.eos import (
    EOS_REGISTRY, EquationOfState, LinearEOS, MieGruneisen, TaitEOS, get_eos,
)
from scion.materials.high_rate import HighRateState, JohnsonCookModel
from scion.materials.hencky import (
    ELASTIC_LAWS, RETURN_MAPS, DruckerPragerSand, FixedCorotated, HenckyElastic,
    PrincipalMaterial, PrincipalPlasticState, PrincipalPlasticity, ReturnMap,
    VonMisesHencky,
)
from scion.materials.fluid import FluidState, WeaklyCompressibleFluid
from scion.materials.eshelby import (
    eshelby_stress, eshelby_stress_small_strain, j_integral,
    kinetic_energy_density, material_force, pseudomomentum,
    total_pseudomomentum,
)

__all__ = list(globals().get("__all__", [])) + [
    "EquationOfState", "LinearEOS", "MieGruneisen", "TaitEOS",
    "EOS_REGISTRY", "get_eos",
    "HighRateState", "JohnsonCookModel",
    # principal-stretch family
    "PrincipalMaterial", "HenckyElastic", "FixedCorotated",
    "ReturnMap", "DruckerPragerSand", "VonMisesHencky",
    "PrincipalPlasticity", "PrincipalPlasticState",
    "ELASTIC_LAWS", "RETURN_MAPS",
    # fluid
    "WeaklyCompressibleFluid", "FluidState",
    # material-manifold quantities
    "eshelby_stress", "eshelby_stress_small_strain", "j_integral",
    "pseudomomentum", "total_pseudomomentum",
    "kinetic_energy_density", "material_force",
]
