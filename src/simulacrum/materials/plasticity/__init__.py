"""
Rate-independent plasticity.

Like the fracture subpackage, a model is a composition of orthogonal choices,
each with its own registry:

    yield criterion   which stress states are admissible
    hardening law     how the elastic domain evolves with plastic strain
    flow rule         the direction of plastic flow (associative or not)

The three axes are complete and the closest-point return map that integrates
them is in place. Both are usable and testable
on their own: a criterion is a statement about stress, a hardening law a
statement about the evolution of the elastic domain, and neither refers to a
mesh, a solver or a time step.

    from simulacrum.materials.plasticity import get_criterion, get_isotropic

    j2   = get_criterion("j2")
    dp   = DruckerPrager.from_mohr_coulomb(phi=30.0, match="plane-strain")
    hard = get_isotropic("voce", kappa_0=250.0, kappa_inf=400.0, delta=20.0)
    kin  = get_kinematic("armstrong-frederick", C=20000.0, gamma=100.0)
    flow = NonAssociative.drucker_prager(psi=10.0)   # psi < phi: less dilatant

    from simulacrum.materials.plasticity import (PlasticState, return_map,
                                                 isotropic_stiffness_mandel)
    C_el   = isotropic_stiffness_mandel(E=210e3, nu=0.3)
    state  = PlasticState.zeros(n_quadrature_points)
    result = return_map(eps_mandel, state, j2, hard, C_el, flow, kin)
    # result.sigma, result.state, result.C_alg

All criteria work on full 3-D stress tensors ``(n, 3, 3)`` regardless of
element dimension -- in plane strain ``sigma_33`` is nonzero and enters ``J2``,
so reducing to the in-plane components would give the wrong surface.
"""

from ..linear_elastic import isotropic_stiffness_mandel
from ..notation import (
    from_mandel,
    mandel_to_voigt_stress,
    mandel_to_voigt_tangent,
    to_mandel,
    to_mandel4,
)
from .model import InelasticMaterial, PlasticityModel
from .return_mapping import PlasticState, ReturnMapResult, return_map
from .flow_rules import (
    FLOW_REGISTRY,
    Associative,
    FlowRule,
    NonAssociative,
    get_flow_rule,
)
from .hardening import (
    ISOTROPIC_REGISTRY,
    KINEMATIC_REGISTRY,
    ArmstrongFrederick,
    IsotropicHardening,
    KinematicHardening,
    LinearIsotropic,
    LinearKinematic,
    Ludwik,
    NoKinematic,
    PerfectPlasticity,
    Swift,
    Voce,
    get_isotropic,
    get_kinematic,
)
from .yield_criteria import (
    CRITERION_REGISTRY,
    DruckerPrager,
    J2_invariant,
    P_DEV,
    VonMises,
    YieldCriterion,
    deviator,
    get_criterion,
    pressure,
    von_mises_stress,
)

__all__ = [
    "YieldCriterion",
    "VonMises",
    "DruckerPrager",
    "get_criterion",
    "CRITERION_REGISTRY",
    "deviator",
    "pressure",
    "J2_invariant",
    "von_mises_stress",
    "P_DEV",
    # hardening -- isotropic
    "IsotropicHardening",
    "PerfectPlasticity",
    "LinearIsotropic",
    "Voce",
    "Swift",
    "Ludwik",
    "get_isotropic",
    "ISOTROPIC_REGISTRY",
    # hardening -- kinematic
    "KinematicHardening",
    "NoKinematic",
    "LinearKinematic",
    "ArmstrongFrederick",
    "get_kinematic",
    "KINEMATIC_REGISTRY",
    # flow rules
    "FlowRule",
    "Associative",
    "NonAssociative",
    "get_flow_rule",
    "FLOW_REGISTRY",
    # notation
    "to_mandel",
    "from_mandel",
    "to_mandel4",
    "mandel_to_voigt_stress",
    "mandel_to_voigt_tangent",
    "isotropic_stiffness_mandel",
    # return mapping
    "PlasticState",
    "ReturnMapResult",
    "return_map",
    # model
    "InelasticMaterial",
    "PlasticityModel",
]
