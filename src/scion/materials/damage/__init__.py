r"""
Continuum damage mechanics.

Scalar isotropic damage: stiffness is lost as an internal variable ``D`` grows,
and the stress is ``(1 - D)`` times what an intact material would carry. Unlike
plasticity nothing is left behind on unloading -- the point returns to the
origin along a reduced secant -- and unlike phase-field fracture there is no
energy functional behind it. It is a phenomenological statement about
degradation, calibrated to a measured softening curve.

A model is a composition of three orthogonal choices, each with its own
registry, the same pattern as ``materials/fracture`` and
``materials/plasticity``:

    equivalent strain   what drives damage        (equivalent.py)
    evolution law       how D grows with history  (evolution.py)
    split               what part of the energy degrades (materials/splits.py)

    from scion.materials.damage import DamageModel, ExponentialSoftening

    model = DamageModel(E=30e3, nu=0.2,
                        evolution=ExponentialSoftening(1e-4, alpha=0.99,
                                                       beta=500.0),
                        equivalent="modified-von-mises",
                        equivalent_kwargs={"k": 10.0},
                        split="spectral")

It implements the same ``InelasticMaterial`` interface as ``PlasticityModel``,
so it runs through ``ElastoplasticAssembler`` and a Newton driver unchanged.

Every model here is ill-posed
-----------------------------
Softening destroys ellipticity. The localisation band has no width of its own,
collapses onto one element, and the dissipated energy goes to zero under mesh
refinement -- so a converged local-damage result is a property of the mesh, not
of the material. This is not a defect of the implementation and cannot be fixed
inside it.

:class:`~scion.materials.damage.model.GradientDamageModel` carries the
internal length that fixes it, solved by
:class:`~scion.multiphysics.gradient_damage.GradientDamageProblem` as a
two-field problem. ``LinearSoftening.from_fracture_energy`` offers the cheaper
crack-band alternative, which preserves the dissipation but not the band.

Use the local model to see the pathology, and the gradient model for anything
you intend to believe.

Deliberate omissions
--------------------
* **Anisotropic damage.** ``D`` is a scalar here, so degradation is the same in
  every direction. Real microcracking is oriented, and the tensor-valued theories
  that capture it need a different state and a different tangent.
* **Damage-plasticity coupling** (Lemaitre, Gurson). Damage driven by *plastic*
  strain through an effective-stress return map, which is a change to the
  plasticity integration rather than another entry in these registries.
* **Rate dependence.** Viscous regularisation would also restore well-posedness,
  and introduces a time scale these quasi-static models do not carry.
"""

from ..splits import (
    NoSplit,
    Spectral,
    Split,
    VolumetricDeviatoric,
    get_split,
)
from .equivalent import (
    EQUIVALENT_REGISTRY,
    EnergyRelease,
    EquivalentStrain,
    Mazars,
    ModifiedVonMises,
    Rankine,
    get_equivalent_strain,
)
from .evolution import (
    EVOLUTION_REGISTRY,
    DamageLaw,
    ExponentialSoftening,
    LinearSoftening,
    MazarsLaw,
    get_damage_law,
)
from .model import (
    DamageModel,
    DamageResult,
    DamageState,
    GradientDamageModel,
)

__all__ = [
    # equivalent strain
    "EquivalentStrain", "Mazars", "ModifiedVonMises", "Rankine",
    "EnergyRelease", "get_equivalent_strain", "EQUIVALENT_REGISTRY",
    # evolution
    "DamageLaw", "LinearSoftening", "ExponentialSoftening", "MazarsLaw",
    "get_damage_law", "EVOLUTION_REGISTRY",
    # splits, re-exported from materials/splits.py
    "Split", "NoSplit", "VolumetricDeviatoric", "Spectral", "get_split",
    # model
    "DamageModel", "GradientDamageModel", "DamageState", "DamageResult",
]
