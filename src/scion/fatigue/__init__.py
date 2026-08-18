r"""
Fatigue life assessment.

Post-processing, not physics: everything here consumes a load history that some
other part of the library produced and returns a life. Nothing in this package
has a tangent, a state at a quadrature point, or any influence on equilibrium.
A model that *does* -- continuum damage, where ``D`` degrades the stiffness
inside the solve -- is a constitutive model and belongs with the others.

An assessment is a composition of four orthogonal choices, each with its own
registry, following the same pattern as ``materials/fracture`` and
``materials/plasticity``:

    reduction     tensor history -> scalar history   (multiaxial.py)
    counting      scalar history -> cycles           (rainflow.py)
    life curve    cycle -> cycles to failure         (curves.py, mean_stress.py)
    accumulation  cycles to failure -> damage        (accumulation.py)

High-cycle, from an elastic analysis::

    from scion.fatigue import Basquin, FatigueAssessment

    curve = Basquin(sigma_f=1200.0, b=-0.095,
                    mean_stress="goodman", mean_stress_kwargs={"sigma_u": 600.0})
    result = FatigueAssessment(curve).evaluate(stress=history, dim=2, nu=0.3)

Low-cycle, from an elastoplastic analysis::

    from scion.fatigue import StrainLife, FatigueAssessment

    curve = StrainLife(E=210e3, sigma_f=1000.0, b=-0.1,
                       eps_f=0.5, c=-0.6, mean_stress="swt")
    result = FatigueAssessment(curve).evaluate(stress=sigma_history,
                                               strain=eps_history, dim=2)

The strain-life curve counts the strain history and reads the mean and maximum
stress of each counted loop from the stress history at that loop's own turning
points, which is why both are passed.

Conventions, all of which are factors of two waiting to happen
--------------------------------------------------------------
* Counting produces **ranges**; curves take **amplitudes**. Converted once, in
  :attr:`~scion.fatigue.rainflow.CycleCounts.amplitudes`.
* Curves return **cycles**. Basquin and Coffin-Manson are published in
  **reversals**; the factor of two is applied inside each curve.
* Code-defined curves (EN 1993-1-9 detail categories) are published in stress
  range and are halved in their constructors.

Deliberate omissions
--------------------
These raise rather than approximate. Implementing one properly is the work.

* **Critical-plane criteria** (Findley, Fatemi-Socie, Brown-Miller). They count
  cycles on each candidate plane instead of reducing the history to a scalar
  first, so they are a different calculation rather than another entry in the
  reduction registry. Without them, non-proportional loading is out of scope:
  check :func:`~scion.fatigue.multiaxial.non_proportionality` before
  trusting any result on a multiaxial history.
* **Crack growth** (Paris, NASGRO, crack closure). Everything here is
  initiation life on an uncracked geometry. A structure with a detectable crack
  is a fracture mechanics problem.
* **Sequence-dependent accumulation.** Rainflow discards the load order before
  any damage rule sees it, so no rule applied to its output can recover a
  sequence effect -- including the nonlinear ones that are usually sold as
  fixing exactly that.
* **Variable-amplitude mean-stress relaxation.** The mean stress of each loop
  is taken as counted; a real material relaxes it under plastic cycling.
"""

from .accumulation import (
    ACCUMULATION_REGISTRY,
    AccumulationRule,
    LinearAccumulation,
    NonlinearAccumulation,
    cycle_damage,
    get_accumulation_rule,
)
from .assessment import FatigueAssessment, FatigueResult
from .curves import (
    CURVE_REGISTRY,
    Basquin,
    CoffinManson,
    LifeCurve,
    StrainLife,
    StressLife,
    TwoSlopeSN,
    get_curve,
)
from .cyclic import RambergOsgood, glinka, glinka_range, neuber, neuber_range
from .mean_stress import (
    MEAN_STRESS_REGISTRY,
    STRAIN_MEAN_STRESS_REGISTRY,
    Gerber,
    Goodman,
    MeanStressCorrection,
    ModifiedMorrow,
    Morrow,
    MorrowStress,
    NoCorrection,
    NoStrainCorrection,
    SmithWatsonTopper,
    Soderberg,
    StrainLifeParams,
    StrainMeanStress,
    SWT,
    Walker,
    get_mean_stress,
    get_strain_mean_stress,
)
from .multiaxial import (
    EQUIVALENT_REGISTRY,
    equivalent_strain,
    get_equivalent,
    mandel_to_tensor,
    max_principal,
    max_shear,
    non_proportionality,
    plane_strain_sigma_33,
    signed_von_mises,
    voigt_strain_to_tensor,
    voigt_to_tensor,
    von_mises,
)
from .rainflow import CycleCounts, rainflow, rainflow_matrix, turning_points

__all__ = [
    # counting
    "CycleCounts", "rainflow", "rainflow_matrix", "turning_points",
    # curves
    "LifeCurve", "StressLife", "Basquin", "TwoSlopeSN", "CoffinManson",
    "StrainLife", "get_curve", "CURVE_REGISTRY",
    # mean stress -- stress-life
    "MeanStressCorrection", "NoCorrection", "Goodman", "Soderberg", "Gerber",
    "MorrowStress", "Walker", "SWT", "get_mean_stress", "MEAN_STRESS_REGISTRY",
    # mean stress -- strain-life
    "StrainMeanStress", "NoStrainCorrection", "Morrow", "ModifiedMorrow",
    "SmithWatsonTopper", "StrainLifeParams", "get_strain_mean_stress",
    "STRAIN_MEAN_STRESS_REGISTRY",
    # accumulation
    "AccumulationRule", "LinearAccumulation", "NonlinearAccumulation",
    "cycle_damage", "get_accumulation_rule", "ACCUMULATION_REGISTRY",
    # multiaxial reduction
    "voigt_to_tensor", "voigt_strain_to_tensor", "mandel_to_tensor",
    "plane_strain_sigma_33", "von_mises", "signed_von_mises", "max_principal",
    "max_shear", "equivalent_strain", "non_proportionality", "get_equivalent",
    "EQUIVALENT_REGISTRY",
    # notch plasticity
    "RambergOsgood", "neuber", "neuber_range", "glinka", "glinka_range",
    # driver
    "FatigueAssessment", "FatigueResult",
]
