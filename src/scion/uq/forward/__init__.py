"""
Forward UQ: pushing a distribution through the model.

    distributions        the inputs, and the transform every design goes through
    estimators           propagate() over crude / LHS / Sobol / Halton /
                         antithetic designs
    variance_reduction   control variates, importance sampling, stratification
    reliability          failure probability: crude, FORM, importance sampling,
                         subset simulation
    sensitivity          Sobol first-order and total indices
    reference            Ishigami and the linear limit state, with the exact
                         answers every method here is tested against

Three things worth knowing before choosing a method
---------------------------------------------------
**The rate is fixed.** Monte Carlo error falls as ``1/sqrt(N)`` whatever the
dimension, and no rearrangement of the average improves it. The lever is the
variance, not the estimator.

**Rare events are a different problem.** Estimating ``P_f`` by counting needs
``N ~ 100 / P_f`` for 10% accuracy, so ``P_f = 1e-6`` is ``1e8`` model
evaluations. That is what ``reliability`` exists for, and the failure of the
naive approach is silent -- too few samples returns zero failures, which reads
as a probability of zero.

**A small first-order Sobol index licenses nothing.** Only a small *total* index
means an input can be ignored. The Ishigami reference exists to make that
concrete: its third input has ``S_3 = 0`` exactly and ``S_T3 = 0.244``.
"""

from scion.uq.forward.distributions import (
    DISTRIBUTION_REGISTRY,
    Beta,
    Exponential,
    Gumbel,
    JointIndependent,
    LogNormal,
    Normal,
    RandomVariable,
    Uniform,
    get_distribution,
)
from scion.uq.forward.estimators import (
    DESIGNS,
    MonteCarloResult,
    propagate,
    unit_design,
)
from scion.uq.forward.reference import (
    Ishigami,
    LinearLimitState,
    QuadraticLimitState,
)
from scion.uq.forward.reliability import (
    ReliabilityResult,
    form,
    importance_sampling_probability,
    monte_carlo_probability,
    required_samples,
    subset_simulation,
)
from scion.uq.forward.sensitivity import SobolResult, sobol_indices
from scion.uq.forward.variance_reduction import (
    ControlVariateResult,
    ImportanceSamplingResult,
    control_variates,
    importance_sampling,
    stratified,
)

__all__ = [
    # inputs
    "RandomVariable",
    "Normal",
    "LogNormal",
    "Uniform",
    "Exponential",
    "Gumbel",
    "Beta",
    "JointIndependent",
    "DISTRIBUTION_REGISTRY",
    "get_distribution",
    # propagation
    "propagate",
    "MonteCarloResult",
    "unit_design",
    "DESIGNS",
    # variance reduction
    "control_variates",
    "ControlVariateResult",
    "importance_sampling",
    "ImportanceSamplingResult",
    "stratified",
    # reliability
    "ReliabilityResult",
    "monte_carlo_probability",
    "form",
    "importance_sampling_probability",
    "subset_simulation",
    "required_samples",
    # sensitivity
    "sobol_indices",
    "SobolResult",
    # references
    "Ishigami",
    "LinearLimitState",
    "QuadraticLimitState",
]
