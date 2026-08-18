"""
Uncertainty quantification.

Two directions, and they are different problems rather than two settings of one.

**Forward** -- push a distribution over inputs through the model and describe
the distribution that comes out. Sampling and surrogates: ``sampling``, ``pce``,
``gp``.

**Inverse** -- start from measurements and recover the distribution over inputs
that could have produced them. Bayesian inference: ``bayesian``.

Submodules
----------
forward     input distributions, sampling designs, variance reduction,
            reliability (FORM, importance sampling, subset simulation) and
            Sobol sensitivity indices
bayesian    MCMC, adaptive Metropolis, TMCMC, HMC, and their diagnostics
sampling    the bare unit-hypercube designs, kept for existing callers
pce         polynomial chaos expansions
gp          Gaussian process emulation

The forward model is a callable
-------------------------------
Nothing here knows what the model is. ``GaussianLikelihood`` takes any
``theta -> prediction``, so an analytic function, a surrogate from ``gp``, and a
full nonlinear FE solve all enter identically, and the inference machinery stays
as physics-free as :mod:`simulacrum.solvers`.
"""

from simulacrum.uq.bayesian import (
    SAMPLER_REGISTRY,
    TMCMC,
    AdaptiveMetropolis,
    ConjugateGaussian,
    GaussianLikelihood,
    HamiltonianMonteCarlo,
    IndependentNormal,
    IndependentUniform,
    InferenceProblem,
    MetropolisHastings,
    PosteriorSamples,
    effective_sample_size,
    get_sampler,
    r_hat,
)
from simulacrum.uq.forward import (
    Ishigami,
    JointIndependent,
    LinearLimitState,
    LogNormal,
    MonteCarloResult,
    RandomVariable,
    ReliabilityResult,
    SobolResult,
    control_variates,
    form,
    importance_sampling_probability,
    monte_carlo_probability,
    propagate,
    required_samples,
    sobol_indices,
    subset_simulation,
)
from simulacrum.uq.gp import GaussianProcessEmulator
from simulacrum.uq.sampling import (
    importance_sampling,
    latin_hypercube,
    monte_carlo,
    sobol_sequence,
)

__all__ = [
    # forward: propagation
    "RandomVariable",
    "LogNormal",
    "JointIndependent",
    "propagate",
    "MonteCarloResult",
    "control_variates",
    # forward: reliability and sensitivity
    "monte_carlo_probability",
    "form",
    "importance_sampling_probability",
    "subset_simulation",
    "required_samples",
    "ReliabilityResult",
    "sobol_indices",
    "SobolResult",
    "Ishigami",
    "LinearLimitState",
    # forward: bare designs and surrogates
    "monte_carlo",
    "latin_hypercube",
    "sobol_sequence",
    "importance_sampling",
    "GaussianProcessEmulator",
    # inverse
    "InferenceProblem",
    "IndependentNormal",
    "IndependentUniform",
    "GaussianLikelihood",
    "PosteriorSamples",
    "MetropolisHastings",
    "AdaptiveMetropolis",
    "TMCMC",
    "HamiltonianMonteCarlo",
    "ConjugateGaussian",
    "SAMPLER_REGISTRY",
    "get_sampler",
    "effective_sample_size",
    "r_hat",
]
