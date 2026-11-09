"""
Uncertainty quantification.

Three questions, and they are different problems rather than three settings of
one.

**Forward** -- push a distribution over inputs through the model and describe
the distribution that comes out, as moments, sensitivity indices or a failure
probability. Sampling and surrogates: ``sampling``, ``pce``, ``gp``.

**Inverse** -- start from measurements and recover the distribution over inputs
that could have produced them. Bayesian inference: ``bayesian``.

**Density** -- the whole PDF of a response and, for a dynamic problem, its
evolution in time, rather than a statistic of it. ``density``, holding the
probability density evolution method. It is separate from ``forward`` because
the output is a function rather than a number, and because it needs the model
to supply a response *velocity* and not only a response.

Submodules
----------
density     representative point sets and the probability density evolution
            method (Li & Chen)
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
as physics-free as :mod:`scion.solvers`.
"""

from scion.uq.bayesian import (
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
from scion.uq.density import (
    DensityEvolution,
    ProbabilityNotConserved,
    RepresentativePoints,
    band_probability,
    exceedance_probability,
    lattice,
    response_histories,
    solve_dpim,
    solve_gdee,
)
from scion.uq.forward import (
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
from scion.uq.gp import GaussianProcessEmulator
from scion.uq.sampling import (
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
    # density
    "RepresentativePoints",
    "lattice",
    "solve_gdee",
    "solve_dpim",
    "exceedance_probability",
    "band_probability",
    "response_histories",
    "DensityEvolution",
    "ProbabilityNotConserved",
]
