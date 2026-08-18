"""
Bayesian inference: recovering parameters from measurements.

Three samplers, one problem interface, one result container.

    MetropolisHastings    the baseline. No gradient, one tuning parameter.
    AdaptiveMetropolis    learns the proposal covariance during burn-in; the
                          right default when parameters are correlated or
                          differ in scale.
    TMCMC                 a tempered particle population. Handles a posterior
                          concentrated in a small corner of the prior, handles
                          multiple modes, and returns the evidence.
    HamiltonianMonteCarlo gradient-guided. By far the best mixing per sample
                          when a gradient is available, and by far the worst
                          when it has to be estimated by finite differences.

Which to reach for
------------------
Cheap forward model and few parameters: ``AdaptiveMetropolis``. Expensive FE
model, or a posterior far from the prior, or model comparison: ``TMCMC``.
Analytic gradient available: ``HamiltonianMonteCarlo``. The deciding question
for HMC is :attr:`InferenceProblem.has_analytic_gradient`, not the dimension.

The problem must be separable
-----------------------------
Every sampler takes an :class:`InferenceProblem`, built from a prior and a
likelihood kept apart -- not a single log-posterior callable. Tempering raises
the likelihood to a power, so TMCMC cannot be written against their product.
See :mod:`~.problem`.

Verification
------------
:class:`ConjugateGaussian` is the closed-form linear-Gaussian problem: exact
posterior mean, covariance and evidence. Every sampler is tested against it,
because two samplers agreeing with each other proves only that they share a
misconception.
"""

from simulacrum.uq.bayesian.chains import PosteriorSamples
from simulacrum.uq.bayesian.diagnostics import (
    autocorrelation,
    effective_sample_size,
    geweke,
    integrated_autocorrelation_time,
    r_hat,
)
from simulacrum.uq.bayesian.hmc import (
    PYRO_AVAILABLE,
    HamiltonianMonteCarlo,
    pyro_nuts,
)
from simulacrum.uq.bayesian.metropolis import (
    AdaptiveMetropolis,
    MetropolisHastings,
)
from simulacrum.uq.bayesian.problem import (
    CallableLikelihood,
    GaussianLikelihood,
    IndependentNormal,
    IndependentUniform,
    InferenceProblem,
    Likelihood,
    Prior,
)
from simulacrum.uq.bayesian.reference import ConjugateGaussian
from simulacrum.uq.bayesian.tmcmc import TMCMC

#: Name -> sampler class, for the same reason every other axis in the library
#: has one: adding a published method should be one class and one entry.
SAMPLER_REGISTRY = {
    "mh": MetropolisHastings,
    "metropolis": MetropolisHastings,
    "metropolis-hastings": MetropolisHastings,
    "am": AdaptiveMetropolis,
    "adaptive-metropolis": AdaptiveMetropolis,
    "tmcmc": TMCMC,
    "hmc": HamiltonianMonteCarlo,
}


def get_sampler(name: str, problem, **kwargs):
    """Build a sampler by name."""
    key = name.lower().replace("_", "-")
    if key not in SAMPLER_REGISTRY:
        raise NotImplementedError(
            f"Unknown sampler '{name}'. Available: "
            f"{sorted(set(SAMPLER_REGISTRY))}. NUTS is available only through "
            f"the optional Pyro backend, uq.bayesian.pyro_nuts, because it "
            f"needs automatic differentiation of the model.")
    return SAMPLER_REGISTRY[key](problem, **kwargs)


__all__ = [
    "InferenceProblem",
    "Prior",
    "IndependentNormal",
    "IndependentUniform",
    "Likelihood",
    "GaussianLikelihood",
    "CallableLikelihood",
    "PosteriorSamples",
    "MetropolisHastings",
    "AdaptiveMetropolis",
    "TMCMC",
    "HamiltonianMonteCarlo",
    "pyro_nuts",
    "PYRO_AVAILABLE",
    "ConjugateGaussian",
    "SAMPLER_REGISTRY",
    "get_sampler",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "r_hat",
    "geweke",
]
