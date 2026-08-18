"""
Verification for the Bayesian inference samplers.

The external benchmark is the conjugate linear-Gaussian problem, whose
posterior mean, covariance and *evidence* are all closed-form. That matters
more here than in most modules: a sampler that is subtly wrong still returns a
plausible-looking histogram, and two samplers agreeing with each other proves
only that they share a misconception. Everything below is measured against the
closed form or against a brute-force quadrature, never against another sampler.

Three checks carry most of the weight.

**Moments against the exact posterior.** Location is the forgiving one -- the
likelihood dominates and almost anything finds the right neighbourhood. The
posterior *standard deviation* is where a mistuned or subtly biased sampler
shows up.

**The evidence.** ``log Z`` depends on the normalisation of every density in
the calculation, so it catches the dropped constants that Metropolis ratios
hide. Only TMCMC produces one.

**Overflow.** Both bugs this module was assembled from were overflow bugs that
are invisible on a toy problem: exponentiating a log-weight before normalising,
and exponentiating an acceptance ratio before comparing. Both are tested at a
magnitude where they actually fail.

python -m pytest tests/test_uq/test_bayesian.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
# not np.trapz / np.trapezoid: NumPy renamed one to the other in 2.0 and
# removed the old name, so either spelling pins the library to one major
# version. SciPy is already a hard dependency and carries both.
from scipy.integrate import trapezoid

from scion.uq.bayesian import (
    PYRO_AVAILABLE,
    AdaptiveMetropolis,
    CallableLikelihood,
    ConjugateGaussian,
    GaussianLikelihood,
    HamiltonianMonteCarlo,
    IndependentNormal,
    IndependentUniform,
    InferenceProblem,
    MetropolisHastings,
    PosteriorSamples,
    TMCMC,
    autocorrelation,
    effective_sample_size,
    geweke,
    get_sampler,
    pyro_nuts,
    r_hat,
)

#: The 1-D problem the incoming code was written against: theta ~ N(0, 2^2),
#: ten measurements with sigma = 0.5.
SCALAR = dict(prior_mean=[0.0], prior_std=[2.0], noise=0.5)

#: A 2-D linear problem whose two parameters are strongly correlated through
#: the design matrix -- the case an isotropic proposal handles badly.
DESIGN_2D = np.array([[1.0, 0.0], [1.0, 0.98], [1.0, 1.02], [0.0, 1.0]] * 5)


@pytest.fixture
def scalar_reference():
    return ConjugateGaussian(design=np.ones((10, 1)), **SCALAR).generate(
        [2.5], seed=42)


@pytest.fixture
def planar_reference():
    return ConjugateGaussian([100.0, 100.0], [20.0, 20.0], 1.0,
                             design=DESIGN_2D).generate([110.0, 95.0], seed=7)


# ---------------------------------------------------------------------------
# The reference itself
# ---------------------------------------------------------------------------

def test_the_scalar_posterior_matches_the_textbook_formula(scalar_reference):
    """``1/sigma_n^2 = 1/sigma_0^2 + n/sigma^2`` and the precision-weighted
    mean. Written out here rather than taken from the class, so this is a check
    on the implementation and not a restatement of it."""
    ref = scalar_reference
    n, sigma, sigma_0 = 10, 0.5, 2.0

    precision = 1.0 / sigma_0 ** 2 + n / sigma ** 2
    expected_std = 1.0 / np.sqrt(precision)
    expected_mean = (ref.data.sum() / sigma ** 2) / precision

    assert ref.posterior_std()[0] == pytest.approx(expected_std, rel=1e-12)
    assert ref.posterior_mean()[0] == pytest.approx(expected_mean, rel=1e-12)
    assert expected_std == pytest.approx(0.157622, abs=1e-6)


def test_the_evidence_matches_a_brute_force_integration(scalar_reference):
    """``log Z`` against the integral it is defined by, on a grid fine enough
    that the quadrature error is negligible. The closed form comes from a
    completely different piece of algebra -- marginalising the Gaussian --
    so agreement is a real check rather than a tautology."""
    ref = scalar_reference
    problem = ref.as_problem()

    grid = np.linspace(-6.0, 12.0, 200001)
    log_integrand = np.array([
        problem.prior.log_pdf([t]) + problem.likelihood.log_pdf([t])
        for t in grid])
    shift = log_integrand.max()
    integral = trapezoid(np.exp(log_integrand - shift), grid)

    assert shift + np.log(integral) == pytest.approx(ref.log_evidence(),
                                                     abs=1e-8)


def test_the_posterior_is_tighter_than_the_prior_and_pulled_towards_the_data(
        scalar_reference):
    ref = scalar_reference
    assert ref.posterior_std()[0] < 2.0
    assert abs(ref.posterior_mean()[0] - ref.data.mean()) < 0.05


def test_a_two_dimensional_reference_is_self_consistent(planar_reference):
    """The exact density integrated over a grid must come to one, and its
    moments must reproduce the closed forms."""
    ref = planar_reference
    mean, std = ref.posterior_mean(), ref.posterior_std()

    axes = [np.linspace(m - 6 * s, m + 6 * s, 301) for m, s in zip(mean, std)]
    X, Y = np.meshgrid(*axes, indexing="ij")
    pdf = ref.posterior_pdf(np.column_stack([X.ravel(), Y.ravel()])
                            ).reshape(X.shape)

    dx = (axes[0][1] - axes[0][0]) * (axes[1][1] - axes[1][0])
    assert pdf.sum() * dx == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# The problem interface
# ---------------------------------------------------------------------------

def test_the_gradient_matches_finite_differences(planar_reference):
    """The finite-difference check this library applies to every analytic
    derivative. HMC is driven entirely by this gradient, and a wrong one still
    produces a chain -- just one that samples the wrong distribution."""
    problem = planar_reference.as_problem()
    theta = np.array([104.0, 97.5])

    analytic = problem.grad_log_posterior(theta)
    numeric = np.zeros(2)
    for i in range(2):
        h = 1e-5
        plus, minus = theta.copy(), theta.copy()
        plus[i] += h
        minus[i] -= h
        numeric[i] = (problem.log_posterior(plus)
                      - problem.log_posterior(minus)) / (2 * h)

    assert problem.has_analytic_gradient
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5)


def test_the_finite_difference_fallback_is_used_when_no_jacobian_is_given():
    """And costs what it claims: 2 * dim model evaluations per gradient."""
    prior = IndependentNormal([0.0, 0.0], [1.0, 1.0])
    likelihood = GaussianLikelihood(model=lambda t: np.array([t[0] + t[1]]),
                                    data=[1.0], noise=0.5)
    problem = InferenceProblem(prior, likelihood)
    assert not problem.has_analytic_gradient

    problem.reset_counter()
    grad = problem.grad_log_posterior(np.array([0.3, -0.2]))
    assert problem.n_evaluations == 4                      # 2 * dim

    exact = np.array([-0.3, 0.2]) + (1.0 - 0.1) / 0.25 * np.array([1.0, 1.0])
    np.testing.assert_allclose(grad, exact, rtol=1e-5)


def test_the_likelihood_is_never_evaluated_outside_a_bounded_prior():
    """A forward model handed a nonsensical parameter may not fail cleanly, so
    the prior short-circuits before the model is called. The counter proves the
    call did not happen."""
    calls = []

    def model(theta):
        calls.append(theta.copy())
        return np.array([theta[0]])

    problem = InferenceProblem(
        prior=IndependentUniform([0.0], [1.0]),
        likelihood=GaussianLikelihood(model, data=[0.5], noise=0.1))

    assert problem.log_posterior([5.0]) == -np.inf
    assert calls == []
    assert problem.n_evaluations == 0

    assert np.isfinite(problem.log_posterior([0.4]))
    assert len(calls) == 1


def test_a_uniform_prior_is_not_a_gaussian_with_the_bounds_as_parameters():
    """The mistake this interface exists to make impossible: passing
    ``(lower, upper)`` into a Gaussian log-density, which silently evaluates a
    different prior entirely instead of a flat one."""
    uniform = IndependentUniform([2.0], [8.0])
    inside = [uniform.log_pdf([x]) for x in (2.5, 5.0, 7.5)]

    assert len(set(inside)) == 1                    # genuinely flat
    assert inside[0] == pytest.approx(-np.log(6.0))
    assert uniform.log_pdf([1.9]) == -np.inf
    assert uniform.log_pdf([8.1]) == -np.inf

    # a Gaussian with mean 2 and std 8 is emphatically not flat on [2, 8]
    wrong = IndependentNormal([2.0], [8.0])
    assert wrong.log_pdf([2.5]) != pytest.approx(wrong.log_pdf([7.5]))


def test_tempering_needs_the_prior_and_likelihood_kept_apart(scalar_reference):
    """``tempered(beta=1)`` is the posterior and ``tempered(beta=0)`` is the
    prior. A sampler handed only their product could produce neither, which is
    why InferenceProblem refuses a bare log-posterior callable."""
    problem = scalar_reference.as_problem()
    theta = [1.7]

    assert problem.tempered_log_posterior(theta, 1.0) == pytest.approx(
        problem.log_posterior(theta))
    assert problem.tempered_log_posterior(theta, 0.0) == pytest.approx(
        problem.log_prior(theta))

    with pytest.raises(TypeError, match="prior must be a Prior"):
        InferenceProblem(lambda t: 0.0, CallableLikelihood(lambda t: 0.0))


def test_the_evaluation_counter_is_per_problem_not_global(scalar_reference):
    """The incoming code counted FE evaluations in a module-level dict, so two
    studies in one session corrupted each other's tally."""
    a = scalar_reference.as_problem()
    b = scalar_reference.as_problem()

    for _ in range(5):
        a.log_posterior([1.0])
    assert (a.n_evaluations, b.n_evaluations) == (5, 0)


# ---------------------------------------------------------------------------
# The samplers, against the closed form
# ---------------------------------------------------------------------------

def _run(name, problem, seed=1):
    if name == "mh":
        return MetropolisHastings(problem, proposal_std=0.4, seed=seed).sample(
            30000, n_burn=2000)
    if name == "am":
        return AdaptiveMetropolis(problem, initial_std=0.4, seed=seed).sample(
            30000, n_burn=2000)
    if name == "hmc":
        return HamiltonianMonteCarlo(problem, step_size=0.1, n_leapfrog=10,
                                     seed=seed).sample(8000, n_burn=500)
    return TMCMC(problem, n_particles=6000, seed=seed).sample()


@pytest.mark.parametrize("name", ["mh", "am", "hmc", "tmcmc"])
def test_every_sampler_reproduces_the_analytic_posterior(name,
                                                         scalar_reference):
    """Mean to within a tenth of a posterior standard deviation, and the
    standard deviation itself to within 5%. Measured over six seeds in
    development: the errors scatter about zero rather than sitting to one side,
    so these tolerances are Monte Carlo noise and not slack for a bias."""
    result = _run(name, scalar_reference.as_problem())
    error = scalar_reference.error(result)

    assert error["mean_error"] < 0.1
    assert error["std_error"] < 0.05


@pytest.mark.parametrize("name", ["mh", "am", "hmc", "tmcmc"])
def test_every_sampler_handles_the_correlated_two_dimensional_posterior(
        name, planar_reference):
    result = _run(name, planar_reference.as_problem(), seed=2)
    error = planar_reference.error(result)

    assert error["mean_error"] < 0.15
    assert error["std_error"] < 0.06


def test_the_samplers_agree_with_each_other_as_well(scalar_reference):
    """Necessary but not sufficient, and only meaningful because each has
    already been checked against the closed form above."""
    problem = scalar_reference.as_problem()
    means = [_run(name, problem).mean()[0]
             for name in ("mh", "am", "hmc", "tmcmc")]
    assert np.ptp(means) < 0.1 * scalar_reference.posterior_std()[0]


# ---------------------------------------------------------------------------
# TMCMC
# ---------------------------------------------------------------------------

def test_tmcmc_recovers_the_exact_evidence(scalar_reference):
    """The check no other sampler here can be given. ``log Z`` is what a Bayes
    factor is built from, and it is sensitive to every normalising constant in
    the calculation."""
    result = TMCMC(scalar_reference.as_problem(), n_particles=8000,
                   seed=1).sample()

    assert result.log_evidence == pytest.approx(
        scalar_reference.log_evidence(), abs=0.1)


def test_the_adaptive_schedule_chooses_its_own_stages(scalar_reference):
    """Betas increase from 0 to exactly 1, and the number of stages is found
    rather than prescribed."""
    result = TMCMC(scalar_reference.as_problem(), n_particles=2000,
                   seed=3).sample()
    betas = result.info["betas"]

    assert betas[0] == 0.0
    assert betas[-1] == 1.0
    assert np.all(np.diff(betas) > 0.0)
    assert 2 <= result.info["n_stages"] <= 30


def test_a_smaller_target_cv_produces_more_and_gentler_stages(
        scalar_reference):
    """The adaptive schedule's one knob, behaving monotonically."""
    problem = scalar_reference.as_problem()
    coarse = TMCMC(problem, n_particles=2000, target_cv=2.0, seed=3).sample()
    fine = TMCMC(problem, n_particles=2000, target_cv=0.3, seed=3).sample()

    assert fine.info["n_stages"] > coarse.info["n_stages"]


def test_the_tempering_weights_do_not_overflow_on_a_sharp_likelihood():
    """The regression for the bug in the incoming code. Exponentiating
    ``d_beta * log_L`` before normalising underflows every weight to zero once
    the log-likelihood reaches a few hundred, and the resampling step then
    divides by zero. Here the log-likelihood is order -1e4, which is ordinary
    for a few hundred observations.
    """
    n_data = 4000
    ref = ConjugateGaussian([0.0], [3.0], 0.05,
                            design=np.ones((n_data, 1))).generate([1.5],
                                                                  seed=5)
    problem = ref.as_problem()
    assert problem.log_likelihood([0.0]) < -1e4         # the dangerous regime

    result = TMCMC(problem, n_particles=1000, seed=1).sample()

    assert np.all(np.isfinite(result.samples))
    assert result.mean()[0] == pytest.approx(ref.posterior_mean()[0],
                                             abs=5 * ref.posterior_std()[0])
    assert np.isfinite(result.log_evidence)


def test_a_prescribed_schedule_is_accepted_and_must_end_at_one(
        scalar_reference):
    problem = scalar_reference.as_problem()
    result = TMCMC(problem, n_particles=1000,
                   schedule=np.linspace(0.0, 1.0, 11), seed=1).sample()
    assert result.info["n_stages"] == 10

    with pytest.raises(ValueError, match="end at exactly 1.0"):
        TMCMC(problem, schedule=[0.2, 0.5, 0.9])
    with pytest.raises(ValueError, match="increase"):
        TMCMC(problem, schedule=[0.5, 0.2, 1.0])


def test_tmcmc_output_is_a_particle_set_and_says_so(scalar_reference):
    """Order carries no information in a resampled population, so the
    chain-only operations refuse rather than returning a meaningless number."""
    result = TMCMC(scalar_reference.as_problem(), n_particles=500,
                   seed=1).sample()

    assert not result.ordered
    for call in (result.ess, result.autocorrelation_time, result.geweke):
        with pytest.raises(ValueError, match="ordered chain"):
            call()
    with pytest.raises(ValueError, match="ordered chain"):
        result.burn(10)

    assert 0.0 < result.unique_fraction <= 1.0


def test_tmcmc_refuses_a_sample_count_it_cannot_honour(scalar_reference):
    with pytest.raises(ValueError, match="population size"):
        TMCMC(scalar_reference.as_problem(), n_particles=500).sample(1000)


# ---------------------------------------------------------------------------
# Metropolis
# ---------------------------------------------------------------------------

def test_the_acceptance_test_is_taken_in_log_space():
    """``log u < log_ratio`` rather than ``u < min(1, exp(log_ratio))``.

    Being precise about what this buys, because the usual claim is too strong:
    the exponentiated form gets the *decision* right anyway, since IEEE
    saturation sends a large positive ratio to ``inf`` (accept) and a large
    negative one to ``0`` (reject). What it does not do is get there without
    an overflow or underflow, which is a genuine error under
    ``np.errstate`` and noise in any log that does not suppress warnings.
    Verified in development: ``min(1, np.exp(1e4))`` raises
    ``FloatingPointError`` under ``errstate(over='raise')`` and
    ``np.exp(-1e4)`` under ``under='raise'``. The log form cannot do either.

    The proposal here is deliberately far wider than the posterior, so nearly
    every step produces a ratio of order -1e4.
    """
    prior = IndependentNormal([0.0], [1000.0])
    likelihood = CallableLikelihood(lambda t: -1e4 * (t[0] - 3.0) ** 2)
    problem = InferenceProblem(prior, likelihood)

    with np.errstate(over="raise", under="raise"):
        MetropolisHastings(problem, proposal_std=1.0, seed=1).sample(
            500, x0=[3.0], n_burn=100)

    # and with a proposal matched to the posterior it recovers it: the
    # likelihood is N(3, 1/sqrt(2e4)) and the prior is effectively flat
    with np.errstate(over="raise", under="raise"):
        result = MetropolisHastings(problem, proposal_std=0.01,
                                    seed=1).sample(20000, x0=[3.0],
                                                   n_burn=2000)

    assert np.all(np.isfinite(result.samples))
    assert result.mean()[0] == pytest.approx(3.0, abs=0.002)
    assert result.std()[0] == pytest.approx(1.0 / np.sqrt(2e4), rel=0.1)


def test_burn_in_is_extra_rather_than_carved_out_of_the_request(
        scalar_reference):
    """The result always holds exactly the number of samples asked for, so two
    runs with different burn-in are comparable."""
    problem = scalar_reference.as_problem()
    for n_burn in (0, 500, 5000):
        result = MetropolisHastings(problem, seed=1).sample(1000,
                                                            n_burn=n_burn)
        assert result.n_samples == 1000


def test_thinning_keeps_every_nth_sample(scalar_reference):
    result = MetropolisHastings(scalar_reference.as_problem(),
                                seed=1).sample(500, n_burn=100, thin=4)
    assert result.n_samples == 500


def test_adaptive_metropolis_learns_a_correlated_proposal(planar_reference):
    """The point of the method: the learned covariance should reflect the
    posterior's correlation, not the isotropic guess it started from."""
    result = AdaptiveMetropolis(planar_reference.as_problem(), initial_std=1.0,
                                seed=2).sample(20000, n_burn=4000)
    learned = result.info["learned_cov"]
    exact = planar_reference.posterior_cov()

    def correlation(c):
        return c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])

    assert correlation(learned) == pytest.approx(correlation(exact), abs=0.15)


def test_adaptation_stops_before_the_kept_samples_begin(scalar_reference):
    """A proposal that keeps depending on the chain's own history makes the
    chain non-Markov and voids the convergence proof, so adaptation is confined
    to burn-in -- and burn-in must be long enough to contain some."""
    problem = scalar_reference.as_problem()
    with pytest.raises(ValueError, match="no room to adapt"):
        AdaptiveMetropolis(problem, adapt_after=1000).sample(100, n_burn=500)


def test_a_starting_point_with_no_posterior_density_is_refused():
    problem = InferenceProblem(
        prior=IndependentUniform([0.0], [1.0]),
        likelihood=CallableLikelihood(lambda t: 0.0))
    with pytest.raises(ValueError, match="zero posterior density"):
        MetropolisHastings(problem, seed=1).sample(10, x0=[7.0])


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def test_hmc_mixes_better_per_model_evaluation_than_random_walk(
        planar_reference):
    """The whole justification for needing a gradient. Measured per *model
    evaluation*, which is the honest currency -- HMC takes many more of them
    per sample."""
    problem = planar_reference.as_problem()

    problem.reset_counter()
    walk = MetropolisHastings(problem, proposal_std=1.0, seed=2).sample(
        40000, n_burn=4000)
    walk_efficiency = walk.ess().min() / walk.n_evaluations

    problem.reset_counter()
    hmc = HamiltonianMonteCarlo(problem, step_size=0.3, n_leapfrog=15,
                                seed=2).sample(8000, n_burn=500)
    hmc_efficiency = hmc.ess().min() / hmc.n_evaluations

    assert hmc_efficiency > 3.0 * walk_efficiency


def test_hmc_warns_when_it_would_have_to_estimate_the_gradient():
    """Without an analytic gradient HMC costs ``2 * dim`` model evaluations per
    leapfrog step. That is a decision the caller should make deliberately, so
    it is said once rather than discovered from the runtime."""
    prior = IndependentNormal([0.0, 0.0], [1.0, 1.0])
    likelihood = GaussianLikelihood(lambda t: np.array([t[0] + t[1]]),
                                    data=[1.0], noise=0.5)
    problem = InferenceProblem(prior, likelihood)

    with pytest.warns(RuntimeWarning, match="no analytic gradient"):
        HamiltonianMonteCarlo(problem, seed=1).sample(20, n_burn=10)


def test_hmc_adapts_its_step_size_towards_the_target_acceptance(
        scalar_reference):
    """0.65, not 0.234 -- the HMC optimum is higher than the random-walk one,
    which is a routine source of mistuning."""
    result = HamiltonianMonteCarlo(scalar_reference.as_problem(),
                                   step_size=0.001, n_leapfrog=10,
                                   seed=1).sample(4000, n_burn=1000)

    assert result.info["step_size"] > 0.01          # grew from the tiny start
    assert 0.5 < result.acceptance_rate < 0.95
    assert result.info["n_divergent"] == 0


def test_an_unstable_step_size_shows_up_as_divergences(scalar_reference):
    """A step size far past the stability limit makes the leapfrog energy blow
    up. Counting those separately is what distinguishes bad tuning from a hard
    posterior; a plain acceptance rate cannot."""
    result = HamiltonianMonteCarlo(scalar_reference.as_problem(),
                                   step_size=50.0, n_leapfrog=20,
                                   adapt_step_size=False,
                                   seed=1).sample(300, n_burn=50)

    assert result.info["n_divergent"] > 0
    assert result.acceptance_rate < 0.5


def test_the_leapfrog_integrator_conserves_energy_far_better_than_euler(
        planar_reference):
    """Symplectic, so the energy error stays bounded instead of drifting. This
    is what makes the Metropolis correction on exp(-dH) close to one, and it is
    a property of the integrator rather than of the target."""
    problem = planar_reference.as_problem()
    hmc = HamiltonianMonteCarlo(problem, step_size=0.2, n_leapfrog=200,
                                seed=1)
    theta = planar_reference.posterior_mean()
    p = np.array([0.4, -0.3])

    start = hmc._potential(theta) + hmc._kinetic(p)
    end_theta, end_p, ok = hmc._leapfrog(theta, p, 0.2, 200)
    end = hmc._potential(end_theta) + hmc._kinetic(end_p)

    assert ok
    assert abs(end - start) < 0.01 * max(1.0, abs(start))


@pytest.mark.skipif(PYRO_AVAILABLE, reason="pyro is installed")
def test_the_pyro_backend_explains_itself_when_absent():
    """The optional path fails with an explanation and a pointer to the
    supported one, rather than an ImportError from a nested module."""
    with pytest.raises(ImportError, match="pure NumPy"):
        pyro_nuts(lambda: None)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_independent_draws_have_an_effective_size_near_their_count():
    rng = np.random.default_rng(0)
    ess = effective_sample_size(rng.normal(size=(20000, 1)))
    assert 0.8 * 20000 < ess[0] <= 1.05 * 20000


def test_a_correlated_chain_is_worth_fewer_samples_than_it_holds():
    """An AR(1) chain with a known autocorrelation time ``(1+rho)/(1-rho)``."""
    rho, n = 0.9, 40000
    rng = np.random.default_rng(0)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0.0, np.sqrt(1 - rho ** 2))

    expected_tau = (1 + rho) / (1 - rho)             # 19
    assert effective_sample_size(x)[0] == pytest.approx(n / expected_tau,
                                                        rel=0.25)


def test_the_autocorrelation_starts_at_one_and_decays():
    rng = np.random.default_rng(1)
    rho = autocorrelation(rng.normal(size=(5000, 2)), max_lag=50)

    np.testing.assert_allclose(rho[0], [1.0, 1.0])
    assert np.abs(rho[1:]).max() < 0.2


def test_r_hat_is_near_one_for_chains_that_mixed(scalar_reference):
    problem = scalar_reference.as_problem()
    chains = [MetropolisHastings(problem, proposal_std=0.4, seed=s).sample(
        4000, n_burn=1000).samples for s in range(4)]

    assert r_hat(chains)[0] < 1.01


def test_r_hat_detects_chains_that_never_met():
    """Four chains sitting in four different places. Each looks perfectly
    converged on its own, which is precisely the failure a single-chain
    diagnostic cannot see."""
    rng = np.random.default_rng(0)
    chains = [rng.normal(loc, 0.1, size=(2000, 1)) for loc in (0, 5, 10, 15)]
    assert r_hat(chains)[0] > 1.5


def test_r_hat_refuses_a_single_chain():
    """Between-chain variance is the whole statistic, and one chain has none."""
    with pytest.raises(ValueError, match="at least two"):
        r_hat(np.random.default_rng(0).normal(size=(100, 1)))


def test_geweke_flags_a_chain_that_is_still_moving():
    drifting = (np.linspace(0.0, 10.0, 4000)
                + np.random.default_rng(0).normal(0, 0.1, 4000))
    settled = np.random.default_rng(0).normal(size=4000)

    assert abs(geweke(drifting)[0]) > 5.0
    assert abs(geweke(settled)[0]) < 3.0


# ---------------------------------------------------------------------------
# Result container and registry
# ---------------------------------------------------------------------------

def test_the_result_reports_moments_and_a_credible_interval():
    rng = np.random.default_rng(0)
    result = PosteriorSamples(samples=rng.normal(3.0, 2.0, size=(40000, 1)),
                              method="test")

    assert result.mean()[0] == pytest.approx(3.0, abs=0.05)
    assert result.std()[0] == pytest.approx(2.0, abs=0.05)

    low, high = result.credible_interval(0.95)
    assert low[0] == pytest.approx(3.0 - 1.96 * 2.0, abs=0.1)
    assert high[0] == pytest.approx(3.0 + 1.96 * 2.0, abs=0.1)
    assert "test" in result.summary()


def test_burn_and_thin_preserve_the_metadata(scalar_reference):
    result = MetropolisHastings(scalar_reference.as_problem(),
                                seed=1).sample(1000, n_burn=100)
    trimmed = result.burn(200).thin(2)

    assert trimmed.n_samples == 400
    assert trimmed.method == result.method
    assert trimmed.acceptance_rate == result.acceptance_rate
    assert trimmed.log_posterior.size == 400


def test_the_registry_builds_every_sampler(scalar_reference):
    problem = scalar_reference.as_problem()
    for name in ("mh", "metropolis-hastings", "am", "tmcmc", "hmc"):
        assert get_sampler(name, problem) is not None

    with pytest.raises(NotImplementedError, match="Pyro"):
        get_sampler("nuts", problem)
