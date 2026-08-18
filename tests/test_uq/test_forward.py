"""
Verification for forward uncertainty quantification.

Two external benchmarks carry the module, both with closed-form answers:

**Ishigami** for sensitivity. Its analytic Sobol indices include the case that
matters -- an input with ``S_3 = 0`` exactly and ``S_T3 = 0.244`` -- so an
implementation that confuses first-order with total, or that silently produces
one when asked for the other, cannot pass.

**The linear limit state** for reliability. ``P_f = Phi(-beta)`` exactly, and
``beta`` dials the probability down to ``1e-6`` where crude Monte Carlo cannot
follow. FORM is exact here by construction, so any error is a bug rather than a
modelling approximation.

Alongside those, the Monte Carlo *rate* is measured rather than assumed. An
error is a number and can be small by luck; the slope of error against sample
count is a property of the method. It comes out at -0.4999 against a theoretical
-0.5, which is the forward-UQ version of the observed-order check this library
applies to every discretisation.

python -m pytest tests/test_uq/test_forward.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from simulacrum.uq.forward import (
    DESIGNS,
    Ishigami,
    JointIndependent,
    LinearLimitState,
    LogNormal,
    Normal,
    QuadraticLimitState,
    Uniform,
    control_variates,
    form,
    get_distribution,
    importance_sampling,
    importance_sampling_probability,
    monte_carlo_probability,
    propagate,
    required_samples,
    sobol_indices,
    stratified,
    subset_simulation,
    unit_design,
)

STANDARD_NORMAL = JointIndependent([Normal(0.0, 1.0)])

#: symmetric about the distribution's centre -- the case antithetic hurts
SQUARE = lambda x: x[:, 0] ** 2
#: monotone -- the case antithetic helps
MONOTONE = lambda x: np.exp(0.7 * x[:, 0])


# ---------------------------------------------------------------------------
# Distributions and the unit-hypercube transform
# ---------------------------------------------------------------------------

def test_the_quantile_transform_reproduces_the_intended_moments():
    """Every design is generated on ``[0,1]^d`` and mapped through ``ppf``, so
    if that transform is wrong every method here is wrong together."""
    joint = JointIndependent([Normal(3.0, 2.0), Uniform(-1.0, 5.0)])
    x = joint.sample(200000, np.random.default_rng(0))

    np.testing.assert_allclose(x.mean(axis=0), joint.mean, atol=0.02)
    np.testing.assert_allclose(x.std(axis=0), joint.std, rtol=0.02)


def test_the_lognormal_moment_parameterisation_is_exact():
    """``from_moments`` inverts the mean/variance relations in closed form, so
    it should round-trip to machine precision, not approximately."""
    variable = LogNormal.from_moments(210.0, 21.0)
    assert variable.mean == pytest.approx(210.0, rel=1e-12)
    assert variable.std == pytest.approx(21.0, rel=1e-12)


def test_the_lognormal_constructor_takes_the_underlying_normals_parameters():
    """The trap the two entry points exist to separate. ``LogNormal(1, 0.3)``
    is not a variable with mean 1 and standard deviation 0.3 -- it has mean
    2.84 and standard deviation 0.87."""
    naive = LogNormal(1.0, 0.3)
    correct = LogNormal.from_moments(1.0, 0.3)

    assert naive.mean == pytest.approx(np.exp(1.0 + 0.045), rel=1e-9)
    assert naive.mean > 2.8
    assert correct.mean == pytest.approx(1.0, rel=1e-12)


def test_the_standard_normal_transform_round_trips():
    """``x = F^-1(Phi(z))`` and back. FORM and subset simulation both work in
    that space, so a broken transform makes both silently wrong."""
    joint = JointIndependent([Normal(3.0, 2.0), Uniform(-1.0, 5.0),
                              LogNormal.from_moments(10.0, 2.0)])
    z = np.random.default_rng(0).standard_normal((500, 3))

    np.testing.assert_allclose(
        joint.to_standard_normal(joint.from_standard_normal(z)), z, atol=1e-8)


def test_an_unknown_distribution_is_refused():
    with pytest.raises(NotImplementedError, match="Unknown distribution"):
        get_distribution("weibull")


def test_a_degenerate_distribution_is_refused():
    for bad in (lambda: Normal(0.0, -1.0), lambda: Uniform(2.0, 1.0),
                lambda: LogNormal.from_moments(-1.0, 1.0)):
        with pytest.raises(ValueError):
            bad()


# ---------------------------------------------------------------------------
# Designs and the rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("design", DESIGNS)
def test_every_design_fills_the_unit_hypercube(design):
    unit = unit_design(design, 256, 3, np.random.default_rng(0))

    assert unit.shape == (256, 3)
    assert unit.min() >= 0.0 and unit.max() <= 1.0
    np.testing.assert_allclose(unit.mean(axis=0), 0.5, atol=0.05)


def test_latin_hypercube_stratifies_each_margin_exactly():
    """The defining property: with ``n`` points and ``n`` equal bins, every bin
    of every margin holds exactly one point. Crude sampling does not."""
    n = 64
    lhs = unit_design("lhs", n, 2, np.random.default_rng(0))
    for column in lhs.T:
        counts = np.histogram(column, bins=n, range=(0.0, 1.0))[0]
        assert np.all(counts == 1)

    crude = unit_design("crude", n, 2, np.random.default_rng(0))
    assert not np.all(np.histogram(crude[:, 0], bins=n,
                                   range=(0.0, 1.0))[0] == 1)


def test_the_antithetic_design_pairs_u_with_one_minus_u():
    unit = unit_design("antithetic", 100, 2, np.random.default_rng(0))
    np.testing.assert_allclose(unit[:50], 1.0 - unit[50:])


def test_sobol_warns_when_the_count_is_not_a_power_of_two():
    """Its equidistribution holds for ``2^m``. Warning beats silently rounding,
    which would change the number of model evaluations behind the caller."""
    with pytest.warns(RuntimeWarning, match="powers of two"):
        unit_design("sobol", 1000, 2, np.random.default_rng(0))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unit_design("sobol", 1024, 2, np.random.default_rng(0))


def test_the_monte_carlo_rate_is_one_over_root_n():
    """Measured, not assumed. The slope is a property of the method and no
    rearrangement of the average improves it -- which is why variance
    reduction, not a cleverer estimator, is the only lever."""
    counts = (256, 1024, 4096, 16384)
    errors = [
        np.sqrt(np.mean([(propagate(SQUARE, STANDARD_NORMAL, n, "crude",
                                    seed=s, vectorised=True).mean() - 1.0) ** 2
                         for s in range(120)]))
        for n in counts]

    slope = float(np.polyfit(np.log(counts), np.log(errors), 1)[0])
    assert slope == pytest.approx(-0.5, abs=0.05)


def test_the_rate_does_not_degrade_with_dimension():
    """The property no quadrature rule has, and the whole reason Monte Carlo
    is used at all. Same integrand mean per dimension, 1-D against 20-D."""
    def mean_of_sum(dim, n):
        joint = JointIndependent([Uniform(0.0, 1.0) for _ in range(dim)])
        model = lambda x: x.sum(axis=1) / dim
        return np.sqrt(np.mean([
            (propagate(model, joint, n, "crude", seed=s,
                       vectorised=True).mean() - 0.5) ** 2
            for s in range(60)]))

    low = mean_of_sum(1, 2048)
    high = mean_of_sum(20, 2048)
    assert high < low                     # averaging 20 inputs reduces variance


@pytest.mark.parametrize("design,better_than_crude",
                         [("lhs", True), ("sobol", True), ("halton", True),
                          ("antithetic", True)])
def test_the_smarter_designs_beat_crude_on_a_smooth_integrand(
        design, better_than_crude):
    """On a smooth, additive, low-dimensional integrand -- the regime they are
    all designed for. None of this transfers automatically to a rough or
    high-dimensional model."""
    joint = JointIndependent([Uniform(0.0, 1.0) for _ in range(4)])
    model = lambda x: np.sum(x, axis=1) ** 2
    exact = 4 * (1.0 / 3.0) + 12 * 0.25

    def rmse(name):
        means = [propagate(model, joint, 4096, name, seed=s,
                           vectorised=True).mean() for s in range(16)]
        return float(np.sqrt(np.mean((np.array(means) - exact) ** 2)))

    assert (rmse(design) < rmse("crude")) is better_than_crude


def test_antithetic_sampling_helps_a_monotone_model_and_hurts_a_symmetric_one():
    """The condition that is usually left unstated. Antithetic variates are
    free and are not safe: guaranteed only for a model monotone in every
    input, and actively harmful for one symmetric about the centre of the
    input distribution."""
    def estimator_variance(model, design):
        means = [propagate(model, STANDARD_NORMAL, 2000, design, seed=s,
                           vectorised=True).mean() for s in range(200)]
        return float(np.var(means))

    assert (estimator_variance(MONOTONE, "antithetic")
            < 0.6 * estimator_variance(MONOTONE, "crude"))
    assert (estimator_variance(SQUARE, "antithetic")
            > 1.3 * estimator_variance(SQUARE, "crude"))


def test_a_quasi_random_result_refuses_to_invent_a_standard_error():
    """The points are deterministic and correlated, so ``s/sqrt(N)``
    understates the error -- the dangerous direction. Replication gives an
    honest one."""
    joint = JointIndependent([Uniform(0.0, 1.0) for _ in range(3)])
    model = lambda x: np.sum(x, axis=1)

    plain = propagate(model, joint, 512, "sobol", seed=1, vectorised=True)
    assert plain.std_error is None
    with pytest.raises(ValueError, match="no valid standard error"):
        plain.confidence_interval()

    replicated = propagate(model, joint, 512, "sobol", n_replications=16,
                           seed=1, vectorised=True)
    assert replicated.std_error is not None
    low, high = replicated.confidence_interval(0.95)
    assert low < 1.5 < high
    assert replicated.n_evaluations == 512 * 16


def test_a_crude_confidence_interval_covers_at_about_its_nominal_rate():
    """The interval leans on the central limit theorem. On a well-behaved
    integrand it should cover 95% of the time; this is what makes the
    qualification on rare events meaningful by contrast."""
    covered = 0
    for seed in range(200):
        result = propagate(SQUARE, STANDARD_NORMAL, 2000, "crude", seed=seed,
                           vectorised=True)
        low, high = result.confidence_interval(0.95)
        covered += int(low <= 1.0 <= high)
    assert 0.90 < covered / 200 < 0.99


def test_an_unknown_design_is_refused():
    with pytest.raises(NotImplementedError, match="Unknown design"):
        propagate(SQUARE, STANDARD_NORMAL, 10, "weno", vectorised=True)


# ---------------------------------------------------------------------------
# Variance reduction
# ---------------------------------------------------------------------------

def test_control_variates_achieve_the_theoretical_variance_ratio():
    """``1 - rho^2``, which is the entire content of the method: a control
    correlated at 0.96 removes 92% of the variance and one correlated at 0.5
    removes 25%."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(8000)
    result = control_variates(np.exp(x), 1.0 + x + 0.5 * x ** 2,
                              control_mean=1.5)

    assert result.variance_ratio == pytest.approx(result.theoretical_ratio,
                                                  rel=0.05)
    assert result.mean == pytest.approx(np.exp(0.5), abs=4 * result.std_error)
    assert result.std_error < 0.4 * result.crude_std_error


def test_control_variates_stay_unbiased_with_a_useless_control():
    """The safety property: unbiased for *any* coefficient, because the control
    is centred. A poor control costs the variance reduction and never the
    answer -- which is what makes it safe to use a surrogate as one."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(6000)
    useless = rng.standard_normal(6000)          # independent of the model

    result = control_variates(np.exp(x), useless, control_mean=0.0)
    assert abs(result.correlation) < 0.05
    assert result.mean == pytest.approx(np.exp(0.5), abs=4 * result.std_error)
    assert result.variance_ratio == pytest.approx(1.0, abs=0.05)


def test_a_constant_control_is_refused():
    with pytest.raises(ValueError, match="constant"):
        control_variates([1.0, 2.0, 3.0], [5.0, 5.0, 5.0], control_mean=5.0)


def test_importance_sampling_recovers_a_tail_expectation():
    """Sampling from a shifted proposal and reweighting. The target expectation
    is dominated by a region the target itself rarely visits."""
    target = JointIndependent([Normal(0.0, 1.0)])
    proposal = JointIndependent([Normal(3.0, 1.0)])
    model = lambda x: (x[:, 0] > 3.0).astype(float)

    result = importance_sampling(model, target, proposal, 20000, seed=1,
                                 vectorised=True)
    exact = float(stats.norm.sf(3.0))

    assert result.mean == pytest.approx(exact, rel=0.05)
    assert result.coefficient_of_variation < 0.05


def test_the_effective_sample_size_is_not_a_verdict_on_the_estimate():
    """The standard misreading, worth pinning down.

    Kish's ESS describes how well the proposal substitutes for the target
    *overall*. On a rare-event problem the proposal is deliberately mismatched
    -- that is the whole method -- so the weights are skewed by design and the
    ESS collapses while the answer is excellent. Here: 19 effective samples out
    of 20000, and an estimate good to under a percent. The coefficient of
    variation, which is built from ``w * f`` rather than ``w`` alone, tells the
    truth.
    """
    target = JointIndependent([Normal(0.0, 1.0)])
    proposal = JointIndependent([Normal(3.0, 1.0)])
    model = lambda x: (x[:, 0] > 3.0).astype(float)

    result = importance_sampling(model, target, proposal, 20000, seed=1,
                                 vectorised=True)

    assert result.efficiency < 0.01                      # looks catastrophic
    assert result.coefficient_of_variation < 0.05        # and is not
    assert result.mean == pytest.approx(float(stats.norm.sf(3.0)), rel=0.02)


def test_importance_sampling_reports_when_it_genuinely_has_failed():
    """The case where the weights *and* the estimate are bad together: a
    proposal six standard deviations from the target, estimating a quantity the
    proposal says nothing about."""
    target = JointIndependent([Normal(0.0, 1.0)])
    far = JointIndependent([Normal(6.0, 1.0)])

    result = importance_sampling(lambda x: x[:, 0], target, far, 5000, seed=1,
                                 vectorised=True)

    assert result.efficiency < 0.02
    assert result.coefficient_of_variation > 0.1         # and this one agrees
    assert abs(result.mean) > 0.05                       # true value is zero


def test_stratification_never_does_worse_than_crude():
    """Unlike antithetic and importance sampling, both of which can. The
    variance splits into within plus between strata and stratifying removes the
    between part exactly, so the comparison is one-sided by construction."""
    joint = JointIndependent([Normal(0.0, 1.0), Normal(0.0, 1.0)])
    model = lambda x: np.exp(0.8 * x[:, 0]) + 0.2 * x[:, 1]
    exact = np.exp(0.32)

    strat = [stratified(model, joint, n_per_stratum=20, n_strata=20, axis=0,
                        seed=s, vectorised=True)[0] for s in range(120)]
    crude = [propagate(model, joint, 400, "crude", seed=s,
                       vectorised=True).mean() for s in range(120)]

    assert np.var(strat) < np.var(crude)
    assert np.mean(strat) == pytest.approx(exact, rel=0.02)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

def test_the_required_sample_count_scales_as_one_over_the_probability():
    """The arithmetic that motivates every other method in the module."""
    assert required_samples(1e-2) == pytest.approx(9900, rel=0.01)
    assert required_samples(1e-6) == pytest.approx(1e8, rel=0.01)


@pytest.mark.parametrize("beta", [2.0, 3.0, 4.0, 4.75])
def test_form_is_exact_for_a_linear_limit_state(beta):
    """Exact by construction: the failure surface is a plane in standard-normal
    space, which is what FORM linearises. Anything other than machine agreement
    here is an implementation error, not a modelling one."""
    limit = LinearLimitState.with_beta(beta, dim=4)
    result = form(limit.evaluate, limit.distribution, vectorised=True)

    assert result.beta == pytest.approx(beta, abs=1e-6)
    assert result.probability == pytest.approx(limit.probability, rel=1e-6)
    np.testing.assert_allclose(result.design_point, limit.design_point,
                               atol=1e-5)


def test_form_is_not_exact_for_a_curved_limit_state():
    """The counterpart, and the reason the test above proves correctness rather
    than accuracy. A parabola is not its own tangent plane: FORM finds the right
    design point and the right ``beta``, and still gets the probability wrong,
    because the surface curves away from the origin either side of it."""
    limit = QuadraticLimitState(threshold=3.0, kappa=0.2, dim=3)
    approximate = form(limit.evaluate, limit.distribution, vectorised=True)
    sampled = monte_carlo_probability(limit.evaluate, limit.distribution,
                                      400000, seed=1, vectorised=True)

    assert approximate.beta == pytest.approx(limit.form_beta, abs=1e-4)
    assert sampled.probability > 0.0
    # positive curvature bends the surface away, so the tangent plane
    # overestimates the failure region
    assert approximate.probability > 1.2 * sampled.probability


def test_a_closed_quadratic_surface_has_no_failure_region_at_all():
    """Why the curvature is applied only to the directions orthogonal to
    ``u_1``. Feeding ``u_1`` into the quadratic term as well closes the surface
    off above ``kappa = 1/(4 * threshold)``, and every sampler then correctly
    reports nothing -- an easy way to produce a reference problem whose answer
    is zero without noticing."""
    threshold, kappa = 3.0, 0.15
    u = np.random.default_rng(0).standard_normal((200000, 2))
    closed = threshold - u[:, 0] + kappa * np.sum(u ** 2, axis=1)
    assert closed.min() > 1.0                       # never reaches zero

    limit = QuadraticLimitState(threshold=threshold, kappa=kappa, dim=2)
    assert limit.evaluate(u).min() < 0.0            # the correct form does


def test_crude_monte_carlo_agrees_where_it_can_afford_to():
    limit = LinearLimitState.with_beta(2.0, dim=3)
    result = monte_carlo_probability(limit.evaluate, limit.distribution,
                                     200000, seed=1, vectorised=True)

    assert result.probability == pytest.approx(limit.probability, rel=0.05)
    assert result.cov < 0.05


def test_zero_failures_is_reported_as_a_bound_and_not_as_a_probability():
    """The silent failure this guards against. Too few samples for a rare event
    returns no failures, and ``0.0`` unqualified reads as certainty of
    survival."""
    limit = LinearLimitState.with_beta(5.0, dim=2)
    result = monte_carlo_probability(limit.evaluate, limit.distribution, 2000,
                                     seed=1, vectorised=True)

    assert result.probability == 0.0
    assert result.cov is None
    assert "upper bound" in result.info["note"]
    assert result.info["upper_bound_95"] == pytest.approx(3.0 / 2000)


@pytest.mark.parametrize("beta,min_saving", [(3.0, 10), (4.0, 500),
                                             (4.75, 10000)])
def test_importance_sampling_reaches_probabilities_crude_cannot(beta,
                                                                min_saving):
    """The saving grows as the event gets rarer, because the cost of importance
    sampling is flat in ``P_f`` and crude's is ``1/P_f``. At ``beta = 4.75``
    (``P_f = 1e-6``) crude needs about 1e8 samples and this needs 5000 -- four
    orders of magnitude. At ``beta = 3`` the saving is only about 15x, which is
    why the method is not worth the trouble for common events."""
    limit = LinearLimitState.with_beta(beta, dim=4)
    result = importance_sampling_probability(
        limit.evaluate, limit.distribution, 5000, seed=1, vectorised=True)

    assert result.probability == pytest.approx(limit.probability, rel=0.08)
    assert result.cov < 0.05
    assert result.n_evaluations < 6000
    saving = required_samples(limit.probability) / result.n_evaluations
    assert saving > min_saving


@pytest.mark.parametrize("beta,min_saving", [(3.0, 5), (4.75, 1000)])
def test_subset_simulation_reaches_them_without_a_design_point(beta,
                                                               min_saving):
    """The method's selling point: no linearisation, no design point, and no
    assumption about the shape of the failure region. Cost grows like
    ``log(P_f)`` -- the level count rises and the sample count does not
    explode."""
    limit = LinearLimitState.with_beta(beta, dim=4)
    result = subset_simulation(limit.evaluate, limit.distribution,
                               n_per_level=3000, seed=1, vectorised=True)

    assert result.probability == pytest.approx(limit.probability, rel=0.35)
    assert (required_samples(limit.probability)
            / result.n_evaluations) > min_saving
    assert result.info["n_levels"] >= 2
    assert np.all(np.diff(result.info["thresholds"]) < 0)   # thresholds fall


def test_subset_simulation_labels_its_coefficient_of_variation_as_optimistic():
    """Its levels are not independent -- each is seeded from the last -- so the
    standard formula understates the spread. Saying so beats reporting it as a
    bound."""
    limit = LinearLimitState.with_beta(3.5, dim=3)
    result = subset_simulation(limit.evaluate, limit.distribution,
                               n_per_level=2000, seed=2, vectorised=True)

    assert "optimistic" in result.info["cov_note"]
    assert result.cov > 0.0


def test_the_reliability_index_and_probability_are_consistent():
    limit = LinearLimitState.with_beta(3.2, dim=2)
    result = form(limit.evaluate, limit.distribution, vectorised=True)

    assert result.beta == pytest.approx(-stats.norm.ppf(result.probability),
                                        rel=1e-9)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def test_the_ishigami_reference_matches_its_published_indices():
    """Before using it to check anything else. The published values are
    ``S = (0.3139, 0.4424, 0)`` and ``S_T = (0.5576, 0.4424, 0.2437)``."""
    ishigami = Ishigami()

    assert ishigami.variance == pytest.approx(13.8446, abs=1e-3)
    np.testing.assert_allclose(ishigami.first_order,
                               [0.3139, 0.4424, 0.0], atol=1e-4)
    np.testing.assert_allclose(ishigami.total,
                               [0.5576, 0.4424, 0.2437], atol=1e-4)


def test_the_reference_variance_matches_a_direct_estimate():
    """The analytic variance against a large sample of the function itself, so
    the closed form is not merely self-consistent."""
    ishigami = Ishigami()
    x = ishigami.distribution.sample(400000, np.random.default_rng(0))
    values = ishigami.evaluate(x)

    assert values.var(ddof=1) == pytest.approx(ishigami.variance, rel=0.02)
    assert values.mean() == pytest.approx(ishigami.mean, abs=0.02)


@pytest.mark.parametrize("estimator", ["jansen", "saltelli"])
def test_sobol_indices_recover_the_analytic_values(estimator):
    ishigami = Ishigami()
    result = sobol_indices(ishigami.evaluate, ishigami.distribution,
                           n_base=8192, estimator=estimator, seed=1,
                           vectorised=True)

    np.testing.assert_allclose(result.first_order, ishigami.first_order,
                               atol=0.02)
    np.testing.assert_allclose(result.total, ishigami.total, atol=0.02)
    assert result.n_evaluations == 8192 * 5


def test_the_third_input_matters_only_through_an_interaction():
    """The distinction the Ishigami function exists to make. ``S_3`` is zero and
    ``S_T3`` is a quarter: a first-order study concludes the input is
    irrelevant, and a total index shows it carries 24% of the variance. Only a
    small *total* index licenses fixing an input."""
    ishigami = Ishigami()
    result = sobol_indices(ishigami.evaluate, ishigami.distribution,
                           n_base=8192, seed=1, vectorised=True)

    assert abs(result.first_order[2]) < 0.03
    assert result.total[2] > 0.2
    assert result.interaction[2] > 0.2
    assert result.interaction[1] == pytest.approx(0.0, abs=0.03)


def test_the_first_order_indices_fall_short_of_one_by_the_interaction_share():
    """``sum S_i = 1`` only for an additive model. The shortfall is a
    measurement of the interaction content, not an error."""
    result = sobol_indices(Ishigami().evaluate, Ishigami().distribution,
                           n_base=8192, seed=1, vectorised=True)

    assert result.additive_fraction == pytest.approx(0.757, abs=0.03)
    assert np.sum(result.total) > 1.0


def test_an_additive_model_has_first_order_indices_summing_to_one():
    """The other side of the same statement, on a model built to be additive."""
    joint = JointIndependent([Normal(0.0, 1.0), Normal(0.0, 2.0),
                              Normal(0.0, 3.0)])
    model = lambda x: x[:, 0] + x[:, 1] + x[:, 2]

    result = sobol_indices(model, joint, n_base=4096, seed=1, vectorised=True)

    assert result.additive_fraction == pytest.approx(1.0, abs=0.02)
    np.testing.assert_allclose(result.first_order, result.total, atol=0.02)
    # variance shares are 1 : 4 : 9
    np.testing.assert_allclose(result.first_order,
                               np.array([1.0, 4.0, 9.0]) / 14.0, atol=0.02)


def test_the_paired_design_is_what_makes_the_estimator_work():
    """``AB_i`` must differ from ``A`` in exactly one column. Rebuilding it from
    fresh points destroys the pairing the correlation argument relies on, and
    the indices become noise."""
    ishigami = Ishigami()
    rng = np.random.default_rng(0)
    joint = ishigami.distribution
    n = 4096

    A = joint.sample(n, rng)
    B = joint.sample(n, rng)
    fA, fB = ishigami.evaluate(A), ishigami.evaluate(B)
    variance = np.var(np.concatenate([fA, fB]), ddof=1)

    paired = A.copy()
    paired[:, 1] = B[:, 1]
    unpaired = joint.sample(n, rng)          # independent, not a column swap

    good = np.mean((fA - ishigami.evaluate(paired)) ** 2) / (2 * variance)
    bad = np.mean((fA - ishigami.evaluate(unpaired)) ** 2) / (2 * variance)

    assert good == pytest.approx(ishigami.total[1], abs=0.05)
    assert bad > 0.9                          # meaningless, near one by chance


def test_bootstrap_intervals_bracket_the_analytic_indices():
    """Without an interval there is nothing to say whether a small index is
    small or is noise."""
    ishigami = Ishigami()
    result = sobol_indices(ishigami.evaluate, ishigami.distribution,
                           n_base=2048, seed=1, vectorised=True,
                           n_bootstrap=200)

    assert result.first_order_ci.shape == (2, 3)
    low, high = result.total_ci
    assert np.all(low <= ishigami.total + 0.05)
    assert np.all(high >= ishigami.total - 0.05)


def test_a_constant_model_cannot_have_its_variance_decomposed():
    joint = JointIndependent([Uniform(0.0, 1.0), Uniform(0.0, 1.0)])
    with pytest.raises(ValueError, match="constant"):
        sobol_indices(lambda x: np.ones(x.shape[0]), joint, n_base=64,
                      vectorised=True)


def test_an_unknown_sensitivity_estimator_is_refused():
    joint = JointIndependent([Uniform(0.0, 1.0)])
    with pytest.raises(NotImplementedError, match="Unknown estimator"):
        sobol_indices(lambda x: x[:, 0], joint, n_base=64, estimator="fast",
                      vectorised=True)


def test_the_summary_names_the_inputs():
    ishigami = Ishigami()
    text = sobol_indices(ishigami.evaluate, ishigami.distribution, n_base=256,
                         seed=1, vectorised=True).summary()
    assert "x3" in text and "S_Ti" in text
