"""
Verification for the representative point sets both density methods start from.

The invariant under test is ``sum_q P_q == 1``, and the reason it gets its own
file is that violating it produces no symptom: every density PDEM or DPIM
builds is then scaled by the same wrong constant, integrates to that constant,
and still looks like a probability density.

Four things are checked, each failing differently:

**The invariant itself**, across distributions, dimensions and both spacings.

**Against the paper's nominal weighting**, as a regression rather than a
description: the midpoint-rule weights Li & Chen use must be *detectably* off
where ours are exact, or the design decision is untested.

**Moment reproduction**, which is the cheapest statement of point-set quality
-- a set that cannot integrate its own input moments will not integrate a
response -- and the analytic ``E[cos(omega t)]`` of the paper's own Example 1,
which is the quadrature PDEM actually performs.

**The refusals**, because a silently accepted bad weight set is the whole
failure mode.

python -m pytest tests/test_uq/test_density_points.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.stats import qmc

from scion.uq.density import (
    PROBABILITY_TOL,
    RepresentativePoints,
    gf_discrepancy,
    gf_points,
    lattice,
)
from scion.uq.forward.distributions import (
    Exponential,
    JointIndependent,
    LogNormal,
    Normal,
    Uniform,
)

SPACINGS = ("quantile", "uniform")


def _distributions():
    return [
        ("normal", Normal(0.0, 1.0)),
        ("uniform", Uniform(5 * np.pi / 4, 7 * np.pi / 4)),
        ("lognormal", LogNormal.from_moments(1.0, 0.3)),
        ("lognormal_cov1", LogNormal.from_moments(1.0, 1.0)),
        ("exponential", Exponential(1.0)),
    ]


class TestTheInvariant:
    @pytest.mark.parametrize("spacing", SPACINGS)
    @pytest.mark.parametrize("name,variable", _distributions())
    def test_assigned_probabilities_sum_to_one(self, name, variable, spacing):
        points = lattice(variable, 16, spacing=spacing)
        assert abs(points.probability.sum() - 1.0) <= PROBABILITY_TOL

    @pytest.mark.parametrize("spacing", SPACINGS)
    def test_the_invariant_survives_the_tensor_product(self, spacing):
        # the product is where a per-axis error would be cubed
        joint = JointIndependent([Normal(0.0, 1.0),
                                  Exponential(1.0),
                                  LogNormal.from_moments(1.0, 1.0)])
        points = lattice(joint, (6, 7, 8), spacing=spacing)
        assert points.n_points == 6 * 7 * 8
        assert abs(points.probability.sum() - 1.0) <= PROBABILITY_TOL

    def test_quantile_weights_are_exactly_equal(self):
        points = lattice(Normal(0.0, 1.0), 25, spacing="quantile")
        assert np.allclose(points.probability, 1.0 / 25, rtol=0, atol=1e-15)

    def test_every_probability_is_non_negative(self):
        for _, variable in _distributions():
            for spacing in SPACINGS:
                p = lattice(variable, 12, spacing=spacing).probability
                assert np.all(p >= 0.0)


class TestAgainstTheNominalWeighting:
    """The paper's ``p_Z(z_q)|dz_q|`` against the exact cell probability.

    This is the one test that would fail if `lattice` were rewritten to follow
    the paper literally, which is what makes it a regression.
    """

    @staticmethod
    def _nominal(variable, count, truncation):
        lo = float(variable.ppf(truncation))
        hi = float(variable.ppf(1.0 - truncation))
        edges = np.linspace(lo, hi, count + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return float((np.asarray(variable.pdf(centres)).ravel()
                      * (edges[1] - edges[0])).sum())

    def test_the_nominal_form_misses_by_percent_on_a_skewed_input(self):
        variable = LogNormal.from_moments(1.0, 1.0)
        nominal = self._nominal(variable, 10, 1e-3)
        # it overshoots here, so a one-sided check would pass by luck
        assert abs(nominal - 1.0) > 0.1
        exact = lattice(variable, 10, spacing="uniform", truncation=1e-3)
        assert abs(exact.probability.sum() - 1.0) <= PROBABILITY_TOL

    def test_tightening_the_truncation_makes_the_nominal_error_worse(self):
        """The nominal error is quadrature, not a discarded tail.

        Exponential(1) at 20 points: ``|1 - sum|`` runs 2.2e-2, 9.0e-3,
        2.0e-2, 3.4e-2, 7.5e-2 as the truncation goes 1e-2 to 1e-12. It is
        non-monotone and ends up *larger*, because a tighter truncation
        stretches the same 20 cells over a longer range and the midpoint rule
        pays for the width. So no choice of truncation rescues the nominal
        weighting, which is the argument for the exact form.
        """
        variable = Exponential(1.0)
        truncations = (1e-2, 1e-4, 1e-6, 1e-8, 1e-12)
        errors = [abs(self._nominal(variable, 20, tr) - 1.0)
                  for tr in truncations]
        assert min(errors) > 5e-3, errors
        assert errors[-1] > errors[0], errors

        # the exact form is flat across the same sweep
        for tr in truncations:
            points = lattice(variable, 20, spacing="uniform", truncation=tr)
            assert abs(points.probability.sum() - 1.0) <= PROBABILITY_TOL


class TestMoments:
    @pytest.mark.parametrize("spacing", SPACINGS)
    def test_a_symmetric_input_recovers_its_mean(self, spacing):
        points = lattice(Normal(2.5, 1.5), 40, spacing=spacing)
        assert abs(float(points.mean[0]) - 2.5) < 1e-10

    @pytest.mark.parametrize("spacing", SPACINGS)
    def test_the_standard_deviation_converges_under_refinement(self, spacing):
        variable = Uniform(5 * np.pi / 4, 7 * np.pi / 4)
        errors = [abs(float(lattice(variable, n, spacing=spacing).std[0])
                      - variable.std) for n in (10, 20, 40, 80)]
        assert errors[-1] < errors[0]
        assert errors[-1] < 1e-3

    def test_it_integrates_the_response_of_the_papers_first_example(self):
        """``E[x0 cos(omega t)]`` for ``omega ~ U[5pi/4, 7pi/4]``.

        The paper's Example 1 mean, which has the closed form
        ``x0 (sin(b t) - sin(a t)) / ((b - a) t)`` and which PDEM must get
        right through this quadrature before any PDE solve can help it.
        """
        a, b, x0 = 5 * np.pi / 4, 7 * np.pi / 4, 0.1
        # NOT t = 1: there sin(b t) == sin(a t), so the exact mean is
        # identically zero and the quadrature returns zero by symmetry
        # whatever its accuracy -- the test would measure nothing. See
        # test_the_mean_is_an_exact_null_at_one_second for that case.
        t = 0.37
        exact = x0 * (np.sin(b * t) - np.sin(a * t)) / ((b - a) * t)

        errors = []
        for n in (20, 40, 80, 160):
            points = lattice(Uniform(a, b), n, spacing="quantile")
            response = x0 * np.cos(points.z[:, 0] * t)
            errors.append(abs(float(points.expectation(response)) - exact))

        assert errors[-1] < 1e-8, errors
        # midpoint quadrature on a smooth integrand: second order
        rate = np.log2(errors[0] / errors[-1]) / 3.0
        assert 1.9 < rate < 2.1, (errors, rate)

    def test_the_mean_is_an_exact_null_at_one_second(self):
        """``sin(7pi/4) == sin(5pi/4)``, so ``E[cos(omega t)] = 0`` exactly at
        ``t = 1`` -- and the quantile points are symmetric about ``3pi/2``, so
        the quadrature cancels pairwise rather than merely converging."""
        a, b, x0 = 5 * np.pi / 4, 7 * np.pi / 4, 0.1
        points = lattice(Uniform(a, b), 64, spacing="quantile")
        value = float(points.expectation(x0 * np.cos(points.z[:, 0] * 1.0)))
        assert abs(value) < 1e-15, value


class TestTheContainer:
    def test_expectation_handles_a_vector_valued_response(self):
        points = lattice(Normal(0.0, 1.0), 9)
        values = np.column_stack([points.z[:, 0], points.z[:, 0] ** 2])
        out = points.expectation(values)
        assert out.shape == (2,)
        assert abs(out[0]) < 1e-12

    def test_a_single_random_variable_needs_no_joint_wrapper(self):
        assert lattice(Normal(0.0, 1.0), 5).dim == 1

    def test_one_point_per_axis_is_the_degenerate_case_not_an_error(self):
        points = lattice(Normal(3.0, 1.0), 1)
        assert points.n_points == 1
        assert points.probability[0] == pytest.approx(1.0)
        assert float(points.z[0, 0]) == pytest.approx(3.0)

    def test_it_is_frozen(self):
        points = lattice(Normal(0.0, 1.0), 4)
        with pytest.raises(Exception):
            points.z = np.zeros((4, 1))


class TestRefusals:
    def test_probabilities_that_do_not_sum_to_one_are_refused(self):
        z = np.zeros((3, 1))
        with pytest.raises(ValueError, match="sum to"):
            RepresentativePoints(z, np.array([0.3, 0.3, 0.3]))

    def test_a_negative_probability_is_refused(self):
        z = np.zeros((3, 1))
        with pytest.raises(ValueError, match="non-negative"):
            RepresentativePoints(z, np.array([1.2, -0.1, -0.1]))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="against"):
            RepresentativePoints(np.zeros((3, 1)), np.array([0.5, 0.5]))

    def test_a_non_finite_point_is_refused(self):
        with pytest.raises(ValueError, match="finite"):
            RepresentativePoints(np.array([[0.0], [np.inf]]),
                                 np.array([0.5, 0.5]))

    def test_an_unknown_spacing_is_refused(self):
        with pytest.raises(ValueError, match="spacing"):
            lattice(Normal(0.0, 1.0), 5, spacing="lattice")

    @pytest.mark.parametrize("truncation", [0.0, 0.5, 0.9, -1e-3])
    def test_a_truncation_outside_the_open_unit_half_is_refused(self,
                                                                truncation):
        with pytest.raises(ValueError, match="truncation"):
            lattice(Normal(0.0, 1.0), 5, spacing="uniform",
                    truncation=truncation)

    def test_zero_points_on_an_axis_is_refused(self):
        with pytest.raises(ValueError, match="at least one point"):
            lattice(Normal(0.0, 1.0), 0)

    def test_quantile_spacing_needs_no_truncation_on_an_unbounded_support(self):
        # the point of the default: no tail to discard, nothing to choose
        points = lattice(Normal(0.0, 1.0), 200, spacing="quantile")
        assert np.all(np.isfinite(points.z))


class TestGFDiscrepancyPoints:
    """Chen & Yang's Appendix A, and the identity that pins it.

    The reason to have this at all is the count: ``N`` points whatever the
    dimension, against the tensor lattice's ``N ** dim``. The reason to trust
    it is that after Eq. (A5) the discrepancy has a closed form,
    ``D_GF = max_q P_q / 2``, so there is something exact to assert rather
    than a number to eyeball.
    """

    def _joint(self, dim):
        pieces = [Normal(0.0, 1.0), Uniform(0.0, 1.0), Normal(2.0, 3.0),
                  LogNormal.from_moments(1.0, 0.3), Exponential(1.0),
                  Uniform(-2.0, 5.0)]
        return JointIndependent(pieces[:dim])

    @pytest.mark.parametrize("dim", [1, 2, 3, 6])
    @pytest.mark.parametrize("n_points", [128, 512])
    def test_the_discrepancy_is_half_the_largest_assigned_probability(
            self, dim, n_points):
        """The identity Eq. (A5) constructs, and the sharpest check here: it
        ties the point placement, the weights and the discrepancy measure
        together, and any one of them being wrong breaks it."""
        distribution = self._joint(dim)
        points = gf_points(distribution, n_points)
        assert gf_discrepancy(points, distribution) == pytest.approx(
            points.probability.max() / 2.0, rel=1e-9)

    def test_one_dimension_attains_the_floor(self):
        variable = Uniform(1.25 * np.pi, 1.75 * np.pi)
        for n_points in (128, 512, 1024):
            points = gf_points(variable, n_points)
            assert gf_discrepancy(points, variable) == pytest.approx(
                1.0 / (2 * n_points), rel=1e-6)

    def test_one_dimension_reproduces_the_quantile_lattice(self):
        """Not a coincidence: Eq. (A4) places the marginal on the
        equal-probability quantiles, and for a uniform input Eq. (A5) is then
        a fixed point. Worth asserting because it means a difference between
        two one-dimensional results cannot be blamed on the point set.
        """
        variable = Uniform(1.25 * np.pi, 1.75 * np.pi)
        gf = gf_points(variable, 512)
        quantile = lattice(variable, 512, spacing="quantile")
        assert np.allclose(np.sort(gf.z[:, 0]), np.sort(quantile.z[:, 0]),
                           rtol=0, atol=1e-12)

    def test_the_point_count_does_not_grow_with_the_dimension(self):
        """The whole reason for this function. A tensor lattice of 512 points
        per axis in six dimensions is 1.4e16 analyses."""
        for dim in (1, 3, 6):
            assert gf_points(self._joint(dim), 256).n_points == 256

    def test_the_discrepancy_halves_when_the_count_doubles(self):
        distribution = self._joint(3)
        previous = None
        for n_points in (128, 256, 512, 1024):
            current = gf_discrepancy(gf_points(distribution, n_points),
                                     distribution)
            if previous is not None:
                assert 1.6 < previous / current < 2.5, (previous, current)
            previous = current

    def test_it_improves_on_the_raw_sobol_set_in_low_dimension(self):
        distribution = self._joint(3)
        n_points = 512
        points = gf_points(distribution, n_points)

        rng = np.random.default_rng(0)
        unit = qmc.Sobol(d=3, scramble=True, seed=rng).random(n_points)
        raw = RepresentativePoints(
            np.column_stack([np.asarray(v.ppf(unit[:, i])).ravel()
                             for i, v in enumerate(distribution.variables)]),
            np.full(n_points, 1.0 / n_points))

        assert (gf_discrepancy(points, distribution)
                < 0.9 * gf_discrepancy(raw, distribution))

    def test_the_probabilities_still_sum_to_one(self):
        for dim in (1, 3, 6):
            points = gf_points(self._joint(dim), 256)
            assert abs(points.probability.sum() - 1.0) <= PROBABILITY_TOL

    def test_it_reproduces_the_input_moments(self):
        distribution = self._joint(3)
        points = gf_points(distribution, 1024)
        assert np.allclose(points.mean, distribution.mean, atol=0.05)
        assert np.allclose(points.std, distribution.std, rtol=0.1)

    def test_a_count_that_is_not_a_power_of_two_warns(self):
        with pytest.warns(RuntimeWarning, match="powers of two"):
            gf_points(Normal(0.0, 1.0), 500)

    def test_too_few_points_is_refused(self):
        with pytest.raises(ValueError, match="at least two points"):
            gf_points(Normal(0.0, 1.0), 1)

    def test_an_empty_voronoi_cell_is_refused(self):
        with pytest.raises(ValueError, match="no sample"):
            gf_points(self._joint(6), 256, n_samples=64)

    def test_the_discrepancy_refuses_a_mismatched_distribution(self):
        points = gf_points(self._joint(3), 64)
        with pytest.raises(ValueError, match="dimensional points"):
            gf_discrepancy(points, self._joint(2))
