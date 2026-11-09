"""
Verification for the probability density evolution method.

The method's output is a *density*, so the checks have to be about a density.
Moments are not enough: a scheme with the wrong dissipation reproduces the
mean and the standard deviation of Li & Chen's Example 1 to a fraction of a
percent while getting the shape visibly wrong, which is the whole quantity the
method exists to produce.

Four things, each failing differently:

**Conservation** of total probability, at round-off. It is structural in the
finite volume path -- the velocity is uniform in space, so the conservative
and non-conservative forms of the GDEE coincide -- which makes a drift a
boundary problem rather than an accuracy one.

**The closed-form PDF** of the appendix, in ``L1``, and that it converges when
the point set and the response grid are refined *together* -- refining the grid
alone past roughly five cells per point makes the answer worse, because the
scheme's dissipation is what smooths the finite set of trajectories into a
density, and that is asserted here too.

**The limiter is load-bearing**, as a regression: the unlimited scheme must
fail, because the paper's reason for using TVD is that Lax-Wendroff cannot
preserve a non-negative PDF.

**A single trajectory**, which is the cheapest check that the velocity enters
with the right sign -- the density's centroid must follow the deterministic
response.

python -m pytest tests/test_uq/test_gdee.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.uq.density import (
    OMEGA_BOUNDS,
    ProbabilityNotConserved,
    X0,
    lattice,
    response_histories,
    sdof_displacement_moments,
    sdof_displacement_pdf,
    sdof_response,
    solve_gdee,
)
from scion.uq.forward.distributions import Uniform

A, B = OMEGA_BOUNDS


def _frequency(n_points):
    return lattice(Uniform(A, B), n_points, spacing="quantile")


def _solve(n_points=40, n_cells=100, t_end=1.0, n_times=101, **kwargs):
    t_grid = np.linspace(0.0, t_end, n_times)
    points = _frequency(n_points)
    x, x_dot = response_histories(points, sdof_response, t_grid)
    return solve_gdee(points, t_grid, x, x_dot, n_cells=n_cells, **kwargs)


def _l1(solution, time):
    """``L1`` distance to the exact density, away from its integrable poles.

    The exact PDF diverges as ``1/sqrt(x0^2 - x^2)`` at ``+-x0``, where a
    cell-centred density is finite by construction, so the comparison is made
    on ``|x| < 0.95 x0``. That is a property of the exact solution and not a
    tolerance being relaxed to pass.
    """
    mask = np.abs(solution.x) < 0.95 * X0
    exact = sdof_displacement_pdf(solution.x, time)
    return float(np.abs(solution.at(time) - exact)[mask].sum() * solution.dx)


class TestConservation:
    def test_total_probability_is_one_at_every_output_time(self):
        solution = _solve()
        assert np.abs(solution.mass - 1.0).max() < 1e-12

    def test_the_density_is_non_negative(self):
        solution = _solve()
        assert solution.pdf.min() > -1e-14

    def test_the_initial_condition_carries_the_assigned_probabilities(self):
        """Eq. (24) puts ``P_q`` in one cell, so the sum over points needs no
        further weight -- and the first row integrates to one exactly."""
        solution = _solve(n_points=17, n_cells=64)
        assert solution.mass[0] == pytest.approx(1.0, abs=1e-14)


class TestAgainstTheAnalyticSolution:
    def test_the_density_matches_the_appendix(self):
        solution = _solve(n_points=40, n_cells=100)
        assert _l1(solution, 1.0) < 0.12

    def test_it_converges_when_both_error_sources_are_refined(self):
        """Both together, which is the only direction that converges -- see
        the measured table in the module docstring. ~1.8x a doubling at
        larger sizes; the cheap sizes here give 1.5-1.8x.
        """
        errors = []
        for n_points, n_cells in ((30, 80), (60, 160), (120, 320)):
            errors.append(_l1(_solve(n_points=n_points, n_cells=n_cells,
                                     t_end=0.5, n_times=51), 0.5))
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        assert all(r > 1.3 for r in ratios), (errors, ratios)
        assert errors[-1] < 0.09, errors

    def test_a_finer_grid_alone_makes_it_worse(self):
        """The implicit-bandwidth property, as a test rather than a remark.

        With only 20 representative points, going from 100 to 400 cells
        resolves the individual smeared trajectories instead of a density and
        the L1 error triples (0.101 -> 0.308 measured). Anyone who "improves"
        this module by refining the default grid would break this.
        """
        coarse = _l1(_solve(n_points=20, n_cells=100), 1.0)
        fine = _l1(_solve(n_points=20, n_cells=400), 1.0)
        assert fine > 2.0 * coarse, (coarse, fine)

    def test_a_grid_too_coarse_erases_the_point_set(self):
        """The other end of the same trade-off: at 50 cells the error is the
        same to three decimals for 20 points and for 320, because dissipation
        has smoothed away everything the extra points bought."""
        few = _l1(_solve(n_points=20, n_cells=50), 1.0)
        many = _l1(_solve(n_points=160, n_cells=50), 1.0)
        assert abs(few - many) < 0.01 * few, (few, many)

    def test_the_first_two_moments_match_the_closed_form(self):
        """The paper's Fig. 2. Note the mean of an *undamped* system decays --
        an ensemble effect, not damping -- so a scheme that merely conserved
        the mean would fail this."""
        solution = _solve()
        mean, std = sdof_displacement_moments(solution.t)
        assert np.abs(solution.mean - mean).max() < 2e-3 * X0 / 0.1
        assert np.abs(solution.std - std).max() < 6e-3 * X0 / 0.1

    def test_the_mean_really_does_decay(self):
        """Guard on the test above: if the exact and computed means were both
        wrong in the same way the comparison would still pass, so assert the
        qualitative signature the paper reports separately."""
        solution = _solve(t_end=4.0, n_times=161, n_cells=100)
        early = np.abs(solution.mean[solution.t < 0.5]).max()
        late = np.abs(solution.mean[solution.t > 3.0]).max()
        assert late < 0.5 * early, (early, late)


class TestTheLimiterIsLoadBearing:
    """The paper's stated reason for TVD, as a regression rather than a note.

    Lax-Wendroff is second order and dispersive and cannot keep a PDF
    non-negative; with the flux limiter it can. If `solve_gdee` were rewritten
    on an unlimited scheme these would be the tests that caught it.
    """

    def test_the_unlimited_scheme_fails_outright(self):
        with pytest.raises(ProbabilityNotConserved):
            _solve(limiter="none")

    def test_a_diffusive_limiter_spreads_the_density_off_the_grid(self):
        # minmod is the most diffusive TVD limiter here; at this resolution it
        # smears the density to the boundary, where the zero-gradient end then
        # feeds the domain -- it comes out at 1.07, not 0.93
        with pytest.raises(ProbabilityNotConserved, match="left the response"):
            _solve(limiter="minmod")

    def test_superbee_survives_the_same_configuration(self):
        assert np.abs(_solve(limiter="superbee").mass - 1.0).max() < 1e-12


class TestASingleTrajectory:
    def test_the_density_follows_the_deterministic_response(self):
        """One representative point: the exact answer is a Dirac riding the
        trajectory, so the centroid is the trajectory. A sign error on the
        velocity sends it the other way and nothing else here would notice.
        """
        t_grid = np.linspace(0.0, 1.0, 101)
        points = lattice(Uniform(1.5 * np.pi, 1.5 * np.pi + 1e-12), 1)
        x, x_dot = response_histories(points, sdof_response, t_grid)
        solution = solve_gdee(points, t_grid, x, x_dot, n_cells=300,
                              margin=0.2)
        assert np.abs(solution.mean - x[0]).max() < 0.01 * X0

    def test_a_zero_velocity_leaves_the_density_where_it_started(self):
        t_grid = np.linspace(0.0, 1.0, 11)
        points = _frequency(5)
        x = np.zeros((5, 11))
        solution = solve_gdee(points, t_grid, x, np.zeros_like(x),
                              n_cells=32, margin=1.0)
        assert np.array_equal(solution.pdf[0], solution.pdf[-1])


class TestRefusals:
    def test_a_zero_margin_is_refused(self):
        with pytest.raises(ValueError, match="margin must be positive"):
            _solve(margin=0.0)

    def test_mismatched_history_shapes_are_refused(self):
        points = _frequency(4)
        t_grid = np.linspace(0.0, 1.0, 11)
        with pytest.raises(ValueError, match="x should be"):
            solve_gdee(points, t_grid, np.zeros((3, 11)), np.zeros((3, 11)))

    def test_a_velocity_history_of_the_wrong_shape_is_refused(self):
        points = _frequency(4)
        t_grid = np.linspace(0.0, 1.0, 11)
        with pytest.raises(ValueError, match="same shape"):
            solve_gdee(points, t_grid, np.zeros((4, 11)), np.zeros((4, 10)))

    def test_non_increasing_times_are_refused(self):
        points = _frequency(4)
        bad = np.array([0.0, 0.1, 0.1, 0.2])
        with pytest.raises(ValueError, match="strictly increasing"):
            solve_gdee(points, bad, np.zeros((4, 4)), np.zeros((4, 4)))

    def test_a_single_output_time_is_refused(self):
        points = _frequency(4)
        with pytest.raises(ValueError, match="at least two output times"):
            solve_gdee(points, np.array([0.0]), np.zeros((4, 1)),
                       np.zeros((4, 1)))

    def test_too_few_cells_is_refused(self):
        with pytest.raises(ValueError, match="at least four cells"):
            _solve(n_cells=3)

    def test_a_non_finite_response_is_refused(self):
        points = _frequency(3)

        def broken(z, t_grid):
            out = np.zeros(np.size(t_grid))
            out[-1] = np.nan
            return out, out

        with pytest.raises(ValueError, match="non-finite"):
            response_histories(points, broken, np.linspace(0.0, 1.0, 5))

    def test_the_pdf_at_time_zero_is_refused_by_the_reference(self):
        with pytest.raises(ValueError, match="Dirac"):
            sdof_displacement_pdf(np.array([0.0]), 0.0)
