"""
Verification for the direct probability integral method.

DPIM is a probability-weighted kernel estimate, so the checks are about the
bandwidth as much as about the density. Six things, each failing differently:

**The closed form**, in ``L1``, on the same SDOF problem PDEM is tested
against -- Chen & Yang's Example 2 is Li & Chen's Example 1.

**Agreement with PDEM**, which is the cross-check neither method gives alone:
two independent discretisations of the same probability statement, one a
transported PDE and one a kernel sum, must land on the same density.

**The variance identity.** A Gaussian kernel adds exactly ``sigma^2`` to the
variance, so ``Var_DPIM - Var_exact - sigma^2`` is independent of ``sigma``.
That is an identity rather than a tolerance, and it pins the normalisation,
the weights and the kernel width together.

**The grid is a rendering choice**, which is the structural difference from
PDEM: doubling the cells changes a DPIM density by 5e-3 relative and a PDEM
density by 5e-1.

**The bandwidth must scale with the point count**, or the method stops
converging -- the paper's fixed rule plateaus where the spacing rule halves.

**The kernel must span a cell**, below which the midpoint quadrature fails
abruptly rather than gradually.

python -m pytest tests/test_uq/test_dpim.py -q
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from scion.uq.density import (
    OMEGA_BOUNDS,
    gf_points,
    knn_bandwidth,
    lattice,
    spacing_uniformity,
    ProbabilityNotConserved,
    X0,
    paper_bandwidth,
    response_histories,
    response_spacing,
    sdof_displacement_moments,
    sdof_displacement_pdf,
    SIGMA_FACTOR,
    sdof_response,
    solve_dpim,
    vanderpol_coefficient,
    vanderpol_pdf,
    solve_gdee,
)
from scion.uq.forward.distributions import (
    JointIndependent,
    LogNormal,
    Normal,
    Uniform,
)

A, B = OMEGA_BOUNDS


def _histories(n_points, t_end=1.0, n_times=101):
    t_grid = np.linspace(0.0, t_end, n_times)
    points = lattice(Uniform(A, B), n_points, spacing="quantile")
    x, x_dot = response_histories(points, sdof_response, t_grid)
    return points, t_grid, x, x_dot


def _l1(solution, time):
    """``L1`` to the exact density away from its integrable poles at ``+-x0``."""
    mask = np.abs(solution.x) < 0.95 * X0
    exact = sdof_displacement_pdf(solution.x, time)
    return float(np.abs(solution.at(time) - exact)[mask].sum() * solution.dx)


class TestAgainstTheAnalyticSolution:
    def test_the_density_matches_the_appendix(self):
        points, t_grid, x, _ = _histories(160)
        assert _l1(solve_dpim(points, x, t_grid), 1.0) < 0.02

    def test_it_converges_in_the_point_count(self):
        errors = []
        for n_points in (40, 80, 160, 320):
            points, t_grid, x, _ = _histories(n_points)
            errors.append(_l1(solve_dpim(points, x, t_grid, n_cells=800), 1.0))
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1)), \
            errors
        assert errors[0] / errors[-1] > 4.0, errors

    def test_the_mean_is_unbiased(self):
        points, t_grid, x, _ = _histories(160)
        solution = solve_dpim(points, x, t_grid, n_cells=800)
        mean, _ = sdof_displacement_moments(t_grid)
        assert np.abs(solution.mean - mean).max() < 1e-5


class TestTheVarianceIdentity:
    """A Gaussian kernel adds exactly ``sigma^2`` to the variance.

    So the residual below is the point set's own quadrature error and must not
    move with ``sigma`` at all. It is the sharpest single check here: it pins
    the kernel normalisation, the assigned probabilities and the width
    together, and any one of them being wrong breaks it.
    """

    def test_the_kernel_adds_exactly_sigma_squared(self):
        points, t_grid, x, _ = _histories(160)
        _, std = sdof_displacement_moments(t_grid)

        residuals = []
        for sigma in (2e-4, 5e-4, 1e-3, 2e-3, 5e-3):
            solution = solve_dpim(points, x, t_grid, n_cells=1200,
                                  sigma=sigma)
            residuals.append(
                np.abs(solution.std ** 2 - std ** 2 - sigma ** 2).max())

        assert max(residuals) < 1e-6, residuals
        # independent of sigma across a 25x range, to three figures
        assert max(residuals) / min(residuals) < 1.01, residuals

    def test_the_variance_is_inflated_not_deflated(self):
        """Direction matters: a smoothed density is wider, never narrower,
        so DPIM's standard deviation is biased high and correcting it means
        subtracting sigma^2 rather than adding it."""
        points, t_grid, x, _ = _histories(80)
        solution = solve_dpim(points, x, t_grid, sigma=4e-3)
        _, std = sdof_displacement_moments(t_grid)
        assert np.all(solution.std >= std - 1e-9)


class TestTheGridIsARenderingChoice:
    """The structural difference from PDEM, asserted on both methods at once.

    In DPIM the bandwidth is ``sigma`` and the grid only samples a continuous
    function; in PDEM the grid *is* the bandwidth, through the scheme's
    dissipation. Refining it therefore changes one answer and not the other.
    """

    @staticmethod
    def _probe(solution, time=1.0):
        probe = np.linspace(-0.065, 0.065, 25)
        return np.interp(probe, solution.x, solution.at(time))

    def test_doubling_the_cells_barely_moves_a_dpim_density(self):
        points, t_grid, x, _ = _histories(40)
        coarse = self._probe(solve_dpim(points, x, t_grid, n_cells=200,
                                        sigma=2e-3))
        fine = self._probe(solve_dpim(points, x, t_grid, n_cells=800,
                                      sigma=2e-3))
        assert np.abs(coarse - fine).max() / np.abs(fine).max() < 0.02

    def test_and_moves_a_pdem_density_a_hundred_times_more(self):
        points, t_grid, x, x_dot = _histories(40)
        coarse = self._probe(solve_gdee(points, t_grid, x, x_dot,
                                        n_cells=200))
        fine = self._probe(solve_gdee(points, t_grid, x, x_dot, n_cells=800))
        assert np.abs(coarse - fine).max() / np.abs(fine).max() > 0.2


class TestTheBandwidthRule:
    def test_the_papers_fixed_rule_stops_converging(self):
        """``sigma = 0.01 max|y|`` does not depend on N, so the smoothing bias
        floors the error: 0.0168 / 0.0143 / 0.0145 at 80 / 160 / 320 points,
        where the spacing rule keeps halving. A regression against anyone
        restoring the published default."""
        fixed, scaled = [], []
        for n_points in (80, 160, 320):
            points, t_grid, x, _ = _histories(n_points)
            fixed.append(_l1(solve_dpim(points, x, t_grid, n_cells=1600,
                                        sigma=paper_bandwidth(x)), 1.0))
            scaled.append(_l1(solve_dpim(points, x, t_grid, n_cells=1600),
                              1.0))

        assert fixed[-1] > 0.5 * fixed[0], fixed        # plateaued
        assert scaled[-1] < 0.4 * scaled[0], scaled     # still converging
        assert scaled[-1] < fixed[-1]

    def test_the_spacing_falls_like_one_over_n(self):
        gaps = []
        for n_points in (40, 80, 160):
            _, _, x, _ = _histories(n_points)
            gaps.append(response_spacing(x))
        assert gaps[0] / gaps[1] == pytest.approx(2.0, rel=0.05)
        assert gaps[1] / gaps[2] == pytest.approx(2.0, rel=0.05)


class TestAgainstPDEM:
    def test_the_two_methods_agree(self):
        """Two independent discretisations of the same probability statement:
        a transported PDE and a kernel sum. Away from the density's jumps,
        where both are smoothing a discontinuity differently, they must agree.
        """
        points, t_grid, x, x_dot = _histories(160)
        dpim = solve_dpim(points, x, t_grid, n_cells=400)
        pdem = solve_gdee(points, t_grid, x, x_dot, n_cells=400)

        probe = np.linspace(-0.065, 0.065, 29)
        a = np.interp(probe, dpim.x, dpim.at(1.0))
        b = np.interp(probe, pdem.x, pdem.at(1.0))
        assert np.abs(a - b).max() / np.abs(b).max() < 0.05

    def test_dpim_needs_no_velocity_history(self):
        """The narrower model seam, asserted by never passing one."""
        points, t_grid, x, _ = _histories(40)
        assert solve_dpim(points, x, t_grid).pdf.shape[0] == t_grid.size


class TestStaticSystems:
    def test_a_time_independent_system_is_one_instant(self):
        """No time derivative in the formulation, so a static problem needs no
        time grid -- which PDEM, being a PDE in t, cannot say."""
        points = lattice(Normal(0.0, 1.0), 200, spacing="quantile")
        y = points.z[:, 0] ** 2          # chi-square with one degree of freedom
        with pytest.warns(RuntimeWarning, match="unevenly spread"):
            solution = solve_dpim(points, y, n_cells=600)

        assert solution.pdf.shape[0] == 1
        assert abs(solution.mass[0] - 1.0) < 1e-3
        # mean of a chi-square(1) is 1; the kernel does not bias the mean
        assert float(solution.mean[0]) == pytest.approx(1.0, rel=0.02)


class TestASecondClosedForm:
    """The SDOF problem is one problem; these are two more, with the second
    marking the method's limit.

    Everything above rests on a single oscillator. A triangular density from
    a two-dimensional input exercises the tensor-product point set as well,
    and a chi-square from an even mapping shows where a global bandwidth
    gives up.
    """

    def test_the_sum_of_two_uniforms_is_triangular(self):
        errors = []
        for per_axis in (30, 60, 120):
            points = lattice(
                JointIndependent([Uniform(0.0, 1.0), Uniform(0.0, 1.0)]),
                per_axis, spacing="quantile")
            y = points.z[:, 0] + points.z[:, 1]
            solution = solve_dpim(points, y, n_cells=2000)

            exact = np.where(solution.x <= 1.0, solution.x, 2.0 - solution.x)
            exact = np.where((solution.x >= 0.0) & (solution.x <= 2.0),
                             np.clip(exact, 0.0, None), 0.0)
            errors.append(float(np.abs(solution.at(0.0) - exact).sum()
                                * solution.dx))
            assert float(solution.mean[0]) == pytest.approx(1.0, abs=1e-6)

        assert errors[-1] < 0.002, errors
        assert errors[0] / errors[-1] > 8.0, errors

    def test_a_strongly_nonlinear_mapping_defeats_a_global_bandwidth(self):
        """``y = z^2`` with ``z`` normal: the L1 error does **not** converge.

        Its spacing uniformity is about 1600, so ``solve_dpim`` warns: a
        sparse tail needs a wide kernel and the ``y^(-1/2)`` singularity at
        the origin needs a narrow one, so one ``sigma`` cannot serve both --
        and the mismatch *grows* with the point count. Documented as a limit
        of the method as published. If an adaptive bandwidth is ever added
        this test should start failing, and that is the point of it.
        """
        errors, ratios = [], []
        for n_points in (100, 200, 400, 800):
            points = lattice(Normal(0.0, 1.0), n_points, spacing="quantile")
            y = points.z[:, 0] ** 2
            # the uniformity warning is the *prediction* here, not noise:
            # this response has a ratio of ~1600, far past the limit
            with pytest.warns(RuntimeWarning, match="unevenly spread"):
                solution = solve_dpim(points, y, n_cells=4000)

            mask = (solution.x > 0.05) & (solution.x < 8.0)
            exact = (np.exp(-solution.x[mask] / 2.0)
                     / np.sqrt(2.0 * np.pi * solution.x[mask]))
            errors.append(float(np.abs(solution.at(0.0)[mask] - exact).sum()
                                * solution.dx))

            gaps = np.diff(np.sort(y))
            gaps = gaps[gaps > 0.0]
            ratios.append(float(gaps.max() / np.median(gaps)))

        # 96 / 189 / 416 / 815 measured: the unevenness *doubles* with the
        # point count, because the tail gap is pinned by the outermost
        # quantile while the median gap halves. Adding points cannot help.
        assert min(ratios) > 50.0, ratios
        assert ratios[-1] > 4.0 * ratios[0], ratios
        assert errors[-1] > 0.5 * errors[0], errors

    def test_but_its_moments_converge_perfectly_well(self):
        """Which is why a moment check would not reveal the stall above."""
        means = []
        for n_points in (100, 400, 1600):
            points = lattice(Normal(0.0, 1.0), n_points, spacing="quantile")
            with pytest.warns(RuntimeWarning, match="unevenly spread"):
                solution = solve_dpim(points, points.z[:, 0] ** 2,
                                      n_cells=4000)
            means.append(abs(float(solution.mean[0]) - 1.0))
        assert means[-1] < means[0] / 4.0, means


class TestAgainstThePublishedTable:
    """Chen & Yang's own Table 2, which is a stronger check than the closed
    form alone.

    The analytical solution says our answer is right; the published table says
    our *implementation* is the same method theirs is. They report K-L
    divergence against the analytical PDF at t = 0.9, 1.0 and 1.1 s with 500
    representative points, for two space steps, and ``sigma = dy``.

    **The K-L divergence must not be evaluated on the computational grid**,
    which is what makes this comparison possible at all. The exact density
    has jump discontinuities at the ends of the mapped interval, so a sum
    over cells depends on where the cell centres fall relative to those
    edges: measured, the same solution scores anywhere from 0.0052 to 0.0133
    at ``t = 1.1`` as the padding moves by a fraction of a cell, and adding
    one cell to the grid moved an earlier run's figures by 50%. The published
    third digits are inside that noise. Evaluating the same integral on a
    grid 16x finer makes it placement-independent (0.0088 at every padding
    tried) and changes nothing about the density, which is a continuous
    function of ``y`` that can be sampled anywhere.

    With that done, ours against theirs at ``dy = 5.6e-4`` is 0.0073/0.0051/
    0.0088 against 0.0078/0.0062/0.0066 -- ratios 0.94, 0.83, 1.33. The
    residual is not the point set: a quantile lattice, a uniformly spaced
    one and one including the endpoints all give the same three figures.
    """

    POINTS = 500
    DY = 5.6e-4
    INSTANTS = (0.9, 1.0, 1.1)
    #: Table 2, DPIM row at dy = 5.6e-4 -- identical at every listed dt.
    PUBLISHED_DPIM = (0.0078, 0.0062, 0.0066)
    #: and the PDEM row at the one dt where the CFL condition let them run it.
    PUBLISHED_PDEM = (0.0478, 0.0159, 0.0247)

    @staticmethod
    def _kl(solution, time_instant, refine: int = 16):
        """``int p_exact log(p_exact / p_num) dy`` on a refined grid.

        ``refine`` samples the computed density between its cell centres --
        legitimate because it is a sum of Gaussians, continuous in ``y`` --
        so the metric stops depending on where the cells fall relative to the
        density's jumps. See the class docstring; with ``refine = 1`` these
        numbers move by factors of two to four for no physical reason.
        """
        fine = np.linspace(solution.x[0], solution.x[-1],
                           solution.x.size * refine)
        numerical = np.interp(fine, solution.x, solution.at(time_instant))
        exact = sdof_displacement_pdf(fine, time_instant)
        mask = exact > 0.0
        return float((exact[mask]
                      * np.log(exact[mask]
                               / np.clip(numerical[mask], 1e-300, None))).sum()
                     * (fine[1] - fine[0]))

    def _setup(self, dt=0.0025):
        t_grid = np.arange(0.0, 1.1 + 0.5 * dt, dt)
        points = lattice(Uniform(A, B), self.POINTS, spacing="quantile")
        x, x_dot = response_histories(points, sdof_response, t_grid)
        n_cells = int(round(2 * X0 * 1.35 / self.DY))
        return points, t_grid, x, x_dot, n_cells

    def test_dpim_reproduces_their_divergences(self):
        points, t_grid, x, _, n_cells = self._setup()
        solution = solve_dpim(points, x, t_grid, n_cells=n_cells,
                              sigma=self.DY)
        ours = [self._kl(solution, t) for t in self.INSTANTS]
        # 0.94, 0.83 and 1.33 measured; the band is wide enough to survive a
        # different point set and narrow enough that a wrong normalisation,
        # a wrong weight or a wrong kernel width would break it
        for mine, theirs in zip(ours, self.PUBLISHED_DPIM):
            assert 0.6 * theirs < mine < 1.6 * theirs, (ours,
                                                        self.PUBLISHED_DPIM)

    def test_the_metric_itself_is_grid_dependent_and_must_be_refined(self):
        """Why the comparison above refines. On the computational grid the
        divergence swings by a factor of two or more under a sub-cell shift
        of the padding, because the exact density jumps at the ends of the
        mapped interval; refined, it is the same number at every padding."""
        raw, refined = [], []
        for margin in (0.10, 0.15, 0.25):
            points, t_grid, x, _, n_cells = self._setup()
            solution = solve_dpim(points, x, t_grid, n_cells=n_cells,
                                  sigma=self.DY, margin=margin)
            raw.append(self._kl(solution, 1.1, refine=1))
            refined.append(self._kl(solution, 1.1, refine=16))

        # 1.58x here from the padding alone (0.0101 / 0.0064 / 0.0096);
        # letting the cell *count* move as well, as a real comparison
        # against a published grid would, reaches 2.6x. Refined it is 1.06x
        # -- and that remainder is not metric noise but the solution itself:
        # holding the cell count while the padding moves changes dx, hence
        # sigma/dx, hence the density by a little.
        assert max(raw) / min(raw) > 1.4, raw
        assert max(refined) / min(refined) < 1.1, refined
        assert (max(raw) / min(raw)) > 10 * (max(refined) / min(refined) - 1.0)

    def test_the_accuracy_does_not_depend_on_the_time_step(self):
        """Their headline claim for DPIM, and here it is exact rather than
        approximate: the PDIE holds instant by instant, so nothing about the
        answer at t can depend on how finely t was reached."""
        results = []
        for dt in (0.1, 0.0025, 0.0005):
            points, t_grid, x, _, n_cells = self._setup(dt)
            solution = solve_dpim(points, x, t_grid, n_cells=n_cells,
                                  sigma=self.DY)
            results.append([self._kl(solution, t) for t in self.INSTANTS])

        # Not bit-identical, and the residual ~7e-7 is not a dependence on
        # dt: `at` returns the nearest *stored* instant, and arange places
        # t = 0.9 at 0.9 exactly on the dt = 0.1 grid and a few ulp away on
        # the others, so the densities being compared are at imperceptibly
        # different times. The method's independence is exact at a fixed
        # instant; the tolerance covers where the instant lands.
        reference = results[0]
        for row in results[1:]:
            assert row == pytest.approx(reference, rel=1e-5), results

    def test_dpim_beats_pdem_on_the_same_grid_as_they_report(self):
        points, t_grid, x, x_dot, n_cells = self._setup()
        dpim = solve_dpim(points, x, t_grid, n_cells=n_cells, sigma=self.DY)
        pdem = solve_gdee(points, t_grid, x, x_dot, n_cells=n_cells)

        for t, theirs in zip(self.INSTANTS, self.PUBLISHED_PDEM):
            assert self._kl(dpim, t) < self._kl(pdem, t)
            assert self._kl(pdem, t) < 1.5 * theirs


class TestResponseSpreadVersusInputDiscrepancy:
    """Low discrepancy in the *input* space does not give well-spread
    *responses*, and it is the responses the bandwidth has to bridge.

    This is the trap in pairing GF-discrepancy point selection with a global
    kernel: the GF set is quasi-random in how it pairs the dimensions, so in
    the response axis it lands like a sample rather than a grid.
    """

    @staticmethod
    def _sum_of_uniforms(point_set):
        return point_set.z.sum(axis=1)

    def test_a_lattice_gives_even_responses_and_a_gf_set_does_not(self):
        distribution = JointIndependent([Uniform(0.0, 1.0)] * 2)
        regular = spacing_uniformity(
            self._sum_of_uniforms(lattice(distribution, 32,
                                          spacing="quantile")))
        scattered = spacing_uniformity(
            self._sum_of_uniforms(gf_points(distribution, 1024)))

        assert regular == pytest.approx(1.0, abs=0.05)
        assert scattered > 10.0, scattered

    def test_the_default_bandwidth_warns_on_a_scattered_response_set(self):
        """It would otherwise undersmooth by a factor of forty in L1,
        silently."""
        distribution = JointIndependent([Uniform(0.0, 1.0)] * 2)
        points = gf_points(distribution, 1024)
        with pytest.warns(RuntimeWarning, match="unevenly spread"):
            solve_dpim(points, self._sum_of_uniforms(points), n_cells=3000)

    def test_and_does_not_warn_on_a_lattice(self):
        distribution = JointIndependent([Uniform(0.0, 1.0)] * 2)
        points = lattice(distribution, 32, spacing="quantile")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve_dpim(points, self._sum_of_uniforms(points), n_cells=3000)

    def test_an_explicit_bandwidth_recovers_the_gf_sets_accuracy(self):
        """The point set is not the problem -- the default rule for it is.
        With sigma chosen for the spread, the scattered set reaches the
        lattice's accuracy."""
        distribution = JointIndependent([Uniform(0.0, 1.0)] * 2)
        points = gf_points(distribution, 1024)
        y = self._sum_of_uniforms(points)

        def l1(sigma):
            solution = solve_dpim(points, y, n_cells=3000, sigma=sigma)
            exact = np.where(solution.x <= 1.0, solution.x, 2.0 - solution.x)
            exact = np.where((solution.x >= 0.0) & (solution.x <= 2.0),
                             np.clip(exact, 0.0, None), 0.0)
            return float(np.abs(solution.at(0.0) - exact).sum() * solution.dx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert l1(None) > 0.2                 # the default, undersmoothed
        assert l1(0.04) < 0.02                    # chosen for the spread

    def test_gf_points_reach_a_dimension_no_lattice_could(self):
        """Eight inputs: 1024 points here, 1024**8 in a tensor product. The
        mean of the Irwin-Hall sum is exact and the density needs a
        hand-chosen bandwidth, which is the honest state of the method."""
        ndim = 8
        distribution = JointIndependent([Uniform(0.0, 1.0)] * ndim)
        points = gf_points(distribution, 1024)
        y = self._sum_of_uniforms(points)
        solution = solve_dpim(points, y, n_cells=2000, sigma=0.05)

        assert float(solution.mean[0]) == pytest.approx(ndim / 2.0, abs=1e-6)
        assert abs(solution.mass[0] - 1.0) < 1e-3
        # variance of a sum of n independent U(0,1) is n/12, plus sigma^2
        assert float(solution.std[0] ** 2) == pytest.approx(
            ndim / 12.0 + 0.05 ** 2, rel=0.02)


class TestRefusals:
    def test_a_kernel_narrower_than_a_cell_is_refused(self):
        points, t_grid, x, _ = _histories(40)
        with pytest.raises(ProbabilityNotConserved, match="cell size"):
            solve_dpim(points, x, t_grid, n_cells=800, sigma=1e-4)

    def test_a_zero_or_negative_bandwidth_is_refused(self):
        points, t_grid, x, _ = _histories(10)
        for sigma in (0.0, -1e-3):
            with pytest.raises(ValueError, match="sigma must be positive"):
                solve_dpim(points, x, t_grid, sigma=sigma)

    def test_mismatched_point_counts_are_refused(self):
        points, t_grid, x, _ = _histories(10)
        with pytest.raises(ValueError, match="against"):
            solve_dpim(points, x[:5], t_grid)

    def test_mismatched_times_are_refused(self):
        points, t_grid, x, _ = _histories(10)
        with pytest.raises(ValueError, match="times against"):
            solve_dpim(points, x, t_grid[:-1])

    def test_a_non_finite_response_is_refused(self):
        points, t_grid, x, _ = _histories(10)
        x = x.copy()
        x[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            solve_dpim(points, x, t_grid)

    def test_a_zero_margin_is_refused(self):
        points, t_grid, x, _ = _histories(10)
        with pytest.raises(ValueError, match="margin must be positive"):
            solve_dpim(points, x, t_grid, margin=0.0)


class TestAdaptiveBandwidth:
    """``knn_bandwidth``: a width per representative point.

    Opt-in, because it is a trade and not an improvement -- it rescues a
    strongly nonlinear mapping and costs a factor of two on a benign one.
    The tests assert both directions, so neither can be quietly lost.
    """

    @staticmethod
    def _chi_square(n_points):
        points = lattice(Normal(0.0, 1.0), n_points, spacing="quantile")
        return points, points.z[:, 0] ** 2

    @staticmethod
    def _chi_square_l1(solution):
        x = np.clip(solution.x, 1e-300, None)
        exact = np.exp(-x / 2.0) / np.sqrt(2.0 * np.pi * x)
        mask = (solution.x > 0.05) & (solution.x < 8.0)
        return float(np.abs(solution.at(0.0)[mask] - exact[mask]).sum()
                     * solution.dx)

    def test_it_rescues_the_case_a_fixed_width_cannot_do(self):
        """``y = z^2``: fixed is flat at 0.21 however many points; this
        converges. The single strongest reason the option exists."""
        errors = []
        for n_points in (100, 400, 800):
            points, y = self._chi_square(n_points)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution = solve_dpim(points, y, n_cells=20000,
                                      sigma=knn_bandwidth(y, k=3))
            errors.append(self._chi_square_l1(solution))

        assert errors[-1] < 0.004, errors
        assert errors[0] / errors[-1] > 8.0, errors       # it converges
        assert errors[-1] < 0.05 * 0.2113, errors         # 20x+ on fixed

    def test_and_costs_a_factor_of_two_where_fixed_was_already_right(self):
        """The other half of the trade, asserted so it cannot be forgotten
        when someone is tempted to make this the default."""
        t_grid = np.linspace(0.0, 1.0, 101)
        points = lattice(Uniform(A, B), 320, spacing="quantile")
        x, _ = response_histories(points, sdof_response, t_grid)

        fixed = solve_dpim(points, x, t_grid, n_cells=1600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adaptive = solve_dpim(points, x, t_grid, n_cells=1600,
                                  sigma=knn_bandwidth(x, k=3))
        assert _l1(adaptive, 1.0) > 1.5 * _l1(fixed, 1.0)

    def test_the_density_still_integrates_to_one(self):
        """The sample-point form keeps the one exact property DPIM has: each
        kernel integrates to one and the weights sum to one, so the density
        does. A balloon bandwidth, varying with the evaluation point, would
        not."""
        points, y = self._chi_square(400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = solve_dpim(points, y, n_cells=20000,
                                  sigma=knn_bandwidth(y, k=3))
        # 3e-9, not machine zero: the chi-square tail is long and the
        # widest kernels there reach past the end of the grid. That is
        # truncation, not a failure of the normalisation.
        assert abs(solution.mass[0] - 1.0) < 1e-7

    def test_it_reduces_to_the_fixed_rule_on_an_evenly_spaced_set(self):
        """The ``1/k`` normalisation is what makes this a generalisation
        rather than a different rule: on an even set ``d_k = k * gap``."""
        even = np.linspace(0.0, 1.0, 200)[:, None]
        sigma = knn_bandwidth(even, k=4)
        expected = SIGMA_FACTOR * (even[1, 0] - even[0, 0])
        assert np.allclose(sigma, expected, rtol=1e-12)

    def test_an_instant_where_every_response_coincides_is_handled(self):
        """``t = 0`` of any dynamic run. There is no local scale there and
        inventing one wrecked the grid through the padding; such instants
        borrow the median width instead."""
        x = np.zeros((50, 3))
        x[:, 1] = np.linspace(0.0, 1.0, 50)
        x[:, 2] = np.linspace(0.0, 2.0, 50)
        sigma = knn_bandwidth(x, k=3)
        assert np.all(np.isfinite(sigma)) and np.all(sigma > 0.0)
        assert sigma[:, 0].max() <= sigma[:, 1:].max()

    def test_a_grid_too_coarse_for_the_bandwidth_says_so(self):
        points, y = self._chi_square(400)
        with pytest.warns(RuntimeWarning, match="widened to the cell size"):
            solve_dpim(points, y, n_cells=2000, sigma=knn_bandwidth(y, k=3))

    def test_bad_arguments_are_refused(self):
        _, y = self._chi_square(50)
        for bad in dict(k=0, factor=0.0).items():
            with pytest.raises(ValueError):
                knn_bandwidth(y, **dict([bad]))
        with pytest.raises(ValueError, match="needs more than"):
            knn_bandwidth(y, k=50)

    def test_a_sigma_of_the_wrong_shape_is_refused(self):
        points, y = self._chi_square(50)
        with pytest.raises(ValueError, match="sigma should be"):
            solve_dpim(points, y, sigma=np.ones(7))


class TestVanDerPol:
    """Chen & Yang's Example 3, and the check that settles what Example 2
    could not.

    A nonlinear oscillator, a *lognormal* input, and exact divergences
    spanning four orders of magnitude across the three instants -- far more
    discriminating than Example 2's narrow range. The response spacing
    uniformity is about 2000, so this is squarely the regime where a single
    bandwidth fails and the per-point one is not a luxury.
    """

    INSTANTS = np.array([3.0, 4.2, 9.0])

    def _setup(self, n_points):
        points = lattice(LogNormal(0.0, 1.0), n_points, spacing="quantile")
        theta = points.z[:, 0]
        x = (np.cos(self.INSTANTS)[None, :]
             - theta[:, None] * vanderpol_coefficient(self.INSTANTS)[None, :]
             / 8.0)
        return points, x

    @staticmethod
    def _kl(solution, t, refine=16):
        fine = np.linspace(solution.x[0], solution.x[-1],
                           solution.x.size * refine)
        numerical = np.interp(fine, solution.x, solution.at(t))
        exact = vanderpol_pdf(fine, t)
        mask = exact > 1e-14
        return float((exact[mask]
                      * np.log(exact[mask]
                               / np.clip(numerical[mask], 1e-300, None))).sum()
                     * (fine[1] - fine[0]))

    def test_the_reference_integrates_to_one(self):
        """The paper's Eq. (46) disagrees with its own Eq. (45) by a factor
        of seven at t = 4.2, so the reference is derived and checked here
        rather than transcribed."""
        for t in self.INSTANTS:
            # theta > 0 only, so the support is one-sided and which side it
            # falls on is the sign of A(t) -- negative at all three instants
            # here, so y runs upward from cos t
            ends = sorted([np.cos(t),
                           np.cos(t) - 80.0 * vanderpol_coefficient(t) / 8.0])
            grid = np.linspace(ends[0] - 1.0, ends[1] + 1.0, 800000)
            assert np.trapz(vanderpol_pdf(grid, t), grid) == \
                pytest.approx(1.0, abs=1e-4)

    def test_it_is_the_regime_a_single_bandwidth_cannot_serve(self):
        _, x = self._setup(500)
        assert spacing_uniformity(x) > 500.0

    def test_our_answer_converges_to_the_analytical_solution(self):
        """The check that validates the implementation, as distinct from
        agreeing with anyone's table: the divergence must go to zero.
        Measured 0.0045 / 0.0021 / 0.0010 / 0.0005 at t = 3 over
        250 / 500 / 1000 / 2000 points -- first order, in all three
        instants at once.
        """
        errors = []
        for n_points, n_cells in ((250, 20000), (500, 30000), (1000, 50000)):
            points, x = self._setup(n_points)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution = solve_dpim(points, x, self.INSTANTS,
                                      n_cells=n_cells,
                                      sigma=knn_bandwidth(x, k=3))
            errors.append([self._kl(solution, t) for t in self.INSTANTS])

        errors = np.array(errors)
        assert np.all(errors[-1] < 0.002), errors
        for column in range(3):
            ratios = errors[:-1, column] / errors[1:, column]
            assert np.all(ratios > 1.6), (column, errors[:, column])

    def test_a_single_bandwidth_cannot_do_it_at_all(self):
        """Using the paper's own prescription, sigma = dy, on this problem:
        no fixed width comes within a factor of 250 of the published row,
        and ours is three orders worse than with a per-point width. That is
        a property of the prescription meeting a uniformity of 2000, not of
        this implementation."""
        points, x = self._setup(500)
        span = float(x.max() - x.min())
        fixed = solve_dpim(points, x, self.INSTANTS,
                           n_cells=int(round(span * 1.4 / 0.012)), sigma=0.012)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adaptive = solve_dpim(points, x, self.INSTANTS, n_cells=30000,
                                  sigma=knn_bandwidth(x, k=3))
        for t in self.INSTANTS:
            assert self._kl(fixed, t) > 100.0 * self._kl(adaptive, t)
