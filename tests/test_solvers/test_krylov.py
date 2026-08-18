"""
Verification for the Krylov solvers and preconditioners.

The external benchmark here is a *rate*, not a number, and it is the same idea
as the observed order of accuracy used for the discretisations: solve a Poisson
problem on a sequence of meshes and watch how the iteration count grows.

    unpreconditioned    grows like 1/h        26 -> 52 -> 102 -> 148
    IC(0)               grows more slowly     12 -> 20 ->  35 ->  50
    AMG                 does not grow         10 -> 11 ->  14 ->  14

The last line is the whole reason multigrid exists and no other preconditioner
here has it. It cannot be faked by tuning a tolerance, which is what makes it a
real test of the hierarchy rather than of the plumbing.

Two facts worth stating because they surprise people, and both are tested:

* **Jacobi does exactly nothing on a constant-diagonal matrix.** Preconditioning
  by a multiple of the identity leaves the Krylov space unchanged, so the
  iteration count is *identical* to unpreconditioned -- not similar, identical.
  The standard Poisson matrix has a constant diagonal, so a Jacobi row in a
  Poisson table that differs from the baseline indicates a bug somewhere.
* **ILU on an SPD system usually works despite not being SPD.** It converged in
  5 iterations on a contrast-1e5 diffusion problem. The problem is that it is
  not guaranteed to, and the failure mode is a meaningless step length rather
  than a warning.

python -m pytest tests/test_solvers/test_krylov.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from simulacrum.solvers.amg import (
    AMGHierarchy,
    aggregate,
    nodal_strength,
    rigid_body_translations,
    smoothed_aggregation,
    strength_of_connection,
)
from simulacrum.solvers.krylov import (
    KrylovResult,
    NotPositiveDefinite,
    conjugate_gradient,
    pcg,
)
from simulacrum.solvers.preconditioners import (
    PRECONDITIONER_REGISTRY,
    PYAMG_AVAILABLE,
    AlgebraicMultigrid,
    IncompleteCholesky,
    IncompleteLU,
    Identity,
    Jacobi,
    get_preconditioner,
)

SPD_PRECONDITIONERS = ["none", "jacobi", "icc", "amg"]


def poisson_2d(n: int) -> sparse.csr_matrix:
    """5-point Laplacian on an ``n x n`` interior grid. SPD, constant diagonal."""
    h = 1.0 / (n + 1)
    identity = sparse.identity(n)
    tri = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n))
    return ((sparse.kron(identity, tri) + sparse.kron(tri, identity))
            / h ** 2).tocsr()


def heterogeneous_diffusion(n: int, contrast: float = 1e5,
                            seed: int = 0) -> sparse.csr_matrix:
    """Cell-wise random conductivity spanning ``contrast`` orders of magnitude.

    The regime where diagonal scaling stops being enough: the diagonal is no
    longer constant, so Jacobi does something, and the near-null space is no
    longer the constant vector alone.
    """
    rng = np.random.default_rng(seed)
    k = 10.0 ** rng.uniform(0.0, np.log10(contrast), n * n)
    h = 1.0 / (n + 1)
    index = lambda i, j: i * n + j
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            centre = index(i, j)
            total = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                inside = 0 <= ii < n and 0 <= jj < n
                face = (0.5 * (k[centre] + k[index(ii, jj)]) if inside
                        else k[centre])
                total += face
                if inside:
                    rows.append(centre)
                    cols.append(index(ii, jj))
                    vals.append(-face / h ** 2)
            rows.append(centre)
            cols.append(centre)
            vals.append(total / h ** 2)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))


@pytest.fixture
def small_spd():
    A = poisson_2d(10)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def test_cg_solves_an_spd_system(small_spd):
    """Against a direct solve, which is a different algorithm entirely."""
    A, b = small_spd
    result = conjugate_gradient(A, b, tol=1e-12)

    assert result.converged
    np.testing.assert_allclose(result.x, spla.spsolve(A.tocsc(), b), rtol=1e-8)


@pytest.mark.parametrize("name", SPD_PRECONDITIONERS)
def test_every_preconditioner_reaches_the_same_solution(name, small_spd):
    """A preconditioner changes the path, never the answer. Anything else is a
    bug in the preconditioner rather than a property of it."""
    A, b = small_spd
    exact = spla.spsolve(A.tocsc(), b)
    result = pcg(A, b, get_preconditioner(name, A), tol=1e-12, max_iter=5000)

    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-7, atol=1e-10)


def test_cg_terminates_in_at_most_n_iterations_in_exact_arithmetic():
    """Not an efficiency claim -- a structural one. CG's search directions are
    A-orthogonal, so after n of them the space is exhausted. On a small
    well-conditioned system round-off does not hide it."""
    rng = np.random.default_rng(0)
    n = 12
    B = rng.standard_normal((n, n))
    A = sparse.csr_matrix(B @ B.T + n * np.eye(n))
    b = rng.standard_normal(n)

    result = conjugate_gradient(A, b, tol=1e-13, max_iter=10 * n)
    assert result.iterations <= n


def test_a_warm_start_reduces_the_work(small_spd):
    A, b = small_spd
    exact = spla.spsolve(A.tocsc(), b)

    cold = conjugate_gradient(A, b, tol=1e-10)
    warm = conjugate_gradient(A, b, x0=exact + 1e-6, tol=1e-10)
    assert warm.iterations < cold.iterations


def test_a_zero_right_hand_side_returns_immediately(small_spd):
    """Guards the division by ``||b||``, which is otherwise a nan."""
    A, _ = small_spd
    result = conjugate_gradient(A, np.zeros(A.shape[0]))

    assert result.converged and result.iterations == 0
    assert not result.x.any()


def test_the_operator_may_be_a_callable_or_a_linear_operator(small_spd):
    """CG needs only the action of A, which is what makes it usable on an
    operator that is never assembled."""
    A, b = small_spd
    exact = spla.spsolve(A.tocsc(), b)

    from_callable = pcg(lambda v: A @ v, b, tol=1e-10)
    from_operator = pcg(spla.LinearOperator(A.shape, matvec=lambda v: A @ v),
                        b, tol=1e-10)

    np.testing.assert_allclose(from_callable.x, exact, rtol=1e-7)
    np.testing.assert_allclose(from_operator.x, exact, rtol=1e-7)


def test_the_callback_sees_every_iteration(small_spd):
    A, b = small_spd
    seen = []
    result = conjugate_gradient(A, b, tol=1e-8,
                               callback=lambda k, x, r: seen.append((k, r)))

    assert [k for k, _ in seen] == list(range(1, result.iterations + 1))
    assert seen[-1][1] == pytest.approx(result.relative_residual)


# ---------------------------------------------------------------------------
# The rate: what actually distinguishes the preconditioners
# ---------------------------------------------------------------------------

def _iterations(A, name, tol=1e-8, **kwargs):
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return pcg(A, b, get_preconditioner(name, A, **kwargs), tol=tol,
               max_iter=20000).iterations


def test_unpreconditioned_cg_scales_like_one_over_h():
    """The baseline behaviour every preconditioner is trying to beat: the
    condition number of a Poisson operator grows like ``h^-2``, so CG's
    iteration count grows like ``h^-1``."""
    counts = [_iterations(poisson_2d(n), "none") for n in (8, 16, 32)]

    ratios = [b / a for a, b in zip(counts, counts[1:])]
    assert all(1.6 < r < 2.4 for r in ratios), counts


def test_amg_iteration_count_is_essentially_mesh_independent():
    """The property that justifies the setup cost, and the one no other
    preconditioner in this module has. Measured 10 / 11 / 14 / 14 against the
    unpreconditioned 26 / 52 / 102 / 148."""
    counts = [_iterations(poisson_2d(n), "amg") for n in (8, 16, 32, 48)]

    assert max(counts) <= 2 * min(counts), counts
    assert counts[-1] < 25
    # and the mesh grew 36-fold in unknowns over that range
    assert counts[-1] <= counts[0] + 8


def test_incomplete_cholesky_grows_more_slowly_than_nothing():
    """IC(0) is not mesh-independent -- it cannot be, it has no coarse level --
    but it is a large constant-factor win."""
    for n in (16, 32):
        assert _iterations(poisson_2d(n), "icc") < 0.5 * _iterations(
            poisson_2d(n), "none")


def test_jacobi_does_exactly_nothing_on_a_constant_diagonal_matrix():
    """Preconditioning by a multiple of the identity leaves the Krylov space
    untouched, so the iteration count is *identical*, not merely similar. The
    Poisson matrix has a constant diagonal, so a Jacobi row that differs from
    the baseline in a Poisson table means something is wrong."""
    A = poisson_2d(16)
    assert np.ptp(np.diff(A.diagonal())) == 0.0        # constant diagonal
    assert _iterations(A, "jacobi") == _iterations(A, "none")


def test_jacobi_helps_once_the_diagonal_varies():
    """The other half of the statement. With coefficients spanning five orders
    of magnitude, diagonal scaling removes a real part of the conditioning."""
    A = heterogeneous_diffusion(24, contrast=1e5)
    assert _iterations(A, "jacobi") < 0.6 * _iterations(A, "none")


def test_the_preconditioners_are_ordered_as_expected_on_a_hard_problem():
    """Iterations only. Wall-clock ordering is different and depends on setup
    cost, which is the trade the accompanying paper is about."""
    A = heterogeneous_diffusion(24, contrast=1e5)
    counts = {name: _iterations(A, name) for name in SPD_PRECONDITIONERS}

    assert counts["amg"] < counts["icc"] < counts["jacobi"] < counts["none"]


# ---------------------------------------------------------------------------
# Failure detection
# ---------------------------------------------------------------------------

def test_a_non_positive_definite_operator_is_refused():
    """``p'Ap <= 0`` means the step length is meaningless. The usual
    implementation adds a small number to that denominator, which turns a
    detectable breakdown into a plausible wrong answer."""
    A = sparse.diags([1.0, -2.0, 3.0]).tocsr()
    with pytest.raises(NotPositiveDefinite, match="not positive definite"):
        conjugate_gradient(A, np.ones(3))


def test_a_non_symmetric_preconditioner_is_caught_when_it_bites():
    """``r'z`` is used as a squared norm, so a negative value means the
    preconditioner is not SPD and everything after it is meaningless."""
    A = poisson_2d(6)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    indefinite = sparse.diags(
        np.where(np.arange(A.shape[0]) % 2, 1.0, -1.0)).tocsr()

    with pytest.raises(NotPositiveDefinite, match="not symmetric positive"):
        pcg(A, b, indefinite)


def test_the_spd_checks_can_be_switched_off():
    """They cost two comparisons an iteration, so there is a way past them --
    but the default is on, because the failure they catch is silent."""
    A = sparse.diags([1.0, -2.0, 3.0]).tocsr()
    result = pcg(A, np.ones(3), check_spd=False, max_iter=5)
    assert isinstance(result, KrylovResult)


def test_failure_to_converge_is_reported_rather_than_raised(small_spd):
    """A capped run is a legitimate outcome -- the comparison tables in the
    paper contain them -- so it returns a result flagged as unconverged."""
    A, b = small_spd
    result = conjugate_gradient(A, b, tol=1e-14, max_iter=3)

    assert not result.converged
    assert result.iterations == 3
    assert result.residual_history.size == 4


def test_both_residuals_are_reported(small_spd):
    """The recurrence residual is what the loop stops on; the true residual is
    what you should believe. On a well-conditioned system they agree, which is
    what makes the drift a useful signal when they do not."""
    A, b = small_spd
    result = conjugate_gradient(A, b, tol=1e-10)

    assert result.true_residual < 1e-8
    assert result.residual_drift == pytest.approx(1.0, abs=0.5)
    assert "converged" in result.summary()


def test_ilu_is_marked_as_not_spd_even_though_it_usually_works():
    """Being precise rather than alarmist: on this problem ILU converges in a
    handful of iterations. It is simply not guaranteed to, and the flag is how
    that shows up in the object rather than only in prose."""
    A = heterogeneous_diffusion(16, contrast=1e5)
    b = np.random.default_rng(1).standard_normal(A.shape[0])

    preconditioner = IncompleteLU().setup(A)
    assert preconditioner.is_spd is False

    result = pcg(A, b, preconditioner, tol=1e-8, max_iter=2000)
    assert result.converged
    assert result.iterations < 30


# ---------------------------------------------------------------------------
# Incomplete Cholesky
# ---------------------------------------------------------------------------

def test_ic0_reproduces_the_matrix_on_its_own_sparsity_pattern():
    """The defining property of IC(0): ``L L'`` agrees with ``A`` exactly where
    ``A`` is nonzero, and the discarded fill is everything else. This is a much
    sharper check than "PCG converges"."""
    A = heterogeneous_diffusion(12, contrast=1e3)
    preconditioner = IncompleteCholesky().setup(A)
    L = preconditioner._L

    difference = (L @ L.T - A)
    error = np.abs(difference[A.nonzero()]).max()
    assert error < 1e-10 * np.abs(A.data).max()
    assert preconditioner.applied_shift == 0.0


def test_ic0_is_lower_triangular_with_positive_diagonal():
    A = poisson_2d(8)
    L = IncompleteCholesky().setup(A)._L

    assert (sparse.triu(L, k=1).nnz) == 0
    assert np.all(L.diagonal() > 0.0)


def test_ic0_shifts_rather_than_failing_when_a_pivot_goes_negative():
    """Dropping fill can destroy definiteness on a matrix whose exact Cholesky
    factorisation exists. The standard remedy is to factor ``A + delta diag(A)``
    and report the shift; failing silently would produce a factor full of nans."""
    # a matrix that is SPD but only barely, so IC(0) breaks down unshifted
    n = 60
    rng = np.random.default_rng(3)
    A = sparse.random(n, n, density=0.3, random_state=rng, format="csr")
    A = (A + A.T).tolil()
    A.setdiag(np.abs(A).sum(axis=1).A1 * 0.5 + 1e-3)   # weakly diagonal
    A = sparse.csr_matrix(A)
    eigenvalues = np.linalg.eigvalsh(A.toarray())
    assume_spd = eigenvalues.min() > 0
    if not assume_spd:
        pytest.skip("random matrix did not come out positive definite")

    preconditioner = IncompleteCholesky().setup(A)
    assert np.all(np.isfinite(preconditioner._L.data))
    b = rng.standard_normal(n)
    assert pcg(A, b, preconditioner, tol=1e-8, max_iter=2000).converged


def test_jacobi_refuses_a_zero_diagonal():
    A = sparse.diags([1.0, 0.0, 1.0]).tocsr()
    with pytest.raises(ValueError, match="zero diagonal"):
        Jacobi().setup(A)


# ---------------------------------------------------------------------------
# The AMG hierarchy itself
# ---------------------------------------------------------------------------

def test_the_strength_graph_is_symmetric_and_scale_invariant():
    """``|A_ij| > theta sqrt(A_ii A_jj)`` compares a coupling against the two
    diagonals it connects, so scaling a row must not change which connections
    are strong. Using ``theta * max|A_ij|`` instead fails exactly here."""
    A = heterogeneous_diffusion(10, contrast=1e4)
    graph = strength_of_connection(A, theta=0.08)

    assert (graph != graph.T).nnz == 0
    assert graph.diagonal().sum() == 0.0

    scaling = sparse.diags(10.0 ** np.random.default_rng(0).uniform(
        -3, 3, A.shape[0]))
    rescaled = strength_of_connection((scaling @ A @ scaling).tocsr(),
                                      theta=0.08)
    assert (graph != rescaled).nnz == 0


def test_aggregation_covers_every_node_exactly_once():
    """Nodes left unaggregated become singleton coarse unknowns and show up as
    near-zero rows in the Galerkin operator, so the second pass is load-bearing
    rather than tidying."""
    graph = strength_of_connection(poisson_2d(12))
    assignment = aggregate(graph)

    assert assignment.min() >= 0
    assert assignment.size == graph.shape[0]
    assert set(np.unique(assignment)) == set(range(assignment.max() + 1))


def test_the_hierarchy_contracts_and_stays_cheap():
    hierarchy = smoothed_aggregation(poisson_2d(32))

    sizes = hierarchy.sizes
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[-1] <= 40
    assert hierarchy.n_levels >= 3
    # smoothed aggregation fills in the coarse operators; above about 2 the
    # V-cycle stops being cheap
    assert hierarchy.operator_complexity < 2.0


def test_the_coarse_operators_stay_symmetric_positive_definite():
    """Galerkin coarsening ``A_c = P' A P`` preserves both, which is what makes
    the V-cycle usable inside PCG. A non-Galerkin restriction would not."""
    hierarchy = smoothed_aggregation(poisson_2d(16))

    for level in hierarchy.levels:
        dense = level.A.toarray()
        assert np.abs(dense - dense.T).max() < 1e-10 * np.abs(dense).max()
        assert np.linalg.eigvalsh(dense).min() > 0.0


def test_the_v_cycle_is_symmetric_as_an_operator():
    """PCG requires it. Equal pre- and post-smoothing with ``R = P'`` gives it;
    an unsymmetric cycle can be a fine standalone iteration and still be an
    invalid preconditioner."""
    A = poisson_2d(8)
    hierarchy = smoothed_aggregation(A, max_coarse=10)
    n = A.shape[0]

    dense = np.column_stack([hierarchy.apply(e) for e in np.eye(n)])
    assert np.abs(dense - dense.T).max() < 1e-9 * np.abs(dense).max()


def test_the_hierarchy_converges_on_its_own():
    """A hierarchy that cannot solve the problem standalone is a poor
    preconditioner too, and it is far easier to diagnose here than inside PCG."""
    A = poisson_2d(24)
    hierarchy = smoothed_aggregation(A)
    b = np.random.default_rng(0).standard_normal(A.shape[0])

    x, history = hierarchy.solve(b, tol=1e-10, max_iter=60)
    assert history[-1] < 1e-8
    contraction = (history[-1] / history[1]) ** (1.0 / (len(history) - 2))
    assert contraction < 0.6


def test_a_single_level_hierarchy_still_applies():
    """A matrix already at the coarse-size threshold gets one level and no
    prolongator, which must not be a special case at the call site."""
    A = poisson_2d(4)
    hierarchy = smoothed_aggregation(A, max_coarse=1000)

    assert hierarchy.n_levels == 1
    assert hierarchy.levels[0].P is None
    assert np.all(np.isfinite(hierarchy.apply(np.ones(A.shape[0]))))


def test_supplying_a_near_null_space_changes_the_hierarchy():
    """The default near-null space is the constant vector, which is right for a
    scalar elliptic operator and wrong for elasticity. Passing modes has to
    actually reach the prolongator."""
    A = poisson_2d(12)
    n = A.shape[0]
    default = smoothed_aggregation(A)
    enriched = smoothed_aggregation(
        A, near_null=np.column_stack([np.ones(n), np.arange(n) / n]))

    assert enriched.sizes[1] > default.sizes[1]
    assert np.all(np.isfinite(enriched.apply(np.ones(n))))


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        smoothed_aggregation(sparse.random(10, 8, density=0.5, format="csr"))


# ---------------------------------------------------------------------------
# Registry and backends
# ---------------------------------------------------------------------------

def test_the_registry_builds_and_sets_up_every_preconditioner(small_spd):
    A, _ = small_spd
    for name in sorted(set(PRECONDITIONER_REGISTRY)):
        preconditioner = get_preconditioner(name, A)
        assert preconditioner.n_unknowns == A.shape[0]
        assert preconditioner.setup_time >= 0.0
        assert np.all(np.isfinite(preconditioner.apply(np.ones(A.shape[0]))))


def test_an_unknown_preconditioner_is_refused():
    with pytest.raises(NotImplementedError, match="Unknown preconditioner"):
        get_preconditioner("ssor")


def test_a_preconditioner_can_be_handed_to_scipy(small_spd):
    """``as_linear_operator`` exists so the same object works with scipy's own
    Krylov routines, which is how these are cross-checked."""
    A, b = small_spd
    M = get_preconditioner("icc", A)

    x, info = spla.cg(A, b, M=M.as_linear_operator(), rtol=1e-10, maxiter=500)
    assert info == 0
    np.testing.assert_allclose(x, spla.spsolve(A.tocsc(), b), rtol=1e-6)


@pytest.mark.skipif(PYAMG_AVAILABLE, reason="pyamg is installed")
def test_the_pyamg_backend_explains_itself_when_absent():
    """The native path is the supported one; the optional backend exists so the
    paper's published numbers stay reproducible where pyamg is installed."""
    with pytest.raises(ImportError, match="pure NumPy"):
        AlgebraicMultigrid(backend="pyamg")


def test_an_unknown_amg_backend_is_refused():
    with pytest.raises(ValueError, match="native"):
        AlgebraicMultigrid(backend="trilinos")


def test_setup_time_is_recorded_and_ordered_as_expected(small_spd):
    """Setup cost is part of the comparison, not an implementation detail: for
    one solve AMG's setup can exceed the entire unpreconditioned iteration,
    and across a hundred right-hand sides it is free."""
    A, _ = small_spd
    cheap = get_preconditioner("jacobi", A)
    expensive = get_preconditioner("amg", A)

    assert cheap.setup_time < expensive.setup_time
    assert Identity().setup(A).setup_time >= 0.0


# ---------------------------------------------------------------------------
# Threshold incomplete Cholesky
# ---------------------------------------------------------------------------

def test_ict_with_a_vanishing_tolerance_is_a_complete_factorisation():
    """The sharpest check available on the ICT recurrence: with nothing dropped
    and no fill cap it *is* Cholesky, so ``L L' == A`` everywhere -- not just on
    A's pattern -- and PCG converges in one iteration."""
    A = heterogeneous_diffusion(12, contrast=1e3)
    preconditioner = IncompleteCholesky(drop_tol=1e-30,
                                        fill_factor=1e9).setup(A)
    L = preconditioner._L

    residual = np.abs((L @ L.T - A).toarray()).max()
    assert residual < 1e-8 * np.abs(A.data).max()

    b = np.random.default_rng(0).standard_normal(A.shape[0])
    assert pcg(A, b, preconditioner, tol=1e-8).iterations == 1


def test_ict_trades_fill_for_iterations_monotonically():
    """The curve that makes the tolerance a useful knob. A criterion that is not
    scale-free gives no range at all -- thresholding against ``||A(:,j)||``
    rather than against the factor's own column gave 278 iterations at every
    tolerance from 1e-1 to 1e-4 on this problem."""
    A = heterogeneous_diffusion(30, contrast=1e6)
    b = np.random.default_rng(1).standard_normal(A.shape[0])

    iterations, fill = [], []
    for drop_tol in (3e-1, 1e-1, 3e-2, 1e-2, 1e-3):
        preconditioner = IncompleteCholesky(drop_tol=drop_tol,
                                            fill_factor=8.0).setup(A)
        iterations.append(pcg(A, b, preconditioner, tol=1e-8,
                              max_iter=50000).iterations)
        fill.append(preconditioner.fill_ratio)

    assert iterations == sorted(iterations, reverse=True), iterations
    assert fill == sorted(fill), fill


def test_ict_beats_ic0_using_less_memory_at_a_coarse_tolerance():
    """The actual argument for thresholding: IC(0) keeps whichever entries lie
    in ``A``'s pattern, ICT keeps the largest wherever they are, and on a
    heterogeneous operator those differ. Measured 62 iterations at fill 0.87
    against IC(0)'s 98 at fill 1.00."""
    A = heterogeneous_diffusion(30, contrast=1e6)
    b = np.random.default_rng(1).standard_normal(A.shape[0])

    ic0 = IncompleteCholesky().setup(A)
    ict = IncompleteCholesky(drop_tol=0.3, fill_factor=8.0).setup(A)

    plain = pcg(A, b, ic0, tol=1e-8, max_iter=50000).iterations
    thresholded = pcg(A, b, ict, tol=1e-8, max_iter=50000).iterations

    assert thresholded < plain
    assert ict.fill_ratio < 1.0


def test_the_fill_cap_bounds_memory_independently_of_the_tolerance():
    """A tolerance alone guarantees nothing about memory, so the count is capped
    too. With the cap at 1 the factor cannot exceed A's own pattern size."""
    A = heterogeneous_diffusion(16, contrast=1e4)
    loose = IncompleteCholesky(drop_tol=1e-12, fill_factor=1.0).setup(A)
    free = IncompleteCholesky(drop_tol=1e-12, fill_factor=50.0).setup(A)

    assert loose.fill_ratio <= 1.05
    assert free.fill_ratio > loose.fill_ratio


def test_the_variant_is_reported_and_selected_by_drop_tol():
    A = poisson_2d(8)
    assert IncompleteCholesky().setup(A).variant == "ic0"
    assert IncompleteCholesky(drop_tol=1e-2).setup(A).variant == "ict"
    assert get_preconditioner("ict", A).variant == "ict"


def test_a_degenerate_ict_configuration_is_refused():
    with pytest.raises(ValueError, match="drop_tol"):
        IncompleteCholesky(drop_tol=-1.0)
    with pytest.raises(ValueError, match="fill_factor"):
        IncompleteCholesky(drop_tol=1e-3, fill_factor=0.5)


def test_ict_solves_the_system_it_preconditions():
    A = heterogeneous_diffusion(20, contrast=1e5)
    b = np.random.default_rng(2).standard_normal(A.shape[0])
    result = pcg(A, b, get_preconditioner("ict", A), tol=1e-10, max_iter=50000)

    assert result.converged
    np.testing.assert_allclose(result.x, spla.spsolve(A.tocsc(), b), rtol=1e-6,
                               atol=1e-10)


# ---------------------------------------------------------------------------
# AMG for systems of PDEs
# ---------------------------------------------------------------------------

def elasticity_operator(nx: int, ny: int):
    """Plane-strain cantilever stiffness with the left edge fixed.

    Returns ``(K, near_null)`` with the three rigid-body modes restricted to the
    free DOFs -- two translations and the in-plane rotation.
    """
    from simulacrum.core.domain import Domain
    from simulacrum.discretization.fem.tables import QuadratureTables
    from simulacrum.materials.linear_elastic import PlaneStrain
    from simulacrum.mesh.generators import rectangle_mesh

    domain = Domain(rectangle_mesh((0.0, 4.0), (0.0, 1.0), nx, ny), dim=2)
    tables = QuadratureTables(domain, thickness=1.0)
    n_dof = 2 * domain.n_nodes
    C = PlaneStrain(E=210e3, nu=0.3).tangent(dim=2)
    n_points = tables.strains_voigt(np.zeros(n_dof)).shape[0]
    K, _ = tables.stiffness_and_force(
        np.zeros((n_points, 3)),
        np.broadcast_to(C, (n_points, 3, 3)).copy(), n_dof)

    nodes = domain.coordinates
    left = np.flatnonzero(nodes[:, 0] < 1e-9)
    fixed = np.concatenate([2 * left, 2 * left + 1])
    free = np.setdiff1d(np.arange(n_dof), fixed)

    coords = np.repeat(nodes, 2, axis=0)[free]
    component = np.tile([0, 1], domain.n_nodes)[free]
    modes = np.zeros((free.size, 3))
    modes[component == 0, 0] = 1.0
    modes[component == 1, 1] = 1.0
    modes[component == 0, 2] = -coords[component == 0, 1]
    modes[component == 1, 2] = coords[component == 1, 0]
    return K.tocsr()[np.ix_(free, free)].tocsr(), modes


def test_nodal_aggregation_keeps_a_nodes_dofs_together():
    """The defect it exists to fix. On the DOF graph 30 of 91 nodes had their
    x- and y-DOF placed in different aggregates; on the node graph that cannot
    happen by construction."""
    K, _ = elasticity_operator(12, 6)
    n_nodes = K.shape[0] // 2

    scattered = aggregate(strength_of_connection(K, 0.08))
    split = sum(scattered[2 * i] != scattered[2 * i + 1]
                for i in range(n_nodes))
    assert split > 0                       # the old behaviour, still reachable

    blocked = np.repeat(aggregate(nodal_strength(K, 2, 0.08)), 2)
    assert all(blocked[2 * i] == blocked[2 * i + 1] for i in range(n_nodes))


def test_nodal_strength_rejects_a_block_size_that_does_not_divide():
    K, _ = elasticity_operator(6, 3)
    # pick a size that genuinely does not divide: this operator has 48 free
    # DOFs, so 2, 3, 4, 6 and 8 all do
    block = next(b for b in range(2, 20) if K.shape[0] % b)
    with pytest.raises(ValueError, match="does not divide"):
        nodal_strength(K, block, 0.08)


def test_the_default_near_null_for_a_system_is_the_translations():
    modes = rigid_body_translations(8, 2)
    assert modes.shape == (8, 2)
    np.testing.assert_array_equal(modes[:, 0], [1, 0, 1, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(modes[:, 1], [0, 1, 0, 1, 0, 1, 0, 1])


def test_amg_on_elasticity_is_mesh_independent_only_with_blocks_and_rotations():
    """The headline. Nodal aggregation plus the rigid-body rotations restores
    the property multigrid exists for; either alone does not.

    Measured 26/33/34/33/34 over a 39-fold growth in unknowns, against the
    scalar path's 54/89/137/201/263 and IC(0)'s 27/50/80/120/158.
    """
    scalar, blocked = [], []
    for nx, ny in ((12, 6), (24, 12), (40, 20)):
        K, modes = elasticity_operator(nx, ny)
        b = np.random.default_rng(0).standard_normal(K.shape[0])
        scalar.append(pcg(K, b, AlgebraicMultigrid().setup(K), tol=1e-8,
                          max_iter=30000).iterations)
        blocked.append(pcg(K, b,
                           AlgebraicMultigrid(block_size=2,
                                              near_null=modes).setup(K),
                           tol=1e-8, max_iter=30000).iterations)

    # the scalar path loses mesh independence; the blocked one keeps it
    assert scalar[-1] > 2 * scalar[0]
    assert blocked[-1] < 1.5 * blocked[0]
    assert blocked[-1] < scalar[-1] / 3


def test_blocked_amg_beats_incomplete_cholesky_on_elasticity():
    """The practical consequence: with the fix AMG is the right choice on an
    elasticity operator, and without it IC(0) was better."""
    K, modes = elasticity_operator(40, 20)
    b = np.random.default_rng(0).standard_normal(K.shape[0])

    blocked = pcg(K, b, AlgebraicMultigrid(block_size=2,
                                           near_null=modes).setup(K),
                  tol=1e-8, max_iter=30000).iterations
    incomplete = pcg(K, b, get_preconditioner("icc", K), tol=1e-8,
                     max_iter=30000).iterations
    assert blocked < incomplete


def test_the_near_null_space_survives_coarsening():
    """``P_tent B_coarse = B`` exactly, which is what carries the rigid-body
    modes past level one. Letting the coarse levels fall back to a constant
    vector throws them away and looks like a partial improvement."""
    from simulacrum.solvers.amg import _tentative_prolongator

    K, modes = elasticity_operator(12, 6)
    assignment = np.repeat(aggregate(nodal_strength(K, 2, 0.08)), 2)
    P, B_coarse = _tentative_prolongator(assignment, modes, K.shape[0])

    assert B_coarse.shape[1] == modes.shape[1]
    np.testing.assert_allclose(P @ B_coarse, modes, atol=1e-10)


def test_a_block_size_that_does_not_divide_the_matrix_is_refused():
    with pytest.raises(ValueError, match="does not divide"):
        smoothed_aggregation(poisson_2d(5), block_size=2)
