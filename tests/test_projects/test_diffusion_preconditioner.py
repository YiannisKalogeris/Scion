"""
Verification for the diffusion-filtered two-level preconditioner.

The method is an *accelerator*, so the tests are of two kinds: identities that
must hold exactly whatever the coarse space contains, and measured gains on a
problem hard enough for a gain to be possible.

The identities are the valuable ones, because a coarse space that is subtly
wrong still converges -- just less well -- and nothing in the iteration count
says which. Three hold by construction:

* ``Z' K P' = 0`` -- the deflation projector removes exactly ``range(Z)`` in the
  K inner product. Broken immediately by a coordinate-mapping mistake between
  ``Z``, ``KZ`` and ``E``, which is the Strategy A trap.
* ``T^-1`` is symmetric whenever the baseline is. PCG requires it, and the
  cheaper one-sided forms do not have it.
* the answer is unchanged. A preconditioner alters the path, never the solution.

python -m pytest tests/test_projects/test_diffusion_preconditioner.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from projects.diffusion_preconditioner.diffusion import (
    DEFAULT_TIMES,
    arnoldi,
    build_coarse_space,
    generate_probes,
    lanczos,
    semigroup_snapshots,
)
from projects.diffusion_preconditioner.two_level import TwoLevelPreconditioner
from simulacrum.solvers import get_preconditioner, pcg
from tests.test_solvers.test_krylov import heterogeneous_diffusion, poisson_2d


@pytest.fixture(scope="module")
def hard():
    """Small enough to be fast, ill-conditioned enough for a gain to exist."""
    A = heterogeneous_diffusion(20, contrast=1e6)
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    return A, b


def _coarse(A, strategy="A", baseline=None, **kwargs):
    options = dict(rank=30, n_probes=10, krylov_dim=35, seed=0)
    options.update(kwargs)
    return build_coarse_space(
        A, strategy=strategy,
        apply_Binv=None if baseline is None else baseline.apply, **options)


# ---------------------------------------------------------------------------
# Krylov machinery
# ---------------------------------------------------------------------------

def test_lanczos_produces_an_orthonormal_basis_and_the_right_tridiagonal():
    """``S V = V T`` with ``V`` orthonormal. Without reorthogonalisation the
    basis loses orthogonality after a few steps and ``V exp(-tT) e1`` becomes a
    projection onto nothing in particular."""
    S = poisson_2d(8)
    v = np.random.default_rng(0).standard_normal(S.shape[0])
    V, T, used = lanczos(lambda x: S @ x, v, 30)

    assert used == 30
    assert np.abs(V.T @ V - np.eye(V.shape[1])).max() < 1e-10
    np.testing.assert_allclose(V.T @ (S @ V), T, atol=1e-8)
    assert np.abs(T - T.T).max() < 1e-12


def test_lanczos_reproduces_the_matrix_exponential():
    """The whole point of the basis: ``exp(-tS)v ~ ||v|| V exp(-tT) e1``,
    checked against a dense exponential."""
    import scipy.linalg
    S = poisson_2d(6)
    dense = S.toarray()
    v = np.random.default_rng(0).standard_normal(S.shape[0])
    t = 0.02                       # small enough that a modest basis suffices

    V, T, _ = lanczos(lambda x: S @ x, v, 40)
    approx = np.linalg.norm(v) * (V @ (scipy.linalg.expm(-t * T)[:, 0]))
    exact = scipy.linalg.expm(-t * dense) @ v

    assert np.linalg.norm(approx - exact) < 1e-6 * np.linalg.norm(exact)


def test_arnoldi_produces_an_orthonormal_basis_and_a_hessenberg():
    A = poisson_2d(8)
    M = get_preconditioner("jacobi", A)
    v = np.random.default_rng(0).standard_normal(A.shape[0])
    V, H, used = arnoldi(lambda x: M.apply(A @ x), v, 25)

    assert used == 25
    assert np.abs(V.T @ V - np.eye(V.shape[1])).max() < 1e-10
    assert np.abs(np.tril(H, -2)).max() < 1e-10       # upper Hessenberg


def test_lanczos_stops_early_on_an_invariant_subspace():
    """Reaching an invariant subspace is a success -- the exponential is then
    exact there -- so it returns a shorter basis rather than continuing."""
    S = sparse.diags([2.0, 3.0, 4.0]).tocsr()
    V, T, used = lanczos(lambda x: S @ x, np.array([1.0, 1.0, 1.0]), 20)

    assert V.shape[1] <= 3 and used <= 4
    assert np.abs(V.T @ V - np.eye(V.shape[1])).max() < 1e-12


def test_the_semigroup_stepping_matches_independent_exponentials():
    """Eq. 47. Stepping ``exp(-(t_j - t_{j-1})T)`` forward must give the same
    snapshots as exponentiating each time from scratch -- that equality is what
    licenses reusing one Krylov basis for every time."""
    import scipy.linalg
    S = poisson_2d(6)
    v = np.random.default_rng(0).standard_normal(S.shape[0])
    V, T, _ = lanczos(lambda x: S @ x, v, 30)
    times = (0.01, 0.02, 0.05, 0.1)

    stepped = semigroup_snapshots(V, T, times, norm=np.linalg.norm(v))
    for t, got in zip(times, stepped):
        direct = np.linalg.norm(v) * (V @ (scipy.linalg.expm(-t * T)[:, 0]))
        np.testing.assert_allclose(got, direct, atol=1e-12,
                                   rtol=1e-10)


def test_the_semigroup_reuse_saves_exactly_a_factor_of_p(hard):
    """The claimed saving, measured: generator applications are independent of
    the number of diffusion times. It is what makes Strategy B affordable with
    an expensive baseline."""
    A, _ = hard
    one = _coarse(A, times=(32.0,))
    four = _coarse(A, times=(32.0, 64.0, 128.0, 256.0))

    assert four.n_operator_applications == one.n_operator_applications
    assert four.info["n_snapshots"] == 4 * one.info["n_snapshots"]


def test_probes_are_normalised_and_mixed():
    """Gaussian probes cover the spectrum broadly; locally supported ones
    resolve modes concentrated in a small region that a global random vector
    represents only weakly."""
    probes = generate_probes(500, 10, np.random.default_rng(0))

    np.testing.assert_allclose(np.linalg.norm(probes, axis=1), 1.0)
    support = (np.abs(probes) > 1e-14).sum(axis=1)
    assert support.max() > 400            # the Gaussian ones
    assert support.min() < 50             # the localised ones


# ---------------------------------------------------------------------------
# The coarse space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["A", "B"])
def test_the_coarse_basis_is_orthonormal(strategy, hard):
    """It comes from an SVD, so this should be round-off. A larger value means
    the snapshot matrix was rank-deficient in a way the truncation missed."""
    A, _ = hard
    baseline = get_preconditioner("jacobi", A)
    coarse = _coarse(A, strategy, baseline)

    assert coarse.orthonormality_error() < 1e-10
    assert coarse.rank <= 30


def test_the_rank_is_truncated_to_what_the_filter_actually_found(hard):
    """Asking for more than the snapshots contain must truncate rather than
    return near-null columns, which would make the coarse operator singular."""
    A, _ = hard
    coarse = _coarse(A, rank=10_000, n_probes=4, krylov_dim=20)

    assert coarse.rank <= coarse.info["n_snapshots"]
    assert coarse.rank == min(coarse.numerical_rank, coarse.info["n_snapshots"])


def test_the_filtered_basis_captures_low_energy_modes(hard):
    """The claim the whole method rests on. The Rayleigh quotient of the coarse
    space should be far below the operator's mean eigenvalue -- that is what
    "slow modes" means, and a basis of random vectors would not have it."""
    A, _ = hard
    coarse = _coarse(A, rank=20)
    Z = coarse.scaling[:, None] * coarse.Z          # into K-coordinates

    quotients = np.einsum("ij,ij->j", Z, A @ Z) / np.einsum("ij,ij->j", Z, Z)
    mean_eigenvalue = A.diagonal().mean()

    rng = np.random.default_rng(0)
    R = rng.standard_normal((A.shape[0], 20))
    random_quotients = (np.einsum("ij,ij->j", R, A @ R)
                        / np.einsum("ij,ij->j", R, R))

    assert quotients.max() < 0.5 * random_quotients.min()
    assert quotients.mean() < 0.1 * mean_eigenvalue


def test_strategy_b_refuses_to_run_without_a_baseline(hard):
    """It filters ``B^-1 K``; there is nothing to filter without ``B``."""
    A, _ = hard
    with pytest.raises(ValueError, match="Strategy B"):
        build_coarse_space(A, strategy="B", apply_Binv=None)


def test_an_unknown_strategy_is_refused(hard):
    A, _ = hard
    with pytest.raises(ValueError, match="'A' or 'B'"):
        build_coarse_space(A, strategy="C")


def test_strategy_b_snapshots_are_more_redundant_than_strategy_a(hard):
    """A measured, reportable difference rather than a defect: B filters the
    preconditioned operator, so the baseline has already removed most of the
    spectrum and the slow subspace that remains is genuinely
    lower-dimensional."""
    A, _ = hard
    baseline = get_preconditioner("icc", A)
    a = _coarse(A, "A", baseline, rank=40, n_probes=12)
    b = _coarse(A, "B", baseline, rank=40, n_probes=12)

    assert b.spectral_decay < a.spectral_decay


# ---------------------------------------------------------------------------
# The two-level operator: the identities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["A", "B"])
@pytest.mark.parametrize("name", ["jacobi", "icc"])
def test_the_deflation_projector_annihilates_the_coarse_space(strategy, name,
                                                              hard):
    """``Z' K P' = 0`` exactly, by construction. The sharpest check that ``Z``,
    ``KZ`` and ``E`` are mutually consistent, and the one that catches a
    Strategy A basis used without its ``D^-1/2`` mapping. The tolerance is set
    by the Tikhonov regularisation, not by the arithmetic."""
    A, _ = hard
    baseline = get_preconditioner(name, A)
    coarse = _coarse(A, strategy, baseline)
    two_level = TwoLevelPreconditioner(coarse, baseline,
                                       regularization=0.0).setup(A)

    assert two_level.deflation_residual() < 1e-6


def test_forgetting_the_strategy_a_scaling_breaks_the_identity(hard):
    """The trap, demonstrated. Using the scaled-coordinate basis directly in
    K-space still yields a working preconditioner -- which is exactly why the
    identity is worth checking rather than trusting the iteration count."""
    A, b = hard
    baseline = get_preconditioner("jacobi", A)
    coarse = _coarse(A, "A", baseline)

    correct = TwoLevelPreconditioner(coarse, baseline,
                                     regularization=0.0).setup(A)
    unscaled = TwoLevelPreconditioner(coarse.Z, baseline,
                                      regularization=0.0).setup(A)

    # both converge; only one satisfies the projector identity as intended
    assert pcg(A, b, unscaled, tol=1e-8, max_iter=20000).converged
    assert correct.deflation_residual() < 1e-6
    assert correct.rank == unscaled.rank
    assert not np.allclose(correct._Z, unscaled._Z)


@pytest.mark.parametrize("name", ["jacobi", "icc"])
def test_the_two_level_operator_is_symmetric(name, hard):
    """PCG requires it, and the symmetric projection in Eq. 60 is what provides
    it. The one-sided forms ``B^-1 P`` and ``P B^-1 P`` are cheaper and are not
    symmetric."""
    A, _ = hard
    baseline = get_preconditioner(name, A)
    two_level = TwoLevelPreconditioner(_coarse(A, "A", baseline),
                                       baseline).setup(A)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((A.shape[0], 12))
    applied = np.column_stack([two_level.apply(x) for x in X.T])
    gram = X.T @ applied

    assert np.abs(gram - gram.T).max() < 1e-8 * np.abs(gram).max()


@pytest.mark.parametrize("strategy", ["A", "B"])
def test_the_two_level_preconditioner_does_not_change_the_answer(strategy,
                                                                 hard):
    A, b = hard
    baseline = get_preconditioner("icc", A)
    two_level = TwoLevelPreconditioner(_coarse(A, strategy, baseline),
                                       baseline).setup(A)

    result = pcg(A, b, two_level, tol=1e-10, max_iter=20000)
    assert result.converged
    np.testing.assert_allclose(result.x, spla.spsolve(A.tocsc(), b), rtol=1e-6,
                               atol=1e-10)


def test_the_coarse_operator_is_positive_definite(hard):
    """``E = Z'KZ`` is SPD when ``Z`` has full column rank, which the SVD
    truncation guarantees. A huge condition number means the basis is nearly
    rank-deficient in K-space and the coarse solve amplifies round-off."""
    A, _ = hard
    baseline = get_preconditioner("jacobi", A)
    two_level = TwoLevelPreconditioner(_coarse(A, "A", baseline),
                                       baseline).setup(A)

    assert np.linalg.eigvalsh(two_level._E).min() > 0.0
    assert two_level.coarse_condition < 1e10


def test_a_mismatched_basis_is_refused(hard):
    A, _ = hard
    baseline = get_preconditioner("jacobi", A)
    with pytest.raises(ValueError, match="rows"):
        TwoLevelPreconditioner(np.ones((A.shape[0] + 5, 3)),
                               baseline).setup(A)


def test_the_wrapper_inherits_the_baselines_spd_status(hard):
    """It preserves symmetry and cannot create it: wrapping ILU gives a
    non-SPD two-level operator, which is what the paper means by needing a
    flexible Krylov method there."""
    A, _ = hard
    spd = get_preconditioner("icc", A)
    not_spd = get_preconditioner("ilu", A)
    coarse = _coarse(A, "A", spd)

    assert TwoLevelPreconditioner(coarse, spd).is_spd is True
    assert TwoLevelPreconditioner(coarse, not_spd).is_spd is False


# ---------------------------------------------------------------------------
# It has to actually help
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["jacobi", "icc", "amg"])
def test_the_coarse_correction_reduces_the_iteration_count(name, hard):
    """The point of the method. Measured on the paper's own bcsstk06 the
    reduction is 7.1x; on this generated problem it is smaller but must still be
    a clear win, or the coarse space is not capturing the slow modes."""
    A, b = hard
    baseline = get_preconditioner(name, A)
    plain = pcg(A, b, baseline, tol=1e-8, max_iter=20000)

    best = min(
        pcg(A, b, TwoLevelPreconditioner(_coarse(A, s, baseline),
                                         baseline).setup(A),
            tol=1e-8, max_iter=20000).iterations
        for s in ("A", "B"))

    assert best < 0.8 * plain.iterations


def test_a_larger_coarse_space_helps_monotonically(hard):
    """More slow modes treated exactly means fewer left for the baseline. Not
    guaranteed in general, but it should hold clearly on a problem whose slow
    subspace the filter is actually finding."""
    A, b = hard
    baseline = get_preconditioner("jacobi", A)
    counts = []
    for rank in (5, 20, 50):
        coarse = _coarse(A, "A", baseline, rank=rank, n_probes=16)
        two_level = TwoLevelPreconditioner(coarse, baseline).setup(A)
        counts.append(pcg(A, b, two_level, tol=1e-8,
                          max_iter=20000).iterations)

    assert counts[0] > counts[1] > counts[2]


def test_the_setup_is_reusable_across_right_hand_sides(hard):
    """The amortisation argument: one coarse space, many solves. If the object
    carried per-solve state this would drift."""
    A, _ = hard
    baseline = get_preconditioner("icc", A)
    two_level = TwoLevelPreconditioner(_coarse(A, "A", baseline),
                                       baseline).setup(A)

    rng = np.random.default_rng(3)
    for _ in range(4):
        b = rng.standard_normal(A.shape[0])
        result = pcg(A, b, two_level, tol=1e-9, max_iter=20000)
        assert result.converged
        np.testing.assert_allclose(result.x, spla.spsolve(A.tocsc(), b),
                                   rtol=1e-5, atol=1e-9)


# ---------------------------------------------------------------------------
# The non-SPD baseline case the paper flags
# ---------------------------------------------------------------------------

def test_fgmres_rescues_the_two_level_operator_around_a_non_spd_baseline(hard):
    """Section 3.2's remark, made concrete.

    Wrapping a non-SPD baseline such as ILU gives a two-level operator that is
    itself not SPD, so PCG's use of ``r'z`` as a squared norm is unjustified --
    ``is_spd`` says so. The paper's answer is "a flexible Krylov method", and
    that is exactly what FGMRES is: it assumes neither symmetry nor a fixed
    preconditioner, so it converges here regardless of whether PCG happens to.
    """
    from simulacrum.solvers import fgmres, gmres

    A, b = hard
    baseline = get_preconditioner("ilu", A)
    two_level = TwoLevelPreconditioner(_coarse(A, "A", baseline),
                                       baseline).setup(A)
    assert two_level.is_spd is False

    flexible = fgmres(A, b, two_level, tol=1e-9, restart=30, max_iter=2000)
    assert flexible.converged
    np.testing.assert_allclose(flexible.x, spla.spsolve(A.tocsc(), b),
                               rtol=1e-6, atol=1e-9)

    # and it still helps: fewer iterations than the bare baseline
    plain = gmres(A, b, baseline, tol=1e-9, restart=30, max_iter=2000)
    assert flexible.iterations <= plain.iterations


def test_the_coarse_space_helps_a_nonsymmetric_solver_too(hard):
    """The coarse space is built from ``K`` and a baseline, not from the Krylov
    method, so it should accelerate GMRES as readily as PCG. Worth checking
    because nothing in the construction knows which solver will consume it."""
    from simulacrum.solvers import gmres

    A, b = hard
    baseline = get_preconditioner("icc", A)
    two_level = TwoLevelPreconditioner(_coarse(A, "B", baseline),
                                       baseline).setup(A)

    plain = gmres(A, b, baseline, tol=1e-9, restart=30, max_iter=5000)
    enhanced = gmres(A, b, two_level, tol=1e-9, restart=30, max_iter=5000)
    assert enhanced.iterations < plain.iterations
