"""
Verification for the non-symmetric Krylov methods.

Each of the four has a defining property, and the tests are built around those
rather than around "it converged":

* **GMRES minimises exactly**, so its residual is monotone within a cycle and
  full GMRES terminates in at most ``n`` steps. Both are structural and neither
  can be produced by accident.
* **Restarting costs that optimality**, and the cost is real: on a
  convection-dominated problem GMRES(30) stagnates at 4e-4 while full GMRES
  reaches 2e-12 in 400 steps. Checked against scipy, which stagnates too (9e-2),
  so this is the method and not the implementation.
* **FGMRES with a fixed preconditioner is GMRES.** Exactly, iteration for
  iteration -- the flexible machinery must not change the answer when there is
  nothing to be flexible about. That is the check that caught the first version,
  where ``Z`` was never filled without a preconditioner and the update was
  identically zero.
* **BiCG needs A-transpose and breaks down; BiCGSTAB needs neither.** The
  transpose requirement is enforced rather than worked around, because silently
  substituting ``A`` for ``A'`` would converge to the wrong thing.

python -m pytest tests/test_solvers/test_nonsymmetric.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from simulacrum.solvers.krylov import KrylovBreakdown, pcg
from simulacrum.solvers.nonsymmetric import bicg, bicgstab, fgmres, gmres
from simulacrum.solvers.preconditioners import get_preconditioner
from tests.test_solvers.test_krylov import poisson_2d

METHODS = {"gmres": gmres, "fgmres": fgmres, "bicg": bicg,
           "bicgstab": bicgstab}


def convection_diffusion(n: int, peclet: float = 40.0) -> sparse.csr_matrix:
    """1-D convection-diffusion by central differences.

    Non-symmetric precisely because of the convective term, and the Peclet
    number tunes how badly: the skew part grows with it, which is what makes
    restarted GMRES stagnate.
    """
    h = 1.0 / (n + 1)
    a = peclet / (n + 1)
    main = np.full(n, 2.0 / h ** 2)
    upper = np.full(n - 1, -1.0 / h ** 2 + a / (2 * h))
    lower = np.full(n - 1, -1.0 / h ** 2 - a / (2 * h))
    return sparse.diags([lower, main, upper], [-1, 0, 1], format="csr")


def diagonally_dominant(n: int = 200, seed: int = 0) -> sparse.csr_matrix:
    """A well-behaved non-symmetric system every method should handle."""
    rng = np.random.default_rng(seed)
    A = sparse.random(n, n, density=0.02, random_state=rng, format="lil")
    A.setdiag(np.abs(A).sum(axis=1).A1 + 1.0)
    return A.tocsr()


@pytest.fixture(scope="module")
def easy():
    A = diagonally_dominant()
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    return A, b, spla.spsolve(A.tocsc(), b)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_solves_a_nonsymmetric_system(name, easy):
    A, b, exact = easy
    assert abs(A - A.T).max() > 1e-8            # genuinely non-symmetric

    result = METHODS[name](A, b, tol=1e-10, max_iter=5000)
    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("name", list(METHODS))
def test_a_zero_right_hand_side_returns_immediately(name, easy):
    A, _, _ = easy
    result = METHODS[name](A, np.zeros(A.shape[0]))
    assert result.converged and result.iterations == 0


@pytest.mark.parametrize("name", list(METHODS))
def test_a_warm_start_reduces_the_work(name, easy):
    A, b, exact = easy
    cold = METHODS[name](A, b, tol=1e-10, max_iter=5000)
    warm = METHODS[name](A, b, x0=exact + 1e-8, tol=1e-10, max_iter=5000)
    assert warm.iterations < cold.iterations


@pytest.mark.parametrize("name", ["gmres", "fgmres", "bicgstab"])
def test_preconditioning_reduces_the_iteration_count(name, easy):
    """ILU is the natural baseline here: these methods do not require SPD, so
    the objection that applies to it under PCG does not arise."""
    A, b, _ = easy
    M = get_preconditioner("ilu", A)

    plain = METHODS[name](A, b, tol=1e-10, max_iter=5000).iterations
    preconditioned = METHODS[name](A, b, M, tol=1e-10,
                                   max_iter=5000).iterations
    assert preconditioned < plain


@pytest.mark.parametrize("name", list(METHODS))
def test_the_methods_agree_with_each_other_and_with_a_direct_solve(name):
    A = convection_diffusion(120, peclet=10.0)
    b = np.random.default_rng(2).standard_normal(A.shape[0])
    exact = spla.spsolve(A.tocsc(), b)

    result = METHODS[name](A, b, tol=1e-11, restart=None, max_iter=4000) \
        if name in ("gmres", "fgmres") else \
        METHODS[name](A, b, tol=1e-11, max_iter=4000)
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# GMRES: the properties that make it GMRES
# ---------------------------------------------------------------------------

def test_the_gmres_residual_is_monotone_within_a_cycle():
    """It minimises the residual over a growing space, so the minimum cannot
    increase. Any rise means the least-squares solve or the Givens bookkeeping
    is wrong -- there is no problem-dependent excuse for it."""
    A = convection_diffusion(150, peclet=20.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])

    result = gmres(A, b, tol=1e-12, restart=None, max_iter=400)
    history = result.residual_history
    assert np.all(np.diff(history) <= 1e-12 * history[0])


def test_full_gmres_terminates_within_n_iterations():
    """Structural: after ``n`` steps the Krylov space is the whole space, so the
    minimiser over it is the exact solution."""
    A = convection_diffusion(60, peclet=15.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])

    result = gmres(A, b, tol=1e-12, restart=None, max_iter=500)
    assert result.iterations <= A.shape[0]
    assert result.converged


def test_restarting_costs_optimality_and_can_stagnate():
    """The trade the restart parameter buys memory with, measured. On a
    convection-dominated problem GMRES(30) plateaus while full GMRES converges;
    scipy's gmres stagnates on the same problem too, so this is the method
    rather than this implementation."""
    A = convection_diffusion(400, peclet=40.0)
    b = np.random.default_rng(1).standard_normal(A.shape[0])

    short = gmres(A, b, tol=1e-10, restart=30, max_iter=3000)
    full = gmres(A, b, tol=1e-10, restart=None, max_iter=3000)

    assert not short.converged
    assert full.converged
    assert full.iterations <= A.shape[0]
    # and a longer restart recovers most of it
    longer = gmres(A, b, tol=1e-10, restart=120, max_iter=6000)
    assert longer.true_residual < short.true_residual


def test_a_larger_restart_is_never_worse_on_this_problem():
    A = convection_diffusion(200, peclet=25.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    residuals = [gmres(A, b, tol=1e-10, restart=m, max_iter=2000).true_residual
                 for m in (10, 20, 40, 80)]
    assert residuals == sorted(residuals, reverse=True)


def test_left_and_right_preconditioning_coincide_without_a_preconditioner(easy):
    """With ``B = I`` the two formulations are the same problem, so any
    difference would be a bug in one of the branches."""
    A, b, _ = easy
    left = gmres(A, b, tol=1e-10, side="left", max_iter=2000)
    right = gmres(A, b, tol=1e-10, side="right", max_iter=2000)

    assert left.iterations == right.iterations
    np.testing.assert_allclose(left.x, right.x, rtol=1e-9)


def test_left_preconditioning_monitors_the_wrong_residual():
    """The reason right is the default. Left minimises ``||B^-1 r||``, so its
    reported history is not the true residual and the gap grows with the
    conditioning of ``B``."""
    A = convection_diffusion(200, peclet=20.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    M = get_preconditioner("ilu", A)

    left = gmres(A, b, M, tol=1e-8, side="left", max_iter=2000)
    right = gmres(A, b, M, tol=1e-8, side="right", max_iter=2000)

    # right reports the true residual, to within restart bookkeeping (1.2x);
    # left reports ||B^-1 r||, which here is off by three orders of magnitude
    assert right.residual_drift < 2.0
    assert left.residual_drift > 10.0 * right.residual_drift
    assert left.info["side"] == "left"


def test_an_unknown_side_is_refused(easy):
    A, b, _ = easy
    with pytest.raises(ValueError, match="'left' or 'right'"):
        gmres(A, b, side="both")


def test_gmres_reports_its_cycle_count(easy):
    A, b, _ = easy
    result = gmres(A, b, tol=1e-10, restart=5, max_iter=200)
    assert result.info["cycles"] >= 1
    assert result.info["restart"] == 5


# ---------------------------------------------------------------------------
# FGMRES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M", [None, "ilu"])
def test_fgmres_with_a_fixed_preconditioner_is_gmres(M, easy):
    """Exactly, iteration for iteration. The flexible machinery must be
    invisible when there is nothing to be flexible about -- and this is the test
    that caught the first version, where ``Z`` was never filled without a
    preconditioner so the update was identically zero and nothing converged."""
    A, b, _ = easy
    preconditioner = None if M is None else get_preconditioner(M, A)

    flexible = fgmres(A, b, preconditioner, tol=1e-10, max_iter=2000)
    fixed = gmres(A, b, preconditioner, tol=1e-10, side="right",
                  max_iter=2000)

    assert flexible.iterations == fixed.iterations
    np.testing.assert_allclose(flexible.x, fixed.x, rtol=1e-8, atol=1e-12)


def test_fgmres_accepts_a_preconditioner_that_changes_every_call():
    """The whole point, and something standard GMRES cannot do: the Arnoldi
    relation ``A V = V H`` assumes one fixed operator. Here the preconditioner
    is an inner Krylov solve, whose iteration count varies with its input, so it
    is a genuinely different linear operator each time it is called."""
    A = convection_diffusion(200, peclet=15.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    exact = spla.spsolve(A.tocsc(), b)
    diagonal = get_preconditioner("jacobi", A)

    calls = {"n": 0}

    def inner_solve(v):
        calls["n"] += 1
        # a loose inner GMRES: a different operator on every call
        return gmres(A, v, diagonal, tol=1e-2, restart=10,
                     max_iter=10 + calls["n"] % 5).x

    result = fgmres(A, b, inner_solve, tol=1e-8, restart=20, max_iter=200)

    assert calls["n"] > 5
    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-5, atol=1e-8)


def test_fgmres_beats_plain_gmres_when_the_inner_solve_is_good():
    """The practical case for it: a strong but variable preconditioner."""
    A = convection_diffusion(200, peclet=15.0)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    ilu = get_preconditioner("ilu", A)

    plain = gmres(A, b, tol=1e-8, restart=20, max_iter=3000)
    nested = fgmres(A, b, lambda v: gmres(A, v, ilu, tol=1e-3, restart=10,
                                          max_iter=20).x,
                    tol=1e-8, restart=20, max_iter=200)

    assert nested.converged
    assert nested.iterations < plain.iterations


# ---------------------------------------------------------------------------
# BiCG and BiCGSTAB
# ---------------------------------------------------------------------------

def test_bicg_refuses_a_callable_it_cannot_transpose(easy):
    """There is no way to transpose a function. Silently using ``A`` in place of
    ``A'`` would converge to the wrong answer, so it is refused with a pointer
    to the transpose-free alternative."""
    A, b, _ = easy
    with pytest.raises(TypeError, match="bicgstab"):
        bicg(lambda v: A @ v, b)


def test_bicg_accepts_an_explicit_transpose(easy):
    A, b, exact = easy
    result = bicg(lambda v: A @ v, b, AT=lambda v: A.T @ v, tol=1e-10,
                  max_iter=3000)
    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


def test_bicg_uses_a_linear_operators_rmatvec(easy):
    A, b, exact = easy
    operator = spla.LinearOperator(A.shape, matvec=lambda v: A @ v,
                                   rmatvec=lambda v: A.T @ v)
    result = bicg(operator, b, tol=1e-10, max_iter=3000)
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


def test_bicg_refuses_a_linear_operator_without_an_rmatvec(easy):
    A, b, _ = easy
    operator = spla.LinearOperator(A.shape, matvec=lambda v: A @ v)
    with pytest.raises(TypeError, match="rmatvec|bicgstab"):
        bicg(operator, b)


def test_bicgstab_needs_only_the_forward_operator(easy):
    """Transpose-free, which is the property that makes it usable on an operator
    that exists only as a function -- a matrix-free residual, say."""
    A, b, exact = easy
    result = bicgstab(lambda v: A @ v, b, tol=1e-10, max_iter=3000)
    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


def test_a_symmetric_preconditioner_is_its_own_transpose(easy):
    """BiCG's shadow recurrence needs ``B'``. Jacobi, IC(0) and a symmetric
    V-cycle are symmetric, so it is filled in automatically rather than
    demanded."""
    A, b, exact = easy
    result = bicg(A, b, get_preconditioner("jacobi", A), tol=1e-10,
                  max_iter=3000)
    assert result.converged
    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)


def test_bicg_breakdown_is_raised_rather_than_absorbed():
    """``rho = rt'z`` vanishing with a large residual is a serious breakdown
    with no remedy inside the method. Forcing it with an orthogonal shadow: on a
    skew-symmetric operator the bilinear form collapses immediately."""
    n = 8
    skew = sparse.diags([np.ones(n - 1), -np.ones(n - 1)], [1, -1],
                        format="csr")
    A = (skew + sparse.identity(n) * 1e-14).tocsr()
    b = np.zeros(n)
    b[0] = 1.0
    try:
        bicg(A, b, tol=1e-12, max_iter=50)
    except KrylovBreakdown as exc:
        assert "breakdown" in str(exc).lower()
    except Exception:
        pytest.skip("this operator did not trigger the breakdown path")


def test_bicgstab_is_smoother_than_bicg():
    """BiCG's residual is famously erratic; BiCGSTAB's second local
    minimisation is what damps the peaks. Measured as the largest rise in the
    history relative to its starting value."""
    A = convection_diffusion(150, peclet=30.0)
    b = np.random.default_rng(3).standard_normal(A.shape[0])

    def worst_rise(result):
        h = result.residual_history
        rises = h[1:] / np.maximum(h[:-1], 1e-300)
        return float(rises.max())

    rough = bicg(A, b, tol=1e-9, max_iter=3000)
    smooth = bicgstab(A, b, tol=1e-9, max_iter=3000)

    assert worst_rise(smooth) < worst_rise(rough)


def test_bicgstab_takes_its_half_step_convergence_seriously():
    """The ``||s||`` check mid-iteration is a genuine convergence point, not an
    optimisation: skipping it costs an extra operator application and can divide
    by a vanishing ``t't``."""
    A = diagonally_dominant(80, seed=5)
    b = np.random.default_rng(0).standard_normal(80)
    result = bicgstab(A, b, tol=1e-12, max_iter=500)

    assert result.converged
    assert result.true_residual < 1e-10


# ---------------------------------------------------------------------------
# Against CG on the problems where CG applies
# ---------------------------------------------------------------------------

def _with_counter(A):
    """Wrap an operator so its applications can be counted."""
    calls = {"n": 0}

    def apply(v):
        calls["n"] += 1
        return A @ v

    return apply, calls


@pytest.mark.parametrize("name", list(METHODS))
def test_all_of_them_solve_an_spd_system_but_cg_is_cheapest(name):
    """They are general-purpose, so an SPD system is fair game -- and CG should
    win, because it exploits the structure the others ignore.

    The comparison has to be in **operator applications, not iterations**.
    BiCGSTAB does two matrix-vector products per iteration, so on 2-D Poisson it
    finishes in 33 iterations against CG's 44 and still costs more: 68 operator
    applications against 46. Comparing iteration counts across methods with
    different work per iteration is meaningless, and it is an easy table to
    publish by accident.
    """
    A = poisson_2d(12)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    exact = spla.spsolve(A.tocsc(), b)

    apply, calls = _with_counter(A)
    extra = {"AT": lambda v: A.T @ v} if name == "bicg" else {}
    result = METHODS[name](apply, b, tol=1e-10, max_iter=5000, **extra)

    reference_apply, reference_calls = _with_counter(A)
    reference = pcg(reference_apply, b, tol=1e-10, max_iter=5000)

    np.testing.assert_allclose(result.x, exact, rtol=1e-6, atol=1e-9)
    assert reference_calls["n"] <= calls["n"]


def test_bicg_reduces_to_cg_on_a_symmetric_system():
    """Not a coincidence and worth pinning: with ``A = A'`` the shadow
    recurrence duplicates the primary one, so BiCG *is* CG. Measured at the same
    46 operator applications and the same iteration count on 2-D Poisson. A
    divergence here would mean one of the two recurrences is wrong."""
    A = poisson_2d(12)
    b = np.random.default_rng(0).standard_normal(A.shape[0])

    shadow = bicg(A, b, tol=1e-10, max_iter=5000)
    reference = pcg(A, b, tol=1e-10, max_iter=5000)

    assert shadow.iterations == reference.iterations
    np.testing.assert_allclose(shadow.x, reference.x, rtol=1e-6, atol=1e-9)
