r"""
Finite-difference verification of a tangent operator.

The tangent is ``K = dR/du`` for whatever residual ``R(u)`` a formulation
defines, so a central difference on the residual reconstructs it without
knowing any of the physics. That makes this the same test for plasticity,
hyperelasticity, contact and phase-field.

Reason for having it in the library
----------------------
A wrong tangent does not usually produce a wrong answer. Newton still converges
if the residual is right -- just linearly instead of quadratically, so the
symptom is "this takes 40 iterations" rather than a failure. That is easy to
misread as a hard problem, a bad initial guess, or a step size that needs
reducing. Plasticity is where this bites hardest: the *consistent* (algorithmic)
tangent ``d(delta sigma)/d(delta eps)`` of the return map differs from the
continuum elastoplastic tangent ``C_ep``, the two agree in the limit of
infinitesimal steps, and substituting one for the other is both a common
mistake and invisible except through the convergence rate.

Cost
----
Central differences need ``2 n`` residual evaluations, so this is a test on a
small mesh, not a production check. A few hundred degrees of freedom takes
seconds; twenty thousand takes hours. 

Usage
-----
    from simulacrum.solvers.tangent_check import check_tangent

    report = check_tangent(residual, tangent, u, free=free_dofs)
    print(report)
    assert report.passed
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["numerical_tangent", "check_tangent", "TangentReport"]


def _to_dense(K) -> np.ndarray:
    if hasattr(K, "toarray"):
        return K.toarray()
    return np.asarray(K)


def numerical_tangent(residual, u, free=None, h=None, scale=None) -> np.ndarray:
    """``dR/du`` by central differences.

    Parameters
    ----------
    residual : callable ``u -> R``, both ``(n,)``. Must be a pure function of
        ``u``: any internal state it updates has to be restored, or the
        differences pick up the state change as well as the derivative. For a
        plasticity residual this means evaluating against a *fixed* committed
        state, never a trial state the call itself advances.
    u : (n,) current configuration
    free : optional index array; only these rows and columns are perturbed and
        returned. Use it to skip constrained degrees of freedom, whose residual
        rows are typically overwritten by the boundary-condition treatment and
        carry no derivative information.
    h : step size. Default ``eps**(1/3) * scale`` per degree of freedom, the
        minimum of the truncation-plus-round-off error for a central difference
        in float64. A step far below this is dominated by cancellation, which
        is the usual reason a correct tangent appears to fail this test.
    scale : characteristic magnitude of ``u``; defaults to ``max(|u|, 1)``.
    """
    u = np.asarray(u, dtype=float)
    n = u.size
    idx = np.arange(n) if free is None else np.asarray(free, dtype=int)

    if scale is None:
        scale = max(float(np.abs(u).max()), 1.0)
    if h is None:
        h = float(np.cbrt(np.finfo(float).eps) * scale)

    K = np.zeros((idx.size, idx.size))
    for c, j in enumerate(idx):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        K[:, c] = (np.asarray(residual(up), dtype=float)[idx]
                   - np.asarray(residual(um), dtype=float)[idx]) / (2.0 * h)
    return K


@dataclass
class TangentReport:
    """Outcome of a tangent check."""

    passed: bool
    max_abs: float
    max_rel: float
    worst: tuple
    norm_ratio: float
    symmetry: float
    n_dof: int
    step: float

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (f"TangentReport({status}, n={self.n_dof}, "
                f"max_rel={self.max_rel:.3e} at {self.worst}, "
                f"max_abs={self.max_abs:.3e}, "
                f"||K_a||/||K_n||={self.norm_ratio:.6f}, "
                f"asym={self.symmetry:.3e}, h={self.step:.2e})")


def check_tangent(residual, tangent, u, free=None, h=None, scale=None,
                  rtol: float = 1.0e-5, verbose: bool = False) -> TangentReport:
    """Compare an analytic tangent against central differences.

    ``tangent`` is either an array (dense or sparse) or a callable ``u -> K``.

    The pass criterion is a *relative* one, entry by entry, normalised by the
    largest entry of the numerical tangent rather than by each entry itself --
    small entries of a stiffness matrix are dominated by round-off and their
    individual relative errors are meaningless.

    ``rtol`` of 1e-5 is about right: a central difference is accurate to
    ``eps**(2/3)``, roughly 4e-11 relative, but the residual assembly itself
    accumulates error over the element loop, so demanding better than 1e-6
    produces false failures on large meshes.

    ``symmetry`` reports ``||K - K^T|| / ||K||`` of the *analytic* tangent.
    This is diagnostic, not a failure: an unsymmetric tangent is correct and
    expected for non-associative plastic flow, and a check that insisted on
    symmetry would reject exactly the models that need testing most.
    """
    u = np.asarray(u, dtype=float)
    idx = np.arange(u.size) if free is None else np.asarray(free, dtype=int)

    K_num = numerical_tangent(residual, u, free=idx, h=h, scale=scale)
    K_ana = _to_dense(tangent(u) if callable(tangent) else tangent)
    if K_ana.shape[0] != idx.size:
        K_ana = K_ana[np.ix_(idx, idx)]

    diff = K_ana - K_num
    denom = max(float(np.abs(K_num).max()), 1.0e-300)
    max_abs = float(np.abs(diff).max())
    max_rel = max_abs / denom
    worst = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(diff)),
                                                   diff.shape))

    n_ana = float(np.linalg.norm(K_ana))
    n_num = float(np.linalg.norm(K_num))
    asym = (float(np.linalg.norm(K_ana - K_ana.T)) / max(n_ana, 1e-300))

    if scale is None:
        scale = max(float(np.abs(u).max()), 1.0)
    step = float(np.cbrt(np.finfo(float).eps) * scale) if h is None else float(h)

    report = TangentReport(
        passed=max_rel <= rtol, max_abs=max_abs, max_rel=max_rel, worst=worst,
        norm_ratio=n_ana / max(n_num, 1e-300), symmetry=asym,
        n_dof=int(idx.size), step=step)

    if verbose:
        print(report)
        if not report.passed:
            i, j = worst
            print(f"  analytic[{i},{j}] = {K_ana[i, j]: .6e}")
            print(f"  numeric [{i},{j}] = {K_num[i, j]: .6e}")
            print(f"  ratio             = "
                  f"{K_ana[i, j] / K_num[i, j] if K_num[i, j] else np.nan: .6f}")
            if abs(report.norm_ratio - 1.0) < 1e-3 and asym > 1e-8:
                print("  norms match but the tangent is unsymmetric: check "
                      "whether a transpose is missing somewhere")
    return report
