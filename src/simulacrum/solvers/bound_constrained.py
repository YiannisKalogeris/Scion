r"""
Bound-constrained solvers.

Needed by AT1, and by any formulation that enforces irreversibility as
``d >= d_prev`` rather than through a history field.

The problem is a linear complementarity system: minimise ``1/2 x'Ax - b'x``
subject to ``lo <= x <= hi`` with ``A`` symmetric positive definite. Solving it
as an unconstrained system and clipping afterwards is *not* equivalent and
gives visibly wrong damage fields -- the clipped values change the residual at
their neighbours, which nobody re-solves.

:func:`solve_bounded` implements a primal-dual active set method (equivalently
a semismooth Newton method on the complementarity function). It converges in a
handful of iterations, terminates finitely, and each iteration is one solve on
the current inactive set -- so it costs about as much as two or three
unconstrained solves, not an order of magnitude more.

A projected Gauss-Seidel fallback is included for the case where ``A`` is not
positive definite and the active-set iteration cycles.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

__all__ = ["solve_bounded", "projected_gauss_seidel", "BoundedSolveResult"]


class BoundedSolveResult:
    """Outcome of a bound-constrained solve."""

    __slots__ = ("x", "converged", "iterations", "n_active", "residual")

    def __init__(self, x, converged, iterations, n_active, residual):
        self.x = x
        self.converged = converged
        self.iterations = iterations
        self.n_active = n_active
        self.residual = residual

    def __repr__(self) -> str:
        return (f"BoundedSolveResult(converged={self.converged}, "
                f"iterations={self.iterations}, n_active={self.n_active}, "
                f"residual={self.residual:.3e})")


def solve_bounded(A, b, lo=None, hi=None, x0=None, c=0.0,
                  max_iter: int | None = None, tol: float = 1.0e-10
                  ) -> BoundedSolveResult:
    """Solve ``A x = b`` subject to ``lo <= x <= hi`` by primal-dual active set.

    Parameters
    ----------
    A   : (n, n) sparse or dense, symmetric positive definite
    b   : (n,)
    lo, hi : bounds, scalars or ``(n,)``; ``None`` means unbounded on that side
    x0  : starting guess; the previous step's solution is a good one and
          usually cuts the iteration count dramatically
    max_iter : default ``max(100, n)``. Primal-dual active set usually
          terminates in a handful of iterations, but it degrades to O(n) when
          the free set grows one unknown at a time -- which is exactly what a
          spreading crack does under AT1. A fixed small cap silently returns a
          truncated damage profile that still looks like a crack.
    c   : complementarity scaling, scalar or ``(n,)``. Affects only which
          constraints are guessed active, not the converged answer -- but the
          guess has to be on the scale of each *row* of ``A``. Leave it at the
          default 0, which uses ``|diag(A)|`` elementwise.

          A single scalar is not good enough when the diagonal is
          heterogeneous, which it is for phase-field fracture: the mass term
          carries ``g''(d) H``, and ``H`` spans orders of magnitude between
          cracked and intact material. With a scalar ``c`` taken from the
          largest row, ``y = x - lam/c`` collapses to ``x`` everywhere else,
          the dual information is discarded, the method degenerates into naive
          clipping, and the active set cycles instead of terminating.

    Notes
    -----
    The active set is predicted from ``y = x - lam/c`` where ``lam = A x - b``
    is the multiplier estimate: a constraint is active when *y* violates the
    bound, not when ``x`` does. That combination of primal and dual information
    is what makes this a Newton method on the complementarity condition rather
    than naive clipping-and-repeat, and it is why the iteration terminates
    finitely instead of oscillating between two candidate sets.
    """
    A = A.tocsr() if sparse.issparse(A) else np.asarray(A)
    n = b.shape[0]
    b = np.asarray(b, dtype=float).ravel()

    lo = np.full(n, -np.inf) if lo is None else np.broadcast_to(
        np.asarray(lo, dtype=float), (n,)).copy()
    hi = np.full(n, np.inf) if hi is None else np.broadcast_to(
        np.asarray(hi, dtype=float), (n,)).copy()

    c = np.asarray(c, dtype=float)
    if c.ndim == 0 and c <= 0.0:
        diag = A.diagonal() if sparse.issparse(A) else np.diag(A)
        c = np.abs(diag)
        floor = c[c > 0.0].min() if (c > 0.0).any() else 1.0
        c = np.where(c > 0.0, c, floor)
    else:
        c = np.broadcast_to(c, (n,)).copy()

    if max_iter is None:
        max_iter = max(100, n)

    x = np.clip(np.zeros(n) if x0 is None
                else np.asarray(x0, dtype=float).ravel(), lo, hi)

    def matvec(v):
        return A @ v

    prev_active = None
    for it in range(1, max_iter + 1):
        lam = matvec(x) - b                       # multiplier estimate
        y = x - lam / c
        at_lo = y < lo
        at_hi = y > hi
        active = at_lo | at_hi

        # finite termination: the predicted active set stopped changing, so the
        # current x already solves the reduced system exactly
        if prev_active is not None and np.array_equal(active, prev_active):
            free = ~active
            res = (float(np.linalg.norm((matvec(x) - b)[free]))
                   if free.any() else 0.0)
            return BoundedSolveResult(x, True, it, int(active.sum()), res)
        prev_active = active

        x_new = np.empty(n)
        x_new[at_lo] = lo[at_lo]
        x_new[at_hi] = hi[at_hi]

        free = ~active
        if free.any():
            if sparse.issparse(A):
                A_ff = A[free][:, free].tocsc()
                rhs = b[free] - (A[free][:, active] @ x_new[active]
                                 if active.any() else 0.0)
                x_new[free] = spla.spsolve(A_ff, rhs)
            else:
                A_ff = A[np.ix_(free, free)]
                rhs = b[free] - (A[np.ix_(free, active)] @ x_new[active]
                                 if active.any() else 0.0)
                x_new[free] = np.linalg.solve(A_ff, rhs)

        if np.linalg.norm(x_new - x) <= tol * max(1.0, np.linalg.norm(x_new)):
            x = x_new
            res = float(np.linalg.norm((matvec(x) - b)[free])) if free.any() else 0.0
            return BoundedSolveResult(x, True, it, int(active.sum()), res)
        x = x_new

    import warnings
    res = float(np.linalg.norm(matvec(x) - b))
    warnings.warn(
        f"bound-constrained solve did not converge in {max_iter} iterations "
        f"(residual {res:.3e}); the result may be a truncated solution",
        RuntimeWarning, stacklevel=2,
    )
    return BoundedSolveResult(x, False, max_iter, int(active.sum()), res)


def projected_gauss_seidel(A, b, lo=None, hi=None, x0=None,
                           max_iter: int = 500, tol: float = 1.0e-10
                           ) -> BoundedSolveResult:
    """Fallback for indefinite ``A``: slow but very robust.

    Sweeps the unknowns one at a time, projecting each onto its bounds
    immediately, so it cannot cycle the way an active-set method can.
    """
    A = A.tocsr() if sparse.issparse(A) else sparse.csr_matrix(A)
    n = b.shape[0]
    b = np.asarray(b, dtype=float).ravel()
    lo = np.full(n, -np.inf) if lo is None else np.broadcast_to(
        np.asarray(lo, dtype=float), (n,)).copy()
    hi = np.full(n, np.inf) if hi is None else np.broadcast_to(
        np.asarray(hi, dtype=float), (n,)).copy()
    x = np.clip(np.zeros(n) if x0 is None
                else np.asarray(x0, dtype=float).ravel(), lo, hi)

    diag = A.diagonal()
    indptr, indices, data = A.indptr, A.indices, A.data

    for it in range(1, max_iter + 1):
        delta = 0.0
        for i in range(n):
            if diag[i] == 0.0:
                continue
            s = b[i]
            for k in range(indptr[i], indptr[i + 1]):
                j = indices[k]
                if j != i:
                    s -= data[k] * x[j]
            xi = min(max(s / diag[i], lo[i]), hi[i])
            delta = max(delta, abs(xi - x[i]))
            x[i] = xi
        if delta <= tol:
            return BoundedSolveResult(x, True, it, -1, delta)
    return BoundedSolveResult(x, False, max_iter, -1, delta)
