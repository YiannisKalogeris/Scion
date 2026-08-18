r"""
Conjugate gradients, with and without a preconditioner.

CG is the method for SPD systems: it needs only matrix-vector products, stores a
handful of vectors, and in exact arithmetic terminates in at most ``n`` steps.
What decides its practical cost is not ``n`` but the spectrum of the operator the
iteration actually sees,

    ||e_k||_K / ||e_0||_K  <=  2 ((sqrt(kappa) - 1)/(sqrt(kappa) + 1))^k,

with ``kappa`` the condition number. Preconditioning replaces ``kappa`` by
``cond(B^-1 K)``, which is the entire point of :mod:`~.preconditioners`.

That bound is worth reading carefully, because it is routinely over-trusted. It
depends on ``kappa`` alone, while real convergence depends on the *whole*
spectrum -- clustering and outliers included. A preconditioner that lowers the
condition number need not reduce the iteration count, and one that leaves it
alone can help a great deal by clustering the interior eigenvalues. Report
iterations, not condition numbers.

Three failure modes, each detected rather than smoothed over
-----------------------------------------------------------
**A is not SPD.** Then some search direction has ``p' K p <= 0``, the step length
is meaningless, and the iteration wanders. The usual implementation adds a small
number to that denominator, which converts a detectable breakdown into a
plausible wrong answer. :func:`pcg` raises instead, and says which quantity went
non-positive.

**The preconditioner is not SPD.** PCG needs ``B`` symmetric positive definite,
because ``r' z`` is being used as a squared norm. A generic ILU is *not* SPD, so
it can produce ``r' z < 0``; the paper's own text notes this and restricts
Strategy B to SPD baselines for the same reason. Detected and reported.

**The recurrence residual drifts.** ``r <- r - alpha K p`` is cheaper than
recomputing ``b - K x`` but accumulates round-off, and on an ill-conditioned
system the recurrence residual can be orders of magnitude smaller than the true
one -- so the solver reports convergence it has not achieved.
:class:`KrylovResult` carries both, and ``true_residual`` is the one to believe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from scion.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "KrylovResult",
    "conjugate_gradient",
    "pcg",
    "KrylovBreakdown",
    "NotPositiveDefinite",
    "as_operator",
    "as_preconditioner",
    "as_transpose_operator",
]


class KrylovBreakdown(RuntimeError):
    """A Krylov recurrence hit a division that has no meaning.

    Base for every breakdown in this package. Each method breaks down in its own
    way -- CG on a non-positive curvature, BiCG on a vanishing bilinear form --
    and all of them share the property that continuing produces numbers rather
    than an error. They are raised, never warned about.
    """


class NotPositiveDefinite(KrylovBreakdown):
    """Raised when CG meets a direction of non-positive curvature.

    Means either the operator or the preconditioner is not SPD. It is a hard
    error rather than a warning because every quantity computed afterwards is
    meaningless -- the step length is a ratio whose denominator has changed
    sign.
    """


@dataclass
class KrylovResult:
    """Solution and the record needed to judge it."""

    x: np.ndarray
    converged: bool
    iterations: int
    #: ``||r_k|| / ||b||`` from the recurrence, one entry per iteration plus the
    #: initial value
    residual_history: np.ndarray
    #: ``||b - K x|| / ||b||`` recomputed once at the end. The honest number.
    true_residual: float
    #: wall-clock seconds spent in the iteration, excluding preconditioner setup
    time: float = 0.0
    preconditioner: str = "none"
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.residual_history = np.asarray(self.residual_history, dtype=float)

    @property
    def relative_residual(self) -> float:
        """The recurrence residual. Compare against :attr:`true_residual`."""
        return float(self.residual_history[-1])

    @property
    def residual_drift(self) -> float:
        """Ratio of true to recurrence residual.

        One means the recurrence stayed honest. A large value means round-off
        has accumulated and the reported convergence is optimistic -- common on
        badly conditioned systems and the reason both numbers are kept.
        """
        recurrence = self.relative_residual
        if recurrence <= 0.0:
            return float("inf") if self.true_residual > 0 else 1.0
        return float(self.true_residual / recurrence)

    def summary(self) -> str:
        state = "converged" if self.converged else "DID NOT CONVERGE"
        return (f"{self.preconditioner}: {state} in {self.iterations} "
                f"iterations, {self.time:.3f} s\n"
                f"  recurrence residual {self.relative_residual:.3e}, "
                f"true {self.true_residual:.3e} "
                f"(drift x{self.residual_drift:.3g})")

    def __repr__(self) -> str:
        return (f"KrylovResult({self.preconditioner}, "
                f"{'ok' if self.converged else 'failed'}, "
                f"{self.iterations} its, rel={self.true_residual:.2e})")


def as_operator(A, n: int) -> Callable[[np.ndarray], np.ndarray]:
    """Anything that can be applied to a vector: matrix, LinearOperator, or
    callable. A Krylov method never needs ``A`` itself, only its action."""
    if callable(A) and not hasattr(A, "shape"):
        return A
    if sparse.issparse(A) or isinstance(A, (np.ndarray, spla.LinearOperator)):
        if A.shape != (n, n):
            raise ValueError(f"operator is {A.shape}, expected ({n}, {n})")
        return lambda v: A @ v
    raise TypeError(f"cannot apply an object of type {type(A).__name__}")


def as_transpose_operator(A, n: int, AT=None):
    """The action of ``A'``, which BiCG needs and nothing else here does.

    A sparse or dense matrix supplies it directly; a ``LinearOperator`` may or
    may not, depending on whether it was given an ``rmatvec``; and a bare
    callable cannot -- there is no way to transpose a function. That last case is
    the reason BiCGSTAB exists and is usually preferred, so it is refused with
    that pointer rather than silently using ``A`` in place of ``A'``.
    """
    if AT is not None:
        return as_operator(AT, n)
    if sparse.issparse(A) or isinstance(A, np.ndarray):
        transpose = A.T
        return lambda v: transpose @ v
    if isinstance(A, spla.LinearOperator):
        try:
            A.rmatvec(np.zeros(n))
        except Exception as exc:               # no rmatvec supplied
            raise TypeError(
                f"this LinearOperator has no usable rmatvec, so A' is "
                f"unavailable: {exc}. Pass `AT=` explicitly, or use bicgstab, "
                f"which is transpose-free.") from None
        return A.rmatvec
    raise TypeError(
        "a bare callable cannot be transposed, and BiCG needs the action of "
        "A'. Pass `AT=` explicitly, or use bicgstab, which needs only A.")


def as_preconditioner(M, n: int):
    """``None``, a callable, a :class:`~.preconditioners.Preconditioner`, a
    ``LinearOperator``, or a matrix -- all reduce to one apply."""
    if M is None:
        return None, "none"
    name = getattr(M, "name", type(M).__name__.lower())
    if hasattr(M, "apply"):
        return M.apply, name
    if callable(M) and not hasattr(M, "shape"):
        return M, name
    if sparse.issparse(M) or isinstance(M, (np.ndarray, spla.LinearOperator)):
        return (lambda v: M @ v), name
    raise TypeError(f"cannot use an object of type {type(M).__name__} as a "
                    f"preconditioner")


def pcg(A, b, M=None, x0=None, tol: float = 1e-8, max_iter: int | None = None,
        callback=None, check_spd: bool = True,
        atol: float = 0.0) -> KrylovResult:
    r"""Preconditioned conjugate gradients for ``A x = b`` with ``A`` SPD.

    Algorithm 1 of Kalogeris, *Two-Level Preconditioning with
    Diffusion-Filtered Coarse Spaces*: the preconditioner enters only through
    ``z = B^-1 r``, so ``M`` is never formed.

    Parameters
    ----------
    A : sparse matrix, dense matrix, ``LinearOperator``, or callable ``v -> Av``.
    b : right-hand side.
    M : preconditioner -- ``None``, a :class:`~.preconditioners.Preconditioner`,
        a callable applying ``B^-1``, or a matrix. Must be SPD; see the module
        docstring.
    tol : relative tolerance on ``||r|| / ||b||``.
    atol : absolute floor, so a zero right-hand side terminates immediately
        rather than dividing by zero.
    max_iter : defaults to ``10 * n``, which is generous by design -- the point
        of these experiments is to count iterations, and a cap that bites
        silently truncates the comparison.
    callback : called as ``callback(k, x, relative_residual)`` each iteration.
    check_spd : detect non-positive curvature. Leave on unless profiling; the
        check is two comparisons per iteration.

    Notes
    -----
    Stopping is on the *recurrence* residual, because recomputing ``b - A x``
    every iteration doubles the matrix-vector cost. The true residual is
    computed once at the end and returned alongside, and
    :attr:`KrylovResult.residual_drift` is how you find out whether the
    recurrence was trustworthy on your problem.
    """
    import time as _time

    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    apply_A = as_operator(A, n)
    apply_M, name = as_preconditioner(M, n)

    if max_iter is None:
        max_iter = 10 * n

    x = (np.zeros(n) if x0 is None
         else np.asarray(x0, dtype=float).ravel().copy())
    if x.size != n:
        raise ValueError(f"x0 has {x.size} entries, expected {n}")

    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return KrylovResult(x=np.zeros(n), converged=True, iterations=0,
                            residual_history=[0.0], true_residual=0.0,
                            preconditioner=name,
                            info={"note": "zero right-hand side"})

    started = _time.perf_counter()
    r = b - apply_A(x)
    z = r.copy() if apply_M is None else np.asarray(apply_M(r), dtype=float).ravel()
    p = z.copy()
    rz = float(np.dot(r, z))

    if check_spd and rz < 0.0:
        raise NotPositiveDefinite(
            f"r'z = {rz:.3e} < 0 at the first iteration, so the preconditioner "
            f"'{name}' is not symmetric positive definite. PCG treats r'z as a "
            f"squared norm. A generic ILU is not SPD -- use incomplete "
            f"Cholesky for an SPD system, or a flexible Krylov method.")

    history = [float(np.linalg.norm(r)) / b_norm]
    threshold = max(tol * b_norm, atol)
    converged = history[-1] * b_norm <= threshold
    k = 0

    while not converged and k < max_iter:
        Ap = apply_A(p)
        pAp = float(np.dot(p, Ap))

        if check_spd and pAp <= 0.0:
            raise NotPositiveDefinite(
                f"p'Ap = {pAp:.3e} <= 0 at iteration {k}, so the operator is "
                f"not positive definite along a search direction. CG does not "
                f"apply; use MINRES for a symmetric indefinite system.")
        if pAp == 0.0:
            break

        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap

        residual = float(np.linalg.norm(r))
        history.append(residual / b_norm)
        k += 1
        if callback is not None:
            callback(k, x, history[-1])

        if residual <= threshold:
            converged = True
            break

        z = (r.copy() if apply_M is None
             else np.asarray(apply_M(r), dtype=float).ravel())
        rz_new = float(np.dot(r, z))
        if check_spd and rz_new < 0.0:
            raise NotPositiveDefinite(
                f"r'z = {rz_new:.3e} < 0 at iteration {k}: the preconditioner "
                f"'{name}' is not SPD")
        if rz == 0.0:
            break

        p = z + (rz_new / rz) * p
        rz = rz_new

    elapsed = _time.perf_counter() - started
    true_residual = float(np.linalg.norm(b - apply_A(x)) / b_norm)

    if not converged:
        logger.warning("PCG (%s) stopped after %d iterations at relative "
                       "residual %.3e", name, k, history[-1])

    return KrylovResult(
        x=x, converged=converged, iterations=k, residual_history=history,
        true_residual=true_residual, time=elapsed, preconditioner=name,
        info={"tol": tol, "max_iter": max_iter, "n": n})


def conjugate_gradient(A, b, x0=None, tol: float = 1e-8,
                       max_iter: int | None = None, callback=None,
                       check_spd: bool = True) -> KrylovResult:
    """Unpreconditioned CG. Exactly :func:`pcg` with ``M=None``.

    Kept as its own name because it is the baseline every preconditioner is
    measured against, and because "how many iterations without a
    preconditioner" is the first number worth knowing about a new system.
    """
    return pcg(A, b, M=None, x0=x0, tol=tol, max_iter=max_iter,
               callback=callback, check_spd=check_spd)
