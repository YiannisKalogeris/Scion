r"""
Krylov methods for systems that are not symmetric positive definite.

CG relies on ``A`` being SPD twice over: to make the three-term recurrence
possible, and to make ``r'z`` a norm. Drop either and it is gone. What replaces
it splits into two families, and the split is a genuine trade rather than a
matter of taste.

**Long recurrence, guaranteed monotone: GMRES.** Build an orthonormal Krylov
basis by Arnoldi and minimise the residual over it exactly. The residual norm
can therefore never increase, and the method cannot break down -- the only thing
that stops it is running out of memory, because iteration ``k`` stores ``k``
basis vectors and orthogonalises against all of them. That is why it is
*restarted*: GMRES(m) throws the basis away every ``m`` steps. Restarting costs
the global optimality, and a restarted run can stagnate where full GMRES would
not.

**Short recurrence, constant memory, no guarantee: BiCG and BiCGSTAB.** Three
vectors regardless of iteration count, at the price of a residual that is not
monotone and a recurrence that can break down outright. BiCG additionally needs
the action of ``A'``, which a matrix-free operator often cannot supply.
BiCGSTAB removes both the transpose and most of the erratic convergence, and is
what actually gets used.

Which to reach for: GMRES(m) when memory allows and robustness matters,
BiCGSTAB when it does not. Neither has CG's optimality, and on an SPD system
both are strictly worse than CG -- use :func:`~.krylov.pcg` there.

Left or right preconditioning is not a detail
---------------------------------------------
For CG the question does not arise; here it changes what is being minimised.

    left     solve  B^-1 A x = B^-1 b     minimises  ||B^-1 (b - Ax)||
    right    solve  A B^-1 y = b, x=B^-1 y   minimises  ||b - Ax||

Left preconditioning monitors the *preconditioned* residual, which is not the
quantity anyone wants and can be orders of magnitude away from the true one when
``B`` is ill-conditioned -- so a left-preconditioned run can report convergence
it has not reached. Right preconditioning minimises the true residual, so its
convergence history means what it says. Right is therefore the default here, and
both are available because the literature uses both.

Flexible GMRES
--------------
:func:`fgmres` allows ``B^-1`` to *change between iterations*. That is impossible
in standard GMRES, where the relation ``A V_k = V_{k+1} H`` assumes one fixed
operator; FGMRES stores the preconditioned vectors ``Z = B_j^-1 v_j`` as well as
the basis, doubling the memory, and builds the solution from ``Z`` instead.

It is the documented fix for the case in the accompanying paper's Section 3.2:
the two-level operator wrapped around a non-SPD baseline such as ILU. There
``T^-1`` is not symmetric and PCG's assumptions fail, but FGMRES does not assume
symmetry *or* a fixed preconditioner, so it converges where PCG breaks down.
FGMRES is right-preconditioned by construction -- there is no left-preconditioned
flexible variant, because the flexibility lives in exactly the place left
preconditioning does not have.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg

from simulacrum.core.logging import get_logger
from simulacrum.solvers.krylov import (
    KrylovBreakdown,
    KrylovResult,
    as_operator,
    as_preconditioner,
    as_transpose_operator,
)

logger = get_logger(__name__)

__all__ = ["gmres", "fgmres", "bicg", "bicgstab"]

_EPS = np.finfo(float).eps


def _givens(a: float, b: float):
    """Rotation zeroing ``b``, guarded against overflow in ``sqrt(a^2+b^2)``."""
    if b == 0.0:
        return 1.0, 0.0
    if abs(b) > abs(a):
        ratio = a / b
        s = 1.0 / np.sqrt(1.0 + ratio * ratio)
        return ratio * s, s
    ratio = b / a
    c = 1.0 / np.sqrt(1.0 + ratio * ratio)
    return c, ratio * c


def _start(A, b, x0, M, tol, max_iter, side="right"):
    """Shared setup: adapters, sizes, initial residual."""
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    apply_A = as_operator(A, n)
    apply_M, name = as_preconditioner(M, n)
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    x = (np.zeros(n) if x0 is None
         else np.asarray(x0, dtype=float).ravel().copy())
    if x.size != n:
        raise ValueError(f"x0 has {x.size} entries, expected {n}")

    b_norm = float(np.linalg.norm(b))
    return b, n, apply_A, apply_M, name, x, b_norm, (10 * n if max_iter is None
                                                    else int(max_iter))


def _trivial(x, name, note):
    return KrylovResult(x=x, converged=True, iterations=0,
                        residual_history=[0.0], true_residual=0.0,
                        preconditioner=name, info={"note": note})


def _finish(x, apply_A, b, b_norm, history, iterations, converged, started,
            name, info) -> KrylovResult:
    true_residual = float(np.linalg.norm(b - apply_A(x)) / b_norm)
    if not converged:
        logger.warning("%s stopped after %d iterations at relative residual "
                       "%.3e", info.get("method", "solver"), iterations,
                       history[-1])
    return KrylovResult(x=x, converged=converged, iterations=iterations,
                        residual_history=history,
                        true_residual=true_residual,
                        time=time.perf_counter() - started,
                        preconditioner=name, info=info)


# ---------------------------------------------------------------------------
# GMRES
# ---------------------------------------------------------------------------

def _gmres_core(A, b, M, x0, tol, max_iter, restart, side, flexible,
                reorthogonalize, method_name):
    (b, n, apply_A, apply_M, name, x, b_norm,
     max_iter) = _start(A, b, x0, M, tol, max_iter, side)

    if b_norm == 0.0:
        return _trivial(np.zeros(n), name, "zero right-hand side")

    identity = apply_M is None
    m = min(int(restart) if restart else max_iter, max_iter, n)
    if m < 1:
        raise ValueError("restart must be at least one")

    started = time.perf_counter()
    history = []
    total, converged, cycles, lucky = 0, False, 0, False

    while total < max_iter and not converged:
        cycles += 1
        residual = b - apply_A(x)
        if side == "left" and not identity:
            residual = apply_M(residual)
        beta = float(np.linalg.norm(residual))

        if not history:
            # for right preconditioning this is the true relative residual; for
            # left it is the preconditioned one, which is the point of the
            # warning in the module docstring
            history.append(beta / b_norm if side == "right"
                           else float(np.linalg.norm(b - apply_A(x))) / b_norm)
        if beta == 0.0:
            converged = True
            break

        V = np.zeros((n, m + 1), order="F")
        Z = np.zeros((n, m), order="F") if flexible else None
        H = np.zeros((m + 1, m))
        cs, sn = np.zeros(m), np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta
        V[:, 0] = residual / beta

        k = 0
        for j in range(m):
            if identity:
                # flexible mode builds its update from Z, so Z must be filled
                # even when there is no preconditioner to apply -- otherwise the
                # update is Z @ y = 0 and the iteration makes no progress at all
                if flexible:
                    Z[:, j] = V[:, j]
                w = apply_A(V[:, j])
            elif side == "right":
                z = apply_M(V[:, j])
                if flexible:
                    Z[:, j] = z
                w = apply_A(z)
            else:
                w = apply_M(apply_A(V[:, j]))

            for i in range(j + 1):              # modified Gram-Schmidt
                H[i, j] = float(np.dot(V[:, i], w))
                w -= H[i, j] * V[:, i]
            if reorthogonalize:
                correction = V[:, :j + 1].T @ w
                H[:j + 1, j] += correction
                w -= V[:, :j + 1] @ correction

            subdiagonal = float(np.linalg.norm(w))
            H[j + 1, j] = subdiagonal
            # A vanishing subdiagonal means the Krylov space is invariant and
            # the exact solution lies in it. Detect it *here*, on ||w||, before
            # the rotations overwrite the entry -- the previous version tested
            # `|c| < eps and |s| < eps`, which is unreachable because Givens
            # rotations satisfy c^2 + s^2 = 1, so the flag was always False.
            # The convergence test below happens to catch the same case (a zero
            # subdiagonal forces sn[j] = 0 and hence g[j+1] = 0 exactly), which
            # is why this was latent rather than a failure.
            invariant = subdiagonal <= _EPS * max(1.0, abs(H[j, j]))

            for i in range(j):                  # previous rotations
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            cs[j], sn[j] = _givens(H[j, j], H[j + 1, j])
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]

            k = j + 1
            total += 1
            estimate = abs(g[j + 1])
            history.append(estimate / b_norm)

            if estimate <= tol * b_norm:
                converged = True
                break
            if invariant:            # a success, not a failure
                lucky = True
                break
            if total >= max_iter:
                break
            V[:, j + 1] = w / subdiagonal

        if k:
            y = scipy.linalg.solve_triangular(H[:k, :k], g[:k], lower=False)
            update = (Z[:, :k] @ y if flexible else V[:, :k] @ y)
            if side == "right" and not flexible and not identity:
                update = apply_M(update)
            x = x + update
        if lucky:
            converged = float(np.linalg.norm(b - apply_A(x))) <= tol * b_norm
            break

    return _finish(x, apply_A, b, b_norm, history, total, converged, started,
                   name, {"method": method_name, "restart": m,
                          "cycles": cycles, "side": side,
                          "flexible": flexible, "lucky_breakdown": lucky})


def gmres(A, b, M=None, x0=None, tol: float = 1e-8, max_iter: int | None = None,
          restart: int | None = 30, side: str = "right",
          reorthogonalize: bool = True) -> KrylovResult:
    r"""Restarted GMRES(m) for a general square system.

    Minimises the residual exactly over the Krylov space, so the residual norm
    is **monotone** within a cycle and the method cannot break down. Its cost is
    memory: ``restart`` basis vectors, and ``O(k)`` inner products at step ``k``.

    Parameters
    ----------
    restart : cycle length ``m``. ``None`` means full GMRES -- optimal, and
        ``O(n * max_iter)`` memory, so only for small problems. 30 is the usual
        default.
    side : ``"right"`` (default) or ``"left"``. Right minimises the true
        residual; left minimises ``||B^-1 r||`` and can therefore claim
        convergence it has not reached. See the module docstring.
    reorthogonalize : second Gram-Schmidt pass. One pass leaves a component of
        size ``O(eps ||w||)`` and the Arnoldi basis loses orthogonality once an
        eigenvalue converges, which corrupts the least-squares solve.

    Notes
    -----
    Restarting is not free. GMRES(m) is optimal only within a cycle; across
    restarts it can stagnate on a problem full GMRES solves, because the
    discarded basis contained the directions it needed. If a run plateaus, the
    first thing to try is a larger ``restart``, not a different method.
    """
    return _gmres_core(A, b, M, x0, tol, max_iter, restart, side, False,
                       reorthogonalize, "gmres")


def fgmres(A, b, M=None, x0=None, tol: float = 1e-8,
           max_iter: int | None = None, restart: int | None = 30,
           reorthogonalize: bool = True) -> KrylovResult:
    r"""Flexible GMRES: the preconditioner may change between iterations.

    Stores the preconditioned basis ``Z = B_j^-1 v_j`` alongside ``V`` -- twice
    the memory of :func:`gmres` -- and builds the update from ``Z``. That is
    what removes the assumption of a single fixed ``B^-1``, so ``M`` may be a
    nonlinear operator, an inner Krylov solve, or a preconditioner that is
    re-tuned as the iteration proceeds.

    Right-preconditioned by construction: the flexibility lives in the place
    left preconditioning does not have, so there is no left-preconditioned
    flexible variant to offer.

    The case it is here for: the two-level operator of the accompanying paper
    wrapped around a non-SPD baseline such as ILU. ``T^-1`` is then not
    symmetric, PCG's use of ``r'z`` as a squared norm is invalid, and FGMRES
    converges where PCG breaks down.
    """
    return _gmres_core(A, b, M, x0, tol, max_iter, restart, "right", True,
                       reorthogonalize, "fgmres")


# ---------------------------------------------------------------------------
# BiCG and BiCGSTAB
# ---------------------------------------------------------------------------

def bicg(A, b, M=None, x0=None, tol: float = 1e-8, max_iter: int | None = None,
         AT=None, MT=None) -> KrylovResult:
    r"""Biconjugate gradients. Short recurrence, needs ``A'``, breaks down.

    Runs two coupled recurrences -- one on ``A``, a shadow one on ``A'`` -- whose
    biorthogonality plays the role CG's ``A``-orthogonality does. The payoff is
    constant memory. The costs are real and both are stated rather than hidden:

    * **It needs the action of ``A'``.** A matrix supplies it; a matrix-free
      operator usually cannot. :func:`bicgstab` needs only ``A``.
    * **It breaks down.** ``rho = rt'r`` can vanish with the residual still
      large (a *serious* breakdown, no cure within the method), and so can
      ``pt'Ap``. Both raise :class:`~.krylov.KrylovBreakdown`.
    * **Its residual is not monotone.** Peaks of several orders of magnitude are
      normal and are not a bug; only the final residual means anything.

    ``MT`` is the transpose of the preconditioner, needed for the shadow
    recurrence. For a symmetric ``B`` -- Jacobi, IC(0), a symmetric V-cycle --
    it equals ``M`` and is filled in automatically. Supply it for anything else.
    """
    (b, n, apply_A, apply_M, name, x, b_norm,
     max_iter) = _start(A, b, x0, M, tol, max_iter, "left")

    if b_norm == 0.0:
        return _trivial(np.zeros(n), name, "zero right-hand side")

    apply_AT = as_transpose_operator(A, n, AT)
    if apply_M is None:
        apply_M = apply_MT = None
    else:
        # a symmetric preconditioner is its own transpose, which covers Jacobi,
        # IC(0) and a symmetric V-cycle; anything else must say so
        apply_MT = apply_M if MT is None else as_preconditioner(MT, n)[0]

    started = time.perf_counter()
    r = b - apply_A(x)
    r_shadow = r.copy()
    history = [float(np.linalg.norm(r)) / b_norm]
    converged = history[-1] <= tol
    p = p_shadow = None
    rho_old = 1.0
    k = 0

    while not converged and k < max_iter:
        z = r if apply_M is None else apply_M(r)
        z_shadow = r_shadow if apply_MT is None else apply_MT(r_shadow)
        rho = float(np.dot(z, r_shadow))

        if rho == 0.0:
            raise KrylovBreakdown(
                f"BiCG serious breakdown at iteration {k}: rho = rt'z vanished "
                f"while the residual is still {history[-1]:.3e}. There is no "
                f"remedy inside the method -- restart from a different shadow "
                f"vector, or use bicgstab.")

        if p is None:
            p, p_shadow = z.copy(), z_shadow.copy()
        else:
            beta = rho / rho_old
            p = z + beta * p
            p_shadow = z_shadow + beta * p_shadow

        q = apply_A(p)
        q_shadow = apply_AT(p_shadow)
        denominator = float(np.dot(p_shadow, q))
        if denominator == 0.0:
            raise KrylovBreakdown(
                f"BiCG breakdown at iteration {k}: pt'Ap vanished, so the step "
                f"length is undefined")

        alpha = rho / denominator
        x = x + alpha * p
        r = r - alpha * q
        r_shadow = r_shadow - alpha * q_shadow
        rho_old = rho
        k += 1

        history.append(float(np.linalg.norm(r)) / b_norm)
        converged = history[-1] <= tol

    return _finish(x, apply_A, b, b_norm, history, k, converged, started, name,
                   {"method": "bicg"})


def bicgstab(A, b, M=None, x0=None, tol: float = 1e-8,
             max_iter: int | None = None) -> KrylovResult:
    r"""BiCGSTAB: BiCG stabilised, transpose-free, and what to actually use.

    Van der Vorst's variant. It replaces BiCG's shadow recurrence with a second
    local minimisation, which buys two things at once: the transpose disappears,
    so a matrix-free operator is enough; and the residual peaks that make BiCG
    unusable are largely smoothed away.

    Two matrix-vector products and two preconditioner applications per
    iteration, against BiCG's two and two -- so an iteration costs about the
    same and is worth much more. Memory stays constant.

    It can still break down, on ``rt'v`` or on ``t't``, and both are raised. The
    residual is still not guaranteed monotone. If it stagnates, GMRES(m) with a
    generous restart is the robust fallback.
    """
    (b, n, apply_A, apply_M, name, x, b_norm,
     max_iter) = _start(A, b, x0, M, tol, max_iter, "left")

    if b_norm == 0.0:
        return _trivial(np.zeros(n), name, "zero right-hand side")

    started = time.perf_counter()
    r = b - apply_A(x)
    r_shadow = r.copy()
    history = [float(np.linalg.norm(r)) / b_norm]
    converged = history[-1] <= tol

    rho_old = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)
    k = 0

    while not converged and k < max_iter:
        rho = float(np.dot(r_shadow, r))
        if rho == 0.0:
            raise KrylovBreakdown(
                f"BiCGSTAB breakdown at iteration {k}: rt'r vanished while the "
                f"residual is {history[-1]:.3e}. Restart, or use GMRES.")

        if k == 0:
            p = r.copy()
        else:
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

        p_hat = p if apply_M is None else apply_M(p)
        v = apply_A(p_hat)
        denominator = float(np.dot(r_shadow, v))
        if denominator == 0.0:
            raise KrylovBreakdown(
                f"BiCGSTAB breakdown at iteration {k}: rt'v vanished")
        alpha = rho / denominator

        s = r - alpha * v
        k += 1
        # the half-step is a genuine convergence point, not an optimisation:
        # skipping it does an extra A-apply and can divide by a zero t't
        if float(np.linalg.norm(s)) <= tol * b_norm:
            x = x + alpha * p_hat
            history.append(float(np.linalg.norm(b - apply_A(x))) / b_norm)
            converged = history[-1] <= tol
            break

        s_hat = s if apply_M is None else apply_M(s)
        t = apply_A(s_hat)
        t_dot = float(np.dot(t, t))
        if t_dot == 0.0:
            raise KrylovBreakdown(
                f"BiCGSTAB breakdown at iteration {k}: t't vanished")
        omega = float(np.dot(t, s)) / t_dot
        if omega == 0.0:
            raise KrylovBreakdown(
                f"BiCGSTAB breakdown at iteration {k}: omega vanished, so the "
                f"stabilising step contributes nothing and the recurrence "
                f"cannot continue")

        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        rho_old = rho

        history.append(float(np.linalg.norm(r)) / b_norm)
        converged = history[-1] <= tol

    return _finish(x, apply_A, b, b_norm, history, k, converged, started, name,
                   {"method": "bicgstab"})
