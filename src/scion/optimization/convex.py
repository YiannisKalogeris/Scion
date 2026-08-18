r"""
Convex problems on a linear subspace, and bisection on a feasibility oracle.

Two pieces, both physics-free, both written for the direct methods of
:mod:`scion.shakedown` and neither specific to them.

Minimising on a subspace
------------------------
A great many mechanics problems are "minimise a convex function over the fields
that satisfy some linear constraint" -- self-equilibrated stress fields,
divergence-free velocities, fields orthogonal to the rigid modes. If the
constraint set is a linear subspace ``S`` and you can project onto it, the
constrained problem needs no constraint machinery at all: minimise ``f(P x)``
over all of ``R^n`` and read the answer off as ``P x``.

**The projector must be orthogonal in the Euclidean inner product**, because
that is the one the optimiser's chain rule uses:

    grad(f . P)(x) = P^T grad f(P x) = P grad f(P x)   only if  P^T = P.

An oblique projection into ``S`` still lands in ``S`` and still looks plausible,
but the gradient handed to the optimiser is then wrong, and the symptom is slow
convergence rather than a wrong answer -- the same failure mode as a wrong
consistent tangent. :func:`check_projector` tests both properties, and
:func:`minimize_on_subspace` will not accept a projector that fails them
without being told to.

A projector orthogonal in some *other* inner product -- an L2 projection with
quadrature weights, say, which is the physically natural one for a stress field
-- can be used by rescaling the variable so that inner product becomes the
Euclidean one. That is a change of variables in the caller, not something this
module can do for you, because only the caller knows the weights.

Bisection on a feasibility oracle
---------------------------------
The other pattern: find the largest parameter for which a convex feasibility
problem is still solvable. Each test is expensive, the answer is monotone, and
what matters is that the value returned is one that was *actually verified* --
not the midpoint of the last bracket. :func:`largest_feasible` returns the last
feasible point and the bracket around it, so a caller reporting a bound reports
something it has a certificate for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "SubspaceResult",
    "BisectionResult",
    "check_projector",
    "minimize_on_subspace",
    "largest_feasible",
]


# ---------------------------------------------------------------------------

@dataclass
class SubspaceResult:
    """Outcome of a subspace-constrained minimisation."""

    x: np.ndarray                # the minimiser, already projected into S
    fun: float
    n_iter: int
    n_fev: int
    converged: bool
    message: str = ""

    def __repr__(self) -> str:
        return (f"SubspaceResult(fun={self.fun:.6g}, iter={self.n_iter}, "
                f"fev={self.n_fev}, converged={self.converged})")


def check_projector(project: Callable[[np.ndarray], np.ndarray],
                    n: int, n_probes: int = 4, seed: int = 0,
                    tol: float = 1.0e-9) -> dict:
    """Test that ``project`` is an orthogonal projector, ``(n,)`` -> ``(n,)``.

    Checks idempotence ``P P = P`` and self-adjointness
    ``<P a, b> = <a, P b>``. Both are cheap and both are needed: idempotence
    alone admits an oblique projection, which is the one that silently breaks
    the gradient.

    Returns the two residuals and a boolean; raises nothing, so a caller can
    report rather than fail.
    """
    rng = np.random.default_rng(seed)
    idem = 0.0
    sym = 0.0
    for _ in range(n_probes):
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        Pa, Pb = project(a), project(b)
        scale = max(np.abs(Pa).max(), 1.0)
        idem = max(idem, float(np.abs(project(Pa) - Pa).max()) / scale)
        sym = max(sym, abs(float(Pa @ b - a @ Pb))
                  / max(abs(float(a @ b)), 1.0))
    return {"idempotent": idem, "self_adjoint": sym,
            "ok": bool(idem < tol and sym < tol)}


def minimize_on_subspace(fun_and_grad, x0: np.ndarray,
                         project: Callable[[np.ndarray], np.ndarray] | None,
                         maxiter: int = 500, gtol: float = 1.0e-10,
                         ftol: float = 1.0e-16,
                         validate: bool = True) -> SubspaceResult:
    """Minimise a convex ``f`` over the subspace ``S`` that ``project`` lands in.

    Parameters
    ----------
    fun_and_grad : ``x -> (value, gradient)`` on the *unconstrained* variable.
        Returning both together, because every use of this has an objective
        whose gradient is nearly free once the value has been computed and
        splitting them doubles the cost.
    x0 : starting point; it is projected before use, so it need not be in ``S``.
    project : Euclidean-orthogonal projection onto ``S``. See the module
        docstring for why orthogonal, and :func:`check_projector`. ``None``
        means the whole space -- worth having, because a search space given by a
        basis of admissible fields carries its constraint in the basis and needs
        no projection at all.
    validate : run :func:`check_projector` once and raise if it fails. Cheap
        next to the solve, and it turns an obscure convergence problem into a
        clear error.

    The minimiser of ``f`` over ``S`` is not unique as a point of ``R^n`` --
    anything in the null space of ``P`` can be added to the iterate without
    changing ``f(Px)``. That is harmless: L-BFGS builds its directions from
    gradients, every gradient here has already been projected, so the iterates
    never leave ``S`` once they start in it.
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    if project is None:
        def project(x):
            return np.asarray(x, dtype=float).ravel()
        validate = False
    if validate:
        report = check_projector(project, x0.size)
        if not report["ok"]:
            raise ValueError(
                f"project is not a Euclidean-orthogonal projector: "
                f"idempotence residual {report['idempotent']:.3e}, "
                f"self-adjointness residual {report['self_adjoint']:.3e}. "
                f"An oblique projection still lands in the subspace but makes "
                f"the gradient wrong; if the natural projector is orthogonal "
                f"in a weighted inner product, rescale the variable first.")

    counter = {"n": 0}

    def wrapped(x):
        counter["n"] += 1
        value, grad = fun_and_grad(project(x))
        return float(value), project(np.asarray(grad, dtype=float).ravel())

    out = minimize(wrapped, project(x0), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": gtol, "ftol": ftol,
                            "maxcor": 20})
    return SubspaceResult(x=project(out.x), fun=float(out.fun),
                          n_iter=int(out.nit), n_fev=counter["n"],
                          converged=bool(out.success),
                          message=str(out.message))


# ---------------------------------------------------------------------------

@dataclass
class BisectionResult:
    """Largest verified-feasible parameter, and the bracket around it."""

    value: float                 # the largest parameter verified feasible
    upper: float                 # the smallest verified infeasible, or inf
    iterations: int
    evaluations: int
    bracketed: bool              # False if the search never found infeasible

    @property
    def bracket(self) -> tuple[float, float]:
        return (self.value, self.upper)

    @property
    def relative_width(self) -> float:
        if not np.isfinite(self.upper) or self.value <= 0.0:
            return float("inf")
        return (self.upper - self.value) / self.value

    def __repr__(self) -> str:
        return (f"BisectionResult(value={self.value:.6g}, "
                f"bracket=[{self.value:.6g}, {self.upper:.6g}], "
                f"width={self.relative_width:.2e}, "
                f"iterations={self.iterations})")


def largest_feasible(oracle: Callable[[float], bool], start: float,
                     growth: float = 2.0, rtol: float = 1.0e-4,
                     max_expand: int = 40, max_bisect: int = 60,
                     ) -> BisectionResult:
    """Largest ``t`` for which ``oracle(t)`` is true, assuming monotonicity.

    ``oracle`` must be monotone: true below some threshold and false above it.
    The value returned is the largest ``t`` the oracle actually accepted, never
    an interpolated one -- so a caller quoting it as a bound is quoting
    something that was checked.

    ``start`` must be feasible; it is the natural place to pass a bound you
    already know, which also warm-starts whatever the oracle does internally.
    The bracket is then found by multiplying by ``growth`` until the oracle
    refuses, and closed by bisection to ``rtol``.

    ``bracketed=False`` means the oracle never refused. That is a real outcome
    -- an unbounded problem, or an oracle that is not actually testing what it
    should -- and it is reported rather than hidden behind a large number.
    """
    if not oracle(start):
        raise ValueError(
            f"largest_feasible needs a feasible starting point; oracle({start:g})"
            f" is already false. Pass a smaller start.")

    lo, hi = float(start), float("inf")
    evaluations = 1

    trial = lo
    for _ in range(max_expand):
        trial = trial * growth
        evaluations += 1
        if oracle(trial):
            lo = trial
        else:
            hi = trial
            break

    if not np.isfinite(hi):
        return BisectionResult(lo, hi, 0, evaluations, False)

    iterations = 0
    while iterations < max_bisect and (hi - lo) > rtol * max(abs(lo), 1e-300):
        mid = 0.5 * (lo + hi)
        evaluations += 1
        iterations += 1
        if oracle(mid):
            lo = mid
        else:
            hi = mid

    return BisectionResult(lo, hi, iterations, evaluations, True)
