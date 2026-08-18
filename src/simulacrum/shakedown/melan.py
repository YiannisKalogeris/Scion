r"""
Melan's static theorem: the shakedown lower bound.

    If there exists a time-independent self-equilibrated residual stress field
    ``rho`` such that ``f(lambda sigma^E_k + rho) <= 0`` at every point and for
    every load ``k`` in the domain, the structure shakes down under ``lambda``.

So the shakedown factor is the largest ``lambda`` for which such a field
exists:

    maximise  lambda
    subject to  A rho = 0
                f(lambda sigma^E_k + rho) <= 0   for all k, all points

The feasible set is convex -- linear equality, convex inequalities -- so this is
a convex program, and for von Mises it is a second-order cone program. There is
no cone solver in this stack and writing one is not the shortest path to a
verified bound.

How it is solved here
---------------------
Bisection on ``lambda``, with a convex *feasibility* problem inside. At fixed
``lambda``, minimise over the residual fields the basis can represent

    Phi(beta) = sum over points and vertices of  w/V * <f_k / kappa>_+^2 ,

which is convex, continuously differentiable, non-negative, and zero exactly
when the load factor is admissible. Its gradient comes from
``criterion.df_dsigma`` -- the same derivative the return map uses, already
tested there -- contracted onto the basis modes. The bisection runs through
:func:`~simulacrum.optimization.convex.largest_feasible`, which knows no
mechanics.

The search space is a reduced basis
-----------------------------------
``rho`` is a combination of a few dozen self-equilibrated fields rather than a
free stress value at every quadrature point. This is not an efficiency measure.
Searching the full discrete residual space does not give a conservative answer,
it gives a meaningless one -- on the thick cylinder it returned between 1.4 and
3 x 10^5 times the analytic limit load, erratically, because weak equilibrium at
quadrature points admits deviatoric fields that no continuum pressure field
could balance. :mod:`~simulacrum.shakedown.basis` has the measurements and the
reasoning; the short version is that the free problem has about three times as
many unknowns as constraints and the optimiser finds the slack.

Because every basis mode is self-equilibrated by construction, the equality
constraint never enters the optimisation: the inner problem is unconstrained in
the coefficients.

Why the answer is a bound and not an estimate
---------------------------------------------
Two different claims, and only one of them holds unconditionally.

**Within the discrete model** the result is a genuine lower bound, and the
implementation is careful about the thing that makes it one: the number reported
is a ``lambda`` that was verified, not one that was interpolated. The bisection
returns the largest factor its oracle actually accepted, together with the field
that certified it, and :meth:`MelanResult.verify` recomputes the equilibrium
residual and the worst yield violation from that stored field. A caller can
check both without trusting anything else here.

**Against the continuum** it is not a strict bound. Yield is enforced at
quadrature points rather than everywhere, and equilibrium is weak rather than
pointwise. A strict bound needs a statically admissible stress discretisation --
element-wise stresses with enforced traction continuity, as in Sloan's
lower-bound limit analysis -- which is a different element family and not
implemented. Treat the result as the shakedown factor of the discrete model:
converging, checkable, and not something to quote as rigorous.

Restricting is safe; the discretisation is what is not
------------------------------------------------------
Everything about the *search* is conservative. A smaller basis, a looser
tolerance, an inner solve stopped early: all of them shrink the feasible set and
lower the answer. That is why enriching the basis can only help, and why a
partial run is still usable. The non-conservative part is entirely in how yield
and equilibrium are discretised, which no amount of optimisation effort
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..materials.notation import from_mandel, to_mandel
from ..optimization.convex import largest_feasible, minimize_on_subspace

__all__ = ["MelanResult", "elastic_factor", "melan_lower_bound"]


# ---------------------------------------------------------------------------

def _violations(vertex_stresses: np.ndarray, rho: np.ndarray,
                lam: float, criterion, kappa) -> np.ndarray:
    """``f_k`` at every vertex and point, ``(n_vertices, n_points)``."""
    out = np.empty(vertex_stresses.shape[:2])
    for k, shat in enumerate(vertex_stresses):
        out[k] = criterion.f(from_mandel(rho + lam * shat), kappa)
    return out


def elastic_factor(vertex_stresses: np.ndarray, criterion, kappa,
                   rtol: float = 1.0e-10) -> float:
    """Largest multiplier that keeps the whole structure elastic.

    ``rho = 0`` is always self-equilibrated, so this is a guaranteed lower bound
    on the shakedown factor and the natural place to start the search. A
    computed shakedown factor below it is a bug, not a conservative answer.

    Found by bisection rather than by dividing: the yield function is positively
    homogeneous in the stress only for pressure-independent criteria, so scaling
    the stress does not scale ``f`` for Drucker-Prager and the closed form would
    be quietly wrong for it.
    """
    zero = np.zeros(vertex_stresses.shape[1:])

    def admissible(lam: float) -> bool:
        return bool(_violations(vertex_stresses, zero, lam,
                                criterion, kappa).max() <= 0.0)

    if not admissible(0.0):
        raise ValueError(
            "the unloaded structure already violates the yield criterion; "
            "check the sign convention of kappa")

    lo, hi = 0.0, 1.0
    for _ in range(200):
        if not admissible(hi):
            break
        lo, hi = hi, hi * 2.0
    else:
        return float("inf")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if admissible(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= rtol * max(lo, 1e-300):
            break
    return float(lo)


# ---------------------------------------------------------------------------

@dataclass
class MelanResult:
    """A certified shakedown lower bound and the field that certifies it."""

    factor: float                     # verified admissible
    upper: float                      # smallest factor found inadmissible
    elastic: float                    # elastic limit, for reference
    residual_stress: np.ndarray = field(repr=False)   # (n_points, 6) Mandel
    iterations: int = 0
    evaluations: int = 0
    bracketed: bool = True
    n_modes: int = 0
    inner: list = field(default_factory=list, repr=False)

    @property
    def bracket(self) -> tuple[float, float]:
        return (self.factor, self.upper)

    @property
    def relative_width(self) -> float:
        if not np.isfinite(self.upper) or self.factor <= 0.0:
            return float("inf")
        return (self.upper - self.factor) / self.factor

    @property
    def reserve(self) -> float:
        """How much the load domain can grow beyond first yield before the
        structure stops shaking down. One means the shakedown limit is the
        elastic limit, which happens under a fixed load and never under a
        varying one."""
        return self.factor / self.elastic if self.elastic > 0 else float("nan")

    def verify(self, vertex_stresses: np.ndarray, equilibrium, criterion,
               kappa) -> dict:
        """Recheck the certificate from the stored field.

        Independent of everything the solver did: it recomputes the equilibrium
        residual of ``rho`` and the worst yield violation at the reported
        factor. A lower bound that fails this is not a bound.
        """
        rho = self.residual_stress
        worst = _violations(vertex_stresses, rho, self.factor,
                            criterion, kappa).max()
        scale = float(np.max(np.abs(kappa))) if np.ndim(kappa) else float(kappa)
        return {
            "equilibrium_residual": equilibrium.residual(rho),
            "max_yield_violation": float(worst),
            "relative_yield_violation": float(worst) / max(scale, 1e-300),
            "certified": bool(equilibrium.residual(rho) < 1e-8
                              and worst <= 1e-6 * scale),
        }

    def __repr__(self) -> str:
        return (f"MelanResult(factor={self.factor:.6g}, "
                f"bracket=[{self.factor:.6g}, {self.upper:.6g}], "
                f"elastic={self.elastic:.6g}, "
                f"reserve={self.reserve:.3f}, "
                f"modes={self.n_modes}, "
                f"width={self.relative_width:.2e})")


# ---------------------------------------------------------------------------

def melan_lower_bound(vertex_stresses: np.ndarray, basis, criterion,
                      kappa, rtol: float = 1.0e-4,
                      feasibility_tol: float = 1.0e-8,
                      inner_maxiter: int = 400, enrichments: int = 3,
                      verbose: bool = False) -> MelanResult:
    """Largest load factor the reduced basis can certify.

    Parameters
    ----------
    vertex_stresses : ``(n_vertices, n_points, 6)`` Mandel elastic stresses,
        from :meth:`~simulacrum.shakedown.elastic.ElasticEnvelope.vertex_stresses`.
    basis : a :class:`~simulacrum.shakedown.basis.ResidualBasis`. Its modes are
        the search space; a basis with no modes gives the elastic limit, which
        is the correct answer for an empty search space rather than a failure.
    criterion : any :class:`~simulacrum.materials.plasticity.yield_criteria.YieldCriterion`.
    kappa : the hardening force -- the yield stress for von Mises. Scalar, or one
        value per point.
    enrichments : outer rounds of basis enrichment. After each bisection the
        basis is grown with eigenstrain modes aimed at whatever is still
        violated just above the current factor, and the search repeats. This is
        the "sequence of reduced subspaces" the method is normally run as; each
        round can only raise the factor, since the previous basis is a subset of
        the new one.

    Returns a :class:`MelanResult`. Call its :meth:`~MelanResult.verify` before
    quoting the number, and read the module docstring on what it does and does
    not bound.
    """
    vertex_stresses = np.asarray(vertex_stresses, dtype=float)
    n_vertices, n_points, _ = vertex_stresses.shape
    kappa_arr = np.broadcast_to(np.asarray(kappa, dtype=float), (n_points,))
    scale = float(np.max(kappa_arr))

    weights = basis.tables.w.ravel()
    wn = weights / float(weights.sum())

    lam_e = elastic_factor(vertex_stresses, criterion, kappa_arr)
    if not np.isfinite(lam_e):
        raise ValueError(
            "the elastic stress field is identically zero at every vertex, so "
            "no load factor is limiting; check that the load cases were built")

    state = {"beta": np.zeros(basis.n_modes),
             "best": np.zeros((n_points, 6)), "log": []}

    def gradient_at(rho: np.ndarray, lam: float):
        total = 0.0
        grad = np.zeros((n_points, 6))
        for shat in vertex_stresses:
            tensor = from_mandel(rho + lam * shat)
            excess = np.maximum(criterion.f(tensor, kappa_arr), 0.0) / kappa_arr
            if not excess.any():
                continue
            total += float((wn * excess ** 2).sum())
            grad += ((2.0 * wn * excess / kappa_arr)[:, None]
                     * to_mandel(criterion.df_dsigma(tensor)))
        return total, grad

    def make_objective(lam: float):
        def fun_and_grad(beta):
            value, grad = gradient_at(basis.field(beta), lam)
            return value, basis.project_gradient(grad)
        return fun_and_grad

    def worst_violation(rho: np.ndarray, lam: float) -> float:
        return float(_violations(vertex_stresses, rho, lam,
                                 criterion, kappa_arr).max())

    def oracle(lam: float) -> bool:
        if not basis.n_modes:
            rho = np.zeros((n_points, 6))
            ok = worst_violation(rho, lam) <= feasibility_tol * scale
            if ok:
                state["best"] = rho
            return ok

        start = state["beta"]
        if start.size != basis.n_modes:            # the basis grew
            start = np.concatenate(
                [start, np.zeros(basis.n_modes - start.size)])
        out = minimize_on_subspace(make_objective(lam), start, project=None,
                                   maxiter=inner_maxiter, validate=False)
        rho = basis.field(out.x)
        worst = worst_violation(rho, lam)
        ok = worst <= feasibility_tol * scale
        state["log"].append((lam, out.fun, worst / scale, out.n_iter, ok))
        if ok:
            state["beta"] = out.x
            state["best"] = rho
        if verbose:
            print(f"    lambda = {lam:12.6g}   Phi = {out.fun:.3e}   "
                  f"worst f/kappa = {worst / scale:+.3e}   "
                  f"{'admissible' if ok else 'no'}")
        return ok

    found = largest_feasible(oracle, lam_e, rtol=rtol)
    total_iterations, total_evaluations = found.iterations, found.evaluations

    for _ in range(int(enrichments)):
        # aim new modes just above the current factor: that is where the
        # constraints that stopped the search are active
        probe = found.upper if np.isfinite(found.upper) else found.value * 1.1
        added = basis.enrich(state["best"], vertex_stresses, probe, criterion,
                             kappa_arr)
        if verbose:
            print(f"  enrichment: +{added} modes (basis now {basis.n_modes})")
        if not added:
            break
        found = largest_feasible(oracle, found.value, rtol=rtol)
        total_iterations += found.iterations
        total_evaluations += found.evaluations

    return MelanResult(
        factor=found.value, upper=found.upper, elastic=lam_e,
        residual_stress=state["best"], iterations=total_iterations,
        evaluations=total_evaluations, bracketed=found.bracketed,
        inner=state["log"], n_modes=basis.n_modes)
