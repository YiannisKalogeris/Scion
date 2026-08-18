r"""
Return mapping.

Integrates the rate equations over a finite strain increment: given a total
strain and the committed state, find the stress, the updated state, and the
tangent that a Newton solver needs.

The algorithm is the closest-point projection, applied to whatever combination
of criterion, hardening and flow rule it is handed. Nothing here is specialised
to von Mises.

Formulation
-----------
Everything is in Mandel notation (see :mod:`..notation`), so ``:`` is a dot
product and fourth-order tensors are ordinary matrices.

The unknowns are the **relative stress** ``xi = sigma - beta`` and the plastic
multiplier increment ``dlambda``. Using ``xi`` rather than ``sigma`` is what
keeps kinematic hardening tractable: substituting the back-stress update into
the stress update leaves a single equation,

    xi = xi_tr - dlambda D m(xi),      D = C_el + (2/3) C_kin I,

in which the back stress no longer appears. The system closed by consistency is

    r1 = xi - xi_tr + dlambda D m(xi)              (6)
    r2 = f(xi, kappa(alpha_n + dlambda))           (1)

solved by Newton, batched over quadrature points.

The consistent tangent
----------------------
Rather than deriving a closed form per model, the converged Jacobian is reused:
differentiating the residual system with respect to the total strain gives

    J [dxi/deps; dlambda/deps] = [C_el; 0],

one linear solve with six right-hand sides, from which

    dsigma/deps = C_el - C_el [ m (x) dlambda/deps + dlambda M dxi/deps ].

This is exact for any criterion, flow rule and hardening law, and it is exact
for the algorithm as actually implemented rather than for the continuum
equations it approximates -- which is the distinction that decides whether
Newton converges quadratically or linearly. It also cannot drift out of step
with the return map, since it is built from the same Jacobian.

Verify it with :func:`scion.solvers.tangent_check.check_tangent`; a
material-point residual is a legitimate residual, and the check applies at that
level as well as at the structural one.

The apex
--------
The Drucker-Prager cone has a singular tip where the flow direction is
undefined, and a predominantly tensile trial state returns there rather than to
the smooth surface. Such points are detected during the Newton loop and handed
to a separate scalar solve, which sets the deviator to zero and finds the
pressure from consistency alone. Their tangent is rank-one and purely
volumetric: all deviatoric stiffness is lost at the apex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flow_rules import Associative, FlowRule
from .hardening import (
    IsotropicHardening,
    KinematicHardening,
    LinearKinematic,
    NoKinematic,
)
from ..linear_elastic import isotropic_stiffness_mandel
from ..notation import from_mandel, to_mandel, to_mandel4
from .yield_criteria import YieldCriterion

__all__ = ["ReturnMapResult", "return_map", "PlasticState"]

#: Mandel image of the second-order identity: the trace/volumetric direction.
_T = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


@dataclass
class PlasticState:
    """Committed internal variables at a set of quadrature points.

    ``eps_p`` and ``beta`` are Mandel ``(n, 6)``; ``alpha`` is ``(n,)``.
    """

    eps_p: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "PlasticState":
        return cls(np.zeros((n, 6)), np.zeros(n), np.zeros((n, 6)))

    def copy(self) -> "PlasticState":
        return PlasticState(self.eps_p.copy(), self.alpha.copy(),
                            self.beta.copy())


@dataclass
class ReturnMapResult:
    """Outcome of a stress update."""

    sigma: np.ndarray          # (n, 6) Mandel
    state: PlasticState
    C_alg: np.ndarray          # (n, 6, 6) Mandel consistent tangent
    plastic: np.ndarray        # (n,) bool
    apex: np.ndarray           # (n,) bool
    iterations: int
    converged: bool
    residual: float

    def __repr__(self) -> str:
        return (f"ReturnMapResult(n={len(self.sigma)}, "
                f"plastic={int(self.plastic.sum())}, "
                f"apex={int(self.apex.sum())}, "
                f"iters={self.iterations}, converged={self.converged}, "
                f"res={self.residual:.2e})")


# ---------------------------------------------------------------------------

def _kinematic_modulus(kin: KinematicHardening) -> float:
    """``C_kin`` such that the back stress contributes ``(2/3) C_kin`` to D."""
    if isinstance(kin, NoKinematic):
        return 0.0
    if isinstance(kin, LinearKinematic):
        return kin.C
    raise NotImplementedError(
        f"{type(kin).__name__} is nonlinear in the back stress, so it cannot "
        f"be eliminated into the relative-stress form used here. Supported: "
        f"NoKinematic, LinearKinematic.")


def return_map(eps: np.ndarray, state: PlasticState,
               criterion: YieldCriterion,
               isotropic: IsotropicHardening,
               C_el: np.ndarray,
               flow: FlowRule | None = None,
               kinematic: KinematicHardening | None = None,
               tol: float = 1.0e-10, max_iter: int = 50,
               apex_tol: float = 1.0e-9) -> ReturnMapResult:
    """Integrate the constitutive equations over one strain increment.

    Parameters
    ----------
    eps : ``(n, 6)`` Mandel total strain at the end of the step.
    state : committed internal variables. Not modified; the updated state is
        returned. That separation is deliberate -- a Newton iterate that is
        later discarded must not advance the plastic strain, exactly as with
        the history field of a phase-field model.
    C_el : ``(6, 6)`` Mandel elastic stiffness, from
        :func:`~scion.materials.linear_elastic.isotropic_stiffness_mandel`.
    flow : defaults to associative.
    kinematic : defaults to none.

    Returns
    -------
    :class:`ReturnMapResult`, with the stress, the new state and the consistent
    tangent for every point, elastic or plastic.
    """
    flow = flow or Associative()
    kinematic = kinematic or NoKinematic()
    C_kin = _kinematic_modulus(kinematic)

    eps = np.asarray(eps, dtype=float)
    n = eps.shape[0]
    C_el = np.asarray(C_el, dtype=float)

    # ---- elastic predictor -------------------------------------------------
    sigma_tr = (eps - state.eps_p) @ C_el.T
    xi_tr = sigma_tr - state.beta

    kappa_n = isotropic.kappa(state.alpha)
    f_tr = criterion.f(from_mandel(xi_tr), kappa_n)
    scale = np.maximum(np.abs(kappa_n), 1.0)
    plastic = f_tr > tol * scale

    sigma = sigma_tr.copy()
    new = state.copy()
    C_alg = np.broadcast_to(C_el, (n, 6, 6)).copy()
    apex = np.zeros(n, dtype=bool)

    if not plastic.any():
        return ReturnMapResult(sigma, new, C_alg, plastic, apex, 0, True, 0.0)

    # ---- admissibility: a cone with too little dilatancy has no return -----
    # If the trial pressure lies beyond the apex, the only way back is to shed
    # volumetric stress -- which purely deviatoric flow cannot do. The model
    # then has no solution for that state at all. This is a property of the
    # constitutive choice, not a numerical failure, and the usual remedy is a
    # tension cutoff (a second surface capping the cone) rather than a better
    # solver.
    eta_c = getattr(criterion, "eta", 0.0)
    if eta_c:
        eta_g = getattr(getattr(flow, "potential", criterion), "eta", eta_c)
        if eta_g <= 0.0:
            xi_c = getattr(criterion, "xi", 1.0)
            beyond = plastic & (
                xi_c * kappa_n - eta_c * (sigma_tr @ _T / 3.0) < 0.0)
            if beyond.any():
                raise RuntimeError(
                    f"{int(beyond.sum())} point(s) have a trial pressure beyond "
                    f"the cone apex, but the flow rule has zero dilatancy, so "
                    f"the plastic strain cannot reduce the pressure and no "
                    f"admissible stress exists. Use a nonzero dilatancy angle "
                    f"or add a tension cutoff.")

    # ---- plastic correction ------------------------------------------------
    idx = np.flatnonzero(plastic)
    D = C_el + (2.0 / 3.0) * C_kin * np.eye(6)

    xi = xi_tr[idx].copy()
    dlam = np.zeros(len(idx))
    alpha_n = state.alpha[idx]

    converged = False
    it = 0
    res_norm = np.inf

    # Classify apex returns BEFORE iterating, not after failing to converge.
    # The smooth system has no solution for a trial state whose pressure is
    # already outside the cone with the deviator removed: plastic flow must
    # then shed volumetric stress, and the return lands on the tip. Detecting
    # this only once the deviator has collapsed never fires, because a Newton
    # with no root does not approach one -- it stalls at whatever residual it
    # reached, which is what a 50-iteration non-convergence looks like.
    at_apex = np.zeros(len(idx), dtype=bool)
    if eta_c:
        xi_c_ = getattr(criterion, "xi", 1.0)
        p_tr = xi_tr[idx] @ _T / 3.0
        at_apex = eta_c * p_tr - xi_c_ * isotropic.kappa(alpha_n) > 0.0

    for it in range(1, max_iter + 1):
        xi_t = from_mandel(xi)
        m = to_mandel(flow.dg_dsigma(criterion, xi_t))
        nrm = to_mandel(criterion.df_dsigma(xi_t))
        M = to_mandel4(flow.d2g_dsigma2(criterion, xi_t))

        alpha = alpha_n + dlam
        kappa = isotropic.kappa(alpha)
        H = isotropic.dkappa_dalpha(alpha)

        r1 = xi - xi_tr[idx] + dlam[:, None] * (m @ D.T)
        r2 = criterion.f(xi_t, kappa)

        smooth = ~at_apex
        if smooth.any():
            res_norm = float(max(np.abs(r1[smooth]).max(),
                                 np.abs(r2[smooth]).max()
                                 / max(scale[idx].max(), 1.0)))
        else:
            res_norm = 0.0
        if res_norm < tol * max(np.abs(xi_tr[idx]).max(), 1.0):
            converged = True
            break

        # a deviator that collapses during iteration is the same situation
        # arrived at from the other side; either way the Newton step is garbage
        at_apex |= criterion.singular_mask(xi_t) | (
            getattr(flow, "singular_mask", lambda s: np.zeros(len(s), bool))(xi_t))

        # 7x7 batched Newton
        J = np.zeros((len(idx), 7, 7))
        J[:, :6, :6] = np.eye(6) + dlam[:, None, None] * (D @ M)
        J[:, :6, 6] = m @ D.T
        J[:, 6, :6] = nrm
        J[:, 6, 6] = H * criterion.df_dkappa(xi_t, kappa)

        rhs = np.concatenate([-r1, -r2[:, None]], axis=1)[:, :, None]
        try:
            dx = np.linalg.solve(J, rhs)[:, :, 0]
        except np.linalg.LinAlgError:
            at_apex[:] = True
            break

        dx[at_apex] = 0.0                 # frozen; handled by the apex branch
        xi = xi + dx[:, :6]
        dlam = np.maximum(dlam + dx[:, 6], 0.0)

    # ---- apex return, where the smooth projection has no solution ----------
    if at_apex.any():
        xi[at_apex], dlam[at_apex] = _apex_return(
            xi_tr[idx][at_apex], alpha_n[at_apex], criterion, isotropic,
            flow, C_el, C_kin)

    # ---- assemble the updated state ---------------------------------------
    xi_t = from_mandel(xi)
    m = to_mandel(flow.dg_dsigma(criterion, xi_t))
    m = np.where(at_apex[:, None],
                 _apex_flow(criterion, flow)[None, :], m)

    new.alpha[idx] = alpha_n + dlam
    new.eps_p[idx] = state.eps_p[idx] + dlam[:, None] * m
    if C_kin:
        new.beta[idx] = state.beta[idx] + (2.0 / 3.0) * C_kin * dlam[:, None] * m
    sigma[idx] = xi + new.beta[idx]

    # ---- consistent tangent ------------------------------------------------
    C_alg[idx] = _consistent_tangent(
        xi, dlam, criterion, isotropic, flow, C_el, D, C_kin,
        alpha_n + dlam, at_apex)

    apex[idx] = at_apex
    return ReturnMapResult(sigma, new, C_alg, plastic, apex, it,
                           bool(converged or at_apex.all()), res_norm)


# ---------------------------------------------------------------------------

def _apex_flow(criterion, flow) -> np.ndarray:
    """Flow direction at the apex: purely volumetric, ``(eta_g/3) I``."""
    eta_g = getattr(getattr(flow, "potential", criterion), "eta", 0.0)
    return (eta_g / 3.0) * _T


def _apex_return(xi_tr, alpha_n, criterion, isotropic, flow, C_el, C_kin):
    """Project onto the cone tip: deviator zero, pressure from consistency.

    The flow direction is undefined at the apex, so the deviatoric part of the
    return is fixed by the requirement ``s = 0`` rather than by ``dg/dsigma``,
    and only the volumetric equation is solved.
    """
    eta = getattr(criterion, "eta", None)
    if not eta:
        raise NotImplementedError(
            "apex return requested for a criterion without a cone apex; only "
            "Drucker-Prager-type criteria have one")
    xi_c = getattr(criterion, "xi", 1.0)
    eta_g = getattr(getattr(flow, "potential", criterion), "eta", eta)

    K = (C_el[0, 0] + 2.0 * C_el[0, 1]) / 3.0           # bulk modulus
    A = 3.0 * K + (2.0 / 9.0) * C_kin

    p_tr = xi_tr @ _T / 3.0
    dlam = np.zeros(len(xi_tr))
    for _ in range(50):
        alpha = alpha_n + dlam
        r = eta * (p_tr - dlam * A * eta_g) - xi_c * isotropic.kappa(alpha)
        if np.abs(r).max() < 1.0e-12 * max(np.abs(p_tr).max(), 1.0):
            break
        drdl = -eta * A * eta_g - xi_c * isotropic.dkappa_dalpha(alpha)
        dlam = np.maximum(dlam - r / drdl, 0.0)

    p = p_tr - dlam * A * eta_g
    return p[:, None] * _T[None, :], dlam


def _consistent_tangent(xi, dlam, criterion, isotropic, flow, C_el, D, C_kin,
                        alpha, at_apex):
    """``dsigma/deps`` for the algorithm as implemented.

    Obtained by differentiating the converged residual system, so it is exact
    for the discrete update rather than for the continuum rate equations.
    """
    m_pts = len(xi)
    xi_t = from_mandel(xi)
    m = to_mandel(flow.dg_dsigma(criterion, xi_t))
    nrm = to_mandel(criterion.df_dsigma(xi_t))
    M = to_mandel4(flow.d2g_dsigma2(criterion, xi_t))
    kappa = isotropic.kappa(alpha)
    H = isotropic.dkappa_dalpha(alpha)

    J = np.zeros((m_pts, 7, 7))
    J[:, :6, :6] = np.eye(6) + dlam[:, None, None] * (D @ M)
    J[:, :6, 6] = m @ D.T
    J[:, 6, :6] = nrm
    J[:, 6, 6] = H * criterion.df_dkappa(xi_t, kappa)

    rhs = np.zeros((m_pts, 7, 6))
    rhs[:, :6, :] = C_el

    C = np.empty((m_pts, 6, 6))
    ok = ~at_apex
    if ok.any():
        sol = np.linalg.solve(J[ok], rhs[ok])
        dxi = sol[:, :6, :]                       # (m, 6, 6)
        dlam_d = sol[:, 6, :]                     # (m, 6)
        inner = (np.einsum("mi,mj->mij", m[ok], dlam_d)
                 + dlam[ok, None, None] * np.einsum("mij,mjk->mik", M[ok], dxi))
        C[ok] = C_el - C_el @ inner

    # at the apex the deviatoric stiffness vanishes and the response is a
    # rank-one volumetric operator
    if at_apex.any():
        eta = getattr(criterion, "eta", 0.0)
        xi_c = getattr(criterion, "xi", 1.0)
        eta_g = getattr(getattr(flow, "potential", criterion), "eta", eta)
        K = (C_el[0, 0] + 2.0 * C_el[0, 1]) / 3.0
        A = 3.0 * K + (2.0 / 9.0) * C_kin
        denom = xi_c * H[at_apex] + eta * A * eta_g
        dlam_d = (eta * K / np.where(denom == 0, 1e-300, denom))[:, None] * _T
        dp = K * _T[None, :] - (A * eta_g) * dlam_d
        if C_kin:
            dp = dp + (2.0 / 9.0) * C_kin * eta_g * dlam_d
        C[at_apex] = np.einsum("i,mj->mij", _T, dp)

    return C
