r"""
The Residual Stress Decomposition Method, and RSDM-S for shakedown.

Spiliopoulos and Panagiotou, CMAME 223-224 (2012) 186-198 for RSDM, and
CMAME 276 (2014) 410-430 for RSDM-S.

Neither of the other two methods here tells you what a structure that does *not*
shake down actually does. Melan and the Linear Matching Method both answer a
yes/no question about a load domain. RSDM answers a different and more physical
one: given a cyclic load history, what asymptotic state does the structure
settle into -- elastic shakedown, alternating plasticity, or ratchetting -- and
what is the residual stress field of that state.

It does it without stepping through cycles. The observation is that in the
asymptotic state everything is periodic, so the residual stress is periodic too,
and can be written as a Fourier series in time whose coefficients are the
unknowns:

    rho(tau) = a_0/2 + sum_k [ a_k cos(2 k pi tau) + b_k sin(2 k pi tau) ]   (10)

Differentiating and using orthogonality gives each coefficient as an integral of
the *rate* over one cycle,

    a_k = -(1/(k pi)) INT_0^1 rho_dot(tau) sin(2 k pi tau) dtau              (13)
    b_k = +(1/(k pi)) INT_0^1 rho_dot(tau) cos(2 k pi tau) dtau              (14)

and the rate at any instant follows from equilibrium and compatibility. That
closes the loop: coefficients give the residual stress, the residual stress
gives the rate, the rate gives the coefficients. It is a fixed point, ``x =
g(x)``, iterated to convergence.

The constant term is the awkward one
------------------------------------
Differentiation annihilates ``a_0``, so no integral of the rate can recover it.
It is instead pinned by requiring the cycle to close, comparing the series at
the start and the end of the cycle (Eq. 15) to give

    a_0(1)/2 = ( a_0(0)/2 + sum_k a_k(0) ) - sum_k a_k(1)
               + INT_0^1 rho_dot(tau) dtau                                   (16)

Three coefficients -- ``a_0``, ``a_1``, ``b_1`` -- are enough in practice, which
is what makes the method cheap.

Plasticity without a return map
-------------------------------
A closest-point projection would return the stress along ``-D eps_dot^pl``.
RSDM uses instead the *radial* excess: the part of the total stress sticking out
of the yield surface along its own direction,

    sigma_p = s (1 - sigma_Y / sigma_eff)   when   sigma_eff > sigma_Y

which the paper calls the plastic stress vector and treats as an equivalent
measure of the plastic strain rate -- both are zero together, and both point the
same way. It costs one division instead of a local Newton solve, and it is what
lets the whole method run on an existing elastic code.

The linear solve
----------------
At each cycle point, with the plastic stress known,

    K r_dot = INT B^T sigma_p dV,     rho_dot = D B r_dot - sigma_p          (19, 23)

``K`` is the ordinary elastic stiffness, factorised once for the entire
analysis. The resulting rate is self-equilibrated by construction:
``INT B^T rho_dot dV = K r_dot - INT B^T sigma_p dV = 0``.

Unlike the Linear Matching Method this needs no B-bar. The matched material
there is exactly incompressible; here the material is the real elastic one, so
ordinary integration is correct and mean dilatation would only soften it.

What is here and what is not
----------------------------
This module gives the **cyclic state for a prescribed load history**: what the
structure settles into, and its residual stress field. That is RSDM.

**RSDM-S is deliberately absent.** The shakedown variant descends a load factor
until the trigonometric content vanishes, and an implementation of it converged
reliably to the wrong answer -- ``2 p_e``, the alternating-plasticity limit,
where the thick cylinder's published value is ``0.8 sigma_Y``, the
collapse-governed one (Spiliopoulos and Panagiotou 2014, Fig. 10, at
``R_2/R_1 = 0.5``). The cause is here rather than in the descent: above
shakedown this fixed point converges too slowly to be read. At a load between
the two limits it was still moving after four thousand iterations, and warm
starting from a higher factor -- which the method requires -- suppressed the
measured plastic stress sixfold at the same load and budget. The descent is
driven by those numbers, so its step decayed geometrically and the sequence
asymptoted onto the alternating limit.

Accelerating the fixed point is the thing to fix; the outer loop was not the
problem. Until then, use :mod:`~simulacrum.shakedown.lmm` for an upper bound and
:mod:`~simulacrum.shakedown.equilibrium_elements` for a lower one -- both are
verified against the same benchmark and bracket it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import linalg as spla

from ..materials.notation import (
    mandel_to_voigt_stress,
    mandel_to_voigt_tangent,
    voigt_strain_to_mandel,
)

__all__ = ["CyclicProgram", "RSDMResult", "rsdm"]

_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _deviator(a: np.ndarray) -> np.ndarray:
    return a - (a[..., :3].sum(axis=-1) / 3.0)[..., None] * _TRACE


def _effective(a: np.ndarray) -> np.ndarray:
    s = _deviator(a)
    return np.sqrt(1.5 * np.einsum("...i,...i->...", s, s))


# ---------------------------------------------------------------------------

class CyclicProgram:
    """A cyclic load history in the time domain: multipliers at cycle points.

    ``multipliers`` is ``(n_time, n_cases)`` giving each basic load case's
    multiplier at each point of the cycle, with ``tau`` running over ``[0, 1)``.

    RSDM needs a *history*, not a domain, which is the substantive difference
    from Melan and LMM. A load domain has to be converted into a path that
    visits its extremes: convexity of the yield surface then guarantees that
    shaking down over the path implies shaking down over the whole convex hull
    of the visited points, so passing through the vertices is enough.
    """

    def __init__(self, multipliers, tau=None):
        self.multipliers = np.atleast_2d(np.asarray(multipliers, dtype=float))
        n = self.multipliers.shape[0]
        self.tau = (np.linspace(0.0, 1.0, n, endpoint=False)
                    if tau is None else np.asarray(tau, dtype=float))
        if self.tau.size != n:
            raise ValueError("tau and multipliers must agree in length")

    # ------------------------------------------------------------------
    @classmethod
    def proportional(cls, n_cases: int = 1, n_time: int = 24) -> "CyclicProgram":
        """All loads rise together to their maximum and back, ``0 -> P* -> 0``.

        A smooth hump rather than a triangle: the paper uses a fourth-order
        polynomial through the same limits, and a smooth program keeps the
        Fourier series from having to represent a corner.
        """
        tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
        alpha = 16.0 * tau ** 2 * (1.0 - tau) ** 2         # peaks at 1, tau=1/2
        return cls(np.tile(alpha[:, None], (1, n_cases)), tau)

    @classmethod
    def through_vertices(cls, domain, per_leg: int = 8) -> "CyclicProgram":
        """A path visiting every vertex of a load domain and returning.

        Convexity does the rest: a structure that shakes down over a path
        through the vertices shakes down over the whole domain they span.
        """
        vertices = np.asarray(domain.vertices, dtype=float)
        legs = list(vertices) + [np.zeros(domain.n_cases)]
        points = []
        for start, end in zip([np.zeros(domain.n_cases)] + legs[:-1], legs):
            for s in np.linspace(0.0, 1.0, per_leg, endpoint=False):
                points.append((1.0 - s) * start + s * end)
        return cls(np.array(points))

    # ------------------------------------------------------------------
    @property
    def n_time(self) -> int:
        return self.multipliers.shape[0]

    @property
    def n_cases(self) -> int:
        return self.multipliers.shape[1]

    def elastic_history(self, envelope) -> np.ndarray:
        """``(n_time, n_points, 6)`` Mandel elastic stress over the cycle."""
        return np.einsum("tc,cpi->tpi", self.multipliers, envelope.stresses)

    def peak_index(self) -> int:
        """The cycle point where the multipliers are largest overall -- the
        ``tau*`` the starting load factor is measured at."""
        return int(np.argmax(np.abs(self.multipliers).sum(axis=1)))

    def __repr__(self) -> str:
        return (f"CyclicProgram({self.n_time} cycle points, "
                f"{self.n_cases} load cases)")


# ---------------------------------------------------------------------------

@dataclass
class RSDMResult:
    """The asymptotic cyclic state for one load factor."""

    a0: np.ndarray = field(repr=False)          # (n_points, 6), the constant
    a: np.ndarray = field(repr=False)           # (n_harmonics, n_points, 6)
    b: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)    # (n_time, n_points, 6)
    plastic_stress: np.ndarray = field(repr=False)
    iterations: int = 0
    converged: bool = False
    phi: float = 0.0                            # trigonometric content
    scale: float = 1.0                          # yield stress, for thresholds

    @property
    def constant_residual(self) -> np.ndarray:
        """``a_0 / 2``: the time-independent part, and the whole field at
        shakedown."""
        return 0.5 * self.a0

    @property
    def plastic_norm(self) -> float:
        """Largest Euclidean norm of the plastic stress over the cycle points.

        The paper's *second* descending function, and the one that separates
        shakedown from ratchetting. ``phi`` going to zero only says the residual
        stress has stopped varying in time; a structure collapsing incrementally
        also has a constant-in-time residual stress, with a steady plastic
        stress driving it. Both have to vanish.
        """
        return float(np.max(np.linalg.norm(
            self.plastic_stress.reshape(self.plastic_stress.shape[0], -1),
            axis=1)))

    @property
    def state(self) -> str:
        """Which of the three asymptotic states this is.

        ``shakedown``      no plastic stress anywhere at any cycle point
        ``alternating``    plastic stress present but integrating to zero over
                           the cycle -- it reverses and cancels
        ``ratchetting``    plastic stress with a nonzero mean, so strain
                           accumulates every cycle without bound

        Judged relative to the yield stress. An absolute threshold is wrong
        here: a converged shakedown state leaves a plastic stress at round-off,
        which is around 1e-9 in units where the yield stress is a few hundred,
        and calling that ratchetting misclassifies every converged run.
        """
        peak = float(np.abs(self.plastic_stress).max())
        if peak <= 1e-8 * self.scale:
            return "shakedown"
        mean = np.abs(self.plastic_stress.mean(axis=0)).max()
        return "ratchetting" if mean > 1e-3 * peak else "alternating"

    def __repr__(self) -> str:
        return (f"RSDMResult(state={self.state}, phi={self.phi:.4g}, "
                f"iterations={self.iterations}, converged={self.converged})")


def rsdm(elastic_history: np.ndarray, tables, free_dofs, C_mandel, sigma_y,
         n_harmonics: int = 1, max_iter: int = 2000, tol: float = 1.0e-8,
         initial=None, factorised=None, verbose: bool = False) -> RSDMResult:
    """The cyclic steady state for a prescribed load history.

    Parameters
    ----------
    elastic_history : ``(n_time, n_points, 6)`` Mandel elastic stress, already
        scaled by the load factor.
    tables, free_dofs, C_mandel : the mesh, its unconstrained degrees of
        freedom and the elastic stiffness in Mandel form.
    n_harmonics : how many ``(a_k, b_k)`` pairs. One is the paper's
        recommendation and is usually enough.
    initial : ``(a0, a, b)`` from a previous run, to warm-start the outer loop
        of RSDM-S.
    max_iter : the fixed point converges linearly and can want a thousand
        iterations from a cold start -- each is cheap, since the stiffness is
        already factorised, but the default is generous for a reason. Cutting
        it short does not fail loudly: it leaves trigonometric content that
        RSDM-S reads as a live ratchetting mechanism and descends on.
    factorised : a prefactorised free-free stiffness, so the outer loop pays for
        the decomposition once.

    Implements steps 1-10 of section 5 of the 2012 paper.
    """
    history = np.asarray(elastic_history, dtype=float)
    n_time, n_points, _ = history.shape
    yield_stress = np.broadcast_to(np.asarray(sigma_y, dtype=float),
                                   (n_points,))
    free = np.asarray(free_dofs, dtype=np.int64)
    n_dof = tables.vector_dofs.n_total
    nv = tables.n_voigt
    C = np.asarray(C_mandel, dtype=float)

    if factorised is None:
        C_v = mandel_to_voigt_tangent(C[None], tables.dim)[0]
        K, _ = tables.stiffness_and_force(
            np.zeros((n_points, nv)),
            np.broadcast_to(C_v, (n_points, nv, nv)), n_dof)
        factorised = spla.factorized(K.tocsr()[free][:, free].tocsc())

    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    k = np.arange(1, n_harmonics + 1)
    cos = np.cos(2.0 * np.pi * np.outer(k, tau))          # (n_harm, n_time)
    sin = np.sin(2.0 * np.pi * np.outer(k, tau))
    dtau = 1.0 / n_time            # uniform: the trapezoid rule on a periodic
                                   # integrand is the rectangle rule

    if initial is None:
        a0 = np.zeros((n_points, 6))
        a = np.zeros((n_harmonics, n_points, 6))
        b = np.zeros((n_harmonics, n_points, 6))
    else:
        a0, a, b = (np.array(x, dtype=float, copy=True) for x in initial)

    rho = np.zeros((n_time, n_points, 6))
    plastic = np.zeros((n_time, n_points, 6))
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        previous = rho.copy()

        # Eq (10): the residual stress at every cycle point
        rho = (0.5 * a0[None] + np.einsum("kt,kpi->tpi", cos, a)
               + np.einsum("kt,kpi->tpi", sin, b))

        rho_dot = np.empty_like(rho)
        for t in range(n_time):
            total = history[t] + rho[t]                            # step 2
            effective = _effective(total)                          # step 3
            excess = effective > yield_stress
            sp = np.zeros((n_points, 6))
            if excess.any():
                scale = np.where(excess,
                                 1.0 - yield_stress / np.maximum(effective,
                                                                 1e-300), 0.0)
                sp = scale[:, None] * _deviator(total)
            plastic[t] = sp

            # step 4-5: K r_dot = INT B^T sigma_p dV
            _, force = tables.stiffness_and_force(
                mandel_to_voigt_stress(sp, tables.dim),
                np.zeros((n_points, nv, nv)), n_dof)
            rate = np.zeros(n_dof)
            rate[free] = factorised(force[free])

            # step 6, Eq (19) in residual form
            eps = voigt_strain_to_mandel(tables.strains_voigt(rate),
                                         tables.dim)
            rho_dot[t] = eps @ C.T - sp

        # step 8: Eqs (13), (14) and (16)
        a_old_sum = a.sum(axis=0)
        a = -(dtau / (np.pi * k))[:, None, None] * np.einsum(
            "kt,tpi->kpi", sin, rho_dot)
        b = +(dtau / (np.pi * k))[:, None, None] * np.einsum(
            "kt,tpi->kpi", cos, rho_dot)
        # the constant term is invisible to rho_dot, so it is pinned by closing
        # the cycle instead
        a0 = a0 + 2.0 * (a_old_sum - a.sum(axis=0)
                         + dtau * rho_dot.sum(axis=0))

        change = np.linalg.norm(rho - previous)
        scale_norm = max(np.linalg.norm(rho), 1e-300)
        if verbose:
            print(f"      rsdm {iteration:3d}: |drho|/|rho| = "
                  f"{change / scale_norm:.3e}")
        if iteration > 1 and change <= tol * scale_norm:
            converged = True
            break

    phi = float(np.linalg.norm(a, axis=(1, 2)).sum()
                + np.linalg.norm(b, axis=(1, 2)).sum())
    return RSDMResult(a0=a0, a=a, b=b, residual=rho, plastic_stress=plastic,
                      iterations=iteration, converged=converged, phi=phi,
                      scale=float(np.max(yield_stress)))


# ---------------------------------------------------------------------------
