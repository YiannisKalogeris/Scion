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

Plasticity without a local Newton solve
---------------------------------------
RSDM works with the *radial* excess: the part of the total stress sticking out
of the yield surface along its own direction,

    sigma_p = s (1 - sigma_Y / sigma_eff)   when   sigma_eff > sigma_Y

which the paper calls the plastic stress vector. It costs one division instead
of a local Newton solve, and it is what lets the whole method run on an
existing elastic code.

**For von Mises this is not an approximation of the closest-point projection,
it IS the closest-point projection** -- an earlier note here calling it "an
equivalent measure of the plastic strain rate" undersold it. Projecting onto
the yield surface in the complementary-energy norm leaves the hydrostatic part
alone, because the criterion does not constrain it, and the metric restricted
to the deviatoric subspace is ``1/(2 mu)`` times the identity, so the
projection of the deviator is radial. Hence

    sigma - PI_K(sigma) = s - s sigma_Y/sigma_eff = sigma_p

exactly, and the plastic strain increment of a backward-Euler return map is
``D^-1 sigma_p``. Verified against
:func:`~scion.materials.plasticity.return_map` on 399 yielding states spanning
``sigma_eff/sigma_Y`` up to 8.1: agreement to **7.2e-16** relative, with the
yielding sets identical. The two routes differ for a criterion whose surface is
not a sphere in the energy metric -- Drucker-Prager, where the projection is
not radial and the apex needs its own branch -- which is why the statement is
specific to von Mises.

Plastic strain, and what it is for
----------------------------------
That identity is what makes the plastic strain recoverable without touching the
iteration. :attr:`RSDMResult.plastic_strain` is the running sum of
``D^-1 sigma_p`` over the cycle, and three indicators are built on it:
:attr:`~RSDMResult.cyclic_amplitude` (the volume-weighted amplitude of the
cyclic residual stress), :attr:`~RSDMResult.plastic_strain_range` and
:attr:`~RSDMResult.ratchet_rate` (the net plastic strain per cycle).

**They are indicators and not magnitudes, and that is a property of the rate
convention rather than of the accumulation.** Driving the residual stress with
``sigma_p`` itself makes the rate scale a free penalty parameter, so the
converged state above shakedown is proportional to it -- on the Bree strip
``plastic_strain_range`` returns 0.084261 where the closed form gives exactly
1.0. The *regime* boundaries are untouched, because ``sigma_p = 0`` is scale
free, which is why these are sound as indicators and why the shakedown limit
comes out to 0.02%. For the magnitude use
:func:`~scion.shakedown.cyclic_steady_state`, which enforces yield instead of
penalising its violation and reproduces that 1.0 to 9.3e-16. :meth:`~RSDMResult.regime` classifies on them, and unlike
:attr:`~RSDMResult.state` it separates the elastic regime from shakedown and
needs no ``1e-3 * peak`` heuristic -- on a fully reversed history the net
plastic strain comes out at **3.0e-13** of the shear yield strain against a
range of 0.088, so alternating plasticity and ratchetting are eleven orders
apart rather than thresholded.

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
``R_2/R_1 = 0.5``). The cause is here rather than in the descent, and it is
not slow convergence: **above the shakedown limit this fixed point does not
exist.** The Fourier series presupposes a periodic asymptotic state, and
ratchetting is not periodic, so what the iteration does above ``lambda_sd`` is
drift. Measured on the thick cylinder, ``|a_0/2|_inf`` over the yield stress
against the iteration count:

    iterations         100     200     400     800    1600    3200
    0.90 p_lim      0.5572  0.5710  0.5725  0.5725  0.5725  0.5725
    1.00 p_lim      0.7435  0.7878  0.8096  0.8129  0.8130  0.8130
    1.04 p_lim      0.8078  0.8589  0.9077  0.9848  1.0960  1.2260
    1.20 p_lim      1.0200  1.2170  1.5410  2.0290  2.7800  3.6940

Below the limit it settles and the cycle-mean plastic stress falls to round-off
(5.8e-17 of the yield stress at 0.90). Above it the constant term grows by a
steady ~1.33x per doubling of the iteration count and passes **3.7 times the
yield stress**, which no admissible residual field can be. So there is nothing
to accelerate onto -- an earlier note here proposing acceleration as the fix is
retracted -- and the descent read a number that was still growing, which is why
its step decayed onto the wrong limit.

What that drift is good for
---------------------------
It is a cleaner indicator than :attr:`RSDMResult.state`, which is unreliable at
any budget short of convergence and errs towards reporting failure: at 500
iterations it calls the 0.90 p_lim case ratchetting, and it needs ~1600 to stop
doing so. The growth ratio does not need convergence at all. With
``g = |a_0/2|_inf(2N) / |a_0/2|_inf(N)`` at ``N = 1600``, 5x5 mesh:

    lam/p_lim   0.70     0.95     1.00     1.005    1.010    1.020    1.20
    g          1.00000  1.00000  1.00000  1.00001  1.00001  1.03344  1.32875

A sharp transition in (1.010, 1.020): 1.00001 to 1.03344 across one grid
interval, with nothing below 1.010 departing from unity in the fifth decimal.

**An earlier entry here claimed agreement to 0.08% with a Koiter bound of
1.0108 from :mod:`~scion.shakedown.lmm` on the same full-integration mesh, and
that is RETRACTED: the figure does not reproduce.** LMM requires b-bar -- its
matched material is exactly incompressible -- and on fully integrated tables it
returns ``nan`` without converging at every ``strain_floor`` from 1e-9 to 1e-3
and every ``penalty`` from 1e2 to 1e4, with an exactly singular matrix. On
b-bar tables it gives 1.00730 p_lim for this mesh, 1.00501 at 6x6 and 1.00113
at 12x12 -- converging from above, as an upper bound should -- and 1.0108
matches none of them. So there is no like-for-like bound for this table, since
RSDM must *not* have b-bar; the fully integrated transition sitting above the
b-bar bound is what locking predicts, and that comparison is indicative only.

The sharp cross-check lives at a fixed discretisation instead:
:func:`~scion.shakedown.cyclic_steady_state` on b-bar tables stops
equilibrating at **1.00747 p_lim against that same 1.00730** -- 0.02%, between
a kinematic bounding method and an incremental elastoplastic analysis.

The iteration count is set by the load level, not the mesh: 593 / 619 / 575 /
578 iterations to a relative change of 1e-8 at 0.90 p_lim over 100 / 256 / 576
/ 1024 quadrature points, at 2.8 / 4.8 / 8.6 / 13.9 ms an iteration.

Until RSDM-S exists, use :mod:`~scion.shakedown.lmm` for an upper bound and
:mod:`~scion.shakedown.equilibrium_elements` for a lower one -- both are
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


def _frobenius(a: np.ndarray) -> np.ndarray:
    """Euclidean norm over the component axis of a Mandel array.

    In Mandel this *is* the tensor norm ``sqrt(a:a)``, because the three shear
    slots carry ``sqrt(2)``. In Voigt it is not, and would under-weight the
    shears by that factor. Worth stating because the indicator definitions say
    only "the Euclidean norm of the components at one Gauss point", which is
    unambiguous only once the notation is fixed.
    """
    return np.linalg.norm(a, axis=-1)


def _strain_range(eps_p: np.ndarray) -> float:
    r"""``max_g max_{i,j} || eps^p_g(t_i) - eps^p_g(t_j) ||`` over a cycle.

    Shared with :mod:`~scion.shakedown.cyclic` so the two methods report the
    same quantity from the same code and a difference between them is a
    difference of state rather than of definition.

    ``n_t`` passes over the history rather than the full
    ``(n_points, n_t, n_t)`` Gram array, which reaches 327 MB at 10,000 points
    and 64 cycle samples.
    """
    worst = 0.0
    for i in range(eps_p.shape[0]):
        worst = max(worst, float(_frobenius(eps_p - eps_p[i]).max()))
    return worst


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
    compliance: np.ndarray | None = field(default=None, repr=False)
    weights: np.ndarray | None = field(default=None, repr=False)
    sigma_0: float = 0.0                        # reference stress for ``A``

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

    # -- plastic strain, and the three asymptotic-cycle indicators --------

    def _require(self, what: str) -> None:
        if self.compliance is None:
            raise ValueError(
                f"{what} needs the elastic compliance, which `rsdm` stores on "
                f"the result. This object was built by hand without it.")

    @property
    def plastic_strain_rate(self) -> np.ndarray:
        r"""``(n_time, n_points, 6)`` Mandel plastic strain rate over the cycle.

        The backward-Euler return map written as a projection,

            eps_dot^p(t_i) = (1/dt) D^-1 [ sigma(t_i) - PI_K(sigma(t_i)) ]

        and for von Mises the bracket **is** the radial plastic stress this
        method already carries: the closest-point projection leaves the
        hydrostatic part alone and scales the deviator by
        ``sigma_Y / sigma_eq``, so the difference is exactly
        ``s (1 - sigma_Y/sigma_eq)``. Checked against
        :func:`~scion.materials.plasticity.return_map` on 399 yielding states
        spanning ``sigma_eq/sigma_Y`` up to 8.1, agreeing to **7.2e-16**
        relative. So the module docstring's "equivalent measure of the plastic
        strain rate" understates it -- for von Mises it is an identity, and
        nothing had to be added to the iteration to recover the strain.

        ``D^-1`` rather than ``1/(2 mu)``: the two agree to the bit here
        because the radial excess is deviatoric, but the general form needs no
        isotropy assumption and costs one 6x6 solve.

        **The rate scaling is a convention and this module does not use the
        one above.** Eq. (eq:epsp) carries a ``1/dt``, which drives the
        residual stress ``n_t`` times harder and therefore converges, as
        ``dt -> 0``, to a state with ``sigma_p -> 0`` -- the exact plasticity
        limit, yield enforced pointwise. The 2012 iteration implemented here
        drives it with ``sigma_p`` itself, so

            eps_dot^p(tau) = D^-1 sigma_p(tau),     tau in [0, 1)

        which is a penalty regularisation with a relaxation time of one cycle.
        The two agree exactly on where shakedown ends -- ``sigma_p = 0`` is
        scale-free -- and disagree on the indicators above it. See the
        ``n_t`` table in :attr:`plastic_strain` before quoting a strain range
        against the published form.
        """
        self._require("plastic_strain_rate")
        return self.plastic_stress @ self.compliance.T

    @property
    def plastic_strain(self) -> np.ndarray:
        r"""``(n_time, n_points, 6)`` plastic strain accumulated over the cycle.

            eps^p_g(t) = INT_0^t eps_dot^p_g(tau) dtau

        on the ``n_t`` samples, so ``dt`` times a running sum. **The ``dt``
        matters and has to match the rate convention the iteration used** --
        dropping it, which is what the published ``1/dt`` form would license,
        makes both indicators below scale with ``n_t`` instead of converging.
        Measured at ``lambda = 1.20 p_lim`` on the thick cylinder, ``n_t`` =
        8 / 16 / 32 / 64:

            without dt   1.06e-4  1.49e-4  3.20e-4  6.39e-4   ~2x a doubling
            with dt      1.33e-5  9.30e-6  1.00e-5  9.98e-6   converged

        and the same at 1.02 p_lim (3.8 / 4.6 / 3.6 / 3.6 e-7). So these are
        quantities rather than artefacts, but only in the convention their own
        fixed point was computed in.

        Measured from the first stored cycle point. The cycle here starts at
        ``tau = 0`` where the published indices start at ``dt``, which is a
        cyclic shift of the samples; both indicators below are invariant to it
        (one is a max over all pairs, the other a sum over the whole cycle).

        **Plastic incompressibility is exact, not approximate.** The radial
        excess is a deviator and ``D^-1`` of a deviator is a deviator, so
        ``tr(eps^p) = 0`` to round-off (measured 1.3e-18) rather than to a
        tolerance. A nonzero trace here is a bug.
        """
        self._require("plastic_strain")
        dt = 1.0 / self.plastic_stress.shape[0]
        return dt * np.cumsum(self.plastic_stress @ self.compliance.T, axis=0)

    @property
    def plastic_strain_range(self) -> float:
        r"""``Delta eps^p``: the largest plastic strain range per cycle.

            Delta eps^p = max_g max_{i,j} || eps^p_g(t_i) - eps^p_g(t_j) ||

        **An indicator, not a magnitude.** Zero in the elastic and shakedown
        regimes, because the cycle then carries no plastic strain at all, and
        monotone in the load above them -- but its *size* carries the rate
        scale of the 2012 iteration, which is a free penalty parameter. On the
        Bree strip at ``x, y = 0, 3`` and ``lambda = 1`` this returns 0.084261
        against a closed-form 1.0.
        :func:`~scion.shakedown.bree.cyclic_strip` returns 1.0 to 9.3e-16 on
        the same problem, and
        :func:`~scion.shakedown.cyclic_steady_state` is the finite element
        counterpart. Do not hand this number to a fatigue curve.

        Computed as ``n_t`` passes over the history rather than by forming the
        full ``(n_points, n_t, n_t)`` Gram array, which reaches 327 MB at
        10,000 points and 64 cycle samples.

        Note this is a *range*, not an amplitude. Every life curve in
        :mod:`scion.fatigue` takes an amplitude, so halve it there -- the same
        factor of two that module is built around.
        """
        self._require("plastic_strain_range")
        return _strain_range(self.plastic_strain)

    @property
    def ratchet_rate(self) -> float:
        r"""``r``: the largest net plastic strain per cycle.

            r = max_g || eps^p_g(T) ||

        Zero for alternating plasticity, where the plastic strain reverses and
        cancels over the cycle, and positive for ratchetting, where it does
        not. It is the physically meaningful version of the ``1e-3 * peak``
        heuristic :attr:`state` uses.
        """
        self._require("ratchet_rate")
        net = self.plastic_stress.mean(axis=0) @ self.compliance.T
        return float(_frobenius(net).max())

    @property
    def plastic_strain_scale(self) -> float:
        """``sigma_0 / (2 mu)``, the shear yield strain -- a scale to read
        :attr:`ratchet_rate` and :attr:`plastic_strain_range` against.

        They are absolute strains, so a bare tolerance on them is not
        transferable between materials. The last diagonal entry of the
        compliance is ``1/(2 mu)`` in Mandel -- all three shear slots carry it
        -- and ``1/E`` for a one-component model such as the Bree strip, which
        is why it is indexed from the end rather than at ``[3, 3]``.
        """
        self._require("plastic_strain_scale")
        return float(self.sigma_0 * self.compliance[-1, -1])

    @property
    def cyclic_amplitude(self) -> float:
        r"""``A``: the normalised amplitude of the cyclic residual stress.

            A = (1/sigma_0) ( sum_k ||a_k||_W^2 + ||b_k||_W^2 )^{1/2}

        with ``||x||_W = (x^T W x)^{1/2}`` over the Gauss-point integration
        weights. Zero in the elastic and shakedown regimes.

        **Not the same quantity as** :attr:`phi`, which is an *unweighted* sum
        of Frobenius norms and so depends on the mesh through the point count
        rather than through the volume each point represents. ``phi`` is kept
        because the 2012 convergence test is written on it; ``A`` is the one to
        report, and the one a first-passage search should bisect.
        """
        self._require("cyclic_amplitude")
        if self.weights is None:
            raise ValueError("cyclic_amplitude needs the integration weights")
        w = self.weights[None, :, None]
        total = float((w * self.a ** 2).sum() + (w * self.b ** 2).sum())
        return np.sqrt(total) / self.sigma_0

    def regime(self, eta_a: float = 1.0e-6,
               eta_r: float = 1.0e-6) -> str:
        """Which of the four regimes, by the indicators rather than by
        :attr:`state`'s heuristic.

        ``eta_a`` is absolute (``A`` is already normalised by ``sigma_0``);
        ``eta_r`` is relative to :attr:`plastic_strain_scale`, since ``r`` is
        an absolute strain.

        ``elastic`` and ``shakedown`` are separated by whether a residual
        stress field developed at all -- both carry zero plastic strain in the
        asymptotic cycle, so the indicators alone cannot tell them apart. The
        published route is to compare against ``lambda_el`` from the elastic
        solution; this is the equivalent statement made from the converged
        cycle, and :func:`~scion.shakedown.elastic_factor` is the other one.
        """
        if self.ratchet_rate > eta_r * self.plastic_strain_scale:
            return "ratchetting"
        if self.cyclic_amplitude > eta_a:
            return "alternating"
        if np.abs(self.constant_residual).max() <= 1e-12 * self.scale:
            return "elastic"
        return "shakedown"

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
    # the compliance is what turns the stored plastic stress back into a
    # plastic strain, and the weights are what make ``A`` a volume-weighted
    # norm rather than a point-count-weighted one
    return RSDMResult(a0=a0, a=a, b=b, residual=rho, plastic_stress=plastic,
                      iterations=iteration, converged=converged, phi=phi,
                      scale=float(np.max(yield_stress)),
                      compliance=np.linalg.inv(C),
                      weights=np.asarray(tables.w, dtype=float).ravel(),
                      sigma_0=float(np.mean(yield_stress)))


# ---------------------------------------------------------------------------
