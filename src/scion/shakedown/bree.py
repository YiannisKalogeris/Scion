r"""
The Bree problem: closed-form regime boundaries for the uniaxial idealization.

Bree, J. Strain Analysis 2 (1967) 226-238.

A thin cylinder wall, or equivalently a strip, under a *constant* primary
membrane stress ``sigma_P`` and a *cyclic* linear through-thickness thermal
stress whose surface value is ``sigma_T``. With ``zeta = 2z/h`` in ``[-1, 1]``
the elastic stress is

    sigma^E(zeta, tau) = sigma_P + sigma_T zeta g(tau),     g: 0 -> 1 -> 0

so the thermal part switches on and off rather than reversing. In the
normalised coordinates ``x = sigma_P/sigma_Y`` and ``y = sigma_T/sigma_Y`` the
plane splits into four regimes -- elastic, shakedown, alternating plasticity
and ratchetting -- separated by curves that are known in closed form. That is
what makes this the calibration problem for any direct method: unlike the
thick cylinder, which pins one load factor, this pins a whole two-parameter
map.

``sigma_T`` is the SURFACE value
--------------------------------
``sigma_T = E alpha DT / [2 (1 - nu)]`` for a plate whose temperature varies
linearly by ``DT`` through the thickness and whose curvature is restrained.
That is the stress at the face, not the peak-to-peak variation across the
thickness, which is twice it. The distinction decides every boundary below by
a factor of two, so it is worth stating rather than inferring: **measured**,
``max |sigma^E|`` over the cycle and the thickness is ``x + y`` exactly
(2.100000 at ``x = 0.1``, ``y = 2.0``), which is only true in the surface
convention.

Three of the four boundaries are derived here, not transcribed
--------------------------------------------------------------
``lambda_el`` is first yield and needs no argument: the peak elastic stress is
``sigma_P + sigma_T`` at the hot face at the top of the cycle, so the elastic
boundary is ``y = 1 - x``.

The shakedown boundary follows from Melan on the strip in closed form. A
residual field ``rho(zeta)`` carries no net force, so admissibility needs, at
every ``zeta``, an interval for ``rho`` that both ends of the cycle fit in:

    lambda sigma_T |zeta| <= 2 sigma_Y                  (the range condition)

worst at the face, giving ``y <= 2``; and the intervals must additionally
admit a zero-force field, ``INT U(zeta) dzeta >= 0``, which works out to

    2 (sigma_Y - lambda sigma_P) - lambda sigma_T / 2 >= 0   ->   y <= 4(1 - x)

so the two familiar curves are the two constraints of one linear program, and
which of them binds is exactly the ``y >= 4x`` branch condition in
:func:`transition_loads`. **Checked against that linear program solved
numerically** over twelve rays spanning both branches: agreement to
**6.6e-14**.

The remaining boundary, alternating plasticity to ratchetting at ``x y = 1``,
is taken from Bree. It is a kinematic statement about the ratchetting
mechanism rather than a Melan one, so the argument above does not reach it;
what *is* checked here is that it meets the other two at ``(0.5, 2)`` and that
it makes ``lambda_rr >= lambda_sd`` precisely when ``y >= 4x``.

A correction to the elastic limit
---------------------------------
A draft of the accompanying manuscript gave the elastic boundary as
``y = 2(1 - x)`` and ``lambda_el = 1/(x + y/2)``, which is the *range*
convention's answer while the other three boundaries are in the surface
convention. The signature is that it destroys the shakedown region at small
primary stress: as ``x -> 0`` the measured window ``lambda_sd / lambda_el``
is **1.0000** under that form and **2.0000** under ``1/(x + y)``, and twice
first yield is the classical answer for a self-equilibrated cyclic stress.
Mixing the two conventions is the trap this module exists to remove.

One source of truth
-------------------
Both the regime map and the transition loads come from
:func:`transition_loads`, and :func:`regime` is defined as the ray through the
query point at ``lambda = 1``. Writing the four regions out as independent
inequalities instead leaves a wedge unclassified -- for ``x > 0.5`` the band
``4(1 - x) < y < 1/x`` is above the shakedown boundary, below ``y = 2`` and
inside ``x y = 1``, so a test built from ``y > 2`` and ``x y > 1`` assigns it
to nothing. ``(0.6, 1.65)`` is such a point and is ratchetting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rsdm import RSDMResult

__all__ = [
    "BREE_CORNER",
    "TransitionLoads",
    "cyclic_strip",
    "elastic_boundary",
    "plastic_strain_range_exact",
    "ratchet_rate_exact",
    "regime",
    "rsdm_strip",
    "shakedown_boundary",
    "strip_quadrature",
    "transition_loads",
]

#: Where the shakedown, alternating-plasticity and ratchetting boundaries meet.
BREE_CORNER = (0.5, 2.0)

REGIMES = ("elastic", "shakedown", "alternating", "ratchetting")


@dataclass(frozen=True)
class TransitionLoads:
    """The three load factors at which a ray changes regime.

    Each is an array shaped like the broadcast of ``x_hat`` and ``y_hat``.
    ``ratchet`` is ``inf`` for a ray with no primary stress, which never
    ratchets however hot it gets, and :attr:`alternating_is_empty` marks the
    rays that go straight from shakedown to ratchetting.
    """

    elastic: np.ndarray
    shakedown: np.ndarray
    ratchet: np.ndarray

    @property
    def alternating_is_empty(self) -> np.ndarray:
        """``lambda_sd == lambda_rr``: the ``y < 4x`` rays, which cross the
        R1 boundary and skip alternating plasticity entirely."""
        return self.ratchet <= self.shakedown * (1.0 + 1.0e-12)

    def regime_at(self, lam) -> np.ndarray:
        """Which regime each ray is in at load factor ``lam``.

        The boundaries are closed on the safe side -- a ray exactly at
        ``lambda_sd`` still shakes down -- because the transition loads are
        defined as first passage *above* a tolerance.
        """
        lam = np.asarray(lam, dtype=float)
        out = np.full(np.broadcast(lam, self.elastic).shape, "ratchetting",
                      dtype="<U11")
        out[lam <= self.ratchet] = "alternating"
        out[lam <= self.shakedown] = "shakedown"
        out[lam <= self.elastic] = "elastic"
        return out if out.ndim else out[()]


def elastic_boundary(x):
    """``y = 1 - x``: first yield at the face, clipped at zero."""
    return np.maximum(1.0 - np.asarray(x, dtype=float), 0.0)


def shakedown_boundary(x):
    """``y = min(2, 4(1 - x))``: the upper edge of the shakedown region."""
    x = np.asarray(x, dtype=float)
    return np.maximum(np.minimum(2.0, 4.0 * (1.0 - x)), 0.0)


def transition_loads(x_hat, y_hat) -> TransitionLoads:
    r"""Transition load factors for the rays ``(x, y) = lambda (x_hat, y_hat)``.

        lambda_el = 1 / (x + y)

        lambda_sd = 2/y                 if y >= 4x        (crosses y = 2)
                    1 / (x + y/4)       otherwise         (crosses y = 4(1-x))

        lambda_rr = 1 / sqrt(x y)       if y >= 4x        (crosses x y = 1)
                    lambda_sd          otherwise

    Both arguments are non-negative and are broadcast together. A ray with
    ``x_hat = y_hat = 0`` carries no load at all and every factor is ``inf``.
    """
    x = np.asarray(x_hat, dtype=float)
    y = np.asarray(y_hat, dtype=float)
    if np.any(x < 0.0) or np.any(y < 0.0):
        raise ValueError(
            "x_hat and y_hat are stresses over a yield stress and cannot be "
            "negative; a reversing thermal load is a different problem from "
            "Bree's, whose thermal stress switches on and off")
    x, y = np.broadcast_arrays(x, y)

    with np.errstate(divide="ignore", invalid="ignore"):
        lam_el = 1.0 / (x + y)
        on_r1 = y < 4.0 * x                      # leaves shakedown at x > 0.5
        lam_sd = np.where(on_r1, 1.0 / (x + 0.25 * y), 2.0 / y)
        lam_rr = np.where(on_r1, lam_sd, 1.0 / np.sqrt(x * y))

    # a ray with no thermal load never meets y = 2, and one with no primary
    # stress never meets x y = 1; both are limits of the formulae above rather
    # than special cases, but 0/0 is not
    lam_el = np.where(x + y > 0.0, lam_el, np.inf)
    lam_sd = np.where(x + y > 0.0, lam_sd, np.inf)
    lam_rr = np.where(x * y > 0.0, lam_rr, np.inf)
    lam_rr = np.where(on_r1, lam_sd, lam_rr)
    return TransitionLoads(elastic=lam_el, shakedown=lam_sd, ratchet=lam_rr)


def regime(x, y) -> np.ndarray:
    """Which regime the point ``(x, y)`` of the Bree plane is in.

    Evaluated as the ray through that point at ``lambda = 1``, which is what
    keeps it consistent with :func:`transition_loads` -- see the module
    docstring for the wedge that an independent set of inequalities misses.
    """
    return transition_loads(x, y).regime_at(1.0)


# ---------------------------------------------------------------------------
# RSDM on the same problem, for comparison against the curves above
# ---------------------------------------------------------------------------

def strip_quadrature(n: int = 101):
    """``(zeta, weight)`` through the thickness, trapezoid on ``[-1, 1]``.

    The faces must be sample points, not cell centres: every boundary in this
    module is attained at ``|zeta| = 1`` and nowhere else, so a midpoint rule
    reports a shakedown limit that is too high by the amount the outermost
    cell centre falls short of the face. Trapezoid puts a node exactly there.
    """
    if n < 3:
        raise ValueError("at least three points through the thickness")
    zeta = np.linspace(-1.0, 1.0, n)
    weight = np.full(n, 2.0 / (n - 1))
    weight[0] = weight[-1] = 1.0 / (n - 1)
    return zeta, weight


def _waveform(name: str, n_time: int) -> np.ndarray:
    """The thermal multiplier over the cycle, ``0 -> 1 -> 0``.

    Only the *primary* stress is constant in the Bree problem; the thermal
    part switches on and off, so this multiplies one of two load cases rather
    than scaling the whole history. ``CyclicProgram.proportional`` would scale
    both and is the wrong program here.
    """
    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    if name == "hump":                      # smooth, so one harmonic suffices
        return 16.0 * tau ** 2 * (1.0 - tau) ** 2
    if name == "triangle":
        return 1.0 - np.abs(2.0 * tau - 1.0)
    raise ValueError(f"unknown waveform {name!r}; use 'hump' or 'triangle'")


def rsdm_strip(x_hat, y_hat, lam: float, *, yield_stress: float = 1.0,
               modulus: float = 1.0, n_thickness: int = 101,
               n_time: int = 24, n_harmonics: int = 1,
               max_iter: int = 2000, tol: float = 1.0e-8,
               waveform: str = "hump", initial=None) -> RSDMResult:
    r"""The asymptotic cycle of the uniaxial Bree strip, by RSDM.

    The point of having this beside the closed form is that the closed form is
    the answer: unlike the thick cylinder, which pins one load factor, this
    pins a whole two-parameter map, so RSDM can be checked over a region
    rather than at a point.

    Why the strip is one generalised degree of freedom
    -------------------------------------------------
    Bree's wall is restrained against bending, so the total strain is uniform
    through the thickness and the only unknown is the membrane strain:
    ``B = 1``, and self-equilibrium of a residual field is ``INT rho dzeta =
    0`` alone. **That is not a modelling convenience, it is what reproduces
    the diagram.** Solving Melan with the Bernoulli kinematics instead --
    membrane *and* curvature, so a residual field must also carry no moment --
    lowers the shakedown factor by 25 to 35% and matches nothing:

        x, y        0.1, 0.8   0.1, 2.0   0.25, 4.0   0.4, 0.5   0.8, 1.0
        Bree         2.50000    1.00000     0.50000    1.90476    0.95238
        force only   2.50000    1.00000     0.50000    1.90476    0.95238
        + moment     1.81054    0.74575     0.37168    1.61322    0.80661

    Worth knowing before the finite element version of this problem: a model
    free to bend is a different problem and will miss the closed form by a
    quarter.

    What the iteration reduces to
    -----------------------------
    With ``B = 1`` and ``K = INT E' dzeta``, the rate equation ``K r_dot =
    INT sigma_p dzeta`` and ``rho_dot = E' r_dot - sigma_p`` collapse to

        rho_dot(zeta) = mean(sigma_p) - sigma_p(zeta)

    a thickness average, with **the modulus cancelling entirely** -- the
    residual stress history of a structure indeterminate to degree one does
    not depend on how stiff it is. ``modulus`` therefore affects only the
    plastic *strain*, through ``eps^p = sigma_p / E'``. Self-equilibrium is
    then exact by construction, since the mean of ``rho_dot`` is zero by
    inspection rather than to a solver tolerance.

    Everything else -- the Fourier series, Eqs. (13), (14) and (16), the
    convergence test -- is the same iteration as :func:`~scion.shakedown.rsdm`
    and the result is the same :class:`~scion.shakedown.RSDMResult`, carrying
    one stress component instead of six. That reuse is deliberate: the
    indicators then come out of exactly the code the finite element path uses,
    so comparing them against the closed form validates the indicators too and
    not merely this solver.

    Parameters
    ----------
    x_hat, y_hat : the ray, ``sigma_P/sigma_Y`` and the *surface*
        ``sigma_T/sigma_Y`` at ``lambda = 1``.
    lam : the load factor.
    modulus : ``E' = E/(1 - nu)``, the biaxially restrained modulus. With the
        default of one, plastic strains are reported in units of
        ``sigma_Y/E'``.
    waveform : ``'hump'`` is smooth, so one harmonic represents it well.
        ``'triangle'`` exists to check that the answer does not depend on it,
        which Melan says it must not -- only the extremes of the cycle enter.
        Measured: identical to five decimals.

    What this reproduces, and what it does not
    -----------------------------------------
    **The regime map, completely.** First-passage bisection on the indicators
    against the closed form, six rays spanning both branches:

        x, y        lambda_sd  exact   ratio   lambda_rr  exact   ratio
        0.10, 0.80   2.50040  2.50000  1.0002   3.53604  3.53553  1.0001
        0.05, 1.00   2.00032  2.00000  1.0002   4.47293  4.47214  1.0002
        0.25, 1.00   2.00003  2.00000  1.0000   2.00003  2.00000  1.0000
        0.40, 0.50   1.90479  1.90476  1.0000   1.90479  1.90476  1.0000
        0.60, 0.30   1.48150  1.48148  1.0000   1.48151  1.48148  1.0000
        0.15, 2.00   1.00016  1.00000  1.0002   1.82594  1.82574  1.0001

    **0.02% on both transition loads**, and 25 of 25 regime classifications
    exact on a coarser sweep. Re-measured independently, the figure is
    **bisection-limited rather than a property of RSDM**: the same six rays
    give a worst ratio of 1.0003 at 101 thickness points, ``n_t`` = 24 and a
    1e-5 bisection tolerance, and 1.0007 at 61 points, ``n_t`` = 16 and 1e-4.
    Every ray approaches from above, which is what a first passage against a
    tolerance must do, so quote the setting with the number -- 0.03% is the
    safe claim and 0.02% is this table's. That is a stronger statement than the thick
    cylinder's single load factor: a two-parameter map, both branches. Note
    `lambda_rr` on the ``AP`` rays comes from the ``x y = 1`` curve, the one
    boundary the Melan derivation above cannot reach -- so this validates the
    only transcribed piece of the closed form.

    **The plastic strain magnitude, not at all -- use** :func:`cyclic_strip`
    **for that.** At ``x = 0``, ``y = 3``, ``lambda = 1`` the exact range is
    ``1.0`` and :attr:`RSDMResult.plastic_strain_range` returns **0.084261**,
    where :func:`cyclic_strip` returns 1.0 to 9.3e-16. The cause is
    that the 2012 iteration drives the residual stress with ``sigma_p``
    itself, so the rate scale is a free penalty parameter rather than the
    backward-Euler ``1/dt``, and the converged state above shakedown moves
    with it: measured, the residual stress range at the face is 0.100 at
    scale 1 and 0.199 at scale 2, exactly proportional, while the iteration
    **diverges** at scale 8 and above (``max|sigma_p|`` reaching 1e153). The
    physical state needs a scale of roughly ten, which is past the stability
    limit. So ``r`` and ``A`` are usable as *indicators* -- ``r`` separates
    alternating plasticity from ratchetting by nine orders here -- and
    :attr:`RSDMResult.plastic_strain_range` is not usable as a *magnitude*.
    :func:`cyclic_strip` is the reformulation -- backward Euler with yield as a
    constraint rather than a penalty, which has no stability limit because
    there is no rate scale left to be stable in.

    A caution on what does not fix it: projecting a *corrected history*
    ``rho_c = rho + INT rho_dot`` onto the basis, rather than projecting the
    rate and dividing by ``k omega``, makes the map admit
    ``rho = -sigma^E`` as a fixed point -- self-equilibrated at zero primary
    stress, and trivially admissible since the total stress vanishes. Damped
    to stability it converges to exactly that family, with
    ``max|sigma_p|`` at 1e-12 and a residual stress range of 3.6 against the
    exact 1.0. The 2012 form is immune because ``sigma_p = 0`` forces every
    harmonic to zero, leaving only the constant term, which is what shakedown
    means.
    """
    zeta, weight = strip_quadrature(n_thickness)
    g = _waveform(waveform, n_time)
    n_points = zeta.size

    # (n_time, n_points, 1): primary constant, thermal switching on and off
    history = (lam * yield_stress
               * (float(x_hat) + float(y_hat) * zeta[None, :] * g[:, None]))
    history = history[:, :, None]

    tau = np.linspace(0.0, 1.0, n_time, endpoint=False)
    k = np.arange(1, n_harmonics + 1)
    cos = np.cos(2.0 * np.pi * np.outer(k, tau))
    sin = np.sin(2.0 * np.pi * np.outer(k, tau))
    dtau = 1.0 / n_time
    share = weight / weight.sum()                  # the thickness average

    if initial is None:
        a0 = np.zeros((n_points, 1))
        a = np.zeros((n_harmonics, n_points, 1))
        b = np.zeros((n_harmonics, n_points, 1))
    else:
        a0, a, b = (np.array(v, dtype=float, copy=True) for v in initial)

    rho = np.zeros((n_time, n_points, 1))
    plastic = np.zeros((n_time, n_points, 1))
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        previous = rho.copy()
        rho = (0.5 * a0[None] + np.einsum("kt,kpi->tpi", cos, a)
               + np.einsum("kt,kpi->tpi", sin, b))

        total = history + rho
        magnitude = np.abs(total)
        # the radial excess of a uniaxial stress is the scalar excess; the
        # closest-point projection onto [-sigma_Y, sigma_Y] is a clip
        plastic = np.where(magnitude > yield_stress,
                           total * (1.0 - yield_stress
                                    / np.maximum(magnitude, 1e-300)), 0.0)
        mean = np.einsum("p,tpi->ti", share, plastic)
        rho_dot = mean[:, None, :] - plastic

        a_old_sum = a.sum(axis=0)
        a = -(dtau / (np.pi * k))[:, None, None] * np.einsum(
            "kt,tpi->kpi", sin, rho_dot)
        b = +(dtau / (np.pi * k))[:, None, None] * np.einsum(
            "kt,tpi->kpi", cos, rho_dot)
        a0 = a0 + 2.0 * (a_old_sum - a.sum(axis=0)
                         + dtau * rho_dot.sum(axis=0))

        change = np.linalg.norm(rho - previous)
        scale_norm = max(np.linalg.norm(rho), 1e-300)
        if iteration > 1 and change <= tol * scale_norm:
            converged = True
            break

    phi = float(np.linalg.norm(a, axis=(1, 2)).sum()
                + np.linalg.norm(b, axis=(1, 2)).sum())
    return RSDMResult(a0=a0, a=a, b=b, residual=rho, plastic_stress=plastic,
                      iterations=iteration, converged=converged, phi=phi,
                      scale=yield_stress,
                      compliance=np.array([[1.0 / modulus]]),
                      weights=weight, sigma_0=yield_stress)


def plastic_strain_range_exact(y_hat, lam, *, yield_stress: float = 1.0,
                               modulus: float = 1.0):
    r"""Closed-form plastic strain range of the strip, at the face.

        Delta eps^p = max(0, lambda sigma_T - 2 sigma_Y) / E'

    Derived rather than quoted. At zero primary stress the problem is
    antisymmetric in ``zeta``, so the membrane strain vanishes and the total
    strain condition at the face reads ``sigma/E' + eps^p = -(lambda
    sigma_T/E') g(tau)``. If the face yields at both ends of the cycle the
    stress is capped at ``-sigma_Y`` when ``g = 1`` and ``+sigma_Y`` when
    ``g = 0``, so

        eps^p(g=1) - eps^p(g=0) = -(lambda sigma_T - 2 sigma_Y)/E'

    and the range is its magnitude. **The primary stress does not enter**: it
    shifts the mean stress, while the range at ``zeta`` is set by the thermal
    part alone, ``lambda sigma_T |zeta|``. That is why this is a function of
    ``y_hat`` only -- and it is a prediction worth testing rather than an
    assumption, because it should fail once the ray ratchets and the cycle
    stops closing.

    Zero at ``lambda y = 2``, the alternating-plasticity boundary, and linear
    above it.

    **Reproduced by** :func:`cyclic_strip` **to 9.3e-16 and by**
    :func:`rsdm_strip` **not at all** (0.084261 for an exact 1.0): the reason
    is in both of their docstrings, and this function was written as the target
    before either was measured against it.

    **The prediction that it fails once the ray ratchets holds.** At
    ``x, y = 0.25, 1.0`` and ``lambda = 2.5``, past ``lambda_rr``, the measured
    range is 1.127273 against this function's 0.500000 -- 2.25 times -- because
    the cycle no longer closes and the net drift enters the range. So this is a
    check on the alternating-plasticity regime and must not be applied outside
    it; :func:`ratchet_rate_exact` is the closed form on the other side.
    """
    y = np.asarray(y_hat, dtype=float)
    return np.maximum(lam * y * yield_stress - 2.0 * yield_stress, 0.0) / modulus


# ---------------------------------------------------------------------------
# the same cycle with yield enforced exactly, which is what gives the magnitude
# ---------------------------------------------------------------------------

def _strip_step(sigma_E, p_old, share, yield_stress, m, tol=1.0e-14,
                max_iter=50):
    """One backward-Euler step on the strip. Returns ``(p_new, sigma, iters)``.

    The strip has one generalised degree of freedom, so the *only* global
    coupling is through the scalar ``m = mean(p)``: with ``p = E' eps^p``
    carried in stress units,

        sigma(zeta) = sigma^E(zeta) + m - p(zeta)

    and the step is a scalar root find. Given ``m``, the trial stress is
    ``sigma~ = sigma^E + m - p_old`` and the exact closest-point projection
    onto ``[-sigma_Y, sigma_Y]`` is a clip, so the increment is the clipped-off
    part. Consistency of ``m`` then needs

        R(m) = m - mean(p_old) - mean(Delta p(m)) = 0

    whose derivative is ``1 - (plastic fraction of the thickness)`` exactly,
    because the overstress has unit slope wherever a point is plastic and zero
    slope where it is not. ``R`` is piecewise linear, so Newton with that
    derivative lands on the root in a handful of steps rather than converging
    to it.

    A fully plastic thickness makes ``R' = 0``, and that is plastic collapse
    rather than a numerical failure: the section can no longer carry the
    membrane force, which for this problem happens exactly at
    ``lambda x_hat = 1``, measured -- ``0.9999994`` runs and ``1.0000006``
    refuses.
    """
    # the root is hit exactly, so what is left is round-off in the thickness
    # average -- which scales with the plastic strain already accumulated, not
    # with the yield stress. A tolerance against sigma_Y alone fails on a
    # heavily ratchetted state, measured at |R| = 1.1e-14 against 1.0e-14.
    reference = max(yield_stress, abs(m), float(np.abs(p_old).max()))
    for it in range(1, max_iter + 1):
        trial = sigma_E + m - p_old
        over = trial - np.clip(trial, -yield_stress, yield_stress)
        residual = m - share @ (p_old + over)
        if abs(residual) <= tol * reference:
            return p_old + over, trial - over, it
        fraction = share[over != 0.0].sum()
        if fraction >= 1.0 - 1.0e-14:
            raise RuntimeError(
                "the whole thickness is plastic, so the section cannot carry "
                "the membrane force: this is plastic collapse, not a solver "
                "failure. The strip's limit load is lambda x_hat = 1.")
        m = m - residual / (1.0 - fraction)
    raise RuntimeError(
        f"the strip step did not close: |R| = {abs(residual):.3e}")


def cyclic_strip(x_hat, y_hat, lam: float, *, yield_stress: float = 1.0,
                 modulus: float = 1.0, n_thickness: int = 101,
                 n_time: int = 24, waveform: str = "hump",
                 max_cycles: int = 200, tol: float = 1.0e-10):
    r"""The asymptotic cycle of the Bree strip with yield enforced exactly.

    The counterpart of :func:`rsdm_strip` on the same problem, and the reason
    it exists is in that function's closing paragraph: the 2012 iteration's
    rate scale is a free penalty parameter, so it gives the regime map to
    0.02% and the plastic strain magnitude not at all. Here yield is a
    constraint rather than a penalty, integrated by backward Euler, and the
    magnitude comes out exact.

    Returns a :class:`~scion.shakedown.CyclicState` carrying the same
    indicator names :class:`~scion.shakedown.RSDMResult` carries.

    What it reproduces
    ------------------
    **The closed-form plastic strain range, to machine precision.** Against
    :func:`plastic_strain_range_exact`, eleven alternating-plasticity rays:

        x, y        lambda   Delta eps^p      exact      rel. error
        0.00, 3.0    1.0     1.000000000   1.000000000    0.0e+00
        0.00, 2.5    1.0     0.500000000   0.500000000    0.0e+00
        0.00, 4.0    1.0     2.000000000   2.000000000    0.0e+00
        0.10, 0.8    2.8     0.240000000   0.240000000    1.9e-15
        0.05, 1.0    3.0     1.000000000   1.000000000    4.4e-16
        0.15, 2.0    1.8     1.600000000   1.600000000    4.2e-16
        0.02, 5.0    1.0     3.000000000   3.000000000    1.5e-16

    and the worst of the eleven is **9.3e-16**, against
    :attr:`~scion.shakedown.RSDMResult.plastic_strain_range`'s **0.084261 for
    an exact 1.0** on the first row. Three cycles in every case, and the yield
    violation is 0.0e+00 in every case -- an exact zero rather than a
    tolerance, because for a scalar stress the clip *is* the projection.

    **And it is discretisation independent, which the penalty form is not.**
    ``Delta eps^p`` at ``x, y = 0.1, 0.8``, ``lambda = 3``: 0.4000000000 at
    every ``n_time`` in 8 / 12 / 16 / 24 / 32 / 48 / 64 and every
    ``n_thickness`` in 11 / 21 / 51 / 101 / 201 / 401, and identical between
    the smooth hump and the triangle. It must be: only the extremes of the
    cycle enter, which is the same statement that makes Melan's condition
    waveform-blind.

    **The ratchet rate has a closed form on the R2 branch and reproduces it.**
    Derived here from ``r = 2 (lambda^2 x y - 1) sigma_Y / E'``, which
    vanishes exactly on ``x y lambda^2 = 1`` as it must:

        lambda^2 x y  1.125   1.152   1.200   1.280   1.440   2.000   4.500
        r             0.2500  0.3040  0.4000  0.5600  0.8800  2.0000  7.0000
        2(lam^2 xy-1) 0.2500  0.3040  0.4000  0.5600  0.8800  2.0000  7.0000

    to **1.1e-6 or better** over fourteen rays spanning ``y/4x`` from 1.1 to
    40 and ``lambda^2 x y`` from 1.125 to 4.5,
    with the single exception of ``y = 4x`` exactly, which is the branch
    boundary and is excluded from :func:`ratchet_rate_exact` for that reason.
    The R1
    branch (``y < 4x``) has a different mechanism and **no closed form is
    derived here**; what is checked there is that ``r`` turns on exactly at
    ``lambda_sd = lambda_rr`` (0.0 at and below, 4.01e-4 at
    ``1.0001 lambda_sd`` on three rays) and that ``r`` equals
    ``Delta eps^p`` identically, which is the definition of one-way
    ratchetting with no reversed plasticity.

    The trap in the cycle sampling
    ------------------------------
    ``n_time`` must be **even** for the smooth hump, which peaks at
    ``tau = 1/2``. An odd count never samples ``g = 1``, so the strip is
    loaded less than asked and the answer is low: 0.9264 / 0.9506 / 0.9904 /
    0.9975 for an exact 1.0 at ``n_time`` = 9 / 11 / 25 / 49. It converges, so
    it does not look like an error -- it looks like a discretisation. Refused
    rather than rounded, because the even counts are exact at 8.

    Parameters
    ----------
    x_hat, y_hat : the ray, ``sigma_P/sigma_Y`` and the *surface*
        ``sigma_T/sigma_Y`` at ``lambda = 1``.
    lam : the load factor. ``lam * x_hat >= 1`` is plastic collapse and raises.
    modulus : ``E' = E/(1 - nu)``. With the default of one, strains are in
        units of ``sigma_Y/E'``.
    max_cycles : three is typical; the cycle is sought to ``tol`` on the
        *stress* history, which is periodic in the ratchetting regime where the
        strain is not.
    tol : relative change in the stress history between successive cycles.
        **Do not tighten below 1e-12.** A converged cycle leaves a residual
        change at the round-off floor, and that floor is set by the thermal
        gradient rather than by the solver -- measured from 4e-16 on a gentle
        ray to 4.4e-13 at ``x, y = 0.02, 5.0`` -- because a quadrature point on
        the elastic-plastic front flips its plastic flag by one ulp from cycle
        to cycle. Asking for 1e-13 makes some rays run to ``max_cycles`` and
        raise while the answer has been correct since cycle three.
    """
    from .cyclic import CyclicState          # local: cyclic imports from rsdm

    x_hat = float(x_hat)
    y_hat = float(y_hat)
    if n_time % 2:
        raise ValueError(
            f"n_time must be even: both waveforms peak at tau = 1/2, which is "
            f"a sample point only for an even count, so {n_time} never reaches "
            f"g = 1 and under-loads the strip. Measured at x, y = 0, 3 and "
            f"lambda = 1, where the exact range is 1: the hump gives 0.9264 / "
            f"0.9506 / 0.9904 at n_time = 9 / 11 / 25 and the triangle 0.7273 "
            f"at 11. It converges, so it reads as a discretisation rather than "
            f"an error, and the even counts are exact from 8 upwards.")
    zeta, weight = strip_quadrature(n_thickness)
    share = weight / weight.sum()
    g = _waveform(waveform, n_time)
    history = lam * yield_stress * (x_hat + y_hat * zeta[None, :] * g[:, None])

    p = np.zeros(zeta.size)
    m = 0.0
    stress = np.empty((n_time, zeta.size))
    plastic = np.empty((n_time, zeta.size))
    previous = None
    converged = False
    cycle = 0
    change = np.inf

    for cycle in range(1, max_cycles + 1):
        p_start = p.copy()
        for t in range(n_time):
            p, s, _ = _strip_step(history[t], p, share, yield_stress, m)
            m = share @ p
            plastic[t] = p
            stress[t] = s
        net = p - p_start
        if previous is not None:
            change = float(np.abs(stress - previous).max()) / yield_stress
        if change <= tol:
            converged = True
            break
        previous = stress.copy()
        # The stress depends on the plastic strain only through ``m - p``, so
        # subtracting a constant from the whole field changes nothing -- and
        # that is worth doing, because a ratchetting run accumulates ``r`` per
        # cycle without bound and the stress then becomes a difference of two
        # large nearly equal numbers. At ``r = 7`` per cycle the field reaches
        # 4e3 within a few hundred cycles and the step residual floors at
        # 2.4e-11, so a tolerance of 1e-13 can never be met and the run never
        # terminates. Shifting the drift out keeps the field O(1) and is exact,
        # not a regularisation; ``net`` is taken first, since it is the one
        # quantity the shift would destroy.
        drift = share @ p
        p -= drift
        m -= drift

    if not converged:
        raise RuntimeError(
            f"the strip's stress cycle was still moving after {max_cycles} "
            f"cycles, by {change:.2e} of the yield stress against a tolerance "
            f"of {tol:.0e}. **If that figure is already near 1e-13, the "
            f"tolerance is below the floor rather than the cycle being "
            f"unconverged**: a quadrature point sitting on the elastic-plastic "
            f"front flips its plastic flag by one ulp from cycle to cycle, and "
            f"the measured floor runs from 4e-16 on a gentle ray to 4.4e-13 at "
            f"x, y = 0.02, 5.0, where the thermal gradient is steepest. Use "
            f"the default 1e-10.")

    return CyclicState(
        elastic=history[:, :, None], stress=stress[:, :, None],
        plastic_strain=(plastic / modulus)[:, :, None],
        net_plastic_strain=(net / modulus)[:, None],
        effective=np.abs(stress),
        yield_stress=np.full(zeta.size, yield_stress),
        compliance=np.array([[1.0 / modulus]]), weights=weight,
        sigma_0=yield_stress, cycles=cycle, converged=True)


def ratchet_rate_exact(x_hat, y_hat, lam, *, yield_stress: float = 1.0,
                       modulus: float = 1.0):
    r"""Closed-form ratchet rate on the R2 branch, ``r = 2(lambda^2 x y - 1)``.

        r = max(0, 2 (lambda^2 x_hat y_hat - 1)) sigma_Y / E'     for y >= 4x

    **Derived here, not transcribed**, and it is the second external benchmark
    this module supplies: it vanishes identically on ``lambda^2 x y = 1``,
    which is the ``x y = 1`` boundary of the Bree diagram -- the only one the
    Melan argument in this module's docstring cannot reach -- so reproducing
    it validates that curve from the plastic strain rather than by bisecting
    an indicator.

    The derivation
    --------------
    Write ``a = lambda x_hat``, ``b = lambda y_hat`` with ``sigma_Y = E' = 1``.
    Only the two extremes of the cycle enter. At the hot extreme the converged
    state has three zones -- tension-plastic for ``zeta >= u``,
    compression-plastic for ``zeta <= v``, elastic between -- and the cold
    extreme is its mirror image in ``zeta``, which is the one structural fact
    taken from the computation rather than argued. Then:

    * the elastic core spans ``2 sigma_Y`` at the thermal gradient ``b``, since
      ``sigma`` is continuous and equal to ``+1`` at one front and ``-1`` at
      the other, so ``u - v = 2/b``;
    * the core's own contribution to the thickness-mean stress vanishes, being
      linear and antisymmetric about its midpoint, so force equilibrium
      ``mean(sigma) = a`` reduces to ``(1/2)[(1 - u) - (1 + v)] = a``, giving
      ``u + v = -2a``;
    * hence ``u = -a + 1/b``, and the front position fixes the membrane strain
      increment over a *half* cycle at ``-b u = a b - 1``;
    * the mirror symmetry makes both half cycles contribute equally, so the
      plastic strain per cycle is ``2 (a b - 1)``.

    The vanishing condition drops out of the third step: ``a b = 1`` puts
    ``u = 0``, the core's upper front exactly at mid-thickness.

    **Measured, and the two relations are checked separately because they are
    pinned by different arguments.** Over seven R2 rays with ``b`` from 2.4 to
    17.5, ``u + v`` matches ``-2a`` to machine precision -- it is an
    equilibrium statement and does not care where the front falls -- while
    ``u - v`` matches ``2/b`` to the thickness spacing, since it *is* a front
    position. :func:`cyclic_strip` then reproduces the rate itself to
    **1.1e-6 or better** over fourteen rays with ``lambda^2 x y`` from 1.125
    to 4.5 and ``y/4x`` from 1.1 to 40.

    **Refused on the R1 branch** (``y < 4x``), where ratchetting is one-way
    with no reversed plasticity and the mechanism is a different one. That
    rate is not derived here; what is checked there instead is that it turns
    on exactly at ``lambda_sd`` and that ``r = Delta eps^p`` identically.
    """
    x = np.asarray(x_hat, dtype=float)
    y = np.asarray(y_hat, dtype=float)
    if np.any(y <= 4.0 * x):
        raise NotImplementedError(
            "this closed form is the R2 branch, y > 4x, and the inequality is "
            "strict. On R1 the ratchetting mechanism is one-way, with no "
            "reversed plasticity, and its rate is not derived here. **The "
            "boundary y = 4x itself is excluded by measurement, not by "
            "caution**: that ray passes exactly through the corner where the "
            "alternating band closes, so both mechanisms are marginal, and "
            "x, y = 0.25, 1.0 at lambda = 2.2 converges to 1.00011965 times "
            "this formula over n_thickness from 101 to 25601 -- a converged "
            "disagreement, not a quadrature error. One step inside, at "
            "y = 4.4x, the ratio is 1.0000000.")
    lam = np.asarray(lam, dtype=float)
    return (np.maximum(2.0 * (lam ** 2 * x * y - 1.0), 0.0)
            * yield_stress / modulus)
