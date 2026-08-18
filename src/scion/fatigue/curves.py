r"""
Life curves: how many cycles of a given amplitude the material survives.

Two families, one interface.

**Stress-life** (:class:`StressLife`) is the high-cycle regime. Nominal stresses
stay elastic, plasticity is confined to microscopic slip, and life is a power
law in the stress amplitude -- Basquin. Above about 10^4 cycles this is the
right model, and below it the stress amplitude is not even a well-defined
quantity, because the material is ratchetting or shaking down and the stress no
longer follows the load.

**Strain-life** (:class:`StrainLife`) is the low-cycle regime, and covers the
high-cycle one as well. The controlled quantity is the local strain, which
remains meaningful under plastic flow, and life follows the
Coffin-Manson-Basquin sum of an elastic and a plastic term. Its elastic term
*is* Basquin, so the two families agree at long life by construction; they are
not two fits of the same data.

Amplitude, not range
--------------------
Every curve here takes an **amplitude**. Rainflow counting produces **ranges**.
The conversion lives in :attr:`~scion.fatigue.rainflow.CycleCounts.amplitudes`
and nowhere else. Code-defined curves such as EN 1993-1-9, whose detail
categories are published as stress *ranges*, halve them in their constructor and
say so.

Cycles, not reversals
---------------------
Every curve here returns **cycles**. Basquin and Coffin-Manson are published in
**reversals** ``2N_f``, and the exponents ``b`` and ``c`` mean nothing without
that convention: reading a published ``b`` as if it applied to cycles moves the
life by ``2^(1/b)``, which at ``b = -0.09`` is a factor of 2000. The factor of
two is applied once, inside each curve.

Runout
------
``cycles`` returns ``inf`` for an amplitude that never fails and ``0`` for one
that fails on the first application. Both propagate correctly through Miner:
``n/inf = 0`` and ``n/0 = inf``. Neither is clamped, because a life of zero is
information -- it says the load exceeded a static limit and the fatigue
calculation is the wrong question.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .mean_stress import (
    MeanStressCorrection,
    NoCorrection,
    NoStrainCorrection,
    StrainLifeParams,
    StrainMeanStress,
    get_mean_stress,
    get_strain_mean_stress,
)

__all__ = [
    "LifeCurve",
    "StressLife",
    "Basquin",
    "TwoSlopeSN",
    "CoffinManson",
    "StrainLife",
    "CURVE_REGISTRY",
    "get_curve",
]


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class LifeCurve(ABC):
    """Cycles to failure as a function of the amplitude of one cycle."""

    name: str = "life-curve"

    #: Which history channel the cycles must be counted on -- ``"stress"``,
    #: ``"strain"`` or ``"plastic-strain"``. The assessment driver reads this to
    #: decide what to rainflow, and counting the wrong channel is not a small
    #: error: under plastic flow the stress history is nearly flat while the
    #: strain history is not.
    variable: str = "stress"

    #: True when the curve needs the stress history *in addition* to the
    #: counted channel, because its mean-stress treatment does.
    needs_stress: bool = False

    @abstractmethod
    def cycles(self, amplitude, mean_stress=0.0,
               max_stress=None) -> np.ndarray:
        """Cycles to failure, ``(n,)``. ``inf`` for runout, ``0`` for immediate
        failure."""

    @abstractmethod
    def amplitude(self, cycles) -> np.ndarray:
        """The fully reversed amplitude that fails at ``cycles``. For plotting
        the curve and for round-trip verification of :meth:`cycles`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class StressLife(LifeCurve):
    """S-N curve plus a mean-stress correction.

    The two are orthogonal: the correction maps ``(sigma_a, sigma_m)`` onto an
    equivalent fully reversed amplitude, and the curve is then evaluated on
    that. Subclasses implement only the fully reversed curve, in both
    directions.
    """

    variable = "stress"

    def __init__(self, mean_stress: str | MeanStressCorrection | None = None,
                 mean_stress_kwargs: dict | None = None):
        self.correction = (
            get_mean_stress(mean_stress, **(mean_stress_kwargs or {}))
            if isinstance(mean_stress, str)
            else (mean_stress or NoCorrection()))

    # ------------------------------------------------------------------
    @abstractmethod
    def _cycles_reversed(self, amplitude: np.ndarray) -> np.ndarray:
        """Life at a fully reversed amplitude."""

    @abstractmethod
    def amplitude(self, cycles) -> np.ndarray:
        ...

    # ------------------------------------------------------------------
    def cycles(self, amplitude, mean_stress=0.0, max_stress=None):
        a = np.asarray(amplitude, dtype=float)
        ar = self.correction.equivalent_amplitude(a, mean_stress)
        return self._cycles_reversed(ar)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mean_stress={self.correction.name})"


# ---------------------------------------------------------------------------
# Stress-life curves
# ---------------------------------------------------------------------------

class Basquin(StressLife):
    r"""``sigma_a = sigma_f' (2 N_f)^b``, inverted to

        ``N_f = 0.5 (sigma_a / sigma_f')^(1/b)``.

    A straight line on log-log axes, which is what an S-N curve is over the
    range where it has been measured. Outside that range it is an
    extrapolation: with no ``endurance_limit`` it predicts failure at any
    amplitude however small, given enough cycles, and for a steel that is wrong
    by an unbounded factor.

    Parameters
    ----------
    sigma_f : fatigue strength coefficient, the amplitude at one reversal.
    b : fatigue strength exponent, negative, typically -0.05 to -0.12.
    endurance_limit : amplitude below which life is infinite. ``None`` extends
        the power law forever, which is right for aluminium and wrong for steel.
    """

    name = "basquin"

    def __init__(self, sigma_f: float, b: float,
                 endurance_limit: float | None = None,
                 mean_stress: str | MeanStressCorrection | None = None,
                 mean_stress_kwargs: dict | None = None):
        super().__init__(mean_stress, mean_stress_kwargs)
        if b >= 0.0:
            raise ValueError("the fatigue strength exponent b must be "
                             "negative; life decreases with amplitude")
        self.sigma_f = float(sigma_f)
        self.b = float(b)
        self.endurance_limit = (None if endurance_limit is None
                                else float(endurance_limit))

    # -- constructors ---------------------------------------------------
    @classmethod
    def from_points(cls, point_1: tuple[float, float],
                    point_2: tuple[float, float], **kw) -> "Basquin":
        """Fit through two ``(amplitude, cycles)`` points.

        The usual way an S-N curve arrives: two points read off a published
        plot. Both are *amplitudes* and *cycles*; the reversal conversion is
        applied here.
        """
        (a1, n1), (a2, n2) = point_1, point_2
        if n1 == n2:
            raise ValueError("the two points must have different lives")
        b = np.log(a1 / a2) / np.log((2.0 * n1) / (2.0 * n2))
        sigma_f = a1 * (2.0 * n1) ** (-b)
        return cls(sigma_f=float(sigma_f), b=float(b), **kw)

    @classmethod
    def from_slope(cls, m: float, amplitude_ref: float,
                   cycles_ref: float, **kw) -> "Basquin":
        """From the code form ``N sigma_a^m = const`` anchored at one point.

        ``m`` is the positive log-log slope used in design codes and relates to
        Basquin's exponent by ``b = -1/m``. Note ``m`` is the slope in
        *cycles*: the reversal factor cancels in a ratio of lives, so no factor
        of two appears here, unlike everywhere else in this module.
        """
        if m <= 0.0:
            raise ValueError("the code slope m must be positive")
        b = -1.0 / float(m)
        sigma_f = amplitude_ref * (2.0 * cycles_ref) ** (-b)
        return cls(sigma_f=float(sigma_f), b=b, **kw)

    @classmethod
    def estimate_steel(cls, sigma_u: float, **kw) -> "Basquin":
        """The textbook estimate for wrought steel: ``0.9 sigma_u`` at 10^3
        cycles, ``0.5 sigma_u`` at 10^6, endurance limit at the latter.

        An estimate, not a measurement. It exists so that a parameter study can
        start before the coupon data arrive, and it carries a scatter of
        roughly a factor of three in life. Replace it with
        :meth:`from_points` on real data before quoting a number.
        """
        c = cls.from_points((0.9 * sigma_u, 1.0e3), (0.5 * sigma_u, 1.0e6),
                            **kw)
        c.endurance_limit = 0.5 * float(sigma_u)
        return c

    # ------------------------------------------------------------------
    def _cycles_reversed(self, amplitude):
        a = np.asarray(amplitude, dtype=float)
        live = (a > 0.0) & np.isfinite(a)
        if self.endurance_limit is not None:
            # the limit itself belongs to the finite branch: the curve reaches
            # it at a definite life, and excluding it makes cycles() fail to
            # invert amplitude() at exactly the knee
            live &= a >= self.endurance_limit
        ratio = np.where(live, a, 1.0) / self.sigma_f
        out = np.where(live, 0.5 * ratio ** (1.0 / self.b), np.inf)
        return np.where(np.isinf(a), 0.0, out)

    def amplitude(self, cycles):
        n = np.asarray(cycles, dtype=float)
        return self.sigma_f * (2.0 * n) ** self.b

    @property
    def slope(self) -> float:
        """The code slope ``m = -1/b``."""
        return -1.0 / self.b

    def __repr__(self) -> str:
        return (f"Basquin(sigma_f={self.sigma_f:g}, b={self.b:g}, "
                f"m={self.slope:.3g}, "
                f"endurance={self.endurance_limit}, "
                f"mean_stress={self.correction.name})")


class TwoSlopeSN(StressLife):
    r"""A bilinear log-log S-N curve with a knee, and a cut-off below it.

    Design codes and the Haibach modification are the same curve with different
    numbers in it:

        N = N_k (a_k / a)^m1     for a >= a_k     (below the knee in life)
        N = N_k (a_k / a)^m2     for a <  a_k
        N = inf                  for a <  a_cut

    Why the second slope exists
    ---------------------------
    Under constant amplitude, cycles below the fatigue limit do no damage. Under
    *variable* amplitude they do: the larger cycles grow a crack, and once it is
    growing, cycles that could never have started one still extend it. A curve
    with a flat cut-off at the knee cannot express that, and Miner's sum then
    misses damage that dominates a long-tailed load spectrum.

    Haibach's answer is to continue past the knee with a shallower slope
    ``m2 = 2 m1 - 1``. EN 1993-1-9 takes ``m2 = 5`` against ``m1 = 3`` -- the
    same value -- and adds a true cut-off five decades further out.
    """

    name = "two-slope"

    def __init__(self, amplitude_knee: float, cycles_knee: float,
                 m1: float = 3.0, m2: float = 5.0,
                 cutoff_amplitude: float | None = None,
                 mean_stress: str | MeanStressCorrection | None = None,
                 mean_stress_kwargs: dict | None = None):
        super().__init__(mean_stress, mean_stress_kwargs)
        if m1 <= 0.0 or m2 <= 0.0:
            raise ValueError("both slopes must be positive")
        if m2 < m1:
            raise ValueError(
                "the slope beyond the knee must be shallower than before it "
                "(m2 >= m1); a steeper one would make small cycles more "
                "damaging than large ones")
        self.a_k = float(amplitude_knee)
        self.N_k = float(cycles_knee)
        self.m1, self.m2 = float(m1), float(m2)
        self.cutoff_amplitude = (None if cutoff_amplitude is None
                                 else float(cutoff_amplitude))

    # -- constructors ---------------------------------------------------
    @classmethod
    def eurocode3(cls, detail_category: float, **kw) -> "TwoSlopeSN":
        r"""EN 1993-1-9 curve for a detail category ``delta_sigma_C``.

        The category is the stress **range** at 2x10^6 cycles, in MPa. The
        constant-amplitude fatigue limit sits at 5x10^6 and the cut-off at
        10^8:

            delta_sigma_D = (2/5)^(1/3) delta_sigma_C     m = 3 above
            delta_sigma_L = (5/100)^(1/5) delta_sigma_D   m = 5 between

        Everything is halved on the way in, because this package works in
        amplitudes throughout. Comparing the resulting ``a_k`` with a published
        ``delta_sigma_D`` will therefore show a factor of two, and that is
        correct.
        """
        d_c = float(detail_category)
        d_d = (2.0 / 5.0) ** (1.0 / 3.0) * d_c
        d_l = (5.0 / 100.0) ** (1.0 / 5.0) * d_d
        return cls(amplitude_knee=0.5 * d_d, cycles_knee=5.0e6,
                   m1=3.0, m2=5.0, cutoff_amplitude=0.5 * d_l, **kw)

    @classmethod
    def haibach(cls, amplitude_knee: float, cycles_knee: float,
                m: float = 3.0, **kw) -> "TwoSlopeSN":
        """Haibach's extension of a single-slope curve past its knee, with
        ``m2 = 2m - 1`` and no cut-off."""
        return cls(amplitude_knee, cycles_knee, m1=m, m2=2.0 * m - 1.0,
                   cutoff_amplitude=None, **kw)

    @classmethod
    def from_basquin(cls, curve: Basquin, cycles_knee: float,
                     **kw) -> "TwoSlopeSN":
        """Haibach-extend an existing Basquin fit at a chosen knee."""
        return cls.haibach(float(curve.amplitude(cycles_knee)), cycles_knee,
                           m=curve.slope, **kw)

    # ------------------------------------------------------------------
    def _cycles_reversed(self, amplitude):
        a = np.asarray(amplitude, dtype=float)
        live = (a > 0.0) & np.isfinite(a)
        ratio = self.a_k / np.where(live, a, 1.0)
        m = np.where(a >= self.a_k, self.m1, self.m2)
        out = np.where(live, self.N_k * ratio ** m, np.inf)
        if self.cutoff_amplitude is not None:
            out = np.where(a < self.cutoff_amplitude, np.inf, out)
        return np.where(np.isinf(a), 0.0, out)

    def amplitude(self, cycles):
        n = np.asarray(cycles, dtype=float)
        m = np.where(n <= self.N_k, self.m1, self.m2)
        return self.a_k * (self.N_k / n) ** (1.0 / m)

    def __repr__(self) -> str:
        return (f"TwoSlopeSN(a_k={self.a_k:g} at N={self.N_k:g}, "
                f"m1={self.m1:g}, m2={self.m2:g}, "
                f"cutoff={self.cutoff_amplitude}, "
                f"mean_stress={self.correction.name})")


# ---------------------------------------------------------------------------
# Strain-life
# ---------------------------------------------------------------------------

def _solve_reversals(rhs, target: np.ndarray, invalid: np.ndarray,
                     max_reversals: float, iterations: int = 80) -> np.ndarray:
    """Solve ``rhs(2N) = target`` for ``2N``, batched.

    Bisection in ``log10(2N)`` over ``[1, max_reversals]``. Not Newton: the
    strain-life equation is a sum of two power laws with very different
    exponents, so its derivative spans decades and a Newton step from a poor
    start overshoots into negative reversals. The function is monotone
    decreasing, which makes bisection unconditionally convergent and its cost --
    80 vectorised function evaluations for the whole mesh -- irrelevant next to
    the finite element solve that produced the history.

    Returns ``1.0`` reversal where the target is above the curve at one reversal
    (the caller turns that into zero life) and ``inf`` where it is below the
    curve at ``max_reversals`` (runout).
    """
    t = np.asarray(target, dtype=float)
    lo = np.zeros(t.shape)                             # log10(1 reversal)
    hi = np.full(t.shape, np.log10(max_reversals))

    below = rhs(np.full(t.shape, max_reversals)) >= t  # runout
    above = rhs(np.ones(t.shape)) <= t                 # fails immediately

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        # monotone decreasing: too much parameter left means go further right
        go_right = rhs(10.0 ** mid) > t
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)

    two_n = 10.0 ** (0.5 * (lo + hi))
    two_n = np.where(below, np.inf, two_n)
    two_n = np.where(above, 1.0, two_n)
    return np.where(invalid, 1.0, two_n)


class CoffinManson(LifeCurve):
    r"""The plastic branch alone: ``eps_pa = eps_f' (2 N_f)^c``.

    Coffin and Manson's original result, and still the right model when the
    plastic strain amplitude is what was measured -- a strain-controlled test
    reports the width of the hysteresis loop directly. Invertible in closed
    form, unlike the total-strain equation.

    It has no elastic term, so it predicts failure at any plastic strain however
    small and says nothing at all about a cycle that is elastic. Below the
    transition life of :class:`StrainLife` it is the whole story; above it, it
    is not a fatigue model. Use it when the *plastic* strain is the input, and
    :class:`StrainLife` when the total strain is.

    ``variable`` is ``"plastic-strain"`` rather than ``"strain"`` so that an
    assessment cannot quietly feed it a total strain history; that substitution
    over-predicts damage without producing any symptom, since both are strains
    of plausible magnitude. Mean stress is not treated at all -- the classical
    form has no term for it.
    """

    name = "coffin-manson"
    variable = "plastic-strain"

    def __init__(self, eps_f: float, c: float):
        if c >= 0.0:
            raise ValueError("the fatigue ductility exponent c must be "
                             "negative")
        self.eps_f = float(eps_f)
        self.c = float(c)

    def cycles(self, amplitude, mean_stress=0.0, max_stress=None):
        a = np.asarray(amplitude, dtype=float)
        live = a > 0.0
        out = np.where(live, np.where(live, a, 1.0) / self.eps_f, 1.0)
        return np.where(live, 0.5 * out ** (1.0 / self.c), np.inf)

    def amplitude(self, cycles):
        n = np.asarray(cycles, dtype=float)
        return self.eps_f * (2.0 * n) ** self.c

    def __repr__(self) -> str:
        return f"CoffinManson(eps_f={self.eps_f:g}, c={self.c:g})"


class StrainLife(LifeCurve):
    r"""Coffin-Manson-Basquin:

        ``eps_a = (sigma_f'/E)(2N)^b + eps_f' (2N)^c``

    Elastic term plus plastic term, each a straight line on log-log axes,
    crossing at the **transition life**. Below it the plastic term dominates and
    ductility governs; above it the elastic term dominates and strength governs.
    That crossover is why a material selected for high-cycle fatigue is often
    the wrong choice for low-cycle fatigue: the two regimes reward opposite
    properties, and the curves of a hard and a soft steel cross.

    The elastic term is Basquin, in exactly the same constants -- see
    :meth:`basquin`.

    Mean stress changes the equation, not its argument, so it is a property of
    the curve here rather than a correction applied before it: pass
    ``mean_stress="morrow"``, ``"modified-morrow"`` or ``"swt"``. See
    :mod:`~scion.fatigue.mean_stress`.

    No endurance limit
    ------------------
    The equation is a sum of two power laws and never becomes flat, so it
    predicts a finite life at any strain amplitude. Real metals have a limit,
    and extrapolating this curve to 10^10 cycles is arithmetic rather than
    prediction. ``max_reversals`` therefore caps the reported life and anything
    beyond it is returned as ``inf`` -- visible as runout rather than as a
    confident number with twelve digits.
    """

    name = "strain-life"
    variable = "strain"

    def __init__(self, E: float, sigma_f: float, b: float,
                 eps_f: float, c: float,
                 mean_stress: str | StrainMeanStress | None = None,
                 max_reversals: float = 1.0e12):
        if b >= 0.0 or c >= 0.0:
            raise ValueError("b and c must both be negative")
        if c >= b:
            raise ValueError(
                "the plastic exponent c must be steeper than the elastic one "
                "(c < b); otherwise the plastic term outlives the elastic one "
                "and there is no transition life")
        self.params = StrainLifeParams(sigma_f=float(sigma_f), b=float(b),
                                       eps_f=float(eps_f), c=float(c),
                                       E=float(E))
        self.correction = (
            get_strain_mean_stress(mean_stress) if isinstance(mean_stress, str)
            else (mean_stress or NoStrainCorrection()))
        self.max_reversals = float(max_reversals)
        self.needs_stress = self.correction.needs_stress

    # -- shorthands -----------------------------------------------------
    @property
    def E(self) -> float:
        return self.params.E

    @property
    def sigma_f(self) -> float:
        return self.params.sigma_f

    @property
    def b(self) -> float:
        return self.params.b

    @property
    def eps_f(self) -> float:
        return self.params.eps_f

    @property
    def c(self) -> float:
        return self.params.c

    # -- constructors ---------------------------------------------------
    @classmethod
    def uniform_material_law(cls, sigma_u: float, E: float,
                             **kw) -> "StrainLife":
        r"""Baeumel and Seeger's estimate for unalloyed and low-alloy steel:

            sigma_f' = 1.50 sigma_u,   b = -0.087
            eps_f'   = 0.59 psi,       c = -0.58

        with ``psi = 1`` for ``sigma_u/E <= 0.003`` and ``1.375 - 125 sigma_u/E``
        above it.

        An estimate from the tensile strength alone, for when no strain-life
        data exist. It cannot distinguish two steels of equal strength and
        different ductility, which is precisely the distinction that decides
        low-cycle life, so treat the low-cycle end as indicative only.
        """
        ratio = float(sigma_u) / float(E)
        psi = 1.0 if ratio <= 0.003 else 1.375 - 125.0 * ratio
        return cls(E=E, sigma_f=1.50 * float(sigma_u), b=-0.087,
                   eps_f=0.59 * psi, c=-0.58, **kw)

    # ------------------------------------------------------------------
    def amplitude(self, cycles):
        """Total strain amplitude at a given life, fully reversed."""
        n = np.asarray(cycles, dtype=float)
        return NoStrainCorrection().curve(2.0 * n, 0.0, self.params)

    def elastic_amplitude(self, cycles):
        n = np.asarray(cycles, dtype=float)
        return (self.sigma_f / self.E) * (2.0 * n) ** self.b

    def plastic_amplitude(self, cycles):
        n = np.asarray(cycles, dtype=float)
        return self.eps_f * (2.0 * n) ** self.c

    def stress_amplitude(self, cycles):
        """The companion Basquin stress amplitude, ``sigma_f' (2N)^b``."""
        return self.E * self.elastic_amplitude(cycles)

    def basquin(self, **kw) -> Basquin:
        """The elastic term as a stand-alone S-N curve.

        Identical constants, so a stress-life and a strain-life assessment of
        the same high-cycle history agree exactly. If they do not, the
        discrepancy is in the counting or the mean-stress treatment, not in the
        material data -- which is what makes this a useful cross-check.
        """
        return Basquin(sigma_f=self.sigma_f, b=self.b, **kw)

    def transition_life(self) -> float:
        """Cycles at which the elastic and plastic terms are equal.

        ``2N_t = (eps_f' E / sigma_f')^(1/(b - c))``.
        """
        two_nt = (self.eps_f * self.E / self.sigma_f) ** (
            1.0 / (self.b - self.c))
        return 0.5 * float(two_nt)

    # ------------------------------------------------------------------
    def cycles(self, amplitude, mean_stress=0.0, max_stress=None):
        """Cycles to failure for a strain amplitude and its stress state.

        ``mean_stress`` and ``max_stress`` come from the **stress** history at
        the same reversal points as the counted strain cycle; SWT needs
        ``max_stress`` and raises rather than guessing it, because the maximum
        stress of a loop is not recoverable from the strain amplitude once the
        material has yielded.
        """
        eps_a = np.asarray(amplitude, dtype=float)
        s_m = np.broadcast_to(np.asarray(mean_stress, dtype=float),
                              eps_a.shape)
        if max_stress is None:
            if self.correction.needs_max_stress:
                raise ValueError(
                    f"{self.correction.name} needs the maximum stress of each "
                    f"loop; pass max_stress. It cannot be inferred from the "
                    f"strain amplitude once the material has yielded.")
            s_max = s_m
        else:
            s_max = np.broadcast_to(np.asarray(max_stress, dtype=float),
                                    eps_a.shape)

        target = self.correction.parameter(eps_a, s_m, s_max, self.params)
        bad = self.correction.invalid(s_m, self.params)

        two_n = _solve_reversals(
            lambda t: self.correction.curve(t, s_m, self.params),
            target, bad, self.max_reversals)

        life = 0.5 * two_n
        # one reversal back from the solver means the bracket bottomed out:
        # the cycle is beyond the curve entirely
        return np.where(np.isfinite(two_n) & (two_n <= 1.0 + 1e-12),
                        0.0, life)

    def __repr__(self) -> str:
        return (f"StrainLife(E={self.E:g}, sigma_f={self.sigma_f:g}, "
                f"b={self.b:g}, eps_f={self.eps_f:g}, c={self.c:g}, "
                f"mean_stress={self.correction.name}, "
                f"N_t={self.transition_life():.3g})")


# ---------------------------------------------------------------------------

CURVE_REGISTRY: dict[str, type[LifeCurve]] = {
    "basquin": Basquin,
    "sn": Basquin,
    "s-n": Basquin,
    "two-slope": TwoSlopeSN,
    "haibach": TwoSlopeSN,
    "coffin-manson": CoffinManson,
    "strain-life": StrainLife,
    "en": StrainLife,
    "e-n": StrainLife,
    "mcb": StrainLife,
}


def get_curve(name: str, **kwargs) -> LifeCurve:
    key = name.lower().replace("_", "-")
    if key not in CURVE_REGISTRY:
        raise NotImplementedError(
            f"Unknown life curve '{name}'. Available: "
            f"{sorted(set(CURVE_REGISTRY))}")
    return CURVE_REGISTRY[key](**kwargs)
