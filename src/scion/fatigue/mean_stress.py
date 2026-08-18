r"""
Mean-stress corrections.

A life curve is measured under fully reversed loading, ``R = -1``. Almost
nothing in service is fully reversed, and a tensile mean stress shortens life by
far more than its share of the peak: it holds the crack open. A correction maps
the pair ``(amplitude, mean)`` onto the equivalent fully reversed amplitude
``sigma_ar`` that would give the same life, after which the measured curve
applies unchanged.

Two families live here, and they are not interchangeable.

**Stress-life** corrections (:class:`MeanStressCorrection`) rescale the
amplitude before it reaches an S-N curve. Goodman, Gerber, Soderberg, Morrow,
Walker, SWT.

**Strain-life** corrections (:class:`StrainMeanStress`) cannot do that: the mean
stress enters the elastic term of the Coffin-Manson-Basquin equation, not the
strain amplitude, so the *curve* changes rather than its argument. Morrow,
modified Morrow, SWT. They are applied inside
:class:`~scion.fatigue.curves.StrainLife`.

Compressive mean stress
-----------------------
A compressive mean is beneficial -- it closes the crack -- but by how much is
poorly characterised, and every formula here extrapolates into that region
without evidence. Goodman predicts unbounded benefit as ``sigma_m -> -inf``;
Gerber, being a symmetric parabola, predicts that compression is as *damaging*
as tension, which is simply wrong.

The linear and parabolic corrections therefore default to
``compressive="ignore"``: for ``sigma_m < 0`` the amplitude passes through
unchanged, taking no credit. That is the standard engineering truncation and
the conservative choice. ``compressive="extrapolate"`` uses the formula as
written; if you choose it, you are asserting that your data covers negative
``R``.

:class:`Walker` and :class:`SWT` default the other way, to ``"extrapolate"``.
Their power-law form is fitted across ``R`` ratios rather than anchored to a
tensile test, negative ``R`` included, so the compressive branch is measured
rather than extrapolated -- and truncating it would discard the reason SWT is
used for variable-amplitude work in the first place.

Static failure
--------------
When the mean stress reaches the ultimate strength there is no fatigue problem
left -- the part fails on the first cycle. The corrections return ``inf`` for the
equivalent amplitude in that case, which propagates to zero life rather than to
a negative amplitude or a silent sign flip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MeanStressCorrection",
    "NoCorrection",
    "Goodman",
    "Soderberg",
    "Gerber",
    "MorrowStress",
    "Walker",
    "SWT",
    "MEAN_STRESS_REGISTRY",
    "get_mean_stress",
    "StrainLifeParams",
    "StrainMeanStress",
    "NoStrainCorrection",
    "Morrow",
    "ModifiedMorrow",
    "SmithWatsonTopper",
    "STRAIN_MEAN_STRESS_REGISTRY",
    "get_strain_mean_stress",
]

_COMPRESSIVE = ("ignore", "extrapolate")


# ===========================================================================
# Stress-life
# ===========================================================================

class MeanStressCorrection(ABC):
    """Maps ``(sigma_a, sigma_m)`` to an equivalent fully reversed amplitude."""

    name: str = "mean-stress"

    def __init__(self, compressive: str = "ignore"):
        if compressive not in _COMPRESSIVE:
            raise ValueError(f"compressive must be one of {_COMPRESSIVE}")
        self.compressive = compressive

    # ------------------------------------------------------------------
    @abstractmethod
    def _corrected(self, amplitude: np.ndarray,
                   mean: np.ndarray) -> np.ndarray:
        """The formula itself, evaluated everywhere."""

    def equivalent_amplitude(self, amplitude, mean=0.0) -> np.ndarray:
        """``sigma_ar``: the ``R = -1`` amplitude of equal life, ``(n,)``.

        ``inf`` where the mean stress alone exceeds the static strength.
        """
        a = np.asarray(amplitude, dtype=float)
        m = np.broadcast_to(np.asarray(mean, dtype=float), a.shape)
        out = self._corrected(a, m)
        if self.compressive == "ignore":
            out = np.where(m < 0.0, a, out)
        return out

    def __call__(self, amplitude, mean=0.0) -> np.ndarray:
        return self.equivalent_amplitude(amplitude, mean)

    # ------------------------------------------------------------------
    @staticmethod
    def _divide(amplitude: np.ndarray, denom: np.ndarray) -> np.ndarray:
        """``a / denom`` with a non-positive denominator meaning static failure."""
        ok = denom > 0.0
        return np.where(ok, amplitude / np.where(ok, denom, 1.0), np.inf)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(compressive='{self.compressive}')"


class NoCorrection(MeanStressCorrection):
    """Mean stress ignored. Correct only for ``R = -1`` loading."""

    name = "none"

    def _corrected(self, amplitude, mean):
        return amplitude


class Goodman(MeanStressCorrection):
    r"""``sigma_a / sigma_ar + sigma_m / sigma_u = 1``.

    The straight line to the ultimate strength. Conservative for ductile metals
    -- most data sit above it -- and the default in most design codes for that
    reason.
    """

    name = "goodman"

    def __init__(self, sigma_u: float, compressive: str = "ignore"):
        super().__init__(compressive)
        self.sigma_u = float(sigma_u)

    def _corrected(self, amplitude, mean):
        return self._divide(amplitude, 1.0 - mean / self.sigma_u)


class Soderberg(MeanStressCorrection):
    r"""Goodman with the yield strength in place of the ultimate.

    Guarantees no yielding as well as no fatigue failure, and is correspondingly
    over-conservative as a life estimate. Use it as a design rule, not as a
    prediction to compare with test data.
    """

    name = "soderberg"

    def __init__(self, sigma_y: float, compressive: str = "ignore"):
        super().__init__(compressive)
        self.sigma_y = float(sigma_y)

    def _corrected(self, amplitude, mean):
        return self._divide(amplitude, 1.0 - mean / self.sigma_y)


class Gerber(MeanStressCorrection):
    r"""``sigma_a / sigma_ar + (sigma_m / sigma_u)^2 = 1``.

    The parabola through the same endpoints as Goodman. It fits ductile-metal
    data better in the tensile-mean region and is unsafe as a design curve
    precisely because it sits above the data's lower bound.

    Being even in ``sigma_m`` it cannot distinguish tension from compression, so
    with ``compressive="extrapolate"`` it claims a compressive mean is as
    damaging as a tensile one. That is not a subtlety of the model, it is a
    defect of using a parabola, and it is why the default truncates.
    """

    name = "gerber"

    def __init__(self, sigma_u: float, compressive: str = "ignore"):
        super().__init__(compressive)
        self.sigma_u = float(sigma_u)

    def _corrected(self, amplitude, mean):
        return self._divide(amplitude, 1.0 - (mean / self.sigma_u) ** 2)


class MorrowStress(MeanStressCorrection):
    r"""Goodman with the true fracture strength ``sigma_f'`` instead of ``sigma_u``.

    Since ``sigma_f' > sigma_u``, this is less conservative than Goodman, and it
    is the consistent choice when the S-N curve is a Basquin fit, because the
    same ``sigma_f'`` is then the curve's own intercept -- the correction and the
    curve share a parameter rather than mixing a fatigue constant with a tensile
    test.
    """

    name = "morrow"

    def __init__(self, sigma_f: float, compressive: str = "ignore"):
        super().__init__(compressive)
        self.sigma_f = float(sigma_f)

    def _corrected(self, amplitude, mean):
        return self._divide(amplitude, 1.0 - mean / self.sigma_f)


class Walker(MeanStressCorrection):
    r"""``sigma_ar = sigma_max^(1 - gamma) sigma_a^gamma``.

    The best of the family when data at several ``R`` ratios exist, because
    ``gamma`` is fitted rather than assumed: ``gamma = 1`` means no mean-stress
    sensitivity at all, ``gamma = 0.5`` reduces exactly to :class:`SWT`, and
    lower ``gamma`` means more sensitivity. For steels it typically falls
    between 0.4 and 0.8, decreasing with strength.

    A cycle whose maximum stress is non-positive never opens a crack, so it is
    reported as non-damaging (``sigma_ar = 0``) rather than raised to a
    fractional power of a negative number.

    Unlike the linear corrections this one defaults to
    ``compressive="extrapolate"``; see the module docstring for why the two
    families differ.
    """

    name = "walker"

    def __init__(self, gamma: float = 0.5,
                 compressive: str = "extrapolate"):
        super().__init__(compressive)
        self.gamma = float(gamma)

    def _corrected(self, amplitude, mean):
        s_max = mean + amplitude
        ok = (s_max > 0.0) & (amplitude > 0.0)
        return np.where(ok,
                        np.where(ok, s_max, 1.0) ** (1.0 - self.gamma)
                        * np.where(ok, amplitude, 1.0) ** self.gamma,
                        0.0)


class SWT(Walker):
    r"""Smith-Watson-Topper: ``sigma_ar = sqrt(sigma_max sigma_a)``.

    Walker with ``gamma = 1/2``, and the only member of the family with no free
    parameter, which is why it is the usual default when no ``R``-ratio data
    exist. It is the stress-life form; the strain-life form is
    :class:`SmithWatsonTopper` and is a different equation, not this one applied
    to strains.
    """

    name = "swt"

    def __init__(self, compressive: str = "extrapolate"):
        super().__init__(gamma=0.5, compressive=compressive)


# ---------------------------------------------------------------------------

MEAN_STRESS_REGISTRY: dict[str, type[MeanStressCorrection]] = {
    "none": NoCorrection,
    "goodman": Goodman,
    "soderberg": Soderberg,
    "gerber": Gerber,
    "morrow": MorrowStress,
    "walker": Walker,
    "swt": SWT,
    "smith-watson-topper": SWT,
}


def get_mean_stress(name: str, **kwargs) -> MeanStressCorrection:
    key = name.lower().replace("_", "-")
    if key not in MEAN_STRESS_REGISTRY:
        raise NotImplementedError(
            f"Unknown mean-stress correction '{name}'. Available: "
            f"{sorted(set(MEAN_STRESS_REGISTRY))}")
    return MEAN_STRESS_REGISTRY[key](**kwargs)


# ===========================================================================
# Strain-life
# ===========================================================================

@dataclass(frozen=True)
class StrainLifeParams:
    """The four Coffin-Manson-Basquin constants and the elastic modulus.

    ``b`` and ``c`` are defined against **reversals** ``2N_f``, which is how
    they are published. Quoting them against cycles instead shifts the life by
    ``2^(1/b)`` -- a factor of thousands at typical ``b`` -- and is the single
    most common way to misuse a strain-life data table.
    """

    sigma_f: float        # fatigue strength coefficient
    b: float              # fatigue strength exponent (negative)
    eps_f: float          # fatigue ductility coefficient
    c: float              # fatigue ductility exponent (negative)
    E: float


class StrainMeanStress(ABC):
    """A mean-stress variant of the strain-life equation.

    Each variant is stated as a damage parameter ``P`` and a curve ``P(2N)``:

        :meth:`parameter`   built from the cycle -- strain amplitude, mean
                            stress, maximum stress
        :meth:`curve`       the right-hand side, a function of reversals

    The life follows from ``curve(2N) = P``, solved numerically because no
    variant is invertible in closed form. Writing all of them this way means the
    root solve, its bracketing and its edge cases live in one place instead of
    three.
    """

    name: str = "strain-mean-stress"

    #: True when :meth:`parameter` reads ``sigma_max``. The mean alone is not
    #: enough for SWT, and the maximum cannot be recovered from the strain
    #: amplitude once the material has yielded, so the caller must supply it.
    needs_max_stress: bool = False

    #: True when the variant reads the stress history at all.
    needs_stress: bool = True

    @abstractmethod
    def parameter(self, eps_a: np.ndarray, sigma_m: np.ndarray,
                  sigma_max: np.ndarray, p: StrainLifeParams) -> np.ndarray:
        """Damage parameter of the counted cycle, ``(n,)``."""

    @abstractmethod
    def curve(self, two_n: np.ndarray, sigma_m: np.ndarray,
              p: StrainLifeParams) -> np.ndarray:
        """The parameter's value at ``two_n`` reversals, ``(n,)``.

        Must be non-increasing in ``two_n`` for the bracketed solve to be
        valid. Implementations that can violate that (Morrow with a mean stress
        above ``sigma_f'``) must report it through :meth:`invalid`.
        """

    def invalid(self, sigma_m: np.ndarray,
                p: StrainLifeParams) -> np.ndarray:
        """Cycles for which the variant has no admissible solution, ``(n,)`` bool.

        Default: none.
        """
        return np.zeros(np.shape(sigma_m), dtype=bool)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NoStrainCorrection(StrainMeanStress):
    r"""``eps_a = (sigma_f'/E)(2N)^b + eps_f' (2N)^c``.

    The Coffin-Manson-Basquin equation as measured, under fully reversed strain.
    """

    name = "none"
    needs_stress = False

    def parameter(self, eps_a, sigma_m, sigma_max, p):
        return eps_a

    def curve(self, two_n, sigma_m, p):
        return (p.sigma_f / p.E) * two_n ** p.b + p.eps_f * two_n ** p.c


class Morrow(StrainMeanStress):
    r"""``eps_a = ((sigma_f' - sigma_m)/E)(2N)^b + eps_f' (2N)^c``.

    The mean stress is subtracted from the elastic term only, on the argument
    that it relaxes away in the plastic regime. That is also its limitation: at
    long lives, where the elastic term dominates, it behaves sensibly; at short
    lives it predicts almost no mean-stress effect, and short-life data show one.

    A mean stress above ``sigma_f'`` makes the elastic term negative and the
    curve non-monotone, at which point the equation has no meaningful root.
    Those cycles are reported through :meth:`invalid` and assigned zero life
    rather than whatever a solver would return.
    """

    name = "morrow"

    def parameter(self, eps_a, sigma_m, sigma_max, p):
        return eps_a

    def curve(self, two_n, sigma_m, p):
        return ((p.sigma_f - sigma_m) / p.E) * two_n ** p.b \
            + p.eps_f * two_n ** p.c

    def invalid(self, sigma_m, p):
        return np.asarray(sigma_m, dtype=float) >= p.sigma_f


class ModifiedMorrow(StrainMeanStress):
    r"""Morrow with the plastic term scaled too.

    ``eps_a = ((sigma_f' - sigma_m)/E)(2N)^b
              + eps_f' ((sigma_f' - sigma_m)/sigma_f')^(c/b) (2N)^c``

    Scaling both terms by the same factor keeps the elastic-plastic ratio -- and
    therefore the transition life -- independent of the mean stress, which the
    plain Morrow form does not. Fits low-cycle data better; the two forms
    coincide as the plastic term dies out.
    """

    name = "modified-morrow"

    def parameter(self, eps_a, sigma_m, sigma_max, p):
        return eps_a

    def curve(self, two_n, sigma_m, p):
        ratio = np.maximum((p.sigma_f - sigma_m) / p.sigma_f, 0.0)
        return (ratio * p.sigma_f / p.E) * two_n ** p.b \
            + p.eps_f * ratio ** (p.c / p.b) * two_n ** p.c

    def invalid(self, sigma_m, p):
        return np.asarray(sigma_m, dtype=float) >= p.sigma_f


class SmithWatsonTopper(StrainMeanStress):
    r"""``sigma_max eps_a = (sigma_f'^2/E)(2N)^(2b) + sigma_f' eps_f' (2N)^(b+c)``.

    A damage parameter rather than a correction: the product of the maximum
    stress and the strain amplitude, obtained by multiplying the strain-life
    equation by the stress-life one. It needs no extra constant, it handles both
    regimes, and it is the usual default for variable-amplitude strain-life.

    It says a cycle with ``sigma_max <= 0`` does no damage at all. That is a
    real prediction, not a numerical guard: a loop entirely in compression
    cannot open a crack. It is also known to be too severe for large compressive
    means, where the parameter goes to zero but the measured life does not go to
    infinity.
    """

    name = "swt"
    needs_max_stress = True

    def parameter(self, eps_a, sigma_m, sigma_max, p):
        return np.maximum(sigma_max, 0.0) * eps_a

    def curve(self, two_n, sigma_m, p):
        return (p.sigma_f ** 2 / p.E) * two_n ** (2.0 * p.b) \
            + p.sigma_f * p.eps_f * two_n ** (p.b + p.c)


# ---------------------------------------------------------------------------

STRAIN_MEAN_STRESS_REGISTRY: dict[str, type[StrainMeanStress]] = {
    "none": NoStrainCorrection,
    "morrow": Morrow,
    "modified-morrow": ModifiedMorrow,
    "swt": SmithWatsonTopper,
    "smith-watson-topper": SmithWatsonTopper,
}


def get_strain_mean_stress(name: str, **kwargs) -> StrainMeanStress:
    key = name.lower().replace("_", "-")
    if key not in STRAIN_MEAN_STRESS_REGISTRY:
        raise NotImplementedError(
            f"Unknown strain-life mean-stress variant '{name}'. Available: "
            f"{sorted(set(STRAIN_MEAN_STRESS_REGISTRY))}. The stress-life "
            f"corrections (Goodman, Gerber, ...) rescale an amplitude and "
            f"cannot be applied to a strain-life curve; see get_mean_stress.")
    return STRAIN_MEAN_STRESS_REGISTRY[key](**kwargs)
