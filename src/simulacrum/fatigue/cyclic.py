r"""
The cyclic stress-strain curve, and getting local plasticity out of an elastic
analysis.

Strain-life needs the *local* elastic-plastic strain at the crack initiation
site. An elastoplastic analysis supplies it directly. An elastic analysis does
not: at a notch it returns a stress that exceeds yield, which the material never
saw, paired with a strain that is too small.

The classical repair is to estimate the real pair from the elastic one plus a
cyclic stress-strain curve. :func:`neuber` and :func:`glinka` do that. They are
cheap -- one scalar solve per point -- and they turn a single linear analysis
into a fatigue calculation that is usually within a factor of two on life, which
is the accuracy of the input data anyway.

The cyclic curve, not the monotonic one
---------------------------------------
Metals do not keep their tensile-test stress-strain curve under cycling. Annealed
alloys harden, cold-worked ones soften, and both settle after a few hundred
cycles onto a stable *cyclic* curve that can differ from the monotonic one by
tens of percent in either direction. Using the monotonic curve here is a common
and invisible error: the arithmetic works, the numbers look reasonable, and the
answer is wrong by whatever the two curves differ by.
:meth:`RambergOsgood.from_strain_life` builds the cyclic curve from the same
four constants as the strain-life curve, which is the one construction
guaranteed to be consistent with it.

Massing's hypothesis
--------------------
The hysteresis loop is the cyclic curve scaled by two in both axes. That single
statement is why this module needs only one solver: a *range* correction is
exactly twice the corresponding *amplitude* correction, so ``neuber_range`` is a
two-line wrapper rather than a second derivation. It is also an approximation --
real loops are not similar to the cyclic curve under large mean strains -- and
it is the reason the corrections below should not be pushed far past general
yielding.

Which rule
----------
Neuber conserves ``sigma eps``; Glinka conserves strain energy density. Neuber
over-estimates the local strain, Glinka under-estimates it, and measurements
generally fall between, closer to Glinka in plane strain and to Neuber in plane
stress. Running both brackets the answer, which is more useful than choosing
one and quoting it to three digits.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RambergOsgood", "neuber", "neuber_range", "glinka", "glinka_range"]

_ITERATIONS = 100


def _bisect(f, lo: np.ndarray, hi: np.ndarray,
            iterations: int = _ITERATIONS) -> np.ndarray:
    """Root of an increasing ``f`` bracketed by ``[lo, hi]``, batched.

    Bisection rather than Newton for the same reason as the strain-life solve:
    the plastic term's derivative diverges as the stress goes to zero for the
    usual ``n < 1``, so a Newton step from a small iterate is unbounded, while
    bisection cannot fail on a bracket that is guaranteed by construction.
    """
    lo = np.array(lo, dtype=float, copy=True)
    hi = np.array(hi, dtype=float, copy=True)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        low = f(mid) < 0.0
        lo = np.where(low, mid, lo)
        hi = np.where(low, hi, mid)
    return 0.5 * (lo + hi)


class RambergOsgood:
    r"""``eps = sigma/E + (sigma/K)^(1/n)``, the cyclic curve.

    ``K`` is the cyclic strength coefficient and ``n`` the cyclic strain
    hardening exponent, typically 0.1 to 0.2. Both are written unprimed here;
    the literature primes them (``K'``, ``n'``) to mark them as cyclic, and
    every value this class should be given is a primed one.

    Odd-symmetric: the curve for compression is the reflection of the curve for
    tension, which is what makes :meth:`hysteresis_strain_range` a scaling of
    it rather than a separate branch.
    """

    def __init__(self, E: float, K: float, n: float):
        if not 0.0 < n <= 1.0:
            raise ValueError("the cyclic hardening exponent n must lie in "
                             "(0, 1]; published values are near 0.15")
        self.E = float(E)
        self.K = float(K)
        self.n = float(n)

    # ------------------------------------------------------------------
    @classmethod
    def from_strain_life(cls, curve) -> "RambergOsgood":
        r"""The cyclic curve implied by a :class:`
        ~simulacrum.fatigue.curves.StrainLife` fit.

            ``n' = b / c``,   ``K' = sigma_f' / eps_f'^(n')``

        These follow from requiring the cyclic curve and the strain-life curve
        to describe the same loops: the plastic strain amplitude of a cycle must
        come out the same whether it is read off one or the other. They are
        "compatibility relations", and a material data sheet that quotes
        ``K'`` and ``n'`` independently will not satisfy them exactly. When they
        disagree, the four strain-life constants are the ones to trust -- they
        are fitted to lives, which is what is being predicted.
        """
        p = curve.params
        n = p.b / p.c
        K = p.sigma_f / p.eps_f ** n
        return cls(E=p.E, K=float(K), n=float(n))

    # ------------------------------------------------------------------
    def strain(self, sigma) -> np.ndarray:
        """Total strain at a stress, ``(...)``, odd in ``sigma``."""
        s = np.asarray(sigma, dtype=float)
        a = np.abs(s)
        return np.sign(s) * (a / self.E + (a / self.K) ** (1.0 / self.n))

    def dstrain_dsigma(self, sigma) -> np.ndarray:
        """``deps/dsigma``, ``(...)``. Diverges as ``sigma -> 0`` for ``n < 1``.

        The elastic-plastic tangent modulus is its reciprocal. The divergence at
        the origin is a property of the power law, not a numerical artefact, and
        it is why nothing here uses a Newton iteration started from zero.
        """
        a = np.abs(np.asarray(sigma, dtype=float))
        with np.errstate(divide="ignore"):
            plastic = (1.0 / (self.n * self.K)) \
                * (a / self.K) ** (1.0 / self.n - 1.0)
        return 1.0 / self.E + plastic

    def stress(self, strain) -> np.ndarray:
        """Invert the curve, ``(...)``.

        Bracketed by ``[0, E|eps|]``: the plastic term is non-negative, so the
        elastic line is always an upper bound on the stress at a given strain.
        """
        e = np.asarray(strain, dtype=float)
        a = np.abs(e)
        s = _bisect(lambda x: self.strain(x) - a, np.zeros_like(a), self.E * a)
        return np.sign(e) * s

    # ------------------------------------------------------------------
    def hysteresis_strain_range(self, delta_sigma) -> np.ndarray:
        r"""Massing: ``d_eps = d_sigma/E + 2 (d_sigma / 2K)^(1/n)``.

        The loop branch, obtained by doubling the cyclic curve. Equivalently
        ``2 * strain(delta_sigma / 2)``, which is the identity the range-form
        notch rules rely on.
        """
        d = np.abs(np.asarray(delta_sigma, dtype=float))
        return 2.0 * self.strain(0.5 * d)

    def energy_density(self, sigma) -> np.ndarray:
        r"""``W = int_0^eps sigma d_eps``, the area under the curve, ``(...)``.

            ``W = sigma^2/(2E) + sigma^(1/n + 1) / (K^(1/n) (1 + n))``

        Glinka's rule conserves this quantity.
        """
        a = np.abs(np.asarray(sigma, dtype=float))
        return (a ** 2 / (2.0 * self.E)
                + a ** (1.0 / self.n + 1.0)
                / (self.K ** (1.0 / self.n) * (1.0 + self.n)))

    def __repr__(self) -> str:
        return f"RambergOsgood(E={self.E:g}, K={self.K:g}, n={self.n:g})"


# ---------------------------------------------------------------------------
# Notch rules
# ---------------------------------------------------------------------------

def neuber(sigma_elastic, ro: RambergOsgood, kt: float = 1.0):
    r"""Neuber's rule: ``sigma eps = sigma_e^2 / E``.

    Returns ``(sigma, eps)``, the local elastic-plastic pair whose product
    matches that of the elastic solution.

    ``sigma_elastic`` is the stress a *linear* analysis reports at the point --
    already containing the geometric stress concentration, so ``kt`` stays at
    one. Pass ``kt`` only when working from a nominal stress that does not,
    in which case the elastic stress is ``kt * S``.

    Neuber over-predicts the local strain, increasingly so as plasticity
    spreads, and the resulting life is conservative. Beyond general yielding
    neither this nor :func:`glinka` means anything: both assume the plastic zone
    is contained by surrounding elastic material, and once it is not, the
    elastic solution it is built from has no relation to the real one.
    """
    se = np.asarray(sigma_elastic, dtype=float) * float(kt)
    a = np.abs(se)
    target = a ** 2 / ro.E
    s = _bisect(lambda x: x * ro.strain(x) - target, np.zeros_like(a), a)
    return np.sign(se) * s, np.sign(se) * ro.strain(s)


def neuber_range(delta_sigma_elastic, ro: RambergOsgood, kt: float = 1.0):
    """Neuber applied to a range, under Massing.

    Returns ``(delta_sigma, delta_eps)``. Exactly twice :func:`neuber` at half
    the input: doubling both axes multiplies ``sigma eps`` by four and
    ``sigma_e^2/E`` by four as well, so the doubled solution is the solution.
    That identity is Massing's hypothesis and nothing else, which is why there
    is no second solver here.
    """
    d = np.asarray(delta_sigma_elastic, dtype=float)
    s, e = neuber(0.5 * d, ro, kt)
    return 2.0 * s, 2.0 * e


def glinka(sigma_elastic, ro: RambergOsgood, kt: float = 1.0):
    r"""Glinka's rule: equal strain energy density, ``W(sigma) = sigma_e^2/(2E)``.

    Returns ``(sigma, eps)``. Derived by assuming the energy density at the
    notch root is unchanged by local yielding, which holds while the plastic
    zone is small and surrounded by elastic material.

    Predicts less local strain than :func:`neuber` and so a longer life. It is
    the better estimate under plane strain, where the constraint suppresses
    notch-root straining; Neuber is closer under plane stress. Running both is
    the honest report.
    """
    se = np.asarray(sigma_elastic, dtype=float) * float(kt)
    a = np.abs(se)
    target = a ** 2 / (2.0 * ro.E)
    s = _bisect(lambda x: ro.energy_density(x) - target, np.zeros_like(a), a)
    return np.sign(se) * s, np.sign(se) * ro.strain(s)


def glinka_range(delta_sigma_elastic, ro: RambergOsgood, kt: float = 1.0):
    """Glinka applied to a range, under Massing. See :func:`neuber_range`."""
    d = np.asarray(delta_sigma_elastic, dtype=float)
    s, e = glinka(0.5 * d, ro, kt)
    return 2.0 * s, 2.0 * e
