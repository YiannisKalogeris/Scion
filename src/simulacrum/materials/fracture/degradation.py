r"""
Degradation functions.

Axis 2. ``g(d)`` multiplies the degradable part of the elastic energy, taking
the material from intact (``g(0) = 1``) to fully broken (``g(1) = k``, a small
residual stiffness that keeps the stiffness matrix non-singular).

Two properties matter beyond the endpoints:

* ``g'(1) = 0``, so the driving force vanishes once the material is broken and
  the crack stops widening rather than smearing.
* the shape between the endpoints sets how much elastic energy is released
  before damage localises, and therefore the peak load. Quadratic degradation
  couples stiffness loss to damage from the very first increment, which is why
  AT2 has no elastic threshold; the cubic and rational families delay it.

The convention here is that ``g`` is applied to the *positive* (degradable)
energy only -- which part that is, is the split's business, not this module's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Degradation",
    "Quadratic",
    "Cubic",
    "Rational",
    "DEGRADATION_REGISTRY",
    "get_degradation",
]


class Degradation(ABC):
    """g(d) with its first two derivatives."""

    name: str = "degradation"

    #: True when g is a polynomial of degree <= 2 in d, so that the
    #: phase-field equation is linear and one Newton step is exact
    quadratic_in_d: bool = False

    def __init__(self, residual: float = 1.0e-9):
        self.residual = float(residual)

    @abstractmethod
    def g(self, d: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def dg(self, d: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def ddg(self, d: np.ndarray) -> np.ndarray: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(residual={self.residual:g})"


class Quadratic(Degradation):
    r"""g(d) = (1 - d)^2 + k.

    The standard choice (Bourdin, Miehe). Together with AT2 it makes the
    phase-field equation linear in ``d``, which is what allows a single linear
    solve per staggered iteration.
    """

    name = "quadratic"
    quadratic_in_d = True

    def g(self, d):
        return (1.0 - d) ** 2 + self.residual

    def dg(self, d):
        return -2.0 * (1.0 - d)

    def ddg(self, d):
        return np.full_like(np.asarray(d, dtype=float), 2.0)


class Cubic(Degradation):
    r"""g(d) = s[(1-d)^3 - (1-d)^2] + 3(1-d)^2 - 2(1-d)^3 + k  (Borden et al.)

    Nearly flat near ``d = 0``, so stiffness is preserved until damage is well
    under way. This produces a more linear pre-peak load-displacement response
    and a sharper peak than quadratic degradation -- closer to a Griffith-type
    response -- at the price of a phase-field equation that is now cubic in
    ``d`` and needs Newton iterations.

    ``s`` controls the initial slope; small positive values (1e-4) are usual.
    """

    name = "cubic"
    quadratic_in_d = False

    def __init__(self, residual: float = 1.0e-9, s: float = 1.0e-4):
        super().__init__(residual)
        self.s = float(s)

    def g(self, d):
        x = 1.0 - np.asarray(d, dtype=float)
        return (self.s - 2.0) * x ** 3 + (3.0 - self.s) * x ** 2 + self.residual

    def dg(self, d):
        x = 1.0 - np.asarray(d, dtype=float)
        return -(3.0 * (self.s - 2.0) * x ** 2 + 2.0 * (3.0 - self.s) * x)

    def ddg(self, d):
        x = 1.0 - np.asarray(d, dtype=float)
        return 6.0 * (self.s - 2.0) * x + 2.0 * (3.0 - self.s)


class Rational(Degradation):
    r"""g(d) = (1-d)^2 / ((1-d)^2 + m d (1 + p d)) + k

    The Lorentz / Wu family. ``m`` is set from the material length and strength
    rather than chosen freely,

        m = 3 G_c / (4 l sigma_c^2 / E)   (one common calibration)

    which is what lets this family reproduce a prescribed peak stress
    independently of ``l`` -- the property AT2 lacks. ``p`` shapes the softening
    branch (p = -0.5 gives a Griffith-like tail, larger p a longer one).
    """

    name = "rational"
    quadratic_in_d = False

    def __init__(self, residual: float = 1.0e-9, m: float = 1.0,
                 p: float = -0.5):
        super().__init__(residual)
        self.m = float(m)
        self.p = float(p)

    def _parts(self, d):
        d = np.asarray(d, dtype=float)
        x = 1.0 - d
        num = x ** 2
        den = num + self.m * d * (1.0 + self.p * d)
        return d, x, num, den

    def g(self, d):
        _, _, num, den = self._parts(d)
        return num / den + self.residual

    def dg(self, d):
        d, x, num, den = self._parts(d)
        dnum = -2.0 * x
        dden = dnum + self.m * (1.0 + 2.0 * self.p * d)
        return (dnum * den - num * dden) / den ** 2

    def ddg(self, d):
        d, x, num, den = self._parts(d)
        dnum, ddnum = -2.0 * x, 2.0
        dden = dnum + self.m * (1.0 + 2.0 * self.p * d)
        ddden = ddnum + 2.0 * self.m * self.p
        return ((ddnum * den - num * ddden) / den ** 2
                - 2.0 * dden * (dnum * den - num * dden) / den ** 3)


DEGRADATION_REGISTRY: dict[str, type[Degradation]] = {
    "quadratic": Quadratic,
    "cubic": Cubic,
    "rational": Rational,
    "lorentz": Rational,
}


def get_degradation(name: str, **kwargs) -> Degradation:
    key = name.lower()
    if key not in DEGRADATION_REGISTRY:
        raise NotImplementedError(
            f"Unknown degradation function '{name}'. Available: "
            f"{sorted(set(DEGRADATION_REGISTRY))}"
        )
    return DEGRADATION_REGISTRY[key](**kwargs)
