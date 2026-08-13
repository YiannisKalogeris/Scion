r"""
Dissipation density profiles (regularization).

Axis 3. The regularized fracture energy is written once, in a form that covers
the whole AT family:

    E_d = (G_c / c_alpha) * INT [ alpha(d) / l  +  l |grad d|^2 ] dx,
    c_alpha = 4 INT_0^1 sqrt(alpha(s)) ds

so a new profile is just a new ``alpha``. The normalisation ``c_alpha`` is what
makes ``G_c`` the actual fracture toughness rather than an arbitrary constant --
get it wrong and every crack propagates at the wrong load.

    AT2:  alpha(d) = d^2,  c_alpha = 2
    AT1:  alpha(d) = d,    c_alpha = 8/3

The difference that matters operationally
-----------------------------------------
AT2 has ``alpha'(0) = 0``: damage grows from the very first load increment, so
there is no purely elastic stage and no well-defined strength -- the peak stress
depends on ``l``. Convenient numerically (the phase-field equation is linear and
homogeneous), physically awkward.

AT1 has ``alpha'(0) = 1``, a constant source term. That is exactly what creates
an elastic threshold and a finite strength. It also means ``d`` is driven
negative wherever the driving energy is below the threshold, so the phase-field
solve becomes a **bound-constrained** problem, ``d >= 0``. That is not optional:
without the constraint AT1 produces negative damage and the results are
meaningless.

:attr:`Dissipation.needs_bound_constraint` is the flag the staggered solver
dispatches on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Dissipation",
    "AT1",
    "AT2",
    "DISSIPATION_REGISTRY",
    "get_dissipation",
]


class Dissipation(ABC):
    """alpha(d), its derivatives, and the normalisation c_alpha."""

    name: str = "dissipation"

    #: 4 * INT_0^1 sqrt(alpha(s)) ds
    c_alpha: float = 1.0

    #: True when alpha'(0) != 0, i.e. d wants to go negative in intact material
    needs_bound_constraint: bool = False

    #: True when alpha is a polynomial of degree <= 2
    quadratic_in_d: bool = True

    @abstractmethod
    def alpha(self, d: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def dalpha(self, d: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def ddalpha(self, d: np.ndarray) -> np.ndarray: ...

    # ------------------------------------------------------------------
    def gradient_coefficient(self, Gc: float, ell: float) -> float:
        """Coefficient of ``grad d . grad w`` in the weak form."""
        return 2.0 * Gc * ell / self.c_alpha

    def local_coefficient(self, Gc: float, ell: float) -> float:
        """Coefficient multiplying ``alpha'(d)`` in the weak form."""
        return Gc / (self.c_alpha * ell)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class AT2(Dissipation):
    r"""alpha(d) = d^2. The Bourdin / Miehe standard.

    No elastic threshold: damage starts immediately, so the peak stress scales
    with ``l`` and is a numerical parameter rather than a material property.
    In exchange, the phase-field equation is linear with a zero source at
    ``d = 0``, so ``d >= 0`` holds automatically and one linear solve per
    staggered iteration is exact.
    """

    name = "AT2"
    c_alpha = 2.0
    needs_bound_constraint = False
    quadratic_in_d = True

    def alpha(self, d):
        return np.asarray(d, dtype=float) ** 2

    def dalpha(self, d):
        return 2.0 * np.asarray(d, dtype=float)

    def ddalpha(self, d):
        return np.full_like(np.asarray(d, dtype=float), 2.0)


class AT1(Dissipation):
    r"""alpha(d) = d. Pham, Amor, Marigo & Maurini.

    Gives a genuine elastic stage: damage nucleates only once the driving
    energy exceeds ``3 G_c / (16 l)``, and the resulting strength

        sigma_c = sqrt(3 E G_c / (8 l))

    is a real material property. Also produces a crack profile with *compact
    support* -- ``d`` reaches exactly zero at a finite distance rather than
    decaying exponentially -- which is closer to a sharp crack.

    Requires ``d >= 0`` to be enforced. See the module docstring.
    """

    name = "AT1"
    c_alpha = 8.0 / 3.0
    needs_bound_constraint = True
    quadratic_in_d = True

    def alpha(self, d):
        return np.asarray(d, dtype=float)

    def dalpha(self, d):
        return np.ones_like(np.asarray(d, dtype=float))

    def ddalpha(self, d):
        return np.zeros_like(np.asarray(d, dtype=float))

    # ------------------------------------------------------------------
    @staticmethod
    def critical_stress(E: float, Gc: float, ell: float) -> float:
        """sigma_c = sqrt(3 E Gc / (8 l)) -- the strength AT1 implies."""
        return float(np.sqrt(3.0 * E * Gc / (8.0 * ell)))

    @staticmethod
    def length_for_strength(E: float, Gc: float, sigma_c: float) -> float:
        """Invert the above: the ``l`` that reproduces a measured strength."""
        return float(3.0 * E * Gc / (8.0 * sigma_c ** 2))


DISSIPATION_REGISTRY: dict[str, type[Dissipation]] = {
    "at1": AT1,
    "at2": AT2,
}


def get_dissipation(name: str) -> Dissipation:
    key = name.lower()
    if key not in DISSIPATION_REGISTRY:
        raise NotImplementedError(
            f"Unknown dissipation profile '{name}'. Available: "
            f"{sorted(DISSIPATION_REGISTRY)}"
        )
    return DISSIPATION_REGISTRY[key]()
