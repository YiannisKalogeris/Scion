r"""
Hardening laws.

Two structurally different mechanisms live
here, and they are not interchangeable:

**Isotropic hardening** grows the elastic domain uniformly. It supplies the
scalar force ``kappa(alpha)`` that the yield criterion consumes -- the yield
stress for von Mises, the cohesion for Drucker-Prager. Which of those it means
is the criterion's business, not this module's, which is why the quantity is
called ``kappa`` throughout rather than ``sigma_y``.

**Kinematic hardening** translates the elastic domain without growing it,
through a back stress ``beta`` that shifts the argument of the yield function
to ``sigma - beta``. This is what produces the Bauschinger effect: after
loading in tension, reverse yielding occurs *below* the original yield stress.
A purely isotropic model predicts the opposite, so any cyclic or reversed-load
problem needs a kinematic component.

Both are optional and combine: ``PerfectPlasticity`` plus ``NoKinematic`` is
non-hardening plasticity, and any mixture of the two is legitimate.

Alpha
-----
``alpha`` is the equivalent plastic strain, defined so that under associative
von Mises flow -- sqrt(2/3 · dε^p : dε^p)-- it accumulates at the rate of the plastic multiplier. It is
monotone non-decreasing by construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "IsotropicHardening",
    "PerfectPlasticity",
    "LinearIsotropic",
    "Voce",
    "Swift",
    "Ludwik",
    "KinematicHardening",
    "NoKinematic",
    "LinearKinematic",
    "ArmstrongFrederick",
    "ISOTROPIC_REGISTRY",
    "KINEMATIC_REGISTRY",
    "get_isotropic",
    "get_kinematic",
]


# ===========================================================================
# Isotropic
# ===========================================================================

class IsotropicHardening(ABC):
    """Scalar hardening force ``kappa(alpha)`` and its slope."""

    name: str = "isotropic"

    #: True when ``kappa`` is linear in ``alpha``, so the consistency condition
    #: of a von Mises return map has a closed-form solution and no local Newton
    #: iteration is required.
    linear: bool = False

    @abstractmethod
    def kappa(self, alpha: np.ndarray) -> np.ndarray:
        """Current hardening force, ``(n,)``."""

    @abstractmethod
    def dkappa_dalpha(self, alpha: np.ndarray) -> np.ndarray:
        """Hardening modulus ``H = dkappa/dalpha``, ``(n,)``."""

    # ------------------------------------------------------------------
    def check(self, alpha_max: float = 1.0, n: int = 256) -> dict:
        """Report properties that affect solvability.

        Softening (``H < 0``) is not rejected -- it is physically real -- but
        it makes the boundary value problem lose ellipticity, so the solution
        localises into a band whose width is set by the element size rather
        than by the material. Results are then mesh-dependent, and a
        regularisation (gradient, viscous or nonlocal) is required for them to
        mean anything.
        """
        a = np.linspace(0.0, alpha_max, n)
        H = self.dkappa_dalpha(a)
        k = self.kappa(a)
        return {
            "kappa_0": float(k[0]),
            "kappa_max": float(k[-1]),
            "H_min": float(H.min()),
            "H_max": float(H.max()),
            "softening": bool((H < 0).any()),
            "saturates": bool(abs(H[-1]) < 1e-6 * max(abs(H[0]), 1.0)),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class PerfectPlasticity(IsotropicHardening):
    """``kappa = kappa_0``. The elastic domain never changes size.

    Once yielding begins the stress cannot rise, so a load-controlled problem
    has no solution beyond the limit load and must be run under displacement
    control.
    """

    name = "perfect"
    linear = True

    def __init__(self, kappa_0: float):
        self.kappa_0 = float(kappa_0)

    def kappa(self, alpha):
        return np.full(np.shape(alpha), self.kappa_0)

    def dkappa_dalpha(self, alpha):
        return np.zeros(np.shape(alpha))


class LinearIsotropic(IsotropicHardening):
    """``kappa = kappa_0 + H alpha``.

    The one case where a von Mises return map is a single division rather than
    a scalar Newton solve, which is why it remains the default for verification
    problems.
    """

    name = "linear"
    linear = True

    def __init__(self, kappa_0: float, H: float):
        self.kappa_0 = float(kappa_0)
        self.H = float(H)

    def kappa(self, alpha):
        return self.kappa_0 + self.H * np.asarray(alpha, dtype=float)

    def dkappa_dalpha(self, alpha):
        return np.full(np.shape(alpha), self.H)


class Voce(IsotropicHardening):
    r"""``kappa = kappa_0 + (kappa_inf - kappa_0)(1 - exp(-delta alpha))``.

    Saturating exponential hardening, the usual fit for annealed metals. The
    modulus decays from ``delta (kappa_inf - kappa_0)`` to zero, so the
    material becomes asymptotically perfectly plastic -- which means a return
    map that divides by ``H`` must not, and a consistency solve must not assume
    ``H`` is bounded away from zero.
    """

    name = "voce"
    linear = False

    def __init__(self, kappa_0: float, kappa_inf: float, delta: float):
        self.kappa_0 = float(kappa_0)
        self.kappa_inf = float(kappa_inf)
        self.delta = float(delta)

    def kappa(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return self.kappa_0 + (self.kappa_inf - self.kappa_0) \
            * (1.0 - np.exp(-self.delta * a))

    def dkappa_dalpha(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return (self.kappa_inf - self.kappa_0) * self.delta \
            * np.exp(-self.delta * a)


class Swift(IsotropicHardening):
    r"""``kappa = K (alpha_0 + alpha)^n``.

    Power-law hardening with an offset, which is what makes it usable: the
    modulus at zero plastic strain is ``K n alpha_0^(n-1)``, finite for any
    ``alpha_0 > 0``. See :class:`Ludwik` for what happens without the offset.

    ``alpha_0`` is fixed by the initial yield stress:
    ``alpha_0 = (kappa_0 / K)^(1/n)``.
    """

    name = "swift"
    linear = False

    def __init__(self, K: float, alpha_0: float, n: float):
        if alpha_0 <= 0.0:
            raise ValueError("Swift requires alpha_0 > 0; use Ludwik for the "
                             "unshifted power law and read its caveat")
        self.K = float(K)
        self.alpha_0 = float(alpha_0)
        self.n = float(n)

    @classmethod
    def from_yield(cls, kappa_0: float, K: float, n: float) -> "Swift":
        """Pick ``alpha_0`` so that ``kappa(0) = kappa_0``."""
        return cls(K=K, alpha_0=(kappa_0 / K) ** (1.0 / n), n=n)

    def kappa(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return self.K * (self.alpha_0 + a) ** self.n

    def dkappa_dalpha(self, alpha):
        a = np.asarray(alpha, dtype=float)
        return self.K * self.n * (self.alpha_0 + a) ** (self.n - 1.0)


class Ludwik(IsotropicHardening):
    r"""``kappa = kappa_0 + K alpha^n``.

    Widely quoted and numerically awkward. For the usual ``n < 1`` the modulus

        ``H = K n alpha^(n-1)``

    diverges as ``alpha -> 0``, which is exactly the state every quadrature
    point is in at first yield. A return map that uses ``H`` in a denominator
    will take a huge first step or fail outright.

    The modulus is therefore floored at ``alpha = alpha_floor``. That is a
    numerical device, not the model: if the floor is visible in your results,
    use :class:`Swift`, which is the same curve shape with the singularity
    removed analytically.
    """

    name = "ludwik"
    linear = False

    def __init__(self, kappa_0: float, K: float, n: float,
                 alpha_floor: float = 1.0e-8):
        self.kappa_0 = float(kappa_0)
        self.K = float(K)
        self.n = float(n)
        self.alpha_floor = float(alpha_floor)

    def kappa(self, alpha):
        a = np.maximum(np.asarray(alpha, dtype=float), 0.0)
        return self.kappa_0 + self.K * a ** self.n

    def dkappa_dalpha(self, alpha):
        a = np.maximum(np.asarray(alpha, dtype=float), self.alpha_floor)
        return self.K * self.n * a ** (self.n - 1.0)


# ===========================================================================
# Kinematic
# ===========================================================================

class KinematicHardening(ABC):
    """Evolution of the back stress ``beta``.

    The yield function is evaluated on the *relative* stress ``sigma - beta``,
    so ``beta`` translates the elastic domain rather than resizing it.

    ``beta`` is deviatoric for pressure-insensitive criteria; the increments
    returned here are deviatoric whenever ``n`` is, which holds for associative
    von Mises flow.
    """

    name: str = "kinematic"

    #: True when ``dbeta`` is linear in ``dalpha`` at fixed direction, so the
    #: update is explicit. Armstrong-Frederick is not.
    linear: bool = False

    @abstractmethod
    def update(self, beta: np.ndarray, n: np.ndarray,
               dalpha: np.ndarray) -> np.ndarray:
        """Return the new back stress.

        ``beta`` ``(m, 3, 3)``, ``n`` the flow direction ``(m, 3, 3)``,
        ``dalpha`` the equivalent plastic strain increment ``(m,)``.
        """

    @abstractmethod
    def dbeta_ddalpha(self, beta: np.ndarray, n: np.ndarray,
                      dalpha: np.ndarray) -> np.ndarray:
        """``d beta_new / d dalpha`` at fixed ``n``, ``(m, 3, 3)``.

        Needed by the consistent tangent; omitting it costs quadratic
        convergence in exactly the way :mod:`simulacrum.solvers.tangent_check`
        is built to detect.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class NoKinematic(KinematicHardening):
    """No back stress. Purely isotropic hardening."""

    name = "none"
    linear = True

    def update(self, beta, n, dalpha):
        return np.zeros_like(beta)

    def dbeta_ddalpha(self, beta, n, dalpha):
        return np.zeros_like(beta)


class LinearKinematic(KinematicHardening):
    r"""Prager's rule: ``dbeta = (2/3) C n dalpha``.

    The ``2/3`` is the von Mises convention. It is chosen so that the
    *equivalent* back stress grows as

        ``q(beta) = sqrt(3/2 beta:beta) = C alpha``,

    which makes ``C`` directly comparable with an isotropic modulus ``H``
    measured from the same test. Note the component along the flow direction is
    smaller: for uniaxial loading ``beta_11 = (2/3) C alpha``, since
    ``q(n) = 3/2`` for the von Mises normal. Comparing ``beta_11`` with ``H``
    instead is a factor-of-1.5 calibration error.

    Produces an exact Bauschinger effect and a closed hysteresis loop, but the
    back stress grows without bound, so cyclic ratchetting is not captured.
    """

    name = "linear"
    linear = True

    def __init__(self, C: float):
        self.C = float(C)

    def update(self, beta, n, dalpha):
        return beta + (2.0 / 3.0) * self.C * dalpha[:, None, None] * n

    def dbeta_ddalpha(self, beta, n, dalpha):
        return (2.0 / 3.0) * self.C * n


class ArmstrongFrederick(KinematicHardening):
    r"""``dbeta = (2/3) C n dalpha - gamma beta dalpha``.

    The recall term bounds the back stress at ``(2/3) C / gamma``, which lets
    the model saturate and reproduce ratchetting under non-symmetric cycles.
    Integrated implicitly here,

        ``beta_new = (beta + (2/3) C n dalpha) / (1 + gamma dalpha)``,

    which is unconditionally stable and preserves the saturation limit for any
    step size -- an explicit update does neither, and overshoots the limit when
    ``gamma dalpha`` approaches one.
    """

    name = "armstrong-frederick"
    linear = False

    def __init__(self, C: float, gamma: float):
        self.C = float(C)
        self.gamma = float(gamma)

    def saturation(self) -> float:
        """Saturated back stress in the von Mises equivalent measure,
        ``q(beta) -> C / gamma``. The component along the flow direction is
        ``(2/3) C / gamma``, smaller by the same factor of 1.5 as in
        :class:`LinearKinematic`."""
        return self.C / self.gamma if self.gamma else np.inf

    def update(self, beta, n, dalpha):
        num = beta + (2.0 / 3.0) * self.C * dalpha[:, None, None] * n
        return num / (1.0 + self.gamma * dalpha)[:, None, None]

    def dbeta_ddalpha(self, beta, n, dalpha):
        d = (1.0 + self.gamma * dalpha)[:, None, None]
        num = beta + (2.0 / 3.0) * self.C * dalpha[:, None, None] * n
        return (2.0 / 3.0) * self.C * n / d - self.gamma * num / d ** 2


# ===========================================================================

ISOTROPIC_REGISTRY: dict[str, type[IsotropicHardening]] = {
    "perfect": PerfectPlasticity,
    "none": PerfectPlasticity,
    "linear": LinearIsotropic,
    "voce": Voce,
    "swift": Swift,
    "ludwik": Ludwik,
}

KINEMATIC_REGISTRY: dict[str, type[KinematicHardening]] = {
    "none": NoKinematic,
    "linear": LinearKinematic,
    "prager": LinearKinematic,
    "armstrong-frederick": ArmstrongFrederick,
    "af": ArmstrongFrederick,
}


def get_isotropic(name: str, **kwargs) -> IsotropicHardening:
    key = name.lower().replace("_", "-")
    if key not in ISOTROPIC_REGISTRY:
        raise NotImplementedError(
            f"Unknown isotropic hardening '{name}'. Available: "
            f"{sorted(set(ISOTROPIC_REGISTRY))}")
    return ISOTROPIC_REGISTRY[key](**kwargs)


def get_kinematic(name: str, **kwargs) -> KinematicHardening:
    key = name.lower().replace("_", "-")
    if key not in KINEMATIC_REGISTRY:
        raise NotImplementedError(
            f"Unknown kinematic hardening '{name}'. Available: "
            f"{sorted(set(KINEMATIC_REGISTRY))}")
    return KINEMATIC_REGISTRY[key](**kwargs)
