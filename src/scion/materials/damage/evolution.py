r"""
Damage evolution: how much stiffness is left at a given history level.

Axis 2. A law is a scalar function ``D(kappa)`` on ``[0, 1]``, zero until the
threshold ``kappa_0`` and non-decreasing after it. ``kappa`` is the largest
equivalent strain the point has ever seen, so ``D`` inherits its
irreversibility: damage never heals, and unloading follows the secant to the
origin rather than the original slope. That secant unloading is the observable
signature of damage and what distinguishes it from plasticity, which unloads on
the elastic slope and leaves a permanent strain.

Every law here softens
----------------------
Past ``kappa_0`` the uniaxial stress ``(1 - D(kappa)) E kappa`` decreases. That
is the physics, and it costs the boundary value problem its ellipticity: the
solution localises into a band whose width is set by the element size and not by
the material, so the computed dissipation is proportional to ``h`` and tends to
zero under mesh refinement. Nothing in this module can fix that -- the fix is
either a regularisation with a length scale
(:class:`~scion.materials.damage.model.GradientDamageModel`) or a crack
band, where the softening slope is scaled by the element size so the dissipation
per unit area is at least held constant.

:meth:`DamageLaw.fracture_energy` exists for that second route, and for
calibration generally: it is the area under the uniaxial curve, and it is the
quantity that has to be preserved.

Residual stress
---------------
The exponential laws carry a parameter (``alpha``, ``A``) that decides whether
the material ends up carrying nothing or keeps a residual stress forever. Less
than one leaves a residual, which makes the fracture energy infinite and the
localisation band never fully broken. That is sometimes deliberate -- it keeps
the tangent better conditioned -- and always worth knowing about, so
:meth:`fracture_energy` returns ``inf`` rather than a large finite number.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "DamageLaw",
    "LinearSoftening",
    "ExponentialSoftening",
    "MazarsLaw",
    "EVOLUTION_REGISTRY",
    "get_damage_law",
]


class DamageLaw(ABC):
    """``D(kappa)`` and its slope."""

    name: str = "damage-law"

    def __init__(self, kappa_0: float):
        if kappa_0 <= 0.0:
            raise ValueError("the damage threshold kappa_0 must be positive")
        self.kappa_0 = float(kappa_0)

    # ------------------------------------------------------------------
    @abstractmethod
    def _D(self, kappa: np.ndarray) -> np.ndarray:
        """The law past the threshold, before clipping."""

    @abstractmethod
    def _dD(self, kappa: np.ndarray) -> np.ndarray:
        """Its derivative past the threshold, before clipping."""

    # ------------------------------------------------------------------
    def D(self, kappa) -> np.ndarray:
        """Damage, ``(n,)``, in ``[0, 1]``. Zero at and below the threshold."""
        k = np.asarray(kappa, dtype=float)
        active = k > self.kappa_0
        safe = np.where(active, k, self.kappa_0)
        return np.clip(np.where(active, self._D(safe), 0.0), 0.0, 1.0)

    def dD_dkappa(self, kappa) -> np.ndarray:
        """``dD/dkappa``, ``(n,)``. Zero below the threshold and once ``D``
        has saturated at one -- both are places where a nonzero slope would put
        a spurious rank-one term into the consistent tangent."""
        k = np.asarray(kappa, dtype=float)
        active = (k > self.kappa_0) & (self._D_raw(k) < 1.0)
        safe = np.where(active, k, self.kappa_0)
        return np.where(active, self._dD(safe), 0.0)

    def _D_raw(self, k: np.ndarray) -> np.ndarray:
        return np.where(k > self.kappa_0, self._D(np.maximum(k, self.kappa_0)),
                        0.0)

    # ------------------------------------------------------------------
    def stress(self, kappa, E: float) -> np.ndarray:
        """Uniaxial stress ``(1 - D) E kappa`` along the loading path.

        The curve the law is usually calibrated against, and the cheapest way to
        see what a set of parameters actually does before running a mesh.
        """
        k = np.asarray(kappa, dtype=float)
        return (1.0 - self.D(k)) * float(E) * k

    def fracture_energy(self, E: float) -> float:
        """Area under the uniaxial curve, per unit volume.

        ``inf`` when the law leaves a residual stress, which is a statement
        about the law rather than a failure to integrate.
        """
        raise NotImplementedError

    def peak_stress(self, E: float) -> float:
        """Largest uniaxial stress the law reaches.

        ``E kappa_0`` for a law that starts softening the instant it damages,
        which is the usual intent and the default here. A law that does not --
        see :class:`MazarsLaw` -- must override this, or every strength
        calibration built on it is wrong by whatever the overshoot is.
        """
        return float(E) * self.kappa_0

    def softens_immediately(self) -> bool:
        """Whether the stress starts decreasing at ``kappa_0``.

        When false, damage initiates while the material is still hardening: the
        peak stress moves off the threshold and ``kappa_0`` stops being the
        strength divided by ``E``.
        """
        return True

    def residual_stress(self, E: float) -> float:
        """Stress the law never sheds, as ``kappa -> inf``.

        Analytic rather than read off a sample, because the sample would have to
        run to a strain nobody applies before an exponential tail looks like its
        own limit -- and a residual missed that way is exactly what makes a
        fracture energy come out finite when it is not.
        """
        raise NotImplementedError

    def _sample_max(self) -> float:
        """Upper end of a ``kappa`` range that shows the whole law."""
        return 40.0 * self.kappa_0

    # ------------------------------------------------------------------
    def check(self, E: float = 1.0, n: int = 512,
              kappa_max: float | None = None) -> dict:
        """Report the properties that decide whether the law is usable."""
        k = np.linspace(0.0, kappa_max or self._sample_max(), n)
        D = self.D(k)
        return {
            "kappa_0": self.kappa_0,
            "D_max": float(D.max()),
            "monotone": bool(np.all(np.diff(D) >= -1e-14)),
            "softens_immediately": self.softens_immediately(),
            "peak_stress": self.peak_stress(E),
            "peak_stress_sampled": float(self.stress(k, E).max()),
            "residual_stress": self.residual_stress(E),
            "fracture_energy": self.fracture_energy(E),
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(kappa_0={self.kappa_0:g})"


# ---------------------------------------------------------------------------

class LinearSoftening(DamageLaw):
    r"""``D = (kappa_u / kappa) (kappa - kappa_0) / (kappa_u - kappa_0)``.

    Chosen so the *stress* is linear, not the damage: substituting into
    ``(1 - D) E kappa`` gives a straight line from ``E kappa_0`` at the
    threshold to exactly zero at ``kappa_u``. That is what makes it the law to
    verify against -- the uniaxial response, the fracture energy
    ``g_f = E kappa_0 kappa_u / 2`` and the localised solution of a bar all have
    closed forms.

    It also reaches ``D = 1`` at a finite strain, so the band is genuinely
    broken and the tangent there is exactly zero. The exponential laws only
    approach it.
    """

    name = "linear"

    def __init__(self, kappa_0: float, kappa_u: float):
        super().__init__(kappa_0)
        if kappa_u <= kappa_0:
            raise ValueError(
                f"kappa_u ({kappa_u:g}) must exceed kappa_0 ({kappa_0:g}); "
                f"otherwise the material is fully broken the moment it damages")
        self.kappa_u = float(kappa_u)

    @classmethod
    def from_fracture_energy(cls, kappa_0: float, E: float,
                             g_f: float) -> "LinearSoftening":
        """Pick ``kappa_u`` to give a specified area under the uniaxial curve.

        ``kappa_u = 2 g_f / (E kappa_0)``. With ``g_f = G_c / h`` and ``h`` the
        element size this is the crack-band method: the dissipation per unit
        *area* is then mesh-independent even though the band is still one
        element wide. It fixes the energy and not the band, so the peak load and
        the post-peak shape still move with the mesh -- an improvement on doing
        nothing and not a substitute for a length scale.
        """
        return cls(kappa_0, 2.0 * float(g_f) / (float(E) * float(kappa_0)))

    def _D(self, kappa):
        return (self.kappa_u / kappa) * (kappa - self.kappa_0) \
            / (self.kappa_u - self.kappa_0)

    def _dD(self, kappa):
        return (self.kappa_u * self.kappa_0
                / ((self.kappa_u - self.kappa_0) * kappa ** 2))

    def fracture_energy(self, E):
        return 0.5 * float(E) * self.kappa_0 * self.kappa_u

    def residual_stress(self, E):
        return 0.0

    def _sample_max(self):
        return 1.2 * self.kappa_u

    def __repr__(self):
        return (f"LinearSoftening(kappa_0={self.kappa_0:g}, "
                f"kappa_u={self.kappa_u:g})")


class ExponentialSoftening(DamageLaw):
    r"""``D = 1 - (kappa_0/kappa) [1 - alpha + alpha exp(-beta (kappa - kappa_0))]``

    Peerlings et al. (1996), and the law the gradient-damage literature is
    written around. ``alpha`` is the fraction of the stress that eventually
    disappears and ``beta`` how fast: large ``beta`` is brittle, small ``beta``
    is a long tail.

    ``alpha = 1`` breaks completely and has a finite fracture energy,
    ``E kappa_0 (kappa_0/2 + 1/beta)``. Anything less leaves a residual stress
    ``(1 - alpha) E kappa_0`` that never goes away, so the material never fully
    loses its stiffness and the fracture energy is infinite.
    """

    name = "exponential"

    def __init__(self, kappa_0: float, alpha: float = 0.99,
                 beta: float = 100.0):
        super().__init__(kappa_0)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def _tail(self, kappa):
        return np.exp(-self.beta * (kappa - self.kappa_0))

    def _D(self, kappa):
        return 1.0 - (self.kappa_0 / kappa) * (
            1.0 - self.alpha + self.alpha * self._tail(kappa))

    def _dD(self, kappa):
        e = self._tail(kappa)
        f = 1.0 - self.alpha + self.alpha * e
        return self.kappa_0 * f / kappa ** 2 \
            + self.kappa_0 * self.alpha * self.beta * e / kappa

    def fracture_energy(self, E):
        if self.alpha < 1.0:
            return float("inf")
        return float(E) * self.kappa_0 * (0.5 * self.kappa_0 + 1.0 / self.beta)

    def residual_stress(self, E):
        return (1.0 - self.alpha) * float(E) * self.kappa_0

    def _sample_max(self):
        return self.kappa_0 + 12.0 / self.beta

    def __repr__(self):
        return (f"ExponentialSoftening(kappa_0={self.kappa_0:g}, "
                f"alpha={self.alpha:g}, beta={self.beta:g})")


class MazarsLaw(DamageLaw):
    r"""``D = 1 - kappa_0 (1 - A) / kappa - A exp(-B (kappa - kappa_0))``.

    Mazars (1986). Almost the exponential law, and not quite: the decaying term
    carries no ``kappa_0/kappa`` factor, which gives a heavier tail at large
    strain. ``A = 1`` breaks completely.

    Mazars' original model applies this twice, with separate ``(A_t, B_t)`` and
    ``(A_c, B_c)``, and blends the two by weights built from the positive and
    negative parts of the strain. That asymmetry is not repeated here: in this
    library it belongs to the split axis, which degrades the tensile part of the
    energy and leaves the compressive part intact. Expressing it in both places
    would mean two independent knobs for one physical effect, and no way to tell
    which one produced a given result.

    ``B kappa_0`` must be at least one
    ----------------------------------
    Unlike the other two laws, this one does not necessarily start softening at
    the threshold. The uniaxial stress is
    ``E [kappa_0 (1-A) + A kappa exp(-B(kappa - kappa_0))]``, whose slope at
    ``kappa_0`` has the sign of ``1 - B kappa_0``. So for ``B kappa_0 < 1`` the
    material *keeps hardening* after damage initiates, peaking at
    ``kappa = 1/B`` with

        ``sigma_peak / (E kappa_0) = (A / (B kappa_0)) exp(B kappa_0 - 1)``,

    so the tensile strength is no longer ``E kappa_0``. The overshoot is not
    small: at ``kappa_0 = 1e-4`` and ``B = 500`` the peak is **7.7 times**
    ``E kappa_0``. Calibrating ``kappa_0`` from a measured strength without
    checking this puts the entire curve in the wrong place.
    :meth:`softens_immediately` reports the condition and :meth:`peak_stress`
    accounts for it.
    """

    name = "mazars"

    def __init__(self, kappa_0: float, A: float = 0.99, B: float = 100.0):
        super().__init__(kappa_0)
        if not 0.0 < A <= 1.0:
            raise ValueError("A must lie in (0, 1]")
        if B <= 0.0:
            raise ValueError("B must be positive")
        self.A = float(A)
        self.B = float(B)

    def _tail(self, kappa):
        return np.exp(-self.B * (kappa - self.kappa_0))

    def _D(self, kappa):
        return 1.0 - self.kappa_0 * (1.0 - self.A) / kappa \
            - self.A * self._tail(kappa)

    def _dD(self, kappa):
        return self.kappa_0 * (1.0 - self.A) / kappa ** 2 \
            + self.A * self.B * self._tail(kappa)

    def fracture_energy(self, E):
        if self.A < 1.0:
            return float("inf")
        return float(E) * (0.5 * self.kappa_0 ** 2
                           + self.kappa_0 / self.B + 1.0 / self.B ** 2)

    def residual_stress(self, E):
        return (1.0 - self.A) * float(E) * self.kappa_0

    def softens_immediately(self) -> bool:
        return self.B * self.kappa_0 >= 1.0

    def peak_stress(self, E):
        """``E kappa_0`` when the law softens at once, otherwise the maximum at
        ``kappa = 1/B``."""
        if self.softens_immediately():
            return float(E) * self.kappa_0
        return float(E) * (self.kappa_0 * (1.0 - self.A)
                           + (self.A / self.B)
                           * np.exp(self.B * self.kappa_0 - 1.0))

    def _sample_max(self):
        return self.kappa_0 + 12.0 / self.B

    def __repr__(self):
        return (f"MazarsLaw(kappa_0={self.kappa_0:g}, A={self.A:g}, "
                f"B={self.B:g})")


# ---------------------------------------------------------------------------

EVOLUTION_REGISTRY: dict[str, type[DamageLaw]] = {
    "linear": LinearSoftening,
    "linear-softening": LinearSoftening,
    "exponential": ExponentialSoftening,
    "peerlings": ExponentialSoftening,
    "mazars": MazarsLaw,
}


def get_damage_law(name: str, **kwargs) -> DamageLaw:
    key = name.lower().replace("_", "-")
    if key not in EVOLUTION_REGISTRY:
        raise NotImplementedError(
            f"Unknown damage evolution law '{name}'. Available: "
            f"{sorted(set(EVOLUTION_REGISTRY))}")
    return EVOLUTION_REGISTRY[key](**kwargs)
