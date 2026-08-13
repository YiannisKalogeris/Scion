r"""
Yield criteria.

A criterion that defines the elastic domain through

    f(sigma, kappa) <= 0,

where ``kappa`` is the current hardening force -- the yield stress for metals,
the cohesion for frictional materials. It supplies ``f`` and its first two
derivatives with respect to stress: the first is the normal to the surface
(and, for associative flow, the plastic flow direction), the second is needed
by the closest-point return map to converge quadratically.

Important: Everything works on **full 3-D stress tensors**, ``(n, 3, 3)``, batched over
quadrature points, regardless of the spatial dimension of the element. This is
not an implementation convenience. In plane strain ``eps_33 = 0`` but
``sigma_33`` is generally nonzero and enters ``J_2`` directly, so a criterion
written on the in-plane components alone gives the wrong yield surface. Voigt
reduction happens at the assembler boundary, not here.

Adding a criterion means implementing :meth:`YieldCriterion.f`,
:meth:`df_dsigma` and :meth:`d2f_dsigma2` and registering the class. The
closest-point return map then works with it unchanged.

Smoothness
----------
:attr:`YieldCriterion.smooth` records whether the surface is differentiable
everywhere it is reached. Mohr-Coulomb and Tresca have edges and corners where
the normal is a cone rather than a vector, and need a multi-surface return map;
they are deliberately absent until that machinery exists.

Separately, a smooth surface can still have an isolated singular point.
:meth:`singular_mask` flags stress states where ``df_dsigma`` is undefined --
for the Drucker-Prager cone this is the apex, which lies **on** the yield
surface and is reached by any predominantly tensile state, so the return map
must treat it as a special case. For von Mises the corresponding point
(``s = 0``) is strictly inside the elastic domain and is never reached.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "YieldCriterion",
    "VonMises",
    "DruckerPrager",
    "deviator",
    "pressure",
    "J2_invariant",
    "von_mises_stress",
    "P_DEV",
    "CRITERION_REGISTRY",
    "get_criterion",
]


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

_I = np.eye(3)

#: Fourth-order deviatoric projector, P_ijkl = I_sym - (1/3) d_ij d_kl.
P_DEV = (0.5 * (np.einsum("ik,jl->ijkl", _I, _I)
                + np.einsum("il,jk->ijkl", _I, _I))
         - np.einsum("ij,kl->ijkl", _I, _I) / 3.0)


def pressure(sigma: np.ndarray) -> np.ndarray:
    """Mean stress ``p = tr(sigma)/3``. Tension positive."""
    return np.einsum("nii->n", sigma) / 3.0


def deviator(sigma: np.ndarray) -> np.ndarray:
    """``s = sigma - p I``."""
    return sigma - pressure(sigma)[:, None, None] * _I


def J2_invariant(sigma: np.ndarray) -> np.ndarray:
    """``J2 = (1/2) s : s``."""
    s = deviator(sigma)
    return 0.5 * np.einsum("nij,nij->n", s, s)


def von_mises_stress(sigma: np.ndarray) -> np.ndarray:
    """``q = sqrt(3 J2)``."""
    return np.sqrt(3.0 * J2_invariant(sigma))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class YieldCriterion(ABC):
    """Defines the elastic domain ``f(sigma, kappa) <= 0``."""

    name: str = "criterion"

    #: False when the surface has edges or corners requiring a multi-surface
    #: return map. Such criteria are rejected by the single-surface algorithm.
    smooth: bool = True

    #: relative floor below which a stress deviator is treated as zero
    DEGENERATE_TOL: float = 1.0e-12

    # ------------------------------------------------------------------
    @abstractmethod
    def f(self, sigma: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """Yield function, ``(n,)``. Negative inside the elastic domain."""

    @abstractmethod
    def df_dsigma(self, sigma: np.ndarray) -> np.ndarray:
        """Outward normal ``df/dsigma``, ``(n, 3, 3)``."""

    @abstractmethod
    def d2f_dsigma2(self, sigma: np.ndarray) -> np.ndarray:
        """``d2f/dsigma2``, ``(n, 3, 3, 3, 3)``, with both minor symmetries."""

    @abstractmethod
    def df_dkappa(self, sigma: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """``df/dkappa``, ``(n,)``. Negative for a hardening material."""

    # ------------------------------------------------------------------
    def singular_mask(self, sigma: np.ndarray) -> np.ndarray:
        """Points where ``df_dsigma`` is undefined, ``(n,)`` bool.

        Default: nowhere. Cone-type criteria override this.
        """
        return np.zeros(sigma.shape[0], dtype=bool)

    def is_yielding(self, sigma: np.ndarray, kappa: np.ndarray,
                    rtol: float = 1.0e-8) -> np.ndarray:
        """``f > 0`` to a relative tolerance, ``(n,)`` bool."""
        scale = np.maximum(np.abs(np.asarray(kappa, dtype=float)), 1.0)
        return self.f(sigma, kappa) > rtol * scale

    def _safe(self, x: np.ndarray) -> np.ndarray:
        """Floor a magnitude before dividing by it."""
        scale = np.maximum(np.abs(x).max(), 1.0)
        return np.where(np.abs(x) < self.DEGENERATE_TOL * scale,
                        self.DEGENERATE_TOL * scale, x)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

class VonMises(YieldCriterion):
    r"""J2 plasticity: ``f = q - kappa``, with ``q = sqrt(3 J2)``.

    Pressure-insensitive and volume-preserving under associative flow, which is
    what makes it right for metals and wrong for soils and concrete.

    Derivatives::

        df/dsigma   = (3/2) s / q
        d2f/dsigma2 = (3/(2q)) [ P_dev - (3/(2 q^2)) s (x) s ]

    ``df_dsigma`` is undefined at ``s = 0``, but ``f = -kappa < 0`` there, so a
    state with zero deviator is strictly elastic and the return map never
    evaluates the normal at that point. :meth:`singular_mask` therefore reports
    nothing, unlike the Drucker-Prager cone.
    """

    name = "von-mises"
    smooth = True

    def f(self, sigma, kappa):
        return von_mises_stress(sigma) - np.asarray(kappa, dtype=float)

    def df_dsigma(self, sigma):
        s = deviator(sigma)
        q = self._safe(von_mises_stress(sigma))
        return 1.5 * s / q[:, None, None]

    def d2f_dsigma2(self, sigma):
        s = deviator(sigma)
        q = self._safe(von_mises_stress(sigma))
        n = sigma.shape[0]
        term = np.broadcast_to(P_DEV, (n,) + P_DEV.shape) \
            - 1.5 * np.einsum("nij,nkl->nijkl", s, s) / (q ** 2)[:, None, None, None, None]
        return 1.5 * term / q[:, None, None, None, None]

    def df_dkappa(self, sigma, kappa):
        return -np.ones(sigma.shape[0])


class DruckerPrager(YieldCriterion):
    r"""Cone in principal stress space: ``f = sqrt(J2) + eta p - xi kappa``.

    ``p = tr(sigma)/3`` with tension positive, so ``eta > 0`` shrinks the
    elastic domain under tension: the pressure sensitivity that von Mises lacks
    and that frictional materials require.

    Derivatives::

        df/dsigma   = s / (2 sqrt(J2))  +  (eta/3) I
        d2f/dsigma2 = (1/(2 sqrt(J2))) [ P_dev - s (x) s / (2 J2) ]

    The apex
    --------
    At ``s = 0`` the normal is undefined, and unlike von Mises this point lies
    **on** the surface, at ``p = xi kappa / eta``. Any sufficiently tensile
    trial state returns there, so a return map that only ever projects along
    ``df/dsigma`` will produce NaNs or fail to converge on such states. Use
    :meth:`singular_mask` to detect them and :meth:`apex_pressure` to place
    them; the apex return is a separate branch that sets ``s = 0`` and solves a
    scalar equation for ``p``.

    Non-associative flow
    --------------------
    Using ``eta`` in the flow direction as well makes the model associative and
    predicts dilatancy far in excess of what is measured. The usual remedy is a
    plastic potential with a smaller dilatancy parameter, which lives in the
    flow-rule module -- and makes the consistent tangent unsymmetric.
    """

    name = "drucker-prager"
    smooth = True

    def __init__(self, eta: float = 0.0, xi: float = 1.0):
        self.eta = float(eta)
        self.xi = float(xi)

    # ------------------------------------------------------------------
    @classmethod
    def from_mohr_coulomb(cls, phi: float, match: str = "outer",
                          degrees: bool = True) -> "DruckerPrager":
        """Calibrate the cone against a Mohr-Coulomb friction angle.

        The Mohr-Coulomb surface is an irregular hexagon; a cone can match it
        only on chosen meridians, and the choice matters by tens of percent in
        strength:

        ``"outer"``       circumscribes, touching the compressive corners
        ``"inner"``       inscribes, touching the tensile corners
        ``"plane-strain"``matches the plane-strain collapse load, which is
                          usually the relevant one for geotechnical problems
        """
        p = np.radians(phi) if degrees else float(phi)
        sp, cp = np.sin(p), np.cos(p)
        if match == "outer":
            eta = 6.0 * sp / (np.sqrt(3.0) * (3.0 - sp))
            xi = 6.0 * cp / (np.sqrt(3.0) * (3.0 - sp))
        elif match == "inner":
            eta = 6.0 * sp / (np.sqrt(3.0) * (3.0 + sp))
            xi = 6.0 * cp / (np.sqrt(3.0) * (3.0 + sp))
        elif match == "plane-strain":
            d = np.sqrt(3.0 + sp ** 2)
            eta = 3.0 * sp / d
            xi = 3.0 * cp / d
        else:
            raise ValueError(f"unknown Mohr-Coulomb match '{match}'")
        return cls(eta=float(eta), xi=float(xi))

    # ------------------------------------------------------------------
    def f(self, sigma, kappa):
        return (np.sqrt(J2_invariant(sigma)) + self.eta * pressure(sigma)
                - self.xi * np.asarray(kappa, dtype=float))

    def df_dsigma(self, sigma):
        s = deviator(sigma)
        r = self._safe(np.sqrt(J2_invariant(sigma)))
        return s / (2.0 * r)[:, None, None] + (self.eta / 3.0) * _I

    def d2f_dsigma2(self, sigma):
        s = deviator(sigma)
        j2 = J2_invariant(sigma)
        r = self._safe(np.sqrt(j2))
        n = sigma.shape[0]
        term = np.broadcast_to(P_DEV, (n,) + P_DEV.shape) \
            - np.einsum("nij,nkl->nijkl", s, s) \
            / (2.0 * self._safe(j2))[:, None, None, None, None]
        return term / (2.0 * r)[:, None, None, None, None]

    def df_dkappa(self, sigma, kappa):
        return np.full(sigma.shape[0], -self.xi)

    # ------------------------------------------------------------------
    def singular_mask(self, sigma: np.ndarray) -> np.ndarray:
        """Stress states at the cone apex, where the normal is undefined."""
        j2 = J2_invariant(sigma)
        scale = max(float(np.abs(sigma).max()), 1.0)
        return np.sqrt(j2) < self.DEGENERATE_TOL * scale

    def apex_pressure(self, kappa) -> np.ndarray:
        """``p`` at the apex, where ``sqrt(J2) = 0`` and ``f = 0``."""
        if self.eta == 0.0:
            return np.full(np.shape(kappa), np.inf)
        return self.xi * np.asarray(kappa, dtype=float) / self.eta


# ---------------------------------------------------------------------------

CRITERION_REGISTRY: dict[str, type[YieldCriterion]] = {
    "von-mises": VonMises,
    "vonmises": VonMises,
    "j2": VonMises,
    "mises": VonMises,
    "drucker-prager": DruckerPrager,
    "druckerprager": DruckerPrager,
    "dp": DruckerPrager,
}


def get_criterion(name: str, **kwargs) -> YieldCriterion:
    key = name.lower().replace("_", "-")
    if key not in CRITERION_REGISTRY:
        raise NotImplementedError(
            f"Unknown yield criterion '{name}'. Available: "
            f"{sorted(set(CRITERION_REGISTRY))}. Mohr-Coulomb and Tresca are "
            f"absent by design: their surfaces have corners and need a "
            f"multi-surface return map."
        )
    return CRITERION_REGISTRY[key](**kwargs)
