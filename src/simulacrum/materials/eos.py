r"""
Equations of state: the pressure a solid carries under compression.

For quasi-static problems the volumetric response is linear elastic and the
bulk modulus is the whole story. Under shock loading it is not. Pressures reach
tens or hundreds of gigapascals, the material stiffens nonlinearly as it
compresses, and the heat deposited by plastic work contributes a pressure of its
own. An equation of state supplies

    p = p(rho, e)

and it replaces the volumetric part of the constitutive law; the deviatoric part
still comes from the plasticity model. That split -- deviator from
:mod:`~.plasticity`, pressure from here -- is why this is a separate module and
not another hardening law.

Sign convention, stated because it is the usual source of a sign error
----------------------------------------------------------------------
:meth:`EquationOfState.pressure` returns ``p`` **positive in compression**, the
convention of the shock-physics literature. Continuum mechanics uses the
opposite sign for mean stress, so assembling the Cauchy stress reads

    sigma = deviator - p I

Getting this backwards produces a material that expands under load and it looks
entirely plausible until the volume history is plotted.

The compression limit is real
-----------------------------
The Mie-Grüneisen denominator ``[eta - S(eta - 1)]^2`` vanishes at
``eta = S/(S - 1)``, which for a typical ``S ~ 1.5`` is a compression ratio of
3. Beyond it the formula returns a negative or infinite pressure rather than a
large one. That is not a numerical artefact to clamp away -- it is the linear
``U_s = c_0 + S u_p`` shock relation being extrapolated past where it holds.
:class:`MieGruneisen` reports the limit and refuses to evaluate past it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "EquationOfState",
    "LinearEOS",
    "MieGruneisen",
    "EOS_REGISTRY",
    "get_eos",
]


class EquationOfState(ABC):
    """``p(rho, e)``, positive in compression."""

    name: str = "eos"

    @abstractmethod
    def pressure(self, compression: np.ndarray,
                 energy: np.ndarray | None = None) -> np.ndarray:
        r"""Pressure from the compression ratio ``eta = rho / rho_0``.

        ``eta > 1`` is compression. ``energy`` is the internal energy **per unit
        reference volume**, which is what makes the Grüneisen term dimensionally
        a pressure; passing energy per unit mass instead is off by ``rho_0``.
        """

    def pressure_from_jacobian(self, J, energy=None) -> np.ndarray:
        r"""Same thing from the volume ratio ``J = det F``.

        ``eta = 1 / J`` by mass conservation, which is the form an MPM or
        finite-strain FE update actually has to hand.
        """
        J = np.asarray(J, dtype=float)
        if np.any(J <= 0.0):
            raise ValueError(
                "det F <= 0: the material has inverted, which is a modelling "
                "failure rather than a state to assign a pressure to")
        return self.pressure(1.0 / J, energy)

    @property
    @abstractmethod
    def bulk_modulus(self) -> float:
        """Small-strain bulk modulus, for a CFL wave speed and for sanity."""

    def sound_speed(self, density: float) -> float:
        """Bulk sound speed ``sqrt(K / rho)``. Not the P-wave speed, which also
        carries the shear stiffness -- see the note in the Taylor bar example."""
        return float(np.sqrt(self.bulk_modulus / density))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(K={self.bulk_modulus:.4g})"


class LinearEOS(EquationOfState):
    r"""``p = K (eta - 1)``, equivalently ``p = K (1 - J)`` to first order.

    The volumetric part of linear elasticity, written as an equation of state so
    that it can be swapped for :class:`MieGruneisen` without touching the caller.
    Correct for small compressions and increasingly wrong beyond a few percent,
    where the real material stiffens and this does not.

    Carries no energy dependence, so plastic heating produces no pressure. That
    is the other thing it gets wrong at high rate.
    """

    name = "linear"

    def __init__(self, bulk_modulus: float):
        self._K = float(bulk_modulus)
        if self._K <= 0.0:
            raise ValueError("the bulk modulus must be positive")

    @property
    def bulk_modulus(self) -> float:
        return self._K

    def pressure(self, compression, energy=None):
        eta = np.asarray(compression, dtype=float)
        return self._K * (eta - 1.0)


class MieGruneisen(EquationOfState):
    r"""Mie-Grüneisen with a linear shock Hugoniot.

        p = rho_0 c_0^2 (eta - 1) [eta - (Gamma_0 / 2)(eta - 1)]
            / [eta - S (eta - 1)]^2                                  + Gamma_0 e

    The first term is the pressure on the Hugoniot -- the locus of states
    reachable by a single shock -- derived from the empirical linear relation
    ``U_s = c_0 + S u_p`` between shock and particle velocity. The second is the
    thermal pressure from internal energy, which is how plastic work feeds back
    into the volumetric response.

    Parameters
    ----------
    rho_0 : reference density.
    c_0 : bulk sound speed at zero pressure, the intercept of the ``U_s(u_p)``
        fit. Note ``rho_0 c_0^2`` **is** the small-strain bulk modulus, which is
        a free consistency check against a handbook value -- for copper it gives
        138.6 GPa against a published 140.
    Gamma_0 : Grüneisen parameter, the thermal pressure coefficient.
    S : slope of the shock relation, typically 1.3 to 1.6.

    Tension
    -------
    The Hugoniot is a compression locus and the fit has no meaning for
    ``eta < 1``. The convention used here, and in the reference implementation
    this was taken from, is to evaluate the same expression anyway: it stays
    finite and gives a restoring pressure of roughly ``-K(1 - eta)``, which is
    the right first-order behaviour. Do not read the magnitude as physical in
    strong tension -- a spall model is what that regime needs.
    """

    name = "mie-gruneisen"

    def __init__(self, rho_0: float, c_0: float, Gamma_0: float,
                 S: float = 1.5):
        self.rho_0 = float(rho_0)
        self.c_0 = float(c_0)
        self.Gamma_0 = float(Gamma_0)
        self.S = float(S)
        for label, value in (("rho_0", self.rho_0), ("c_0", self.c_0)):
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")

    @property
    def bulk_modulus(self) -> float:
        """``rho_0 c_0^2``: the small-strain limit of the Hugoniot term."""
        return self.rho_0 * self.c_0 ** 2

    @property
    def compression_limit(self) -> float:
        r"""``eta`` at which the denominator vanishes, ``S / (S - 1)``.

        The linear shock relation predicts infinite compression at finite
        pressure here, which is where the model stops being extrapolatable.
        Infinite for ``S <= 1``, where the denominator never vanishes.
        """
        return float("inf") if self.S <= 1.0 else self.S / (self.S - 1.0)

    def pressure(self, compression, energy=None):
        eta = np.asarray(compression, dtype=float)
        if np.any(eta <= 0.0):
            raise ValueError("the compression ratio must be positive")

        limit = self.compression_limit
        if np.any(eta >= limit):
            worst = float(np.max(eta))
            raise ValueError(
                f"compression ratio {worst:.3f} reaches the Mie-Gruneisen limit "
                f"{limit:.3f} = S/(S-1), where the denominator vanishes. The "
                f"linear shock relation U_s = c_0 + S u_p has been extrapolated "
                f"past where it holds; this is a modelling limit, not a "
                f"numerical one.")

        mu = eta - 1.0
        denominator = (eta - self.S * mu) ** 2
        hugoniot = (self.rho_0 * self.c_0 ** 2 * mu
                    * (eta - 0.5 * self.Gamma_0 * mu) / denominator)

        if energy is None:
            return hugoniot
        return hugoniot + self.Gamma_0 * np.asarray(energy, dtype=float)

    def hugoniot_pressure(self, compression) -> np.ndarray:
        """The cold (zero internal energy) part alone."""
        return self.pressure(compression, energy=None)

    def __repr__(self) -> str:
        return (f"MieGruneisen(rho_0={self.rho_0:g}, c_0={self.c_0:g}, "
                f"Gamma_0={self.Gamma_0:g}, S={self.S:g}, "
                f"K={self.bulk_modulus:.4g})")


# ---------------------------------------------------------------------------

EOS_REGISTRY: dict[str, type[EquationOfState]] = {
    "linear": LinearEOS,
    "mie-gruneisen": MieGruneisen,
    "mg": MieGruneisen,
}


def get_eos(name: str, **kwargs) -> EquationOfState:
    key = name.lower().replace("_", "-")
    if key not in EOS_REGISTRY:
        raise NotImplementedError(
            f"Unknown equation of state '{name}'. Available: "
            f"{sorted(set(EOS_REGISTRY))}. Tillotson and JWL are not "
            f"implemented; both need a different functional form rather than "
            f"another parameter set.")
    return EOS_REGISTRY[key](**kwargs)
