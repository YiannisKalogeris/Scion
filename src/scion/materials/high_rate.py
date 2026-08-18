r"""
High-rate metal plasticity: Johnson-Cook flow stress on a Mie-Grüneisen
pressure, with adiabatic heating.

This is the constitutive model for impact: a Taylor bar, a penetrator, a shaped
charge. What distinguishes it from :mod:`~.plasticity` is not the yield surface
-- it is still von Mises -- but three couplings that quasi-static plasticity does
without:

**The deviator and the pressure come from different places.** The deviatoric
stress is integrated hypoelastically and returned radially to the Johnson-Cook
flow stress; the pressure comes from an :mod:`~.eos`, which is nonlinear in
compression and depends on internal energy. Splitting them is the whole point:
at 10 GPa the volumetric response of a metal is nothing like ``K tr(eps)``.

**Plastic work heats the material, and the heat softens it.** A fraction ``chi``
of the plastic work (Taylor-Quinney, ~0.9) becomes internal energy, which raises
the temperature, which lowers the Johnson-Cook flow stress, which changes the
plastic work. Adiabatic because the timescale is microseconds and conduction has
no time to act -- that is an assumption, and it fails for a slow process.

**The rate matters.** The flow stress carries ``1 + C ln(edot*)``, evaluated at
the total deviatoric strain rate measured from the velocity gradient. See
:class:`~.plasticity.hardening.JohnsonCook` for why that is frozen over the
step rather than solved for.

Objectivity is exact, and a rate-dependent surface is not a violation of it
---------------------------------------------------------------------------
A rigid rotation of a pre-stressed state leaves the von Mises stress unchanged
to ``4e-16`` and rotates the tensor correctly -- provided the flow stress is
rate-independent. Turn the Johnson-Cook rate term on and the same rotation makes
the stress *fall*, from 217.2 MPa to 170.6 MPa. That is not a loss of
objectivity: the rotation happens at zero strain rate, so the yield surface
shrinks to its quasi-static size and the stress returns onto it. See
:class:`~.plasticity.hardening.JohnsonCook` for why that is the model rather
than the integration.

Hypoelastic, and objective by polar decomposition
--------------------------------------------------
The update is a rate form: ``sigma^d`` is carried as state and incremented by
``2G dt d^d``. A rate form is not automatically frame-indifferent -- a rigid
rotation would generate spurious stress -- so the standard fix is applied here:
the deformation gradient is polar-decomposed, ``F = R U``, and the update is
performed in the *unrotated* configuration, with ``R`` applied only when the
Cauchy stress is assembled. That is the Green-Naghdi rate, and it is why
:meth:`JohnsonCookModel.update` needs ``F`` and not only the strain increment.

Not a tangent operator
----------------------
This model returns stress, not a consistent tangent, because it is built for
explicit dynamics where none is needed. Using it inside an implicit Newton solve
would give at best linear convergence -- and the library's own discipline says a
wrong tangent still converges, linearly, with no symptom but the iteration
count. :class:`~.plasticity.PlasticityModel` is the implicit path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scion.core import small_linalg as sla
from scion.materials.eos import EquationOfState, LinearEOS
from scion.materials.plasticity.hardening import JohnsonCook

__all__ = ["HighRateState", "JohnsonCookModel"]


@dataclass
class HighRateState:
    """Per-particle history. Everything here is path-dependent.

    Held separately from the model so that a trial step can be discarded --
    the same discipline as ``QuadratureState.commit``/``restore`` on the FEM
    side. The model never mutates a state it was handed.
    """

    #: ``(n, 3, 3)`` unrotated deviatoric Cauchy stress
    deviator: np.ndarray
    #: ``(n,)`` equivalent plastic strain
    alpha: np.ndarray
    #: ``(n,)`` internal energy per unit *reference* volume
    energy: np.ndarray
    #: ``(n,)`` absolute temperature
    temperature: np.ndarray

    @classmethod
    def zeros(cls, n: int, temperature: float = 293.0) -> "HighRateState":
        return cls(deviator=np.zeros((n, 3, 3)), alpha=np.zeros(n),
                   energy=np.zeros(n),
                   temperature=np.full(n, float(temperature)))

    def copy(self) -> "HighRateState":
        return HighRateState(self.deviator.copy(), self.alpha.copy(),
                             self.energy.copy(), self.temperature.copy())

    @property
    def n(self) -> int:
        return int(self.alpha.size)

    def __repr__(self) -> str:
        return (f"HighRateState({self.n} points, "
                f"max alpha {self.alpha.max():.4g}, "
                f"max T {self.temperature.max():.1f} K)")


class JohnsonCookModel:
    """Johnson-Cook plasticity with an equation of state and adiabatic heating.

    Parameters
    ----------
    shear_modulus : ``G``. The deviatoric stiffness; the volumetric one comes
        from ``eos`` instead, which is the whole point of the split.
    hardening : a :class:`~.plasticity.hardening.JohnsonCook`.
    eos : an :class:`~.eos.EquationOfState`. Defaults to a linear one built
        from ``bulk_modulus`` if that is given instead.
    density : reference density, for the temperature update and the wave speed.
    specific_heat : ``C_p``, for the temperature update.
    taylor_quinney : ``chi``, the fraction of plastic work that becomes heat.
        0.9 is the conventional value and is an approximation; the true value
        depends on strain and is not constant.
    """

    name = "johnson-cook"

    def __init__(self, shear_modulus: float, hardening: JohnsonCook,
                 eos: EquationOfState | None = None,
                 bulk_modulus: float | None = None,
                 density: float = 1.0, specific_heat: float = 1.0,
                 taylor_quinney: float = 0.9):
        self.G = float(shear_modulus)
        if self.G <= 0.0:
            raise ValueError("the shear modulus must be positive")
        if not isinstance(hardening, JohnsonCook):
            raise TypeError(
                "this model exists for the Johnson-Cook flow stress; for a "
                "rate-independent law use PlasticityModel, which also gives a "
                "consistent tangent")
        self.hardening = hardening

        if eos is None:
            if bulk_modulus is None:
                raise ValueError("supply either an eos or a bulk_modulus")
            eos = LinearEOS(bulk_modulus)
        self.eos = eos

        self.density = float(density)
        self.specific_heat = float(specific_heat)
        self.taylor_quinney = float(taylor_quinney)

    # ------------------------------------------------------------------
    @property
    def wave_speed(self) -> float:
        """P-wave speed ``sqrt((K + 4G/3) / rho)``.

        The dilatational speed, not the bulk speed: it carries the shear
        stiffness too, and it is the one that sets the CFL limit. Using
        ``sqrt(K/rho)`` gives a step about 15% too large for a typical metal,
        which is exactly the margin an explicit scheme does not have.
        """
        return float(np.sqrt((self.eos.bulk_modulus + 4.0 * self.G / 3.0)
                             / self.density))

    # ------------------------------------------------------------------
    def update(self, F: np.ndarray, L: np.ndarray, dt: float,
               state: HighRateState):
        r"""One step of the constitutive update. Returns ``(sigma, new_state)``.

        Parameters
        ----------
        F : ``(n, 3, 3)`` deformation gradient at the *end* of the step.
        L : ``(n, 3, 3)`` velocity gradient over the step.
        dt : time increment.
        state : committed history. Not modified.

        Returns
        -------
        sigma : ``(n, 3, 3)`` Cauchy stress, rotated back to the current frame.
        new_state : the updated history, to be committed by the caller.
        """
        F = np.asarray(F, dtype=float)
        L = np.asarray(L, dtype=float)
        n = F.shape[0]
        if F.shape[1:] != (3, 3):
            raise ValueError(
                f"F must be (n, 3, 3); got {F.shape}. Constitutive code here is "
                f"always full 3-D, even for a plane-strain problem, because "
                f"sigma_33 is nonzero and enters J2")

        # --- kinematics: rate of deformation and the rotation ------------
        D = 0.5 * (L + sla.transpose(L))
        R = _polar_rotation(F)
        # unrotated rate of deformation, the Green-Naghdi frame
        d = sla.transpose(R) @ D @ R
        trace_d = np.einsum("pii->p", d)
        d_dev = d - (trace_d / 3.0)[:, None, None] * np.eye(3)

        # equivalent strain rate, the JC rate argument
        rate = np.sqrt(2.0 / 3.0 * np.einsum("pij,pij->p", d_dev, d_dev))

        # --- deviatoric trial and radial return ---------------------------
        trial = state.deviator + 2.0 * self.G * dt * d_dev
        trial_eq = np.sqrt(1.5 * np.einsum("pij,pij->p", trial, trial))

        frozen = self.hardening.at(
            rate=rate,
            temperature=(state.temperature
                         if self.hardening.T_melt is not None else None))
        flow_stress = frozen.kappa(state.alpha)

        plastic = trial_eq > flow_stress
        d_alpha = np.zeros(n)
        deviator = trial.copy()
        if np.any(plastic):
            # radial return: scale the trial deviator onto the yield surface.
            # dalpha = (trial_eq - sigma_f) / (3G) is the closed form for von
            # Mises with the flow stress frozen -- no Newton iteration needed,
            # which is what makes this affordable per particle per step.
            excess = trial_eq[plastic] - flow_stress[plastic]
            d_alpha[plastic] = excess / (3.0 * self.G)
            scale = flow_stress[plastic] / trial_eq[plastic]
            deviator[plastic] = scale[:, None, None] * trial[plastic]

        alpha = state.alpha + d_alpha

        # --- energy, pressure, temperature --------------------------------
        J = sla.det(F)
        if np.any(J <= 0.0):
            raise ValueError(
                f"det F <= 0 at {int(np.sum(J <= 0.0))} points: the material "
                f"has inverted. In explicit dynamics this almost always means "
                f"the time step exceeded the CFL limit.")

        # plastic work per unit reference volume; J converts from current to
        # reference, which is what makes the EOS energy argument consistent
        energy = state.energy + J * self.taylor_quinney * flow_stress * d_alpha
        pressure = self.eos.pressure(1.0 / J, energy)

        temperature = state.temperature
        if self.specific_heat > 0.0:
            density = self.density / J
            temperature = state.temperature + (
                self.taylor_quinney * flow_stress * d_alpha
                / (density * self.specific_heat))

        # --- assemble and rotate back --------------------------------------
        # sigma = deviator - p I, with p positive in compression
        unrotated = deviator - pressure[:, None, None] * np.eye(3)
        sigma = R @ unrotated @ sla.transpose(R)

        return sigma, HighRateState(deviator=deviator, alpha=alpha,
                                    energy=energy, temperature=temperature)

    # ------------------------------------------------------------------
    def equivalent_stress(self, state: HighRateState) -> np.ndarray:
        """von Mises stress from the stored deviator, ``sqrt(3/2 s:s)``."""
        return np.sqrt(1.5 * np.einsum("pij,pij->p", state.deviator,
                                       state.deviator))

    def __repr__(self) -> str:
        return (f"JohnsonCookModel(G={self.G:.4g}, {self.hardening!r}, "
                f"{self.eos!r})")


def _polar_rotation(F: np.ndarray) -> np.ndarray:
    r"""``R`` from the polar decomposition ``F = R U``, by SVD.

    ``F = U_s Sigma V'`` gives ``R = U_s V'``. Done by SVD rather than by
    ``R = F U^-1`` with ``U = sqrt(F'F)``, because the square root of a nearly
    singular ``F'F`` loses most of its digits while the SVD does not.

    This is the objectivity fix: without it a rigidly rotating body accumulates
    spurious deviatoric stress, growing with the rotation angle, and the failure
    looks like a material effect rather than an integration error.
    """
    U, _, Vt = np.linalg.svd(np.asarray(F, dtype=float))
    R = U @ Vt
    # guard the reflection branch: an SVD may return det = -1, which is a
    # reflection rather than a rotation and would flip the stress
    flipped = np.linalg.det(R) < 0.0
    if np.any(flipped):
        U = U.copy()
        U[flipped, :, -1] *= -1.0
        R = U @ Vt
    return R
