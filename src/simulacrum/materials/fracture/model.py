r"""
The phase-field fracture model: composition of the four axes.

    model = PhaseFieldModel(E, nu, Gc, ell,
                            split="spectral", degradation="quadratic",
                            dissipation="AT2", coupling="hybrid")

Nothing here re-implements any of the four; this class only wires them together
and owns the one piece of state they share, the history field ``H``.

The two equations
-----------------
Momentum (weak form, per quadrature point):

    INT  sigma(eps, d) : eps(v)  dx  =  external work

with ``sigma`` and its tangent supplied by the coupling.

Phase field, in a form that covers every combination of alpha and g:

    r(d, H)  :=  (Gc / (c_alpha l)) alpha'(d)  +  g'(d) H
    INT [ (2 Gc l / c_alpha) grad d . grad w  +  r(d, H) w ] dx  =  0

Linearising ``r`` about the current ``d`` gives the Newton system; when both
``alpha`` and ``g`` are quadratic (AT1 or AT2 with quadratic degradation)
``r`` is affine in ``d``, so one step from any starting point is exact and the
"Newton loop" costs a single linear solve. That is the case the staggered
scheme is usually run in, and :attr:`linear_phase_field` reports it.

Irreversibility
---------------
``H(x, t) = max over tau <= t of  Psi_drive(eps(x, tau))``. Because ``H`` never
decreases and ``g'`` is negative, ``d`` cannot decrease either: cracks do not
heal. This is Miehe's device, and it is also what decouples the two equations
enough for a staggered scheme to be robust.
"""

from __future__ import annotations

import numpy as np

from .coupling import Coupling, Hybrid, get_coupling
from .degradation import Degradation, get_degradation
from .dissipation import Dissipation, get_dissipation
from ..splits import Split, get_split, voigt_stress, voigt_tangent

__all__ = ["PhaseFieldModel"]


def _resolve(value, getter, kwargs=None):
    if isinstance(value, str):
        return getter(value, **(kwargs or {}))
    return value


class PhaseFieldModel:
    """Isotropic linear elasticity + phase-field fracture.

    Parameters
    ----------
    E, nu     : elastic constants
    Gc        : critical energy release rate
    ell       : regularization length
    split, degradation, dissipation, coupling
              : names or instances; see the corresponding modules
    plane_strain : 2-D interpretation (plane stress is not supported by the
              splits, which assume eps_33 = 0)
    """

    def __init__(self, E: float, nu: float, Gc: float, ell: float,
                 split: str | Split = "spectral",
                 degradation: str | Degradation = "quadratic",
                 dissipation: str | Dissipation = "AT2",
                 coupling: str | Coupling = "hybrid",
                 degradation_kwargs: dict | None = None,
                 plane_strain: bool = True):
        self.E, self.nu = float(E), float(nu)
        self.Gc, self.ell = float(Gc), float(ell)
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = E / (2.0 * (1.0 + nu))
        self.plane_strain = bool(plane_strain)
        if not plane_strain:
            raise NotImplementedError(
                "the splits assume eps_33 = 0; plane stress needs a separate "
                "treatment of the out-of-plane strain"
            )

        self.split = _resolve(split, get_split)
        self.degradation = _resolve(degradation, get_degradation,
                                    degradation_kwargs)
        self.dissipation = _resolve(dissipation, get_dissipation)
        self.coupling = _resolve(coupling, get_coupling)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"PhaseFieldModel(E={self.E:g}, nu={self.nu:g}, "
                f"Gc={self.Gc:g}, ell={self.ell:g}, "
                f"split={self.split.name}, "
                f"degradation={self.degradation.name}, "
                f"dissipation={self.dissipation.name}, "
                f"coupling={self.coupling.name})")

    @property
    def linear_momentum(self) -> bool:
        return self.coupling.linear_momentum

    @property
    def linear_phase_field(self) -> bool:
        """True when one linear solve gives the exact phase-field update."""
        return (self.degradation.quadratic_in_d
                and self.dissipation.quadratic_in_d)

    @property
    def needs_bound_constraint(self) -> bool:
        return self.dissipation.needs_bound_constraint

    # ------------------------------------------------------------------
    # Momentum equation
    # ------------------------------------------------------------------
    def stress_tangent(self, eps: np.ndarray, d: np.ndarray):
        """Degraded stress ``(n, dim, dim)`` and tangent ``(n, dim, ...)``."""
        gd = self.degradation.g(np.asarray(d, dtype=float))
        return self.coupling.stress_tangent(self.split, eps, self.lam,
                                            self.mu, gd)

    def stress_tangent_voigt(self, eps: np.ndarray, d: np.ndarray):
        """The same in Voigt form, ready for ``B.T @ C @ B`` assembly."""
        sigma, C = self.stress_tangent(eps, d)
        return voigt_stress(sigma), voigt_tangent(C)

    # ------------------------------------------------------------------
    # Phase-field equation
    # ------------------------------------------------------------------
    def driving_energy(self, eps: np.ndarray) -> np.ndarray:
        return self.coupling.driving_energy(self.split, eps, self.lam, self.mu)

    def update_history(self, H: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """H <- max(H, Psi_drive). Returns the updated array."""
        return np.maximum(H, self.driving_energy(eps))

    def gradient_coefficient(self) -> float:
        return self.dissipation.gradient_coefficient(self.Gc, self.ell)

    def reaction(self, d: np.ndarray, H: np.ndarray) -> np.ndarray:
        """r(d, H), the non-gradient part of the phase-field residual."""
        d = np.asarray(d, dtype=float)
        c = self.dissipation.local_coefficient(self.Gc, self.ell)
        return c * self.dissipation.dalpha(d) + self.degradation.dg(d) * H

    def dreaction(self, d: np.ndarray, H: np.ndarray) -> np.ndarray:
        """dr/dd, the mass-matrix coefficient of the linearised problem."""
        d = np.asarray(d, dtype=float)
        c = self.dissipation.local_coefficient(self.Gc, self.ell)
        return c * self.dissipation.ddalpha(d) + self.degradation.ddg(d) * H

    # ------------------------------------------------------------------
    def apply_compression_constraint(self, d: np.ndarray,
                                     eps: np.ndarray) -> np.ndarray:
        """Zero the damage where compression dominates (hybrid only).

        No-op for every other coupling, so callers need not branch.
        """
        if not self.coupling.enforce_compression_constraint:
            return d
        mask = Hybrid.compression_mask(self.split, eps, self.lam, self.mu)
        out = np.array(d, dtype=float, copy=True)
        out[mask] = 0.0
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def elastic_energy_density(self, eps: np.ndarray,
                               d: np.ndarray) -> np.ndarray:
        """g(d) Psi_plus + Psi_minus, or g(d) Psi_0 for isotropic coupling."""
        pp, pm = self.split.energies(eps, self.lam, self.mu)
        gd = self.degradation.g(np.asarray(d, dtype=float))
        if self.coupling.name == "anisotropic":
            return gd * pp + pm
        return gd * (pp + pm)

    def fracture_energy_density(self, d: np.ndarray,
                                grad_d: np.ndarray) -> np.ndarray:
        """(Gc / c_alpha) [alpha(d)/l + l |grad d|^2]."""
        d = np.asarray(d, dtype=float)
        gsq = np.einsum("ni,ni->n", grad_d, grad_d)
        return (self.Gc / self.dissipation.c_alpha) * (
            self.dissipation.alpha(d) / self.ell + self.ell * gsq)

    def recommended_mesh_size(self, safety: float = 2.0) -> float:
        """``l / safety``. Below this the regularized crack is unresolved and
        the effective toughness is wrong -- silently, with a plausible-looking
        load-displacement curve."""
        return self.ell / safety

    def check_mesh_size(self, h: float, safety: float = 2.0) -> None:
        import warnings
        h_max = self.recommended_mesh_size(safety)
        if h > h_max:
            warnings.warn(
                f"element size h={h:g} exceeds l/{safety:g}={h_max:g}: the "
                "regularized crack is under-resolved and the effective Gc "
                "will be too high",
                RuntimeWarning, stacklevel=2,
            )
