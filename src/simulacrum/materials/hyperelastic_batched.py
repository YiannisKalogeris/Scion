"""
Batched (vectorised over particles) hyperelastic models.

These mirror the single-point models in ``simulacrum/materials/hyperelastic.py``
but operate on ``(n, d, d)`` stacks of deformation gradients, which is what
particle methods need. The single-point versions stay as the readable reference
implementation; ``tests/mpm/test_material_consistency.py`` asserts the two agree
to machine precision on random F, so the duplication is guarded rather than a
second source of truth.

Interface shared by anything usable as an MPM constitutive law:

    pk1(F) -> (P, J)     P: (n, d, d) first Piola-Kirchhoff,  J: (n,) det(F)

Keeping PK1 (rather than Cauchy) as the interface means the stepper never has to
divide by J, and it matches the signature already used by the FEM assembler.
"""

from __future__ import annotations

import numpy as np

from simulacrum.core import small_linalg as sla


class BatchedMaterial:
    """Interface marker. Subclasses implement ``pk1``."""

    def pk1(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class NeoHookeanCompressible(BatchedMaterial):
    r"""W = (mu/2)(I1 - d) - mu ln J + (lambda/2)(ln J)^2

    P = mu (F - F^-T) + lambda ln(J) F^-T

    The standard workhorse for explicit MPM: cheap, robust in compression, and
    its tangent never needs to be formed explicitly.
    """

    def __init__(self, mu: float, lam: float):
        self.mu = float(mu)
        self.lam = float(lam)

    @classmethod
    def from_E_nu(cls, E: float, nu: float) -> "NeoHookeanCompressible":
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return cls(mu, lam)

    @property
    def wave_speed_factor(self) -> float:
        """sqrt(lambda + 2 mu); divide by sqrt(rho) for the P-wave speed."""
        return float(np.sqrt(self.lam + 2.0 * self.mu))

    def pk1(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        J = sla.det(F)
        FinvT = sla.inv_transpose(F, J)
        P = self.mu * (F - FinvT) + self.lam * np.log(J)[:, None, None] * FinvT
        return P, J

    def energy(self, F: np.ndarray) -> np.ndarray:
        """Strain-energy density, for energy-conservation checks."""
        J = sla.det(F)
        I1 = np.einsum("pij,pij->p", F, F)
        d = F.shape[1]
        lnJ = np.log(J)
        return 0.5 * self.mu * (I1 - d) - self.mu * lnJ + 0.5 * self.lam * lnJ ** 2


class NeoHookeanDecoupledBatched(BatchedMaterial):
    r"""W = (mu/2)(J^{-2/3} I1 - 3) + (K/2)(J - 1)^2

    Batched counterpart of the decoupled deviatoric/volumetric model already in
    ``hyperelastic.py``. Expects F embedded in 3-D (F33 = 1 for plane strain);
    see the single-point implementation for the F33 handling.
    """

    def __init__(self, mu: float, K: float):
        self.mu = float(mu)
        self.K = float(K)

    @classmethod
    def from_E_nu(cls, E: float, nu: float) -> "NeoHookeanDecoupledBatched":
        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        return cls(mu, K)

    @property
    def wave_speed_factor(self) -> float:
        return float(np.sqrt(self.K + 4.0 * self.mu / 3.0))

    def pk1(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        J = sla.det(F)
        G = sla.inv_transpose(F, J)
        I1 = np.einsum("pij,pij->p", F, F)
        g = J ** (-2.0 / 3.0)
        P = (
            self.mu * g[:, None, None] * (F - (I1 / 3.0)[:, None, None] * G)
            + (self.K * J * (J - 1.0))[:, None, None] * G
        )
        return P, J
