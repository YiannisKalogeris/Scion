"""
Batched (vectorised over particles) hyperelastic models.

These mirror the single-point models in ``scion/materials/hyperelastic.py``
but operate on ``(n, d, d)`` stacks of deformation gradients, which is what
particle methods need. The single-point versions stay as the readable reference
implementation; ``tests/mpm/test_material_consistency.py`` asserts the two agree
to machine precision on random F, so the duplication is guarded rather than a
second source of truth.

Interface shared by anything usable as an MPM constitutive law:

    pk1(F) -> (P, J)     P: (n, d, d) first Piola-Kirchhoff,  J: (n,) det(F)

Keeping PK1 (rather than Cauchy) as the interface means the stepper never has to
divide by J, and it matches the signature already used by the FEM assembler.

The implicit stepper needs one thing more:

    pk1_and_tangent(F) -> (P, J, A)   A: (n, d, d, d, d), A_iJkL = dP_iJ/dF_kL

which is the *first* elasticity tensor, differentiated with respect to F itself
rather than to a strain measure. That choice is what makes the structural
tangent a single contraction with no separate geometric-stiffness term -- see
``discretization/mpm/implicit.py``, where the initial-stress contribution
cancels identically. Writing A as dS/dE instead would put that term back and
it would have to be added by hand.

Index convention, because it is easy to get backwards: the first index pair
``iJ`` is spatial-then-material (as in ``P_iJ``), and the same for ``kL``. At
``F = I`` every model here must reduce to
``lambda d_iJ d_kL + mu (d_ik d_JL + d_iL d_Jk)``, which is the cheapest
sanity check there is and is asserted in the tests.
"""

from __future__ import annotations

import numpy as np

from scion.core import small_linalg as sla


class BatchedMaterial:
    """Interface marker. Subclasses implement ``pk1``.

    ``pk1_and_tangent`` is optional and only the implicit stepper asks for it,
    so it raises here rather than being abstract: an explicit-only material is
    complete without one.
    """

    def pk1(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def pk1_and_tangent(self, F: np.ndarray):
        raise NotImplementedError(
            f"{type(self).__name__} has no dP/dF, so it cannot be used with "
            f"ImplicitMPMStepper. Deriving it in closed form is the work; "
            f"finite-differencing it here would be silently slow and would "
            f"defeat the tangent check that is supposed to catch errors in it.")


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

    def pk1_and_tangent(self, F: np.ndarray):
        r"""``(P, J, A)`` with ``A_iJkL = dP_iJ/dF_kL``.

        Differentiating ``P = mu (F - F^-T) + lambda ln(J) F^-T`` needs only
        ``d(F^-T)_iJ/dF_kL = -G_kJ G_iL`` (with ``G = F^-T``) and
        ``d ln J/dF_kL = G_kL``, giving

            A = mu I (x) I + (mu - lambda ln J) G_kJ G_iL + lambda G_iJ G_kL.

        Note the middle term's index pattern is *transposed* relative to the
        last one; writing both as ``G (x) G`` in the same order is the natural
        slip and produces a tangent that is still symmetric and still converges,
        linearly.
        """
        J = sla.det(F)
        G = sla.inv_transpose(F, J)
        lnJ = np.log(J)
        P = self.mu * (F - G) + self.lam * lnJ[:, None, None] * G

        d = F.shape[1]
        eye = np.eye(d)
        A = np.broadcast_to(
            self.mu * np.einsum("ik,JL->iJkL", eye, eye),
            (F.shape[0], d, d, d, d)).copy()
        A += ((self.mu - self.lam * lnJ)[:, None, None, None, None]
              * np.einsum("pkJ,piL->piJkL", G, G))
        A += self.lam * np.einsum("piJ,pkL->piJkL", G, G)
        return P, J, A

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

    def pk1_and_tangent(self, F: np.ndarray):
        r"""``(P, J, A)`` with ``A_iJkL = dP_iJ/dF_kL``.

            A = mu g [ d_ik d_JL - (2/3)(G_kL F_iJ + F_kL G_iJ)
                       + (2/9) I1 G_iJ G_kL + (I1/3) G_kJ G_iL ]
                + K [ J(2J - 1) G_iJ G_kL - J(J - 1) G_kJ G_iL ]

        The volumetric and deviatoric parts each contribute *both* index
        patterns of ``G (x) G``, with opposite signs, and the two nearly cancel
        near incompressibility -- which is exactly where a slip in either sign
        stops showing up as a wrong answer and starts showing up only as a
        Newton iteration count.
        """
        J = sla.det(F)
        G = sla.inv_transpose(F, J)
        I1 = np.einsum("pij,pij->p", F, F)
        g = J ** (-2.0 / 3.0)
        P = (
            self.mu * g[:, None, None] * (F - (I1 / 3.0)[:, None, None] * G)
            + (self.K * J * (J - 1.0))[:, None, None] * G
        )

        d = F.shape[1]
        eye = np.eye(d)
        GG = np.einsum("piJ,pkL->piJkL", G, G)       # G_iJ G_kL
        GGt = np.einsum("pkJ,piL->piJkL", G, G)      # G_kJ G_iL

        A = np.broadcast_to(np.einsum("ik,JL->iJkL", eye, eye),
                            (F.shape[0], d, d, d, d)).copy()
        A -= (2.0 / 3.0) * (np.einsum("pkL,piJ->piJkL", G, F)
                            + np.einsum("pkL,piJ->piJkL", F, G))
        A += (2.0 / 9.0) * I1[:, None, None, None, None] * GG
        A += (I1 / 3.0)[:, None, None, None, None] * GGt
        A *= (self.mu * g)[:, None, None, None, None]
        A += self.K * ((J * (2.0 * J - 1.0))[:, None, None, None, None] * GG
                       - (J * (J - 1.0))[:, None, None, None, None] * GGt)
        return P, J, A
