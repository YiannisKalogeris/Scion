"""
Hyperelastic material models (scaffold).

Planned models
--------------
- Neo-Hookean
- Mooney-Rivlin
- Ogden
- Arruda-Boyce
"""

from __future__ import annotations

import numpy as np

from simulacrum.materials.base import Material


class NeoHookean(Material):
    """
    Compressible Neo-Hookean hyperelastic model.
 
    Strain energy (3D):
        W(F) = (μ/2)(I₁ - 3) - μ ln J + (λ/2)(ln J)²
 
    where I₁ = tr(FᵀF), J = det F.
 
    First Piola-Kirchhoff stress:
        P = μ F + (λ ln J - μ) F⁻ᵀ
 
    Parameters
    ----------
    mu : float
        Shear modulus (Lamé's second parameter)
    lam : float
        Lamé's first parameter
        Related to bulk modulus: κ = λ + 2μ/3
    """
 
    name = "NeoHookean"
 
    def __init__(self, mu: float, lam: float):
        self.mu = mu
        self.lam = lam
 
    @classmethod
    def from_E_nu(cls, E: float, nu: float):
        """Create from Young's modulus and Poisson's ratio."""
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return cls(mu=mu, lam=lam)
 
    @property
    def bulk_modulus(self):
        return self.lam + 2.0 * self.mu / 3.0
 
    # -----------------------------------------------------------------
    # Plane stress: solve for out-of-plane stretch λ₃
    # -----------------------------------------------------------------
    def _solve_lambda3(self, J2d, tol=1e-12, max_iter=50):
        """
        Solve σ₃₃ = 0 for λ₃ (plane stress condition).
        """
        mu, lam = self.mu, self.lam
        lam3 = 1.0

        for _ in range(max_iter):
            J3d = J2d * lam3
            if J3d <= 0:
                lam3 = 0.5 * lam3  # pull back if we overshoot
                continue
            f = mu * (lam3**2 - 1.0) + lam * np.log(J3d)
            df = 2.0 * mu * lam3 + lam / lam3
            dlam3 = -f / df
            # Limit step to prevent going negative
            if lam3 + dlam3 <= 0:
                lam3 *= 0.5
            else:
                lam3 += dlam3
            if abs(dlam3) < tol * abs(lam3):
                break

        return lam3
    # -----------------------------------------------------------------
    # PK1 stress from 2D deformation gradient
    # -----------------------------------------------------------------
    def pk1_stress_2d(self, F, plane_stress=True):
        """
        Compute 2D First Piola-Kirchhoff stress.
 
        Args:
            F: (2, 2) deformation gradient
            plane_stress: if True, solve for λ₃ such that σ₃₃=0
 
        Returns:
            P: (2, 2) First PK stress
            lambda3: out-of-plane stretch (1.0 for plane strain)
        """
        mu, lam = self.mu, self.lam
        J2d = np.linalg.det(F)

                # Guard against non-physical deformation
        if J2d <= 1e-14:
            P = np.zeros((2, 2))
            A = np.zeros((2, 2, 2, 2))
            # Return identity-like tangent to help NR recover
            for i in range(2):
                for J in range(2):
                    A[i, J, i, J] = mu
            return P, A, 1.0
        
        F_inv_T = np.linalg.inv(F).T
 
        if plane_stress:
            lam3 = self._solve_lambda3(J2d)
        else:
            lam3 = 1.0
 
        J3d = J2d * lam3
 
        # P = μ F + (λ ln J - μ) F⁻ᵀ  (2D components)
        P = mu * F + (lam * np.log(J3d) - mu) * F_inv_T
 
        return P, lam3
 
    # -----------------------------------------------------------------
    # PK1 stress and tangent (for Newton-Raphson)
    # -----------------------------------------------------------------
    def pk1_stress_and_tangent_2d(self, F, plane_stress=True):
        """
        Compute 2D PK1 stress and material tangent A_iJkL = ∂P_iJ/∂F_kL.
 
        Args:
            F: (2, 2) deformation gradient
 
        Returns:
            P: (2, 2) First PK stress
            A: (2, 2, 2, 2) tangent tensor ∂P/∂F
            lambda3: out-of-plane stretch
        """
        mu, lam = self.mu, self.lam
        J2d = np.linalg.det(F)

        if J2d <= 1e-14:
            P = np.zeros((2, 2))
            A = np.zeros((2, 2, 2, 2))
            for i in range(2):
                for J in range(2):
                    A[i, J, i, J] = mu
            return P, A, 1.0  

        F_inv = np.linalg.inv(F)
        F_inv_T = F_inv.T
 
        if plane_stress:
            lam3 = self._solve_lambda3(J2d)
        else:
            lam3 = 1.0
 
        J3d = J2d * lam3
 
        # PK1 stress
        P = mu * F + (lam * np.log(J3d) - mu) * F_inv_T
 
        # Tangent: A_iJkL = ∂P_iJ / ∂F_kL
        # A_iJkL = μ δ_ik δ_JL
        #        + (λ ln J - μ) F⁻¹_Li F⁻¹_Jk   (from ∂F⁻ᵀ/∂F)
        #        + λ F⁻ᵀ_iJ F⁻ᵀ_kL              (from ∂(ln J)/∂F)
        A = np.zeros((2, 2, 2, 2))
        coeff = lam * np.log(J3d) - mu
 
        for i in range(2):
            for J in range(2):
                for k in range(2):
                    for L in range(2):
                        A[i, J, k, L] = (
                            mu * (i == k) * (J == L)
                            - coeff * F_inv[L, i] * F_inv[J, k]
                            + lam * F_inv_T[i, J] * F_inv_T[k, L]
                        )
 
        # Plane stress correction to tangent
        # (condensation of the out-of-plane DOF)
        if plane_stress:
            # ∂λ₃/∂F_kL from implicit differentiation of σ₃₃=0
            # d/dF_kL [μ(λ₃² - 1) + λ ln(J₂D λ₃)] = 0
            # → (2μλ₃ + λ/λ₃) dλ₃/dF_kL + λ (1/J₂D)(∂J₂D/∂F_kL) = 0
            # → dλ₃/dF_kL = -λ/(2μλ₃ + λ/λ₃) * F⁻ᵀ_kL
            denom = 2.0 * mu * lam3 + lam / lam3
            dlam3_dF = np.zeros((2, 2))
            for k in range(2):
                for L in range(2):
                    dlam3_dF[k, L] = -lam / denom * F_inv_T[k, L]
 
            # Correction to A from λ₃ dependence:
            # ∂P_iJ/∂F_kL += ∂P_iJ/∂λ₃ * ∂λ₃/∂F_kL
            # ∂P_iJ/∂λ₃ = (λ/λ₃) F⁻ᵀ_iJ  (via ∂ln(J₃D)/∂λ₃ = 1/λ₃)
            dP_dlam3 = (lam / lam3) * F_inv_T
 
            for i in range(2):
                for J in range(2):
                    for k in range(2):
                        for L in range(2):
                            A[i, J, k, L] += dP_dlam3[i, J] * dlam3_dF[k, L]
 
        return P, A, lam3
 
    # -----------------------------------------------------------------
    # Convenience: Cauchy stress from F
    # -----------------------------------------------------------------
    def cauchy_stress_2d(self, F, plane_stress=True):
        """Compute Cauchy stress: σ = (1/J) P Fᵀ."""
        P, lam3 = self.pk1_stress_2d(F, plane_stress)
        J = np.linalg.det(F) * lam3
        sigma = (1.0 / J) * P @ F.T
        return sigma
 
    # -----------------------------------------------------------------
    # Strain energy density
    # -----------------------------------------------------------------
    def strain_energy(self, F, plane_stress=True):
        """Compute strain energy density W(F)."""
        mu, lam = self.mu, self.lam
        J2d = np.linalg.det(F)
 
        if plane_stress:
            lam3 = self._solve_lambda3(J2d)
        else:
            lam3 = 1.0
 
        J3d = J2d * lam3
        C = F.T @ F
        I1_2d = np.trace(C) + lam3**2  # full 3D trace
 
        W = 0.5 * mu * (I1_2d - 3.0) - mu * np.log(J3d) + 0.5 * lam * np.log(J3d)**2
        return W
 
    # -----------------------------------------------------------------
    # Legacy interface (for compatibility)
    # -----------------------------------------------------------------
    def tangent(self, dim: int = 3, strain=None):
        """Linear tangent at zero strain (for small-strain compatibility)."""
        mu, lam = self.mu, self.lam
        if dim == 2:
            # Plane strain Voigt tangent
            C = np.array([
                [lam + 2*mu, lam,       0      ],
                [lam,       lam + 2*mu, 0      ],
                [0,         0,          mu     ],
            ])
            return C
        else:
            raise NotImplementedError("3D tangent not implemented")
 
    def stress(self, strain, dim=3):
        raise NotImplementedError("Use pk1_stress_2d with deformation gradient F")
