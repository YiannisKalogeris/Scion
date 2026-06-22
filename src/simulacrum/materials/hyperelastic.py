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


class NeoHookeanDecoupled(Material):
    """
    Compressible Neo-Hookean with a decoupled deviatoric-volumetric split.

    This is the form used by ANSYS / Abaqus (and the one requested in
    Εργασία 3). Strain energy (3D):

        W(F) = (μ/2)(Ī₁ - 3) + (K/2)(J - 1)²

    with the isochoric (volume-preserving) first invariant

        Ī₁ = J^(-2/3) I₁,   I₁ = tr(C) = F:F,   J = det F,

    μ the initial shear modulus and K the initial bulk modulus.

    First Piola-Kirchhoff stress (P = ∂W/∂F):

        P = μ J^(-2/3) (F - (1/3) I₁ F⁻ᵀ)  +  K J (J - 1) F⁻ᵀ

    The material tangent A_iJkL = ∂P_iJ/∂F_kL is computed in closed form
    (see _pk1_tangent_3d). At F = I the tangent reduces exactly to the
    isotropic linear elastic tensor with shear modulus μ and Lamé
    λ = K - 2μ/3, so the *initial* response matches linear elasticity.

    Supports three idealisations through one 3D kernel:
        - 3D            : pk1_stress_and_tangent_3d(F)            (F is 3x3)
        - plane strain  : pk1_stress_and_tangent_2d(F, False)    (λ₃ = 1)
        - plane stress  : pk1_stress_and_tangent_2d(F, True)     (solve σ₃₃=0)

    Note: this is a DIFFERENT model from `NeoHookean` in this module, which
    uses the coupled ln(J) (Ciarlet) energy. They diverge at large strain.

    Parameters
    ----------
    mu : float
        Initial shear modulus.
    K : float
        Initial bulk modulus.
    """

    name = "NeoHookeanDecoupled"

    def __init__(self, mu: float, K: float):
        self.mu = mu
        self.K = K
        self.lam = K - 2.0 * mu / 3.0  # equivalent Lamé first parameter

    @classmethod
    def from_E_nu(cls, E: float, nu: float):
        """
        Create from Young's modulus and Poisson's ratio.

            μ = E / (2(1+ν)),   K = E / (3(1-2ν))

        These make the initial tangent coincide with the isotropic linear
        elastic material defined by (E, ν).
        """
        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        return cls(mu=mu, K=K)

    @property
    def bulk_modulus(self):
        return self.K

    @property
    def shear_modulus(self):
        return self.mu

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _embed(F2, lam3):
        """Embed a 2x2 deformation gradient into 3x3 with F_33 = lam3."""
        F3 = np.eye(3)
        F3[:2, :2] = F2
        F3[2, 2] = lam3
        return F3

    # -----------------------------------------------------------------
    # 3D kernel: PK1 stress and consistent tangent
    # -----------------------------------------------------------------
    def _pk1_tangent_3d(self, F):
        """
        Core routine. Returns (P, A) for a full 3x3 deformation gradient F.

            P : (3, 3)        First Piola-Kirchhoff stress
            A : (3,3,3,3)     Tangent A_iJkL = ∂P_iJ/∂F_kL
        """
        mu, K = self.mu, self.K
        J = np.linalg.det(F)

        # Guard against non-physical / inverted states (help NR recover)
        if J <= 1e-14:
            P = np.zeros((3, 3))
            A = mu * np.einsum('ik,JL->iJkL', np.eye(3), np.eye(3))
            return P, A

        Finv = np.linalg.inv(F)
        G = Finv.T                       # G_iJ = (F⁻ᵀ)_iJ
        I1 = float(np.sum(F * F))        # tr(FᵀF)
        g = J ** (-2.0 / 3.0)

        # PK1 stress
        P = mu * g * (F - (1.0 / 3.0) * I1 * G) + K * J * (J - 1.0) * G

        # Consistent tangent A_iJkL = ∂P_iJ/∂F_kL
        I3 = np.eye(3)
        A = (mu * g) * (
            np.einsum('ik,JL->iJkL', I3, I3)
            - (2.0 / 3.0) * (np.einsum('iJ,kL->iJkL', F, G)
                             + np.einsum('kL,iJ->iJkL', F, G))
            + (2.0 / 9.0) * I1 * np.einsum('iJ,kL->iJkL', G, G)
            + (1.0 / 3.0) * I1 * np.einsum('iL,kJ->iJkL', G, G)
        )
        A += K * J * (2.0 * J - 1.0) * np.einsum('iJ,kL->iJkL', G, G)
        A -= K * J * (J - 1.0) * np.einsum('iL,kJ->iJkL', G, G)
        return P, A

    def pk1_stress_and_tangent_3d(self, F):
        """3D PK1 stress and tangent. F is (3, 3)."""
        return self._pk1_tangent_3d(F)

    def pk1_stress_3d(self, F):
        """3D PK1 stress only. F is (3, 3)."""
        P, _ = self._pk1_tangent_3d(F)
        return P

    # -----------------------------------------------------------------
    # Plane stress: solve σ₃₃ = 0 (⇔ P₃₃ = 0) for the out-of-plane stretch
    # -----------------------------------------------------------------
    def _solve_lambda3(self, F2, tol=1e-12, max_iter=50):
        """Newton solve of P₃₃(λ₃) = 0 for the plane-stress out-of-plane stretch."""
        lam3 = 1.0
        for _ in range(max_iter):
            P3, A3 = self._pk1_tangent_3d(self._embed(F2, lam3))
            r = P3[2, 2]
            drdl = A3[2, 2, 2, 2]
            if abs(drdl) < 1e-30:
                break
            dlam3 = -r / drdl
            if lam3 + dlam3 <= 0.0:      # keep the stretch positive
                lam3 *= 0.5
            else:
                lam3 += dlam3
            if abs(dlam3) < tol * abs(lam3):
                break
        return lam3

    # -----------------------------------------------------------------
    # 2D interface (plane strain / plane stress), drop-in compatible
    # with the rest of the materials in this module.
    # -----------------------------------------------------------------
    def pk1_stress_and_tangent_2d(self, F, plane_stress=True):
        """
        2D PK1 stress and condensed tangent. F is (2, 2).

        Returns:
            P2  : (2, 2)        in-plane First PK stress
            A2  : (2,2,2,2)     in-plane tangent ∂P_iJ/∂F_kL
            lam3: out-of-plane stretch (1.0 for plane strain)
        """
        if plane_stress:
            lam3 = self._solve_lambda3(F)
        else:
            lam3 = 1.0

        P3, A3 = self._pk1_tangent_3d(self._embed(F, lam3))
        P2 = P3[:2, :2].copy()
        A2 = A3[:2, :2, :2, :2].copy()

        if plane_stress:
            # Static condensation of the out-of-plane DOF:
            #   dλ₃/dF_kL = -A₃₃kL / A₃₃₃₃,   then
            #   A2_iJkL += ∂P_iJ/∂λ₃ · dλ₃/dF_kL = A_iJ33 · dλ₃/dF_kL
            denom = A3[2, 2, 2, 2]
            dlam3_dF = -A3[2, 2, :2, :2] / denom
            A2 += np.einsum('iJ,kL->iJkL', A3[:2, :2, 2, 2], dlam3_dF)

        return P2, A2, lam3

    def pk1_stress_2d(self, F, plane_stress=True):
        """
        2D PK1 stress only. F is (2, 2).

        Returns:
            P2  : (2, 2) in-plane First PK stress
            lam3: out-of-plane stretch (1.0 for plane strain)

        (No condensation term is needed for the stress itself — by the
        envelope theorem ∂W/∂λ₃ = P₃₃ = 0 at the plane-stress solution.)
        """
        if plane_stress:
            lam3 = self._solve_lambda3(F)
        else:
            lam3 = 1.0
        P3, _ = self._pk1_tangent_3d(self._embed(F, lam3))
        return P3[:2, :2].copy(), lam3

    # -----------------------------------------------------------------
    # 2nd Piola-Kirchhoff stress and material tangent in Voigt form
    # (for the B-matrix / Total Lagrangian PK2 formulation)
    # -----------------------------------------------------------------
    def pk2_voigt_and_tangent_2d(self, F, plane_stress=True):
        """
        2nd Piola-Kirchhoff stress and material tangent in Voigt notation,
        for the B-matrix (Total Lagrangian) formulation. F is (2, 2).

            S_voigt = [S_XX, S_YY, S_XY]      (2nd PK stress)
            C_voigt = dS/dE  (3x3 Voigt, engineering shear)
            lam3    = out-of-plane stretch (1.0 for plane strain)

        Obtained by pulling back the (condensed) PK1 stress and tangent
        returned by pk1_stress_and_tangent_2d:

            S       = F^{-1} P
            A_iJkL  = d_ik S_LJ + F_iM F_kN C_MJNL
            =>  C_MJNL = F^{-1}_Mi F^{-1}_Nk (A_iJkL - d_ik S_LJ)

        This identity is exact, so the PK2/B-matrix assembly reproduces the
        PK1 assembly to machine precision. Valid for both plane strain
        (lam3 = 1) and plane stress (where A, S are the condensed quantities).
        """
        P2, A2, lam3 = self.pk1_stress_and_tangent_2d(F, plane_stress)
        Finv = np.linalg.inv(F)
        S2 = Finv @ P2                                  # 2nd PK (2x2)

        I2 = np.eye(2)
        geom = np.einsum('ik,LJ->iJkL', I2, S2)         # d_ik S_LJ
        C4 = np.einsum('Mi,Nk,iJkL->MJNL', Finv, Finv, A2 - geom)  # dS/dE

        vm = [(0, 0), (1, 1), (0, 1)]
        S_voigt = np.array([S2[0, 0], S2[1, 1], S2[0, 1]])
        C_voigt = np.zeros((3, 3))
        for a, (M, J) in enumerate(vm):
            for b, (N, L) in enumerate(vm):
                C_voigt[a, b] = C4[M, J, N, L]
        return S_voigt, C_voigt, lam3

    # -----------------------------------------------------------------
    # Cauchy stress
    # -----------------------------------------------------------------
    def cauchy_stress_3d(self, F):
        """Cauchy stress σ = (1/J) P Fᵀ. F is (3, 3); returns (3, 3)."""
        P, _ = self._pk1_tangent_3d(F)
        J = np.linalg.det(F)
        return (1.0 / J) * P @ F.T

    def cauchy_stress_2d(self, F, plane_stress=True):
        """
        Cauchy stress for a 2D problem. F is (2, 2).

        Returns the FULL 3x3 Cauchy tensor: [:2,:2] is the in-plane stress
        and [2,2] is σ₃₃ (zero for plane stress, generally nonzero for plane
        strain — needed for a correct plane-strain von Mises).
        """
        if plane_stress:
            lam3 = self._solve_lambda3(F)
        else:
            lam3 = 1.0
        F3 = self._embed(F, lam3)
        P3, _ = self._pk1_tangent_3d(F3)
        J3 = np.linalg.det(F3)
        return (1.0 / J3) * P3 @ F3.T

    # -----------------------------------------------------------------
    # Strain energy
    # -----------------------------------------------------------------
    def strain_energy_3d(self, F):
        """Strain energy density W(F). F is (3, 3)."""
        J = np.linalg.det(F)
        I1 = float(np.sum(F * F))
        I1_bar = J ** (-2.0 / 3.0) * I1
        return 0.5 * self.mu * (I1_bar - 3.0) + 0.5 * self.K * (J - 1.0) ** 2

    def strain_energy(self, F, plane_stress=True):
        """Strain energy density for a 2D problem. F is (2, 2)."""
        if plane_stress:
            lam3 = self._solve_lambda3(F)
        else:
            lam3 = 1.0
        return self.strain_energy_3d(self._embed(F, lam3))

    # -----------------------------------------------------------------
    # Small-strain (initial) tangent in Voigt form, for compatibility
    # and for building the matching linear-elastic comparison material.
    # -----------------------------------------------------------------
    def tangent(self, dim: int = 3, plane_stress: bool = False, strain=None):
        """
        Initial linear-elastic tangent in Voigt notation.

            dim=3                      -> 6x6  [xx,yy,zz,xy,yz,zx]
            dim=2, plane_stress=False  -> 3x3  plane strain  [xx,yy,xy]
            dim=2, plane_stress=True   -> 3x3  plane stress   [xx,yy,xy]
        """
        mu, lam = self.mu, self.lam
        if dim == 3:
            C = np.zeros((6, 6))
            C[:3, :3] = lam
            for i in range(3):
                C[i, i] += 2.0 * mu
            for i in range(3, 6):
                C[i, i] = mu
            return C
        if dim == 2:
            if not plane_stress:
                return np.array([
                    [lam + 2 * mu, lam,          0.0],
                    [lam,          lam + 2 * mu, 0.0],
                    [0.0,          0.0,          mu ],
                ])
            # plane stress: recover (E, ν) from (μ, λ)
            E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
            nu = lam / (2.0 * (lam + mu))
            c = E / (1.0 - nu ** 2)
            return np.array([
                [c,      c * nu, 0.0           ],
                [c * nu, c,      0.0           ],
                [0.0,    0.0,    c * (1 - nu) / 2.0],
            ])
        raise NotImplementedError(f"tangent for dim={dim} not implemented")

    def stress(self, strain, dim=3):
        raise NotImplementedError(
            "Use pk1_stress_*_2d / _3d with the deformation gradient F")