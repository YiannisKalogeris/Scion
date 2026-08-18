"""
Geometrically nonlinear material models.

These models use a linear stress-strain relationship but account for
large displacements and rotations through the Green-Lagrange strain
and the Second Piola-Kirchhoff stress (Total Lagrangian formulation).

The nonlinearity is purely geometric — not material.

Supported models:
    - StVenantKirchhoff: S = C:E with E = (F^T F - I)/2

All models implement the same interface as hyperelastic materials:
    P, A, lam3 = mat.pk1_stress_and_tangent_2d(F, plane_stress=True)

This allows the nonlinear assembly code to be material-agnostic.

Usage:
    from scion.materials.geometric_nonlinear import StVenantKirchhoff

    mat = StVenantKirchhoff.from_E_nu(E=50e9, nu=0.3)
    P, A, lam3 = mat.pk1_stress_and_tangent_2d(F, plane_stress=True)
"""

import numpy as np
from scion.materials.base import Material


class StVenantKirchhoff(Material):
    """
    St. Venant-Kirchhoff model for geometric nonlinearity.

    Uses a linear constitutive law (S = C:E) with the nonlinear
    Green-Lagrange strain measure E = (F^T F - I)/2.

    Strain energy:
        W(E) = (1/2) E : C : E

    Second Piola-Kirchhoff stress:
        S = C : E    (linear in E, but E is nonlinear in u)

    First Piola-Kirchhoff stress:
        P = F S      (nonlinear through F)

    At small strains this reduces exactly to linear elasticity.

    WARNING: SVK exhibits unphysical behaviour under large compression
    (the strain energy is not polyconvex). For large compressive strains,
    use Neo-Hookean from scion.materials.hyperelastic instead.

    Parameters
    ----------
    E_young : float
        Young's modulus
    nu : float
        Poisson's ratio
    """

    name = "StVenantKirchhoff"

    def __init__(self, E_young: float, nu: float):
        self.E_young = E_young
        self.nu = nu
        self.mu = E_young / (2.0 * (1.0 + nu))
        self.lam = E_young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    @classmethod
    def from_E_nu(cls, E: float, nu: float):
        """Create from Young's modulus and Poisson's ratio."""
        return cls(E_young=E, nu=nu)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _plane_stress_C(self):
        """Plane stress elasticity matrix (3x3 Voigt)."""
        E, nu = self.E_young, self.nu
        c = E / (1.0 - nu**2)
        return np.array([
            [c,      c * nu, 0            ],
            [c * nu, c,      0            ],
            [0,      0,      c * (1-nu)/2 ],
        ])

    def _green_lagrange_2d(self, F):
        """Green-Lagrange strain E = (F^T F - I)/2."""
        return 0.5 * (F.T @ F - np.eye(2))

    def _E_to_voigt(self, E):
        """Symmetric 2x2 to Voigt [E11, E22, 2*E12]."""
        return np.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]])

    def _voigt_to_S(self, S_v):
        """Voigt [S11, S22, S12] to symmetric 2x2."""
        return np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])

    def _build_C4(self, C_mat):
        """Build 4th-order elasticity tensor from 3x3 Voigt matrix."""
        C4 = np.zeros((2, 2, 2, 2))
        voigt_map = [(0, 0), (1, 1), (0, 1)]
        for a in range(3):
            I, J = voigt_map[a]
            for b in range(3):
                K, L = voigt_map[b]
                val = C_mat[a, b]
                C4[I, J, K, L] = val
                C4[J, I, K, L] = val
                C4[I, J, L, K] = val
                C4[J, I, L, K] = val
        return C4

    # -----------------------------------------------------------------
    # PK1 stress
    # -----------------------------------------------------------------

    def pk1_stress_2d(self, F, plane_stress=True):
        """
        PK1 stress: P = F S, where S = C:E.

        Args:
            F: (2, 2) deformation gradient
            plane_stress: if True, use plane stress C matrix

        Returns:
            P: (2, 2) First PK stress
            lambda3: 1.0 (interface compatibility)
        """
        E_gl = self._green_lagrange_2d(F)
        E_voigt = self._E_to_voigt(E_gl)

        if plane_stress:
            C_mat = self._plane_stress_C()
        else:
            raise NotImplementedError("Plane strain SVK not yet implemented")

        S_voigt = C_mat @ E_voigt
        S = self._voigt_to_S(S_voigt)
        P = F @ S
        return P, 1.0

    # -----------------------------------------------------------------
    # PK1 stress and tangent (for Newton-Raphson)
    # -----------------------------------------------------------------

    def pk1_stress_and_tangent_2d(self, F, plane_stress=True):
        """
        PK1 stress and tangent for SVK.

        Tangent:
            A_iJkL = dP_iJ/dF_kL
                   = delta_ik S_JL  +  F_iM C_MJNP dE_NP/dF_kL

        where:
            dE_NP/dF_kL = (1/2)(delta_NL F_kP + F_kN delta_PL)

        Args:
            F: (2, 2) deformation gradient

        Returns:
            P: (2, 2) First PK stress
            A: (2, 2, 2, 2) tangent dP/dF
            lambda3: 1.0 (interface compatibility)
        """
        E_gl = self._green_lagrange_2d(F)
        E_voigt = self._E_to_voigt(E_gl)

        if plane_stress:
            C_mat = self._plane_stress_C()
        else:
            raise NotImplementedError("Plane strain SVK not yet implemented")

        S_voigt = C_mat @ E_voigt
        S = self._voigt_to_S(S_voigt)
        P = F @ S

        C4 = self._build_C4(C_mat)

        A = np.zeros((2, 2, 2, 2))
        for i in range(2):
            for J_idx in range(2):
                for k in range(2):
                    for L_idx in range(2):
                        # Geometric part: delta_ik S_JL
                        val = float(i == k) * S[J_idx, L_idx]
                        # Material part: F_iM C_MJNP dE_NP/dF_kL
                        for M in range(2):
                            for N in range(2):
                                for P_idx in range(2):
                                    dE = 0.5 * (
                                        float(N == L_idx) * F[k, P_idx]
                                        + F[k, N] * float(P_idx == L_idx)
                                    )
                                    val += F[i, M] * C4[M, J_idx, N, P_idx] * dE
                        A[i, J_idx, k, L_idx] = val

        return P, A, 1.0

    # -----------------------------------------------------------------
    # 2nd Piola-Kirchhoff stress and material tangent in Voigt form
    # (for the B-matrix / Total Lagrangian formulation)
    # -----------------------------------------------------------------

    def pk2_voigt_and_tangent_2d(self, F, plane_stress=True):
        """
        2nd Piola-Kirchhoff stress and material tangent in Voigt notation.

        Intended for the B-matrix Total Lagrangian formulation:

            S_voigt = [S_XX, S_YY, S_XY]   (2nd PK stress)
            C_voigt = dS/dE                (3x3 Voigt material tangent)

        For St. Venant-Kirchhoff the law is linear in the Green-Lagrange
        strain, S = C:E, so the material tangent C_voigt is the *constant*
        plane-stress elasticity matrix. All displacement dependence enters
        the element matrices through B_L (via A_L) and the geometric term.

        Args:
            F: (2, 2) deformation gradient
            plane_stress: if True, use the plane-stress C matrix

        Returns:
            S_voigt: (3,) [S_XX, S_YY, S_XY]
            C_voigt: (3, 3) material tangent dS/dE in Voigt form
        """
        E_gl = self._green_lagrange_2d(F)
        E_voigt = self._E_to_voigt(E_gl)

        if plane_stress:
            C_mat = self._plane_stress_C()
        else:
            raise NotImplementedError("Plane strain SVK not yet implemented")

        S_voigt = C_mat @ E_voigt
        return S_voigt, C_mat

    # -----------------------------------------------------------------
    # Cauchy stress
    # -----------------------------------------------------------------

    def cauchy_stress_2d(self, F, plane_stress=True):
        """Cauchy stress sigma = (1/J) P F^T."""
        P, _ = self.pk1_stress_2d(F, plane_stress)
        J = np.linalg.det(F)
        return (1.0 / J) * P @ F.T

    # -----------------------------------------------------------------
    # Linear interface (small-strain compatibility)
    # -----------------------------------------------------------------

    def tangent(self, dim: int = 2, strain=None):
        """Linear tangent (for small-strain compatibility)."""
        if dim == 2:
            return self._plane_stress_C()
        else:
            raise NotImplementedError("3D tangent not implemented")

    def stress(self, strain, dim=2):
        """Small-strain stress sigma = C epsilon (Voigt)."""
        return self.tangent(dim=dim) @ strain