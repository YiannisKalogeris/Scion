"""
Linear elastic constitutive models.

Provides full 3-D isotropic elasticity, and plane-stress / plane-strain
specialisations for 2-D problems.
"""

from __future__ import annotations

import numpy as np

from simulacrum.materials.base import Material


def isotropic_stiffness_mandel(E: float, nu: float) -> np.ndarray:
    """Isotropic elastic stiffness as a ``(6, 6)`` **Mandel** matrix.

    Distinct from :meth:`LinearElastic3D.tangent`, which returns the Voigt form
    that pairs with an engineering-shear strain vector. The two differ on the
    shear block -- Voigt carries ``mu`` there and Mandel ``2 mu`` -- so the name
    says which one it is rather than leaving it to the caller to remember.
    Mandel is what the constitutive integrators use, because it turns tensor
    contractions into ordinary matrix algebra; see
    :mod:`simulacrum.materials.notation`.

    Always the full 3-D operator, whatever the element dimension. Plane strain
    is a constraint on the *strain* (``eps_33 = 0``), not a different material,
    and imposing it by deleting rows would lose ``sigma_33`` -- which is nonzero
    and enters every pressure-sensitive yield function.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = 2.0 * mu
    return C


class LinearElastic3D(Material):
    """Isotropic linear elasticity (3-D).

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    """

    name = "LinearElastic3D"

    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    def tangent(self, dim: int = 3, strain: np.ndarray | None = None) -> np.ndarray:
        if dim != 3:
            raise ValueError(f"{self.name} only supports dim=3, got {dim}")

        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        return np.array([
            [lam + 2 * mu, lam,          lam,          0,  0,  0],
            [lam,          lam + 2 * mu, lam,          0,  0,  0],
            [lam,          lam,          lam + 2 * mu, 0,  0,  0],
            [0,            0,            0,            mu, 0,  0],
            [0,            0,            0,            0,  mu, 0],
            [0,            0,            0,            0,  0,  mu],
        ])

    def stress(self, strain: np.ndarray, dim: int = 3) -> np.ndarray:
        return self.tangent(dim=dim) @ strain

    def stiffness_mandel(self) -> np.ndarray:
        """The same operator in Mandel notation, for constitutive integrators."""
        return isotropic_stiffness_mandel(self.E, self.nu)

    def __repr__(self) -> str:
        return f"LinearElastic3D(E={self.E}, nu={self.nu})"


class PlaneStress(Material):
    """Isotropic linear elasticity under plane-stress conditions."""

    name = "PlaneStress"

    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    def tangent(self, dim: int = 2, strain: np.ndarray | None = None) -> np.ndarray:
        if dim != 2:
            raise ValueError(f"{self.name} only supports dim=2, got {dim}")

        E, nu = self.E, self.nu
        factor = E / (1 - nu**2)

        return factor * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1 - nu) / 2],
        ])

    def stress(self, strain: np.ndarray, dim: int = 2) -> np.ndarray:
        return self.tangent(dim=dim) @ strain

    def __repr__(self) -> str:
        return f"PlaneStress(E={self.E}, nu={self.nu})"


class PlaneStrain(Material):
    """Isotropic linear elasticity under plane-strain conditions."""

    name = "PlaneStrain"

    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    def tangent(self, dim: int = 2, strain: np.ndarray | None = None) -> np.ndarray:
        if dim != 2:
            raise ValueError(f"{self.name} only supports dim=2, got {dim}")

        E, nu = self.E, self.nu
        factor = E / ((1 + nu) * (1 - 2 * nu))

        return factor * np.array([
            [1 - nu, nu,     0],
            [nu,     1 - nu, 0],
            [0,      0,      (1 - 2 * nu) / 2],
        ])

    def stress(self, strain: np.ndarray, dim: int = 2) -> np.ndarray:
        return self.tangent(dim=dim) @ strain

    def __repr__(self) -> str:
        return f"PlaneStrain(E={self.E}, nu={self.nu})"
