"""
Linear elastic constitutive models.

Provides full 3-D isotropic elasticity, and plane-stress / plane-strain
specialisations for 2-D problems.
"""

from __future__ import annotations

import numpy as np

from simulacrum.materials.base import Material


class LinearElastic(Material):
    """Isotropic linear elasticity (3-D).

    Parameters
    ----------
    E : float   Young's modulus
    nu : float  Poisson's ratio
    """

    name = "LinearElastic"

    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu

    def tangent(self, dim: int = 3, strain: np.ndarray | None = None) -> np.ndarray:
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        if dim == 3:
            C = np.array([
                [lam + 2 * mu, lam,          lam,          0,  0,  0],
                [lam,          lam + 2 * mu, lam,          0,  0,  0],
                [lam,          lam,          lam + 2 * mu, 0,  0,  0],
                [0,            0,            0,            mu, 0,  0],
                [0,            0,            0,            0,  mu, 0],
                [0,            0,            0,            0,  0,  mu],
            ])
        elif dim == 2:
            # Default to plane strain for 2-D
            C = self._plane_strain()
        else:
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        return C

    def _plane_strain(self) -> np.ndarray:
        E, nu = self.E, self.nu
        factor = E / ((1 + nu) * (1 - 2 * nu))
        return factor * np.array([
            [1 - nu, nu,     0],
            [nu,     1 - nu, 0],
            [0,      0,      (1 - 2 * nu) / 2],
        ])

    def stress(self, strain: np.ndarray, dim: int = 3) -> np.ndarray:
        return self.tangent(dim) @ strain

    def __repr__(self) -> str:
        return f"LinearElastic(E={self.E}, nu={self.nu})"


class PlaneStress(LinearElastic):
    """Plane-stress specialisation."""

    name = "PlaneStress"

    def tangent(self, dim: int = 2, strain: np.ndarray | None = None) -> np.ndarray:
        E, nu = self.E, self.nu
        factor = E / (1 - nu**2)
        return factor * np.array([
            [1,  nu, 0],
            [nu, 1,  0],
            [0,  0,  (1 - nu) / 2],
        ])


class PlaneStrain(LinearElastic):
    """Plane-strain specialisation."""

    name = "PlaneStrain"

    def tangent(self, dim: int = 2, strain: np.ndarray | None = None) -> np.ndarray:
        return self._plane_strain()
