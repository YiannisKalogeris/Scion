"""Thermal conductivity models for heat-transfer problems."""

from __future__ import annotations

import numpy as np

from simulacrum.materials.base import Material


class IsotropicThermal(Material):
    """Isotropic thermal conductivity.

    Parameters
    ----------
    k : float
        Thermal conductivity [W/(m·K)].
    rho : float
        Density [kg/m^3] (needed for transient problems).
    cp : float
        Specific heat capacity [J/(kg·K)].
    """

    name = "IsotropicThermal"

    def __init__(self, k: float, rho: float = 1.0, cp: float = 1.0):
        self.conductivity = k
        self.rho = rho
        self.cp = cp

    @property
    def diffusivity(self) -> float:
        return self.conductivity / (self.rho * self.cp)

    def tangent(self, dim: int = 2, strain: np.ndarray | None = None) -> np.ndarray:
        return self.conductivity * np.eye(dim)

    def stress(self, strain: np.ndarray, dim: int = 2) -> np.ndarray:
        """For thermal: 'stress' ≡ heat flux q = -k * grad(T)."""
        return -self.conductivity * strain

    def __repr__(self) -> str:
        return f"IsotropicThermal(k={self.conductivity})"
