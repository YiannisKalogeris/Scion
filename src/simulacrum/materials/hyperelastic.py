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
    """Compressible Neo-Hookean hyperelastic model.

    Parameters
    ----------
    mu : float   Shear modulus
    kappa : float  Bulk modulus
    """

    name = "NeoHookean"

    def __init__(self, mu: float, kappa: float):
        self.mu = mu
        self.kappa = kappa

    def tangent(self, dim: int = 3, strain: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError("NeoHookean tangent: requires deformation gradient F")

    def stress(self, strain: np.ndarray, dim: int = 3) -> np.ndarray:
        raise NotImplementedError("NeoHookean stress: requires deformation gradient F")
