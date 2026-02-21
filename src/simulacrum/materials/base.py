"""
Abstract base for constitutive models.

Every material must provide:
- ``tangent(dim)``  → the constitutive (tangent) matrix C
- ``stress(strain)`` → stress from a given strain state

For nonlinear materials the tangent is strain-dependent; for linear
materials it is constant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Material(ABC):
    """Base class for all constitutive models."""

    name: str = "Material"

    @abstractmethod
    def tangent(self, dim: int, strain: np.ndarray | None = None) -> np.ndarray:
        """Return the tangent modulus matrix (Voigt notation).

        Parameters
        ----------
        dim : int
            Spatial dimension (2 or 3).
        strain : array, optional
            Current strain state (needed for nonlinear models).

        Returns
        -------
        C : (n_voigt, n_voigt) array
        """
        ...

    @abstractmethod
    def stress(self, strain: np.ndarray, dim: int) -> np.ndarray:
        """Compute stress from strain (Voigt notation)."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
