"""
SPH smoothing kernel functions.

Each kernel provides ``W(r, h)`` and ``dW_dr(r, h)`` where *r* is the
inter-particle distance and *h* is the smoothing length.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class SPHKernel(ABC):
    """Base class for SPH smoothing kernels."""

    @abstractmethod
    def W(self, r: float | np.ndarray, h: float) -> float | np.ndarray:
        """Kernel value."""
        ...

    @abstractmethod
    def dW_dr(self, r: float | np.ndarray, h: float) -> float | np.ndarray:
        """Kernel gradient magnitude (dW/dr)."""
        ...


class CubicSplineKernel(SPHKernel):
    """Standard cubic spline kernel (Monaghan, 1992)."""

    def _sigma(self, dim: int, h: float) -> float:
        if dim == 1:
            return 2 / (3 * h)
        elif dim == 2:
            return 10 / (7 * np.pi * h**2)
        else:
            return 1 / (np.pi * h**3)

    def W(self, r: float | np.ndarray, h: float, dim: int = 2) -> float | np.ndarray:
        q = np.asarray(r) / h
        sigma = self._sigma(dim, h)
        val = np.where(
            q <= 1,
            sigma * (1 - 1.5 * q**2 + 0.75 * q**3),
            np.where(q <= 2, sigma * 0.25 * (2 - q) ** 3, 0.0),
        )
        return float(val) if np.ndim(r) == 0 else val

    def dW_dr(self, r: float | np.ndarray, h: float, dim: int = 2) -> float | np.ndarray:
        q = np.asarray(r) / h
        sigma = self._sigma(dim, h)
        val = np.where(
            q <= 1,
            sigma / h * (-3 * q + 2.25 * q**2),
            np.where(q <= 2, sigma / h * (-0.75 * (2 - q) ** 2), 0.0),
        )
        return float(val) if np.ndim(r) == 0 else val
