r"""
SPH smoothing kernels.

Conventions, because they are the first thing to get wrong
----------------------------------------------------------
``q = r/h`` and the support is ``q <= cutoff_ratio``, so a kernel with
``cutoff_ratio = 2`` reaches ``2h`` and *not* ``h``. The literature is split
on whether ``h`` is the support radius or the support is ``2h``; SPHinXsys
uses the second, and so does this module, because the whole point of the
2-D heat-transfer example is to reproduce theirs.

``normalization`` is the dimension-dependent factor that makes
``INT W dV = 1`` over the whole space. It is *not* interchangeable between
dimensions even when the radial shape is -- the Wendland C2 shape function is
literally the same expression in 1-D, 2-D and 3-D and its three constants are
all different, which is exactly the kind of thing that produces a plausible
answer that is wrong by a factor.

``dW_dr`` is the derivative with respect to the *distance*, so the gradient of
the kernel is ``dW_dr(r) * (x_i - x_j)/r``. It carries one ``1/h`` more than
``W`` does; dropping it leaves a gradient that is wrong by the smoothing
length and right in its units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["SPHKernel", "WendlandC2", "CubicSpline", "KERNELS", "get_kernel"]


class SPHKernel(ABC):
    """A radial kernel in one, two or three dimensions.

    Subclasses supply the shape function of ``q`` and its derivative, plus the
    three normalizing constants; everything public is built from those, so a
    new kernel cannot disagree with itself about where the support ends.
    """

    #: support radius in units of the smoothing length
    cutoff_ratio: float = 2.0
    name: str = "kernel"

    @abstractmethod
    def _w(self, q: np.ndarray) -> np.ndarray:
        """The shape function, normalized so ``_w(0)`` is finite and O(1)."""

    @abstractmethod
    def _dw(self, q: np.ndarray) -> np.ndarray:
        """``d _w / dq``."""

    @abstractmethod
    def _factor(self, dim: int) -> float:
        """The normalizing constant for ``h = 1``."""

    def normalization(self, dim: int, h: float) -> float:
        if dim not in (1, 2, 3):
            raise ValueError(f"dimension must be 1, 2 or 3, got {dim}")
        return self._factor(dim) / h**dim

    def support(self, h: float) -> float:
        return self.cutoff_ratio * h

    def W(self, r, h: float, dim: int = 2):
        """Kernel value at distance ``r``; ``r`` is a distance, so ``|r|``."""
        q = np.abs(np.asarray(r, dtype=float)) / h
        out = np.where(q < self.cutoff_ratio,
                       self.normalization(dim, h) * self._w(np.minimum(q, self.cutoff_ratio)),
                       0.0)
        return float(out) if np.ndim(r) == 0 else out

    def dW_dr(self, r, h: float, dim: int = 2):
        """``dW/dr`` at distance ``r``; negative on the whole support.

        Taken at ``|r|``, as ``W`` is, because the gradient is assembled as
        ``dW_dr(r) * (x_i - x_j)/r`` and the sign lives in that factor.
        """
        q = np.abs(np.asarray(r, dtype=float)) / h
        out = np.where(q < self.cutoff_ratio,
                       self.normalization(dim, h) / h
                       * self._dw(np.minimum(q, self.cutoff_ratio)),
                       0.0)
        return float(out) if np.ndim(r) == 0 else out


class WendlandC2(SPHKernel):
    r"""Wendland's C2 kernel, ``(1 - q/2)^4 (1 + 2q)`` on ``q <= 2``.

    SPHinXsys's default, and the one the heat-transfer example runs on. The
    shape is the same in every dimension; only the constant changes.
    """

    cutoff_ratio = 2.0
    name = "wendland_c2"

    def _w(self, q):
        return (1.0 - 0.5 * q) ** 4 * (1.0 + 2.0 * q)

    def _dw(self, q):
        # d/dq [(1-q/2)^4 (1+2q)] = -5 q (1-q/2)^3, written as SPHinXsys does
        return 0.625 * (q - 2.0) ** 3 * q

    def _factor(self, dim: int) -> float:
        return {1: 0.75, 2: 7.0 / (4.0 * np.pi), 3: 21.0 / (16.0 * np.pi)}[dim]


class CubicSpline(SPHKernel):
    """The Monaghan (1992) cubic B-spline, support ``2h``.

    Kept beside Wendland because a kernel-independent check -- partition of
    unity, first-order consistency after correction -- tests the framework
    only if it runs on more than one kernel. Its second derivative changes
    sign inside the support, which is the classical pairing-instability
    argument for preferring Wendland; nothing here depends on that.
    """

    cutoff_ratio = 2.0
    name = "cubic_spline"

    def _w(self, q):
        return np.where(q < 1.0,
                        1.0 - 1.5 * q**2 + 0.75 * q**3,
                        0.25 * (2.0 - q) ** 3)

    def _dw(self, q):
        return np.where(q < 1.0,
                        -3.0 * q + 2.25 * q**2,
                        -0.75 * (2.0 - q) ** 2)

    def _factor(self, dim: int) -> float:
        return {1: 2.0 / 3.0, 2: 10.0 / (7.0 * np.pi), 3: 1.0 / np.pi}[dim]


KERNELS = {k.name: k for k in (WendlandC2(), CubicSpline())}


def get_kernel(name: str | SPHKernel) -> SPHKernel:
    if isinstance(name, SPHKernel):
        return name
    try:
        return KERNELS[name]
    except KeyError:
        raise KeyError(f"unknown kernel {name!r}; have {sorted(KERNELS)}") from None
