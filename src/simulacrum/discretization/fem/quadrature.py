"""
Gauss quadrature rules for reference elements.

Provides ``(points, weights)`` for line, triangle, quad, tet, and hex domains.
"""

from __future__ import annotations

import numpy as np


def gauss_1d(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss–Legendre points and weights on [-1, 1]."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    return pts.reshape(-1, 1), wts


def gauss_triangle(order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature on the reference triangle {(s,t) : s,t >= 0, s+t <= 1}."""
    if order == 1:
        pts = np.array([[1 / 3, 1 / 3]])
        wts = np.array([0.5])
    elif order == 2:
        pts = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        wts = np.full(3, 1 / 6)
    else:
        raise NotImplementedError(f"Triangle quadrature order {order}")
    return pts, wts


def gauss_quad(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss quadrature on [-1,1]^2."""
    pts1, wts1 = np.polynomial.legendre.leggauss(n)
    pts = np.array([[xi, eta] for xi in pts1 for eta in pts1])
    wts = np.array([wi * wj for wi in wts1 for wj in wts1])
    return pts, wts


def gauss_hex(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product Gauss quadrature on [-1,1]^3."""
    pts1, wts1 = np.polynomial.legendre.leggauss(n)
    pts = np.array([[x, y, z] for x in pts1 for y in pts1 for z in pts1])
    wts = np.array([wx * wy * wz for wx in wts1 for wy in wts1 for wz in wts1])
    return pts, wts


# =====================================================================
# Registry: element-type → default quadrature
# =====================================================================

def get_quadrature(etype: str, order: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return default quadrature for a given element type."""
    defaults = {
        "line2": lambda: gauss_1d(2),
        "tri3": lambda: gauss_triangle(1),
        "quad4": lambda: gauss_quad(2),
        "hex8": lambda: gauss_hex(2),
    }
    if etype not in defaults:
        raise NotImplementedError(f"No default quadrature for '{etype}'")
    return defaults[etype]()
