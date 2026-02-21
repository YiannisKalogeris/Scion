"""
Finite-difference stencil definitions.

Each stencil is a dictionary mapping relative offsets → coefficients.
"""

from __future__ import annotations

import numpy as np


# =====================================================================
# 1-D stencils
# =====================================================================

def laplacian_1d(order: int = 2) -> dict[int, float]:
    """1-D Laplacian stencil: u''(x) ≈ (1/h^2) * sum(coeff * u(x + k*h))."""
    if order == 2:
        return {-1: 1.0, 0: -2.0, 1: 1.0}
    elif order == 4:
        return {-2: -1 / 12, -1: 4 / 3, 0: -5 / 2, 1: 4 / 3, 2: -1 / 12}
    raise NotImplementedError(f"Order-{order} 1-D Laplacian stencil")


# =====================================================================
# 2-D stencils (Kronecker product of 1-D stencils on structured grids)
# =====================================================================

def laplacian_2d_5pt() -> dict[tuple[int, int], float]:
    """Standard 5-point Laplacian on a uniform 2-D grid."""
    return {
        (0, 0): -4.0,
        (-1, 0): 1.0, (1, 0): 1.0,
        (0, -1): 1.0, (0, 1): 1.0,
    }


def laplacian_2d_9pt() -> dict[tuple[int, int], float]:
    """9-point compact Laplacian (Mehrstellen) on a uniform 2-D grid."""
    return {
        (0, 0): -20 / 6,
        (-1, 0): 4 / 6, (1, 0): 4 / 6,
        (0, -1): 4 / 6, (0, 1): 4 / 6,
        (-1, -1): 1 / 6, (-1, 1): 1 / 6,
        (1, -1): 1 / 6, (1, 1): 1 / 6,
    }
