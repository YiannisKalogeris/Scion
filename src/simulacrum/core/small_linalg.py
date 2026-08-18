"""
Closed-form linear algebra for stacks of small (2x2, 3x3) matrices.

Deliberately avoids ``np.linalg``:

* it is several times faster than ``np.linalg.inv`` on a stack of 10^5 tiny
  matrices, which is the regime particle methods live in;
* every routine maps 1:1 onto ``wp.mat22`` / ``wp.mat33`` intrinsics, so the
  Warp port of any caller is mechanical rather than a rewrite.

All functions take arrays shaped ``(n, d, d)``.

Singular matrices are not checked
---------------------------------
:func:`inv` divides by the determinant and does nothing else. A singular input
gives ``±inf`` or ``nan`` with a numpy warning, and that is deliberate: the
callers are per-particle loops running at 1e5 matrices a step, and a check
there would cost more than the inversion. The physical guard belongs upstream --
a deformation gradient with ``J <= 0`` means the material has inverted, which is
a modelling failure the constitutive model should catch and report, not
something to paper over with a pseudo-inverse.
"""

from __future__ import annotations

import numpy as np


def det(F: np.ndarray) -> np.ndarray:
    """Determinants of a stack of 2x2 or 3x3 matrices. ``(n, d, d) -> (n,)``."""
    d = F.shape[-1]
    if d == 2:
        return F[:, 0, 0] * F[:, 1, 1] - F[:, 0, 1] * F[:, 1, 0]
    if d == 3:
        return (
            F[:, 0, 0] * (F[:, 1, 1] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 1])
            - F[:, 0, 1] * (F[:, 1, 0] * F[:, 2, 2] - F[:, 1, 2] * F[:, 2, 0])
            + F[:, 0, 2] * (F[:, 1, 0] * F[:, 2, 1] - F[:, 1, 1] * F[:, 2, 0])
        )
    raise ValueError(f"det: expected 2x2 or 3x3, got {d}x{d}")


def inv(F: np.ndarray, J: np.ndarray | None = None) -> np.ndarray:
    """Inverses of a stack of 2x2 or 3x3 matrices.

    Pass ``J`` when the determinants are already known, to avoid recomputing.
    """
    if J is None:
        J = det(F)
    d = F.shape[-1]
    out = np.empty_like(F)
    if d == 2:
        out[:, 0, 0] = F[:, 1, 1]
        out[:, 1, 1] = F[:, 0, 0]
        out[:, 0, 1] = -F[:, 0, 1]
        out[:, 1, 0] = -F[:, 1, 0]
    elif d == 3:
        for i in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            for j in range(3):
                j1, j2 = (j + 1) % 3, (j + 2) % 3
                # adjugate = transpose of the cofactor matrix
                out[:, j, i] = (
                    F[:, i1, j1] * F[:, i2, j2] - F[:, i1, j2] * F[:, i2, j1]
                )
    else:
        raise ValueError(f"inv: expected 2x2 or 3x3, got {d}x{d}")
    return out / J[:, None, None]


def inv_transpose(F: np.ndarray, J: np.ndarray | None = None) -> np.ndarray:
    """``F^-T`` for a stack of small matrices."""
    return np.swapaxes(inv(F, J), 1, 2)


def transpose(F: np.ndarray) -> np.ndarray:
    """``F^T`` for a stack of small matrices."""
    return np.swapaxes(F, 1, 2)


def identity(n: int, dim: int, dtype=np.float64) -> np.ndarray:
    """A stack of ``n`` identity matrices."""
    return np.tile(np.eye(dim, dtype=dtype), (n, 1, 1))
