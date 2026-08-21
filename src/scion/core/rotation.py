r"""
Finite rotations: SO(3) as batched unit quaternions.

Why quaternions and not the rotation vector
-------------------------------------------
A rotation vector ``phi`` (axis times angle) is the obvious parameterisation and
is wrong for anything that accumulates rotation. Three separate reasons, all of
which show up in a micropolar solve:

* **Rotations compose multiplicatively.** ``Q^{N+1} = dQ Q^N``, never
  ``phi^{N+1} = phi^N + dw``. The additive update is a first-order approximation
  that is exact only in 2-D or for infinitesimal rotations, and it degrades
  silently -- the answer stays plausible and drifts.
* **The axis becomes indeterminate at ``phi = n pi``.** Rebuilding ``Q`` from
  ``phi`` through the Rodrigues formula divides by ``phi``; a body that turns
  through a full revolution passes exactly through the singularity.
* **``arccos`` is not injective.** Recovering ``phi`` from ``Q`` can only return
  an angle in ``(-pi, pi]`` from the matrix logarithm, or ``(-2 pi, 2 pi]``
  through :func:`rotvec_from_quat`. Either way the rotation vector is an
  *output*, and using it as state caps the accumulated rotation.

So the state is the quaternion, and the rotation vector is only ever produced
for plotting. This mirrors O'Hare, Gourgiotis, Coombs & Augarde, CMAME 419
(2024) 116668, Section 3.2.2, whose torsion example exists to demonstrate a
shaft twisting past ``2 pi`` without ill conditioning.

Layout and conventions
----------------------
A quaternion is ``(n, 4)`` with the **scalar part first**: ``q[:, 0]`` is the
real component, ``q[:, 1:]`` the vector. Rotation matrices are ``(n, 3, 3)``.
``skew(v) @ w == cross(v, w)``, i.e. ``skew(v)_ij = -e_ijk v_k``.

Small angles are handled by Taylor series, not by clamping. Every function here
has a ``0/0`` at ``theta = 0`` and every one of those limits is finite; a naive
implementation returns ``nan`` on the very first step of any problem that starts
from rest, which is the most common way this is got wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "skew",
    "axial",
    "quat_identity",
    "quat_from_rotvec",
    "quat_multiply",
    "quat_to_matrix",
    "rotvec_from_quat",
    "spin_map",
]

#: below this angle the series are used instead of the closed forms. Chosen so
#: the truncated term is below machine precision relative to the leading one:
#: the series here are all in theta^2, so 1e-4 leaves a relative error of ~1e-9
#: in the *correction*, which is itself O(theta^2).
_SMALL = 1.0e-4


def skew(v: np.ndarray) -> np.ndarray:
    """``(n, 3) -> (n, 3, 3)`` with ``skew(v) @ w == cross(v, w)``."""
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    S = np.zeros((n, 3, 3), dtype=v.dtype)
    S[:, 0, 1] = -v[:, 2]
    S[:, 0, 2] = v[:, 1]
    S[:, 1, 0] = v[:, 2]
    S[:, 1, 2] = -v[:, 0]
    S[:, 2, 0] = -v[:, 1]
    S[:, 2, 1] = v[:, 0]
    return S


def axial(A: np.ndarray) -> np.ndarray:
    """Inverse of :func:`skew`, taking the skew part of a general ``A``."""
    A = np.asarray(A, dtype=float)
    return 0.5 * np.stack([A[:, 2, 1] - A[:, 1, 2],
                           A[:, 0, 2] - A[:, 2, 0],
                           A[:, 1, 0] - A[:, 0, 1]], axis=1)


def quat_identity(n: int, dtype=np.float64) -> np.ndarray:
    """``n`` copies of the identity rotation, ``(1, 0, 0, 0)``."""
    q = np.zeros((n, 4), dtype=dtype)
    q[:, 0] = 1.0
    return q


def quat_from_rotvec(w: np.ndarray) -> np.ndarray:
    """Quaternion of the rotation ``|w|`` about ``w``. ``(n, 3) -> (n, 4)``."""
    w = np.asarray(w, dtype=float)
    theta = np.linalg.norm(w, axis=1)
    small = theta < _SMALL
    # sin(theta/2)/theta, which is 1/2 at theta = 0 and not 0/0 in the limit
    half = np.where(small,
                    0.5 - theta ** 2 / 48.0,
                    np.sin(np.where(small, 1.0, theta) / 2.0)
                    / np.where(small, 1.0, theta))
    q = np.empty((w.shape[0], 4), dtype=w.dtype)
    q[:, 0] = np.cos(theta / 2.0)
    q[:, 1:] = half[:, None] * w
    return q


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p o q``, whose matrix is ``Q(p) Q(q)``.

    Order matters and is the single easiest thing to get backwards: a
    multiplicative update ``Q^{N+1} = dQ Q^N`` is ``quat_multiply(dq, q)``.
    Reversing it applies the increment in the *material* frame instead of the
    spatial one, which is a different (and also self-consistent) scheme -- so it
    does not blow up, it just answers a different question.
    """
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    p0, pv = p[:, 0], p[:, 1:]
    q0, qv = q[:, 0], q[:, 1:]
    out = np.empty(np.broadcast_shapes(p.shape, q.shape), dtype=float)
    out[:, 0] = p0 * q0 - np.einsum("ni,ni->n", pv, qv)
    out[:, 1:] = (p0[:, None] * qv + q0[:, None] * pv + np.cross(pv, qv))
    return out


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(n, 4) -> (n, 3, 3)``, ``Q = (2 q0^2 - 1) I + 2 q0 skew(qv)
    + 2 qv (x) qv``."""
    q = np.asarray(q, dtype=float)
    q0, qv = q[:, 0], q[:, 1:]
    return ((2.0 * q0 ** 2 - 1.0)[:, None, None] * np.eye(3)
            + 2.0 * q0[:, None, None] * skew(qv)
            + 2.0 * np.einsum("ni,nj->nij", qv, qv))


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """``(n, 4) -> (n, 3)``. **Output only** -- see the module docstring.

    Covers ``(-2 pi, 2 pi]`` rather than the matrix logarithm's ``(-pi, pi]``,
    and is still not injective. At a whole revolution the axis is genuinely
    indeterminate and the components returned there are meaningless; that is a
    property of the representation, not of the caller's rotation.
    """
    q = np.asarray(q, dtype=float)
    q0 = np.clip(q[:, 0], -1.0, 1.0)
    s = np.sqrt(np.maximum(1.0 - q0 ** 2, 0.0))
    scale = np.where(s < _SMALL, 2.0, 2.0 * np.arccos(q0) / np.where(s < _SMALL, 1.0, s))
    return scale[:, None] * q[:, 1:]


def spin_map(w: np.ndarray) -> np.ndarray:
    """Right-trivialised differential of the exponential map.

    ``H = I + (1 - cos t)/t^2 skew(w) + (t - sin t)/t^3 skew(w)^2``.

    Measured, not assumed: with ``skew(v) w = v x w``, this ``H`` satisfies
    ``dw_spatial = H dphi`` -- it is verified in the tests against
    ``axial(dQ Q^T)`` for ``Q = exp(skew(phi))``. The transpose is the *body*
    map ``H(-w)``, and the reference (O'Hare et al. Eq. 55) states the relation
    with a transpose the sign pattern of its own Eq. 55 does not support. The
    distinction only bites at finite increment size, since ``H -> I`` as
    ``w -> 0``, so it changes the accuracy of a wryness update and never the
    consistent tangent; ``test_micropolar.py`` settles it by accumulating the
    update against the definition of the wryness tensor.
    """
    w = np.asarray(w, dtype=float)
    t = np.linalg.norm(w, axis=1)
    small = t < _SMALL
    ts = np.where(small, 1.0, t)
    c1 = np.where(small, 0.5 - t ** 2 / 24.0, (1.0 - np.cos(ts)) / ts ** 2)
    c2 = np.where(small, 1.0 / 6.0 - t ** 2 / 120.0,
                  (ts - np.sin(ts)) / ts ** 3)
    S = skew(w)
    return (np.eye(3) + c1[:, None, None] * S
            + c2[:, None, None] * np.einsum("nij,njk->nik", S, S))
