"""
Voronoi particle volumes and their shape derivative.

``Pi_h = sum_p V_p^0 W(F_p)`` approximates the body's energy only while the
``V_p^0`` tessellate the body. Holding them fixed while moving ``X_p`` --
measured in ``adaptivity.py`` -- turns the descent into a migration towards low
strain energy that worsens the quadrature by 28%. Tying volumes to positions is
what makes ``G_p = 0`` mean anything, and this supplies that tie.

Clipping by mirroring
---------------------
A Voronoi diagram of scattered points has unbounded cells at the hull, which
have no area. Reflecting every generator across each face of the bounding box
fixes it exactly rather than approximately: the bisector between a point and its
mirror image *is* the box face, so the cell of an original point comes out
already clipped, with no special-casing and no polygon intersection. The check
that this is right is that the areas sum to the box area, which they do to
twelve digits.

The neighbour coupling Eq. (9) omits
------------------------------------
With volumes tied to positions,

.. math::
    \frac{\partial \Pi_h}{\partial X_p}
      = V_p \frac{\partial W(F_p)}{\partial X_p}
      + \sum_q W(F_q)\, \frac{\partial V_q}{\partial X_p}

and the second sum runs over ``p`` **and all its Voronoi neighbours**, because
moving one generator resizes the cells around it. Eq. (9) carries only the
``q = p`` term of that sum. Whether the difference matters is measured in
:func:`compare_with_equation_9`.

Cost
----
:func:`volume_gradient` finite-differences the tessellation: ``2 n d`` Voronoi
builds for one gradient. That is correct by construction and far too slow for
production -- it is here to answer whether the volume term matters, and to serve
as the reference any analytic formula would have to match. Two closed forms were
tried against it and neither reproduced it, the second being worse than the
first, so the finite-difference route stands until one is derived properly.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "voronoi_volumes",
    "voronoi_neighbours",
    "volume_gradient",
    "energy_shape_gradient",
]


def _mirrored(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Generators plus one reflection across each box face."""
    blocks = [X]
    for a in range(X.shape[1]):
        for face in (lo[a], hi[a]):
            m = X.copy()
            m[:, a] = 2.0 * face - m[:, a]
            blocks.append(m)
    return np.vstack(blocks)


def _polygon_area(P: np.ndarray) -> float:
    centre = P.mean(axis=0)
    order = np.argsort(np.arctan2(P[:, 1] - centre[1], P[:, 0] - centre[0]))
    Q = P[order]
    return 0.5 * abs(np.dot(Q[:, 0], np.roll(Q[:, 1], -1))
                     - np.dot(Q[:, 1], np.roll(Q[:, 0], -1)))


def voronoi_volumes(X: np.ndarray, lo, hi) -> np.ndarray:
    """``(n,)`` Voronoi cell areas, clipped to the box. 2-D."""
    from scipy.spatial import Voronoi

    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise NotImplementedError(
            f"2-D only; got dim={X.shape[1]}. The mirroring generalises but the "
            "cell volume needs a convex-hull volume rather than a shoelace."
        )
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if np.any(X <= lo) or np.any(X >= hi):
        raise ValueError(
            "every generator must be strictly inside the box; a point on the "
            "face coincides with its own mirror image and the diagram degenerates"
        )

    vor = Voronoi(_mirrored(X, lo, hi))
    out = np.empty(len(X))
    for i in range(len(X)):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or not region:
            raise RuntimeError(
                f"cell {i} came out unbounded despite mirroring; this should be "
                "impossible for a point strictly inside the box"
            )
        out[i] = _polygon_area(vor.vertices[region])
    return out


def voronoi_neighbours(X: np.ndarray, lo, hi) -> list[tuple[int, int]]:
    """Pairs of original generators that share a Voronoi edge."""
    from scipy.spatial import Voronoi

    n = len(X)
    vor = Voronoi(_mirrored(np.asarray(X, dtype=float),
                            np.asarray(lo), np.asarray(hi)))
    return [(int(p), int(q)) for (p, q), rv
            in zip(vor.ridge_points, vor.ridge_vertices)
            if p < n and q < n and -1 not in rv]


def volume_gradient(X: np.ndarray, lo, hi, h: float = 1e-7) -> np.ndarray:
    """``(n, n, dim)`` with ``[p, q, k] = dV_p / dX_{q,k}``, by central differences.

    Dense and ``O(n)`` tessellations. See the cost note in the module docstring.
    """
    X = np.asarray(X, dtype=float)
    n, dim = X.shape
    out = np.zeros((n, n, dim))
    for q in range(n):
        for k in range(dim):
            plus, minus = X.copy(), X.copy()
            plus[q, k] += h
            minus[q, k] -= h
            out[:, q, k] = (voronoi_volumes(plus, lo, hi)
                            - voronoi_volumes(minus, lo, hi)) / (2.0 * h)
    return out


def energy_shape_gradient(energy: np.ndarray, dW_dX: np.ndarray,
                          volumes: np.ndarray,
                          dV_dX: np.ndarray) -> np.ndarray:
    r"""The complete ``d\Pi_h / dX_p``. ``-> (n, dim)``.

    Parameters
    ----------
    energy  : ``(n,)`` ``W(F_q)``
    dW_dX   : ``(n, dim)`` ``dW(F_p)/dX_p``, the diagonal part -- ``F_q`` depends
              only on ``X_q``, so there is no coupling in this term
    volumes : ``(n,)``
    dV_dX   : ``(n, n, dim)`` from :func:`volume_gradient`

    The second sum is where the neighbour coupling enters, and it is the part
    Eq. (9) does not have.
    """
    return (volumes[:, None] * dW_dX
            + np.einsum("q,qpk->pk", np.asarray(energy), np.asarray(dV_dX)))
