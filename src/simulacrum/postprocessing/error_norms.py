"""
Consistent error norms for direct and pullback finite element solutions.

The point of this module is that the *same* functional must be evaluated for
both discretisations, or the comparison in a results table is not a comparison.

Why nodal RMS is not good enough
--------------------------------
``sqrt(mean((u_h - u_exact)**2))`` over the degrees of freedom is a discrete
l2 norm weighted by node density. For a direct solve on a body-fitted mesh that
density is roughly uniform in the physical domain, so the value happens to
approximate the L2 norm up to a constant. For a pullback solve it is weighted by
*reference*-square node density, and the map deliberately compresses and expands
regions -- so the same number means something different. On a 30x30 ellipse
pullback the two differ by more than a factor of two.

The norm computed here is the genuine

    ||u_h - u||_L2(Omega)^2 = INT_Omega (u_h - u)^2 dx

evaluated by quadrature. On the physical mesh that is the usual element loop.
On the reference square the change of variables gives

    INT_Omega (u_h - u)^2 dx = INT_square (uhat_h(xi) - u(psi(xi)))^2 J(xi) dxi

so the *same* Jacobian determinant that appears in the load vector reappears in
the error norm. Forgetting it is the most common way to make a pullback look
better than it is.

Quadrature order
----------------
Use a higher rule for the error than for assembly. The error field oscillates at
the element scale and vanishes at the nodes for a nodally-interpolant solution,
so a 2x2 rule on Q1 elements systematically under-reports it. ``order=4``
(4x4 Gauss) is cheap and safe; the default here.

Usage
-----
    from error_norms import l2_error_direct, l2_error_pullback

    e_abs, e_rel, e_inf = l2_error_direct(mesh_d, u_d, u_exact_fn)
    e_abs, e_rel, e_inf = l2_error_pullback(mesh_sq, u_sq, mapping, u_exact_fn)

``u_exact_fn`` takes an ``(n, 2)`` array of physical coordinates and returns
``(n,)`` values.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gauss_rule",
    "l2_error_direct",
    "l2_error_pullback",
    "l2_norm_exact",
]


# ---------------------------------------------------------------------------
# Quadrature and shape functions
# ---------------------------------------------------------------------------

def gauss_rule(etype: str, order: int = 4):
    """High-order rule for error evaluation, independent of the assembly rule."""
    if etype in ("quad4", "quad8", "quad9"):
        p, w = np.polynomial.legendre.leggauss(order)
        pts = np.array([[a, b] for a in p for b in p])
        wts = np.array([wa * wb for wa in w for wb in w])
        return pts, wts
    if etype in ("tri3", "tri6"):
        # symmetric 6-point rule, degree 4 -- enough for a quadratic error field
        a, b = 0.445948490915965, 0.091576213509771
        w1, w2 = 0.111690794839005, 0.054975871827661
        pts = np.array([[a, a], [1 - 2 * a, a], [a, 1 - 2 * a],
                        [b, b], [1 - 2 * b, b], [b, 1 - 2 * b]])
        wts = np.array([w1, w1, w1, w2, w2, w2])
        return pts, wts
    raise NotImplementedError(f"no error rule for element type '{etype}'")


def _shape(etype: str, xi):
    """(N, dN_dxi) for the element types used here."""
    if etype == "quad4":
        s, t = float(xi[0]), float(xi[1])
        N = 0.25 * np.array([(1 - s) * (1 - t), (1 + s) * (1 - t),
                             (1 + s) * (1 + t), (1 - s) * (1 + t)])
        dN = 0.25 * np.array([[-(1 - t), -(1 - s)], [(1 - t), -(1 + s)],
                              [(1 + t), (1 + s)], [-(1 + t), (1 - s)]])
        return N, dN
    if etype == "tri3":
        s, t = float(xi[0]), float(xi[1])
        N = np.array([1 - s - t, s, t])
        dN = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        return N, dN
    raise NotImplementedError(f"no shape functions for '{etype}'")


# ---------------------------------------------------------------------------
# Direct solve: mesh lives in the physical domain
# ---------------------------------------------------------------------------

def l2_error_direct(mesh, u, u_exact_fn, order: int = 4):
    """``(abs, rel, inf)`` error of a solution on a body-fitted mesh.

    ``inf`` is the maximum over quadrature points, not over nodes: a nodally
    exact interpolant can have a large interior error that a nodal maximum
    never sees.
    """
    num = den = 0.0
    einf = 0.0

    for block in mesh.element_blocks:
        qpts, qwts = gauss_rule(block.etype, order)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh.nodes[enodes]
            ue = u[enodes]
            for qp, qw in zip(qpts, qwts):
                N, dN = _shape(block.etype, qp)
                detJ = abs(np.linalg.det(dN.T @ coords))
                x = N @ coords
                diff = float(N @ ue) - float(u_exact_fn(x[None, :])[0])
                num += diff ** 2 * detJ * qw
                den += float(u_exact_fn(x[None, :])[0]) ** 2 * detJ * qw
                einf = max(einf, abs(diff))

    return np.sqrt(num), np.sqrt(num / den) if den > 0 else np.nan, einf


# ---------------------------------------------------------------------------
# Pullback solve: mesh lives on the reference square
# ---------------------------------------------------------------------------

def l2_error_pullback(mesh_sq, u_sq, mapping, u_exact_fn, order: int = 4,
                      batch: int = 200_000):
    """``(abs, rel, inf)`` error of a pullback solution, measured on Omega.

    ``mapping`` must expose ``psi(y_torch) -> x`` and
    ``jacobian_matrix(y_torch) -> F``, i.e. the wrapper already used to build
    the metric. The map is evaluated at the *error* quadrature points, which are
    not the assembly points, so it cannot reuse the cached metric.
    """
    import torch

    # ---- collect all error quadrature points on the square -----------------
    ys, dets_iso, weights, elems = [], [], [], []
    for block in mesh_sq.element_blocks:
        qpts, qwts = gauss_rule(block.etype, order)
        for e in range(block.connectivity.shape[0]):
            enodes = block.connectivity[e]
            coords = mesh_sq.nodes[enodes]
            for qp, qw in zip(qpts, qwts):
                N, dN = _shape(block.etype, qp)
                ys.append(N @ coords)
                dets_iso.append(abs(np.linalg.det(dN.T @ coords)) * qw)
                weights.append(float(N @ u_sq[enodes]))
                elems.append(0)

    ys = np.asarray(ys, dtype=np.float32)
    dets_iso = np.asarray(dets_iso)
    uh = np.asarray(weights)

    # ---- map them, in batches, and get the Jacobian determinant ------------
    x_phys = np.empty((len(ys), 2))
    detF = np.empty(len(ys))
    for i in range(0, len(ys), batch):
        yb = torch.tensor(ys[i:i + batch], dtype=torch.float32)
        x_phys[i:i + batch] = mapping.psi(yb).cpu().numpy()
        F = mapping.jacobian_matrix(yb)
        detF[i:i + batch] = torch.det(F).abs().cpu().numpy()

    ux = u_exact_fn(x_phys)
    diff = uh - ux
    # the change of variables: dx = |det F| dxi
    w = dets_iso * detF

    num = float(np.sum(diff ** 2 * w))
    den = float(np.sum(ux ** 2 * w))
    return np.sqrt(num), np.sqrt(num / den) if den > 0 else np.nan, \
        float(np.max(np.abs(diff)))


# ---------------------------------------------------------------------------

def l2_norm_exact(mesh, u_exact_fn, order: int = 6):
    """``||u||_L2`` on a body-fitted mesh -- a reference denominator if you
    prefer one common normaliser for every row of a results table."""
    total = 0.0
    for block in mesh.element_blocks:
        qpts, qwts = gauss_rule(block.etype, order)
        for e in range(block.connectivity.shape[0]):
            coords = mesh.nodes[block.connectivity[e]]
            for qp, qw in zip(qpts, qwts):
                N, dN = _shape(block.etype, qp)
                detJ = abs(np.linalg.det(dN.T @ coords))
                x = N @ coords
                total += float(u_exact_fn(x[None, :])[0]) ** 2 * detJ * qw
    return np.sqrt(total)
