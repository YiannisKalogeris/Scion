r"""
The discrete configurational force, Eq. (9).

Differentiating the discrete potential energy

.. math::
    \Pi_h(\{u_i\}; \{X_p\}) = \sum_p V_p^0 W(F_p) - \Pi_h^{\rm ext}

with respect to the *reference* position of a material point gives

.. math::
    \mathcal{G}_{p,K} = -\frac{\partial \Pi_h}{\partial X_{p,K}}
      = -V_p^0 P_{p,iJ} \sum_a u_{a,i}
        \frac{\partial^2 w_{ap}}{\partial X_J \partial X_K}
      - W(F_p) \frac{\partial V_p^0}{\partial X_K}.

The first term is the paper's mechanism: the force is driven by the **second**
material gradient of the grid basis. For a piecewise-multilinear basis that
object is a singular measure supported on cell boundaries, so the force is
unbounded exactly at a cell crossing; for a ``C^{k-1}`` B-spline of degree
``k >= 2`` it is bounded. Cell-crossing error is thereby identified with
spurious configurational force, and the regularity of the *grid basis* -- rather
than the particle-domain representation -- emerges as the governing quantity.

The second term needs the particle volumes to be a differentiable function of
the reference positions, i.e. a tessellation. With the fixed per-particle
volumes MPM normally carries, ``dV^0/dX = 0`` and the term drops; pass
``dvol_dX`` explicitly when a tessellation is available.

Total-Lagrangian only
---------------------
Every gradient here is with respect to the reference configuration, which is
unambiguous only when the grid is anchored there. Under the updated-Lagrangian
stepper the basis is evaluated at current positions and the reference manifold
is not represented at all, so passing its stencil computes a different object.
The paper defers that case (its Section 7) and so does this.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "displacement_gradient",
    "second_displacement_gradient",
    "configurational_force",
]


def displacement_gradient(u_nodes: np.ndarray, node_id: np.ndarray,
                          grad_w: np.ndarray) -> np.ndarray:
    r"""``H_{p,iJ} = sum_a u_{a,i} dw_{ap}/dX_J``. ``-> (n, dim, dim)``.

    The deformation gradient is ``F = I + H``; this returns ``H`` alone because
    Eq. (9) contracts the *second* gradient of the same sum, and keeping the two
    parallel makes the pair easy to check against each other.
    """
    return np.einsum("pka,pkJ->paJ", u_nodes[node_id], grad_w)


def second_displacement_gradient(u_nodes: np.ndarray, node_id: np.ndarray,
                                 hess: np.ndarray) -> np.ndarray:
    r"""``sum_a u_{a,i} d2w_{ap}/dX_J dX_K``. ``-> (n, dim, dim, dim)``.

    The gradient of the discrete displacement gradient. Symmetric in its last
    two indices because the basis Hessian is.
    """
    if hess is None:
        raise ValueError(
            "no second gradients on this stencil; build it with "
            "CartesianGrid.build_stencil(..., hessian=True)"
        )
    return np.einsum("pka,pkJK->paJK", u_nodes[node_id], hess)


def configurational_force(P: np.ndarray, volume: np.ndarray,
                          u_nodes: np.ndarray, node_id: np.ndarray,
                          hess: np.ndarray,
                          energy: np.ndarray | None = None,
                          dvol_dX: np.ndarray | None = None) -> np.ndarray:
    r"""Equation (9). ``-> (n, dim)``, one vector per material point.

    Parameters
    ----------
    P        : ``(n, d, d)`` first Piola-Kirchhoff stress at each point
    volume   : ``(n,)`` reference volume ``V_p^0``
    u_nodes  : ``(n_nodes, d)`` nodal displacements
    node_id  : ``(n, k)`` stencil node indices
    hess     : ``(n, k, d, d)`` second material gradients of the basis
    energy   : ``(n,)`` ``W(F_p)``; required only with ``dvol_dX``
    dvol_dX  : ``(n, d)`` ``dV_p^0/dX_K``. Omit when particle volumes are fixed,
               which is the usual case and makes the term vanish identically.

    Vanishing is the useful property
    --------------------------------
    For the exact solution of a homogeneous body this is zero, so ``||G_p||`` is
    an admissible error indicator: whatever it returns is discretisation error,
    not physics. In particular it is zero for *any* uniform deformation on any
    basis with linear completeness, because then
    ``sum_a X_a d2w_a/dXdX = 0`` -- so a non-zero value on a uniform state is a
    bug in the basis, not a small quadrature residual.
    """
    d2u = second_displacement_gradient(u_nodes, node_id, hess)
    out = -volume[:, None] * np.einsum("piJ,piJK->pK", P, d2u)

    if dvol_dX is not None:
        if energy is None:
            raise ValueError("dvol_dX given without the energy W(F_p)")
        out = out - energy[:, None] * np.asarray(dvol_dX)
    return out
