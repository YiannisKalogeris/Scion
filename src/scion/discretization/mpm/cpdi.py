r"""
Convected Particle Domain Interpolation (Sadeghirad, Brannon & Burghardt 2011).

The particle domain is a parallelogram that convects with the deformation,
``r_k^{n+1} = F^{n+1} r_k^0``, and the grid basis is replaced by its average
over that domain. Sadeghirad et al. show (their Eqs. 25-26) that the average and
its gradient collapse to a weighted sum of the *grid* basis sampled at the four
domain **corners**:

.. math::
    \varphi_{ip} = \tfrac14 \sum_\alpha S_i(x_\alpha)

.. math::
    \nabla\varphi_{ip} = \frac{1}{V_p}\Big\{
      (S_1 - S_3)\begin{bmatrix} r_{1y}-r_{2y} \\ r_{2x}-r_{1x}\end{bmatrix}
    + (S_2 - S_4)\begin{bmatrix} r_{1y}+r_{2y} \\ -r_{1x}-r_{2x}\end{bmatrix}
    \Big\}

with corners ``x_p \pm r_1 \pm r_2`` counterclockwise and ``V_p = 4|r_1 \times r_2|``.
Scion's ``r_k`` are *half*-edge vectors, matching ``ParticleSet.lp``; the paper's
are full edges, which is why its prefactor reads ``1/(2V_p)`` and this one reads
``1/V_p``. The gradient formula is exactly the divergence theorem
``\int_\Omega \nabla S = \oint_{\partial\Omega} S n`` with the trapezoid rule on
each edge, which is how the coefficients were checked here.

Why this is not a ``Basis``
---------------------------
``CartesianGrid.build_stencil`` is strictly tensor-product: it forms 1-D weights
per axis and multiplies them. A CPDI domain stops being axis-aligned the moment
``F`` contains any rotation, so its weight is **not separable** and cannot be
written as ``weights_1d`` at all. Hence a grid subclass that overrides the
stencil construction rather than another entry in ``BASIS_REGISTRY``.

The construction is: build the ordinary tensor-product stencil at each of the
four corners, then concatenate. Node indices repeat between corners and that is
deliberately left alone -- every consumer downstream is either an ``np.add.at``
scatter, which accumulates duplicates correctly, or a gather, which reads the
same value twice against weights that already sum correctly. Deduplicating would
cost a sort per step and buy nothing.

On pairing this with a B-spline (it is not BSCPDI)
--------------------------------------------------
Nothing here fixes the grid basis, so ``CPDIGrid(..., basis=get_basis(
"quadratic"))`` is constructible -- and it is **not** B-spline CPDI, and it is
unstable. Measured on a rotating, drifting disk over 300 steps: kinetic energy
grew by a factor of 80, ``max|J - 1|`` reached 0.94 and the particle domains
inflated 2.2x, while the same run with the linear grid basis stayed at
``KE_f/KE_0 = 0.71`` and ``max|J - 1| = 9e-4``.

The reason is in the derivation. Eq. 26 is the divergence theorem with the
**trapezoid rule on each edge**, which is exact only when ``S`` is linear along
that edge, and Eq. 25 averages the corner values, which equals the domain
average only for a linear ``S``. Both assumptions are properties of the *linear*
grid basis, not of the corner construction, so swapping in a B-spline leaves
``phi`` and ``grad phi`` mutually inconsistent even though each separately still
reproduces linear fields -- which the consistency checks confirm, and which is
why the failure does not show up until a run destabilises.

Published BSCPDI is a different construction, not this substitution, and is not
implemented here.
"""

from __future__ import annotations

import numpy as np

from .background_grid import CartesianGrid, StencilData

__all__ = ["CPDIGrid", "corner_offsets"]

#: signs of (r1, r2) at each corner, counterclockwise
_CORNER_SIGNS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def corner_offsets(domain: np.ndarray) -> np.ndarray:
    """``(n, 4, dim)`` corner positions relative to the particle centre.

    ``domain`` is ``(n, dim, dim)`` with ``domain[:, k, :]`` the k-th half-edge
    vector.
    """
    return np.einsum("cd,pdi->pci", _CORNER_SIGNS, domain)


class CPDIGrid(CartesianGrid):
    """Cartesian grid whose stencil averages the basis over a convected domain.

    Takes the same arguments as :class:`~.background_grid.CartesianGrid`. Pass a
    B-spline basis to get BSCPDI.

    ``build_stencil`` requires ``domain``; the particle-domain half-widths
    ``lp`` are not enough, because once ``F`` rotates the domain the axis-aligned
    half-widths no longer describe it.
    """

    def build_stencil(self, xp: np.ndarray, lp: np.ndarray | None = None,
                      hessian: bool = False,
                      domain: np.ndarray | None = None) -> StencilData:
        if domain is None:
            raise ValueError(
                "CPDIGrid needs the convected particle domain vectors. Pass "
                "domain=particles.domain, and keep it convected by calling "
                "ParticleSet.update_domain('cpdi') after each step."
            )
        if self.dim != 2:
            raise NotImplementedError(
                f"CPDI here is 2-D; got dim={self.dim}. The corner "
                "construction generalises to 8 corners in 3-D but the gradient "
                "coefficients are a different set and are not implemented."
            )

        domain = np.asarray(domain, dtype=self.dtype)
        offsets = corner_offsets(domain)                   # (n, 4, dim)
        r1, r2 = domain[:, 0, :], domain[:, 1, :]
        area = 4.0 * np.abs(r1[:, 0] * r2[:, 1] - r1[:, 1] * r2[:, 0])
        if np.any(area <= 0.0):
            raise ValueError(
                "a particle domain has collapsed to zero area; CPDI cannot "
                "average a basis over it. This is the degeneracy CPDI is "
                "supposed to survive, so it means F has become singular."
            )

        # Eq. 26: coefficient multiplying S at corners 1,3 and at 2,4
        coef_a = np.stack([r1[:, 1] - r2[:, 1], r2[:, 0] - r1[:, 0]], axis=1)
        coef_b = np.stack([r1[:, 1] + r2[:, 1], -r1[:, 0] - r2[:, 0]], axis=1)
        per_corner = [coef_a, coef_b, -coef_a, -coef_b]

        ids, ws, gws, dps, hss = [], [], [], [], []
        for c in range(4):
            corner = xp + offsets[:, c, :]
            # hessian=False deliberately: the CPDI second gradient below is
            # built from grad_w at the corners, not from the basis hessian, so
            # requesting one would refuse on a linear grid basis for no reason.
            st = super().build_stencil(corner, lp=None, hessian=False)
            ids.append(st.node_id)
            ws.append(0.25 * st.w)

            coef = per_corner[c] / area[:, None]           # (n, dim)
            gws.append(st.w[:, :, None] * coef[:, None, :])
            # node positions are the corner stencil's, but dpos must be
            # measured from the particle centre for APIC's affine term
            dps.append(st.dpos + offsets[:, c, None, :])

            if hessian:
                # Differentiating Eq. 26 with respect to x_p. The coefficients
                # are constant over the step, so only S is differentiated -- and
                # only to *first* order, because Eq. 26 already carries one
                # derivative in the form of the boundary integral.
                #
                # The consequence is the asymmetry the paper predicts: CPDI's
                # second material gradient is **bounded but discontinuous** even
                # on a linear grid basis, where the raw hat's is a Dirac
                # measure. It is not continuous, as a B-spline's is. So CPDI
                # and BSMPM improve different terms of Eq. (9).
                hss.append(np.einsum("pa,pkb->pkab", coef, st.grad_w))

        return StencilData(
            np.concatenate(ids, axis=1),
            np.concatenate(ws, axis=1),
            np.concatenate(gws, axis=1),
            np.concatenate(dps, axis=1),
            np.concatenate(hss, axis=1) if hessian else None,
        )
