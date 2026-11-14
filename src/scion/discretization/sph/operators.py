r"""
SPH operators on a :class:`~scion.discretization.sph.neighbors.NeighborList`.

The correction matrix is what makes the gradient first-order
------------------------------------------------------------
A bare SPH gradient does not reproduce a linear field, because the kernel
moments of a finite, irregular neighbourhood are not the moments of the
continuum. ``linear_gradient_correction`` builds

    A_i = sum_j V_j (-dW/dr) r_ij (e_ij (x) e_ij),     B_i = A_i^{-1}

so that ``B_i sum_j V_j (phi_j - phi_i) grad W_ij`` is *exact* for any linear
field -- an identity, not a tolerance, and that is how it is tested. ``A`` is
symmetric positive semi-definite by construction, which is the cheap check
that the assembly has its signs right: ``dW/dr`` is negative on the whole
support, so the ``-dW/dr`` above is positive.

``alpha`` follows SPHinXsys: at the default 0 the correction is the plain
(Tikhonov-regularized) inverse, and a positive value blends towards the
identity wherever ``det A`` falls below it -- a free-surface guard, since a
particle whose support is half empty has a nearly singular ``A`` and
inverting it amplifies noise instead of correcting anything.

The diffusion operator is a weighted graph Laplacian
----------------------------------------------------
The Cleary--Monaghan second-derivative form is

    dphi_i/dt = sum_j D_ij (phi_i - phi_j) * 2 (B_ij grad W_ij . e_ij) / r_ij * V_j

and everything in it except ``phi`` is fixed by the particle positions. So it
splits: ``pair_conductance`` computes the edge weight once, and
``laplacian_rate`` is then one subtraction and one scatter per pair. On a
fixed particle set that is not a micro-optimization -- it is **5x** on the
step (measured, 15.8 -> 3.2 ms at 32k particles), because the projection it
hoists materialises a ``(n_pairs, dim, dim)`` gather that costs more than the
scatter does.

Written pairwise, so particle ``j`` receives the same term with ``V_i`` and
the opposite sign. Hence ``sum_i V_i dphi_i/dt`` telescopes to zero at
round-off on any particle distribution -- the SPH counterpart of the exact
conservation the FVM assembler gets from face cancellation. A drift above
~1e-14 there is a bookkeeping bug in the pair loop, not inaccuracy.
"""

from __future__ import annotations

import numpy as np

from .kernels import SPHKernel, get_kernel
from .neighbors import NeighborList

__all__ = ["kernel_pair_values", "linear_gradient_correction", "sph_gradient",
           "pair_conductance", "laplacian_rate", "number_density",
           "shepard_interpolate"]


def kernel_pair_values(pairs: NeighborList, kernel, h: float, dim: int):
    """``W`` and ``dW/dr`` at every stored pair."""
    kernel = get_kernel(kernel)
    return kernel.W(pairs.r, h, dim), kernel.dW_dr(pairs.r, h, dim)


def number_density(pairs: NeighborList, volumes, w_ij) -> np.ndarray:
    """``sum_j V_j W_ij`` over the *neighbours*, without the self term.

    The pair list holds no self pairs, so a caller after the partition of
    unity has to add ``W(0) V_i`` -- which is a third of the total in 2-D at
    the usual ``h = 1.3 dp``, i.e. not a correction. One is the residual on a
    lattice away from a boundary, and the shortfall at a free surface is the
    whole reason the corrected gradient exists.
    """
    vol = np.asarray(volumes, dtype=float)
    out = np.zeros(pairs.n_particles)
    np.add.at(out, pairs.i, vol[pairs.j] * w_ij)
    np.add.at(out, pairs.j, vol[pairs.i] * w_ij)
    return out


def linear_gradient_correction(pairs: NeighborList, volumes, dw_ij, dim: int,
                               alpha: float = 0.0,
                               epsilon: float | None = None) -> np.ndarray:
    """The per-particle correction matrices ``B``, ``(n, dim, dim)``."""
    vol = np.asarray(volumes, dtype=float)
    weight = -dw_ij * pairs.r                      # positive on the support
    ee = pairs.e[:, :, None] * pairs.e[:, None, :]
    A = np.zeros((pairs.n_particles, dim, dim))
    np.add.at(A, pairs.i, (weight * vol[pairs.j])[:, None, None] * ee)
    np.add.at(A, pairs.j, (weight * vol[pairs.i])[:, None, None] * ee)

    eps = np.sqrt(np.finfo(float).eps) if epsilon is None else float(epsilon)
    eye = np.eye(dim)
    At = np.swapaxes(A, 1, 2)
    inverse = np.linalg.solve(At @ A + eps * eye, At)
    det = np.linalg.det(A)
    if alpha <= 0.0:
        return inverse
    deficit = np.maximum(alpha - det, 0.0)
    total = det + deficit
    w1 = np.where(total > 0.0, det / np.where(total > 0.0, total, 1.0), 0.0)
    return w1[:, None, None] * inverse + (1.0 - w1)[:, None, None] * eye


def sph_gradient(pairs: NeighborList, values, volumes, dw_ij, dim: int,
                 B: np.ndarray | None = None) -> np.ndarray:
    """``grad phi`` at the particles, corrected if ``B`` is given.

    Exact for a linear field with the matching ``B``; off by the boundary
    deficit without it, which is what the test asserts as a pair.
    """
    phi = np.asarray(values, dtype=float)
    vol = np.asarray(volumes, dtype=float)
    grad = np.zeros((pairs.n_particles, dim))
    gw = dw_ij[:, None] * pairs.e                  # grad_i W_ij
    dphi = phi[pairs.j] - phi[pairs.i]
    np.add.at(grad, pairs.i, (vol[pairs.j] * dphi)[:, None] * gw)
    np.add.at(grad, pairs.j, (vol[pairs.i] * -dphi)[:, None] * -gw)
    if B is None:
        return grad
    return np.einsum("nab,nb->na", B, grad)


def pair_conductance(pairs: NeighborList, dw_ij, diffusivity_ij,
                     B: np.ndarray | None = None) -> np.ndarray:
    """The edge weight ``2 D_ij (B_ij e . e) dW/dr / r``, one per pair.

    Negative, because ``dW/dr`` is. ``diffusivity_ij`` is the inter-particle
    coefficient; for an anisotropic tensor it is the directional projection
    ``1 / (e . D^-1 e)`` that :mod:`.diffusion` supplies.
    """
    if B is None:
        projection = np.ones_like(pairs.r)
    else:
        M = 0.5 * (B[pairs.i] + B[pairs.j])
        projection = np.einsum("na,nab,nb->n", pairs.e, M, pairs.e)
    return 2.0 * diffusivity_ij * dw_ij * projection / pairs.r


def laplacian_rate(pairs: NeighborList, values, volumes,
                   conductance) -> np.ndarray:
    """``d phi / dt`` for ``div(D grad phi)``, from the edge weights.

    ``np.bincount`` rather than ``np.add.at``: measured identical to 2.3e-17
    on a 316k-pair set and the same speed here, but it is the form that does
    not degrade when the pair list grows, and it states that the scatter is a
    segmented sum rather than an in-place update.
    """
    phi = np.asarray(values, dtype=float)
    vol = np.asarray(volumes, dtype=float)
    flux = np.asarray(conductance, dtype=float) * (phi[pairs.i] - phi[pairs.j])
    n = pairs.n_particles
    return (np.bincount(pairs.i, weights=flux * vol[pairs.j], minlength=n)
            - np.bincount(pairs.j, weights=flux * vol[pairs.i], minlength=n))


def shepard_interpolate(points, positions, values, volumes, kernel, h: float,
                        dim: int, lower=None, upper=None, periodic=None):
    """Kernel interpolation of a particle field at arbitrary points.

    Shepard-normalized, ``sum_j W V_j phi_j / sum_j W V_j``, which is what an
    SPHinXsys observer body reports: without the normalization the value at a
    point near a free surface reads low in proportion to the missing
    neighbours rather than being merely noisy.

    Brute force over the particles, because the observer sets this exists for
    are a handful of points.
    """
    kernel = get_kernel(kernel)
    p = np.atleast_2d(np.asarray(points, dtype=float))
    x = np.asarray(positions, dtype=float)
    phi = np.asarray(values, dtype=float)
    vol = np.asarray(volumes, dtype=float)
    periodic = (False,) * dim if periodic is None else tuple(bool(q) for q in periodic)
    out = np.zeros(p.shape[0])
    for n, point in enumerate(p):
        d = x - point
        for axis, flag in enumerate(periodic):
            if flag:
                length = upper[axis] - lower[axis]
                d[:, axis] -= length * np.round(d[:, axis] / length)
        r = np.sqrt((d * d).sum(axis=1))
        near = r < kernel.support(h)
        w = kernel.W(r[near], h, dim) * vol[near]
        total = w.sum()
        out[n] = (w * phi[near]).sum() / total if total > 0.0 else np.nan
    return out
