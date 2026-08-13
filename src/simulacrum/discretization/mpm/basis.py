"""
MPM weighting functions.

This is the one part of MPM that cannot reuse ``discretization/fem/elements.py``.
FEM shape functions are element-local and evaluated at fixed quadrature points;
MPM weights are node-local and evaluated at arbitrary particle positions, and
their *gradients must be continuous across cell boundaries*. Bilinear FEM hats
have discontinuous gradients there, which produces cell-crossing noise -- an
artefact that looks like a physical instability but is not.

Two families here, addressing two different columns of the usual MPM-variant
comparison table:

* B-splines (:class:`QuadraticBSpline`, :class:`CubicBSpline`) fix cell crossing
  and reduce quadrature error by making the weights C1/C2 continuous. They keep
  treating each particle as a *point*.
* GIMP (:class:`uGIMP`) instead gives each particle a *domain* of half-width
  ``lp`` and integrates the linear hat over it. This fixes cell crossing too,
  and additionally suppresses numerical fracture -- the spurious tearing that
  happens when point particles drift more than a cell apart under stretching.

Each basis exposes:

    stencil_size            nodes touched per particle per axis
    requires_domain         True if ``weights_1d`` needs the ``lp`` argument
    base_offset()           shift applied before ``floor`` to find the first node
    weights_1d(fx, lp)      -> (w, dw), each (n, stencil_size, dim)
    apic_inv_D(dx)          inverse of the APIC normalisation tensor D

``fx`` is the particle position relative to the first stencil node, in units of
``dx``; ``lp`` is the particle domain half-width, also in units of ``dx``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Basis(ABC):
    """Tensor-product MPM weighting function on a Cartesian grid."""

    stencil_size: int
    requires_domain: bool = False

    @abstractmethod
    def base_offset(self) -> float:
        """Shift applied before ``floor`` to locate the first stencil node."""

    @abstractmethod
    def weights_1d(self, fx: np.ndarray,
                   lp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """1-D weights and their derivatives w.r.t. ``fx``.

        Parameters
        ----------
        fx : (n, dim) particle position relative to the first stencil node,
             in units of ``dx``
        lp : (n, dim) particle domain half-width in units of ``dx``; required
             only when ``requires_domain`` is True, ignored otherwise

        Returns
        -------
        w, dw : each (n, stencil_size, dim)
        """

    @abstractmethod
    def apic_inv_D(self, dx: float) -> float:
        """Inverse of the APIC normalisation scalar, D = c * dx^2 * I."""


# ---------------------------------------------------------------------------
# Point-particle bases
# ---------------------------------------------------------------------------

class LinearHat(Basis):
    """Original MPM weights (Sulsky 1994). Cheap, but gradients jump when a
    particle crosses a cell boundary. Useful only as a reference/debug basis --
    and to reproduce the cell-crossing artefact deliberately."""

    stencil_size = 2

    def base_offset(self) -> float:
        return 0.0

    def weights_1d(self, fx, lp=None):
        w = np.stack([1.0 - fx, fx], axis=1)
        dw = np.stack([-np.ones_like(fx), np.ones_like(fx)], axis=1)
        return w, dw

    def apic_inv_D(self, dx: float) -> float:
        return 3.0 / (dx * dx)


class QuadraticBSpline(Basis):
    """Quadratic B-spline weights (Steffen et al. 2008).

    C1-continuous, so gradients are continuous across cell boundaries. The
    sensible default for explicit MPM with point particles.
    """

    stencil_size = 3

    def base_offset(self) -> float:
        return 0.5

    def weights_1d(self, fx, lp=None):
        w = np.stack(
            [
                0.50 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.50 * (fx - 0.5) ** 2,
            ],
            axis=1,
        )
        dw = np.stack([fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5], axis=1)
        return w, dw

    def apic_inv_D(self, dx: float) -> float:
        return 4.0 / (dx * dx)


class CubicBSpline(Basis):
    """Cubic B-spline weights. C2-continuous, 4-node stencil per axis, so 16
    nodes in 2-D and 64 in 3-D -- noticeably more expensive than quadratic."""

    stencil_size = 4

    def base_offset(self) -> float:
        return 1.0

    def weights_1d(self, fx, lp=None):
        d0, d1, d2, d3 = fx, fx - 1.0, 2.0 - fx, 3.0 - fx
        w = np.stack(
            [
                (1.0 / 6.0) * (2.0 - d0) ** 3,
                0.5 * d1 ** 3 - d1 ** 2 + 2.0 / 3.0,
                0.5 * d2 ** 3 - d2 ** 2 + 2.0 / 3.0,
                (1.0 / 6.0) * (2.0 - d3) ** 3,
            ],
            axis=1,
        )
        dw = np.stack(
            [
                -0.5 * (2.0 - d0) ** 2,
                1.5 * d1 ** 2 - 2.0 * d1,
                -(1.5 * d2 ** 2 - 2.0 * d2),
                0.5 * (2.0 - d3) ** 2,
            ],
            axis=1,
        )
        return w, dw

    def apic_inv_D(self, dx: float) -> float:
        return 3.0 / (dx * dx)


# ---------------------------------------------------------------------------
# Domain-particle basis
# ---------------------------------------------------------------------------

class uGIMP(Basis):
    r"""Uniform Generalized Interpolation Material Point (Bardenhagen & Kober 2004).

    The particle is a segment (rectangle, box) of half-width ``lp`` rather than
    a point, and the weight is the linear hat averaged over that domain:

        S_i(x_p) = (1 / 2 lp) \int_{x_p - lp}^{x_p + lp} N_i(x) dx

    In units of the cell size, with :math:`r = (x_p - x_i)/h` and
    :math:`\lambda = l_p/h`:

        S(r) = 0                              |r| >= 1 + lam
               (1 + lam + r)^2 / (4 lam)      -(1+lam) < r <= -(1-lam)
               1 + r                          -(1-lam) < r <= -lam
               1 - (r^2 + lam^2)/(2 lam)      -lam < r <= lam
               1 - r                          lam < r <= 1 - lam
               (1 + lam - r)^2 / (4 lam)      1 - lam < r < 1 + lam

    The "uniform" in uGIMP means the domain keeps its initial size; see
    ``ParticleSet.update_domain`` for the contiguous-particle (cpGIMP) variant
    that stretches it with the deformation.

    Stencil size is 3, the same as the quadratic B-spline: the support half-width
    is ``h + lp <= 1.5 h`` provided ``lp <= h/2``, which the centred 3-node
    stencil covers exactly. That bound is checked, because exceeding it silently
    truncates the support and breaks partition of unity.
    """

    stencil_size = 3
    requires_domain = True

    #: below this lam the formula is ill-conditioned; fall back to the linear hat
    LAM_FLOOR = 1.0e-9

    def base_offset(self) -> float:
        return 0.5

    def weights_1d(self, fx, lp=None):
        if lp is None:
            raise ValueError(
                "uGIMP needs the particle domain half-width. Pass lp to "
                "CartesianGrid.build_stencil, or give ParticleSet an `lp` array."
            )
        lam = np.asarray(lp)
        if lam.shape != fx.shape:
            lam = np.broadcast_to(lam, fx.shape)
        if np.any(lam > 0.5 + 1e-12):
            raise ValueError(
                f"uGIMP requires lp <= dx/2; got max lp/dx = {lam.max():.4f}. "
                "Either refine the grid, reduce particles-per-cell, or widen "
                "the stencil to 4 nodes."
            )

        # r_k = fx - k is the signed particle-to-node distance in units of dx
        k = np.arange(3).reshape(1, 3, 1)
        r = fx[:, None, :] - k                      # (n, 3, dim)
        lam = lam[:, None, :]                       # (n, 1, dim) -> broadcast

        a = np.abs(r)
        w = np.zeros_like(r)
        dw = np.zeros_like(r)

        # branch masks, written on |r| so each is stated once
        outer = (a > 1.0 - lam) & (a < 1.0 + lam)   # quadratic tail
        flat = (a > lam) & (a <= 1.0 - lam)         # linear part
        inner = a <= lam                            # node inside the domain

        s = np.sign(r)
        s[s == 0] = 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            t = 1.0 + lam - a
            w = np.where(outer, t * t / (4.0 * lam), w)
            dw = np.where(outer, -s * t / (2.0 * lam), dw)

            w = np.where(flat, 1.0 - a, w)
            dw = np.where(flat, -s, dw)

            w = np.where(inner, 1.0 - (r * r + lam * lam) / (2.0 * lam), w)
            dw = np.where(inner, -r / lam, dw)

        # lam -> 0 degenerates to the linear hat; avoid 0/0
        tiny = lam < self.LAM_FLOOR
        if np.any(tiny):
            hat_w = np.where(a < 1.0, 1.0 - a, 0.0)
            hat_dw = np.where(a < 1.0, -s, 0.0)
            w = np.where(tiny, hat_w, w)
            dw = np.where(tiny, hat_dw, dw)

        return w, dw

    def apic_inv_D(self, dx: float) -> float:
        # GIMP is not a quadratic B-spline, so D is not exactly (dx^2/4) I.
        # 4/dx^2 is the standard practical choice and is what APIC-GIMP
        # implementations use; the affine term is a correction, not the
        # leading-order transfer, so the approximation is benign.
        return 4.0 / (dx * dx)


BASIS_REGISTRY: dict[str, type[Basis]] = {
    "linear": LinearHat,
    "quadratic": QuadraticBSpline,
    "cubic": CubicBSpline,
    "gimp": uGIMP,
    "ugimp": uGIMP,
}


def get_basis(name: str) -> Basis:
    if name not in BASIS_REGISTRY:
        raise NotImplementedError(f"Unknown MPM basis '{name}'")
    return BASIS_REGISTRY[name]()
