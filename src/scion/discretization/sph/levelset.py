r"""
Level-set geometry for SPH: a solid boundary with no solid particles.

The idea, and why it is not just a convenience
----------------------------------------------
A wall built from particles has to be meshed, stored, searched against and
kept thick enough that no fluid particle sees past it. A *static confinement*
replaces all of that with a signed distance field: the solid is wherever
``phi > 0``, and every boundary term the fluid needs is an integral of the
kernel over that region,

    K(x) = INT_solid W(|x - y|) dy,      G(x) = INT_solid grad W(x - y) dy

which depend on the geometry alone and are therefore computed **once**, on a
background grid, and interpolated. The fluid then behaves as though the solid
were filled with mirror particles -- same pressure, opposite velocity --
without any of them existing.

Sign convention, which is the first thing to get wrong
------------------------------------------------------
``phi > 0`` is **inside the solid** here. SPHinXsys's ``Shape`` returns the
opposite (negative inside the shape) and its wall case passes the *fluid*
region as the shape, so the two conventions meet and cancel; stating ours
explicitly is cheaper than tracing theirs. The normal ``grad phi / |grad phi|``
then points **into** the solid, which is what the surface bounding relies on.

The cut cell is what makes a grid integral sharp
------------------------------------------------
A grid cell straddling the surface is neither in nor out, and counting it as
either puts a staircase on a smooth wall. ``cut_cell_fraction`` is
SPHinXsys's estimate of the solid fraction of a cell from ``phi`` and its
gradient: a direction-weighted smoothed Heaviside, exact for a plane through
the cell and degrading gracefully for a corner. Without it the kernel
integrals jump by a whole cell's worth as the surface crosses a grid line,
and the pressure on a particle sitting there jumps with them.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .kernels import get_kernel

__all__ = ["LevelSet", "box_interior_distance", "polygon_distance",
           "cut_cell_fraction"]


def box_interior_distance(points, lower, upper) -> np.ndarray:
    """Signed distance to a box, **positive outside** it.

    So a tank whose fluid region is the box becomes a solid everywhere else,
    which is the static-confinement reading of a wall.
    """
    x = np.atleast_2d(np.asarray(points, dtype=float))
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    # per axis, how far outside the box the point is; negative when inside
    excess = np.maximum(lo - x, x - hi)
    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
    inside = excess.max(axis=1)
    return np.where(outside > 0.0, outside, inside)


def polygon_distance(points, polygon) -> np.ndarray:
    """Signed distance to a closed polygon, **positive inside** it.

    So a polygon describing an obstacle is a solid. Brute force over the
    edges: these are tens of vertices evaluated once on a grid, and a
    cleverer scheme would be a second place for the sign to be wrong.
    """
    x = np.atleast_2d(np.asarray(points, dtype=float))
    poly = np.asarray(polygon, dtype=float)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    starts, ends = poly, np.roll(poly, -1, axis=0)
    edge = ends - starts
    length_sq = (edge * edge).sum(axis=1)

    diff = x[:, None, :] - starts[None, :, :]
    t = np.clip((diff * edge[None]).sum(axis=2) / length_sq[None], 0.0, 1.0)
    closest = starts[None] + t[:, :, None] * edge[None]
    distance = np.linalg.norm(x[:, None, :] - closest, axis=2).min(axis=1)

    # crossing number, with the ray cast along +x
    y0, y1 = starts[None, :, 1], ends[None, :, 1]
    straddles = (y0 > x[:, None, 1]) != (y1 > x[:, None, 1])
    x_cross = (starts[None, :, 0] + (x[:, None, 1] - y0) / (y1 - y0 + 1e-300)
               * edge[None, :, 0])
    inside = (straddles & (x[:, None, 0] < x_cross)).sum(axis=1) % 2 == 1
    return np.where(inside, distance, -distance)


def cut_cell_fraction(phi, gradient, spacing: float) -> np.ndarray:
    """The solid fraction of a cell, from ``phi`` and ``grad phi``.

    ``sum_d (g_d/|g|)^2 H(phi / |g_d|, dx/2)`` with ``H`` the smoothed
    Heaviside clamped to ``[0, 1]``. Exact for a plane crossing the cell
    perpendicular to an axis and a weighted blend otherwise, which is the
    point: a plain ``phi > 0`` test makes a smooth wall into a staircase.
    """
    g = np.asarray(gradient, dtype=float)
    squared = (g * g).sum(axis=-1)
    inverse = 1.0 / (squared + 1e-300)
    half_width = 0.5 * spacing
    fraction = np.zeros_like(np.asarray(phi, dtype=float))
    for d in range(g.shape[-1]):
        measure = phi / (np.abs(g[..., d]) + 1e-300)
        heaviside = np.clip(0.5 + 0.5 * measure / half_width, 0.0, 1.0)
        fraction += g[..., d] ** 2 * inverse * heaviside
    return fraction


class LevelSet:
    """A solid region on a background grid, with its kernel integrals.

    ``phi`` is a callable taking ``(n, dim)`` points and returning the signed
    distance, positive inside the solid. The grid covers ``[lower, upper]``
    and should extend at least one cutoff beyond anything the fluid reaches,
    since the integrals at the edge are completed by edge replication.
    """

    def __init__(self, phi, lower, upper, spacing, kernel, h, dim: int = 2):
        if dim != 2:
            raise NotImplementedError("the level set is two-dimensional")
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.spacing = float(spacing)
        self.kernel = get_kernel(kernel)
        self.h = float(h)
        self.dim = dim

        counts = np.maximum(np.rint((self.upper - self.lower) / self.spacing)
                            .astype(int), 1)
        self.counts = counts
        axes = [self.lower[d] + (np.arange(counts[d]) + 0.5) * self.spacing
                for d in range(dim)]
        self.axes = axes
        grid = np.meshgrid(*axes, indexing="ij")
        points = np.stack([g.ravel() for g in grid], axis=1)

        self.phi = np.asarray(phi(points), dtype=float).reshape(counts)
        self.gradient = np.stack(np.gradient(self.phi, self.spacing, edge_order=2),
                                 axis=-1)
        self.fraction = cut_cell_fraction(self.phi, self.gradient, self.spacing)
        self._build_integrals()

    # -- construction ---------------------------------------------------
    def _build_integrals(self) -> None:
        """Convolve the solid fraction with the kernel and its gradient.

        ``ndimage.correlate`` and not ``convolve``: correlate evaluates
        ``sum_d alpha(x + d) w(d)``, which is the integral as written. The
        convolution would flip the stencil, which is invisible for ``W`` --
        it is even -- and silently negates ``G``.
        """
        support = self.kernel.support(self.h)
        depth = int(np.ceil(support / self.spacing))
        offsets = np.arange(-depth, depth + 1) * self.spacing
        dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
        r = np.sqrt(dx * dx + dy * dy)
        cell_volume = self.spacing ** self.dim

        w = self.kernel.W(r, self.h, self.dim) * cell_volume
        self.kernel_integral = ndimage.correlate(self.fraction, w, mode="nearest")

        dw = self.kernel.dW_dr(r, self.h, self.dim) * cell_volume
        unit = np.stack([dx, dy], axis=-1) / (r[..., None] + 1e-300)
        # grad_x W(x - y) with y = x + d is -d/|d| dW/dr
        gradient = []
        for d in range(self.dim):
            stencil = -dw * unit[..., d]
            gradient.append(ndimage.correlate(self.fraction, stencil, mode="nearest"))
        self.kernel_gradient_integral = np.stack(gradient, axis=-1)

    # -- probing --------------------------------------------------------
    def _interpolate(self, field, points) -> np.ndarray:
        """Bilinear interpolation, with the edge value outside the grid."""
        x = np.atleast_2d(np.asarray(points, dtype=float))
        index = ((x - self.lower) / self.spacing - 0.5).T
        trailing = field.shape[self.dim:]
        if trailing:
            out = [ndimage.map_coordinates(field[..., k], index, order=1,
                                           mode="nearest")
                   for k in range(trailing[0])]
            return np.stack(out, axis=1)
        return ndimage.map_coordinates(field, index, order=1, mode="nearest")

    def signed_distance(self, points) -> np.ndarray:
        return self._interpolate(self.phi, points)

    def normal(self, points) -> np.ndarray:
        """The unit normal, pointing **into** the solid."""
        g = self._interpolate(self.gradient, points)
        return g / (np.linalg.norm(g, axis=1)[:, None] + 1e-300)

    def kernel_integral_at(self, points) -> np.ndarray:
        """``K(x)``: the kernel mass that falls inside the solid."""
        return self._interpolate(self.kernel_integral, points)

    def kernel_gradient_at(self, points) -> np.ndarray:
        """``G(x)``: the kernel gradient integrated over the solid."""
        return self._interpolate(self.kernel_gradient_integral, points)
