"""
Cartesian background grid for MPM.

Kept separate from ``scion.mesh`` on purpose. An MPM grid is regular and
disposable: no stored connectivity, node indices found by integer division, and
all nodal state zeroed every step. Forcing it into the general unstructured Mesh
would cost exactly the performance the method depends on.

The grid is padded by ``stencil_size`` cells on each side so that particles near
the domain edge never index out of bounds.
"""

from __future__ import annotations

import numpy as np

from .basis import Basis


class StencilData:
    """Per-particle node stencil for one configuration of the particles.

    This is the central data structure of the whole method, and the reason the
    NumPy code ports cleanly to Warp: everything downstream is a gather or a
    scatter through ``node_id``.

    Attributes
    ----------
    node_id : (n, k) int64      flat grid-node indices, k = stencil_size**dim
    w       : (n, k) float      weights
    grad_w  : (n, k, dim) float weight gradients w.r.t. physical coordinates
    dpos    : (n, k, dim) float node position minus particle position (APIC)
    """

    __slots__ = ("node_id", "w", "grad_w", "dpos")

    def __init__(self, node_id, w, grad_w, dpos):
        self.node_id = node_id
        self.w = w
        self.grad_w = grad_w
        self.dpos = dpos


class CartesianGrid:
    """Uniform Cartesian grid covering ``[origin, origin + shape * dx]``."""

    def __init__(self, shape: tuple[int, ...], dx: float,
                 origin=None, basis: Basis | None = None, dtype=np.float64):
        self.shape = tuple(int(s) for s in shape)
        self.dim = len(self.shape)
        self.dx = float(dx)
        self.inv_dx = 1.0 / self.dx
        self.dtype = dtype
        self.origin = (np.zeros(self.dim, dtype=dtype) if origin is None
                       else np.asarray(origin, dtype=dtype))

        if basis is None:
            from .basis import QuadraticBSpline
            basis = QuadraticBSpline()
        self.basis = basis

        self.pad = basis.stencil_size
        # padded node counts per axis
        self.node_shape = tuple(s + 1 + 2 * self.pad for s in self.shape)
        self.n_nodes = int(np.prod(self.node_shape))

        # flat-index strides and the 3^dim (or 4^dim) stencil offsets
        self._strides = np.array(
            [int(np.prod(self.node_shape[i + 1:])) for i in range(self.dim)],
            dtype=np.int64,
        )
        k = basis.stencil_size
        self._offsets = np.array(
            np.meshgrid(*[np.arange(k) for _ in range(self.dim)], indexing="ij")
        ).reshape(self.dim, -1).T.astype(np.int64)  # (k**dim, dim)

    @property
    def stencil_width(self) -> int:
        return self._offsets.shape[0]

    # ------------------------------------------------------------------
    def build_stencil(self, xp: np.ndarray,
                      lp: np.ndarray | None = None) -> StencilData:
        """Weights, gradients and node indices for every particle.

        One pass, fully vectorised, no Python loop over particles.

        Parameters
        ----------
        xp : (n, dim) particle positions
        lp : (n, dim) particle domain half-widths in *physical* units. Required
             by domain bases such as :class:`uGIMP`; ignored by point bases.
        """
        grid_pos = (xp - self.origin) * self.inv_dx
        base = np.floor(grid_pos - self.basis.base_offset()).astype(np.int64)
        fx = grid_pos - base                                  # (n, dim)

        lam = None
        if self.basis.requires_domain:
            if lp is None:
                raise ValueError(
                    f"{type(self.basis).__name__} needs particle domain "
                    "half-widths; pass lp=particles.lp to build_stencil."
                )
            lam = np.asarray(lp) * self.inv_dx                # -> units of dx

        w1, dw1 = self.basis.weights_1d(fx, lam)              # (n, k, dim)
        dw1 = dw1 * self.inv_dx                               # -> d/dx_physical

        off = self._offsets                                   # (K, dim)
        node_idx = base[:, None, :] + off[None, :, :] + self.pad   # (n, K, dim)
        node_id = (node_idx * self._strides).sum(axis=2)           # (n, K)

        # tensor products over axes
        n = xp.shape[0]
        K = off.shape[0]
        w = np.ones((n, K), dtype=self.dtype)
        parts = []
        for a in range(self.dim):
            wa = w1[:, off[:, a], a]          # (n, K)
            da = dw1[:, off[:, a], a]
            w *= wa
            parts.append((wa, da))

        grad_w = np.empty((n, K, self.dim), dtype=self.dtype)
        for a in range(self.dim):
            g = parts[a][1].copy()
            for b in range(self.dim):
                if b != a:
                    g = g * parts[b][0]
            grad_w[:, :, a] = g

        node_pos = (node_idx - self.pad) * self.dx + self.origin
        dpos = node_pos - xp[:, None, :]

        return StencilData(node_id, w, grad_w, dpos)

    # ------------------------------------------------------------------
    def zeros_scalar(self) -> np.ndarray:
        return np.zeros(self.n_nodes, dtype=self.dtype)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.dim), dtype=self.dtype)

    def node_coordinates(self) -> np.ndarray:
        """(n_nodes, dim) physical coordinates, for boundary conditions."""
        axes = [np.arange(s, dtype=self.dtype) - self.pad for s in self.node_shape]
        mg = np.meshgrid(*axes, indexing="ij")
        return (np.stack([m.ravel() for m in mg], axis=1) * self.dx + self.origin)
