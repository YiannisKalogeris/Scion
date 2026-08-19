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
    hess    : (n, k, dim, dim) float second gradients, or None unless
              ``build_stencil(..., hessian=True)`` asked for them
    """

    __slots__ = ("node_id", "w", "grad_w", "dpos", "hess")

    def __init__(self, node_id, w, grad_w, dpos, hess=None):
        self.node_id = node_id
        self.w = w
        self.grad_w = grad_w
        self.dpos = dpos
        self.hess = hess


class CartesianGrid:
    """Uniform Cartesian grid covering ``[origin, origin + shape * dx]``."""

    def __init__(self, shape: tuple[int, ...], dx: float,
                 origin=None, basis: Basis | None = None, dtype=np.float64,
                 periodic=False):
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

        self.periodic = np.broadcast_to(
            np.asarray(periodic, dtype=bool), (self.dim,)).copy()
        self.any_periodic = bool(self.periodic.any())

        # A periodic axis needs no padding -- the stencil wraps instead -- and
        # carries exactly ``shape[a]`` distinct nodes, because the node at
        # ``x = L`` *is* the node at ``x = 0``. Keeping the extra one would
        # double-count every quantity scattered onto the seam.
        self.pad = np.where(self.periodic, 0, basis.stencil_size).astype(np.int64)
        self.node_shape = tuple(
            int(s) if p else int(s + 1 + 2 * q)
            for s, p, q in zip(self.shape, self.periodic, self.pad)
        )
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
    def build_stencil(self, xp: np.ndarray, lp: np.ndarray | None = None,
                      hessian: bool = False) -> StencilData:
        """Weights, gradients and node indices for every particle.

        One pass, fully vectorised, no Python loop over particles.

        Parameters
        ----------
        xp : (n, dim) particle positions
        lp : (n, dim) particle domain half-widths in *physical* units. Required
             by domain bases such as :class:`uGIMP`; ignored by point bases.
        hessian : also fill ``StencilData.hess`` with the second gradients
             ``d2w/dx_J dx_K``. Off by default because nothing in the time step
             needs it and it costs another ``dim**2`` tensor products; it exists
             for configurational-force diagnostics. Raises for a basis whose
             second derivative is not a function -- see
             :meth:`~.basis.Basis.second_derivatives_1d`.
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

        # Node *positions* come from the unwrapped indices, below, and the
        # wrapping happens only to the index used for gather/scatter. Wrapping
        # first would make ``dpos`` jump by a domain length across the seam and
        # hand APIC an affine velocity field with a discontinuity in it.
        if self.any_periodic:
            wrapped = node_idx.copy()
            for a in np.flatnonzero(self.periodic):
                wrapped[:, :, a] %= self.node_shape[a]
            node_id = (wrapped * self._strides).sum(axis=2)
        else:
            node_id = (node_idx * self._strides).sum(axis=2)       # (n, K)

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

        hess = None
        if hessian:
            d2 = self.basis.second_derivatives_1d(fx, lam) * self.inv_dx ** 2
            second = [d2[:, off[:, a], a] for a in range(self.dim)]
            # tensor product: the JK entry takes the second derivative along J
            # when J == K, and one first derivative along each otherwise
            hess = np.empty((n, K, self.dim, self.dim), dtype=self.dtype)
            for j in range(self.dim):
                for l in range(self.dim):
                    g = np.ones((n, K), dtype=self.dtype)
                    for a in range(self.dim):
                        order = (j == a) + (l == a)
                        g = g * (parts[a][0] if order == 0
                                 else parts[a][1] if order == 1
                                 else second[a])
                    hess[:, :, j, l] = g

        return StencilData(node_id, w, grad_w, dpos, hess)

    # ------------------------------------------------------------------
    def zeros_scalar(self) -> np.ndarray:
        return np.zeros(self.n_nodes, dtype=self.dtype)

    def zeros_vector(self) -> np.ndarray:
        return np.zeros((self.n_nodes, self.dim), dtype=self.dtype)

    def node_coordinates(self) -> np.ndarray:
        """(n_nodes, dim) physical coordinates, for boundary conditions."""
        axes = [np.arange(s, dtype=self.dtype) - p
                for s, p in zip(self.node_shape, self.pad)]
        mg = np.meshgrid(*axes, indexing="ij")
        return (np.stack([m.ravel() for m in mg], axis=1) * self.dx + self.origin)

    # ------------------------------------------------------------------
    def wrap_positions(self, x: np.ndarray) -> np.ndarray:
        """Fold positions back into the domain on periodic axes, **in place**.

        Returns ``x`` for chaining. A no-op on a fully non-periodic grid.

        This has to be called after every position update. The stencil wraps
        indices, so an unwrapped particle would still find *some* node and the
        run would look healthy while the body silently tiled itself across the
        domain again and again.
        """
        if not self.any_periodic:
            return x
        for a in np.flatnonzero(self.periodic):
            length = self.shape[a] * self.dx
            x[:, a] = self.origin[a] + (x[:, a] - self.origin[a]) % length
        return x

    def minimum_image(self, d: np.ndarray) -> np.ndarray:
        """Shortest separation vector under periodicity. ``(..., dim)``.

        For diagnostics that difference two particle positions -- a centre of
        mass, a radius of gyration -- across a periodic seam, where the naive
        difference is a domain length too long.
        """
        if not self.any_periodic:
            return d
        d = np.array(d, copy=True)
        for a in np.flatnonzero(self.periodic):
            length = self.shape[a] * self.dx
            d[..., a] -= length * np.round(d[..., a] / length)
        return d
