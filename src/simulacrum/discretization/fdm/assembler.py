r"""
Finite-difference assembly on uniform Cartesian grids.

Builds the discrete Laplacian and the right-hand side of

    lap(u) = f   inside,      u = g   on the boundary,

in one or two dimensions, at second or fourth order.

The operator is a sum, not a stencil
------------------------------------
A central Laplacian separates: it is one 1-D second difference per direction,
each divided by *its own* spacing squared,

    A = D_xx / hx^2  (x) I  +  I (x) D_yy / hy^2 ,

so the assembler builds it as a Kronecker sum rather than by scattering a 2-D
stencil dictionary. That is not a refactor for elegance. Writing the 5-point
stencil as ``{-4, 1, 1, 1, 1}`` and dividing the lot by one number is only
correct when ``hx == hy``, and on an anisotropic grid it produces an operator
that does not converge to the Laplacian at all -- the error sits at 20% and
stays there under refinement, because the scheme is consistent with the wrong
equation. Building per direction makes anisotropy and higher order fall out for
free.

Where the boundary is
---------------------
``grid_shape`` counts **interior** points. Interior point ``i`` of a direction
sits at ``origin + (i+1) h``, so the boundaries are at ``origin`` and at
``origin + (n+1) h`` and no unknown lives on them. A stencil reaching past the
last interior point therefore reaches a *boundary* value, which is known: it
moves to the right-hand side rather than into the matrix.

That is a genuine Dirichlet condition, and it is worth stating because the old
behaviour -- dropping out-of-range neighbours and leaving the right-hand side
zero -- is what you get for ``g = 0`` and nothing else. Silently. There was no
way to pose a problem with a source term or a nonzero boundary value at all.

Fourth order reaches outside the domain
---------------------------------------
The fourth-order stencil spans two cells, so at the first interior point it
wants a value at ``origin - h``, one cell *beyond* the boundary. There is no
avoiding that without one-sided stencils, which cost the order they were meant
to buy. The assembler evaluates ``boundary`` wherever the stencil actually
reaches, including outside; for a manufactured solution that is free, since the
exact solution is defined everywhere. For a problem where ``g`` is only known on
the boundary itself, use second order, or accept the reduction near the edge.

Sign
----
``A`` approximates ``lap``, so it is **negative** definite and the system is
``A u = f``. Solvers that want a positive definite matrix want ``-A`` and
``-f``; the sign is not hidden inside the assembly, because a scheme that
silently negates is impossible to compose with anything.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from simulacrum.core.domain import Domain
from simulacrum.discretization.base import DiscretizationBase, LinearSystem
from simulacrum.discretization.fdm.stencils import (
    laplacian_1d,
    laplacian_2d_9pt,
    mehrstellen_rhs_stencil,
)

__all__ = ["FDMAssembler"]


class FDMAssembler(DiscretizationBase):
    """Structured-grid finite-difference assembly of the Laplacian.

    Parameters
    ----------
    domain : a ``Domain``, or ``None``. The grid is described entirely by
        ``grid_shape``, ``spacing`` and ``origin``; the domain is accepted for
        interface consistency with the other discretizations and is not read.
    grid_shape : number of **interior** points per direction, ``(nx,)`` or
        ``(nx, ny)``.
    spacing : grid spacing, scalar or one per direction. Anisotropic spacing is
        supported and tested; it is the case the previous implementation got
        wrong.
    origin : coordinate of the lower boundary per direction, default zero. The
        domain then spans ``origin`` to ``origin + (n+1) h``.
    order : accuracy of the central difference, 2 or 4.
    scheme : ``"central"`` for the separable stencil of the given order, or
        ``"compact"`` for the 9-point Mehrstellen operator, which is 2-D only
        and needs a square grid.
    """

    def __init__(self, domain: Domain | None, grid_shape, spacing=1.0,
                 origin=0.0, order: int = 2, scheme: str = "central"):
        if domain is not None:
            super().__init__(domain)
        else:
            self.domain = None

        self.grid_shape = tuple(int(n) for n in np.atleast_1d(grid_shape))
        self.dim = len(self.grid_shape)
        if self.dim not in (1, 2):
            raise NotImplementedError(f"{self.dim}-D FDM assembly")

        self.h = tuple(float(x) for x in np.broadcast_to(
            np.asarray(spacing, dtype=float), (self.dim,)))
        self.origin = tuple(float(x) for x in np.broadcast_to(
            np.asarray(origin, dtype=float), (self.dim,)))
        self.order = int(order)

        if scheme not in ("central", "compact"):
            raise ValueError(f"unknown scheme {scheme!r}; "
                             f"use 'central' or 'compact'")
        if scheme == "compact":
            if self.dim != 2:
                raise NotImplementedError(
                    "the compact 9-point stencil is two-dimensional")
            if not np.isclose(self.h[0], self.h[1]):
                raise ValueError(
                    f"the compact stencil mixes the corners and is only the "
                    f"Laplacian on a square grid; got hx={self.h[0]:g}, "
                    f"hy={self.h[1]:g}. Use scheme='central', which builds each "
                    f"direction separately and handles any spacing.")
        self.scheme = scheme

    # ------------------------------------------------------------------
    @property
    def n_unknowns(self) -> int:
        return int(np.prod(self.grid_shape))

    def axis(self, d: int) -> np.ndarray:
        """Interior coordinates along direction ``d``, ``(n_d,)``."""
        n, h, o = self.grid_shape[d], self.h[d], self.origin[d]
        return o + h * np.arange(1, n + 1)

    def coordinates(self) -> np.ndarray:
        """Interior point coordinates, ``(n_unknowns, dim)``, in solve order.

        Row-major: the last direction varies fastest, matching
        :meth:`recover_fields`.
        """
        axes = [self.axis(d) for d in range(self.dim)]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1)

    def _sample(self, value, points: np.ndarray) -> np.ndarray:
        """Evaluate a scalar, an array, or a callable at ``points``."""
        if value is None:
            return np.zeros(points.shape[0])
        if callable(value):
            return np.asarray(value(*points.T), dtype=float).ravel()
        arr = np.asarray(value, dtype=float)
        return (np.full(points.shape[0], float(arr)) if arr.ndim == 0
                else arr.ravel())

    # ------------------------------------------------------------------
    def _line(self, d: int) -> sparse.csr_matrix:
        """1-D operator along direction ``d``, boundary columns dropped."""
        n, h = self.grid_shape[d], self.h[d]
        stencil = laplacian_1d(self.order)
        rows, cols, vals = [], [], []
        for i in range(n):
            for offset, c in stencil.items():
                j = i + offset
                if 0 <= j < n:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c / h ** 2)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def _central_operator(self) -> sparse.csr_matrix:
        if self.dim == 1:
            return self._line(0)
        nx, ny = self.grid_shape
        return (sparse.kron(self._line(0), sparse.eye(ny))
                + sparse.kron(sparse.eye(nx), self._line(1))).tocsr()

    def _compact_operator(self) -> sparse.csr_matrix:
        nx, ny = self.grid_shape
        h = self.h[0]
        stencil = laplacian_2d_9pt()
        rows, cols, vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                for (di, dj), c in stencil.items():
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        rows.append(i * ny + j)
                        cols.append(ii * ny + jj)
                        vals.append(c / h ** 2)
        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(nx * ny, nx * ny))

    def operator(self) -> sparse.csr_matrix:
        """The discrete Laplacian alone, without any right-hand side."""
        return (self._compact_operator() if self.scheme == "compact"
                else self._central_operator())

    # ------------------------------------------------------------------
    def _stencil_entries(self):
        """``(offset_tuple, coefficient)`` pairs including the spacing."""
        if self.scheme == "compact":
            h2 = self.h[0] ** 2
            return [(o, c / h2) for o, c in laplacian_2d_9pt().items()]

        entries = []
        for d in range(self.dim):
            for offset, c in laplacian_1d(self.order).items():
                key = [0] * self.dim
                key[d] = offset
                if offset == 0:
                    continue                      # centre handled once below
                entries.append((tuple(key), c / self.h[d] ** 2))
        centre = sum(laplacian_1d(self.order)[0] / self.h[d] ** 2
                     for d in range(self.dim))
        entries.append((tuple([0] * self.dim), centre))
        return entries

    def _boundary_load(self, boundary) -> np.ndarray:
        """Ghost contributions moved to the right-hand side.

        For each stencil offset that leaves the interior, the neighbour's value
        is known, so its term ``c * u_ghost`` is subtracted from the equation
        rather than kept as an unknown.
        """
        if boundary is None:
            return np.zeros(self.n_unknowns)

        load = np.zeros(self.grid_shape)
        index = np.indices(self.grid_shape)
        axes = [self.axis(d) for d in range(self.dim)]
        base = np.meshgrid(*axes, indexing="ij")

        for offset, coeff in self._stencil_entries():
            if not any(offset):
                continue
            neighbour = [index[d] + offset[d] for d in range(self.dim)]
            outside = np.zeros(self.grid_shape, dtype=bool)
            for d in range(self.dim):
                outside |= (neighbour[d] < 0) | (neighbour[d] >=
                                                 self.grid_shape[d])
            if not outside.any():
                continue

            # the coordinate the stencil actually reaches, which for a
            # fourth-order stencil may be a full cell outside the domain
            where = np.nonzero(outside)
            points = np.stack(
                [base[d][where] + offset[d] * self.h[d]
                 for d in range(self.dim)], axis=1)
            load[where] += coeff * self._sample(boundary, points)

        return load.ravel()

    def _mehrstellen(self, source) -> np.ndarray:
        """``f + (h^2/12) lap f``, sampled with the correction stencil."""
        nx, ny = self.grid_shape
        axes = [self.axis(d) for d in range(2)]
        base = np.meshgrid(*axes, indexing="ij")
        out = np.zeros(self.grid_shape)
        for (di, dj), c in mehrstellen_rhs_stencil().items():
            points = np.stack([(base[0] + di * self.h[0]).ravel(),
                               (base[1] + dj * self.h[1]).ravel()], axis=1)
            out += c * self._sample(source, points).reshape(nx, ny)
        return out.ravel()

    # ------------------------------------------------------------------
    def assemble(self, source=None, boundary=None,
                 mehrstellen: bool = False) -> LinearSystem:
        """``A u = b`` for ``lap(u) = f`` with Dirichlet data.

        Parameters
        ----------
        source : ``f`` at the interior points. Scalar, array of ``grid_shape``,
            or a callable of the coordinates. ``None`` gives Laplace's equation.
        boundary : the Dirichlet value ``g``, same forms. ``None`` or ``0.0``
            gives the homogeneous condition, which is what the assembler used to
            do unconditionally.
        mehrstellen : apply the ``f + (h^2/12) lap f`` correction that makes the
            compact scheme fourth order. Without it the 9-point stencil is only
            second order and costs more than the 5-point one; see
            :func:`~simulacrum.discretization.fdm.stencils.laplacian_2d_9pt`.
        """
        if mehrstellen and self.scheme != "compact":
            raise ValueError(
                "the Mehrstellen correction belongs to the compact scheme; "
                "for scheme='central' raise `order` instead")

        points = self.coordinates()
        rhs = (self._mehrstellen(source) if mehrstellen
               else self._sample(source, points))
        return LinearSystem(matrix=self.operator(),
                            rhs=rhs - self._boundary_load(boundary))

    def recover_fields(self, solution: np.ndarray) -> dict[str, np.ndarray]:
        return {"primary": np.asarray(solution).reshape(self.grid_shape)}

    def __repr__(self) -> str:
        shape = " x ".join(str(n) for n in self.grid_shape)
        spacing = ", ".join(f"{x:g}" for x in self.h)
        return (f"FDMAssembler({shape} interior points, h=({spacing}), "
                f"order={self.order}, scheme='{self.scheme}')")
