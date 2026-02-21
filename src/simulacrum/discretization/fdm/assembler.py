"""
Finite-difference assembler for structured grids.

Applies stencil operators to build sparse system matrices.
Currently supports uniform Cartesian grids in 1-D and 2-D.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from simulacrum.core.domain import Domain
from simulacrum.discretization.base import DiscretizationBase, LinearSystem
from simulacrum.discretization.fdm.stencils import laplacian_1d, laplacian_2d_5pt


class FDMAssembler(DiscretizationBase):
    """Structured-grid finite-difference assembly.

    Parameters
    ----------
    domain : Domain
    grid_shape : tuple
        Number of *interior* grid points in each direction, e.g. (nx,) or (nx, ny).
    spacing : float | tuple[float, ...]
        Grid spacing (uniform).  Scalar or per-direction.
    """

    def __init__(
        self,
        domain: Domain,
        grid_shape: tuple[int, ...],
        spacing: float | tuple[float, ...] = 1.0,
    ):
        super().__init__(domain)
        self.grid_shape = grid_shape
        self.dim = len(grid_shape)
        self.h = spacing if isinstance(spacing, (tuple, list)) else (spacing,) * self.dim

    def assemble(self) -> LinearSystem:
        if self.dim == 1:
            return self._assemble_1d()
        elif self.dim == 2:
            return self._assemble_2d()
        raise NotImplementedError(f"{self.dim}-D FDM assembly")

    def _assemble_1d(self) -> LinearSystem:
        nx = self.grid_shape[0]
        h = self.h[0]
        stencil = laplacian_1d(order=2)
        diags, offsets = [], []
        for offset, coeff in stencil.items():
            diags.append(np.full(nx, coeff / h**2))
            offsets.append(offset)
        A = sparse.diags(diags, offsets, shape=(nx, nx), format="csr")
        rhs = np.zeros(nx)
        return LinearSystem(matrix=A, rhs=rhs)

    def _assemble_2d(self) -> LinearSystem:
        nx, ny = self.grid_shape
        hx, hy = self.h
        N = nx * ny
        stencil = laplacian_2d_5pt()

        rows, cols, vals = [], [], []
        for i in range(nx):
            for j in range(ny):
                row = i * ny + j
                for (di, dj), coeff in stencil.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        col = ni * ny + nj
                        rows.append(row)
                        cols.append(col)
                        vals.append(coeff / (hx * hy))

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        rhs = np.zeros(N)
        return LinearSystem(matrix=A, rhs=rhs)

    def recover_fields(self, solution: np.ndarray) -> dict[str, np.ndarray]:
        return {"primary": solution.reshape(self.grid_shape)}
