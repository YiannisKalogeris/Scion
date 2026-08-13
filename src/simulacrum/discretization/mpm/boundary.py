"""
MPM boundary conditions.

These do not share an interface with the FEM Dirichlet machinery, and shouldn't:
FEM constrains *displacement* degrees of freedom on a fixed mesh, while MPM
constrains *grid velocities* after P2G, on nodes that are re-selected every step
because the material moves through the grid.

Each condition is a callable ``(grid, grid_v, grid_m) -> None`` applied in order
after the grid velocity update.
"""

from __future__ import annotations

import numpy as np

from .background_grid import CartesianGrid


class GridBC:
    """Base class. Subclasses implement ``apply``."""

    def apply(self, grid: CartesianGrid, grid_v: np.ndarray,
              grid_m: np.ndarray) -> None:
        raise NotImplementedError

    def __call__(self, grid, grid_v, grid_m):
        self.apply(grid, grid_v, grid_m)


class HalfSpace(GridBC):
    """Wall at ``x[axis] = level``, material confined to one side.

    ``mode``:
        "sticky"     all velocity components zeroed
        "slip"       normal component zeroed, tangential free
        "separating" normal component zeroed only when moving into the wall
    """

    def __init__(self, axis: int, level: float, sign: int = +1,
                 mode: str = "separating"):
        self.axis = int(axis)
        self.level = float(level)
        self.sign = int(sign)   # +1: material lives at x > level
        self.mode = mode
        self._nodes = None

    def _select(self, grid):
        if self._nodes is None:
            coords = grid.node_coordinates()[:, self.axis]
            self._nodes = (coords < self.level if self.sign > 0
                           else coords > self.level)
        return self._nodes

    def apply(self, grid, grid_v, grid_m):
        sel = self._select(grid)
        if self.mode == "sticky":
            grid_v[sel] = 0.0
            return
        vn = grid_v[sel, self.axis]
        if self.mode == "separating":
            into = (vn * self.sign) < 0.0
            idx = np.where(sel)[0][into]
            grid_v[idx, self.axis] = 0.0
        else:  # slip
            grid_v[sel, self.axis] = 0.0


class Box(GridBC):
    """Convenience: six (or four) half-spaces forming a closed container."""

    def __init__(self, lo, hi, mode: str = "separating"):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        self.walls = []
        for a in range(len(lo)):
            self.walls.append(HalfSpace(a, lo[a], sign=+1, mode=mode))
            self.walls.append(HalfSpace(a, hi[a], sign=-1, mode=mode))

    def apply(self, grid, grid_v, grid_m):
        for w in self.walls:
            w.apply(grid, grid_v, grid_m)


class PrescribedVelocity(GridBC):
    """Fixed velocity on nodes selected by a predicate on node coordinates."""

    def __init__(self, predicate, value):
        self.predicate = predicate
        self.value = np.asarray(value, float)
        self._nodes = None

    def apply(self, grid, grid_v, grid_m):
        if self._nodes is None:
            self._nodes = self.predicate(grid.node_coordinates())
        grid_v[self._nodes] = self.value
