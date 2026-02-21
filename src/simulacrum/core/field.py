"""
Discrete field living on nodes, elements, or quadrature points.

A Field wraps a NumPy array with metadata about where it lives (nodal vs
cell-centred vs quadrature) and how many components it has.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np


class FieldLocation(Enum):
    """Where a discrete field is defined."""
    NODAL = auto()
    CELL = auto()
    QUADRATURE = auto()


class Field:
    """A discrete scalar or vector field on a mesh.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"displacement"``, ``"temperature"``).
    n_points : int
        Number of points (nodes, cells, or quad points) the field is defined on.
    n_components : int
        Number of components per point (1 for scalar, dim for vector, etc.).
    location : FieldLocation
        Where the field lives on the mesh.
    """

    def __init__(
        self,
        name: str,
        n_points: int,
        n_components: int = 1,
        location: FieldLocation = FieldLocation.NODAL,
    ):
        self.name = name
        self.location = location
        self.n_components = n_components
        self.values = np.zeros((n_points, n_components), dtype=np.float64)

    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def flat(self) -> np.ndarray:
        """Return a flat (1-D) view for assembly into global vectors."""
        return self.values.ravel()

    # ------------------------------------------------------------------
    def set(self, data: np.ndarray) -> None:
        self.values[:] = np.asarray(data).reshape(self.values.shape)

    def __repr__(self) -> str:
        return (
            f"Field('{self.name}', n_points={self.values.shape[0]}, "
            f"n_comp={self.n_components}, loc={self.location.name})"
        )
