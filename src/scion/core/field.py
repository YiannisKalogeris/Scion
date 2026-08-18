"""
A discrete field on nodes, cells, or quadrature points.

Wraps a ``(n_points, n_components)`` array with the one piece of metadata that
cannot be recovered from its shape: *where* the values live. A nodal field and a
cell field on the same mesh have different lengths, but a nodal field and a
quadrature field can easily have the same one, and interpolating the wrong one
produces a plausible answer.

``flat`` is a view, not a copy
-----------------------------
:attr:`Field.flat` returns ``values.ravel()``, and because the array is
C-contiguous and only ever written through ``[:]``, that is a genuine view --
writing into it writes into :attr:`values`. Assembly routines rely on this to
scatter into a global vector without a copy. It is worth stating because
``ravel`` silently returns a *copy* when its input is not contiguous, so the
guarantee is a property of how this class manages its array and not of
``ravel`` itself.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

__all__ = ["Field", "FieldLocation"]


class FieldLocation(Enum):
    """Where a discrete field is defined."""

    NODAL = auto()
    CELL = auto()
    QUADRATURE = auto()


class Field:
    """A discrete scalar or vector field.

    Parameters
    ----------
    name : human-readable, e.g. ``"displacement"``.
    n_points : number of nodes, cells, or quadrature points.
    n_components : 1 for a scalar, ``dim`` for a vector.
    location : which of those ``n_points`` counts.
    """

    def __init__(self, name: str, n_points: int, n_components: int = 1,
                 location: FieldLocation = FieldLocation.NODAL):
        n_points, n_components = int(n_points), int(n_components)
        if n_points < 0:
            raise ValueError(f"n_points must not be negative, got {n_points}")
        if n_components < 1:
            raise ValueError(
                f"n_components must be at least one, got {n_components}")

        self.name = name
        self.location = location
        self.n_components = n_components
        self.values = np.zeros((n_points, n_components), dtype=np.float64)

    # ------------------------------------------------------------------
    @property
    def n_points(self) -> int:
        return int(self.values.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def flat(self) -> np.ndarray:
        """A flat **view** for assembly into global vectors. Writes through."""
        return self.values.ravel()

    # ------------------------------------------------------------------
    def set(self, data) -> None:
        """Overwrite the values, from any array of matching size.

        Accepts either the ``(n_points, n_components)`` shape or a flat one,
        since both turn up -- a solver returns a flat vector and a
        post-processor a two-dimensional block.
        """
        data = np.asarray(data, dtype=float)
        if data.size != self.values.size:
            raise ValueError(
                f"field '{self.name}' holds {self.values.size} values "
                f"({self.n_points} x {self.n_components}) but was given "
                f"{data.size}")
        self.values[:] = data.reshape(self.values.shape)

    def fill(self, value: float) -> None:
        self.values[:] = float(value)

    def copy(self) -> "Field":
        other = Field(self.name, self.n_points, self.n_components,
                      self.location)
        other.values[:] = self.values
        return other

    def __array__(self, dtype=None):
        return self.values if dtype is None else self.values.astype(dtype)

    def __repr__(self) -> str:
        return (f"Field('{self.name}', n_points={self.n_points}, "
                f"n_comp={self.n_components}, loc={self.location.name})")
