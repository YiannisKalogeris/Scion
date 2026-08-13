r"""
Persistent quadrature-point state.

The FEM assembler is stateless: it rebuilds everything from nodal values each
call. That is fine for elasticity and wrong for anything with memory --
phase-field fracture needs the history field ``H``, plasticity needs equivalent
plastic strain and back stress, viscoelasticity needs internal variables. All of
them live at Gauss points, all of them must survive from one load step to the
next, and all of them need a *committed* value that a failed or rejected
iteration does not corrupt.

Hence this being general rather than phase-field specific.

Layout
------
Fields are stored flat, ``(n_elements * n_qp, ...)``, in the same order the
assembler visits quadrature points -- element-major, then quadrature point.
That makes every constitutive call a single batched operation over the whole
mesh, which is what the batched materials in ``materials/`` expect, and it
keeps the arrays contiguous.

Trial versus committed
----------------------
``state[name]`` is the working (trial) value. ``commit()`` copies trial into
the committed store once a load step is accepted; ``restore()`` throws the
trial away and goes back to the last committed value. A staggered scheme that
iterates on ``(u, d)`` within a load step must not let a discarded iterate
leave a permanent mark on ``H`` -- the maximum in ``H = max(H, Psi)`` is taken
against the *committed* value, not the trial one, or a non-converged excursion
would be baked in forever.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QuadratureState"]


class QuadratureState:
    """Named arrays defined at every quadrature point of a mesh.

    Parameters
    ----------
    n_elements : number of elements
    n_qp       : quadrature points per element
    """

    def __init__(self, n_elements: int, n_qp: int, dtype=np.float64):
        self.n_elements = int(n_elements)
        self.n_qp = int(n_qp)
        self.n_points = self.n_elements * self.n_qp
        self.dtype = dtype
        self._trial: dict[str, np.ndarray] = {}
        self._committed: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def add(self, name: str, shape: tuple[int, ...] = (),
            init: float | np.ndarray = 0.0) -> np.ndarray:
        """Declare a field. ``shape`` is per point: ``()``, ``(2,)``, ``(2,2)``."""
        if name in self._trial:
            raise KeyError(f"field '{name}' already exists")
        full = (self.n_points,) + tuple(shape)
        arr = (np.full(full, float(init), dtype=self.dtype)
               if np.isscalar(init)
               else np.broadcast_to(np.asarray(init, dtype=self.dtype),
                                    full).copy())
        self._trial[name] = arr
        self._committed[name] = arr.copy()
        return arr

    # ------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._trial

    def __getitem__(self, name: str) -> np.ndarray:
        return self._trial[name]

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        if name not in self._trial:
            raise KeyError(f"unknown field '{name}'; declare it with add()")
        self._trial[name][...] = value

    @property
    def names(self) -> list[str]:
        return sorted(self._trial)

    def committed(self, name: str) -> np.ndarray:
        """Last accepted value. Read-only by convention."""
        return self._committed[name]

    # ------------------------------------------------------------------
    def commit(self) -> None:
        """Accept the trial state. Call once per converged load step."""
        for k, v in self._trial.items():
            self._committed[k][...] = v

    def restore(self) -> None:
        """Discard the trial state and reset to the last committed one."""
        for k, v in self._committed.items():
            self._trial[k][...] = v

    # ------------------------------------------------------------------
    def element_view(self, name: str) -> np.ndarray:
        """``(n_elements, n_qp, ...)`` view of a field, for per-element work."""
        arr = self._trial[name]
        return arr.reshape((self.n_elements, self.n_qp) + arr.shape[1:])

    def element_average(self, name: str) -> np.ndarray:
        """Mean over the quadrature points of each element -- the natural
        quantity to write as VTK cell data."""
        return self.element_view(name).mean(axis=1)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        fields = ", ".join(f"{k}{tuple(v.shape[1:])}"
                           for k, v in sorted(self._trial.items()))
        return (f"QuadratureState({self.n_elements} elements x {self.n_qp} qp"
                f"{', ' + fields if fields else ''})")
