"""
Boundary condition definitions and application.

Supports Dirichlet (essential) and Neumann (natural) conditions.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse

from scion.core.dof import DOFMap


class BoundaryCondition(ABC):
    """Base for all boundary conditions."""
    pass


@dataclass
class DirichletBC(BoundaryCondition):
    """Fixed-value (essential) boundary condition.

    Parameters
    ----------
    node_ids : array-like
        Mesh nodes where the BC is applied.
    component : int
        DOF component (0 = x, 1 = y, ...).
    value : float | Callable
        Prescribed value or function ``f(coords) -> value``.
    """
    node_ids: np.ndarray
    component: int = 0
    value: float | Callable = 0.0

    def get_value(self, coords: np.ndarray | None = None) -> float | np.ndarray:
        if callable(self.value):
            return self.value(coords)
        return self.value


@dataclass
class NeumannBC(BoundaryCondition):
    """Flux / traction (natural) boundary condition.

    Parameters
    ----------
    node_ids : array-like
        Boundary nodes.
    component : int
        DOF component.
    value : float | Callable
        Prescribed flux / traction.
    """
    node_ids: np.ndarray
    component: int = 0
    value: float | Callable = 0.0


def apply_dirichlet_penalty(
    K: sparse.spmatrix,
    f: np.ndarray,
    dof_map: DOFMap,
    bcs: list[DirichletBC],
    penalty: float = 1e20,
) -> tuple[sparse.spmatrix, np.ndarray]:
    """Apply Dirichlet BCs via the penalty method."""
    K = K.tolil()
    for bc in bcs:
        for node in bc.node_ids:
            dof = dof_map.dof(node, bc.component)
            val = bc.get_value()
            K[dof, dof] += penalty
            f[dof] += penalty * val
    return K.tocsr(), f


def apply_dirichlet_elimination(
    K: sparse.spmatrix,
    f: np.ndarray,
    dof_map: DOFMap,
    bcs: list[DirichletBC],
) -> tuple[sparse.spmatrix, np.ndarray]:
    """Apply Dirichlet BCs by row/column elimination."""
    K = K.tolil()
    for bc in bcs:
        for node in bc.node_ids:
            dof = dof_map.dof(node, bc.component)
            val = bc.get_value() if not callable(bc.value) else 0.0
            # Zero row and column, set diagonal to 1
            K[dof, :] = 0
            K[:, dof] = 0
            K[dof, dof] = 1.0
            f[dof] = val
    return K.tocsr(), f
