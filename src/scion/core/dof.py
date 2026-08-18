"""
Degree-of-freedom mapping.

Translates between ``(node, component)`` pairs and global equation numbers, and
tracks which of them are constrained.

Ordering is **node-major**: node ``i``, component ``j`` is global DOF
``i * n_components + j``, so a node's components are contiguous. That is what
lets an element's DOFs be built by broadcasting rather than by a loop, and it is
what the ``[0::2]`` / ``[1::2]`` slicing all over the tests relies on to pick
out the x- and y-components of a boundary. Component-major numbering would be
equally valid and would silently break every one of those slices, so the choice
is stated rather than left to be inferred.

Indices are validated
---------------------
``constrain`` and ``dof`` check their arguments, which is not fussiness. Python
indexing accepts negatives, so ``constrain(-1, 0)`` on a 5-node 2-component map
used to remove global DOF 8 from the free set -- a real DOF, belonging to a
different node, with nothing to show anything had gone wrong. And an index past
the end left the map inconsistent: ``n_free`` counted the bogus constraint while
``free_dofs`` raised ``IndexError`` much later, somewhere else entirely.

Neither is hot. ``element_dofs`` is the only routine on this class that runs per
element, every caller caches its result, and it does its checking with one
vectorised comparison.
"""

from __future__ import annotations

import numpy as np

__all__ = ["DOFMap"]


class DOFMap:
    """Map ``(node, component)`` to a global equation number.

    Parameters
    ----------
    n_nodes : total number of mesh nodes.
    n_components : DOFs per node -- 2 for 2-D elasticity, 1 for a scalar field.
    """

    def __init__(self, n_nodes: int, n_components: int = 1):
        self.n_nodes = int(n_nodes)
        self.n_components = int(n_components)
        if self.n_nodes < 0:
            raise ValueError(f"n_nodes must not be negative, got {n_nodes}")
        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least one, got {n_components}")

        self._total = self.n_nodes * self.n_components
        self._constrained = np.zeros(self._total, dtype=bool)

    # ------------------------------------------------------------------
    def _check(self, node: int, component: int) -> None:
        if not 0 <= node < self.n_nodes:
            raise IndexError(
                f"node {node} is outside [0, {self.n_nodes}); a negative index "
                f"would silently wrap onto a different node's DOF")
        if not 0 <= component < self.n_components:
            raise IndexError(
                f"component {component} is outside [0, {self.n_components})")

    def dof(self, node: int, component: int = 0) -> int:
        """Global DOF index for one ``(node, component)`` pair."""
        node, component = int(node), int(component)
        self._check(node, component)
        return node * self.n_components + component

    def element_dofs(self, connectivity: np.ndarray) -> np.ndarray:
        """DOF indices for an element, from its node connectivity.

        ``(n_element_nodes,) -> (n_element_nodes * n_components,)``, node-major,
        so the result reads ``[n0_x, n0_y, n1_x, n1_y, ...]`` in 2-D.
        """
        connectivity = np.asarray(connectivity, dtype=np.int64).ravel()
        if connectivity.size and (connectivity.min() < 0
                                  or connectivity.max() >= self.n_nodes):
            raise IndexError(
                f"connectivity refers to nodes outside [0, {self.n_nodes}): "
                f"min {connectivity.min()}, max {connectivity.max()}")

        base = np.repeat(connectivity * self.n_components, self.n_components)
        offsets = np.tile(np.arange(self.n_components), connectivity.size)
        return base + offsets

    # ------------------------------------------------------------------
    def constrain(self, node: int, component: int) -> None:
        """Mark one DOF as constrained (Dirichlet). Idempotent."""
        self._constrained[self.dof(node, component)] = True

    def release(self, node: int, component: int) -> None:
        """Undo :meth:`constrain`. Idempotent."""
        self._constrained[self.dof(node, component)] = False

    def is_constrained(self, node: int, component: int = 0) -> bool:
        return bool(self._constrained[self.dof(node, component)])

    # ------------------------------------------------------------------
    @property
    def free_dofs(self) -> np.ndarray:
        """Global indices of the unconstrained DOFs, ascending."""
        return np.flatnonzero(~self._constrained).astype(np.int64)

    @property
    def constrained_dofs(self) -> np.ndarray:
        """Global indices of the constrained DOFs, ascending."""
        return np.flatnonzero(self._constrained).astype(np.int64)

    @property
    def n_free(self) -> int:
        return int(self._total - self._constrained.sum())

    @property
    def n_constrained(self) -> int:
        return int(self._constrained.sum())

    @property
    def n_total(self) -> int:
        return self._total

    def __len__(self) -> int:
        return self._total

    def __repr__(self) -> str:
        return (f"DOFMap({self.n_nodes} nodes x {self.n_components} "
                f"components, {self.n_free}/{self._total} free)")
