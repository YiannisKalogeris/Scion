"""
Degree-of-freedom (DOF) mapping.

Translates between (node, component) pairs and global equation numbers,
handling constrained (Dirichlet) DOFs.
"""

from __future__ import annotations

import numpy as np


class DOFMap:
    """Map (node_id, component) → global equation number.

    Parameters
    ----------
    n_nodes : int
        Total number of mesh nodes.
    n_components : int
        DOFs per node (e.g. 2 for 2-D elasticity, 1 for scalar problems).
    """

    def __init__(self, n_nodes: int, n_components: int = 1):
        self.n_nodes = n_nodes
        self.n_components = n_components
        self._total = n_nodes * n_components
        # full numbering: node i, comp j → i*n_comp + j
        self._all_dofs = np.arange(self._total, dtype=np.int64)
        self._constrained: set[int] = set()

    # ------------------------------------------------------------------
    def dof(self, node: int, component: int = 0) -> int:
        """Global DOF index for a (node, component) pair."""
        return node * self.n_components + component

    def element_dofs(self, connectivity: np.ndarray) -> np.ndarray:
        """Return DOF indices for an element given its node connectivity.

        Parameters
        ----------
        connectivity : (n_el_nodes,) array of node indices

        Returns
        -------
        (n_el_nodes * n_components,) array of global DOF indices
        """
        base = np.repeat(connectivity * self.n_components, self.n_components)
        offsets = np.tile(np.arange(self.n_components), len(connectivity))
        return base + offsets

    # ------------------------------------------------------------------
    def constrain(self, node: int, component: int) -> None:
        """Mark a DOF as constrained (Dirichlet)."""
        self._constrained.add(self.dof(node, component))

    @property
    def free_dofs(self) -> np.ndarray:
        mask = np.ones(self._total, dtype=bool)
        for d in self._constrained:
            mask[d] = False
        return self._all_dofs[mask]

    @property
    def n_free(self) -> int:
        return self._total - len(self._constrained)

    @property
    def n_total(self) -> int:
        return self._total
