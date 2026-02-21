"""
Core mesh data structure.

Stores nodes, element connectivity, boundary tags, and provides
topology queries needed by all discretization schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# Canonical element-type names used throughout Simulacrum
ELEMENT_TYPES = {
    "line2", "line3",
    "tri3", "tri6",
    "quad4", "quad8", "quad9",
    "tet4", "tet10",
    "hex8", "hex20", "hex27",
}


@dataclass
class ElementBlock:
    """A group of elements sharing the same topology.

    Attributes
    ----------
    etype : str
        Element type name (e.g. ``"tri3"``, ``"hex8"``).
    connectivity : np.ndarray
        (n_elements, nodes_per_element) integer array.
    tag : str
        Optional label for material / subdomain assignment.
    """
    etype: str
    connectivity: np.ndarray
    tag: str = "default"


@dataclass
class Mesh:
    """Unstructured mesh.

    Parameters
    ----------
    nodes : np.ndarray
        (n_nodes, dim) coordinate array.
    element_blocks : list[ElementBlock]
        One or more groups of elements.
    boundary_tags : dict[str, np.ndarray]
        Named sets of boundary node indices.
    """

    nodes: np.ndarray
    element_blocks: list[ElementBlock] = field(default_factory=list)
    boundary_tags: dict[str, np.ndarray] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.nodes.shape[1]

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return sum(blk.connectivity.shape[0] for blk in self.element_blocks)

    # ------------------------------------------------------------------
    def get_block(self, tag: str) -> ElementBlock:
        for blk in self.element_blocks:
            if blk.tag == tag:
                return blk
        raise KeyError(f"No element block with tag '{tag}'")

    def boundary_nodes(self, name: str) -> np.ndarray:
        return self.boundary_tags[name]

    def add_boundary(self, name: str, node_ids: Sequence[int] | np.ndarray) -> None:
        self.boundary_tags[name] = np.asarray(node_ids, dtype=np.int64)
