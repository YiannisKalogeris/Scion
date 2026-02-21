"""
Computational domain definition.

A Domain ties together a Mesh with the physics (fields, BCs, material zones)
that live on it.  It is the top-level object handed to any discretization scheme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulacrum.mesh.mesh import Mesh
    from simulacrum.boundary.bc import BoundaryCondition
    from simulacrum.materials.base import Material


@dataclass
class Domain:
    """A computational domain combining geometry, physics, and material data.

    Parameters
    ----------
    mesh : Mesh
        The underlying spatial discretization.
    dim : int
        Spatial dimension (1, 2, or 3).
    materials : dict[str, Material]
        Named material models assigned to element groups / subdomains.
    bcs : list[BoundaryCondition]
        Boundary conditions imposed on the domain.
    """

    mesh: Mesh
    dim: int = 2
    materials: dict[str, Material] = field(default_factory=dict)
    bcs: list[BoundaryCondition] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.mesh.n_nodes

    @property
    def n_elements(self) -> int:
        return self.mesh.n_elements

    @property
    def coordinates(self) -> np.ndarray:
        return self.mesh.nodes

    # ------------------------------------------------------------------
    def add_material(self, name: str, material: Material, region: str | None = None) -> None:
        """Assign a material model to a named subdomain / element group."""
        self.materials[name] = material

    def add_bc(self, bc: BoundaryCondition) -> None:
        self.bcs.append(bc)
