"""
The computational domain: geometry plus the physics that lives on it.

A :class:`Domain` is what gets handed to a discretisation scheme. It carries the
mesh, the spatial dimension, the material assigned to each region, and the
boundary conditions.

Regions
-------
A *region* names a subdomain -- an element group, a layer, a component. Each
region has exactly one material, which is the physical statement: a point of
the body is made of one thing.

The distinction that makes the bookkeeping work is between a material *model*
and a material *assignment*. The same steel can be assigned to three regions
without being registered three times, and two regions can hold genuinely
different models. Both are ordinary, and an earlier version of this class
handled neither: ``add_material`` took a ``region`` argument and discarded it,
so two calls with the same name silently collapsed into one entry and the
second model won.

    domain.add_material("steel", steel, region="flange")
    domain.add_material("steel", steel, region="web")      # same model, fine
    domain.add_material("concrete", concrete, region="slab")

Re-registering a *name* with a different model is refused, because that is the
case where one of the two is a mistake and neither is recoverable afterwards.
``region=None`` means the whole domain and is the default, so the single-material
case needs no region at all.

Note that nothing in the library reads these assignments yet -- every assembler
is currently handed its material directly. The structure is here so that when
one does, the meaning is already fixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulacrum.boundary.bc import BoundaryCondition
    from simulacrum.materials.base import Material
    from simulacrum.mesh.mesh import Mesh

__all__ = ["Domain"]


@dataclass
class Domain:
    """Geometry, materials and boundary conditions for one problem.

    Parameters
    ----------
    mesh : the spatial discretisation.
    dim : spatial dimension. Checked against the mesh's node coordinates, since
        a mismatch produces an assembler that runs and returns nonsense.
    materials : material models by name. Prefer :meth:`add_material`.
    bcs : boundary conditions.
    assignments : region name to material name. ``None`` keys the whole domain.
    """

    mesh: Mesh
    dim: int = 2
    materials: dict[str, Material] = field(default_factory=dict)
    bcs: list[BoundaryCondition] = field(default_factory=list)
    assignments: dict[str | None, str] = field(default_factory=dict)

    def __post_init__(self):
        self.dim = int(self.dim)
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2 or 3, got {self.dim}")

        nodes = getattr(self.mesh, "nodes", None)
        if nodes is not None and np.ndim(nodes) == 2:
            mesh_dim = int(np.shape(nodes)[1])
            if mesh_dim != self.dim:
                raise ValueError(
                    f"dim={self.dim} but the mesh has {mesh_dim}-D node "
                    f"coordinates; an assembler built on the mismatch runs and "
                    f"returns nonsense rather than failing")

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
    def add_material(self, name: str, material: Material,
                     region: str | None = None) -> None:
        """Register ``material`` under ``name`` and assign it to ``region``.

        ``region=None`` means the whole domain. Assigning the same model to
        several regions is the normal case and costs one call each; registering
        one *name* against two different models is refused, and so is assigning
        two materials to the same region.
        """
        existing = self.materials.get(name)
        if existing is not None and existing is not material:
            raise ValueError(
                f"material name {name!r} is already registered to a different "
                f"model; use a distinct name, or pass the same object to "
                f"assign it to another region")

        assigned = self.assignments.get(region)
        if assigned is not None and assigned != name:
            raise ValueError(
                f"region {region!r} already holds material {assigned!r}; a "
                f"region is made of one material, so reassigning is a mistake "
                f"rather than an update")

        self.materials[name] = material
        self.assignments[region] = name

    def material_for(self, region: str | None = None) -> Material:
        """The material assigned to ``region``.

        Falls back to the whole-domain assignment (``region=None``) when the
        region has none of its own, so a single-material problem needs no
        regions at all.
        """
        name = self.assignments.get(region, self.assignments.get(None))
        if name is None:
            known = sorted(r for r in self.assignments if r is not None)
            raise KeyError(
                f"no material assigned to region {region!r} and no "
                f"whole-domain default; assigned regions: {known or 'none'}")
        return self.materials[name]

    @property
    def regions(self) -> tuple[str, ...]:
        """Named regions, excluding the whole-domain assignment."""
        return tuple(sorted(r for r in self.assignments if r is not None))

    def add_bc(self, bc: BoundaryCondition) -> None:
        self.bcs.append(bc)

    def __repr__(self) -> str:
        try:
            shape = f"{self.n_nodes} nodes, {self.n_elements} elements"
        except AttributeError:
            shape = "no mesh"
        return (f"Domain({self.dim}-D, {shape}, "
                f"{len(self.materials)} materials, {len(self.bcs)} bcs)")
