"""
Thin wrapper around ``gmsh`` and ``meshio`` for mesh import.

Usage
-----
>>> mesh = mesh_from_gmsh("plate_with_hole.geo")
>>> mesh = mesh_from_file("mesh.msh")   # or .vtu, .xdmf, …
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from simulacrum.mesh.mesh import Mesh, ElementBlock

# meshio cell-type → Simulacrum element-type
_MESHIO_MAP: dict[str, str] = {
    "line": "line2",
    "line3": "line3",
    "triangle": "tri3",
    "triangle6": "tri6",
    "quad": "quad4",
    "quad8": "quad8",
    "quad9": "quad9",
    "tetra": "tet4",
    "tetra10": "tet10",
    "hexahedron": "hex8",
    "hexahedron20": "hex20",
    "hexahedron27": "hex27",
}


def mesh_from_gmsh(geo_or_msh: str | Path, dim: int = 2) -> Mesh:
    """Generate or read a mesh via the Gmsh Python API.

    Parameters
    ----------
    geo_or_msh : path to a ``.geo`` or ``.msh`` file.
    dim : target mesh dimension (used when meshing a .geo file).
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    path = Path(geo_or_msh)
    if path.suffix == ".geo":
        gmsh.open(str(path))
        gmsh.model.mesh.generate(dim)
    elif path.suffix == ".msh":
        gmsh.open(str(path))
    else:
        gmsh.finalize()
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Extract nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_map = {int(t): i for i, t in enumerate(node_tags)}
    nodes = coords.reshape(-1, 3)[:, :dim]

    # Extract elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=dim)
    blocks = []
    for etype_id, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        name, _, _, n_per_elem, _, _ = gmsh.model.mesh.getElementProperties(etype_id)
        conn = np.array([node_map[int(n)] for n in enodes]).reshape(-1, n_per_elem)
        gmsh_name = name.lower().replace(" ", "")
        sim_etype = _MESHIO_MAP.get(gmsh_name, gmsh_name)
        blocks.append(ElementBlock(etype=sim_etype, connectivity=conn))

    gmsh.finalize()
    return Mesh(nodes=nodes, element_blocks=blocks)


def mesh_from_file(path: str | Path) -> Mesh:
    """Import a mesh from any format supported by ``meshio``."""
    import meshio

    m = meshio.read(str(path))
    nodes = m.points[:, : m.points.shape[1]]
    blocks = []
    for cell_block in m.cells:
        sim_etype = _MESHIO_MAP.get(cell_block.type, cell_block.type)
        blocks.append(ElementBlock(etype=sim_etype, connectivity=cell_block.data))
    return Mesh(nodes=nodes, element_blocks=blocks)
