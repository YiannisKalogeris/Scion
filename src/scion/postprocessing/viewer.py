"""
Mesh and field visualisation using PyVista.

Provides convenience wrappers that convert Scion meshes / fields
into PyVista objects and launch interactive plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scion.mesh.mesh import Mesh


def _to_pyvista(mesh: Mesh) -> Any:
    """Convert a Scion Mesh to a PyVista UnstructuredGrid."""
    import pyvista as pv

    # VTK cell-type constants
    _VTK_TYPES = {
        "line2": 3, "tri3": 5, "quad4": 9,
        "tet4": 10, "hex8": 12, "tri6": 22,
        "quad8": 23, "tet10": 24, "hex20": 25,
    }

    nodes = mesh.nodes
    if nodes.shape[1] < 3:
        nodes = np.hstack([nodes, np.zeros((nodes.shape[0], 3 - nodes.shape[1]))])

    cells, celltypes = [], []
    for blk in mesh.element_blocks:
        vtype = _VTK_TYPES.get(blk.etype)
        if vtype is None:
            raise ValueError(f"No VTK mapping for element type '{blk.etype}'")
        n_per = blk.connectivity.shape[1]
        for conn in blk.connectivity:
            cells.extend([n_per, *conn])
            celltypes.append(vtype)

    grid = pv.UnstructuredGrid(
        np.array(cells, dtype=np.int64),
        np.array(celltypes, dtype=np.uint8),
        nodes,
    )
    return grid


def plot_mesh(mesh: Mesh, **kwargs: Any) -> None:
    """Plot the mesh wireframe."""
    grid = _to_pyvista(mesh)
    grid.plot(show_edges=True, color="white", **kwargs)


def plot_field(
    mesh: Mesh,
    field_data: np.ndarray,
    field_name: str = "field",
    component: int | None = None,
    **kwargs: Any,
) -> None:
    """Plot a scalar or vector field on the mesh.

    Parameters
    ----------
    mesh : Mesh
    field_data : (n_nodes,) or (n_nodes, n_comp) array
    field_name : label for the colour bar
    component : if vector field, which component to display (None = magnitude)
    """
    grid = _to_pyvista(mesh)

    data = np.asarray(field_data)
    if data.ndim == 1:
        grid.point_data[field_name] = data
    elif component is not None:
        grid.point_data[field_name] = data[:, component]
    else:
        grid.point_data[field_name] = np.linalg.norm(data, axis=1)

    grid.plot(scalars=field_name, show_edges=True, cmap="viridis", **kwargs)
