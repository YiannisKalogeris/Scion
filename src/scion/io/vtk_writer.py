"""
VTK / VTU output via meshio.

Writes simulation results to VTK files for visualisation in ParaView.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scion.mesh.mesh import Mesh

# meshio element-type mapping (Scion → meshio)
_SIM_TO_MESHIO: dict[str, str] = {
    "line2": "line",
    "tri3": "triangle",
    "tri6": "triangle6",
    "quad4": "quad",
    "quad8": "quad8",
    "quad9": "quad9",
    "tet4": "tetra",
    "tet10": "tetra10",
    "hex8": "hexahedron",
    "hex20": "hexahedron20",
    "hex27": "hexahedron27",
}


def write_vtk(
    path: str | Path,
    mesh: Mesh,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, list[np.ndarray]] | None = None,
) -> None:
    """Write a single VTU file.

    Parameters
    ----------
    path : output file path (.vtu)
    mesh : Scion Mesh
    point_data : {name: (n_nodes, ...) array}
    cell_data : {name: [array_per_block, ...]}
    """
    import meshio

    # Pad to 3-D if needed (VTK requires 3-D coordinates)
    nodes = mesh.nodes
    if nodes.shape[1] < 3:
        pad = np.zeros((nodes.shape[0], 3 - nodes.shape[1]))
        nodes = np.hstack([nodes, pad])

    cells = []
    for blk in mesh.element_blocks:
        mtype = _SIM_TO_MESHIO.get(blk.etype, blk.etype)
        cells.append(meshio.CellBlock(mtype, blk.connectivity))

    m = meshio.Mesh(
        points=nodes,
        cells=cells,
        point_data=point_data or {},
        cell_data=cell_data or {},
    )
    m.write(str(path))


def write_vtk_time_series(
    directory: str | Path,
    basename: str,
    mesh: Mesh,
    time_steps: list[float],
    fields: list[dict[str, np.ndarray]],
) -> list[Path]:
    """Write a sequence of VTU files for transient results.

    Also writes a ``.pvd`` collection file for ParaView.

    Returns
    -------
    List of written VTU file paths.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vtu_paths = []
    for i, (t, fdata) in enumerate(zip(time_steps, fields)):
        vtu_path = directory / f"{basename}_{i:06d}.vtu"
        write_vtk(vtu_path, mesh, point_data=fdata)
        vtu_paths.append(vtu_path)

    # Write PVD collection
    pvd_path = directory / f"{basename}.pvd"
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write("  <Collection>\n")
        for i, (t, vp) in enumerate(zip(time_steps, vtu_paths)):
            f.write(f'    <DataSet timestep="{t}" file="{vp.name}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

    return vtu_paths
