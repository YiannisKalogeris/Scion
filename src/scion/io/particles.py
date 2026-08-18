"""
Particle-cloud output.

Separate from ``vtk_writer.py`` because that module writes a :class:`Mesh`, and
a particle cloud has no mesh: no connectivity, no element blocks, and a node
count that can change between frames. Same meshio backend, so no new dependency.

Format note: particles are written as ``.vtu`` with ``vertex`` cells rather than
``.vtp`` (PolyData). ``.vtp`` is arguably the more natural VTK type for a point
cloud and ParaView prefers it, but meshio does not read it -- and neither, as a
result, does the Blender Sequence Loader addon, which bundles meshio. The extra
cost of ``.vtu`` is one index array per frame.

Field naming conventions that pay off downstream:

* keep vectors as ``(n, dim)`` arrays -- they are padded to three components here
* write tensors as separate scalar components (``sigma_xx``, ``sigma_xy``, ...)
  rather than one 9-component array; meshio silently drops arrays with more than
  three components, so a packed tensor survives ParaView but vanishes in Blender
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["write_particles_vtu", "pad_to_3d"]


def pad_to_3d(a: np.ndarray) -> np.ndarray:
    """Pad an ``(n, 2)`` array to ``(n, 3)``. VTK vectors are always 3-D."""
    a = np.asarray(a)
    if a.ndim != 2 or a.shape[1] == 3:
        return a
    out = np.zeros((a.shape[0], 3), dtype=a.dtype)
    out[:, : a.shape[1]] = a
    return out


def write_particles_vtu(
    path: str | Path,
    positions: np.ndarray,
    point_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a particle cloud as an UnstructuredGrid of vertex cells.

    Parameters
    ----------
    path       : output file (.vtu)
    positions  : (n, 2) or (n, 3) particle positions
    point_data : {name: (n,) or (n, dim) array}

    Returns
    -------
    The path written, so this composes with ``TimeSeriesWriter.add``.
    """
    import meshio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = pad_to_3d(np.asarray(positions, dtype=np.float64))
    n = pts.shape[0]

    cells = [meshio.CellBlock("vertex", np.arange(n, dtype=np.int64).reshape(-1, 1))]

    pdata: dict[str, np.ndarray] = {}
    for name, arr in (point_data or {}).items():
        arr = np.asarray(arr)
        if arr.shape[0] != n:
            raise ValueError(
                f"point_data['{name}'] has {arr.shape[0]} rows, expected {n}"
            )
        if arr.ndim == 2 and arr.shape[1] == 2:
            arr = pad_to_3d(arr)
        if arr.ndim == 2 and arr.shape[1] > 3:
            raise ValueError(
                f"point_data['{name}'] has {arr.shape[1]} components; meshio "
                "drops arrays with more than 3. Split it into scalar components."
            )
        pdata[name] = arr

    meshio.Mesh(points=pts, cells=cells, point_data=pdata).write(str(path))
    return path
