"""
Uniform-grid (VTK ImageData, ``.vti``) output.

Why this is not routed through meshio like everything else in ``io/``: meshio
has no ImageData writer. It is a small enough format to emit directly, and the
alternative -- storing a uniform grid as an UnstructuredGrid -- is genuinely
expensive. ImageData records only origin, spacing and dimensions; a ``.vtu``
would store every node coordinate plus full connectivity. For a 128^3 MPM grid
that is a few hundred kB against a few hundred MB per frame.

Written as raw XML with an appended binary payload, so this module depends on
nothing but numpy. ParaView, PyVista, VTK and OVITO all read the result.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

__all__ = ["write_image_vti", "flatten_for_vtk"]

_NUMPY_TO_VTK = {
    "float32": "Float32",
    "float64": "Float64",
    "int32": "Int32",
    "int64": "Int64",
    "uint8": "UInt8",
    "uint32": "UInt32",
    "uint64": "UInt64",
}


class _Appended:
    """Collects arrays into one appended block of ``(uint64 n_bytes)(bytes)``."""

    def __init__(self):
        self._blocks: list[bytes] = []
        self._offset = 0

    def add(self, array: np.ndarray) -> int:
        arr = np.ascontiguousarray(array)
        if arr.dtype.byteorder == ">":
            arr = arr.astype(arr.dtype.newbyteorder("<"))
        payload = arr.tobytes()
        offset = self._offset
        self._blocks.append(struct.pack("<Q", len(payload)))
        self._blocks.append(payload)
        self._offset += 8 + len(payload)
        return offset

    def payload(self) -> bytes:
        return b"".join(self._blocks)


def flatten_for_vtk(array: np.ndarray) -> np.ndarray:
    """Reorder a C-ordered ``(nx, ny[, nz])`` array into VTK's x-fastest order.

    Scion grids store the last axis fastest; VTK expects the first. Getting
    this wrong produces a transposed field that still *looks* plausible, which
    is exactly the kind of error that survives visual inspection.
    """
    a = np.asarray(array)
    return np.transpose(a, axes=tuple(reversed(range(a.ndim)))).ravel()


def write_image_vti(
    path: str | Path,
    origin,
    spacing,
    dimensions,
    point_data: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write a uniform grid as VTK ImageData.

    Parameters
    ----------
    path       : output file (.vti)
    origin     : (dim,) coordinates of node (0, 0[, 0])
    spacing    : scalar or (dim,) cell size
    dimensions : (dim,) number of *nodes* per axis
    point_data : {name: array}; arrays must already be in VTK's x-fastest order
                 -- pass them through :func:`flatten_for_vtk` if they are not
    """
    dims = [int(d) for d in dimensions]
    dim = len(dims)
    origin = [float(o) for o in np.atleast_1d(origin)]
    spacing = ([float(spacing)] * dim if np.isscalar(spacing)
               else [float(s) for s in np.atleast_1d(spacing)])

    while len(dims) < 3:
        dims.append(1)
        origin.append(0.0)
        spacing.append(0.0)

    extent = f"0 {dims[0]-1} 0 {dims[1]-1} 0 {dims[2]-1}"
    n_expected = dims[0] * dims[1] * dims[2]

    app = _Appended()
    root = ET.Element(
        "VTKFile", type="ImageData", version="1.0",
        byte_order="LittleEndian", header_type="UInt64",
    )
    img = ET.SubElement(
        root, "ImageData", WholeExtent=extent,
        Origin=" ".join(f"{o:.17g}" for o in origin),
        Spacing=" ".join(f"{s:.17g}" for s in spacing),
    )
    piece = ET.SubElement(img, "Piece", Extent=extent)

    pd = ET.SubElement(piece, "PointData")
    for name, arr in (point_data or {}).items():
        arr = np.asarray(arr)
        ncomp = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
        if arr.size // ncomp != n_expected:
            raise ValueError(
                f"point_data['{name}'] has {arr.size // ncomp} tuples, "
                f"expected {n_expected} for dimensions {dims}"
            )
        vtk_type = _NUMPY_TO_VTK.get(arr.dtype.name)
        if vtk_type is None:
            arr, vtk_type = arr.astype(np.float64), "Float64"
        offset = app.add(arr.ravel())
        ET.SubElement(
            pd, "DataArray", type=vtk_type, Name=name,
            NumberOfComponents=str(ncomp), format="appended", offset=str(offset),
        )
    ET.SubElement(piece, "CellData")

    appended = ET.SubElement(root, "AppendedData", encoding="raw")
    # the leading underscore is part of the format: readers skip exactly one
    # character after the opening tag before the first block header
    appended.text = "_@@PAYLOAD@@"

    xml = ET.tostring(root, encoding="unicode")
    head, tail = xml.split("@@PAYLOAD@@")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b'<?xml version="1.0"?>\n')
        f.write(head.encode("ascii"))
        f.write(app.payload())
        f.write(tail.encode("ascii"))
    return path
