"""
Time-series output and the ParaView ``.pvd`` collection.

Supersedes the frame-list approach of ``write_vtk_time_series``: that signature
takes ``fields: list[dict]``, so every frame of the run has to be held in memory
until the end. For a transient FEM job that is merely wasteful; for MPM, where
you might write 2000 frames of 10^6 particles, it is impossible.

This writer streams instead -- one frame written and forgotten at a time -- and
rewrites the collection after each, so an interrupted or crashed run still
leaves a loadable series. It is writer-agnostic: any callable with the signature
``writer(path, *args, **kwargs)`` works, so the same class serves meshes,
particles and grids.

The ``.pvd`` file is what gives ParaView a real time axis instead of frame
indices, which matters as soon as the step size is adaptive.

Example
-------
>>> from simulacrum.io.particles import write_particles_vtu
>>> series = TimeSeriesWriter("results/drop", "particles", suffix=".vtu")
>>> series.add(t, write_particles_vtu, particles.x, point_data=fields)
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = ["TimeSeriesWriter"]


class TimeSeriesWriter:
    """Numbered frame files plus their ``.pvd`` collection.

    Parameters
    ----------
    directory : output directory, created if absent
    name      : basename; frames are ``<name>_000000<suffix>``
    suffix    : frame file extension (``.vtu``, ``.vti``, ...)
    """

    def __init__(self, directory: str | Path, name: str, suffix: str = ".vtu"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.suffix = suffix
        self.entries: list[tuple[float, str]] = []

    # ------------------------------------------------------------------
    def frame_path(self, index: int) -> Path:
        return self.directory / f"{self.name}_{index:06d}{self.suffix}"

    def add(self, time: float, writer, *args, **kwargs) -> Path:
        """Write one frame via ``writer(path, *args, **kwargs)``."""
        path = self.frame_path(len(self.entries))
        writer(path, *args, **kwargs)
        self.entries.append((float(time), path.name))
        self.write_collection()
        return path

    # ------------------------------------------------------------------
    def write_collection(self) -> Path:
        """(Re)write the ``.pvd``. Called automatically after every frame."""
        root = ET.Element("VTKFile", type="Collection", version="1.0",
                          byte_order="LittleEndian")
        coll = ET.SubElement(root, "Collection")
        for t, fname in self.entries:
            ET.SubElement(coll, "DataSet", timestep=f"{t:.17g}",
                          group="", part="0", file=fname)
        path = self.directory / f"{self.name}.pvd"
        with open(path, "wb") as f:
            f.write(b'<?xml version="1.0"?>\n')
            f.write(ET.tostring(root))
        return path

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return (f"TimeSeriesWriter({self.name!r}, {len(self.entries)} frames, "
                f"dir={os.fspath(self.directory)!r})")
