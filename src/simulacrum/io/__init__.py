"""
File I/O: reading and writing simulation results.

Primary format: VTK/VTU via ``meshio``.

Three writers, one per kind of data:

* :func:`write_vtk` / :func:`write_vtk_time_series` -- FEM meshes (``.vtu``)
* :func:`write_particles_vtu`                       -- particle clouds (``.vtu``)
* :func:`write_image_vti`                           -- uniform grids (``.vti``)

:class:`TimeSeriesWriter` sits above all three: it streams frames and maintains
the ``.pvd`` collection that gives ParaView a real time axis.

``uniform_grid`` is the one module that bypasses meshio, because meshio has no
ImageData writer; it depends on numpy alone.
"""

from simulacrum.io.particles import pad_to_3d, write_particles_vtu
from simulacrum.io.time_series import TimeSeriesWriter
from simulacrum.io.uniform_grid import flatten_for_vtk, write_image_vti
from simulacrum.io.vtk_writer import write_vtk, write_vtk_time_series

__all__ = [
    # meshes
    "write_vtk",
    "write_vtk_time_series",
    # particle clouds
    "write_particles_vtu",
    "pad_to_3d",
    # uniform grids
    "write_image_vti",
    "flatten_for_vtk",
    # time series / .pvd
    "TimeSeriesWriter",
]
