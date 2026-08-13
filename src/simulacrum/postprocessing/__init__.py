"""
Post-processing: visualisation and derived quantities.
 
Uses ``pyvista`` for interactive 3-D plots and VTK pipelines.
 
Two families, split by what they draw:
 
* :func:`plot_mesh` / :func:`plot_field` -- FEM meshes and nodal fields
* :func:`plot_particles` / :func:`save_particle_frames` -- particle clouds
 
They are separate because a particle cloud is a ``pv.PolyData`` of bare points
with no cells, so none of the mesh-conversion machinery applies to it.
 
These are for the development loop and for scripted figures. For interactive
exploration of a full run, write the series with ``simulacrum.io`` and open the
``.pvd`` in ParaView or OVITO instead.
 
``pyvista`` is imported lazily inside each function, so importing this package
does not pull in VTK.
"""
 
from simulacrum.postprocessing.particle_viewer import (
    plot_particles,
    save_particle_frames,
    to_pyvista_points,
)
from simulacrum.postprocessing.viewer import plot_field, plot_mesh
 
__all__ = [
    # meshes and nodal fields
    "plot_mesh",
    "plot_field",
    # particle clouds
    "plot_particles",
    "save_particle_frames",
    "to_pyvista_points",
]
