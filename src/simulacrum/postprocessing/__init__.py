"""
Post-processing: visualisation and derived quantities.

Uses ``pyvista`` for interactive 3-D plots and VTK pipelines.
"""

from simulacrum.postprocessing.viewer import plot_field, plot_mesh

__all__ = ["plot_field", "plot_mesh"]
