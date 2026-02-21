"""Mesh handling: generation, import/export, topology queries."""

from simulacrum.mesh.mesh import Mesh
from simulacrum.mesh.generators import line_mesh, rectangle_mesh
from simulacrum.mesh.gmsh_interface import mesh_from_gmsh

__all__ = ["Mesh", "line_mesh", "rectangle_mesh", "mesh_from_gmsh"]
