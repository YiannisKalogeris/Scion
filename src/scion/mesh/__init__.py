"""Mesh handling: generation, import/export, topology queries."""

from scion.mesh.mesh import Mesh
from scion.mesh.generators import line_mesh, rectangle_mesh
from scion.mesh.gmsh_interface import mesh_from_gmsh

__all__ = ["Mesh", "line_mesh", "rectangle_mesh", "mesh_from_gmsh"]
