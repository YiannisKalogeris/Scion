"""Simple structured mesh generators for quick prototyping."""

from __future__ import annotations

import numpy as np

from simulacrum.mesh.mesh import Mesh, ElementBlock


def line_mesh(x0: float, x1: float, n_elements: int) -> Mesh:
    """Generate a 1-D line mesh of ``line2`` elements."""
    nodes = np.linspace(x0, x1, n_elements + 1).reshape(-1, 1)
    conn = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    block = ElementBlock(etype="line2", connectivity=conn)
    boundary_tags = {
        "left": np.array([0]),
        "right": np.array([n_elements]),
    }
    return Mesh(nodes=nodes, element_blocks=[block], boundary_tags=boundary_tags)


def rectangle_mesh(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int,
    ny: int,
    etype: str = "quad4",
) -> Mesh:
    """Generate a 2-D structured quad or triangle mesh.

    Parameters
    ----------
    x_range, y_range : (min, max)
    nx, ny : number of elements in each direction
    etype : ``"quad4"`` or ``"tri3"``
    """
    x = np.linspace(*x_range, nx + 1)
    y = np.linspace(*y_range, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    def nid(i: int, j: int) -> int:
        return i * (ny + 1) + j

    elements = []
    for i in range(nx):
        for j in range(ny):
            n0, n1, n2, n3 = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if etype == "quad4":
                elements.append([n0, n1, n2, n3])
            elif etype == "tri3":
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
            else:
                raise ValueError(f"Unsupported etype '{etype}' for rectangle_mesh")

    conn = np.array(elements, dtype=np.int64)
    block = ElementBlock(etype=etype, connectivity=conn)

    # Boundary node sets
    left = np.array([nid(0, j) for j in range(ny + 1)])
    right = np.array([nid(nx, j) for j in range(ny + 1)])
    bottom = np.array([nid(i, 0) for i in range(nx + 1)])
    top = np.array([nid(i, ny) for i in range(nx + 1)])

    boundary_tags = {"left": left, "right": right, "bottom": bottom, "top": top}
    return Mesh(nodes=nodes, element_blocks=[block], boundary_tags=boundary_tags)
