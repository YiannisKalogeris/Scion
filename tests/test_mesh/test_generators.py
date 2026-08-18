"""Tests for built-in mesh generators."""

import numpy as np
import pytest

from scion.mesh import line_mesh, rectangle_mesh


def test_line_mesh_nodes():
    m = line_mesh(0.0, 1.0, 10)
    assert m.n_nodes == 11
    assert m.n_elements == 10
    assert m.dim == 1


def test_rectangle_mesh_quad4():
    m = rectangle_mesh((0, 1), (0, 1), 4, 4, etype="quad4")
    assert m.n_nodes == 25
    assert m.n_elements == 16
    assert "left" in m.boundary_tags
    assert "right" in m.boundary_tags


def test_rectangle_mesh_tri3():
    m = rectangle_mesh((0, 1), (0, 1), 3, 3, etype="tri3")
    assert m.n_nodes == 16
    assert m.n_elements == 18  # 3*3 quads * 2 triangles each


def test_boundary_tags():
    m = rectangle_mesh((0, 2), (0, 1), 4, 2)
    assert len(m.boundary_nodes("bottom")) == 5
    assert len(m.boundary_nodes("top")) == 5
    assert len(m.boundary_nodes("left")) == 3
    assert len(m.boundary_nodes("right")) == 3
