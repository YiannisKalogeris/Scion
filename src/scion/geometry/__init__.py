"""
Geometry subpackage: shape generation, signed distance fields, and harmonic maps.

Modules
-------
shapes          Parameterized closed curves (Ellipse, Superellipse, FourierBlob, etc.)
sdf             Signed distance field computation and rendering
harmonic_maps   Harmonic map solver for domain-to-square mappings
"""

from scion.geometry.shapes import (
    ClosedCurve,
    Ellipse,
    Superellipse,
    FourierBlob,
    StarPolygon,
    SmoothPolygon,
    ShapeGenerator,
    SHAPE_CLASSES,
    SHAPE_REGISTRY,
)
from scion.geometry.sdf import SDFRenderer, sample_points_in_shape, render_shape_to_sdf
from scion.geometry.harmonic_maps import HarmonicMapSolver, interpolate_harmonic_map

__all__ = [
    "ClosedCurve", 
    "Ellipse",
    "Superellipse",
    "FourierBlob",
    "StarPolygon",
    "SmoothPolygon",
    "ShapeGenerator",
    "SHAPE_CLASSES",
    "SHAPE_REGISTRY",
    "SDFRenderer",
    "sample_points_in_shape",
    "render_shape_to_sdf",
    "HarmonicMapSolver",
    "interpolate_harmonic_map",
]