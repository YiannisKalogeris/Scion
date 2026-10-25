"""
Native acceleration backends (Rust, C++).

This package provides a dispatch layer that routes performance-critical
operations to compiled backends when available, falling back to
pure-Python/NumPy/SciPy otherwise.

Rust Backend
------------
The kernels live in ``crates/`` at the repo root, outside the Python package
so the core crate can be unit tested with ``cargo test`` and no interpreter in
the loop. Built with ``maturin`` and exposed as the top-level ``scion_rust``.
Currently: the MPM particle-to-grid scatter, wrapped for use in
``scion.discretization.mpm.rust``.

C++ Backend
-----------
Not implemented. ``cpp/`` holds a pybind11 stub and ``rust/`` a dormant
element-stiffness demo, neither of which is built or called by anything.
"""

from scion._backends.dispatch import get_backend

__all__ = ["get_backend"]
