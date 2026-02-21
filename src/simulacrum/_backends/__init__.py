"""
Native acceleration backends (Rust, C++).

This package provides a dispatch layer that routes performance-critical
operations to compiled backends when available, falling back to
pure-Python/NumPy/SciPy otherwise.

Rust Backend
------------
Built with ``maturin`` and exposed as ``simulacrum._backends._rust_core``.
Intended for: sparse matrix assembly, kernel evaluations, neighbour search.

C++ Backend
-----------
Built with ``pybind11`` and exposed as ``simulacrum._backends._cpp_core``.
Intended for: element-level stiffness routines, quadrature loops.
"""

from simulacrum._backends.dispatch import get_backend

__all__ = ["get_backend"]
