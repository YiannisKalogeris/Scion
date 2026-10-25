"""
Backend dispatch: try compiled backends, fall back to Python.

Usage
-----
>>> backend = get_backend()
>>> backend.assemble_element_stiffness(...)   # routes to Rust/C++/Python
"""

from __future__ import annotations

from enum import Enum, auto

from scion.core.logging import get_logger

logger = get_logger(__name__)


class BackendType(Enum):
    PYTHON = auto()
    RUST = auto()
    CPP = auto()


def _try_import_rust() -> bool:
    try:
        import scion_rust  # noqa: F401
        return True
    except ImportError:
        return False


def _try_import_cpp() -> bool:
    try:
        import scion._backends._cpp_core  # noqa: F401
        return True
    except ImportError:
        return False


def get_backend(preferred: str | None = None) -> BackendType:
    """Detect the best available backend, or validate a requested one.

    Parameters
    ----------
    preferred : ``"rust"``, ``"cpp"``, ``"python"``, or ``None`` (auto-detect).

    Raises
    ------
    ImportError
        If a compiled backend is named explicitly and is not importable. The
        earlier version returned the requested enum without checking, so a
        caller on a machine with no extension got a backend that failed later
        at the call site rather than here, where the cause is still visible.
    """
    probes = {"rust": _try_import_rust, "cpp": _try_import_cpp}

    if preferred is not None:
        if preferred == "python":
            return BackendType.PYTHON
        if preferred not in probes:
            raise ValueError(
                f"unknown backend {preferred!r}; expected 'rust', 'cpp' or 'python'")
        if not probes[preferred]():
            raise ImportError(
                f"the {preferred} backend was requested but is not built; "
                f"build it from crates/ (see README) or pass preferred=None "
                f"to fall back to Python")
        logger.info("Using %s backend (requested)", preferred)
        return BackendType.RUST if preferred == "rust" else BackendType.CPP

    for name, probe in probes.items():
        if probe():
            logger.info("Using %s backend", name)
            return BackendType.RUST if name == "rust" else BackendType.CPP

    logger.info("Using pure-Python backend")
    return BackendType.PYTHON
