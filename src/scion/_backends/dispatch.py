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
        import scion._backends._rust_core  # noqa: F401
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
    """Detect the best available backend.

    Parameters
    ----------
    preferred : ``"rust"``, ``"cpp"``, ``"python"``, or ``None`` (auto-detect).
    """
    if preferred == "rust" or (preferred is None and _try_import_rust()):
        logger.info("Using Rust backend")
        return BackendType.RUST
    if preferred == "cpp" or (preferred is None and _try_import_cpp()):
        logger.info("Using C++ backend")
        return BackendType.CPP

    logger.info("Using pure-Python backend")
    return BackendType.PYTHON
