"""
Rust backend for MPM particle-to-grid assembly.

Import is deliberately NOT re-exported from the parent ``mpm`` package, for
the same reason as the Warp backend: ``import scion.discretization.mpm`` must
not require a compiled extension on every machine. Import it explicitly:

    from scion.discretization.mpm.rust import Scatter, scatter_add

Build with ``maturin build --release`` in ``crates/scion-mpm-py`` and install
the resulting wheel. Without it every function here falls back to the NumPy
implementation, so the library behaves exactly as it did before -- the
fallback is the reference, and the Rust path must reproduce it.

The kernel replaces ``np.add.at(out[:, col:col+k], node_id, vals)``, which is
unbuffered and single-threaded. The accumulation order is identical, so the
Rust result matches NumPy's to the bit rather than to a tolerance; see the
note in ``crates/scion-mpm/src/scatter.rs`` for why that was worth paying for.
"""
from __future__ import annotations

import numpy as np

try:
    import scion_rust as _rs
except ImportError:                                   # pragma: no cover
    _rs = None

__all__ = ["available", "backend", "Scatter", "scatter_add"]


def available() -> bool:
    """True when the compiled extension is importable."""
    return _rs is not None


def backend() -> str:
    return "rust" if _rs is not None else "numpy"


def _as_entries(vals: np.ndarray) -> np.ndarray:
    """Flatten ``(n_p, npe, k)`` or ``(n_p, npe)`` to ``(n_entries, k)``.

    ``ascontiguousarray`` is a no-op for the arrays the assembly builds, which
    are freshly allocated einsum results; it is here for the caller who slices
    one, where a silent copy is cheaper than a wrong answer.
    """
    v = np.ascontiguousarray(vals, dtype=np.float64)
    return v.reshape(-1, 1) if v.ndim == 2 else v.reshape(-1, v.shape[-1])


class Scatter:
    """Reusable particle-to-grid scatter for one stencil.

    Build once per stencil rebuild and call ``add`` for each field: the
    node-major transpose costs a pass over the stencil and the assembly
    scatters through it several times per Newton iteration.
    """

    def __init__(self, node_id: np.ndarray, n_nodes: int):
        self.node_id = np.ascontiguousarray(node_id, dtype=np.int64)
        self.n_nodes = int(n_nodes)
        self._inner = _rs.Scatter(self.node_id, self.n_nodes) if _rs else None

    def add(self, out: np.ndarray, vals: np.ndarray, col: int = 0) -> None:
        """Accumulate ``vals`` into ``out[:, col : col + k]``, in place."""
        if self._inner is None:
            k = 1 if vals.ndim == 2 else vals.shape[-1]
            np.add.at(out[:, col:col + k], self.node_id,
                      vals if vals.ndim == 3 else vals)
            return
        self._inner.add(out, _as_entries(vals), col)


def scatter_add(out: np.ndarray, node_id: np.ndarray, vals: np.ndarray,
                col: int = 0) -> None:
    """One-shot drop-in for ``np.add.at``; prefer ``Scatter`` when reusing."""
    Scatter(node_id, out.shape[0]).add(out, vals, col)
