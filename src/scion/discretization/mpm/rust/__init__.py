"""
Rust backend for MPM assembly: the particle-to-grid scatter, and the tangent's
cached sparsity pattern.

Import is deliberately NOT re-exported from the parent ``mpm`` package, for
the same reason as the Warp backend: ``import scion.discretization.mpm`` must
not require a compiled extension on every machine. Import it explicitly:

    from scion.discretization.mpm.rust import Scatter, scatter_add

Build with ``maturin build --release`` in ``crates/scion-mpm-py`` and install
the resulting wheel. Without it every function here falls back to the NumPy
implementation, so the library behaves exactly as it did before -- the
fallback is the reference, and the Rust path must reproduce it.

``Scatter`` replaces ``np.add.at(out[:, col:col+k], node_id, vals)``, which is
unbuffered and single-threaded. ``BlockPattern`` replaces building the
tangent's triplets and sorting them into CSR on every Newton iteration, when
``node_id`` -- and so the pattern -- is fixed for the whole load step. Both
accumulate in the order NumPy would, so both match it to the bit rather than
to a tolerance; see the notes in ``crates/scion-mpm/src/scatter.rs`` and
``block_csr.rs`` for why that was worth paying for.
"""
from __future__ import annotations

import numpy as np

try:
    import scion_rust as _rs
except ImportError:                                   # pragma: no cover
    _rs = None

__all__ = ["available", "backend", "Scatter", "scatter_add", "coo_to_csr",
           "BlockPattern", "native_csr_enabled", "enable_native_csr"]


#: Opt-in, because unlike `Scatter` the native CSR path is not result-neutral:
#: it differs from SciPy by about an ulp and keeps a few entries SciPy prunes
#: as exact zeros (11,068 against 10,817 nonzeros on the small micropolar
#: block). Default off, so the published numerics are what runs unless asked.
_USE_NATIVE_CSR = False


def native_csr_enabled() -> bool:
    return _USE_NATIVE_CSR


def enable_native_csr(on: bool = True) -> bool:
    """Switch `coo_to_csr` onto the native path. Returns the old value.

    **This no longer reaches the micropolar tangent.** That assembly went to
    `BlockPattern`, which converts no COO at all, so the flag now affects only
    `coo_to_csr`'s own callers -- the two chunk helpers kept as references.
    Measured on a load step at ``a/dx = 6``: 9.57 s with it on against 9.64
    with it off, i.e. nothing.
    """
    global _USE_NATIVE_CSR
    was, _USE_NATIVE_CSR = _USE_NATIVE_CSR, bool(on)
    return was


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
            # a (n_p, npe) field needs its component axis back: out[:, c:c+1]
            # gathers to (n_p, npe, 1), which (n_p, npe) does not broadcast to
            v = vals[..., None] if vals.ndim == 2 else vals
            np.add.at(out[:, col:col + v.shape[-1]], self.node_id, v)
            return
        self._inner.add(out, _as_entries(vals), col)


def scatter_add(out: np.ndarray, node_id: np.ndarray, vals: np.ndarray,
                col: int = 0) -> None:
    """One-shot drop-in for ``np.add.at``; prefer ``Scatter`` when reusing."""
    Scatter(node_id, out.shape[0]).add(out, vals, col)


def coo_to_csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray,
               shape: tuple[int, int], native: bool | None = None):
    """``coo_matrix((vals, (rows, cols)), shape).tocsr()``, in parallel.

    **Not bit-identical to SciPy**, unlike `Scatter`: SciPy sorts each row's
    columns with an unstable sort, so the order in which it adds entries
    sharing a ``(row, col)`` is not reproducible. This sums in ascending
    original-triplet order, which is deterministic and independent of the
    thread count, and differs by about one ulp. The sparsity pattern is
    identical.
    """
    from scipy import sparse
    if native is None:
        native = _USE_NATIVE_CSR
    if not native:
        # the default: SciPy's own conversion, so the numbers are the ones
        # every published result was produced with
        return sparse.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    if _rs is None:
        # NOT scipy's tocsr(): that sorts each row with an unstable sort, so
        # it would disagree with the Rust path by an ulp and the backend
        # switch would stop being result-neutral. lexsort is stable, so this
        # reproduces the kernel's order -- ascending original index within a
        # (row, col) group -- exactly. Slower than tocsr; it runs only where
        # the extension is absent.
        order = np.lexsort((cols, rows))
        r, c, v = rows[order], cols[order], vals[order]
        first = np.empty(r.size, bool)
        first[0] = True
        first[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
        seg = np.cumsum(first) - 1
        data = np.zeros(seg[-1] + 1)
        np.add.at(data, seg, v)
        indices = c[first].astype(np.int32)
        counts = np.bincount(r[first], minlength=shape[0])
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    indptr, indices, data = _rs.coo_to_csr_py(
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(cols, dtype=np.int64),
        np.ascontiguousarray(vals, dtype=np.float64),
        int(shape[0]), int(shape[1]))
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


class BlockPattern:
    """Fixed CSR pattern for a block tangent on one particle stencil.

    ``node_id`` is rebuilt once per load step while the tangent is assembled
    once per Newton iteration, so the sparsity pattern is computed once here
    and the assembly becomes an accumulation into a fixed ``data`` array --
    no COO triplets, no per-chunk sort, no merge tree.

    **Bit-identical to a flat ascending COO sum**, and therefore to a
    *single-chunk* run of the path it replaces. It differs in the last bits
    from a multi-chunk run of that path, which summed each chunk separately
    and then added the partial sums in merge-tree order; the consequence is
    that the result no longer moves with ``chunk_entries``, which it used to.

    The NumPy branch is the reference the kernel is tested against, and is
    slower than the COO path it is a reference for -- it exists to pin the
    contract on a machine with no compiler, not to run a production assembly.
    """

    def __init__(self, node_id: np.ndarray, n_nodes: int, nc: int):
        self.node_id = np.ascontiguousarray(node_id, dtype=np.int64)
        if self.node_id.ndim != 2:
            raise ValueError("node_id must be (n_particles, npe)")
        self.n_particles, self.npe = self.node_id.shape
        self.n_nodes, self.nc = int(n_nodes), int(nc)
        self.n_dof = self.n_nodes * self.nc
        if _rs is not None:
            self._inner = _rs.BlockPattern(self.node_id, self.n_nodes, self.nc)
            self.nnz = self._inner.nnz
            self._indptr = self._indices = None
            return
        self._inner = None
        self._build_numpy()

    # -- pattern ----------------------------------------------------------
    def _build_numpy(self) -> None:
        npe, nc, n_nodes = self.npe, self.nc, self.n_nodes
        nid = self.node_id
        if nid.size and (nid.min() < 0 or nid.max() >= n_nodes):
            raise ValueError(f"node id outside 0..{n_nodes}")
        # every (node, neighbour) pair sharing a particle, deduplicated. The
        # key is sorted, so each node's neighbour list comes out ascending.
        nK = np.repeat(nid, npe, axis=1).ravel()
        nL = np.tile(nid, (1, npe)).ravel()
        self._uk = np.unique(nK * n_nodes + nL)
        owner = self._uk // n_nodes
        self._nbr = (self._uk % n_nodes).astype(np.int64)
        counts = np.bincount(owner, minlength=n_nodes)
        self._nbr_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._row_len = counts * nc
        # every row of a node holds the same columns, so its length repeats nc
        self._indptr = np.concatenate(
            ([0], np.cumsum(np.repeat(self._row_len, nc)))).astype(np.int64)
        self.nnz = int(self._indptr[-1])
        blk = (self._nbr[:, None] * nc + np.arange(nc)[None, :]).ravel()
        starts = self._nbr_ptr[:-1] * nc
        lens = np.diff(self._indptr)
        rowno = np.repeat(np.arange(self.n_dof), lens)
        off = np.arange(self.nnz) - np.repeat(self._indptr[:-1], lens)
        self._indices = blk[starts[rowno // nc] + off].astype(np.int32)

    @property
    def indptr(self) -> np.ndarray:
        if self._indptr is None:
            self._indptr = self._inner.indptr()
        return self._indptr

    @property
    def indices(self) -> np.ndarray:
        if self._indices is None:
            self._indices = self._inner.indices()
        return self._indices

    # -- assembly ---------------------------------------------------------
    def accumulate(self, data: np.ndarray, blocks: np.ndarray,
                   lo: int, hi: int) -> None:
        """Add ``blocks`` for particles ``lo:hi`` into ``data``, in place.

        ``blocks`` is ``(hi - lo, npe, nc, npe, nc)``: particle, row slot, row
        component, column slot, column component.
        """
        b = np.ascontiguousarray(blocks, dtype=np.float64)
        if self._inner is not None:
            self._inner.accumulate(data, b.reshape(-1), int(lo), int(hi))
            return
        npe, nc = self.npe, self.nc
        if hi == lo:
            return
        nid = self.node_id[lo:hi]
        nK = np.repeat(nid, npe, axis=1)
        nL = np.tile(nid, (1, npe))
        # the pair key is in the deduplicated list by construction, so this
        # search locates the block rather than bracketing it
        g = np.searchsorted(self._uk, nK * self.n_nodes + nL)
        col = (g - self._nbr_ptr[nK]) * nc
        base = self.indptr[nK * nc]
        width = self._row_len[nK]
        i = np.arange(nc)[None, None, :, None]
        dest = (base[..., None, None] + i * width[..., None, None]
                + col[..., None, None] + np.arange(nc)[None, None, None, :])
        # (p, K, L, i, j) order, so a repeated destination is summed with p,
        # then K, then L ascending -- the kernel's order, and COO's
        vals = b.reshape(hi - lo, npe, nc, npe, nc).transpose(0, 1, 3, 2, 4)
        np.add.at(data, dest.ravel(), np.ascontiguousarray(vals).ravel())

    def matrix(self, data: np.ndarray):
        """Wrap ``data`` as a CSR matrix, pruning the structural zeros.

        The pattern arrays are copied because ``eliminate_zeros`` compacts
        them in place, and they belong to this reusable pattern.
        """
        from scipy import sparse
        K = sparse.csr_matrix(
            (data, self.indices.copy(), self.indptr.copy()),
            shape=(self.n_dof, self.n_dof))
        K.eliminate_zeros()
        return K
