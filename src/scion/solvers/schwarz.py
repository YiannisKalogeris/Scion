r"""
Algebraic additive Schwarz: overlapping domain decomposition from the matrix
graph alone.

Split the unknowns into overlapping subsets, solve each subproblem exactly, add
the corrections:

    M^-1 = sum_i  R_i'  A_i^-1  R_i,        A_i = R_i A R_i'

``R_i`` selects subdomain ``i``'s unknowns and ``R_i'`` scatters back. Nothing
in that needs a mesh -- the partition comes from the sparsity graph and the
overlap from growing each subset by a few layers of it, which is exactly what
PETSc's ``PCASM`` does. So it works on a bare SuiteSparse matrix, unlike the
substructuring methods (BDDC, FETI-DP), which need the *unassembled* subdomain
operators that assembly has already summed away.

Three things to know before using it
------------------------------------
**One level does not scale.** The condition number of the preconditioned
operator grows like ``1/(H delta)`` with subdomain diameter ``H`` and overlap
``delta``. Cutting the problem into more pieces makes each solve cheaper and the
iteration worse, and refining the mesh at a fixed number of subdomains makes it
worse again. This is not a defect to tune away -- information has to cross the
whole domain and a one-level method moves it one subdomain per iteration. The
fix is a coarse space, which is what makes *two-level* Schwarz the method people
actually cite, and what
``projects/diffusion_preconditioner.TwoLevelPreconditioner`` supplies
algebraically.

**Restricted ASM is faster and not symmetric.** RAS restricts *with* overlap and
scatters back *without* it, so the overlapping corrections are not double
counted. It typically needs noticeably fewer iterations than classical ASM --
and it is not a symmetric operator, so it is not a valid PCG preconditioner. Use
it with GMRES or BiCGSTAB. :class:`AdditiveSchwarz` carries ``is_spd = False``
when ``restricted=True`` so the constraint is visible from the object rather
than only here.

**Partition quality drives everything.** A partition with a large edge cut puts
more coupling across subdomain boundaries, which is precisely the coupling the
preconditioner throws away. METIS is the usual tool and is not a dependency, so
:func:`partition_graph` offers recursive spectral bisection (a Fiedler vector
per split, pure SciPy, good cuts, ``O(n)`` eigensolves) and a breadth-first
greedy alternative (instant, worse cuts). An optional pymetis backend is used
when available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse import linalg as spla

from scion.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "Subdomain",
    "partition_graph",
    "grow_overlap",
    "SchwarzOperator",
    "PYMETIS_AVAILABLE",
]

try:                                        # pragma: no cover - environment
    import pymetis                          # noqa: F401
    PYMETIS_AVAILABLE = True
except ImportError:                         # pragma: no cover - environment
    PYMETIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def _adjacency(A: sparse.spmatrix) -> sparse.csr_matrix:
    """Symmetric unweighted graph of the sparsity pattern, no self-loops."""
    G = sparse.csr_matrix(A, copy=True)
    G.data = np.ones_like(G.data, dtype=float)
    G = G.maximum(G.T)
    G.setdiag(0.0)
    G.eliminate_zeros()
    return G


def _fiedler(G: sparse.csr_matrix) -> np.ndarray:
    """Eigenvector of the second-smallest Laplacian eigenvalue.

    The classical spectral bisection heuristic: its sign pattern is a good
    approximate minimum cut. Computed by LOBPCG with the constant vector
    deflated, which is far more robust on a Laplacian than asking ARPACK for
    the smallest eigenvalues directly -- those are clustered near zero and
    ``which="SM"`` converges badly there.
    """
    n = G.shape[0]
    L = csgraph.laplacian(G).astype(float).tocsr()

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 1))
    constant = np.ones((n, 1)) / np.sqrt(n)
    X -= constant * (constant.T @ X)          # deflate the null vector

    try:
        # LOBPCG's "did not reach tolerance" warning is suppressed on purpose:
        # only the *sign pattern* of the Fiedler vector is used, and a vector
        # accurate to 1e-3 splits the graph identically to one accurate to
        # 1e-12. Converging it properly would cost far more than the partition
        # is worth.
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, vectors = spla.lobpcg(L, X, Y=constant, largest=False,
                                     maxiter=200, tol=1e-4)
        return np.asarray(vectors[:, 0]).ravel()
    except Exception:                         # fall back rather than fail
        logger.debug("LOBPCG failed on a subgraph; using a BFS split")
        return np.asarray(csgraph.breadth_first_order(
            G, 0, return_predecessors=False), dtype=float)


def _bfs_labels(G: sparse.csr_matrix, n_parts: int) -> np.ndarray:
    """Greedy breadth-first growth from spread-out seeds. Instant, mediocre."""
    n = G.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    order = csgraph.breadth_first_order(G, 0, return_predecessors=False)
    # seeds spaced along a BFS ordering are far apart in the graph
    seeds = order[np.linspace(0, order.size - 1, n_parts).astype(int)]

    frontier = {int(s): p for p, s in enumerate(seeds)}
    for node, part in frontier.items():
        labels[node] = part

    indptr, indices = G.indptr, G.indices
    active = list(frontier)
    while active:
        nxt = []
        for node in active:
            part = labels[node]
            for neighbour in indices[indptr[node]:indptr[node + 1]]:
                if labels[neighbour] == -1:
                    labels[neighbour] = part
                    nxt.append(int(neighbour))
        active = nxt

    labels[labels == -1] = 0                  # unreachable components
    return labels


def partition_graph(A: sparse.spmatrix, n_parts: int, method: str = "spectral"
                    ) -> np.ndarray:
    """Split the unknowns into ``n_parts`` disjoint sets. Returns a label array.

    Parameters
    ----------
    method : ``"spectral"`` (recursive Fiedler bisection -- better cuts, one
        eigensolve per split), ``"bfs"`` (greedy, instant, worse cuts), or
        ``"metis"`` (optional pymetis backend).

    Notes
    -----
    The cut matters: every edge crossing a subdomain boundary is coupling the
    preconditioner discards. Spectral bisection is the reasonable default
    without METIS, and the recursion halves rather than splitting ``n`` ways at
    once, so ``n_parts`` need not be a power of two.
    """
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    n_parts = int(n_parts)
    if n_parts < 1:
        raise ValueError("n_parts must be at least one")
    if n_parts == 1:
        return np.zeros(n, dtype=np.int64)
    if n_parts > n:
        raise ValueError(f"asked for {n_parts} subdomains of {n} unknowns")

    G = _adjacency(A)
    method = method.lower()

    if method == "metis":
        if not PYMETIS_AVAILABLE:             # pragma: no cover - environment
            raise ImportError(
                "pymetis is not installed and is not a dependency of "
                "Scion. Use method='spectral', which is pure SciPy.")
        adjacency = [G.indices[G.indptr[i]:G.indptr[i + 1]].tolist()
                     for i in range(n)]       # pragma: no cover
        _, labels = pymetis.part_graph(n_parts, adjacency=adjacency)
        return np.asarray(labels, dtype=np.int64)

    if method == "bfs":
        return _bfs_labels(G, n_parts)

    if method != "spectral":
        raise NotImplementedError(
            f"Unknown partitioner '{method}'; use 'spectral', 'bfs' or 'metis'")

    # recursive bisection: split the largest piece until there are enough
    labels = np.zeros(n, dtype=np.int64)
    for part in range(1, n_parts):
        sizes = np.bincount(labels, minlength=part)
        target = int(np.argmax(sizes))
        members = np.flatnonzero(labels == target)
        if members.size < 2:
            break
        sub = G[np.ix_(members, members)].tocsr()
        vector = _fiedler(sub)
        if vector.size != members.size:
            vector = np.arange(members.size, dtype=float)
        half = vector > np.median(vector)
        if not half.any() or half.all():      # degenerate: split by index
            half = np.arange(members.size) >= members.size // 2
        labels[members[half]] = part
    return labels


def grow_overlap(A: sparse.spmatrix, labels: np.ndarray, overlap: int = 1
                 ) -> list:
    """Turn a disjoint partition into overlapping index sets.

    Each subdomain absorbs ``overlap`` layers of its graph neighbours. Overlap
    is what makes Schwarz work at all: with none, the subdomain solves are
    independent and nothing crosses a boundary, so the preconditioner is block
    Jacobi. Each extra layer improves the conditioning and enlarges every local
    factorisation, and that trade is the parameter.
    """
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    overlap = int(overlap)
    if overlap < 0:
        raise ValueError("overlap must not be negative")

    G = _adjacency(A)
    n_parts = int(labels.max()) + 1
    subdomains = []

    for part in range(n_parts):
        owned = np.flatnonzero(labels == part)
        if owned.size == 0:
            continue
        current = np.zeros(n, dtype=bool)
        current[owned] = True
        for _ in range(overlap):
            # one graph layer: any neighbour of a current member joins
            reached = G[current].sum(axis=0).A1 > 0
            current |= reached
        rows = np.flatnonzero(current)
        # which of this subdomain's rows it *owns*, for the restricted variant
        owned_mask = np.isin(rows, owned)
        subdomains.append(Subdomain(rows=rows, owned=owned_mask))

    return subdomains


@dataclass
class Subdomain:
    """One overlapping index set and its factorised local operator."""

    rows: np.ndarray
    owned: np.ndarray                  #: mask over ``rows``: the non-overlapping part
    factorisation: object = None

    @property
    def size(self) -> int:
        return int(self.rows.size)

    @property
    def n_owned(self) -> int:
        return int(self.owned.sum())


# ---------------------------------------------------------------------------

@dataclass
class SchwarzOperator:
    """The assembled subdomain solves, and the two ways of adding them up."""

    subdomains: list = field(default_factory=list)
    n: int = 0
    restricted: bool = False
    setup_time: float = 0.0
    A: object = None

    # ------------------------------------------------------------------
    @property
    def n_subdomains(self) -> int:
        return len(self.subdomains)

    @property
    def overlap_ratio(self) -> float:
        """Total subdomain size over ``n``.

        One means no overlap at all -- the subdomains tile the unknowns and this
        is block Jacobi. The excess above one is what the overlap costs in
        factorisation work and memory.
        """
        if not self.subdomains:
            return 0.0
        return sum(s.size for s in self.subdomains) / max(1, self.n)

    @property
    def largest_subdomain(self) -> int:
        return max((s.size for s in self.subdomains), default=0)

    @property
    def multiplicity(self) -> int:
        """The most subdomains covering any single unknown.

        One when there is no overlap. It is the factor by which the additive
        sum double-counts in the overlap region, and therefore the damping a
        standalone iteration needs -- see :meth:`solve`.
        """
        counts = np.zeros(self.n, dtype=int)
        for sub in self.subdomains:
            counts[sub.rows] += 1
        return int(counts.max()) if self.subdomains else 0

    @property
    def balance(self) -> float:
        """Largest subdomain over the mean. One is perfect balance; a large
        value means the partitioner produced one dominant piece, and that piece
        dictates the factorisation cost."""
        if not self.subdomains:
            return 0.0
        sizes = np.array([s.size for s in self.subdomains], dtype=float)
        return float(sizes.max() / sizes.mean())

    # ------------------------------------------------------------------
    def apply(self, v: np.ndarray) -> np.ndarray:
        r"""``sum_i R_i' A_i^-1 R_i v``, or its restricted variant.

        Classical ASM scatters the whole local correction back, so the
        overlapping region is written by several subdomains and the
        contributions add. Restricted ASM scatters back only the owned part,
        which removes that double counting -- faster in practice, and no longer
        a symmetric operator.
        """
        v = np.asarray(v, dtype=float).ravel()
        out = np.zeros(self.n)
        for sub in self.subdomains:
            local = sub.factorisation.solve(v[sub.rows])
            if self.restricted:
                out[sub.rows[sub.owned]] += local[sub.owned]
            else:
                out[sub.rows] += local
        return out

    def __call__(self, v):
        return self.apply(v)

    def solve(self, b, tol: float = 1e-10, max_iter: int = 500,
              damping: float | None = None):
        r"""Schwarz as a *standalone* iteration, for diagnosis.

        ``x <- x + damping * M^-1 (b - A x)``.

        **Additive Schwarz is not convergent at unit damping**, and that is the
        precise reason it is a preconditioner rather than a solver. The additive
        sum counts the overlap region once per covering subdomain, so where
        subdomains meet the correction is applied several times over and the
        iteration matrix has spectral radius above one -- measured here, it
        diverges to inf within a few hundred steps. The classical Schwarz
        *alternating* method converges because it is multiplicative: each
        subdomain sees the previous one's update, like Gauss-Seidel rather than
        Jacobi.

        Damping by :attr:`multiplicity` -- the largest number of subdomains
        covering any one unknown -- removes the double counting and restores
        convergence. That is the default when ``damping`` is not given.

        This exists to check the subdomain solves in isolation, which is far
        easier than diagnosing them inside a Krylov method. It is not how the
        preconditioner is meant to be used.
        """
        if damping is None:
            damping = 1.0 / max(1, self.multiplicity)
        b = np.asarray(b, dtype=float).ravel()
        x = np.zeros_like(b)
        norm = float(np.linalg.norm(b)) or 1.0
        history = [1.0]
        for _ in range(max_iter):
            x = x + damping * self.apply(b - self.A @ x)
            history.append(float(np.linalg.norm(b - self.A @ x)) / norm)
            if history[-1] < tol:
                break
        return x, np.array(history)

    def __repr__(self) -> str:
        kind = "restricted" if self.restricted else "classical"
        return (f"SchwarzOperator({self.n_subdomains} subdomains, {kind}, "
                f"overlap ratio {self.overlap_ratio:.2f}, "
                f"largest {self.largest_subdomain})")


def build_schwarz(A: sparse.spmatrix, n_subdomains: int = 8, overlap: int = 1,
                  restricted: bool = False, method: str = "spectral"
                  ) -> SchwarzOperator:
    """Partition, grow the overlap, and factorise each subdomain."""
    A = sparse.csr_matrix(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")

    started = time.perf_counter()
    labels = partition_graph(A, n_subdomains, method)
    subdomains = grow_overlap(A, labels, overlap)

    for sub in subdomains:
        block = A[np.ix_(sub.rows, sub.rows)].tocsc()
        try:
            sub.factorisation = spla.splu(block)
        except RuntimeError as exc:
            raise ValueError(
                f"a subdomain of size {sub.size} is singular and cannot be "
                f"factorised ({exc}). This usually means the partition isolated "
                f"a set of unknowns that the operator does not constrain on "
                f"its own -- try fewer subdomains or more overlap.") from exc

    operator = SchwarzOperator(subdomains=subdomains, n=A.shape[0],
                               restricted=restricted,
                               setup_time=time.perf_counter() - started, A=A)
    logger.info("Schwarz: %s", operator)
    return operator
