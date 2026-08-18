r"""
Algebraic multigrid by smoothed aggregation.

Multigrid works because a smoother kills oscillatory error cheaply and is
useless on smooth error, while a coarse grid represents smooth error cheaply and
cannot see oscillatory error. Alternating the two attacks the whole spectrum, and
for an elliptic operator the result is a solver whose iteration count does not
grow with the mesh -- which is the property nothing else in this module has, and
the one the tests check.

*Algebraic* multigrid builds the hierarchy from the matrix alone, with no
geometry. That matters here because the systems this is aimed at arrive as bare
sparse matrices from SuiteSparse or from an assembler, with no grid to coarsen.

The construction, and why each step is there
--------------------------------------------
**Strength of connection.** Not every nonzero represents a coupling worth
preserving. For an SPD matrix the standard measure is

    |A_ij| > theta * sqrt(A_ii * A_jj),

which is scale-invariant -- it compares a coupling against the two diagonals it
connects rather than against an absolute size. Using ``|A_ij| > theta * max|A_ij|``
instead breaks on any matrix whose rows differ in magnitude, which for an
elasticity operator with mixed units is all of them.

**Aggregation.** Greedy: pick a node whose whole neighbourhood is unaggregated,
make it and its strong neighbours an aggregate, repeat; then sweep the leftovers
into an adjacent aggregate. This is Vanek's standard two-pass scheme. The second
pass is not tidying -- without it, isolated nodes get their own singleton
aggregates, the coarse operator picks up near-zero rows, and the hierarchy
degrades.

**Tentative prolongator.** One column per aggregate, carrying the near-null
space. For a scalar elliptic operator the near-null space is the constant
vector, so ``P_tent`` is a normalised indicator matrix. For elasticity it is the
rigid-body modes, and getting it wrong is one reason AMG stalls on elasticity
while working on Poisson -- pass ``near_null`` to supply them.

**Systems of PDEs need `block_size`.** With one unknown per node the default is
right. With two or three, aggregation on the DOF graph splits a node's DOFs
across different aggregates -- measured, 30 of 91 nodes on a plane-strain
cantilever -- because the strength measure compares individual matrix entries
and cannot see that they belong together. Setting ``block_size`` switches to
:func:`nodal_strength`, which collapses each ``d x d`` sub-block to its
Frobenius norm and aggregates *nodes*.

The other half is the near-null space, and it has to survive coarsening.
:func:`_tentative_prolongator` returns the ``R`` of its local QR as the coarse
modes, satisfying ``P_tent B_coarse = B`` exactly; letting the coarse levels
fall back to a constant vector throws the rigid-body modes away after level one
and looks like a partial improvement rather than a bug.

PCG iterations to 1e-8 on a plane-strain cantilever, both fixes in place:

    dof     scalar   +rot   nodal   nodal+rot   IC(0)   plain CG
    168         54     27      41          26      27        136
    624         89     43      62          33      50        289
    1680       137     50      62          34      80        483
    3720       201     58      70          33     120        718
    6560       263     66      66          34     158       1006

Read the last two useful columns against each other. ``nodal+rot`` is
**26, 33, 34, 33, 34** over a 39-fold growth in unknowns -- mesh-independent,
which is the whole point of multigrid and what the scalar path (54 to 263) had
lost. IC(0) grows steadily throughout. Nodal aggregation alone helps and is not
enough; the rotations need geometry and must be passed as ``near_null``.

**Prolongator smoothing.** ``P = (I - omega D^-1 A) P_tent``. This is what makes
smoothed aggregation better than plain aggregation: the tentative columns are
discontinuous across aggregate boundaries, the smoothing spreads them, and the
resulting coarse space represents smooth functions far more accurately. The cost
is a denser ``P`` and therefore a denser coarse operator.

**Galerkin coarsening.** ``A_c = P' A P``, which preserves symmetry and positive
definiteness. Using a non-Galerkin restriction on an SPD problem gives a
non-symmetric coarse operator and a preconditioner PCG cannot use.

The V-cycle is symmetric by construction -- same number of pre- and
post-smoothing sweeps, ``R = P'`` -- because an unsymmetric cycle is not a valid
PCG preconditioner however well it works as a standalone iteration.

Optional pyamg backend
----------------------
:func:`smoothed_aggregation` is pure numpy/scipy and is the default. If
``pyamg`` is installed, :class:`~.preconditioners.AlgebraicMultigrid` can
dispatch to it with ``backend="pyamg"``, which is what the published results in
the accompanying paper used. The two are not expected to produce identical
iteration counts -- coarsening heuristics differ in detail -- and the comparison
is reported rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from scion.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "AMGLevel",
    "AMGHierarchy",
    "smoothed_aggregation",
    "strength_of_connection",
    "nodal_strength",
    "rigid_body_translations",
    "aggregate",
]


def strength_of_connection(A: sparse.spmatrix, theta: float = 0.08
                           ) -> sparse.csr_matrix:
    r"""Symmetric strength graph: ``|A_ij| > theta sqrt(A_ii A_jj)``.

    Scale-invariant by construction, which is what makes it usable on an
    operator whose rows differ by orders of magnitude. The diagonal is dropped
    -- a node is not its own neighbour -- and the result is symmetrised, since
    aggregation needs an undirected graph.
    """
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    diagonal = np.abs(A.diagonal())
    # a zero diagonal would make every connection from that row "strong";
    # floor it instead, which isolates the row rather than corrupting the graph
    diagonal[diagonal <= 0.0] = 1.0
    scale = np.sqrt(diagonal)

    coo = A.tocoo()
    off = coo.row != coo.col
    rows, cols, vals = coo.row[off], coo.col[off], np.abs(coo.data[off])
    strong = vals > theta * scale[rows] * scale[cols]

    graph = sparse.coo_matrix(
        (np.ones(int(strong.sum())), (rows[strong], cols[strong])),
        shape=(n, n)).tocsr()
    graph = graph.maximum(graph.T)          # undirected
    graph.eliminate_zeros()
    return graph


def nodal_strength(A: sparse.spmatrix, block_size: int,
                   theta: float = 0.08) -> sparse.csr_matrix:
    r"""Strength graph over *nodes* rather than DOFs, for a system of PDEs.

    Each ``block_size x block_size`` sub-block is collapsed to its Frobenius
    norm, and the same scale-invariant criterion is then applied to the node
    graph. The point is that a node's DOFs are one entity: an elasticity node
    couples to its neighbour through a 2x2 or 3x3 block, and asking whether an
    individual entry of that block is "strong" is asking the wrong question.

    Measured consequence of getting this wrong: with DOF-level aggregation on a
    plane-strain cantilever, 30 of 91 nodes had their x- and y-DOF assigned to
    *different* aggregates, and AMG lost mesh independence entirely.
    """
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    if block_size < 1 or n % block_size:
        raise ValueError(
            f"block_size {block_size} does not divide the {n} unknowns; for a "
            f"system of PDEs the DOF count must be a whole number of nodes")
    n_nodes = n // block_size

    coo = A.tocoo()
    blocks = sparse.coo_matrix(
        (coo.data ** 2, (coo.row // block_size, coo.col // block_size)),
        shape=(n_nodes, n_nodes)).tocsr()
    blocks.data = np.sqrt(blocks.data)

    diagonal = np.abs(blocks.diagonal())
    diagonal[diagonal <= 0.0] = 1.0
    scale = np.sqrt(diagonal)

    coo = blocks.tocoo()
    off = coo.row != coo.col
    rows, cols, vals = coo.row[off], coo.col[off], coo.data[off]
    strong = vals > theta * scale[rows] * scale[cols]

    graph = sparse.coo_matrix(
        (np.ones(int(strong.sum())), (rows[strong], cols[strong])),
        shape=(n_nodes, n_nodes)).tocsr()
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return graph


def aggregate(graph: sparse.csr_matrix) -> np.ndarray:
    """Greedy two-pass aggregation. Returns an aggregate index per node.

    Pass one takes only nodes whose entire strong neighbourhood is still free,
    which produces well-separated aggregates of a sensible size. Pass two
    assigns whatever is left to an adjacent aggregate; skipping it leaves
    singletons that show up later as near-zero rows in the coarse operator.
    """
    n = graph.shape[0]
    assignment = np.full(n, -1, dtype=np.int64)
    indptr, indices = graph.indptr, graph.indices
    n_aggregates = 0

    for i in range(n):
        if assignment[i] != -1:
            continue
        neighbours = indices[indptr[i]:indptr[i + 1]]
        if neighbours.size and np.any(assignment[neighbours] != -1):
            continue
        assignment[i] = n_aggregates
        assignment[neighbours] = n_aggregates
        n_aggregates += 1

    # second pass: attach the remainder to a neighbouring aggregate
    for i in np.flatnonzero(assignment == -1):
        neighbours = indices[indptr[i]:indptr[i + 1]]
        taken = neighbours[assignment[neighbours] != -1] if neighbours.size \
            else np.empty(0, dtype=np.int64)
        if taken.size:
            assignment[i] = assignment[taken[0]]
        else:
            assignment[i] = n_aggregates       # genuinely isolated node
            n_aggregates += 1

    return assignment


def _tentative_prolongator(assignment: np.ndarray, near_null: np.ndarray | None,
                           n: int):
    r"""One block of columns per aggregate, carrying the near-null space.

    Returns ``(P_tent, B_coarse)``. For a scalar problem ``near_null`` is the
    constant vector and ``P_tent`` is a normalised indicator matrix; for a
    system it is one column per mode per aggregate.

    ``B_coarse`` is the second half of the algorithm and is easy to omit. The
    defining property of the tentative prolongator is that it *represents the
    near-null space exactly*, ``P_tent B_coarse = B``, and ``B_coarse`` is the
    ``R`` of the local QR. Dropping it and letting the coarse levels fall back
    to a constant vector throws the rigid-body modes away after level one, so
    the hierarchy is only correct at its finest level -- which looks like a
    partial improvement rather than a bug.
    """
    n_aggregates = int(assignment.max()) + 1

    if near_null is None:
        near_null = np.ones((n, 1))
    near_null = np.atleast_2d(np.asarray(near_null, dtype=float))
    if near_null.shape[0] != n:
        near_null = near_null.T
    if near_null.shape[0] != n:
        raise ValueError(f"near_null must have {n} rows")
    n_modes = near_null.shape[1]

    rows, cols, data = [], [], []
    coarse_rows, coarse_blocks = [], []
    column = 0
    for a in range(n_aggregates):
        members = np.flatnonzero(assignment == a)
        if members.size == 0:
            continue
        block = near_null[members]                      # (len, n_modes)
        # local orthonormalisation keeps P_tent well conditioned, which the
        # Galerkin product depends on; R carries the modes to the coarse level
        q, r = np.linalg.qr(block)
        keep = min(n_modes, q.shape[1], members.size)
        for j in range(keep):
            rows.extend(members.tolist())
            cols.extend([column + j] * members.size)
            data.extend(q[:, j].tolist())
        coarse_rows.append(r[:keep, :])
        coarse_blocks.append(keep)
        column += keep

    P = sparse.coo_matrix((data, (rows, cols)), shape=(n, column)).tocsr()
    B_coarse = (np.vstack(coarse_rows) if coarse_rows
                else np.zeros((0, n_modes)))
    return P, B_coarse


def _smooth_prolongator(A: sparse.csr_matrix, P: sparse.csr_matrix,
                        omega: float) -> sparse.csr_matrix:
    r"""``P <- (I - omega D^-1 A) P``, one Jacobi sweep on the columns.

    The step that distinguishes smoothed aggregation from plain aggregation.
    ``omega`` is scaled by an estimate of the spectral radius of ``D^-1 A``,
    for which the row-sum bound is used -- cheap, always an upper bound, and
    therefore safe.
    """
    diagonal = A.diagonal().copy()
    diagonal[diagonal == 0.0] = 1.0
    D_inv = sparse.diags(1.0 / diagonal)

    # Gershgorin bound on rho(D^-1 A); an overestimate only makes the sweep
    # gentler, never unstable
    row_sums = np.abs(A).dot(np.ones(A.shape[0])) / np.abs(diagonal)
    rho = max(float(row_sums.max()), 1.0)

    return (P - (omega / rho) * (D_inv @ (A @ P))).tocsr()


@dataclass
class AMGLevel:
    """One level of the hierarchy."""

    A: sparse.csr_matrix
    P: sparse.csr_matrix | None = None      #: prolongator to the next-coarser
    diagonal_inverse: np.ndarray | None = None
    n_aggregates: int = 0

    def __post_init__(self):
        diagonal = self.A.diagonal().copy()
        diagonal[diagonal == 0.0] = 1.0
        self.diagonal_inverse = 1.0 / diagonal


@dataclass
class AMGHierarchy:
    """A smoothed-aggregation hierarchy and its V-cycle."""

    levels: list = field(default_factory=list)
    n_presmooth: int = 1
    n_postsmooth: int = 1
    smoother_weight: float = 0.7
    coarse_factorisation: object = None

    # ------------------------------------------------------------------
    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def sizes(self) -> list:
        return [level.A.shape[0] for level in self.levels]

    @property
    def operator_complexity(self) -> float:
        """Total nonzeros across levels divided by the fine-level count.

        The standard measure of what the hierarchy costs. Above about 2 the
        coarse operators have filled in badly and each V-cycle is expensive;
        smoothed aggregation trades some of this for a better coarse space.
        """
        total = sum(level.A.nnz for level in self.levels)
        return total / max(1, self.levels[0].A.nnz)

    # ------------------------------------------------------------------
    def _smooth(self, level: AMGLevel, x: np.ndarray, b: np.ndarray,
                sweeps: int) -> np.ndarray:
        """Weighted Jacobi. Symmetric, which a V-cycle used inside PCG needs;
        Gauss-Seidel is a better smoother and is not symmetric unless the
        sweeps are alternated."""
        for _ in range(sweeps):
            x = x + self.smoother_weight * level.diagonal_inverse * (
                b - level.A @ x)
        return x

    def _vcycle(self, index: int, b: np.ndarray) -> np.ndarray:
        level = self.levels[index]

        if level.P is None:                          # coarsest
            if self.coarse_factorisation is not None:
                return self.coarse_factorisation(b)
            return self._smooth(level, np.zeros_like(b), b, 20)

        x = self._smooth(level, np.zeros_like(b), b, self.n_presmooth)
        residual = b - level.A @ x
        coarse = self._vcycle(index + 1, level.P.T @ residual)
        x = x + level.P @ coarse
        return self._smooth(level, x, b, self.n_postsmooth)

    def apply(self, b: np.ndarray) -> np.ndarray:
        """One symmetric V-cycle, i.e. the action of ``B^-1``."""
        b = np.asarray(b, dtype=float).ravel()
        return self._vcycle(0, b)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        return self.apply(b)

    def solve(self, b, tol: float = 1e-10, max_iter: int = 100):
        """Standalone multigrid iteration, for checking the hierarchy.

        Used to verify that the hierarchy converges *on its own* -- if it does
        not, it is a poor preconditioner too, and the diagnosis is much easier
        here than inside PCG.
        """
        b = np.asarray(b, dtype=float).ravel()
        A = self.levels[0].A
        x = np.zeros_like(b)
        norm = float(np.linalg.norm(b)) or 1.0
        history = [1.0]
        for _ in range(max_iter):
            x = x + self.apply(b - A @ x)
            history.append(float(np.linalg.norm(b - A @ x)) / norm)
            if history[-1] < tol:
                break
        return x, np.array(history)

    def __repr__(self) -> str:
        chain = " -> ".join(str(s) for s in self.sizes)
        return (f"AMGHierarchy({self.n_levels} levels: {chain}, "
                f"complexity {self.operator_complexity:.2f})")


def rigid_body_translations(n: int, block_size: int) -> np.ndarray:
    """The default near-null space for a system: one constant per component.

    ``block_size`` columns, the ``c``-th being one on every DOF of component
    ``c``. These are the rigid-body *translations*, which need no geometry. The
    rotations do, so they are not supplied automatically -- pass them through
    ``near_null`` when they matter, which for a slender elastic body they do.
    """
    modes = np.zeros((n, block_size))
    component = np.arange(n) % block_size
    for c in range(block_size):
        modes[component == c, c] = 1.0
    return modes


def smoothed_aggregation(A: sparse.spmatrix, theta: float = 0.08,
                         max_levels: int = 12, max_coarse: int = 40,
                         omega: float = 4.0 / 3.0, near_null=None,
                         n_presmooth: int = 1, n_postsmooth: int = 1,
                         smoother_weight: float = 0.7,
                         block_size: int = 1) -> AMGHierarchy:
    """Build a smoothed-aggregation hierarchy for an SPD matrix.

    Parameters
    ----------
    theta : strength-of-connection threshold. Smaller keeps more couplings and
        gives smaller, more numerous aggregates.
    max_coarse : stop coarsening at this size and factorise directly. A dense
        solve on 40 unknowns is free and removes the coarsest-level smoothing
        error entirely.
    omega : Jacobi weight for prolongator smoothing; ``4/3`` is the standard
        choice, scaled internally by an estimate of ``rho(D^-1 A)``.
    near_null : ``(n, k)`` near-null-space modes. Defaults to the constant
        vector for ``block_size = 1`` and to the rigid-body translations
        otherwise. For a slender elastic body the rotations matter too and need
        geometry, so pass them explicitly.
    block_size : DOFs per node. **Set this for a system of PDEs** -- 2 for 2-D
        elasticity, 3 for 3-D. It switches aggregation from the DOF graph to
        the node graph, which is what keeps a node's DOFs in one aggregate. The
        default of 1 is a scalar AMG and degrades badly on elasticity; see the
        table in the module docstring.

    Coarsening stops early, with a warning, if a level fails to shrink; a
    hierarchy that is not contracting is worse than one fewer level, since each
    extra level costs a smoothing sweep and buys nothing.
    """
    A0 = sparse.csr_matrix(A, dtype=float)
    if A0.shape[0] != A0.shape[1]:
        raise ValueError(f"A must be square, got {A0.shape}")

    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if A0.shape[0] % block_size:
        raise ValueError(
            f"block_size {block_size} does not divide the {A0.shape[0]} "
            f"unknowns")

    levels: list[AMGLevel] = []
    current = A0
    modes = (near_null if near_null is not None or block_size == 1
             else rigid_body_translations(A0.shape[0], block_size))
    block = block_size

    for depth in range(max_levels - 1):
        if current.shape[0] <= max_coarse:
            break

        if block > 1 and current.shape[0] % block == 0:
            # aggregate nodes, then expand back to DOFs: node-major ordering
            # means DOF i*block + c belongs to node i
            nodes = aggregate(nodal_strength(current, block, theta))
            assignment = np.repeat(nodes, block)
        else:
            assignment = aggregate(strength_of_connection(current, theta))

        P_tent, B_coarse = _tentative_prolongator(assignment, modes,
                                                  current.shape[0])
        if P_tent.shape[1] == 0:
            break
        P = _smooth_prolongator(current, P_tent, omega)

        coarse = (P.T @ current @ P).tocsr()
        coarse.eliminate_zeros()

        if coarse.shape[0] >= current.shape[0]:
            logger.warning(
                "AMG stopped at level %d: coarsening produced %d unknowns from "
                "%d, so the hierarchy is not contracting", depth,
                coarse.shape[0], current.shape[0])
            break

        level = AMGLevel(A=current, P=P,
                         n_aggregates=int(assignment.max()) + 1)
        levels.append(level)
        current = coarse
        # carry the near-null space up: B_coarse satisfies P_tent B_coarse = B
        # exactly, so the coarse operator sees the same modes. Falling back to a
        # constant here would discard the rigid-body modes after level one.
        modes = B_coarse
        # each coarse unknown block now holds one entry per retained mode
        block = B_coarse.shape[1] if B_coarse.size else 1

    levels.append(AMGLevel(A=current, P=None))

    hierarchy = AMGHierarchy(levels=levels, n_presmooth=n_presmooth,
                             n_postsmooth=n_postsmooth,
                             smoother_weight=smoother_weight)

    if current.shape[0] <= max(max_coarse, 200):
        dense = current.toarray()
        try:
            from scipy.linalg import cho_factor, cho_solve
            factor = cho_factor(dense, lower=True)
            hierarchy.coarse_factorisation = lambda v: cho_solve(factor, v)
        except np.linalg.LinAlgError:
            # a singular coarse operator means the near-null space was not
            # represented; smoothing there is a poor but safe fallback
            logger.warning("AMG coarse operator is not positive definite; "
                           "falling back to smoothing on the coarsest level")

    logger.info("AMG hierarchy: %s", hierarchy)
    return hierarchy
