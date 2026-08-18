r"""
Preconditioners for SPD Krylov solves.

All of them are the same object from the solver's point of view: something that
applies ``B^-1`` to a vector. ``B`` is never formed. What separates them is
where they sit on the curve between setup cost and iteration reduction, and
none is uniformly best -- that observation is the premise of the two-level work
in ``projects/diffusion_preconditioner``.

    Identity            nothing. The baseline every other number is read against.
    Jacobi              diag(A)^-1. One divide per unknown, fully parallel, and
                        weak on anything with heterogeneous coefficients.
    IncompleteCholesky  IC(0) -- sparse triangular factors on A's own pattern.
                        Symmetric, so it is the natural SPD choice.
    IncompleteLU        scipy's spilu. Stronger than IC(0) and **not SPD**, so
                        PCG's assumptions are violated; see below.
    AlgebraicMultigrid  a symmetric V-cycle. Most expensive to set up and the
                        only one whose iteration count is mesh-independent.

ILU is not SPD, and that is a real caveat rather than a fatal one
----------------------------------------------------------------
PCG uses ``r' z`` as a squared norm, so it assumes ``B`` is symmetric positive
definite. A general ILU factorisation of an SPD matrix is neither symmetric nor
guaranteed definite, so the assumption is violated -- but being precise about
what follows: it usually works anyway. Measured on a heterogeneous diffusion
problem with contrast 1e5, ILU-preconditioned PCG converged in 4 iterations
without complaint. The failure is not certain, it is *unpredictable*: ``r' z``
can go negative, and when it does the step lengths are meaningless, so
:func:`~.krylov.pcg` raises at that point rather than returning a wrong answer.

:class:`IncompleteLU` is here because the accompanying paper compares against
it, and it carries ``is_spd = False`` so the caveat is visible from the object
rather than only from a docstring. On an SPD system prefer
:class:`IncompleteCholesky`, which is symmetric by construction and costs the
same to apply.

IC(0) can break down, and the fix is a shift
--------------------------------------------
Incomplete Cholesky discards fill, which can drive a pivot non-positive on a
matrix whose exact Cholesky factorisation exists. The standard remedy is to
factor ``A + delta*diag(A)`` instead, which is what
:class:`IncompleteCholesky` does automatically -- doubling ``delta`` until the
factorisation succeeds and reporting the value it settled on. A silent failure
here would produce a preconditioner containing NaNs and a solver that reports
zero iterations.

Setup cost is part of the comparison
------------------------------------
Every preconditioner records :attr:`Preconditioner.setup_time`. For a single
solve, AMG setup can exceed the entire unpreconditioned iteration; across a
hundred right-hand sides it is free. The whole argument for a reusable coarse
space is amortisation, so the setup number has to be visible or the comparison
is dishonest.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from simulacrum.core.logging import get_logger
from simulacrum.solvers.amg import smoothed_aggregation
from simulacrum.solvers.schwarz import build_schwarz

logger = get_logger(__name__)

__all__ = [
    "Preconditioner",
    "Identity",
    "Jacobi",
    "IncompleteCholesky",
    "IncompleteLU",
    "AlgebraicMultigrid",
    "AdditiveSchwarz",
    "PRECONDITIONER_REGISTRY",
    "get_preconditioner",
    "PYAMG_AVAILABLE",
]

try:                                        # pragma: no cover - environment
    import pyamg                            # noqa: F401
    PYAMG_AVAILABLE = True
except ImportError:                         # pragma: no cover - environment
    PYAMG_AVAILABLE = False


class Preconditioner(ABC):
    """Something that applies ``B^-1``. The only interface PCG needs."""

    name: str = "preconditioner"

    #: Whether ``B`` is symmetric positive definite. False means PCG's
    #: assumptions are violated and it may raise; it is not a suggestion.
    is_spd: bool = True

    def __init__(self):
        self.setup_time: float = 0.0
        self.n_unknowns: int = 0

    # ------------------------------------------------------------------
    def setup(self, A: sparse.spmatrix) -> "Preconditioner":
        """Build whatever the preconditioner needs, and time it."""
        A = sparse.csr_matrix(A, dtype=float)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got {A.shape}")
        self.n_unknowns = A.shape[0]
        started = time.perf_counter()
        self._setup(A)
        self.setup_time = time.perf_counter() - started
        return self

    @abstractmethod
    def _setup(self, A: sparse.csr_matrix) -> None:
        ...

    @abstractmethod
    def apply(self, v: np.ndarray) -> np.ndarray:
        """``B^-1 v``."""

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

    def as_linear_operator(self) -> spla.LinearOperator:
        """For handing to scipy's own Krylov routines."""
        n = self.n_unknowns
        return spla.LinearOperator((n, n), matvec=self.apply)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(n={self.n_unknowns}, "
                f"setup={self.setup_time:.3f}s)")


class Identity(Preconditioner):
    """No preconditioning. ``B = I``."""

    name = "none"

    def _setup(self, A):
        pass

    def apply(self, v):
        return np.asarray(v, dtype=float).ravel()


class Jacobi(Preconditioner):
    r"""``B = diag(A)``.

    The cheapest thing that is not nothing, and the reference point for
    "cheap". It removes scale differences between rows and nothing else, so on
    a heterogeneous-coefficient or anisotropic operator it barely moves the
    iteration count -- which is exactly the regime a coarse-space correction is
    for.
    """

    name = "jacobi"

    def _setup(self, A):
        diagonal = A.diagonal().copy()
        bad = diagonal == 0.0
        if bad.any():
            # a zero diagonal on an SPD matrix means a structurally empty row
            raise ValueError(
                f"{int(bad.sum())} zero diagonal entries: the matrix is not "
                f"positive definite, or rows are structurally empty")
        self._inverse = 1.0 / diagonal

    def apply(self, v):
        return self._inverse * np.asarray(v, dtype=float).ravel()


class IncompleteCholesky(Preconditioner):
    r"""IC(0): incomplete Cholesky on ``A``'s own sparsity pattern.

    ``A ~ L L'`` with ``L`` restricted to the lower triangle of ``A``, so the
    preconditioner is two sparse triangular solves per application and needs no
    more memory than ``A`` itself. Symmetric by construction, which is why it is
    the natural choice for PCG over ILU.

    Parameters
    ----------
    shift : initial diagonal shift as a fraction of ``diag(A)``. Zero means try
        the unshifted factorisation first and escalate only if it breaks down.
    max_attempts : how many times to double the shift before giving up. The
        default reaches a shift of about 800 times the diagonal, which sounds
        absurd until you meet ``nos1`` or ``msc01050`` from the paper's own
        benchmark set: both need more than 0.2, and a cap of 12 attempts made
        them fail outright rather than converge slowly.

    Notes
    -----
    The factorisation is a straightforward left-looking IC(0) in Python. It is
    the *setup* that is slow, not the apply: measured on 2-D Poisson the cost is
    linear in the nonzero count at about 5.5 microseconds per nonzero, so
    roughly 0.3 s at 1e4 unknowns, 3 s at 1e5, and half a minute at 1e6. The
    apply is SuperLU triangular solves and is C-speed. If the setup dominates
    your problem, the answer is AMG -- 5x faster to build here and far better
    per iteration -- not a faster IC(0).

    Threshold variant
    -----------------
    ``drop_tol > 0`` switches from IC(0) to **ICT**: fill outside ``A``'s pattern
    is allowed, and entries are discarded by magnitude instead. Each column of
    ``L`` is computed in full and then thinned, keeping ``|l_ij| >= drop_tol *
    ||A(:,j)||`` and at most ``fill_factor`` times the original column count.

    The two controls do different jobs and both are needed. The tolerance adapts
    to the column's own scale, so it behaves sensibly on an operator with mixed
    units; the fill cap bounds memory, because a tolerance alone gives no
    guarantee at all -- on a matrix with slow decay a small tolerance can fill a
    column densely. Together they are Lin and Moré's IC(tau, p).

    Measured on a 900-unknown diffusion problem with contrast 1e6,
    ``fill_factor = 8``, PCG to 1e-8:

        variant            iterations   nnz(L)/nnz(tril A)
        IC(0)                      98                 1.00
        ICT  drop_tol=0.3          62                 0.87
        ICT  drop_tol=0.1          34                 1.04
        ICT  drop_tol=0.03         24                 1.25
        ICT  drop_tol=0.01         15                 1.51
        ICT  drop_tol=0.001         9                 2.53

    The first ICT row is the interesting one: **fewer nonzeros than IC(0) and
    still 1.6x fewer iterations.** That is the real argument for thresholding
    over level-zero fill -- IC(0) keeps whichever entries happen to lie in
    ``A``'s pattern, while ICT keeps the largest wherever they turn up, and on a
    heterogeneous operator those are not the same set.

    Dropping fill makes a non-positive pivot *more* likely, not less, so the
    shift escalation matters more here than for IC(0).
    """

    name = "icc"

    def __init__(self, shift: float = 0.0, max_attempts: int = 24,
                 drop_tol: float = 0.0, fill_factor: float = 5.0):
        super().__init__()
        self.shift = float(shift)
        self.max_attempts = int(max_attempts)
        self.drop_tol = float(drop_tol)
        self.fill_factor = float(fill_factor)
        if self.drop_tol < 0.0:
            raise ValueError("drop_tol must not be negative")
        if self.fill_factor < 1.0:
            raise ValueError("fill_factor must be at least 1, or the "
                             "factorisation cannot keep A's own pattern")
        self.applied_shift: float = 0.0
        self.fill_ratio: float = 1.0

    @property
    def variant(self) -> str:
        return "ic0" if self.drop_tol == 0.0 else "ict"

    def _setup(self, A):
        diagonal = A.diagonal().copy()
        shift = self.shift
        for attempt in range(self.max_attempts):
            shifted = A if shift == 0.0 else (
                A + sparse.diags(shift * np.abs(diagonal)))
            L = (_ic0(sparse.csc_matrix(shifted)) if self.drop_tol == 0.0
                 else _ict(sparse.csc_matrix(shifted), self.drop_tol,
                           self.fill_factor))
            if L is not None:
                self.fill_ratio = L.nnz / max(1, sparse.tril(A).nnz)
                self.applied_shift = shift
                self._L = L.tocsr()
                # SuperLU rather than spsolve_triangular: the factorisation of
                # an already-triangular matrix is nearly free, and its solve
                # runs in C. Measured on a 576-unknown heterogeneous diffusion
                # problem, spsolve_triangular made the *apply* 100x slower than
                # an AMG V-cycle and dominated the whole comparison.
                self._forward = spla.splu(sparse.csc_matrix(self._L),
                                          permc_spec="NATURAL",
                                          options={"SymmetricMode": False})
                self._backward = spla.splu(sparse.csc_matrix(self._L.T),
                                           permc_spec="NATURAL",
                                           options={"SymmetricMode": False})
                if attempt:
                    logger.info("IC(0) needed a diagonal shift of %.3g", shift)
                return
            shift = 1e-4 if shift == 0.0 else 2.0 * shift

        raise ValueError(
            f"{self.variant.upper()} broke down for every shift up to "
            f"{shift:.3g}: a pivot went non-positive, which means dropping fill "
            f"has destroyed definiteness. Try a smaller `drop_tol` (keeping more "
            f"fill makes breakdown less likely, not more), a larger initial "
            f"`shift`, or AlgebraicMultigrid.")

    def apply(self, v):
        v = np.asarray(v, dtype=float).ravel()
        return self._backward.solve(self._forward.solve(v))


def _ic0(A: sparse.csc_matrix):
    r"""Left-looking IC(0). Returns ``L`` or ``None`` on a non-positive pivot.

    Restricted to the lower-triangular pattern of ``A``: entries generated by
    elimination outside that pattern are discarded, which is what makes it
    incomplete and what can cost definiteness.
    """
    n = A.shape[0]
    A = sparse.tril(A, format="csr")
    indptr, indices, data = A.indptr, A.indices.copy(), A.data.copy().astype(float)

    # dense row workspace: fine for the bandwidths this is used on
    for i in range(n):
        row_start, row_end = indptr[i], indptr[i + 1]
        cols = indices[row_start:row_end]

        for slot in range(row_start, row_end):
            j = indices[slot]
            if j == i:
                continue
            # data[slot] -= sum_k L[i,k] L[j,k] over the shared pattern
            j_start, j_end = indptr[j], indptr[j + 1]
            shared, i_pos, j_pos = np.intersect1d(
                cols[cols < j], indices[j_start:j_end][
                    indices[j_start:j_end] < j],
                assume_unique=True, return_indices=True)
            if shared.size:
                left = data[row_start:row_end][cols < j][i_pos]
                right = data[j_start:j_end][
                    indices[j_start:j_end] < j][j_pos]
                data[slot] -= float(np.dot(left, right))
            pivot = data[indptr[j + 1] - 1]
            if pivot <= 0.0:
                return None
            data[slot] /= pivot

        diagonal_slot = row_end - 1
        off = data[row_start:diagonal_slot]
        value = data[diagonal_slot] - float(np.dot(off, off))
        if value <= 0.0:
            return None
        data[diagonal_slot] = np.sqrt(value)

    return sparse.csr_matrix((data, indices, indptr), shape=(n, n))


def _ict(A: sparse.csc_matrix, drop_tol: float, fill_factor: float):
    r"""Threshold incomplete Cholesky, IC(tau, p). ``L`` or ``None``.

    Left-looking and column by column: column ``j`` is assembled in a dense
    workspace, every previously computed column with a nonzero in row ``j``
    subtracts its contribution, and only then is the result thinned. Computing
    the column in full before dropping is the point -- dropping earlier changes
    which contributions later columns see, and the factorisation stops
    approximating ``A`` in any controlled way.

    Two structures make this affordable. ``row_columns[i]`` lists the columns
    with a nonzero at row ``i``, which is what turns "find the contributing
    columns" from a scan of everything computed so far into a lookup. And the
    dense workspace is reset only at the entries actually touched, since
    rezeroing ``n`` floats per column would make the whole thing quadratic.
    """
    n = A.shape[0]
    lower = sparse.tril(A).tocsc()
    indptr, indices, data = lower.indptr, lower.indices, lower.data

    columns: list = []
    row_columns: list = [[] for _ in range(n)]
    workspace = np.zeros(n)

    for j in range(n):
        start, end = indptr[j], indptr[j + 1]
        original = indices[start:end]
        workspace[original] = data[start:end]
        touched = [original]

        for k in row_columns[j]:
            rows_k, vals_k = columns[k]
            position = int(np.searchsorted(rows_k, j))
            l_jk = vals_k[position]
            if l_jk == 0.0:
                continue
            tail = rows_k[position:]
            workspace[tail] -= vals_k[position:] * l_jk
            touched.append(tail)

        pivot = workspace[j]
        if pivot <= 0.0:
            for block in touched:
                workspace[block] = 0.0
            return None
        diagonal = np.sqrt(pivot)

        candidates = np.unique(np.concatenate(touched))
        candidates = candidates[candidates > j]
        values = workspace[candidates] / diagonal

        # Threshold against the norm of *this column of L*, not of A(:,j).
        # MATLAB's ichol('ict') uses the latter, and it only behaves sensibly
        # when A is reasonably scaled: L's entries carry a 1/sqrt(pivot), so on
        # an operator with entries of order 1e8 a tolerance of 1e-3 against
        # ||A(:,j)|| discards nearly the whole column. Measured that way, ICT
        # produced a *sparser* factor than IC(0) (fill 0.34) and 278 iterations
        # against IC(0)'s 98 at every tolerance from 1e-1 to 1e-4 -- the
        # parameter had no useful range at all. Self-relative is scale-free.
        scale = float(np.linalg.norm(values)) or 1.0
        keep = np.abs(values) >= drop_tol * scale
        candidates, values = candidates[keep], values[keep]

        # The tolerance alone bounds nothing: on a matrix with slow decay a
        # small tau can fill a column densely, so cap the count as well. The
        # budget counts *off-diagonals*, against A's own off-diagonal count, so
        # that fill_factor = 1 means "no more nonzeros than tril(A)" exactly.
        # Measuring it against the full column instead lets the diagonal through
        # on top and gives a fill ratio of 1.29 at fill_factor = 1.
        budget = max(1, int(fill_factor * max(1, end - start - 1)))
        if candidates.size > budget:
            largest = np.argpartition(-np.abs(values), budget - 1)[:budget]
            largest.sort()
            candidates, values = candidates[largest], values[largest]

        columns.append((np.concatenate([[j], candidates]).astype(np.int64),
                        np.concatenate([[diagonal], values])))
        for row in candidates:
            row_columns[row].append(j)

        for block in touched:
            workspace[block] = 0.0

    lengths = [rows.size for rows, _ in columns]
    return sparse.csc_matrix(
        (np.concatenate([vals for _, vals in columns]),
         np.concatenate([rows for rows, _ in columns]),
         np.concatenate([[0], np.cumsum(lengths)])), shape=(n, n)).tocsr()


class IncompleteLU(Preconditioner):
    r"""scipy's ``spilu``. Effective, and **not SPD**.

    Included because it is one of the baselines in the accompanying paper. For a
    symmetric matrix the factors are still computed without exploiting or
    preserving symmetry, so ``B^-1`` is not symmetric and PCG's use of ``r'z``
    as a squared norm is unjustified. :func:`~.krylov.pcg` detects the
    consequence and raises; that is the honest outcome, not a bug to work
    around. Use :class:`IncompleteCholesky` on an SPD system, or a flexible
    Krylov method.
    """

    name = "ilu"
    is_spd = False

    def __init__(self, drop_tol: float = 1e-4, fill_factor: float = 10.0):
        super().__init__()
        self.drop_tol = float(drop_tol)
        self.fill_factor = float(fill_factor)

    def _setup(self, A):
        self._ilu = spla.spilu(sparse.csc_matrix(A), drop_tol=self.drop_tol,
                               fill_factor=self.fill_factor)

    def apply(self, v):
        return self._ilu.solve(np.asarray(v, dtype=float).ravel())


class AlgebraicMultigrid(Preconditioner):
    r"""A symmetric V-cycle as ``B^-1``.

    The only preconditioner here whose iteration count does not grow with the
    mesh on an elliptic problem, and the most expensive to set up. See
    :mod:`~.amg` for the construction.

    Parameters
    ----------
    backend : ``"native"`` (default, numpy/scipy) or ``"pyamg"``. The published
        results in the accompanying paper used pyamg; the two are not expected
        to give identical iteration counts, since coarsening heuristics differ.
    near_null : ``(n, k)`` near-null-space modes. Defaults to the constant
        vector for ``block_size = 1``, and to the rigid-body translations
        otherwise.
    block_size : DOFs per node. **Set it for a system of PDEs** -- 2 for 2-D
        elasticity, 3 for 3-D. It is what keeps a node's DOFs in one aggregate,
        and without it AMG on elasticity is worse than IC(0); see the table in
        :mod:`~.amg`.
    """

    name = "amg"

    def __init__(self, backend: str = "native", theta: float = 0.08,
                 near_null=None, n_presmooth: int = 1, n_postsmooth: int = 1,
                 max_coarse: int = 40, smoother_weight: float = 0.7,
                 pyamg_kind: str = "sa", cycle: str = "V",
                 block_size: int = 1):
        super().__init__()
        if backend not in ("native", "pyamg"):
            raise ValueError(f"backend must be 'native' or 'pyamg', got "
                             f"{backend!r}")
        if backend == "pyamg" and not PYAMG_AVAILABLE:
            raise ImportError(
                "pyamg is not installed and is not a dependency of Simulacrum. "
                "Use backend='native', which is pure NumPy/SciPy, or install "
                "pyamg to reproduce the paper's published numbers.")
        self.backend = backend
        self.theta = float(theta)
        self.near_null = near_null
        self.n_presmooth = int(n_presmooth)
        self.n_postsmooth = int(n_postsmooth)
        self.max_coarse = int(max_coarse)
        self.smoother_weight = float(smoother_weight)
        self.pyamg_kind = pyamg_kind
        self.cycle = cycle
        self.block_size = int(block_size)
        self.hierarchy = None

    def _setup(self, A):
        if self.backend == "pyamg":            # pragma: no cover - environment
            import pyamg
            builder = (pyamg.ruge_stuben_solver
                       if self.pyamg_kind.lower() in ("rs", "ruge-stuben")
                       else pyamg.smoothed_aggregation_solver)
            self.hierarchy = builder(A.tocsr())
            self._operator = self.hierarchy.aspreconditioner(cycle=self.cycle)
            self._apply = self._operator.matvec
            return

        self.hierarchy = smoothed_aggregation(
            A, theta=self.theta, near_null=self.near_null,
            n_presmooth=self.n_presmooth, n_postsmooth=self.n_postsmooth,
            max_coarse=self.max_coarse, smoother_weight=self.smoother_weight,
            block_size=self.block_size)
        self._apply = self.hierarchy.apply

    def apply(self, v):
        return np.asarray(self._apply(np.asarray(v, dtype=float).ravel()),
                          dtype=float).ravel()

    @property
    def levels(self) -> int:
        if self.backend == "pyamg":            # pragma: no cover
            return len(self.hierarchy.levels)
        return self.hierarchy.n_levels

    def __repr__(self) -> str:
        return (f"AlgebraicMultigrid({self.backend}, {self.levels} levels, "
                f"setup={self.setup_time:.3f}s)")


class AdditiveSchwarz(Preconditioner):
    r"""Overlapping domain decomposition, built from the matrix graph.

    ``M^-1 = sum_i R_i' A_i^-1 R_i``. See :mod:`~.schwarz` for the construction
    and for why one level does not scale.

    Parameters
    ----------
    n_subdomains : how many pieces. More means cheaper local factorisations and
        a worse iteration count -- that is the whole trade, and it is why a
        coarse space is needed for a method that scales.
    overlap : graph layers added to each subdomain. Zero is block Jacobi.
    restricted : RAS. Faster, and **not symmetric**, so not valid under PCG --
        ``is_spd`` is set to False and GMRES or BiCGSTAB should be used.
    method : ``"spectral"``, ``"bfs"`` or ``"metis"``; see
        :func:`~.schwarz.partition_graph`.
    """

    name = "schwarz"

    def __init__(self, n_subdomains: int = 8, overlap: int = 1,
                 restricted: bool = False, method: str = "spectral"):
        super().__init__()
        self.n_subdomains = int(n_subdomains)
        self.overlap = int(overlap)
        self.restricted = bool(restricted)
        self.method = method
        # RAS restricts with overlap and scatters back without it, which is
        # exactly what destroys symmetry
        self.is_spd = not self.restricted
        self.operator = None

    def _setup(self, A):
        self.operator = build_schwarz(A, self.n_subdomains, self.overlap,
                                      self.restricted, self.method)

    def apply(self, v):
        return self.operator.apply(v)

    @property
    def overlap_ratio(self) -> float:
        return self.operator.overlap_ratio

    def __repr__(self) -> str:
        kind = "RAS" if self.restricted else "ASM"
        return (f"AdditiveSchwarz({kind}, {self.n_subdomains} subdomains, "
                f"overlap {self.overlap}, setup={self.setup_time:.3f}s)")


class _ThresholdCholesky(IncompleteCholesky):
    """:class:`IncompleteCholesky` with dropping on by default.

    Registered as ``"ict"`` so ``get_preconditioner("ict", A)`` gives a sensible
    threshold variant without having to remember the parameter name.
    """

    name = "ict"

    def __init__(self, shift: float = 0.0, max_attempts: int = 24,
                 drop_tol: float = 1e-3, fill_factor: float = 5.0):
        super().__init__(shift=shift, max_attempts=max_attempts,
                         drop_tol=drop_tol, fill_factor=fill_factor)


class _RestrictedSchwarz(AdditiveSchwarz):
    """RAS by default. Registered as ``"ras"``; not SPD, so use GMRES."""

    name = "ras"

    def __init__(self, n_subdomains: int = 8, overlap: int = 1,
                 restricted: bool = True, method: str = "spectral"):
        super().__init__(n_subdomains, overlap, restricted, method)


# ---------------------------------------------------------------------------

PRECONDITIONER_REGISTRY: dict[str, type[Preconditioner]] = {
    "none": Identity,
    "identity": Identity,
    "jacobi": Jacobi,
    "diagonal": Jacobi,
    "icc": IncompleteCholesky,
    "ic0": IncompleteCholesky,
    "ichol": IncompleteCholesky,
    "ict": _ThresholdCholesky,
    "ilu": IncompleteLU,
    "amg": AlgebraicMultigrid,
    "multigrid": AlgebraicMultigrid,
    "schwarz": AdditiveSchwarz,
    "asm": AdditiveSchwarz,
    "ras": _RestrictedSchwarz,
}


def get_preconditioner(name: str, A=None, **kwargs) -> Preconditioner:
    """Build a preconditioner by name, and set it up if ``A`` is given."""
    key = name.lower().replace("_", "-")
    if key not in PRECONDITIONER_REGISTRY:
        raise NotImplementedError(
            f"Unknown preconditioner '{name}'. Available: "
            f"{sorted(set(PRECONDITIONER_REGISTRY))}")
    preconditioner = PRECONDITIONER_REGISTRY[key](**kwargs)
    return preconditioner if A is None else preconditioner.setup(A)
