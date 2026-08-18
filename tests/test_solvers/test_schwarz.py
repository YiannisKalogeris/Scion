"""
Verification for algebraic additive Schwarz.

The tests are built around what the theory predicts, because for a domain
decomposition method the interesting properties are all *trends*:

* **One level does not scale**, in two separate directions. More subdomains at
  a fixed problem makes it worse (14, 20, 29, 34, 39 for 2 to 32 pieces), and
  refining the mesh at a fixed number of subdomains makes it worse again
  (22, 26, 29, 33) where AMG stays near flat (11, 12, 13, 14). Both follow from
  the ``1/(H delta)`` condition-number bound, and neither is a defect to tune
  away -- information has to cross the domain and a one-level method moves it
  one subdomain per iteration.
* **Overlap is the parameter that trades the two.** Zero overlap is block
  Jacobi (41 iterations); one, two and three layers give 29, 25, 24 while the
  total subdomain size grows 1.00, 1.25, 1.51, 1.79 times ``n``.
* **RAS is faster and not symmetric.** Measured: classical ASM is symmetric to
  2e-16 and takes 28 GMRES iterations, RAS is asymmetric at 2e-1 and takes 22.
  The asymmetry is the reason it must not be used under PCG.

python -m pytest tests/test_solvers/test_schwarz.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from simulacrum.solvers import gmres, pcg
from simulacrum.solvers.preconditioners import (
    AdditiveSchwarz,
    get_preconditioner,
)
from simulacrum.solvers.schwarz import (
    PYMETIS_AVAILABLE,
    build_schwarz,
    grow_overlap,
    partition_graph,
)
from tests.test_solvers.test_krylov import heterogeneous_diffusion, poisson_2d


@pytest.fixture(scope="module")
def square():
    A = poisson_2d(24)
    b = np.random.default_rng(0).standard_normal(A.shape[0])
    return A, b


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["spectral", "bfs"])
def test_the_partition_covers_every_unknown_exactly_once(method, square):
    A, _ = square
    labels = partition_graph(A, 8, method)

    assert labels.shape == (A.shape[0],)
    assert set(np.unique(labels)) <= set(range(8))
    assert labels.min() >= 0


@pytest.mark.parametrize("method", ["spectral", "bfs"])
def test_the_partition_is_reasonably_balanced(method, square):
    """A dominant piece dictates the factorisation cost, so balance is not
    cosmetic -- it is what the subdomain solves cost."""
    A, _ = square
    sizes = np.bincount(partition_graph(A, 8, method), minlength=8)

    assert (sizes > 0).all()
    assert sizes.max() / sizes.mean() < 2.5


def test_spectral_bisection_cuts_fewer_edges_than_greedy_growth(square):
    """The reason it is the default. Every edge crossing a boundary is coupling
    the preconditioner discards, so a smaller cut is a better preconditioner."""
    A, _ = square
    graph = sparse.csr_matrix(A, copy=True)
    graph.data = np.ones_like(graph.data)

    def edge_cut(labels):
        coo = graph.tocoo()
        return int(np.count_nonzero(labels[coo.row] != labels[coo.col]))

    assert edge_cut(partition_graph(A, 8, "spectral")) < \
        edge_cut(partition_graph(A, 8, "bfs"))


def test_a_single_partition_is_the_whole_matrix(square):
    A, _ = square
    assert not partition_graph(A, 1).any()


def test_a_degenerate_partition_request_is_refused(square):
    A, _ = square
    with pytest.raises(ValueError, match="at least one"):
        partition_graph(A, 0)
    with pytest.raises(ValueError, match="subdomains"):
        partition_graph(A, A.shape[0] + 1)
    with pytest.raises(NotImplementedError, match="Unknown partitioner"):
        partition_graph(A, 4, "kernighan-lin")


@pytest.mark.skipif(PYMETIS_AVAILABLE, reason="pymetis is installed")
def test_the_metis_backend_explains_itself_when_absent(square):
    A, _ = square
    with pytest.raises(ImportError, match="pure SciPy"):
        partition_graph(A, 4, "metis")


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def test_zero_overlap_reproduces_the_disjoint_partition(square):
    """With no overlap the subdomains tile the unknowns and the method is block
    Jacobi -- nothing crosses a boundary."""
    A, _ = square
    labels = partition_graph(A, 8)
    subdomains = grow_overlap(A, labels, overlap=0)

    total = sum(s.size for s in subdomains)
    assert total == A.shape[0]
    assert all(s.owned.all() for s in subdomains)


def test_each_overlap_layer_enlarges_every_subdomain(square):
    A, _ = square
    labels = partition_graph(A, 8)
    sizes = [sum(s.size for s in grow_overlap(A, labels, overlap=k))
             for k in (0, 1, 2, 3)]

    assert sizes == sorted(sizes)
    assert sizes[0] == A.shape[0]
    assert sizes[-1] > 1.5 * sizes[0]


def test_the_owned_mask_always_selects_the_original_partition(square):
    """RAS scatters back only the owned part, so a wrong mask silently changes
    the operator rather than failing."""
    A, _ = square
    labels = partition_graph(A, 6)
    subdomains = grow_overlap(A, labels, overlap=2)

    recovered = np.full(A.shape[0], -1)
    for part, sub in enumerate(subdomains):
        recovered[sub.rows[sub.owned]] = part
    np.testing.assert_array_equal(recovered, labels)


def test_a_negative_overlap_is_refused(square):
    A, _ = square
    with pytest.raises(ValueError, match="overlap"):
        grow_overlap(A, partition_graph(A, 4), overlap=-1)


# ---------------------------------------------------------------------------
# The preconditioner
# ---------------------------------------------------------------------------

def test_schwarz_solves_the_system_it_preconditions(square):
    A, b = square
    M = AdditiveSchwarz(n_subdomains=8, overlap=1).setup(A)
    result = pcg(A, b, M, tol=1e-10, max_iter=9000)

    assert result.converged
    np.testing.assert_allclose(result.x, spla.spsolve(A.tocsc(), b), rtol=1e-6,
                               atol=1e-9)


def test_classical_schwarz_is_symmetric_and_ras_is_not(square):
    """The property that decides which Krylov method is valid. Symmetry is
    exact for classical ASM because every subdomain contributes
    ``R' A_i^-1 R``, a congruence; RAS drops part of the scatter and loses it."""
    A, _ = square
    n = A.shape[0]

    def asymmetry(M):
        dense = np.column_stack([M.apply(e) for e in np.eye(n)])
        return float(np.abs(dense - dense.T).max() / np.abs(dense).max())

    classical = AdditiveSchwarz(n_subdomains=6, overlap=1).setup(A)
    restricted = AdditiveSchwarz(n_subdomains=6, overlap=1,
                                 restricted=True).setup(A)

    assert classical.is_spd is True
    assert asymmetry(classical) < 1e-12
    assert restricted.is_spd is False
    assert asymmetry(restricted) > 1e-3


def test_ras_needs_fewer_iterations_than_classical_schwarz(square):
    """Its selling point: no double counting in the overlap. Compared under
    GMRES, because RAS is not a legal PCG preconditioner."""
    A, b = square
    classical = AdditiveSchwarz(n_subdomains=8, overlap=1).setup(A)
    restricted = AdditiveSchwarz(n_subdomains=8, overlap=1,
                                 restricted=True).setup(A)

    assert gmres(A, b, restricted, tol=1e-8, restart=30,
                 max_iter=9000).iterations < \
        gmres(A, b, classical, tol=1e-8, restart=30, max_iter=9000).iterations


def test_more_overlap_means_fewer_iterations_and_more_work(square):
    """The trade the parameter exists for: 41, 29, 25, 24 iterations against
    subdomain totals of 1.00, 1.25, 1.51, 1.79 times n."""
    A, b = square
    iterations, ratios = [], []
    for overlap in (0, 1, 2, 3):
        M = AdditiveSchwarz(n_subdomains=8, overlap=overlap).setup(A)
        iterations.append(pcg(A, b, M, tol=1e-8, max_iter=9000).iterations)
        ratios.append(M.overlap_ratio)

    assert iterations == sorted(iterations, reverse=True)
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0)


def test_more_subdomains_makes_the_iteration_worse(square):
    """``kappa ~ 1/(H delta)``: smaller subdomains, worse conditioning. This is
    the half of non-scalability that shows up at fixed problem size."""
    A, b = square
    counts = [pcg(A, b, AdditiveSchwarz(n_subdomains=p, overlap=1).setup(A),
                  tol=1e-8, max_iter=9000).iterations
              for p in (2, 4, 8, 16)]

    assert counts == sorted(counts)
    assert counts[-1] > 1.5 * counts[0]


def test_one_level_schwarz_is_not_mesh_independent_and_amg_is():
    """The other half, and the reason two-level Schwarz is what gets cited.
    Measured 22/26/29/33 for ASM against 11/12/13/14 for AMG."""
    schwarz, multigrid = [], []
    for n in (16, 24, 32):
        A = poisson_2d(n)
        b = np.random.default_rng(0).standard_normal(A.shape[0])
        schwarz.append(pcg(A, b, AdditiveSchwarz(n_subdomains=8,
                                                 overlap=1).setup(A),
                           tol=1e-8, max_iter=9000).iterations)
        multigrid.append(pcg(A, b, get_preconditioner("amg", A), tol=1e-8,
                             max_iter=9000).iterations)

    growth = schwarz[-1] / schwarz[0]
    assert growth > 1.2
    assert multigrid[-1] / multigrid[0] < growth


def test_zero_overlap_is_block_jacobi(square):
    """Not an analogy -- with no overlap the subdomain solves are independent
    and the preconditioner is exactly a block-diagonal inverse."""
    A, _ = square
    M = AdditiveSchwarz(n_subdomains=4, overlap=0).setup(A)
    labels = np.full(A.shape[0], -1)
    for part, sub in enumerate(M.operator.subdomains):
        labels[sub.rows] = part

    blocks = [np.flatnonzero(labels == p) for p in range(4)]
    dense = np.zeros((A.shape[0], A.shape[0]))
    for rows in blocks:
        dense[np.ix_(rows, rows)] = np.linalg.inv(
            A[np.ix_(rows, rows)].toarray())

    v = np.random.default_rng(0).standard_normal(A.shape[0])
    np.testing.assert_allclose(M.apply(v), dense @ v, atol=1e-8)


def test_schwarz_works_on_a_heterogeneous_operator():
    """No geometry is used, so a jumping-coefficient problem is no harder to
    set up -- though the coefficient jumps do land inside subdomains and the
    iteration count reflects that."""
    A = heterogeneous_diffusion(20, contrast=1e5)
    b = np.random.default_rng(1).standard_normal(A.shape[0])
    M = AdditiveSchwarz(n_subdomains=8, overlap=1).setup(A)

    result = pcg(A, b, M, tol=1e-8, max_iter=20000)
    assert result.converged
    assert result.iterations < pcg(A, b, tol=1e-8,
                                   max_iter=20000).iterations


def test_additive_schwarz_diverges_as_a_stationary_iteration(square):
    """The precise reason it is a preconditioner and not a solver.

    The additive sum counts the overlap once per covering subdomain, so where
    subdomains meet the correction is applied several times over and the
    iteration matrix has spectral radius above one. Measured: at unit damping
    it reaches ``inf`` within 400 steps. The classical Schwarz *alternating*
    method converges because it is multiplicative -- each subdomain sees the
    previous one's update.
    """
    A, b = square
    operator = build_schwarz(A, n_subdomains=4, overlap=2)

    _, history = operator.solve(b, tol=1e-8, max_iter=400, damping=1.0)
    assert not np.isfinite(history[-1]) or history[-1] > history[0]


def test_damping_by_the_multiplicity_restores_convergence(square):
    """Dividing by the largest number of subdomains covering any one unknown
    removes the double counting exactly. Measured: multiplicity 4, so damping
    0.25, converging in 184 steps where 1.0 diverges and 0.5 stalls."""
    A, b = square
    operator = build_schwarz(A, n_subdomains=4, overlap=2)
    assert operator.multiplicity >= 2

    x, history = operator.solve(b, tol=1e-8, max_iter=600)   # default damping
    assert history[-1] < 1e-6
    np.testing.assert_allclose(x, spla.spsolve(A.tocsc(), b), rtol=1e-4,
                               atol=1e-7)


def test_multiplicity_is_one_without_overlap(square):
    A, _ = square
    assert build_schwarz(A, n_subdomains=6, overlap=0).multiplicity == 1


def test_the_registry_offers_both_variants(square):
    A, _ = square
    assert get_preconditioner("asm", A).is_spd is True
    assert get_preconditioner("ras", A).is_spd is False
    assert get_preconditioner("schwarz", A).name == "schwarz"


def test_the_operator_reports_its_balance_and_overlap(square):
    A, _ = square
    M = AdditiveSchwarz(n_subdomains=8, overlap=1).setup(A)

    assert M.operator.n_subdomains == 8
    assert 1.0 < M.overlap_ratio < 2.0
    assert 1.0 <= M.operator.balance < 2.5
    assert M.operator.largest_subdomain < A.shape[0]
    assert "subdomains" in repr(M.operator)
