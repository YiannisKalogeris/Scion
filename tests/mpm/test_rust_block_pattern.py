"""The cached tangent pattern against the COO path it replaces.

``node_id`` is fixed for a whole load step while the tangent is assembled once
per Newton iteration, so `BlockPattern` computes the sparsity pattern once and
the assembly becomes an accumulation into a fixed ``data`` array.

Two claims, and they are not the same one. The *accumulation* is bit-identical
to a flat ascending COO sum, because each node is owned by one thread and
walked in ascending ``p * npe + K`` order -- the order `coo_to_csr` sums a
``(row, col)`` group in. The *assembled tangent* is only close to the old
path's, and the reason is upstream of this code: `optimize=True` picks its
contraction order from the operand shapes, so the per-particle blocks
themselves move in the last bits when the chunk size changes. Asserting bit
equality on the whole tangent would therefore be testing the einsum path
search, not this kernel.
"""
import numpy as np
import pytest
from scipy import sparse

import scion.discretization.mpm.rust as rust
from scion.discretization.mpm.rust import BlockPattern
from scion.discretization.mpm import (
    CartesianGrid,
    ParticleSet,
    get_basis,
)
from scion.discretization.mpm.micropolar import (
    MicropolarMPMStepper,
    attach_micropolar_state,
)
from scion.materials.micropolar import MicropolarHooke

pytestmark = pytest.mark.skipif(not rust.available(),
                                reason="scion_rust extension not built")

MATERIAL = MicropolarHooke(lam=3.333e6, mu=5.0e6, a=1.0 / 3.0, alpha=0.0,
                           beta=1.0e5, gam=1.0e5)


def numpy_pattern(node_id, n_nodes, nc):
    """The same class with the extension taken away: the reference."""
    p = BlockPattern.__new__(BlockPattern)
    p.node_id = np.ascontiguousarray(node_id, dtype=np.int64)
    p.n_particles, p.npe = p.node_id.shape
    p.n_nodes, p.nc = int(n_nodes), int(nc)
    p.n_dof = p.n_nodes * p.nc
    p._inner = None
    p._indptr = p._indices = None
    p._build_numpy()
    return p


def blocks_for(rng, n_p, npe, nc):
    # several magnitudes per particle, so a changed summation order shows up
    # in the last bits instead of cancelling
    return (rng.standard_normal((n_p, npe, nc, npe, nc))
            * 10.0 ** rng.integers(-4, 4, size=(n_p, 1, 1, 1, 1)))


@pytest.mark.parametrize("n_p,npe,n_nodes,nc",
                         [(400, 8, 97, 6),
                          (200, 27, 203, 6),
                          (120, 4, 41, 3),
                          (64, 8, 23, 1)])
def test_matches_the_numpy_reference_bit_for_bit(n_p, npe, n_nodes, nc):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, n_nodes, size=(n_p, npe)).astype(np.int64)
    blk = blocks_for(rng, n_p, npe, nc)

    fast = BlockPattern(ids, n_nodes, nc)
    ref = numpy_pattern(ids, n_nodes, nc)
    assert fast._inner is not None, "fell back without being asked"

    a = np.zeros(fast.nnz)
    fast.accumulate(a, blk, 0, n_p)
    b = np.zeros(ref.nnz)
    ref.accumulate(b, blk, 0, n_p)

    assert np.array_equal(fast.indptr, ref.indptr)
    assert np.array_equal(fast.indices, ref.indices)
    assert np.array_equal(a, b), "must be bit-identical, not merely close"


def test_pattern_agrees_with_scipys_coo_conversion():
    """The pattern has to be the one the triplets would have produced, or the
    accumulation is writing into the wrong places with no symptom but a wrong
    answer."""
    rng = np.random.default_rng(1)
    n_p, npe, n_nodes, nc = 300, 8, 61, 6
    ids = rng.integers(0, n_nodes, size=(n_p, npe)).astype(np.int64)
    blk = blocks_for(rng, n_p, npe, nc)

    pat = BlockPattern(ids, n_nodes, nc)
    data = np.zeros(pat.nnz)
    pat.accumulate(data, blk, 0, n_p)
    got = sparse.csr_matrix((data, pat.indices, pat.indptr),
                            shape=(pat.n_dof,) * 2)

    dofs = (ids[:, :, None] * nc + np.arange(nc)).reshape(n_p, -1)
    n_loc = dofs.shape[1]
    want = sparse.coo_matrix(
        (blk.ravel(), (np.repeat(dofs, n_loc, axis=1).ravel(),
                       np.tile(dofs, (1, n_loc)).ravel())),
        shape=(pat.n_dof,) * 2).tocsr()

    assert got.nnz >= want.nnz                   # unpruned, so never fewer
    assert abs(got - want).max() <= 1e-15 * abs(want).max()


@pytest.mark.parametrize("chunk", [1, 7, 64, 299, 300])
def test_accumulation_is_chunk_independent_to_the_bit(chunk):
    """The property the old merge tree could not offer. The tree added partial
    sums pairwise, so the answer moved with ``chunk_entries``; here the
    destination of an entry is a function of the pattern alone."""
    rng = np.random.default_rng(2)
    n_p, npe, n_nodes, nc = 300, 8, 61, 6
    ids = rng.integers(0, n_nodes, size=(n_p, npe)).astype(np.int64)
    blk = blocks_for(rng, n_p, npe, nc)
    pat = BlockPattern(ids, n_nodes, nc)

    whole = np.zeros(pat.nnz)
    pat.accumulate(whole, blk, 0, n_p)

    got = np.zeros(pat.nnz)
    for lo in range(0, n_p, chunk):
        hi = min(lo + chunk, n_p)
        pat.accumulate(got, blk[lo:hi], lo, hi)

    assert np.array_equal(got, whole)


def test_a_duplicated_stencil_node_lands_twice():
    """CPDI concatenates per-corner stencils, so one node can occupy two slots
    of a particle and both copies have to be summed. Deduplicating the slots
    would lose the second, with no shape error to say so."""
    ids = np.array([[0, 1, 1, 2]], dtype=np.int64)
    blk = np.arange(16.0).reshape(1, 4, 1, 4, 1)
    pat = BlockPattern(ids, 3, 1)
    data = np.zeros(pat.nnz)
    pat.accumulate(data, blk, 0, 1)
    K = sparse.csr_matrix((data, pat.indices, pat.indptr), shape=(3, 3))
    # node 1 occupies slots 1 and 2, so K[1, 1] collects four of the sixteen
    assert K[1, 1] == blk[0, 1, 0, 1, 0] + blk[0, 1, 0, 2, 0] \
                      + blk[0, 2, 0, 1, 0] + blk[0, 2, 0, 2, 0]
    assert K.sum() == blk.sum()


def test_unsupported_nodes_give_empty_rows():
    """An MPM grid is larger than the body, and a node no particle touches
    must contribute no row -- those are the identically zero rows that made
    the phase-field factorisation singular."""
    ids = np.array([[3, 4]], dtype=np.int64)
    pat = BlockPattern(ids, 6, 2)
    lengths = np.diff(pat.indptr)
    assert lengths[[0, 1, 2, 3, 10, 11]].tolist() == [0, 0, 0, 0, 0, 0]
    assert pat.nnz == (2 * 2) ** 2


def test_matrix_prunes_the_structural_zeros_without_losing_the_pattern():
    """`matrix` must not compact the cached arrays in place: the pattern is
    reused for every later Newton iteration of the same load step."""
    ids = np.array([[0, 1], [1, 2]], dtype=np.int64)
    pat = BlockPattern(ids, 3, 2)
    before = pat.nnz
    data = np.zeros(pat.nnz)
    data[0] = 1.0
    K = pat.matrix(data)
    assert K.nnz == 1
    assert pat.nnz == before
    assert pat.indptr[-1] == before and pat.indices.size == before


def test_rejects_a_bad_stencil():
    with pytest.raises(ValueError):
        BlockPattern(np.array([0, 1], dtype=np.int64), 4, 6)
    with pytest.raises(ValueError):
        BlockPattern(np.array([[0, 9]], dtype=np.int64), 4, 6)


# ----------------------------------------------------------------------
def notched_block(dx=0.5, ppc=2):
    grid = CartesianGrid(shape=(7, 6, 6), dx=dx, basis=get_basis("linear"))
    sub = dx / ppc
    axes = [np.arange(lo + sub / 2, hi, sub)
            for lo, hi in ((0.5, 2.0), (0.5, 1.5), (0.5, 1.5))]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    return grid, attach_micropolar_state(
        ParticleSet(pts, vol0=sub ** 3, density=1.0))


class TestOnARealTangent:
    """The claim the change rests on: the assembled operator is the same one.

    Taken on a *deformed* state. At ``F = I`` whole terms of the tangent
    vanish, so a destination computed into the wrong slot could land on a zero
    and the comparison would pass.
    """

    def _stepper(self, chunk_entries=4_000_000):
        grid, parts = notched_block()
        st = MicropolarMPMStepper(grid, MATERIAL, chunk_entries=chunk_entries)
        st.prepare(parts)
        rng = np.random.default_rng(5)
        st.apply_increment(parts, 1e-3 * rng.standard_normal(st._n_dof))
        return st, parts

    def _coo_tangent(self, st, parts):
        """The path this replaces, reproduced here rather than kept in the
        module: triplets per chunk, a CSR per chunk, added up."""
        c = st.constitutive(parts)
        n, k = st._st.node_id.shape
        chunk = max(1, int(st.chunk_entries // (st.n_components * k) ** 2))
        K = None
        for lo in range(0, n, chunk):
            part = st._tangent_chunk_general(slice(lo, min(lo + chunk, n)),
                                             c, parts)
            K = part if K is None else K + part
        K.eliminate_zeros()
        return K if st._ghost is None else K + st._ghost

    def test_the_tangent_really_goes_through_the_pattern(self):
        """Guard against a test that compares the COO path with itself."""
        st, parts = self._stepper()
        st.tangent(parts)
        assert st._pattern is not None
        assert st._pattern._inner is not None

    def test_agrees_with_the_coo_path(self):
        st, parts = self._stepper()
        new = st.tangent(parts)
        old = self._coo_tangent(st, parts)
        assert new.shape == old.shape
        assert abs(new - old).max() <= 1e-13 * abs(old).max()
        assert abs(old).max() > 0.0              # not a comparison of zeros

    def test_is_bit_identical_to_the_native_coo_path(self, monkeypatch):
        """The sharp version of the comparison above.

        SciPy's per-row sort is unstable, so its summation order inside a
        ``(row, col)`` group is not reproducible and the default COO path can
        only be matched to an ulp. The native kernel sums ascending, which is
        this path's order exactly, so against *that* reference the agreement
        is to the bit -- at one chunk, which is what removes the merge tree
        and the einsum path search from the comparison.
        """
        monkeypatch.setattr(rust, "_USE_NATIVE_CSR", True)
        st, parts = self._stepper(chunk_entries=1 << 40)
        new = st.tangent(parts)
        old = self._coo_tangent(st, parts)
        assert np.array_equal(new.indptr, old.indptr)
        assert np.array_equal(new.indices, old.indices)
        assert np.array_equal(new.data, old.data)

    def test_prunes_to_the_same_sparsity_as_the_coo_path(self):
        st, parts = self._stepper(chunk_entries=1 << 40)
        new = st.tangent(parts)
        old = self._coo_tangent(st, parts)
        assert new.nnz == old.nnz
        assert np.array_equal(new.indptr, old.indptr)

    def test_the_pattern_is_rebuilt_when_the_stencil_is(self):
        """`prepare` is the only place ``node_id`` changes, so it is the only
        place the cache may be dropped -- and it must be, or a moved body
        assembles into the previous step's pattern."""
        st, parts = self._stepper()
        st.tangent(parts)
        first = st._pattern
        st.prepare(parts)
        assert st._pattern is None
        st.tangent(parts)
        assert st._pattern is not None and st._pattern is not first

    def test_the_ghost_penalty_still_reaches_the_tangent(self):
        """The penalty is added after the pattern's own entries, and it must
        be: a penalty in the residual and not the tangent diverges on the
        first iterate and reports as inverted particles."""
        grid, parts = notched_block()
        bare = MicropolarMPMStepper(grid, MATERIAL)
        bare.prepare(parts)
        k0 = bare.tangent(parts)
        grid2, parts2 = notched_block()
        stab = MicropolarMPMStepper(grid2, MATERIAL, ghost=10.0)
        stab.prepare(parts2)
        k1 = stab.tangent(parts2)
        assert abs(k1 - k0).max() > 0.0
