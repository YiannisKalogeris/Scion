"""The Rust scatter against NumPy, which is the reference implementation.

The claim under test is stronger than agreement to a tolerance: the Rust
kernel sums each node's contributions in the same order ``np.add.at`` does,
so the two must agree *to the bit*. That is what lets the backend be switched
without weakening the checkpoint test, which asserts a resumed run reproduces
the load path at exactly 0.00e+00.
"""
import numpy as np
import pytest

from scion.discretization.mpm.rust import Scatter, available, scatter_add

pytestmark = pytest.mark.skipif(not available(),
                                reason="scion_rust extension not built")


def stencil(rng, n_p, npe, n_nodes):
    return rng.integers(0, n_nodes, size=(n_p, npe)).astype(np.int64)


@pytest.mark.parametrize("n_p,npe,n_nodes,ncomp,k,col",
                         [(500, 8, 97, 6, 3, 0),
                          (500, 27, 97, 6, 3, 3),
                          (1000, 27, 2003, 3, 3, 0),
                          (64, 4, 17, 1, 1, 0)])
def test_matches_numpy_bit_for_bit(n_p, npe, n_nodes, ncomp, k, col):
    rng = np.random.default_rng(0)
    ids = stencil(rng, n_p, npe, n_nodes)
    # several magnitudes, so a changed summation order would show up rather
    # than cancel; uniform values are the weakest possible input here
    vals = (rng.standard_normal((n_p, npe, k))
            * 10.0 ** rng.integers(-4, 4, size=(n_p, npe, 1)))

    want = np.zeros((n_nodes, ncomp))
    np.add.at(want[:, col:col + k], ids, vals)

    got = np.zeros((n_nodes, ncomp))
    Scatter(ids, n_nodes).add(got, vals, col)

    assert np.array_equal(got, want), "must be bit-identical, not merely close"


def test_accumulates_rather_than_overwrites():
    # the assembly scatters twice into the same array (force, then couple)
    ids = np.array([[0, 2], [2, 1]], dtype=np.int64)
    vals = np.arange(4.0).reshape(2, 2, 1) + 1.0
    out = np.full((3, 2), 10.0)
    Scatter(ids, 3).add(out, vals, 1)
    assert out[:, 1].tolist() == [11.0, 14.0, 15.0]
    assert out[:, 0].tolist() == [10.0, 10.0, 10.0]


def test_two_scatters_into_disjoint_columns():
    """The micropolar internal force writes :3 and 3: of the same array."""
    rng = np.random.default_rng(3)
    ids = stencil(rng, 200, 8, 51)
    f = rng.standard_normal((200, 8, 3))
    c = rng.standard_normal((200, 8, 3))

    want = np.zeros((51, 6))
    np.add.at(want[:, :3], ids, f)
    np.add.at(want[:, 3:], ids, c)

    got = np.zeros((51, 6))
    sc = Scatter(ids, 51)
    sc.add(got, f, 0)
    sc.add(got, c, 3)
    assert np.array_equal(got, want)


def test_reused_transpose_matches_one_shot():
    rng = np.random.default_rng(5)
    ids = stencil(rng, 300, 27, 83)
    vals = rng.standard_normal((300, 27, 3))
    a, b = np.zeros((83, 3)), np.zeros((83, 3))
    Scatter(ids, 83).add(a, vals, 0)
    scatter_add(b, ids, vals, 0)
    assert np.array_equal(a, b)


def test_thread_count_cannot_change_the_answer():
    """Run the same scatter repeatedly: a racy kernel drifts, this must not."""
    rng = np.random.default_rng(7)
    ids = stencil(rng, 20000, 27, 1500)
    vals = rng.standard_normal((20000, 27, 3)) * 1e3
    sc = Scatter(ids, 1500)
    first = np.zeros((1500, 3))
    sc.add(first, vals, 0)
    for _ in range(5):
        again = np.zeros((1500, 3))
        sc.add(again, vals, 0)
        assert np.array_equal(again, first)


def test_rejects_an_out_of_range_node():
    ids = np.array([[0, 99]], dtype=np.int64)
    with pytest.raises(ValueError, match="outside"):
        Scatter(ids, 3)


# ----------------------------------------------------------------------
class TestNativeCsr:
    """The CSR path is opt-in because it is NOT result-neutral.

    It differs from SciPy by about an ulp and keeps entries SciPy prunes as
    exact zeros, so the default must stay on SciPy and the switch must be
    explicit.
    """

    def test_default_is_off(self):
        from scion.discretization.mpm.rust import native_csr_enabled
        assert native_csr_enabled() is False

    def test_off_reproduces_scipy_exactly(self):
        from scipy import sparse
        from scion.discretization.mpm.rust import coo_to_csr
        rng = np.random.default_rng(1)
        r = rng.integers(0, 80, 20000).astype(np.int64)
        c = rng.integers(0, 80, 20000).astype(np.int64)
        v = rng.standard_normal(20000)
        want = sparse.coo_matrix((v, (r, c)), shape=(80, 80)).tocsr()
        got = coo_to_csr(r, c, v, (80, 80))            # default: off
        assert np.array_equal(got.data, want.data)
        assert np.array_equal(got.indices, want.indices)

    def test_on_agrees_with_scipy_to_an_ulp(self):
        from scipy import sparse
        from scion.discretization.mpm.rust import coo_to_csr
        rng = np.random.default_rng(2)
        r = rng.integers(0, 80, 20000).astype(np.int64)
        c = rng.integers(0, 80, 20000).astype(np.int64)
        v = rng.standard_normal(20000)
        want = sparse.coo_matrix((v, (r, c)), shape=(80, 80)).tocsr()
        got = coo_to_csr(r, c, v, (80, 80), native=True)
        assert abs(got - want).max() / abs(want).max() < 1e-15

    def test_native_and_its_fallback_agree_to_the_bit(self, monkeypatch):
        """Switching the extension off must not change the opted-in answer."""
        import scion.discretization.mpm.rust as rust
        rng = np.random.default_rng(3)
        r = rng.integers(0, 60, 15000).astype(np.int64)
        c = rng.integers(0, 60, 15000).astype(np.int64)
        v = rng.standard_normal(15000)
        a = rust.coo_to_csr(r, c, v, (60, 60), native=True)
        monkeypatch.setattr(rust, "_rs", None)
        b = rust.coo_to_csr(r, c, v, (60, 60), native=True)
        assert np.array_equal(a.data, b.data)
        assert np.array_equal(a.indices, b.indices)
