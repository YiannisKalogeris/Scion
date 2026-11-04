"""A real micropolar MPM solve driven through the Rust scatter.

``test_rust_scatter.py`` checks the kernel in isolation. This is the check the
backend's licence actually rests on, and it is a different claim: a per-kernel
comparison cannot catch an error that moves the load path, because every
Newton iterate feeds the next and the stencil is rebuilt from the state the
previous step left. The same argument was what licensed ``KrylovSolver`` --
the per-tangent residual agreed long before the end-to-end peak torques did.

Equality is asserted to the bit. The scatter sums each node's contributions in
``np.add.at``'s own order, so switching backend is supposed to change nothing
whatsoever, and a tolerance here would hide exactly the drift that matters.
"""
import numpy as np
import pytest

import scion.discretization.mpm.rust as rust
from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
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


def block(dx=0.5, ppc=2, lo=(0.5, 0.5, 0.5), hi=(2.0, 1.5, 1.5), shape=(7, 6, 6)):
    grid = CartesianGrid(shape=shape, dx=dx, basis=get_basis("linear"))
    sub = dx / ppc
    axes = [np.arange(lo[k] + sub / 2, hi[k], sub) for k in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    return grid, attach_micropolar_state(
        ParticleSet(pts, vol0=sub ** 3, density=1.0))


def solve(n_steps=2):
    """Clamp one end, drive the other, and return everything the step moved."""
    grid, parts = block()
    bcs = [GridDirichlet(lambda c: c[:, 0] < 0.5 + 1e-9, np.zeros(6)),
           GridDirichlet(lambda c: c[:, 0] > 2.0 - 1e-9,
                         np.array([0.08, 0.04, 0.0, 0.0, 0.0, 0.02]))]
    stepper = MicropolarMPMStepper(grid, MATERIAL, dirichlet=bcs)
    stepper.run(parts, n_steps=n_steps)
    return parts, stepper


def test_the_assembly_really_goes_through_the_extension():
    """Guard against a test that silently compares NumPy with NumPy."""
    _, stepper = solve(n_steps=1)
    assert stepper._scatter is not None
    assert stepper._scatter._inner is not None, "fell back without being asked"


def test_solve_is_bit_identical_to_the_numpy_path(monkeypatch):
    parts_rs, step_rs = solve()

    # Scatter reads the module global at construction, so this reaches the
    # instance the stepper builds inside prepare()
    monkeypatch.setattr(rust, "_rs", None)
    parts_np, step_np = solve()
    assert step_np._scatter._inner is None, "fallback did not take effect"

    for name in ("x", "F", "quat", "wryness"):
        a, b = getattr(parts_rs, name), getattr(parts_np, name)
        assert np.array_equal(a, b), f"{name} differs between backends"


def test_the_newton_path_itself_is_identical(monkeypatch):
    """Not just the answer: the same iterates, so the residuals match too.

    A backend that changed the assembly slightly could still converge to the
    same state from a different path, which is a weaker property than the one
    the bit-identical scatter is supposed to buy.
    """
    _, step_rs = solve()
    monkeypatch.setattr(rust, "_rs", None)
    _, step_np = solve()

    assert len(step_rs.history) == len(step_np.history)
    for r, n in zip(step_rs.history, step_np.history):
        assert r.converged and n.converged
        assert r.iterations == n.iterations
        assert np.array_equal(np.asarray(r.residuals), np.asarray(n.residuals))


def test_internal_force_matches_on_the_deformed_state(monkeypatch):
    """The scatter is in internal_force; compare it away from the rest state,
    where every term of the assembly is nonzero."""
    parts, stepper = solve()
    f_rs = stepper.internal_force(parts)

    # Rebuild ONLY the scatter. Calling prepare() again would also reset
    # ``_grad``, which Newton advances every iteration, so the two forces
    # would be evaluated on different states and the test would fail for a
    # reason that has nothing to do with the backend.
    monkeypatch.setattr(rust, "_rs", None)
    stepper._scatter = rust.Scatter(stepper._st.node_id, stepper._n_nodes_eff)
    assert stepper._scatter._inner is None
    f_np = stepper.internal_force(parts)

    assert np.array_equal(f_rs, f_np)
    assert np.abs(f_rs).max() > 0.0              # not a comparison of zeros


# ----------------------------------------------------------------------
class TestChunkedAssembly:
    """The tangent must not depend on how it was chunked.

    The guarantee is numerical, not structural, and the distinction is real:
    whether a cancelling entry lands on exactly 0.0 (and is pruned) or on
    1e-20 (and is kept) depends on which particles were summed together, so
    the stored-entry count genuinely moves with the chunk size -- 10,817
    against 9,932 on this block. That predates the merge tree; the old
    `K = K + chunk` loop pruned at every addition for the same reason. What
    must not move is the operator itself.

    The cached pattern narrowed this without closing it: the *pattern* is now
    fixed, and the accumulation into it is chunk-independent to the bit, so
    what is left is `optimize=True` choosing its contraction order from the
    operand shapes -- the per-particle blocks themselves move by ~2e-16 when
    the chunk size changes. Hence a tolerance here and not equality.
    """

    def _tangent(self, chunk_entries):
        grid, parts = block()
        st = MicropolarMPMStepper(grid, MATERIAL, chunk_entries=chunk_entries)
        st.prepare(parts)
        return st.tangent(parts)

    def test_one_chunk_matches_many(self):
        big = self._tangent(4_000_000)          # a single chunk for this size
        small = self._tangent(4_000)            # forces dozens of chunks
        assert big.shape == small.shape
        assert abs(big - small).max() / abs(big).max() < 1e-13

    def test_a_single_particle_per_chunk_still_assembles(self):
        """The degenerate end of the range: every chunk is one particle, so
        the merge tree runs at its deepest."""
        one = self._tangent(1)                  # chunk clamps to 1 particle
        many = self._tangent(4_000_000)
        assert abs(one - many).max() / abs(many).max() < 1e-13
