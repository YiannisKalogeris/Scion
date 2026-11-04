"""The phase-field system on an iterative solver instead of a factorisation.

The direct solve is ``N^2.46`` on a 3-D MPM operator and is the wall at
production resolution, so these pin the two claims that license replacing it:
the isolated-unknown split is *exact*, and the Krylov path moves a converged
result only in the last digits.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as spla

from scion.multiphysics.phase_field_fracture import (isolated_unknowns,
                                                     solve_phase_field_system)
from scion.solvers.linear import KrylovSolver

from test_phase_field_mpm import notched_micropolar


def relative(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def pcg():
    return KrylovSolver(method="pcg", preconditioner="jacobi", tol=1e-10)


def laplacian_1d(n, shift=1.0):
    """A scalar SPD reaction-diffusion operator, the shape of the real one."""
    A = sparse.diags([-1.0, 2.0 + shift, -1.0], [-1, 0, 1],
                     shape=(n, n), format="csr")
    return A.tocsr()


def embed_isolated(A, where):
    """Grow ``A`` by unit-diagonal rows that couple to nothing, as the MPM
    Laplacian does for a node with no particle support."""
    n = A.shape[0] + len(where)
    keep = np.setdiff1d(np.arange(n), where)
    B = sparse.lil_matrix((n, n))
    B[np.ix_(keep, keep)] = A.toarray()
    for i in where:
        B[i, i] = 1.0
    return B.tocsr(), keep


class TestTheSolverSeam:
    def test_none_is_the_direct_solve(self):
        A = laplacian_1d(40)
        b = np.arange(40, dtype=float)
        assert np.array_equal(solve_phase_field_system(A, b, None),
                              spla.spsolve(A.tocsc(), b))

    def test_krylov_agrees_with_the_factorisation(self):
        A = laplacian_1d(200)
        b = np.cos(np.arange(200.0))
        x = solve_phase_field_system(A, b, pcg())
        assert relative(x, spla.spsolve(A.tocsc(), b)) <= 1e-9

    def test_a_matrix_with_no_isolated_rows_is_passed_through(self):
        """The mask must not cost a copy where there is nothing to mask."""
        A = laplacian_1d(30)
        seen = {}

        class Recording(KrylovSolver):
            def solve(self, M, b):
                seen["n"] = M.shape[0]
                return super().solve(M, b)

        solve_phase_field_system(A, np.ones(30),
                                 Recording(method="pcg", tol=1e-10))
        assert seen["n"] == 30


class TestItRefusesRatherThanReturningAnIterate:
    """The whole argument for `KrylovSolver` over scipy's wrappers: a damage
    increment that missed its tolerance must not reach the stagger loop, which
    would re-converge around it and report success."""

    def test_an_unconverged_solve_raises(self):
        A = laplacian_1d(400, shift=0.0)
        with pytest.raises(RuntimeError, match="did not reach"):
            solve_phase_field_system(
                A, np.cos(np.arange(400.0)),
                KrylovSolver(method="pcg", tol=1e-14, max_iter=3))

    def test_an_indefinite_operator_is_named(self):
        """``dr = c alpha'' + g'' H`` is strictly positive, so this cannot
        happen -- but if it ever does, PCG says which quantity went wrong
        instead of dividing by a negative curvature."""
        from scion.solvers.krylov import NotPositiveDefinite
        A = sparse.diags([1.0, -3.0, 1.0], [-1, 0, 1],
                         shape=(20, 20), format="csr")
        with pytest.raises(NotPositiveDefinite):
            solve_phase_field_system(A, np.ones(20),
                                     KrylovSolver(method="pcg", tol=1e-10))


class TestIsolatedUnknowns:
    """A node with no particle support carries a unit diagonal and nothing
    else, and taking it out of the Krylov space has to be exact rather than
    nearly so -- it is 74% of the rows on a real torsion grid."""

    where = np.array([0, 5, 17, 31])

    def test_the_split_reproduces_the_full_direct_solve(self):
        A, keep = embed_isolated(laplacian_1d(40), self.where)
        b = np.sin(np.arange(float(A.shape[0])))
        x = solve_phase_field_system(A, b, pcg())
        # on the norm, not componentwise: a Krylov solve stops on ||r||, so a
        # component small against ||x|| carries no relative accuracy and a
        # componentwise bound is a statement about the test, not the code
        assert relative(x, spla.spsolve(A.tocsc(), b)) <= 1e-9

    def test_isolated_entries_are_exact_not_iterated(self):
        A, keep = embed_isolated(laplacian_1d(40), self.where)
        b = np.sin(np.arange(float(A.shape[0])))
        x = solve_phase_field_system(A, b, pcg())
        # one equation in one unknown, so there is no tolerance to allow
        assert np.array_equal(x[self.where], b[self.where])

    def test_only_the_coupled_block_reaches_the_krylov_solver(self):
        A, keep = embed_isolated(laplacian_1d(40), self.where)
        seen = {}

        class Recording(KrylovSolver):
            def solve(self, M, b):
                seen["n"] = M.shape[0]
                return super().solve(M, b)

        solve_phase_field_system(A, np.ones(A.shape[0]),
                                 Recording(method="pcg", tol=1e-10))
        assert seen["n"] == 40

    def test_it_finds_exactly_the_embedded_rows(self):
        A, keep = embed_isolated(laplacian_1d(40), self.where)
        assert np.array_equal(np.flatnonzero(isolated_unknowns(A)), self.where)

    def test_a_lone_off_diagonal_row_is_not_one(self):
        """``nnz == 1`` alone is not the test: the entry has to BE the
        diagonal, or the row is split off and divided by a structural zero."""
        A = laplacian_1d(6).tolil()
        A[2, :] = 0.0
        A[2, 4] = 3.0          # one stored entry, off the diagonal
        assert not isolated_unknowns(A.tocsr())[2]

    def test_a_row_lone_but_coupled_by_its_column_is_not_one(self):
        """Row ``i`` holding only its diagonal is not enough -- if column
        ``i`` has entries the unknown still appears in the free block, so
        removing its equation would leave the block singular."""
        A = laplacian_1d(6).tolil()
        A[2, :] = 0.0
        A[2, 2] = 1.0          # the row is lone, the column is not
        A = A.tocsr()
        assert A[:, 2].nnz > 1
        assert not isolated_unknowns(A)[2]


class TestOnACoupledLoadStep:
    """The check that licenses the change: an inexact inner solve must move a
    *reported* quantity only in the last digits, not merely the increment it
    returns."""

    @staticmethod
    def one_step(solver):
        pf, st, parts = notched_micropolar()
        pf.damage_solver = solver
        pf.solve_step(1.0, 1.0)
        f = st.internal_force(parts)
        return {"d_max": float(pf.d.max()), "d_sum": float(pf.d.sum()),
                "H_max": float(pf.H.max()),
                "F_fro": float(np.linalg.norm(parts.F)),
                "reaction": float(np.linalg.norm(f))}

    def test_krylov_reproduces_the_direct_damage_solve(self):
        new, old = self.one_step(pcg()), self.one_step(None)
        # the damage maximum and the history are pinned by the clip and the
        # max, so they come out exactly equal; the sums and the reaction carry
        # the round-off of a different path to the same converged field
        assert new["d_max"] == old["d_max"]
        assert new["H_max"] == old["H_max"]
        assert new["F_fro"] == old["F_fro"]
        for k in ("d_sum", "reaction"):
            assert abs(new[k] - old[k]) <= 1e-10 * abs(old[k]), k

    def test_the_default_is_still_the_direct_solve(self):
        pf, _, _ = notched_micropolar()
        assert pf.damage_solver is None


class TestTheGradientOperatorCache:
    def test_it_is_built_once_and_reused(self):
        pf, _, _ = notched_micropolar()
        assert pf._K_grad is None
        pf.solve_damage()
        first = pf._K_grad
        assert first is not None
        pf.solve_damage()
        assert pf._K_grad is first

    def test_it_matches_a_freshly_assembled_operator(self):
        """Safe only because the tables are built at the reference
        positions and never refreshed. If that changes, this fails."""
        pf, _, _ = notched_micropolar()
        pf.solve_damage()
        fresh = pf.tables.laplacian(pf.model.gradient_coefficient(),
                                    pf.n_nodes).tolil()
        fresh[pf._unsupported, pf._unsupported] = 1.0
        assert (pf._K_grad != fresh.tocsr()).nnz == 0
