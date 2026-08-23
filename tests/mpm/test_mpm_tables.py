"""The MPM backend for the two-field quadrature tables.

This is an *adapter*, so the tests are consistency checks against properties the
finite-element tables have by construction. The one that carries real weight is
the strain operator: ``B`` is assembled by hand from the stencil gradients, and
a linear displacement field is the only cheap check that pins both the Voigt
ordering and the engineering-shear convention at once.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from scion.discretization.mpm import CartesianGrid, ParticleSet, get_basis
from scion.discretization.mpm.tables import MPMQuadratureTables


def patch(dx=0.25, ppc=2, L=1.0, pad=2, dim=2):
    sub = dx / ppc
    grid = CartesianGrid(shape=(int(L / dx) + 2 * pad,) * dim, dx=dx,
                         origin=np.full(dim, -pad * dx),
                         basis=get_basis("linear"))
    ax = [np.arange(sub / 2, L, sub)] * dim
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")],
                   axis=1)
    parts = ParticleSet(pts, vol0=sub ** dim, density=1.0)
    return grid, parts, MPMQuadratureTables(grid, parts)


class TestConsistency:
    def test_partition_of_unity_and_volume(self):
        """The weights must reproduce the body's volume exactly, or every
        integrated quantity carries the error."""
        grid, parts, t = patch()
        assert np.abs(t.N[:, 0].sum(axis=1) - 1.0).max() < 1e-14
        assert_allclose(t.integrate(np.ones(t.n_points)), 1.0, rtol=1e-12)

    def test_gradients_sum_to_zero(self):
        grid, parts, t = patch()
        assert np.abs(t.dN[:, 0].sum(axis=1)).max() < 1e-14

    @pytest.mark.parametrize("dim", [2, 3])
    def test_strain_operator_against_a_linear_field(self, dim):
        """``u = A x`` gives ``eps = sym(A)`` exactly.

        The check that validates ``B``. It pins the Voigt row ordering *and*
        the engineering-shear factor together: a wrong ordering shows up in the
        normal components, a factor of two in the shears, and neither survives
        a comparison against ``sym(A)``.
        """
        grid, parts, t = patch(dim=dim)
        rng = np.random.default_rng(0)
        A = 0.3 * rng.standard_normal((dim, dim))
        xy = np.asarray(grid.node_coordinates())[:, :dim]
        u = (xy @ A.T).ravel()
        eps = t.strain_tensors(u)
        # assert_allclose does not broadcast, so the target is expanded rather
        # than compared against a (dim, dim) block
        assert_allclose(eps, np.broadcast_to(0.5 * (A + A.T), eps.shape),
                        atol=1e-13)

    def test_interpolation_and_gradient_are_exact_for_a_linear_scalar(self):
        grid, parts, t = patch()
        xy = np.asarray(grid.node_coordinates())[:, :2]
        c = np.array([2.0, -3.0])
        f = xy @ c + 1.0
        assert_allclose(t.interpolate(f), parts.x[:, :2] @ c + 1.0, atol=1e-13)
        assert_allclose(t.gradient(f), np.broadcast_to(c, (t.n_points, 2)),
                        atol=1e-13)

    def test_laplacian_annihilates_a_constant_and_is_symmetric(self):
        """Both are structural. A Laplacian that does not kill a constant has a
        gradient error; an unsymmetric one cannot come from ``grad u . grad v``
        and would break the phase-field solve's conjugacy."""
        grid, parts, t = patch()
        L = t.laplacian(1.0, grid.n_nodes)
        assert np.abs(L @ np.ones(grid.n_nodes)).max() < 1e-12
        assert abs(L - L.T).max() < 1e-14

    def test_mass_integrates_to_the_volume(self):
        grid, parts, t = patch()
        assert_allclose(t.mass(1.0, grid.n_nodes).sum(), 1.0, rtol=1e-12)

    def test_load_of_a_constant_matches_the_mass_row_sums(self):
        grid, parts, t = patch()
        f = t.load(np.ones(t.n_points), grid.n_nodes)
        assert_allclose(f.sum(), 1.0, rtol=1e-12)


class TestSupport:
    def test_only_nodes_reached_by_particles_are_supported(self):
        """An MPM grid is larger than the body. The count is exact and worth
        asserting: a unit body at ``dx`` has ``(1/dx + 1)^dim`` nodes in
        range, and everything else must be excluded or both field solves are
        singular."""
        grid, parts, t = patch(dx=0.25, L=1.0)
        assert int(t.supported_nodes().sum()) == 5 ** 2
        assert grid.n_nodes > 5 ** 2          # the grid really is bigger

    def test_stiffness_assembles_only_onto_supported_nodes(self):
        grid, parts, t = patch()
        nd = grid.n_nodes * t.dim
        nv = t.n_voigt
        C = np.broadcast_to(np.eye(nv), (t.n_points, nv, nv)).copy()
        s = np.zeros((t.n_points, nv))
        K, f = t.stiffness_and_force(s, C, nd)
        sup = np.repeat(t.supported_nodes(), t.dim)
        assert np.abs(K.tocsr()[~sup].data).sum() == 0.0
        assert np.abs(f[~sup]).max() == 0.0
