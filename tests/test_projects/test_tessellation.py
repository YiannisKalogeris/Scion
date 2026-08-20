"""
Voronoi particle volumes and the shape derivative Eq. (9) is missing.

The headline assertion is the last class: once volumes are tied to positions,
Eq. (9) is no longer the shape gradient of the discrete energy, and the term it
omits is larger than the whole gradient. Asserted deliberately.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.discretization.mpm import CartesianGrid
from scion.discretization.mpm.basis import get_basis
from scion.materials.hyperelastic_batched import NeoHookeanCompressible
from projects.configurational_mpm import configurational_forces
from projects.configurational_mpm.adaptivity import _kinematics
from projects.configurational_mpm.tessellation import (
    energy_shape_gradient, volume_gradient, voronoi_neighbours,
    voronoi_volumes,
)

LO = np.array([0.0, 0.0])
HI = np.array([1.0, 1.0])


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def lattice(k=5, jitter=0.0, rng=None):
    a = (np.arange(k) + 0.5) / k
    X = np.stack(np.meshgrid(a, a, indexing="ij"), axis=-1).reshape(-1, 2)
    if jitter:
        X = X + rng.uniform(-jitter, jitter, X.shape) / k
    return X


class TestVolumes:
    @pytest.mark.parametrize("jitter", [0.0, 0.2, 0.4])
    def test_areas_partition_the_box_exactly(self, jitter, rng):
        """The check that the mirror-clipping is right: nothing lost, nothing
        double-counted, whatever the layout."""
        V = voronoi_volumes(lattice(6, jitter, rng), LO, HI)
        assert np.isclose(V.sum(), 1.0, atol=1e-12)

    def test_uniform_lattice_gives_equal_areas(self):
        V = voronoi_volumes(lattice(6), LO, HI)
        assert np.allclose(V, V[0])
        assert np.isclose(V[0], 1.0 / 36)

    def test_all_areas_positive(self, rng):
        assert np.all(voronoi_volumes(lattice(6, 0.4, rng), LO, HI) > 0)

    def test_generator_on_the_boundary_is_rejected(self):
        X = np.array([[0.5, 0.5], [0.0, 0.5]])
        with pytest.raises(ValueError, match="strictly inside"):
            voronoi_volumes(X, LO, HI)

    def test_three_dimensions_refused(self):
        with pytest.raises(NotImplementedError, match="2-D only"):
            voronoi_volumes(np.full((4, 3), 0.5), np.zeros(3), np.ones(3))

    def test_clustering_shrinks_the_cells_involved(self):
        X = lattice(5)
        V0 = voronoi_volumes(X, LO, HI)
        X = X.copy()
        X[12] = X[13] - 0.01                      # pull two together
        V1 = voronoi_volumes(X, LO, HI)
        assert V1[12] < V0[12] and V1[13] < V0[13]
        assert np.isclose(V1.sum(), 1.0, atol=1e-12)


class TestNeighbours:
    def test_uniform_lattice_interior_point_has_four_neighbours(self):
        X = lattice(5)
        pairs = voronoi_neighbours(X, LO, HI)
        deg = np.zeros(len(X), dtype=int)
        for p, q in pairs:
            deg[p] += 1
            deg[q] += 1
        centre = 12                                # middle of a 5x5 lattice
        assert deg[centre] == 4

    def test_pairs_reference_original_generators_only(self, rng):
        X = lattice(5, 0.3, rng)
        for p, q in voronoi_neighbours(X, LO, HI):
            assert 0 <= p < len(X) and 0 <= q < len(X)


class TestVolumeGradient:
    def test_matches_a_direct_finite_difference(self, rng):
        X = lattice(4, 0.25, rng)
        dV = volume_gradient(X, LO, HI)
        h = 1e-7
        for q in (0, 5, 11):
            for k in range(2):
                a, b = X.copy(), X.copy()
                a[q, k] += h
                b[q, k] -= h
                ref = (voronoi_volumes(a, LO, HI)
                       - voronoi_volumes(b, LO, HI)) / (2 * h)
                assert np.allclose(dV[:, q, k], ref, atol=1e-6)

    def test_total_volume_is_conserved_by_any_move(self, rng):
        """The areas always sum to the box, so their gradients must sum to 0."""
        dV = volume_gradient(lattice(4, 0.25, rng), LO, HI)
        assert np.abs(dV.sum(axis=0)).max() < 1e-5

    def test_a_move_affects_neighbours_not_just_itself(self, rng):
        """The coupling Eq. (9) leaves out."""
        X = lattice(4, 0.2, rng)
        dV = volume_gradient(X, LO, HI)
        diag = np.abs(np.einsum("ppk->pk", dV)).max()
        off = dV.copy()
        for p in range(len(X)):
            off[p, p, :] = 0.0
        assert np.abs(off).max() > 0.5 * diag


class TestEquationNineIsIncompleteWithTessellatedVolumes:
    """The finding this module was built to establish.

    With ``V^0`` fixed, Eq. (9) is the exact shape gradient -- verified to 1e-8
    in ``test_configurational_adaptivity.py``. Tie the volumes to the positions,
    which is what makes ``Pi_h`` an approximation of the body's energy at all,
    and Eq. (9) is no longer the gradient: the neighbour coupling it omits is
    larger than the whole gradient.
    """

    def setup_problem(self, rng):
        n = 16
        dx = 1.0 / n
        grid = CartesianGrid(shape=(n, n), dx=dx, basis=get_basis("quadratic"))
        s = (np.arange(2) + 0.5) / 2
        ax = ((np.arange(5, 11)[:, None] + s[None, :]).ravel()) * dx
        Xg, Yg = np.meshgrid(ax, ax, indexing="ij")
        X = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
        X = X + rng.uniform(-0.2, 0.2, X.shape) * dx
        lo = np.array([ax.min() - 0.25 * dx] * 2)
        hi = np.array([ax.max() + 0.25 * dx] * 2)

        Xn = grid.node_coordinates()
        u = np.stack([0.05 * np.sin(2 * np.pi * Xn[:, 0]) * Xn[:, 1],
                      0.03 * np.cos(2 * np.pi * Xn[:, 1]) * Xn[:, 0]], axis=1)
        return grid, X, u, lo, hi

    def test_complete_gradient_matches_finite_differences(self, rng):
        grid, X, u, lo, hi = self.setup_problem(rng)
        mat = NeoHookeanCompressible.from_E_nu(1.0e4, 0.3)

        def Pi(Xq):
            _, F = _kinematics(grid, Xq, u, None)
            return float((voronoi_volumes(Xq, lo, hi) * mat.energy(F)).sum())

        V = voronoi_volumes(X, lo, hi)
        _, F = _kinematics(grid, X, u, None)
        G9 = configurational_forces(mat, grid, X, u, V)
        full = -energy_shape_gradient(mat.energy(F), -G9 / V[:, None], V,
                                      volume_gradient(X, lo, hi))

        h = 1e-6
        for p in (3, 40, 100):
            for k in range(2):
                a, b = X.copy(), X.copy()
                a[p, k] += h
                b[p, k] -= h
                fd = (Pi(a) - Pi(b)) / (2 * h)
                assert np.isclose(-full[p, k], fd, rtol=1e-4, atol=1e-5)

    def test_equation_nine_alone_does_not(self, rng):
        """And not marginally -- it is wrong by more than the gradient itself."""
        grid, X, u, lo, hi = self.setup_problem(rng)
        mat = NeoHookeanCompressible.from_E_nu(1.0e4, 0.3)

        def Pi(Xq):
            _, F = _kinematics(grid, Xq, u, None)
            return float((voronoi_volumes(Xq, lo, hi) * mat.energy(F)).sum())

        V = voronoi_volumes(X, lo, hi)
        G9 = configurational_forces(mat, grid, X, u, V)

        h = 1e-6
        errs, scale = [], []
        for p in (3, 40, 100):
            for k in range(2):
                a, b = X.copy(), X.copy()
                a[p, k] += h
                b[p, k] -= h
                fd = (Pi(a) - Pi(b)) / (2 * h)
                errs.append(abs(-G9[p, k] - fd))
                scale.append(abs(fd))
        assert max(errs) > 0.5 * max(scale)
