"""
Material-manifold r-adaptivity.

Two things are asserted here that matter more than the descent working.

The first is that ``G_p`` really is ``-dPi_h/dX_p``: checked by
finite-differencing the discrete energy in each particle's reference
coordinate. That is a far stronger statement than matching an analytic
displacement field, because it would catch a missing term in Eq. (9).

The second is that the descent, with particle volumes held fixed, *degrades the
quadrature while reducing the energy monotonically*. That is not a bug and the
test asserts it deliberately: it is the measured evidence that the ``dV^0/dX``
term is load-bearing rather than a refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.discretization.mpm import CartesianGrid
from scion.discretization.mpm.basis import get_basis
from scion.materials.hencky import (
    DruckerPragerSand, HenckyElastic, PrincipalPlasticity,
)
from scion.materials.hyperelastic_batched import NeoHookeanCompressible
from projects.configurational_mpm import (
    configurational_forces, discrete_energy, relax_reference_positions,
)

N = 16
DX = 1.0 / N


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def scene(basis="quadratic", jitter=0.0, rng=None):
    grid = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis(basis))
    s = (np.arange(2) + 0.5) / 2
    ax = ((np.arange(4, 12)[:, None] + s[None, :]).ravel()) * DX
    Xg, Yg = np.meshgrid(ax, ax, indexing="ij")
    X = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
    if jitter:
        X = X + rng.uniform(-jitter, jitter, X.shape) * DX
    vol = np.full(X.shape[0], (DX / 2) ** 2)

    Xn = grid.node_coordinates()
    u = np.stack([0.05 * np.sin(2 * np.pi * Xn[:, 0]) * Xn[:, 1],
                  0.03 * np.cos(2 * np.pi * Xn[:, 1]) * Xn[:, 0]], axis=1)
    return grid, X, u, vol


def material():
    return NeoHookeanCompressible.from_E_nu(1.0e4, 0.3)


class TestShapeGradient:
    @pytest.mark.parametrize("basis", ["quadratic", "cubic"])
    def test_G_is_the_finite_differenced_energy_gradient(self, basis, rng):
        """The definitional check: G_p = -dPi_h/dX_p."""
        grid, X, u, vol = scene(basis, jitter=0.2, rng=rng)
        mat = material()
        G = configurational_forces(mat, grid, X, u, vol)

        h = 1e-7
        idx = rng.choice(X.shape[0], 10, replace=False)
        for a, p in enumerate(idx):
            for k in range(2):
                Xp, Xm = X.copy(), X.copy()
                Xp[p, k] += h
                Xm[p, k] -= h
                dPi = (discrete_energy(mat, grid, Xp, u, vol)
                       - discrete_energy(mat, grid, Xm, u, vol)) / (2 * h)
                assert np.isclose(G[p, k], -dPi, rtol=1e-5,
                                  atol=1e-5 * np.abs(G).max())

    def test_vanishes_for_a_uniform_deformation(self):
        grid, X, _, vol = scene()
        A = np.array([[0.05, 0.0], [0.0, -0.02]])
        u = grid.node_coordinates() @ A.T
        assert np.abs(configurational_forces(material(), grid, X, u, vol)).max() \
            < 1e-12

    def test_energy_is_the_volume_weighted_strain_energy(self):
        grid, X, u, vol = scene()
        from projects.configurational_mpm.adaptivity import _kinematics
        _, F = _kinematics(grid, X, u, None)
        assert np.isclose(discrete_energy(material(), grid, X, u, vol),
                          float((vol * material().energy(F)).sum()))


class TestDescent:
    def test_energy_decreases_monotonically(self, rng):
        """Guaranteed by the line search -- a rise would mean G is not the
        gradient it claims to be."""
        grid, X, u, vol = scene(jitter=0.25, rng=rng)
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=40)
        assert all(b <= a + 1e-14 for a, b in zip(r.energy, r.energy[1:]))
        assert r.energy_reduction > 0.0

    def test_reports_its_history(self, rng):
        grid, X, u, vol = scene(jitter=0.2, rng=rng)
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=10)
        assert len(r.energy) == len(r.separation_cv)
        assert r.n_iterations <= 10
        assert "Pi" in repr(r)

    def test_particles_stay_inside_the_grid(self, rng):
        grid, X, u, vol = scene(jitter=0.25, rng=rng)
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=40,
                                      trust_radius=0.3)
        lo = grid.origin
        hi = lo + np.asarray(grid.shape) * grid.dx
        assert np.all(r.X > lo) and np.all(r.X < hi)

    def test_barrier_stops_particles_merging(self, rng):
        from scipy.spatial import cKDTree
        grid, X, u, vol = scene(jitter=0.25, rng=rng)
        floor = 0.25 * cKDTree(X).query(X, k=2)[0][:, 1].min()
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=60,
                                      trust_radius=0.3)
        after = cKDTree(r.X).query(r.X, k=2)[0][:, 1].min()
        assert after >= floor * (1 - 1e-9)

    def test_zero_gradient_state_converges_immediately(self):
        grid, X, _, vol = scene()
        A = np.array([[0.05, 0.0], [0.0, -0.02]])
        u = grid.node_coordinates() @ A.T
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=20)
        assert r.converged
        assert np.allclose(r.X, X)


class TestFixedVolumeDegeneracy:
    """Asserted on purpose: with fixed V^0 the descent worsens the quadrature.

    ``Pi_h`` approximates ``int W dV`` only while the volumes tessellate the
    body. Held fixed, minimising it just migrates particles to low strain
    energy. This is the evidence that Eq. (9)'s second term is load-bearing.
    """

    def reference_integral(self, grid, u):
        from projects.configurational_mpm.adaptivity import _kinematics
        s = (np.arange(8) + 0.5) / 8
        ax = ((np.arange(4, 12)[:, None] + s[None, :]).ravel()) * DX
        FX, FY = np.meshgrid(ax, ax, indexing="ij")
        Xf = np.stack([FX.ravel(), FY.ravel()], axis=1)
        _, F = _kinematics(grid, Xf, u, None)
        return float((material().energy(F) * (DX / 8) ** 2).sum())

    def test_uniform_lattice_is_already_an_excellent_quadrature(self):
        grid, X, u, vol = scene()
        ref = self.reference_integral(grid, u)
        got = discrete_energy(material(), grid, X, u, vol)
        assert abs(got - ref) / ref < 0.01

    def test_descent_moves_the_quadrature_away_from_the_truth(self):
        grid, X, u, vol = scene()
        ref = self.reference_integral(grid, u)
        before = abs(discrete_energy(material(), grid, X, u, vol) - ref) / ref

        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=40)
        after = abs(r.energy[-1] - ref) / ref

        assert after > 10 * before          # markedly worse, not marginally
        assert r.energy[-1] < r.energy[0]   # while the objective fell

    def test_layout_degradation_is_reported(self):
        grid, X, u, vol = scene()
        r = relax_reference_positions(material(), grid, X, u, vol, n_iter=40)
        assert r.separation_cv[0] < 1e-12   # started a perfect lattice
        assert r.layout_degraded
        assert "DEGRADED" in repr(r)


class TestRefusals:
    def test_state_carrying_material_is_refused(self):
        grid, X, u, vol = scene()
        plastic = PrincipalPlasticity(HenckyElastic.from_E_nu(1e4, 0.3),
                                      DruckerPragerSand(30.0))
        with pytest.raises(TypeError, match="carries history"):
            configurational_forces(plastic, grid, X, u, vol)

    def test_refusal_explains_why_rather_than_just_refusing(self):
        grid, X, u, vol = scene()
        plastic = PrincipalPlasticity(HenckyElastic.from_E_nu(1e4, 0.3),
                                      DruckerPragerSand(30.0))
        with pytest.raises(TypeError, match="not unique"):
            relax_reference_positions(plastic, grid, X, u, vol)

    def test_material_without_energy_is_refused(self):
        class NoEnergy:
            def pk1(self, F):
                return F, np.ones(len(F))

        grid, X, u, vol = scene()
        with pytest.raises(TypeError, match="has no energy"):
            configurational_forces(NoEnergy(), grid, X, u, vol)
