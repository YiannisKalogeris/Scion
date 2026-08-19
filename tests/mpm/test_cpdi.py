"""
Convected Particle Domain Interpolation.

CPDI cannot be a ``Basis``: ``CartesianGrid.build_stencil`` is tensor-product,
and a convected parallelogram stops being axis-aligned as soon as ``F`` rotates.
So it is a grid subclass that samples the grid basis at the four domain corners
and concatenates the four resulting stencils.

The four consistency identities below are the real test. They are checked on
*rotated and sheared* domains specifically, because an axis-aligned domain would
pass even if the corner coefficients were wrong -- the axis-aligned case is the
one where CPDI degenerates to something GIMP-like.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC, CartesianGrid, CPDIGrid, ExplicitMPMStepper, ParticleSet,
)
from scion.discretization.mpm.basis import get_basis
from scion.discretization.mpm.cpdi import corner_offsets
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

N, DX = 24, 1.0 / 24
RHO = 1000.0


def rotate(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


DOMAINS = {
    "axis-aligned": np.eye(2),
    "rotated": rotate(0.7),
    "sheared": np.array([[1.3, 0.4], [0.0, 0.8]]),
    "rotated+sheared": rotate(0.7) @ np.array([[1.3, 0.4], [0.0, 0.8]]),
}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def scene(basis, shape_matrix, rng, n_particles=200):
    grid = CPDIGrid(shape=(N, N), dx=DX, basis=get_basis(basis))
    xp = rng.uniform(0.3, 0.7, (n_particles, 2))
    domain = np.tile(0.4 * DX * shape_matrix, (n_particles, 1, 1))
    return grid, xp, grid.build_stencil(xp, domain=domain)


class TestStencilConsistency:
    @pytest.mark.parametrize("shape", list(DOMAINS))
    def test_partition_of_unity(self, shape, rng):
        _, _, st = scene("linear", DOMAINS[shape], rng)
        assert np.abs(st.w.sum(axis=1) - 1.0).max() < 1e-13

    @pytest.mark.parametrize("shape", list(DOMAINS))
    def test_gradients_sum_to_zero(self, shape, rng):
        _, _, st = scene("linear", DOMAINS[shape], rng)
        assert np.abs(st.grad_w.sum(axis=1)).max() < 1e-12

    @pytest.mark.parametrize("shape", list(DOMAINS))
    def test_linear_completeness(self, shape, rng):
        grid, xp, st = scene("linear", DOMAINS[shape], rng)
        Xn = grid.node_coordinates()
        got = np.einsum("pk,pka->pa", st.w, Xn[st.node_id])
        assert np.abs(got - xp).max() < 1e-13

    @pytest.mark.parametrize("shape", list(DOMAINS))
    def test_gradient_reproduces_a_linear_field(self, shape, rng):
        """sum_i grad w_i (x) x_i = I -- the identity Eq. 26 exists to satisfy."""
        grid, _, st = scene("linear", DOMAINS[shape], rng)
        Xn = grid.node_coordinates()
        got = np.einsum("pka,pkb->pab", Xn[st.node_id], st.grad_w)
        assert np.abs(got - np.eye(2)).max() < 1e-12


class TestStencilStructure:
    def test_four_corners_are_concatenated(self, rng):
        grid, _, st = scene("linear", np.eye(2), rng, n_particles=1)
        per_corner = get_basis("linear").stencil_size ** 2
        assert st.node_id.shape[1] == 4 * per_corner

    def test_duplicate_node_indices_are_kept(self, rng):
        """Deliberate: every consumer is a scatter or a gather, both of which
        handle repeats, and deduplicating would cost a sort per step."""
        _, _, st = scene("linear", np.eye(2), rng, n_particles=1)
        assert len(np.unique(st.node_id)) < st.node_id.shape[1]

    def test_dpos_is_measured_from_the_particle_not_the_corner(self, rng):
        """APIC's affine term needs the offset to the particle centre."""
        grid, xp, st = scene("linear", np.eye(2), rng, n_particles=50)
        Xn = grid.node_coordinates()
        assert np.allclose(st.dpos, Xn[st.node_id] - xp[:, None, :], atol=1e-12)

    def test_corner_offsets_are_the_four_sign_combinations(self):
        domain = np.array([[[1.0, 0.0], [0.0, 2.0]]])
        got = corner_offsets(domain)[0]
        expected = np.array([[-1.0, -2.0], [1.0, -2.0], [1.0, 2.0], [-1.0, 2.0]])
        assert np.allclose(got, expected)

    def test_domain_is_required(self):
        grid = CPDIGrid(shape=(N, N), dx=DX, basis=get_basis("linear"))
        with pytest.raises(ValueError, match="convected particle domain"):
            grid.build_stencil(np.array([[0.5, 0.5]]))

    def test_collapsed_domain_is_rejected(self):
        grid = CPDIGrid(shape=(N, N), dx=DX, basis=get_basis("linear"))
        flat = np.array([[[0.001, 0.0], [0.002, 0.0]]])   # parallel -> zero area
        with pytest.raises(ValueError, match="collapsed to zero area"):
            grid.build_stencil(np.array([[0.5, 0.5]]), domain=flat)

    def test_three_dimensions_are_refused(self):
        grid = CPDIGrid(shape=(8, 8, 8), dx=DX, basis=get_basis("linear"))
        with pytest.raises(NotImplementedError, match="2-D"):
            grid.build_stencil(np.zeros((1, 3)), domain=np.zeros((1, 3, 3)))


class TestDomainConvection:
    def test_rotation_is_tracked_and_area_preserved(self):
        """The reason CPDI exists: cpGIMP cannot represent a rotated domain."""
        p = ParticleSet(np.array([[0.5, 0.5]]), vol0=(0.5 * DX) ** 2)
        a0 = np.linalg.det(p.domain[0])
        p.F = rotate(np.pi / 4)[None]
        p.update_domain("cpdi")

        assert np.isclose(np.linalg.det(p.domain[0]), a0)      # area exact
        assert not np.allclose(p.domain[0], np.diag(np.diag(p.domain[0])))

    def test_cpgimp_shrinks_the_domain_under_pure_rotation(self):
        """Contrast: the diagonal of F is not a stretch."""
        p = ParticleSet(np.array([[0.5, 0.5]]), vol0=(0.5 * DX) ** 2)
        p.F = rotate(np.pi / 4)[None]
        p.update_domain("stretch")
        assert np.all(p.lp[0] < 0.75 * p.lp0[0])

    def test_shear_is_tracked(self):
        p = ParticleSet(np.array([[0.5, 0.5]]), vol0=(0.5 * DX) ** 2)
        p.F = np.array([[[1.0, 0.5], [0.0, 1.0]]])
        p.update_domain("cpdi")
        assert p.domain[0, 1, 0] != 0.0                        # r2 gained x

    def test_unknown_mode_still_rejected(self):
        p = ParticleSet(np.array([[0.5, 0.5]]), vol0=1e-4)
        with pytest.raises(ValueError, match="Unknown domain update"):
            p.update_domain("nonsense")


def disk(cls, basis, spin=3.0, drift=0.4):
    grid = cls(shape=(48, 48), dx=1.0 / 48, basis=get_basis(basis))
    p = ParticleSet.from_sampler(
        lambda q: ((q[:, 0] - 0.4) ** 2 + (q[:, 1] - 0.5) ** 2) < 0.12 ** 2,
        ((0.0, 0.0), (1.0, 1.0)), 1.0 / 48, ppc=2, density=RHO)
    d = p.x - np.array([0.4, 0.5])
    p.v[:, 0] = -spin * d[:, 1] + drift
    p.v[:, 1] = spin * d[:, 0]
    mat = NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)
    c = mat.wave_speed_factor / np.sqrt(RHO)
    return p, ExplicitMPMStepper(grid, mat, APIC(), dt=0.25 / 48 / c)


class TestSimulation:
    def test_runs_and_conserves_momentum(self):
        p, s = disk(CPDIGrid, "linear")
        m = p.mass
        p0 = (m[:, None] * p.v).sum(axis=0)
        with quiet():
            s.run(p, n_steps=300)
        drift = np.linalg.norm((m[:, None] * p.v).sum(axis=0) - p0) \
            / np.linalg.norm(p0)
        assert drift < 1e-12
        assert np.all(np.isfinite(p.F))

    def test_the_domain_actually_convects_during_a_run(self):
        p, s = disk(CPDIGrid, "linear")
        before = p.domain.copy()
        with quiet():
            s.run(p, n_steps=200)
        assert not np.allclose(p.domain, before)

    def test_survives_a_rotation_that_NaNs_the_linear_hat(self):
        """The linear hat is a documented debug basis and fails here at HEAD;
        CPDI built on the same basis does not."""
        p_hat, s_hat = disk(CartesianGrid, "linear")
        with quiet():
            s_hat.run(p_hat, n_steps=300)
        assert not np.all(np.isfinite(p_hat.F))

        p_cpdi, s_cpdi = disk(CPDIGrid, "linear")
        with quiet():
            s_cpdi.run(p_cpdi, n_steps=300)
        assert np.all(np.isfinite(p_cpdi.F))
        assert np.abs(np.linalg.det(p_cpdi.F) - 1.0).max() < 0.05


class TestBSplineCombinationIsUnstable:
    """Asserted deliberately. Pairing the CPDI corners with a B-spline grid
    basis is constructible, is *not* published BSCPDI, and blows up -- Eq. 25-26
    assume a linear ``S`` over the domain. Documented so the substitution is not
    mistaken for the real method."""

    def test_consistency_still_holds_which_is_why_it_looks_fine(self, rng):
        grid, xp, st = scene("quadratic", DOMAINS["rotated+sheared"], rng)
        Xn = grid.node_coordinates()
        assert np.abs(st.w.sum(axis=1) - 1.0).max() < 1e-13
        got = np.einsum("pka,pkb->pab", Xn[st.node_id], st.grad_w)
        assert np.abs(got - np.eye(2)).max() < 1e-12

    def test_but_the_run_destabilises(self):
        p, s = disk(CPDIGrid, "quadratic")
        e0 = p.kinetic_energy()
        try:
            with quiet():
                s.run(p, n_steps=300)
        except IndexError:
            return                       # left the grid: also a failure
        assert p.kinetic_energy() / e0 > 5.0 or not np.all(np.isfinite(p.F))
