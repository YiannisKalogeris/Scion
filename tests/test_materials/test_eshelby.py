"""
Tests for the Eshelby stress, pseudomomentum and material forces.

These are material-manifold quantities, and the tests that matter are the ones
that would fail if a spatial quantity had been implemented by mistake: ``Sigma``
must be *invariant* under a spatial rotation rather than conjugated by it, and
the pseudomomentum must transform through ``F`` rather than through ``Q``.
Getting either backwards produces something with the right shape, the right
units, and the wrong meaning.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core import small_linalg as sla
from scion.discretization.mpm import CartesianGrid
from scion.materials.eshelby import (
    eshelby_stress,
    eshelby_stress_small_strain,
    j_integral,
    kinetic_energy_density,
    material_force,
    pseudomomentum,
    total_pseudomomentum,
)
from scion.materials.hencky import HenckyElastic
from scion.materials.hyperelastic_batched import NeoHookeanCompressible


@pytest.fixture
def rng():
    return np.random.default_rng(20250819)


@pytest.fixture
def material():
    return NeoHookeanCompressible.from_E_nu(1.0e4, 0.3)


def random_F(rng, n=16, scale=0.2):
    return sla.identity(n, 3) + scale * rng.standard_normal((n, 3, 3))


def sigma_of(mat, F, kinetic=None):
    P, _ = mat.pk1(F)
    return eshelby_stress(mat.energy(F), F, P, kinetic)


# ---------------------------------------------------------------------------
# Eshelby stress
# ---------------------------------------------------------------------------

class TestEshelbyStress:
    def test_vanishes_for_an_unloaded_undeformed_body(self, material):
        assert np.allclose(sigma_of(material, sla.identity(4, 3)), 0.0)

    def test_matches_an_explicit_per_point_assembly(self, material, rng):
        F = random_F(rng, 10)
        P, _ = material.pk1(F)
        W = material.energy(F)
        expected = np.stack([W[p] * np.eye(3) - F[p].T @ P[p] for p in range(10)])
        assert np.allclose(sigma_of(material, F), expected, atol=1e-10)

    def test_symmetric_for_an_isotropic_material(self, material, rng):
        """F^T P = C S, and C and S commute only under isotropy."""
        S = sigma_of(material, random_F(rng))
        assert np.allclose(S, sla.transpose(S), atol=1e-10)

    def test_not_assumed_symmetric_in_general(self):
        """Locks the trap: an anisotropic P must give a non-symmetric Sigma."""
        F = np.array([[[1.2, 0.3, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]]])
        P = np.array([[[1.0, 5.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
        S = eshelby_stress(np.zeros(1), F, P)
        assert not np.allclose(S, sla.transpose(S))

    def test_is_invariant_under_a_spatial_rotation(self, material, rng):
        """The discriminator against a spatial stress.

        Cauchy and Kirchhoff conjugate as ``Q tau Q^T``. Sigma has both indices
        on the material manifold, which a spatial rotation does not touch, so it
        must come back *unchanged*.
        """
        th = 0.6
        c, s = np.cos(th), np.sin(th)
        Q = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        F = random_F(rng)
        assert np.allclose(sigma_of(material, np.einsum("ij,pjk->pik", Q, F)),
                           sigma_of(material, F), atol=1e-10)

    def test_kinetic_term_enters_as_minus_K_identity(self, material, rng):
        F = random_F(rng, 10)
        K = kinetic_energy_density(2.5, rng.standard_normal((10, 3)))
        delta = sigma_of(material, F, kinetic=K) - sigma_of(material, F)
        assert np.allclose(delta, -K[:, None, None] * np.eye(3), atol=1e-10)

    def test_static_by_default(self, material, rng):
        """Omitting the kinetic term must not silently assume one."""
        F = random_F(rng, 6)
        assert np.allclose(sigma_of(material, F),
                           sigma_of(material, F, kinetic=np.zeros(6)))

    @pytest.mark.parametrize("cls", [NeoHookeanCompressible, HenckyElastic])
    def test_works_for_any_material_with_energy_and_pk1(self, cls, rng):
        mat = cls.from_E_nu(1.0e4, 0.3)
        S = sigma_of(mat, random_F(rng, 8, 0.1))
        assert np.all(np.isfinite(S))
        assert np.allclose(S, sla.transpose(S), atol=1e-9)

    def test_pure_dilation_gives_an_isotropic_sigma(self, material):
        F = np.tile(np.eye(3) * 1.1, (1, 1, 1))
        S = sigma_of(material, F)[0]
        assert np.allclose(S - np.diag(np.diag(S)), 0.0, atol=1e-10)
        assert np.allclose(np.diag(S), np.diag(S)[0])


class TestKineticEnergyDensity:
    def test_scalar_density(self):
        v = np.array([[3.0, 4.0, 0.0]])
        assert np.isclose(kinetic_energy_density(2.0, v)[0], 0.5 * 2.0 * 25.0)

    def test_per_point_density(self):
        v = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        got = kinetic_energy_density(np.array([1.0, 3.0]), v)
        assert np.allclose(got, [0.5, 6.0])

    def test_zero_velocity(self):
        assert np.allclose(kinetic_energy_density(5.0, np.zeros((4, 3))), 0.0)


# ---------------------------------------------------------------------------
# Pseudomomentum
# ---------------------------------------------------------------------------

class TestPseudomomentum:
    def test_rigid_translation_gives_minus_rho_v(self):
        v = np.tile([1.0, 2.0, 3.0], (4, 1))
        got = pseudomomentum(2.0, sla.identity(4, 3), v)
        assert np.allclose(got, -2.0 * v)

    def test_stretch_scales_the_component_along_it(self):
        """Transforms through F -- this is what makes it a material vector."""
        F = np.tile(np.diag([2.0, 1.0, 1.0]), (1, 1, 1))
        got = pseudomomentum(1.0, F, np.array([[1.0, 0.0, 0.0]]))
        assert np.allclose(got, [[-2.0, 0.0, 0.0]])

    def test_per_point_density(self):
        F = sla.identity(2, 3)
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        got = pseudomomentum(np.array([1.0, 4.0]), F, v)
        assert np.allclose(got, [[-1.0, 0.0, 0.0], [0.0, -4.0, 0.0]])

    def test_zero_at_rest(self, rng):
        assert np.allclose(pseudomomentum(3.0, random_F(rng, 5), np.zeros((5, 3))),
                           0.0)

    def test_total_is_the_mass_weighted_sum(self):
        m = np.array([1.0, 3.0])
        v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        got = total_pseudomomentum(m, sla.identity(2, 3), v)
        assert np.allclose(got, [-1.0, -3.0, 0.0])

    def test_total_differs_from_spatial_momentum_once_deformed(self):
        """The whole reason to measure it separately."""
        m = np.array([2.0])
        v = np.array([[1.0, 0.0, 0.0]])
        F = np.tile(np.diag([3.0, 1.0, 1.0]), (1, 1, 1))

        spatial = (m[:, None] * v).sum(axis=0)
        material = total_pseudomomentum(m, F, v)
        assert np.allclose(spatial, [2.0, 0.0, 0.0])
        assert np.allclose(material, [-6.0, 0.0, 0.0])

    def test_total_matches_the_sum_of_densities(self, rng):
        n = 12
        F = random_F(rng, n)
        v = rng.standard_normal((n, 3))
        vol0 = rng.uniform(0.5, 1.5, n)
        rho0 = 2.0
        by_density = (pseudomomentum(rho0, F, v) * vol0[:, None]).sum(axis=0)
        by_mass = total_pseudomomentum(rho0 * vol0, F, v)
        assert np.allclose(by_density, by_mass)


# ---------------------------------------------------------------------------
# Nodal material force
# ---------------------------------------------------------------------------

class TestMaterialForce:
    def test_contracts_the_second_index(self):
        """Fixed by the traction form Sigma_KL N_L; a non-symmetric Sigma is
        the only input that can tell the two conventions apart."""
        sigma = np.array([[[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        grad_w = np.array([[[0.0, 1.0, 0.0]]])          # one point, one node
        got = material_force(sigma, np.ones(1), grad_w)
        assert np.allclose(got[0, 0], [2.0, 0.0, 0.0])   # not [0, 0, 0]

    def test_scales_with_reference_volume(self, rng):
        sigma = random_F(rng, 4)
        grad_w = rng.standard_normal((4, 3, 3))
        single = material_force(sigma, np.ones(4), grad_w)
        double = material_force(sigma, np.full(4, 2.0), grad_w)
        assert np.allclose(double, 2.0 * single)

    def test_shape_is_unscattered(self, rng):
        got = material_force(random_F(rng, 7), np.ones(7),
                             rng.standard_normal((7, 9, 3)))
        assert got.shape == (7, 9, 3)

    def test_node_sum_is_zero_by_partition_of_unity(self, material, rng):
        """Sum over *nodes* at a fixed point, which is ``V Sigma sum_i Grad w``.

        Identically zero for any Sigma whatever, so this checks the basis and
        nothing about the material. It is *not* the homogeneous-body statement
        -- that one is the sum over points at a fixed node, below.
        """
        grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16)
        st = grid.build_stencil(rng.uniform(0.3, 0.7, (30, 2)))
        assert np.abs(st.grad_w.sum(axis=1)).max() < 1e-9

        F = sla.identity(30, 2) + 0.2 * rng.standard_normal((30, 2, 2))
        P, _ = material.pk1(F)
        sigma = eshelby_stress(material.energy(F), F, P)
        contrib = material_force(sigma, np.full(30, 1e-3), st.grad_w)
        assert np.abs(contrib.sum(axis=1)).max() < 1e-12


def _lattice(n, dx, ppc, jitter=0.0, rng=None):
    s = np.arange(ppc) / ppc + 0.5 / ppc
    g = (np.arange(n)[:, None] + s[None, :]).ravel() * dx
    X, Y = np.meshgrid(g, g, indexing="ij")
    xp = np.stack([X.ravel(), Y.ravel()], axis=1)
    if jitter:
        xp = xp + rng.uniform(-jitter, jitter, xp.shape) * dx
    return xp, (dx / ppc) ** 2


def _interior_max(grid, r, pad=8):
    ns = grid.node_shape
    mask = np.zeros(ns, dtype=bool)
    mask[pad:-pad, pad:-pad] = True
    return np.abs(r.reshape(*ns, r.shape[-1])[mask]).max()


class TestHomogeneousBodyResidual:
    """``G_i = Sigma . sum_p V_p Grad w_ip`` for a homogeneous body.

    The bracket is the discrete divergence residual: it approximates
    ``\\int Grad w_i dV = 0``, exactly zero for the continuum, so whatever
    survives is particle-quadrature error. This is the quantity a
    configurational-force error indicator is built on.

    Nodes are masked 8 deep. A shallower mask includes nodes whose support
    reaches past the edge of the particle region, and the boundary truncation
    there is orders of magnitude larger than the effect being measured -- it
    looks like a quadrature error that *grows* with particle count.
    """

    N, PPC = 32, 2

    def residual(self, basis, jitter, rng):
        from scion.discretization.mpm.basis import get_basis
        dx = 1.0 / self.N
        grid = CartesianGrid(shape=(self.N, self.N), dx=dx,
                             basis=get_basis(basis))
        xp, vol = _lattice(self.N, dx, self.PPC, jitter, rng)
        st = grid.build_stencil(xp)
        r = np.zeros((grid.n_nodes, 2))
        np.add.at(r, st.node_id, vol * st.grad_w)
        return _interior_max(grid, r)

    @pytest.mark.parametrize("basis", ["linear", "quadratic", "cubic"])
    def test_uniform_lattice_is_exact(self, basis, rng):
        """A regular particle lattice integrates these bases exactly."""
        assert self.residual(basis, 0.0, rng) < 1e-15

    @pytest.mark.parametrize("basis", ["linear", "quadratic", "cubic"])
    def test_disorder_produces_a_residual(self, basis, rng):
        """Which is the point: the error comes from the particles drifting off
        the lattice, not from the basis."""
        assert self.residual(basis, 0.25, rng) > 1e-4

    @pytest.mark.parametrize("basis", ["linear", "quadratic", "cubic"])
    def test_residual_grows_with_disorder(self, basis, rng):
        small = self.residual(basis, 0.05, rng)
        large = self.residual(basis, 0.25, rng)
        assert large > 2.0 * small

    def test_scales_with_the_eshelby_stress(self, material, rng):
        """For a homogeneous body G_i factorises as Sigma . r_i exactly."""
        from scion.discretization.mpm.basis import get_basis
        dx = 1.0 / self.N
        grid = CartesianGrid(shape=(self.N, self.N), dx=dx,
                             basis=get_basis("quadratic"))
        xp, vol = _lattice(self.N, dx, self.PPC, 0.25, rng)
        n = xp.shape[0]
        st = grid.build_stencil(xp)

        F = np.tile(np.diag([1.3, 0.9]), (n, 1, 1))
        P, _ = material.pk1(F)
        sigma = eshelby_stress(material.energy(F), F, P)
        assert np.abs(sigma - sigma[0]).max() < 1e-12          # truly uniform

        G = np.zeros((grid.n_nodes, 2))
        np.add.at(G, st.node_id,
                  material_force(sigma, np.full(n, vol), st.grad_w))

        r = np.zeros((grid.n_nodes, 2))
        np.add.at(r, st.node_id, vol * st.grad_w)
        expected = np.einsum("ij,nj->ni", sigma[0], r)
        assert np.allclose(G, expected, atol=1e-12)

    def test_scatters_into_a_nodal_field(self, material, rng):
        """The documented usage pattern, mirroring internal_force."""
        grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16)
        xp = rng.uniform(0.3, 0.7, (30, 2))
        st = grid.build_stencil(xp)

        F = sla.identity(30, 2) + 0.15 * rng.standard_normal((30, 2, 2))
        P, _ = material.pk1(F)
        sigma = eshelby_stress(material.energy(F), F, P)

        G = np.zeros((grid.n_nodes, 2))
        np.add.at(G, st.node_id, material_force(sigma, np.full(30, 1e-3),
                                                st.grad_w))
        assert np.all(np.isfinite(G))
        assert np.abs(G).max() > 0.0


# ---------------------------------------------------------------------------
# J-integral, against the Williams K-field
# ---------------------------------------------------------------------------

class WilliamsField:
    """Mode-I near-tip solution, plane strain, with an *analytic* gradient.

    Finite-differencing this field is a trap: ``arctan2`` has a branch cut along
    the crack faces, and a stencil that straddles it differences across the
    crack opening. Parametrising by ``(r, theta)`` and differentiating in closed
    form avoids the question.

    ``u = A sqrt(r) (kappa - cos t) (cos(t/2), sin(t/2))`` with
    ``A = K_I / (2 mu sqrt(2 pi))``.
    """

    def __init__(self, E=210e9, nu=0.3, K_I=1.0e6):
        self.E, self.nu, self.K_I = E, nu, K_I
        self.mu = E / (2 * (1 + nu))
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.kappa = 3.0 - 4.0 * nu                    # plane strain
        self.A = K_I / (2 * self.mu * np.sqrt(2 * np.pi))

    @property
    def J_exact(self):
        """``K_I^2 / E'`` with ``E' = E/(1-nu^2)`` in plane strain."""
        return self.K_I ** 2 * (1 - self.nu ** 2) / self.E

    def grad_u(self, r, t):
        s, b = np.sqrt(r), self.kappa - np.cos(t)
        ur_x = self.A / (2 * s) * np.cos(t / 2) * b
        ur_y = self.A / (2 * s) * np.sin(t / 2) * b
        ut_x = self.A * s * (-0.5 * np.sin(t / 2) * b
                             + np.cos(t / 2) * np.sin(t))
        ut_y = self.A * s * (0.5 * np.cos(t / 2) * b
                             + np.sin(t / 2) * np.sin(t))
        ct, st = np.cos(t), np.sin(t)
        H = np.zeros(r.shape + (2, 2))
        H[..., 0, 0] = ct * ur_x - st / r * ut_x
        H[..., 0, 1] = st * ur_x + ct / r * ut_x
        H[..., 1, 0] = ct * ur_y - st / r * ut_y
        H[..., 1, 1] = st * ur_y + ct / r * ut_y
        return H

    def stress_and_energy(self, H):
        eps = 0.5 * (H + np.swapaxes(H, -1, -2))
        tr = np.einsum("...ii->...", eps)
        sig = 2 * self.mu * eps + self.lam * tr[..., None, None] * np.eye(2)
        return sig, 0.5 * np.einsum("...ij,...ij->...", sig, eps)

    def ring(self, r_in, r_out, n_r=600):
        """Points, weights and grad q on the annulus where q ramps 1 -> 0."""
        r = np.linspace(r_in, r_out, n_r)
        t = np.linspace(-np.pi, np.pi, 2 * n_r)
        R, T = np.meshgrid(r, t, indexing="ij")
        area = (R * (r[1] - r[0]) * (t[1] - t[0])).ravel()
        R, T = R.ravel(), T.ravel()
        grad_q = np.stack([-np.cos(T), -np.sin(T)], axis=1) / (r_out - r_in)
        return R, T, area, grad_q


@pytest.fixture
def williams():
    return WilliamsField()


class TestJIntegral:
    def small_strain_J(self, w, r_in=0.02, r_out=0.04, n_r=600):
        R, T, area, gq = w.ring(r_in, r_out, n_r)
        H = w.grad_u(R, T)
        sig, W = w.stress_and_energy(H)
        return j_integral(eshelby_stress_small_strain(W, H, sig), area, gq)[0]

    def finite_strain_J(self, w, r_in=0.02, r_out=0.04, n_r=600):
        R, T, area, gq = w.ring(r_in, r_out, n_r)
        H = w.grad_u(R, T)
        sig, W = w.stress_and_energy(H)
        # P = dW/dF, which for this W is exactly sigma -- see the trap test
        return j_integral(eshelby_stress(W, np.eye(2) + H, sig), area, gq)[0]

    def test_recovers_K_squared_over_E_prime(self, williams):
        """The external check the whole module rested on until now."""
        got = self.small_strain_J(williams)
        assert np.isclose(got, williams.J_exact, rtol=5e-3), got

    def test_is_path_independent(self, williams):
        """Div Sigma = 0, so J must not depend on where the ring is."""
        vals = [self.small_strain_J(williams, a, b)
                for a, b in [(0.01, 0.02), (0.02, 0.04), (0.05, 0.10)]]
        assert max(vals) - min(vals) < 1e-3 * williams.J_exact

    def test_scales_as_K_squared(self):
        a = WilliamsField(K_I=1.0e6)
        b = WilliamsField(K_I=2.0e6)
        assert np.isclose(self.small_strain_J(b) / self.small_strain_J(a),
                          4.0, rtol=1e-6)

    def test_J_2_vanishes_for_pure_mode_I(self, williams):
        R, T, area, gq = williams.ring(0.02, 0.04)
        H = williams.grad_u(R, T)
        sig, W = williams.stress_and_energy(H)
        J = j_integral(eshelby_stress_small_strain(W, H, sig), area, gq)
        assert abs(J[1]) < 1e-3 * abs(J[0])

    def test_finite_strain_agrees_but_converges_far_more_slowly(self, williams):
        """Both tensors are correct; only one is usable at coarse quadrature.

        The finite-strain form carries a term of order ``sigma``, which near a
        crack tip is ~1e5 times ``J`` and cancels analytically. What survives is
        quadrature error of the same size as the answer.
        """
        coarse_small = self.small_strain_J(williams, n_r=300)
        coarse_finite = self.finite_strain_J(williams, n_r=300)
        fine_finite = self.finite_strain_J(williams, n_r=2400)

        assert np.isclose(coarse_small, williams.J_exact, rtol=5e-3)
        assert abs(coarse_finite - williams.J_exact) > 0.1 * williams.J_exact
        # but it does converge to the same answer
        assert abs(fine_finite - williams.J_exact) < abs(
            coarse_finite - williams.J_exact)
        assert np.isclose(fine_finite, williams.J_exact, rtol=2e-2)

    def test_mismatched_P_gives_exactly_half(self, williams):
        """The trap worth a regression test.

        ``P = J sigma F^-T`` is the correct Cauchy-to-Piola relation, but it is
        not ``dW/dF`` for a linear-elastic ``W``. Pairing them converges to half
        the true J, which reads as a factor-of-two convention slip and is not.
        """
        R, T, area, gq = williams.ring(0.02, 0.04, n_r=2400)
        H = williams.grad_u(R, T)
        sig, W = williams.stress_and_energy(H)

        F = np.eye(2) + H
        Jdet = np.linalg.det(F)
        P_wrong = Jdet[:, None, None] * np.einsum(
            "pij,pkj->pik", sig, np.linalg.inv(F))
        got = j_integral(eshelby_stress(W, F, P_wrong), area, gq)[0]
        assert np.isclose(got / williams.J_exact, 0.5, rtol=0.05), got

    def test_vanishes_for_an_unstressed_body(self, williams):
        R, T, area, gq = williams.ring(0.02, 0.04, n_r=200)
        n = R.size
        zero = np.zeros((n, 2, 2))
        assert np.allclose(j_integral(zero, area, gq), 0.0)
