"""
Total-Lagrangian MPM.

The defining properties, each asserted directly rather than inferred from a
benchmark: the stencil is built once and never changes, ``F`` is reconstructed
from the nodal displacement field instead of accumulated, and the reference grid
has to contain the undeformed body for the whole run.

What is *not* tested here is that TL beats UL on accuracy. Rigid translation
cannot show it -- a uniform velocity field is reproduced exactly by any
partition-of-unity basis -- and on a genuinely deforming motion neither scheme
is the exact answer, so a difference between them is not evidence either way.
The structural claims below are the ones that can be made honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core import small_linalg as sla
from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC,
    CartesianGrid,
    ExplicitMPMStepper,
    FLIP,
    PIC,
    ParticleSet,
    TotalLagrangianStepper,
)
from scion.discretization.mpm.basis import get_basis
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

RHO = 1000.0
N = 32
DX = 1.0 / N


def material():
    return NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)


def block(dx=DX):
    return ParticleSet.from_sampler(
        lambda q: ((q[:, 0] > 0.3) & (q[:, 0] < 0.7)
                   & (q[:, 1] > 0.4) & (q[:, 1] < 0.6)),
        ((0.0, 0.0), (1.0, 1.0)), dx, ppc=2, density=RHO)


# NOT named `setup`: pytest treats a module-level `setup` as the
# xunit module hook and calls it with the module object, which
# arrives here as `basis` and errors out every test in the file.
def make_case(basis="quadratic", transfer=None, hessian=False, n_grid=N):
    dx = 1.0 / n_grid
    grid = CartesianGrid(shape=(n_grid, n_grid), dx=dx, basis=get_basis(basis))
    parts = block(dx)
    mat = material()
    c = mat.wave_speed_factor / np.sqrt(RHO)
    stepper = TotalLagrangianStepper(grid, mat, transfer or APIC(),
                                     dt=0.2 * dx / c, hessian=hessian)
    return grid, parts, stepper


class TestReferencePositions:
    def test_X_defaults_to_the_initial_placement(self):
        p = block()
        assert np.array_equal(p.X, p.x)

    def test_X_is_a_copy_not_a_view(self):
        """Advecting x must never drag the reference configuration with it."""
        p = block()
        p.x += 1.0
        assert not np.allclose(p.X, p.x)

    def test_X_can_be_given_explicitly(self):
        x = np.array([[0.5, 0.5]])
        X = np.array([[0.1, 0.2]])
        p = ParticleSet(x, vol0=1e-3, X=X)
        assert np.allclose(p.X, X)

    def test_displacement(self):
        p = block()
        p.x = p.X + 0.25
        assert np.allclose(p.displacement(), 0.25)

    def test_concatenate_carries_X(self):
        a, b = block(), block()
        b.X = b.X + 5.0
        merged = a.concatenate(b)
        assert merged.X.shape == merged.x.shape
        assert np.allclose(merged.X[a.n:], b.X)


class TestFixedStencil:
    def test_prepare_is_idempotent(self):
        _, p, s = make_case()
        assert s.prepare(p) is s.prepare(p)

    def test_stencil_is_none_before_the_first_step(self):
        _, _, s = make_case()
        assert s.stencil is None

    def test_stencil_does_not_change_as_the_body_moves(self):
        """The literal statement of 'no cell crossing'."""
        _, p, s = make_case()
        p.v[:] = np.array([1.0, 0.0])
        before = s.prepare(p).node_id.copy()
        with quiet():
            s.run(p, n_steps=400)

        travelled = np.abs(p.x - p.X)[:, 0].max()
        assert travelled > 5 * DX                      # it really moved
        assert np.array_equal(before, s.prepare(p).node_id)

    def test_the_updated_lagrangian_stencil_does_change(self):
        """Guards the previous test against passing for a trivial reason."""
        grid = CartesianGrid(shape=(N, N), dx=DX)
        p = block()
        p.v[:] = np.array([1.0, 0.0])
        mat = material()
        c = mat.wave_speed_factor / np.sqrt(RHO)
        s = ExplicitMPMStepper(grid, mat, APIC(), dt=0.2 * DX / c)

        before = grid.build_stencil(p.x).node_id.copy()
        with quiet():
            s.run(p, n_steps=400)
        assert not np.array_equal(before, grid.build_stencil(p.x).node_id)

    def test_grid_must_contain_the_body(self):
        grid = CartesianGrid(shape=(4, 4), dx=DX)      # covers [0, 0.125]^2
        s = TotalLagrangianStepper(grid, material(), APIC(), dt=1e-6)
        with pytest.raises(ValueError, match="does not contain the undeformed"):
            s.prepare(block())

    def test_hessian_is_available_when_requested(self):
        _, p, s = make_case(hessian=True)
        st = s.prepare(p)
        assert st.hess is not None
        assert st.hess.shape == (p.n, grid_k(st), 2, 2)
        assert np.all(np.isfinite(st.hess))

    def test_hessian_absent_by_default(self):
        _, p, s = make_case()
        assert s.prepare(p).hess is None


def grid_k(st):
    return st.node_id.shape[1]


class TestKinematics:
    def test_a_body_at_rest_stays_exactly_at_rest(self):
        _, p, s = make_case()
        with quiet():
            s.run(p, n_steps=200)
        assert np.abs(p.x - p.X).max() == 0.0
        assert np.abs(p.F - sla.identity(p.n, 2)).max() == 0.0
        assert np.abs(p.v).max() == 0.0

    def test_F_is_reconstructed_from_the_nodal_field(self):
        """A uniform nodal displacement u = A X must give F = I + A exactly.

        This is the property that distinguishes total- from updated-Lagrangian:
        no accumulation, so no integration error in F at all.
        """
        grid, p, s = make_case()
        st = s.prepare(p)
        A = np.array([[0.10, 0.03], [0.00, -0.05]])
        s.u_nodes = grid.node_coordinates() @ A.T
        assert np.allclose(s.deformation_gradient(st), np.eye(2) + A,
                           atol=1e-13)

    def test_zero_displacement_gives_identity(self):
        _, p, s = make_case()
        st = s.prepare(p)
        assert np.allclose(s.deformation_gradient(st), np.eye(2))

    def test_position_follows_the_same_nodal_field_as_F(self):
        """x and F must be read from one field, or they can disagree."""
        grid, p, s = make_case()
        p.v[:] = np.array([0.5, 0.2])
        with quiet():
            s.run(p, n_steps=100)
        st = s.stencil
        expected = p.X + np.einsum("pk,pka->pa", st.w, s.u_nodes[st.node_id])
        assert np.allclose(p.x, expected)


class TestConservation:
    @pytest.mark.parametrize("name,transfer", [
        ("pic", PIC()), ("flip", FLIP(alpha=0.95)), ("apic", APIC()),
    ])
    def test_momentum_conserved_without_forces_or_walls(self, name, transfer):
        _, p, s = make_case(transfer=transfer)
        p.v[:] = np.array([0.7, -0.3])
        p0 = p.linear_momentum()
        with quiet():
            s.run(p, n_steps=250)
        drift = np.linalg.norm(p.linear_momentum() - p0) / np.linalg.norm(p0)
        assert drift < 1e-12

    def test_mass_is_conserved(self):
        _, p, s = make_case()
        p.v[:] = np.array([0.4, 0.0])
        m0 = p.mass.sum()
        with quiet():
            s.run(p, n_steps=200)
        assert np.isclose(p.mass.sum(), m0)

    @pytest.mark.parametrize("basis", ["linear", "quadratic", "cubic"])
    def test_runs_on_every_point_basis(self, basis):
        _, p, s = make_case(basis=basis)
        p.v[:] = np.array([0.3, 0.1])
        with quiet():
            s.run(p, n_steps=100)
        assert np.all(np.isfinite(p.x)) and np.all(np.isfinite(p.F))


class TestMaterialInterface:
    def test_state_without_update_is_rejected(self):
        grid = CartesianGrid(shape=(N, N), dx=DX)
        with pytest.raises(TypeError, match="no update"):
            TotalLagrangianStepper(grid, material(), APIC(), dt=1e-6,
                                   state=object())

    def test_path_dependent_material_runs(self):
        from scion.materials.hencky import (
            HenckyElastic, PrincipalPlasticity, PrincipalPlasticState,
            VonMisesHencky,
        )
        grid = CartesianGrid(shape=(N, N), dx=DX)
        p = block()
        p.v[:, 0] = 0.5 * (p.x[:, 1] - 0.5)
        mat = PrincipalPlasticity(HenckyElastic.from_E_nu(5.0e4, 0.3),
                                  VonMisesHencky.from_von_mises(1.0e3))
        c = mat.wave_speed_factor / np.sqrt(RHO)
        s = TotalLagrangianStepper(grid, mat, APIC(), dt=0.2 * DX / c,
                                   state=PrincipalPlasticState.identity(p.n))
        with quiet():
            s.run(p, n_steps=150)
        assert np.all(np.isfinite(p.F))
        assert np.all(np.isfinite(s.state.F_e))

    def test_cfl_dt_is_positive(self):
        _, p, s = make_case()
        assert s.cfl_dt(p, RHO) > 0.0


class TestConfigurationalForcesOnARun:
    """The reason total-Lagrangian was built: Eq. (9) on a real simulation.

    Previously ``G_p`` could only be evaluated on a synthetic nodal displacement
    field, because the updated-Lagrangian stepper carries no reference
    configuration to differentiate with respect to.
    """

    def evaluate(self, basis, shear, n_steps=200):
        from scion.materials.eshelby import eshelby_stress, material_force
        from projects.configurational_mpm.forces import configurational_force

        grid, p, s = make_case(basis=basis, hessian=True)
        if shear:
            p.v[:, 0] = shear * (p.x[:, 1] - 0.5)
        st = s.prepare(p)
        with quiet():
            s.run(p, n_steps=n_steps)

        mat = material()
        P, _ = mat.pk1(p.F)
        G_p = configurational_force(P, p.vol0, s.u_nodes, st.node_id, st.hess)

        sigma = eshelby_stress(mat.energy(p.F), p.F, P)
        G_i = np.zeros((grid.n_nodes, 2))
        np.add.at(G_i, st.node_id, material_force(sigma, p.vol0, st.grad_w))
        return G_p, G_i, np.abs(p.F - np.eye(2)).max()

    @pytest.mark.parametrize("basis", ["quadratic", "cubic"])
    def test_both_vanish_for_an_undeformed_body(self, basis):
        G_p, G_i, F_err = self.evaluate(basis, shear=0.0)
        assert F_err == 0.0
        assert np.abs(G_p).max() == 0.0
        assert np.abs(G_i).max() == 0.0

    @pytest.mark.parametrize("basis", ["quadratic", "cubic"])
    def test_finite_and_nonzero_once_deformed(self, basis):
        G_p, G_i, F_err = self.evaluate(basis, shear=4.0)
        assert F_err > 0.05
        assert np.all(np.isfinite(G_p)) and np.all(np.isfinite(G_i))
        assert np.abs(G_p).max() > 0.0

    def test_smoother_basis_produces_less_spurious_force(self):
        """Eq. (9) is driven by the second material gradient of the basis, so a
        C2 spline should produce less of it than a C1 one on the same motion."""
        quad, _, _ = self.evaluate("quadratic", shear=4.0)
        cubic, _, _ = self.evaluate("cubic", shear=4.0)
        assert np.abs(cubic).max() < np.abs(quad).max()
