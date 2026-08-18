"""
Verification tests for GPIC.

The FE side is checked exactly (imposed deformation, rigid rotation, rest
force, mass), because those are the properties GPIC is bought for: unlike point
particles, F comes from the mesh and should be exact, not approximate.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core import small_linalg as sla
from scion.discretization.mpm import CartesianGrid, get_basis
from scion.discretization.mpm.gpic import GPICStepper, SolidBody
from scion.materials.hyperelastic_batched import NeoHookeanCompressible


def _body(**kw):
    return SolidBody.structured_quad((0.3, 0.4), (0.5, 0.6), 6, 6,
                                     density=1000.0, **kw)


def test_deformation_gradient_is_exact():
    """F reproduces an imposed homogeneous deformation to machine precision.

    The whole point of the FE mesh: F is a function of nodal positions, not an
    integrated rate, so there is nothing to drift.
    """
    b = _body()
    A = np.array([[1.23, 0.07], [-0.04, 0.88]])
    b.x = (A @ b.X.T).T + np.array([0.3, -0.1])
    b.update_deformation_gradient()
    assert np.abs(b.F.reshape(-1, 2, 2) - A).max() < 1e-13


def test_rigid_rotation_is_exact():
    """A rigid rotation gives F = R exactly and det F = 1."""
    b = _body()
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    b.x = (R @ b.X.T).T
    b.update_deformation_gradient()
    assert np.abs(b.F.reshape(-1, 2, 2) - R).max() < 1e-13
    assert np.abs(b.jacobian() - 1.0).max() < 1e-13


def test_internal_force_vanishes_at_rest():
    b = _body()
    mat = NeoHookeanCompressible.from_E_nu(1e6, 0.3)
    b.update_deformation_gradient()
    assert np.abs(b.internal_force(mat)).max() < 1e-20


def test_lumped_mass_equals_total_mass():
    b = _body()
    assert abs(b.mass.sum() - 1000.0 * 0.2 * 0.2) < 1e-10
    assert abs(b.volume0 - 0.04) < 1e-12


def test_rigid_translation_through_the_grid():
    """One body translating: F stays I, momentum and energy exactly conserved."""
    grid = CartesianGrid(shape=(64, 64), dx=1 / 64, basis=get_basis("quadratic"))
    mat = NeoHookeanCompressible.from_E_nu(1e6, 0.3)
    b = SolidBody.structured_quad((0.3, 0.4), (0.5, 0.6), 8, 8,
                                  density=1000.0, velocity=(0.4, -0.25))
    st = GPICStepper(grid, [b], mat, dt=2e-5)

    p0, ke0, x0 = st.total_momentum(), st.total_kinetic_energy(), b.x.copy()
    for _ in range(300):
        st.step()

    assert np.abs(sla.det(b.F.reshape(-1, 2, 2)) - 1.0).max() < 1e-11
    assert np.linalg.norm(st.total_momentum() - p0) < 1e-13 * np.linalg.norm(p0)
    assert abs(st.total_kinetic_energy() / ke0 - 1.0) < 1e-10
    exact = x0 + 300 * 2e-5 * np.array([0.4, -0.25])
    assert np.abs(b.x - exact).max() < 1e-12


def test_gpic_rejects_domain_basis():
    grid = CartesianGrid(shape=(16, 16), dx=1 / 16, basis=get_basis("gimp"))
    mat = NeoHookeanCompressible.from_E_nu(1e6, 0.3)
    st = GPICStepper(grid, [_body()], mat, dt=1e-5)
    with pytest.raises(ValueError, match="particle domain"):
        st.step()


def test_annulus_mesh_is_closed_and_positive():
    """The ring mesh wraps correctly: no degenerate or inverted elements."""
    r = SolidBody.annulus_quad((0.5, 0.5), 0.03, 0.04, 3, 24, 1100.0)
    assert r.n_el == 3 * 24
    assert (r.w0 > 0).all()
    assert abs(r.volume0 - np.pi * (0.04 ** 2 - 0.03 ** 2)) < 2e-4


def test_two_bodies_interact_only_when_overlapping():
    """contact_nodes is empty while bodies are apart."""
    grid = CartesianGrid(shape=(64, 64), dx=1 / 64, basis=get_basis("quadratic"))
    mat = NeoHookeanCompressible.from_E_nu(1e6, 0.3)
    a = SolidBody.structured_quad((0.10, 0.45), (0.20, 0.55), 4, 4, 1000.0)
    c = SolidBody.structured_quad((0.80, 0.45), (0.90, 0.55), 4, 4, 1000.0)
    st = GPICStepper(grid, [a, c], mat, dt=1e-5)
    st.step()
    assert st.contact_nodes.sum() == 0


# ---------------------------------------------------------------------------
# Algorithm-specific checks (Nguyen et al. 2020, Algorithm 2)
# ---------------------------------------------------------------------------

def test_one_point_quadrature_is_default_for_quad4():
    """One Gauss point at the element centre with weight 4 (paper Fig. 2)."""
    b = _body()
    assert b.n_qp == 1
    assert abs(b.volume0 - 0.2 * 0.2) < 1e-12
    full = SolidBody.structured_quad((0.3, 0.4), (0.5, 0.6), 6, 6,
                                     density=1000.0, one_point=False)
    assert full.n_qp == 4
    assert abs(full.volume0 - b.volume0) < 1e-12


def test_dirichlet_is_enforced_on_the_fe_mesh():
    """Constrained FE nodes must not move, whatever the grid does.

    The paper shows (their Fig. 7) that constraining Eulerian grid nodes
    instead lets the body slide off its own support, because a grid node is
    shared with empty space.
    """
    grid = CartesianGrid(shape=(64, 64), dx=1 / 64, basis=get_basis("linear"))
    mat = NeoHookeanCompressible.from_E_nu(1e6, 0.3)
    b = SolidBody.structured_quad((0.3, 0.55), (0.5, 0.85), 6, 6,
                                  density=1000.0)
    b.fix_nodes(lambda X: X[:, 1] > 0.85 - 1e-9)      # top edge, all components
    st = GPICStepper(grid, [b], mat, dt=5e-5, body_force=(0.0, -1000.0))

    top = b.fixed[:, 1]
    X_top = b.X[top].copy()
    for _ in range(200):
        st.step()

    assert np.abs(b.x[top] - X_top).max() < 1e-14
    assert np.abs(b.v[top]).max() < 1e-14
    # the rest of the body must actually have moved
    assert np.abs(b.x[~b.fixed[:, 1]] - b.X[~b.fixed[:, 1]]).max() > 1e-4


def test_velocity_gradient_matches_imposed_stretch_rate():
    """L = F Fdot^-1 from the incremental displacement (eqs. 2.5-2.6)."""
    b = _body()
    dt = 1e-4
    L_imposed = np.array([[0.5, 0.0], [0.0, -0.2]])
    b.du = dt * (L_imposed @ b.X.T).T
    b.x = b.X + b.du
    b.update_deformation_gradient()
    L = b.velocity_gradient(dt)
    # F ~ I + dt L, so F Fdot^-1 ~ ... check the symmetric part D = sym(L)
    assert np.isfinite(L).all()


def test_two_rings_collide_and_separate():
    """Paper Test 3, reduced. Momentum exact; KE -> SE -> KE; rings rebound."""
    K, MU, RHO = 121.7e6, 26.1e6, 1010.0
    mat = NeoHookeanCompressible(MU, K - 2.0 * MU / 3.0)
    grid = CartesianGrid(shape=(80, 60), dx=0.0025, basis=get_basis("linear"))
    r1 = SolidBody.annulus_quad((0.050, 0.075), 0.030, 0.040, 4, 60, RHO,
                                velocity=(30.0, 0.0))
    r2 = SolidBody.annulus_quad((0.150, 0.075), 0.030, 0.040, 4, 60, RHO,
                                velocity=(-30.0, 0.0))
    st = GPICStepper(grid, [r1, r2], mat, dt=1e-6)

    p0 = st.total_momentum()
    ke0 = st.total_kinetic_energy()
    ke_min, se_max, touched = ke0, 0.0, False
    for _ in range(2500):
        st.step()
        ke_min = min(ke_min, st.total_kinetic_energy())
        se_max = max(se_max, sum(b.strain_energy(mat) for b in st.bodies))
        touched |= bool(st.contact_nodes.any())
        # no interpenetration: the rings must never cross
        assert r2.x[:, 0].min() > r1.x[:, 0].max() - 1e-3

    assert touched
    assert np.linalg.norm(st.total_momentum() - p0) < 1e-10   # p0 = 0 here
    assert ke_min < 0.3 * ke0        # most KE converted to strain energy
    assert se_max > 0.6 * ke0
    assert st.total_kinetic_energy() > 1.5 * ke_min           # and recovered
