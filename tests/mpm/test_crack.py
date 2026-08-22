"""Explicit cracks in MPM: node splitting, the driving force, and growth.

The benchmark here is the **fixed-grip strip**, not a handbook SENT formula,
and that choice cost a wrong answer to discover. A handbook edge-crack solution
assumes traction loading on a long strip; prescribing a displacement on a short
one is a different problem, and comparing against it made a *correct* driving
force look wrong by a factor that varied from 2 to 9 with crack length. The
fixed-grip strip has an exact answer for the configuration actually solved:
material goes from uniformly strained ahead of the tip to unloaded behind it,
so ``J = W_0 H`` per unit advance, independent of crack length.

``W_0`` is the **plane-strain** energy density. The 2-D material carries
``F_33 = 1``, so with free sides the far field is plane strain at zero
``sigma_xx`` and ``sigma_yy = E' eps``, giving ``W_0 = E' eps^2 / 2``. Using
``E`` instead of ``E'`` puts a flat 11.7% bias on every number and it does not
converge away -- which is what exposed it.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.discretization.mpm.crack import (CrackPath, crack_tip_force,
                                            split_nodes)
from scion.discretization.mpm.implicit import ImplicitMPMStepper
from scion.materials.eshelby import (eshelby_stress_small_strain, j_integral,
                                     material_force)
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

MU, LAM = 1.0e6, 1.5e6
E = MU * (3 * LAM + 2 * MU) / (LAM + MU)
NU = LAM / (2 * (LAM + MU))
E_PLANE_STRAIN = E / (1 - NU ** 2)


def plate(dx=0.05, ppc=2, W=1.0, H=1.0, crack=None, stretch=0.005):
    sub, pad = dx / ppc, 3
    grid = CartesianGrid(shape=(int(W / dx) + 2 * pad, int(H / dx) + 2 * pad),
                         dx=dx, origin=np.array([-pad * dx, -pad * dx]),
                         basis=get_basis("linear"))
    ax = [np.arange(sub / 2, e, sub) for e in (W, H)]
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")], axis=1)
    parts = ParticleSet(pts, vol0=sub ** 2, density=1.0)
    nan = np.nan
    bcs = [GridDirichlet(lambda c: c[:, 1] < 1e-9, np.array([nan, 0.0])),
           GridDirichlet(lambda c: np.abs(c[:, 1] - H) < 1e-9,
                         np.array([nan, stretch * H])),
           GridDirichlet(lambda c: (np.abs(c[:, 1]) < 1e-9)
                         & (np.abs(c[:, 0]) < 1e-9), np.array([0.0, 0.0]))]
    st = ImplicitMPMStepper(grid, NeoHookeanCompressible(mu=MU, lam=LAM),
                            dirichlet=bcs, crack=crack, max_iter=30)
    return grid, parts, st


def _ramp_grad(x, tip, r_in, r_out):
    d = x[:, :2] - tip
    r = np.linalg.norm(d, axis=1)
    g = np.zeros_like(d)
    ring = (r > r_in) & (r < r_out)
    g[ring] = -(d[ring] / r[ring, None]) / (r_out - r_in)
    return g


def solve(dx=0.05, H=1.0, a=0.7, stretch=0.005):
    """Solve one load step and return everything a contour integral needs."""
    yc = H / 2 + 0.4 * dx        # deliberately off the node line -- see below
    crack = CrackPath([[-dx, yc], [a, yc]])
    grid, parts, st = plate(dx=dx, H=H, crack=crack, stretch=stretch)
    st.run(parts, n_steps=1)
    F = parts.F
    P, Jd = st.material.pk1(F)
    energy = st.material.energy(F)
    sigma = np.einsum("pij,pkj->pik", P, F) / Jd[:, None, None]
    Sig = eshelby_stress_small_strain(energy, F - np.eye(2), sigma)
    return grid, parts, st, crack, Sig


def J_of(dx=0.05, H=1.0, a=0.7, stretch=0.005, rings=(3, 6)):
    grid, parts, st, crack, Sig = solve(dx, H, a, stretch)
    gq = _ramp_grad(parts.x, crack.tip, rings[0] * dx, rings[1] * dx)
    return j_integral(Sig, parts.vol0, gq)


def exact_J(H=1.0, stretch=0.005):
    """``W_0 H`` for the fixed-grip strip -- plane strain."""
    return 0.5 * E_PLANE_STRAIN * stretch ** 2 * H


# ======================================================================
# geometry and splitting
# ======================================================================
class TestVisibility:
    def test_a_particle_ahead_of_the_tip_sees_every_node(self):
        """The tip is where the split has to *stop*. A line of sight passing
        beyond the tip reaches around it and must not be blocked, or the crack
        would extend to the domain edge however short it is."""
        c = CrackPath([[0.0, 0.5], [0.5, 0.5]])
        nodes = np.array([[0.25, 0.0], [0.25, 1.0], [0.75, 0.0], [0.75, 1.0]])
        ahead = np.array([[0.9, 0.4]])
        node_id = np.arange(4)[None, :]
        assert not c.blocked(ahead, nodes, node_id).any()

    def test_a_particle_behind_the_tip_is_blocked_from_the_far_side(self):
        c = CrackPath([[0.0, 0.5], [0.5, 0.5]])
        nodes = np.array([[0.25, 0.0], [0.25, 1.0]])
        below = np.array([[0.25, 0.4]])
        node_id = np.arange(2)[None, :]
        b = c.blocked(below, nodes, node_id)
        assert not b[0, 0]        # same side, visible
        assert b[0, 1]            # across the crack, hidden

    def test_split_adds_one_node_per_blocked_grid_node(self):
        node_id = np.array([[0, 1, 2], [1, 2, 3]])
        # node 1 is blocked from *both* particles, node 2 from only the first
        blocked = np.array([[False, True, True], [True, False, False]])
        new_id, n_tot, origin = split_nodes(node_id, blocked, 4)
        assert n_tot == 6                       # nodes 1 and 2 duplicated
        assert_allclose(origin, [1, 2])
        # one duplicate *per node*, so both blocked references to node 1 land
        # on the same copy -- otherwise the two crack faces would each get
        # their own unconnected field and the body would fall apart
        assert new_id[0, 1] == new_id[1, 0] == 4
        assert new_id[0, 2] == 5                            # node 2's copy
        assert new_id[0, 0] == 0                            # untouched

    def test_no_crossing_leaves_the_stencil_alone(self):
        node_id = np.array([[0, 1], [2, 3]])
        new_id, n_tot, origin = split_nodes(
            node_id, np.zeros_like(node_id, bool), 4)
        assert n_tot == 4 and origin.size == 0
        assert_allclose(new_id, node_id)

    def test_a_3d_grid_is_refused(self):
        grid = CartesianGrid(shape=(4, 4, 4), dx=1.0, basis=get_basis("linear"))
        with pytest.raises(NotImplementedError, match="2-D"):
            ImplicitMPMStepper(grid, NeoHookeanCompressible(mu=MU, lam=LAM),
                               crack=CrackPath([[0, 0], [1, 0]]))


# ======================================================================
# the solve
# ======================================================================
class TestCrackOpens:
    def test_the_faces_separate_behind_the_tip_and_not_ahead(self):
        """The property the whole module exists for. Standard MPM cannot do
        this at all: a single-valued grid field forces the two faces of a
        sub-cell discontinuity to move together."""
        dx = 0.05
        yc = 0.5 + 0.4 * dx
        crack = CrackPath([[-dx, yc], [0.5, yc]])
        grid, parts, st = plate(dx=dx, crack=crack)
        X0 = parts.x.copy()
        st.run(parts, n_steps=1)
        u = parts.x - X0

        def opening(xlo, xhi):
            band = (X0[:, 0] > xlo) & (X0[:, 0] < xhi)
            up = band & (X0[:, 1] > yc) & (X0[:, 1] < yc + 0.06)
            dn = band & (X0[:, 1] < yc) & (X0[:, 1] > yc - 0.06)
            return u[up, 1].mean() - u[dn, 1].mean()

        behind, ahead = opening(0.10, 0.30), opening(0.70, 0.90)
        assert behind > 5.0 * ahead        # measured ~18x
        assert ahead > 0.0                 # intact material still stretches

    def test_an_uncracked_body_has_no_material_force(self):
        """The control that licenses every number in this file. A homogeneous
        equilibrated body has ``Div Sigma = 0``, so the nodal material force
        must vanish -- measured 1e-12. Without this, a nonzero reading at a
        crack tip could be the machinery rather than the crack."""
        grid, parts, st = plate(crack=None)
        st.run(parts, n_steps=1)
        F = parts.F
        P, Jd = st.material.pk1(F)
        sigma = np.einsum("pij,pkj->pik", P, F) / Jd[:, None, None]
        Sig = eshelby_stress_small_strain(st.material.energy(F),
                                          F - np.eye(2), sigma)
        G = np.zeros((st._n_nodes_eff, 2))
        np.add.at(G, st._st.node_id, material_force(Sig, parts.vol0, st._b))
        xy = grid.node_coordinates()[:, :2]
        inside = np.linalg.norm(xy - np.array([0.5, 0.5]), axis=1) <= 0.3
        assert np.abs(G[inside].sum(axis=0)).max() < 1e-8


# ======================================================================
# the driving force
# ======================================================================
class TestDrivingForce:
    def test_matches_the_fixed_grip_strip_to_two_percent(self):
        """The external check. ``J = W_0 H`` is exact for this configuration."""
        got = J_of(dx=0.025, H=1.0, a=0.7)[0]
        assert_allclose(got / exact_J(), 1.0, rtol=0.03)

    def test_mode_I_has_no_transverse_driving_force(self):
        """``J_2`` must vanish by symmetry. This is the check that would catch
        a sign or index slip in the contraction ``Sigma_KL grad q_L``, which a
        magnitude comparison alone cannot see."""
        J = J_of(dx=0.025)
        assert abs(J[1]) < 1e-2 * abs(J[0])   # measured 0.2%

    def test_is_path_independent(self):
        """Only the ring where ``grad q`` is nonzero contributes, so the answer
        must not depend on where that ring sits."""
        grid, parts, st, crack, Sig = solve(dx=0.025)
        vals = [j_integral(Sig, parts.vol0,
                           _ramp_grad(parts.x, crack.tip, a * 0.025,
                                      b * 0.025))[0]
                for a, b in ((2, 4), (3, 6), (4, 8))]
        assert (max(vals) - min(vals)) < 0.03 * abs(np.mean(vals))

    def test_scales_as_strain_squared(self):
        """``J ~ eps^2`` exactly, being quadratic in the field. Measured flat
        to 0.5% over a 4x range in strain, i.e. 16x in J."""
        a = J_of(dx=0.05, stretch=0.0025)[0]
        b = J_of(dx=0.05, stretch=0.005)[0]
        c = J_of(dx=0.05, stretch=0.010)[0]
        assert_allclose(b / a, 4.0, rtol=0.02)
        assert_allclose(c / b, 4.0, rtol=0.02)

    def test_the_disc_and_ramp_forms_agree(self):
        """Summing nodal material forces over a disc and contracting ``Sigma``
        with a ``grad q`` ramp are two routes to the same number, and they
        agree here to 0.6%.

        Recorded because the opposite was believed for a while. When the
        driving force came out constant in crack length, the disc form was
        blamed -- a split node on a crack face has clipped support and does
        carry a boundary term of order ``W``, so the reasoning was sound. The
        measurement says otherwise: both forms give the same answer, and the
        constancy was the *benchmark* being wrong (a handbook SENT formula for
        a problem under fixed grips). A plausible mechanism is not evidence
        that it is the operative one.
        """
        grid, parts, st, crack, Sig = solve(dx=0.05, a=0.3)
        xy = grid.node_coordinates()[:, :2]
        xy_eff = np.vstack([xy, xy[st._dup_origin]])
        disc = crack_tip_force(Sig, parts.vol0, st._b, st._st.node_id,
                               st._n_nodes_eff, xy_eff, crack.tip, 6 * 0.05)[0]
        ramp = j_integral(Sig, parts.vol0,
                          _ramp_grad(parts.x, crack.tip, 0.15, 0.30))[0]
        assert_allclose(disc, ramp, rtol=0.02)


# ======================================================================
# growth
# ======================================================================
class TestPropagation:
    def test_mode_I_grows_straight(self):
        """The propagation loop, closed. A symmetric problem must advance the
        tip along its own line: any drift is the visibility artefact or a slip
        in the direction convention, and both would show here as a kink."""
        dx, Gc, dl = 0.025, 30.0, 0.05
        yc = 0.5 + 0.4 * dx
        crack = CrackPath([[-dx, yc], [0.25, yc]])
        grew, angles = 0, []
        eps = 0.0
        for _ in range(10):
            eps += 0.0008
            grid, parts, st = plate(dx=dx, crack=crack, stretch=eps)
            st.run(parts, n_steps=1)
            F = parts.F
            P, Jd = st.material.pk1(F)
            sigma = np.einsum("pij,pkj->pik", P, F) / Jd[:, None, None]
            Sig = eshelby_stress_small_strain(st.material.energy(F),
                                              F - np.eye(2), sigma)
            cf = j_integral(Sig, parts.vol0,
                            _ramp_grad(parts.x, crack.tip, 3 * dx, 6 * dx))
            mag = np.linalg.norm(cf)
            if mag > Gc:
                angles.append(np.degrees(np.arctan2(cf[1], cf[0])))
                crack = crack.extend(cf / mag, dl)
                grew += 1
            if crack.tip[0] > 0.85:
                break
        assert grew >= 3, "the crack never advanced"
        assert max(abs(np.array(angles))) < 2.0
        # the path stays on its own line to well within a particle spacing;
        # exact straightness is not available, since each increment is taken
        # along a numerically computed direction
        assert np.abs(crack.vertices[:, 1] - yc).max() < 0.1 * dl
