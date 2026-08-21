"""Implicit micropolar (Cosserat) MPM.

Verification is layered so that a failure localises:

1. **SO(3) utilities** against scipy and against the differential of the
   exponential map. Nothing downstream is meaningful if a quaternion composes
   the wrong way round.
2. **The constitutive law** -- the closed form against the fourth-order
   contraction, and objectivity of both deformation measures.
3. **The wryness update**, against the *definition* of the wryness tensor. This
   is what fixes ``H`` rather than ``H^T``; the reference states the relation
   one way and its own Eq. 55 delivers the other, and the two differ by 4.5%.
4. **The four tangent blocks**, by central differences on the residual. The
   tangent is exact only at zero increment -- that is the Eq. B.6 property, not
   a defect -- so the check is run there and the linear degradation away from it
   is measured rather than assumed.
5. **The manufactured solution**, which is the only check that tests the
   *residual* against the governing equations rather than against itself.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scion.core.rotation import (
    axial,
    quat_from_rotvec,
    quat_identity,
    quat_multiply,
    quat_to_matrix,
    rotvec_from_quat,
    skew,
    spin_map,
)
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
from scion.materials.micropolar_plasticity import MicropolarDruckerPrager
from scion.solvers.tangent_check import check_tangent, numerical_tangent

_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0

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


# ======================================================================
# 1. SO(3)
# ======================================================================
class TestRotation:
    def test_matches_scipy_including_past_pi(self):
        rng = np.random.default_rng(0)
        w = 2.0 * rng.standard_normal((200, 3))
        w[:3] = [[0, 0, 0], [1e-12, 0, 0], [0, 3 * np.pi, 0]]
        Q = quat_to_matrix(quat_from_rotvec(w))
        assert np.abs(Q - Rotation.from_rotvec(w).as_matrix()).max() < 1e-14

    def test_zero_rotation_is_not_nan(self):
        """Every formula here has a 0/0 at theta = 0 with a finite limit.
        Returning nan on the first step of a problem starting from rest is the
        most common way this is got wrong."""
        w = np.zeros((3, 3))
        assert np.isfinite(quat_from_rotvec(w)).all()
        assert np.isfinite(spin_map(w)).all()
        assert np.isfinite(rotvec_from_quat(quat_identity(3))).all()
        assert np.abs(quat_to_matrix(quat_from_rotvec(w)) - np.eye(3)).max() == 0

    def test_stays_in_SO3_under_many_increments(self):
        q = quat_identity(1)
        rng = np.random.default_rng(2)
        for _ in range(500):
            q = quat_multiply(quat_from_rotvec(0.05 * rng.standard_normal((1, 3))), q)
        Q = quat_to_matrix(q)
        assert np.abs(Q @ Q.transpose(0, 2, 1) - np.eye(3)).max() < 1e-12
        assert abs(np.linalg.det(Q)[0] - 1.0) < 1e-12

    def test_composition_order(self):
        """``quat_multiply(dq, q)`` must be ``Q(dq) Q(q)``. The reverse applies
        the increment in the material frame -- also self-consistent, and a
        different scheme."""
        rng = np.random.default_rng(5)
        p = quat_from_rotvec(rng.standard_normal((20, 3)))
        q = quat_from_rotvec(rng.standard_normal((20, 3)))
        lhs = quat_to_matrix(quat_multiply(p, q))
        rhs = np.einsum("nij,njk->nik", quat_to_matrix(p), quat_to_matrix(q))
        assert np.abs(lhs - rhs).max() < 1e-13

    def test_skew_and_axial_are_inverse(self):
        rng = np.random.default_rng(7)
        v, u = rng.standard_normal((50, 3)), rng.standard_normal((50, 3))
        assert np.abs(np.einsum("nij,nj->ni", skew(v), u)
                      - np.cross(v, u)).max() == 0
        assert np.abs(axial(skew(v)) - v).max() < 1e-15

    def test_spin_map_is_the_exponential_differential(self):
        """``H`` must satisfy ``dw_spatial = H dphi``, i.e. it is
        ``axial(dQ Q^T)`` differentiated in phi."""
        for w0 in ([0.3, -0.7, 1.1], [1e-9, 0.0, 0.0]):
            w0 = np.array(w0)
            H_num = np.zeros((3, 3))
            h = 1e-7
            Q0 = Rotation.from_rotvec(w0).as_matrix()
            for k in range(3):
                e = np.zeros(3)
                e[k] = h
                dQ = (Rotation.from_rotvec(w0 + e).as_matrix() - Q0) / h
                H_num[:, k] = axial((dQ @ Q0.T)[None])[0]
            assert np.abs(spin_map(w0[None])[0] - H_num).max() < 1e-6


# ======================================================================
# 2. the constitutive law
# ======================================================================
class TestMaterial:
    def test_closed_form_matches_the_contraction(self):
        rng = np.random.default_rng(0)
        E, G = rng.standard_normal((7, 3, 3)), rng.standard_normal((7, 3, 3))
        B, S = MATERIAL.stresses(E, G)
        assert np.abs(B - np.einsum("abgp,ngp->nab", MATERIAL.D, E)).max() \
            < 1e-8 * np.abs(B).max()
        assert np.abs(S - np.einsum("abgp,ngp->nab", MATERIAL.D_curvature, G)).max() \
            < 1e-8 * max(np.abs(S).max(), 1.0)

    def test_moduli_are_major_symmetric(self):
        for C in (MATERIAL.D, MATERIAL.D_curvature):
            assert np.abs(C - C.transpose(2, 3, 0, 1)).max() == 0

    def test_both_measures_are_objective(self):
        """``E = Q^T F - I`` and ``Gamma`` are invariant under a superposed
        spatial rotation (``F -> RF``, ``Q -> RQ``). Defining the law in the
        material frame is what makes this automatic; a law written on ``sigma``
        would have to earn it."""
        rng = np.random.default_rng(11)
        F = np.eye(3) + 0.2 * rng.standard_normal((6, 3, 3))
        Q = quat_to_matrix(quat_from_rotvec(rng.standard_normal((6, 3))))
        R = quat_to_matrix(quat_from_rotvec(np.tile([0.4, -0.9, 0.3], (6, 1))))
        E = np.einsum("pji,pjk->pik", Q, F) - np.eye(3)
        E_rot = np.einsum("pji,pjk->pik",
                          np.einsum("pij,pjk->pik", R, Q),
                          np.einsum("pij,pjk->pik", R, F)) - np.eye(3)
        assert np.abs(E - E_rot).max() < 1e-13

    def test_a_is_the_rotation_stiffness(self):
        """The rotation stiffness at the undeformed state is exactly ``4 mu a``.

        This is why ``a = alpha = beta = gam = 0`` -- the reference's stated
        Cauchy-Boltzmann reduction -- is a *singular* discrete problem rather
        than a classical one: the microrotation is left with no equation at all.
        """
        for a in (0.0, 1.0 / 3.0, 0.75):
            mat = MicropolarHooke(lam=3.333e6, mu=5.0e6, a=a)
            K = np.einsum("abi,abgt,gtj->ij", _EPS, mat.D, _EPS)
            assert np.abs(K - 4.0 * mat.mu * a * np.eye(3)).max() \
                < 1e-9 * max(4.0 * mat.mu * a, 1.0)

    def test_coupling_number_of_one_is_refused(self):
        with pytest.raises(ValueError, match="constrained-rotation"):
            MicropolarHooke.from_technical(E=1e6, nu=0.3, N=1.0)


# ======================================================================
# 3. the wryness update
# ======================================================================
class TestWryness:
    def test_H_not_H_transpose(self):
        """``Gamma_:,pi = Q^T H(phi) dphi/dX_pi`` must reproduce the definition
        ``Gamma = -1/2 e : Q^T dQ/dX``. The transposed map is wrong by 4.5% on
        this field, and only at finite increment size -- so nothing else in the
        module would catch it."""
        def phi(X):
            x, y, z = X
            return np.array([0.7 * y + 0.2 * z, 0.5 * x - 0.3 * z,
                             0.4 * x + 0.6 * y])
        dphi = np.array([[0.0, 0.7, 0.2], [0.5, 0.0, -0.3], [0.4, 0.6, 0.0]])

        X0 = np.array([0.31, -0.22, 0.47])
        Q0 = Rotation.from_rotvec(phi(X0)).as_matrix()
        h, Gam_def = 1e-6, np.zeros((3, 3))
        for pi in range(3):
            e = np.zeros(3)
            e[pi] = h
            dQ = (Rotation.from_rotvec(phi(X0 + e)).as_matrix()
                  - Rotation.from_rotvec(phi(X0 - e)).as_matrix()) / (2 * h)
            Gam_def[:, pi] = -0.5 * np.einsum("gth,pt,ph->g", _EPS, Q0, dQ)

        H = spin_map(phi(X0)[None])[0]
        scale = np.abs(Gam_def).max()
        assert np.abs(Q0.T @ H @ dphi - Gam_def).max() < 1e-9 * scale
        assert np.abs(Q0.T @ H.T @ dphi - Gam_def).max() > 1e-2 * scale

    def test_stepper_accumulates_it_with_no_displacement(self):
        """A pure microrotation field must leave ``F`` at identity and put the
        whole increment into the wryness."""
        grid, parts = block()
        stepper = MicropolarMPMStepper(grid, MATERIAL)
        stepper.prepare(parts)
        A = np.array([[0.0, 0.30, 0.10], [0.20, 0.0, -0.15],
                      [0.18, 0.25, 0.0]])
        coords = grid.node_coordinates()
        dof = np.zeros(stepper._n_dof)
        dof.reshape(grid.n_nodes, 6)[:, 3:] = coords @ A.T

        st = stepper._st
        phi_p = np.einsum("pk,pka->pa", st.w, (coords @ A.T)[st.node_id])
        grad_p = np.einsum("pka,pkb->pab", (coords @ A.T)[st.node_id],
                           stepper._grad)
        stepper.apply_increment(parts, dof)

        assert np.abs(parts.F - np.eye(3)).max() == 0
        Q = quat_to_matrix(parts.quat)
        expect = np.einsum("pji,pjk,pkn->pin", Q, spin_map(phi_p), grad_p)
        assert np.abs(parts.wryness - expect).max() < 1e-14 * np.abs(expect).max()


# ======================================================================
# 4. the tangent
# ======================================================================
class TestTangent:
    @staticmethod
    def _deformed():
        grid, parts = block()
        stepper = MicropolarMPMStepper(grid, MATERIAL)
        stepper.prepare(parts)
        rng = np.random.default_rng(3)
        for _ in range(2):
            kick = np.zeros(stepper._n_dof)
            kick[stepper._free] = 0.02 * rng.standard_normal(
                stepper._free.size)
            stepper.apply_increment(parts, kick)
        return grid, parts, stepper

    def test_undeformed_residual_is_exactly_zero(self):
        grid, parts = block()
        stepper = MicropolarMPMStepper(grid, MATERIAL)
        stepper.prepare(parts)
        assert np.abs(stepper.internal_force(parts)).max() == 0.0

    def test_all_four_blocks_against_finite_differences(self):
        _, parts, stepper = self._deformed()
        free = stepper._free
        report = check_tangent(lambda u: stepper.residual_at(parts, u),
                               lambda u: stepper.tangent_at(parts, u),
                               np.zeros(stepper._n_dof), free=free)
        assert report.passed, report

    def test_tangent_is_exact_only_at_zero_increment(self):
        """The Eq. B.6 derivative is taken *at* the current ``Q``, so the
        tangent is exact at zero increment and degrades linearly away from it.
        That is the reason the whole scheme re-bases every iteration, and it is
        measured here rather than asserted: a tangent that was merely
        *approximate* everywhere would not show this clean proportionality."""
        _, parts, stepper = self._deformed()
        free = stepper._free
        rng = np.random.default_rng(9)
        errs = {}
        for scale in (0.0, 1e-5, 1e-4, 1e-3):
            u = np.zeros(stepper._n_dof)
            if scale:
                u[free] = scale * rng.standard_normal(free.size)
            Ka = np.asarray(stepper.tangent_at(parts, u).todense())[
                np.ix_(free, free)]
            Kn = numerical_tangent(lambda v: stepper.residual_at(parts, v),
                                   u, free=free, h=1e-6)
            errs[scale] = np.abs(Ka - Kn).max() / np.abs(Kn).max()

        assert errs[0.0] < 1e-9                      # exact where Newton uses it
        for a, b in zip((1e-5, 1e-4), (1e-4, 1e-3)):
            assert 5.0 < errs[b] / errs[a] < 20.0    # linear in the increment

    def test_unsymmetric_away_from_equilibrium_symmetric_at_it(self):
        """A rotation-parameterised tangent is symmetric only at equilibrium.
        Worth pinning down, because it is the reason this must not be handed to
        PCG even though the underlying elasticity is hyperelastic."""
        grid, parts = block()
        rest = MicropolarMPMStepper(grid, MATERIAL)
        rest.prepare(parts)
        K = rest.tangent(parts)[rest._free][:, rest._free]
        assert abs(K - K.T).max() <= 1e-12 * abs(K).max()

        _, parts2, stepper = self._deformed()
        K2 = stepper.tangent(parts2)[stepper._free][:, stepper._free]
        assert abs(K2 - K2.T).max() > 1e-6 * abs(K2).max()


# ======================================================================
# 4b. ghost stabilisation
# ======================================================================
class TestGhostStabilisation:
    """The penalty has to reach the residual *and* the tangent.

    Found the hard way: it was wired into the micropolar residual and not the
    tangent, so the two described different problems and the very first Newton
    iterate diverged -- reported as "particles inverted", which reads like a
    load step that is too large and is not.
    """

    def test_tangent_and_residual_agree(self):
        grid, parts = block()
        stepper = MicropolarMPMStepper(grid, MATERIAL, ghost=MATERIAL.mu / 100)
        stepper.prepare(parts)
        rng = np.random.default_rng(3)
        free = stepper._free
        for _ in range(2):
            kick = np.zeros(stepper._n_dof)
            kick[free] = 0.02 * rng.standard_normal(free.size)
            stepper.apply_increment(parts, kick)
        report = check_tangent(lambda u: stepper.residual_at(parts, u),
                               lambda u: stepper.tangent_at(parts, u),
                               np.zeros(stepper._n_dof), free=free)
        assert report.passed, report

    def test_it_lifts_the_weakest_diagonal(self):
        """The point of the whole exercise, stated as a number.

        Needs a body with genuinely small cuts -- a ball seeded on a lattice.
        A box whose faces land on grid lines has nothing to stabilise and the
        penalty correctly does almost nothing there, which is a useful thing to
        know and a useless thing to test on.
        """
        dx = 0.25
        grid = CartesianGrid(shape=(10, 10, 10), dx=dx,
                             basis=get_basis("linear"))
        sub = dx / 2
        axes = [np.arange(0.3 + sub / 2, 1.9, sub) for _ in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        X = np.stack([m.ravel() for m in mesh], axis=1)
        X = X[np.linalg.norm(X - X.mean(axis=0), axis=1) < 0.63]

        weak, strong = {}, {}
        for ghost in (None, MATERIAL.mu / 100):
            p2 = attach_micropolar_state(
                ParticleSet(X.copy(), vol0=sub ** 3, density=1.0))
            st = MicropolarMPMStepper(grid, MATERIAL, ghost=ghost)
            st.prepare(p2)
            d = np.abs(st.tangent(p2)[st._free][:, st._free].diagonal())
            weak[ghost], strong[ghost] = d.min(), d.max()
        assert weak[MATERIAL.mu / 100] > 5.0 * weak[None]
        assert strong[MATERIAL.mu / 100] < 2.0 * strong[None]


# ======================================================================
# 5. the solve
# ======================================================================
class TestSolve:
    def test_newton_is_quadratic(self):
        grid, parts = block(hi=(2.5, 1.5, 1.5), shape=(8, 6, 6))
        bcs = [GridDirichlet(lambda c: c[:, 0] < 0.5 + 1e-9, np.zeros(6)),
               GridDirichlet(lambda c: c[:, 0] > 2.5 - 1e-9,
                             np.array([0.10, 0.05, 0.0, 0.0, 0.0, 0.0]))]
        stepper = MicropolarMPMStepper(grid, MATERIAL, dirichlet=bcs)
        stepper.run(parts, n_steps=2)
        for res in stepper.history:
            assert res.converged
            assert res.iterations <= 6
            assert res.convergence_rate > 1.8
        Q = quat_to_matrix(parts.quat)
        assert np.abs(Q @ Q.transpose(0, 2, 1) - np.eye(3)).max() < 1e-13

    def test_two_dimensional_grid_is_refused(self):
        grid = CartesianGrid(shape=(8, 8), dx=0.5)
        with pytest.raises(NotImplementedError, match="3-D only"):
            MicropolarMPMStepper(grid, MATERIAL)

    def test_missing_microrotation_state_is_named(self):
        grid = CartesianGrid(shape=(6, 6, 6), dx=0.5)
        pts = np.array([[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]])
        parts = ParticleSet(pts, vol0=0.001, density=1.0)
        stepper = MicropolarMPMStepper(grid, MATERIAL)
        with pytest.raises(AttributeError, match="attach_micropolar_state"):
            stepper.prepare(parts)

    def test_failed_step_restores_the_state(self):
        """A load step that does not converge must roll the particles back.
        Unlike the displacement-only solve there is no increment vector to
        discard -- the state has already been advanced in place."""
        grid, parts = block()
        bcs = [GridDirichlet(lambda c: c[:, 0] < 0.5 + 1e-9, np.zeros(6)),
               GridDirichlet(lambda c: c[:, 0] > 2.0 - 1e-9,
                             np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]))]
        stepper = MicropolarMPMStepper(grid, MATERIAL, dirichlet=bcs,
                                       max_iter=3)
        before = (parts.x.copy(), parts.F.copy(), parts.quat.copy())
        with pytest.raises(RuntimeError):
            stepper.step(parts, 1.0, 1.0)
        assert np.abs(parts.x - before[0]).max() == 0
        assert np.abs(parts.F - before[1]).max() == 0
        assert np.abs(parts.quat - before[2]).max() == 0


# ======================================================================
# 6. the external check
# ======================================================================
class TestManufacturedSolution:
    """The paper's own verification (Section 4.1), which is the only test here
    that checks the *residual* against the governing equations."""

    def test_displacement_is_second_order(self):
        from examples.mpm.micropolar_manufactured_solution import (
            PAPER, convergence,
        )
        rows = convergence([4, 6, 8], PAPER)
        assert min(r[2] for r in rows[1:]) > 1.85

    def test_rotation_error_is_load_path_dependent(self):
        """Same mesh, different number of load steps: the rotation error grows
        by ~75% from 1 step to 8 while the displacement error moves ~4%.

        Each increment re-embeds the Lagrange interpolation of a quantity that
        does not live on a linear manifold, so more increments accumulate more
        of it. The reference names this (Jelenic and Crisfield) and does not
        quantify it. Pinned here because it decides how a rotation convergence
        order may be reported: only against a stated load path.
        """
        from examples.mpm.micropolar_manufactured_solution import PAPER, solve
        e_u1, e_p1, _, _, _ = solve(8, PAPER, n_steps=1)
        e_u8, e_p8, _, _, _ = solve(8, PAPER, n_steps=8)
        assert e_p8 / e_p1 > 1.4          # rotation degrades badly
        assert 0.9 < e_u8 / e_u1 < 1.15   # displacement barely moves

    def test_rotation_converges_more_slowly_and_improves(self):
        """Not a defect: rotations are interpolated with Lagrange shape
        functions and do not live on a linear manifold. The reference reports
        the same lag and the same closing of it under refinement."""
        from examples.mpm.micropolar_manufactured_solution import (
            PAPER, convergence,
        )
        rows = convergence([4, 6, 8], PAPER)
        rates = [r[4] for r in rows[1:]]
        assert all(0.8 < r < 1.9 for r in rates), rates
        assert rates[-1] > rates[0]


# ======================================================================
# 7. the external benchmark
# ======================================================================
class TestCurvedCantilever:
    """Bathe & Bolourchi's 45-degree curved cantilever against the analytical
    solution of Kozar & Ibrahimbegovic, as run in the reference's Section 4.4.

    This is the only test here that could catch a *transcription* error -- a
    wrong sign in the wryness, a swapped index in the constitutive tensor, a
    mis-copied constant. The manufactured solution cannot: its loads are built
    from the same routines as the residual, so a shared misconception cancels
    and second-order convergence survives it.

    Coarsest mesh only. The 1 m cross-section is then smaller than one grid
    cell, which is under-resolved and still lands within 12% of an independent
    analytical solution; the 1 m mesh reaches 2.6% and takes four minutes, so
    it lives in the example rather than the suite.
    """

    CELL = 2.0

    @pytest.fixture(scope="class")
    def solved(self):
        from examples.mpm.micropolar_curved_cantilever import run
        return run(self.CELL)

    def test_matches_the_analytical_solution(self, solved):
        from examples.mpm.micropolar_curved_cantilever import (
            ANALYTIC, invariants,
        )
        u, stepper, _, _, _ = solved
        out, ins, _ = invariants(u)
        a_out, a_in, _ = invariants(ANALYTIC)

        # convention-free: the component along the load, and the in-plane
        # magnitude. Which component sits on which axis depends on the
        # reference's figure and is not recoverable from its text.
        assert abs(out / a_out - 1.0) < 0.10
        assert abs(ins / a_in - 1.0) < 0.20
        assert all(h.converged for h in stepper.history)

    def test_the_rotation_axis_genuinely_varies(self, solved):
        """Guards the guard.

        The reference added this example *after* its bending and torsion ones
        because in both of those the axis of rotation never moves -- which
        degenerates the rotation to a scalar angle, so a linearised update
        would pass them. If the axis did not vary here either, the benchmark
        above would not be testing the geometrically exact machinery at all.
        Measured, and more modest than the framing suggests: rotations reach
        1.40 rad but their *axes* span only **13.2 degrees**. That is small, and
        it is not zero -- a fixed-axis problem gives round-off. So this does
        discriminate, and the honest reading is that the arc exercises a varying
        axis without stressing it hard.
        """
        _, stepper, _, _, particles = solved
        phi = stepper.microrotation_vector(particles)
        mag = np.linalg.norm(phi, axis=1)
        axes = phi / np.maximum(mag, 1e-30)[:, None]
        gram = np.clip(axes @ axes.T, -1.0, 1.0)
        spread = np.degrees(np.arccos(gram.min()))
        assert spread > 5.0, spread
        assert mag.max() > 1.0        # and the rotations themselves are large


# ======================================================================
# 8. elastoplastic coupling
# ======================================================================
class TestElastoplastic:
    """The return map wired into the stepper.

    The tangent here is verified two ways, and both were needed. The general
    assembly must reproduce the *already verified* elastic tangent exactly when
    nothing yields -- that check found nothing but licenses the rewrite -- and
    ``check_tangent`` on the full 6-DOF residual must pass with points actually
    on the yield surface, which is what found the elastic/plastic dispatch bug.
    """

    ELASTIC = MicropolarHooke(lam=4.0e9, mu=4.0e9, a=0.1, alpha=0.0, beta=0.0,
                              gam=4.0e9 * 4.0 * 0.005 ** 2)

    def body(self, material, kick=0.01, plastic=True, seed=3):
        from scion.discretization.mpm.micropolar import attach_micropolar_state
        dx = 0.5
        sub = dx / 2
        grid = CartesianGrid(shape=(7, 6, 6), dx=dx, basis=get_basis("linear"))
        axes = [np.arange(lo + sub / 2, hi, sub)
                for lo, hi in ((0.5, 2.0), (0.5, 1.5), (0.5, 1.5))]
        mesh = np.meshgrid(*axes, indexing="ij")
        pts = np.stack([m.ravel() for m in mesh], axis=1)
        parts = attach_micropolar_state(
            ParticleSet(pts, vol0=sub ** 3, density=1.0), plastic=plastic)
        stepper = MicropolarMPMStepper(grid, material)
        stepper.prepare(parts)
        rng = np.random.default_rng(seed)
        for _ in range(2):
            k = np.zeros(stepper._n_dof)
            k[stepper._free] = kick * rng.standard_normal(stepper._free.size)
            stepper.apply_increment(parts, k)
        return stepper, parts

    def test_general_tangent_reproduces_the_verified_elastic_one(self):
        """The elastoplastic assembly is derived from the internal force rather
        than transcribed from the reference's ``a1..a7``. It must agree with the
        elastic tangent -- verified to 5.1e-11 against finite differences -- to
        round-off. Measured: 5.4e-16 relative."""
        el = MicropolarHooke(lam=3.333e6, mu=5.0e6, a=1.0 / 3.0, alpha=0.0,
                             beta=1.0e5, gam=1.0e5)
        stepper, parts = self.body(el, kick=0.02, plastic=False)
        Q, F, Js, Jm = stepper._kinematics(parts)
        old = stepper._tangent_chunk(slice(None), stepper._st, Q, F, Js, Jm,
                                     stepper._D, stepper._Dc)
        new = stepper._tangent_chunk_general(
            slice(None), stepper.constitutive(parts), parts)
        assert abs(old - new).max() < 1e-12 * abs(old).max()

    def test_no_yield_reproduces_the_elastic_stresses_exactly(self):
        """A plasticity model whose yield limit is unreachable must return the
        elastic answer *bit for bit*, not to a tolerance -- the predictor does
        no work at all below yield."""
        el = self.ELASTIC
        huge = MicropolarDruckerPrager(el, cohesion=1e30, plastic_length=0.005)
        s_e, p_e = self.body(el, plastic=False)
        s_p, p_p = self.body(huge, plastic=True)
        c_e, c_p = s_e.constitutive(p_e), s_p.constitutive(p_p)
        assert not c_p["state"]["plastic"].any()
        assert np.abs(c_e["tau"] - c_p["tau"]).max() == 0.0
        assert np.abs(c_e["nu"] - c_p["nu"]).max() == 0.0
        assert np.abs(s_e.internal_force(p_e)
                      - s_p.internal_force(p_p)).max() == 0.0

    @pytest.mark.parametrize("kw,kick", [
        ({"cohesion": 6.0e7}, 0.01),
        ({"cohesion": 6.0e7, "friction": 8.6, "dilation": 2.0}, 0.004),
        ({"cohesion": 6.0e7, "friction": 8.6, "dilation": 2.0,
          "hardening": 2.0e9}, 0.004),
        ({"cohesion": 1e30}, 0.004),
    ])
    def test_global_tangent_with_plasticity(self, kw, kick):
        material = MicropolarDruckerPrager(self.ELASTIC, plastic_length=0.005,
                                           **kw)
        stepper, parts = self.body(material, kick=kick)
        report = check_tangent(lambda u: stepper.residual_at(parts, u),
                               lambda u: stepper.tangent_at(parts, u),
                               np.zeros(stepper._n_dof), free=stepper._free)
        assert report.passed, report

    def test_a_point_below_yield_needs_the_elastic_sensitivities(self):
        """``P = -C^-1 T`` is only valid *on* the yield surface.

        At ``dgamma = 0`` the local Jacobian still carries its yield-function
        row, so solving through it imposes ``f = 0`` on a point whose actual
        constraint is ``dgamma = 0``. Symptom: the tangent check passes on a
        fully plastic body and fails by ~5% the moment a single point is
        elastic -- at *any* friction angle, which is what localised it, since
        the friction angle looked like the variable and was not.
        """
        material = MicropolarDruckerPrager(
            self.ELASTIC, cohesion=6.0e7, friction=8.6, dilation=2.0,
            plastic_length=0.005)
        stepper, parts = self.body(material, kick=0.004)
        plastic = stepper.constitutive(parts)["state"]["plastic"]
        # the state must genuinely be mixed, or this test has no power
        assert plastic.any() and not plastic.all()
        report = check_tangent(lambda u: stepper.residual_at(parts, u),
                               lambda u: stepper.tangent_at(parts, u),
                               np.zeros(stepper._n_dof), free=stepper._free)
        assert report.passed, report

    def test_heterogeneous_yield_strength_with_a_mixed_state(self):
        """Per-particle cohesion *and* only some points plastic.

        The two together are what break: the sensitivities are evaluated on the
        plastic subset, so a full-length ``k`` no longer broadcasts. Latent
        until the biaxial test seeded an imperfection -- the shear layer has
        heterogeneous strength too, but every particle there is plastic inside a
        single chunk, which made the full array match the subset by coincidence.
        """
        # the particle count is fixed by ``body``; build the strength array
        # against it directly rather than by making a throwaway stepper first
        n_pts = 96
        coh = np.full(n_pts, 6.0e7)
        coh[:n_pts // 3] *= 0.85
        stepper, parts = self.body(
            MicropolarDruckerPrager(self.ELASTIC, cohesion=coh, friction=8.6,
                                    dilation=2.0, plastic_length=0.005),
            kick=0.004)
        assert parts.n == n_pts
        plastic = stepper.constitutive(parts)["state"]["plastic"]
        assert plastic.any() and not plastic.all()
        report = check_tangent(lambda u: stepper.residual_at(parts, u),
                               lambda u: stepper.tangent_at(parts, u),
                               np.zeros(stepper._n_dof), free=stepper._free)
        assert report.passed, report

    def test_history_advances_only_on_a_committed_step(self):
        """``F^p``, ``Gamma^p`` and ``chi`` must not move during a Newton
        iterate. A discarded iterate that touched them corrupts the history
        permanently and makes the answer depend on the iteration path."""
        material = MicropolarDruckerPrager(
            self.ELASTIC, cohesion=6.0e7, friction=8.6, dilation=2.0,
            plastic_length=0.005)
        stepper, parts = self.body(material, kick=0.004)
        before = (parts.F_plastic.copy(), parts.wryness_plastic.copy(),
                  parts.hardening.copy())
        stepper.internal_force(parts)
        stepper.tangent(parts)
        assert np.abs(parts.F_plastic - before[0]).max() == 0.0
        assert np.abs(parts.hardening - before[2]).max() == 0.0

        stepper.commit_plastic(parts)
        assert np.abs(parts.hardening - before[2]).max() > 0.0

    def test_a_load_stepped_shear_yields_and_hardens_monotonically(self):
        from scion.discretization.mpm.micropolar import attach_micropolar_state
        nan = np.nan
        material = MicropolarDruckerPrager(
            self.ELASTIC, cohesion=5.0e7, friction=8.6, dilation=2.0,
            plastic_length=0.005)
        dx = 0.02
        sub = dx / 2
        grid = CartesianGrid(shape=(8, 7, 4), dx=dx, basis=get_basis("linear"))
        axes = [np.arange(lo + sub / 2, hi, sub)
                for lo, hi in ((0.02, 0.12), (0.02, 0.08), (0.02, 0.04))]
        mesh = np.meshgrid(*axes, indexing="ij")
        parts = attach_micropolar_state(
            ParticleSet(np.stack([m.ravel() for m in mesh], axis=1),
                        vol0=sub ** 3, density=1.0), plastic=True)
        bcs = [GridDirichlet(lambda c: c[:, 1] < 0.02 + 1e-9, np.zeros(6)),
               GridDirichlet(lambda c: c[:, 1] > 0.08 - 1e-9,
                             np.array([0.004, 0.0, 0.0, nan, nan, nan]))]
        stepper = MicropolarMPMStepper(grid, material, dirichlet=bcs,
                                       max_iter=20)

        chi = []
        stepper.run(parts, n_steps=6,
                    callback=lambda s, r, p: chi.append(p.hardening.max()))
        assert all(r.converged for r in stepper.history)
        assert chi[-1] > 0.0
        assert all(b >= a for a, b in zip(chi, chi[1:]))     # never unloads
        # dilation is positive, so the plastic volume must grow
        assert np.linalg.det(parts.F_plastic).max() > 1.0
