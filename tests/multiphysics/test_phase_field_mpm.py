"""Phase-field fracture driven from an MPM grid, classical and micropolar.

The classical half is an *adapter* over the finite-element path, so the test
that matters is that it agrees with that path on the same problem -- Kakouris &
Triantafyllou's whole claim (IJNME 2017) is that PF-MPM and PF-FEM give
practically identical load paths.

The micropolar half is not an adapter, because the micropolar solve does not go
through the quadrature tables, and the tests reflect that: they check the energy
decomposition that makes the split legal, and the degradation hook that carries
damage into the Newton solve.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from scion.core.logging import quiet
from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.discretization.mpm.micropolar import (MicropolarMPMStepper,
                                                 attach_micropolar_state)
from scion.materials.fracture import PhaseFieldModel
from scion.materials.micropolar import MicropolarHooke
from scion.multiphysics.micropolar_phase_field import MicropolarPhaseFieldMPM
from scion.multiphysics.phase_field_mpm import PhaseFieldMPMProblem
from scion.solvers.tangent_check import check_tangent

E_MOD, NU, GC = 2.1e5, 0.30, 2.7e-3


def plate(L=1.0, dx=0.1, ppc=2, pad=2, ell=0.15, dissipation="AT2",
          free_lateral=False):
    sub = dx / ppc
    grid = CartesianGrid(shape=(int(L / dx) + 2 * pad,) * 2, dx=dx,
                         origin=np.array([-pad * dx, -pad * dx]),
                         basis=get_basis("linear"))
    ax = [np.arange(sub / 2, L, sub)] * 2
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")],
                   axis=1)
    parts = ParticleSet(pts, vol0=sub ** 2, density=1.0)
    model = PhaseFieldModel(E_MOD, NU, GC, ell, split="spectral",
                            coupling="hybrid", dissipation=dissipation)
    p = PhaseFieldMPMProblem(grid, parts, model, thickness=1.0)
    xy = np.asarray(grid.node_coordinates())[:, :2]
    bot = np.flatnonzero(np.abs(xy[:, 1]) < 1e-9)
    top = np.flatnonzero(np.abs(xy[:, 1] - L) < 1e-9)
    # A dx that does not divide L lands no node on the edge, the selection is
    # empty, and every reaction reads exactly 0.0 -- which looks like a solve
    # returning nothing rather than a boundary that was never found.
    assert bot.size and top.size, f"dx={dx} puts no grid node on y=0 or y={L}"

    p.u.prescribe(bot, 1, 0.0)
    if free_lateral:
        # only enough to stop rigid translation, so the sides really are free
        # and sigma_xx = 0 -- otherwise the state sits between uniaxial stress
        # and uniaxial strain and no simple closed form applies
        p.u.prescribe(bot[:1], 0, 0.0)
    else:
        p.u.prescribe(bot, 0, 0.0)
        p.u.prescribe(top, 0, 0.0)
    return p, grid, top


class TestPhaseFieldMPM:
    def test_the_unsupported_nodes_make_it_solvable(self):
        """The regression guard on the one MPM-only failure.

        ``solve_second`` factorises the **full** phase-field system rather than
        its free part -- on a mesh every node carries support, so there is
        nothing to exclude. An MPM grid is larger than the body, those rows are
        identically zero, and without the identity block the factorisation is
        exactly singular and the staggered loop returns nan. Prescribing the
        dofs does *not* help: that path never consults the prescribed set.
        """
        p, grid, top = plate()
        assert p._unsupported.size > 0, "no unsupported nodes: test has no power"
        K = p._laplacian(p.model.gradient_coefficient())
        d = np.abs(K.diagonal())
        assert (d > 0).all(), "a zero diagonal makes the phase-field solve singular"

    def test_a_uniform_stretch_is_reproduced_exactly(self):
        """A patch test through the coupled machinery: prescribing ``u`` on the
        top edge must give exactly that strain at every particle."""
        p, grid, top = plate()
        p.u.prescribe(top, 1, 1.0e-4)
        with quiet():
            res = p.solve_step(1, 1.0, 0.0)
        assert res.converged
        eps = p.strains()
        assert_allclose(eps[:, 1, 1].mean(), 1.0e-4, rtol=1e-6)

    def test_the_undamaged_reaction_matches_plane_strain_elasticity(self):
        """An exact target, and self-contained.

        With no damage the plate is linear elastic in plane strain, so a
        prescribed stretch ``eps`` gives a net edge force ``E' eps W`` with
        ``E' = E/(1-nu^2)``. This replaces a cross-check against the
        finite-element path: that comparison is worth making, but `fem.build`
        constructs its own *graded* mesh refined near the crack, so the two
        discretisations are not the same problem and the agreement is not a
        property of the adapter. The controlled comparison -- identical
        parameters, 0.49% on the peak load -- belongs in an example.
        """
        eps = 1.0e-4
        # AT1, not AT2: AT2 has no elastic threshold and damages from the very
        # first increment, so an undamaged reference state does not exist in it
        p, grid, top = plate(dx=0.1, ell=0.15, dissipation="AT1",
                             free_lateral=True)
        p.u.prescribe(top, 1, eps)
        with quiet():
            p.solve_step(1, 1.0, 0.0)
        assert p.d.values.max() < 1e-6, "the plate must stay undamaged here"
        want = E_MOD / (1.0 - NU ** 2) * eps * 1.0
        assert_allclose(p.reaction(top, 1), want, rtol=0.02)


class TestMicropolarDegradation:
    """The hook that carries damage into the micropolar Newton solve."""

    @staticmethod
    def body():
        dx, sub = 0.5, 0.25
        grid = CartesianGrid(shape=(7, 6, 6), dx=dx, basis=get_basis("linear"))
        ax = [np.arange(lo + sub / 2, hi, sub)
              for lo, hi in ((0.5, 2.0), (0.5, 1.5), (0.5, 1.5))]
        pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")],
                       axis=1)
        parts = attach_micropolar_state(
            ParticleSet(pts, vol0=sub ** 3, density=1.0))
        el = MicropolarHooke(lam=4e9, mu=4e9, a=0.1, alpha=0.0, beta=0.0,
                             gam=4e9 * 4 * 0.005 ** 2)
        st = MicropolarMPMStepper(grid, el, max_iter=20)
        st.prepare(parts)
        rng = np.random.default_rng(3)
        k = np.zeros(st._n_dof)
        k[st._free] = 0.01 * rng.standard_normal(st._free.size)
        st.apply_increment(parts, k)
        return st, parts

    def test_a_uniform_degradation_scales_force_and_tangent_exactly(self):
        """Both the residual and the tangent are *linear* in the moduli, so a
        scalar multiplier per particle is exact -- which is why no per-particle
        modulus tensor is needed. Asserted to round-off, not to a tolerance."""
        st0, p0 = self.body()
        f0, K0 = st0.internal_force(p0), st0.tangent(p0)
        for c in (0.5, 0.1):
            st, p = self.body()
            st.degradation = np.full(p.n, c)
            f, K = st.internal_force(p), st.tangent(p)
            assert np.abs(f - c * f0).max() < 1e-14 * np.abs(f0).max()
            assert abs(K - c * K0).max() < 1e-14 * abs(K0).max()

    def test_default_is_inert(self):
        st0, p0 = self.body()
        st1, p1 = self.body()
        st1.degradation = None
        assert np.abs(st0.internal_force(p0)
                      - st1.internal_force(p1)).max() == 0.0

    def test_the_tangent_stays_consistent_under_non_uniform_damage(self):
        """The case that actually occurs. A uniform factor could hide an error
        that a spatially varying one exposes, since it would scale out of both
        sides of the comparison."""
        st, p = self.body()
        rng = np.random.default_rng(7)
        st.degradation = 0.2 + 0.6 * rng.random(p.n)
        report = check_tangent(lambda u: st.residual_at(p, u),
                               lambda u: st.tangent_at(p, u),
                               np.zeros(st._n_dof), free=st._free)
        assert report.passed, report


class TestMicropolarEnergyDecomposition:
    """What makes the tension/compression split legal on a Cosserat strain."""

    def test_the_energy_decouples_into_sym_skew_and_curvature(self):
        r"""``W = W_iso(sym E) + mu a |skew E|^2 + W_curv``, with no cross
        terms.

        This is the whole reason the standard splits apply: they presuppose a
        *symmetric* strain, and ``E = Q^T F - I`` is not symmetric. The
        decomposition is exact rather than a small-strain approximation, and it
        holds for every coupling number -- so the split goes on ``sym(E)``
        alone and the other two terms degrade unconditionally.
        """
        rng = np.random.default_rng(0)
        Ee = 0.01 * rng.standard_normal((64, 3, 3))
        Gam = 0.5 * rng.standard_normal((64, 3, 3))
        S = 0.5 * (Ee + Ee.transpose(0, 2, 1))
        A = 0.5 * (Ee - Ee.transpose(0, 2, 1))
        trS = np.einsum("nii->n", S)
        for N in (0.1, 0.35, 0.9):
            el = MicropolarHooke.from_technical(E=E_MOD, nu=NU, N=N,
                                                l_t=0.02, l_b=0.01, psi=1.5)
            _, couple = el.stresses(np.zeros_like(Ee), Gam)
            split = (0.5 * el.lam * trS ** 2
                     + el.mu * np.einsum("nij,nij->n", S, S)
                     + el.mu * el.a * np.einsum("nij,nij->n", A, A)
                     + 0.5 * np.einsum("nij,nij->n", couple, Gam))
            full = el.energy(Ee, Gam)
            assert np.abs(full - split).max() < 1e-12 * np.abs(full).max()

    def test_a_homogeneous_deformation_has_no_micropolar_driving_force(self):
        """Both micropolar terms vanish identically under uniform strain -- no
        relative rotation, no wryness. So the internal length can only enter
        through *gradients*, and a uniform tension test cannot exercise this
        model at all. Not a defect: a statement about where ``l_b`` can matter.
        """
        dx, sub, nz, pad = 0.1, 0.05, 1, 3
        grid = CartesianGrid(shape=(4 + 2 * pad, 4 + 2 * pad, nz + 2 * pad),
                             dx=dx, origin=np.array([-pad * dx] * 3),
                             basis=get_basis("linear"))
        ax = [np.arange(sub / 2, e, sub) for e in (0.4, 0.4, nz * dx)]
        pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")],
                       axis=1)
        parts = attach_micropolar_state(
            ParticleSet(pts, vol0=sub ** 3, density=1.0))
        el = MicropolarHooke.from_technical(E=E_MOD, nu=NU, N=0.35,
                                            l_t=0.08, l_b=0.04, psi=1.5)
        st = MicropolarMPMStepper(grid, el, max_iter=20)
        st.prepare(parts)
        # a uniform stretch applied directly to the grid
        xy = np.asarray(grid.node_coordinates())
        inc = np.zeros(st._n_dof)
        inc[1::6] = 0.01 * xy[:, 1]
        st.apply_increment(parts, inc)

        model = PhaseFieldModel(E_MOD, NU, GC, 0.08, split="spectral",
                                coupling="hybrid", dissipation="AT2")
        pf = MicropolarPhaseFieldMPM(st, parts, model)
        shares = pf.driving_shares()
        assert shares["relative_rotation"] < 1e-12
        assert shares["curvature"] < 1e-12
        assert_allclose(shares["symmetric"], 1.0, atol=1e-10)


class TestRefresh:
    """Rebuilding the tables when particles move."""

    def test_refresh_updates_the_weights(self):
        """The tables go stale when particles move, and nothing errors if they
        are not rebuilt -- stale shape functions still assemble and converge,
        to the wrong problem.

        The check is on ``N``, not on ``conn``: node *ids* change only when a
        particle crosses a cell, and any crossing at the body's edge also
        changes the support set. The shape-function *values* move with any
        displacement at all, which is what makes them the stale quantity."""
        p, grid, top = plate(dx=0.1)
        N0, conn0 = p.tables.N.copy(), p.tables.conn.copy()
        p.particles.x[:, 0] += 0.02        # a fifth of a cell: no crossing
        p.refresh()
        assert np.array_equal(conn0, p.tables.conn), "unexpected cell crossing"
        assert np.abs(p.tables.N - N0).max() > 1e-3, "weights did not update"
        assert p.N is p.tables.N, "the alias still points at the old table"

    def test_refresh_refuses_when_a_node_gains_support(self):
        """``Field`` can prescribe a dof but cannot release one, so a node
        pinned for having no support would stay pinned once it gains some --
        clamping material at the body's edge while the solve carried on
        converging. Refused loudly rather than carried silently."""
        p, grid, top = plate(dx=0.1)
        p.particles.x[:, 0] += 0.35
        with pytest.raises(NotImplementedError, match="gained particle support"):
            p.refresh()


def notched_micropolar(L=0.4, dx=0.1, ppc=2, ell=0.2, l_b=0.1, u_top=1.0e-3):
    """A small notched slab that genuinely staggers, for the history tests.

    One cell thick with the plane-strain components constrained, and the
    elongation split between the two edges rather than clamping one and driving
    the other -- on a fixed background grid those are not the same discrete
    problem, because a drifting body's weight operator stops commuting with the
    mirror.
    """
    sub, pad, zpad, nz = dx / ppc, 2, 2, 1
    n = int(round(L / dx))
    grid = CartesianGrid(shape=(n + 2 * pad, n + 2 * pad, nz + 2 * zpad),
                         dx=dx,
                         origin=np.array([-pad * dx, -pad * dx, -zpad * dx]),
                         basis=get_basis("linear"))
    ax = [np.arange(sub / 2, e, sub) for e in (L, L, nz * dx)]
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")],
                   axis=1)
    parts = attach_micropolar_state(
        ParticleSet(pts, vol0=sub ** 3, density=1.0))
    el = MicropolarHooke.from_technical(E=E_MOD, nu=NU, N=0.35,
                                        l_t=2.0 * l_b, l_b=l_b, psi=1.5)
    nan = np.nan
    bcs = [
        GridDirichlet(lambda c: c[:, 1] < 1e-9,
                      np.array([0.0, -0.5 * u_top, nan, nan, nan, nan])),
        GridDirichlet(lambda c: np.abs(c[:, 1] - L) < 1e-9,
                      np.array([0.0, 0.5 * u_top, nan, nan, nan, nan])),
        GridDirichlet(lambda c: np.ones(len(c), bool),
                      np.array([nan, nan, 0.0, 0.0, 0.0, nan])),
    ]
    st = MicropolarMPMStepper(grid, el, dirichlet=bcs, max_iter=25)
    st.prepare(parts)
    model = PhaseFieldModel(E_MOD, NU, GC, ell, split="spectral",
                            coupling="hybrid", dissipation="AT2")
    pf = MicropolarPhaseFieldMPM(st, parts, model, max_stagger=30)
    x = parts.x[:, :2]
    behind = x[:, 0] <= L / 2
    dist = np.where(behind, np.abs(x[:, 1] - L / 2),
                    np.hypot(x[:, 0] - L / 2, x[:, 1] - L / 2))
    width = max(0.25 * ell, dx)
    inside = dist <= width
    H0 = np.zeros(parts.n)
    H0[inside] = 1.0e3 * GC / (2.0 * ell) * (1.0 - dist[inside] / width)
    pf.H = H0
    return pf, st, parts


class TestStaggeredMonotoneBounds:
    """Both monotone bounds belong to the load step, not to the sweep.

    The history variable is defined as a maximum over the *solution*
    trajectory. Taking ``H = max(H, W+)`` in place inside the staggered loop
    maximises over the iterates as well, which are not points on that
    trajectory, and clamping ``d`` against the previous sweep does the same to
    the damage. The cost is not just a bias towards damage: the converged
    answer then depends on how the stagger loop reached it, so changing
    ``damage_tol`` or the starting ``d`` changes the solution of the same
    boundary value problem.

    Both tests fail against the pre-fix code by factors of 40 to 1600 on most
    loaded particles -- an early sweep carries less damage, so at the same
    prescribed displacement it is stiffer and its ``W+`` at the notch is far
    larger than the converged one, and that value was being locked in.
    """

    def test_history_is_a_maximum_over_converged_states_only(self):
        pf, st, parts = notched_micropolar()
        H_n = pf.H.copy()
        with quiet():
            info = pf.solve_step(1.0, 1.0)
        # the loop has to have iterated, or the invariant is vacuous
        assert info["stagger"] > 1, "no staggering, so nothing is being tested"
        assert info["converged"] and info["equilibrium"]
        assert_allclose(pf.H, np.maximum(H_n, pf.driving_energy()),
                        rtol=1e-12, atol=0.0)

    def test_the_damage_bound_is_the_last_converged_step(self):
        """The converged damage must be reproducible from the committed bound.

        One phase-field solve at the converged kinematics, started from
        ``d^n``, has to land back on the converged field. AT2 is linear at
        fixed history, so that solve is exact and independent of its starting
        guess -- which makes this a statement about the *bound* and not about
        the iteration. Clamping against the previous sweep instead leaves the
        converged ``d`` carrying whatever an intermediate sweep overshot, and
        it cannot be recovered this way.
        """
        pf, st, parts = notched_micropolar()
        d_n = pf.d.copy()
        with quiet():
            info = pf.solve_step(1.0, 1.0)
        assert info["stagger"] > 1
        d_conv, H_conv = pf.d.copy(), pf.H.copy()
        assert (d_conv >= d_n - 1e-14).all(), "irreversibility violated"
        pf.d = d_n.copy()
        with quiet():
            pf.solve_damage()
        assert_allclose(pf.H, H_conv, rtol=1e-12, atol=0.0)
        assert_allclose(pf.d, d_conv, rtol=0.0, atol=1e-12)
