"""
Section 4: conservation properties as Noether corollaries.

The claim is that casting the transfers in a discrete Lagrange-d'Alembert
framework makes APIC's and PolyPIC's conservation properties *corollaries* of
which symmetry group acts on the enlarged space, rather than numerical
observations. The checkable consequence is the spatial rotation group: APIC's
affine augmentation should make the corresponding momentum map -- total angular
momentum -- exactly conserved, and PIC, which has no augmentation, should not.

The measurement trap, and the reason this file exists
-----------------------------------------------------
``sum_p m_p x_p ^ v_p`` is **not** the total angular momentum of an APIC state.
The augmentation stores spin in ``C_p``, so the total is

    L = sum_p m_p ( x_p ^ v_p  +  tr(D_p) omega_p )

with ``omega_p`` the antisymmetric part of ``C_p`` and ``D_p`` the APIC
normalisation. Measuring only the first term reports APIC losing ~1% per few
hundred steps and looking no better than FLIP. Counting both gives 1e-14. The
enlargement is exactly where the conserved quantity lives, which is the
paper's point stated as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC, CartesianGrid, ExplicitMPMStepper, FLIP, MLSMPM, PIC, ParticleSet,
)
from scion.discretization.mpm.basis import get_basis
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

RHO = 1000.0
ORIGIN = np.array([0.5, 0.5])
SPIN = 4.0


class NullMaterial:
    """Zero stress, to isolate the transfer from the internal-force step."""

    wave_speed_factor = float(np.sqrt(5.0e4))

    def pk1(self, F):
        return np.zeros_like(F), np.ones(len(F))

    def energy(self, F):
        return np.zeros(len(F))


def bulk_angular_momentum(p):
    d = p.x - ORIGIN
    return float((p.mass * (d[:, 0] * p.v[:, 1] - d[:, 1] * p.v[:, 0])).sum())


def affine_angular_momentum(p, grid):
    """The half APIC stores in ``C``: ``m tr(D) omega``."""
    trD = 2.0 / grid.basis.apic_inv_D(grid.dx)          # 2-D
    two_omega = p.C[:, 1, 0] - p.C[:, 0, 1]
    return float((p.mass * trD * two_omega * 0.5).sum())


def spin_up(basis="quadratic", n=48):
    dx = 1.0 / n
    grid = CartesianGrid(shape=(n, n), dx=dx, basis=get_basis(basis))
    p = ParticleSet.from_sampler(
        lambda q: ((q[:, 0] - 0.5) ** 2 + (q[:, 1] - 0.5) ** 2) < 0.14 ** 2,
        ((0.0, 0.0), (1.0, 1.0)), dx, ppc=2, density=RHO)
    d = p.x - ORIGIN
    p.v[:, 0] = -SPIN * d[:, 1]
    p.v[:, 1] = SPIN * d[:, 0]
    p.C = np.tile(np.array([[0.0, -SPIN], [SPIN, 0.0]]), (p.n, 1, 1))
    return grid, p


def drift(transfer, material, steps=300, basis="quadratic"):
    """Returns (bulk-only drift, total drift), both relative."""
    grid, p = spin_up(basis)
    c = material.wave_speed_factor / np.sqrt(RHO)
    s = ExplicitMPMStepper(grid, material, transfer, dt=0.25 * grid.dx / c)

    b0 = bulk_angular_momentum(p)
    total0 = b0 + affine_angular_momentum(p, grid)
    with quiet():
        s.run(p, n_steps=steps)
    b1 = bulk_angular_momentum(p)
    total1 = b1 + affine_angular_momentum(p, grid)
    return abs(b1 - b0) / abs(total0), abs(total1 - total0) / abs(total0)


class TestTheMeasurementTrap:
    def test_bulk_alone_understates_APIC(self):
        """Counting only ``m x ^ v`` makes APIC look like it loses ~1%."""
        bulk, total = drift(APIC(), NullMaterial())
        assert bulk > 1e-3
        assert total < 1e-10
        assert bulk > 1e6 * total

    def test_the_affine_term_is_where_the_rest_lives(self):
        grid, p = spin_up()
        assert abs(affine_angular_momentum(p, grid)) > 0.0


class TestRotationGroupCorollary:
    """APIC's augmentation makes the rotation momentum map conserved."""

    def test_apic_conserves_total_angular_momentum(self):
        _, total = drift(APIC(), NullMaterial())
        assert total < 1e-10

    def test_mls_mpm_inherits_it(self):
        """MLS-MPM is APIC transfers, so the corollary must carry over."""
        _, total = drift(MLSMPM(), NullMaterial())
        assert total < 1e-10

    def test_pic_has_no_augmentation_and_destroys_it(self):
        _, total = drift(PIC(), NullMaterial())
        assert total > 0.5

    def test_pure_flip_conserves_it_in_the_transfer(self):
        """FLIP transfers the increment, so it carries the spin implicitly."""
        _, total = drift(FLIP(alpha=1.0), NullMaterial())
        assert total < 1e-10

    @pytest.mark.parametrize("basis", ["quadratic", "cubic"])
    def test_apic_conserves_on_the_bsplines(self, basis):
        """Only these have a position-independent second moment, which is what
        makes the constant ``apic_inv_D`` exact."""
        _, total = drift(APIC(), NullMaterial(), steps=150, basis=basis)
        assert total < 1e-9, basis

    def test_apic_does_not_conserve_on_the_linear_hat(self):
        """Recorded deliberately: the hat's second moment is ``xi(1-xi) dx^2``,
        which varies by 46% across a cell, so no constant ``D`` can be right and
        the corollary does not apply. 53% of the angular momentum is gone in 150
        steps. APIC is specified with a quadratic B-spline for this reason."""
        _, total = drift(APIC(), NullMaterial(), steps=150, basis="linear")
        assert total > 0.1

    def test_second_moment_is_constant_only_for_the_bsplines(self):
        from scion.discretization.mpm.basis import get_basis
        rng = np.random.default_rng(0)
        spread = {}
        for name in ("linear", "quadratic", "cubic"):
            grid = CartesianGrid(shape=(32, 32), dx=1.0 / 32,
                                 basis=get_basis(name))
            st = grid.build_stencil(rng.uniform(0.3, 0.7, (300, 2)))
            m2 = np.einsum("pk,pk,pk->p", st.w,
                           st.dpos[..., 0], st.dpos[..., 0])
            spread[name] = m2.std() / m2.mean()
        assert spread["quadratic"] < 1e-12
        assert spread["cubic"] < 1e-12
        assert spread["linear"] > 0.1


class TestPolyPICCorollary:
    r"""Section 4 claims the corollary covers PolyPIC as well as APIC.

    Derivation: for a tensor-product basis the bilinear mode carries no angular
    momentum, because ``sum_i w z_x^2 z_y = (sum w_x z_x^2)(sum w_y z_y) = D
    \cdot 0``. PolyPIC's degree-one momentum map therefore reduces to APIC's,
    and is conserved for the same reason.
    """

    def seed(self, grid, p, spin):
        """Initialise the polynomial coefficients from the affine field."""
        from scion.discretization.mpm import PolyPIC
        tr = PolyPIC(1)
        st = grid.build_stencil(p.x)
        tr._modes(grid, st)
        ex = tr._exponents
        c = np.zeros((p.n, len(ex), 2))
        c[:, ex.index((0, 0)), :] = p.v
        for a in range(2):
            e = [0, 0]
            e[a] = 1
            c[:, ex.index(tuple(e)), :] = p.C[:, :, a] * grid.dx
        p.poly_c = c
        p.poly_pullback = np.tile(np.eye(2), (p.n, 1, 1))
        return tr, st

    def test_bilinear_mode_carries_no_angular_momentum(self):
        """The separability argument, measured."""
        grid, p = spin_up()
        st = grid.build_stencil(p.x)
        z = st.dpos
        mixed = np.einsum("pk,pk,pk,pk->p", st.w, z[..., 0], z[..., 0], z[..., 1])
        assert np.abs(mixed).max() < 1e-18

    def test_polypic_conserves_the_same_map_as_apic(self):
        grid, p = spin_up()
        tr, st = self.seed(grid, p, SPIN)
        L0 = bulk_angular_momentum(p) + affine_angular_momentum(p, grid)

        m, mv = tr.p2g(p, grid, st)
        active = m > 1e-12
        gv = np.zeros_like(mv)
        gv[active] = mv[active] / m[active, None]
        tr.g2p(p, grid, st, gv, 0.0)

        L1 = bulk_angular_momentum(p) + affine_angular_momentum(p, grid)
        assert abs(L1 - L0) / abs(L0) < 1e-12

    def test_cold_start_no_longer_discards_the_affine_state(self):
        """Falling back to PIC momentum when ``poly_c`` is unset cost 2.2% of
        the angular momentum on the first step -- exactly the loss PolyPIC
        exists to avoid."""
        from scion.discretization.mpm import PolyPIC
        grid, p = spin_up()
        tr = PolyPIC(1)
        st = grid.build_stencil(p.x)
        L0 = bulk_angular_momentum(p) + affine_angular_momentum(p, grid)

        m, mv = tr.p2g(p, grid, st)          # no poly_c set: the cold path
        active = m > 1e-12
        gv = np.zeros_like(mv)
        gv[active] = mv[active] / m[active, None]
        tr.g2p(p, grid, st, gv, 0.0)

        L1 = bulk_angular_momentum(p) + affine_angular_momentum(p, grid)
        assert abs(L1 - L0) / abs(L0) < 1e-12


class TestPseudomomentumIsConsistent:
    """Measured: the pseudomomentum defect *does* vanish under refinement.

    Recorded because the opposite would have been the stronger result. At fixed
    physical time in a periodic cell the defect converges at rate 1.4 to 4.8
    across bases and transfers, so MPM is pseudomomentum-consistent and the
    non-conservation is ordinary discretisation error rather than a structural
    cost of the Eulerian grid.
    """

    def test_defect_decreases_under_refinement(self):
        from scion.core.logging import quiet
        from scion.discretization.mpm import CartesianGrid, ParticleSet
        from scion.materials.eshelby import total_pseudomomentum

        def defect(n):
            dx = 1.0 / n
            grid = CartesianGrid(shape=(n, n), dx=dx,
                                 basis=get_basis("quadratic"), periodic=True)
            p = ParticleSet.from_sampler(
                lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
                dx, ppc=2, density=RHO)
            p.v[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * p.x[:, 1])
            p.v[:, 1] = 0.2 * np.sin(2 * np.pi * p.x[:, 0])
            mat = NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)
            c = mat.wave_speed_factor / np.sqrt(RHO)
            dt = 0.25 * dx / c
            s = ExplicitMPMStepper(grid, mat, APIC(), dt=dt)
            m = p.mass
            scale = m.sum() * np.sqrt((p.v ** 2).sum(axis=1).mean())
            P0 = total_pseudomomentum(m, p.F, p.v)
            with quiet():
                s.run(p, n_steps=int(round(0.05 / dt)))
            return np.linalg.norm(
                total_pseudomomentum(m, p.F, p.v) - P0) / scale

        coarse, fine = defect(12), defect(32)
        assert fine < 0.2 * coarse


class TestThroughTheForceStep:
    """The stronger statement: the corollary survives the internal force."""

    def material(self):
        return NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)

    def test_apic_still_conserves_with_stress(self):
        _, total = drift(APIC(), self.material())
        assert total < 1e-10

    def test_mls_mpm_still_conserves_with_stress(self):
        _, total = drift(MLSMPM(), self.material())
        assert total < 1e-10

    def test_flip_does_not_survive_the_force_step(self):
        """Exact through the transfer, not through the force -- which is the
        practical difference between FLIP and APIC."""
        _, transfer_only = drift(FLIP(alpha=1.0), NullMaterial())
        _, with_stress = drift(FLIP(alpha=1.0), self.material())
        assert transfer_only < 1e-10
        assert with_stress > 1e-4

    def test_pic_is_hopeless_either_way(self):
        _, total = drift(PIC(), self.material())
        assert total > 0.5
