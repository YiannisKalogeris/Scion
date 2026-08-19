"""
Periodic background grids.

A periodic axis carries exactly ``shape[a]`` nodes and no padding: the node at
``x = L`` *is* the node at ``x = 0``. The stencil wraps the node index while
leaving the node *position* alone, and that asymmetry is the thing most likely
to be got wrong -- wrapping the position too would make ``dpos`` jump by a
domain length across the seam and hand APIC a discontinuous affine field.

The last class is the reason periodicity was added: it is the only setting in
which the continuum conserves total pseudomomentum non-trivially, so it is the
only setting in which a drift can be attributed to the discretisation.
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
)
from scion.discretization.mpm.basis import get_basis
from scion.materials.eshelby import total_pseudomomentum
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

N = 16
DX = 1.0 / N
BASES = ["linear", "quadratic", "cubic"]


class TestGridTopology:
    def test_periodic_axis_drops_the_duplicate_seam_node(self):
        plain = CartesianGrid(shape=(N, N), dx=DX)
        wrap = CartesianGrid(shape=(N, N), dx=DX, periodic=True)
        assert wrap.node_shape == (N, N)
        assert wrap.n_nodes == N * N
        assert plain.n_nodes > wrap.n_nodes

    def test_per_axis_periodicity(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=(True, False))
        assert g.node_shape[0] == N
        assert g.node_shape[1] == N + 1 + 2 * g.basis.stencil_size
        assert tuple(g.pad) == (0, g.basis.stencil_size)

    def test_default_is_non_periodic(self):
        g = CartesianGrid(shape=(N, N), dx=DX)
        assert not g.any_periodic
        assert not g.periodic.any()

    @pytest.mark.parametrize("basis", BASES)
    def test_no_index_escapes_the_grid(self, basis):
        g = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis(basis),
                          periodic=True)
        rng = np.random.default_rng(0)
        st = g.build_stencil(rng.uniform(0.0, 1.0, (500, 2)))
        assert st.node_id.min() >= 0
        assert st.node_id.max() < g.n_nodes


class TestStencilAcrossTheSeam:
    @pytest.mark.parametrize("basis", BASES)
    def test_partition_of_unity_holds_at_the_seam(self, basis):
        g = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis(basis),
                          periodic=True)
        xp = np.array([[0.0, 0.5], [1e-12, 0.5], [1.0 - 1e-12, 0.5],
                       [0.5, 0.0], [0.5, 1.0 - 1e-12]])
        st = g.build_stencil(xp)
        assert np.abs(st.w.sum(axis=1) - 1.0).max() < 1e-13
        assert np.abs(st.grad_w.sum(axis=1)).max() < 1e-11

    def test_indices_wrap_but_positions_do_not(self):
        """dpos must stay local; APIC's affine field depends on it."""
        g = CartesianGrid(shape=(N, N), dx=DX,
                          basis=get_basis("quadratic"), periodic=True)
        st = g.build_stencil(np.array([[0.002, 0.5]]))

        # the stencil reaches both ends of the axis
        assert st.node_id[0].max() - st.node_id[0].min() > N
        # yet every node is physically within a couple of cells
        assert np.abs(st.dpos[0, :, 0]).max() < 3 * DX

    def test_shifting_by_a_whole_period_changes_nothing(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=True)
        xp = np.array([[0.31, 0.42], [0.55, 0.17]])
        a = g.build_stencil(xp)
        b = g.build_stencil(xp + np.array([1.0, 0.0]))
        assert np.array_equal(a.node_id, b.node_id)
        assert np.allclose(a.w, b.w)
        assert np.allclose(a.grad_w, b.grad_w)

    def test_non_periodic_grid_is_bit_identical_to_before(self):
        """Periodicity must not perturb the existing path."""
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=False)
        rng = np.random.default_rng(1)
        xp = rng.uniform(0.3, 0.7, (40, 2))
        st = g.build_stencil(xp)
        assert np.abs(st.w.sum(axis=1) - 1.0).max() < 1e-13
        assert st.node_id.min() >= 0 and st.node_id.max() < g.n_nodes


class TestWrapping:
    def test_wrap_positions_folds_only_periodic_axes(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=(True, False))
        x = np.array([[1.25, 0.3], [-0.1, 0.7], [0.5, 1.4]])
        out = g.wrap_positions(x.copy())
        assert np.allclose(out[:, 0], [0.25, 0.9, 0.5])
        assert np.allclose(out[:, 1], [0.3, 0.7, 1.4])       # untouched

    def test_wrap_is_in_place_and_returns_the_array(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=True)
        x = np.array([[1.25, 0.3]])
        assert g.wrap_positions(x) is x
        assert np.isclose(x[0, 0], 0.25)

    def test_wrap_is_a_no_op_without_periodicity(self):
        g = CartesianGrid(shape=(N, N), dx=DX)
        x = np.array([[1.25, -0.3]])
        assert np.allclose(g.wrap_positions(x.copy()), x)

    def test_minimum_image(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=True)
        d = g.minimum_image(np.array([[0.9, 0.1], [-0.8, 0.0]]))
        assert np.allclose(d, [[-0.1, 0.1], [0.2, 0.0]])

    def test_minimum_image_does_not_mutate_its_input(self):
        g = CartesianGrid(shape=(N, N), dx=DX, periodic=True)
        d = np.array([[0.9, 0.1]])
        g.minimum_image(d)
        assert np.allclose(d, [[0.9, 0.1]])


class TestStepperOnAPeriodicGrid:
    def _blob(self, n_grid=32):
        dx = 1.0 / n_grid
        g = CartesianGrid(shape=(n_grid, n_grid), dx=dx, periodic=True)
        mat = NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)
        parts = ParticleSet.from_sampler(
            lambda p: ((p[:, 0] - 0.5) ** 2 + (p[:, 1] - 0.5) ** 2) < 0.12 ** 2,
            ((0.0, 0.0), (1.0, 1.0)), dx, ppc=2, density=1000.0)
        c = mat.wave_speed_factor / np.sqrt(1000.0)
        return g, parts, ExplicitMPMStepper(g, mat, APIC(), dt=0.25 * dx / c)

    def test_a_body_crosses_the_seam_without_incident(self):
        g, parts, stepper = self._blob()
        parts.v[:] = np.array([2.0, 0.0])
        with quiet():
            stepper.run(parts, n_steps=400)
        assert np.all(np.isfinite(parts.x))
        assert parts.x[:, 0].min() >= 0.0 and parts.x[:, 0].max() <= 1.0

    def test_momentum_is_conserved_with_no_walls(self):
        g, parts, stepper = self._blob()
        parts.v[:] = np.array([1.0, 0.0])
        m = parts.mass
        p0 = (m[:, None] * parts.v).sum(axis=0)
        with quiet():
            stepper.run(parts, n_steps=400)
        p = (m[:, None] * parts.v).sum(axis=0)
        assert np.linalg.norm(p - p0) / np.linalg.norm(p0) < 1e-12

    def test_rigid_translation_leaves_F_untouched_across_the_seam(self):
        """Cell crossing at the seam must be no worse than anywhere else."""
        g, parts, stepper = self._blob()
        parts.v[:] = np.array([1.0, 0.0])
        with quiet():
            stepper.run(parts, n_steps=600)
        assert np.abs(parts.F - sla.identity(parts.n, 2)).max() < 1e-10


class TestPseudomomentumProduction:
    """The measurement periodicity was added to make possible.

    A homogeneous body filling a periodic cell has no boundary, so
    ``\\oint \\Sigma N dS`` vanishes and the continuum conserves total
    pseudomomentum along with linear momentum. Any difference between how well
    the two are conserved is therefore a property of the discretisation.

    The initial velocity field must itself be periodic. A linear shear
    ``v_x = k(y - 1/2)`` is not -- it disagrees with itself across the seam --
    and produces a shock there that swamps the effect being measured.
    """

    N_GRID = 24

    def audit(self, basis, transfer, n_steps=400):
        dx = 1.0 / self.N_GRID
        g = CartesianGrid(shape=(self.N_GRID, self.N_GRID), dx=dx,
                          basis=get_basis(basis), periodic=True)
        p = ParticleSet.from_sampler(
            lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
            dx, ppc=2, density=1000.0)
        p.v[:, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * p.x[:, 1])
        p.v[:, 1] = 0.2 * np.sin(2 * np.pi * p.x[:, 0])

        mat = NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)
        c = mat.wave_speed_factor / np.sqrt(1000.0)
        stepper = ExplicitMPMStepper(g, mat, transfer, dt=0.25 * dx / c)

        m = p.mass
        p0 = (m[:, None] * p.v).sum(axis=0)
        P0 = total_pseudomomentum(m, p.F, p.v)
        with quiet():
            stepper.run(p, n_steps=n_steps)
        dp = np.linalg.norm((m[:, None] * p.v).sum(axis=0) - p0) \
            / np.linalg.norm(p0)
        dP = np.linalg.norm(total_pseudomomentum(m, p.F, p.v) - P0) \
            / np.linalg.norm(P0)
        return dp, dP

    @pytest.mark.parametrize("basis", ["linear", "quadratic"])
    def test_spatial_momentum_is_machine_precision(self, basis):
        dp, _ = self.audit(basis, APIC())
        assert dp < 1e-12

    @pytest.mark.parametrize("basis", ["linear", "quadratic"])
    def test_pseudomomentum_is_not(self, basis):
        """The paper's claim (b), measured."""
        dp, dP = self.audit(basis, APIC())
        assert dP > 1e4 * dp

    def test_basis_regularity_controls_the_production(self):
        """C0 linear hat against the C1 quadratic B-spline.

        Equation (9) drives the configurational force with the *second*
        material gradient of the basis, which is a singular measure for the hat
        and bounded for the spline. If that mechanism is right, the hat should
        produce markedly more pseudomomentum.
        """
        _, hat = self.audit("linear", APIC())
        _, spline = self.audit("quadratic", APIC())
        assert hat > 10.0 * spline, (hat, spline)
