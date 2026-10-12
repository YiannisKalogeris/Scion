"""
PolyPIC and MLS-MPM transfers.

Two of the four variants the configurational-forces paper ranks. The other two,
CPDI and BSCPDI, are particle-domain schemes and cannot be expressed through the
tensor-product ``Basis`` interface at all: a convected parallelogram stops being
axis-aligned the moment ``F`` rotates, so its weight is not separable.

What is asserted here is bounded by what was actually verified. PolyPIC is
supported at degree 1 only; degree 2 and above are refused, and that refusal is
tested rather than papered over.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC, CartesianGrid, ExplicitMPMStepper, MLSMPM, ParticleSet, PolyPIC,
)
from scion.discretization.mpm.basis import get_basis
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

RHO = 1000.0
N, DX = 48, 1.0 / 48


def material():
    return NeoHookeanCompressible.from_E_nu(5.0e4, 0.3)


def blob(basis="quadratic", transfer=None, flow="affine"):
    grid = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis(basis))
    p = ParticleSet.from_sampler(
        lambda q: ((q[:, 0] - 0.5) ** 2 + (q[:, 1] - 0.5) ** 2) < 0.15 ** 2,
        ((0.0, 0.0), (1.0, 1.0)), DX, ppc=2, density=RHO)
    d = p.x - np.array([0.5, 0.5])
    if flow == "affine":
        p.v[:, 0] = -4.0 * d[:, 1] + 0.3
        p.v[:, 1] = 4.0 * d[:, 0]
    else:
        p.v[:, 0] = 6.0 * d[:, 1] ** 2
        p.v[:, 1] = 6.0 * d[:, 0] ** 2
    mat = material()
    c = mat.wave_speed_factor / np.sqrt(RHO)
    return grid, p, ExplicitMPMStepper(grid, mat, transfer or APIC(),
                                       dt=0.25 * DX / c)


class TestMLSKernel:
    """MLS-MPM is defined by replacing grad_w with w D^-1 (x_i - x_p)."""

    def kernels(self, basis):
        grid = CartesianGrid(shape=(32, 32), dx=1.0 / 32,
                             basis=get_basis(basis))
        rng = np.random.default_rng(1)
        xp = rng.uniform(0.3, 0.7, (200, 2))
        st = grid.build_stencil(xp)
        return grid, st, MLSMPM().force_kernel(grid, st)

    @pytest.mark.parametrize("basis", ["quadratic", "cubic"])
    def test_reproduces_a_linear_field_gradient(self, basis):
        """The MLS reproducing property, and why the substitution is legal."""
        grid, st, mls = self.kernels(basis)
        A = np.array([[0.7, -0.2], [0.3, 1.1]])
        f = grid.node_coordinates() @ A.T + np.array([0.05, -0.1])
        got = np.einsum("pka,pkb->pab", f[st.node_id], mls)
        assert np.abs(got - A).max() < 1e-12

    def test_differs_from_the_basis_gradient_on_a_quadratic_basis(self):
        grid, st, mls = self.kernels("quadratic")
        Xn = grid.node_coordinates()
        f = np.stack([Xn[:, 0] ** 2, Xn[:, 1] ** 2], axis=1)
        a = np.einsum("pka,pkb->pab", f[st.node_id], st.grad_w)
        b = np.einsum("pka,pkb->pab", f[st.node_id], mls)
        assert np.abs(a - b).max() > 1e-4

    def test_is_identical_to_the_basis_gradient_on_a_cubic_basis(self):
        """The cubic's third weight moment vanishes, so the two agree.

        Worth knowing before ranking MLS-MPM against BSMPM-cubic: on that basis
        they are the same scheme.
        """
        grid, st, mls = self.kernels("cubic")
        Xn = grid.node_coordinates()
        f = np.stack([Xn[:, 0] ** 2, Xn[:, 1] ** 2], axis=1)
        a = np.einsum("pka,pkb->pab", f[st.node_id], st.grad_w)
        b = np.einsum("pka,pkb->pab", f[st.node_id], mls)
        assert np.abs(a - b).max() < 1e-12

    def test_default_transfers_use_the_basis_gradient(self):
        grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16)
        st = grid.build_stencil(np.array([[0.5, 0.5]]))
        assert APIC().force_kernel(grid, st) is st.grad_w


class TestMLSSimulation:
    def test_runs_and_conserves_momentum(self):
        _, p, s = blob(transfer=MLSMPM())
        m = p.mass
        p0 = (m[:, None] * p.v).sum(axis=0)
        with quiet():
            s.run(p, n_steps=300)
        drift = np.linalg.norm((m[:, None] * p.v).sum(axis=0) - p0) \
            / np.linalg.norm(p0)
        assert drift < 1e-11
        assert np.all(np.isfinite(p.F))

    def test_retains_energy_like_APIC_not_like_PIC(self):
        out = {}
        for name, tr in [("apic", APIC()), ("mls", MLSMPM())]:
            _, p, s = blob(transfer=tr)
            e0 = p.kinetic_energy()
            with quiet():
                s.run(p, n_steps=300)
            out[name] = p.kinetic_energy() / e0
        assert out["mls"] > 0.9
        assert abs(out["mls"] - out["apic"]) < 0.05


class TestPolyPIC:
    def test_degree_one_gives_the_multilinear_modes(self):
        grid, p, s = blob(transfer=PolyPIC(1))
        tr = s.transfer
        st = grid.build_stencil(p.x)
        tr._modes(grid, st)
        assert tr.n_modes == 4                    # {1, x, y, xy} in 2-D

    def test_runs_and_conserves_momentum(self):
        _, p, s = blob(transfer=PolyPIC(1))
        m = p.mass
        p0 = (m[:, None] * p.v).sum(axis=0)
        with quiet():
            s.run(p, n_steps=300)
        drift = np.linalg.norm((m[:, None] * p.v).sum(axis=0) - p0) \
            / np.linalg.norm(p0)
        assert drift < 1e-7
        assert np.all(np.isfinite(p.F))

    def test_matches_APIC_on_a_purely_affine_flow(self):
        """It must: a rigid rotation plus translation has no bilinear content,
        so the extra mode carries nothing."""
        out = {}
        for name, tr in [("apic", APIC()), ("poly", PolyPIC(1))]:
            _, p, s = blob(transfer=tr, flow="affine")
            e0 = p.kinetic_energy()
            with quiet():
                s.run(p, n_steps=300)
            out[name] = p.kinetic_energy() / e0
        assert abs(out["poly"] - out["apic"]) < 1e-3

    def test_retains_more_energy_than_APIC_on_a_non_affine_flow(self):
        """Where the extra mode has something to represent."""
        out = {}
        for name, tr in [("apic", APIC()), ("poly", PolyPIC(1))]:
            _, p, s = blob(transfer=tr, flow="quadratic")
            e0 = p.kinetic_energy()
            with quiet():
                s.run(p, n_steps=300)
            out[name] = p.kinetic_energy() / e0
        assert out["poly"] > out["apic"]

    def test_effective_rank_is_reported(self):
        _, p, s = blob(transfer=PolyPIC(1))
        with quiet():
            s.run(p, n_steps=20)
        assert s.transfer.effective_rank == pytest.approx(4.0)

    @pytest.mark.parametrize("degree", [2, 3])
    def test_higher_degrees_are_refused(self, degree):
        with pytest.raises(NotImplementedError, match="not supported"):
            PolyPIC(degree)

    def test_refusal_names_what_was_tried(self):
        with pytest.raises(NotImplementedError, match="CFL"):
            PolyPIC(2)

    def test_high_degree_can_be_forced_for_experiments(self):
        tr = PolyPIC(2, allow_high_degree=True)
        assert tr.degree == 2

    def test_degree_zero_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            PolyPIC(0)


class TestRegistry:
    @pytest.mark.parametrize("name", ["pic", "flip", "apic", "polypic",
                                      "mls", "mls-mpm"])
    def test_registry_entries_construct(self, name):
        from scion.discretization.mpm.transfer import get_transfer
        assert get_transfer(name) is not None
