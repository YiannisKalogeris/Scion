"""
Verification tests for the MPM kernels.

These are the MPM analogue of the FEM patch test: cheap, exact, and they fail
loudly when a basis function, a stencil index, or a transfer kernel is wrong.
Both failures caught here look like physical instabilities if you only inspect
the deformed shape, which is exactly why they need to be assertions.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core import small_linalg as sla
from scion.discretization.mpm import (
    APIC,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
    get_basis,
)
from scion.materials.hyperelastic_batched import NeoHookeanCompressible

BASES = ["linear", "quadratic", "cubic", "gimp"]


@pytest.mark.parametrize("name", BASES)
@pytest.mark.parametrize("dim", [2, 3])
def test_partition_of_unity(name, dim):
    """sum_i w_ip = 1 and sum_i grad w_ip = 0, for arbitrary particle positions.

    Catches wrong B-spline coefficients, a wrong base-node offset, and a
    mismatched stencil size.
    """
    rng = np.random.default_rng(0)
    grid = CartesianGrid(shape=(8,) * dim, dx=0.125, basis=get_basis(name))
    x = rng.uniform(0.2, 0.8, size=(500, dim))
    lp = np.full((500, dim), 0.125 / 4)

    st = grid.build_stencil(x, lp=lp)
    assert np.abs(st.w.sum(axis=1) - 1.0).max() < 1e-12
    assert np.abs(st.grad_w.sum(axis=1)).max() < 1e-10


@pytest.mark.parametrize("name", BASES)
def test_linear_field_reproduction(name):
    """The weights reproduce a linear field and its gradient exactly.

    This is the MPM patch test: sum_i w_ip x_i = x_p and sum_i x_i (x) grad w_ip = I.
    """
    rng = np.random.default_rng(1)
    grid = CartesianGrid(shape=(8, 8), dx=0.125, basis=get_basis(name))
    x = rng.uniform(0.2, 0.8, size=(200, 2))
    st = grid.build_stencil(x, lp=np.full((200, 2), 0.125 / 4))

    node_pos = x[:, None, :] + st.dpos
    interp = np.einsum("pk,pka->pa", st.w, node_pos)
    assert np.abs(interp - x).max() < 1e-12

    grad = np.einsum("pka,pkb->pab", node_pos, st.grad_w)
    assert np.abs(grad - np.eye(2)).max() < 1e-10


def test_p2g_conserves_mass_and_momentum():
    """Scattering to the grid must not create or destroy mass or momentum."""
    rng = np.random.default_rng(2)
    grid = CartesianGrid(shape=(8, 8), dx=0.125)
    x = rng.uniform(0.2, 0.8, size=(300, 2))
    p = ParticleSet(x, v=rng.normal(size=(300, 2)), vol0=1e-4, density=1000.0)

    st = grid.build_stencil(p.x)
    m, mv = APIC().p2g(p, grid, st)

    assert abs(m.sum() - p.mass.sum()) < 1e-9 * p.mass.sum()
    assert np.abs(mv.sum(axis=0) - p.linear_momentum()).max() < 1e-9


def test_apic_roundtrip_preserves_affine_velocity():
    """P2G -> G2P with no forces returns an affine velocity field unchanged.

    This is the defining property of APIC and fails for PIC.
    """
    rng = np.random.default_rng(3)
    grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16)
    x = rng.uniform(0.25, 0.75, size=(400, 2))

    A = np.array([[0.3, -0.7], [0.5, 0.2]])   # affine (incl. rotation) field
    b = np.array([0.11, -0.04])
    p = ParticleSet(x, v=x @ A.T + b, vol0=1e-4, density=1000.0)
    p.C = np.tile(A, (p.n, 1, 1))

    st = grid.build_stencil(p.x)
    m, mv = APIC().p2g(p, grid, st)
    active = m > 1e-12
    gv = np.zeros_like(mv)
    gv[active] = mv[active] / m[active, None]

    v_before = p.v.copy()
    APIC().g2p(p, grid, st, gv, dt=0.0)
    assert np.abs(p.v - v_before).max() < 1e-10
    assert np.abs(p.C - A).max() < 1e-8


def test_rigid_translation_leaves_F_unchanged():
    """A uniformly translating body must keep F = I and lose no momentum."""
    grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16)
    mat = NeoHookeanCompressible.from_E_nu(1e3, 0.3)
    p = ParticleSet.from_sampler(
        lambda q: np.linalg.norm(q - 0.5, axis=1) < 0.15,
        ((0.0, 0.0), (1.0, 1.0)), 1.0 / 16, ppc=2,
        velocity=(0.05, -0.03), density=1000.0,
    )
    stepper = ExplicitMPMStepper(grid, mat, APIC(), dt=1e-3)

    p0 = p.linear_momentum()
    for _ in range(50):
        stepper.step(p)

    assert np.abs(sla.det(p.F) - 1.0).max() < 1e-9
    assert np.linalg.norm(p.linear_momentum() - p0) < 1e-9 * np.linalg.norm(p0)


def test_material_pk1_matches_finite_difference():
    """Guards the batched material against the single-point reference."""
    rng = np.random.default_rng(4)
    mat = NeoHookeanCompressible.from_E_nu(50e9, 0.3)
    F = np.eye(2) + 0.1 * rng.normal(size=(5, 2, 2))

    def energy(Fs):
        J = sla.det(Fs)
        I1 = np.einsum("pij,pij->p", Fs, Fs)
        return 0.5 * mat.mu * (I1 - 2) - mat.mu * np.log(J) \
            + 0.5 * mat.lam * np.log(J) ** 2

    P, _ = mat.pk1(F)
    eps = 1e-7
    for i in range(2):
        for j in range(2):
            Fp, Fm = F.copy(), F.copy()
            Fp[:, i, j] += eps
            Fm[:, i, j] -= eps
            fd = (energy(Fp) - energy(Fm)) / (2 * eps)
            assert np.abs(P[:, i, j] - fd).max() < 1e-4 * np.abs(P).max()


# ---------------------------------------------------------------------------
# GIMP-specific
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.05, 0.125, 0.25, 0.5])
def test_gimp_reduces_to_hat_and_stays_exact(lam):
    """uGIMP weights are exact for any admissible domain size.

    Partition of unity, linear reproduction and gradient reproduction all hold
    to machine precision, independently of lp -- which is what makes lp a free
    quadrature parameter rather than something that has to be tuned.
    """
    from scion.discretization.mpm.basis import uGIMP

    rng = np.random.default_rng(7)
    b = uGIMP()
    fx = rng.uniform(0.5, 1.5, size=(2000, 1))
    w, dw = b.weights_1d(fx, np.full_like(fx, lam))

    k = np.arange(3).reshape(1, 3, 1)
    assert np.abs(w.sum(axis=1) - 1.0).max() < 1e-14
    assert np.abs(dw.sum(axis=1)).max() < 1e-13
    assert np.abs((w * k).sum(axis=1) - fx).max() < 1e-14
    assert np.abs((dw * k).sum(axis=1) - 1.0).max() < 1e-13


def test_gimp_rejects_oversized_domain():
    """lp > dx/2 would silently truncate the support, so it must raise."""
    from scion.discretization.mpm.basis import uGIMP

    b = uGIMP()
    fx = np.full((10, 1), 1.0)
    with pytest.raises(ValueError, match="lp <= dx/2"):
        b.weights_1d(fx, np.full_like(fx, 0.75))


def test_grid_requires_lp_for_gimp():
    grid = CartesianGrid(shape=(8, 8), dx=0.125, basis=get_basis("gimp"))
    with pytest.raises(ValueError, match="domain"):
        grid.build_stencil(np.full((5, 2), 0.5))


def test_gimp_runs_and_conserves_momentum():
    """A translating body under GIMP keeps F = I and its momentum."""
    grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16, basis=get_basis("gimp"))
    mat = NeoHookeanCompressible.from_E_nu(1e3, 0.3)
    p = ParticleSet.from_sampler(
        lambda q: np.linalg.norm(q - 0.5, axis=1) < 0.15,
        ((0.0, 0.0), (1.0, 1.0)), 1.0 / 16, ppc=2,
        velocity=(0.05, -0.03), density=1000.0,
    )
    stepper = ExplicitMPMStepper(grid, mat, APIC(), dt=1e-3)

    p0 = p.linear_momentum()
    for _ in range(50):
        stepper.step(p)

    assert np.abs(sla.det(p.F) - 1.0).max() < 1e-9
    assert np.linalg.norm(p.linear_momentum() - p0) < 1e-9 * np.linalg.norm(p0)


def test_cpgimp_domain_stretches_with_F():
    """Under "stretch", lp follows the diagonal of F."""
    p = ParticleSet.from_sampler(
        lambda q: np.ones(len(q), bool), ((0.0, 0.0), (0.5, 0.5)),
        0.1, ppc=2, density=1000.0,
    )
    p.F[:, 0, 0] = 1.6
    p.F[:, 1, 1] = 0.8
    p.update_domain("stretch")
    assert np.allclose(p.lp[:, 0], p.lp0[:, 0] * 1.6)
    assert np.allclose(p.lp[:, 1], p.lp0[:, 1] * 0.8)
