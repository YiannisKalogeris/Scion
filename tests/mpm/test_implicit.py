"""Implicit (quasi-static) MPM.

The four checks that would each catch a different class of error:

1. ``dP/dF`` against a central difference on ``P``, per material.
2. The *structural* tangent against a central difference on the residual, which
   is a different statement -- an exact material tangent assembled with the
   wrong contraction passes (1) and fails (2).
3. Observed Newton convergence rate. A wrong tangent with a right residual
   still converges, linearly; nothing but the rate distinguishes it.
4. The affine patch test, exact to round-off. This is the one that says the
   discretisation reproduces a homogeneous state, which neither tangent check
   can see.
"""

import numpy as np
import pytest

from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
    ImplicitMPMStepper,
    InvertedParticle,
    ParticleSet,
    get_basis,
)
from scion.materials.hyperelastic_batched import (
    NeoHookeanCompressible,
    NeoHookeanDecoupledBatched,
)
from scion.solvers.tangent_check import check_tangent


MATERIALS = [
    NeoHookeanCompressible.from_E_nu(1000.0, 0.30),
    NeoHookeanCompressible.from_E_nu(1000.0, 0.45),
    NeoHookeanDecoupledBatched.from_E_nu(1000.0, 0.30),
]


def block(dx=0.25, ppc=2, lo=(0.5, 0.5), hi=(2.0, 1.5), shape=(16, 16),
          basis="linear"):
    """A rectangular body whose edges land on grid lines, so every cell it
    occupies is completely filled. Partially filled cells are the MPM
    quadrature error and would mask everything these tests are looking for."""
    grid = CartesianGrid(shape=shape, dx=dx, basis=get_basis(basis))
    sub = dx / ppc
    axes = [np.arange(lo[k] + sub / 2, hi[k], sub) for k in range(2)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    return grid, ParticleSet(pts, vol0=sub ** 2, density=1.0)


# ----------------------------------------------------------------------
# 1. material tangent
# ----------------------------------------------------------------------
@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("dim", [2, 3])
def test_material_tangent_matches_finite_difference(material, dim):
    if isinstance(material, NeoHookeanDecoupledBatched) and dim == 2:
        pytest.skip("the decoupled model's J^(-2/3) presumes a 3-D F")
    rng = np.random.default_rng(4)
    F = np.eye(dim)[None] + 0.2 * rng.standard_normal((6, dim, dim))

    P, J, A = material.pk1_and_tangent(F)
    assert np.allclose(P, material.pk1(F)[0])
    assert (J > 0).all()

    h = 1e-6
    num = np.zeros_like(A)
    for k in range(dim):
        for L in range(dim):
            up, dn = F.copy(), F.copy()
            up[:, k, L] += h
            dn[:, k, L] -= h
            num[:, :, :, k, L] = (material.pk1(up)[0]
                                  - material.pk1(dn)[0]) / (2 * h)
    assert np.abs(A - num).max() / np.abs(num).max() < 1e-7


@pytest.mark.parametrize("material", MATERIALS)
def test_tangent_at_identity_is_the_isotropic_tensor(material):
    """``A_iJkL = lam d_iJ d_kL + mu (d_ik d_JL + d_iL d_Jk)`` at F = I.

    The cheapest possible check and it pins the index convention: a tangent
    written with the two ``G (x) G`` patterns swapped is still symmetric, still
    converges, and fails this exactly.
    """
    d = np.eye(3)
    _, _, A = material.pk1_and_tangent(d[None])
    mu = material.mu
    lam = getattr(material, "lam", None)
    if lam is None:
        lam = material.K - 2.0 * mu / 3.0
    ref = (lam * np.einsum("iJ,kL->iJkL", d, d)
           + mu * (np.einsum("ik,JL->iJkL", d, d)
                   + np.einsum("iL,Jk->iJkL", d, d)))
    assert np.abs(A[0] - ref).max() < 1e-10 * np.abs(ref).max()


@pytest.mark.parametrize("material", MATERIALS)
def test_tangent_is_major_symmetric(material):
    """``A_iJkL = A_kLiJ``, because P derives from a potential. Loses nothing
    if it holds and localises the error immediately if it does not."""
    rng = np.random.default_rng(11)
    F = np.eye(3)[None] + 0.15 * rng.standard_normal((5, 3, 3))
    _, _, A = material.pk1_and_tangent(F)
    assert np.abs(A - A.transpose(0, 3, 4, 1, 2)).max() < 1e-12 * np.abs(A).max()


def test_material_without_a_tangent_is_refused():
    class Explicit:
        def pk1(self, F):
            return F, np.ones(len(F))

    grid, _ = block()
    with pytest.raises(TypeError, match="pk1_and_tangent"):
        ImplicitMPMStepper(grid, Explicit())


def test_path_dependent_material_is_refused():
    grid, _ = block()
    with pytest.raises(NotImplementedError, match="algorithmic tangent"):
        ImplicitMPMStepper(grid, MATERIALS[0], state={})


# ----------------------------------------------------------------------
# 2. structural tangent
# ----------------------------------------------------------------------
@pytest.mark.parametrize("basis", ["linear", "quadratic"])
def test_structural_tangent_matches_finite_difference(basis):
    grid, parts = block(basis=basis)
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0],
        dirichlet=[GridDirichlet(lambda c: c[:, 0] < 0.5 + 1e-9,
                                 np.array([0.0, 0.0]))])
    stepper.prepare(parts)

    free = stepper._free
    u = np.zeros(stepper._n_dof)
    u[free] = 1e-3 * np.random.default_rng(1).standard_normal(free.size)

    report = check_tangent(stepper.internal_force, stepper.tangent, u,
                           free=free)
    assert report.passed, report
    assert report.symmetry < 1e-12


# ----------------------------------------------------------------------
# 3. Newton convergence rate
# ----------------------------------------------------------------------
def test_newton_convergence_is_quadratic():
    """The rate, not the count. A tangent that is merely close converges
    linearly and takes only a few more iterations, which is invisible without
    this."""
    grid, parts = block()
    A = np.eye(2) + 0.03 * np.array([[1.0, 0.4], [-0.25, -0.5]])
    interior = lambda c: ~((c[:, 0] > 0.5 + 1e-9) & (c[:, 0] < 2.0 - 1e-9)
                           & (c[:, 1] > 0.5 + 1e-9) & (c[:, 1] < 1.5 - 1e-9))
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0],
        dirichlet=[GridDirichlet(interior, lambda c: c @ (A - np.eye(2)).T)])
    stepper.run(parts, n_steps=2)

    for res in stepper.history:
        assert res.converged
        assert res.iterations <= 6
        assert res.convergence_rate > 1.8


# ----------------------------------------------------------------------
# 4. patch test
# ----------------------------------------------------------------------
@pytest.mark.parametrize("basis", ["linear", "quadratic"])
def test_affine_patch_test_is_exact(basis):
    """Prescribe ``u = (A - I) x`` on every node outside the body and the
    interior must reproduce ``F = A`` exactly, in one step.

    One step matters. The prescribed field is attached to *grid* coordinates,
    so it is a spatial condition; ramping it over ``n`` steps drives the body
    along ``(I + (A-I)/n)^n`` instead of to ``A``, which is a different
    (equally homogeneous) state. See the next test.
    """
    grid, parts = block(basis=basis, hi=(2.5, 2.0), shape=(20, 20))
    A = np.eye(2) + 0.02 * np.array([[1.0, 0.4], [-0.25, -0.5]])
    # The constrained ring has to be at least as thick as the basis support.
    # A free node whose support is clipped by the edge of the *particle* region
    # has sum_p V_p Grad N_I != 0, so a constant stress does not equilibrate
    # there and the patch test fails by ~1e-3 -- which reads as a quadrature
    # error and is really a mis-sized boundary layer. The linear hat reaches one
    # cell, the quadratic B-spline one and a half.
    pad = 0.0 if basis == "linear" else grid.dx
    interior = lambda c: ~((c[:, 0] > 0.5 + pad + 1e-9)
                           & (c[:, 0] < 2.5 - pad - 1e-9)
                           & (c[:, 1] > 0.5 + pad + 1e-9)
                           & (c[:, 1] < 2.0 - pad - 1e-9))
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0],
        dirichlet=[GridDirichlet(interior, lambda c: c @ (A - np.eye(2)).T)])
    stepper.prepare(parts)
    assert stepper._free.size >= 12          # a real solve, not a kinematic one
    stepper.step(parts, 1.0, 1.0)

    assert np.abs(parts.F - A).max() < 1e-13


def test_ramped_grid_dirichlet_follows_the_compounded_path():
    """A grid-attached displacement field is *spatial*, so a linear ramp of it
    compounds multiplicatively: ``F = (I + (A-I)/n)^n``, not ``A``.

    Recorded as a test because the difference is small, entirely physical, and
    reads exactly like a convergence bug -- the error grows with the number of
    load steps, which is the opposite of what an integration error does.
    """
    grid, parts = block()
    n = 4
    A = np.eye(2) + 0.02 * np.array([[1.0, 0.4], [-0.25, -0.5]])
    interior = lambda c: ~((c[:, 0] > 0.5 + 1e-9) & (c[:, 0] < 2.0 - 1e-9)
                           & (c[:, 1] > 0.5 + 1e-9) & (c[:, 1] < 1.5 - 1e-9))
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0],
        dirichlet=[GridDirichlet(interior, lambda c: c @ (A - np.eye(2)).T)])
    stepper.run(parts, n_steps=n)

    compounded = np.linalg.matrix_power(np.eye(2) + (A - np.eye(2)) / n, n)
    assert np.abs(parts.F - compounded).max() < 1e-12
    assert np.abs(parts.F - A).max() > 1e-6      # and *not* A
    assert np.ptp(parts.F, axis=0).max() < 1e-12  # still perfectly homogeneous


# ----------------------------------------------------------------------
# failure modes are named, not silent
# ----------------------------------------------------------------------
def test_inverted_trial_state_is_named():
    grid, parts = block()
    stepper = ImplicitMPMStepper(grid, MATERIALS[0])
    stepper.prepare(parts)
    du = np.zeros(stepper._n_dof)
    du[0::2] = -10.0 * grid.node_coordinates()[:, 0]
    with pytest.raises(InvertedParticle, match="inverted"):
        stepper.internal_force(du)


def test_load_stepping_cuts_back_rather_than_failing():
    """A single increment that inverts must be recovered by halving, not
    reported as non-convergence."""
    grid, parts = block()
    A = np.eye(2) + 0.20 * np.array([[1.0, 0.4], [-0.25, -0.5]])
    interior = lambda c: ~((c[:, 0] > 0.5 + 1e-9) & (c[:, 0] < 2.0 - 1e-9)
                           & (c[:, 1] > 0.5 + 1e-9) & (c[:, 1] < 1.5 - 1e-9))
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0],
        dirichlet=[GridDirichlet(interior, lambda c: c @ (A - np.eye(2)).T)])
    stepper.run(parts, n_steps=1)
    assert len(stepper.history) > 1          # it did cut back
    assert all(r.converged for r in stepper.history)


def test_body_force_produces_a_symmetric_response():
    """Self-weight on a block held at its base. Not a benchmark -- it checks
    that the external force path is wired at all, and that a load with no
    horizontal component produces no net horizontal drift."""
    grid, parts = block()
    stepper = ImplicitMPMStepper(
        grid, MATERIALS[0], body_force=np.array([0.0, -20.0]),
        dirichlet=[GridDirichlet(lambda c: c[:, 1] < 0.5 + 1e-9,
                                 np.array([0.0, 0.0]))])
    x0 = parts.x.copy()
    stepper.run(parts, n_steps=4)
    u = parts.x - x0
    assert u[:, 1].mean() < 0.0                       # it sagged
    assert abs(u[:, 0].mean()) < 1e-3 * abs(u[:, 1].mean())


# ----------------------------------------------------------------------
# 5. the external check: observed order of accuracy
# ----------------------------------------------------------------------
class TestManufacturedSolution:
    """Verification against the governing equations rather than against the
    library. An error can be small for the wrong reasons; the rate it falls at
    is a property of the scheme, and it is the only check here that would catch
    the code consistently solving a *different* equation.

    Mirrors Fig. 3(a) of O'Hare, Gourgiotis, Coombs & Augarde, CMAME 419 (2024)
    116668, reduced to the classical continuum.
    """

    def test_linear_basis_is_second_order(self):
        from examples.mpm.implicit_manufactured_solution import convergence
        rows = convergence([8, 16, 32], "linear")
        rates = [r[2] for r in rows[1:]]
        assert min(rates) > 1.9, rates

    def test_newton_is_quadratic_on_the_manufactured_problem(self):
        from examples.mpm.implicit_manufactured_solution import solve
        _, _, stepper, _, _ = solve(16, basis="linear")
        res = stepper.history[-1]
        assert res.iterations <= 6
        assert res.convergence_rate > 1.9

    def test_bspline_dirichlet_costs_an_order_globally(self):
        """A non-interpolatory basis puts the Dirichlet condition O(h) off the
        boundary, and the resulting first-order error pollutes the *interior*
        at the same rate -- so it is not a boundary layer that could be excused
        by measuring away from it. Recorded as a test because it is the price
        the smooth bases charge on a Dirichlet problem, and the configurational
        work wants those bases for other reasons."""
        from examples.mpm.implicit_manufactured_solution import convergence
        rows = convergence([8, 16, 32], "quadratic")
        assert 0.7 < rows[-1][2] < 1.3          # whole body
        assert 0.6 < rows[-1][4] < 1.3          # and the interior too
