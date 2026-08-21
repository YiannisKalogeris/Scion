"""Ghost stabilisation.

The operator has two independent halves and each needs its own check, because
the obvious test only exercises one of them:

* the **jump** ``G_g = G^+ - G^-``, tested by the affine kernel;
* the **normal matrix** ``N``, tested against ``sigma n``.

The affine kernel says nothing at all about ``N`` -- for an affine field the
jump is already zero, so *any* ``N`` annihilates it. That was found by trying
it: the reference's printed Eq. 148 has ``n3`` where ``n1`` belongs, and the
affine test passes either way.
"""

import numpy as np
import pytest

from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
    ImplicitMPMStepper,
    ParticleSet,
    get_basis,
)
from scion.discretization.mpm.ghost import (
    _normal_matrix,
    ghost_faces,
    ghost_penalty,
    populated_cells,
)
from scion.materials.hyperelastic_batched import NeoHookeanCompressible


def ragged_body(dim=3, dx=0.25, radius=0.62, basis="linear"):
    """A ball seeded on a Cartesian lattice: every cell at the rim is cut, and
    some are cut very badly. That is the situation ghost stabilisation is for;
    a box whose faces land on grid lines has no small cuts at all."""
    n = 8 if dim == 3 else 10
    grid = CartesianGrid(shape=(n,) * dim, dx=dx, basis=get_basis(basis))
    sub = dx / 2
    axes = [np.arange(0.3 + sub / 2, 1.6, sub) for _ in range(dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    X = X[np.linalg.norm(X - X.mean(axis=0), axis=1) < radius]
    return grid, X, sub


# ----------------------------------------------------------------------
# the normal matrix
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_normal_matrix_is_the_traction_map(dim):
    """``N voigt(sigma) == sigma n`` for random symmetric tensors and normals.

    This is the *only* check with power over ``N``. The reference's Eq. 148
    prints ``n3`` in the slot that must hold ``n1``; that version fails here and
    passes every other test in this file.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        S = rng.standard_normal((dim, dim))
        S = 0.5 * (S + S.T)
        n = rng.standard_normal(dim)
        n /= np.linalg.norm(n)
        voigt = ([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[2, 0]]
                 if dim == 3 else [S[0, 0], S[1, 1], S[0, 1]])
        assert np.allclose(_normal_matrix(n, dim) @ np.array(voigt), S @ n)


def test_the_printed_normal_matrix_fails_that():
    """Guards the guard: if this ever passes, the check above has lost its
    power and the transcription is no longer being tested."""
    N = _normal_matrix(np.array([0.0, 0.0, 1.0]), 3)
    N[0, 5] = 1.0
    N[2, 5] = 1.0                      # the reference's printed Eq. 148
    S = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    voigt = np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[2, 0]])
    assert not np.allclose(N @ voigt, S @ np.array([0.0, 0.0, 1.0]))


# ----------------------------------------------------------------------
# the operator
# ----------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 3])
def test_annihilates_affine_fields(dim):
    """An affine field has a constant gradient, so there is no jump to
    penalise. This is what makes the stabilisation free: it cannot perturb the
    states a consistent discretisation has to reproduce exactly."""
    grid, X, _ = ragged_body(dim)
    J = ghost_penalty(grid, X)
    coords = grid.node_coordinates()
    rng = np.random.default_rng(2)
    scale = abs(J).max()
    for _ in range(5):
        A = rng.standard_normal((dim, dim))
        b = rng.standard_normal(dim)
        u = (coords @ A.T + b).ravel()
        assert np.linalg.norm(J @ u) < 1e-12 * scale * np.linalg.norm(u)


@pytest.mark.parametrize("dim", [2, 3])
def test_does_not_annihilate_a_quadratic_field(dim):
    """Guards the guard again -- an operator of all zeros would pass every
    other test here."""
    grid, X, _ = ragged_body(dim)
    J = ghost_penalty(grid, X)
    coords = grid.node_coordinates()
    u = np.repeat((coords[:, 0] ** 2)[:, None], dim, axis=1).ravel()
    assert np.linalg.norm(J @ u) > 1e-6 * abs(J).max() * np.linalg.norm(u)


@pytest.mark.parametrize("dim", [2, 3])
def test_symmetric_positive_semidefinite(dim):
    grid, X, _ = ragged_body(dim)
    J = ghost_penalty(grid, X)
    assert abs(J - J.T).max() < 1e-14 * abs(J).max()
    live = np.unique(J.nonzero()[0])
    ev = np.linalg.eigvalsh(J.toarray()[np.ix_(live, live)])
    assert ev.min() > -1e-12 * ev.max()


def test_bspline_basis_is_refused():
    """A B-spline grid space is C^1: its gradient does not jump across a cell
    face, so this penalty is identically zero there and would silently do
    nothing."""
    grid, X, _ = ragged_body(3, basis="quadratic")
    with pytest.raises(NotImplementedError, match="C\\^1"):
        ghost_penalty(grid, X)


def test_gimp_occupancy_uses_the_particle_domain(self=None):
    """A GIMP particle supports every cell its *domain* overlaps. Counting only
    the cell holding its centre marks those cells empty, and an empty cell gets
    no penalised face -- which is exactly the cell that needed stabilising."""
    grid, X, sub = ragged_body(3)
    centres = populated_cells(grid, X)
    domains = populated_cells(grid, X, lp=np.full_like(X, sub / 2))
    assert domains.sum() >= centres.sum()
    assert (domains | centres == domains).all()


def test_faces_exist_only_at_a_cut_boundary(self=None):
    """A body filling whole cells exactly has no small cuts, so nothing to
    stabilise. A ragged one has many faces."""
    dx = 0.25
    grid = CartesianGrid(shape=(12, 12), dx=dx, basis=get_basis("linear"))
    sub = dx / 2
    axis = np.arange(0.5 + sub / 2, 1.5, sub)      # edges on grid lines
    mesh = np.meshgrid(axis, axis, indexing="ij")
    box = np.stack([m.ravel() for m in mesh], axis=1)
    ragged = ragged_body(2)[1]
    assert len(ghost_faces(grid, ragged)) > len(ghost_faces(grid, box))


# ----------------------------------------------------------------------
# in the solver
# ----------------------------------------------------------------------
def test_patch_test_survives_the_penalty():
    """The affine patch test must stay exact with stabilisation on. This is the
    end-to-end version of the kernel property and it is the reason the penalty
    can be left on by default."""
    dx = 0.25
    grid = CartesianGrid(shape=(20, 20), dx=dx, basis=get_basis("linear"))
    sub = dx / 2
    axes = [np.arange(0.5 + sub / 2, e, sub) for e in (2.5, 2.0)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    material = NeoHookeanCompressible.from_E_nu(1000.0, 0.30)
    A = np.eye(2) + 0.02 * np.array([[1.0, 0.4], [-0.25, -0.5]])
    interior = lambda c: ~((c[:, 0] > 0.5 + 1e-9) & (c[:, 0] < 2.5 - 1e-9)
                           & (c[:, 1] > 0.5 + 1e-9) & (c[:, 1] < 2.0 - 1e-9))
    bcs = [GridDirichlet(interior, lambda c: c @ (A - np.eye(2)).T)]

    for ghost in (None, 10.0):
        parts = ParticleSet(pts.copy(), vol0=sub ** 2, density=1.0)
        stepper = ImplicitMPMStepper(grid, material, dirichlet=bcs,
                                     ghost=ghost)
        stepper.step(parts, 1.0, 1.0)
        assert np.abs(parts.F - A).max() < 1e-13, ghost


def test_penalty_enters_residual_and_tangent_consistently():
    """If the penalty were added to the tangent only, Newton would still
    converge -- linearly. The rate is the only symptom."""
    dx = 0.25
    grid = CartesianGrid(shape=(20, 20), dx=dx, basis=get_basis("linear"))
    sub = dx / 2
    axes = [np.arange(0.5 + sub / 2, e, sub) for e in (2.0, 1.5)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    parts = ParticleSet(pts, vol0=sub ** 2, density=1.0)
    material = NeoHookeanCompressible.from_E_nu(1000.0, 0.30)
    stepper = ImplicitMPMStepper(
        grid, material, ghost=50.0,
        dirichlet=[GridDirichlet(lambda c: c[:, 0] < 0.5 + 1e-9,
                                 np.array([0.0, 0.0]))])
    stepper.prepare(parts)

    from scion.solvers.tangent_check import check_tangent
    free = stepper._free
    u = np.zeros(stepper._n_dof)
    u[free] = 1e-3 * np.random.default_rng(1).standard_normal(free.size)
    report = check_tangent(stepper.internal_force, stepper.tangent, u,
                           free=free)
    assert report.passed, report


def test_it_targets_the_small_cut_and_leaves_the_rest_alone():
    """The small-cut issue, quantified, and the fix, quantified.

    A disc seeded on a lattice has badly cut cells at its rim. Measured on the
    weakest free-DOF diagonal of the tangent:

        linear, no penalty      2.6e+01
        GIMP,   no penalty      4.5e-02      <- 580x weaker: the small cut
        GIMP,   gamma = 10      6.7e+00      <- 149x lift
        GIMP,   gamma = 100     6.7e+01

    GIMP is the worse case because its support is wider, so a particle near the
    rim reaches nodes it barely weights. The point of the test is the *pair* of
    claims: the penalty lifts the weak diagonal by orders of magnitude while
    leaving a well-conditioned basis essentially untouched. A penalty that
    simply stiffened everything would pass the first half and fail the second.
    """
    dx, ppc = 0.25, 2
    sub = dx / ppc
    axis = np.arange(0.5 + sub / 2, 2.5, sub)
    mesh = np.meshgrid(axis, axis, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    X = X[np.linalg.norm(X - np.array([1.5, 1.5]), axis=1) < 0.83]
    material = NeoHookeanCompressible.from_E_nu(1000.0, 0.30)
    bcs = [GridDirichlet(lambda c: c[:, 0] < 0.70, np.array([0.05, 0.0])),
           GridDirichlet(lambda c: c[:, 0] > 2.30, np.array([-0.05, 0.0]))]

    weakest = {}
    for basis in ("linear", "gimp"):
        for ghost in (None, 10.0):
            grid = CartesianGrid(shape=(16, 16), dx=dx, basis=get_basis(basis))
            parts = ParticleSet(X.copy(), vol0=sub ** 2, density=1.0)
            stepper = ImplicitMPMStepper(grid, material, dirichlet=bcs,
                                         ghost=ghost, max_iter=20)
            stepper.run(parts, n_steps=2)
            K = stepper.tangent(np.zeros(stepper._n_dof))
            d = np.abs(K[stepper._free][:, stepper._free].diagonal())
            weakest[(basis, ghost)] = d.min()

    # GIMP without stabilisation is the ill-conditioned case
    assert weakest[("gimp", None)] < 0.05 * weakest[("linear", None)]
    # the penalty rescues it by orders of magnitude
    assert weakest[("gimp", 10.0)] > 20.0 * weakest[("gimp", None)]
    # and barely touches the basis that was already fine
    assert weakest[("linear", 10.0)] < 3.0 * weakest[("linear", None)]
