"""
Verification for gradient damage as a boundary value problem.

Two things are being checked. First that the Helmholtz filter is assembled
correctly, which has an exact answer: a constant field must pass through it
unchanged. Second, and the reason the whole regularisation exists, that the
structural response converges under mesh refinement -- and that the local model
it replaces does not.

The mesh study is the external-facing claim of Peerlings et al. (1996). It is
reproduced here at small scale so it runs in seconds; the example does it
properly.

python -m pytest tests/test_damage/test_gradient.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.domain import Domain
from scion.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
    StatefulOperators,
)
from scion.materials.damage import (
    DamageModel,
    ExponentialSoftening,
    GradientDamageModel,
)
from scion.mesh.generators import rectangle_mesh
from scion.multiphysics import GradientDamageProblem
from scion.solvers.nonlinear import DisplacementControl

E, NU = 20.0e3, 0.2
K0 = 1.0e-4
ELL = 0.5
LENGTH, HEIGHT = 10.0, 1.0


def _law():
    return ExponentialSoftening(K0, alpha=0.999, beta=500.0)


def _mesh(nx, ny=4, neck=True):
    mesh = rectangle_mesh((0.0, LENGTH), (0.0, HEIGHT), nx, ny)
    if neck:
        x = mesh.nodes[:, 0]
        s = 1.0 - 0.1 * np.exp(-((x - 0.5 * LENGTH) / 0.5) ** 2)
        mesh.nodes[:, 1] = (mesh.nodes[:, 1] - 0.5 * HEIGHT) * s + 0.5 * HEIGHT
    return Domain(mesh, dim=2)


def _problem(nx, ny=4, neck=True, ell=ELL):
    domain = _mesh(nx, ny, neck)
    model = GradientDamageModel(E, NU, _law(), ell=ell,
                                equivalent="modified-von-mises",
                                equivalent_kwargs={"k": 10.0})
    problem = GradientDamageProblem(domain, model)
    nodes = domain.coordinates
    left = np.where(nodes[:, 0] < 1e-9)[0]
    right = np.where(nodes[:, 0] > LENGTH - 1e-9)[0]
    problem.u.prescribe(left, 0, 0.0)
    problem.u.prescribe([left[0]], 1, 0.0)
    problem.set_displacement_control(right, 0, 1.0)
    return problem


def _run(problem, u_max=3.0e-3, n_steps=15):
    history = []
    problem.run(np.linspace(u_max / n_steps, u_max, n_steps),
                callback=lambda r, p: history.append(
                    (p.control_displacement(), p.control_force())),
                stop_on_failure=False)
    return np.array(history)


def _band_width(problem, nx):
    """Second moment of the damage profile along the bar, times two.

    A width measured this way does not depend on the number of elements used to
    resolve it, which a count of damaged elements or a threshold crossing both
    would -- and the whole question here is whether a width is mesh-dependent.
    """
    d = problem.damage_by_element().reshape(nx, -1).mean(axis=1)
    x = (np.arange(nx) + 0.5) * LENGTH / nx
    w = np.maximum(d, 0.0)
    if w.sum() <= 0.0:
        return 0.0
    mean = (w * x).sum() / w.sum()
    return float(2.0 * np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))


# ---------------------------------------------------------------------------
# The Helmholtz filter
# ---------------------------------------------------------------------------

def test_a_constant_field_passes_through_the_filter_unchanged():
    """``ebar - ell^2 laplacian(ebar) = eps_eq`` with a constant right-hand
    side has the same constant as its exact solution, and the natural boundary
    condition is satisfied identically. Any error in the mass matrix, the
    Laplacian, the load vector or their relative scaling breaks it, so this one
    check covers the whole assembly.
    """
    problem = _problem(nx=12, neck=False)
    nodes = problem.domain.coordinates

    # a uniform axial stretch: eps_eq is then the same at every point
    stretch = 4.0e-4
    problem.u.values = np.zeros(problem.u.n_dof)
    problem.u.values[0::2] = stretch * nodes[:, 0]

    eps = problem.tables.strains_voigt(problem.u.values)
    eps_eq = problem.model.equivalent_strain(eps, dim=2)
    assert np.allclose(eps_eq, eps_eq[0])

    problem.solve_second()
    assert np.allclose(problem.e.values, eps_eq[0], rtol=1e-10)


def test_the_filter_lowers_a_peak_and_spreads_it():
    """What regularisation does: the nonlocal field is a smeared version of the
    local one, so its maximum is lower and its support wider."""
    problem = _problem(nx=60)
    nodes = problem.domain.coordinates
    # a strain concentrated near mid-span
    bump = np.exp(-((nodes[:, 0] - 0.5 * LENGTH) / 0.4) ** 2)
    problem.u.values = np.zeros(problem.u.n_dof)
    problem.u.values[0::2] = 1.0e-3 * np.cumsum(bump) / len(bump)

    eps = problem.tables.strains_voigt(problem.u.values)
    local = problem.model.equivalent_strain(eps, dim=2)
    problem.solve_second()
    nonlocal_ = problem.tables.interpolate(problem.e.values)

    assert nonlocal_.max() < local.max()
    assert nonlocal_.sum() == pytest.approx(local.sum(), rel=0.05)


def test_a_longer_length_smooths_more():
    peaks = []
    for ell in (0.25, 1.0):
        problem = _problem(nx=60, ell=ell)
        nodes = problem.domain.coordinates
        problem.u.values = np.zeros(problem.u.n_dof)
        problem.u.values[0::2] = 1.0e-3 * np.exp(
            -((nodes[:, 0] - 0.5 * LENGTH) / 0.4) ** 2)
        problem.solve_second()
        peaks.append(problem.tables.interpolate(problem.e.values).max())
    assert peaks[1] < peaks[0]


def test_the_filter_operator_is_built_once():
    """It depends on neither field, so it is factorised once for the whole
    analysis -- the reason the regularised problem costs less per iteration
    than the local one."""
    problem = _problem(nx=12, neck=False)
    assert problem._filter is None
    problem.solve_second()
    first = problem._filter
    problem.solve_second()
    assert problem._filter is first


# ---------------------------------------------------------------------------
# Mesh convergence -- the point of the exercise
# ---------------------------------------------------------------------------

def test_the_gradient_model_converges_under_mesh_refinement():
    """Peak load, dissipated energy and band width all settle. This is the
    claim the internal length is there to deliver."""
    results = []
    for nx, ny in ((80, 8), (160, 8)):
        problem = _problem(nx, ny)
        problem.check_mesh()
        history = _run(problem)
        results.append((history[:, 1].max(), problem.dissipated_energy(),
                        _band_width(problem, nx)))

    peak, dissipated, width = (np.array(x) for x in zip(*results))
    assert peak[1] == pytest.approx(peak[0], rel=1e-3)
    assert dissipated[1] == pytest.approx(dissipated[0], rel=0.02)
    assert width[1] == pytest.approx(width[0], rel=0.02)


def test_the_band_is_several_times_the_internal_length_and_not_the_mesh():
    problem = _problem(80, 8)
    _run(problem)
    width = _band_width(problem, 80)
    assert width > 2.0 * ELL
    assert width > 20.0 * problem.element_size


def test_a_mesh_too_coarse_for_the_length_is_refused():
    problem = _problem(nx=8, ny=2)
    with pytest.raises(ValueError, match="resolve the internal length"):
        problem.check_mesh()


# ---------------------------------------------------------------------------
# The local model, for contrast
# ---------------------------------------------------------------------------

def _run_local(nx, u_max, n_steps, ny=4):
    domain = _mesh(nx, ny)
    model = DamageModel(E, NU, _law(), equivalent="modified-von-mises",
                        equivalent_kwargs={"k": 10.0})
    asm = ElastoplasticAssembler(domain, model)
    ops = StatefulOperators(asm)
    nodes, dm = domain.coordinates, asm.dof_map
    left = np.where(nodes[:, 0] < 1e-9)[0]
    right = np.where(nodes[:, 0] > LENGTH - 1e-9)[0]
    fixed = np.concatenate([dm.element_dofs(left)[0::2],
                            [dm.dof(int(left[0]), 1)]])
    prescribed = {dm.dof(int(n), 0): u_max for n in right}
    free = np.setdiff1d(np.arange(asm.n_dof),
                        np.union1d(fixed, list(prescribed)))
    driver = DisplacementControl(ops.internal_force, ops.tangent, asm.n_dof,
                                 free, prescribed, fixed, n_steps=n_steps,
                                 tol=1e-8, max_iter=40, line_search=True)
    driver.on_step_converged = ops.commit
    result = driver.solve()
    damage = model.evolution.D(ops.state.kappa).reshape(asm.n_elements,
                                                        asm.n_qp).mean(axis=1)
    return result, damage, asm


def test_the_local_model_runs_through_the_standard_assembler():
    """No new assembly code: the damage model implements the same interface as
    the plasticity model, so ElastoplasticAssembler drives it unchanged --
    including the unsymmetric tangent, which it forwards from the material."""
    result, damage, asm = _run_local(20, 1.0e-3, 16)
    assert all(s.converged for s in result.step_info)
    assert not asm.symmetric
    assert damage.max() > 0.0


def test_the_local_band_collapses_onto_the_mesh():
    """The pathology, and the reason the gradient model exists. The damaged
    region is a fixed number of *elements* however fine the mesh, so its
    physical length halves whenever the element size does.
    """
    lengths = {}
    for nx in (40, 80):
        _, damage, _ = _run_local(nx, 1.2e-3, 30)
        profile = damage.reshape(nx, -1).mean(axis=1)
        n_damaged = int((profile > 0.05 * profile.max()).sum())
        lengths[nx] = n_damaged * LENGTH / nx

    assert lengths[40] > 0.0
    assert lengths[80] < 0.75 * lengths[40]
