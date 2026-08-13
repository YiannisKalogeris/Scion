"""
Verification for the incremental-analysis path.

The pieces are individually tested elsewhere; what is checked here is the
wiring: that a load path runs, that the state advances exactly once per
accepted step, and that a rejected step leaves no trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.core.domain import Domain
from simulacrum.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
    StatefulOperators,
)
from simulacrum.materials.notation import from_mandel, voigt_strain_to_mandel
from simulacrum.materials.plasticity import (
    LinearIsotropic,
    PlasticityModel,
    von_mises_stress,
)
from simulacrum.mesh.mesh import ElementBlock, Mesh
from simulacrum.solvers.nonlinear import DisplacementControl

E, NU, Y, H = 210e3, 0.3, 250.0, 2000.0


def _stretch(asm, nodes, factor):
    """Displacement field of a uniform stretch, ``u_x = factor * x``.

    Setting every x-degree-of-freedom to the same value is a rigid translation
    and produces no strain at all -- an easy way to write a test that passes
    for the wrong reason.
    """
    u = np.zeros(asm.n_dof)
    u[0::2] = factor * nodes[:, 0]
    return u


def _problem(n=4, disp=0.004, n_steps=20):
    xs = np.linspace(0.0, 1.0, n + 1)
    nodes = np.array([[x, y] for y in xs for x in xs])
    conn = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1,
                      j * (n + 1) + i + n + 2, j * (n + 1) + i + n + 1]
                     for j in range(n) for i in range(n)])
    dom = Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)
    _problem.nodes = nodes

    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm = ElastoplasticAssembler(dom, model)
    ops = StatefulOperators(asm)

    dm, nd = asm.dof_map, dom.coordinates
    left = np.where(nd[:, 0] < 1e-12)[0]
    right = np.where(nd[:, 0] > 1 - 1e-12)[0]
    bot = np.where(nd[:, 1] < 1e-12)[0]
    fixed = np.concatenate([dm.element_dofs(left)[0::2],
                            dm.element_dofs(bot)[1::2]])
    pres = {int(d): disp for d in dm.element_dofs(right)[0::2]}
    free = np.setdiff1d(np.arange(asm.n_dof),
                        np.union1d(fixed, list(pres)))

    driver = DisplacementControl(
        ops.internal_force, ops.tangent, asm.n_dof, free, pres, fixed,
        n_steps=n_steps, tol=1e-9, max_iter=20, line_search=False,
        on_step_converged=ops.commit)
    return asm, ops, driver, model


# ---------------------------------------------------------------------------

def test_incremental_analysis_runs_to_completion():
    asm, ops, driver, model = _problem()
    result = driver.solve()

    assert all(s.converged for s in result.step_info)
    assert max(s.nr_iterations for s in result.step_info) <= 6
    assert ops.state.alpha.max() > 0.0            # it actually yielded


def test_the_converged_state_satisfies_the_yield_condition():
    """q = kappa(alpha) at every plastic point, which is the one statement that
    holds independently of the stress state."""
    asm, ops, driver, model = _problem()
    u = driver.solve().u_history[-1]

    eps_m = voigt_strain_to_mandel(asm.strains(u), 2)
    sigma_m = (eps_m - ops.state.eps_p) @ model.C_el.T
    q = von_mises_stress(from_mandel(sigma_m))
    kappa = model.isotropic.kappa(ops.state.alpha)

    assert np.abs(q - kappa).max() < 1e-8 * kappa.mean()


def test_state_advances_once_per_accepted_step():
    """Not once per Newton iteration. Committing inside the assembly would let
    a discarded iterate advance the plastic strain permanently."""
    asm, ops, driver, model = _problem(n_steps=5)
    seen = []
    original = ops.commit

    def spy(u):
        seen.append(ops.state.alpha.copy())
        original(u)

    driver.on_step_converged = spy
    result = driver.solve()

    assert len(seen) == len(result.step_info)
    # alpha is monotone across steps: plastic strain never unwinds
    for a, b in zip(seen, seen[1:]):
        assert np.all(b >= a - 1e-14)


def test_assembly_is_cached_within_an_iterate():
    """The driver calls f_int(u) and K_T(u) separately at the same u; without
    the cache each Newton iteration would run the return map twice."""
    asm, ops, driver, model = _problem(n_steps=10)
    result = driver.solve()
    total_iters = sum(s.nr_iterations for s in result.step_info)
    assert ops.n_assemblies < 2 * total_iters


def test_uncommitted_trial_assemblies_leave_the_state_alone():
    asm, ops, driver, model = _problem()
    before = ops.state.alpha.copy()

    for trial in (0.001, 0.005, 0.002):
        u = _stretch(asm, _problem.nodes, trial)
        ops.internal_force(u)
        ops.tangent(u)

    assert np.array_equal(ops.state.alpha, before)


def test_commit_promotes_the_trial_state_and_clears_the_cache():
    asm, ops, driver, model = _problem()
    u = _stretch(asm, _problem.nodes, 0.004)

    ops.internal_force(u)
    assert ops.last is not None
    trial_alpha = ops.last.state.alpha.copy()
    assert trial_alpha.max() > 0.0

    ops.commit()
    assert np.array_equal(ops.state.alpha, trial_alpha)
    assert ops.last is None                       # cache invalidated


def test_unloading_from_the_committed_state_is_elastic():
    """After a load path the material has hardened and sits exactly on its
    yield surface, so any unloading increment is elastic while continued
    loading is not.

    Reloading from *zero* total strain would not test this: the stored plastic
    strain makes ``eps - eps_p`` strongly negative there, so the point yields
    again in compression -- correct physics, and the opposite of what the name
    would suggest.
    """
    asm, ops, driver, model = _problem(disp=0.003, n_steps=10)
    result = driver.solve()
    u_final = result.u_history[-1]

    alpha = ops.state.alpha.max()
    assert alpha > 0.0

    unloaded = asm.assemble(0.999 * u_final, ops.state)
    assert not unloaded.plastic.any()

    loaded = asm.assemble(1.02 * u_final, ops.state)
    assert loaded.plastic.any()

    kappa = model.isotropic.kappa(ops.state.alpha).max()
    assert kappa == pytest.approx(Y + H * alpha, rel=1e-8)
    assert kappa > Y                              # it hardened
