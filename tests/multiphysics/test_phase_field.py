"""
Verification for the coupled phase-field driver.

The constitutive pieces are already covered in ``tests/fracture``. What is
tested here is the FEM assembly and the two-field wiring, against solutions
that are known in closed form.

python -m pytest tests/multiphysics/test_phase_field.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.core.domain import Domain
from simulacrum.mesh.mesh import ElementBlock, Mesh
from simulacrum.materials.fracture import PhaseFieldModel
from simulacrum.multiphysics import Field, PhaseFieldFractureProblem

E, NU = 210e3, 0.3


def grid(nx, ny, lx=1.0, ly=1.0):
    xs, ys = np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    conn = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1,
                      j * (nx + 1) + i + nx + 2, j * (nx + 1) + i + nx + 1]
                     for j in range(ny) for i in range(nx)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)


def strip(n, L=4.0, h=0.05):
    """One-element-thick strip, for the 1-D crack profile."""
    xs = np.linspace(-L / 2, L / 2, n + 1)
    nodes = np.array([[x, y] for y in (0.0, h) for x in xs])
    conn = np.array([[i, i + 1, i + n + 2, i + n + 1] for i in range(n)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2), xs


# ---------------------------------------------------------------------------
# Field
# ---------------------------------------------------------------------------

def test_field_prescribe_and_scale():
    f = Field("u", 6, 2)
    f.prescribe([0, 1], 0, 0.0)
    f.prescribe([4, 5], 0, 1.0)
    base = {f.dof_map.dof(4, 0): 1.0, f.dof_map.dof(5, 0): 1.0}
    f.scale_prescribed(0.25, base)
    f.apply_prescribed()
    assert f.nodal()[4, 0] == pytest.approx(0.25)
    assert f.nodal()[0, 0] == 0.0
    assert set(f.free_dofs) == set(range(12)) - set(f.prescribed_dofs)


# ---------------------------------------------------------------------------
# Momentum equation
# ---------------------------------------------------------------------------

def test_undamaged_problem_is_a_patch_test():
    """With d = 0 and no residual stiffness the momentum equation must
    reproduce linear elasticity exactly -- displacement field, Poisson ratio
    and reaction."""
    dom = grid(6, 6)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.1,
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.set_displacement_control(np.where(nd[:, 0] > 1 - 1e-9)[0], 0, 1e-4)

    p.begin_step(1, 1.0, 0.0)
    p.solve_first()

    ux = p.u.nodal()[:, 0]
    assert np.abs(ux - 1e-4 * nd[:, 0]).max() < 1e-14

    eps = p.strains()
    assert eps[:, 0, 0].std() < 1e-16                       # uniform state
    # plane strain with sigma_yy = 0
    assert eps[:, 1, 1].mean() / eps[:, 0, 0].mean() == pytest.approx(
        -NU / (1 - NU), rel=1e-10)
    assert p.report()["reactions"]["control"] == pytest.approx(
        E / (1 - NU ** 2) * 1e-4, rel=1e-10)


@pytest.mark.parametrize("coupling", ["isotropic", "anisotropic", "hybrid"])
def test_all_couplings_agree_when_undamaged(coupling):
    dom = grid(4, 4)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.1, coupling=coupling,
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.set_displacement_control(np.where(nd[:, 0] > 1 - 1e-9)[0], 0, 1e-4)
    p.begin_step(1, 1.0, 0.0)
    p.solve_first()
    assert np.abs(p.u.nodal()[:, 0] - 1e-4 * nd[:, 0]).max() < 1e-12


def test_uniform_damage_scales_stiffness_by_g():
    """A uniformly damaged body is a softer elastic body, by exactly g(d)."""
    dom = grid(4, 4)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.1, coupling="hybrid",
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.set_displacement_control(np.where(nd[:, 0] > 1 - 1e-9)[0], 0, 1e-4)
    p.begin_step(1, 1.0, 0.0)

    p.solve_first()
    r0 = p.report()["reactions"]["control"]
    p.d.values[:] = 0.4
    p.solve_first()
    r1 = p.report()["reactions"]["control"]
    assert r1 / r0 == pytest.approx((1 - 0.4) ** 2, rel=1e-10)


# ---------------------------------------------------------------------------
# Phase-field equation: the analytic 1-D crack profiles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diss,exact", [
    ("AT2", lambda x, l: np.exp(-np.abs(x) / l)),
    ("AT1", lambda x, l: np.maximum(1 - np.abs(x) / (2 * l), 0.0) ** 2),
])
def test_one_dimensional_crack_profile(diss, exact):
    """The optimal profile is known in closed form for both AT models, and it
    is the sharpest available check on the phase-field assembly:

        AT2:  d(x) = exp(-|x|/l)                 exponential tails
        AT1:  d(x) = (1 - |x|/2l)^2 on |x| < 2l  compact support

    The compact support is what distinguishes AT1, and it only appears if the
    d >= 0 bound is actually enforced rather than clipped.
    """
    n, ell, L, height = 800, 0.25, 4.0, 0.05
    dom, xs = strip(n, L, height)
    model = PhaseFieldModel(E, NU, 1.0, ell, dissipation=diss,
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)

    # force d = 1 at the centre through the history field
    xq = np.einsum("eqa,ea->eq", p.N, dom.coordinates[p.conn][:, :, 0]).ravel()
    p.state["H"] = np.where(np.abs(xq) < L / n + 1e-9, 1e12, 0.0)
    p.state.commit()
    p.solve_second()

    d = p.d.values[:n + 1]
    assert np.abs(d - exact(xs, ell)).max() < 0.03
    # surface energy / Gc must recover the crack length (1 here, per height)
    assert p.surface_energy() / height == pytest.approx(1.0, abs=0.03)


def test_at1_profile_has_compact_support():
    n, ell, L = 800, 0.25, 4.0
    dom, xs = strip(n, L)
    model = PhaseFieldModel(E, NU, 1.0, ell, dissipation="AT1",
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)
    xq = np.einsum("eqa,ea->eq", p.N, dom.coordinates[p.conn][:, :, 0]).ravel()
    p.state["H"] = np.where(np.abs(xq) < L / n + 1e-9, 1e12, 0.0)
    p.state.commit()
    p.solve_second()

    d = p.d.values[:n + 1]
    assert (d >= -1e-12).all()                      # the bound is respected
    support = xs[d > 1e-6]
    assert support.max() == pytest.approx(2 * ell, abs=0.02)
    assert d[np.abs(xs) > 2.5 * ell].max() < 1e-9   # exactly zero outside


def test_at2_profile_has_exponential_tails():
    """AT2 damage never reaches zero -- the contrast with AT1."""
    n, ell, L = 800, 0.25, 4.0
    dom, xs = strip(n, L)
    model = PhaseFieldModel(E, NU, 1.0, ell, dissipation="AT2",
                            degradation_kwargs={"residual": 0.0})
    p = PhaseFieldFractureProblem(dom, model)
    xq = np.einsum("eqa,ea->eq", p.N, dom.coordinates[p.conn][:, :, 0]).ravel()
    p.state["H"] = np.where(np.abs(xq) < L / n + 1e-9, 1e12, 0.0)
    p.state.commit()
    p.solve_second()
    d = p.d.values[:n + 1]
    far = d[np.abs(xs) > 2.5 * ell]
    assert far.max() > 1e-6


# ---------------------------------------------------------------------------
# Coupled behaviour
# ---------------------------------------------------------------------------

def test_history_is_irreversible_across_steps():
    dom = grid(4, 4)
    model = PhaseFieldModel(E, NU, 2.7e-5, 0.2)
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.set_displacement_control(np.where(nd[:, 0] > 1 - 1e-9)[0], 0, 1e-2)

    p.run([1.0], stop_on_failure=False)
    d_loaded = p.d.values.copy()
    assert d_loaded.max() > 0.0

    p.run([0.0], stop_on_failure=False)          # unload completely
    assert (p.d.values >= d_loaded - 1e-12).all()


def test_rejected_step_does_not_pollute_the_history():
    """The whole reason QuadratureState separates trial from committed."""
    dom = grid(3, 3)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.2)
    p = PhaseFieldFractureProblem(dom, model)
    p.state["H"] = np.full(p.state.n_points, 5.0)
    p.state.commit()
    p.state["H"] = np.full(p.state.n_points, 1e6)      # a bad iterate
    p.reject_step(1)
    assert np.allclose(p.state["H"], 5.0)


def test_mesh_size_check_fires_when_ell_is_unresolved():
    dom = grid(4, 4)
    p = PhaseFieldFractureProblem(dom, PhaseFieldModel(E, NU, 2.7e-3, 0.05))
    with pytest.warns(RuntimeWarning, match="under-resolved"):
        p.check_mesh()


def test_energy_is_the_sum_of_both_contributions():
    """The staggered criterion monitors this; one part alone can decrease
    while the total does not."""
    dom = grid(4, 4)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.2)
    p = PhaseFieldFractureProblem(dom, model)
    p.u.values[:] = 1e-3 * np.tile([1.0, 0.0], dom.n_nodes)
    p.d.values[:] = 0.3
    total = p.total_energy()
    assert total == pytest.approx(p._energies["elastic"]
                                  + p._energies["fracture"])
    assert p._energies["fracture"] > 0.0


def test_one_way_strategy_solves_each_field_once():
    dom = grid(3, 3)
    model = PhaseFieldModel(E, NU, 2.7e-3, 0.2)
    p = PhaseFieldFractureProblem(dom, model, strategy="one_way")
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.set_displacement_control(np.where(nd[:, 0] > 1 - 1e-9)[0], 0, 1e-4)
    r = p.solve_step(1, 1.0, 0.0)
    assert r.converged and r.staggered.iterations == 1


def test_monolithic_strategy_is_refused_rather_than_faked():
    dom = grid(2, 2)
    with pytest.raises(NotImplementedError, match="coupled tangent"):
        PhaseFieldFractureProblem(dom, PhaseFieldModel(E, NU, 2.7e-3, 0.2),
                                  strategy="monolithic")


# ---------------------------------------------------------------------------
# Staggered convergence: the two modes the energy criterion cannot see
# ---------------------------------------------------------------------------

def test_energy_criterion_survives_a_converged_state():
    """An elastic step leaves the energy constant to round-off. Normalising by
    a round-off span divides noise by noise; the guard must call it converged
    rather than iterating to the cap."""
    from simulacrum.solvers.staggered import normalized_slope
    E = 1.9361352264e-3 + 1e-13 * np.array([1.0, -0.4, 0.6, -0.5, 0.5])
    assert normalized_slope(E) == 0.0


def test_energy_criterion_is_blind_to_linear_drift():
    """gamma measures shape, not magnitude: a steady creep pins it near 45 deg
    no matter how small the creep is. This is why a second test is needed."""
    from simulacrum.solvers.staggered import normalized_slope
    # slopes above the round-off guard: the drift is real but gamma cannot
    # see how small it is
    for slope in (1e-3, 1e-6, 1e-8):
        E = 1.0 - slope * np.arange(50)
        assert 44.0 < normalized_slope(E) < 47.0
    # below the guard the drift is round-off and the guard reports converged
    assert normalized_slope(1.0 - 1e-14 * np.arange(50)) == 0.0


def test_field_change_test_stops_a_linear_drift():
    from simulacrum.solvers.staggered import StaggeredSolver
    state = {"d": 1.0, "E": 1.0}

    def second():
        state["d"] *= 0.5
        return 1

    def energy():
        state["E"] -= 1e-9          # steady creep: gamma stays near 45
        return state["E"]

    res = StaggeredSolver(lambda: 1, second, energy, tol=10.0,
                          max_iterations=60, change=lambda: state["d"],
                          change_tol=1e-4).run()
    assert res.converged
    assert res.stopped_on == "field"
    assert res.criterion > 40.0            # the energy test never fired


def test_damage_change_is_recorded_and_decreases():
    dom = grid(6, 6)
    model = PhaseFieldModel(E, NU, 2.7e-5, 0.2)
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.set_displacement_control(np.where(nd[:, 1] > 1 - 1e-9)[0], 1, 1e-2)

    assert p.field_change() == float("inf")     # before any solve
    p.begin_step(1, 1.0, 0.0)
    p.solve_first()
    p.solve_second()
    first = p.field_change()
    assert 0.0 <= first <= 1.0
    p.solve_first()
    p.solve_second()
    assert p.field_change() <= first + 1e-12    # settles, does not grow


def test_hybrid_damage_is_monotone_across_steps():
    """Regression for the compression constraint: writing it back onto the
    nodal field let it undo irreversibility, and damage flickered 1 -> 0 -> 1
    between staggered iterations."""
    dom = grid(8, 8)
    model = PhaseFieldModel(E, NU, 2.7e-5, 0.2, coupling="hybrid")
    p = PhaseFieldFractureProblem(dom, model)
    nd = dom.coordinates
    p.u.prescribe(np.where(nd[:, 1] < 1e-9)[0], 1, 0.0)
    p.u.prescribe(np.where(nd[:, 0] < 1e-9)[0], 0, 0.0)
    p.set_displacement_control(np.where(nd[:, 1] > 1 - 1e-9)[0], 1, 1.0)

    previous = p.d.values.copy()
    for factor in (2e-3, 4e-3, 6e-3, 8e-3):
        p.solve_step(1, factor, 0.0)
        assert (p.d.values >= previous - 1e-12).all()
        previous = p.d.values.copy()
    assert previous.max() > 0.0                 # something actually damaged
