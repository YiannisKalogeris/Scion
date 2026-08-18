"""
Verification for the phase-field fracture components.

Each of the four axes is tested independently, because that is the whole point
of keeping them orthogonal: a new split, degradation function or dissipation
profile should be provable correct on its own, before any solver exists.

python -m pytest tests/fracture/test_fracture.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
# not np.trapezoid / np.trapz: NumPy renamed one to the other in 2.0 and
# removed the old name, so either spelling pins the library to one major
# version. SciPy is already a hard dependency and carries both.
from scipy.integrate import trapezoid

from simulacrum.discretization.fem.quadrature_state import QuadratureState
from simulacrum.materials.fracture import (
    AT1,
    AT2,
    PhaseFieldModel,
    get_coupling,
    get_degradation,
    get_dissipation,
    get_split,
    voigt_tangent,
)
from simulacrum.solvers.bound_constrained import projected_gauss_seidel, solve_bounded
from simulacrum.solvers.staggered import (
    StaggeredSolver,
    normalized_slope,
    relative_energy_change,
)

LAM, MU = 121.15, 80.77
SPLITS = ["none", "amor", "spectral"]


def _random_strains(n=200, dim=2, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    A = scale * rng.standard_normal((n, dim, dim))
    return 0.5 * (A + np.swapaxes(A, 1, 2))


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SPLITS)
@pytest.mark.parametrize("dim", [2, 3])
def test_split_energies_sum_to_total(name, dim):
    """Psi_plus + Psi_minus = Psi_0, exactly. A split partitions; it does not
    create or destroy energy."""
    sp = get_split(name)
    eps = _random_strains(dim=dim)
    pp, pm = sp.energies(eps, LAM, MU)
    total = sp.total_energy(eps, LAM, MU)
    assert np.abs(pp + pm - total).max() < 1e-12 * np.abs(total).max()


@pytest.mark.parametrize("name", SPLITS)
@pytest.mark.parametrize("dim", [2, 3])
def test_split_stress_is_energy_derivative(name, dim):
    """sigma_pm = d Psi_pm / d eps, by central differences."""
    sp = get_split(name)
    eps = _random_strains(dim=dim)
    sig_p, sig_m, _, _ = sp.stress_tangent(eps, LAM, MU)

    h = 1e-6
    for i in range(dim):
        for j in range(dim):
            E = np.zeros((dim, dim))
            E[i, j] = E[j, i] = 1.0 if i == j else 0.5
            pp_p, pm_p = sp.energies(eps + h * E, LAM, MU)
            pp_m, pm_m = sp.energies(eps - h * E, LAM, MU)
            assert np.abs((pp_p - pp_m) / (2 * h) - sig_p[:, i, j]).max() < 1e-5
            assert np.abs((pm_p - pm_m) / (2 * h) - sig_m[:, i, j]).max() < 1e-5


@pytest.mark.parametrize("name", SPLITS)
@pytest.mark.parametrize("dim", [2, 3])
def test_split_tangent_is_stress_derivative(name, dim):
    """C_pm = d sigma_pm / d eps. This is where the spectral split's
    eigenprojection derivative gets checked."""
    sp = get_split(name)
    eps = _random_strains(dim=dim)
    _, _, Cp, Cm = sp.stress_tangent(eps, LAM, MU)

    h = 1e-6
    for k in range(dim):
        for l in range(dim):
            E = np.zeros((dim, dim))
            E[k, l] = E[l, k] = 1.0 if k == l else 0.5
            a = sp.stress_tangent(eps + h * E, LAM, MU)
            b = sp.stress_tangent(eps - h * E, LAM, MU)
            assert np.abs((a[0] - b[0]) / (2 * h) - Cp[:, :, :, k, l]).max() < 1e-4
            assert np.abs((a[1] - b[1]) / (2 * h) - Cm[:, :, :, k, l]).max() < 1e-4


@pytest.mark.parametrize("name", SPLITS)
@pytest.mark.parametrize("dim", [2, 3])
def test_split_tangents_recover_elasticity(name, dim):
    """C_plus + C_minus = C_elastic, so an undamaged body is exactly elastic
    whatever split it was built with."""
    sp = get_split(name)
    eps = _random_strains(dim=dim)
    _, _, Cp, Cm = sp.stress_tangent(eps, LAM, MU)
    I = np.eye(dim)
    C_el = (LAM * np.einsum("ij,kl->ijkl", I, I)
            + 2 * MU * 0.5 * (np.einsum("ik,jl->ijkl", I, I)
                              + np.einsum("il,jk->ijkl", I, I)))
    assert np.abs(Cp + Cm - C_el).max() < 1e-10 * np.abs(C_el).max()


@pytest.mark.parametrize("gap", [1e-2, 1e-8, 1e-14, 0.0])
def test_spectral_split_handles_repeated_eigenvalues(gap):
    """theta_IJ = (f_I - f_J)/(e_I - e_J) is 0/0 under equibiaxial strain.
    The limit branch must be taken, and must still give the right tangent."""
    sp = get_split("spectral")
    eps = np.zeros((1, 2, 2))
    eps[0, 0, 0], eps[0, 1, 1] = 0.01, 0.01 + gap
    _, _, Cp, Cm = sp.stress_tangent(eps, LAM, MU)
    assert np.isfinite(Cp).all() and np.isfinite(Cm).all()

    h = 1e-7
    for k in range(2):
        for l in range(2):
            E = np.zeros((2, 2))
            E[k, l] = E[l, k] = 1.0 if k == l else 0.5
            a = sp.stress_tangent(eps + h * E, LAM, MU)[0]
            b = sp.stress_tangent(eps - h * E, LAM, MU)[0]
            assert np.abs((a - b) / (2 * h) - Cp[:, :, :, k, l]).max() < 1e-4


def test_splits_disagree_only_where_they_should():
    """Under pure tension every split gives the same answer; the differences
    appear in compression, which is the entire reason splits exist."""
    tension = np.zeros((1, 2, 2))
    tension[0, 0, 0] = tension[0, 1, 1] = 0.01
    psi = [get_split(s).energies(tension, LAM, MU)[0][0] for s in SPLITS]
    assert max(psi) - min(psi) < 1e-12 * max(psi)

    compression = -tension
    psi_c = [get_split(s).energies(compression, LAM, MU)[0][0] for s in SPLITS]
    assert psi_c[0] > 0.0          # NoSplit still drives damage
    assert psi_c[1] < 1e-12        # Amor: pure volumetric compression
    assert psi_c[2] < 1e-12        # spectral: all principal strains negative


def test_amor_degrades_shear_under_full_compression():
    """The documented weakness of Amor's split: with all principal strains
    negative but a nonzero deviator, Psi_plus is still positive, so damage can
    grow in compression. The spectral split gives zero there."""
    eps = np.zeros((1, 2, 2))
    eps[0, 0, 0], eps[0, 1, 1] = -0.02, -0.005
    psi_amor = get_split("amor").energies(eps, LAM, MU)[0][0]
    psi_spec = get_split("spectral").energies(eps, LAM, MU)[0][0]
    assert psi_amor > 0.0
    assert psi_spec < 1e-14


def test_voigt_tangent_is_symmetric_and_matches_tensor_form():
    """The Voigt matrix must pair with the engineering-shear B matrix."""
    sp = get_split("spectral")
    eps = _random_strains(n=20, dim=2)
    _, _, Cp, _ = sp.stress_tangent(eps, LAM, MU)
    Cv = voigt_tangent(Cp)
    assert np.abs(Cv - np.swapaxes(Cv, 1, 2)).max() < 1e-8 * np.abs(Cv).max()

    # sigma_voigt = C_voigt @ eps_voigt with gamma = 2 eps_xy
    ev = np.stack([eps[:, 0, 0], eps[:, 1, 1], 2 * eps[:, 0, 1]], axis=1)
    sv = np.einsum("nab,nb->na", Cv, ev)
    ref = np.einsum("nijkl,nkl->nij", Cp, eps)
    assert np.abs(sv[:, 0] - ref[:, 0, 0]).max() < 1e-8
    assert np.abs(sv[:, 1] - ref[:, 1, 1]).max() < 1e-8
    assert np.abs(sv[:, 2] - ref[:, 0, 1]).max() < 1e-8


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["quadratic", "cubic", "rational"])
def test_degradation_endpoints_and_derivatives(name):
    """g(0) = 1, g(1) = k, g'(1) = 0 -- the last one is what stops a fully
    broken point from continuing to drive damage."""
    g = get_degradation(name, residual=0.0)
    assert abs(g.g(np.array([0.0]))[0] - 1.0) < 1e-12
    assert abs(g.g(np.array([1.0]))[0]) < 1e-12
    assert abs(g.dg(np.array([1.0]))[0]) < 1e-12

    d = np.linspace(0.0, 1.0, 41)
    h = 1e-6
    assert np.abs((g.g(d + h) - g.g(d - h)) / (2 * h) - g.dg(d)).max() < 1e-5
    assert np.abs((g.dg(d + h) - g.dg(d - h)) / (2 * h) - g.ddg(d)).max() < 1e-4


@pytest.mark.parametrize("name", ["quadratic", "cubic", "rational"])
def test_degradation_is_monotone(name):
    g = get_degradation(name, residual=0.0)
    d = np.linspace(0.0, 1.0, 201)
    assert np.all(np.diff(g.g(d)) <= 1e-12)


def test_only_quadratic_degradation_keeps_the_problem_linear():
    assert get_degradation("quadratic").quadratic_in_d
    assert not get_degradation("cubic").quadratic_in_d
    assert not get_degradation("rational").quadratic_in_d


# ---------------------------------------------------------------------------
# Dissipation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [("AT2", 2.0), ("AT1", 8.0 / 3.0)])
def test_c_alpha_matches_its_definition(name, expected):
    """c_alpha = 4 INT_0^1 sqrt(alpha(s)) ds. Getting this wrong rescales Gc
    and every crack propagates at the wrong load."""
    diss = get_dissipation(name)
    s = np.linspace(0.0, 1.0, 200001)
    integral = trapezoid(np.sqrt(diss.alpha(s)), s)
    assert abs(4.0 * integral - expected) < 1e-5
    assert abs(diss.c_alpha - expected) < 1e-12


@pytest.mark.parametrize("name", ["AT1", "AT2"])
def test_dissipation_derivatives(name):
    diss = get_dissipation(name)
    d = np.linspace(0.0, 1.0, 41)
    h = 1e-6
    assert np.abs((diss.alpha(d + h) - diss.alpha(d - h)) / (2 * h)
                  - diss.dalpha(d)).max() < 1e-5


def test_at1_needs_bounds_and_at2_does_not():
    """alpha'(0) != 0 is exactly the condition that drives d negative."""
    assert get_dissipation("AT1").dalpha(np.array([0.0]))[0] != 0.0
    assert get_dissipation("AT1").needs_bound_constraint
    assert get_dissipation("AT2").dalpha(np.array([0.0]))[0] == 0.0
    assert not get_dissipation("AT2").needs_bound_constraint


def test_at1_strength_relation_is_self_consistent():
    E, Gc = 210e3, 2.7e-3
    ell = 0.02
    sc = AT1.critical_stress(E, Gc, ell)
    assert abs(AT1.length_for_strength(E, Gc, sc) - ell) < 1e-12 * ell


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------

def test_hybrid_borrows_stress_from_isotropic_and_energy_from_anisotropic():
    """The defining property of the hybrid formulation, as a test."""
    sp = get_split("spectral")
    eps = _random_strains(n=50, dim=2)
    gd = np.full(eps.shape[0], 0.6)
    iso, ani, hyb = (get_coupling(n) for n in ("isotropic", "anisotropic", "hybrid"))

    s_i, C_i = iso.stress_tangent(sp, eps, LAM, MU, gd)
    s_h, C_h = hyb.stress_tangent(sp, eps, LAM, MU, gd)
    assert np.abs(s_i - s_h).max() < 1e-12 * np.abs(s_i).max()
    assert np.abs(C_i - C_h).max() < 1e-12 * np.abs(C_i).max()

    assert np.abs(ani.driving_energy(sp, eps, LAM, MU)
                  - hyb.driving_energy(sp, eps, LAM, MU)).max() < 1e-14

    s_a, _ = ani.stress_tangent(sp, eps, LAM, MU, gd)
    assert np.abs(s_a - s_h).max() > 1e-6      # and it is genuinely different


def test_only_hybrid_needs_the_compression_constraint():
    assert get_coupling("hybrid").enforce_compression_constraint
    assert not get_coupling("isotropic").enforce_compression_constraint
    assert not get_coupling("anisotropic").enforce_compression_constraint


def test_compression_constraint_zeroes_damage_under_compression():
    model = PhaseFieldModel(210e3, 0.3, 2.7e-3, 0.02, coupling="hybrid")
    eps = np.zeros((2, 2, 2))
    eps[0] = 0.01 * np.eye(2)          # tension
    eps[1] = -0.01 * np.eye(2)         # compression
    d = model.apply_compression_constraint(np.array([0.7, 0.7]), eps)
    assert d[0] == pytest.approx(0.7)
    assert d[1] == 0.0


def test_isotropic_coupling_leaves_damage_alone():
    model = PhaseFieldModel(210e3, 0.3, 2.7e-3, 0.02, coupling="isotropic")
    eps = -0.01 * np.tile(np.eye(2), (2, 1, 1))
    d = model.apply_compression_constraint(np.array([0.7, 0.3]), eps)
    assert np.allclose(d, [0.7, 0.3])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_undamaged_model_is_exactly_linear_elastic():
    """d = 0 with zero residual stiffness must reproduce Hooke's law for every
    combination of split and coupling."""
    eps = _random_strains(n=50, dim=2)
    I = np.eye(2)
    C_el = (LAM * np.einsum("ij,kl->ijkl", I, I)
            + 2 * MU * 0.5 * (np.einsum("ik,jl->ijkl", I, I)
                              + np.einsum("il,jk->ijkl", I, I)))
    sigma_el = np.einsum("ijkl,nkl->nij", C_el, eps)

    E = MU * (3 * LAM + 2 * MU) / (LAM + MU)
    nu = LAM / (2 * (LAM + MU))
    for split in SPLITS:
        for coupling in ("isotropic", "anisotropic", "hybrid"):
            m = PhaseFieldModel(E, nu, 1.0, 0.1, split=split,
                                coupling=coupling,
                                degradation_kwargs={"residual": 0.0})
            sigma, C = m.stress_tangent(eps, np.zeros(eps.shape[0]))
            assert np.abs(sigma - sigma_el).max() < 1e-8 * np.abs(sigma_el).max()
            assert np.abs(C - C_el).max() < 1e-8 * np.abs(C_el).max()


def test_history_field_never_decreases():
    """Irreversibility. H is what makes cracks unable to heal."""
    m = PhaseFieldModel(210e3, 0.3, 2.7e-3, 0.02)
    H = np.zeros(10)
    big = 0.02 * np.tile(np.eye(2), (10, 1, 1))
    H = m.update_history(H, big)
    peak = H.copy()
    assert (peak > 0).all()
    H = m.update_history(H, 0.1 * big)          # unload
    assert np.allclose(H, peak)


def test_reaction_is_derivative_of_itself_and_linear_where_promised():
    for diss in ("AT1", "AT2"):
        m = PhaseFieldModel(210e3, 0.3, 2.7e-3, 0.02, dissipation=diss)
        d = np.linspace(0.0, 0.9, 25)
        H = np.linspace(0.0, 1.0, 25)
        h = 1e-6
        fd = (m.reaction(d + h, H) - m.reaction(d - h, H)) / (2 * h)
        assert np.abs(fd - m.dreaction(d, H)).max() < 1e-4 * max(
            1.0, np.abs(m.dreaction(d, H)).max())
        assert m.linear_phase_field          # quadratic degradation


def test_mesh_size_check_warns():
    m = PhaseFieldModel(210e3, 0.3, 2.7e-3, 0.02)
    with pytest.warns(RuntimeWarning, match="under-resolved"):
        m.check_mesh_size(0.05)
    m.check_mesh_size(0.005)                 # must not warn


# ---------------------------------------------------------------------------
# Quadrature state
# ---------------------------------------------------------------------------

def test_quadrature_state_commit_and_restore():
    """A rejected iterate must not leave a permanent mark on the history."""
    st = QuadratureState(n_elements=4, n_qp=4)
    st.add("H")
    assert st["H"].shape == (16,)

    st["H"] = np.arange(16.0)
    st.commit()
    st["H"] = np.full(16, 999.0)             # a bad iterate
    st.restore()
    assert np.allclose(st["H"], np.arange(16.0))
    assert np.allclose(st.committed("H"), np.arange(16.0))


def test_quadrature_state_shapes_and_views():
    st = QuadratureState(3, 4)
    st.add("eps_p", shape=(2, 2))
    assert st["eps_p"].shape == (12, 2, 2)
    assert st.element_view("eps_p").shape == (3, 4, 2, 2)
    st["eps_p"] = np.ones((12, 2, 2))
    assert st.element_average("eps_p").shape == (3, 2, 2)
    assert np.allclose(st.element_average("eps_p"), 1.0)


def test_quadrature_state_rejects_unknown_field():
    st = QuadratureState(2, 1)
    with pytest.raises(KeyError):
        st["nope"] = np.zeros(2)


# ---------------------------------------------------------------------------
# Bound-constrained solver
# ---------------------------------------------------------------------------

def _spd(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def test_bounded_solve_reduces_to_linear_when_inactive():
    A, n = _spd(30), 30
    x_exact = np.random.default_rng(1).standard_normal(n)
    b = A @ x_exact
    r = solve_bounded(A, b, lo=x_exact.min() - 1, hi=x_exact.max() + 1)
    assert r.converged and r.n_active == 0
    assert np.abs(r.x - x_exact).max() < 1e-8


def test_bounded_solve_satisfies_kkt():
    """Clipping an unconstrained solve is *not* the answer: the constrained
    optimum differs, and this test is what catches that shortcut."""
    n = 40
    A = _spd(n, seed=3)
    rng = np.random.default_rng(4)
    b = rng.standard_normal(n)
    r = solve_bounded(A, b, lo=0.0)
    assert r.converged
    assert (r.x >= -1e-12).all()

    lam = A @ r.x - b
    assert (lam[r.x <= 1e-12] >= -1e-8).all()          # multiplier sign
    assert np.abs(lam[r.x > 1e-12]).max() < 1e-8       # stationarity

    clipped = np.clip(np.linalg.solve(A, b), 0.0, None)
    energy = lambda x: 0.5 * x @ A @ x - b @ x
    assert energy(r.x) <= energy(clipped) + 1e-12


def test_bounded_solve_sparse_matches_dense():
    from scipy import sparse as sp
    n = 50
    A = _spd(n, seed=5)
    b = np.random.default_rng(6).standard_normal(n)
    rd = solve_bounded(A, b, lo=0.0, hi=0.5)
    rs = solve_bounded(sp.csr_matrix(A), b, lo=0.0, hi=0.5)
    assert rd.converged and rs.converged
    assert np.abs(rd.x - rs.x).max() < 1e-9


def test_projected_gauss_seidel_agrees_with_active_set():
    n = 25
    A = _spd(n, seed=7)
    b = np.random.default_rng(8).standard_normal(n)
    r1 = solve_bounded(A, b, lo=0.0)
    r2 = projected_gauss_seidel(A, b, lo=0.0, max_iter=5000, tol=1e-12)
    assert r2.converged
    assert np.abs(r1.x - r2.x).max() < 1e-6


# ---------------------------------------------------------------------------
# Staggered driver
# ---------------------------------------------------------------------------

def test_normalized_criterion_does_not_stop_early():
    """The paper's central algorithmic point: on a slowly-decaying energy the
    naive measures stop after two iterations while the normalized slope keeps
    going. Reproduced here on a synthetic sequence with the same character."""
    k = np.arange(1, 62)
    E = 3.14 + 0.11 * np.exp(-(k - 1) / 12.0)

    stops_naive = next(i for i in range(2, len(E))
                       if relative_energy_change(E[:i]) <= 1.0)
    stops_gamma = next(i for i in range(3, len(E))
                       if normalized_slope(E[:i]) <= 10.0)
    assert stops_naive <= 3
    assert stops_gamma > 3 * stops_naive


def test_staggered_solver_converges_and_records():
    state = {"x": 1.0, "y": 1.0}

    def first():
        state["x"] *= 0.5
        return 1

    def second():
        state["y"] *= 0.5
        return 2

    energy = lambda: state["x"] ** 2 + state["y"] ** 2

    res = StaggeredSolver(first, second, energy, tol=45.0,
                          max_iterations=50).run()
    assert res.converged
    assert res.iterations >= 2
    assert len(res.energies) == res.iterations
    assert res.energies == sorted(res.energies, reverse=True)
    assert res.inner_iterations[0] == (1, 2)


def test_staggered_solver_reports_failure():
    res = StaggeredSolver(lambda: None, lambda: None,
                          lambda: 1.0, tol=-1.0, max_iterations=5).run()
    assert not res.converged and res.iterations == 5
