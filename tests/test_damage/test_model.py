"""
Verification for the damage model at a material point.

The consistent tangent is finite-differenced for every combination of the three
axes, in both 2-D and 3-D. A wrong tangent still converges, linearly, so
nothing but this check finds it.

The other half of the file is about what makes damage *damage*: secant
unloading, no permanent strain, and a history variable that never decreases.
A test written on the loading branch alone passes for a plasticity model too.

python -m pytest tests/test_damage/test_model.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.materials.damage import (
    DamageModel,
    DamageState,
    EnergyRelease,
    ExponentialSoftening,
    GradientDamageModel,
    LinearSoftening,
)

E, NU = 30.0e3, 0.2
K0 = 1.0e-4
LAW = ExponentialSoftening(K0, alpha=0.99, beta=500.0)

COMBOS = [
    ("mazars", None),
    ("modified-von-mises", None),
    ("rankine", None),
    ("energy-release", None),
    ("mazars", "spectral"),
    ("mazars", "amor"),
    ("modified-von-mises", "spectral"),
    ("energy-release", "spectral"),
]


def _model(equivalent, split):
    kw = {"k": 10.0} if equivalent == "modified-von-mises" else {}
    return DamageModel(E, NU, LAW, equivalent=equivalent,
                       equivalent_kwargs=kw, split=split)


TARGET_EQ = 5.0 * K0


def _loading_strains(model, dim, seed=0, n=6, target=TARGET_EQ):
    """Random states rescaled so every one has ``eps_eq == target``.

    Every measure here is positively homogeneous of degree one in the strain, so
    scaling a state scales its equivalent strain by the same factor. That makes
    it possible to put all the sample points at a known distance above the
    threshold rather than hoping a random draw lands there -- with an unbiased
    draw, roughly a third of them do not, and a tangent check that silently
    includes an unloading point is checking the wrong branch.
    """
    rng = np.random.default_rng(seed)
    nv = 3 if dim == 2 else 6
    eps = rng.normal(scale=3.0e-4, size=(n, nv))
    eps[:, :(2 if dim == 2 else 3)] += 4.0e-4          # bias into tension
    eq = model.equivalent_strain(eps, dim=dim)
    assert np.all(eq > 0.0)
    return eps * (target / eq)[:, None]


def _loading_state(model, eps_voigt, dim):
    """A committed history strictly below the current equivalent strain.

    The tangent is only defined on the loading branch, and the response has a
    kink exactly at ``eps_eq = kappa``: a central difference taken there
    straddles the kink and averages the loading tangent with the secant. Sitting
    well inside the branch is not weakening the test, it is testing the branch
    the tangent claims to describe.
    """
    eq = model.equivalent_strain(eps_voigt, dim=dim)
    return DamageState(np.maximum(0.5 * eq, 1.01 * K0))


def _fd_tangent(model, eps_voigt, state, dim, h=1.0e-10):
    nv = eps_voigt.shape[1]
    C = np.zeros((eps_voigt.shape[0], nv, nv))
    for b in range(nv):
        p = np.zeros_like(eps_voigt)
        p[:, b] = h
        sp, _, _ = model.update(eps_voigt + p, state, dim=dim)
        sm, _, _ = model.update(eps_voigt - p, state, dim=dim)
        C[:, :, b] = (sp - sm) / (2.0 * h)
    return C


# ---------------------------------------------------------------------------
# The consistent tangent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("equivalent,split", COMBOS,
                         ids=[f"{e}-{s}" for e, s in COMBOS])
def test_the_tangent_is_the_derivative_of_the_stress(equivalent, split, dim):
    model = _model(equivalent, split)
    eps = _loading_strains(model, dim)
    state = _loading_state(model, eps, dim)
    _, C, res = model.update(eps, state, dim=dim)

    assert res.loading.all(), "the check is meaningless off the loading branch"
    fd = _fd_tangent(model, eps, state, dim)
    assert np.abs(C - fd).max() < 1e-5 * np.abs(fd).max()


@pytest.mark.parametrize("equivalent,split", COMBOS,
                         ids=[f"{e}-{s}" for e, s in COMBOS])
def test_the_symmetry_flag_tells_the_truth(equivalent, split):
    """Claiming symmetry wrongly makes a Cholesky factorisation of an
    unsymmetric matrix return nonsense, so the flag has to be checked against
    the matrix rather than trusted."""
    model = _model(equivalent, split)
    eps = _loading_strains(model, 3, seed=1)
    _, C, _ = model.update(eps, _loading_state(model, eps, 3), dim=3)
    asymmetry = np.abs(C - np.swapaxes(C, 1, 2)).max() / np.abs(C).max()
    if model.symmetric_tangent:
        assert asymmetry < 1e-12
    else:
        assert asymmetry > 1e-3


def test_only_the_energy_measure_without_a_split_is_symmetric():
    assert _model("energy-release", None).symmetric_tangent
    assert not _model("energy-release", "spectral").symmetric_tangent
    assert not _model("mazars", None).symmetric_tangent


def test_the_secant_is_recovered_on_unloading():
    """No rank-one term when the point is not loading, and the tangent is then
    exactly the degraded elastic one."""
    model = _model("mazars", None)
    eps = _loading_strains(model, 3, seed=2)
    eq = model.equivalent_strain(eps, dim=3)
    state = DamageState(2.0 * np.maximum(eq, K0))        # everything unloading

    _, C, res = model.update(eps, state, dim=3)
    assert not res.loading.any()

    secant = (1.0 - res.damage)[:, None, None] * model.elastic_tangent_voigt(3)
    assert np.allclose(C, secant)
    assert np.allclose(C, np.swapaxes(C, 1, 2))          # and symmetric


# ---------------------------------------------------------------------------
# What makes damage damage
# ---------------------------------------------------------------------------

def test_unloading_returns_to_the_origin_with_no_permanent_strain():
    """The property that separates damage from plasticity. A plastic model
    unloads on the *elastic* slope and keeps a permanent strain; a damaged one
    unloads on a reduced secant straight to the origin.

    A test written only on the loading branch cannot tell the two apart.
    """
    model = _model("mazars", None)
    state = model.initial_state(1)

    peak = np.zeros((1, 6))
    peak[0, 0] = 8.0e-4
    s_peak, _, res = model.update(peak, state, dim=3)
    state = res.state
    assert res.damage[0] > 0.1

    # unload to zero strain: the stress must vanish exactly
    s_zero, _, res0 = model.update(np.zeros((1, 6)), state, dim=3)
    assert np.abs(s_zero).max() < 1e-20
    assert res0.damage[0] == pytest.approx(res.damage[0])   # and stays damaged

    # halfway back down, the stress is halfway: a straight secant
    s_half, _, _ = model.update(0.5 * peak, state, dim=3)
    assert np.allclose(s_half, 0.5 * s_peak, rtol=1e-12)


def test_damage_never_decreases_and_reloading_follows_the_secant():
    model = _model("mazars", None)
    state = model.initial_state(1)

    peak = np.zeros((1, 6))
    peak[0, 0] = 8.0e-4
    s_peak, _, res = model.update(peak, state, dim=3)
    state = res.state
    kappa_peak, d_peak = state.kappa[0], res.damage[0]

    # a smaller strain must not grow the damage or the history
    _, _, res_small = model.update(0.3 * peak, state, dim=3)
    assert res_small.state.kappa[0] == pytest.approx(kappa_peak)
    assert res_small.damage[0] == pytest.approx(d_peak)
    assert not res_small.loading[0]

    # back to the previous maximum: the same stress, nothing new dissipated
    s_back, _, res_back = model.update(peak, state, dim=3)
    assert np.allclose(s_back, s_peak, rtol=1e-12)
    assert not res_back.loading[0]

    # beyond it: damage grows again
    _, _, res_more = model.update(1.5 * peak, state, dim=3)
    assert res_more.damage[0] > d_peak
    assert res_more.loading[0]


def test_the_committed_state_is_never_mutated():
    """A Newton iterate that is later discarded must not advance kappa."""
    model = _model("mazars", None)
    state = model.initial_state(4)
    before = state.kappa.copy()
    eps = np.zeros((4, 6))
    eps[:, 0] = 1.0e-3
    result = model.update(eps, state, dim=3)[2]
    assert np.array_equal(state.kappa, before)
    assert np.all(result.state.kappa > before)


def test_the_initial_history_sits_at_the_threshold():
    """Starting from zero would report every point as loading on the first
    increment, since any strain at all exceeds it."""
    model = _model("mazars", None)
    state = model.initial_state(3)
    assert np.all(state.kappa == K0)
    tiny = np.zeros((3, 6))
    tiny[:, 0] = 1.0e-6
    _, _, res = model.update(tiny, state, dim=3)
    assert not res.loading.any()
    assert np.all(res.damage == 0.0)


# ---------------------------------------------------------------------------
# Closed forms and conventions
# ---------------------------------------------------------------------------

def test_uniaxial_stress_matches_the_law_by_hand():
    """``sigma = (1 - D(kappa)) E eps`` on a uniaxial *stress* path, which is
    the curve the evolution law is calibrated against."""
    law = LinearSoftening(K0, 2.0e-3)
    model = DamageModel(E, NU, law, equivalent="mazars")
    state = model.initial_state(1)

    for axial in (5.0e-5, 3.0e-4, 1.0e-3, 1.9e-3):
        eps = np.zeros((1, 6))
        eps[0, 0] = axial
        eps[0, 1] = eps[0, 2] = -NU * axial
        sigma, _, res = model.update(eps, state, dim=3)
        assert res.damage[0] == pytest.approx(law.D(np.array([axial]))[0])
        assert sigma[0, 0] == pytest.approx(law.stress(axial, E), rel=1e-12)
        state = res.state


def test_plane_strain_agrees_with_the_three_dimensional_model():
    """The 2-D Voigt sub-block is exact, not a reduction: with
    eps_33 = gamma_13 = gamma_23 = 0 the dropped columns contribute nothing to
    the kept rows."""
    model = _model("modified-von-mises", None)
    eps2 = _loading_strains(model, 2, seed=5)
    eps3 = np.zeros((eps2.shape[0], 6))
    eps3[:, 0], eps3[:, 1], eps3[:, 5] = eps2[:, 0], eps2[:, 1], eps2[:, 2]

    state = _loading_state(model, eps2, 2)
    s2, C2, r2 = model.update(eps2, state, dim=2)
    s3, C3, r3 = model.update(eps3, state, dim=3)

    assert np.allclose(r2.eps_eq, r3.eps_eq)
    assert np.allclose(r2.damage, r3.damage)
    assert np.allclose(s2, s3[:, [0, 1, 5]])
    assert np.allclose(C2, C3[:, [0, 1, 5]][:, :, [0, 1, 5]])


def test_the_equivalent_strain_uses_the_full_three_dimensional_tensor():
    """In plane strain eps_33 = 0 but the deviator's e_33 is not, so a measure
    evaluated on the 2-D tensor would give a different number. Checked through
    the modified von Mises measure, which reads the deviator."""
    model = _model("modified-von-mises", None)
    eps2 = np.array([[1.0e-3, 0.0, 0.0]])
    got = model.equivalent_strain(eps2, dim=2)[0]

    plane_strain_state = np.zeros((1, 3, 3))
    plane_strain_state[0, 0, 0] = 1.0e-3
    expected = model.equivalent.value(plane_strain_state)[0]
    assert got == pytest.approx(expected)
    # and it is *not* the uniaxial-stress value, which is what a 2-D-only
    # measure would accidentally produce
    assert got != pytest.approx(model.equivalent.uniaxial(1.0e-3)[0], rel=1e-3)


def test_the_undamaged_tangent_is_the_elastic_one():
    for dim, nv in ((2, 3), (3, 6)):
        model = _model("mazars", None)
        C = model.elastic_tangent_voigt(dim)
        assert C.shape == (nv, nv)
        state = model.initial_state(1)
        eps = np.zeros((1, nv))
        eps[0, 0] = 1.0e-6                    # below the threshold
        _, C_update, _ = model.update(eps, state, dim=dim)
        assert np.allclose(C_update[0], C)


def test_the_assembler_facing_fields_are_present():
    """ElastoplasticAssembler reads plastic/apex/iterations/converged off any
    inelastic result; damage has no apex and no local solve."""
    model = _model("mazars", None)
    eps = np.zeros((2, 6))
    eps[:, 0] = 1.0e-3
    _, _, res = model.update(eps, model.initial_state(2), dim=3)
    assert np.array_equal(res.plastic, res.loading)
    assert not res.apex.any()
    assert res.iterations == 0 and res.converged


# ---------------------------------------------------------------------------
# Refusals and the gradient model
# ---------------------------------------------------------------------------

def test_plane_stress_is_refused():
    with pytest.raises(NotImplementedError, match="plane stress"):
        DamageModel(E, NU, LAW, plane_strain=False)


def test_a_measure_contradicting_the_model_is_refused():
    with pytest.raises(ValueError, match="must agree"):
        DamageModel(E, NU, LAW, equivalent=EnergyRelease(E=E, nu=0.35))


def test_the_split_decides_whether_momentum_is_linear():
    assert _model("mazars", None).linear_momentum
    assert not _model("mazars", "spectral").linear_momentum


def test_driving_damage_externally_removes_the_rank_one_term():
    """What gradient damage relies on: at fixed driver the damage does not
    respond to the local strain, so the tangent is the secant and symmetric."""
    model = _model("mazars", None)
    eps = _loading_strains(model, 3, seed=7)
    state = _loading_state(model, eps, 3)
    driver = 3.0 * np.maximum(model.equivalent_strain(eps, dim=3), K0)

    _, C, res = model.update(eps, state, dim=3, driver=driver)
    secant = (1.0 - res.damage)[:, None, None] * model.elastic_tangent_voigt(3)
    assert np.allclose(C, secant)
    assert np.allclose(res.state.kappa, driver)


def test_the_gradient_model_carries_a_length_and_checks_the_mesh():
    model = GradientDamageModel(E, NU, LAW, ell=0.5)
    assert model.gradient_coefficient() == pytest.approx(0.25)
    assert model.recommended_mesh_size(3.0) == pytest.approx(0.5 / 3.0)
    model.check_mesh_size(0.1)
    with pytest.raises(ValueError, match="resolve the internal length"):
        model.check_mesh_size(0.4)


def test_a_non_positive_length_is_refused():
    with pytest.raises(ValueError, match="ell must be positive"):
        GradientDamageModel(E, NU, LAW, ell=0.0)
