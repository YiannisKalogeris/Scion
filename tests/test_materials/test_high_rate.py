"""
Verification for high-rate plasticity: Johnson-Cook, the equations of state,
and the model that composes them.

Adapted from the MUSL elasto-plastic MPM library in ``_incoming``, whose README
is a complete algorithm specification. The code here is a batched reimplementation
in Scion's idiom rather than a port -- the source integrates one particle at
a time in a Python loop, which this library does not do.

The external benchmarks are the ones that need no simulation:

* ``rho_0 c_0^2`` **is** the small-strain bulk modulus, so the Mie-Gruneisen
  parameters can be checked against handbook values without running anything.
  Copper comes out 138.6 GPa against a published 140, aluminium 78.3 against 76.
* Johnson-Cook reduces to ``A`` at zero plastic strain, quasi-static and at the
  reference temperature. Every factor is a product of terms that are one at
  their reference state, so each can be switched on alone and checked.

The subtle one, and the reason it is tested explicitly: Johnson-Cook is a
rate-dependent *yield surface*, not a viscoplastic law. Stop straining and the
stress drops onto the quasi-static surface within one step. That looks like an
objectivity failure under a rigid rotation and is not.

python -m pytest tests/test_materials/test_high_rate.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.materials.eos import (
    EOS_REGISTRY,
    LinearEOS,
    MieGruneisen,
    get_eos,
)
from scion.materials.high_rate import HighRateState, JohnsonCookModel
from scion.materials.plasticity import JohnsonCook, get_isotropic

#: OFHC copper, Johnson & Cook (1983), with shock data from the same literature.
COPPER = dict(rho=8930.0, E=124e9, nu=0.34, A=90e6, B=292e6, n=0.31, C=0.025,
              m=1.09, T_melt=1356.0, c_0=3940.0, Gamma_0=2.02, S=1.489,
              specific_heat=383.0)
#: Aluminium 2024-T3, the parameter set from the source library's Taylor bar.
ALUMINIUM = dict(rho=2700.0, E=73.1e9, nu=0.33, A=369e6, B=684e6, n=0.73,
                 C=0.0083, m=1.0, T_melt=775.0, c_0=5386.0, Gamma_0=1.97,
                 S=1.339, specific_heat=875.0)


def build(spec, C=None, with_temperature=True):
    G = spec["E"] / (2.0 * (1.0 + spec["nu"]))
    hardening = JohnsonCook(
        A=spec["A"], B=spec["B"], n=spec["n"],
        C=spec["C"] if C is None else C, m=spec["m"], eps_dot_0=1.0,
        T_reference=293.0, T_melt=spec["T_melt"] if with_temperature else None)
    eos = MieGruneisen(spec["rho"], spec["c_0"], spec["Gamma_0"], spec["S"])
    return JohnsonCookModel(G, hardening, eos, density=spec["rho"],
                            specific_heat=spec["specific_heat"]), hardening, eos


def rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def shear_to_yield(model, rate=1e5, steps=300, dt=1e-9):
    """Load a single point in simple shear. Returns (stress, state, F)."""
    state = HighRateState.zeros(1)
    F = np.tile(np.eye(3), (1, 1, 1))
    L = np.zeros((1, 3, 3))
    L[0, 0, 1] = rate
    for _ in range(steps):
        F_new = (np.eye(3) + L * dt) @ F
        sigma, state = model.update(F_new, L, dt, state)
        F = F_new
    return sigma, state, F


# ---------------------------------------------------------------------------
# Johnson-Cook flow stress
# ---------------------------------------------------------------------------

def test_johnson_cook_reduces_to_A_at_the_reference_state():
    """Every factor is one at its reference, so the flow stress at zero plastic
    strain, the reference rate and the reference temperature is exactly ``A``."""
    jc = JohnsonCook(A=90e6, B=292e6, n=0.31, C=0.025, m=1.09,
                     T_melt=1356.0)
    assert jc.kappa(np.zeros(1))[0] == pytest.approx(90e6)
    assert jc.at(rate=np.ones(1)).kappa(np.zeros(1))[0] == pytest.approx(90e6)
    assert jc.at(rate=np.ones(1),
                 temperature=np.full(1, 293.0)).kappa(np.zeros(1))[0] \
        == pytest.approx(90e6)


def test_each_factor_can_be_checked_alone():
    """``[A + B a^n] [1 + C ln edot*] [1 - T*^m]`` -- a product, so each term
    is verifiable against a hand calculation with the others at unity."""
    jc = JohnsonCook(A=369e6, B=684e6, n=0.73, C=0.0083, m=1.0,
                     eps_dot_0=1.0, T_reference=293.0, T_melt=775.0)
    alpha = np.array([0.1])

    strain_only = jc.kappa(alpha)[0]
    assert strain_only == pytest.approx(369e6 + 684e6 * 0.1 ** 0.73)

    rate = jc.at(rate=np.array([1e4])).kappa(alpha)[0]
    assert rate / strain_only == pytest.approx(1.0 + 0.0083 * np.log(1e4))

    hot = jc.at(rate=np.array([1e4]), temperature=np.array([534.0]))
    homologous = (534.0 - 293.0) / (775.0 - 293.0)
    assert hot.kappa(alpha)[0] / rate == pytest.approx(1.0 - homologous)


def test_the_rate_factor_never_softens_below_the_reference_rate():
    """``edot* = max(edot/edot_0, 1)``. Without the clamp a slow process gets a
    negative logarithm and the material would weaken, which the model does not
    mean."""
    jc = JohnsonCook(A=90e6, B=0.0, n=1.0, C=0.025, eps_dot_0=1.0)
    for slow in (1e-6, 1e-3, 0.5, 1.0):
        assert jc.at(rate=np.array([slow])).kappa(np.zeros(1))[0] \
            == pytest.approx(90e6)
    assert jc.at(rate=np.array([10.0])).kappa(np.zeros(1))[0] > 90e6


def test_thermal_softening_reaches_zero_at_the_melting_point():
    jc = JohnsonCook(A=90e6, B=292e6, n=0.31, m=1.09, T_melt=1356.0)
    alpha = np.array([0.2])

    assert jc.at(temperature=np.array([1356.0])).kappa(alpha)[0] == \
        pytest.approx(0.0, abs=1e-6)
    # and above it, clamped rather than negative
    assert jc.at(temperature=np.array([2000.0])).kappa(alpha)[0] == \
        pytest.approx(0.0, abs=1e-6)


def test_binding_returns_a_copy_so_batches_cannot_corrupt_each_other():
    """One material object is routinely shared between particle batches at
    different rates; mutating in place would let one leak into another with no
    symptom but a slightly wrong stress."""
    jc = JohnsonCook(A=90e6, B=292e6, n=0.31, C=0.025)
    fast = jc.at(rate=np.array([1e6]))
    slow = jc.at(rate=np.array([1.0]))

    assert jc._rate is None
    assert fast.kappa(np.zeros(1))[0] > slow.kappa(np.zeros(1))[0]


def test_the_hardening_modulus_is_floored_like_ludwik():
    """``H = B n a^(n-1)`` diverges at ``a = 0`` for ``n < 1``, which is the
    state every point is in at first yield."""
    jc = JohnsonCook(A=90e6, B=292e6, n=0.31)
    assert np.isfinite(jc.dkappa_dalpha(np.zeros(1))[0])
    assert jc.dkappa_dalpha(np.zeros(1))[0] > jc.dkappa_dalpha(np.ones(1))[0]


def test_a_temperature_without_a_melting_point_is_refused():
    jc = JohnsonCook(A=90e6, B=1.0, n=1.0)
    with pytest.raises(ValueError, match="T_melt"):
        jc.at(temperature=np.array([500.0]))


def test_a_melting_point_below_the_reference_is_refused():
    with pytest.raises(ValueError, match="T_melt"):
        JohnsonCook(A=1.0, B=1.0, n=1.0, T_reference=300.0, T_melt=250.0)


def test_johnson_cook_is_in_the_hardening_registry():
    assert get_isotropic("jc", A=1.0, B=1.0, n=0.5).name == "johnson-cook"
    assert get_isotropic("johnson-cook", A=1.0, B=1.0, n=0.5).rate_dependent


# ---------------------------------------------------------------------------
# Equations of state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec,handbook,tolerance", [
    (COPPER, 140e9, 0.03), (ALUMINIUM, 76e9, 0.05)])
def test_rho_c0_squared_is_the_handbook_bulk_modulus(spec, handbook,
                                                     tolerance):
    """The external benchmark that needs no simulation. ``rho_0 c_0^2`` is the
    small-strain limit of the Hugoniot, so it must agree with the measured bulk
    modulus -- and it does, to 1% for copper and 3% for aluminium.

    Iron is the instructive exception and is deliberately not in this list: its
    linear Hugoniot fit is distorted by the alpha-to-epsilon phase transition at
    13 GPa, so ``rho_0 c_0^2`` gives 100 GPa against a true 170.
    """
    eos = MieGruneisen(spec["rho"], spec["c_0"], spec["Gamma_0"], spec["S"])
    assert eos.bulk_modulus == pytest.approx(handbook, rel=tolerance)


def test_mie_gruneisen_reduces_to_linear_elasticity_at_small_compression():
    """It must, or the model disagrees with elasticity where both apply."""
    eos = MieGruneisen(COPPER["rho"], COPPER["c_0"], COPPER["Gamma_0"],
                       COPPER["S"])
    linear = LinearEOS(eos.bulk_modulus)

    for eta, tolerance in ((1.0001, 1e-3), (1.001, 2e-3), (1.01, 2e-2)):
        assert eos.pressure(eta) == pytest.approx(linear.pressure(eta),
                                                  rel=tolerance)


def test_mie_gruneisen_stiffens_beyond_linear_under_real_compression():
    """The reason it exists: at 20% compression the real material is much
    stiffer than ``K (eta - 1)`` predicts."""
    eos = MieGruneisen(COPPER["rho"], COPPER["c_0"], COPPER["Gamma_0"],
                       COPPER["S"])
    linear = LinearEOS(eos.bulk_modulus)
    assert eos.pressure(1.2) > 1.2 * linear.pressure(1.2)


def test_internal_energy_contributes_exactly_gamma_times_e():
    eos = MieGruneisen(COPPER["rho"], COPPER["c_0"], COPPER["Gamma_0"],
                       COPPER["S"])
    cold = eos.pressure(1.1, 0.0)
    hot = eos.pressure(1.1, 1e9)
    assert hot - cold == pytest.approx(COPPER["Gamma_0"] * 1e9)


def test_zero_compression_gives_only_the_thermal_pressure():
    eos = MieGruneisen(COPPER["rho"], COPPER["c_0"], COPPER["Gamma_0"],
                       COPPER["S"])
    assert eos.pressure(1.0, 0.0) == pytest.approx(0.0, abs=1e-6)
    assert eos.pressure(1.0, 5e8) == pytest.approx(COPPER["Gamma_0"] * 5e8)


def test_the_compression_limit_is_reported_and_enforced():
    """``eta = S/(S-1)`` is where the linear shock relation predicts infinite
    compression. Past it the formula returns nonsense rather than a large
    number, so it raises instead."""
    eos = MieGruneisen(8930.0, 3940.0, 2.02, 1.5)
    assert eos.compression_limit == pytest.approx(3.0)

    assert np.isfinite(eos.pressure(2.9))
    with pytest.raises(ValueError, match="limit"):
        eos.pressure(3.1)


def test_a_shock_slope_below_one_has_no_compression_limit():
    assert MieGruneisen(1000.0, 1000.0, 2.0, 0.8).compression_limit == \
        float("inf")


def test_pressure_from_the_jacobian_matches_the_compression_ratio():
    """``eta = 1/J`` by mass conservation. The Jacobian form is what a
    finite-strain update actually has to hand."""
    eos = MieGruneisen(COPPER["rho"], COPPER["c_0"], COPPER["Gamma_0"],
                       COPPER["S"])
    J = np.array([0.9, 0.95, 1.0, 1.05])
    np.testing.assert_allclose(eos.pressure_from_jacobian(J),
                               eos.pressure(1.0 / J))


def test_an_inverted_element_is_refused():
    eos = LinearEOS(1e9)
    with pytest.raises(ValueError, match="inverted"):
        eos.pressure_from_jacobian(np.array([-0.5]))


def test_the_eos_registry_and_its_refusals():
    assert get_eos("mg", rho_0=1.0, c_0=1.0, Gamma_0=2.0).name == \
        "mie-gruneisen"
    assert set(EOS_REGISTRY) >= {"linear", "mie-gruneisen", "mg"}
    with pytest.raises(NotImplementedError, match="Tillotson"):
        get_eos("tillotson")


# ---------------------------------------------------------------------------
# The composed model
# ---------------------------------------------------------------------------

def test_the_radial_return_lands_on_the_flow_surface():
    """The defining property of a return map. After loading well past yield the
    von Mises stress must equal the Johnson-Cook flow stress at the current
    plastic strain, rate and temperature."""
    model, hardening, _ = build(COPPER)
    sigma, state, _ = shear_to_yield(model)

    rate = np.array([1e5 / np.sqrt(3.0)])          # equivalent rate in shear
    flow = hardening.at(rate=rate,
                        temperature=state.temperature).kappa(state.alpha)
    assert model.equivalent_stress(state)[0] == pytest.approx(flow[0],
                                                              rel=0.02)
    assert state.alpha[0] > 0.0


def test_a_rigid_rotation_preserves_the_stress_exactly():
    """Objectivity, with the rate term off so the yield surface holds still.
    The Green-Naghdi update rotates the stress and changes nothing else --
    measured at 4e-16 relative, which is round-off."""
    model, _, _ = build(COPPER, C=0.0)
    sigma0, state, F0 = shear_to_yield(model)
    reference = model.equivalent_stress(state)[0]
    alpha0 = state.alpha[0]

    omega, steps = 1e5, 200
    dt = (np.pi / 2) / omega / steps
    W = np.zeros((1, 3, 3))
    W[0, 0, 1], W[0, 1, 0] = -omega, omega

    for k in range(1, steps + 1):
        F = np.array([rotation(omega * k * dt) @ F0[0]])
        sigma, state = model.update(F, W, dt, state)

    assert model.equivalent_stress(state)[0] == pytest.approx(reference,
                                                              rel=1e-10)
    assert state.alpha[0] == pytest.approx(alpha0, abs=1e-15)
    turned = rotation(np.pi / 2)
    np.testing.assert_allclose(sigma[0], turned @ sigma0[0] @ turned.T,
                               atol=1e-6 * abs(sigma0).max())


def test_a_rate_dependent_surface_relaxes_when_the_straining_stops():
    """Not an objectivity failure -- the model. Johnson-Cook's surface moves
    with the *current* rate and has no relaxation time, so holding the material
    still drops it onto the quasi-static surface within one step. Measured for
    copper: 217.2 MPa at 1e5 /s falls to 170.6 MPa, which is exactly the
    quasi-static flow stress at that plastic strain."""
    model, hardening, _ = build(COPPER, C=0.025)
    _, state, F0 = shear_to_yield(model)
    loaded = model.equivalent_stress(state)[0]

    still = np.zeros((1, 3, 3))
    _, state = model.update(F0, still, 1e-9, state)
    relaxed = model.equivalent_stress(state)[0]

    quasi_static = hardening.at(
        rate=np.zeros(1), temperature=state.temperature).kappa(state.alpha)[0]
    assert relaxed < loaded
    # the relaxation step itself advances alpha slightly, so the surface it
    # lands on is evaluated a hair further along the hardening curve
    assert relaxed == pytest.approx(quasi_static, rel=1e-2)
    assert loaded / relaxed == pytest.approx(
        1.0 + 0.025 * np.log(1e5 / np.sqrt(3.0)), rel=0.05)


def test_plastic_work_raises_the_temperature():
    """Adiabatic heating: a fraction ``chi`` of the plastic work becomes heat.
    For copper at these strains the rise is small but must be positive and
    monotone."""
    model, _, _ = build(COPPER)
    _, state, _ = shear_to_yield(model, steps=600)

    assert state.temperature[0] > 293.0
    assert state.energy[0] > 0.0


def test_thermal_softening_lowers_the_stress_and_the_heating():
    """The feedback loop, and it runs the way that surprises people.

    ``with_temperature=False`` removes the softening *term*, not the temperature
    tracking, so both runs heat up. The one that softens ends at a lower stress
    -- and therefore *cooler*, 335.00 K against 335.59 K, because a weaker
    material does less plastic work for the same imposed strain. The heating and
    the softening oppose each other, which is why the coupling has to be solved
    together rather than applied as a correction.
    """
    unsoftened, _, _ = build(COPPER, with_temperature=False)
    softening, _, _ = build(COPPER, with_temperature=True)

    _, stiff, _ = shear_to_yield(unsoftened, rate=1e6, steps=800)
    _, soft, _ = shear_to_yield(softening, rate=1e6, steps=800)

    assert softening.equivalent_stress(soft)[0] < \
        unsoftened.equivalent_stress(stiff)[0]
    assert soft.temperature[0] < stiff.temperature[0]
    assert soft.temperature[0] > 293.0


def test_pure_compression_gives_the_eos_pressure_and_no_deviator():
    """The split: volumetric response entirely from the EOS, nothing from the
    plasticity. A hydrostatic state must produce zero deviatoric stress."""
    model, _, eos = build(COPPER)
    state = HighRateState.zeros(1)
    F = np.array([np.diag([0.95, 0.95, 0.95])])

    sigma, new_state = model.update(F, np.zeros((1, 3, 3)), 1e-9, state)

    J = float(np.linalg.det(F[0]))
    mean_stress = float(np.trace(sigma[0]) / 3.0)
    assert mean_stress == pytest.approx(-eos.pressure(1.0 / J, 0.0))
    assert np.abs(new_state.deviator).max() < 1e-6
    assert new_state.alpha[0] == 0.0


def test_elastic_loading_does_not_accumulate_plastic_strain():
    model, _, _ = build(COPPER)
    _, state, _ = shear_to_yield(model, rate=1.0, steps=5, dt=1e-12)
    assert state.alpha[0] == 0.0


def test_the_state_handed_in_is_never_modified():
    """Trial-vs-committed, the same discipline as QuadratureState. A discarded
    step must not advance the plastic strain."""
    model, _, _ = build(COPPER)
    state = HighRateState.zeros(1)
    before = state.copy()
    L = np.zeros((1, 3, 3))
    L[0, 0, 1] = 1e6

    model.update(np.array([np.eye(3) + L[0] * 1e-9]), L, 1e-9, state)
    np.testing.assert_array_equal(state.alpha, before.alpha)
    np.testing.assert_array_equal(state.deviator, before.deviator)


def test_the_wave_speed_carries_the_shear_stiffness():
    """``sqrt((K + 4G/3)/rho)``, not ``sqrt(K/rho)``. The difference is about
    15% for a metal, which is the whole CFL margin."""
    model, _, eos = build(COPPER)
    bulk_only = np.sqrt(eos.bulk_modulus / COPPER["rho"])

    assert model.wave_speed > bulk_only
    assert model.wave_speed == pytest.approx(4736.0, rel=0.01)


def test_an_inverted_deformation_gradient_is_refused():
    model, _, _ = build(COPPER)
    with pytest.raises(ValueError, match="inverted"):
        model.update(np.array([np.diag([-1.0, 1.0, 1.0])]),
                     np.zeros((1, 3, 3)), 1e-9, HighRateState.zeros(1))


def test_a_two_dimensional_tensor_is_refused():
    """Constitutive code here is always full 3-D, because sigma_33 is nonzero
    in plane strain and enters J2."""
    model, _, _ = build(COPPER)
    with pytest.raises(ValueError, match=r"\(n, 3, 3\)"):
        model.update(np.array([np.eye(2)]), np.zeros((1, 2, 2)), 1e-9,
                     HighRateState.zeros(1))


def test_a_rate_independent_hardening_law_is_refused():
    """This model exists for the rate term; without it PlasticityModel is the
    right path and also gives a consistent tangent."""
    from scion.materials.plasticity import LinearIsotropic
    with pytest.raises(TypeError, match="Johnson-Cook"):
        JohnsonCookModel(1e9, LinearIsotropic(1e6, 1e6), bulk_modulus=1e9)


def test_the_model_runs_on_a_batch_of_particles():
    """Everything is vectorised over the particle axis -- the source library
    integrates one particle at a time, which is what had to change."""
    model, _, _ = build(ALUMINIUM)
    n = 500
    state = HighRateState.zeros(n)
    rng = np.random.default_rng(0)
    L = rng.standard_normal((n, 3, 3)) * 1e4
    F = np.tile(np.eye(3), (n, 1, 1)) + L * 1e-9

    sigma, state = model.update(F, L, 1e-9, state)
    assert sigma.shape == (n, 3, 3)
    assert np.all(np.isfinite(sigma))
    assert state.alpha.shape == (n,)
