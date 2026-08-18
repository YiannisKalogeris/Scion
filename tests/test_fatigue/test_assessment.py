"""
Verification for the multiaxial reduction, the accumulation rules and the
driver that runs them.

The anchor is constant amplitude: a history of ``n`` identical cycles must give
a damage of exactly ``n/N`` with ``N`` from the curve evaluated by hand. Every
piece of wiring -- reduction, counting, the range-to-amplitude halving, the
mean-stress path -- sits between those two numbers, so anything misplaced shows
up as a clean factor.

python -m pytest tests/test_fatigue/test_assessment.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.fatigue.accumulation import (
    LinearAccumulation,
    NonlinearAccumulation,
    cycle_damage,
    get_accumulation_rule,
)
from scion.fatigue.assessment import FatigueAssessment
from scion.fatigue.curves import Basquin, CoffinManson, StrainLife
from scion.fatigue.multiaxial import (
    equivalent_strain,
    get_equivalent,
    max_principal,
    max_shear,
    non_proportionality,
    signed_von_mises,
    voigt_strain_to_tensor,
    voigt_to_tensor,
    von_mises,
)
from scion.fatigue.rainflow import rainflow

E, NU = 210.0e3, 0.3
CURVE = Basquin(sigma_f=1200.0, b=-0.095)


def _uniaxial_history(amplitude, cycles=10, steps_per_cycle=64, mean=0.0):
    """A sinusoidal uniaxial 3-D Voigt stress history, ``(1, n_steps, 6)``."""
    t = np.linspace(0.0, cycles * 2.0 * np.pi, cycles * steps_per_cycle + 1)
    h = np.zeros((1, t.size, 6))
    h[0, :, 0] = mean + amplitude * np.sin(t)
    return h


# ---------------------------------------------------------------------------
# The anchor
# ---------------------------------------------------------------------------

def test_constant_amplitude_reproduces_the_curve_exactly():
    """Ten cycles of amplitude 200 must give ``D = 10/N(200)``. Everything the
    driver does lies between those two numbers."""
    job = FatigueAssessment(CURVE)
    r = job.evaluate(stress=_uniaxial_history(200.0, cycles=10))

    N = CURVE.cycles(np.array([200.0]))[0]
    assert r.cycles_per_block[0] == pytest.approx(10.0, abs=0.5)
    assert r.damage[0] == pytest.approx(10.0 / N, rel=1e-2)
    assert r.blocks_to_failure[0] == pytest.approx(N / 10.0, rel=1e-2)


def test_a_scalar_history_needs_no_reduction_and_gives_the_same_answer():
    """The driver accepts an already-reduced history; for uniaxial stress the
    signed von Mises of the tensor is the same signal, so the two routes must
    agree to the last digit."""
    h = _uniaxial_history(200.0)
    job = FatigueAssessment(CURVE)
    tensor = job.evaluate(stress=h)
    scalar = job.evaluate(stress=h[:, :, 0])
    assert tensor.damage[0] == pytest.approx(scalar.damage[0], rel=1e-12)


def test_damage_scales_linearly_with_the_number_of_cycles():
    job = FatigueAssessment(CURVE)
    one = job.evaluate(stress=_uniaxial_history(200.0, cycles=10))
    ten = job.evaluate(stress=_uniaxial_history(200.0, cycles=10), repeats=10.0)
    assert ten.damage[0] == pytest.approx(10.0 * one.damage[0], rel=1e-12)


def test_a_tensile_mean_stress_shortens_the_life():
    plain = Basquin(sigma_f=1200.0, b=-0.095)
    corrected = Basquin(sigma_f=1200.0, b=-0.095, mean_stress="goodman",
                        mean_stress_kwargs={"sigma_u": 600.0})
    h = _uniaxial_history(200.0, mean=150.0)
    d0 = FatigueAssessment(plain).evaluate(stress=h).damage[0]
    d1 = FatigueAssessment(corrected).evaluate(stress=h).damage[0]
    assert d1 > d0


# ---------------------------------------------------------------------------
# The rectification trap
# ---------------------------------------------------------------------------

def test_unsigned_von_mises_halves_the_range_and_doubles_the_count():
    """The reason signed von Mises is the default. Rectifying a fully reversed
    signal turns one cycle of range 2a into two of range a, and on a slope-3
    curve that is roughly a factor of four too much life."""
    h = _uniaxial_history(200.0, cycles=10)
    t = voigt_to_tensor(h[0], dim=3)

    signed = rainflow(signed_von_mises(t), residue="repeat")
    unsigned = rainflow(von_mises(t), residue="repeat")

    assert signed.ranges.max() == pytest.approx(400.0, rel=1e-3)
    assert unsigned.ranges.max() == pytest.approx(200.0, rel=1e-3)
    assert unsigned.total == pytest.approx(2.0 * signed.total, abs=1.0)

    job = FatigueAssessment(CURVE)
    d_signed = job.evaluate(stress=h).damage[0]
    d_unsigned = FatigueAssessment(CURVE, equivalent="von-mises") \
        .evaluate(stress=h).damage[0]
    assert d_signed > 100.0 * d_unsigned


def test_the_hydrostatic_sign_falls_back_where_there_is_no_pressure():
    """Isochoric plastic flow and pure shear have zero trace. Reporting a sign
    of zero there would collapse the cycle to nothing, so those points take the
    principal rule instead."""
    shear = np.zeros((2, 3, 3))
    shear[0, 0, 1] = shear[0, 1, 0] = 100.0
    shear[1, 0, 1] = shear[1, 1, 0] = -100.0
    q = signed_von_mises(shear)
    assert np.all(np.abs(q) == pytest.approx(np.sqrt(3.0) * 100.0))
    assert np.all(q != 0.0)


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------

def test_the_reducers_on_a_known_state():
    t = np.diag([200.0, 50.0, -100.0])[None]
    assert max_principal(t)[0] == pytest.approx(200.0)
    assert max_shear(t)[0] == pytest.approx(150.0)
    assert von_mises(t)[0] == pytest.approx(
        np.sqrt(0.5 * (150.0 ** 2 + 150.0 ** 2 + 300.0 ** 2)))
    assert signed_von_mises(t)[0] > 0.0            # tensile trace


def test_a_two_dimensional_voigt_stress_is_refused_without_the_out_of_plane():
    """sigma_33 is neither in the vector nor zero in plane strain, and filling
    it with zero would silently be plane stress."""
    v = np.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="sigma_33"):
        voigt_to_tensor(v, dim=2)
    with pytest.raises(ValueError, match="sigma_33"):
        FatigueAssessment(CURVE).evaluate(stress=v, dim=2)


def test_the_plane_strain_out_of_plane_stress_changes_the_equivalent_stress():
    """Not a rounding difference: at nu = 0.3 an in-plane uniaxial state comes
    out 12% high without it, and an equibiaxial one 2.5 times high."""
    uni = np.array([[200.0, 0.0, 0.0]])
    bi = np.array([[200.0, 200.0, 0.0]])

    q_uni = von_mises(voigt_to_tensor(uni, dim=2, nu=NU))[0]
    q_bi = von_mises(voigt_to_tensor(bi, dim=2, nu=NU))[0]

    assert q_uni == pytest.approx(200.0 * np.sqrt(1.0 - NU + NU ** 2))
    assert 200.0 / q_uni == pytest.approx(1.125, rel=1e-2)
    assert q_bi == pytest.approx(200.0 * (1.0 - 2.0 * NU))
    assert 200.0 / q_bi == pytest.approx(2.5, rel=1e-2)


def test_an_explicit_out_of_plane_stress_overrides_the_elastic_relation():
    """What an elastoplastic run supplies, where the elastic relation is
    wrong."""
    v = np.array([[200.0, 0.0, 0.0]])
    t = voigt_to_tensor(v, dim=2, sigma_33=np.array([100.0]))
    assert t[0, 2, 2] == pytest.approx(100.0)


def test_the_strain_conversion_halves_the_engineering_shear():
    """Reusing the stress conversion on a strain vector doubles every shear.
    Both the 2-D and 3-D paths are checked, because they are separate code."""
    e2 = np.array([[0.0, 0.0, 0.004]])              # gamma_12
    assert voigt_strain_to_tensor(e2, dim=2)[0, 0, 1] == pytest.approx(0.002)

    e3 = np.zeros((1, 6))
    e3[0, 3] = e3[0, 4] = e3[0, 5] = 0.004
    t = voigt_strain_to_tensor(e3, dim=3)
    assert t[0, 1, 2] == pytest.approx(0.002)
    assert t[0, 0, 2] == pytest.approx(0.002)
    assert t[0, 0, 1] == pytest.approx(0.002)


def test_plane_strain_needs_no_out_of_plane_strain_because_it_is_zero():
    """The asymmetry against the stress conversion: the kinematic constraint is
    prescribed, the static response to it is not."""
    e = np.array([[1e-3, -5e-4, 0.0]])
    assert voigt_strain_to_tensor(e, dim=2)[0, 2, 2] == 0.0


def test_the_equivalent_strain_is_normalised_to_the_uniaxial_value():
    """Returns the applied strain exactly, for whatever nu describes the state
    -- and under-reads by (1+nu)/1.5 if told the wrong one."""
    e = np.diag([1e-3, -NU * 1e-3, -NU * 1e-3])[None]
    assert equivalent_strain(e, nu=NU)[0] == pytest.approx(1e-3)
    assert equivalent_strain(e, nu=0.5)[0] == pytest.approx(
        1e-3 * (1.0 + NU) / 1.5)


def _axial_torsion(phase, tau):
    t = np.linspace(0.0, 2.0 * np.pi, 256)
    a = np.zeros((1, t.size, 3, 3))
    a[0, :, 0, 0] = 200.0 * np.cos(t)
    a[0, :, 0, 1] = a[0, :, 1, 0] = tau * np.cos(t - phase)
    return a


def test_non_proportionality_is_zero_under_proportional_loading():
    """Including through a full reversal, which is the case a naive metric
    built on the largest *signed* principal gets wrong: in compression that
    direction jumps to one of the degenerate zero eigenvectors and reports a
    90 degree rotation for the most proportional history there is."""
    t = np.linspace(0.0, 2.0 * np.pi, 128)
    prop = np.zeros((1, t.size, 3, 3))
    prop[0, :, 0, 0] = 200.0 * np.sin(t)
    prop[0, :, 1, 1] = 100.0 * np.sin(t)
    assert non_proportionality(prop)[0] == pytest.approx(0.0, abs=1e-9)

    tau = 200.0 / np.sqrt(3.0)
    assert non_proportionality(_axial_torsion(0.0, tau))[0] == \
        pytest.approx(0.0, abs=1e-9)


def test_the_canonical_out_of_phase_case_sets_the_scale():
    """Axial and torsional loading 90 degrees out of phase at equal von Mises
    amplitude is the standard non-proportional benchmark, and it scores 0.11 --
    not something near one, because the dominant principal direction of an
    axial-torsion state sweeps 45 degrees rather than 90. The calibration
    matters: without it 0.11 reads as mild when it is severe.
    """
    tau = 200.0 / np.sqrt(3.0)
    assert non_proportionality(_axial_torsion(0.5 * np.pi, tau))[0] == \
        pytest.approx(0.11, abs=0.01)


def test_more_rotation_scores_higher():
    tau = 200.0 / np.sqrt(3.0)
    scores = [non_proportionality(_axial_torsion(p, tau))[0]
              for p in (0.0, 0.25 * np.pi, 0.5 * np.pi)]
    assert scores[0] < scores[1] < scores[2]


def test_the_equivalent_registry_resolves_binds_and_rejects():
    assert get_equivalent("tresca") is max_shear
    bound = get_equivalent("signed-von-mises", sign="principal")
    t = np.diag([-200.0, 0.0, 0.0])[None]
    assert bound(t)[0] < 0.0
    with pytest.raises(NotImplementedError, match="[Cc]ritical-plane"):
        get_equivalent("fatemi-socie")


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

def test_runout_and_immediate_failure_propagate_rather_than_clipping():
    d = cycle_damage(np.array([1.0, 1.0, 0.0]), np.array([np.inf, 0.0, 0.0]))
    assert d[0] == 0.0
    assert np.isinf(d[1])
    assert d[2] == 0.0


def test_one_cycle_off_the_end_of_the_curve_cannot_be_averaged_away():
    """An infinite contribution must survive the sum, or a static overload
    disappears into a large spectrum of harmless cycles."""
    rule = LinearAccumulation()
    d = cycle_damage(np.ones(1000), np.concatenate([[0.0], np.full(999, 1e12)]))
    assert np.isinf(rule.accumulate(d))
    assert rule.blocks_to_failure(rule.accumulate(d)) == 0.0


def test_miner_is_a_plain_sum_and_the_critical_value_is_exposed():
    rule = LinearAccumulation(critical=0.5)
    d = np.array([0.1, 0.2, 0.05])
    assert rule.accumulate(d) == pytest.approx(0.35)
    assert rule.blocks_to_failure(0.35) == pytest.approx(0.5 / 0.35)
    assert not rule.failed(0.35)
    assert rule.failed(0.6)


def test_miner_is_indifferent_to_order():
    """Its defining limitation, asserted so that nobody mistakes the module for
    handling sequence effects."""
    rule = LinearAccumulation()
    d = np.array([0.1, 0.4, 0.05, 0.2])
    assert rule.accumulate(d) == pytest.approx(rule.accumulate(d[::-1]))


def test_marco_starkey_reduces_to_miner_at_unit_exponent():
    d = np.array([0.1, 0.2, 0.3])
    assert NonlinearAccumulation(1.0).accumulate(d) == \
        pytest.approx(LinearAccumulation().accumulate(d))


def test_marco_starkey_is_also_order_independent():
    """Changing the exponent reweights amplitudes; it does not recover
    sequence, because rainflow already discarded the order."""
    rule = NonlinearAccumulation(1.6)
    d = np.array([0.1, 0.4, 0.05, 0.2])
    assert rule.accumulate(d) == pytest.approx(rule.accumulate(d[::-1]))


def test_the_accumulation_registry_resolves_and_rejects():
    assert isinstance(get_accumulation_rule("miner"), LinearAccumulation)
    with pytest.raises(NotImplementedError):
        get_accumulation_rule("corten-dolan")


# ---------------------------------------------------------------------------
# Driver wiring
# ---------------------------------------------------------------------------

def test_the_curve_decides_which_channel_is_counted():
    sl = StrainLife(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60)
    with pytest.raises(ValueError, match="strain"):
        FatigueAssessment(sl).evaluate(stress=_uniaxial_history(200.0))
    with pytest.raises(ValueError, match="plastic"):
        FatigueAssessment(CoffinManson(0.45, -0.6)).evaluate(
            strain=np.zeros((1, 8)))


def test_a_curve_that_needs_the_stress_history_says_so():
    swt = StrainLife(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60,
                     mean_stress="swt")
    with pytest.raises(ValueError, match="stress"):
        FatigueAssessment(swt).evaluate(strain=np.zeros((1, 8)))


def test_strain_life_reads_the_mean_stress_at_the_loops_own_turning_points():
    """The reason the counts carry indices. Here the strain history is fully
    reversed while the stress history is offset, so a correct pairing sees the
    offset and a positional one would not.
    """
    t = np.linspace(0.0, 10.0 * 2 * np.pi, 641)
    strain = (0.003 * np.sin(t))[None, :]
    reversed_stress = (E * 0.003 * np.sin(t))[None, :]
    offset_stress = reversed_stress + 200.0

    curve = StrainLife(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60,
                       mean_stress="swt")
    job = FatigueAssessment(curve)
    d0 = job.evaluate(strain=strain, stress=reversed_stress).damage[0]
    d1 = job.evaluate(strain=strain, stress=offset_stress).damage[0]
    assert d1 > d0


def test_mismatched_channel_shapes_are_refused():
    curve = StrainLife(E=E, sigma_f=1000.0, b=-0.095, eps_f=0.45, c=-0.60,
                       mean_stress="morrow")
    with pytest.raises(ValueError, match="same points and steps"):
        FatigueAssessment(curve).evaluate(strain=np.zeros((2, 8)),
                                          stress=np.zeros((2, 9)))


def test_points_restricts_the_assessment_and_scatter_puts_it_back():
    """The result arrays are indexed by assessment order, so writing them
    straight to a mesh field would mis-place every value."""
    h = np.concatenate([_uniaxial_history(a) for a in (50.0, 300.0, 150.0)])
    job = FatigueAssessment(CURVE)

    full = job.evaluate(stress=h)
    part = job.evaluate(stress=h, points=[1])

    assert part.damage.size == 1
    assert part.n_points == 3
    assert part.damage[0] == pytest.approx(full.damage[1], rel=1e-12)
    assert part.critical == 1

    field = part.scatter()
    assert field.shape == (3,)
    assert field[1] == pytest.approx(part.damage[0])
    assert field[0] == 0.0 and field[2] == 0.0


def test_hotspots_ranks_by_the_span_of_the_reduced_history():
    h = np.concatenate([_uniaxial_history(a) for a in (50.0, 300.0, 150.0)])
    idx = FatigueAssessment(CURVE).hotspots(h, n=2)
    assert list(idx) == [1, 2]


def test_the_critical_point_is_the_most_damaged_one():
    h = np.concatenate([_uniaxial_history(a) for a in (50.0, 300.0, 150.0)])
    r = FatigueAssessment(CURVE).evaluate(stress=h)
    assert r.critical == 1
    assert r.max_damage == pytest.approx(r.damage[1])
    assert np.all(np.diff(r.blocks_to_failure[[1, 2, 0]]) > 0.0)


def test_counts_are_retained_only_on_request():
    h = _uniaxial_history(200.0)
    job = FatigueAssessment(CURVE)
    assert job.evaluate(stress=h).counts is None
    kept = job.evaluate(stress=h, keep_counts=True).counts
    assert kept is not None and len(kept) == 1


def test_an_untouched_point_accumulates_no_damage():
    h = np.zeros((1, 64, 6))
    r = FatigueAssessment(CURVE).evaluate(stress=h)
    assert r.damage[0] == 0.0
    assert np.isinf(r.blocks_to_failure[0])
