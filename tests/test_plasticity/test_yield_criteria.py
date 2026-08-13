"""
Verification for the yield criteria.

Each criterion is checked independently of any return map or assembler: the
derivatives against finite differences, and the surface itself against
properties that follow from its definition. That is the point of keeping the
criteria a separate axis -- they are provably correct before a solver exists.

python -m pytest tests/test_plasticity/test_yield_criteria.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.plasticity.yield_criteria import (
    DruckerPrager,
    J2_invariant,
    P_DEV,
    VonMises,
    deviator,
    get_criterion,
    pressure,
    von_mises_stress,
)

CRITERIA = ["von-mises", "drucker-prager"]


def _stresses(n=200, scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    A = scale * rng.standard_normal((n, 3, 3))
    return 0.5 * (A + np.swapaxes(A, 1, 2))


def _uniaxial(value, axis=0):
    s = np.zeros((1, 3, 3))
    s[0, axis, axis] = value
    return s


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_deviator_is_traceless_and_splits_the_stress():
    sig = _stresses()
    s = deviator(sig)
    assert np.abs(np.einsum("nii->n", s)).max() < 1e-10
    rebuilt = s + pressure(sig)[:, None, None] * np.eye(3)
    assert np.abs(rebuilt - sig).max() < 1e-10


def test_von_mises_stress_of_uniaxial_state_is_the_axial_stress():
    assert von_mises_stress(_uniaxial(137.0))[0] == pytest.approx(137.0)
    assert von_mises_stress(_uniaxial(-137.0))[0] == pytest.approx(137.0)


def test_deviatoric_projector_annihilates_pressure():
    sig = _stresses()
    s = np.einsum("ijkl,nkl->nij", P_DEV, sig)
    assert np.abs(s - deviator(sig)).max() < 1e-10


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", CRITERIA)
def test_normal_is_the_gradient_of_f(name):
    crit = get_criterion(name)
    sig = _stresses()
    kappa = np.full(len(sig), 50.0)
    n = crit.df_dsigma(sig)

    h = 1e-4
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0 if i == j else 0.5
            fd = (crit.f(sig + h * E, kappa) - crit.f(sig - h * E, kappa)) / (2 * h)
            assert np.abs(fd - n[:, i, j]).max() < 1e-6


@pytest.mark.parametrize("name", CRITERIA)
def test_hessian_is_the_gradient_of_the_normal(name):
    """This is what the closest-point return map needs for quadratic
    convergence; an error here shows up only as a slow inner loop."""
    crit = get_criterion(name)
    sig = _stresses()
    H = crit.d2f_dsigma2(sig)

    h = 1e-4
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0 if i == j else 0.5
            fd = (crit.df_dsigma(sig + h * E)
                  - crit.df_dsigma(sig - h * E)) / (2 * h)
            assert np.abs(fd - H[:, :, :, i, j]).max() < 1e-6


@pytest.mark.parametrize("name", CRITERIA)
def test_hessian_has_both_minor_symmetries(name):
    crit = get_criterion(name)
    H = crit.d2f_dsigma2(_stresses(n=20))
    assert np.abs(H - np.swapaxes(H, 1, 2)).max() < 1e-10
    assert np.abs(H - np.swapaxes(H, 3, 4)).max() < 1e-10


@pytest.mark.parametrize("name", CRITERIA)
def test_df_dkappa_is_negative(name):
    """Hardening must shrink the yield function, or the return map moves the
    wrong way."""
    crit = get_criterion(name)
    sig = _stresses(n=20)
    kappa = np.full(len(sig), 50.0)
    assert (crit.df_dkappa(sig, kappa) < 0).all()

    h = 1e-3
    fd = (crit.f(sig, kappa + h) - crit.f(sig, kappa - h)) / (2 * h)
    assert np.abs(fd - crit.df_dkappa(sig, kappa)).max() < 1e-8


# ---------------------------------------------------------------------------
# Von Mises: the defining physical properties
# ---------------------------------------------------------------------------

def test_von_mises_is_pressure_insensitive():
    """Adding hydrostatic stress must not change f. This is the property that
    makes J2 right for metals and wrong for soils."""
    crit = VonMises()
    sig = _stresses()
    kappa = np.full(len(sig), 50.0)
    f0 = crit.f(sig, kappa)
    for dp in (-500.0, 500.0, 5000.0):
        f1 = crit.f(sig + dp * np.eye(3), kappa)
        assert np.abs(f1 - f0).max() < 1e-8 * max(np.abs(f0).max(), 1.0)


def test_von_mises_flow_is_volume_preserving():
    """tr(df/dsigma) = 0, so associative flow produces no plastic dilatation."""
    crit = VonMises()
    n = crit.df_dsigma(_stresses())
    assert np.abs(np.einsum("nii->n", n)).max() < 1e-10


def test_von_mises_yields_at_the_uniaxial_yield_stress():
    crit = VonMises()
    y = np.array([250.0])
    assert crit.f(_uniaxial(249.0), y)[0] < 0
    assert crit.f(_uniaxial(251.0), y)[0] > 0
    assert crit.f(_uniaxial(-251.0), y)[0] > 0        # symmetric in tension


def test_von_mises_shear_yield_is_sigma_y_over_sqrt3():
    crit = VonMises()
    tau = 250.0 / np.sqrt(3.0)
    s = np.zeros((1, 3, 3))
    s[0, 0, 1] = s[0, 1, 0] = tau
    assert abs(crit.f(s, np.array([250.0]))[0]) < 1e-9


def test_von_mises_reports_no_singular_points():
    """s = 0 is inside the elastic domain, so the normal is never needed
    there -- unlike the Drucker-Prager apex."""
    crit = VonMises()
    assert not crit.singular_mask(np.zeros((3, 3, 3))).any()


# ---------------------------------------------------------------------------
# Drucker-Prager
# ---------------------------------------------------------------------------

def test_drucker_prager_is_pressure_sensitive():
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    kappa = np.array([50.0])
    sig = _uniaxial(0.0)
    f_neutral = crit.f(sig, kappa)[0]
    f_tension = crit.f(sig + 100.0 * np.eye(3), kappa)[0]
    f_compress = crit.f(sig - 100.0 * np.eye(3), kappa)[0]
    assert f_tension > f_neutral > f_compress


def test_drucker_prager_flow_is_dilatant():
    """tr(df/dsigma) = eta > 0: associative flow on a cone predicts plastic
    volume increase, which is why non-associative flow exists."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    n = crit.df_dsigma(_stresses())
    assert np.abs(np.einsum("nii->n", n) - crit.eta).max() < 1e-10


def test_drucker_prager_is_stronger_in_compression():
    """The asymmetry that distinguishes a frictional material from a metal."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    kappa = np.array([50.0])

    def strength(sign):
        lo, hi = 0.0, 1e5
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if crit.f(_uniaxial(sign * mid), kappa)[0] < 0:
                lo = mid
            else:
                hi = mid
        return lo

    ft, fc = strength(+1), strength(-1)
    assert fc > ft
    # for the outer cone at phi = 30 deg the ratio is (3+eta)/(3-eta)... just
    # check it is the documented 2.33 to a few digits
    assert fc / ft == pytest.approx(2.333, rel=1e-3)


def test_drucker_prager_apex_is_on_the_surface_and_flagged():
    """The apex lies ON the yield surface and the normal is undefined there,
    so the return map needs a separate branch. Both facts are asserted."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    kappa = 50.0
    p = crit.apex_pressure(kappa)
    sig = float(p) * np.eye(3)[None]

    assert abs(crit.f(sig, np.array([kappa]))[0]) < 1e-10
    assert crit.singular_mask(sig)[0]
    assert not crit.singular_mask(_stresses(n=50)).any()


def test_drucker_prager_reduces_to_von_mises_at_zero_friction():
    crit = DruckerPrager(eta=0.0, xi=1.0)
    sig = _stresses()
    kappa = np.full(len(sig), 50.0)
    # f_DP = sqrt(J2) - kappa,  f_VM = sqrt(3 J2) - kappa: same surface for
    # kappa scaled by sqrt(3), so compare the yield locus rather than f itself
    vm = VonMises()
    assert np.abs(crit.f(sig, kappa * 0 + 1.0) * np.sqrt(3.0)
                  - (vm.f(sig, kappa * 0 + np.sqrt(3.0)))).max() < 1e-8


@pytest.mark.parametrize("match", ["outer", "inner", "plane-strain"])
def test_calibrated_cones_match_the_closed_form_strength_ratio(match):
    """For a uniaxial state ``sqrt(J2) = |sigma|/sqrt(3)`` and ``p = sigma/3``,
    so the compressive/tensile strength ratio of the cone is

        (sqrt(3) + eta) / (sqrt(3) - eta),

    independent of the cohesion. Checking the solved strengths against that
    expression tests ``f`` rather than restating the constructor.

    Note this is NOT the Mohr-Coulomb ratio ``(1+sin phi)/(1-sin phi)``, which
    is 3.0 at phi = 30 deg: uniaxial tension and compression sit at the two
    corner families of the hexagon, and a cone cannot pass through both. The
    three calibrations give 2.333, 1.800 and 2.849 respectively -- a spread of
    over 50%, which is why the choice has to be deliberate.
    """
    crit = DruckerPrager.from_mohr_coulomb(30.0, match=match)
    kappa = np.array([50.0])

    def strength(sign):
        lo, hi = 0.0, 1e5
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if crit.f(_uniaxial(sign * mid), kappa)[0] < 0:
                lo = mid
            else:
                hi = mid
        return lo

    expected = (np.sqrt(3.0) + crit.eta) / (np.sqrt(3.0) - crit.eta)
    assert strength(-1) / strength(+1) == pytest.approx(expected, rel=1e-6)
    assert strength(-1) > strength(+1)          # always stronger in compression


def test_inner_cone_lies_inside_the_outer_cone():
    """Both the aperture and the zero-pressure radius are ordered; the
    plane-strain cone is not between them, being calibrated on a different
    quantity entirely."""
    phi = 30.0
    outer = DruckerPrager.from_mohr_coulomb(phi, "outer")
    inner = DruckerPrager.from_mohr_coulomb(phi, "inner")
    assert inner.eta < outer.eta
    assert inner.xi < outer.xi

    ps = DruckerPrager.from_mohr_coulomb(phi, "plane-strain")
    assert ps.eta > outer.eta and ps.xi > outer.xi


def test_zero_friction_angle_gives_a_pressure_insensitive_cone():
    crit = DruckerPrager.from_mohr_coulomb(0.0)
    assert crit.eta == pytest.approx(0.0)
    sig = _stresses(n=20)
    kappa = np.full(len(sig), 50.0)
    f0 = crit.f(sig, kappa)
    f1 = crit.f(sig + 300.0 * np.eye(3), kappa)
    assert np.abs(f1 - f0).max() < 1e-9


def test_unknown_calibration_is_rejected():
    with pytest.raises(ValueError, match="match"):
        DruckerPrager.from_mohr_coulomb(30.0, match="middle")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_aliases_resolve():
    for alias in ("j2", "J2", "von-mises", "vonmises", "mises"):
        assert isinstance(get_criterion(alias), VonMises)
    for alias in ("dp", "drucker-prager", "drucker_prager"):
        assert isinstance(get_criterion(alias), DruckerPrager)


def test_unknown_criterion_explains_the_omissions():
    with pytest.raises(NotImplementedError, match="multi-surface"):
        get_criterion("mohr-coulomb")


def test_all_criteria_declare_smoothness():
    for name in CRITERIA:
        assert get_criterion(name).smooth is True
