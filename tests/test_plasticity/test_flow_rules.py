"""
Verification for the flow rules.

The gradient of the potential is checked against the potential itself, which is
the one test that catches ``g`` and ``dg/dsigma`` drifting apart. Beyond that
the tests pin the physical content: how much plastic volume change each rule
predicts, and whether the resulting tangent is symmetric.

python -m pytest tests/test_plasticity/test_flow_rules.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.plasticity.flow_rules import (
    Associative,
    NonAssociative,
    get_flow_rule,
)
from simulacrum.materials.plasticity.yield_criteria import (
    DruckerPrager,
    VonMises,
)

PHI = 35.0


def _stresses(n=200, scale=100.0, seed=0):
    rng = np.random.default_rng(seed)
    A = scale * rng.standard_normal((n, 3, 3))
    return 0.5 * (A + np.swapaxes(A, 1, 2))


def _rules():
    return [
        ("associative", Associative()),
        ("dp-psi-10", NonAssociative.drucker_prager(10.0)),
        ("volume-preserving", NonAssociative.volume_preserving()),
    ]


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,rule", _rules(), ids=[r[0] for r in _rules()])
def test_flow_direction_is_the_gradient_of_the_potential(label, rule):
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses()
    kappa = np.full(len(sig), 50.0)
    n = rule.dg_dsigma(crit, sig)

    h = 1e-4
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0 if i == j else 0.5
            fd = (rule.g(crit, sig + h * E, kappa)
                  - rule.g(crit, sig - h * E, kappa)) / (2 * h)
            assert np.abs(fd - n[:, i, j]).max() < 1e-6


@pytest.mark.parametrize("label,rule", _rules(), ids=[r[0] for r in _rules()])
def test_potential_hessian_is_the_gradient_of_the_flow_direction(label, rule):
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses()
    H = rule.d2g_dsigma2(crit, sig)

    h = 1e-4
    for i in range(3):
        for j in range(3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = 1.0 if i == j else 0.5
            fd = (rule.dg_dsigma(crit, sig + h * E)
                  - rule.dg_dsigma(crit, sig - h * E)) / (2 * h)
            assert np.abs(fd - H[:, :, :, i, j]).max() < 1e-6


@pytest.mark.parametrize("label,rule", _rules(), ids=[r[0] for r in _rules()])
def test_potential_hessian_has_both_minor_symmetries(label, rule):
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    H = rule.d2g_dsigma2(crit, _stresses(n=20))
    assert np.abs(H - np.swapaxes(H, 1, 2)).max() < 1e-10
    assert np.abs(H - np.swapaxes(H, 3, 4)).max() < 1e-10


# ---------------------------------------------------------------------------
# Associative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("crit", [VonMises(),
                                  DruckerPrager.from_mohr_coulomb(PHI)],
                         ids=["von-mises", "drucker-prager"])
def test_associative_flow_is_exactly_the_yield_surface_normal(crit):
    sig = _stresses()
    rule = Associative()
    assert np.abs(rule.dg_dsigma(crit, sig) - crit.df_dsigma(sig)).max() == 0.0
    assert np.abs(rule.d2g_dsigma2(crit, sig)
                  - crit.d2f_dsigma2(sig)).max() == 0.0


def test_associative_von_mises_flow_is_volume_preserving():
    """Metals do not dilate plastically, and J2 with associative flow gets this
    right without any extra assumption."""
    rule = Associative()
    d = rule.dilatancy(VonMises(), _stresses())
    assert np.abs(d).max() < 1e-12


def test_associative_drucker_prager_dilates_at_the_friction_angle():
    """The over-prediction that motivates non-associative flow: the plastic
    volume change is tied to the friction angle, not to measured dilatancy."""
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    d = Associative().dilatancy(crit, _stresses())
    assert np.abs(d - crit.eta).max() < 1e-12
    assert crit.eta > 0.8                     # phi = 35 deg is strongly dilatant


# ---------------------------------------------------------------------------
# Non-associative
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("psi,expected_sign", [(0.0, 0), (5.0, 1), (20.0, 1)])
def test_dilatancy_is_set_by_the_dilatancy_angle_not_the_friction_angle(
        psi, expected_sign):
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    rule = NonAssociative.drucker_prager(psi)
    d = rule.dilatancy(crit, _stresses())
    expected = DruckerPrager.from_mohr_coulomb(psi).eta
    assert np.abs(d - expected).max() < 1e-12
    if expected_sign == 0:
        assert np.abs(d).max() < 1e-12
    else:
        assert d.mean() > 0 and d.mean() < crit.eta


def test_dilatancy_increases_monotonically_with_psi():
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses(n=20)
    d = [NonAssociative.drucker_prager(p).dilatancy(crit, sig).mean()
         for p in (0.0, 5.0, 10.0, 20.0, 35.0)]
    assert np.all(np.diff(d) > 0)


def test_psi_equal_to_phi_recovers_associative_flow():
    """The limiting case must agree exactly, or the two paths disagree on a
    problem that should be identical."""
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses()
    rule = NonAssociative.drucker_prager(PHI)
    assert np.abs(rule.dg_dsigma(crit, sig) - crit.df_dsigma(sig)).max() < 1e-12
    assert np.abs(rule.dilatancy(crit, sig) - crit.eta).max() < 1e-12


def test_volume_preserving_flow_is_deviatoric():
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    n = NonAssociative.volume_preserving().dg_dsigma(crit, _stresses())
    assert np.abs(np.einsum("nii->n", n)).max() < 1e-12


def test_non_associative_flow_differs_from_the_surface_normal():
    """If it did not, the whole axis would be pointless."""
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses()
    rule = NonAssociative.drucker_prager(10.0)
    assert np.abs(rule.dg_dsigma(crit, sig) - crit.df_dsigma(sig)).max() > 0.1


def test_non_associative_potential_has_its_own_singular_set():
    """The flow direction can be undefined where the yield surface is regular,
    so the return map must test both."""
    rule = NonAssociative.drucker_prager(10.0)
    apex = np.eye(3)[None] * 40.0            # zero deviator
    assert rule.singular_mask(apex)[0]
    assert not rule.singular_mask(_stresses(n=50)).any()


# ---------------------------------------------------------------------------
# Tangent symmetry
# ---------------------------------------------------------------------------

def test_symmetry_flag_matches_associativity():
    """The assembler picks a linear solver from this, so it must not lie: a
    non-associative model stored in a symmetric format loses half the tangent.
    """
    assert Associative().symmetric_tangent is True
    assert Associative().associative is True
    for rule in (NonAssociative.drucker_prager(10.0),
                 NonAssociative.volume_preserving()):
        assert rule.symmetric_tangent is False
        assert rule.associative is False


def test_elastoplastic_tangent_is_symmetric_only_for_associative_flow():
    """The operator contains dg/dsigma (x) df/dsigma, symmetric exactly when
    the two coincide. Built here from the pieces rather than asserted."""
    crit = DruckerPrager.from_mohr_coulomb(PHI)
    sig = _stresses(n=30)

    def outer_product(rule):
        m = rule.dg_dsigma(crit, sig)
        n = crit.df_dsigma(sig)
        return np.einsum("nij,nkl->nijkl", m, n)

    T = outer_product(Associative())
    assert np.abs(T - np.einsum("nijkl->nklij", T)).max() < 1e-9

    T = outer_product(NonAssociative.drucker_prager(10.0))
    assert np.abs(T - np.einsum("nijkl->nklij", T)).max() > 1e-3


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_resolves_and_rejects():
    assert isinstance(get_flow_rule("associative"), Associative)
    assert isinstance(get_flow_rule("normal"), Associative)
    rule = get_flow_rule("non-associative",
                         potential=DruckerPrager.from_mohr_coulomb(10.0))
    assert isinstance(rule, NonAssociative)
    with pytest.raises(NotImplementedError):
        get_flow_rule("radial")
