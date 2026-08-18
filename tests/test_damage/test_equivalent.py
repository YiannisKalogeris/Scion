"""
Verification for the equivalent strain measures.

Every measure gets a finite-difference check on its derivative -- the term that
makes the damage tangent unsymmetric and the one whose absence shows up only as
a Newton iteration count -- plus a check that it means what its calibration
says on the states where the answer is known by hand.

python -m pytest tests/test_damage/test_equivalent.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.damage.equivalent import (
    EnergyRelease,
    Mazars,
    ModifiedVonMises,
    Rankine,
    get_equivalent_strain,
)

E, NU, K = 30.0e3, 0.2, 10.0


def _measures():
    m = [Mazars(), ModifiedVonMises(k=K), Rankine(), EnergyRelease()]
    for x in m:
        x.bind(E, NU)
    return m


def _random_states(seed=0, n=8):
    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=1.0e-3, size=(n, 3, 3))
    return 0.5 * (eps + np.swapaxes(eps, 1, 2))


def _fd_derivative(measure, eps, h=1.0e-9):
    """``deps_eq/deps`` by central differences, respecting symmetry.

    A perturbation of an off-diagonal component has to touch both ``(i,j)`` and
    ``(j,i)`` to stay symmetric, which makes the directional derivative twice
    the component of the gradient -- hence the halving. Getting that factor
    wrong makes a correct derivative look 2x wrong on the shear terms only.
    """
    out = np.zeros_like(eps)
    for i in range(3):
        for j in range(3):
            p = np.zeros((3, 3))
            if i == j:
                p[i, i] = h
            else:
                p[i, j] = p[j, i] = h
            d = (measure.value(eps + p) - measure.value(eps - p)) / (2.0 * h)
            out[:, i, j] = d if i == j else 0.5 * d
    return out


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("measure", _measures(), ids=lambda m: m.name)
def test_the_derivative_is_the_derivative(measure):
    eps = _random_states()
    analytic = measure.dvalue_deps(eps)
    assert np.abs(analytic - _fd_derivative(measure, eps)).max() \
        < 1e-6 * np.abs(analytic).max()


@pytest.mark.parametrize("measure", _measures(), ids=lambda m: m.name)
def test_the_derivative_is_symmetric(measure):
    """It contracts with a symmetric strain increment, so only its symmetric
    part is meaningful -- and an unsymmetric one would break the Voigt tangent
    silently."""
    d = measure.dvalue_deps(_random_states(1))
    assert np.allclose(d, np.swapaxes(d, 1, 2))


@pytest.mark.parametrize("measure", _measures(), ids=lambda m: m.name)
def test_the_measure_is_non_negative(measure):
    assert np.all(measure.value(_random_states(2)) >= 0.0)


@pytest.mark.parametrize("measure", _measures(), ids=lambda m: m.name)
def test_an_unstrained_point_is_undamaging_with_a_finite_derivative(measure):
    """Zero strain is where several of these are non-differentiable. The value
    must be zero and the reported derivative finite, or the first increment of
    every analysis produces a NaN."""
    eps = np.zeros((1, 3, 3))
    assert measure.value(eps)[0] == 0.0
    assert np.all(np.isfinite(measure.dvalue_deps(eps)))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("measure", _measures(), ids=lambda m: m.name)
def test_uniaxial_tension_returns_the_axial_strain(measure):
    """The shared normalisation. Without it, comparing two measures compares
    two arbitrary scalings as well as two theories."""
    assert measure.uniaxial(1.0e-3)[0] == pytest.approx(1.0e-3, rel=1e-12)


def test_what_each_measure_makes_of_uniaxial_compression():
    """Where the four theories actually differ, on the one state that
    distinguishes them."""
    m = {x.name: x for x in _measures()}
    e = -1.0e-3

    # Mazars sees only the two tensile lateral strains, nu |e| each
    assert m["mazars"].uniaxial(e)[0] == pytest.approx(
        np.sqrt(2.0) * NU * 1.0e-3)
    # the whole point of k: compression is k times less damaging
    assert m["modified-von-mises"].uniaxial(e)[0] == pytest.approx(1.0e-3 / K)
    # no tensile principal stress at all
    assert m["rankine"].uniaxial(e)[0] == 0.0
    # energy is quadratic, so it cannot tell the two apart
    assert m["energy-release"].uniaxial(e)[0] == pytest.approx(1.0e-3)


def test_modified_von_mises_at_k_equals_one_is_symmetric():
    m = ModifiedVonMises(k=1.0)
    m.bind(E, NU)
    assert m.uniaxial(1e-3)[0] == pytest.approx(m.uniaxial(-1e-3)[0])


def test_mazars_ignores_hydrostatic_compression_entirely():
    """However large. The statement that a material fails by opening cracks."""
    m = Mazars()
    m.bind(E, NU)
    eps = -0.05 * np.eye(3)[None]
    assert m.value(eps)[0] == 0.0


def test_mazars_cannot_distinguish_shear_from_tension():
    """Pure shear has principal strains (g/2, 0, -g/2), so Mazars returns g/2 --
    the same as uniaxial tension of g/2. A real limitation, asserted so it is
    not mistaken for a bug later."""
    m = Mazars()
    m.bind(E, NU)
    g = 2.0e-3
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 1] = eps[0, 1, 0] = 0.5 * g
    assert m.value(eps)[0] == pytest.approx(0.5 * g)


def test_rankine_and_the_largest_principal_strain_are_different_measures():
    """Both are 'the most tensile direction'; the Poisson coupling makes them
    disagree in a biaxial state, which is why the class documents the
    distinction."""
    m = Rankine()
    m.bind(E, NU)
    eps = np.diag([1.0e-3, 1.0e-3, 0.0])[None]      # equibiaxial in plane
    assert m.value(eps)[0] != pytest.approx(1.0e-3, rel=1e-3)


def test_energy_release_is_the_stored_energy():
    m = EnergyRelease()
    m.bind(E, NU)
    eps = _random_states(3)
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    tr = np.einsum("nii->n", eps)
    sig = lam * tr[:, None, None] * np.eye(3) + 2.0 * mu * eps
    psi = 0.5 * np.einsum("nij,nij->n", sig, eps)
    assert np.allclose(m.value(eps), np.sqrt(2.0 * psi / E))


# ---------------------------------------------------------------------------
# Elastic constants and the tangent symmetry flag
# ---------------------------------------------------------------------------

def test_bind_fills_missing_constants_and_rejects_a_conflict():
    """A measure carrying a different nu from its model would disagree about
    when the material damages while reporting the same stiffness."""
    m = ModifiedVonMises(k=K)
    m.bind(E, NU)
    assert m.nu == NU and m.E == E

    held = ModifiedVonMises(k=K, nu=0.3)
    with pytest.raises(ValueError, match="must agree"):
        held.bind(E, NU)
    held.bind(E, 0.3)                                # consistent, so fine


def test_a_measure_that_needs_nu_says_so_rather_than_guessing():
    with pytest.raises(ValueError, match="needs nu"):
        ModifiedVonMises(k=K).value(np.zeros((1, 3, 3)))


def test_the_incompressible_limit_is_refused_by_modified_von_mises():
    m = ModifiedVonMises(k=K, nu=0.5)
    with pytest.raises(ValueError, match="incompressible"):
        m.value(np.zeros((1, 3, 3)))


def test_only_the_energy_measure_gives_a_symmetric_tangent_term():
    """Its derivative is parallel to the effective stress; the others are not,
    and the damage tangent is genuinely unsymmetric for them."""
    flags = {m.name: m.symmetric_rank_one for m in _measures()}
    assert flags == {"mazars": False, "modified-von-mises": False,
                     "rankine": False, "energy-release": True}


def test_the_registry_resolves_and_rejects():
    assert isinstance(get_equivalent_strain("de-vree", k=5.0), ModifiedVonMises)
    assert isinstance(get_equivalent_strain("simo_ju"), EnergyRelease)
    with pytest.raises(NotImplementedError):
        get_equivalent_strain("lemaitre")


def test_a_negative_k_is_refused():
    with pytest.raises(ValueError, match="k = f_c/f_t"):
        ModifiedVonMises(k=-1.0)
