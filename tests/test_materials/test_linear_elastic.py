"""
Verification for the linear elastic models.

The Mandel stiffness is checked against the tensor form and against the elastic
constants, and against the Voigt tangent of the same material -- the two differ
on the shear block, which is exactly the confusion the explicit name is there
to prevent.

python -m pytest tests/test_materials/test_linear_elastic.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from simulacrum.materials.linear_elastic import (
    LinearElastic3D,
    PlaneStrain,
    PlaneStress,
    isotropic_stiffness_mandel,
)
from simulacrum.materials.notation import to_mandel, to_mandel4

E, NU = 210e3, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
BULK = E / (3 * (1 - 2 * NU))


def _elastic_tensor():
    I = np.eye(3)
    return (LAM * np.einsum("ij,kl->ijkl", I, I)
            + 2 * MU * 0.5 * (np.einsum("ik,jl->ijkl", I, I)
                              + np.einsum("il,jk->ijkl", I, I)))


def test_mandel_stiffness_matches_the_fourth_order_tensor():
    assert np.abs(to_mandel4(_elastic_tensor()[None])[0]
                  - isotropic_stiffness_mandel(E, NU)).max() < 1e-9


def test_mandel_stiffness_reproduces_the_elastic_constants():
    C = isotropic_stiffness_mandel(E, NU)
    assert C[5, 5] == pytest.approx(2 * MU)
    assert (C[0, 0] + 2 * C[0, 1]) / 3 == pytest.approx(BULK)
    assert C[0, 0] == pytest.approx(LAM + 2 * MU)


def test_mandel_and_voigt_shear_blocks_differ_by_two():
    """The reason the Mandel form carries an explicit name: Voigt puts mu on
    the shear diagonal and Mandel 2 mu, and picking the wrong one gives a shear
    response off by a factor of two."""
    voigt = LinearElastic3D(E, NU).tangent(dim=3)
    mandel = isotropic_stiffness_mandel(E, NU)
    assert voigt[5, 5] == pytest.approx(MU)
    assert mandel[5, 5] == pytest.approx(2 * MU)
    assert np.abs(voigt[:3, :3] - mandel[:3, :3]).max() < 1e-9


def test_the_method_and_the_function_agree():
    assert np.abs(LinearElastic3D(E, NU).stiffness_mandel()
                  - isotropic_stiffness_mandel(E, NU)).max() == 0.0


def test_mandel_stiffness_maps_strain_to_stress():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((20, 3, 3))
    eps = 1e-4 * 0.5 * (A + np.swapaxes(A, 1, 2))
    sigma = to_mandel(eps) @ isotropic_stiffness_mandel(E, NU).T
    expected = to_mandel(np.einsum("ijkl,nkl->nij", _elastic_tensor(), eps))
    assert np.abs(sigma - expected).max() < 1e-9 * np.abs(expected).max()


def test_plane_models_reject_the_wrong_dimension():
    with pytest.raises(ValueError, match="dim"):
        PlaneStress(E, NU).tangent(dim=3)
    with pytest.raises(ValueError, match="dim"):
        PlaneStrain(E, NU).tangent(dim=3)
    with pytest.raises(ValueError, match="dim"):
        LinearElastic3D(E, NU).tangent(dim=2)
