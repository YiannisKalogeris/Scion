"""Tests for material models."""

import numpy as np
import pytest

from simulacrum.materials import LinearElastic, PlaneStress, PlaneStrain


def test_linear_elastic_3d_symmetry():
    mat = LinearElastic(E=210e9, nu=0.3)
    C = mat.tangent(dim=3)
    assert C.shape == (6, 6)
    np.testing.assert_allclose(C, C.T, atol=1e-6)


def test_plane_stress_vs_plane_strain():
    E, nu = 200e9, 0.3
    ps = PlaneStress(E=E, nu=nu).tangent()
    pe = PlaneStrain(E=E, nu=nu).tangent()
    assert ps.shape == (3, 3)
    assert pe.shape == (3, 3)
    # Plane strain should be stiffer
    assert pe[0, 0] > ps[0, 0]


def test_stress_strain_consistency():
    mat = LinearElastic(E=1.0, nu=0.0)
    strain = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    stress = mat.stress(strain, dim=3)
    # With nu=0: sigma_xx = E * eps_xx = 1
    np.testing.assert_allclose(stress[0], 1.0, atol=1e-12)
