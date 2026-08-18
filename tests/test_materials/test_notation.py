"""
Verification for the tensor-notation conversions.

The point of Mandel notation is that tensor operations become ordinary linear
algebra, so every test here asserts exactly that: the same quantity computed on
3x3 tensors and on 6-vectors must agree to machine precision.

python -m pytest tests/test_materials/test_notation.py -q 
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.materials.notation import (
    from_mandel,
    from_mandel4,
    mandel_to_voigt_stress,
    mandel_to_voigt_tangent,
    to_mandel,
    to_mandel4,
)

E, NU = 210e3, 0.3


def _sym(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, 3, 3))
    return 0.5 * (A + np.swapaxes(A, 1, 2))


def _stiffness():
    """Mandel elastic stiffness, built here so the notation tests depend on no
    material model."""
    return to_mandel4(_elastic_tensor()[None])[0]


def _elastic_tensor():
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    mu = E / (2 * (1 + NU))
    I = np.eye(3)
    return (lam * np.einsum("ij,kl->ijkl", I, I)
            + 2 * mu * 0.5 * (np.einsum("ik,jl->ijkl", I, I)
                              + np.einsum("il,jk->ijkl", I, I)))


def test_roundtrip_is_exact():
    A = _sym(50)
    assert np.abs(from_mandel(to_mandel(A)) - A).max() < 1e-15


def test_double_contraction_becomes_a_dot_product():
    """The defining property, and the reason Voigt is unusable inside a return
    map: there A:B is not a dot product and stress and strain scale
    differently."""
    A, B = _sym(50, 0), _sym(50, 1)
    tensor = np.einsum("nij,nij->n", A, B)
    vector = np.einsum("ni,ni->n", to_mandel(A), to_mandel(B))
    assert np.abs(tensor - vector).max() < 1e-12 * np.abs(tensor).max()


def test_frobenius_norm_becomes_the_two_norm():
    A = _sym(50)
    assert np.abs(np.linalg.norm(A, axis=(1, 2))
                  - np.linalg.norm(to_mandel(A), axis=1)).max() < 1e-13


def test_fourth_order_contraction_becomes_matrix_multiplication():
    C4 = _elastic_tensor()
    A = _sym(50)
    tensor = to_mandel(np.einsum("ijkl,nkl->nij", C4, A))
    matrix = to_mandel(A) @ to_mandel4(C4[None])[0].T
    assert np.abs(tensor - matrix).max() < 1e-10 * np.abs(tensor).max()


def test_symmetric_tensor_gives_a_symmetric_matrix():
    """A property Voigt does not have, and one the Newton solve relies on."""
    C = to_mandel4(_elastic_tensor()[None])[0]
    assert np.abs(C - C.T).max() == 0.0


def test_fourth_order_roundtrip():
    C = _stiffness()
    assert np.abs(to_mandel4(from_mandel4(C[None]))[0] - C).max() < 1e-12


# ---------------------------------------------------------------------------
# The Voigt boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [2, 3])
def test_voigt_tangent_pairs_with_the_engineering_strain_vector(dim):
    """sigma_voigt = C_voigt @ eps_voigt with gamma = 2 eps. Doubling the shear
    columns as well -- the obvious mistake -- gives a shear stiffness twice too
    large, which is exactly the bug this test exists to catch."""
    C = _stiffness()
    eps = _sym(20)
    if dim == 2:                                  # plane strain
        eps[:, 2, :] = eps[:, :, 2] = 0.0

    sigma = to_mandel(eps) @ C.T
    Cv = mandel_to_voigt_tangent(np.broadcast_to(C, (20, 6, 6)), dim=dim)

    if dim == 3:
        ev = np.stack([eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2],
                       2 * eps[:, 1, 2], 2 * eps[:, 0, 2], 2 * eps[:, 0, 1]], 1)
    else:
        ev = np.stack([eps[:, 0, 0], eps[:, 1, 1], 2 * eps[:, 0, 1]], 1)

    predicted = np.einsum("nab,nb->na", Cv, ev)
    assert np.abs(predicted - mandel_to_voigt_stress(sigma, dim=dim)).max() \
        < 1e-9 * np.abs(sigma).max()


def test_voigt_shear_stiffness_is_mu_not_two_mu():
    C = mandel_to_voigt_tangent(_stiffness()[None], dim=3)[0]
    assert C[5, 5] == pytest.approx(E / (2 * (1 + NU)))


def test_plane_strain_keeps_sigma_33_in_the_computation():
    """The 2-D Voigt output drops sigma_33 from the *return*, not from the
    state: it is nonzero in plane strain and enters every pressure-sensitive
    yield function."""
    C = _stiffness()
    eps = np.zeros((1, 3, 3))
    eps[0, 0, 0] = 1e-3
    sigma = to_mandel(eps) @ C.T
    full = mandel_to_voigt_stress(sigma, dim=3)
    plane = mandel_to_voigt_stress(sigma, dim=2)

    assert abs(full[0, 2]) > 0.1 * abs(full[0, 0])       # sigma_33 is large
    assert plane.shape == (1, 3)
    assert np.allclose(plane[0], [full[0, 0], full[0, 1], full[0, 5]])


def test_bad_dimension_is_rejected():
    C = _stiffness()[None]
    with pytest.raises(ValueError, match="dim"):
        mandel_to_voigt_tangent(C, dim=1)
    with pytest.raises(ValueError, match="dim"):
        mandel_to_voigt_stress(np.zeros((1, 6)), dim=4)
