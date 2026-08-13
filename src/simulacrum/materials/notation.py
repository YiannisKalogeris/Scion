r"""
Tensor notation for constitutive models.

Symmetric second-order tensors have six independent components and symmetric
fourth-order tensors thirty-six, so both can be stored as vectors and matrices.
Which storage convention you pick decides how much factor-of-two bookkeeping
the rest of the code carries.

Voigt notation stores ``[s11, s22, s33, s23, s13, s12]`` and compensates by
defining the *strain* vector with engineering shears, ``gamma = 2 eps``. It is
what finite element codes use at the B-matrix, and it is fine there. It is a
poor choice inside a return map, because stress and strain then transform
differently: the double contraction ``A : B`` is not the dot product of their
Voigt vectors, a fourth-order tensor is not the matrix product, and the
transpose of a Voigt stiffness is not the transpose of the tensor.

Mandel notation scales the off-diagonal components by ``sqrt(2)``:

    [s11, s22, s33, sqrt2 s23, sqrt2 s13, sqrt2 s12]

and with that single change everything becomes ordinary linear algebra:

    A : B          -> a . b            (plain dot product)
    C : A          -> C @ a            (plain matrix-vector)
    A : C : B      -> a @ C @ b
    ||A||          -> ||a||            (Frobenius norm is the 2-norm)
    C symmetric    <-> matrix symmetric

Stress and strain use the *same* map, so there is no separate convention to
remember and no place for a factor of two to go missing. That matters here
because the closest-point return map solves a 7x7 Newton system built from
``C``, ``d2g/dsigma2`` and ``df/dsigma`` together; in Voigt, three different
scalings would have to be tracked through it.

Conversion to Voigt happens once, at the boundary with the assembler, and both
directions live here so there is a single tested implementation rather than one
per constitutive subpackage.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "SQ2",
    "to_mandel",
    "from_mandel",
    "to_mandel4",
    "from_mandel4",
    "voigt_strain_to_mandel",
    "mandel_to_voigt_stress",
    "mandel_to_voigt_tangent",
    "identity_mandel",
    "MANDEL_INDEX",
]

SQ2 = np.sqrt(2.0)

#: (i, j) of each Mandel slot, and the scale applied to it.
MANDEL_INDEX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
_SCALE = np.array([1.0, 1.0, 1.0, SQ2, SQ2, SQ2])


def to_mandel(A: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` symmetric tensor -> ``(n, 6)``."""
    return np.stack([A[:, i, j] for i, j in MANDEL_INDEX], axis=1) * _SCALE


def from_mandel(a: np.ndarray) -> np.ndarray:
    """``(n, 6)`` -> ``(n, 3, 3)`` symmetric tensor."""
    a = np.asarray(a, dtype=float)
    out = np.zeros((a.shape[0], 3, 3))
    for k, (i, j) in enumerate(MANDEL_INDEX):
        v = a[:, k] / _SCALE[k]
        out[:, i, j] = v
        out[:, j, i] = v
    return out


def to_mandel4(C: np.ndarray) -> np.ndarray:
    """``(n, 3, 3, 3, 3)`` -> ``(n, 6, 6)``.

    The tensor must have both minor symmetries; only the symmetric part is
    representable, and an unsymmetric input is silently projected.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    out = np.empty((n, 6, 6))
    for a, (i, j) in enumerate(MANDEL_INDEX):
        for b, (k, l) in enumerate(MANDEL_INDEX):
            out[:, a, b] = C[:, i, j, k, l] * _SCALE[a] * _SCALE[b]
    return out


def from_mandel4(C: np.ndarray) -> np.ndarray:
    """``(n, 6, 6)`` -> ``(n, 3, 3, 3, 3)`` with both minor symmetries."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    out = np.zeros((n, 3, 3, 3, 3))
    for a, (i, j) in enumerate(MANDEL_INDEX):
        for b, (k, l) in enumerate(MANDEL_INDEX):
            v = C[:, a, b] / (_SCALE[a] * _SCALE[b])
            for ii, jj in ((i, j), (j, i)):
                for kk, ll in ((k, l), (l, k)):
                    out[:, ii, jj, kk, ll] = v
    return out


# ---------------------------------------------------------------------------
# Boundary with the assembler
# ---------------------------------------------------------------------------

def voigt_strain_to_mandel(eps: np.ndarray, dim: int = 3) -> np.ndarray:
    """Voigt engineering strain -> Mandel ``(n, 6)``.

    The inverse of :func:`mandel_to_voigt_tangent`'s convention on the strain
    side: the engineering shear ``gamma = 2 eps`` first becomes the tensor
    component, then picks up the Mandel ``sqrt(2)``, for a net ``1/sqrt(2)``.

    In 2-D the input is ``[e11, e22, gamma12]`` and the result is the **plane
    strain** tensor, with ``eps_33 = eps_13 = eps_23 = 0``. Plane *stress* is a
    different constraint -- ``sigma_33 = 0`` with ``eps_33`` unknown -- and
    cannot be imposed by filling in zeros here; it needs the constitutive
    update to solve for the out-of-plane strain.
    """
    eps = np.asarray(eps, dtype=float)
    out = np.zeros((eps.shape[0], 6))
    if dim == 3:
        out[:, :3] = eps[:, :3]
        out[:, 3:] = eps[:, 3:] / SQ2
    elif dim == 2:
        out[:, 0] = eps[:, 0]
        out[:, 1] = eps[:, 1]
        out[:, 5] = eps[:, 2] / SQ2
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    return out


def mandel_to_voigt_stress(a: np.ndarray, dim: int = 3) -> np.ndarray:
    """Mandel ``(n, 6)`` -> Voigt stress, ``(n, 6)`` in 3-D or ``(n, 3)`` in 2-D.

    In 2-D the components returned are ``[s11, s22, s12]``, the plane-strain
    triple that pairs with the assembler's B matrix. ``s33`` is dropped from
    the *return*, not from the computation -- it is nonzero in plane strain,
    it entered ``J2``, and it stays in the stored state.
    """
    v = a / _SCALE
    if dim == 3:
        return v
    if dim == 2:
        return np.stack([v[:, 0], v[:, 1], v[:, 5]], axis=1)
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def mandel_to_voigt_tangent(C: np.ndarray, dim: int = 3) -> np.ndarray:
    """Mandel ``(n, 6, 6)`` -> Voigt tangent for ``B.T @ C @ B`` assembly.

    Working the scalings through, ``sigma_v[a] = sum_b C_m[a,b] * s_b /
    (s_a * f_b) * eps_v[b]`` with ``s = (1,1,1,sqrt2,sqrt2,sqrt2)`` the Mandel
    scale and ``f_b = 2`` on the shear slots. The engineering factor of two in
    the strain and the ``sqrt2`` of the Mandel column cancel to leave simply
    ``1 / (s_a s_b)`` -- so no extra doubling belongs here, and adding one
    gives a shear stiffness twice too large.

    In 2-D the plane-strain sub-block is exact rather than approximate:
    ``eps_33 = eps_13 = eps_23 = 0``, so the omitted columns contribute
    nothing to the retained rows.
    """
    n = C.shape[0]
    if dim == 3:
        idx = list(range(6))
    elif dim == 2:
        idx = [0, 1, 5]
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    m = len(idx)
    out = np.empty((n, m, m))
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            out[:, a, b] = C[:, ia, ib] / (_SCALE[ia] * _SCALE[ib])
    return out


# ---------------------------------------------------------------------------
# Common operators
# ---------------------------------------------------------------------------

def identity_mandel() -> np.ndarray:
    """Fourth-order symmetric identity as a ``(6, 6)`` matrix: the identity."""
    return np.eye(6)
