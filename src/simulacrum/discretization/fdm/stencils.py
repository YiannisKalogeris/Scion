r"""
Finite-difference stencil definitions.

Each stencil maps relative offsets to coefficients, *before* division by the
grid spacing. A 1-D stencil ``{-1: 1, 0: -2, 1: 1}`` means

    u''(x) ~ (1/h^2) sum_k c_k u(x + k h)

Consistency is one sum
----------------------
The coefficients of any Laplacian stencil must sum to zero, or the operator
returns something nonzero for a constant field and the scheme is not consistent
at all. It is one line to check and it catches a mistyped coefficient
immediately, which is why every stencil here is tested against it.

Separable and compact stencils are different animals
----------------------------------------------------
A multi-dimensional central Laplacian is a *sum* of 1-D operators, one per
direction, each with its own spacing:

    lap ~ D_xx / hx^2 + D_yy / hy^2

so it costs nothing to support anisotropic grids or a higher order -- see
:class:`~simulacrum.discretization.fdm.assembler.FDMAssembler`, which builds
exactly that. The 2-D dictionaries below are the *assembled* form of that sum on
a **square** grid, kept because they are the readable statement of what the
scheme is, and they carry a factor that is only correct when ``hx == hy``.

The 9-point compact stencil is not separable and never will be: it mixes the
corners deliberately. It is also not fourth-order on its own -- see
:func:`laplacian_2d_9pt`.
"""

from __future__ import annotations

__all__ = [
    "laplacian_1d",
    "laplacian_2d_5pt",
    "laplacian_2d_9pt",
    "mehrstellen_rhs_stencil",
    "is_consistent",
]


# =====================================================================
# 1-D
# =====================================================================

def laplacian_1d(order: int = 2) -> dict[int, float]:
    """Central second difference, ``u'' ~ (1/h^2) sum c_k u(x + k h)``.

    ``order`` is the order of accuracy of the truncation error, 2 or 4. The
    fourth-order stencil reaches two points either side, which matters at a
    boundary: the first interior point needs a value one cell *outside* the
    domain. See the assembler's ``boundary`` argument.
    """
    if order == 2:
        return {-1: 1.0, 0: -2.0, 1: 1.0}
    if order == 4:
        return {-2: -1 / 12, -1: 4 / 3, 0: -5 / 2, 1: 4 / 3, 2: -1 / 12}
    raise NotImplementedError(
        f"order-{order} 1-D Laplacian stencil; available: 2, 4")


# =====================================================================
# 2-D, assembled on a square grid
# =====================================================================

def laplacian_2d_5pt() -> dict[tuple[int, int], float]:
    """Standard 5-point Laplacian, **on a uniform square grid**.

    The single ``-4`` at the centre is ``-2/hx^2 - 2/hy^2`` with ``hx == hy``
    factored out. On a rectangular grid the two directions carry different
    factors and this dictionary cannot express that; the assembler builds the
    operator per direction instead and does not use this.
    """
    return {
        (0, 0): -4.0,
        (-1, 0): 1.0, (1, 0): 1.0,
        (0, -1): 1.0, (0, 1): 1.0,
    }


def laplacian_2d_9pt() -> dict[tuple[int, int], float]:
    r"""9-point compact (Mehrstellen) Laplacian, on a uniform square grid.

    Fourth order **only with a corrected right-hand side.** The operator itself
    satisfies

        A_9 u = lap(u) + (h^2/12) lap^2(u) + O(h^4),

    and since ``lap^2 u = lap f`` for ``lap u = f``, solving ``A_9 u = f`` leaves
    an ``O(h^2)`` error -- no better than the 5-point stencil, for twice the
    work. Solving

        A_9 u = f + (h^2/12) lap(f)

    instead cancels the leading term and gives fourth order. That correction is
    :func:`mehrstellen_rhs_stencil`, and the assembler applies it when asked.

    The exception is a harmonic right-hand side, ``lap f = 0``, where the
    correction vanishes and the plain form is already fourth order. That is a
    narrow enough case to be a trap rather than a feature: a scheme that is
    fourth order on your test problem and second order on the real one is worse
    than one that is honestly second order everywhere.
    """
    return {
        (0, 0): -20 / 6,
        (-1, 0): 4 / 6, (1, 0): 4 / 6,
        (0, -1): 4 / 6, (0, 1): 4 / 6,
        (-1, -1): 1 / 6, (-1, 1): 1 / 6,
        (1, -1): 1 / 6, (1, 1): 1 / 6,
    }


def mehrstellen_rhs_stencil() -> dict[tuple[int, int], float]:
    """The ``f + (h^2/12) lap f`` correction, as a stencil applied to ``f``.

    ``(1/12)`` on the four edge neighbours and ``(8/12)`` at the centre, which
    is ``I + (h^2/12) lap_5`` written out. Applied to ``f`` sampled on the grid,
    so it needs ``f`` one cell beyond the boundary as well.
    """
    return {
        (0, 0): 8 / 12,
        (-1, 0): 1 / 12, (1, 0): 1 / 12,
        (0, -1): 1 / 12, (0, 1): 1 / 12,
    }


# =====================================================================

def is_consistent(stencil, tol: float = 1.0e-12) -> bool:
    """Whether the coefficients sum to zero, as a Laplacian's must.

    A stencil that fails this returns something nonzero for a constant field,
    which is not an approximation of a second derivative in any limit.
    """
    return abs(sum(stencil.values())) < tol
