"""
Verification for the finite-difference assembler.

The characteristic check for a discretisation is the **observed order of
accuracy**: solve a problem whose exact answer is known, refine, and measure the
slope. It is a number the code cannot fake -- a scheme with a mistyped
coefficient still converges, just at the wrong rate or to the wrong thing -- and
it is the only test here that would have caught the anisotropic-spacing bug
this module was carrying.

That bug is worth stating, because it is the shape of thing this file exists to
prevent. The 5-point Laplacian was assembled by dividing the whole stencil by
``hx*hy``, which equals ``h^2`` on a square grid and nothing meaningful
otherwise. On a square grid the scheme was exactly right; on any other the
operator was consistent with a different equation, and the error sat at 20%
through three refinements without moving. Every isotropic test passed.

python -m pytest tests/test_discretization/test_fdm.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from scion.discretization.fdm import FDMAssembler
from scion.discretization.fdm.stencils import (
    is_consistent,
    laplacian_1d,
    laplacian_2d_5pt,
    laplacian_2d_9pt,
    mehrstellen_rhs_stencil,
)


def _solve(assembler, source, boundary, mehrstellen=False):
    system = assembler.assemble(source=source, boundary=boundary,
                                mehrstellen=mehrstellen)
    return spsolve(system.matrix.tocsc(), system.rhs)


def _error(n, Lx=1.0, Ly=1.0, **kw):
    """Relative max error of a manufactured sine solution on an n x n grid."""
    mehrstellen = kw.pop("mehrstellen", False)
    a = FDMAssembler(None, (n, n), spacing=(Lx / (n + 1), Ly / (n + 1)), **kw)
    u = lambda x, y: np.sin(2 * np.pi * x / Lx) * np.sin(3 * np.pi * y / Ly)
    f = lambda x, y: -((2 * np.pi / Lx) ** 2
                       + (3 * np.pi / Ly) ** 2) * u(x, y)
    got = _solve(a, f, u, mehrstellen)
    exact = u(*a.coordinates().T)
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def _observed_order(errors):
    return [float(np.log2(a / b)) for a, b in zip(errors, errors[1:])]


# ---------------------------------------------------------------------------
# Stencils
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil", [
    laplacian_1d(2), laplacian_1d(4), laplacian_2d_5pt(), laplacian_2d_9pt()],
    ids=["1d-o2", "1d-o4", "2d-5pt", "2d-9pt"])
def test_every_laplacian_stencil_annihilates_a_constant(stencil):
    """The coefficients must sum to zero, or the operator returns something
    nonzero for a constant field and approximates no derivative at all."""
    assert is_consistent(stencil)
    assert sum(stencil.values()) == pytest.approx(0.0, abs=1e-12)


def test_the_right_hand_side_correction_is_a_weighted_average():
    """``I + (h^2/12) lap`` applied to f, so its weights sum to one -- unlike a
    Laplacian stencil, which sums to zero."""
    assert sum(mehrstellen_rhs_stencil().values()) == pytest.approx(1.0)


def test_an_unavailable_order_is_refused():
    with pytest.raises(NotImplementedError, match="order-3"):
        laplacian_1d(3)


# ---------------------------------------------------------------------------
# The operator itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Lx,Ly", [(1.0, 1.0), (1.0, 3.0), (4.0, 0.5)],
                         ids=["square", "tall", "wide"])
def test_the_operator_reproduces_a_known_laplacian(Lx, Ly):
    """Applied to a field whose Laplacian is known, on square *and* rectangular
    grids. The rectangular cases are the regression: they used to sit at 20%
    error regardless of refinement."""
    n = 40
    a = FDMAssembler(None, (n, n), spacing=(Lx / (n + 1), Ly / (n + 1)))
    points = a.coordinates()
    u = np.sin(np.pi * points[:, 0] / Lx) * np.sin(np.pi * points[:, 1] / Ly)
    exact = -((np.pi / Lx) ** 2 + (np.pi / Ly) ** 2) * u

    got = a.operator() @ u
    assert np.abs(got - exact).max() < 0.02 * np.abs(exact).max()


def test_the_operator_is_symmetric_and_negative_definite():
    """``A`` approximates ``lap``, so it is negative definite and ``A u = f``
    needs a solver that knows it -- or a sign flip the caller applies."""
    a = FDMAssembler(None, (6, 5), spacing=(0.3, 0.7))
    A = a.operator().toarray()

    assert np.abs(A - A.T).max() < 1e-12
    assert np.linalg.eigvalsh(A).max() < 0.0


def test_a_uniform_grid_still_gives_the_classical_five_point_stencil():
    """The Kronecker construction has to reproduce ``{-4, 1, 1, 1, 1}/h^2`` on a
    square grid, or it is a different scheme wearing the same name."""
    h = 0.25
    a = FDMAssembler(None, (5, 5), spacing=h)
    A = a.operator().toarray()
    middle = 2 * 5 + 2                              # an interior-interior point

    assert A[middle, middle] == pytest.approx(-4.0 / h ** 2)
    for neighbour in (middle - 5, middle + 5, middle - 1, middle + 1):
        assert A[middle, neighbour] == pytest.approx(1.0 / h ** 2)
    assert np.count_nonzero(A[middle]) == 5


# ---------------------------------------------------------------------------
# Exactness on polynomials
# ---------------------------------------------------------------------------

def test_second_order_is_exact_for_a_cubic():
    """The truncation error of the central second difference is
    ``h^2 u'''' / 12``, which vanishes for any polynomial of degree three or
    less. Exact to round-off, not merely convergent -- a much sharper statement
    than an error norm."""
    n = 7
    a = FDMAssembler(None, (n, n), spacing=(0.3, 0.11), order=2)
    u = lambda x, y: x ** 3 + y ** 3 + 2 * x * y + x - 5.0
    f = lambda x, y: 6.0 * x + 6.0 * y

    got = _solve(a, f, u)
    assert np.abs(got - u(*a.coordinates().T)).max() < 1e-9


def test_fourth_order_is_exact_for_a_quintic():
    """Same argument one order up: the leading error is ``h^4 u^(6) / 90``."""
    n = 9
    a = FDMAssembler(None, (n, n), spacing=(0.2, 0.35), order=4)
    u = lambda x, y: x ** 5 - 2 * y ** 5 + x ** 2 * y
    f = lambda x, y: 20.0 * x ** 3 - 40.0 * y ** 3 + 2.0 * y

    got = _solve(a, f, u)
    assert np.abs(got - u(*a.coordinates().T)).max() < 1e-7


def test_a_constant_boundary_with_no_source_gives_a_constant_field():
    """The simplest check that the boundary data reaches the right-hand side
    with the right sign. Before, the right-hand side was unconditionally zero
    and this returned zero for any boundary value."""
    a = FDMAssembler(None, (6, 4), spacing=(0.2, 0.5))
    got = _solve(a, None, 3.75)
    assert np.abs(got - 3.75).max() < 1e-10


# ---------------------------------------------------------------------------
# Observed order of accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order,expected", [(2, 2.0), (4, 4.0)])
def test_the_observed_order_matches_the_scheme_in_one_dimension(order, expected):
    L = 1.0
    u = lambda x: np.sin(np.pi * x / L) * np.exp(x)
    f = lambda x: np.exp(x) * ((1.0 - (np.pi / L) ** 2) * np.sin(np.pi * x / L)
                               + 2.0 * (np.pi / L) * np.cos(np.pi * x / L))
    errors = []
    for n in (10, 20, 40, 80):
        a = FDMAssembler(None, (n,), spacing=L / (n + 1), order=order)
        got = _solve(a, f, u)
        exact = u(a.coordinates()[:, 0])
        errors.append(float(np.abs(got - exact).max() / np.abs(exact).max()))

    assert _observed_order(errors)[-1] == pytest.approx(expected, abs=0.25)


@pytest.mark.parametrize("order,expected", [(2, 2.0), (4, 4.0)])
def test_the_observed_order_matches_the_scheme_in_two_dimensions(order,
                                                                 expected):
    errors = [_error(n, order=order) for n in (10, 20, 40, 80)]
    assert _observed_order(errors)[-1] == pytest.approx(expected, abs=0.25)


@pytest.mark.parametrize("Lx,Ly", [(1.0, 3.0), (5.0, 0.4)],
                         ids=["tall", "wide"])
def test_anisotropic_spacing_converges_at_the_right_order(Lx, Ly):
    """The regression test for the bug. Under the old assembly these errors did
    not fall at all -- 2.01e-01, 2.00e-01, 2.00e-01 across three refinements --
    because the operator was consistent with the wrong equation."""
    errors = [_error(n, Lx=Lx, Ly=Ly) for n in (10, 20, 40, 80)]

    assert _observed_order(errors)[-1] == pytest.approx(2.0, abs=0.25)
    assert errors[-1] < 0.02 * errors[0]


# ---------------------------------------------------------------------------
# The compact scheme
# ---------------------------------------------------------------------------

def test_the_compact_stencil_is_only_second_order_without_the_correction():
    """The trap the stencil's docstring describes. Nine points, more work than
    five, and the same order -- unless the right-hand side is corrected too."""
    errors = [_error(n, scheme="compact") for n in (10, 20, 40, 80)]
    assert _observed_order(errors)[-1] == pytest.approx(2.0, abs=0.25)


def test_the_mehrstellen_correction_buys_the_fourth_order():
    errors = [_error(n, scheme="compact", mehrstellen=True)
              for n in (10, 20, 40)]
    assert _observed_order(errors)[-1] == pytest.approx(4.0, abs=0.4)
    assert errors[-1] < 1e-6


def test_the_compact_scheme_refuses_a_grid_it_cannot_represent():
    """It mixes the corners, so it is the Laplacian only when the cells are
    square. Refusing beats returning a plausible wrong operator."""
    with pytest.raises(ValueError, match="square grid"):
        FDMAssembler(None, (5, 5), spacing=(0.1, 0.2), scheme="compact")
    with pytest.raises(NotImplementedError, match="two-dimensional"):
        FDMAssembler(None, (5,), scheme="compact")


def test_the_correction_belongs_to_the_compact_scheme_only():
    a = FDMAssembler(None, (5, 5), spacing=0.1, order=2)
    with pytest.raises(ValueError, match="Mehrstellen"):
        a.assemble(mehrstellen=True)


# ---------------------------------------------------------------------------
# Grid bookkeeping
# ---------------------------------------------------------------------------

def test_the_grid_holds_interior_points_only():
    """``grid_shape`` counts unknowns; the boundaries carry data, not degrees of
    freedom, and sit half a stencil beyond the first and last point."""
    a = FDMAssembler(None, (4,), spacing=0.2, origin=1.0)
    np.testing.assert_allclose(a.axis(0), [1.2, 1.4, 1.6, 1.8])
    assert a.n_unknowns == 4


def test_coordinates_are_in_solve_order():
    a = FDMAssembler(None, (2, 3), spacing=(1.0, 10.0))
    points = a.coordinates()
    assert points.shape == (6, 2)
    np.testing.assert_allclose(points[:, 1][:3], [10.0, 20.0, 30.0])
    assert a.recover_fields(np.arange(6))["primary"].shape == (2, 3)


def test_an_unknown_scheme_or_dimension_is_refused():
    with pytest.raises(ValueError, match="unknown scheme"):
        FDMAssembler(None, (4, 4), scheme="upwind")
    with pytest.raises(NotImplementedError, match="3-D"):
        FDMAssembler(None, (4, 4, 4))
