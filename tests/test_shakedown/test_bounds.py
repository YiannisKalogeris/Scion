"""
Verification for the two formulations that really do bound, and for the
selective integration one of them needs.

The thick cylinder is the anchor again, and the point of this file is the
*direction* of each error:

* :mod:`~scion.shakedown.equilibrium_elements` bounds from **below**, and
  rigorously -- stress is the unknown, equilibrium is exact inside each element,
  traction is continuous across every edge, and yield holds at every point by
  convexity. It must never exceed the analytic limit load.
* :mod:`~scion.shakedown.lmm` bounds from **above**, and also rigorously,
  because any admissible mechanism gives a Koiter bound and a finite element
  displacement field is one. It must never fall below.

Together they bracket. That is what neither the Melan implementation nor either
of them alone can claim.

python -m pytest tests/test_shakedown/test_bounds.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.domain import Domain
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.linear_elastic import isotropic_stiffness_mandel
from scion.materials.notation import mandel_to_voigt_tangent
from scion.mesh.generators import rectangle_mesh
from scion.mesh.mesh import ElementBlock, Mesh
from scion.shakedown import (
    ElasticEnvelope,
    EquilibriumTriangles,
    LoadDomain,
    linear_matching,
    lower_bound_limit,
    solve_elastic,
)

from .cylinder import (
    A_IN,
    B_OUT,
    C_MANDEL,
    LIMIT_PRESSURE,
    YIELD,
    bore_pressure_forces,
    quarter_annulus,
)


# ---------------------------------------------------------------------------
# Selective integration
# ---------------------------------------------------------------------------

def _cantilever_tip(nu, bbar, nx=24, ny=3):
    from scipy.sparse import linalg as spla
    L, H, E, P = 10.0, 1.0, 1000.0, 1.0
    domain = Domain(rectangle_mesh((0.0, L), (0.0, H), nx, ny), dim=2)
    tables = QuadratureTables(domain, bbar=bbar)
    dm = tables.vector_dofs
    nv, n_dof = tables.n_voigt, dm.n_total

    C = mandel_to_voigt_tangent(isotropic_stiffness_mandel(E, nu)[None], 2)[0]
    K, _ = tables.stiffness_and_force(
        np.zeros((tables.n_points, nv)),
        np.broadcast_to(C, (tables.n_points, nv, nv)), n_dof)

    nodes = domain.coordinates
    left = np.where(nodes[:, 0] < 1e-9)[0]
    right = np.where(nodes[:, 0] > L - 1e-9)[0]
    fixed = dm.element_dofs(left)
    free = np.setdiff1d(np.arange(n_dof), fixed)
    f = np.zeros(n_dof)
    for n in right:
        f[dm.dof(int(n), 1)] = -P / len(right)

    u = np.zeros(n_dof)
    u[free] = spla.spsolve(K.tocsr()[free][:, free].tocsc(), f[free])
    return abs(u[1::2]).max()


def test_full_integration_locks_as_the_material_becomes_incompressible():
    """The failure B-bar exists to prevent. A bilinear quadrilateral asked to
    deform at constant volume at four points has nowhere to go and reports a
    stiffness an order of magnitude too high."""
    soft = _cantilever_tip(0.3, bbar=False)
    locked = _cantilever_tip(0.4999, bbar=False)
    assert locked / soft < 0.05


def test_bbar_removes_the_locking_and_lands_on_the_right_answer():
    """Plane-strain bending stiffness scales as ``1/(1 - nu^2)``, so the tip
    deflection ratio between two Poisson ratios is known in closed form. B-bar
    has to reproduce it, not merely avoid collapsing."""
    # deflection scales as (1 - nu^2), the inverse of the stiffness. The ratio
    # carries the usual discretisation error through the depth -- 3% at two
    # elements, 0.15% at eight -- so it is checked on a mesh fine enough to
    # separate that from the locking this is testing for.
    expected = (1.0 - 0.4999 ** 2) / (1.0 - 0.3 ** 2)
    ratios = [_cantilever_tip(0.4999, bbar=True, ny=ny)
              / _cantilever_tip(0.3, bbar=True, ny=ny) for ny in (2, 4, 8)]

    assert ratios[-1] == pytest.approx(expected, rel=0.005)
    assert abs(ratios[-1] - expected) < abs(ratios[0] - expected)


def test_bbar_makes_the_dilatation_constant_over_the_element():
    """The mechanism: one volumetric constraint per element instead of one per
    quadrature point."""
    domain = Domain(rectangle_mesh((0.0, 4.0), (0.0, 1.0), 3, 2), dim=2)
    plain = QuadratureTables(domain, bbar=False)
    barred = QuadratureTables(domain, bbar=True)

    m = np.array([1.0, 1.0, 0.0])
    for tables, constant in ((plain, False), (barred, True)):
        dilatation = np.einsum("a,eqad->eqd", m, tables.B)
        spread = np.abs(dilatation - dilatation.mean(axis=1, keepdims=True)).max()
        # bool(): a numpy scalar comparison is np.False_, which is not the
        # Python singleton and fails an `is` test
        assert bool(spread < 1e-12) is constant


# ---------------------------------------------------------------------------
# Linear matching: the upper bound
# ---------------------------------------------------------------------------

def _lmm(n, domain=None):
    dom = quarter_annulus(n, n)
    tables = QuadratureTables(dom, thickness=1.0, bbar=True)
    dm = tables.vector_dofs
    nodes = dom.coordinates
    on_x = np.where(nodes[:, 1] < 1e-9 * B_OUT)[0]
    on_y = np.where(nodes[:, 0] < 1e-9 * B_OUT)[0]
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x)[1::2],
                                      dm.element_dofs(on_y)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)
    u, = solve_elastic(tables, C_MANDEL, free,
                       [bore_pressure_forces(dom, dm)])
    envelope = ElasticEnvelope(tables, C_MANDEL).add_case(u)
    vertex = envelope.vertex_stresses(domain or LoadDomain.fixed(1.0))
    return linear_matching(vertex, tables, free, YIELD, max_iter=200,
                           tol=1e-10), tables, free


def test_the_upper_bound_stays_above_and_converges_down_to_it():
    """Above is mandatory -- an upper bound below the true value is wrong --
    and it must tighten with the mesh."""
    factors = []
    for n in (8, 12, 16):
        result, _, _ = _lmm(n)
        assert result.factor > LIMIT_PRESSURE, f"mesh {n}"
        factors.append(result.factor)

    assert np.all(np.diff(factors) < 0.0)
    assert factors[-1] < 1.002 * LIMIT_PRESSURE


def test_the_sequence_of_bounds_decreases_monotonically():
    """Ponter's convergence result. A rise means the strain floor is active
    where it matters, and the bound stops being trustworthy."""
    result, _, _ = _lmm(12)
    assert result.monotone
    assert result.converged
    assert np.all(np.diff(result.history) <= 1e-9 * result.factor)


def test_the_residual_field_is_self_equilibrated():
    """It is the field the iteration converged onto, and the reason it is worth
    exporting is that it can seed a lower-bound basis -- which requires it to
    be admissible, not merely small.

    The subtlety this catches: the linear solve makes ``C:eps(du) - sigma_init``
    equilibrated, *including* the penalty pressure. Exporting only the
    deviatoric part leaves a field out of equilibrium by order one.
    """
    from scion.shakedown import SelfEquilibrium
    result, tables, free = _lmm(10)
    equilibrium = SelfEquilibrium(tables, free)
    assert equilibrium.residual(result.residual_stress) < 1e-9


def test_locking_tables_are_refused_unless_asked_for():
    """The matched material is exactly incompressible, so a locked mesh gives a
    bound that is too high while still converging monotonically and looking
    healthy. Refusing is better than warning."""
    dom = quarter_annulus(6, 6)
    tables = QuadratureTables(dom, thickness=1.0, bbar=False)
    free = np.arange(tables.vector_dofs.n_total)
    vertex = np.zeros((1, tables.n_points, 6))
    with pytest.raises(ValueError, match="incompressible"):
        linear_matching(vertex, tables, free, YIELD)


# ---------------------------------------------------------------------------
# Equilibrium elements: the rigorous lower bound
# ---------------------------------------------------------------------------

def _annulus_triangles(n_r, n_theta):
    r = A_IN + (B_OUT - A_IN) * np.linspace(0.0, 1.0, n_r + 1) ** 1.2
    th = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    nodes = np.array([[ri * np.cos(tj), ri * np.sin(tj)]
                      for tj in th for ri in r])
    tris = []
    for j in range(n_theta):
        for i in range(n_r):
            a = j * (n_r + 1) + i
            b, c, d = a + 1, (j + 1) * (n_r + 1) + i + 1, (j + 1) * (n_r + 1) + i
            tris += [[a, b, c], [a, c, d]]
    return Domain(Mesh(nodes, [ElementBlock("tri3", np.array(tris))]), dim=2)


def _annulus_bcs(elements, domain):
    nodes = domain.coordinates
    radius = np.hypot(nodes[:, 0], nodes[:, 1])
    tol = 1e-8 * B_OUT
    conditions = {}
    for (a, b), _ in elements.boundary:
        # classified by the nodes: on a curved boundary the midpoint of a
        # straight chord sits well inside the true radius, and classifying by
        # it silently turns the free outer surface into a symmetry plane --
        # which leaves the hydrostatic stress unconstrained and the linear
        # program unbounded
        if abs(radius[a] - A_IN) < tol and abs(radius[b] - A_IN) < tol:
            mid = 0.5 * (nodes[a] + nodes[b])
            n = mid / np.hypot(*mid)
            conditions[(a, b)] = ("load", n[0], n[1])
        elif abs(radius[a] - B_OUT) < tol and abs(radius[b] - B_OUT) < tol:
            conditions[(a, b)] = ("free",)
        else:
            conditions[(a, b)] = ("symmetry",)
    return conditions


def _sloan(n, n_facets=24):
    domain = _annulus_triangles(n, n)
    elements = EquilibriumTriangles(domain)
    conditions = _annulus_bcs(elements, domain)
    return lower_bound_limit(elements, YIELD, conditions,
                             n_facets=n_facets), elements, conditions


def test_the_rigorous_lower_bound_never_exceeds_the_limit_load():
    """The property the whole element family exists for, and the one the
    quadrature-point formulation cannot deliver."""
    for n in (4, 6, 8):
        result, _, _ = _sloan(n)
        assert result.factor < LIMIT_PRESSURE, f"mesh {n}"


def test_it_converges_upwards_under_refinement():
    factors = [_sloan(n)[0].factor for n in (4, 6, 8, 10)]
    assert np.all(np.diff(factors) > 0.0)
    assert factors[-1] > 0.8 * LIMIT_PRESSURE


def test_the_returned_field_is_admissible_against_the_exact_yield_surface():
    """Solved against an inscribed polygon, checked against the true disk. If
    the polygon were circumscribed instead the bound would be a fraction of a
    percent better and would stop being a bound, so the check has to use the
    circle."""
    result, elements, conditions = _sloan(6)
    report = result.verify(elements, YIELD, conditions)
    assert report["admissible"]
    assert report["equilibrium_residual"] < 1e-6
    assert report["max_yield_violation"] <= 0.0


def test_more_facets_tighten_the_bound_by_the_predicted_factor():
    """An inscribed regular polygon keeps ``cos(pi/n)`` of the disk, so dividing
    the bound by that should give the same number for any facet count. It does,
    to three digits -- which says the remaining gap is discretisation and not
    linearisation."""
    corrected = []
    for n_facets in (12, 24, 48):
        result, _, _ = _sloan(6, n_facets)
        assert result.facet_efficiency == pytest.approx(
            np.cos(np.pi / n_facets))
        corrected.append(result.factor / result.facet_efficiency)

    spread = (max(corrected) - min(corrected)) / np.mean(corrected)
    assert spread < 0.01


def test_a_coarser_polygon_is_more_conservative_never_less():
    coarse, _, _ = _sloan(6, n_facets=8)
    fine, _, _ = _sloan(6, n_facets=32)
    assert coarse.factor < fine.factor < LIMIT_PRESSURE


def test_a_missing_boundary_condition_is_refused():
    """An unconstrained boundary is a free traction the optimiser will invent,
    and the symptom is an unbounded program rather than a wrong number."""
    domain = _annulus_triangles(4, 4)
    elements = EquilibriumTriangles(domain)
    conditions = _annulus_bcs(elements, domain)
    conditions.pop(next(iter(conditions)))
    with pytest.raises(KeyError, match="no condition"):
        lower_bound_limit(elements, YIELD, conditions)


def test_only_triangles_are_accepted():
    domain = Domain(rectangle_mesh((0.0, 1.0), (0.0, 1.0), 2, 2), dim=2)
    with pytest.raises(NotImplementedError, match="tri3"):
        EquilibriumTriangles(domain)


# ---------------------------------------------------------------------------
# The two together
# ---------------------------------------------------------------------------

def test_the_two_rigorous_methods_bracket_the_answer():
    """The point of having both. One is guaranteed below and one guaranteed
    above, on the same problem, so the true collapse load is trapped between
    them without either needing to be trusted on its own.
    """
    lower, _, _ = _sloan(8)
    upper, _, _ = _lmm(12)

    assert lower.factor < LIMIT_PRESSURE < upper.factor
    assert lower.factor < upper.factor
