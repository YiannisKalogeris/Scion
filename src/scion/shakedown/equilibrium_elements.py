r"""
Sloan-style equilibrium elements: a lower bound that is actually a bound.

Everything else in this package computes stresses at quadrature points from a
displacement field and enforces equilibrium *weakly*, against the displacement
test space. That is enough for an upper bound -- any admissible mechanism gives
one -- and it is not enough for a lower bound. Measured, on the thick cylinder:
with a good enough residual basis the Melan factor comes out at 200.49 against
an analytic limit load of 200.094, having passed every check the formulation
knows how to make. The field is self-equilibrated in the discrete sense and
satisfies yield at every quadrature point; it is simply not a stress state the
continuum admits.

The static theorem needs three things to hold *everywhere*, not on average:

    div sigma + b = 0        inside the body
    [[sigma]] . n = 0        across every internal surface
    f(sigma) <= 0            at every point

This module gets all three, by making stress the unknown instead of deriving it.

The discretisation (Sloan 1988)
-------------------------------
Linear stress over three-noded triangles, with **each element carrying its own
nodal stresses**, so the field is discontinuous between elements. That sounds
like giving up on equilibrium and is the opposite:

* **Inside an element** the stress is linear, so its divergence is constant and
  ``div sigma = 0`` is two equations per element, satisfied exactly rather than
  in the mean.
* **Across an edge** only the *traction* has to be continuous, not the whole
  tensor. Statically admissible stress discontinuities are physical -- they are
  what a slip line is -- and allowing them is what lets a coarse mesh represent
  a collapse mechanism. Traction is linear along the edge, so matching it at the
  two endpoints matches it everywhere.
* **Yield** is enforced at the three nodes. Because the stress is linear and the
  yield set is convex, a linear field whose endpoints lie inside a convex set
  lies inside it throughout the element. Three constraints buy admissibility at
  every point of the element, which is the property the quadrature-point
  formulation can never have.

Plane strain, von Mises, and why the cone becomes a disk
--------------------------------------------------------
``sigma_33`` appears in no in-plane equilibrium equation, so in plane strain it
is a free field -- any value at any point is admissible. A lower bound may
therefore choose it, and choosing it to minimise the von Mises stress gives
``sigma_33 = (sigma_11 + sigma_22)/2`` and

    q^2 = (3/4)(sigma_11 - sigma_22)^2 + 3 sigma_12^2 .

So ``q <= sigma_y`` becomes

    || (sigma_11 - sigma_22,  2 sigma_12) ||  <=  2 sigma_y / sqrt(3),

a disk in two variables. That radius is the same ``Y' = 2Y/sqrt(3)`` that appears
in the closed-form thick-cylinder solution, which is not a coincidence: plane
strain von Mises *is* Tresca with that factor. Eliminating ``sigma_33`` this way
loses nothing and removes a variable per node.

Linearising the disk keeps the bound
------------------------------------
There is no cone solver in this stack, so the disk is replaced by an
**inscribed** regular polygon of ``n_facets`` sides, whose supporting lines sit
at ``R cos(pi/n_facets)``. Inscribed, not circumscribed: the feasible set
shrinks, so the bound stays valid and merely becomes conservative, and it
tightens as ``cos(pi/n)`` towards one. Circumscribing would be a fraction of a
percent more accurate and would stop being a bound.

What is left is a linear program, which ``scipy.optimize.linprog`` solves
directly.

Scope
-----
Limit analysis: one load case, scaled. That is the one-vertex shakedown domain
and the case with a closed-form answer, so it is what the implementation can be
checked against.

Full shakedown needs ``lambda sigma^E(t) + rho`` to be statically admissible,
which needs ``sigma^E`` in *this* space rather than in the displacement one --
a complementary-energy elastic solve. Not implemented, and flagged rather than
faked: mixing a displacement-based ``sigma^E`` into an equilibrium-element
residual field would quietly give back the non-rigorous bound this module exists
to replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

__all__ = ["EquilibriumTriangles", "LimitAnalysisResult", "lower_bound_limit"]


def _triangle_gradients(coords: np.ndarray):
    """``(dN/dx, dN/dy)`` and the area of a 3-noded triangle, both constant."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * abs(det)
    dNdx = np.array([y2 - y3, y3 - y1, y1 - y2]) / det
    dNdy = np.array([x3 - x2, x1 - x3, x2 - x1]) / det
    return dNdx, dNdy, area


class EquilibriumTriangles:
    """Linear, element-discontinuous stress field on a triangular mesh.

    Unknowns are ``(sigma_11, sigma_22, sigma_12)`` at each of the three nodes
    of each element, numbered ``9 * element + 3 * local_node + component``, with
    the load multiplier last. ``sigma_33`` is eliminated analytically; see the
    module docstring.
    """

    def __init__(self, domain):
        blocks = domain.mesh.element_blocks
        if len(blocks) != 1 or blocks[0].etype != "tri3":
            raise NotImplementedError(
                "equilibrium elements are implemented for a single tri3 block; "
                "linear stress on a triangle is what makes the divergence "
                "constant and the yield check exact at three points")
        self.domain = domain
        self.conn = np.asarray(blocks[0].connectivity, dtype=np.int64)
        self.coords = domain.coordinates
        self.n_elements = self.conn.shape[0]
        self.n_stress = 9 * self.n_elements
        self.n_var = self.n_stress + 1
        self.lam = self.n_stress                      # index of the multiplier

        self._geometry()
        self._edges()

    # ------------------------------------------------------------------
    def _geometry(self) -> None:
        self.dNdx = np.empty((self.n_elements, 3))
        self.dNdy = np.empty((self.n_elements, 3))
        self.area = np.empty(self.n_elements)
        for e, nodes in enumerate(self.conn):
            self.dNdx[e], self.dNdy[e], self.area[e] = _triangle_gradients(
                self.coords[nodes])

    def _edges(self) -> None:
        """Interior edges with their two elements, and the boundary edges."""
        seen: dict[tuple[int, int], list] = {}
        for e, nodes in enumerate(self.conn):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((int(nodes[a]), int(nodes[b]))))
                seen.setdefault(key, []).append((e, a, b))
        self.interior = [(k, v[0], v[1]) for k, v in seen.items() if len(v) == 2]
        self.boundary = [(k, v[0]) for k, v in seen.items() if len(v) == 1]

    def var(self, element: int, local: int, component: int) -> int:
        return 9 * element + 3 * local + component

    def local_index(self, element: int, node: int) -> int:
        return int(np.flatnonzero(self.conn[element] == node)[0])

    def edge_normal(self, node_a: int, node_b: int) -> np.ndarray:
        d = self.coords[node_b] - self.coords[node_a]
        n = np.array([d[1], -d[0]])
        return n / np.linalg.norm(n)

    # ------------------------------------------------------------------
    # Constraint rows
    # ------------------------------------------------------------------
    def _traction_rows(self, element: int, node: int, normal: np.ndarray):
        """The two rows giving ``sigma . n`` at one node of one element."""
        i = self.local_index(element, node)
        nx, ny = normal
        rows = ([(self.var(element, i, 0), nx), (self.var(element, i, 2), ny)],
                [(self.var(element, i, 2), nx), (self.var(element, i, 1), ny)])
        return rows

    def equality_rows(self, boundary_conditions):
        """Assemble ``A_eq x = b_eq``: element equilibrium, traction continuity
        and the stress boundary conditions.

        ``boundary_conditions`` maps a boundary edge key ``(node_a, node_b)`` to
        either ``("free",)``, ``("symmetry",)`` or ``("load", tx, ty)`` where the
        traction is ``lambda * (tx, ty)`` per unit multiplier, expressed in the
        global frame at that edge.

        A symmetry edge constrains only the *shear* traction to vanish: the
        normal traction there is a reaction and must stay free, exactly as a
        constrained degree of freedom is exempt from the weak form elsewhere in
        this package.
        """
        rows, cols, vals, rhs = [], [], [], []

        def add(entries, value):
            r = len(rhs)
            for c, v in entries:
                rows.append(r)
                cols.append(c)
                vals.append(v)
            rhs.append(value)

        # 1. div sigma = 0 inside each element -- exact, since the stress is
        #    linear and its divergence therefore constant
        for e in range(self.n_elements):
            gx, gy = self.dNdx[e], self.dNdy[e]
            add([(self.var(e, i, 0), gx[i]) for i in range(3)]
                + [(self.var(e, i, 2), gy[i]) for i in range(3)], 0.0)
            add([(self.var(e, i, 2), gx[i]) for i in range(3)]
                + [(self.var(e, i, 1), gy[i]) for i in range(3)], 0.0)

        # 2. traction continuity across every interior edge, at both endpoints
        for (na, nb), (ea, _, _), (eb, _, _) in self.interior:
            normal = self.edge_normal(na, nb)
            for node in (na, nb):
                left = self._traction_rows(ea, node, normal)
                right = self._traction_rows(eb, node, normal)
                for lo, ro in zip(left, right):
                    add(lo + [(c, -v) for c, v in ro], 0.0)

        # 3. stress boundary conditions
        for (na, nb), (e, _, _) in self.boundary:
            key = (na, nb)
            spec = boundary_conditions.get(key)
            if spec is None:
                raise KeyError(
                    f"boundary edge {key} has no condition; every boundary edge "
                    f"needs one, because an unconstrained boundary is a free "
                    f"traction the optimiser will happily invent")
            normal = self.edge_normal(na, nb)
            # point the normal out of the element
            centroid = self.coords[self.conn[e]].mean(axis=0)
            midpoint = 0.5 * (self.coords[na] + self.coords[nb])
            if normal @ (midpoint - centroid) < 0.0:
                normal = -normal

            for node in (na, nb):
                tx_row, ty_row = self._traction_rows(e, node, normal)
                if spec[0] == "free":
                    add(tx_row, 0.0)
                    add(ty_row, 0.0)
                elif spec[0] == "load":
                    add(tx_row + [(self.lam, -float(spec[1]))], 0.0)
                    add(ty_row + [(self.lam, -float(spec[2]))], 0.0)
                elif spec[0] == "symmetry":
                    tangent = np.array([-normal[1], normal[0]])
                    combined = [(c, v * tangent[0]) for c, v in tx_row]
                    combined += [(c, v * tangent[1]) for c, v in ty_row]
                    add(combined, 0.0)
                else:
                    raise ValueError(f"unknown boundary condition {spec[0]!r}")

        A = sparse.csr_matrix((vals, (rows, cols)),
                              shape=(len(rhs), self.n_var))
        return A, np.array(rhs)

    def yield_rows(self, yield_stress, n_facets: int = 24):
        """``A_ub x <= b_ub``: the linearised plane-strain von Mises disk.

        Enforced at every nodal stress point. Convexity does the rest -- a
        linear field with all three nodes inside a convex set is inside it
        everywhere, which is the whole reason this discretisation gives a bound.
        """
        if n_facets < 3:
            raise ValueError("need at least three facets to bound the disk")
        radius = 2.0 * np.asarray(yield_stress, dtype=float) / np.sqrt(3.0)
        radius = np.broadcast_to(radius, (self.n_elements,))
        inscribed = float(np.cos(np.pi / n_facets))

        angles = 2.0 * np.pi * np.arange(n_facets) / n_facets
        rows, cols, vals, rhs = [], [], [], []
        for e in range(self.n_elements):
            for i in range(3):
                a = self.var(e, i, 0)
                b = self.var(e, i, 1)
                s = self.var(e, i, 2)
                for theta in angles:
                    c, sn = np.cos(theta), np.sin(theta)
                    r = len(rhs)
                    rows += [r, r, r]
                    cols += [a, b, s]
                    vals += [c, -c, 2.0 * sn]
                    rhs.append(radius[e] * inscribed)

        A = sparse.csr_matrix((vals, (rows, cols)),
                              shape=(len(rhs), self.n_var))
        return A, np.array(rhs)

    def __repr__(self) -> str:
        return (f"EquilibriumTriangles({self.n_elements} tri3, "
                f"{self.n_var} unknowns, {len(self.interior)} interior edges, "
                f"{len(self.boundary)} boundary edges)")


# ---------------------------------------------------------------------------

@dataclass
class LimitAnalysisResult:
    """A rigorous lower bound on the collapse multiplier."""

    factor: float
    stresses: np.ndarray = field(repr=False)      # (n_elements, 3, 3) nodal
    n_facets: int = 0
    status: str = ""
    converged: bool = True

    @property
    def facet_efficiency(self) -> float:
        """``cos(pi/n)``: the fraction of the true yield disk the polygon keeps.

        The bound is conservative by at most this factor from the linearisation
        alone, so dividing by it gives the value an exact cone solve would
        approach from below.
        """
        return float(np.cos(np.pi / self.n_facets)) if self.n_facets else 1.0

    def verify(self, elements: "EquilibriumTriangles", yield_stress,
               boundary_conditions) -> dict:
        """Recheck admissibility from the returned stresses alone.

        The three conditions of the static theorem, recomputed rather than
        trusted: element equilibrium, traction continuity and the boundary
        conditions come back as one residual, and the yield check is the *exact*
        disk rather than the polygon that was solved with.
        """
        x = np.concatenate([self.stresses.ravel(), [self.factor]])
        A, b = elements.equality_rows(boundary_conditions)
        residual = float(np.abs(A @ x - b).max())

        s = self.stresses
        q = np.sqrt((s[:, :, 0] - s[:, :, 1]) ** 2 + 4.0 * s[:, :, 2] ** 2)
        radius = 2.0 * float(np.max(yield_stress)) / np.sqrt(3.0)
        worst = float(q.max() - radius)
        return {
            "equilibrium_residual": residual,
            "max_yield_violation": worst,
            "admissible": bool(residual < 1e-6 * max(self.factor, 1.0)
                               and worst <= 1e-8 * radius),
        }

    def __repr__(self) -> str:
        return (f"LimitAnalysisResult(factor={self.factor:.6g}, "
                f"facets={self.n_facets}, "
                f"efficiency={self.facet_efficiency:.5f}, "
                f"converged={self.converged})")


def lower_bound_limit(elements: EquilibriumTriangles, yield_stress,
                      boundary_conditions, n_facets: int = 24,
                      verbose: bool = False) -> LimitAnalysisResult:
    """Maximise the load multiplier over statically admissible stress fields.

    A linear program: maximise ``lambda`` subject to exact element equilibrium,
    traction continuity, the stress boundary conditions, and the linearised
    yield condition at every nodal stress point.

    The answer is a lower bound on the true collapse multiplier of the
    *continuum*, not merely of a discretisation. Every approximation here --
    a coarse mesh, fewer facets -- restricts the set of admissible fields and
    therefore lowers the answer.
    """
    A_eq, b_eq = elements.equality_rows(boundary_conditions)
    A_ub, b_ub = elements.yield_rows(yield_stress, n_facets)

    c = np.zeros(elements.n_var)
    c[elements.lam] = -1.0                       # maximise lambda
    bounds = [(None, None)] * elements.n_stress + [(0.0, None)]

    out = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if verbose:
        print(f"    linprog: {out.message} ({A_eq.shape[0]} equalities, "
              f"{A_ub.shape[0]} inequalities, {elements.n_var} unknowns)")
    if not out.success:
        raise RuntimeError(
            f"the lower-bound linear program failed: {out.message}. An "
            f"infeasible problem usually means a boundary condition is missing "
            f"or over-specified; an unbounded one means the structure is not "
            f"restrained against a rigid-body motion by its traction "
            f"conditions.")

    x = out.x
    return LimitAnalysisResult(
        factor=float(x[elements.lam]),
        stresses=x[:elements.n_stress].reshape(elements.n_elements, 3, 3),
        n_facets=n_facets, status=str(out.message), converged=bool(out.success))
