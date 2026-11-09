r"""
Square plate with a central hole: the model of Example 2.

A quarter of the plate by symmetry, plane strain, von Mises,
elastic--perfectly plastic, with two independent edge tractions. The yield
stress is a field, which is the point: everything else in this package carries
a scalar.

Why plane strain and not plane stress
-------------------------------------
The whole shakedown package is built on the three-dimensional Mandel
stiffness, which in two dimensions is plane *strain* -- ``eps_33 = 0``, with
``sigma_33`` carrying the out-of-plane constraint and entering ``J2``. Plane
stress is not reachable by substituting another ``C``: the condition
``sigma_33 = 0`` leaves the ``33`` direction a null direction of the
constitutive matrix, so the compliance the plastic strain is computed from
does not exist, and plane-stress plasticity needs a nested return map for
``eps_33``, which this library refuses deliberately. A thick plate is
therefore the problem solved here. The consequence worth stating is that
published values for this benchmark are nearly all plane stress and are *not*
a like-for-like comparison; the verification below is against the model's own
elastic solution and against a second direct method on the same mesh.

What the mesh is checked against
--------------------------------
Kirsch's solution, which is external to this library and closed form. At the
innermost ring of quadrature points the uniaxial peak converges to 3.0 and the
equibiaxial hoop to 2.0 from below, as the ring approaches the hole:

    n          4       8      12      16      24      32
    uniaxial 2.130   2.738   2.903   2.972   3.029   3.051
    hoop     1.846   1.982   2.002   2.009   2.014   2.016

The quadrature points sit *inside* the elements, so a coarse mesh reads low
rather than wrong, and the area is exact to 5e-5 at ``n = 8``.

Two cost reductions, both measured rather than assumed
------------------------------------------------------
**The cycle needs only the vertices.** Melan's condition sees the extremes of
the load domain and nothing between them, so refining the path through the
domain cannot move the shakedown factor. Measured, on this plate:
``lambda_sd`` is 2.543732 at ``per_leg`` = 1, 2 and 4 -- identical to six
decimals -- and 2.541102 at 8, the last being the bisection's own resolution.
``per_leg = 1`` is therefore used throughout, at 5 cycle points instead of 40.

**The ratio to the elastic limit is mesh-converged where the factor is not.**
``lambda_sd`` itself falls with refinement because the elastic peak at the hole
is still sharpening (Kirsch above), but ``lambda_sd/lambda_el`` settles:
1.4805 / 1.5117 / 1.5102 at ``n`` = 6 / 8 / 12. So the bisection can be
bracketed tightly off the elastic factor, which costs nothing to compute.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import linalg as spla

from scion.core.domain import Domain
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.linear_elastic import isotropic_stiffness_mandel
from scion.materials.notation import mandel_to_voigt_tangent
from scion.mesh.mesh import ElementBlock, Mesh
from scion.shakedown import CyclicProgram, ElasticEnvelope, LoadDomain

__all__ = ["DOMAINS", "E_REF", "L_HALF", "NU", "R_HOLE", "TRACTION", "YIELD",
           "build", "hoop", "quarter_plate"]

L_HALF, R_HOLE = 1.0, 0.1           # hole diameter is 0.2 of the full side
E_REF, NU, YIELD = 200.0e3, 0.3, 355.0
TRACTION = 100.0                    # reference amplitude of each edge traction

#: The load domains. ``box`` is the two tractions varying independently and
#: is Example 2's stated case; ``bree`` holds the first constant and reverses
#: the second, which is the Bree structure and the one with a non-empty
#: alternating band.
DOMAINS = {
    "box": LoadDomain.box([0.0, 0.0], [1.0, 1.0]),
    "bree": LoadDomain.box([1.0, -1.0], [1.0, 1.0]),
}


def quarter_plate(n_r: int = 8, n_t: int = 8, grade: float = 1.6) -> Domain:
    """Transfinite map from the hole quadrant to the square corner.

    Graded towards the hole, where the stress concentration and therefore the
    plastic zone live. ``grade = 1`` is the ungraded map.
    """
    th = np.linspace(0.0, 0.5 * np.pi, n_t + 1)
    c, s = np.cos(th), np.sin(th)
    inner = np.stack([R_HOLE * c, R_HOLE * s], axis=1)
    # along each ray, the point on whichever edge of the square it reaches
    outer = (L_HALF / np.maximum(c, s))[:, None] * np.stack([c, s], axis=1)
    t = np.linspace(0.0, 1.0, n_r + 1) ** grade
    nodes = np.array([(1.0 - ti) * inner[j] + ti * outer[j]
                      for j in range(n_t + 1) for ti in t])
    conn = np.array([[j * (n_r + 1) + i, j * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i + 1, (j + 1) * (n_r + 1) + i]
                     for j in range(n_t) for i in range(n_r)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)


def _edge_forces(domain, dof_map, which: int) -> np.ndarray:
    """Consistent nodal forces for a unit traction on one outer edge.

    ``which`` is 0 for the ``x = L`` edge, loaded in ``+x``, and 1 for
    ``y = L`` in ``+y``.
    """
    nodes = domain.coordinates
    on = np.where(np.abs(nodes[:, which] - L_HALF) < 1e-9 * L_HALF)[0]
    other = 1 - which
    on = on[np.argsort(nodes[on, other])]
    f = np.zeros(dof_map.n_total)
    for p, q in zip(on[:-1], on[1:]):
        length = abs(nodes[q, other] - nodes[p, other])
        for node in (p, q):
            f[dof_map.dof(int(node), which)] += 0.5 * length
    return f


def build(n_r: int = 8, n_t: int = 8, bbar: bool = True,
          per_leg: int = 1) -> dict:
    """The plate, its two elastic load cases and one cycle per load domain.

    ``bbar`` defaults to True because the constrained cyclic solve is
    displacement-based with exactly incompressible plastic flow and locks
    otherwise -- by 17% on the thick cylinder, with nothing in the run looking
    wrong. Pass False for a Melan or RSDM comparison, which must *not* have
    it.
    """
    domain = quarter_plate(n_r, n_t)
    tables = QuadratureTables(domain, thickness=1.0, bbar=bbar)
    dm, nodes = tables.vector_dofs, domain.coordinates
    on_x = np.where(np.abs(nodes[:, 1]) < 1e-9 * L_HALF)[0]      # u_y = 0
    on_y = np.where(np.abs(nodes[:, 0]) < 1e-9 * L_HALF)[0]      # u_x = 0
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x)[1::2],
                                      dm.element_dofs(on_y)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)

    C = isotropic_stiffness_mandel(E_REF, NU)
    nv, nd = tables.n_voigt, dm.n_total
    K, _ = tables.stiffness_and_force(
        np.zeros((tables.n_points, nv)),
        np.broadcast_to(mandel_to_voigt_tangent(C[None], 2)[0],
                        (tables.n_points, nv, nv)), nd)
    solve = spla.factorized(K.tocsr()[free][:, free].tocsc())

    envelope = ElasticEnvelope(tables, C)
    for which in (0, 1):
        u = np.zeros(nd)
        u[free] = solve(_edge_forces(domain, dm, which)[free])
        envelope.add_case(TRACTION * u)

    xq = np.stack([tables.interpolate(nodes[:, 0]),
                   tables.interpolate(nodes[:, 1])], axis=1)
    history, vertices = {}, {}
    for name, dom in DOMAINS.items():
        history[name] = CyclicProgram.through_vertices(
            dom, per_leg=per_leg).elastic_history(envelope)
        vertices[name] = envelope.vertex_stresses(dom)
    return dict(domain=domain, tables=tables, free=free, C=C,
                envelope=envelope, history=history, vertices=vertices,
                xq=xq, weights=np.asarray(tables.w, dtype=float).ravel())


def hoop(stress_field: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """The hoop component about the hole, for the Kirsch comparison."""
    th = np.arctan2(xq[:, 1], xq[:, 0])
    t = np.stack([-np.sin(th), np.cos(th)], axis=1)
    sxy = stress_field[:, 5] / np.sqrt(2.0)
    return (t[:, 0] ** 2 * stress_field[:, 0]
            + t[:, 1] ** 2 * stress_field[:, 1]
            + 2.0 * t[:, 0] * t[:, 1] * sxy)
