"""
The thick-walled cylinder, set up for shakedown.

Shared by the tests because it is the one problem here with a closed-form
answer: under a load domain with a single vertex, shakedown degenerates to
plastic collapse, and the limit pressure of a von Mises cylinder in plane strain
is ``2 Y / sqrt(3) * ln(b/a)`` with no elastic constant in it. The same geometry
and the same number are already used by ``examples/plasticity/thick_cylinder``,
where an incremental elastoplastic analysis reaches it to 0.09% -- so the two
routes to it are independent.
"""

from __future__ import annotations

import numpy as np

from simulacrum.core.domain import Domain
from simulacrum.discretization.fem.tables import QuadratureTables
from simulacrum.materials.linear_elastic import isotropic_stiffness_mandel
from simulacrum.mesh.mesh import ElementBlock, Mesh
from simulacrum.shakedown import ElasticEnvelope, SelfEquilibrium, solve_elastic

A_IN, B_OUT = 100.0, 200.0
E, NU, YIELD = 210.0e3, 0.3, 250.0

#: ``p_lim = 2 Y / sqrt(3) * ln(b/a)``, independent of E and nu.
LIMIT_PRESSURE = 2.0 * YIELD / np.sqrt(3.0) * np.log(B_OUT / A_IN)

C_MANDEL = isotropic_stiffness_mandel(E, NU)


def quarter_annulus(n_r: int, n_theta: int) -> Domain:
    """Graded quad4 mesh of a quarter annulus, biased towards the bore."""
    t = np.linspace(0.0, 1.0, n_r + 1) ** 1.3
    r = A_IN + (B_OUT - A_IN) * t
    th = np.linspace(0.0, 0.5 * np.pi, n_theta + 1)
    nodes = np.array([[ri * np.cos(tj), ri * np.sin(tj)]
                      for tj in th for ri in r])
    conn = np.array([[j * (n_r + 1) + i,
                      j * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i]
                     for j in range(n_theta) for i in range(n_r)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)


def bore_pressure_forces(domain, dof_map) -> np.ndarray:
    """Consistent nodal forces for unit internal pressure on the bore."""
    nodes = domain.coordinates
    radius = np.hypot(nodes[:, 0], nodes[:, 1])
    bore = np.where(np.abs(radius - A_IN) < 1e-9 * B_OUT)[0]
    bore = bore[np.argsort(np.arctan2(nodes[bore, 1], nodes[bore, 0]))]

    f = np.zeros(dof_map.n_total)
    for a, b in zip(bore[:-1], bore[1:]):
        xa, xb = nodes[a], nodes[b]
        length = np.linalg.norm(xb - xa)
        normal = xa / np.linalg.norm(xa) + xb / np.linalg.norm(xb)
        normal /= np.linalg.norm(normal)
        for n in (a, b):
            f[dof_map.dof(int(n), 0)] += 0.5 * length * normal[0]
            f[dof_map.dof(int(n), 1)] += 0.5 * length * normal[1]
    return f


def build_cylinder(n_r: int = 12, n_theta: int = 12):
    """``(tables, equilibrium, envelope)`` for unit bore pressure.

    Symmetry conditions on the two straight edges; the bore is loaded by
    traction, so its degrees of freedom stay free.
    """
    domain = quarter_annulus(n_r, n_theta)
    tables = QuadratureTables(domain, thickness=1.0)
    dm = tables.vector_dofs
    nodes = domain.coordinates

    on_x_axis = np.where(nodes[:, 1] < 1e-9 * B_OUT)[0]
    on_y_axis = np.where(nodes[:, 0] < 1e-9 * B_OUT)[0]
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x_axis)[1::2],
                                      dm.element_dofs(on_y_axis)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)

    f = bore_pressure_forces(domain, dm)
    u, = solve_elastic(tables, C_MANDEL, free, [f])

    equilibrium = SelfEquilibrium(tables, free)
    envelope = ElasticEnvelope(tables, C_MANDEL).add_case(u)
    return tables, equilibrium, envelope
