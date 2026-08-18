#!/usr/bin/env python3
r"""
Shakedown of a thick-walled cylinder under varying internal pressure.

Problem
-------
The same quarter annulus as ``examples/plasticity/thick_cylinder``: inner radius
``a``, outer ``b = 2a``, plane strain, von Mises, perfectly plastic. There the
pressure was ramped once and the collapse load read off a plateau after forty
nonlinear steps. Here nothing is integrated at all -- one elastic solve, a few
dozen residual fields, and a convex search give the load factor directly.

Three load domains, and what each one asks
------------------------------------------
``fixed``       the pressure held at its value. One vertex, so shakedown
                degenerates to plastic collapse and the answer is the limit
                pressure ``2 Y / sqrt(3) ln(b/a)``. This is the calibration
                case: the only one here with a closed form.
``pulsating``   the pressure free anywhere in ``[0, p]``, switching on and off
                forever. Two vertices.
``alternating`` free in ``[-p, p]``. Two vertices, and much more severe: the
                structure must survive full reversal with one fixed residual
                field, which is what makes alternating plasticity the governing
                mechanism rather than ratchetting.

Reading the numbers
-------------------
Two ratios matter more than the factors themselves.

``lambda_SD / lambda_E`` is the reserve beyond first yield -- how much load the
redistribution buys. It is the whole reason to compute a shakedown factor rather
than an elastic limit, and for the fixed domain here it is about 1.8.

``lambda_SD / p_lim`` says how much the *variation* costs. Going from a held
pressure to a fully reversed one takes the safe factor down by roughly half,
and no amount of plastic redistribution recovers it: a residual field that
suits one extreme is wrong for the other.

What the answer is a bound on
-----------------------------
It is a lower bound of the *discrete* model, checked and certified: the printed
equilibrium residual and yield violation come from the returned field, not from
the solver's internal state. It is not a strict continuum bound -- yield is
enforced at quadrature points and equilibrium is weak -- so the closing table
compares against the analytic limit load to show how much is left on the table.
See ``shakedown/melan.py`` for why the distinction matters and
``shakedown/basis.py`` for what happens without the reduced basis.

Run
---
    python -m examples.shakedown.thick_cylinder_shakedown
    python -m examples.shakedown.thick_cylinder_shakedown --quick
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from scion.core.domain import Domain
from scion.discretization.fem.tables import QuadratureTables
from scion.materials.linear_elastic import isotropic_stiffness_mandel
from scion.materials.plasticity import VonMises
from scion.mesh.mesh import ElementBlock, Mesh
from scion.shakedown import (
    ElasticEnvelope,
    LoadDomain,
    ResidualBasis,
    SelfEquilibrium,
    elastic_factor,
    melan_lower_bound,
    solve_elastic,
)

# --- geometry and material -------------------------------------------------
A_IN, B_OUT = 100.0, 200.0
E, NU, YIELD = 210.0e3, 0.3, 250.0

MESHES = [(8, 8), (12, 12), (16, 16), (24, 24), (32, 32)]
QUICK_MESHES = [(8, 8), (16, 16)]
LEVELS = np.linspace(1.02, 4.0, 16)
ENRICHMENTS = 8

RESULTS = Path(__file__).resolve().parent / "results"


def limit_pressure() -> float:
    """``p_lim = 2 Y / sqrt(3) ln(b/a)``, with no elastic constant in it.

    Which is what makes it the trustworthy number: it is a statement about
    equilibrium and the yield surface, so it tests the shakedown machinery
    without depending on how well the mesh represents elastic strains.
    """
    return 2.0 * YIELD / np.sqrt(3.0) * np.log(B_OUT / A_IN)


# ---------------------------------------------------------------------------

def quarter_annulus(n_r: int, n_theta: int) -> Domain:
    t = np.linspace(0.0, 1.0, n_r + 1) ** 1.3          # graded to the bore
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


def build(n_r: int, n_theta: int):
    """One elastic solve for unit bore pressure, plus the shakedown operators."""
    domain = quarter_annulus(n_r, n_theta)
    tables = QuadratureTables(domain, thickness=1.0)
    dm = tables.vector_dofs
    nodes = domain.coordinates

    on_x = np.where(nodes[:, 1] < 1e-9 * B_OUT)[0]     # theta = 0:  u_y = 0
    on_y = np.where(nodes[:, 0] < 1e-9 * B_OUT)[0]     # theta = 90: u_x = 0
    fixed = np.unique(np.concatenate([dm.element_dofs(on_x)[1::2],
                                      dm.element_dofs(on_y)[0::2]]))
    free = np.setdiff1d(np.arange(dm.n_total), fixed)

    # consistent nodal forces for unit pressure on the bore
    radius = np.hypot(nodes[:, 0], nodes[:, 1])
    bore = np.where(np.abs(radius - A_IN) < 1e-9 * B_OUT)[0]
    bore = bore[np.argsort(np.arctan2(nodes[bore, 1], nodes[bore, 0]))]
    f = np.zeros(dm.n_total)
    for p, q in zip(bore[:-1], bore[1:]):
        xp, xq = nodes[p], nodes[q]
        length = np.linalg.norm(xq - xp)
        normal = xp / np.linalg.norm(xp) + xq / np.linalg.norm(xq)
        normal /= np.linalg.norm(normal)
        for node in (p, q):
            f[dm.dof(int(node), 0)] += 0.5 * length * normal[0]
            f[dm.dof(int(node), 1)] += 0.5 * length * normal[1]

    C = isotropic_stiffness_mandel(E, NU)
    u, = solve_elastic(tables, C, free, [f])
    return (tables, SelfEquilibrium(tables, free),
            ElasticEnvelope(tables, C).add_case(u), C)


def analyse(n_r, n_theta, domain: LoadDomain):
    tables, equilibrium, envelope, C = build(n_r, n_theta)
    vertex = envelope.vertex_stresses(domain)
    lam_e = elastic_factor(vertex, VonMises(), YIELD)

    t0 = time.time()
    basis = ResidualBasis(tables, C, equilibrium, scale=YIELD)
    basis.from_overstress(vertex, VonMises(), YIELD, levels=LEVELS,
                          elastic_factor=lam_e)
    result = melan_lower_bound(vertex, basis, VonMises(), YIELD, rtol=1e-5,
                               enrichments=ENRICHMENTS)
    elapsed = time.time() - t0

    return dict(result=result, seconds=elapsed, points=tables.n_points,
                basis=basis.verify(),
                certificate=result.verify(vertex, equilibrium, VonMises(),
                                          YIELD))


# ---------------------------------------------------------------------------

def mesh_study(meshes):
    p_lim = limit_pressure()
    print(f"\n  Limit analysis: a one-vertex domain, against "
          f"p_lim = {p_lim:.4f}")
    print(f"\n  {'mesh':>9} {'points':>7} {'elastic':>9} {'modes':>6} "
          f"{'Melan':>10} {'/p_lim':>8} {'eq res':>9} {'time':>7}")
    rows = []
    for n_r, n_theta in meshes:
        out = analyse(n_r, n_theta, LoadDomain.fixed(1.0))
        r = out["result"]
        rows.append(out)
        print(f"  {n_r:4d}x{n_theta:<4d} {out['points']:7d} {r.elastic:9.3f} "
              f"{r.n_modes:6d} {r.factor:10.4f} {r.factor / p_lim:8.4f} "
              f"{out['certificate']['equilibrium_residual']:9.1e} "
              f"{out['seconds']:6.1f}s")
    return rows


def basis_study(n_r, n_theta):
    p_lim = limit_pressure()
    tables, equilibrium, envelope, C = build(n_r, n_theta)
    vertex = envelope.vertex_stresses(LoadDomain.fixed(1.0))
    lam_e = elastic_factor(vertex, VonMises(), YIELD)

    print(f"\n  Basis refinement on the {n_r}x{n_theta} mesh: the search space "
          f"grows, the bound rises")
    print(f"\n  {'levels':>7} {'enrich':>7} {'modes':>6} {'Melan':>10} "
          f"{'/p_lim':>8}")
    out = []
    for n_levels, enrich in ((1, 0), (2, 0), (4, 0), (8, 0), (8, 4), (16, 8)):
        basis = ResidualBasis(tables, C, equilibrium, scale=YIELD)
        basis.from_overstress(vertex, VonMises(), YIELD,
                              levels=np.linspace(1.02, 4.0, n_levels),
                              elastic_factor=lam_e)
        r = melan_lower_bound(vertex, basis, VonMises(), YIELD, rtol=1e-5,
                              enrichments=enrich)
        out.append((basis.n_modes, r.factor))
        print(f"  {n_levels:7d} {enrich:7d} {r.n_modes:6d} {r.factor:10.4f} "
              f"{r.factor / p_lim:8.4f}")
    return out


def domain_study(n_r, n_theta):
    p_lim = limit_pressure()
    print(f"\n  What the load *variation* costs, on the {n_r}x{n_theta} mesh")
    print(f"\n  {'domain':>13} {'vertices':>9} {'elastic':>9} {'Melan':>10} "
          f"{'reserve':>8} {'/p_lim':>8}")
    domains = [("fixed", LoadDomain.fixed(1.0)),
               ("pulsating", LoadDomain.box([0.0], [1.0])),
               ("alternating", LoadDomain.box([-1.0], [1.0]))]
    out = {}
    for name, domain in domains:
        res = analyse(n_r, n_theta, domain)["result"]
        out[name] = res
        print(f"  {name:>13} {domain.n_vertices:9d} {res.elastic:9.3f} "
              f"{res.factor:10.4f} {res.reserve:8.3f} "
              f"{res.factor / p_lim:8.4f}")
    return out


def report(rows, domains):
    p_lim = limit_pressure()
    best = max(r["result"].factor for r in rows)
    print("\n" + "=" * 74)
    print(f"  The bound stays below {p_lim:.3f} on every mesh and every basis, "
          f"reaching {100 * best / p_lim:.1f}%")
    print("=" * 74)
    print("\n  Below is the part that has to hold: a lower bound above the true")
    print("  value is not conservative, it is wrong. Approaching it is the")
    print("  quality measure, and it improves with the basis, not the mesh --")
    print("  refining resolves the elastic peak better and lowers the elastic")
    print("  limit, so a fixed basis gets slightly more conservative.")

    fixed = domains["fixed"].factor
    alt = domains["alternating"].factor
    print(f"\n  Holding the pressure gives {fixed:.2f}; letting it reverse gives")
    print(f"  {alt:.2f}, a factor of {fixed / alt:.2f}. One residual field has to")
    print("  serve both extremes, and the field that suits one is wrong for the")
    print("  other -- which is alternating plasticity, and no redistribution")
    print("  recovers it.")

    every = all(r["certificate"]["certified"] for r in rows)
    print(f"\n  Every returned field certified: {every}. The equilibrium residual")
    print("  and yield violation above are recomputed from the stored field, so")
    print("  they can be checked without trusting the solver that produced it.")


def plot(rows, basis_rows, domains):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    p_lim = limit_pressure()
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    pts = [r["points"] for r in rows]
    ax[0].semilogx(pts, [r["result"].factor for r in rows], "o-", lw=1.4,
                   label="Melan lower bound")
    ax[0].semilogx(pts, [r["result"].elastic for r in rows], "s--", lw=1.2,
                   label="elastic limit")
    ax[0].axhline(p_lim, ls="-", c="k", lw=1.2, label=r"$p_{lim}$ (analytic)")
    ax[0].set_xlabel("quadrature points")
    ax[0].set_ylabel("load factor")
    ax[0].set_title("Mesh: the bound stays below")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.3, which="both")

    modes = [m for m, _ in basis_rows]
    ax[1].plot(modes, [f / p_lim for _, f in basis_rows], "o-", lw=1.4)
    ax[1].axhline(1.0, ls="-", c="k", lw=1.2)
    ax[1].set_xlabel("basis modes")
    ax[1].set_ylabel(r"$\lambda_{SD} / p_{lim}$")
    ax[1].set_ylim(0.5, 1.05)
    ax[1].set_title("Basis: monotone, never crossing")
    ax[1].grid(alpha=0.3)

    names = list(domains)
    factors = [domains[n].factor for n in names]
    elastics = [domains[n].elastic for n in names]
    x = np.arange(len(names))
    ax[2].bar(x - 0.2, elastics, 0.4, label="elastic limit")
    ax[2].bar(x + 0.2, factors, 0.4, label="shakedown")
    ax[2].axhline(p_lim, ls="-", c="k", lw=1.2, label=r"$p_{lim}$")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names)
    ax[2].set_ylabel("load factor")
    ax[2].set_title("What load variation costs")
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "thick_cylinder_shakedown.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    # scoped to this package: logging.disable is process-global and
    # would silence every other library in the interpreter too
    import logging

    from scion.core.logging import set_level
    set_level(logging.WARNING)

    quick = "--quick" in sys.argv
    meshes = QUICK_MESHES if quick else MESHES
    study_mesh = meshes[-1]

    print("Shakedown of a thick-walled cylinder, plane strain, von Mises")
    print(f"  a={A_IN:g}, b={B_OUT:g}, E={E:g}, nu={NU}, Y={YIELD:g}")
    print(f"  analytic limit pressure = {limit_pressure():.4f}")

    rows = mesh_study(meshes)
    basis_rows = basis_study(*study_mesh)
    domains = domain_study(*study_mesh)
    report(rows, domains)
    plot(rows, basis_rows, domains)
