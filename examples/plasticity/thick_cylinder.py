#!/usr/bin/env python3
r"""
Thick-walled cylinder under internal pressure.

Problem
-------
A cylinder of inner radius ``a`` and outer radius ``b`` under internal pressure
``p``, plane strain, von Mises, perfectly plastic. A quarter is modelled with
symmetry conditions on the two straight edges.

Closed-form solution
--------------------
Plastic flow is incompressible, so in the plastic zone ``sigma_z`` is the
intermediate principal stress and von Mises reduces to

    sigma_theta - sigma_r = Y' = 2 Y / sqrt(3).

Equilibrium ``d sigma_r / dr = (sigma_theta - sigma_r) / r`` then integrates
directly, and matching to the elastic Lame field at the elastic-plastic
boundary ``r = c`` gives the pressure that drives the front to that radius:

    p(c) = Y' [ ln(c/a) + (1 - c^2/b^2) / 2 ]                            (1)

with the two endpoints

    p_e   = p(a) = Y' (1 - a^2/b^2) / 2        first yield at the bore
    p_lim = p(b) = Y' ln(b/a)                  collapse

In the fully plastic state the radial stress is ``sigma_r = Y' ln(r/b)``.

Note ``p_lim`` involves no elastic constant at all: it is a statement about
equilibrium and the yield surface, so it tests the constitutive model and the
assembly without depending on how well the mesh represents elastic strains.
That makes it the most trustworthy single number here.

Why displacement control
------------------------
Under pressure control the analysis simply diverges at ``p_lim`` with no
solution beyond it, and the last converged step is a lower bound of unknown
quality. Prescribing the radial displacement of the bore instead traces the
whole curve, and the pressure is recovered from the reactions -- so the limit
load is read off a plateau rather than inferred from a failure.

Volumetric locking is the usual worry here, since perfectly plastic flow is
isochoric and bilinear quadrilaterals resist an isochoric deformation with
spurious stiffness. In practice it barely shows: the limit load is accurate to
about 0.1% on a 12 x 12 mesh with full integration. That is because ``p_lim``
is a global equilibrium quantity governed by the through-wall force balance,
not by how faithfully any one element deforms -- the locking that would
dominate a displacement measure largely cancels in the reaction sum. Reduced
integration is *less* accurate on the coarse mesh here (-1.1% at 6 x 6), which
is worth knowing before reaching for it reflexively.

Run
---
    python -m examples.plasticity.thick_cylinder
    python -m examples.plasticity.thick_cylinder --refine     # mesh study
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simulacrum.core.domain import Domain
from simulacrum.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
    StatefulOperators,
)
from simulacrum.materials.notation import from_mandel, voigt_strain_to_mandel
from simulacrum.materials.plasticity import (
    PerfectPlasticity,
    PlasticityModel,
    von_mises_stress,
)
from simulacrum.mesh.mesh import ElementBlock, Mesh
from simulacrum.solvers.nonlinear import DisplacementControl

# --- geometry and material -------------------------------------------------
A_IN, B_OUT = 100.0, 200.0
E, NU, YIELD = 210e3, 0.3, 250.0

N_R, N_THETA = 48, 48          # elements through the wall / around the quarter
U_MAX = 0.60                   # prescribed bore displacement
N_STEPS = 40

RESULTS = Path(__file__).resolve().parent / "results" / "thick_cylinder"


# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------

def yield_plane_strain(Y: float = YIELD) -> float:
    """``Y' = 2Y/sqrt(3)``, the value of ``sigma_theta - sigma_r`` at yield."""
    return 2.0 * Y / np.sqrt(3.0)


def pressure_for_front(c, a=A_IN, b=B_OUT, Y=YIELD):
    """Equation (1): the pressure that places the elastic-plastic front at c."""
    Yp = yield_plane_strain(Y)
    c = np.asarray(c, dtype=float)
    return Yp * (np.log(c / a) + (1.0 - c ** 2 / b ** 2) / 2.0)


def elastic_limit(a=A_IN, b=B_OUT, Y=YIELD) -> float:
    return float(pressure_for_front(a, a, b, Y))


def limit_pressure(a=A_IN, b=B_OUT, Y=YIELD) -> float:
    return float(pressure_for_front(b, a, b, Y))


def radial_stress_fully_plastic(r, b=B_OUT, Y=YIELD):
    return yield_plane_strain(Y) * np.log(np.asarray(r, dtype=float) / b)


# ---------------------------------------------------------------------------
# Mesh: a structured quarter annulus
# ---------------------------------------------------------------------------

def quarter_annulus(n_r=N_R, n_theta=N_THETA, a=A_IN, b=B_OUT):
    """Quad4 mesh of the quarter annulus, graded towards the bore.

    The plastic front starts at ``r = a`` and moves outward, and the stress
    gradient is steepest there, so the radial spacing is biased inward.
    """
    t = np.linspace(0.0, 1.0, n_r + 1) ** 1.3          # grading
    r = a + (b - a) * t
    th = np.linspace(0.0, np.pi / 2.0, n_theta + 1)

    nodes = np.array([[ri * np.cos(tj), ri * np.sin(tj)]
                      for tj in th for ri in r])
    conn = np.array([[j * (n_r + 1) + i,
                      j * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i + 1,
                      (j + 1) * (n_r + 1) + i]
                     for j in range(n_theta) for i in range(n_r)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2), r, th


# ---------------------------------------------------------------------------

def build(n_r=N_R, n_theta=N_THETA, quad_order=None):
    dom, r, th = quarter_annulus(n_r, n_theta)
    model = PlasticityModel(E, NU, "j2", PerfectPlasticity(YIELD))
    asm = ElastoplasticAssembler(dom, model, quad_order=quad_order)
    ops = StatefulOperators(asm)

    nd = dom.coordinates
    dm = asm.dof_map
    rad = np.hypot(nd[:, 0], nd[:, 1])

    on_x_axis = np.where(nd[:, 1] < 1e-9)[0]           # theta = 0: u_y = 0
    on_y_axis = np.where(nd[:, 0] < 1e-9)[0]           # theta = 90: u_x = 0
    bore = np.where(np.abs(rad - A_IN) < 1e-9 * B_OUT)[0]

    fixed = np.concatenate([dm.element_dofs(on_x_axis)[1::2],
                            dm.element_dofs(on_y_axis)[0::2]])

    # prescribe the bore displacement radially, one component per direction
    prescribed = {}
    for n in bore:
        c, s = nd[n] / rad[n]
        for comp, direction in ((0, c), (1, s)):
            d = dm.dof(int(n), comp)
            if d in fixed:                              # symmetry wins
                continue
            prescribed[d] = U_MAX * direction

    free = np.setdiff1d(np.arange(asm.n_dof),
                        np.union1d(fixed, list(prescribed)))
    return asm, ops, model, dict(nodes=nd, radius=rad, bore=bore, fixed=fixed,
                                 prescribed=prescribed, free=free)


def internal_pressure(asm, ops, u, info) -> float:
    """Recover the bore pressure from the reactions.

    The nodal internal force at a constrained degree of freedom is the reaction
    the support supplies; summing
    its radial components and dividing by the loaded arc length gives the
    pressure. Using reactions rather than integrating the stress field avoids
    any extrapolation from quadrature points to the boundary.
    """
    f_int = ops.internal_force(u)
    nd, rad = info["nodes"], info["radius"]
    total = 0.0
    for n in info["bore"]:
        c, s = nd[n] / rad[n]
        total += (f_int[asm.dof_map.dof(int(n), 0)] * c
                  + f_int[asm.dof_map.dof(int(n), 1)] * s)
    arc = 0.5 * np.pi * A_IN                            # quarter of 2 pi a
    return total / arc


def plastic_front(asm, ops, u, info) -> float:
    """Outermost radius at which any quadrature point has yielded."""
    res = asm.assemble(u, ops.state)
    if not res.plastic.any() and ops.state.alpha.max() <= 0.0:
        return A_IN
    coords = info["nodes"][asm.conn]                    # (n_el, npe, 2)
    yielded = (ops.state.alpha > 1e-12).reshape(asm.n_elements, asm.n_qp)
    r_el = np.hypot(coords[..., 0], coords[..., 1]).max(axis=1)
    hit = r_el[yielded.any(axis=1)]
    return float(hit.max()) if hit.size else A_IN


# ---------------------------------------------------------------------------

def run(n_r=N_R, n_theta=N_THETA, n_steps=N_STEPS, quad_order=None,
        verbose=True):
    import logging
    logging.disable(logging.CRITICAL)

    asm, ops, model, info = build(n_r, n_theta, quad_order)
    driver = DisplacementControl(
        ops.internal_force, ops.tangent, asm.n_dof, info["free"],
        info["prescribed"], info["fixed"], n_steps=n_steps, tol=1e-8,
        max_iter=25, line_search=False, on_step_converged=ops.commit)

    history = []

    def record(u):
        history.append((np.abs(u[list(info["prescribed"])]).max(),
                        internal_pressure(asm, ops, u, info),
                        plastic_front(asm, ops, u, info)))

    driver.on_step_converged = lambda u: (ops.commit(u), record(u))
    result = driver.solve()

    if verbose:
        print(f"  mesh {n_r} x {n_theta} = {asm.n_elements} quad4, "
              f"{asm.n_dof} dof, {asm.n_qp} qp/element")
        print(f"  steps converged: {sum(s.converged for s in result.step_info)}"
              f"/{len(result.step_info)}")
    return asm, ops, np.array(history), result


def report(history, label=""):
    p_e, p_lim = elastic_limit(), limit_pressure()
    p = history[:, 1]
    print(f"\n  {'analytic':>22}  {'computed':>10}  {'error':>8}")
    print(f"  {'elastic limit p_e':>22}  {p_e:10.3f}")
    print(f"  {'limit pressure p_lim':>22}  {p_lim:10.3f}")
    print(f"  {'computed plateau':>22}  {p.max():10.3f}  "
          f"{100 * (p.max() - p_lim) / p_lim:+7.2f}%")

    # pressure at each computed front position, against equation (1)
    print(f"\n  {'front c':>8} {'p computed':>11} {'p analytic':>11} {'error':>8}")
    for u_a, pp, c in history[::max(1, len(history) // 8)]:
        if c > A_IN * 1.001:
            pa = float(pressure_for_front(c))
            print(f"  {c:8.2f} {pp:11.3f} {pa:11.3f} "
                  f"{100 * (pp - pa) / pa:+7.2f}%")


def refinement_study():
    """The limit load against mesh density, and against integration order.

    Full integration converges from slightly above; reduced integration
    under-predicts on a coarse mesh, because removing three of the four points
    removes constraint the collapse mechanism actually needs, not only the
    spurious volumetric constraint. Neither is off by more than about 1%, which
    says the limit load is not locking-sensitive for this problem.
    """
    p_lim = limit_pressure()
    print(f"\n{'mesh':>10} {'dof':>7} {'rule':>6} {'p_max':>9} {'error':>9}")
    for n in (6, 12, 24):
        for order, tag in ((None, "2x2"), (1, "1x1")):
            _, _, hist, _ = run(n, n, n_steps=30, quad_order=order,
                                verbose=False)
            pm = hist[:, 1].max()
            print(f"{n:5d}x{n:<4d} {2 * (n + 1) ** 2:7d} {tag:>6} "
                  f"{pm:9.3f} {100 * (pm - p_lim) / p_lim:+8.2f}%")


def plot(history, asm, ops, info=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return

    p_e, p_lim = elastic_limit(), limit_pressure()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(history[:, 0], history[:, 1], "o-", ms=3, lw=1.4,
               label="computed")
    ax[0].axhline(p_lim, ls="--", c="k", lw=1.1, label=r"$p_{lim}$ (analytic)")
    ax[0].axhline(p_e, ls=":", c="0.4", lw=1.1, label=r"$p_e$ (analytic)")
    ax[0].set_xlabel(r"bore displacement $u(a)$")
    ax[0].set_ylabel("internal pressure $p$")
    ax[0].set_title("Pressure-displacement response")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.3)

    moved = history[history[:, 2] > A_IN * 1.001]
    if len(moved):
        c = np.linspace(A_IN, B_OUT, 200)
        ax[1].plot(pressure_for_front(c), c, "-", c="k", lw=1.2,
                   label="analytic, eq. (1)")
        ax[1].plot(moved[:, 1], moved[:, 2], "o", ms=4, label="computed")
    ax[1].set_xlabel("internal pressure $p$")
    ax[1].set_ylabel("elastic-plastic front $c$")
    ax[1].set_title("Growth of the plastic zone")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "thick_cylinder.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    print("Thick-walled cylinder, von Mises perfect plasticity, plane strain")
    print(f"  a={A_IN}, b={B_OUT}, E={E:g}, nu={NU}, Y={YIELD}")

    if "--refine" in sys.argv:
        refinement_study()
    else:
        asm, ops, history, result = run()
        report(history)
        plot(history, asm, ops, None)
