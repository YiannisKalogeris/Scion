"""
Single-edge notched tension test.

Section 4.1 of Ambati, Gerasimov & De Lorenzis, "A review on phase-field models
of brittle fracture and a new fast hybrid formulation", Comput Mech 55 (2015)
383-405 -- itself the benchmark from Miehe, Hofacker & Welschinger (2010).

A 1 x 1 mm square with a horizontal notch of length 0.5 mm at mid-height of the
left edge, pulled vertically on the top edge. The crack runs straight across to
the right edge.

Why this one first: all four formulations agree on it. Isotropic, Amor, Miehe
and hybrid give the same crack path and load-displacement curve (their Figs. 8
and 9), so it isolates the assembly and the solver from the formulation. The
shear test is the discriminating one and should be attempted only after this
reproduces.

Paper parameters
----------------
    lambda = 121.15 kN/mm^2      mu = 80.77 kN/mm^2
    Gc     = 2.7e-3  kN/mm       l  = 4.0e-3 mm
    12735 quadrilaterals, refined in the central band
    du = 1e-5 mm up to u = 5e-3 mm, then du = 1e-6 mm to failure
    peak load about 0.7 kN at u about 5.5e-3 mm

Run
---
    python -m examples.phase_field.single_edge_notched_tension            # quick
    python -m examples.phase_field.single_edge_notched_tension --paper    # full
    python -m examples.phase_field.single_edge_notched_tension --model=amor

``--quick`` (the default) coarsens ``l`` and the mesh so the run finishes in a
couple of minutes. It reproduces the *behaviour* -- straight crack, brittle load
drop -- but not the paper's peak load, which depends on ``l``. Use ``--paper``
for the quantitative comparison, and expect it to take a long time in pure
Python: 12735 elements and well over a thousand load steps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from scion.core.domain import Domain
from scion.materials.fracture import PhaseFieldModel
from scion.mesh.mesh import ElementBlock, Mesh
from scion.multiphysics import PhaseFieldFractureProblem

# --- paper material --------------------------------------------------------
LAM, MU = 121.15, 80.77                     # kN/mm^2
E = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
GC = 2.7e-3                                  # kN/mm

NOTCH_Y = 0.5
NOTCH_X = 0.5

RESULTS = Path(__file__).resolve().parent / "results" / "sent"


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def graded_axis(lo, hi, band, h_fine, ratio=1.15):
    """Coordinates uniform at ``h_fine`` inside ``band``, geometrically graded
    outside. Uniform refinement everywhere would be wasted: only the strip the
    crack passes through has to resolve ``l``."""
    b0, b1 = band
    fine = np.arange(b0, b1 + 0.5 * h_fine, h_fine)

    def grade(start, stop, sign):
        pts, x, h = [], start, h_fine
        while (stop - x) * sign > 1e-12:
            h *= ratio
            x = x + sign * h
            if (stop - x) * sign <= 0:
                break
            pts.append(x)
        return pts

    left = sorted(grade(b0, lo, -1))
    right = grade(b1, hi, +1)
    pts = np.unique(np.concatenate([[lo], left, fine, right, [hi]]))
    # drop points that would leave a sliver against a domain edge: a graded
    # sequence rarely lands exactly on the boundary, and the leftover gap
    # becomes an element orders of magnitude smaller than its neighbours,
    # which wrecks the conditioning for no benefit
    keep = np.ones(pts.size, dtype=bool)
    gaps = np.diff(pts)
    for i, g in enumerate(gaps):
        if g < 0.3 * h_fine:
            keep[i + 1 if i + 1 < pts.size - 1 else i] = False
    keep[0] = keep[-1] = True
    return pts[keep]


def build_mesh(h_fine: float, ell: float):
    """Structured quad4 grid, refined in a horizontal band about the crack path
    and in the right half where the crack travels."""
    # The refined band must cover the whole crack LINE, not just the part the
    # crack propagates through. The notch carries a damage band of width ~2l as
    # well, and if the mesh is coarse there the band smears over an element:
    # the specimen comes out too stiff, the load-displacement curve develops a
    # staircase as coarse elements damage one at a time, and the staggered
    # cycle stops converging because each iteration damages a little more.
    # So x is uniform across the full width; only y is graded away from y = 0.5.
    half = max(4.0 * ell, 3.0 * h_fine)
    xs = np.arange(0.0, 1.0 + 0.5 * h_fine, h_fine)
    xs = np.unique(np.append(xs, 1.0))
    ys = graded_axis(0.0, 1.0, (NOTCH_Y - half, NOTCH_Y + half), h_fine)

    nodes = np.array([[x, y] for y in ys for x in xs])
    nx = len(xs) - 1
    conn = np.array([[j * (nx + 1) + i, j * (nx + 1) + i + 1,
                      j * (nx + 1) + i + nx + 2, j * (nx + 1) + i + nx + 1]
                     for j in range(len(ys) - 1) for i in range(nx)])

    mesh = Mesh(nodes, [ElementBlock("quad4", conn)])
    mesh.add_boundary("bottom", np.where(nodes[:, 1] < 1e-12)[0])
    mesh.add_boundary("top", np.where(nodes[:, 1] > 1.0 - 1e-12)[0])
    return Domain(mesh, dim=2)


# ---------------------------------------------------------------------------
# Notch
# ---------------------------------------------------------------------------

def notch_history(problem, ell: float, B: float = 1.0e3,
                  width: float | None = None) -> np.ndarray:
    r"""Initial history field representing the pre-existing notch.

    Rather than cutting the mesh -- which needs duplicated nodes along the notch
    and complicates every later step -- the notch is imposed as a large initial
    value of H along the notch line, following Borden et al. (2012):

        H_0(x) = B Gc / (2 l) * (1 - dist(x) / w)   for dist <= w

    ``B`` controls how sharply the damage approaches 1, ``w`` how wide the
    seeded band is. Both matter more than they look: with ``B`` large over a
    band of half-width ``l/2`` the damage saturates at 1 across the *whole*
    band rather than only on the line, and the regularized surface energy comes
    out around 40% too high -- the seeded notch is then not a crack of length
    0.5 but a fat damaged strip. Calibrating against the exact value
    ``surface_energy / Gc = 0.5`` gives ``w`` of order ``l/4``, floored at the
    element size so the band is never missed between quadrature points.

    This is the device the paper uses for the shear test ("the pre-existing
    crack is modeled by defining an initial strain-history field").
    """
    coords = problem.domain.coordinates[problem.conn]           # (n_el, npe, 2)
    xq = np.einsum("eqa,ead->eqd", problem.N, coords).reshape(-1, 2)

    on_segment = xq[:, 0] <= NOTCH_X
    dist = np.where(on_segment,
                    np.abs(xq[:, 1] - NOTCH_Y),
                    np.hypot(xq[:, 0] - NOTCH_X, xq[:, 1] - NOTCH_Y))

    if width is None:
        width = max(0.25 * ell, problem.element_size)
    H0 = np.zeros(xq.shape[0])
    inside = dist <= width
    H0[inside] = B * GC / (2.0 * ell) * (1.0 - dist[inside] / width)
    return H0


# ---------------------------------------------------------------------------

def build(coupling="hybrid", split="spectral", dissipation="AT2",
          ell=0.015, h_fine=0.006):
    domain = build_mesh(h_fine, ell)
    model = PhaseFieldModel(E, NU, GC, ell, split=split, coupling=coupling,
                            dissipation=dissipation)
    problem = PhaseFieldFractureProblem(domain, model, thickness=1.0,
                                        stagger_tol=10.0, max_stagger=50)

    mesh = domain.mesh
    bottom, top = mesh.boundary_nodes("bottom"), mesh.boundary_nodes("top")
    problem.u.prescribe(bottom, 0, 0.0)
    # The elongation is split symmetrically between the two edges, matching the
    # micropolar rows, so the whole study describes one boundary value problem.
    # Here it is a rigid translation and changes nothing -- this is a mesh, with no background grid to drift against --
    # which is exactly what makes it safe to unify on.
    problem.set_displacement_control(bottom, 1, -0.5)
    problem.set_displacement_control(top, 1, 0.5)     # factor = u in mm
    # The reported reaction is the force on the LOADED edge. Both edges are now
    # control dofs and their reactions cancel by equilibrium, so the control set
    # is narrowed back to the top before any step runs.
    problem._control_dofs = np.array(
        sorted(problem.u.dof_map.dof(int(n), 1) for n in top), dtype=np.int64)

    H0 = notch_history(problem, ell)
    problem.state["H"] = H0
    problem.state.commit()
    return domain, model, problem


# ---------------------------------------------------------------------------

def main(coupling="hybrid", split="spectral", dissipation="AT2",
         paper=False, plot=True):
    if paper:
        ell, h_fine = 4.0e-3, 2.0e-3
        steps = np.concatenate([np.arange(0.0, 5.0e-3, 1e-5),
                                np.arange(5.0e-3, 7.0e-3, 1e-6)])
    else:
        ell, h_fine = 0.02, 0.008
        steps = np.concatenate([np.arange(0.0, 6.0e-3, 5.0e-4),
                                np.arange(6.0e-3, 2.4e-2, 1.0e-4)])

    domain, model, problem = build(coupling, split, dissipation, ell, h_fine)
    print(model)
    print(f"nodes {domain.n_nodes}   elements {problem.n_elements}   "
          f"h_min {problem.element_size:.4g}   l/h {ell / problem.element_size:.2f}")
    problem.check_mesh()

    disp, load, iters = [], [], []
    t0 = time.time()

    def record(res, prob):
        if not res.converged:
            return
        disp.append(res.factor)
        load.append(res.reactions["control"])
        iters.append(res.staggered.iterations)
        if len(disp) % 20 == 0 or prob.d.values.max() > 0.95:
            print(f"  u={res.factor:9.6f} mm  P={load[-1]:8.5f} kN  "
                  f"stag={iters[-1]:3d} ({res.staggered.stopped_on})  "
                  f"d_max={prob.d.values.max():.3f}")

    # run() rather than a manual loop over solve_step: the increment has to be
    # cut where the crack starts moving, and where that is depends on the
    # notch, the mesh and l -- it is not knowable when the schedule is written.
    problem.run(steps, callback=record, adaptive=True, stop_on_failure=True)

    disp, load = np.array(disp), np.array(load)
    k = int(np.argmax(load))
    print(f"\npeak load {load[k]:.5f} kN at u = {disp[k]:.6f} mm")
    print(f"crack length (surface energy / Gc) = "
          f"{problem.surface_energy() / GC:.4f} mm   (notch-to-edge = 0.5)")
    print(f"staggered iterations: mean {np.mean(iters):.1f}, max {max(iters)}")
    print(f"wall time {time.time() - t0:.1f} s")

    if plot:
        make_figure(problem, disp, load, coupling)
    return problem, disp, load


def make_figure(problem, disp, load, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return

    nodes = problem.domain.coordinates
    conn = problem.conn
    tri = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]])
    T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tri)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(disp, load, lw=1.5)
    ax[0].set_xlabel("displacement [mm]")
    ax[0].set_ylabel("force [kN]")
    ax[0].set_title("Load-displacement")
    ax[0].grid(alpha=0.3)

    c = ax[1].tricontourf(T, problem.d.values, levels=np.linspace(0, 1, 21),
                          cmap="turbo")
    fig.colorbar(c, ax=ax[1], label="d")
    ax[1].set_aspect("equal")
    ax[1].set_title("Crack phase field")

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"sent_{tag}.png"
    fig.savefig(path, dpi=130)
    print(f"figure -> {path}")


if __name__ == "__main__":
    kw = {"paper": "--paper" in sys.argv, "plot": "--no-figure" not in sys.argv}
    for a in sys.argv[1:]:
        if a.startswith("--model="):
            name = a.split("=")[1]
            kw["coupling"] = "anisotropic" if name in ("amor", "miehe") else name
            if name == "amor":
                kw["split"] = "amor"
        elif a.startswith("--dissipation="):
            kw["dissipation"] = a.split("=")[1]
    main(**kw)
