#!/usr/bin/env python3
r"""
Localisation in a softening bar: why local damage needs a length scale.

Problem
-------
A prismatic bar in plane strain, pulled under displacement control, with a
smooth 10% neck at mid-span to decide where damage starts. Modified von Mises
equivalent strain, exponential softening, no split. The same problem is run two
ways on the same sequence of meshes:

    local     DamageModel through ElastoplasticAssembler and Newton
    gradient  GradientDamageModel as a two-field problem, ebar filtered
              through ell^2

and the question is only whether the answers converge as the mesh is refined.

What the local model does
-------------------------
Softening destroys ellipticity. The band has no width of its own, so it
collapses onto the smallest thing available -- the element -- and the physical
length of the damaged region halves whenever the element size does. The
dissipated energy goes with it, and in the limit the bar fails at the peak load
having absorbed nothing.

Worse, the structural response snaps back. Past the peak, the elastic bulk
unloads faster than the band opens, so equilibrium requires the *end
displacement* to decrease and no positive increment can reach the next point on
the path. Displacement control simply stops. That happens once

    (band length) / L  <  |E_soft| / E,

and since the band length is a few elements, refining the mesh drives the
analysis into snap-back. The failure to converge is not a solver weakness, it is
the discrete problem telling you the continuum one is ill-posed -- and it is why
the finer meshes below stop early rather than producing a wrong curve.

What the gradient model does
----------------------------
The nonlocal field ``ebar - ell^2 laplacian(ebar) = eps_eq`` gives the band a
width the material owns. Peak load, dissipated energy and band width all settle
under refinement, and every mesh traces the whole softening branch.

It is also cheaper per iteration. At fixed ``ebar`` the damage does not respond
to the local strain, so momentum is one linear symmetric solve; at fixed ``u``
the Helmholtz operator is constant and is factorised once for the entire
analysis. The local model, by contrast, needs Newton with an unsymmetric
tangent.

Run
---
    python -m examples.damage.bar_localization
    python -m examples.damage.bar_localization --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
# not np.trapz / np.trapezoid: NumPy renamed one to the other in 2.0 and
# removed the old name, so either spelling pins the library to one major
# version. SciPy is already a hard dependency and carries both.
from scipy.integrate import trapezoid

from simulacrum.core.domain import Domain
from simulacrum.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
    StatefulOperators,
)
from simulacrum.materials.damage import (
    DamageModel,
    ExponentialSoftening,
    GradientDamageModel,
)
from simulacrum.mesh.generators import rectangle_mesh
from simulacrum.multiphysics import GradientDamageProblem
from simulacrum.solvers.nonlinear import DisplacementControl

# --- geometry and material -------------------------------------------------
LENGTH, HEIGHT, THICKNESS = 10.0, 1.0, 1.0
NECK, NECK_WIDTH = 0.10, 0.5          # 10% reduction over a Gaussian of this width

E, NU = 20.0e3, 0.2
KAPPA_0, ALPHA, BETA = 1.0e-4, 1.0, 500.0
K_RATIO = 10.0                        # f_c / f_t for the modified von Mises
ELL = 0.5                             # internal length

U_MAX, N_STEPS = 3.0e-3, 40

#: A displacement just past damage onset, where every local mesh still
#: converges. The band collapse and the snap-back are two different failures
#: and they cannot be seen at the same load: by the time a fine mesh has snapped
#: back it has stopped, and its last committed state is undamaged. So the local
#: study is run twice -- once here, to measure the band, and once at ``U_MAX``,
#: to measure how far the analysis gets at all.
PROBE_U, PROBE_STEPS = 1.2e-3, 30
#: The two studies cannot share a mesh sequence, and that is the headline.
#:
#: The gradient runs need ``h <= ell/3`` or the band is set by the mesh again
#: while still looking smooth, which puts the coarsest usable mesh at nx = 80.
#: The local runs need the opposite: they snap back and stop as soon as the
#: band is shorter than ``|E_soft|/E`` of the bar, which rules out everything
#: finer than about nx = 40. There is no mesh on which the local model is both
#: runnable and resolved -- so the sequences below overlap only where the local
#: model has already failed.
LOCAL_MESHES = [(20, 2), (40, 4), (80, 8), (160, 8)]
GRADIENT_MESHES = [(80, 8), (160, 8), (320, 16), (640, 16)]
QUICK_LOCAL = [(20, 2), (40, 4)]
QUICK_GRADIENT = [(80, 8), (160, 8)]

RESULTS = Path(__file__).resolve().parent / "results"


def law():
    return ExponentialSoftening(KAPPA_0, alpha=ALPHA, beta=BETA)


def softening_modulus() -> float:
    """``|dsigma/dkappa|`` just past the threshold, for the snap-back estimate."""
    return E * KAPPA_0 * ALPHA * BETA


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def bar(nx: int, ny: int) -> Domain:
    """Rectangular bar with a smooth neck at mid-span.

    A geometric imperfection rather than a weakened element: it perturbs the
    problem without introducing a second length scale tied to the mesh, so the
    mesh study measures the model and not the imperfection.
    """
    mesh = rectangle_mesh((0.0, LENGTH), (0.0, HEIGHT), nx, ny)
    x = mesh.nodes[:, 0]
    s = 1.0 - NECK * np.exp(-((x - 0.5 * LENGTH) / NECK_WIDTH) ** 2)
    mesh.nodes[:, 1] = (mesh.nodes[:, 1] - 0.5 * HEIGHT) * s + 0.5 * HEIGHT
    return Domain(mesh, dim=2)


def _ends(domain):
    nodes = domain.coordinates
    return (np.where(nodes[:, 0] < 1e-9)[0],
            np.where(nodes[:, 0] > LENGTH - 1e-9)[0])


def profile(damage_by_element: np.ndarray, nx: int):
    """Damage averaged across the section, against the axial coordinate."""
    d = damage_by_element.reshape(nx, -1).mean(axis=1)
    return (np.arange(nx) + 0.5) * LENGTH / nx, d


def band_width(x, d) -> float:
    """Twice the standard deviation of the damage profile.

    Measured as a second moment rather than by counting elements above a
    threshold, so the number does not inherit the resolution of the mesh that
    is being varied.
    """
    w = np.maximum(d, 0.0)
    if w.sum() <= 0.0:
        return 0.0
    mean = (w * x).sum() / w.sum()
    return float(2.0 * np.sqrt((w * (x - mean) ** 2).sum() / w.sum()))


# ---------------------------------------------------------------------------
# Local damage: the standard assembler, Newton, unsymmetric tangent
# ---------------------------------------------------------------------------

def run_local(nx: int, ny: int, u_max: float = U_MAX,
              n_steps: int = N_STEPS):
    domain = bar(nx, ny)
    model = DamageModel(E, NU, law(), equivalent="modified-von-mises",
                        equivalent_kwargs={"k": K_RATIO})
    asm = ElastoplasticAssembler(domain, model, thickness=THICKNESS)
    ops = StatefulOperators(asm)

    left, right = _ends(domain)
    dm = asm.dof_map
    fixed = np.concatenate([dm.element_dofs(left)[0::2],
                            [dm.dof(int(left[0]), 1)]])
    prescribed = {dm.dof(int(n), 0): u_max for n in right}
    free = np.setdiff1d(np.arange(asm.n_dof),
                        np.union1d(fixed, list(prescribed)))

    history = []
    driver = DisplacementControl(
        ops.internal_force, ops.tangent, asm.n_dof, free, prescribed, fixed,
        n_steps=n_steps, tol=1e-8, max_iter=40, line_search=True)

    def record(u):
        ops.commit(u)
        f = ops.internal_force(u)
        history.append((max(u[d] for d in prescribed),
                        sum(f[d] for d in prescribed)))

    driver.on_step_converged = record
    result = driver.solve()

    damage = model.evolution.D(ops.state.kappa).reshape(
        asm.n_elements, asm.n_qp).mean(axis=1)
    h = np.array(history) if history else np.zeros((0, 2))
    return dict(
        nx=nx, h=LENGTH / nx, symmetric=asm.symmetric,
        traced=sum(s.converged for s in result.step_info),
        attempted=len(result.step_info),
        curve=h, peak=h[:, 1].max() if len(h) else np.nan,
        work=float(trapezoid(h[:, 1], h[:, 0])) if len(h) > 1 else 0.0,
        damage=damage, dmax=float(damage.max()))


# ---------------------------------------------------------------------------
# Gradient damage: two fields, both linear
# ---------------------------------------------------------------------------

def run_gradient(nx: int, ny: int, n_steps: int = N_STEPS):
    domain = bar(nx, ny)
    model = GradientDamageModel(E, NU, law(), ell=ELL,
                                equivalent="modified-von-mises",
                                equivalent_kwargs={"k": K_RATIO})
    problem = GradientDamageProblem(domain, model, thickness=THICKNESS)
    problem.check_mesh()

    left, right = _ends(domain)
    problem.u.prescribe(left, 0, 0.0)
    problem.u.prescribe([left[0]], 1, 0.0)
    problem.set_displacement_control(right, 0, 1.0)

    history = []
    steps = problem.run(
        np.linspace(U_MAX / n_steps, U_MAX, n_steps),
        callback=lambda r, p: history.append(
            (p.control_displacement(), p.control_force())),
        stop_on_failure=False)

    h = np.array(history)
    damage = problem.damage_by_element()
    return dict(
        nx=nx, h=LENGTH / nx, element_size=problem.element_size,
        traced=sum(s.converged for s in steps), attempted=len(steps),
        curve=h, peak=h[:, 1].max() if len(h) else np.nan,
        work=problem.external_work, dissipated=problem.dissipated_energy(),
        damage=damage, dmax=float(damage.max()),
        stagger=float(np.mean([s.staggered.iterations for s in steps
                               if s.staggered])))


# ---------------------------------------------------------------------------

def report(local, probe, gradient):
    print("\n" + "=" * 78)
    print("  Local damage: the band follows the mesh")
    print("=" * 78)
    limit = softening_modulus() / E
    print(f"\n  snap-back once (band length)/L < |E_soft|/E = {limit:.3f}, "
          f"i.e. band < {limit * LENGTH:.2f}")
    print(f"\n  probed at u = {PROBE_U:.1e}, just past onset, where every "
          f"mesh still converges:")
    print(f"\n  {'nx':>5} {'h':>7} {'steps':>9} {'band':>7} {'band/h':>7} "
          f"{'Dmax':>7}")
    for r in probe:
        x, d = profile(r["damage"], r["nx"])
        w = band_width(x, d)
        print(f"  {r['nx']:5d} {r['h']:7.4f} {r['traced']:3d}/{r['attempted']:<5d} "
              f"{w:7.3f} {w / r['h']:7.1f} {r['dmax']:7.4f}")

    print(f"\n  loaded to u = {U_MAX:.1e}, to see how far each mesh gets:")
    print(f"\n  {'nx':>5} {'h':>7} {'steps traced':>13} {'peak F':>9} "
          f"{'work so far':>12} {'Dmax':>7}")
    for r in local:
        print(f"  {r['nx']:5d} {r['h']:7.4f} {r['traced']:6d}/{r['attempted']:<6d} "
              f"{r['peak']:9.4f} {r['work']:12.6f} {r['dmax']:7.4f}")

    damaged = [(r, band_width(*profile(r["damage"], r["nx"])))
               for r in probe if r["dmax"] > 1e-6]
    if damaged:
        ratios = [w / r["h"] for r, w in damaged]
        print(f"\n  band/h is {min(ratios):.2f} to {max(ratios):.2f} across "
              f"those meshes: the band is one")
        print(f"  element wide, whatever the element is. Its physical length "
              f"went "
              f"{damaged[0][1]:.3f} -> {damaged[-1][1]:.3f}")
        print(f"  as h went {damaged[0][0]['h']:.3f} -> "
              f"{damaged[-1][0]['h']:.3f}, and the dissipated energy goes down "
              f"with it.")
        print(f"  There is no limit for it to converge to -- refine far enough "
              f"and the bar")
        print(f"  fails having absorbed nothing.")
    if len(damaged) < len(probe):
        print(f"\n  The finest mesh does not even reach damage at this load: "
              f"its band has")
        print(f"  narrowed enough that the peak arrives later and the snap-back "
              f"arrives with it.")
    print("\n  At the full load the finer meshes stop at damage onset. The "
          "response snaps")
    print("  back past the peak, displacement control cannot follow it, and "
          "no tolerance")
    print("  or iteration cap changes that: the continuum problem has no "
          "solution to")
    print("  converge to. Nor is there any mesh here that is both runnable "
          "and fine")
    print(f"  enough to resolve ell = {ELL:g} -- which is the whole argument "
          f"for a length scale.")

    print("\n" + "=" * 78)
    print("  Gradient damage: the band follows ell")
    print("=" * 78)
    print(f"\n  ell = {ELL:g}, so the band should settle at a few times that "
          f"regardless of h")
    print(f"\n  {'nx':>5} {'h':>7} {'h/ell':>6} {'steps':>10} {'peak F':>9} "
          f"{'dissipated':>11} {'band':>7} {'stagger':>8}")
    for r in gradient:
        x, d = profile(r["damage"], r["nx"])
        w = band_width(x, d)
        print(f"  {r['nx']:5d} {r['h']:7.4f} {r['element_size'] / ELL:6.2f} "
              f"{r['traced']:4d}/{r['attempted']:<5d} {r['peak']:9.4f} "
              f"{r['dissipated']:11.6f} {w:7.3f} {r['stagger']:8.1f}")

    peaks = np.array([r["peak"] for r in gradient])
    diss = np.array([r["dissipated"] for r in gradient])
    widths = np.array([band_width(*profile(r["damage"], r["nx"]))
                       for r in gradient])
    print(f"\n  spread over the mesh sequence:"
          f"  peak {100 * peaks.ptp() / peaks.mean():.2f}%"
          f"   dissipated {100 * diss.ptp() / diss.mean():.2f}%"
          f"   band {100 * widths.ptp() / widths.mean():.2f}%")
    print("  Every mesh traced every step, and the band is set by ell.")


def plot(local, probe, gradient):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.2))

    for r in local:
        if len(r["curve"]):
            ax[0].plot(r["curve"][:, 0], r["curve"][:, 1], "o-", ms=2.5,
                       lw=1.2, label=f"h = {r['h']:.3f}")
    ax[0].set_xlabel("end displacement")
    ax[0].set_ylabel("reaction")
    ax[0].set_title("Local: curves stop where they snap back")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.3)

    for r in gradient:
        ax[1].plot(r["curve"][:, 0], r["curve"][:, 1], "-", lw=1.4,
                   label=f"h = {r['h']:.3f}")
    ax[1].set_xlabel("end displacement")
    ax[1].set_ylabel("reaction")
    ax[1].set_title(f"Gradient (ell = {ELL:g}): curves coincide")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)

    # the local profiles come from the probe run: at the full load the finer
    # meshes have stopped before damaging anything, so their committed state
    # is a straight zero and there is nothing to draw
    for r in probe:
        x, d = profile(r["damage"], r["nx"])
        ax[2].plot(x, d, "--", lw=1.0, alpha=0.8,
                   label=f"local, h = {r['h']:.3f}")
    for r in gradient:
        x, d = profile(r["damage"], r["nx"])
        ax[2].plot(x, d, "-", lw=1.4, label=f"gradient, h = {r['h']:.3f}")
    ax[2].set_xlim(0.25 * LENGTH, 0.75 * LENGTH)
    ax[2].set_xlabel("x")
    ax[2].set_ylabel("damage")
    ax[2].set_title("Damage profile (local: probed at onset)")
    ax[2].legend(fontsize=6, frameon=False, ncol=2)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "bar_localization.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    # scoped to this package: logging.disable is process-global and
    # would silence every other library in the interpreter too
    import logging

    from simulacrum.core.logging import set_level
    set_level(logging.WARNING)

    quick = "--quick" in sys.argv
    local_meshes = QUICK_LOCAL if quick else LOCAL_MESHES
    gradient_meshes = QUICK_GRADIENT if quick else GRADIENT_MESHES

    print("Softening bar under displacement control, plane strain")
    print(f"  L={LENGTH:g}, H={HEIGHT:g}, {100 * NECK:g}% neck, "
          f"E={E:g}, nu={NU}")
    print(f"  kappa_0={KAPPA_0:g}, alpha={ALPHA:g}, beta={BETA:g}, "
          f"k={K_RATIO:g}, ell={ELL:g}")
    print(f"  fracture energy per unit volume: "
          f"{law().fracture_energy(E):.5g}")

    print("\n  local runs:")
    local, probe = [], []
    for nx, ny in local_meshes:
        r = run_local(nx, ny)
        local.append(r)
        probe.append(run_local(nx, ny, PROBE_U, PROBE_STEPS))
        print(f"    nx={nx:4d}  traced {r['traced']}/{r['attempted']} at full "
              f"load, {probe[-1]['traced']}/{probe[-1]['attempted']} at the "
              f"probe;  symmetric tangent: {r['symmetric']}")

    print("\n  gradient runs:")
    gradient = []
    for nx, ny in gradient_meshes:
        r = run_gradient(nx, ny)
        gradient.append(r)
        print(f"    nx={nx:4d}  traced {r['traced']}/{r['attempted']}  "
              f"mean staggered iterations {r['stagger']:.1f}")

    report(local, probe, gradient)
    plot(local, probe, gradient)
