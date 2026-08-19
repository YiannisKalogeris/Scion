"""
Granular column collapse with Drucker-Prager sand.

A rectangular column of dry sand, backed against a wall, released under gravity.
The classic granular benchmark (Lube et al. 2005; Lajeunesse et al. 2005),
because it has published experimental scaling laws rather than an eyeballed
animation to compare against.

Run:  python -m examples.mpm.sand_column_collapse
      python -m examples.mpm.sand_column_collapse --quick
      python -m examples.mpm.sand_column_collapse --no-vtk

What this validates, and what it does not
-----------------------------------------
**The angle of repose tracks the friction angle.** This is the material test,
and it passes: the settled pile slopes at roughly ``phi - 5 deg`` across
``phi = 20, 30, 40``, converging as ``phi`` grows. The residual gap is
resolution -- the free surface is smeared over the ~10 cells that span the
initial column.

**The runout does not match the experiment, and the reason is a missing
boundary condition, not the material.** Runout in this benchmark is controlled
by friction at the *base*, and ``mpm.boundary`` offers only two limits:

* ``sticky`` (no-slip) settles, but under-predicts the published runout by
  30-50% and gives ``dr/r0 ~ a^1.0`` where the experiment gives ``a^0.67``.
  No-slip is more friction than a real floor.
* ``slip`` (frictionless) does not settle **at all** -- with no basal traction
  there is nothing to arrest the flow, so there is no final runout to measure.
  A number read off at some arbitrary end time is not a result.

The experiment lives between the two, and reaching it needs a Coulomb friction
wall, which is a boundary condition rather than a constitutive model. Until
that exists this example is a demonstration of the material, and the repose
angle is the part to trust.

Outputs
-------
results/sand_column/particles.pvd    ParaView / OVITO time series
results/sand_column/collapse.png     contact sheet + the repose-angle sweep
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import (
    APIC,
    Box,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
)
from scion.discretization.mpm.output import ParticleOutput
from scion.materials.hencky import (
    DruckerPragerSand,
    HenckyElastic,
    PrincipalPlasticity,
    PrincipalPlasticState,
)

# The domain is wide and short on purpose: at 1.0 wide the a >= 3 runouts reach
# the far wall, and the saturation that produces looks exactly like a physical
# arrest until you check where the front is.
NY = 96
DX = 1.0 / NY
NX = 288                                    # 3.0 x 1.0
R0 = 0.10                                   # column half-width, ~10 cells
ASPECT = 2.0
RHO0 = 1500.0                               # dry sand
E, NU = 1.0e6, 0.3
FRICTION_ANGLE = 30.0
GRAVITY = np.array([0.0, -9.81])
N_STEPS = 6000

FLOOR = 2 * DX
LEFT = 2 * DX
X_MAX = NX * DX - 2 * DX

RESULTS = Path(__file__).resolve().parent / "results" / "sand_column"


def lube_runout(a: float) -> float:
    """Published 2-D scaling (Lube et al., Phys. Rev. E 72, 041301, 2005)."""
    return 1.2 * a if a < 1.8 else 1.9 * a ** (2.0 / 3.0)


# ---------------------------------------------------------------------------

def build(aspect: float = ASPECT, friction_angle: float = FRICTION_ANGLE):
    h0 = aspect * R0
    grid = CartesianGrid(shape=(NX, NY), dx=DX)

    material = PrincipalPlasticity(
        HenckyElastic.from_E_nu(E, NU),
        DruckerPragerSand(friction_angle=friction_angle),
    )

    def column(p):
        return ((p[:, 0] > LEFT) & (p[:, 0] < LEFT + R0)
                & (p[:, 1] > FLOOR) & (p[:, 1] < FLOOR + h0))

    particles = ParticleSet.from_sampler(
        column, ((0.0, 0.0), (NX * DX, NY * DX)), DX, ppc=2,
        velocity=(0.0, 0.0), density=RHO0,
    )

    walls = Box(lo=(LEFT, FLOOR), hi=(X_MAX, NY * DX - 2 * DX), mode="sticky")

    c = material.wave_speed_factor / np.sqrt(RHO0)
    stepper = ExplicitMPMStepper(
        grid, material, APIC(), dt=0.3 * DX / c,
        body_force=GRAVITY, boundaries=[walls],
        state=PrincipalPlasticState.identity(particles.n),
    )
    return grid, material, particles, stepper, h0


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def free_surface(particles: ParticleSet) -> np.ndarray:
    """Highest particle in each column of cells. ``(m, 2)``, floor-relative."""
    x = particles.x[:, 0] - LEFT
    y = particles.x[:, 1] - FLOOR
    edges = np.arange(0.0, x.max() + DX, DX)
    which = np.digitize(x, edges)
    return np.array([[edges[i - 1] + 0.5 * DX, y[which == i].max()]
                     for i in np.unique(which) if (which == i).sum() >= 2])


def repose_angle(particles: ParticleSet) -> float:
    """Slope of the settled free surface, in degrees.

    Fitted over the middle of the height range only: the crest is flat and the
    toe tapers into the floor, and including either biases the angle down.
    """
    surf = free_surface(particles)
    h = surf[:, 1]
    band = (h > 0.15 * h.max()) & (h < 0.85 * h.max())
    if band.sum() < 3:
        return float("nan")
    slope = np.polyfit(surf[band, 0], surf[band, 1], 1)[0]
    return float(np.degrees(np.arctan(abs(slope))))


def runout(particles: ParticleSet) -> float:
    """Front position in units of ``R0``. The 99.5th percentile, not the max,
    so one ejected particle does not set the answer."""
    return (np.percentile(particles.x[:, 0], 99.5) - LEFT) / R0


# ---------------------------------------------------------------------------

def friction_sweep(angles=(20.0, 30.0, 40.0), n_steps=N_STEPS):
    """Repose angle against input friction angle -- the material-level check."""
    out = []
    for phi in angles:
        _, _, particles, stepper, _ = build(friction_angle=phi)
        with quiet():
            stepper.run(particles, n_steps=n_steps)
        out.append((phi, repose_angle(particles),
                    float(np.abs(particles.v).max())))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(snapshots, sweep, path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return None

    times = sorted(snapshots)
    n = len(times)
    fig = plt.figure(figsize=(4.2 * min(n, 3), 3.0 * (1 + (n + 2) // 3)))
    gs = fig.add_gridspec((n + 2) // 3 + 1, 3, hspace=0.45, wspace=0.25)

    x_hi = max(s[:, 0].max() for s in snapshots.values())
    for k, t in enumerate(times):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        pts = snapshots[t]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, c="#c8a06a", linewidths=0)
        ax.set_xlim(0.0, x_hi + 4 * DX)
        ax.set_ylim(0.0, ASPECT * R0 * 1.15)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t:.2f} s", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[-1, :])
    phi = [s[0] for s in sweep]
    rep = [s[1] for s in sweep]
    ax.plot(phi, rep, "o-", color="#b4611f", label="measured repose angle")
    ax.plot(phi, phi, "k--", lw=1, label=r"$\theta = \phi$")
    ax.set_xlabel("input friction angle $\\phi$ [deg]")
    ax.set_ylabel("repose angle [deg]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Repose angle tracks the friction angle "
                 "(offset is resolution)", fontsize=10)

    path = Path(path or RESULTS / "collapse.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")
    return path


# ---------------------------------------------------------------------------

def main(write_vtk: bool = True, make_figure: bool = True, quick: bool = False):
    n_steps = 1500 if quick else N_STEPS
    grid, material, particles, stepper, h0 = build()

    print(f"domain {NX * DX:.1f} x {NY * DX:.1f} m, dx = {DX:.4f} "
          f"({R0 / DX:.0f} cells across the column)")
    print(f"particles: {particles.n}   dt: {stepper.dt:.2e}   "
          f"aspect a = {ASPECT}   phi = {FRICTION_ANGLE} deg")

    snapshots: dict[float, np.ndarray] = {}
    vtk_out = (ParticleOutput(RESULTS, body_force=GRAVITY, verbose=False)
               if write_vtk else None)

    def log(step, t, pts):
        snapshots[round(t, 3)] = pts.x.copy()
        if vtk_out is not None:
            vtk_out(step, t, pts)

    stepper.run(particles, n_steps=n_steps, callback=log,
                callback_every=max(1, n_steps // 5))

    r = runout(particles)
    v_max = float(np.abs(particles.v).max())
    hit_wall = particles.x[:, 0].max() > X_MAX - 3 * DX

    print()
    print(f"  settled          : |v|max = {v_max:.2e} m/s "
          f"({'yes' if v_max < 1e-2 else 'NOT SETTLED'})")
    print(f"  reached far wall : {hit_wall}")
    print(f"  runout dr/r0     : {r - 1:.3f}   "
          f"(Lube 2005: {lube_runout(ASPECT):.3f})")
    print(f"  final height     : {np.percentile(particles.x[:, 1], 99.5) - FLOOR:.4f} m "
          f"of {h0:.4f}")
    print(f"  repose angle     : {repose_angle(particles):.2f} deg "
          f"(phi = {FRICTION_ANGLE})")
    print()
    print("  runout falls short because a no-slip floor is more friction than")
    print("  a real one; see the module docstring.")

    sweep = []
    if not quick:
        print()
        print("friction sweep (the material-level check):")
        sweep = friction_sweep()
        for phi, theta, vm in sweep:
            print(f"  phi = {phi:4.1f} deg  ->  repose {theta:5.2f} deg   "
                  f"|v|max {vm:.1e}")

    if make_figure and snapshots:
        render(snapshots, sweep or [(FRICTION_ANGLE, repose_angle(particles), v_max)])
    return snapshots


if __name__ == "__main__":
    main(
        write_vtk="--no-vtk" not in sys.argv,
        make_figure="--no-figure" not in sys.argv,
        quick="--quick" in sys.argv,
    )
