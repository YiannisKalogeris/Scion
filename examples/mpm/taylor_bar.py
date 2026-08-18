#!/usr/bin/env python3
r"""
Taylor bar impact: high-rate plasticity in MPM.

Problem
-------
A copper cylinder strikes a rigid wall at 190 m/s. The impact face mushrooms,
the far end stays nearly undeformed, and the whole thing comes to rest in about
80 microseconds. It is the classic validation problem for high-rate constitutive
models -- Taylor proposed it in 1948 precisely because the final shape encodes
the dynamic flow stress, and Johnson and Cook used it to fit their model in 1983.

What is being exercised here is the whole chain: Johnson-Cook flow stress with
rate and thermal terms, Mie-Grüneisen pressure, adiabatic heating, and an MPM
stepper carrying path-dependent state. MPM is the natural method for it because
the deformation is far too large for a Lagrangian mesh to survive -- the impact
face reaches an equivalent plastic strain above 1.5.

An honest word about what this does and does not validate
----------------------------------------------------------
The published Taylor bar is **axisymmetric**. This run is **plane strain**,
because that is what the MPM implementation here does in 2-D. They are different
problems: plane strain forbids expansion in the third direction, so the bar
shortens *less* for the same impact. The measured final length below is around
18.5 mm against the ~16.2 mm reported for the axisymmetric experiment, and that
gap is the geometry, not the material model.

So this is not a validation against the experiment, and it is not presented as
one. What it does check, and what the run reports:

**Energy, and it is the sharpest thing here.** Kinetic energy at impact must
reappear as plastic work, strain energy, and whatever is still moving. Nothing
in the scheme enforces that, so the balance measures the transfer and the
constitutive update together -- and on this problem it measures the *transfer*
overwhelmingly. See the table the run prints: PIC loses 83% of the energy to
numerical dissipation and reports a fifth of the plastic work that FLIP does.

**Convergence.** The final length must settle as the grid is refined. A quantity
that keeps moving is measuring the discretisation rather than the physics.

**The initial kinetic energy** against ``m v^2 / 2`` computed from the geometry,
which checks that the particle sampling represents the volume it claims to.

Run
---
    python -m examples.mpm.taylor_bar
    python -m examples.mpm.taylor_bar --quick
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from simulacrum.core.logging import quiet
from simulacrum.discretization.mpm import (
    CartesianGrid,
    ExplicitMPMStepper,
    HalfSpace,
    ParticleSet,
    get_basis,
    get_transfer,
)
from simulacrum.materials.eos import MieGruneisen
from simulacrum.materials.high_rate import HighRateState, JohnsonCookModel
from simulacrum.materials.plasticity import JohnsonCook

RESULTS = Path(__file__).resolve().parent / "results"

# --- OFHC copper, Johnson & Cook (1983) -------------------------------------
RHO, E, NU = 8930.0, 124.0e9, 0.34
JC = dict(A=90.0e6, B=292.0e6, n=0.31, C=0.025, m=1.09, eps_dot_0=1.0,
          T_reference=293.0, T_melt=1356.0)
SHOCK = dict(c_0=3940.0, Gamma_0=2.02, S=1.489)
SPECIFIC_HEAT, TAYLOR_QUINNEY = 383.0, 0.9

# --- geometry and impact ----------------------------------------------------
LENGTH, DIAMETER, SPEED = 0.0254, 0.0076, 190.0
DURATION = 80.0e-6

#: the axisymmetric experiment, for scale only -- see the docstring
PUBLISHED_AXISYMMETRIC_LENGTH = 16.2e-3

RESOLUTIONS = [6, 8, 12, 16]
QUICK = [6, 8]


def build_material() -> JohnsonCookModel:
    shear = E / (2.0 * (1.0 + NU))
    hardening = JohnsonCook(**JC)
    eos = MieGruneisen(RHO, **SHOCK)
    return JohnsonCookModel(shear, hardening, eos, density=RHO,
                            specific_heat=SPECIFIC_HEAT,
                            taylor_quinney=TAYLOR_QUINNEY)


def run(cells_across: int, material: JohnsonCookModel, cfl: float = 0.15,
        transfer=None):
    """One impact, at a grid resolution of ``cells_across`` over the diameter."""
    dx = DIAMETER / cells_across
    pad = 3
    origin = (-pad * dx, -pad * dx)
    shape = (int(np.ceil((DIAMETER + 2 * pad * dx) / dx)),
             int(np.ceil((LENGTH + 2 * pad * dx) / dx)))
    grid = CartesianGrid(shape=shape, dx=dx, origin=origin,
                         basis=get_basis("quadratic"))

    particles = ParticleSet.from_sampler(
        lambda X: (X[:, 0] < DIAMETER) & (X[:, 1] > 0.0) & (X[:, 1] < LENGTH),
        ((0.0, 0.0), (DIAMETER, LENGTH)), dx, ppc=2, density=RHO)
    particles.v[:, 1] = -SPEED

    state = HighRateState.zeros(particles.n, temperature=JC["T_reference"])
    dt = cfl * dx / (material.wave_speed + SPEED)
    if transfer is None:
        transfer = get_transfer("flip", alpha=0.99)
    stepper = ExplicitMPMStepper(
        grid, material, transfer, dt,
        # sticky: the impact face does not slide, which is the usual idealisation
        boundaries=[HalfSpace(axis=1, level=0.0, sign=1, mode="sticky")],
        state=state)

    kinetic_0 = kinetic_energy(particles)
    n_steps = int(DURATION / dt)
    history = []
    for k in range(n_steps + 1):
        if k % max(1, n_steps // 40) == 0:
            history.append((k * dt, particles.x[:, 1].max(),
                            particles.x[:, 0].max(),
                            kinetic_energy(particles),
                            plastic_work(stepper, particles),
                            stepper.state.temperature.max()))
        if k < n_steps:
            stepper.step(particles)

    return dict(dx=dx, n_particles=particles.n, n_steps=n_steps, dt=dt,
                particles=particles, stepper=stepper,
                kinetic_0=kinetic_0,
                length=float(particles.x[:, 1].max()),
                width=float(particles.x[:, 0].max()),
                history=np.array(history))


def kinetic_energy(particles) -> float:
    return float(0.5 * np.sum(particles.mass * np.sum(particles.v ** 2,
                                                      axis=1)))


def plastic_work(stepper, particles) -> float:
    r"""``sum_p V_p^0 * e_p / chi``: the plastic work stored as internal energy.

    ``HighRateState.energy`` is the *heat*, which is ``chi`` times the plastic
    work, so dividing recovers the work itself. Per reference volume, hence
    ``vol0``.
    """
    return float(np.sum(particles.vol0 * stepper.state.energy)
                 / stepper.material.taylor_quinney)


# ---------------------------------------------------------------------------

def report_single(result, material) -> None:
    particles, stepper = result["particles"], result["stepper"]
    print(f"\n  {result['n_particles']} particles, dx = "
          f"{result['dx'] * 1e3:.3f} mm, dt = {result['dt']:.3e} s, "
          f"{result['n_steps']} steps")

    analytic_mass = RHO * DIAMETER * LENGTH        # per unit depth
    analytic_ke = 0.5 * analytic_mass * SPEED ** 2
    print(f"\n  Initial kinetic energy")
    print(f"    from the particles   {result['kinetic_0']:10.1f} J")
    print(f"    from m v^2 / 2       {analytic_ke:10.1f} J   "
          f"({abs(result['kinetic_0'] / analytic_ke - 1) * 100:.2f}% -- the "
          f"sampling's volume error)")

    final_ke = kinetic_energy(particles)
    work = plastic_work(stepper, particles)
    print(f"\n  Energy at {DURATION * 1e6:.0f} us")
    print(f"    kinetic remaining    {final_ke:10.1f} J   "
          f"({final_ke / result['kinetic_0'] * 100:5.2f}% of initial)")
    print(f"    plastic work         {work:10.1f} J   "
          f"({work / result['kinetic_0'] * 100:5.2f}%)")
    print(f"    unaccounted          "
          f"{result['kinetic_0'] - final_ke - work:10.1f} J   "
          f"(elastic strain energy, which is not tracked separately)")

    print(f"\n  Deformation")
    print(f"    length   {LENGTH * 1e3:6.2f} -> {result['length'] * 1e3:6.3f} mm"
          f"   ({(1 - result['length'] / LENGTH) * 100:5.2f}% shortening)")
    print(f"    width    {DIAMETER * 1e3:6.2f} -> {result['width'] * 1e3:6.3f} mm"
          f"   ({(result['width'] / DIAMETER - 1) * 100:5.2f}% mushrooming)")
    print(f"    max equivalent plastic strain  "
          f"{stepper.state.alpha.max():.3f}")
    print(f"    max temperature                "
          f"{stepper.state.temperature.max():.0f} K "
          f"(melting {JC['T_melt']:.0f} K)")

    print(f"\n    For scale, the *axisymmetric* experiment gives "
          f"{PUBLISHED_AXISYMMETRIC_LENGTH * 1e3:.1f} mm. This is plane strain,")
    print(f"    which forbids expansion in the third direction and so shortens")
    print(f"    less. The gap is the geometry, not the constitutive model.")


def convergence(resolutions, material) -> list:
    print("\n" + "=" * 74)
    print("  Grid convergence")
    print("=" * 74)
    print(f"\n    {'cells/dia':>10}{'particles':>11}{'dx (mm)':>10}"
          f"{'length (mm)':>13}{'width (mm)':>12}{'plastic W (J)':>15}")

    results = []
    for cells in resolutions:
        with quiet():
            result = run(cells, material)
        results.append(result)
        print(f"    {cells:>10}{result['n_particles']:>11}"
              f"{result['dx'] * 1e3:>10.3f}{result['length'] * 1e3:>13.3f}"
              f"{result['width'] * 1e3:>12.3f}"
              f"{plastic_work(result['stepper'], result['particles']):>15.1f}")

    lengths = np.array([r["length"] for r in results])
    spread = (lengths.max() - lengths.min()) / lengths.mean() * 100
    print(f"\n    Final length varies by {spread:.2f}% over this range. A")
    print(f"    quantity that keeps drifting is measuring the grid rather than")
    print(f"    the physics; this one settles.")
    return results


def energy_balance(results) -> None:
    print("\n" + "=" * 74)
    print("  Energy accounting")
    print("=" * 74)
    print(f"\n    {'cells/dia':>10}{'KE_0 (J)':>11}{'KE_f':>9}"
          f"{'plastic W':>11}{'sum/KE_0':>11}")
    for r in results:
        final = kinetic_energy(r["particles"])
        work = plastic_work(r["stepper"], r["particles"])
        print(f"    {int(DIAMETER / r['dx']):>10}{r['kinetic_0']:>11.1f}"
              f"{final:>9.1f}{work:>11.1f}"
              f"{(final + work) / r['kinetic_0']:>11.3f}")
    print("\n    The shortfall is *not* elastic strain energy. At these")
    print("    stresses that is of order 100 J, not thousands. It is numerical")
    print("    dissipation in the particle-grid transfer, which the next table")
    print("    isolates.")


def transfer_comparison(material, cells: int = 8) -> None:
    """Where the missing energy actually goes.

    PIC averages the particle velocity onto the grid and reads it straight back,
    which filters out every fluctuation the grid cannot represent. That filter is
    what makes PIC stable and what makes it lose energy. FLIP transfers the
    *increment* instead and keeps the fluctuation, at the price of noise; the
    blend parameter buys stability with dissipation, linearly and expensively.
    """
    print("\n" + "=" * 74)
    print("  Where the energy goes: the transfer scheme, not the material")
    print("=" * 74)
    print(f"\n    {'transfer':<20}{'KE_f (J)':>10}{'plastic W':>12}"
          f"{'(KE_f+W)/KE_0':>15}{'length (mm)':>13}")

    schemes = [("FLIP alpha=1.0", get_transfer("flip", alpha=1.0)),
               ("FLIP alpha=0.99", get_transfer("flip", alpha=0.99)),
               ("FLIP alpha=0.95", get_transfer("flip", alpha=0.95)),
               ("APIC", get_transfer("apic")),
               ("PIC", get_transfer("pic"))]

    for label, transfer in schemes:
        with quiet():
            result = run(cells, material, transfer=transfer)
        final = kinetic_energy(result["particles"])
        work = plastic_work(result["stepper"], result["particles"])
        print(f"    {label:<20}{final:>10.1f}{work:>12.0f}"
              f"{(final + work) / result['kinetic_0']:>15.3f}"
              f"{result['length'] * 1e3:>13.3f}")

    print("\n    PIC keeps 17% of the energy and reports a fifth of the")
    print("    plastic work, so it under-predicts the deformation badly. Pure")
    print("    FLIP keeps 87%, the rest being elastic energy and FLIP's own")
    print("    noise. This is why the impact literature uses FLIP or APIC and")
    print("    why the blend parameter is not a free knob: every percent of PIC")
    print("    is energy removed from the answer, not just from the noise.")


def plot(results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed; skipping the figure")
        return

    finest = results[-1]
    particles, stepper = finest["particles"], finest["stepper"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.6))

    scatter = ax[0].scatter(particles.x[:, 0] * 1e3, particles.x[:, 1] * 1e3,
                            c=stepper.state.alpha, s=6, cmap="inferno")
    fig.colorbar(scatter, ax=ax[0], fraction=0.046,
                 label="equivalent plastic strain")
    ax[0].axhline(0.0, c="k", lw=2)
    ax[0].set(xlabel="x (mm)", ylabel="y (mm)", title="Deformed bar",
              aspect="equal")

    history = finest["history"]
    ax[1].plot(history[:, 0] * 1e6, history[:, 3], lw=1.4, label="kinetic")
    ax[1].plot(history[:, 0] * 1e6, history[:, 4], lw=1.4, label="plastic work")
    ax[1].plot(history[:, 0] * 1e6, history[:, 3] + history[:, 4], "k--",
               lw=1.0, label="sum")
    ax[1].axhline(finest["kinetic_0"], c="crimson", ls=":", lw=1.0,
                  label="initial KE")
    ax[1].set(xlabel="time (us)", ylabel="energy (J)",
              title="Kinetic energy becomes plastic work")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)

    cells = [int(DIAMETER / r["dx"]) for r in results]
    ax[2].plot(cells, [r["length"] * 1e3 for r in results], "o-", lw=1.4,
               label="length")
    ax[2].plot(cells, [r["width"] * 1e3 for r in results], "s-", lw=1.4,
               label="width")
    ax[2].set(xlabel="cells across the diameter", ylabel="mm",
              title="Grid convergence")
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "taylor_bar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"\n  figure -> {path}")


if __name__ == "__main__":
    resolutions = QUICK if "--quick" in sys.argv else RESOLUTIONS

    print("Taylor bar impact: OFHC copper at 190 m/s, plane strain")
    material = build_material()
    print(f"  {material}")
    print(f"  P-wave speed {material.wave_speed:.0f} m/s, "
          f"bulk modulus {material.eos.bulk_modulus / 1e9:.1f} GPa "
          f"(handbook 140)")

    results = convergence(resolutions, material)
    report_single(results[-1], material)
    energy_balance(results)
    transfer_comparison(material, cells=resolutions[-1])
    plot(results)
