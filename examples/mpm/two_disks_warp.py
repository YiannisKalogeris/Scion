"""
Two-disk collision on the Warp backend, with a NumPy cross-check.

Same benchmark as ``two_disks.py`` -- Sulsky's colliding elastic disks -- run
through ``WarpMPMStepper`` instead of ``ExplicitMPMStepper``. Momentum is
conserved exactly with no forcing, so the physics check is unchanged; what this
example adds is a like-for-like timing and an agreement check between the two
backends.

Run:  python -m examples.mpm.two_disks_warp                  # auto device
      python -m examples.mpm.two_disks_warp --device=cpu
      python -m examples.mpm.two_disks_warp --device=gpu
      python -m examples.mpm.two_disks_warp --compare        # + NumPy timing
      python -m examples.mpm.two_disks_warp --ppc=8          # more particles

On a Mac, ``--device=gpu`` raises: Warp is CPU-only there. That is deliberate --
falling back silently would hide a large slowdown.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from simulacrum.discretization.mpm import (
    APIC,
    CartesianGrid,
    ExplicitMPMStepper,
    ParticleSet,
)
from simulacrum.discretization.mpm.warp import (
    WarpMPMStepper,
    WarpParticleSet,
    device_info,
    get_device,
    sync,
)
from simulacrum.materials.hyperelastic_batched import NeoHookeanCompressible

N_GRID = 64
DX = 1.0 / N_GRID
E, NU, RHO0 = 1.0e3, 0.3, 1.0e3
RADIUS = 0.2
C1, C2 = np.array([0.3, 0.3]), np.array([0.7, 0.7])
V1, V2 = (0.1, 0.1), (-0.1, -0.1)
DT = 1.0e-3
T_END = 3.0


def seed(ppc: int = 4) -> ParticleSet:
    bounds = ((0.0, 0.0), (1.0, 1.0))
    d1 = ParticleSet.from_sampler(
        lambda p: np.linalg.norm(p - C1, axis=1) < RADIUS,
        bounds, DX, ppc=ppc, velocity=V1, density=RHO0)
    d2 = ParticleSet.from_sampler(
        lambda p: np.linalg.norm(p - C2, axis=1) < RADIUS,
        bounds, DX, ppc=ppc, velocity=V2, density=RHO0)
    return d1.concatenate(d2)


def run_warp(device: str, ppc: int, n_steps: int, verbose: bool = True):
    mat = NeoHookeanCompressible.from_E_nu(E, NU)
    p = WarpParticleSet.from_numpy(seed(ppc), device)
    stepper = WarpMPMStepper((N_GRID, N_GRID), DX, mat.mu, mat.lam, DT,
                             device=device)

    # first launch pays JIT compilation; time the steady state instead
    stepper.step(p)
    p = WarpParticleSet.from_numpy(seed(ppc), device)

    p0 = p.linear_momentum()
    t0 = time.time()
    for s in range(n_steps + 1):
        if verbose and s % (n_steps // 6) == 0:
            sync(device)
            print(f"  t={s * DT:5.2f}  |p - p0| = "
                  f"{np.linalg.norm(p.linear_momentum() - p0):.3e}   "
                  f"KE = {p.kinetic_energy():.6e}   "
                  f"minJ = {p.jacobian().min():.4f}")
        if s < n_steps:
            stepper.step(p)
    sync(device)
    return p, time.time() - t0


def run_numpy(ppc: int, n_steps: int):
    mat = NeoHookeanCompressible.from_E_nu(E, NU)
    p = seed(ppc)
    grid = CartesianGrid(shape=(N_GRID, N_GRID), dx=DX)
    stepper = ExplicitMPMStepper(grid, mat, APIC(), dt=DT)
    t0 = time.time()
    for _ in range(n_steps):
        stepper.step(p)
    return p, time.time() - t0


def main(device: str = "auto", ppc: int = 4, compare: bool = False):
    dev = get_device(device)
    info = device_info()
    print(f"warp {info['warp_version']}   device = {dev}   "
          f"cuda available = {info['cuda_available']}")

    n_steps = int(T_END / DT)
    pw, tw = run_warp(dev, ppc, n_steps)
    print(f"\nwarp ({dev}): {pw.n} particles, {n_steps} steps in {tw:.2f} s "
          f"({1e6 * tw / (pw.n * n_steps):.3f} us/particle-step)")

    if compare:
        pn, tn = run_numpy(ppc, n_steps)
        r = pw.to_numpy()
        scale = np.abs(pn.x).max()
        print(f"numpy: {tn:.2f} s      speedup {tn / tw:.1f}x")
        print(f"agreement:  |dx|/scale = {np.abs(r['x'] - pn.x).max() / scale:.2e}"
              f"   |dF| = {np.abs(r['F'] - pn.F).max():.2e}")
        print("(float32 vs float64 -- ~1e-5 relative is the floor, not an error)")
    return pw


if __name__ == "__main__":
    dev = "auto"
    ppc = 4
    for a in sys.argv[1:]:
        if a.startswith("--device="):
            dev = a.split("=")[1]
        elif a.startswith("--ppc="):
            ppc = int(a.split("=")[1])
    main(device=dev, ppc=ppc, compare="--compare" in sys.argv)
