"""Shared setup for the two micropolar phase-field examples.

Kept in one place because the two problems differ only in geometry and boundary
conditions: the grid construction, the plane-strain reduction, the seeded notch
and the load path are identical, and a divergence between them would be a
difference in the *method* rather than in the problem.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scion.core.logging import quiet
from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.discretization.mpm.micropolar import (MicropolarMPMStepper,
                                                 attach_micropolar_state)
from scion.materials.fracture import PhaseFieldModel
from scion.materials.micropolar import MicropolarHooke
from scion.multiphysics.micropolar_phase_field import MicropolarPhaseFieldMPM

RESULTS = Path(__file__).resolve().parent / "results"

PAD = 3          # in-plane grid rings outside the body; the body may bulge
NZ = 1           # one cell through the thickness -- see plane_strain_bc

#: Padding through the thickness. One ring, not ``PAD``: the body is one cell
#: thick and cannot move in z at all (``plane_strain_bc`` pins it), so the only
#: requirement is that the linear basis, which reaches one cell, has somewhere
#: to write. Padding z like the in-plane directions gives 7 node layers where 3
#: carry any support -- 2.3x the nodes, and every one of them enters assembly
#: and the sparsity pattern. On the notched plate that is 52k nodes instead of
#: 22k, for nothing.
ZPAD = 1

#: Resolution floor. An internal length spread over fewer cells than this is
#: not resolved, and the answer then moves with the *particle count* at fixed
#: grid rather than converging: measured at a factor of 2.3 for l_b/dx = 1,
#: 3% at 2 and 1.7% at 4. Below it the numbers are meaningless rather than
#: merely inaccurate, which is why this is checked and not just documented.
MIN_CELLS_PER_LENGTH = 4.0


def check_resolution(dx: float, ell_pf: float, l_b: float) -> None:
    """Refuse a discretisation that cannot represent its own length scales."""
    for name, val in (("l (phase field)", ell_pf), ("l_b (micropolar)", l_b)):
        if val <= 0.0:
            continue                      # l_b = 0 is the classical limit
        if val / dx < MIN_CELLS_PER_LENGTH - 1e-9:
            raise ValueError(
                f"{name} = {val:g} is only {val / dx:.2f} cells at "
                f"dx = {dx:g}; at least {MIN_CELLS_PER_LENGTH:g} are needed. "
                f"Refine to dx <= {val / MIN_CELLS_PER_LENGTH:g}, or raise the "
                f"length. Results below this floor move with the particle "
                f"count instead of converging.")


def plane_strain_bc():
    """Constrain the out-of-plane translation and the two in-plane rotations.

    The micropolar stepper is 3-D by construction, because SO(3) is the point:
    a plane-strain Cosserat continuum is a genuine three-degree-of-freedom
    reduction and a different discretisation. It is obtained here by
    constraining the components that the reduction removes -- ``u_z`` and the
    rotations about ``x`` and ``y`` -- leaving the drilling rotation ``phi_z``
    free, which is the one a two-dimensional Cosserat solid keeps.
    """
    nan = np.nan
    return GridDirichlet(lambda c: np.ones(len(c), bool),
                         np.array([nan, nan, 0.0, 0.0, 0.0, nan]))


def build_grid(lengths, dx: float):
    """Background grid and a lattice of particles filling ``lengths``."""
    nx, ny = (int(round(L / dx)) for L in lengths[:2])
    grid = CartesianGrid(shape=(nx + 2 * PAD, ny + 2 * PAD, NZ + 2 * ZPAD),
                         dx=dx,
                         origin=np.array([-PAD * dx, -PAD * dx, -ZPAD * dx]),
                         basis=get_basis("linear"))
    return grid, nx, ny


def seed_particles(lengths, dx: float, ppc: int):
    sub = dx / ppc
    ax = [np.arange(sub / 2, e, sub)
          for e in (lengths[0], lengths[1], NZ * dx)]
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")], axis=1)
    return attach_micropolar_state(ParticleSet(pts, vol0=sub ** 3,
                                               density=1.0))


def seed_notch(pf, parts, dx, ell_pf, gc, tip, along,
               amplitude=1.0e3, width_factor=0.25):
    """Impose the notch as an initial history field, not by cutting the body.

    Following Borden et al. (2012). ``along`` selects the axis the notch runs
    along and ``tip`` its tip; the distance is measured to the *segment*, so
    the field is a half-plane behind the tip and radial ahead of it.

    The seeded width is ``max(0.3 l, dx)``: floored at the cell size so the
    band cannot fall between particles, and capped well below ``l`` so the
    notch is a crack rather than a fat damaged strip -- with a wide band the
    damage saturates across the whole of it and the regularised surface energy
    comes out far too high.
    """
    x = parts.x[:, :2]
    a, b = (0, 1) if along == "x" else (1, 0)
    behind = x[:, a] <= tip[a]
    dist = np.where(behind, np.abs(x[:, b] - tip[b]),
                    np.hypot(x[:, 0] - tip[0], x[:, 1] - tip[1]))
    width = max(width_factor * ell_pf, dx)
    H0 = np.zeros(parts.n)
    inside = dist <= width
    H0[inside] = amplitude * gc / (2.0 * ell_pf) * (1.0 - dist[inside] / width)
    pf.H = H0
    return pf


def make_problem(grid, parts, *, E, nu, N, l_b, gc, ell_pf, bcs,
                 max_iter=25, max_stagger=25):
    elastic = MicropolarHooke.from_technical(E=E, nu=nu, N=N, l_t=2.0 * l_b,
                                             l_b=l_b, psi=1.5)
    stepper = MicropolarMPMStepper(grid, elastic, dirichlet=bcs,
                                   max_iter=max_iter)
    stepper.prepare(parts)
    model = PhaseFieldModel(E, nu, gc, ell_pf, split="spectral",
                            coupling="hybrid", dissipation="AT2")
    return MicropolarPhaseFieldMPM(stepper, parts, model,
                                   max_stagger=max_stagger), stepper


def run_load_path(pf, stepper, parts, react_nodes, comp, n_steps,
                  min_fraction=1.0 / 64.0, growth=1.5, progress=None):
    """Step to the full prescribed displacement, recording the load path.

    Adaptive, and it has to be. Past the peak load the crack releases more
    energy in one increment than it can absorb, and the failure is not a
    tolerance that can be loosened: once a band is fully broken the degraded
    tangent is near-singular, the Newton increment is noise, and the *energy*
    half of the stepper's convergence test can never be satisfied however small
    the residual gets. The only remedy is a smaller increment, so a failed step
    is rolled back and retried at half the size, down to ``min_fraction`` of
    the nominal one.

    Rolling back means the phase field too. ``stepper.step`` restores its own
    state when it raises, but ``d`` and ``H`` have already been advanced by the
    staggered sweeps that preceded the failure, and ``H`` is monotone by
    construction -- so leaving them would carry damage from a rejected step
    into the retry, permanently.

    The reaction is the internal force on the loaded nodes, which at a
    converged state differs from the residual only by the prescribed reaction.
    The three driving-force shares are recorded per step so the curvature
    contribution can be read at the *peak* rather than at the end, where the
    specimen has already lost most of its stiffness.
    """
    disp, load, dmax, shares, fields = [], [], [], [], []
    lam, increment, floor = 0.0, 1.0 / n_steps, min_fraction / n_steps
    cutbacks, t0 = 0, time.time()
    with quiet():
        while lam < 1.0 - 1e-12:
            step = min(increment, 1.0 - lam)
            snap = stepper.snapshot(parts)
            d_save, h_save = pf.d.copy(), pf.H.copy()
            try:
                pf.solve_step(lam + step, step)
            except RuntimeError:
                stepper.restore(parts, snap)
                pf.d, pf.H = d_save, h_save
                increment *= 0.5
                cutbacks += 1
                if increment < floor:
                    break          # the path has genuinely run out
                continue
            lam += step
            f = stepper.internal_force(parts)
            disp.append(lam)
            load.append(float(f[react_nodes * 6 + comp].sum()))
            dmax.append(float(pf.d.max()))
            shares.append(pf.driving_shares())
            # the damage field at every converged step, so the figures can be
            # taken at a load level COMMON to every row. Comparing each row at
            # its own final state is not like-for-like when the rows stop at
            # different lambda, and it makes a row that merely stopped early
            # look like one that resisted cracking.
            fields.append(pf.tables.interpolate(pf.d).copy())
            increment = min(increment * growth, 1.0 / n_steps)
            if progress:
                el = time.time() - t0
                print(f"      {progress}  lambda {lam:5.3f}  "
                      f"load {load[-1]:11.4g}  d_max {dmax[-1]:5.3f}  "
                      f"cuts {cutbacks:2d}  {el / 60:6.1f} min elapsed, "
                      f"~{el * (1.0 - lam) / max(lam, 1e-9) / 60:5.1f} min "
                      f"left", flush=True)
    if not disp:
        raise RuntimeError("no load step converged; the problem is not "
                           "solvable as posed, not merely badly stepped")
    return {"disp": np.array(disp), "load": np.array(load),
            "max_damage": np.array(dmax), "shares": shares,
            "reached": lam, "cutbacks": cutbacks, "fields": fields,
            "seconds": time.time() - t0}
