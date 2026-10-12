"""Shared setup for the two micropolar phase-field examples.

Kept in one place because the two problems differ only in geometry and boundary
conditions: the grid construction, the plane-strain reduction, the seeded notch
and the load path are identical, and a divergence between them would be a
difference in the *method* rather than in the problem.
"""
from __future__ import annotations

import time
import os
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
                 max_iter=25, max_stagger=25, linear_solver=None):
    # linear_solver defaults to the stepper's own DirectSolver, which is what
    # every plane-strain example uses. It exists for the three-dimensional
    # example, where the direct factorisation's fill-in dominates.
    elastic = MicropolarHooke.from_technical(E=E, nu=nu, N=N, l_t=2.0 * l_b,
                                             l_b=l_b, psi=1.5)
    stepper = MicropolarMPMStepper(grid, elastic, dirichlet=bcs,
                                   max_iter=max_iter,
                                   linear_solver=linear_solver)
    stepper.prepare(parts)
    model = PhaseFieldModel(E, nu, gc, ell_pf, split="spectral",
                            coupling="hybrid", dissipation="AT2")
    return MicropolarPhaseFieldMPM(stepper, parts, model,
                                   max_stagger=max_stagger), stepper


def run_load_path(pf, stepper, parts, react_nodes, comp, n_steps,
                  min_fraction=1.0 / 64.0, growth=1.5, progress=None,
                  stop_past_peak=None, on_step=None, reaction=None,
                  checkpoint=None, resume=True):
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
    disp, load, dmax, shares, fields, extra = [], [], [], [], [], []
    lam, increment, floor = 0.0, 1.0 / n_steps, min_fraction / n_steps
    cutbacks, t0, past_peak = 0, time.time(), False
    # Resume, if a checkpoint from an interrupted run is present. The recorded
    # history is restored with it, so the returned result is the same object a
    # run without an interruption would have produced.
    if checkpoint is not None and resume and Path(checkpoint).exists():
        lam, increment, ex = load_checkpoint(checkpoint, pf, parts)
        stepper.prepare(parts)
        disp = list(ex["hist_disp"]); load = list(ex["hist_load"])
        dmax = list(ex["hist_dmax"]); cutbacks = int(ex["cutbacks"])
        fields = [np.asarray(f) for f in ex["hist_fields"]]
        shares = [dict(zip(ex["share_keys"].tolist(), row))
                  for row in ex["hist_shares"]]
        if "hist_extra" in ex and np.size(ex["hist_extra"]):
            extra = [np.asarray(r) for r in ex["hist_extra"]]
        if progress:
            print(f"      {progress}  resumed from {Path(checkpoint).name} at "
                  f"lambda {lam:5.3f} ({len(disp)} steps already done)",
                  flush=True)
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
            # ``reaction`` exists for torsion, where the conjugate quantity is a
            # torque rather than a force component and cannot be read off one
            # slice of the nodal force vector.
            load.append(float(f[react_nodes * 6 + comp].sum())
                        if reaction is None else float(reaction(f)))
            dmax.append(float(pf.d.max()))
            shares.append(pf.driving_shares())
            # the damage field at every converged step, so the figures can be
            # taken at a load level COMMON to every row. Comparing each row at
            # its own final state is not like-for-like when the rows stop at
            # different lambda, and it makes a row that merely stopped early
            # look like one that resisted cracking.
            fields.append(pf.tables.interpolate(pf.d).copy())
            # anything else a caller wants at every converged step -- the
            # per-particle energy densities for a map at the peak, say --
            # without this function having to know what
            if on_step is not None:
                extra.append(on_step(pf, stepper, parts))
            if checkpoint is not None:
                keys = list(shares[0].keys())
                save_checkpoint(
                    checkpoint, pf, parts, lam, increment,
                    hist_disp=np.array(disp), hist_load=np.array(load),
                    hist_dmax=np.array(dmax), cutbacks=cutbacks,
                    hist_fields=np.array(fields),
                    share_keys=np.array(keys),
                    hist_shares=np.array([[sh[k] for k in keys]
                                          for sh in shares]),
                    hist_extra=(np.array(extra) if extra else np.zeros(0)))
            increment = min(increment * growth, 1.0 / n_steps)
            # Optional early stop, once the load has fallen clear of its own
            # maximum. Everything reported *at the peak* -- the peak load
            # itself and the three driving-force shares -- is then already
            # determined, and what is skipped is the post-peak branch, which is
            # most of the cost because that is where the crack actually runs.
            #
            # The test is placed after the increment update and changes no
            # increment before it, so the state at every recorded step is
            # bit-identical to the same run without this argument. That is the
            # whole point: a cheaper run that returned slightly different
            # numbers would not be a substitute for the full one.
            if (stop_past_peak is not None and len(load) >= 3
                    and abs(load[-1]) < stop_past_peak * max(abs(v)
                                                             for v in load)):
                past_peak = True
            if progress:
                el = time.time() - t0
                print(f"      {progress}  lambda {lam:5.3f}  "
                      f"load {load[-1]:11.4g}  d_max {dmax[-1]:5.3f}  "
                      f"cuts {cutbacks:2d}  {el / 60:6.1f} min elapsed, "
                      f"~{el * (1.0 - lam) / max(lam, 1e-9) / 60:5.1f} min "
                      f"left{'  [past peak, stopping]' if past_peak else ''}",
                      flush=True)
            if past_peak:
                break
    if not disp:
        raise RuntimeError("no load step converged; the problem is not "
                           "solvable as posed, not merely badly stepped")
    return {"disp": np.array(disp), "load": np.array(load),
            "max_damage": np.array(dmax), "shares": shares,
            "reached": lam, "cutbacks": cutbacks, "fields": fields,
            "stopped_past_peak": past_peak, "extra": extra,
            "seconds": time.time() - t0}

# ---------------------------------------------------------------------------
# checkpointing
#
# A 39-hour run that writes its results only at the end loses everything to a
# shutdown, and one of these was lost at 17 hours to exactly that. The state
# that has to survive is the particle state, the phase field and the load
# factor; the stepper is rebuilt from the particles by ``prepare``, so it needs
# nothing of its own. Saved after every converged step, which costs a few
# seconds against a load step of over an hour.
# ---------------------------------------------------------------------------

#: Where checkpoints go. Deliberately *not* ``RESULTS``: that sits under
#: ~/Desktop, which macOS syncs to iCloud by default, and writing a 200 MB
#: checkpoint there after every load step both risks the rename above and feeds
#: a sync queue for no reason. Override with SCION_CKPT_DIR.
CHECKPOINT_DIR = Path(os.environ.get(
    "SCION_CKPT_DIR", "/tmp/scion_checkpoints"))
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


#: Particle arrays carrying history. ``x`` and ``F`` are the kinematics, the
#: quaternion and wryness the micropolar state, and the plastic fields are
#: present but unused by the elastic-plus-damage path; they are saved anyway so
#: a checkpoint does not silently become invalid if a plastic model is used.
_STATE_FIELDS = ("x", "X", "F", "C", "v", "vol0", "mass", "quat", "wryness",
                 "F_plastic", "wryness_plastic", "hardening")


def save_checkpoint(path, pf, parts, lam, increment, **extra):
    """Everything needed to resume, written atomically."""
    import os
    data = {f"p_{k}": np.asarray(getattr(parts, k))
            for k in _STATE_FIELDS if hasattr(parts, k)}
    data.update(d=np.asarray(pf.d), H=np.asarray(pf.H),
                lam=float(lam), increment=float(increment), **extra)
    # The suffix must stay ".npz": savez_compressed appends one when the name
    # lacks it, and the rename then looks for a file that was never written.
    #
    # The rename is refused intermittently on an iCloud-synced directory --
    # ~/Desktop is one by default -- and that killed a 14-hour run at its fifth
    # checkpoint with EPERM. Two defences. Prefer a directory outside the synced
    # tree (see CHECKPOINT_DIR), and if the rename is refused anyway, fall back
    # to writing in place: that loses atomicity for one save, which is strictly
    # better than losing the run. The window is small and the next step rewrites
    # it; a torn checkpoint is detected on load because np.load raises.
    tmp = Path(str(path) + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    try:
        os.replace(tmp, path)      # atomic, so a crash mid-write cannot corrupt
    except (PermissionError, OSError) as exc:
        print(f"      [checkpoint] rename refused ({exc}); writing in place. "
              f"Is {Path(path).parent} inside an iCloud-synced folder?",
              flush=True)
        np.savez_compressed(path, **data)
        tmp.unlink(missing_ok=True)


def load_checkpoint(path, pf, parts):
    """Restore in place; returns ``(lam, increment, extra)``.

    The caller must call ``stepper.prepare(parts)`` afterwards -- the stencil is
    a function of the particle positions and is not saved.
    """
    m = np.load(path, allow_pickle=False)
    for k in _STATE_FIELDS:
        key = f"p_{k}"
        if key in m.files and hasattr(parts, k):
            getattr(parts, k)[...] = m[key]
    pf.d[...] = m["d"]
    pf.H[...] = m["H"]
    extra = {k: m[k] for k in m.files
             if not k.startswith("p_") and k not in ("d", "H", "lam", "increment")}
    return float(m["lam"]), float(m["increment"]), extra
