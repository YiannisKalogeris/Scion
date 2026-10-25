r"""Torsion of a circular micropolar bar: the one configuration whose rotation
axis varies by construction, with an exact classical limit to check it against.

    python -m examples.micropolar_phase_field.torsion_3d --verify
    python -m examples.micropolar_phase_field.torsion_3d --size
    python -m examples.micropolar_phase_field.torsion_3d --time

Why torsion. Every plane-strain example here has ``phi_x = phi_y = 0``, so
``Q`` is a rotation about ``e_3`` everywhere and the wryness update cannot be
distinguished from its first-order alternatives. Giving a plate a thickness
helps only through its free surfaces. Torsion is different in kind: the
classical rotation field of a bar twisted about ``e_3`` is

    omega ~ ( -tau x / 2 ,  -tau y / 2 ,  tau z )

for twist rate ``tau``, so the rotation *axis* depends on position across the
section, and the wryness

    Gamma ~ diag( -tau/2, -tau/2, tau )

is symmetric, traceless and nonzero. Two consequences worth stating. The
curvature energy is excited in the part of it this work actually retains: with
``l_t = 2 l_b`` and ``psi = 3/2`` the curvature energy reduces to
``beta ||dev sym Gamma||^2``, whose null space is the purely skew and purely
spherical wryness, and the torsion field lies in neither. And torsion is the
configuration in which the Cosserat length was first *measured* -- Gauthier &
Jahsman's size-effect experiment on cylinders, and Lakes' later work on
cellular solids -- so the size effect here is the one with an experimental
tradition behind it.

**The circular section is chosen for the benchmark, not for looks.**
Saint-Venant torsion of a circle has zero warping, so imposing a rigid-body
twist at both ends *is* the exact solution and the classical torsional rigidity

    K = T L / Theta = G J ,     J = pi a^4 / 2

holds for any length. A square section would need the tabulated 0.1406
coefficient and would also restrain warping at the clamped ends, putting a
length-dependent error of a few percent on the very quantity being checked.
The price is a staircased lattice boundary, which misrepresents the outermost
material where ``r^2`` is largest, so ``--verify`` reports the discrete second
polar moment of the particle set against ``pi a^4 / 2`` as well. That number is
the floor on the benchmark: a rigidity error no smaller than it says nothing
about the solver.

**The microrotation is left free at both ends.** Prescribing it on a boundary
that carries couple stress manufactures a boundary layer, which is precisely
what contaminated an earlier shear-layer study in this project. The
translations are prescribed at both ends and the microrotation at neither, so
the reported torque is a pure force moment with no couple reaction to
disentangle. ``--clamp-rotation`` imposes it instead, and the difference is
worth measuring rather than assuming.
"""
from __future__ import annotations

import sys
import time
from collections import deque

import numpy as np

from scion.core.logging import quiet
from scion.core.rotation import rotvec_from_quat
from scion.discretization.mpm import (CartesianGrid, GridDirichlet,
                                      ParticleSet, get_basis)
from scion.discretization.mpm.micropolar import attach_micropolar_state

from ._common import PAD, RESULTS, make_problem, run_load_path

# --- material ---------------------------------------------------------------
E_MOD, NU, N_COUPLING = 10.0e6, 0.25, 0.5
GC = 5.0
MU = E_MOD / (2.0 * (1.0 + NU))

# --- geometry and discretisation --------------------------------------------
RADIUS = 0.10
LENGTH_OVER_A = 3.0
PPC = 2
ZPAD = 2
#: ``a/dx``. Fixed at 8 so that every internal length in the sweep below clears
#: the ``l_b/h_cell >= 4`` floor this work uses throughout. The clipped lattice
#: misrepresents the circle by 1.9% in the second polar moment at this
#: resolution (6.8% at a/dx = 4, 0.6% at 12), which is why the rigidity is
#: reported against the *discrete* polar moment as well as the exact one.
CELLS_PER_RADIUS = 8            # a / dx
#: Peak engineering shear at the surface, ``a Theta / L``. Small, because the
#: rigidity is a *linear* property and the check against ``G J`` is only exact
#: in that limit; the finite-strain solve would otherwise stiffen and the
#: benchmark would appear to fail for a reason that is not an error.
SURFACE_SHEAR = 1.0e-3
STEPS = 4

#: The sweep, as ``l_b / a``. Varying the internal length at fixed geometry
#: traces the *same* curve a fixed-material size series would, and this is not a
#: convenience: ``K/GJ`` is dimensionless, so at fixed ``nu``, ``N`` and ``psi``,
#: with ``l_t = 2 l_b`` tied, it can depend only on ``a / l_b``. A series over
#: radii at fixed ``dx`` would cost ``a^3`` and one at fixed ``a/dx`` would let
#: ``l_b/dx`` fall below the floor at the large end -- the trap recorded for the
#: shear layer, where an unresolved internal length reads as a physical effect.
L_B_OVER_A = (0.0, 0.5, 0.75, 1.0)


def classical_rigidity(a):
    """``G J`` with ``J = pi a^4 / 2``, exact for a circular bar."""
    return MU * 0.5 * np.pi * a ** 4


def build(a, l_b, *, cells_per_radius=CELLS_PER_RADIUS, ppc=PPC,
          length_over_a=LENGTH_OVER_A, surface_shear=SURFACE_SHEAR,
          clamp_rotation=False, ell_pf=None, notch_depth=None):
    dx = a / cells_per_radius
    L = length_over_a * a
    nx = ny = int(round(2.0 * a / dx))
    nz = int(round(L / dx))
    grid = CartesianGrid(
        shape=(nx + 2 * PAD, ny + 2 * PAD, nz + 2 * ZPAD), dx=dx,
        origin=np.array([-PAD * dx, -PAD * dx, -ZPAD * dx]),
        basis=get_basis("linear"))

    sub = dx / ppc
    ax = [np.arange(sub / 2, 2.0 * a, sub), np.arange(sub / 2, 2.0 * a, sub),
          np.arange(sub / 2, L, sub)]
    pts = np.stack([m.ravel() for m in np.meshgrid(*ax, indexing="ij")], axis=1)
    c = np.array([a, a])
    keep = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) <= a
    parts = attach_micropolar_state(
        ParticleSet(pts[keep], vol0=sub ** 3, density=1.0))

    theta = surface_shear * L / a          # total twist, radians
    nan = np.nan
    rot_lo = np.zeros(3) if clamp_rotation else np.full(3, nan)
    rot_hi = (np.array([0.0, 0.0, theta]) if clamp_rotation
              else np.full(3, nan))

    def twist_field(coords):
        """Exact rigid rotation of the end plane, not its linearisation.

        ``Theta z_hat x r`` is the linearised form and introduces a spurious
        radial extension of order ``Theta^2`` that the finite-strain solve would
        resist, biasing the rigidity upward at larger twist.
        """
        ct, st = np.cos(theta), np.sin(theta)
        r = coords[:, :2] - c
        out = np.empty((len(coords), 6))
        out[:, 0] = ct * r[:, 0] - st * r[:, 1] - r[:, 0]
        out[:, 1] = st * r[:, 0] + ct * r[:, 1] - r[:, 1]
        out[:, 2] = 0.0
        out[:, 3:] = rot_hi
        return out

    bcs = [
        GridDirichlet(lambda q: q[:, 2] < 1e-12,
                      np.concatenate([np.zeros(3), rot_lo])),
        GridDirichlet(lambda q: np.abs(q[:, 2] - L) < 1e-12, twist_field),
    ]
    pf, stepper = make_problem(grid, parts, E=E_MOD, nu=NU, N=N_COUPLING,
                               l_b=l_b, gc=GC,
                               ell_pf=ell_pf or 4.0 * dx, bcs=bcs)
    if notch_depth is not None:
        seed_circumferential_notch(pf, parts, c, a, L, dx,
                                   ell_pf or 4.0 * dx, GC, notch_depth)
    xyz = np.asarray(grid.node_coordinates())
    loaded = np.flatnonzero(np.abs(xyz[:, 2] - L) < 1e-12)
    return dict(pf=pf, stepper=stepper, parts=parts, grid=grid, loaded=loaded,
                a=a, L=L, dx=dx, theta=theta, centre=c, nodes=xyz)


def notch_distance(x, c, a, L, depth):
    """Distance to the annular notch, in the ``(r, z)`` half-plane.

    The notch is the segment from the root at ``r = a - depth`` out to the free
    surface at ``r = a``, so the distance is measured in a half-plane behind
    the root and radially ahead of it -- a blunt annular crack rather than a
    damaged torus.

    Factored out because two callers need the *same* predicate: the seeding
    below, and the regional energy shares in `regional_sums`. A mask that drifts
    from the band it is meant to exclude would silently misattribute energy.
    """
    r = np.hypot(x[:, 0] - c[0], x[:, 1] - c[1])
    root, zc = a - depth, 0.5 * L
    return np.where(r >= root, np.abs(x[:, 2] - zc),
                    np.hypot(r - root, x[:, 2] - zc))


def notch_width(dx, ell, width_factor=0.25):
    """Half-thickness of the seeded band -- the ``r_0`` the other maps use."""
    return max(width_factor * ell, dx)


def seed_circumferential_notch(pf, parts, c, a, L, dx, ell, gc, depth,
                               amplitude=1.0e3, width_factor=0.25):
    """An annular notch at mid-length, seeded through the history field."""
    dist = notch_distance(parts.x, c, a, L, depth)
    width = notch_width(dx, ell, width_factor)
    H0 = np.zeros(parts.n)
    inside = dist <= width
    H0[inside] = amplitude * gc / (2.0 * ell) * (1.0 - dist[inside] / width)
    pf.H = H0
    return pf


#: Regions the driving energy is summed over, in the order `regional_sums`
#: returns them. ``whole`` is what the tables report.
REGIONS = ("whole", "band", "beyond_band", "beyond_2band")


def regional_sums(psi, dist, r0):
    """``(3, 4)``: each driving-energy term summed over each of `REGIONS`.

    Twelve floats per load step, and that is the point. The full per-particle
    densities of the production bar are 25 MB a step, `run_load_path` rewrites
    its checkpoint after *every* step, and the torsion runs checkpoint into
    ``results/`` under ~/Desktop, which macOS syncs -- carrying every step's
    densities through there would add some 400 MB to a file already implicated
    in the loss of one run to a refused rename. The map itself is wanted at the
    peak alone, which `_DensityRecorder` keeps in memory instead.
    """
    masks = (slice(None), dist <= r0 + 1e-9, dist > r0 + 1e-9, dist > 2.0 * r0)
    return np.stack([psi[:, m].sum(axis=1) for m in masks], axis=1)


class _DensityRecorder:
    """``on_step`` for `run_load_path`: regional sums every step, map at the peak.

    The returned sums go into the history and the checkpoint. The full
    densities are held for the last ``keep`` steps only, so the caller can
    write the one at the peak once ``argmax`` is known; ``stop_past_peak``
    stops the path a step or two past the turnover, which is what keeps the
    peak inside that window.
    """

    def __init__(self, dist, r0, keep=4):
        self.dist, self.r0 = dist, r0
        self.maps, self.n = deque(maxlen=keep), 0

    def __call__(self, pf, stepper, parts):
        psi = np.stack(pf.driving_densities())          # (3, n_particles)
        self.maps.append((self.n, psi))
        self.n += 1
        return regional_sums(psi, self.dist, self.r0)

    def map_at(self, k):
        """Densities at step ``k``, or None once it has left the window."""
        return next((m for i, m in self.maps if i == k), None)


def torque(s):
    """Axial torque on the loaded end, as the moment of the nodal forces.

    With the microrotation free at that end there is no couple reaction to
    add; ``--clamp-rotation`` does prescribe it, and the nodal moment about
    ``e_3`` then has to be included or the torque is understated.
    """
    f = s["stepper"].internal_force(s["parts"])
    i = s["loaded"]
    r = s["nodes"][i, :2] - s["centre"]
    fx, fy = f[i * 6 + 0], f[i * 6 + 1]
    T = float((r[:, 0] * fy - r[:, 1] * fx).sum())
    if s.get("clamped_rotation"):
        T += float(f[i * 6 + 5].sum())
    return T


def polar_moment(s):
    """``sum V_p r_p^2 / L`` against ``pi a^4 / 2``: the discretisation floor."""
    p = s["parts"]
    r2 = ((p.x[:, 0] - s["centre"][0]) ** 2 + (p.x[:, 1] - s["centre"][1]) ** 2)
    return float((p.vol0 * r2).sum() / s["L"])


def out_of_plane(parts):
    """In-plane over drilling microrotation magnitude; 0 in plane strain."""
    w = rotvec_from_quat(parts.quat)
    ip, dr = np.hypot(w[:, 0], w[:, 1]).max(), np.abs(w[:, 2]).max()
    return float(ip / dr) if dr > 0 else 0.0, float(dr)


def twist_elastic(a, l_b, steps=STEPS, **kw):
    """Elastic twist in ``steps`` increments; returns rigidity and diagnostics."""
    s = build(a, l_b, **kw)
    s["clamped_rotation"] = kw.get("clamp_rotation", False)
    st, parts = s["stepper"], s["parts"]
    with quiet():
        for k in range(steps):
            r = st.step(parts, (k + 1) / steps, 1.0 / steps)
            if not r.converged:
                raise RuntimeError(f"twist step {k + 1}/{steps} did not "
                                   f"converge (a={a}, l_b={l_b})")
    T = torque(s)
    K = T * s["L"] / s["theta"]
    Jd = polar_moment(s)
    ratio, drill = out_of_plane(parts)
    return dict(a=a, l_b=l_b, dx=s["dx"], L=s["L"], theta=s["theta"],
                torque=T, rigidity=K, J_discrete=Jd,
                J_exact=0.5 * np.pi * a ** 4,
                K_classical=classical_rigidity(a),
                K_discrete=MU * Jd, inplane_ratio=ratio, drill_max=drill,
                n_particles=parts.n, n_nodes=s["grid"].n_nodes)


# ---------------------------------------------------------------------------

def verify(clamp_rotation=False):
    """``l_b = 0`` must reproduce ``G J``; then the size effect switches on."""
    print("Elastic torsion of a circular bar, a = "
          f"{RADIUS:g}, L/a = {LENGTH_OVER_A:g}, a/dx = {CELLS_PER_RADIUS}")
    print(f"  E = {E_MOD:g}, nu = {NU:g}, N = {N_COUPLING:g}, "
          f"surface shear = {SURFACE_SHEAR:g}"
          f"{', microrotation CLAMPED at both ends' if clamp_rotation else ''}")
    rows = []
    for f in L_B_OVER_A:
        l_b = f * RADIUS
        t0 = time.time()
        r = twist_elastic(RADIUS, l_b, clamp_rotation=clamp_rotation)
        r["minutes"] = (time.time() - t0) / 60
        rows.append(r)
        print(f"    l_b/a = {l_b / RADIUS:4.2f}  done in "
              f"{r['minutes']:.1f} min", flush=True)
    j = rows[0]
    print(f"\n  discretisation floor: J_discrete/J_exact = "
          f"{j['J_discrete'] / j['J_exact']:.4f}  "
          f"({j['n_particles']} particles, {j['n_nodes']} nodes)")
    print(f"  the staircased boundary is what this measures; a rigidity error "
          f"no larger\n  than it carries no information about the solver.\n")
    print(f"  {'l_b/a':>6} {'K':>12} {'K/GJ_exact':>11} {'K/GJ_disc':>10} "
          f"{'|w_ip|/|w_z|':>12} {'|w_z|max':>10}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['l_b'] / r['a']:6.2f} {r['rigidity']:12.5g} "
              f"{r['rigidity'] / r['K_classical']:11.4f} "
              f"{r['rigidity'] / r['K_discrete']:10.4f} "
              f"{r['inplane_ratio']:12.4f} {r['drill_max']:10.3g}")
    print(f"\n  The l_b = 0 row is the external benchmark: it must return "
          f"G J.\n  Measured K/GJ_discrete = "
          f"{rows[0]['rigidity'] / rows[0]['K_discrete']:.4f}.")
    print(f"  |w_ip|/|w_z| is identically zero in every plane-strain example "
          f"in this work;\n  here the rotation axis leaves e_3 by construction.")
    np.savez_compressed(RESULTS / "torsion3d_verify.npz",
                        l_b=np.array([r["l_b"] for r in rows]),
                        K=np.array([r["rigidity"] for r in rows]),
                        K_cl=np.array([r["K_classical"] for r in rows]),
                        K_disc=np.array([r["K_discrete"] for r in rows]),
                        ratio=np.array([r["inplane_ratio"] for r in rows]),
                        a=RADIUS, dx=rows[0]["dx"])
    return rows


def size_series():
    """Deprecated: superseded by ``verify``, which traces the same curve.

    A series over radii needs either a fixed ``dx``, which costs ``a^3``, or a
    fixed ``a/dx``, which lets ``l_b/dx`` fall to 1.5 at the large end. Since
    ``K/GJ`` depends only on ``a/l_b``, the sweep in ``verify`` is the same
    curve with the circle equally resolved at every point.
    """
    raise SystemExit(size_series.__doc__)

NOTCH_DEPTH_OVER_A = 0.30
FRACTURE_SHEAR = 8.0e-3      # past the AT2 critical shear of ~4e-3 at l = a/2
FRACTURE_STEPS = 16


def fracture(l_b_over_a, *, cells_per_radius=CELLS_PER_RADIUS,
             length_over_a=LENGTH_OVER_A, steps=FRACTURE_STEPS,
             surface_shear=FRACTURE_SHEAR, progress=None, stop_past_peak=0.9,
             notch_depth_over_a=NOTCH_DEPTH_OVER_A, linear_solver=None,
             checkpoint=None, tag=""):
    """Circumferentially notched bar twisted to failure.

    The conjugate quantity is a torque, so ``run_load_path`` is given an
    explicit ``reaction`` rather than a component of the nodal force.
    """
    a = RADIUS
    s = build(a, l_b_over_a * a, cells_per_radius=cells_per_radius,
              length_over_a=length_over_a, surface_shear=surface_shear,
              notch_depth=notch_depth_over_a * a)
    s["clamped_rotation"] = False
    if linear_solver is not None:
        s["stepper"].linear_solver = linear_solver
    i, r = s["loaded"], s["nodes"][s["loaded"], :2] - s["centre"]

    def torque_of(f):
        return (r[:, 0] * f[i * 6 + 1] - r[:, 1] * f[i * 6 + 0]).sum()

    dist = notch_distance(s["parts"].x, s["centre"], a, s["L"],
                          notch_depth_over_a * a)
    r0 = notch_width(s["dx"], 4.0 * s["dx"])
    rec = _DensityRecorder(dist, r0)
    out = run_load_path(s["pf"], s["stepper"], s["parts"], i, 1, steps,
                        progress=progress or f"torsion l_b/a={l_b_over_a:g}",
                        reaction=torque_of, stop_past_peak=stop_past_peak,
                        checkpoint=checkpoint, on_step=rec)
    T = np.abs(np.asarray(out["load"]))
    k = int(np.argmax(T))
    out.update(l_b_over_a=l_b_over_a, theta=s["theta"], a=a, L=s["L"],
               dx=s["dx"], peak_torque=float(T[k]), peak_step=k,
               torque=T, shares_at_peak=out["shares"][k],
               d=s["pf"].tables.interpolate(s["pf"].d),
               X0=s["parts"].x.copy(), centre=s["centre"],
               notch_r0=r0, regional_at_peak=None)
    # A run resumed from a checkpoint written before this recorder existed
    # restores an empty extra history and then appends from the resume step, so
    # ``extra`` is shorter than the load path and indexing it by the global step
    # would read the wrong one. Loud rather than plausible.
    ex = out["extra"]
    if len(ex) == len(T):
        out["regional_at_peak"] = np.asarray(ex[k])
    else:
        print(f"      regional shares unavailable: {len(ex)} recorded against "
              f"{len(T)} steps (resumed across the change that added them)",
              flush=True)
    psi = rec.map_at(k)
    if psi is not None:
        np.savez_compressed(RESULTS / f"torsion3d_dens{tag}_lb{l_b_over_a:g}.npz",
                            X0=s["parts"].x.copy(), dens=psi, dist=dist, r0=r0,
                            t=out["disp"][k], peak_step=k,
                            l_b_over_a=l_b_over_a, centre=s["centre"],
                            a=a, L=s["L"], dx=s["dx"])
    return out


def fields(frac=0.5):
    """One twist at a single ``l_b``, saving the per-particle rotation vectors.

    Kept separate from ``verify`` because the fields are only needed for the
    figure and carrying them through the whole sweep would store four copies of
    a 39k-particle array for no use. ``frac`` is ``l_b/a``.
    """
    t0 = time.time()
    l_b = frac * RADIUS
    s = build(RADIUS, l_b)
    s["clamped_rotation"] = False
    st, parts = s["stepper"], s["parts"]
    X0 = parts.x.copy()
    with quiet():
        for k in range(STEPS):
            r = st.step(parts, (k + 1) / STEPS, 1.0 / STEPS)
            if not r.converged:
                raise RuntimeError(f"twist step {k + 1} did not converge")
    w = rotvec_from_quat(parts.quat)
    np.savez_compressed(RESULTS / f"torsion3d_fields_{frac:g}.npz",
                        X0=X0, x=parts.x.copy(), w=w,
                        a=s["a"], L=s["L"], dx=s["dx"], centre=s["centre"],
                        theta=s["theta"], l_b=l_b,
                        minutes=(time.time() - t0) / 60)
    ip = np.hypot(w[:, 0], w[:, 1])
    print(f"  saved {parts.n} particles, l_b/a = {frac:g}, "
          f"max |w_in-plane| = {ip.max():.4g}, max |w_z| = "
          f"{np.abs(w[:, 2]).max():.4g}, {(time.time() - t0) / 60:.1f} min")


def run_fracture(cells_per_radius=CELLS_PER_RADIUS,
                 length_over_a=LENGTH_OVER_A, sweep=(0.0, 0.5),
                 notch_depth_over_a=NOTCH_DEPTH_OVER_A, krylov=False,
                 tag=""):
    """The notched bar twisted to failure, with and without the internal length.

    A pilot at the coarse resolution first: the question it answers is whether
    the crack initiates at the notch root rather than at the clamped end and
    whether the solve survives the peak, neither of which is worth paying for
    at a finer cell size before it is known.
    """
    t0 = time.time()
    print(f"Notched bar in torsion, a/dx = {cells_per_radius}, "
          f"L/a = {length_over_a:g}, notch depth {NOTCH_DEPTH_OVER_A:g}a")
    print(f"  notch depth {notch_depth_over_a:g}a, "
          f"notch/l = {notch_depth_over_a * cells_per_radius / 4:.2f}"
          f"{', KrylovSolver' if krylov else ', direct solver'}")
    print(f"  l = {4 * RADIUS / cells_per_radius:.4g} "
          f"(l/a = {4 / cells_per_radius:.2f}) -- coarse, see the caption\n")
    rows = []
    for f in sweep:
        # A fresh solver per row, so its iteration counters describe one run.
        ls = None
        if krylov:
            from scion.solvers.linear import KrylovSolver
            ls = KrylovSolver()
        r = fracture(f, cells_per_radius=cells_per_radius,
                     length_over_a=length_over_a,
                     notch_depth_over_a=notch_depth_over_a,
                     linear_solver=ls, tag=tag,
                     checkpoint=RESULTS / f"ckpt_torsion{tag}_lb{f:g}.npz")
        if ls is not None:
            r["krylov_solves"], r["krylov_iters"] = ls.n_solves, ls.total_iterations
        rows.append(r)
        # The question the pilot exists to answer: does it crack at the notch
        # or at the clamped end? A beam in this paper once cracked at the clamp
        # because the phase-field length was half the section, and l/a is 0.5
        # here, so this is not a formality.
        X0, c = r["X0"], r["centre"]
        rad = np.hypot(X0[:, 0] - c[0], X0[:, 1] - c[1])
        z, d = X0[:, 2], r["d"]
        at_notch = (rad > 0.6 * r["a"]) & (np.abs(z - 0.5 * r["L"])
                                           < 0.15 * r["L"])
        at_clamp = z < 0.12 * r["L"]
        r["d_notch"], r["d_clamp"] = float(d[at_notch].max()), float(d[at_clamp].max())
        print(f"    l_b/a = {f:g}: peak torque {r['peak_torque']:.5g} N m at "
              f"step {r['peak_step']}, reached t = {r['reached']:.3f}, "
              f"d_notch {r['d_notch']:.3f} vs d_clamp {r['d_clamp']:.3f}, "
              f"{(time.time() - t0) / 60:.0f} min so far", flush=True)
    base = rows[0]["peak_torque"]
    print(f"\n  {'l_b/a':>6} {'peak torque':>13} {'normalised':>11} "
          f"{'W_rel':>10} {'W_curv':>10} {'reached':>8}")
    for r in rows:
        sh = r["shares_at_peak"]
        print(f"  {r['l_b_over_a']:6.2f} {r['peak_torque']:13.5g} "
              f"{r['peak_torque'] / base:11.4f} "
              f"{sh['relative_rotation']:10.3g} {sh['curvature']:10.3g} "
              f"{r['reached']:8.3f}")
    print(f"\n  cracked at the notch in every row: "
          f"{all(r['d_notch'] > r['d_clamp'] for r in rows)}")
    # The whole-specimen shares above include the seeded band, and inside it
    # g(d) -> 0 has removed the very stress whose symmetry would force the
    # microrotation onto the continuum rotation -- so skew(E) there is
    # evaluated on nearly free kinematics while W_rel is reported undegraded.
    # At l_b = 0 that is the whole of the W_rel column: measured on an elastic
    # surrogate, 0.229 over the whole bar against 0.0012 with no notch at all.
    # Read the micropolar contribution from the band-excluded rows.
    if all(r["regional_at_peak"] is not None for r in rows):
        print(f"\n  shares at the peak by region (r_0 = seeded "
              f"half-thickness {rows[0]['notch_r0']:.4g})")
        print(f"  {'l_b/a':>6} {'region':>13} {'W_iso+':>10} {'W_rel':>10} "
              f"{'W_curv':>10}")
        for r in rows:
            for j, name in enumerate(REGIONS):
                g = r["regional_at_peak"][:, j]
                tot = max(float(g.sum()), 1e-300)
                print(f"  {r['l_b_over_a']:6.2f} {name:>13} "
                      f"{g[0] / tot:10.4f} {g[1] / tot:10.4g} "
                      f"{g[2] / tot:10.4g}")
    np.savez_compressed(RESULTS / f"torsion3d_fracture{tag}.npz",
                        l_b_over_a=np.array([r["l_b_over_a"] for r in rows]),
                        peak=np.array([r["peak_torque"] for r in rows]),
                        disp=np.array([r["disp"] for r in rows], dtype=object),
                        torque=np.array([r["torque"] for r in rows], dtype=object),
                        d=np.array([r["d"] for r in rows]),
                        d_notch=np.array([r["d_notch"] for r in rows]),
                        d_clamp=np.array([r["d_clamp"] for r in rows]),
                        X0=rows[0]["X0"], centre=rows[0]["centre"],
                        a=RADIUS, dx=rows[0]["dx"], L=rows[0]["L"],
                        cells_per_radius=cells_per_radius,
                        notch_r0=rows[0]["notch_r0"],
                        regions=np.array(REGIONS),
                        regional_at_peak=np.array(
                            [r["regional_at_peak"] for r in rows]
                            if all(r["regional_at_peak"] is not None
                                   for r in rows) else []),
                        minutes=(time.time() - t0) / 60)
    print(f"\n  {(time.time() - t0) / 60:.0f} min")
    return rows


def time_fracture():
    """One staggered load step of the fracture problem, to price the path."""
    for cpr, loa in ((8, 3.0), (10, 2.0), (12, 2.0)):
        a = RADIUS
        s = build(a, 0.5 * a, cells_per_radius=cpr, length_over_a=loa,
                  surface_shear=FRACTURE_SHEAR,
                  notch_depth=NOTCH_DEPTH_OVER_A * a)
        dx = s["dx"]
        t0 = time.time()
        with quiet():
            s["pf"].solve_step(1.0 / FRACTURE_STEPS, 1.0 / FRACTURE_STEPS)
        dt = time.time() - t0
        print(f"  a/dx={cpr:2d} L/a={loa:g}  l={4*dx:.4f} (l/a={4*dx/a:.2f})  "
              f"nodes {s['grid'].n_nodes:6d} parts {s['parts'].n:7d}  "
              f"{dt/60:6.2f} min/step -> ~{dt*FRACTURE_STEPS/3600:.1f} h/run",
              flush=True)


def timing():
    for cpr in (4, 6, 8):
        s = build(RADIUS, 0.5 * RADIUS, cells_per_radius=cpr)
        t0 = time.time()
        with quiet():
            s["stepper"].step(s["parts"], 0.25, 0.25)
        print(f"  a/dx={cpr}  nodes {s['grid'].n_nodes:6d}  "
              f"particles {s['parts'].n:7d}  {time.time() - t0:6.1f} s/Newton",
              flush=True)


if __name__ == "__main__":
    if "--time" in sys.argv:
        timing()
    elif "--fields" in sys.argv:
        k = sys.argv.index("--fields")
        fields(float(sys.argv[k + 1]) if len(sys.argv) > k + 1
               and not sys.argv[k + 1].startswith("-") else 0.5)
    elif "--time-fracture" in sys.argv:
        time_fracture()
    elif "--fracture" in sys.argv:
        run_fracture()
    elif "--size" in sys.argv:
        size_series()
    else:
        verify(clamp_rotation="--clamp-rotation" in sys.argv)
