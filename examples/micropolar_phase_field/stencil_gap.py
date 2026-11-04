"""How far the reference and current stencils separate, in background cells.

The phase field is interpolated by ``N_I(X_p)`` and the mechanics by
``N_I(x_p)`` on the same grid, so the two share the node numbering but not the
configuration they are sampled in. That is admissible because the fields are
never coupled node to node -- the mechanics sees damage only as the particle
value ``d_p`` and the phase field sees the mechanics only as ``H_p`` -- but the
paper still has to say how far apart the stencils get, and the honest bound is
``|x_p - X_p|`` measured against the cell size.

The band column is the one that matters: a large separation at a beam tip where
``d = 0`` costs nothing, while the same separation inside the diffuse crack
would mean the damage field and the stress that drives it are resolved on
visibly different stencils.

    python -m examples.micropolar_phase_field.stencil_gap

Reads the saved final states; every run that wrote one is reported and the rest
are skipped with a note, so a partial results directory still gives a table.
"""
from pathlib import Path

import numpy as np

from ._common import seed_particles
from . import notched_beam as nb
from . import notched_tension as nt

RESULTS = Path(__file__).resolve().parent / "results"
BAND = 0.5          # d above this counts as inside the diffuse crack


def row(name, X0, x, dx, d=None):
    """One line: particle displacement in cells, overall and in the band."""
    u = np.linalg.norm(x[:, :X0.shape[1]] - X0, axis=1) / dx
    band = "" if d is None else "      --      --"
    if d is not None and (d > BAND).any():
        m = d > BAND
        band = f"  {u[m].max():8.3f}  {u[m].mean():6.3f}"
    print(f"  {name:22s} {u.max():8.3f}  {u.mean():6.3f}{band}")


def header(title):
    print(f"\n{title}")
    print(f"  {'':22s} {'max':>8s}  {'mean':>6s}  {'band max':>8s}  "
          f"{'band mean':>6s}")


def tension():
    f = RESULTS / "tension_data.npz"
    if not f.exists():
        print("\nnotched tension: no tension_data.npz")
        return
    # the reference lattice is deterministic and order preserving, so it can be
    # regenerated rather than stored -- the same trick diagnose_symmetry uses
    X0 = seed_particles((nt.SIDE, nt.SIDE), nt.DX, nt.PPC).x[:, :2]
    d = np.load(f, allow_pickle=True)
    header(f"notched tension  (h_cell = {nt.DX:g} mm)")
    for tag in ("classical", "lb0", "lb1", "lb2", "lb4"):
        if f"{tag}_x" not in d.files:
            continue
        # the classical row stores REFERENCE positions -- it is small strain and
        # never advects its particles -- so its separation is identically zero
        # and reporting it would be meaningless rather than reassuring
        if tag == "classical":
            continue
        row(tag, X0, d[f"{tag}_x"], nt.DX,
            np.asarray(d[f"{tag}_fields"])[-1])


def beam():
    f = RESULTS / "beam_data.npz"
    if not f.exists():
        print("\nnotched beam: no beam_data.npz")
        return
    X0 = seed_particles((nb.SPAN, nb.DEPTH), nb.DX, nb.PPC).x[:, :2]
    d = np.load(f, allow_pickle=True)
    header(f"notched beam  (h_cell = {nb.DX:g} m)")
    for i, v in enumerate(np.asarray(d["l_b_over_h"]).ravel()):
        row(f"l_b/h = {v:g}", X0, np.asarray(d[f"row{i}_x"]), nb.DX,
            np.asarray(d[f"row{i}_fields"])[-1])


def double_notch():
    found = False
    for n in ("0", "0.3", "0.5", "0.7"):
        f = RESULTS / f"double_notch_N{n}_dx1.npz"
        if not f.exists():
            continue
        q = np.load(f, allow_pickle=True)
        if not found:
            header(f"double-notched plate  (h_cell = {float(q['dx']):g} mm)")
            found = True
        row(f"N = {n}", np.asarray(q["X0"])[:, :2], np.asarray(q["x"]),
            float(q["dx"]), np.asarray(q["fields"])[-1])
    if not found:
        print("\ndouble-notched plate: no double_notch_N*_dx1.npz")


def torsion():
    found = False
    for frac in (0.0, 0.5):
        f = RESULTS / f"torsion3d_fields_{frac:g}.npz"
        if not f.exists():
            continue
        t = np.load(f)
        if not found:
            header(f"torsion, elastic  (h_cell = {float(t['dx']):g} m)")
            found = True
        row(f"l_b/a = {frac:g}", t["X0"], t["x"], float(t["dx"]))
    if not found:
        print("\ntorsion: no torsion3d_fields_*.npz")


def main():
    print("Separation of the reference and current stencils, "
          "|x_p - X_p| / h_cell,\nat the final state of each production run.")
    tension()
    beam()
    double_notch()
    torsion()


if __name__ == "__main__":
    main()
