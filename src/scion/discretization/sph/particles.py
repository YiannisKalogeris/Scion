r"""
Particle generation.

``lattice`` places a particle at the centre of every cell of a regular grid
over the box, which is what SPHinXsys's ``Lattice`` generator does: positions
are ``lower + (i + 1/2) dp``, never on the box face. Two consequences worth
stating because both are easy to trip over:

* the particle volume is ``dp**dim`` exactly, so the volumes sum to the box
  volume with no correction;
* the lattice is compatible with a periodic axis only when the box side is an
  integer number of spacings -- the wrapped gap between the first and last
  column is then ``dp`` like any other. A box side that is not a multiple of
  ``dp`` leaves a short seam that reads as a stationary defect in the
  solution, so it is refused rather than rounded.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lattice"]


def lattice(lower, upper, spacing, periodic=None, inside=None):
    """Cell-centred particles filling the box, with their volumes.

    ``inside(x) -> bool array`` optionally carves a shape out of the lattice.
    Returns ``(positions, volumes)``.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    dp = float(spacing)
    dim = lo.size
    periodic = (False,) * dim if periodic is None else tuple(bool(p) for p in periodic)
    counts = np.rint((hi - lo) / dp).astype(int)
    for d in range(dim):
        exact = abs((hi[d] - lo[d]) / dp - counts[d]) < 1e-9
        if periodic[d] and not exact:
            raise ValueError(
                f"axis {d} is periodic but {(hi[d] - lo[d]) / dp:.6g} spacings "
                "wide; the wrapped seam would not match the lattice")
        if not periodic[d]:
            counts[d] = int(np.floor((hi[d] - lo[d]) / dp + 1e-9))
    axes = [lo[d] + (np.arange(counts[d]) + 0.5) * dp for d in range(dim)]
    grid = np.meshgrid(*axes, indexing="ij")
    x = np.stack([g.ravel() for g in grid], axis=1)
    if inside is not None:
        x = x[np.asarray(inside(x), dtype=bool)]
    return x, np.full(x.shape[0], dp**dim)
