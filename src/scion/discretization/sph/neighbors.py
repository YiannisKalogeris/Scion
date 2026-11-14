r"""
Neighbour search on a cell-linked list, with per-axis periodicity.

Each unordered pair is stored ONCE
----------------------------------
``NeighborList`` holds ``i``, ``j`` with ``i != j`` and every pair appearing
a single time, together with the separation ``dx = x_i - x_j`` already
reduced to its minimum image. Both particles of a pair are then updated in
one pass with opposite signs, which is not merely an optimization: it makes
the discrete operators *exactly* antisymmetric, so the diffusion operator
conserves the total species at round-off rather than at a tolerance (see
``operators.laplacian_rate``). Storing both directions and summing them
independently gives the same answer up to the last bits and loses that.

The minimum image is applied at build time
------------------------------------------
``dx`` is the wrapped separation, so nothing downstream has to know which
axes are periodic. A consumer that recomputes ``x_i - x_j`` from the
positions gets the unwrapped vector and is wrong only for the pairs that
straddle the boundary -- a handful of particles out of thousands, which is
small enough to look like noise and large enough to change an answer.

The cell size is the cutoff, and a periodic axis must hold at least three
cells, or the three-cell stencil would visit the same cell twice and
double-count. That is checked rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

__all__ = ["NeighborList", "build_neighbors", "brute_force_neighbors"]


@dataclass(frozen=True)
class NeighborList:
    """Pairs within the cutoff, each listed once.

    ``i``, ``j``   particle indices, ``(n_pairs,)``
    ``dx``         ``x_i - x_j``, minimum image, ``(n_pairs, dim)``
    ``r``          ``|dx|``, ``(n_pairs,)``
    ``e``          ``dx / r``, ``(n_pairs, dim)``
    """

    i: np.ndarray
    j: np.ndarray
    dx: np.ndarray
    r: np.ndarray
    e: np.ndarray
    n_particles: int
    cutoff: float

    def __len__(self) -> int:
        return int(self.i.size)

    def counts(self) -> np.ndarray:
        """Neighbours per particle, both directions counted."""
        out = np.zeros(self.n_particles, dtype=np.int64)
        np.add.at(out, self.i, 1)
        np.add.at(out, self.j, 1)
        return out


def _wrap(positions, lower, upper, periodic):
    x = np.array(positions, dtype=float, copy=True)
    for d, flag in enumerate(periodic):
        if flag:
            length = upper[d] - lower[d]
            x[:, d] = lower[d] + np.mod(x[:, d] - lower[d], length)
    return x


def build_neighbors(positions, cutoff, lower=None, upper=None,
                    periodic=None) -> NeighborList:
    """Pairs closer than ``cutoff``, by a cell-linked list.

    ``lower``/``upper`` bound the cell grid and are required for any periodic
    axis; they default to the particle extent. ``periodic`` is a per-axis
    bool sequence.
    """
    x = np.asarray(positions, dtype=float)
    n, dim = x.shape
    cutoff = float(cutoff)
    periodic = (False,) * dim if periodic is None else tuple(bool(p) for p in periodic)
    lo = x.min(axis=0) if lower is None else np.asarray(lower, dtype=float)
    hi = x.max(axis=0) if upper is None else np.asarray(upper, dtype=float)
    if len(periodic) != dim:
        raise ValueError("periodic must have one entry per dimension")
    if any(periodic) and (lower is None or upper is None):
        raise ValueError("a periodic axis needs explicit lower and upper bounds")

    x = _wrap(x, lo, hi, periodic)
    length = np.maximum(hi - lo, 1e-300)
    n_cells = np.maximum((length / cutoff).astype(np.int64), 1)
    for d, flag in enumerate(periodic):
        if flag and n_cells[d] < 3:
            raise ValueError(
                f"axis {d} is periodic and {length[d] / cutoff:.3g} cutoffs wide; "
                "the minimum image needs at least three cells")
    cell = np.clip(((x - lo) / length * n_cells).astype(np.int64), 0, n_cells - 1)

    strides = np.cumprod(np.append(1, n_cells[:-1])).astype(np.int64)
    cid = (cell * strides).sum(axis=1)

    order = np.argsort(cid, kind="stable")
    cid_sorted = cid[order]
    uniq, start = np.unique(cid_sorted, return_index=True)
    count = np.diff(np.append(start, n))

    # the self cell plus half the shifts, so a pair is generated once; which
    # half does not matter as long as it is a half-space of the offsets
    offsets = [o for o in product((-1, 0, 1), repeat=dim)]
    forward = [o for o in offsets if o > (0,) * dim]

    pi_all, pj_all = [], []
    for offset in [(0,) * dim] + forward:
        target = cell + np.asarray(offset, dtype=np.int64)
        keep = np.ones(n, dtype=bool)
        for d in range(dim):
            if periodic[d]:
                target[:, d] = np.mod(target[:, d], n_cells[d])
            else:
                keep &= (target[:, d] >= 0) & (target[:, d] < n_cells[d])
        if not keep.any():
            continue
        src = np.flatnonzero(keep)
        tid = (target[src] * strides).sum(axis=1)
        slot = np.searchsorted(uniq, tid)
        slot = np.clip(slot, 0, uniq.size - 1)
        hit = uniq[slot] == tid
        src, slot = src[hit], slot[hit]
        if src.size == 0:
            continue
        k = count[slot]
        # ragged expansion: each src repeated k times against its cell's slice
        pi = np.repeat(src, k)
        base = np.repeat(start[slot], k)
        within = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
        pj = order[base + within]
        if offset == (0,) * dim:
            m = pi < pj
            pi, pj = pi[m], pj[m]
        pi_all.append(pi)
        pj_all.append(pj)

    if pi_all:
        pi = np.concatenate(pi_all)
        pj = np.concatenate(pj_all)
    else:
        pi = pj = np.zeros(0, dtype=np.int64)

    dx = x[pi] - x[pj]
    for d, flag in enumerate(periodic):
        if flag:
            dx[:, d] -= length[d] * np.round(dx[:, d] / length[d])
    r = np.sqrt((dx * dx).sum(axis=1))
    m = (r < cutoff) & (r > 0.0)
    pi, pj, dx, r = pi[m], pj[m], dx[m], r[m]
    return NeighborList(i=pi, j=pj, dx=dx, r=r, e=dx / r[:, None],
                        n_particles=n, cutoff=cutoff)


def brute_force_neighbors(positions, cutoff, lower=None, upper=None,
                          periodic=None) -> NeighborList:
    """The same thing in ``O(N^2)``, as the reference the cell list is tested
    against. Never call it on a production run."""
    x = np.asarray(positions, dtype=float)
    n, dim = x.shape
    periodic = (False,) * dim if periodic is None else tuple(bool(p) for p in periodic)
    lo = x.min(axis=0) if lower is None else np.asarray(lower, dtype=float)
    hi = x.max(axis=0) if upper is None else np.asarray(upper, dtype=float)
    x = _wrap(x, lo, hi, periodic)
    length = hi - lo
    pi, pj = np.triu_indices(n, k=1)
    dx = x[pi] - x[pj]
    for d, flag in enumerate(periodic):
        if flag:
            dx[:, d] -= length[d] * np.round(dx[:, d] / length[d])
    r = np.sqrt((dx * dx).sum(axis=1))
    m = (r < cutoff) & (r > 0.0)
    pi, pj, dx, r = pi[m], pj[m], dx[m], r[m]
    return NeighborList(i=pi, j=pj, dx=dx, r=r, e=dx / r[:, None],
                        n_particles=n, cutoff=float(cutoff))
