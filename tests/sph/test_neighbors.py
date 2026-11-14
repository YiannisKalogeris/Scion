"""
The neighbour search, against brute force.

A cell list is the part of an SPH code that is easy to get subtly wrong and
hard to notice: a missed pair at a periodic seam, or a pair counted twice
where the stencil wraps onto itself, changes the operator by a few percent in
a thin region and reads as a boundary effect. The reference here is the
``O(N^2)`` search, and the comparison is on the pair *set*, not on a count.
"""

import numpy as np
import pytest

from scion.discretization.sph import (brute_force_neighbors, build_neighbors,
                                      lattice)

BOX = (np.array([0.0, 0.0]), np.array([2.0, 0.4]))


def cloud(n=900, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(BOX[0], BOX[1], size=(n, 2))


def pair_set(neighbors):
    return set(map(tuple, np.sort(np.stack([neighbors.i, neighbors.j], axis=1), axis=1)))


@pytest.mark.parametrize("periodic", [(False, False), (True, False), (True, True)])
def test_cell_list_matches_brute_force(periodic):
    x = cloud()
    fast = build_neighbors(x, 0.1, lower=BOX[0], upper=BOX[1], periodic=periodic)
    slow = brute_force_neighbors(x, 0.1, lower=BOX[0], upper=BOX[1], periodic=periodic)
    assert pair_set(fast) == pair_set(slow)
    assert len(fast) == len(slow)


@pytest.mark.parametrize("periodic", [(False, False), (True, False)])
def test_each_pair_is_stored_once(periodic):
    """Both operators update ``i`` and ``j`` from a single entry, so a
    duplicated pair is a doubled diffusivity for those two particles."""
    x = cloud(400, seed=3)
    neighbors = build_neighbors(x, 0.15, lower=BOX[0], upper=BOX[1], periodic=periodic)
    pairs = np.sort(np.stack([neighbors.i, neighbors.j], axis=1), axis=1)
    assert len(np.unique(pairs, axis=0)) == len(neighbors)
    assert (neighbors.i != neighbors.j).all()


def test_separation_is_the_minimum_image():
    x = np.array([[0.005, 0.2], [1.995, 0.2]])
    neighbors = build_neighbors(x, 0.1, lower=BOX[0], upper=BOX[1],
                                periodic=(True, False))
    assert len(neighbors) == 1
    assert neighbors.r[0] == pytest.approx(0.01, abs=1e-15)
    # and the stored separation is the wrapped one, not x_i - x_j
    assert abs(neighbors.dx[0, 0]) == pytest.approx(0.01, abs=1e-15)


def test_unit_vectors_are_consistent_with_the_separation():
    x = cloud(300, seed=7)
    neighbors = build_neighbors(x, 0.12, lower=BOX[0], upper=BOX[1],
                                periodic=(True, False))
    assert np.abs(np.linalg.norm(neighbors.e, axis=1) - 1.0).max() < 1e-14
    assert np.abs(neighbors.e * neighbors.r[:, None] - neighbors.dx).max() < 1e-15


def test_counts_both_directions():
    x, _ = lattice([0.0, 0.0], [0.2, 0.2], 0.02)
    neighbors = build_neighbors(x, 0.05, lower=[0.0, 0.0], upper=[0.2, 0.2])
    assert neighbors.counts().sum() == 2 * len(neighbors)


def test_refuses_a_periodic_axis_thinner_than_three_cells():
    """Two cells would make the three-cell stencil meet the same cell twice,
    which doubles those pairs rather than failing."""
    x = cloud(50)
    with pytest.raises(ValueError, match="three cells"):
        build_neighbors(x, 1.0, lower=BOX[0], upper=BOX[1], periodic=(True, False))


def test_refuses_a_periodic_axis_without_bounds():
    with pytest.raises(ValueError, match="bounds"):
        build_neighbors(cloud(50), 0.1, periodic=(True, False))
