"""
Verification for the load domain and the convex-optimisation additions.

The load domain is small but it is where the ``2^P`` lives, and getting a
vertex wrong means silently checking the wrong loads. The optimisation pieces
are physics-free and are tested against problems with closed-form answers.

python -m pytest tests/test_shakedown/test_load_domain.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.optimization.convex import (
    check_projector,
    largest_feasible,
    minimize_on_subspace,
)
from scion.shakedown import LoadDomain


# ---------------------------------------------------------------------------
# Load domain
# ---------------------------------------------------------------------------

def test_the_vertices_are_every_combination_of_the_bounds():
    domain = LoadDomain.box([0.0, -1.0], [1.0, 2.0])
    assert domain.n_vertices == 4
    got = {tuple(v) for v in domain.vertices}
    assert got == {(0.0, -1.0), (1.0, -1.0), (0.0, 2.0), (1.0, 2.0)}


@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_the_vertex_count_is_two_to_the_number_of_varying_cases(p):
    domain = LoadDomain.pulsating(p)
    assert domain.n_vertices == 2 ** p
    assert domain.vertices.shape == (2 ** p, p)
    assert len({tuple(v) for v in domain.vertices}) == 2 ** p


def test_a_fixed_case_contributes_no_vertex_pair():
    """Limit analysis is the degenerate domain, and it has to come out with one
    vertex rather than two identical ones -- otherwise every yield constraint is
    duplicated and the solve does twice the work for the same answer."""
    assert LoadDomain.fixed(1.0).n_vertices == 1
    assert LoadDomain.fixed([1.0, 0.5]).n_vertices == 1
    np.testing.assert_allclose(LoadDomain.fixed([1.0, 0.5]).vertices,
                               [[1.0, 0.5]])

    mixed = LoadDomain.box([1.0, 0.0], [1.0, 2.0])       # one fixed, one free
    assert mixed.n_vertices == 2
    assert np.all(mixed.vertices[:, 0] == 1.0)


def test_the_alternating_and_pulsating_shortcuts():
    assert set(LoadDomain.alternating(1).vertices.ravel()) == {-1.0, 1.0}
    assert set(LoadDomain.pulsating(1).vertices.ravel()) == {0.0, 1.0}


def test_every_vertex_is_inside_the_domain():
    domain = LoadDomain.box([-1.0, 0.0, 2.0], [1.0, 3.0, 2.0])
    assert domain.contains(domain.vertices).all()
    assert not domain.contains([[2.0, 0.0, 2.0]])[0]


def test_a_domain_too_large_to_hold_is_refused():
    """Each vertex carries a stress field at every quadrature point, so this is
    a memory decision rather than a tolerance and it is made loudly."""
    with pytest.raises(ValueError, match="vertices"):
        LoadDomain.pulsating(20)
    LoadDomain.pulsating(20, max_vertices=2 ** 21)       # deliberate override


def test_inconsistent_bounds_are_refused():
    with pytest.raises(ValueError, match="below lower"):
        LoadDomain.box([1.0], [0.0])
    with pytest.raises(ValueError, match="same length"):
        LoadDomain.box([1.0, 2.0], [3.0])
    with pytest.raises(ValueError, match="names"):
        LoadDomain.box([0.0], [1.0], names=["a", "b"])


# ---------------------------------------------------------------------------
# minimize_on_subspace
# ---------------------------------------------------------------------------

def _sum_to_zero_projector(n):
    """Orthogonal projection onto ``{x : sum x = 0}``."""
    def project(x):
        x = np.asarray(x, dtype=float).ravel()
        return x - x.mean()
    return project


def test_minimising_on_a_subspace_hits_the_constrained_optimum():
    """``min |x - a|^2`` over ``sum x = 0`` has the answer ``a - mean(a)``."""
    n = 12
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)

    def fun_and_grad(x):
        d = x - a
        return float(d @ d), 2.0 * d

    out = minimize_on_subspace(fun_and_grad, np.zeros(n),
                               _sum_to_zero_projector(n))
    assert out.converged
    np.testing.assert_allclose(out.x, a - a.mean(), atol=1e-8)
    assert abs(out.x.sum()) < 1e-10


def test_a_projector_that_is_not_orthogonal_is_refused():
    """An oblique projection lands in the subspace and still breaks the chain
    rule, so idempotence alone is not enough to accept one."""
    def oblique(x):
        x = np.asarray(x, dtype=float).ravel()
        y = x.copy()
        y[0] = 0.0                       # idempotent, but not self-adjoint
        y[1] += x[0]
        return y

    assert check_projector(oblique, 5)["idempotent"] < 1e-12
    assert check_projector(oblique, 5)["self_adjoint"] > 1e-3
    assert not check_projector(oblique, 5)["ok"]

    with pytest.raises(ValueError, match="orthogonal projector"):
        minimize_on_subspace(lambda x: (0.0, np.zeros_like(x)), np.zeros(5),
                             oblique)


def test_no_projector_means_the_whole_space():
    """A search space given by a basis of admissible fields carries its
    constraint in the basis, so there is nothing to project."""
    def fun_and_grad(x):
        d = x - 3.0
        return float(d @ d), 2.0 * d

    out = minimize_on_subspace(fun_and_grad, np.zeros(4), project=None)
    np.testing.assert_allclose(out.x, 3.0, atol=1e-8)


# ---------------------------------------------------------------------------
# largest_feasible
# ---------------------------------------------------------------------------

def test_bisection_returns_a_value_it_actually_verified():
    """Not the midpoint of the last bracket. A bound has to be a checked
    value, or it is an interpolation being passed off as a certificate."""
    accepted = []

    def oracle(t):
        accepted.append(t)
        return t <= 7.0

    out = largest_feasible(oracle, start=1.0, rtol=1e-6)
    assert out.value in accepted
    assert oracle(out.value)
    assert out.value <= 7.0 < out.upper
    assert out.relative_width < 1e-5
    assert out.bracketed


def test_bisection_expands_until_it_finds_the_far_side():
    out = largest_feasible(lambda t: t <= 1000.0, start=1.0, rtol=1e-3)
    assert out.value == pytest.approx(1000.0, rel=1e-3)


def test_an_oracle_that_never_refuses_is_reported_not_hidden():
    """An unbounded problem, or an oracle that is not testing what it should.
    Either way it is a real outcome and should not come back as a big number."""
    out = largest_feasible(lambda t: True, start=1.0, max_expand=5)
    assert not out.bracketed
    assert np.isinf(out.upper)
    assert np.isinf(out.relative_width)


def test_an_infeasible_start_is_refused():
    with pytest.raises(ValueError, match="feasible starting point"):
        largest_feasible(lambda t: False, start=1.0)
