"""
The discrete configurational force, Eq. (9).

Verified against a closed form rather than against itself. For a displacement
field ``u(X) = c X^2`` and a B-spline of degree >= 2 -- which reproduces
quadratics exactly -- the second discrete gradient is exactly ``2c``, so with
constant stress and volume Eq. (9) collapses to ``G_p = -V_p P_p (2c)`` with no
discretisation error at all. Anything the implementation adds to that shows up
immediately.

The companion property is that ``G_p`` vanishes identically for a *uniform*
deformation on any basis with linear completeness. That is what makes ``||G_p||``
usable as an error indicator, and it is the assertion that would fail first if
the basis Hessian were wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.discretization.mpm import CartesianGrid
from scion.discretization.mpm.basis import LinearHat, get_basis
from projects.configurational_mpm.forces import (
    configurational_force,
    displacement_gradient,
    second_displacement_gradient,
)

N = 24
DX = 1.0 / N
SMOOTH = ["quadratic", "cubic"]


def setup_1d(basis, u_of_X, ppc=1):
    grid = CartesianGrid(shape=(N,), dx=DX, basis=get_basis(basis))
    s = (np.arange(ppc) + 0.5) / ppc
    Xp = ((np.arange(N)[:, None] + s[None, :]).ravel() * DX)[:, None]
    u_nodes = u_of_X(grid.node_coordinates())
    st = grid.build_stencil(Xp, hessian=True)
    return grid, Xp, u_nodes, st


def interior(Xp):
    return (Xp[:, 0] > 0.2) & (Xp[:, 0] < 0.8)


class TestDiscreteGradients:
    @pytest.mark.parametrize("basis", SMOOTH)
    def test_quadratic_field_is_reproduced_exactly(self, basis):
        c = 0.3
        _, Xp, u, st = setup_1d(basis, lambda X: c * X ** 2)
        m = interior(Xp)

        H = displacement_gradient(u, st.node_id, st.grad_w)[:, 0, 0]
        d2 = second_displacement_gradient(u, st.node_id, st.hess)[:, 0, 0, 0]

        assert np.abs(H[m] - 2 * c * Xp[m, 0]).max() < 1e-12
        assert np.abs(d2[m] - 2 * c).max() < 1e-11

    @pytest.mark.parametrize("basis", SMOOTH)
    def test_linear_field_has_zero_second_gradient(self, basis):
        _, Xp, u, st = setup_1d(basis, lambda X: 0.05 * X)
        d2 = second_displacement_gradient(u, st.node_id, st.hess)
        assert np.abs(d2).max() < 1e-12

    def test_second_gradient_needs_a_hessian_stencil(self):
        grid = CartesianGrid(shape=(N,), dx=DX)
        st = grid.build_stencil(np.array([[0.5]]))
        with pytest.raises(ValueError, match="hessian=True"):
            second_displacement_gradient(np.zeros((grid.n_nodes, 1)),
                                         st.node_id, st.hess)

    @pytest.mark.parametrize("basis", SMOOTH)
    def test_symmetric_in_the_two_material_indices(self, basis):
        grid = CartesianGrid(shape=(12, 12), dx=1.0 / 12,
                             basis=get_basis(basis))
        rng = np.random.default_rng(0)
        Xp = rng.uniform(0.3, 0.7, (20, 2))
        st = grid.build_stencil(Xp, hessian=True)
        u = rng.standard_normal((grid.n_nodes, 2)) * 1e-3
        d2 = second_displacement_gradient(u, st.node_id, st.hess)
        assert np.allclose(d2, np.swapaxes(d2, 2, 3))


class TestEquationNine:
    @pytest.mark.parametrize("basis", SMOOTH)
    def test_matches_the_closed_form(self, basis):
        c, stress = 0.3, 2.5
        _, Xp, u, st = setup_1d(basis, lambda X: c * X ** 2)
        n = Xp.shape[0]

        P = np.full((n, 1, 1), stress)
        V = np.full(n, DX)
        G = configurational_force(P, V, u, st.node_id, st.hess)[:, 0]

        exact = -V * stress * 2.0 * c
        m = interior(Xp)
        assert np.abs(G[m] - exact[m]).max() < 1e-13
        assert np.abs(exact).max() > 1e-3          # not vacuously zero

    @pytest.mark.parametrize("basis", SMOOTH)
    def test_vanishes_for_a_uniform_deformation(self, basis):
        """The property that makes ||G_p|| an error indicator."""
        _, Xp, u, st = setup_1d(basis, lambda X: 0.05 * X)
        n = Xp.shape[0]
        G = configurational_force(np.full((n, 1, 1), 2.5), np.full(n, DX),
                                  u, st.node_id, st.hess)
        assert np.abs(G).max() < 1e-14

    @pytest.mark.parametrize("basis", SMOOTH)
    def test_vanishes_for_an_undeformed_body(self, basis):
        _, Xp, u, st = setup_1d(basis, lambda X: np.zeros_like(X))
        n = Xp.shape[0]
        G = configurational_force(np.full((n, 1, 1), 2.5), np.full(n, DX),
                                  u, st.node_id, st.hess)
        assert np.abs(G).max() < 1e-15

    def test_scales_linearly_with_stress_and_volume(self):
        c = 0.3
        _, Xp, u, st = setup_1d("quadratic", lambda X: c * X ** 2)
        n = Xp.shape[0]

        base = configurational_force(np.full((n, 1, 1), 1.0), np.full(n, DX),
                                     u, st.node_id, st.hess)
        twice_P = configurational_force(np.full((n, 1, 1), 2.0),
                                        np.full(n, DX), u, st.node_id, st.hess)
        twice_V = configurational_force(np.full((n, 1, 1), 1.0),
                                        np.full(n, 2 * DX), u, st.node_id,
                                        st.hess)
        assert np.allclose(twice_P, 2.0 * base)
        assert np.allclose(twice_V, 2.0 * base)

    def test_volume_gradient_term(self):
        c = 0.3
        _, Xp, u, st = setup_1d("quadratic", lambda X: c * X ** 2)
        n = Xp.shape[0]
        P, V = np.full((n, 1, 1), 2.5), np.full(n, DX)
        W = np.full(n, 7.0)
        dV = np.full((n, 1), 0.5)

        without = configurational_force(P, V, u, st.node_id, st.hess)
        with_ = configurational_force(P, V, u, st.node_id, st.hess,
                                      energy=W, dvol_dX=dV)
        assert np.allclose(with_ - without, -W[:, None] * dV)

    def test_volume_gradient_needs_the_energy(self):
        _, Xp, u, st = setup_1d("quadratic", lambda X: 0.3 * X ** 2)
        n = Xp.shape[0]
        with pytest.raises(ValueError, match="without the energy"):
            configurational_force(np.zeros((n, 1, 1)), np.full(n, DX), u,
                                  st.node_id, st.hess,
                                  dvol_dX=np.zeros((n, 1)))


class TestBasisRegularityGate:
    """Eq. (9) is unavailable exactly where the paper says the force is
    unbounded, and that refusal is the finding rather than a limitation."""

    def test_linear_hat_reports_first_order_only(self):
        assert LinearHat.max_derivative == 1

    def test_linear_hat_refuses_a_second_derivative(self):
        with pytest.raises(NotImplementedError, match="singular measure"):
            LinearHat().second_derivatives_1d(np.array([[0.3]]))

    def test_a_hessian_stencil_on_the_hat_refuses(self):
        grid = CartesianGrid(shape=(N,), dx=DX, basis=LinearHat())
        with pytest.raises(NotImplementedError, match="singular measure"):
            grid.build_stencil(np.array([[0.5]]), hessian=True)

    @pytest.mark.parametrize("basis", SMOOTH + ["gimp"])
    def test_smooth_bases_report_second_order(self, basis):
        assert get_basis(basis).max_derivative == 2

    def test_gimp_bound_diverges_as_the_domain_shrinks(self):
        """GIMP bounds d2w only through the particle width, so it recovers the
        hat's singular measure in the limit -- a particle-domain remedy and a
        grid-regularity remedy are not the same fix."""
        b = get_basis("gimp")
        fx = np.linspace(0.52, 1.48, 41).reshape(-1, 1)
        peaks = []
        for lam in (0.45, 0.25, 0.10):
            d2 = b.second_derivatives_1d(fx, np.full_like(fx, lam))
            peaks.append(np.abs(d2).max())
            assert np.isclose(peaks[-1], 1.0 / lam, rtol=1e-12)
        assert peaks[0] < peaks[1] < peaks[2]

    def test_gimp_refuses_at_zero_domain_width(self):
        with pytest.raises(ValueError, match="diverges"):
            get_basis("gimp").second_derivatives_1d(
                np.array([[1.0]]), np.array([[1e-12]]))
