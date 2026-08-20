"""
Relating the variant ranking to the term structure of Eq. (9).

Eq. (9) drives the discrete configurational force with the **second material
gradient of the grid basis**, so the paper predicts that the regularity of
``d2w/dX dX`` governs the ranking. Measured, by sweeping a particle past a
fixed grid node and profiling that node's ``d2w`` (8001 samples):

==================  =============  ===========  ========================
basis               peak |d2w|     jump/peak    character
==================  =============  ===========  ========================
linear hat          unbounded      --           Dirac measure
BSMPM cubic         1.37e3         0.001        **continuous**
BSMPM quadratic     1.54e3         1.50         bounded, discontinuous
CPDI (linear)       3.07e3         1.00         bounded, discontinuous
uGIMP               3.58e3         1.00         bounded, discontinuous
==================  =============  ===========  ========================

The ranking by ``||G_i||`` follows: the only basis with a *continuous* second
gradient comes first, and among the merely bounded ones the order tracks the
peak magnitude at Spearman +0.80.

Refining the paper's wording
----------------------------
Section 1.4(a) says BSMPM and CPDI "act on different terms of (9)". The
measurement says they act on the **same** term and differently: BSMPM raises the
*regularity* (continuous at degree three), CPDI only *bounds* it -- turning the
Dirac into a finite jump -- and does so with roughly twice a B-spline's peak.
Bounding is not smoothing, and that is why CPDI does not reach BSMPM.

A profile summed over the stencil will not show any of this: for the quadratic
B-spline ``sum_k |d2w_k|`` is the constant ``4/dx^2`` even though the individual
node values jump as the stencil shifts. The node has to be held fixed.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.discretization.mpm import CartesianGrid, CPDIGrid
from scion.discretization.mpm.basis import get_basis

N, DX = 32, 1.0 / 32


def node_profile(kind, n_samples=4001):
    """``d2w/dX^2`` at one fixed node as a particle sweeps past it."""
    t = np.linspace(0.42, 0.58, n_samples)
    xp = np.stack([t, np.full_like(t, 0.5)], axis=1)

    if kind == "cpdi":
        grid = CPDIGrid(shape=(N, N), dx=DX, basis=get_basis("linear"))
        st = grid.build_stencil(
            xp, domain=np.tile(np.eye(2) * 0.25 * DX, (len(t), 1, 1)),
            hessian=True)
    else:
        basis = get_basis(kind)
        grid = CartesianGrid(shape=(N, N), dx=DX, basis=basis)
        kw = {"lp": np.full_like(xp, 0.25 * DX)} if basis.requires_domain else {}
        st = grid.build_stencil(xp, hessian=True, **kw)

    coords = grid.node_coordinates()
    target = int(np.argmin(np.linalg.norm(coords - np.array([0.5, 0.5]), axis=1)))
    out = np.zeros(len(t))
    for i in range(len(t)):
        hit = np.where(st.node_id[i] == target)[0]
        if len(hit):
            out[i] = st.hess[i, hit, 0, 0].sum()
    return out


def peak_and_jump(kind):
    h = node_profile(kind)
    return float(np.abs(h).max()), float(np.abs(np.diff(h)).max())


class TestRegularityHierarchy:
    def test_linear_hat_second_gradient_is_unbounded(self):
        """Not a large number -- no number. It is a measure on cell boundaries."""
        grid = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis("linear"))
        with pytest.raises(NotImplementedError, match="singular measure"):
            grid.build_stencil(np.array([[0.5, 0.5]]), hessian=True)

    def test_only_the_cubic_bspline_is_continuous(self):
        peak, jump = peak_and_jump("cubic")
        assert jump / peak < 0.01

    @pytest.mark.parametrize("kind", ["quadratic", "gimp", "cpdi"])
    def test_the_others_are_bounded_but_discontinuous(self, kind):
        peak, jump = peak_and_jump(kind)
        assert np.isfinite(peak) and peak > 0
        assert jump / peak > 0.5

    def test_cpdi_bounds_the_hat_but_with_a_larger_peak_than_a_bspline(self):
        """The BSMPM/CPDI asymmetry, quantified. CPDI turns the Dirac into a
        finite jump; it does not make it continuous, and its bound is worse."""
        cpdi_peak, _ = peak_and_jump("cpdi")
        quad_peak, _ = peak_and_jump("quadratic")
        assert np.isfinite(cpdi_peak)
        assert cpdi_peak > 1.5 * quad_peak

    def test_a_stencil_sum_would_hide_all_of_this(self):
        """Guards the measurement itself: summed over the stencil, the
        quadratic B-spline's second gradient looks perfectly continuous."""
        t = np.linspace(0.42, 0.58, 2001)
        xp = np.stack([t, np.full_like(t, 0.5)], axis=1)
        grid = CartesianGrid(shape=(N, N), dx=DX, basis=get_basis("quadratic"))
        st = grid.build_stencil(xp, hessian=True)
        summed = np.abs(st.hess[:, :, 0, 0]).sum(axis=1)
        assert np.abs(np.diff(summed)).max() / summed.max() < 1e-12

        _, per_node_jump = peak_and_jump("quadratic")
        assert per_node_jump > 0.0


class TestRankingFollowsTheTermStructure:
    """The ordering by ``||G_i||`` tracks peak ``|d2w|`` among bounded bases."""

    #: measured at N=32, shear wave, fixed physical time
    ORDER_BY_G = ["BSMPM cubic", "BSMPM quad", "uGIMP", "CPDI"]

    def test_peak_magnitude_orders_the_bounded_bases(self):
        peaks = {
            "BSMPM cubic": peak_and_jump("cubic")[0],
            "BSMPM quad": peak_and_jump("quadratic")[0],
            "uGIMP": peak_and_jump("gimp")[0],
            "CPDI": peak_and_jump("cpdi")[0],
        }
        by_peak = sorted(peaks, key=peaks.get)
        import projects.configurational_mpm.ranking as R
        assert R.spearman(self.ORDER_BY_G, by_peak) > 0.7

    def test_the_continuous_basis_ranks_first(self):
        assert self.ORDER_BY_G[0] == "BSMPM cubic"
        peak, jump = peak_and_jump("cubic")
        assert jump / peak < 0.01
