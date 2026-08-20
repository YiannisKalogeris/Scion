"""
The variant-ranking machinery (Section 6).

The full sweep is nine variants over five benchmarks and takes minutes, so it
lives in ``projects/configurational_mpm/ranking.py`` and is run by hand. What is
regression-tested here is the machinery and the few conclusions that are cheap
to reproduce: that the measurement is valid at all (the spatial-momentum control
sits at machine precision, so any pseudomomentum drift is real), and the one
ordering Eq. (9) predicts outright.
"""

from __future__ import annotations

import numpy as np
import pytest

from projects.configurational_mpm import ranking as R


@pytest.fixture(scope="module")
def small(monkeypatch_module=None):
    """Shrink the problem so the suite can afford it."""
    return dict(n_steps=60)


class TestSpearman:
    def test_identical_orderings(self):
        a = ["x", "y", "z", "w"]
        assert R.spearman(a, a) == pytest.approx(1.0)

    def test_reversed_orderings(self):
        a = ["x", "y", "z", "w"]
        assert R.spearman(a, a[::-1]) == pytest.approx(-1.0)

    def test_ignores_labels_missing_from_one_side(self):
        assert R.spearman(["a", "b", "c"], ["a", "b", "c", "d"]) \
            == pytest.approx(1.0)

    def test_too_few_common_labels_is_nan(self):
        assert np.isnan(R.spearman(["a", "b"], ["a", "b"]))


class TestBenchmarks:
    def test_every_flow_is_periodic(self):
        """A field that disagrees with itself across the seam puts a shock
        there and swamps the quantity being measured."""
        for name, flow in R.BENCHMARKS.items():
            lo = np.array([[0.3, 0.0], [0.0, 0.4]])
            hi = np.array([[0.3, 1.0], [1.0, 0.4]])
            assert np.allclose(flow(lo), flow(hi), atol=1e-12), name

    def test_every_flow_has_nonzero_net_momentum(self):
        """Otherwise the drift normalisation divides by ~0 -- the mistake that
        produced ratios of 1e2 in the first attempt."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, (4000, 2))
        for name, flow in R.BENCHMARKS.items():
            assert np.linalg.norm(flow(x).mean(axis=0)) > 0.1, name

    def test_classes_cover_the_benchmarks(self):
        assert set(R.DEFORMATION_CLASS) == set(R.BENCHMARKS)


class TestMeasurement:
    @pytest.mark.parametrize("label", ["BSMPM quad", "BSMPM cubic", "PIC"])
    def test_diagnostics_are_finite(self, label, small):
        v = next(v for v in R.variants() if v.label == label)
        dP, dp, err = R.run_one(v, R.shear_wave, **small)
        assert err is None
        assert np.isfinite(dP) and np.isfinite(dp)

    def test_spatial_momentum_control_is_machine_precision(self, small):
        """The validity check on the whole measurement: if linear momentum did
        not conserve, a pseudomomentum drift would prove nothing."""
        v = next(v for v in R.variants() if v.label == "BSMPM quad")
        _, dp, err = R.run_one(v, R.shear_wave, **small)
        assert err is None
        assert dp < 1e-12

    def test_pseudomomentum_drifts_far_more_than_linear_momentum(self, small):
        """Claim (b), in one assertion."""
        v = next(v for v in R.variants() if v.label == "BSMPM quad")
        dP, dp, err = R.run_one(v, R.shear_wave, **small)
        assert err is None
        assert dP > 1e3 * dp

    def test_linear_hat_is_worse_than_a_cubic_bspline(self, small):
        """Eq. (9) drives G_p with the second material gradient of the basis,
        singular for the hat and continuous for the cubic spline. This is the
        one ordering the mechanism predicts outright."""
        hat = next(v for v in R.variants() if v.label == "MPM (linear)")
        spline = next(v for v in R.variants() if v.label == "BSMPM cubic")
        d_hat, _, e1 = R.run_one(hat, R.shear_wave, **small)
        d_spl, _, e2 = R.run_one(spline, R.shear_wave, **small)
        assert e1 is None and e2 is None
        assert d_hat > 10.0 * d_spl


class TestPICIsAFalsePositive:
    """Recorded deliberately.

    PIC ranks at or near the top on this diagnostic on every benchmark while
    being the most dissipative scheme available. The diagnostic does not
    penalise dissipation, so a ranking that reads it as a quality ordering puts
    the worst transfer near the top. Measured: PIC scores better than FLIP on
    the drift and retains markedly less energy.
    """

    #: PIC only overtakes FLIP on the drift after ~150 steps -- at 60 the two
    #: are within a factor of 1.3 and the ordering has not developed. Using the
    #: shorter run and relaxing the assertion would test nothing.
    N_STEPS = 150

    def test_pic_scores_well_while_dissipating_most(self):
        from scion.core.logging import quiet
        from scion.discretization.mpm import ExplicitMPMStepper, ParticleSet
        from scion.materials.hyperelastic_batched import NeoHookeanCompressible

        out = {}
        for label in ("PIC", "FLIP 0.95"):
            v = next(x for x in R.variants() if x.label == label)
            grid = v.build()
            mat = NeoHookeanCompressible.from_E_nu(R.E, R.NU)
            dx = 1.0 / R.N_GRID
            p = ParticleSet.from_sampler(
                lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
                dx, ppc=R.PPC, density=R.RHO)
            p.v[:] = R.vortex_array(p.x)
            c = mat.wave_speed_factor / np.sqrt(R.RHO)
            s = ExplicitMPMStepper(grid, mat, v.transfer, dt=0.25 * dx / c)
            e0 = p.kinetic_energy()
            with quiet():
                s.run(p, n_steps=self.N_STEPS)
            drift, _, _ = R.run_one(v, R.vortex_array, n_steps=self.N_STEPS)
            out[label] = (drift, p.kinetic_energy() / e0)

        assert out["PIC"][0] < out["FLIP 0.95"][0]      # PIC "wins"
        assert out["PIC"][1] < out["FLIP 0.95"][1]      # while dissipating more
