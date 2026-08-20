"""
Which of the two diagnostics in claim (c) actually indicates solution quality.

Claim (c) offers ``||G_p||`` and the production rate of ``P_h`` as two
indicators of the same thing. Measured over three flows and three integration
times, they are indicators of *different* things, and only one of them tracks
quality:

======================  ==========================================
diagnostic              rank correlation with energy retention
======================  ==========================================
``||G_i||`` (Eq. 10)    +0.68 to +0.95, mean +0.81
``P_h`` production      -0.27 to +0.30, mean +0.01
======================  ==========================================

PIC is the clearest single case: worst scheme by energy retention, ranked 2nd of
9 by ``P_h`` at every run length, and 6th-9th by ``G_i``.

Energy retention is used as the quality proxy because it is independent of both
diagnostics and is the axis the MPM literature already agrees on.
"""

from __future__ import annotations

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import ExplicitMPMStepper, ParticleSet
from scion.materials.hyperelastic_batched import NeoHookeanCompressible
import projects.configurational_mpm.ranking as R

STEPS = 150


def energy_ordering(flow, steps=STEPS):
    """Variants ordered best-retention-first."""
    ke = {}
    for v in R.variants():
        grid = v.build()
        mat = NeoHookeanCompressible.from_E_nu(R.E, R.NU)
        dx = 1.0 / R.N_GRID
        p = ParticleSet.from_sampler(
            lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
            dx, ppc=R.PPC, density=R.RHO)
        p.v[:] = flow(p.x)
        c = mat.wave_speed_factor / np.sqrt(R.RHO)
        s = ExplicitMPMStepper(grid, mat, v.transfer, dt=0.25 * dx / c)
        e0 = p.kinetic_energy()
        try:
            with quiet():
                s.run(p, n_steps=steps)
            if np.all(np.isfinite(p.F)):
                ke[v.label] = p.kinetic_energy() / e0
        except Exception:                                    # noqa: BLE001
            pass
    return [k for k, _ in sorted(ke.items(), key=lambda t: -t[1])], ke


@pytest.fixture(scope="module")
def measured():
    flow = R.vortex_array
    rows = []
    for v in R.variants():
        dP, gi, err = R.run_one_with_G(v, flow, STEPS)
        if err is None:
            rows.append((v.label, dP, gi))
    by_P = [r[0] for r in sorted(rows, key=lambda t: t[1])]
    by_G = [r[0] for r in sorted(rows, key=lambda t: t[2])]
    by_ke, ke = energy_ordering(flow)
    return dict(by_P=by_P, by_G=by_G, by_ke=by_ke, ke=ke)


class TestDiagnosticsDisagree:
    def test_the_two_orderings_are_not_the_same(self, measured):
        """Claim (c) treats them as interchangeable. They are not."""
        rho = R.spearman(measured["by_P"], measured["by_G"])
        assert rho < 0.5, rho

    def test_configurational_force_tracks_quality(self, measured):
        rho = R.spearman(measured["by_G"], measured["by_ke"])
        assert rho > 0.6, rho

    def test_pseudomomentum_does_not(self, measured):
        rho = R.spearman(measured["by_P"], measured["by_ke"])
        assert rho < 0.5, rho

    def test_G_beats_P_as_a_quality_indicator(self, measured):
        g = R.spearman(measured["by_G"], measured["by_ke"])
        p = R.spearman(measured["by_P"], measured["by_ke"])
        assert g > p + 0.3, (g, p)


class TestPICIsTheDiscriminatingCase:
    def test_pic_retains_the_least_energy(self, measured):
        assert measured["by_ke"][-1] == "PIC"

    def test_pseudomomentum_ranks_pic_near_the_top(self, measured):
        assert measured["by_P"].index("PIC") <= 2

    def test_configurational_force_ranks_pic_near_the_bottom(self, measured):
        n = len(measured["by_G"])
        assert measured["by_G"].index("PIC") >= n - 4


class TestNodalMaterialForce:
    def test_vanishes_for_an_undeformed_body(self):
        """The property that makes it admissible at all."""
        from scion.discretization.mpm import APIC, CartesianGrid
        from scion.discretization.mpm.basis import get_basis

        grid = CartesianGrid(shape=(16, 16), dx=1.0 / 16,
                             basis=get_basis("quadratic"), periodic=True)
        mat = NeoHookeanCompressible.from_E_nu(R.E, R.NU)
        p = ParticleSet.from_sampler(
            lambda q: np.ones(len(q), dtype=bool), ((0.0, 0.0), (1.0, 1.0)),
            1.0 / 16, ppc=2, density=R.RHO)
        st = grid.build_stencil(p.x)
        assert R.nodal_material_force_norm(p, grid, st, mat) < 1e-12

    def test_is_positive_once_deformed(self, measured):
        _, gi, err = R.run_one_with_G(R.variants()[1], R.vortex_array, 60)
        assert err is None and gi > 0.0
