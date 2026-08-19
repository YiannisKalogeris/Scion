"""Tests for the weakly compressible fluid and the Tait equation of state."""

from __future__ import annotations

import numpy as np
import pytest

from scion.materials.eos import LinearEOS, TaitEOS
from scion.materials.fluid import FluidState, WeaklyCompressibleFluid


class TestTaitEOS:
    def test_zero_pressure_at_rest_density(self):
        assert TaitEOS(1000.0).pressure(np.array([1.0]))[0] == 0.0

    def test_matches_the_closed_form(self):
        eos = TaitEOS(1000.0, gamma=7.0)
        for J in (0.95, 0.99, 1.01, 1.05):
            expected = 1000.0 * ((1.0 / J) ** 7 - 1.0)
            assert np.isclose(eos.pressure_from_jacobian(np.array([J]))[0], expected)

    def test_compression_is_positive_tension_negative(self):
        eos = TaitEOS(1000.0)
        assert eos.pressure_from_jacobian(np.array([0.99]))[0] > 0.0
        assert eos.pressure_from_jacobian(np.array([1.01]))[0] < 0.0

    def test_stiffens_faster_than_linear(self):
        """The point of gamma = 7."""
        tait, lin = TaitEOS(1000.0, 7.0), LinearEOS(1000.0)
        mild = (tait.pressure_from_jacobian(np.array([0.99]))[0]
                / lin.pressure_from_jacobian(np.array([0.99]))[0])
        hard = (tait.pressure_from_jacobian(np.array([0.90]))[0]
                / lin.pressure_from_jacobian(np.array([0.90]))[0])
        assert hard > mild > 1.0

    def test_gamma_one_reduces_to_linear(self):
        tait, lin = TaitEOS(1000.0, gamma=1.0), LinearEOS(1000.0)
        eta = np.array([0.9, 1.0, 1.1])
        assert np.allclose(tait.pressure(eta), lin.pressure(eta))

    def test_bulk_modulus_is_the_tangent_at_rest(self):
        eos = TaitEOS(1000.0, gamma=7.0)
        h = 1e-7
        slope = (eos.pressure(np.array([1.0 + h]))[0]
                 - eos.pressure(np.array([1.0 - h]))[0]) / (2 * h)
        assert np.isclose(slope, eos.bulk_modulus, rtol=1e-6)

    def test_for_flow_speed_targets_the_requested_mach(self):
        eos = TaitEOS.for_flow_speed(density=1000.0, flow_speed=2.0, mach=0.1)
        assert np.isclose(eos.sound_speed(1000.0), 20.0)

    def test_inverted_jacobian_is_rejected(self):
        with pytest.raises(ValueError, match="inverted"):
            TaitEOS(1000.0).pressure_from_jacobian(np.array([-0.5]))

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_invalid_parameters(self, bad):
        with pytest.raises(ValueError):
            TaitEOS(bad)
        with pytest.raises(ValueError):
            TaitEOS(1000.0, gamma=bad)


class TestFluidState:
    def test_starts_at_unit_volume(self):
        st = FluidState.ones(6)
        assert np.allclose(st.J, 1.0)
        assert st.n == 6

    def test_copy_is_independent(self):
        st = FluidState.ones(3)
        other = st.copy()
        other.J[0] = 0.5
        assert st.J[0] == 1.0

    def test_repr_reports_the_range(self):
        assert "3 points" in repr(FluidState.ones(3))


class TestWeaklyCompressibleFluid:
    def test_requires_a_pressure_closure(self):
        with pytest.raises(ValueError, match="eos or a bulk_modulus"):
            WeaklyCompressibleFluid()

    def test_negative_viscosity_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            WeaklyCompressibleFluid(bulk_modulus=1.0, viscosity=-1.0)

    def test_accepts_any_equation_of_state(self):
        f = WeaklyCompressibleFluid(eos=LinearEOS(500.0))
        assert np.isclose(f.pressure(np.array([1.0]))[0], 0.0)
        assert np.isclose(f.wave_speed_factor, np.sqrt(500.0))

    def test_at_rest_there_is_no_stress(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=2.0)
        sigma, st = f.update(None, np.zeros((4, 3, 3)), 0.01, FluidState.ones(4))
        assert np.allclose(sigma, 0.0)
        assert np.allclose(st.J, 1.0)

    def test_pure_shear_is_volume_preserving_and_purely_viscous(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=2.0)
        L = np.zeros((1, 3, 3))
        L[0, 0, 1] = 0.5

        sigma, st = f.update(None, L, 0.01, FluidState.ones(1))
        assert np.isclose(st.J[0], 1.0)
        assert np.isclose(np.trace(sigma[0]), 0.0, atol=1e-12)
        assert np.isclose(sigma[0, 0, 1], 2.0 * 0.5)      # nu * (L + L^T)_xy

    def test_stress_is_symmetric(self):
        rng = np.random.default_rng(3)
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=1.5)
        L = rng.standard_normal((16, 3, 3))
        sigma, _ = f.update(None, L, 0.001, FluidState.ones(16))
        assert np.allclose(sigma, np.swapaxes(sigma, 1, 2))

    def test_compression_produces_a_compressive_mean_stress(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=0.0)
        L = -np.eye(3)[None] * 0.1

        sigma, st = f.update(None, L, 0.01, FluidState.ones(1))
        assert st.J[0] < 1.0
        assert np.isclose(np.trace(sigma[0]) / 3.0, -f.pressure(st.J)[0])

    def test_inviscid_fluid_carries_no_shear(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=0.0)
        L = np.zeros((1, 3, 3))
        L[0, 0, 1] = 0.5
        sigma, _ = f.update(None, L, 0.01, FluidState.ones(1))
        assert np.allclose(sigma, 0.0)

    def test_volume_follows_the_trace_of_the_velocity_gradient(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0)
        L = np.eye(3)[None] * 0.2                  # tr L = 0.6
        _, st = f.update(None, L, 0.1, FluidState.ones(1))
        assert np.isclose(st.J[0], 1.0 + 0.1 * 0.6)

    def test_state_is_created_when_none_is_given(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0)
        _, st = f.update(None, np.zeros((5, 3, 3)), 0.01, None)
        assert st.n == 5

    def test_the_input_state_is_never_mutated(self):
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0)
        st = FluidState.ones(2)
        f.update(None, -np.tile(np.eye(3), (2, 1, 1)) * 0.5, 0.1, st)
        assert np.allclose(st.J, 1.0)

    def test_the_deformation_gradient_argument_is_ignored(self):
        """A fluid has no reference configuration; the signature only exists
        so it shares the stepper path with the solids."""
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0, viscosity=1.0)
        L = np.zeros((1, 3, 3))
        L[0, 0, 1] = 0.3

        a, _ = f.update(None, L, 0.01, FluidState.ones(1))
        b, _ = f.update(np.full((1, 3, 3), 17.0), L, 0.01, FluidState.ones(1))
        assert np.allclose(a, b)

    def test_first_order_volume_update_is_not_det_F(self):
        """Documents a real limitation: J drifts and nothing corrects it."""
        f = WeaklyCompressibleFluid(bulk_modulus=1000.0)
        L = np.eye(3)[None] * 0.1
        dt = 0.05

        st = FluidState.ones(1)
        for _ in range(20):
            _, st = f.update(None, L, dt, st)

        exact = np.exp(3.0 * 0.1 * 20 * dt)         # det of exp(L t)
        assert not np.isclose(st.J[0], exact, rtol=1e-3)
        assert np.isclose(st.J[0], (1.0 + dt * 0.3) ** 20)
