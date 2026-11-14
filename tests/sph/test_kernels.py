"""
The kernels, checked against their defining property rather than against
their own formulas.

A transcribed normalising constant is the classic silent error here: every
qualitative check -- positive, smooth, compactly supported, peaked at zero --
passes with the wrong constant, and the answer is then wrong by a factor that
looks like a physical effect. The only check with power over it is the
integral, so that is what these tests do, in all three dimensions, because
the three constants are different even where the radial shape is the same.
"""

import numpy as np
import pytest
from scipy import integrate

from scion.discretization.sph import KERNELS, CubicSpline, WendlandC2, get_kernel

KERNEL_LIST = [WendlandC2(), CubicSpline()]
H = 0.013


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
def test_integrates_to_one_in_every_dimension(kernel):
    support = kernel.support(H)
    one_d, _ = integrate.quad(lambda r: kernel.W(r, H, 1), -support, support,
                              limit=200)
    two_d, _ = integrate.quad(lambda r: kernel.W(r, H, 2) * 2 * np.pi * r,
                              0.0, support, limit=200)
    three_d, _ = integrate.quad(lambda r: kernel.W(r, H, 3) * 4 * np.pi * r * r,
                                0.0, support, limit=200)
    assert one_d == pytest.approx(1.0, abs=1e-10)
    assert two_d == pytest.approx(1.0, abs=1e-10)
    assert three_d == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_derivative_against_central_differences(kernel, dim):
    r = np.linspace(0.05, 1.95, 40) * H
    step = 1e-7 * H
    finite = (kernel.W(r + step, H, dim) - kernel.W(r - step, H, dim)) / (2 * step)
    exact = kernel.dW_dr(r, H, dim)
    assert np.abs(finite - exact).max() <= 1e-5 * np.abs(exact).max()


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
def test_support_is_two_smoothing_lengths(kernel):
    """Not ``h``. Half the SPH literature uses the other convention, and a
    kernel truncated at ``h`` loses most of its neighbours while still
    looking like a kernel."""
    assert kernel.support(H) == pytest.approx(2.0 * H)
    assert kernel.W(2.0 * H, H, 2) == 0.0
    assert kernel.dW_dr(2.0 * H, H, 2) == 0.0
    assert kernel.W(1.999 * H, H, 2) > 0.0
    assert kernel.dW_dr(1.999 * H, H, 2) < 0.0


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
def test_gradient_is_negative_over_the_whole_support(kernel):
    """The sign the correction matrix relies on: ``A = sum (-dW/dr) r V ee``
    is positive semi-definite only because of it."""
    r = np.linspace(1e-6, 2.0, 200) * H
    inside = r < kernel.support(H)
    assert (kernel.dW_dr(r[inside], H, 2) < 0.0).all()


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
def test_distance_is_unsigned(kernel):
    """``W`` takes a distance, so a negative argument is its magnitude; a
    one-dimensional caller passing a signed separation otherwise evaluates
    the shape function outside its domain and gets a plausible number."""
    assert kernel.W(-0.5 * H, H, 1) == kernel.W(0.5 * H, H, 1)


def test_registry_round_trips():
    for name, kernel in KERNELS.items():
        assert get_kernel(name) is kernel
        assert get_kernel(kernel) is kernel
    with pytest.raises(KeyError):
        get_kernel("gaussian")


@pytest.mark.parametrize("kernel", KERNEL_LIST, ids=lambda k: k.name)
def test_refuses_an_unsupported_dimension(kernel):
    with pytest.raises(ValueError):
        kernel.W(0.5 * H, H, 4)
