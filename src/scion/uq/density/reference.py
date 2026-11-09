"""Closed-form solutions the density methods are tested against.

Mirrors ``uq/forward/reference.py`` and ``uq/bayesian/reference.py``: every
method in this subpackage is checked against a problem whose answer is known
analytically rather than against another numerical method.

The problem is Li & Chen's Example 1 -- the free oscillation of an undamped
SDOF system whose natural frequency is a random variable:

    X'' + w^2 X = 0,   X(0) = x0,  X'(0) = 0   =>   X(t) = x0 cos(w t)

Its appendix gives the time-dependent PDF of the displacement, which is the
only check here that tests a *density* rather than a moment. That matters: a
scheme with the wrong dissipation reproduces the mean and the standard
deviation and still gets the shape wrong, and the shape is the whole point of
the method.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import lognorm

#: The paper's Example 1: frequency uniform on this interval, in rad/s.
OMEGA_BOUNDS = (1.25 * np.pi, 1.75 * np.pi)
#: and the deterministic initial displacement, in metres.
X0 = 0.1


def sdof_displacement_pdf(x, t, x0: float = X0, bounds=OMEGA_BOUNDS):
    """PDF of ``x0 cos(w t)`` for ``w`` uniform on ``bounds``, at time ``t``.

    The appendix's Eq. (A3). Substituting ``eta = w t`` and inverting
    ``x = x0 cos(eta)`` branch by branch gives

        f_X(x, t) = sum_k f_eta(eta_k) / sqrt(x0^2 - x^2)

    over every ``eta_k >= 0`` with ``x0 cos(eta_k) = x``, i.e.
    ``eta_k = 2 l pi + arccos(x/x0)`` and ``2 l pi + 2 pi - arccos(x/x0)``.
    With a uniform frequency, ``f_eta`` is the constant ``1/(t (b - a))`` on
    ``a <= eta/t <= b`` and zero outside, so the density is a piecewise
    constant *count of branches* divided by ``sqrt(x0^2 - x^2)``.

    Two consequences worth knowing before comparing against a numerical
    solution, because both are properties of the exact answer and not of the
    scheme:

    - it is **singular** at ``x = +-x0``, as every ``cos`` transformation is,
      so a cell-centred numerical density can never match it pointwise there;
    - it is **discontinuous** wherever a branch enters or leaves the frequency
      band, which is what makes the PDF "much irregular and far from
      well-known regular distribution types" in the paper's words.

    ``t = 0`` is not supported: the density is a Dirac there, which is the
    initial condition rather than a function.
    """
    t = float(t)
    if t <= 0.0:
        raise ValueError("the PDF at t = 0 is a Dirac delta, not a function")
    a, b = (float(v) for v in bounds)
    if not b > a:
        raise ValueError(f"frequency bounds must be increasing; got {bounds}")

    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape)
    inside = np.abs(x) < x0            # the open interval; the ends are poles
    if not inside.any():
        return out

    principal = np.arccos(np.clip(x[inside] / x0, -1.0, 1.0))
    # enough wraps to cover eta up to b t, plus one for safety at the edge
    n_wrap = int(np.ceil(b * t / (2.0 * np.pi))) + 1
    count = np.zeros(principal.shape)
    for wrap in range(n_wrap + 1):
        for eta in (2.0 * np.pi * wrap + principal,
                    2.0 * np.pi * (wrap + 1) - principal):
            count += ((eta >= a * t) & (eta <= b * t)).astype(float)

    out[inside] = count / (t * (b - a) * np.sqrt(x0 ** 2 - x[inside] ** 2))
    return out


def sdof_displacement_moments(t, x0: float = X0, bounds=OMEGA_BOUNDS):
    """``(mean, std)`` of ``x0 cos(w t)``, exactly.

    ``E[cos(w t)]`` and ``E[cos^2(w t)]`` are elementary for a uniform ``w``,
    which makes this the Fig. 2 comparison. Note the mean decays like ``1/t``
    while the standard deviation settles -- the paper's observation that an
    *undamped* system's mean response attenuates, which is a property of the
    ensemble and not of any damping.
    """
    t = np.asarray(t, dtype=float)
    a, b = (float(v) for v in bounds)
    width = b - a

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = x0 * np.where(
            t == 0.0, 1.0,
            (np.sin(b * t) - np.sin(a * t)) / (width * np.where(t == 0.0, 1.0, t)))
        second = x0 ** 2 * 0.5 * (1.0 + np.where(
            t == 0.0, 1.0,
            (np.sin(2 * b * t) - np.sin(2 * a * t))
            / (2.0 * width * np.where(t == 0.0, 1.0, t))))
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.0))


def sdof_response(z, t_grid, x0: float = X0):
    """The deterministic analysis for one frequency: ``(x, x_dot)``.

    This is the ``model`` seam for the SDOF example -- the analytical solution
    stands in for a time integration, so a GDEE test is not also a test of
    somebody's Newmark implementation. ``z`` is ``(1,)``, holding ``w``.
    """
    omega = float(np.asarray(z).ravel()[0])
    t_grid = np.asarray(t_grid, dtype=float)
    return x0 * np.cos(omega * t_grid), -x0 * omega * np.sin(omega * t_grid)


# ----------------------------------------------------------------------
# Chen & Yang's Example 3: Van der Pol oscillator with random damping
# ----------------------------------------------------------------------
#
# A second closed form, and a much more discriminating one: the input is
# skewed rather than uniform, the mapping is nonlinear, and the exact K-L
# divergences span four orders of magnitude across the three instants the
# paper reports. Example 1 alone could not tell a good implementation from a
# lucky one.


def vanderpol_response(z, t_grid):
    """``Y(t) = cos t - (1/8) Theta (t cos^3 t + sin^3 t)``, their Eq. (45).

    Jiang and Li's approximate analytical solution of
    ``Y'' + Y + Theta Y^2 Y' = 0``. Affine in ``Theta`` at fixed ``t``, which
    is what makes the density below elementary -- the nonlinearity is in the
    oscillator, not in the map from the random variable.
    """
    theta = float(np.asarray(z).ravel()[0])
    t_grid = np.asarray(t_grid, dtype=float)
    return np.cos(t_grid) - theta * vanderpol_coefficient(t_grid) / 8.0


def vanderpol_coefficient(t):
    """``A(t) = t cos^3 t + sin^3 t``, the factor multiplying the damping."""
    t = np.asarray(t, dtype=float)
    return t * np.cos(t) ** 3 + np.sin(t) ** 3


def vanderpol_pdf(y, t):
    """PDF of the response at ``t`` for ``Theta`` lognormal with ``s = 1``.

    Derived here rather than transcribed. The paper's Eq. (46) prints
    ``t cos^3 t - sin^3 t`` where its Eq. (45) has a plus, and the two
    disagree by a factor of seven at ``t = 4.2`` -- so one of them is a typo
    and the safe route is the change of variables, which for an affine map
    is unambiguous:

        y = cos t - A(t) theta / 8   =>   p_Y(y) = p_Theta(8 (cos t - y) / A)
                                                   * 8 / |A|

    Checked rather than assumed: it integrates to 1.0000 and its mean agrees
    with a 400,000-sample Monte Carlo to three decimals at ``t`` = 3, 4.2
    and 9.
    """
    coefficient = float(vanderpol_coefficient(t))
    theta = 8.0 * (np.cos(float(t)) - np.asarray(y, dtype=float)) / coefficient
    density = np.where(theta > 0.0,
                       lognorm.pdf(np.clip(theta, 1e-300, None), s=1.0), 0.0)
    return density * 8.0 / abs(coefficient)
