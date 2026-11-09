"""The direct probability integral method: a probability-weighted kernel
estimate, with no PDE at all.

Chen & Yang observe that the probability density integral equation

    p_Y(y, t) = int p_Theta(theta) delta[y - g(theta, t)] dtheta

holds *at every instant independently*, so there is nothing to march. Replace
the Dirac by a Gaussian of width ``sigma`` and discretise the integral on the
same representative points PDEM uses, and the whole method is their Eq. (36):

    p(y, t) = sum_q P_q N(y; g(theta_q, t), sigma)

That is a weighted kernel density estimate whose weights are assigned
probabilities rather than ``1/N``, and whose sample points are deterministic
responses rather than random draws.

**What it buys over PDEM**, and each of these is structural rather than a
matter of degree:

- **No CFL condition and no time step at all.** Each instant is computed from
  the responses at that instant, so the output times are whatever the
  deterministic analysis produced and the cost is one pass over them.
- **The bandwidth is explicit.** PDEM's smoothing is its scheme's numerical
  dissipation, so refining the response grid narrows the kernel and, past a
  point, resolves the individual trajectories and ruins the answer -- see the
  table in :mod:`scion.uq.density.gdee`. Here ``sigma`` and the grid are
  independent: the grid only decides how finely the answer is *drawn*.
- **Static problems are the same calculation.** With no time derivative in
  the formulation, a time-independent system is one instant, which is why the
  paper's title covers both. ``t_grid`` is optional for that reason.
- **The velocity is not needed.** PDEM advects at ``Xdot_l``; DPIM evaluates
  ``X_l``. So the model seam is narrower, and a response history from any
  source will do.

**What it costs**, stated because the paper is quiet about it: the conserved
total probability is no longer structural. PDEM transports a density with a
space-uniform velocity, so a finite volume scheme conserves it to round-off;
here ``sum_j p_j dy`` is a midpoint quadrature of a sum of Gaussians, and it
is one only to the extent that the kernels are resolved by the grid and
contained by it. The returned ``mass`` is the diagnostic, and a tolerance is
checked rather than assumed.

**The kernel must span at least about one and a half cells**, and then the
quadrature is not merely adequate but exact to round-off. Midpoint
integration of a Gaussian converges geometrically rather than
algebraically, so the transition is abrupt rather than gradual. Measured,
``max|int p dy - 1|`` against ``sigma/dy``:

    sigma/dy      0.19     0.37     0.75     1.49     2.99
    mass error    1.1e+0   1.1e-1   3.2e-5   8.4e-15  8.4e-15

So there is no accuracy argument for a grid finer than ``sigma``, only a
drawing one, and there is a hard failure in going finer while holding
``sigma`` fixed. ``solve_dpim`` reports the ratio when it refuses.

**The paper's bandwidth rule does not depend on ``N``, so DPIM as published
stops converging.** Chen & Yang take ``sigma = r max|y|`` with ``r`` between
0.005 and 0.01. A kernel width is a smoothing scale and has to shrink as the
points close up, or the smoothing bias floors the error. Example 1 at
``t = 1 s``, ``L1`` against the closed form:

    points               20       40       80      160      320
    sigma = 0.01 max|y|  0.8766   0.2697   0.0168   0.0143   0.0145
    sigma = 2 x spacing  0.0634   0.0311   0.0158   0.0083   0.0047

The paper's rule is 14x worse at 20 points, happens to be near optimal at 80,
and **plateaus at about 0.014 thereafter** -- adding points buys nothing.
Scaling instead with the median gap between adjacent representative responses
tracks the measured optimum to within 14% at every count and keeps halving.
The default here is therefore ``2 x response_spacing(x)``;
:func:`paper_bandwidth` reproduces the published rule for comparison.

The factor of two is fitted on this problem and the transferable part is the
*scaling*, not the constant: the optimum sat at 1.58 to 2.09 gaps over a 16x
range in ``N``. The reason the gap is the right scale is that these are
deterministic, well-spread points rather than random draws, so the spacing
falls like ``1/N`` and not like ``N^(-1/2)``, and the usual Silverman and
Scott rules -- written for samples -- do not apply.

**And a fixed ``sigma`` is a global bandwidth, which decides what the method
can do.** The representative responses bunch wherever ``dg/dtheta`` is small
and spread where it is large, so one width is right everywhere only if they
are nearly evenly spaced. The useful diagnostic is the **largest gap over the
median gap**, and it separates the cases cleanly:

    y = z1 + z2, uniform inputs     max/median gap  1.0     L1 converges
                                    (900 / 3600 / 14400 points:
                                     0.0085 / 0.0021 / 0.0005)

    y = z^2,     normal input       max/median gap  96 -> 815   L1 does not
                                    (100 / 200 / 400 / 800 points:
                                     0.228 / 0.211 / 0.218 / 0.212)

The second case stalls because its tail is sparse -- the largest gap between
adjacent responses is 1.99 where the median is 0.0048 -- so the tail needs a
wide kernel while the ``y^(-1/2)`` singularity at the origin needs a narrow
one. Worse, that ratio **doubles with the point count** (96, 189, 416, 815 at
100 to 800 points), because the largest gap is pinned by the outermost
quantile while the median gap halves. Adding points therefore cannot rescue a
global bandwidth here; it makes the mismatch wider. Sweeping ``sigma`` by hand at 400 points gives 0.72, 0.38, 0.21, 0.11,
0.059 for 0.002 to 0.05: still falling at the widest, i.e. no interior
optimum, which is the signature of one bandwidth being asked to do two jobs.
Note the moments converge perfectly well there (mean 0.9873 to 0.9984) --
only the density stalls, so a moment check would not reveal this.

An adaptive bandwidth is the fix and is not implemented. Until it is, check
the ratio before trusting a DPIM density on a strongly nonlinear mapping.
"""
from __future__ import annotations

import warnings

import numpy as np

from scion.uq.density.gdee import DensityEvolution, ProbabilityNotConserved
from scion.uq.density.points import RepresentativePoints

#: Default bandwidth, as a multiple of the median gap between adjacent
#: representative responses. See the module docstring for why this replaces
#: the paper's ``sigma = 0.01 max|y|``.
SIGMA_FACTOR = 2.0

#: The paper's own ratio, kept so :func:`paper_bandwidth` can reproduce it.
SIGMA_RATIO = 0.01


def paper_bandwidth(x, ratio: float = SIGMA_RATIO) -> float:
    """Chen & Yang's rule, ``sigma = r max|y|`` with ``r`` in 0.005 to 0.01.

    Provided because it is what the paper specifies and a comparison against
    the published results needs it, not because it is recommended -- it does
    not depend on the number of representative points, which is what makes
    the method stop converging. See the module docstring.
    """
    return float(ratio * np.max(np.abs(np.asarray(x, dtype=float))))


#: Above this ratio of largest to median gap between adjacent responses, a
#: single bandwidth cannot serve the whole axis and the spacing rule
#: undersmooths badly. Chosen from the measured cases: a tensor lattice sits
#: at 1.0, a Sobol-based GF set in two dimensions at 33.
UNIFORMITY_LIMIT = 5.0


def spacing_uniformity(x) -> float:
    """Largest gap over median gap between adjacent distinct responses.

    The diagnostic that says whether one bandwidth can serve the whole axis.
    A tensor lattice of a smooth mapping gives 1.0 -- perfectly even -- and
    anything built on a quasi-random point set gives tens, because low
    discrepancy in the *input* space says nothing about how evenly the
    responses land.
    """
    x = np.asarray(x, dtype=float)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    gaps = np.diff(np.sort(x, axis=0), axis=0)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return 1.0
    return float(positive.max() / np.median(positive))


def response_spacing(x) -> float:
    """Median gap between adjacent *distinct* representative responses.

    Zero gaps are dropped rather than averaged in, and that is not a detail.
    An even function of a symmetric input -- ``y = z^2`` on a lattice
    symmetric about zero, say -- maps points in pairs, so half the gaps are
    exactly zero and a plain median returns zero, which is not a bandwidth.
    A dynamic run has the same problem at ``t = 0``, where every realisation
    starts from the same deterministic initial condition. Dropping them
    measures the spacing of the distinct response *locations*, which is what
    a kernel has to bridge.

    Returns ``0.0`` when every response coincides, leaving the caller to
    choose a fallback; there is no spacing in that case to scale by.
    """
    x = np.asarray(x, dtype=float)
    # a 1-D array is one response per point, as in the static case -- NOT one
    # point at many times, which is what atleast_2d would make of it
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    if x.ndim != 2:
        raise ValueError(f"responses must be (n_points, n_times) or "
                         f"(n_points,); got shape {x.shape}")
    if x.shape[0] < 2:
        raise ValueError("a spacing needs at least two representative points")
    gaps = np.diff(np.sort(x, axis=0), axis=0)
    positive = gaps[gaps > 0.0]
    return float(np.median(positive)) if positive.size else 0.0


def knn_bandwidth(x, k: int = 3, factor: float = SIGMA_FACTOR) -> np.ndarray:
    """Per-point bandwidth from the k-th nearest neighbouring response.

    ``sigma_q = (factor / k) * d_k(x_q)``, where ``d_k`` is the distance to
    the k-th nearest other response at that instant. The ``1/k`` is what
    makes this a strict generalisation rather than a different rule: on an
    evenly spaced set ``d_k = k * gap``, so it reduces exactly to
    ``factor * gap``, the fixed rule :func:`response_spacing` implements.

    Returns ``(n_points, n_times)``, which :func:`solve_dpim` accepts as
    ``sigma``. The *sample-point* form -- a width per representative point
    rather than per evaluation point -- is the one that matters: each kernel
    still integrates to one, so ``sum_q P_q = 1`` still makes the density
    integrate to one. The balloon form, with the width depending on ``y``,
    does not and would cost the one exact property the method has.

    **Use it where the mapping stretches the points unevenly, not
    everywhere.** It is opt-in for that reason. ``L1`` against the closed
    forms, ``k = 3``:

        case                       fixed      knn      verdict
        chi-square, y = z^2        0.2281    0.0211    and it converges:
          N = 100 / 400 / 800      0.2181    0.0042    0.0211 / 0.0042 /
                                   0.2113    0.0018    0.0018, where fixed
                                                       is flat. 120x at 800
        SDOF, N = 160              0.0086    0.0203    2.3x worse, at any
        SDOF, N = 320              0.0044    0.0100    grid resolution
        GF point set, 1024         0.3788    0.3299    barely helps

    So it rescues a strongly nonlinear mapping of a *lattice*, where the
    fixed rule cannot converge at all, and costs a factor of 2.3 where the
    fixed rule was already near optimal -- the SDOF responses are dense
    exactly where the density is large, so a uniform width is already right
    there and making it vary only adds a varying bias.

    It does **not** rescue a scattered point set, which was the hope. There
    the responses are a sample rather than a grid, so the right bandwidth
    scales as ``N^(-1/5)`` and is tens of times the local spacing; following
    the local spacing, which is what this does, is the wrong instinct
    entirely. High-dimensional GF sets plus densities remain open.
    """
    x = np.asarray(x, dtype=float)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    n_points, n_times = x.shape
    if k < 1:
        raise ValueError("k must be at least one")
    if k >= n_points:
        raise ValueError(f"k = {k} needs more than {n_points} points")
    if not factor > 0.0:
        raise ValueError("factor must be positive")

    index = np.arange(n_points)
    low = np.clip(index - k, 0, n_points - 1)
    high = np.clip(index + k, 0, n_points - 1)
    sigma = np.empty((n_points, n_times))

    # An instant where every realisation coincides -- t = 0 of any dynamic
    # problem, where the initial condition is deterministic -- has no local
    # scale at all, and inventing one there is how this goes wrong. A
    # constant unrelated to the problem put sigma at 2.0 on a response of
    # size 0.1; a tiny fraction of the global span put it below the cell
    # size and forced an eightfold finer grid for one useless instant. The
    # density there is a Dirac whatever is chosen, so such instants borrow
    # the median width from the instants that do have a scale.
    global_span = float(x.max() - x.min())
    floor = 0.0 if global_span > 0.0 else 1.0
    for j in range(n_times):
        values = x[:, j]
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        # one-sided at the ends, where there is no k-th neighbour below or
        # above; the wider of the two sides is the honest local scale
        gap = np.maximum(ordered[index] - ordered[low],
                         ordered[high] - ordered[index]) / k
        column = np.empty(n_points)
        column[order] = gap
        sigma[:, j] = factor * np.maximum(column, floor)

    degenerate = sigma.max(axis=0) <= 0.0
    if degenerate.all():
        return np.full_like(sigma, factor if global_span <= 0.0
                            else factor * global_span)
    if degenerate.any():
        sigma[:, degenerate] = np.median(sigma[:, ~degenerate])
    return sigma


def silverman_bandwidth(points: RepresentativePoints, values,
                        factor: float = 1.0) -> float:
    r"""Sample-family bandwidth, for responses that are scattered rather than
    laid out on a lattice.

        h = 0.9 min(s, IQR/1.34) N_eff^{-1/5}

    with ``s`` and the interquartile range taken against the assigned
    probabilities, and ``N_eff = 1 / sum_q P_q^2``.

    **Which rule to use is decided by the response, not by the method.**
    :func:`response_spacing` scales with the median gap between adjacent
    responses, which is right when those responses inherit a lattice's
    regularity and fall like ``1/N``. Push a low-discrepancy set through a
    nonlinear map of several inputs and they do not: they land like a sample,
    whose spacing falls like ``N^{-1/2}`` while the optimal bandwidth falls
    like ``N^{-1/5}``, so following the local spacing undersmooths by a factor
    that grows with ``N``. :func:`spacing_uniformity` is the diagnostic --
    above :data:`UNIFORMITY_LIMIT` the spacing rules are the wrong family and
    this one is right.

    ``N_eff`` rather than ``N`` because the points carry unequal weights, and
    Kish's effective sample size is exactly the statement that a set with a
    few dominant weights behaves like a smaller one. That is the use it is
    valid for -- describing the weights -- as opposed to judging an
    importance-sampling estimate, which it cannot do.

    ``factor`` scales the result, for the sensitivity study a bandwidth choice
    should always carry.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.shape[0] != points.n_points:
        raise ValueError(f"{v.shape[0]} values against {points.n_points} "
                         f"representative points")
    w = points.probability
    mean = float(w @ v)
    std = float(np.sqrt(max(w @ (v - mean) ** 2, 0.0)))

    order = np.argsort(v)
    cdf = np.cumsum(w[order])
    quartiles = np.interp([0.25, 0.75], cdf, v[order])
    spread = float(quartiles[1] - quartiles[0]) / 1.34

    scale = min(std, spread) if spread > 0.0 else std
    if scale <= 0.0:
        raise ValueError(
            "the responses have no spread, so no bandwidth is defined; this "
            "is the instant where every realisation coincides")
    n_eff = 1.0 / float(np.sum(w ** 2))
    return factor * 0.9 * scale * n_eff ** (-0.2)


def exceedance_probability(points: RepresentativePoints, values,
                           thresholds, *, strict: bool = True) -> np.ndarray:
    r"""DPIM-H: ``Pr[Z > t]`` straight from the probability density integral
    equation, with no kernel and no bandwidth.

        Pr[G <= 0] = INT H(-G) p_Theta dtheta  ~  sum_q P_q H(-G(theta_q))

    so an event probability is a probability-weighted count of the
    representative points inside it. That is the whole method: DPIM-S smooths
    the Diracs of the discrete PDIE into a density, DPIM-H integrates them
    over a set instead, and integrating a Dirac needs no smoothing. Hence no
    bandwidth, no grid and no ``mass`` diagnostic -- and hence also none of
    the smoothing bias this module's docstring is largely about, which is why
    this is the right tool for a probability and :func:`solve_dpim` for a
    density.

    Parameters
    ----------
    values : ``(n_points,)`` the response at each representative point. May
        contain ``inf``, which is meaningful here rather than an error: a
        realisation that never leaves a regime has an infinite transition
        load.
    thresholds : scalar or ``(m,)``; the result takes their shape.
    strict : ``True`` gives ``Pr[Z > t]``, ``False`` gives ``Pr[Z >= t]``.

    **The strictness is not pedantry here.** For a continuous response a
    threshold landing exactly on a representative value is a coincidence, but
    the intended use has load factors as thresholds and *transition* loads as
    values, so a sweep that evaluates at a point's own transition load hits
    the tie deliberately. With the convention ``H(u) = 1`` for ``u >= 0``,
    a band's lower edge needs ``strict=False`` and an indicator exceedance
    needs ``strict=True``; swapping them moves a regime curve by one point's
    probability, which at ``N = 64`` is 1.6%.

    The error is the point set's quadrature error and nothing else, so it
    inherits whatever the set gives -- :func:`~scion.uq.density.gf_points`
    for a low-discrepancy one.
    """
    v = np.asarray(values, dtype=float).ravel()
    if v.shape[0] != points.n_points:
        raise ValueError(f"{v.shape[0]} values against {points.n_points} "
                         f"representative points")
    t = np.asarray(thresholds, dtype=float)
    inside = (v[None, :] > t.reshape(-1, 1) if strict
              else v[None, :] >= t.reshape(-1, 1))
    out = inside @ points.probability
    return out.reshape(t.shape) if t.ndim else out[0]


def band_probability(points: RepresentativePoints, lower, upper,
                     thresholds) -> np.ndarray:
    r"""``Pr[lower < t <= upper]``, for thresholds that are per-point.

    The shape every regime band takes -- ``Pr[t <= lambda_sd]`` minus
    ``Pr[t <= lambda_el]`` -- written once so the Heaviside convention cannot
    drift between bands. ``lower`` and ``upper`` are ``(n_points,)`` arrays of
    per-point transition loads with ``upper >= lower`` pointwise, and each
    band is closed on its upper side, which is what makes a stack of them sum
    to exactly one.
    """
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    if np.any(upper < lower):
        raise ValueError("a band needs upper >= lower at every point; the "
                         "transition loads are ordered lambda_el <= "
                         "lambda_sd <= lambda_rr")
    return (exceedance_probability(points, upper, thresholds, strict=False)
            - exceedance_probability(points, lower, thresholds, strict=False))


def solve_dpim(points: RepresentativePoints, x, t_grid=None, *,
               n_cells: int = 400, margin: float = 0.15,
               sigma: float | None = None,
               sigma_factor: float = SIGMA_FACTOR,
               mass_tol: float = 1e-3) -> DensityEvolution:
    """Evaluate the PDIE on a response grid, at every instant.

    The kernel here is unbounded, so a response living on a half line is
    read **low at its bound** -- exactly half the true density there in the
    narrow-bandwidth limit, since the kernel loses the half of its mass that
    falls outside the support. A response with a bound, or with a point mass
    at one, belongs in :func:`~scion.uq.density.mixed.mixed_density` instead.

    Parameters
    ----------
    points : the representative set; ``P_q`` enters as the kernel weight, so
        the exact assigned probabilities from :func:`~.points.lattice` are
        what Chen & Yang's Eq. (34) asks for -- they say so explicitly,
        improving on the nominal ``p(theta_q) V_q`` of the earlier work.
    x : responses, ``(n_points, n_times)``, or ``(n_points,)`` for a static
        system.
    t_grid : the output times. Optional, and unused by the calculation: the
        PDIE holds instant by instant, so this only labels the result.
    n_cells : cells across the response axis. **This is a rendering choice,
        not an accuracy one** -- the density is a continuous function of
        ``y`` and the grid merely samples it. Compare with PDEM, where the
        same parameter sets the smoothing.
    sigma : the kernel width. ``None`` applies ``sigma_factor`` times
        :func:`response_spacing`; pass ``paper_bandwidth(x)`` for Chen &
        Yang's own rule, or :func:`knn_bandwidth` for a per-point width,
        which may be given as an ``(n_points,)`` or ``(n_points, n_times)``
        array.
    mass_tol : how far ``int p dy`` may sit from one before
        :class:`~.gdee.ProbabilityNotConserved` is raised. Looser than PDEM's
        by default because here it is a quadrature error rather than a
        structural property -- a kernel narrower than the cell size, or one
        sitting at the edge of the grid, is what makes it fail.
    """
    x = np.asarray(x, dtype=float)
    static = x.ndim == 1
    if static:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"responses must be (n_points, n_times) or "
                         f"(n_points,); got shape {x.shape}")
    if x.shape[0] != points.n_points:
        raise ValueError(f"{x.shape[0]} response histories against "
                         f"{points.n_points} points")
    if not np.all(np.isfinite(x)):
        raise ValueError("the responses contain a non-finite value")

    if t_grid is None:
        t_grid = np.zeros(1) if static else np.arange(x.shape[1], dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if t_grid.size != x.shape[1]:
        raise ValueError(f"{t_grid.size} times against {x.shape[1]} columns "
                         f"of response")
    if n_cells < 4:
        raise ValueError("at least four cells are needed")
    if margin <= 0.0:
        raise ValueError("margin must be positive; the kernels must fit "
                         "inside the response axis")

    peak = float(np.max(np.abs(x)))
    if sigma is None:
        if not sigma_factor > 0.0:
            raise ValueError("sigma_factor must be positive")
        gap = response_spacing(x) if points.n_points > 1 else 0.0
        if gap > 0.0:
            sigma = sigma_factor * gap
        else:
            # one point, or every response identical: there is no spacing to
            # scale by, so fall back to the paper's rule
            sigma = paper_bandwidth(x) if peak > 0.0 else SIGMA_RATIO
        ratio = spacing_uniformity(x)
        if ratio > UNIFORMITY_LIMIT:
            warnings.warn(
                f"the representative responses are unevenly spread (largest "
                f"gap is {ratio:.0f}x the median), so the default bandwidth "
                f"-- {SIGMA_FACTOR} x the median gap -- will undersmooth the "
                f"sparse part badly. Measured on a Sobol-based point set at "
                f"a ratio of 33, the default gave an L1 error of 0.39 where "
                f"the best fixed sigma gave 0.0098. Set sigma explicitly, or "
                f"use a tensor `lattice` if the dimension allows one.",
                RuntimeWarning, stacklevel=2)
    sigma = np.asarray(sigma, dtype=float)
    per_point = sigma.ndim > 0
    if sigma.ndim == 0:
        sigma = np.full(x.shape, float(sigma))
    elif sigma.ndim == 1:
        sigma = np.repeat(sigma[:, None], x.shape[1], axis=1)
    if sigma.shape != x.shape:
        raise ValueError(f"sigma should be a scalar, {(x.shape[0],)} or "
                         f"{x.shape}; got {sigma.shape}")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must be positive and finite everywhere. The "
                         "Dirac limit is not computable -- that is what the "
                         "smoothing exists to avoid")

    lo, hi = float(x.min()), float(x.max())
    width = hi - lo
    scale = width if width > 0.0 else (peak if peak > 0.0 else 1.0)
    # the kernels must fit as well as the responses, or their tails are cut
    # off and the quadrature loses probability at the ends
    pad = margin * scale + 4.0 * float(sigma.max())
    lo, hi = lo - pad, hi + pad
    dx = (hi - lo) / n_cells
    centres = lo + (np.arange(n_cells) + 0.5) * dx

    # A per-point bandwidth can ask for kernels the grid cannot integrate:
    # on a dynamic problem with a deterministic initial condition the early
    # instants have barely separated responses, and the k-th neighbour
    # distance there is tiny -- measured, sigma spanning 9.7e-8 to 4.3e-3
    # across one run. Those kernels are floored at the cell scale, because a
    # feature narrower than a cell cannot be drawn on this grid whatever is
    # done, and the quadrature would otherwise be wrong rather than coarse.
    #
    # The floor is reported, not silent: flooring *most* kernels means the
    # grid is too coarse to exploit a per-point bandwidth at all, and the
    # result is then the fixed-bandwidth answer with extra steps -- measured,
    # a chi-square L1 of 0.0042 became 2.46 that way before this warned.
    # ...and only for a bandwidth that varies. A scalar sigma is the
    # caller's choice, so a kernel narrower than a cell is refused below
    # rather than silently widened -- quietly repairing it would move every
    # published comparison in this module.
    floored = int(np.sum(sigma < 1.5 * dx)) if per_point else 0
    if floored:
        sigma = np.maximum(sigma, 1.5 * dx)
        fraction = floored / sigma.size
        if fraction > 0.10:
            warnings.warn(
                f"{100 * fraction:.0f}% of the kernels were widened to the "
                f"cell size ({1.5 * dx:.2e}), so the grid is too coarse for "
                f"this bandwidth and the density is smoother than asked for. "
                f"Raise n_cells (now {n_cells}) until the fraction is small, "
                f"or use a fixed sigma.",
                RuntimeWarning, stacklevel=2)

    pdf = np.empty((t_grid.size, n_cells))
    for k in range(t_grid.size):
        # (n_cells, n_points): every kernel evaluated on every cell centre
        width = sigma[None, :, k]
        z = (centres[:, None] - x[None, :, k]) / width
        pdf[k] = ((np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * width))
                  @ points.probability)

    mass = pdf.sum(axis=1) * dx
    drift = float(np.abs(mass - 1.0).max())
    if drift > mass_tol:
        worst = int(np.argmax(np.abs(mass - 1.0)))
        raise ProbabilityNotConserved(
            f"the density integrates to {mass[worst]!r} at t = "
            f"{t_grid[worst]!r} ({drift:.3e} off one). With a Gaussian kernel "
            f"this is quadrature, not transport. sigma runs "
            f"{float(sigma.min()):.3e} to {float(sigma.max()):.3e} against a "
            f"cell size of {dx:.3e} (ratios {float(sigma.min()) / dx:.2f} to "
            f"{float(sigma.max()) / dx:.2f}; about 1.5 is where midpoint "
            f"integration of a Gaussian becomes exact). The widest kernel "
            f"may instead be reaching the ends of the axis: raise margin "
            f"(now {margin})")
    return DensityEvolution(x=centres, t=t_grid, pdf=pdf, mass=mass, dx=dx)
