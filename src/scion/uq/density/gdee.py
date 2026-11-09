"""The probability density evolution method: solve the GDEE per representative
point and sum.

Li & Chen reduce the high-dimensional density evolution equation to a *scalar*
advection equation, their Eq. (23), one per representative point ``z_q``:

    dp/dt + Xdot_l(z_q, t) dp/dx_l = 0,    p(x_l, 0) = P_q delta(x_l - X_{0,l})

and the response PDF is the sum over ``q``. Everything structural about the
method lives in that velocity: it is the *deterministic* response velocity of
the l-th state component, obtained from one ordinary time integration at
``Z = z_q``. The PDE knows nothing about the structure.

**The equation is indifferent to how the velocity was obtained**, which is the
property worth building on. Precise time integration is what the paper uses;
Newmark, a nonlinear Newton solve or a full finite element run are equally
admissible suppliers. So nonlinearity is a question about the *model*, not
about this module, and nothing here needs to change to meet one.

**The precondition, which the paper states and which is easy to miss.**
Eq. (23) is Eq. (20) with ``A_ll = 0``. That holds for a *displacement*
response (the state matrix's top-left block is zero) and for any response of
an *undamped* system. When ``A_ll != 0`` -- a velocity response of a damped
system -- the velocity acquires a term in ``x_l`` and the equation acquires a
source ``A_ll p``, so what is solved here is the wrong equation. The honest
reading is that this module transports a density along a trajectory supplied
to it, which is exactly the later GDEE's trajectory form; a caller handing it
velocity histories outside that scope gets no warning, because the matrices
are never passed in.

Two consequences of the velocity being uniform in space:

- the conservative and non-conservative forms coincide, so the finite volume
  path's *structural* conservation is conservation of probability -- the
  reported ``mass`` holds at round-off rather than to the order of the scheme;
- the exact solution for each point is pure translation of a Dirac, so every
  numerical ``p_q`` is a smeared delta and the sum is smoother than the truth.
  Dissipation is therefore the error that matters, which is why the limiter
  matters and why DPIM exists.

**The grid's dissipation is an implicit kernel bandwidth, so a finer grid is
not a better one.** The computed density is a sum of ``N`` smeared Diracs, one
per representative point, and what smooths them into a density is the scheme's
own numerical dissipation. Refining the response axis narrows every spike, so
past a point the individual trajectories become resolved and the answer gets
*worse* -- not asymptotically, but immediately and badly. Example 1 at
``t = 1 s``, ``L1`` error of the density over ``|x| < 0.95 x0`` against the
closed form:

    points / cells      50      100      200      400      800
        20           0.1621   0.1013   0.1685   0.3075   0.9620
        40           0.1628   0.0877   0.0645   0.1457   0.2100
        80           0.1626   0.0872   0.0518   0.0361   0.1117
       160           0.1629   0.0873   0.0519   0.0293   0.0225
       320           0.1629   0.0873   0.0519   0.0288   0.0161

Read along a row: 20 points at 800 cells is **ten times worse** than the same
20 points at 100. Read down a column: at 50 cells the point count does not
matter at all, because dissipation has erased it. The optimum tracks
``n_cells ~ 5 N`` on this problem, and along that diagonal the error falls
about 1.8x per doubling of both -- first order, which is what a TVD scheme
gives on a density that is discontinuous wherever a branch enters the
frequency band and has integrable poles at ``+-x0``.

The factor of five is not universal -- it depends on how fast the response
spreads the points, ``dX/dz`` -- so it is documented rather than enforced. But
the shape of the table is the method's, not this implementation's, and it is
the single most useful thing to know before choosing a resolution: **refine
the point set and the response grid together, or neither.**

This is also the cleanest statement of what DPIM changes. Chen & Yang replace
the implicit bandwidth with an explicit Gaussian smoothing parameter, which
turns the accidental coupling above into a parameter that can be chosen -- and
removes the CFL condition and the TVD scheme with it.

**The response axis must contain the density with room to spare, and this is
enforced rather than advised.** Eq. (17) puts ``p -> 0`` at ``x -> +-inf``,
which no finite grid has. Of the conditions the finite volume module offers,
a zero-gradient end is a clean non-reflecting outflow *while the density
there is negligible* and a mass source as soon as it is not: with the grid cut
exactly to the response range, the oscillator's turning points sit on the
boundary, the velocity points inward there, and the ghost cell feeds the
domain -- measured, the total probability reaches **88** instead of 1. A
Dirichlet end is no better, injecting negative density by reflection. So the
margin is not a tuning parameter, and ``solve_gdee`` checks the total
probability afterwards and raises :class:`ProbabilityNotConserved` rather than
returning a density that integrates to something else.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scion.discretization.fvm import FVMAssembler
from scion.uq.density.points import RepresentativePoints


class ProbabilityNotConserved(RuntimeError):
    """The computed density does not integrate to one.

    Raised rather than warned because the failure is silent otherwise: a
    density off by a few percent still has the right shape, the right peak
    location and plausible moments. The cause is almost always the response
    axis -- either the density reached a boundary, or the scheme is diffusive
    enough at this resolution to have spread it there.
    """


@dataclass(frozen=True)
class DensityEvolution:
    """The PDF of one response component and its evolution in time."""

    x: np.ndarray                 # (n_cells,) cell centres
    t: np.ndarray                 # (n_times,)
    pdf: np.ndarray               # (n_times, n_cells)
    mass: np.ndarray              # (n_times,) integral of pdf: the diagnostic
    dx: float

    @property
    def mean(self) -> np.ndarray:
        """``E[X_l(t)]`` from the computed density, ``(n_times,)``."""
        return (self.pdf * self.x) .sum(axis=1) * self.dx

    @property
    def std(self) -> np.ndarray:
        second = (self.pdf * self.x ** 2).sum(axis=1) * self.dx
        return np.sqrt(np.maximum(second - self.mean ** 2, 0.0))

    def at(self, time: float) -> np.ndarray:
        """The density at the stored time nearest ``time``."""
        return self.pdf[int(np.argmin(np.abs(self.t - float(time))))]

    def __repr__(self) -> str:
        return (f"DensityEvolution(n_times={self.t.size}, "
                f"n_cells={self.x.size}, "
                f"mass in [{self.mass.min():.12f}, {self.mass.max():.12f}])")


def response_histories(points: RepresentativePoints, model, t_grid):
    """Run the deterministic analysis at every representative point.

    ``model(z, t_grid) -> (x_l, x_l_dot)``, each ``(n_times,)``. One plain
    callable, as everywhere else in ``uq`` -- which is what lets a finite
    element solve enter without this module knowing any physics.

    This is the expensive half of PDEM and it is embarrassingly parallel; the
    loop is left serial because a process pool would have to pickle the
    model, which is the caller's decision rather than ours.
    """
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    x = np.empty((points.n_points, t_grid.size))
    x_dot = np.empty_like(x)
    for q in range(points.n_points):
        xq, vq = model(points.z[q], t_grid)
        x[q] = np.asarray(xq, dtype=float).ravel()
        x_dot[q] = np.asarray(vq, dtype=float).ravel()
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(x_dot))):
        raise ValueError("the model returned a non-finite response")
    return x, x_dot


def solve_gdee(points: RepresentativePoints, t_grid, x, x_dot, *,
               n_cells: int = 400, margin: float = 0.15, cfl: float = 0.4,
               limiter: str = "superbee", flux: str = "upwind",
               stages: int = 2,
               mass_tol: float = 1e-6) -> DensityEvolution:
    """Solve the GDEE for every point and sum into the response PDF.

    Parameters
    ----------
    points : the representative set; its assigned probabilities enter through
        the initial condition, so the sum over points needs no further weight.
    t_grid : output times, ``(n_times,)``, increasing and starting at ``t=0``.
    x, x_dot : response and response velocity, ``(n_points, n_times)``.
    n_cells : cells across the response axis.
    margin : padding of the response range, as a fraction of its width. It
        exists because conservation is only conservation *inside* the grid: a
        density whose support reaches a boundary loses probability through the
        outflow condition, silently and in one direction.
    cfl : safety factor on the advective step. The paper writes the condition
        as ``r_L |g| <= 1`` and fixes the lattice ratio once from
        ``|g|_max``; here each sub-step takes its own stable step from the
        local velocity, which is the same condition applied less
        conservatively.
    mass_tol : the density must integrate to one within this at every output
        time, or :class:`ProbabilityNotConserved` is raised. Conservation is
        structural in the finite volume path, so a drift means probability
        left the grid; the tolerance is therefore about the *boundary*, not
        about the scheme's accuracy.

    The sub-stepping is the one place this departs from the paper's
    discretisation deliberately: the output times come from the deterministic
    analysis and have no reason to satisfy the CFL condition, so each output
    interval is covered by as many stable sub-steps as it needs, with the
    velocity interpolated linearly to each sub-step midpoint. At one sub-step
    per interval that reduces to the paper's ``g_m = (g~_m + g~_{m-1})/2``.
    """
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    x = np.atleast_2d(np.asarray(x, dtype=float))
    x_dot = np.atleast_2d(np.asarray(x_dot, dtype=float))

    if t_grid.size < 2:
        raise ValueError("at least two output times are needed")
    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("output times must be strictly increasing")
    if x.shape != (points.n_points, t_grid.size):
        raise ValueError(f"x should be {(points.n_points, t_grid.size)}; "
                         f"got {x.shape}")
    if x_dot.shape != x.shape:
        raise ValueError("x and x_dot must have the same shape")
    if n_cells < 4:
        raise ValueError("at least four cells are needed")
    if margin <= 0.0:
        # zero is not a conservative choice but a broken one: the response
        # extremes land exactly on the boundary, where the ghost cells feed
        # the domain. See the module docstring.
        raise ValueError("margin must be positive; the density may not reach "
                         "the ends of the response axis")

    lo, hi = float(x.min()), float(x.max())
    # A response with no range at all is not contrived -- a static problem, a
    # system at rest, or simply a component that does not move -- and padding
    # by a fraction of a zero width collapses the grid. Fall back to the
    # response's own magnitude for the length scale, and to one only when
    # there is no scale in the problem whatsoever.
    width = hi - lo
    if width > 0.0:
        scale = width
    elif np.max(np.abs(x)) > 0.0:
        scale = float(np.max(np.abs(x)))
    else:
        scale = 1.0
    pad = margin * scale
    lo, hi = lo - pad, hi + pad
    dx = (hi - lo) / n_cells
    if not dx > 0.0:
        raise ValueError(f"the response axis came out empty: [{lo}, {hi}]")
    assembler = FVMAssembler(grid_shape=(n_cells,), spacing=dx, velocity=0.0,
                             flux=flux, limiter=limiter, origin=lo)
    centres = assembler.centres(0)
    conditions = assembler.default_conditions()

    pdf = np.zeros((t_grid.size, n_cells))

    # one point at a time. The grid is shared and the equation is linear, so
    # this could be one batched solve -- the assembler's operators already
    # pass a leading batch axis through untouched -- but the velocity and
    # hence the sub-step count differ per point, so the batched form would
    # have to step every point at the slowest one's rate.
    for q in range(points.n_points):
        # Eq. (24): the whole assigned probability in the single cell holding
        # this point's initial response, so the cell integral is exactly P_q
        cell = int(np.clip((x[q, 0] - lo) / dx, 0, n_cells - 1))
        p = np.zeros(n_cells)
        p[cell] = points.probability[q] / dx
        pdf[0] += p

        for m in range(t_grid.size - 1):
            span = t_grid[m + 1] - t_grid[m]
            peak = max(abs(x_dot[q, m]), abs(x_dot[q, m + 1]))
            stable = assembler.stable_dt(cfl, peak)
            n_sub = max(1, int(np.ceil(span / stable))) if np.isfinite(stable) \
                else 1
            h = span / n_sub
            for s in range(n_sub):
                # linear interpolation of the velocity to the sub-step midpoint
                frac = (s + 0.5) / n_sub
                speed = (1.0 - frac) * x_dot[q, m] + frac * x_dot[q, m + 1]
                p = assembler.advance(p, h, conditions, stages, speed)
            pdf[m + 1] += p

    mass = pdf.sum(axis=1) * dx
    drift = float(np.abs(mass - 1.0).max())
    if drift > mass_tol:
        worst = int(np.argmax(np.abs(mass - 1.0)))
        raise ProbabilityNotConserved(
            f"the density integrates to {mass[worst]!r} at t = "
            f"{t_grid[worst]!r} ({drift:.3e} off one). Probability has left "
            f"the response axis: raise margin (now {margin}) or n_cells (now "
            f"{n_cells}), or use a less diffusive limiter (now {limiter!r})")
    return DensityEvolution(x=centres, t=t_grid, pdf=pdf, mass=mass, dx=dx)
