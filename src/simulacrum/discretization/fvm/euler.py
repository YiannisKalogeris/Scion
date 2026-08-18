r"""
The 1-D Euler equations by finite volume.

    d/dt [rho, rho u, E] + d/dx [rho u, rho u^2 + p, u (E + p)] = 0

closed by the ideal-gas law ``p = (gamma - 1)(E - rho u^2 / 2)``. Three
conservation laws -- mass, momentum, energy -- and the reason to solve them with
a finite volume method rather than anything else is that the solutions contain
shocks, and a shock is where conservation stops being a nicety.

Why conservation is not optional here
-------------------------------------
Across a shock the solution is discontinuous, so the differential form of the
equations does not hold there at all; only the integral form does, and the
jump conditions that fix the shock speed come from it. A scheme that discretises
the differential form -- any finite difference scheme, however accurate --
converges to *a* weak solution, but not necessarily the right one: the
Lax-Wendroff theorem guarantees the correct shock speed only for a scheme in
conservation form. A non-conservative scheme will happily produce a
beautifully sharp shock in the wrong place, and refine to a sharper shock still
in the wrong place.

That is the whole argument for this module. It is also why every test here
checks the shock *position* against the exact solution and not merely the
smoothness of the profile.

Structure
---------
:class:`EulerEquations` is the algebra: conserved from primitive and back,
the flux, the sound speed. It knows nothing about grids.

:class:`EulerSolver1D` is the scheme: MUSCL reconstruction, a Riemann solver
from :mod:`~.riemann` at each face, SSP Runge-Kutta in time.

Reconstruct primitive, not conserved
------------------------------------
The slopes are limited on ``(rho, u, p)`` and converted to conserved variables
afterwards. Limiting the conserved variables instead is defensible and common,
but momentum and total energy both mix velocity into density, so a limiter that
clips them independently can produce a face state with negative pressure in a
strong rarefaction -- from two perfectly valid neighbours. Reconstructing
primitives and enforcing positivity on ``rho`` and ``p`` directly makes that
failure impossible to express.

Even so, MUSCL on a system is not TVD, and the scalar theory does not extend:
total variation is not defined for a vector. Limiting each characteristic field
would come close; limiting each *component*, as here and nearly everywhere, is
only an approximation to that. Measured on Sod at 200 cells, every limiter holds
density and pressure to their initial bounds exactly and every one overshoots
the exact peak velocity by about 0.3% -- first order included, so it is the
system, not the reconstruction. Expect small overshoots and do not expect a
theorem to forbid them.

Without a limiter it is not a question of overshoot. On Sod the unlimited scheme
breaks the density and pressure bounds by 3 to 5% and the velocity by 50%; on
the stronger Lax tube it goes unstable altogether, and
positivity has to be forced 4211 times to keep the run finite. That is why
:attr:`EulerResult.repairs` is reported rather than swallowed -- a solver that
silently floors its own state produces a plausible picture out of a diverged
one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulacrum.discretization.fvm.limiters import get_limiter
from simulacrum.discretization.fvm.riemann import (
    exact_riemann,
    get_riemann_solver,
)

__all__ = [
    "EulerEquations",
    "EulerSolver1D",
    "EulerResult",
    "ShockTube",
    "SOD",
    "SHOCK_TUBES",
    "get_shock_tube",
]

_EPS = 1.0e-12
_FLOOR = 1.0e-12


class EulerEquations:
    """Ideal-gas algebra for the 1-D Euler system. No grid, no scheme.

    State arrays are ``(n, 3)`` -- cells along the first axis, the conserved
    components ``(rho, rho u, E)`` along the last.
    """

    def __init__(self, gamma: float = 1.4):
        if gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        self.gamma = float(gamma)

    # -- conversions ----------------------------------------------------
    def conserved(self, rho, u, p) -> np.ndarray:
        rho, u, p = (np.atleast_1d(np.asarray(v, dtype=float))
                     for v in (rho, u, p))
        rho, u, p = np.broadcast_arrays(rho, u, p)
        energy = p / (self.gamma - 1.0) + 0.5 * rho * u * u
        return np.stack([rho, rho * u, energy], axis=-1)

    def primitive(self, U) -> np.ndarray:
        """``(rho, u, p)`` from the conserved state, same shape."""
        U = np.asarray(U, dtype=float)
        rho = U[..., 0]
        u = U[..., 1] / rho
        p = (self.gamma - 1.0) * (U[..., 2] - 0.5 * rho * u * u)
        return np.stack([rho, u, p], axis=-1)

    # -- physics --------------------------------------------------------
    def flux(self, U) -> np.ndarray:
        U = np.asarray(U, dtype=float)
        W = self.primitive(U)
        rho, u, p = W[..., 0], W[..., 1], W[..., 2]
        return np.stack([rho * u, rho * u * u + p, u * (U[..., 2] + p)],
                        axis=-1)

    def sound_speed(self, U) -> np.ndarray:
        W = self.primitive(U)
        return np.sqrt(self.gamma * np.maximum(W[..., 2], 0.0) / W[..., 0])

    def max_speed(self, U) -> float:
        """``max(|u| + a)``: what the CFL condition is measured against."""
        W = self.primitive(U)
        return float(np.max(np.abs(W[..., 1]) + self.sound_speed(U)))

    def pressure_positive(self, U) -> bool:
        W = self.primitive(U)
        return bool(np.all(W[..., 0] > 0.0) and np.all(W[..., 2] > 0.0))

    def enforce_positivity(self, W: np.ndarray) -> tuple[np.ndarray, int]:
        """Floor density and pressure in a *primitive* array.

        Returns the repaired array and how many entries were touched, because a
        solver that silently repairs its own state is a solver whose failures
        never surface.
        """
        W = np.array(W, dtype=float, copy=True)
        bad = int(np.count_nonzero(W[..., 0] < _FLOOR)
                  + np.count_nonzero(W[..., 2] < _FLOOR))
        np.maximum(W[..., 0], _FLOOR, out=W[..., 0])
        np.maximum(W[..., 2], _FLOOR, out=W[..., 2])
        return W, bad

    def __repr__(self) -> str:
        return f"EulerEquations(gamma={self.gamma})"


# ---------------------------------------------------------------------------

@dataclass
class ShockTube:
    """A named Riemann problem: two constant primitive states and a diaphragm.

    ``left`` and ``right`` are ``(rho, u, p)``.
    """

    name: str
    left: tuple[float, float, float]
    right: tuple[float, float, float]
    t_end: float
    x0: float = 0.5
    length: float = 1.0
    gamma: float = 1.4
    note: str = ""

    def initial_state(self, x: np.ndarray, eos: EulerEquations) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        left = np.array(self.left)
        right = np.array(self.right)
        W = np.where((x < self.x0)[:, None], left, right)
        return eos.conserved(W[:, 0], W[:, 1], W[:, 2])

    def exact(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """Exact primitive solution at time ``t``, shape ``(len(x), 3)``."""
        t = self.t_end if t is None else t
        x = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.where((x < self.x0)[:, None], np.array(self.left),
                            np.array(self.right))
        return exact_riemann(self.left, self.right, (x - self.x0) / t,
                             self.gamma)


#: Sod's problem. The standard first test for a compressible solver, and it is
#: standard because its solution contains one of each wave -- a left
#: rarefaction, a contact, and a right shock -- so a scheme cannot get a
#: passable answer by being good at only one of them.
SOD = ShockTube(
    name="sod",
    left=(1.0, 0.0, 1.0),
    right=(0.125, 0.0, 0.1),
    t_end=0.2,
    note="Sod (1978): left rarefaction, contact, right shock",
)

SHOCK_TUBES: dict[str, ShockTube] = {
    "sod": SOD,
    "lax": ShockTube(
        name="lax",
        left=(0.445, 0.698, 3.528),
        right=(0.5, 0.0, 0.571),
        t_end=0.14,
        note="Lax: a stronger shock than Sod, with a moving left state",
    ),
    "123": ShockTube(
        name="123",
        left=(1.0, -2.0, 0.4),
        right=(1.0, 2.0, 0.4),
        t_end=0.15,
        note="Toro's 123 problem: two rarefactions, near-vacuum in the middle. "
             "Pressure there is small and positive, so schemes that are not "
             "positivity preserving fail outright",
    ),
    "blast": ShockTube(
        name="blast",
        left=(1.0, 0.0, 1000.0),
        right=(1.0, 0.0, 0.01),
        t_end=0.012,
        note="Toro's left half-blast: pressure ratio 1e5, a severe test of the "
             "Riemann solver rather than of the reconstruction",
    ),
}


def get_shock_tube(name: str) -> ShockTube:
    key = name.lower()
    if key not in SHOCK_TUBES:
        raise KeyError(f"unknown shock tube '{name}'; available: "
                       f"{sorted(SHOCK_TUBES)}")
    return SHOCK_TUBES[key]


# ---------------------------------------------------------------------------

@dataclass
class EulerResult:
    """What a run produced, and what it had to do to get there."""

    U: np.ndarray                     #: conserved state, ``(n, 3)``
    x: np.ndarray                     #: cell centres
    t: float                          #: time actually reached
    steps: int
    mass: float                       #: final integral of rho
    mass_drift: float                 #: |final - initial|, should be round-off
    energy_drift: float
    repairs: int = 0                  #: how often positivity had to be forced

    @property
    def conservative(self) -> bool:
        """Whether the drifts are at round-off.

        Only meaningful when nothing crossed the boundary: periodic walls,
        reflective walls, or a transmissive run stopped before any wave reached
        the ends. On Sod at ``t = 0.2`` the drift is exactly zero; on Toro's
        123 problem it is 0.6, because the two rarefactions carry mass out
        through the transmissive boundaries and that is the correct answer, not
        a leak.
        """
        return max(self.mass_drift, self.energy_drift) < 1e-10 * max(
            1.0, abs(self.mass))

    def primitive(self, eos: EulerEquations) -> np.ndarray:
        return eos.primitive(self.U)


class EulerSolver1D:
    """MUSCL finite volume for the 1-D Euler equations on a uniform grid.

    Parameters
    ----------
    n_cells, length, origin : the grid. Cells tile the domain and centres sit
        at ``(i + 1/2) h``, as everywhere else in this package.
    gamma : ratio of specific heats.
    riemann : ``"hllc"`` (default), ``"hll"`` or ``"rusanov"``.
    limiter : any name from :mod:`~.limiters`; ``"first-order"`` disables
        reconstruction entirely and is the reference every other setting should
        beat.
    boundary : ``"transmissive"``, ``"reflective"`` or ``"periodic"``.
    """

    def __init__(self, n_cells: int = 200, length: float = 1.0,
                 origin: float = 0.0, gamma: float = 1.4,
                 riemann: str = "hllc", limiter: str = "minmod",
                 boundary: str = "transmissive"):
        if n_cells < 4:
            raise ValueError("MUSCL needs two ghost layers, so at least four "
                             "cells")
        self.n = int(n_cells)
        self.length = float(length)
        self.origin = float(origin)
        self.h = self.length / self.n

        self.eos = EulerEquations(gamma)
        self.riemann = (get_riemann_solver(riemann, gamma)
                        if isinstance(riemann, str) else riemann)
        self.limiter = (get_limiter(limiter) if isinstance(limiter, str)
                        else limiter)
        if boundary not in ("transmissive", "reflective", "periodic"):
            raise ValueError(f"unknown boundary {boundary!r}")
        self.boundary = boundary
        self._repairs = 0

    # ------------------------------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return self.origin + self.h * (np.arange(self.n) + 0.5)

    @property
    def faces(self) -> np.ndarray:
        return self.origin + self.h * np.arange(self.n + 1)

    # ------------------------------------------------------------------
    def _pad(self, W: np.ndarray) -> np.ndarray:
        """Two ghost cells per side, on *primitive* variables.

        ``reflective`` flips the sign of the velocity and mirrors the rest,
        which is what makes a wall a wall: zero mass flux and a pressure force.
        Copying the interior instead (a transmissive wall) lets the shock walk
        straight out of the domain.
        """
        if self.boundary == "periodic":
            return np.concatenate([W[-2:], W, W[:2]], axis=0)

        if self.boundary == "reflective":
            low, high = W[[1, 0]].copy(), W[[-1, -2]].copy()
            low[:, 1] *= -1.0
            high[:, 1] *= -1.0
        else:                                   # transmissive: zero gradient
            low, high = np.stack([W[0], W[0]]), np.stack([W[-1], W[-1]])
        return np.concatenate([low, W, high], axis=0)

    def _reconstruct(self, Wp: np.ndarray):
        """Limited primitive face states from a padded array, componentwise."""
        diff = Wp[1:] - Wp[:-1]                       # (n+3, 3)
        forward = diff[1:]                            # (n+2, 3)
        safe = np.where(np.abs(forward) < _EPS,
                        np.where(forward < 0.0, -_EPS, _EPS), forward)
        slope = 0.5 * self.limiter(diff[:-1] / safe) * forward

        n = self.n
        left = Wp[1:n + 2] + slope[:n + 1]
        right = Wp[2:n + 3] - slope[1:n + 2]
        return left, right

    def residual(self, U: np.ndarray) -> np.ndarray:
        """``-dF/dx`` at every cell: the right-hand side of ``dU/dt``."""
        W, bad = self.eos.enforce_positivity(self.eos.primitive(U))
        self._repairs += bad
        Wp = self._pad(W)

        WL, WR = self._reconstruct(Wp)
        WL, nl = self.eos.enforce_positivity(WL)
        WR, nr = self.eos.enforce_positivity(WR)
        self._repairs += nl + nr

        UL = self.eos.conserved(WL[:, 0], WL[:, 1], WL[:, 2])
        UR = self.eos.conserved(WR[:, 0], WR[:, 1], WR[:, 2])
        F = self.riemann(UL, UR)                       # (n+1, 3)
        return -(F[1:] - F[:-1]) / self.h

    def stable_dt(self, U: np.ndarray, cfl: float = 0.4) -> float:
        speed = self.eos.max_speed(U)
        return float(cfl * self.h / speed) if speed > 0.0 else float("inf")

    def advance(self, U: np.ndarray, dt: float, stages: int = 2) -> np.ndarray:
        if stages == 1:
            return U + dt * self.residual(U)
        if stages == 2:
            U1 = U + dt * self.residual(U)
            return 0.5 * U + 0.5 * (U1 + dt * self.residual(U1))
        if stages == 3:
            U1 = U + dt * self.residual(U)
            U2 = 0.75 * U + 0.25 * (U1 + dt * self.residual(U1))
            return (U + 2.0 * (U2 + dt * self.residual(U2))) / 3.0
        raise ValueError("SSP Runge-Kutta is implemented for 1, 2 or 3 stages")

    # ------------------------------------------------------------------
    def solve(self, U0: np.ndarray, t_end: float, cfl: float = 0.4,
              stages: int = 2, max_steps: int = 100000) -> EulerResult:
        """Integrate to ``t_end`` under a CFL-limited step.

        The step is recomputed every iteration from the current maximum wave
        speed. Fixing it in advance from the initial data is a common shortcut
        and fails on any problem where a shock accelerates -- the blast tube in
        :data:`SHOCK_TUBES` does exactly that.
        """
        U = np.array(U0, dtype=float, copy=True)
        self._repairs = 0
        mass0 = float(np.sum(U[:, 0]) * self.h)
        energy0 = float(np.sum(U[:, 2]) * self.h)

        t, steps = 0.0, 0
        while t < t_end - 1e-14:
            if steps >= max_steps:
                raise RuntimeError(
                    f"stopped at t = {t:g} of {t_end:g} after {steps} steps; "
                    f"the CFL step has collapsed, which usually means the "
                    f"state has gone unphysical")
            dt = min(self.stable_dt(U, cfl), t_end - t)
            U = self.advance(U, dt, stages)
            t += dt
            steps += 1

        mass = float(np.sum(U[:, 0]) * self.h)
        energy = float(np.sum(U[:, 2]) * self.h)
        return EulerResult(
            U=U, x=self.x, t=t, steps=steps, mass=mass,
            mass_drift=abs(mass - mass0), energy_drift=abs(energy - energy0),
            repairs=self._repairs)

    def run_tube(self, tube: ShockTube, cfl: float = 0.4,
                 stages: int = 2) -> EulerResult:
        """Set the tube's initial data on this grid and integrate to its
        ``t_end``. The grid must already span the tube's length."""
        if not np.isclose(self.length, tube.length) or \
                not np.isclose(self.eos.gamma, tube.gamma):
            raise ValueError(
                f"the solver spans {self.length:g} with gamma "
                f"{self.eos.gamma:g}; tube '{tube.name}' needs "
                f"{tube.length:g} and {tube.gamma:g}")
        U0 = tube.initial_state(self.x, self.eos)
        return self.solve(U0, tube.t_end, cfl=cfl, stages=stages)

    def __repr__(self) -> str:
        return (f"EulerSolver1D({self.n} cells, riemann="
                f"'{self.riemann.name}', limiter='{self.limiter.name}', "
                f"{self.boundary})")
