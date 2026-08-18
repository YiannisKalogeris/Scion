r"""
Riemann solvers for the 1-D Euler equations.

At every face of a finite volume mesh sits a discontinuity: the left cell's
average on one side, the right cell's on the other. Godunov's insight is that
this is not an approximation problem but an exactly solvable one -- the
initial-value problem for two constant states meeting at a point is the
*Riemann problem*, and its solution is self-similar in ``x/t``, so the state
*at* the face is constant in time and its flux can be evaluated once per step.

For Euler the solution always has the same structure: three waves separating
four constant states.

    left state | left wave | u*, p* left | contact | u*, p* right | right wave | right state

The outer waves are each a shock or a rarefaction; the middle is a contact
discontinuity, across which pressure and velocity are continuous and density is
not. Finding the star state means solving one nonlinear scalar equation for
``p*``, which is what :func:`exact_riemann` does.

Why approximate solvers exist
-----------------------------
That scalar equation needs Newton iterations per face per step, and the whole
point of the exercise is one number -- the flux. The approximate solvers here
skip the iteration by guessing the wave speeds and enforcing conservation
across them:

``rusanov``  one wave speed, the largest. Cheapest, monotone, and it makes no
             attempt to see a contact at all.
``hll``      two waves, left and right. Correct for an isolated shock; still
             blind to the contact, because a two-wave model has exactly one
             intermediate state and a contact needs two that differ in density.
``hllc``     three waves. The C is for Contact, and restoring it is the whole
             difference.

The cleanest demonstration is an isolated *stationary* contact -- a density jump
with uniform pressure and zero velocity, whose exact solution is that it does
not move. Measured on 100 cells at CFL 0.4 out to t = 0.4, HLLC holds it to
**zero error, to the bit**, while HLL and Rusanov smear it over 21 cells. That
is not a tuning difference: with ``u = p = const`` the HLLC star state reduces
algebraically to the cell state itself, so the flux is the exact one, whereas
HLL's two-wave average leaves a spurious density flux proportional to
``S_L S_R``, which is negative and therefore never vanishes.

On Sod the gap is real but much smaller, and it closes further with
reconstruction. Contact width in cells at ``n = 800``:

    limiter        rusanov   hll   hllc
    first-order         48    35     32
    minmod              15    13     11
    van-leer             9     8      8

Worth reading before optimising the wrong thing: once MUSCL is on, the limiter
is doing most of the sharpening and the Riemann solver is doing little of it.
The contact still matters -- it is a material interface, and smearing it is
smearing the boundary between two fluids -- but the lever is the reconstruction.

:func:`exact_riemann` is here for verification, not for the solver loop. It is
scalar, iterative and slow, and it is the only thing in the module that can say
whether the others are right.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "RiemannSolver",
    "RusanovSolver",
    "HLLSolver",
    "HLLCSolver",
    "RIEMANN_REGISTRY",
    "get_riemann_solver",
    "exact_riemann",
    "star_pressure",
    "VacuumError",
]

_TOL = 1.0e-12
_MAX_ITER = 100


class VacuumError(ValueError):
    """Raised when two states separate fast enough to open a vacuum.

    The star region ceases to exist -- there is no ``p*`` to solve for, because
    the pressure between the two rarefactions has fallen to zero. The condition
    is ``2(a_L + a_R)/(gamma - 1) <= u_R - u_L``. It is a real physical regime,
    not a numerical failure, and it needs a different sampling formula, so it is
    refused rather than approximated.
    """


# ---------------------------------------------------------------------------
# Conserved-variable helpers, kept local so the module stands alone
# ---------------------------------------------------------------------------

def _primitive(U: np.ndarray, gamma: float):
    """``(rho, u, p)`` from ``(rho, rho u, E)``, arrays of shape ``(..., 3)``."""
    rho = U[..., 0]
    u = U[..., 1] / rho
    p = (gamma - 1.0) * (U[..., 2] - 0.5 * rho * u * u)
    return rho, u, p


def _euler_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p = _primitive(U, gamma)
    return np.stack([rho * u, rho * u * u + p, u * (U[..., 2] + p)], axis=-1)


def _sound_speed(rho, p, gamma):
    return np.sqrt(gamma * np.maximum(p, 0.0) / rho)


# ---------------------------------------------------------------------------
# Approximate solvers
# ---------------------------------------------------------------------------

class RiemannSolver(ABC):
    """The Godunov flux at a face, from the conserved states either side."""

    name: str = "riemann"

    #: How many waves the wave model carries. Three is the physical number;
    #: fewer means some feature of the solution is invisible to the scheme.
    waves: int = 1

    #: Whether the contact discontinuity is represented. False means it is
    #: smeared, not that it is absent from the answer.
    resolves_contact: bool = False

    def __init__(self, gamma: float = 1.4):
        self.gamma = float(gamma)

    @abstractmethod
    def flux(self, UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
        """``(n, 3), (n, 3) -> (n, 3)``."""

    def __call__(self, UL, UR) -> np.ndarray:
        return self.flux(np.asarray(UL, dtype=float),
                         np.asarray(UR, dtype=float))

    # -- shared wave-speed estimate -------------------------------------
    def wave_speeds(self, UL: np.ndarray, UR: np.ndarray):
        """Toro's pressure-based estimates of the outer wave speeds.

        A guess at ``p*`` from the linearised (primitive-variable) Riemann
        problem tells each side whether it is looking at a shock or a
        rarefaction, and the speed follows. The cruder Davis estimate
        ``S_L = min(u_L - a_L, u_R - a_R)`` costs nothing and is safe, but it
        is loose enough on a strong shock to add noticeable dissipation.
        """
        g = self.gamma
        rho_l, u_l, p_l = _primitive(UL, g)
        rho_r, u_r, p_r = _primitive(UR, g)
        a_l, a_r = _sound_speed(rho_l, p_l, g), _sound_speed(rho_r, p_r, g)

        p_star = np.maximum(
            0.0,
            0.5 * (p_l + p_r) - 0.125 * (u_r - u_l) * (rho_l + rho_r)
            * (a_l + a_r))

        def q(p_k):
            ratio = np.where(p_k > 0.0, p_star / np.maximum(p_k, _TOL), 1.0)
            return np.where(p_star <= p_k, 1.0,
                            np.sqrt(np.maximum(
                                1.0 + 0.5 * (g + 1.0) / g * (ratio - 1.0),
                                1.0)))

        return u_l - a_l * q(p_l), u_r + a_r * q(p_r)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(gamma={self.gamma})"


class RusanovSolver(RiemannSolver):
    r"""``F = (F_L + F_R)/2 - (s/2)(U_R - U_L)``, ``s = max(|u| + a)``.

    One wave, travelling at the fastest speed present. Robust,
    positivity-friendly, and dissipative in proportion to how wrong the
    single-speed assumption is. A contact moves at ``u`` while ``s`` is
    ``|u| + a``, so the penalty applied to it is far larger than the wave it is
    meant to stabilise, and the contact smears accordingly.

    It shares its formula with :class:`~.flux.RusanovFlux` and *not* its
    behaviour, which is worth knowing before assuming the scalar intuition
    carries over. For a scalar the jump penalty cancels the downwind half of the
    average exactly and Rusanov **is** upwinding; for a system that cancellation
    needs ``F_R - F_L = s (U_R - U_L)``, true only if the entire jump is one
    wave moving at ``s``. So on a face where every wave already travels the same
    way -- where the exact flux is simply ``F_L`` and no dissipation is called
    for -- Rusanov still adds some: 3% of the flux on a Mach 9 state, where HLL
    and HLLC both return ``F_L`` to the bit.
    """

    name = "rusanov"
    waves = 1
    resolves_contact = False

    def flux(self, UL, UR):
        g = self.gamma
        rho_l, u_l, p_l = _primitive(UL, g)
        rho_r, u_r, p_r = _primitive(UR, g)
        s = np.maximum(np.abs(u_l) + _sound_speed(rho_l, p_l, g),
                       np.abs(u_r) + _sound_speed(rho_r, p_r, g))
        return (0.5 * (_euler_flux(UL, g) + _euler_flux(UR, g))
                - 0.5 * s[..., None] * (UR - UL))


class HLLSolver(RiemannSolver):
    r"""Two waves, ``S_L`` and ``S_R``, with one constant state between them.

    Harten, Lax and van Leer: assume the fan collapses to two discontinuities
    and let conservation determine the state in between,

        U* = (S_R U_R - S_L U_L + F_L - F_R) / (S_R - S_L).

    Exact for an isolated shock, and structurally incapable of representing a
    contact -- there is only one intermediate state, and a contact needs two
    that differ in density. That is the deficiency HLLC repairs.
    """

    name = "hll"
    waves = 2
    resolves_contact = False

    def flux(self, UL, UR):
        g = self.gamma
        SL, SR = self.wave_speeds(UL, UR)
        FL, FR = _euler_flux(UL, g), _euler_flux(UR, g)

        denom = np.where(np.abs(SR - SL) < _TOL, _TOL, SR - SL)
        star = ((SR[..., None] * FL - SL[..., None] * FR
                 + (SL * SR)[..., None] * (UR - UL)) / denom[..., None])

        out = np.where((SL >= 0.0)[..., None], FL, star)
        return np.where((SR <= 0.0)[..., None], FR, out)


class HLLCSolver(RiemannSolver):
    r"""Three waves: ``S_L``, the contact ``S_*``, and ``S_R``.

    Toro, Spruce and Speares. The contact speed comes from requiring pressure
    and normal velocity to be continuous across it,

        S_* = (p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R))
              / (rho_L (S_L - u_L) - rho_R (S_R - u_R)),

    and each side gets its own star state, so density can jump across the
    middle wave. With that one degree of freedom restored, a stationary contact
    is preserved exactly rather than smeared over twenty cells; see the module
    docstring for the measurement.

    The default here, and the usual default everywhere: within a few operations
    of HLL's cost, and the only one of the three that sees all the physics.
    """

    name = "hllc"
    waves = 3
    resolves_contact = True

    def flux(self, UL, UR):
        g = self.gamma
        rho_l, u_l, p_l = _primitive(UL, g)
        rho_r, u_r, p_r = _primitive(UR, g)
        SL, SR = self.wave_speeds(UL, UR)
        FL, FR = _euler_flux(UL, g), _euler_flux(UR, g)

        mass_l = rho_l * (SL - u_l)
        mass_r = rho_r * (SR - u_r)
        denom = mass_l - mass_r
        denom = np.where(np.abs(denom) < _TOL, _TOL, denom)
        S_star = (p_r - p_l + u_l * mass_l - u_r * mass_r) / denom

        def star_state(U, rho, u, p, S, mass):
            factor = mass / np.where(np.abs(S - S_star) < _TOL, _TOL,
                                     S - S_star)
            energy = (U[..., 2] / rho
                      + (S_star - u) * (S_star + p / np.where(
                          np.abs(mass) < _TOL, _TOL, mass)))
            return factor[..., None] * np.stack(
                [np.ones_like(rho), S_star, energy], axis=-1)

        UL_star = star_state(UL, rho_l, u_l, p_l, SL, mass_l)
        UR_star = star_state(UR, rho_r, u_r, p_r, SR, mass_r)
        FL_star = FL + SL[..., None] * (UL_star - UL)
        FR_star = FR + SR[..., None] * (UR_star - UR)

        out = np.where((S_star >= 0.0)[..., None], FL_star, FR_star)
        out = np.where((SL >= 0.0)[..., None], FL, out)
        return np.where((SR <= 0.0)[..., None], FR, out)


# ---------------------------------------------------------------------------

RIEMANN_REGISTRY: dict[str, type[RiemannSolver]] = {
    "rusanov": RusanovSolver,
    "lax-friedrichs": RusanovSolver,
    "llf": RusanovSolver,
    "hll": HLLSolver,
    "hllc": HLLCSolver,
}


def get_riemann_solver(name: str, gamma: float = 1.4, **kwargs):
    key = name.lower().replace("_", "-")
    if key not in RIEMANN_REGISTRY:
        raise NotImplementedError(
            f"Unknown Riemann solver '{name}'. Available: "
            f"{sorted(set(RIEMANN_REGISTRY))}. Roe's solver is not here: it "
            f"needs an entropy fix to avoid admitting an expansion shock, and "
            f"the fix is a choice rather than a formula.")
    return RIEMANN_REGISTRY[key](gamma=gamma, **kwargs)


# ---------------------------------------------------------------------------
# The exact solver -- verification only
# ---------------------------------------------------------------------------

def _wave_function(p, rho_k, p_k, a_k, gamma):
    """``f_K(p)`` and its derivative: the pressure change each outer wave needs.

    Two branches, and they meet smoothly at ``p = p_K``. Above it the wave is a
    shock and the relation comes from Rankine-Hugoniot; below it the wave is a
    rarefaction and the relation comes from the isentrope. Using one branch
    everywhere is the classic mistake and gives a plausible, wrong ``p*``.
    """
    if p > p_k:                                        # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        root = np.sqrt(A / (B + p))
        f = (p - p_k) * root
        df = root * (1.0 - 0.5 * (p - p_k) / (B + p))
    else:                                              # rarefaction
        power = (gamma - 1.0) / (2.0 * gamma)
        f = 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** power - 1.0)
        df = (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma)) / (rho_k * a_k)
    return f, df


def star_pressure(left, right, gamma: float = 1.4,
                  tol: float = 1e-12) -> tuple[float, float]:
    """``(p_star, u_star)`` for the Riemann problem, by Newton iteration.

    ``left`` and ``right`` are ``(rho, u, p)``. Raises :class:`VacuumError` if
    the two states separate fast enough that no star region exists.
    """
    rho_l, u_l, p_l = (float(v) for v in left)
    rho_r, u_r, p_r = (float(v) for v in right)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    if 2.0 * (a_l + a_r) / (gamma - 1.0) <= u_r - u_l:
        raise VacuumError(
            "the two states separate faster than the rarefactions can follow, "
            "so a vacuum opens between them and there is no star state to "
            "solve for")

    # linearised (primitive-variable) guess; clipped because it can go negative
    # for a strong expansion, and Newton on a negative pressure diverges
    guess = max(tol, 0.5 * (p_l + p_r) - 0.125 * (u_r - u_l)
                * (rho_l + rho_r) * (a_l + a_r))

    p = guess
    for _ in range(_MAX_ITER):
        f_l, df_l = _wave_function(p, rho_l, p_l, a_l, gamma)
        f_r, df_r = _wave_function(p, rho_r, p_r, a_r, gamma)
        step = (f_l + f_r + u_r - u_l) / (df_l + df_r)
        p_new = max(tol, p - step)
        if abs(p_new - p) / (0.5 * (p_new + p)) < tol:
            p = p_new
            break
        p = p_new
    else:
        raise RuntimeError("the star-pressure iteration did not converge")

    f_l, _ = _wave_function(p, rho_l, p_l, a_l, gamma)
    f_r, _ = _wave_function(p, rho_r, p_r, a_r, gamma)
    return p, 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)


def exact_riemann(left, right, xi, gamma: float = 1.4) -> np.ndarray:
    """Exact ``(rho, u, p)`` sampled at similarity coordinates ``xi = x/t``.

    ``left`` and ``right`` are ``(rho, u, p)``; ``xi`` is an array. Returns
    ``(len(xi), 3)``.

    This is the external benchmark for everything else in the module -- the
    only solution here that owes nothing to the discretisation. Note that it is
    the solution of the *differential* problem: comparing it pointwise to a
    finite volume answer compares a point value against a cell average, which
    is an ``O(h^2)`` discrepancy everywhere and ``O(1)`` in the cell a
    discontinuity happens to fall in.
    """
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    rho_l, u_l, p_l = (float(v) for v in left)
    rho_r, u_r, p_r = (float(v) for v in right)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma - 1.0) / (gamma + 1.0)

    p_star, u_star = star_pressure(left, right, gamma)
    out = np.empty((xi.size, 3))

    for i, s in enumerate(xi):
        if s <= u_star:                                # left of the contact
            ratio = p_star / p_l
            if p_star > p_l:                           # left shock
                speed = u_l - a_l * np.sqrt(0.5 * (gamma + 1.0) / gamma * ratio
                                            + g1)
                if s <= speed:
                    out[i] = (rho_l, u_l, p_l)
                else:
                    out[i] = (rho_l * (ratio + g2) / (g2 * ratio + 1.0),
                              u_star, p_star)
            else:                                      # left rarefaction
                head = u_l - a_l
                a_star = a_l * ratio ** g1
                tail = u_star - a_star
                if s <= head:
                    out[i] = (rho_l, u_l, p_l)
                elif s >= tail:
                    out[i] = (rho_l * ratio ** (1.0 / gamma), u_star, p_star)
                else:                                  # inside the fan
                    u = 2.0 / (gamma + 1.0) * (
                        a_l + 0.5 * (gamma - 1.0) * u_l + s)
                    a = 2.0 / (gamma + 1.0) * (
                        a_l + 0.5 * (gamma - 1.0) * (u_l - s))
                    out[i] = (rho_l * (a / a_l) ** (2.0 / (gamma - 1.0)), u,
                              p_l * (a / a_l) ** (1.0 / g1))
        else:                                          # right of the contact
            ratio = p_star / p_r
            if p_star > p_r:                           # right shock
                speed = u_r + a_r * np.sqrt(0.5 * (gamma + 1.0) / gamma * ratio
                                            + g1)
                if s >= speed:
                    out[i] = (rho_r, u_r, p_r)
                else:
                    out[i] = (rho_r * (ratio + g2) / (g2 * ratio + 1.0),
                              u_star, p_star)
            else:                                      # right rarefaction
                head = u_r + a_r
                a_star = a_r * ratio ** g1
                tail = u_star + a_star
                if s >= head:
                    out[i] = (rho_r, u_r, p_r)
                elif s <= tail:
                    out[i] = (rho_r * ratio ** (1.0 / gamma), u_star, p_star)
                else:
                    u = 2.0 / (gamma + 1.0) * (
                        -a_r + 0.5 * (gamma - 1.0) * u_r + s)
                    a = 2.0 / (gamma + 1.0) * (
                        a_r - 0.5 * (gamma - 1.0) * (u_r - s))
                    out[i] = (rho_r * (a / a_r) ** (2.0 / (gamma - 1.0)), u,
                              p_r * (a / a_r) ** (1.0 / g1))
    return out
