r"""
Numerical fluxes for a scalar conservation law.

The finite volume method never differentiates anything. It integrates

    du/dt + div F(u) = 0

over a cell and applies the divergence theorem, leaving a balance between the
cell average and the fluxes through its faces:

    V du_i/dt = - sum over faces of  F_face * A_face

Every face is shared by exactly two cells and contributes to each with opposite
sign, so whatever leaves one enters the other and the total is conserved
**exactly** -- not to truncation error, to round-off, on any mesh. That is the
property the method exists for, and it is why the only thing a scheme has to
supply is a single number per face.

That single number is the difficulty. The face sits between two different cell
values and the scheme has to decide what flux to attribute to it. The choices
here differ only in that decision.

``central``   the average of the two. Second order, and unstable for advection
              without enough diffusion to hold it up -- see the Peclet limit
              below.
``upwind``    the value from whichever side the flow comes from. First order,
              unconditionally monotone, and diffusive enough that the numerical
              diffusion can exceed the physical kind.
``rusanov``   the average plus a jump penalty scaled by the fastest wave speed.
              Upwind's logic without needing to know the sign in advance, which
              is what makes it the one scheme here that generalises unchanged to
              a system.

The Peclet limit is not a tolerance
-----------------------------------
For steady advection-diffusion the cell Peclet number ``Pe = a h / D`` measures
how much a cell transports against how much it diffuses. Central differencing
produces a matrix that stops being diagonally dominant at ``Pe = 2``, and beyond
it the solution oscillates cell to cell -- not because the mesh is too coarse in
some vague sense, but because the discrete operator has changed character. The
oscillations do not decay with refinement at fixed ``Pe``; they vanish only when
refinement pushes ``Pe`` back below two.

:func:`peclet_limit` reports it, and the example demonstrates it, because a
scheme that is exact on smooth problems and wrong on a boundary layer is worth
recognising by sight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "FluxScheme",
    "CentralFlux",
    "UpwindFlux",
    "RusanovFlux",
    "FLUX_REGISTRY",
    "get_flux",
    "peclet_limit",
]


def peclet_limit(velocity: float, diffusivity: float, spacing: float) -> float:
    """Cell Peclet number ``|a| h / D``. Central differencing oscillates above 2.

    ``inf`` for pure advection, which is the honest answer: there is no
    diffusion to stabilise anything, and an unlimited central scheme has no
    stable steady state at all.
    """
    if diffusivity <= 0.0:
        return float("inf")
    return abs(float(velocity)) * float(spacing) / float(diffusivity)


class FluxScheme(ABC):
    """The convective flux at a face, from the states either side."""

    name: str = "flux"

    #: Whether the scheme is monotone -- cannot create a new extremum. First
    #: order in every case, by Godunov's theorem.
    monotone: bool = False

    @abstractmethod
    def face_flux(self, left: np.ndarray, right: np.ndarray,
                  velocity: np.ndarray) -> np.ndarray:
        """Flux through faces whose left and right states are given.

        ``velocity`` is the advecting speed normal to the face, positive from
        left to right.
        """

    def __call__(self, left, right, velocity) -> np.ndarray:
        return self.face_flux(np.asarray(left, dtype=float),
                              np.asarray(right, dtype=float),
                              np.asarray(velocity, dtype=float))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CentralFlux(FluxScheme):
    r"""``F = a (u_L + u_R) / 2``.

    Second-order accurate and the source of every oscillation in the module.
    It carries no information about which way the flow goes, so at a steep
    front it happily interpolates through it.
    """

    name = "central"
    monotone = False

    def face_flux(self, left, right, velocity):
        return velocity * 0.5 * (left + right)


class UpwindFlux(FluxScheme):
    r"""``F = a u_L`` if ``a > 0``, else ``a u_R``.

    Takes the value from the side the flow arrives from, which is the only
    side that can physically influence the face. First order, monotone, and its
    truncation error is a diffusion term with coefficient ``|a| h / 2`` -- so on
    a coarse mesh the scheme's own numerical diffusion can dwarf the physical
    diffusivity it is meant to be resolving. Worth computing before trusting a
    result: that is what :func:`peclet_limit` is comparing against.
    """

    name = "upwind"
    monotone = True

    def face_flux(self, left, right, velocity):
        return velocity * np.where(velocity > 0.0, left, right)


class RusanovFlux(FluxScheme):
    r"""``F = (F_L + F_R)/2 - (s/2)(u_R - u_L)``, with ``s`` the fastest speed.

    Local Lax-Friedrichs. The average, plus a jump penalty proportional to the
    largest wave speed at the face, which is exactly enough dissipation to make
    it monotone. For a scalar with constant velocity it reduces to upwinding
    identically; its value is that the same expression works for a nonlinear
    flux and for a system, where "which side is upwind" has no single answer
    because different characteristics go different ways.
    """

    name = "rusanov"
    monotone = True

    def face_flux(self, left, right, velocity):
        speed = np.abs(velocity)
        return 0.5 * velocity * (left + right) - 0.5 * speed * (right - left)


# ---------------------------------------------------------------------------

FLUX_REGISTRY: dict[str, type[FluxScheme]] = {
    "central": CentralFlux,
    "upwind": UpwindFlux,
    "rusanov": RusanovFlux,
    "lax-friedrichs": RusanovFlux,
    "llf": RusanovFlux,
}


def get_flux(name: str, **kwargs) -> FluxScheme:
    key = name.lower().replace("_", "-")
    if key not in FLUX_REGISTRY:
        raise NotImplementedError(
            f"Unknown flux scheme '{name}'. Available: "
            f"{sorted(set(FLUX_REGISTRY))}. Riemann solvers for the Euler "
            f"equations are a different signature and live in "
            f"scion.discretization.fvm.riemann.")
    return FLUX_REGISTRY[key](**kwargs)
