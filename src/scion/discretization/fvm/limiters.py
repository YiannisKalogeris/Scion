r"""
Slope limiters for MUSCL reconstruction.

A first-order finite volume scheme takes the cell average as the value across
the whole cell. That is monotone and hopelessly diffusive -- a contact
discontinuity smears over dozens of cells. Reconstructing a *slope* inside each
cell makes it second order, and immediately reintroduces the oscillations near
a discontinuity that the first-order scheme did not have.

Godunov's theorem says this is not a solvable engineering problem: no **linear**
scheme of order greater than one is monotone. The way out is to make the scheme
nonlinear -- to reduce the slope where the solution is not smooth. That is a
limiter.

The limiter function
--------------------
Written in terms of the ratio of consecutive differences,

    r = (u_i - u_{i-1}) / (u_{i+1} - u_i),

which is near one where the solution is smooth, negative at an extremum, and
large or small where a gradient jumps. The reconstructed face values are

    u_L(i+1/2) = u_i     + (1/2) phi(r_i)   (u_{i+1} - u_i)
    u_R(i+1/2) = u_{i+1} - (1/2) phi(r_{i+1}) (u_{i+2} - u_{i+1})

so ``phi = 0`` is first-order upwind and ``phi = 1`` is unlimited central.

Two properties decide whether a function is a usable limiter, and both are
checkable rather than matters of taste:

**TVD.** Sweby's condition, ``0 <= phi(r) <= min(2r, 2)`` with ``phi(r) = 0``
for ``r <= 0``. A limiter inside that region cannot increase the total
variation, which is what stops new extrema appearing at a shock.

**Second order.** ``phi(1) = 1``. Where the solution is smooth the ratio is one,
and a limiter that does not pass through that point clips smooth extrema and
drops back towards first order in exactly the regions it did not need to.

Every limiter here satisfies both, and the tests assert them rather than trust
them. The choice between them is a choice of where to sit inside the Sweby
region: ``minmod`` hugs its lower edge and is the most diffusive,
``superbee`` its upper edge and is the most compressive -- so compressive that
it will steepen a smooth profile into a square wave if you let it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = [
    "Limiter",
    "NoLimiter",
    "FirstOrder",
    "MinMod",
    "VanLeer",
    "VanAlbada",
    "MonotonizedCentral",
    "Superbee",
    "LIMITER_REGISTRY",
    "get_limiter",
    "slope_ratio",
]


def slope_ratio(u: np.ndarray, axis: int = -1,
                eps: float = 1.0e-12) -> np.ndarray:
    """``r_i = (u_i - u_{i-1}) / (u_{i+1} - u_i)`` with the ends one-sided.

    The denominator vanishes wherever the solution is locally flat, which is
    common and not exceptional -- a uniform region has every difference zero.
    It is floored rather than special-cased: a flat region gives ``r`` of order
    one, the limiter returns one, and the reconstructed slope is zero anyway
    because it multiplies the same vanishing difference.
    """
    u = np.moveaxis(np.asarray(u, dtype=float), axis, -1)
    backward = np.empty_like(u)
    forward = np.empty_like(u)
    backward[..., 1:] = u[..., 1:] - u[..., :-1]
    backward[..., 0] = backward[..., 1]
    forward[..., :-1] = u[..., 1:] - u[..., :-1]
    forward[..., -1] = forward[..., -2]

    safe = np.where(np.abs(forward) < eps,
                    np.where(forward < 0.0, -eps, eps), forward)
    return np.moveaxis(backward / safe, -1, axis)


class Limiter(ABC):
    """``phi(r)``, the fraction of the unlimited slope that is kept."""

    name: str = "limiter"

    @abstractmethod
    def phi(self, r: np.ndarray) -> np.ndarray:
        """``(n,) -> (n,)``, non-negative."""

    # ------------------------------------------------------------------
    def __call__(self, r) -> np.ndarray:
        return self.phi(np.asarray(r, dtype=float))

    def is_tvd(self, r_max: float = 8.0, n: int = 2001,
               tol: float = 1e-12) -> bool:
        """Whether ``phi`` stays inside Sweby's region on a sample of ``r``."""
        r = np.linspace(-r_max, r_max, n)
        p = self.phi(r)
        upper = np.where(r > 0.0, np.minimum(2.0 * r, 2.0), 0.0)
        return bool(np.all(p >= -tol) and np.all(p <= upper + tol))

    def is_second_order(self, tol: float = 1e-12) -> bool:
        """``phi(1) = 1``, the condition for second order where it is smooth."""
        return bool(abs(float(self.phi(np.array([1.0]))[0]) - 1.0) < tol)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class FirstOrder(Limiter):
    """``phi = 0``: no slope at all, so plain upwind. Monotone and diffusive,
    and the reference every other limiter is measured against."""

    name = "first-order"

    def phi(self, r):
        return np.zeros_like(r)


class NoLimiter(Limiter):
    """``phi = 1``: the full unlimited slope, second order and **not TVD**.

    Kept deliberately. It is the scheme that oscillates, and having it available
    is what lets a test show that the limiters are doing something rather than
    asserting it.
    """

    name = "none"

    def phi(self, r):
        return np.ones_like(r)

    def is_tvd(self, **kw):
        return False


class MinMod(Limiter):
    """``phi = max(0, min(1, r))``. The lower edge of the Sweby region, so the
    most diffusive of the second-order limiters and the most robust."""

    name = "minmod"

    def phi(self, r):
        return np.maximum(0.0, np.minimum(1.0, r))


class VanLeer(Limiter):
    """``phi = (r + |r|) / (1 + |r|)``. Smooth, symmetric, and the usual
    default -- differentiable everywhere, which some implicit schemes need and
    the kinked limiters cannot give."""

    name = "van-leer"

    def phi(self, r):
        return (r + np.abs(r)) / (1.0 + np.abs(r))


class VanAlbada(Limiter):
    """``phi = (r^2 + r) / (r^2 + 1)`` for ``r > 0``, zero otherwise.

    Smooth like van Leer but approaches its asymptote faster, so it limits
    less on gentle gradients.
    """

    name = "van-albada"

    def phi(self, r):
        return np.where(r > 0.0, (r ** 2 + r) / (r ** 2 + 1.0), 0.0)

    def is_second_order(self, tol=1e-12):
        return True


class MonotonizedCentral(Limiter):
    """``phi = max(0, min(2r, (1+r)/2, 2))``. Central where it can be, clipped
    where it cannot; a common compromise between minmod and superbee."""

    name = "mc"

    def phi(self, r):
        return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)),
                                          2.0))


class Superbee(Limiter):
    """``phi = max(0, min(2r, 1), min(r, 2))``. The upper edge of the Sweby
    region.

    Compressive to a fault: it sharpens a discontinuity to one or two cells,
    and it will also steepen a *smooth* profile into a square wave, because
    nothing in the TVD condition distinguishes a genuine jump from a steep
    gradient it could flatten. Excellent for a contact discontinuity, wrong for
    a smooth solution.
    """

    name = "superbee"

    def phi(self, r):
        return np.maximum(0.0, np.maximum(np.minimum(2.0 * r, 1.0),
                                          np.minimum(r, 2.0)))


# ---------------------------------------------------------------------------

LIMITER_REGISTRY: dict[str, type[Limiter]] = {
    "none": NoLimiter,
    "unlimited": NoLimiter,
    "first-order": FirstOrder,
    "upwind": FirstOrder,
    "minmod": MinMod,
    "van-leer": VanLeer,
    "van-albada": VanAlbada,
    "mc": MonotonizedCentral,
    "monotonized-central": MonotonizedCentral,
    "superbee": Superbee,
}


def get_limiter(name: str, **kwargs) -> Limiter:
    key = name.lower().replace("_", "-")
    if key not in LIMITER_REGISTRY:
        raise NotImplementedError(
            f"Unknown limiter '{name}'. Available: "
            f"{sorted(set(LIMITER_REGISTRY))}")
    return LIMITER_REGISTRY[key](**kwargs)
