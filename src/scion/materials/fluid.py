"""
Weakly compressible fluid.

The one material in ``materials/`` that never forms a deformation gradient.
A fluid has no reference configuration to measure ``F`` against -- it forgets
its rest shape by definition -- so the state is the volume ratio ``J`` alone,
advanced by the trace of the velocity gradient, and the deviatoric response
comes from the *rate* of deformation rather than from strain.

That is also why it does not live in ``hencky.py``: there is no SVD, no
principal decomposition, and no elastic strain. It shares only the stateful
``update(F, L, dt, state)`` signature.

The pressure closure is an :class:`~scion.materials.eos.EquationOfState`, so
:class:`~scion.materials.eos.TaitEOS` is the default but a linear or
Mie-Grüneisen closure drops in unchanged.

Provenance: the formulation follows CK-MPM's fluid model (Liu, Wang & Li,
ACM TOG 44(4), 2025; Apache-2.0), reimplemented in batched NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scion.core import small_linalg as sla
from scion.materials.eos import EquationOfState, TaitEOS

__all__ = ["FluidState", "WeaklyCompressibleFluid"]


@dataclass
class FluidState:
    """Per-particle volume ratio ``J = V / V_0``.

    Tracked incrementally rather than as ``det F``, because there is no ``F``.
    The consequence is that ``J`` accumulates its own integration error over a
    long run and does not self-correct -- see the note on
    :meth:`WeaklyCompressibleFluid.update`.
    """

    #: ``(n,)`` volume ratio
    J: np.ndarray

    @classmethod
    def ones(cls, n: int) -> "FluidState":
        return cls(J=np.ones(n))

    def copy(self) -> "FluidState":
        return FluidState(self.J.copy())

    @property
    def n(self) -> int:
        return int(self.J.size)

    def __repr__(self) -> str:
        return (f"FluidState({self.n} points, "
                f"J in [{self.J.min():.6f}, {self.J.max():.6f}])")


class WeaklyCompressibleFluid:
    r"""``sigma = 2 nu sym(L) - p I``, with ``p`` from an equation of state.

    Parameters
    ----------
    eos : pressure closure. Defaults to :class:`~scion.materials.eos.TaitEOS`
        built from ``bulk_modulus``.
    viscosity : dynamic viscosity ``nu``. Zero gives an inviscid fluid, which
        in MPM is not as unstable as it sounds -- the grid transfer supplies its
        own dissipation.
    density : reference density, for the CFL wave speed.

    Why the deviatoric part uses ``L`` and not a strain
    ---------------------------------------------------
    ``sym(L)`` is the rate of deformation, so the shear stress depends on how
    fast the fluid is deforming *now* and not on how far it has deformed --
    which is the definition of a fluid. It also means this model is only as
    good as the velocity gradient the transfer gives it: with plain PIC that
    gradient is heavily smoothed, and the effective viscosity will exceed
    ``viscosity``. APIC or a FLIP blend is the right pairing.
    """

    name = "weakly-compressible-fluid"

    def __init__(self, eos: EquationOfState | None = None,
                 bulk_modulus: float | None = None,
                 gamma: float = 7.0,
                 viscosity: float = 0.0,
                 density: float = 1.0):
        if eos is None:
            if bulk_modulus is None:
                raise ValueError("give either an eos or a bulk_modulus")
            eos = TaitEOS(bulk_modulus, gamma)
        self.eos = eos
        self.viscosity = float(viscosity)
        self.density = float(density)
        if self.viscosity < 0.0:
            raise ValueError("the viscosity must be non-negative")

    @property
    def wave_speed_factor(self) -> float:
        """``sqrt(K)``; divide by ``sqrt(rho)`` for the sound speed.

        No shear stiffness to add, unlike a solid: a fluid's fastest signal is
        the pressure wave.
        """
        return float(np.sqrt(self.eos.bulk_modulus))

    def pressure(self, J: np.ndarray) -> np.ndarray:
        """Pressure from the volume ratio, positive in compression."""
        return self.eos.pressure_from_jacobian(J)

    def update(self, F: np.ndarray, L: np.ndarray, dt: float,
               state: FluidState | None):
        """Advance one step and return ``(cauchy_stress, new_state)``.

        ``F`` is accepted and ignored, so that this model plugs into the same
        stepper path as the solids.

        On the ``J`` update
        -------------------
        ``J <- J (1 + dt tr L)`` is the first-order form, and it is what the
        source this follows uses. It is *not* the same as ``det`` of the updated
        deformation gradient, and the difference accumulates: a fluid circulating
        for many thousands of steps will drift in volume with no mechanism to
        pull it back, because nothing else in the model knows what ``J`` should
        be. If a long run loses or gains mass, this is where it went.
        """
        n = L.shape[0]
        if state is None:
            state = FluidState.ones(n)

        trace_L = np.einsum("pii->p", L)
        J = state.J * (1.0 + dt * trace_L)

        p = self.pressure(J)

        sigma = self.viscosity * (L + sla.transpose(L))
        idx = np.arange(3)
        sigma[:, idx, idx] -= p[:, None]

        return sigma, FluidState(J=J)
