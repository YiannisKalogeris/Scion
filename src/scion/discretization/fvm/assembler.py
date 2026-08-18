r"""
Cell-centred finite volume for scalar advection-diffusion.

Solves

    du/dt + a . grad(u) = D lap(u) + s

in conservation form on a uniform Cartesian grid, either as a steady linear
system or by explicit time stepping.

Cells, not points
-----------------
This is the difference from :mod:`scion.discretization.fdm` that catches
people, and the two take the same ``grid_shape`` and ``spacing`` arguments with
different meanings.

    FDM   ``grid_shape`` counts *interior points*; the boundary sits one full
          step beyond the first unknown and carries data, so ``h = L/(n+1)``
    FVM   ``grid_shape`` counts *cells*, which tile the whole domain; centres
          are at ``(i + 1/2) h`` and the boundary faces are the domain edges,
          so ``h = L/n``

:meth:`FVMAssembler.centres` and :meth:`FVMAssembler.faces` return both sets, so
neither has to be reconstructed by hand.

Conservation is structural
--------------------------
Every interior face appears in exactly two cell balances with opposite signs, so
the total ``sum u_i V_i`` can only change through the boundary faces --
exactly, to round-off, on any mesh and at any resolution, whatever the flux
scheme or limiter. :meth:`FVMAssembler.total` and
:meth:`FVMAssembler.boundary_flux` exist so that can be measured rather than
believed.

That property is *not* shared by the FDM or FEM assemblers here: they
approximate derivatives, and their global balance holds only to truncation
error. It is the entire reason to reach for FVM.

Boundaries
----------
Ghost cells, one layer for the first-order operator and two for MUSCL.
``"neumann"`` copies the interior cell; ``"periodic"`` wraps; ``("dirichlet",
value)`` reflects the ghost about the face so that the *average* across it is
the boundary value.

The reflection is not cosmetic. Setting the ghost *equal* to the boundary value
places the condition half a cell inside the domain and costs an order of
accuracy; the reflected ghost puts it on the face, and it reproduces the
one-sided diffusive gradient over ``h/2`` exactly.

The boundary faces are then left to the ghost, and the tempting alternative is
worse. A Dirichlet face has a known value, so its convective flux is exactly
``a u_b`` and it looks obviously right to write that in directly instead of
upwinding the ghost's ``a (2 u_b - u_0)``. Doing so changes nothing for the
central scheme -- the reflection already produces ``a u_b`` there, to the bit --
and for upwinding it removes the ``a/h`` that the outflow cell's diagonal was
relying on. Measured on the boundary layer at ``Pe = 10``: reflection gives a
monotone profile with a max error of 0.067, the "exact" boundary flux gives a
non-monotone one that undershoots to -4.1. Upwind's own order is one, so the
reflected flux's ``O(h)`` error costs nothing it was not already paying, and it
keeps the operator an M-matrix, which is the whole point of upwinding.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from scion.discretization.base import DiscretizationBase, LinearSystem
from scion.discretization.fvm.flux import get_flux, peclet_limit
from scion.discretization.fvm.limiters import get_limiter

__all__ = ["FVMAssembler"]

_KINDS = ("dirichlet", "neumann", "periodic")
_SIDES = (("left", "right"), ("bottom", "top"))
_EPS = 1.0e-12


class FVMAssembler(DiscretizationBase):
    """Cell-centred finite volume on a uniform Cartesian grid, 1-D or 2-D.

    Parameters
    ----------
    domain : accepted for interface consistency; the grid is described by the
        arguments below and ``None`` is fine.
    grid_shape : number of **cells** per direction.
    spacing : cell size, scalar or one per direction.
    velocity : advecting velocity, scalar or one component per direction.
    diffusivity : ``D``. Zero gives pure advection, and then :meth:`peclet` is
        infinite -- which is the honest answer, not a failure.
    flux : name or :class:`~.flux.FluxScheme` for the convective face flux.
    limiter : name or :class:`~.limiters.Limiter` for MUSCL reconstruction.
        Used by :meth:`divergence` and :meth:`advance`. The steady
        :meth:`assemble` ignores it: a limiter is a nonlinear function of the
        solution, so a limited scheme has no linear system at all.
    origin : coordinate of the lower boundary per direction.
    """

    def __init__(self, domain=None, grid_shape=(10,), spacing=1.0,
                 velocity=1.0, diffusivity=0.0, flux="upwind",
                 limiter="minmod", origin=0.0):
        self.domain = domain

        self.grid_shape = tuple(int(n) for n in np.atleast_1d(grid_shape))
        self.dim = len(self.grid_shape)
        if self.dim not in (1, 2):
            raise NotImplementedError(
                f"{self.dim}-D finite volume; 1-D and 2-D are implemented")
        if min(self.grid_shape) < 2:
            raise ValueError("at least two cells are needed per direction")

        def _per_direction(value):
            return tuple(float(x) for x in np.broadcast_to(
                np.asarray(value, dtype=float), (self.dim,)))

        self.h = _per_direction(spacing)
        self.origin = _per_direction(origin)
        self.velocity = _per_direction(velocity)
        self.diffusivity = float(diffusivity)

        self.flux = get_flux(flux) if isinstance(flux, str) else flux
        self.limiter = (get_limiter(limiter) if isinstance(limiter, str)
                        else limiter)

        # directions are addressed by *negative* axis so a leading batch
        # dimension passes through every operator untouched -- which is what
        # turns the matrix assembly into a few blocked probes instead of one
        # call per unknown
        self._axis = tuple(d - self.dim for d in range(self.dim))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(np.prod(self.grid_shape))

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.h))

    def centres(self, d: int = 0) -> np.ndarray:
        """Cell centres along ``d``, at ``(i + 1/2) h``."""
        return self.origin[d] + self.h[d] * (np.arange(self.grid_shape[d])
                                             + 0.5)

    def faces(self, d: int = 0) -> np.ndarray:
        """The ``n + 1`` face coordinates along ``d``, ends included."""
        return self.origin[d] + self.h[d] * np.arange(self.grid_shape[d] + 1)

    def length(self, d: int = 0) -> float:
        return self.h[d] * self.grid_shape[d]

    def coordinates(self) -> np.ndarray:
        """Cell centres, ``(n_cells, dim)``, in solve order."""
        grids = np.meshgrid(*[self.centres(d) for d in range(self.dim)],
                            indexing="ij")
        return np.stack([g.ravel() for g in grids], axis=1)

    def peclet(self, d: int = 0) -> float:
        """Cell Peclet number ``|a| h / D`` along ``d``.

        Central differencing loses diagonal dominance at two and oscillates
        beyond it. The oscillation does not decay with refinement at fixed
        ``Pe``; it goes away only when refinement drops ``Pe`` below two.
        """
        return peclet_limit(self.velocity[d], self.diffusivity, self.h[d])

    # ------------------------------------------------------------------
    def total(self, u) -> float:
        """``sum u_i V_i`` -- the quantity the method conserves exactly."""
        return float(np.sum(np.asarray(u, dtype=float)) * self.cell_volume)

    def default_conditions(self, kind="neumann") -> dict:
        return {s: kind for pair in _SIDES[:self.dim] for s in pair}

    # ------------------------------------------------------------------
    # Ghost cells
    # ------------------------------------------------------------------
    def _resolve(self, conditions, d):
        """``((kind, value), (kind, value))`` for the two ends of direction
        ``d``, validated."""
        names = _SIDES[d]
        out = []
        for name in names:
            if name not in conditions:
                raise KeyError(
                    f"no boundary condition for {name!r}; this grid needs "
                    f"{[s for p in _SIDES[:self.dim] for s in p]}")
            spec = conditions[name]
            kind = spec[0] if isinstance(spec, tuple) else spec
            if kind not in _KINDS:
                raise ValueError(f"unknown boundary condition {kind!r}; "
                                 f"use one of {_KINDS}")
            out.append((kind, float(spec[1]) if kind == "dirichlet" else None))
        if (out[0][0] == "periodic") != (out[1][0] == "periodic"):
            raise ValueError("a periodic direction must be periodic at both "
                             "ends")
        return out[0], out[1]

    def _pad(self, u: np.ndarray, conditions, width: int) -> np.ndarray:
        """Add ``width`` ghost layers per side. ``u`` may carry batch axes."""
        padded = u
        for d in range(self.dim):
            axis = self._axis[d]
            (lo_kind, lo_value), (hi_kind, hi_value) = self._resolve(
                conditions, d)

            if lo_kind == "periodic":
                head = np.take(padded, range(-width, 0), axis=axis)
                tail = np.take(padded, range(width), axis=axis)
                padded = np.concatenate([head, padded, tail], axis=axis)
                continue

            first = np.take(padded, [0] * width, axis=axis)
            last = np.take(padded, [padded.shape[axis] - 1] * width, axis=axis)
            # reflect about the face, so the mean of ghost and neighbour is the
            # boundary value; the ghost set to the value itself would place the
            # condition half a cell inside the domain
            low = first if lo_kind == "neumann" else 2.0 * lo_value - first
            high = last if hi_kind == "neumann" else 2.0 * hi_value - last
            padded = np.concatenate([low, padded, high], axis=axis)
        return padded

    # ------------------------------------------------------------------
    # Face fluxes
    # ------------------------------------------------------------------
    def _muscl(self, v: np.ndarray):
        """Limited face states from a two-ghost-layer array, along the last
        axis.

        ``v`` has ``n + 4`` entries and the interior has ``n + 1`` faces. Each
        face takes a left state from the cell below and a right state from the
        cell above, both extrapolated half a cell along the limited slope.
        """
        n = v.shape[-1] - 4
        diff = v[..., 1:] - v[..., :-1]                        # (..., n+3)
        forward = diff[..., 1:]                                # (..., n+2)
        safe = np.where(np.abs(forward) < _EPS,
                        np.where(forward < 0.0, -_EPS, _EPS), forward)
        slope = 0.5 * self.limiter(diff[..., :-1] / safe) * forward
        return (v[..., 1:n + 2] + slope[..., :n + 1],
                v[..., 2:n + 3] - slope[..., 1:n + 2])

    def _faces(self, trimmed: np.ndarray, d: int, width: int) -> np.ndarray:
        """Total flux at the ``n + 1`` faces along ``d``.

        ``trimmed`` carries ghosts along ``d`` only, and every boundary
        condition has already been absorbed into them -- see the note on
        reflection in the module docstring for why the boundary faces are not
        special-cased here.
        """
        v = np.moveaxis(trimmed, self._axis[d], -1)
        n = v.shape[-1] - 2 * width

        if width == 2:
            left, right = self._muscl(v)
        else:
            left, right = v[..., :-1], v[..., 1:]

        face = self.flux(left, right, np.full(left.shape, self.velocity[d]))

        if self.diffusivity != 0.0:
            inner = v[..., width - 1:width + n + 1]
            face = face - self.diffusivity * np.diff(inner, axis=-1) / self.h[d]

        return np.moveaxis(face, -1, self._axis[d])

    def _divergence(self, u: np.ndarray, conditions, width: int) -> np.ndarray:
        """``-div F``: the right-hand side of ``du/dt``, shaped like ``u``."""
        u = np.asarray(u, dtype=float)
        padded = self._pad(u, conditions, width)
        out = np.zeros(u.shape)

        for d in range(self.dim):
            trimmed = padded
            for other in range(self.dim):          # strip the other ghosts
                if other != d:
                    axis = self._axis[other]
                    trimmed = np.take(
                        trimmed, range(width, trimmed.shape[axis] - width),
                        axis=axis)
            face = self._faces(trimmed, d, width)
            out = out - np.diff(face, axis=self._axis[d]) / self.h[d]
        return out

    # ------------------------------------------------------------------
    # Explicit time stepping
    # ------------------------------------------------------------------
    def divergence(self, u, conditions=None) -> np.ndarray:
        """``-div F`` with MUSCL reconstruction: second order where smooth."""
        conditions = conditions or self.default_conditions()
        u = np.asarray(u, dtype=float).reshape(self.grid_shape)
        return self._divergence(u, conditions, width=2)

    def boundary_flux(self, u, conditions=None) -> float:
        """Net outflow through the domain boundary, per unit time.

        The only route by which :meth:`total` can change; comparing the two is
        the conservation check.
        """
        return -float(np.sum(self.divergence(u, conditions)) * self.cell_volume)

    def stable_dt(self, cfl: float = 0.4) -> float:
        """Largest stable explicit step, advective and diffusive limits
        together.

        Both matter. The diffusive limit scales as ``h^2`` where the advective
        one scales as ``h``, so on a fine enough mesh diffusion sets the step
        no matter how small the diffusivity is.
        """
        rate = sum(abs(self.velocity[d]) / self.h[d] for d in range(self.dim))
        if self.diffusivity > 0.0:
            rate += 2.0 * self.diffusivity * sum(1.0 / self.h[d] ** 2
                                                 for d in range(self.dim))
        return float(cfl / rate) if rate > 0.0 else float("inf")

    def advance(self, u, dt: float, conditions=None,
                stages: int = 2) -> np.ndarray:
        """One explicit step by SSP Runge-Kutta of 1, 2 or 3 stages.

        Strong-stability-preserving means every stage is a convex combination
        of forward Euler steps, so a scheme that is TVD under forward Euler
        stays TVD under these. An arbitrary Runge-Kutta method of the same
        order does not, and will put the oscillations back into a limited
        scheme.
        """
        conditions = conditions or self.default_conditions()
        u = np.asarray(u, dtype=float).reshape(self.grid_shape)

        def L(w):
            return self._divergence(w, conditions, width=2)

        if stages == 1:
            return u + dt * L(u)
        if stages == 2:
            u1 = u + dt * L(u)
            return 0.5 * u + 0.5 * (u1 + dt * L(u1))
        if stages == 3:
            u1 = u + dt * L(u)
            u2 = 0.75 * u + 0.25 * (u1 + dt * L(u1))
            return (u + 2.0 * (u2 + dt * L(u2))) / 3.0
        raise ValueError("SSP Runge-Kutta is implemented for 1, 2 or 3 stages")

    def solve_unsteady(self, u0, t_end: float, conditions=None,
                       cfl: float = 0.4, stages: int = 2):
        """Integrate to ``t_end``; returns ``(field, n_steps)``."""
        conditions = conditions or self.default_conditions()
        u = np.asarray(u0, dtype=float).reshape(self.grid_shape).copy()
        dt_max = self.stable_dt(cfl)
        t, steps = 0.0, 0
        while t < t_end - 1e-14:
            dt = min(dt_max, t_end - t)
            u = self.advance(u, dt, conditions, stages)
            t += dt
            steps += 1
        return u, steps

    # ------------------------------------------------------------------
    # Steady linear system
    # ------------------------------------------------------------------
    def assemble(self, source=None, conditions=None,
                 block: int = 128) -> LinearSystem:
        """``A u = b`` for the steady problem ``a . grad(u) - D lap(u) = s``.

        Built from the first-order (unreconstructed) operator, because a
        limiter is a nonlinear function of the solution and a limited scheme
        has no linear system at all. ``flux="upwind"`` gives a monotone
        operator; ``flux="central"`` gives second order and, above ``Pe = 2``,
        the oscillations that come with it.

        The matrix is obtained by probing the same operator the explicit
        stepper uses, ``block`` columns at a time. That operator is affine --
        ``R(u) = A u + c``, with ``c`` the inhomogeneous boundary data -- so
        ``c = R(0)`` is subtracted from every probe and moved to the
        right-hand side. Probing rather than writing the stencil out a second
        time keeps the matrix and the residual from ever disagreeing about a
        boundary.
        """
        conditions = conditions or self.default_conditions("dirichlet")
        conditions = {k: (("dirichlet", 0.0) if v == "dirichlet" else v)
                      for k, v in conditions.items()}

        def residual(u):
            return -self._divergence(u, conditions, width=1)

        n = self.n_cells
        offset = residual(np.zeros(self.grid_shape)).ravel()

        rows = []
        for start in range(0, n, block):
            stop = min(start + block, n)
            probe = np.zeros((stop - start, n))
            probe[np.arange(stop - start), np.arange(start, stop)] = 1.0
            batch = residual(probe.reshape((stop - start,) + self.grid_shape))
            rows.append(batch.reshape(stop - start, n) - offset)

        # probe j produced row j of A-transpose, i.e. column j of A
        A = sparse.csr_matrix(np.concatenate(rows, axis=0).T)
        A.eliminate_zeros()

        s = _sample(source, self.coordinates())
        return LinearSystem(matrix=A, rhs=s - offset)

    def operator(self, conditions=None) -> sparse.csr_matrix:
        """The steady matrix alone, without source or boundary data."""
        return self.assemble(conditions=conditions).matrix

    def recover_fields(self, solution) -> dict[str, np.ndarray]:
        return {"primary": np.asarray(solution).reshape(self.grid_shape)}

    def __repr__(self) -> str:
        shape = " x ".join(str(n) for n in self.grid_shape)
        return (f"FVMAssembler({shape} cells, flux='{self.flux.name}', "
                f"limiter='{self.limiter.name}', Pe={self.peclet():.4g})")


# ---------------------------------------------------------------------------

def _sample(value, points: np.ndarray) -> np.ndarray:
    if value is None:
        return np.zeros(points.shape[0])
    if callable(value):
        return np.broadcast_to(np.asarray(value(*points.T), dtype=float),
                               (points.shape[0],)).ravel().copy()
    arr = np.asarray(value, dtype=float)
    return (np.full(points.shape[0], float(arr)) if arr.ndim == 0
            else arr.ravel())
