"""Phase-field brittle fracture on an MPM background grid.

Kakouris & Triantafyllou, IJNME 2017. The physics is *not* re-derived here:
:class:`~scion.multiphysics.phase_field_fracture.PhaseFieldFractureProblem`
already carries the model, the staggered solve, the energy split, the
degradation function and the AT1/AT2 dissipation, all verified against
Ambati et al. (2015) on a finite-element mesh. What that class never touches is
a mesh -- it reaches the discretisation only through ``QuadratureTables``. So
this subclass swaps in the MPM tables and inherits everything else.

Two things genuinely differ from the finite-element case, and neither is a
rename:

- **The grid is larger than the body.** A node with no particle support gives an
  identically zero row in both field solves, so it must be constrained out or
  the system is singular. A finite-element mesh has no such nodes, which is why
  the FEM path needs no equivalent.
- **The tables depend on where the particles are.** They are built once at
  construction. A body that deforms enough for particles to cross cells needs
  them rebuilt -- :meth:`refresh` -- which is the same cost the reference
  reports as re-factorising the stiffness matrix on cell crossing (their
  measured 13% overhead against PF-FEM).
"""

from __future__ import annotations

import numpy as np

from ..discretization.mpm.tables import MPMQuadratureTables
from .phase_field_fracture import PhaseFieldFractureProblem

__all__ = ["PhaseFieldMPMProblem"]


class _GridDomain:
    """The little that ``PhaseFieldFractureProblem`` asks of a domain.

    It wants ``n_nodes`` to size the fields and ``coordinates`` to know the
    dimension. On an MPM grid those are the *background* nodes -- the unknowns
    live there, exactly as they do on a mesh.
    """

    def __init__(self, grid):
        self.grid = grid
        self.coordinates = np.asarray(grid.node_coordinates())[:, :grid.dim]
        self.n_nodes = int(grid.n_nodes)
        self.dim = int(grid.dim)


class PhaseFieldMPMProblem(PhaseFieldFractureProblem):
    """Phase-field fracture with the fields carried on an MPM grid.

    Parameters
    ----------
    grid, particles : the background grid and the particle set. Particles are
        the quadrature points; ``vol0`` supplies the integration weights.
    model : :class:`~scion.materials.fracture.PhaseFieldModel`, unchanged from
        the finite-element path.
    constrain_unsupported : pin nodes with no particle support in both fields.
        Leave it on: without it the tangent is singular, and the failure is a
        LinAlgError from inside the linear solve rather than anything that names
        the cause.
    """

    def __init__(self, grid, particles, model, thickness: float = 1.0,
                 strategy: str = "staggered", stagger_tol: float = 10.0,
                 max_stagger: int = 100, damage_tol: float = 1.0e-4,
                 verbose: bool = False, constrain_unsupported: bool = True):
        self.grid = grid
        self.particles = particles
        super().__init__(_GridDomain(grid), model, thickness, strategy,
                         stagger_tol, max_stagger, damage_tol, verbose)
        self._unsupported = np.flatnonzero(~self.tables.supported_nodes())
        if constrain_unsupported and self._unsupported.size:
            for c in range(self.dim):
                self.u.prescribe(self._unsupported, c, 0.0)
            self.d.prescribe(self._unsupported, 0, 0.0)

    # ------------------------------------------------------------------
    def _build_quadrature_tables(self) -> None:
        """Same aliases as the finite-element path, MPM tables underneath."""
        self.tables = MPMQuadratureTables(self.grid, self.particles,
                                          self.thickness)
        t = self.tables
        self.conn = t.conn
        self.n_elements, self.npe, self.n_qp = t.n_elements, t.npe, t.n_qp
        self.N, self.dN, self.w, self.B = t.N, t.dN, t.w, t.B
        self.u_edofs = t.u_edofs
        self.d_edofs = t.scalar_edofs
        self.element_size = t.element_size

    # ------------------------------------------------------------------
    def _laplacian(self, coeff):
        """The gradient operator, with unsupported nodes made solvable.

        ``solve_second`` factorises the **full** phase-field system rather than
        its free part -- on a finite-element mesh every node carries support, so
        there is nothing to exclude. An MPM grid is larger than the body, and
        those nodes give identically zero rows in both the Laplacian and the
        mass term, so the factorisation is exactly singular and the staggered
        loop returns nan.

        An identity block on exactly those rows is not a regularisation and
        does not perturb the answer: their columns are zero too, so they are
        uncoupled from the supported system, and the right-hand side vanishes
        there as well -- so they solve to zero, which is the value a node with
        no material should carry. Prescribing them on the ``d`` field does not
        work, because that path never consults the prescribed set.
        """
        K = super()._laplacian(coeff).tolil()
        if self._unsupported.size:
            K[self._unsupported, self._unsupported] = 1.0
        return K.tocsr()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the tables against the current particle positions.

        Needed once particles have moved appreciably relative to the grid --
        the reference's own 13% overhead against PF-FEM is exactly this. The
        history ``H`` is indexed by particle and survives, since the particle
        ordering does not change; it is the *stencil* that does.

        **Refuses when a node gains support.** ``Field`` can prescribe a degree
        of freedom but cannot release one, so a node that was unsupported (and
        therefore pinned) and is now reached by particles would stay pinned --
        material clamped at the body's edge, silently, with the solve still
        converging. Rebuilding the problem is the workaround. Nodes that *lose*
        support are handled: they are pinned here.

        Not calling this at all is the other silent failure: stale shape
        functions still assemble and still converge, to the wrong problem.
        """
        was_unsupported = set(self._unsupported.tolist())
        self._build_quadrature_tables()
        now = ~self.tables.supported_nodes()
        gained = sorted(set(np.flatnonzero(~now).tolist()) & was_unsupported)
        if gained:
            raise NotImplementedError(
                f"{len(gained)} nodes gained particle support on refresh "
                f"(e.g. {gained[:5]}). They are still pinned from when they "
                f"had none, and Field has no way to release a prescribed "
                f"degree of freedom -- so they would clamp material at the "
                f"body's edge while the solve carried on converging. Rebuild "
                f"the problem instead.")
        self._unsupported = np.flatnonzero(now)
        fresh = sorted(set(self._unsupported.tolist()) - was_unsupported)
        if fresh:
            idx = np.asarray(fresh, dtype=int)
            for c in range(self.dim):
                self.u.prescribe(idx, c, 0.0)
            self.d.prescribe(idx, 0, 0.0)

    def reaction(self, nodes, component: int) -> float:
        """Net internal force on ``nodes`` in ``component``.

        The load-displacement curve the reference plots is this, taken on the
        loaded edge. It is the internal force rather than a residual because at
        a converged state they differ only by the prescribed reaction.
        """
        _, f_int = self._assemble_u()
        idx = np.asarray(nodes, dtype=int) * self.dim + int(component)
        return float(f_int[idx].sum())
