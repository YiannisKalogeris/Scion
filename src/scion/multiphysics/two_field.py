r"""
Two-field coupled problems: the orchestration layer.

What lives here is only the wiring that neither ``materials`` nor ``solvers``
can see: two fields on one mesh, their DOF maps and Dirichlet data, the loop
over load or time steps, and the choice of how the two equations are solved
relative to each other.

Deliberately thin. Constitutive behaviour stays in ``materials`` so a non-FEM
discretization can reuse it; solution algorithms stay in ``solvers`` so
``StaggeredSolver`` serves every physics that lands here.

Coupling strategies
-------------------
``"staggered"``  alternate until an energy functional stops decreasing. The
                 default, and the robust choice for fracture: the monolithic
                 tangent loses positive definiteness as soon as damage
                 localises.
``"one_way"``    solve the second field once per step, then the first. Correct
                 whenever the back-coupling is genuinely absent -- most
                 thermomechanics -- and wasteful to iterate on.
``"monolithic"`` reserved; a subclass must supply the coupled tangent.

Transient support
-----------------
``dt`` is carried through the step loop even though phase-field fracture is
rate-independent, because poroelasticity and thermomechanics are not, and
retrofitting time into a quasi-static base is unpleasant. Quasi-static
subclasses simply ignore it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as _field

import numpy as np

from scion.core.dof import DOFMap
from scion.solvers.staggered import StaggeredResult, StaggeredSolver

__all__ = ["Field", "StepResult", "TwoFieldProblem"]


class Field:
    """One nodal field: its values, its DOF map and its Dirichlet data.

    ``DOFMap`` records *which* DOFs are constrained but not what they are
    constrained to, so the prescribed values are kept here alongside it. That
    also lets a value be changed per load step -- which is the whole point of
    displacement control -- without touching the map.
    """

    def __init__(self, name: str, n_nodes: int, n_components: int = 1,
                 dtype=np.float64):
        self.name = name
        self.n_nodes = int(n_nodes)
        self.n_components = int(n_components)
        self.dof_map = DOFMap(self.n_nodes, self.n_components)
        self.values = np.zeros(self.n_nodes * self.n_components, dtype=dtype)
        self._prescribed: dict[int, float] = {}

    # ------------------------------------------------------------------
    def prescribe(self, nodes, component: int, value=0.0) -> None:
        """Constrain ``component`` at ``nodes`` to ``value``.

        ``value`` may be a scalar or an array matching ``nodes``. Re-calling
        with a new value updates it; the constraint itself is registered once.
        """
        nodes = np.atleast_1d(np.asarray(nodes, dtype=np.int64))
        values = np.broadcast_to(np.asarray(value, dtype=float), nodes.shape)
        for n, v in zip(nodes, values):
            d = self.dof_map.dof(int(n), component)
            if d not in self._prescribed:
                self.dof_map.constrain(int(n), component)
            self._prescribed[d] = float(v)

    def scale_prescribed(self, factor: float, base: dict[int, float]) -> None:
        """Set prescribed values to ``factor * base`` -- displacement control."""
        for d, v in base.items():
            self._prescribed[d] = factor * v

    # ------------------------------------------------------------------
    @property
    def prescribed_dofs(self) -> np.ndarray:
        return np.array(sorted(self._prescribed), dtype=np.int64)

    @property
    def prescribed_values(self) -> np.ndarray:
        return np.array([self._prescribed[d] for d in sorted(self._prescribed)])

    @property
    def free_dofs(self) -> np.ndarray:
        return self.dof_map.free_dofs

    @property
    def n_dof(self) -> int:
        return self.dof_map.n_total

    def nodal(self) -> np.ndarray:
        """``(n_nodes, n_components)`` view of the values."""
        return self.values.reshape(self.n_nodes, self.n_components)

    def apply_prescribed(self) -> None:
        """Write the prescribed values into the solution vector."""
        if self._prescribed:
            self.values[self.prescribed_dofs] = self.prescribed_values

    def __repr__(self) -> str:
        return (f"Field({self.name!r}, {self.n_nodes} nodes x "
                f"{self.n_components}, {len(self._prescribed)} prescribed)")


@dataclass
class StepResult:
    """Outcome of one load or time step."""

    step: int
    factor: float
    converged: bool
    staggered: StaggeredResult | None = None
    reactions: dict = _field(default_factory=dict)
    energies: dict = _field(default_factory=dict)

    def __repr__(self) -> str:
        it = self.staggered.iterations if self.staggered else 0
        return (f"StepResult(step={self.step}, factor={self.factor:.4g}, "
                f"converged={self.converged}, staggered_iters={it})")


class TwoFieldProblem(ABC):
    """Base for coupled problems with two nodal fields on one mesh.

    Subclasses implement :meth:`solve_first`, :meth:`solve_second` and
    :meth:`total_energy`; everything else -- stepping, the staggered driver,
    history commit/restore -- is handled here.
    """

    #: names of the two fields, for logging
    field_names: tuple[str, str] = ("first", "second")

    def __init__(self, domain, strategy: str = "staggered",
                 stagger_tol: float = 10.0, max_stagger: int = 100,
                 change_tol: float = 1.0e-4, verbose: bool = False):
        if strategy not in ("staggered", "one_way", "monolithic"):
            raise ValueError(f"unknown coupling strategy '{strategy}'")
        if strategy == "monolithic":
            raise NotImplementedError(
                "monolithic coupling needs the coupled tangent; subclass and "
                "override solve_step() to provide it"
            )
        self.domain = domain
        self.strategy = strategy
        self.stagger_tol = float(stagger_tol)
        self.max_stagger = int(max_stagger)
        self.change_tol = float(change_tol)
        self.verbose = bool(verbose)
        self.history: list[StepResult] = []

    # ------------------------------------------------------------------
    # To implement
    # ------------------------------------------------------------------
    @abstractmethod
    def solve_first(self) -> int:
        """Update field one at fixed field two. Returns an iteration count."""

    @abstractmethod
    def solve_second(self) -> int:
        """Update field two at fixed field one. Returns an iteration count."""

    @abstractmethod
    def total_energy(self) -> float:
        """Energy functional the staggered cycle monitors.

        Must include *both* contributions. One of them alone can decrease while
        the total does not, which silently stops the cycle early.
        """

    # Optional hooks -----------------------------------------------------
    def field_change(self) -> float:
        """Largest change in the second field over the last iteration.

        The energy criterion measures the *shape* of the energy decay and is
        blind to a slow steady drift, which is exactly what happens near a
        crack tip. This gives the staggered driver a second, independent test.
        Default ``inf`` disables it.
        """
        return float("inf")

    def begin_step(self, step: int, factor: float, dt: float) -> None:
        """Apply the load level for this step."""

    def end_step(self, result: StepResult) -> None:
        """Accept the converged state -- commit quadrature history here."""

    def reject_step(self, step: int) -> None:
        """Discard a failed step -- restore quadrature history here."""

    def report(self) -> dict:
        """Per-step quantities to record. Reactions, crack length, whatever."""
        return {}

    # ------------------------------------------------------------------
    def solve_step(self, step: int, factor: float, dt: float) -> StepResult:
        self.begin_step(step, factor, dt)

        if self.strategy == "one_way":
            n2 = self.solve_second()
            n1 = self.solve_first()
            res = StaggeredResult(True, 1, [self.total_energy()], 0.0,
                                  [(n1, n2)], self.field_change(), "one_way")
        else:
            res = StaggeredSolver(
                self.solve_first, self.solve_second, self.total_energy,
                tol=self.stagger_tol, max_iterations=self.max_stagger,
                change=self.field_change, change_tol=self.change_tol,
                verbose=self.verbose,
            ).run()

        out = StepResult(step, factor, res.converged, res, **self.report())
        if res.converged:
            self.end_step(out)
        else:
            self.reject_step(step)
        self.history.append(out)
        return out

    # ------------------------------------------------------------------
    def run(self, load_factors, dt: float = 0.0, callback=None,
            stop_on_failure: bool = True, adaptive: bool = True,
            min_increment: float = 1.0e-9, growth: float = 1.5) -> list[StepResult]:
        """Step through ``load_factors``, cutting the increment when a step fails.

        ``load_factors`` is any iterable of scalars -- typically an increasing
        displacement multiplier, but a cyclic history works too and is what
        exercises the hybrid model's compression constraint.

        Adaptive stepping matters for brittle fracture specifically. Past the
        peak load the crack releases more energy in one increment than it can
        absorb, the staggered cycle stops converging, and no number of
        iterations helps -- the increment itself is too large. Where that
        happens is not known in advance (it depends on the notch, the mesh and
        ``l``), so a hand-tuned refinement schedule either wastes steps in the
        elastic range or misses the onset entirely.

        On failure the step is rejected -- ``reject_step`` restores the
        committed quadrature state, so nothing of the failed attempt survives
        -- the increment is halved and the step retried, down to
        ``min_increment``. After a success the increment grows again by
        ``growth``, capped at the originally requested spacing, so the elastic
        range still runs at full speed.

        This handles a steep post-peak drop. It cannot handle a true snap-back,
        where equilibrium requires the *displacement* to decrease: no positive
        increment reaches the next point on the path. That needs arc-length
        control, with the load level itself an unknown.
        """
        targets = [float(f) for f in np.atleast_1d(load_factors)]
        results: list[StepResult] = []
        current = 0.0
        step = 0

        for i, target in enumerate(targets):
            requested = target - current
            if requested <= 0.0:                      # unloading or repeat
                step += 1
                r = self.solve_step(step, target, dt)
                results.append(r)
                current = target
                if callback is not None:
                    callback(r, self)
                if stop_on_failure and not r.converged:
                    break
                continue

            increment = requested
            while current < target - 1e-15:
                trial = min(current + increment, target)
                step += 1
                r = self.solve_step(step, trial, dt)

                if r.converged:
                    results.append(r)
                    current = trial
                    if callback is not None:
                        callback(r, self)
                    increment = min(increment * growth, requested)
                    continue

                if not adaptive or increment <= min_increment:
                    results.append(r)
                    if callback is not None:
                        callback(r, self)
                    if self.verbose:
                        print(f"  step failed at factor {trial:.6g} with the "
                              f"smallest allowed increment; arc-length control "
                              f"is needed here")
                    if stop_on_failure:
                        return results
                    current = trial
                    continue

                increment *= 0.5
                if self.verbose:
                    print(f"  cutting increment to {increment:.3e} "
                          f"at factor {current:.6g}")
            else:
                continue
            break

        return results
