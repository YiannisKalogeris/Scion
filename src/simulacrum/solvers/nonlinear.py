"""
Newton-Raphson nonlinear solver with load/displacement control.

Solves R(u) = f_ext - f_int(u) = 0 by repeated linearisation:
    K_T(u_k) δu = -R(u_k)
    u_{k+1} = u_k + δu

Strategies:
    - LoadControl: increment external force in N steps
    - DisplacementControl: increment prescribed DOFs in N steps

Usage:
    from simulacrum.solvers.nonlinear import (
        NewtonRaphson, LoadControl, DisplacementControl
    )

    # Basic NR
    nr = NewtonRaphson(residual_fn, tangent_fn)
    u, info = nr.solve(u0)

    # Displacement control (e.g., nonlinear elasticity)
    dc = DisplacementControl(
        internal_force_fn=f_int_fn,
        tangent_fn=K_T_fn,
        free_dofs=free_dofs,
        prescribed_dofs=prescribed_dofs,
        prescribed_values=u_prescribed_total,
        n_steps=20,
    )
    u_history, reaction_history, info = dc.solve()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from simulacrum.solvers.linear import LinearSolverBase, DirectSolver
from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


# ====================================================================
# DATA CLASSES
# ====================================================================

@dataclass
class ConvergenceInfo:
    """Convergence information for a single NR solve."""
    converged: bool
    iterations: int
    residual_norms: list[float] = field(default_factory=list)


@dataclass
class StepInfo:
    """Information about a single load/displacement step."""
    step: int
    load_fraction: float
    converged: bool
    nr_iterations: int
    residual_final: float


@dataclass
class AnalysisResult:
    """Complete result of a multi-step nonlinear analysis."""
    u_history: list[np.ndarray]
    reaction_history: list[float]
    load_fractions: list[float]
    step_info: list[StepInfo]
    completed: bool  # True if all steps converged


# ====================================================================
# BASIC NEWTON-RAPHSON
# ====================================================================

class NewtonRaphson:
    """
    Full Newton-Raphson iteration.

    Solves R(u) = 0 where R is the residual vector.

    Parameters
    ----------
    residual_fn : callable
        R(u) -> np.ndarray — evaluates the residual vector
    tangent_fn : callable
        K(u) -> sparse matrix — evaluates the tangent (Jacobian) matrix
    linear_solver : LinearSolverBase, optional
        Solver for the linearised system
    tol : float
        Relative residual norm tolerance
    max_iter : int
        Maximum Newton iterations
    """

    def __init__(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        tangent_fn: Callable[[np.ndarray], sparse.spmatrix],
        linear_solver: LinearSolverBase | None = None,
        tol: float = 1e-8,
        max_iter: int = 50,
    ):
        self.residual = residual_fn
        self.tangent = tangent_fn
        self.linear_solver = linear_solver or DirectSolver()
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, u0: np.ndarray) -> tuple[np.ndarray, ConvergenceInfo]:
        u = u0.copy()
        norms: list[float] = []

        for it in range(self.max_iter):
            R = self.residual(u)
            norm_R = np.linalg.norm(R)
            norms.append(norm_R)
            logger.info("Newton iter %d: ||R|| = %.6e", it, norm_R)

            if norm_R < self.tol:
                return u, ConvergenceInfo(
                    converged=True, iterations=it, residual_norms=norms
                )

            K = self.tangent(u)
            du = self.linear_solver.solve(K, -R)
            u += du

        logger.warning(
            "Newton-Raphson did not converge after %d iterations", self.max_iter
        )
        return u, ConvergenceInfo(
            converged=False, iterations=self.max_iter, residual_norms=norms
        )


# ====================================================================
# DISPLACEMENT CONTROL
# ====================================================================

class DisplacementControl:
    """
    Incremental displacement control with Newton-Raphson.

    Incrementally applies prescribed displacements and solves for
    equilibrium at each step. Suitable for nonlinear problems with
    displacement-driven loading (e.g., prescribed boundary motion).

    Parameters
    ----------
    internal_force_fn : callable
        f_int(u) -> np.ndarray — internal force vector
    tangent_fn : callable
        K_T(u) -> sparse matrix — tangent stiffness matrix
    n_dof : int
        Total number of DOFs
    free_dofs : array-like
        Indices of free (unconstrained) DOFs
    prescribed_dofs : dict
        {dof_index: total_prescribed_value} for DOFs with nonzero
        prescribed displacement
    fixed_dofs : array-like
        Indices of DOFs fixed at zero (e.g., clamped boundary)
    n_steps : int
        Number of load increments
    tol : float
        Relative residual tolerance for NR convergence
    max_iter : int
        Maximum NR iterations per step
    line_search : bool
        Enable line search with backtracking
    line_search_check : callable, optional
        f(u) -> bool — additional check during line search
        (e.g., verify det(F) > 0)
    """

    def __init__(
        self,
        internal_force_fn: Callable[[np.ndarray], np.ndarray],
        tangent_fn: Callable[[np.ndarray], sparse.spmatrix],
        n_dof: int,
        free_dofs: np.ndarray,
        prescribed_dofs: dict,
        fixed_dofs: np.ndarray,
        n_steps: int = 20,
        tol: float = 1e-8,
        max_iter: int = 30,
        line_search: bool = True,
        line_search_check: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.f_int = internal_force_fn
        self.K_T = tangent_fn
        self.n_dof = n_dof
        self.free_dofs = np.asarray(free_dofs)
        self.prescribed_dofs = prescribed_dofs
        self.fixed_dofs = np.asarray(fixed_dofs)
        self.n_steps = n_steps
        self.tol = tol
        self.max_iter = max_iter
        self.line_search = line_search
        self.line_search_check = line_search_check

        # Precompute DOF sets
        self.prescribed_dof_ids = np.array(sorted(prescribed_dofs.keys()))
        self.all_constrained = np.union1d(self.fixed_dofs, self.prescribed_dof_ids)

        # Incremental prescribed values
        self.du_prescribed = {
            dof: val / n_steps for dof, val in prescribed_dofs.items()
        }

        logger.info(
            f"DisplacementControl: n_dof={n_dof}, "
            f"free={len(self.free_dofs)}, "
            f"prescribed={len(self.prescribed_dof_ids)}, "
            f"fixed={len(self.fixed_dofs)}, "
            f"n_steps={n_steps}"
        )

    def solve(self, u0: Optional[np.ndarray] = None) -> AnalysisResult:
        """
        Run the incremental analysis.

        Returns:
            AnalysisResult with displacement history, reactions, etc.
        """
        if u0 is None:
            u0 = np.zeros(self.n_dof)

        u = u0.copy()
        u_history = [u.copy()]
        reaction_history = [0.0]
        load_fractions = [0.0]
        step_info_list = []

        for step in range(1, self.n_steps + 1):
            frac = step / self.n_steps
            logger.info(f"Load step {step}/{self.n_steps} (frac={frac:.3f})")

            # Apply displacement increment (with linearised predictor)
            self._predictor(u)

            # Newton-Raphson
            converged, nr_iters, res_final = self._newton_step(u)

            si = StepInfo(
                step=step,
                load_fraction=frac,
                converged=converged,
                nr_iterations=nr_iters,
                residual_final=res_final,
            )
            step_info_list.append(si)

            if not converged:
                logger.warning(f"Step {step} did not converge. Stopping.")
                return AnalysisResult(
                    u_history=u_history,
                    reaction_history=reaction_history,
                    load_fractions=load_fractions,
                    step_info=step_info_list,
                    completed=False,
                )

            # Compute reaction forces
            f_int = self.f_int(u)
            rx = sum(f_int[dof] for dof in self.prescribed_dof_ids)

            u_history.append(u.copy())
            reaction_history.append(rx)
            load_fractions.append(frac)

            logger.info(
                f"  Step {step}: converged in {nr_iters} iters, "
                f"reaction={rx:.4e}"
            )

        return AnalysisResult(
            u_history=u_history,
            reaction_history=reaction_history,
            load_fractions=load_fractions,
            step_info=step_info_list,
            completed=True,
        )
    
    def _predictor(self, u: np.ndarray) -> None:
        """
        Linearised (elastic) predictor for one displacement increment.
        """
        pid = self.prescribed_dof_ids
        if len(pid) == 0:
            return

        du_p = np.array([self.du_prescribed[d] for d in pid])

        K_dense = self.K_T(u).toarray()
        K_ff = K_dense[np.ix_(self.free_dofs, self.free_dofs)]
        rhs = -K_dense[np.ix_(self.free_dofs, pid)] @ du_p

        u[pid] += du_p
        try:
            u[self.free_dofs] += np.linalg.solve(K_ff, rhs)
        except np.linalg.LinAlgError:
            logger.warning("Predictor: singular tangent — continuing without it")

    def _newton_step(self, u: np.ndarray) -> Tuple[bool, int, float]:
        """
        Run NR iterations for the current load step.

        Modifies u in-place for the free DOFs.

        Returns:
            (converged, n_iterations, final_residual_norm)
        """
        res_norm_0 = None

        for nr_iter in range(self.max_iter):
            # Internal force and residual
            f_int = self.f_int(u)
            R = f_int.copy()

            # Zero residual at constrained DOFs
            R[self.all_constrained] = 0.0

            # Convergence check
            res_norm = np.linalg.norm(R[self.free_dofs])
            if nr_iter == 0:
                res_norm_0 = max(res_norm, 1e-15)
            scale = max(np.linalg.norm(f_int), 1e-15)

            rel_res = res_norm / res_norm_0

            logger.info(
                f"    NR iter {nr_iter+1}: |R| = {res_norm:.4e}, "
                f"rel = {rel_res:.4e}, |R|/|f| = {res_norm/scale:.4e}"
            )

            if nr_iter > 0 and (rel_res < self.tol or res_norm / scale < self.tol):
                logger.info(f"    Converged in {nr_iter+1} iterations")
                return True, nr_iter + 1, res_norm

            # Tangent and solve
            K = self.K_T(u)
            K_dense = K.toarray()
            K_ff = K_dense[np.ix_(self.free_dofs, self.free_dofs)]
            R_f = R[self.free_dofs]

            try:
                du = np.linalg.solve(K_ff, -R_f)
            except np.linalg.LinAlgError:
                logger.warning("    Singular tangent matrix!")
                return False, nr_iter + 1, res_norm

            # Apply correction with optional line search
            if self.line_search:
                alpha = self._line_search(u, du)
            else:
                alpha = 1.0

            u[self.free_dofs] += alpha * du

        return False, self.max_iter, res_norm

    def _line_search(self, u: np.ndarray, du: np.ndarray) -> float:
        """
        Backtracking line search.

        Reduces step size if the line_search_check fails
        (e.g., det(F) <= 0 at some quadrature point).
        """
        alpha = 1.0
        for ls_iter in range(10):
            u_trial = u.copy()
            u_trial[self.free_dofs] += alpha * du

            if self.line_search_check is not None:
                if self.line_search_check(u_trial):
                    return alpha
                alpha *= 0.5
                logger.info(f"      Line search: alpha = {alpha:.4f}")
            else:
                return alpha

        logger.warning(f"      Line search failed, using alpha = {alpha:.4f}")
        return alpha


# ====================================================================
# LOAD CONTROL
# ====================================================================

class LoadControl:
    """
    Incremental load control with Newton-Raphson.

    Incrementally applies a fraction of the external force vector
    and solves for equilibrium at each step.

    Parameters
    ----------
    internal_force_fn : callable
        f_int(u) -> np.ndarray — internal force vector
    tangent_fn : callable
        K_T(u) -> sparse matrix — tangent stiffness matrix
    f_ext : np.ndarray
        Total external force vector at full load
    n_dof : int
        Total number of DOFs
    constrained_dofs : array-like
        Indices of DOFs fixed at zero
    n_steps : int
        Number of load increments
    tol : float
        Relative residual tolerance
    max_iter : int
        Maximum NR iterations per step
    line_search : bool
        Enable line search
    line_search_check : callable, optional
        f(u) -> bool — validity check during line search
    """

    def __init__(
        self,
        internal_force_fn: Callable[[np.ndarray], np.ndarray],
        tangent_fn: Callable[[np.ndarray], sparse.spmatrix],
        f_ext: np.ndarray,
        n_dof: int,
        constrained_dofs: np.ndarray,
        n_steps: int = 20,
        tol: float = 1e-8,
        max_iter: int = 30,
        line_search: bool = True,
        line_search_check: Optional[Callable[[np.ndarray], bool]] = None,
    ):
        self.f_int = internal_force_fn
        self.K_T = tangent_fn
        self.f_ext_total = f_ext
        self.n_dof = n_dof
        self.constrained_dofs = np.asarray(constrained_dofs)
        self.free_dofs = np.array(
            [d for d in range(n_dof) if d not in self.constrained_dofs]
        )
        self.n_steps = n_steps
        self.tol = tol
        self.max_iter = max_iter
        self.line_search = line_search
        self.line_search_check = line_search_check

        logger.info(
            f"LoadControl: n_dof={n_dof}, "
            f"free={len(self.free_dofs)}, "
            f"constrained={len(self.constrained_dofs)}, "
            f"n_steps={n_steps}"
        )

    def solve(self, u0: Optional[np.ndarray] = None) -> AnalysisResult:
        """Run the incremental analysis."""
        if u0 is None:
            u0 = np.zeros(self.n_dof)

        u = u0.copy()
        u_history = [u.copy()]
        reaction_history = [0.0]
        load_fractions = [0.0]
        step_info_list = []

        df_ext = self.f_ext_total / self.n_steps

        for step in range(1, self.n_steps + 1):
            frac = step / self.n_steps
            f_ext_step = frac * self.f_ext_total

            logger.info(f"Load step {step}/{self.n_steps} (frac={frac:.3f})")

            self._predictor(u, df_ext)
            
            converged, nr_iters, res_final = self._newton_step(u, f_ext_step)

            si = StepInfo(
                step=step,
                load_fraction=frac,
                converged=converged,
                nr_iterations=nr_iters,
                residual_final=res_final,
            )
            step_info_list.append(si)

            if not converged:
                logger.warning(f"Step {step} did not converge. Stopping.")
                return AnalysisResult(
                    u_history=u_history,
                    reaction_history=reaction_history,
                    load_fractions=load_fractions,
                    step_info=step_info_list,
                    completed=False,
                )

            # Reaction: f_int at constrained DOFs
            f_int = self.f_int(u)
            rx = sum(f_int[dof] for dof in self.constrained_dofs)

            u_history.append(u.copy())
            reaction_history.append(rx)
            load_fractions.append(frac)

        return AnalysisResult(
            u_history=u_history,
            reaction_history=reaction_history,
            load_fractions=load_fractions,
            step_info=step_info_list,
            completed=True,
        )

    def _predictor(self, u: np.ndarray, df_ext: np.ndarray) -> None:
        """Linearised predictor: K_ff du_f = df_ext_f at the last converged state."""
        K_ff = self.K_T(u).toarray()[np.ix_(self.free_dofs, self.free_dofs)]
        try:
            u[self.free_dofs] += np.linalg.solve(K_ff, df_ext[self.free_dofs])
        except np.linalg.LinAlgError:
            logger.warning("Predictor: singular tangent — continuing without it")

    def _newton_step(
        self, u: np.ndarray, f_ext: np.ndarray
    ) -> Tuple[bool, int, float]:
        """NR iterations for one load step."""
        res_norm_0 = None

        for nr_iter in range(self.max_iter):
            f_int = self.f_int(u)
            R = f_int - f_ext
            R[self.constrained_dofs] = 0.0

            res_norm = np.linalg.norm(R[self.free_dofs])
            if nr_iter == 0:
                res_norm_0 = max(res_norm, 1e-15)

            rel_res = res_norm / res_norm_0

            logger.info(
                f"    NR iter {nr_iter+1}: |R| = {res_norm:.4e}, "
                f"rel = {rel_res:.4e}"
            )

            if rel_res < self.tol and nr_iter > 0:
                return True, nr_iter + 1, res_norm

            K = self.K_T(u)
            K_dense = K.toarray()
            K_ff = K_dense[np.ix_(self.free_dofs, self.free_dofs)]
            R_f = R[self.free_dofs]

            try:
                du = np.linalg.solve(K_ff, -R_f)
            except np.linalg.LinAlgError:
                logger.warning("    Singular tangent matrix!")
                return False, nr_iter + 1, res_norm

            # Line search
            alpha = 1.0
            if self.line_search and self.line_search_check is not None:
                for _ in range(10):
                    u_trial = u.copy()
                    u_trial[self.free_dofs] += alpha * du
                    if self.line_search_check(u_trial):
                        break
                    alpha *= 0.5

            u[self.free_dofs] += alpha * du

        return False, self.max_iter, res_norm