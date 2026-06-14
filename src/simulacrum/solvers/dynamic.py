"""
Time integration methods for structural dynamics.

Solves the equation of motion:
    M ü + C u̇ + K u = F(t)

Methods:
    - Newmark-β (implicit, unconditionally stable for β≥1/4, γ=1/2)
    - Central Difference (explicit, conditionally stable: Δt < 2/ω_max)

Usage:
    from simulacrum.solvers.dynamic import NewmarkBeta, CentralDifference

    integrator = NewmarkBeta(M, C, K, dt, beta=0.25, gamma=0.5)
    t_hist, u_hist, v_hist, a_hist = integrator.solve(
        force_fn, t_end, u0, v0, dof_out=[node_dof]
    )
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Callable, Optional, Tuple, List

from simulacrum.core.logging import get_logger

logger = get_logger(__name__)


class NewmarkBeta:
    """
    Newmark-β time integration (implicit).

    Standard parameters:
        β = 1/4, γ = 1/2  → average acceleration (unconditionally stable)
        β = 1/6, γ = 1/2  → linear acceleration (conditionally stable)

    Algorithm:
        1. Predict: ũ = u_n + Δt v_n + Δt²(0.5-β) a_n
                    ṽ = v_n + Δt(1-γ) a_n
        2. Solve:   K_eff a_{n+1} = F_{n+1} - C ṽ - K ũ
                    where K_eff = M + γ Δt C + β Δt² K
        3. Correct: u_{n+1} = ũ + β Δt² a_{n+1}
                    v_{n+1} = ṽ + γ Δt a_{n+1}
    """

    def __init__(
        self,
        M: sparse.csr_matrix,
        C: sparse.csr_matrix,
        K: sparse.csr_matrix,
        dt: float,
        beta: float = 0.25,
        gamma: float = 0.5,
    ):
        self.M = M
        self.C = C
        self.K = K
        self.dt = dt
        self.beta = beta
        self.gamma = gamma
        self.n_dof = M.shape[0]

        # Effective stiffness (factored once, reused every step)
        self.K_eff = M + gamma * dt * C + beta * dt**2 * K
        self.K_eff = self.K_eff.tocsc()

        logger.info(f"Newmark-β: dt={dt}, beta={beta}, gamma={gamma}, "
                    f"n_dof={self.n_dof}")

    def solve(
        self,
        force_fn: Callable[[float], np.ndarray],
        t_end: float,
        u0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        dof_out: Optional[List[int]] = None,
        store_every: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run time integration.

        Args:
            force_fn: Function f(t) → F (n_dof,) returning the force vector
            t_end: End time
            u0: Initial displacement (default: zero)
            v0: Initial velocity (default: zero)
            dof_out: List of DOF indices to store (default: all)
            store_every: Store results every N steps

        Returns:
            t_hist: (n_stored,) time values
            u_hist: (n_stored, n_out) displacement history
            v_hist: (n_stored, n_out) velocity history
            a_hist: (n_stored, n_out) acceleration history
        """
        dt = self.dt
        n_steps = int(np.ceil(t_end / dt))

        if u0 is None:
            u0 = np.zeros(self.n_dof)
        if v0 is None:
            v0 = np.zeros(self.n_dof)

        if dof_out is None:
            dof_out = list(range(self.n_dof))

        # Initial acceleration: M a0 = F(0) - C v0 - K u0
        F0 = force_fn(0.0)
        rhs0 = F0 - self.C @ v0 - self.K @ u0
        a0 = spsolve(self.M.tocsc(), rhs0)

        # Storage
        n_out = len(dof_out)
        n_stored = n_steps // store_every + 1
        t_hist = np.zeros(n_stored)
        u_hist = np.zeros((n_stored, n_out))
        v_hist = np.zeros((n_stored, n_out))
        a_hist = np.zeros((n_stored, n_out))

        # Initial values
        t_hist[0] = 0.0
        u_hist[0] = u0[dof_out]
        v_hist[0] = v0[dof_out]
        a_hist[0] = a0[dof_out]

        u_n, v_n, a_n = u0.copy(), v0.copy(), a0.copy()
        store_idx = 1

        logger.info(f"Newmark-β: {n_steps} steps, t_end={t_end}")

        for step in range(1, n_steps + 1):
            t = step * dt

            # Predict
            u_pred = u_n + dt * v_n + dt**2 * (0.5 - self.beta) * a_n
            v_pred = v_n + dt * (1.0 - self.gamma) * a_n

            # Effective force
            F = force_fn(t)
            rhs = F - self.C @ v_pred - self.K @ u_pred

            # Solve for acceleration
            a_new = spsolve(self.K_eff, rhs)

            # Correct
            u_new = u_pred + self.beta * dt**2 * a_new
            v_new = v_pred + self.gamma * dt * a_new

            # Store
            if step % store_every == 0 and store_idx < n_stored:
                t_hist[store_idx] = t
                u_hist[store_idx] = u_new[dof_out]
                v_hist[store_idx] = v_new[dof_out]
                a_hist[store_idx] = a_new[dof_out]
                store_idx += 1

            u_n, v_n, a_n = u_new, v_new, a_new

        t_hist = t_hist[:store_idx]
        u_hist = u_hist[:store_idx]
        v_hist = v_hist[:store_idx]
        a_hist = a_hist[:store_idx]

        logger.info(f"Newmark-β complete: {store_idx} stored steps")

        return t_hist, u_hist, v_hist, a_hist


class CentralDifference:
    """
    Central Difference time integration (explicit).

    Classical displacement-based formulation:
        M_eff * u_{n+1} = Δt² F_n + (2M - Δt²K) u_n - M_eff_minus * u_{n-1}

    where:
        M_eff      = M + (Δt/2) C
        M_eff_minus = M - (Δt/2) C

    Conditionally stable: Δt < 2/ω_max
    """

    def __init__(self, M, C, K, dt, constrained_dofs=None):
        self.M = M
        self.C = C
        self.K = K
        self.dt = dt
        self.n_dof = M.shape[0]
        self.constrained_dofs = np.array(
            constrained_dofs if constrained_dofs is not None else [], dtype=int
        )

        # Effective mass matrices
        self.M_eff = M + (dt / 2.0) * C          # M + Δt/2 C
        self.M_eff_minus = M - (dt / 2.0) * C    # M - Δt/2 C
        self.A_mid = 2.0 * M - dt**2 * K          # 2M - Δt²K

        # Check if M_eff is diagonal (lumped mass + proportional damping)
        self.M_eff_diag = self.M_eff.diagonal()
        self.is_lumped = (M.nnz <= M.shape[0] * 1.1)  # roughly diagonal

        logger.info(f"Central Difference: dt={dt}, n_dof={self.n_dof}, "
                    f"lumped={self.is_lumped}")

    def critical_timestep(self, K, M):
        """Estimate critical time step: Δt_cr = 2/ω_max."""
        from scipy.sparse.linalg import eigsh
        try:
            lambda_max = eigsh(K, k=1, M=M, which="LM",
                               return_eigenvectors=False, maxiter=10000)[0]
            omega_max = np.sqrt(abs(lambda_max))
            return 2.0 / omega_max
        except Exception:
            logger.warning("Could not estimate critical timestep")
            return self.dt

    def solve(
        self,
        force_fn,
        t_end,
        u0=None,
        v0=None,
        dof_out=None,
        store_every=1,
    ):
        dt = self.dt
        n_steps = int(np.ceil(t_end / dt))

        if u0 is None:
            u0 = np.zeros(self.n_dof)
        if v0 is None:
            v0 = np.zeros(self.n_dof)
        if dof_out is None:
            dof_out = list(range(self.n_dof))

        # Initial acceleration: M a0 = F(0) - C v0 - K u0
        F0 = force_fn(0.0)
        rhs0 = F0 - self.C @ v0 - self.K @ u0
        a0 = self._solve_mass_raw(self.M, rhs0)

        # Bootstrap: u_{-1} = u_0 - Δt v_0 + (Δt²/2) a_0
        u_prev = u0 - dt * v0 + 0.5 * dt**2 * a0
        u_curr = u0.copy()

        # Enforce BCs on initial conditions
        if len(self.constrained_dofs) > 0:
            u_prev[self.constrained_dofs] = 0.0
            u_curr[self.constrained_dofs] = 0.0

        # Storage
        n_out = len(dof_out)
        n_stored = n_steps // store_every + 1
        t_hist = np.zeros(n_stored)
        u_hist = np.zeros((n_stored, n_out))
        v_hist = np.zeros((n_stored, n_out))
        a_hist = np.zeros((n_stored, n_out))

        t_hist[0] = 0.0
        u_hist[0] = u0[dof_out]
        v_hist[0] = v0[dof_out]
        a_hist[0] = a0[dof_out]

        store_idx = 1

        logger.info(f"Central Difference: {n_steps} steps, t_end={t_end}")

        for step in range(1, n_steps + 1):
            t = step * dt
            F = force_fn(t - dt)  # F at t_n (not t_{n+1})

            # rhs = Δt² F_n + (2M - Δt²K) u_n - (M - Δt/2 C) u_{n-1}
            rhs = dt**2 * F + self.A_mid @ u_curr - self.M_eff_minus @ u_prev

            # Solve: (M + Δt/2 C) u_{n+1} = rhs
            u_next = self._solve_M_eff(rhs)

            # Enforce BCs
            if len(self.constrained_dofs) > 0:
                u_next[self.constrained_dofs] = 0.0

            # Velocity and acceleration (for output only)
            if step % store_every == 0 and store_idx < n_stored:
                v_n = (u_next - u_prev) / (2.0 * dt)
                a_n = (u_next - 2.0 * u_curr + u_prev) / dt**2

                if len(self.constrained_dofs) > 0:
                    v_n[self.constrained_dofs] = 0.0
                    a_n[self.constrained_dofs] = 0.0

                t_hist[store_idx] = t
                u_hist[store_idx] = u_next[dof_out]
                v_hist[store_idx] = v_n[dof_out]
                a_hist[store_idx] = a_n[dof_out]
                store_idx += 1

            # Advance
            u_prev = u_curr
            u_curr = u_next

        t_hist = t_hist[:store_idx]
        u_hist = u_hist[:store_idx]
        v_hist = v_hist[:store_idx]
        a_hist = a_hist[:store_idx]

        logger.info(f"Central Difference complete: {store_idx} stored steps")
        return t_hist, u_hist, v_hist, a_hist

    def _solve_M_eff(self, rhs):
        """Solve M_eff * x = rhs."""
        return spsolve(self.M_eff.tocsc(), rhs)

    def _solve_mass_raw(self, M, rhs):
        """Solve M * x = rhs (for initial acceleration)."""
        diag = M.diagonal()
        if M.nnz <= M.shape[0] * 1.1:
            return rhs / diag
        else:
            return spsolve(M.tocsc(), rhs)
