"""
Optimal Experimental Design (OED).

Selects sensor locations, loading conditions, or measurement times
that maximise the information gained about uncertain model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from scion.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OEDResult:
    """Result of an OED problem."""
    design: np.ndarray          # optimal design variables
    criterion_value: float      # value of the optimality criterion
    n_fev: int


class OptimalExperimentalDesign:
    """Framework for optimal experimental design.

    Maximises an information criterion (A-, D-, or E-optimality) over
    the design space by evaluating the Fisher Information Matrix (FIM).

    Parameters
    ----------
    forward_model : callable
        ``(params, design) -> observables`` — the PDE forward model.
    sensitivity_fn : callable
        ``(params, design) -> jacobian`` — sensitivity of observables
        w.r.t. parameters. Shape ``(n_obs, n_params)``.
    prior_params : np.ndarray
        Nominal / prior-mean parameter values.
    noise_cov : np.ndarray
        Observation noise covariance matrix ``(n_obs, n_obs)``.
    criterion : str
        Optimality criterion: ``"D"`` (log-det FIM), ``"A"`` (trace of FIM inverse),
        ``"E"`` (smallest eigenvalue of FIM).
    """

    def __init__(
        self,
        forward_model: Callable,
        sensitivity_fn: Callable,
        prior_params: np.ndarray,
        noise_cov: np.ndarray,
        criterion: str = "D",
    ):
        self.forward_model = forward_model
        self.sensitivity_fn = sensitivity_fn
        self.prior_params = prior_params
        self.noise_cov = noise_cov
        self.criterion = criterion.upper()
        self._noise_cov_inv = np.linalg.inv(noise_cov)

    def fisher_information(self, design: np.ndarray) -> np.ndarray:
        """Compute the Fisher Information Matrix at a given design.

        FIM = J^T Σ^{-1} J

        where J is the sensitivity matrix and Σ is the noise covariance.
        """
        J = self.sensitivity_fn(self.prior_params, design)
        return J.T @ self._noise_cov_inv @ J

    def _criterion(self, design: np.ndarray) -> float:
        """Evaluate the optimality criterion (to be maximised)."""
        FIM = self.fisher_information(design)
        eigvals = np.linalg.eigvalsh(FIM)
        eigvals = np.maximum(eigvals, 1e-16)  # numerical safety

        if self.criterion == "D":
            return np.sum(np.log(eigvals))  # log-det
        elif self.criterion == "A":
            return -np.sum(1.0 / eigvals)  # negative trace of inverse
        elif self.criterion == "E":
            return eigvals[0]  # smallest eigenvalue
        raise ValueError(f"Unknown criterion '{self.criterion}'")

    def optimize(
        self,
        design0: np.ndarray,
        bounds: np.ndarray | None = None,
        method: str = "L-BFGS-B",
        maxiter: int = 200,
    ) -> OEDResult:
        """Find the optimal design.

        Parameters
        ----------
        design0 : initial design variables
        bounds : (n_design, 2) array or None
        method : SciPy optimisation method
        maxiter : max iterations
        """
        # Negate because we maximise the criterion but scipy minimises
        def neg_criterion(d):
            return -self._criterion(d)

        result = minimize(
            neg_criterion,
            design0,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
        )

        return OEDResult(
            design=result.x,
            criterion_value=-result.fun,
            n_fev=result.nfev,
        )

    def evaluate_designs(self, designs: np.ndarray) -> np.ndarray:
        """Evaluate the criterion at multiple candidate designs.

        Parameters
        ----------
        designs : (n_candidates, n_design_vars) array

        Returns
        -------
        criterion_values : (n_candidates,) array
        """
        return np.array([self._criterion(d) for d in designs])
