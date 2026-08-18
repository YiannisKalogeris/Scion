"""
Polynomial Chaos Expansion (PCE) for uncertainty propagation.

Represents a model output as a series expansion in orthogonal polynomials
of the random inputs. Enables cheap computation of statistical moments
once the expansion coefficients are determined.
"""

from __future__ import annotations

from typing import Callable
from itertools import product

import numpy as np
from numpy.polynomial import hermite_e, legendre


class PolynomialChaosExpansion:
    """Non-intrusive PCE via least-squares regression.

    Parameters
    ----------
    dim : int
        Number of input random variables.
    max_degree : int
        Maximum total polynomial degree.
    distribution : str
        Input distribution type: ``"uniform"`` (Legendre basis) or
        ``"gaussian"`` (Hermite basis).
    """

    def __init__(self, dim: int, max_degree: int = 3, distribution: str = "uniform"):
        self.dim = dim
        self.max_degree = max_degree
        self.distribution = distribution
        self.multi_indices = self._build_multi_indices()
        self.n_terms = len(self.multi_indices)
        self.coefficients: np.ndarray | None = None

    def _build_multi_indices(self) -> list[tuple[int, ...]]:
        """Build multi-index set with total degree truncation."""
        indices = []
        for combo in product(range(self.max_degree + 1), repeat=self.dim):
            if sum(combo) <= self.max_degree:
                indices.append(combo)
        return sorted(indices, key=sum)

    def _evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all basis polynomials at sample points.

        Parameters
        ----------
        x : (n_samples, dim) array in the standard domain

        Returns
        -------
        Phi : (n_samples, n_terms) design matrix
        """
        n_samples = x.shape[0]
        Phi = np.ones((n_samples, self.n_terms))

        for j, alpha in enumerate(self.multi_indices):
            for d in range(self.dim):
                if alpha[d] == 0:
                    continue
                if self.distribution == "uniform":
                    # Legendre polynomials on [-1, 1]
                    coeffs = np.zeros(alpha[d] + 1)
                    coeffs[-1] = 1.0
                    Phi[:, j] *= legendre.legval(x[:, d], coeffs)
                elif self.distribution == "gaussian":
                    # Probabilist's Hermite polynomials
                    coeffs = np.zeros(alpha[d] + 1)
                    coeffs[-1] = 1.0
                    Phi[:, j] *= hermite_e.hermeval(x[:, d], coeffs)
        return Phi

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Compute PCE coefficients via least-squares regression.

        Parameters
        ----------
        x : (n_samples, dim) input samples
        y : (n_samples,) model outputs
        """
        Phi = self._evaluate_basis(x)
        self.coefficients, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the PCE at new points."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() before predict()")
        Phi = self._evaluate_basis(x)
        return Phi @ self.coefficients

    @property
    def mean(self) -> float:
        """Mean of the PCE (first coefficient for orthonormal bases)."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() first")
        return float(self.coefficients[0])

    @property
    def variance(self) -> float:
        """Variance of the PCE (sum of squared coefficients, excluding mean)."""
        if self.coefficients is None:
            raise RuntimeError("Call fit() first")
        return float(np.sum(self.coefficients[1:] ** 2))

    def sobol_indices(self) -> dict[int, float]:
        """First-order Sobol sensitivity indices.

        Returns
        -------
        dict mapping dimension index → first-order Sobol index.
        """
        if self.coefficients is None:
            raise RuntimeError("Call fit() first")

        total_var = self.variance
        if total_var < 1e-16:
            return {d: 0.0 for d in range(self.dim)}

        indices = {}
        for d in range(self.dim):
            partial_var = 0.0
            for j, alpha in enumerate(self.multi_indices):
                # Terms that depend ONLY on dimension d
                if alpha[d] > 0 and sum(alpha) == alpha[d]:
                    partial_var += self.coefficients[j] ** 2
            indices[d] = partial_var / total_var
        return indices
