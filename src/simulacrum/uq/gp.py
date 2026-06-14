"""
Gaussian Process regression / emulation.

A GP emulator wraps expensive PDE solves: train on a small design of
experiments, then predict (with uncertainty) at untried parameter values.
Useful for Bayesian optimisation, sensitivity analysis, and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve


@dataclass
class GPPrediction:
    """GP prediction with uncertainty."""
    mean: np.ndarray
    std: np.ndarray
    variance: np.ndarray


class GaussianProcessEmulator:
    """Scalar-output GP with squared-exponential (RBF) kernel.

    Parameters
    ----------
    length_scale : float or array
        Kernel length scale(s). Scalar for isotropic, array for ARD.
    signal_variance : float
        Kernel signal variance (σ_f²).
    noise_variance : float
        Observation noise variance (σ_n²).
    """

    def __init__(
        self,
        length_scale: float | np.ndarray = 1.0,
        signal_variance: float = 1.0,
        noise_variance: float = 1e-6,
    ):
        self.length_scale = np.atleast_1d(length_scale).astype(np.float64)
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._L: np.ndarray | None = None  # Cholesky factor
        self._alpha: np.ndarray | None = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Squared-exponential (RBF) kernel with ARD."""
        X1_s = X1 / self.length_scale
        X2_s = X2 / self.length_scale
        sq_dist = cdist(X1_s, X2_s, metric="sqeuclidean")
        return self.signal_variance * np.exp(-0.5 * sq_dist)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the GP on observed data.

        Parameters
        ----------
        X : (n_train, dim) input locations
        y : (n_train,) observations
        """
        self._X_train = np.atleast_2d(X)
        self._y_train = y.ravel()

        K = self._kernel(self._X_train, self._X_train)
        K += self.noise_variance * np.eye(len(K))

        self._L, low = cho_factor(K, lower=True)
        self._alpha = cho_solve((self._L, low), self._y_train)

    def predict(self, X: np.ndarray) -> GPPrediction:
        """Predict at new locations with uncertainty.

        Parameters
        ----------
        X : (n_test, dim)

        Returns
        -------
        GPPrediction with mean, std, and variance arrays.
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict()")

        X = np.atleast_2d(X)
        K_star = self._kernel(X, self._X_train)
        K_ss = self._kernel(X, X)

        mean = K_star @ self._alpha
        v = cho_solve((self._L, True), K_star.T)
        variance = np.diag(K_ss) - np.sum(K_star * v.T, axis=1)
        variance = np.maximum(variance, 0)  # numerical safety

        return GPPrediction(mean=mean, std=np.sqrt(variance), variance=variance)

    def __call__(self, X: np.ndarray) -> GPPrediction:
        return self.predict(X)
