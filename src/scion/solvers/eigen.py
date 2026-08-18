"""
Eigenvalue solver for structural dynamics.

Solves the generalised eigenvalue problem:
    K φ = ω² M φ

Returns natural frequencies and mode shapes.

Usage:
    from scion.solvers.eigen import solve_eigenvalue_problem

    freqs, modes = solve_eigenvalue_problem(K, M, n_modes=10)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Optional, Tuple

from scion.core.logging import get_logger

logger = get_logger(__name__)


def solve_eigenvalue_problem(
    K: sparse.csr_matrix,
    M: sparse.csr_matrix,
    n_modes: int = 10,
    sigma: float = 0.0,
    which: str = "SM",
    maxiter: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:

    n_dof = K.shape[0]
    n_modes = min(n_modes, n_dof - 2)

    logger.info(f"Eigenvalue solve: n_dof={n_dof}, n_modes={n_modes}")

    try:
        eigenvalues, eigenvectors = eigsh(
            K, k=n_modes, M=M, sigma=sigma, which=which,
            maxiter=maxiter,
        )
    except Exception as e:
        logger.warning(f"eigsh with sigma={sigma} failed: {e}")
        logger.info("Retrying with sigma=1.0 ...")
        eigenvalues, eigenvectors = eigsh(
            K, k=n_modes, M=M, sigma=1.0, which="LM",
            maxiter=maxiter,
        )

    # Sort by ascending eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Convert to frequencies
    omega = np.sqrt(np.maximum(eigenvalues, 0.0))
    frequencies = omega / (2.0 * np.pi)

    logger.info(f"First {min(5, n_modes)} frequencies (Hz): "
                f"{frequencies[:5]}")

    return frequencies, eigenvectors


def compute_rayleigh_damping(
    M: sparse.csr_matrix,
    K: sparse.csr_matrix,
    omega1: float,
    omega2: float,
    zeta1: float,
    zeta2: float,
) -> Tuple[sparse.csr_matrix, float, float]:
    """
    Compute Rayleigh damping matrix C = a*M + b*K.

    The coefficients a and b are determined from two target frequencies
    and their corresponding damping ratios:

        ζ_i = a/(2ω_i) + b*ω_i/2

    This gives a 2×2 system:
        [1/(2ω₁)  ω₁/2] [a]   [ζ₁]
        [1/(2ω₂)  ω₂/2] [b] = [ζ₂]

    Args:
        M: Mass matrix
        K: Stiffness matrix
        omega1, omega2: Two circular frequencies (rad/s)
        zeta1, zeta2: Damping ratios at those frequencies

    Returns:
        C: Damping matrix
        a: Mass-proportional coefficient
        b: Stiffness-proportional coefficient
    """
    # Solve 2×2 system for a, b
    A = np.array([
        [1.0 / (2.0 * omega1), omega1 / 2.0],
        [1.0 / (2.0 * omega2), omega2 / 2.0],
    ])
    rhs = np.array([zeta1, zeta2])
    coeffs = np.linalg.solve(A, rhs)
    a, b = coeffs[0], coeffs[1]

    C = a * M + b * K

    logger.info(f"Rayleigh damping: a={a:.6e}, b={b:.6e}")
    logger.info(f"  ω₁={omega1:.2f} rad/s (ζ₁={zeta1:.4f}), "
                f"ω₂={omega2:.2f} rad/s (ζ₂={zeta2:.4f})")

    return C, a, b
