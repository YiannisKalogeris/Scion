r"""
The two-level operator that couples a coarse space to a baseline (Section 3.2).

Given a coarse basis ``Z`` and any baseline ``B^-1``, the preconditioner is

    T^-1 = P' B^-1 P + Z E^-1 Z',        E = Z' K Z,   P = I - K Z E^-1 Z'.

Two terms doing two different jobs. The second is an *exact* solve on
``range(Z)`` -- a Galerkin coarse correction. The first applies the baseline to
the complement, with ``P`` deflating the coarse content out first so that the
baseline is not asked to re-smooth components already handled exactly. That
separation is the point: without the projector the two terms fight, and the
coarse correction is repeatedly undone.

Why this particular form
------------------------
``T^-1`` as written is **symmetric** whenever ``B^-1`` is, because ``P' B^-1 P``
is a congruence and ``Z E^-1 Z'`` is symmetric outright. That is not decoration:
PCG requires it, and the obvious cheaper alternatives are not symmetric. Using
``P B^-1 P`` or the one-sided ``B^-1 P`` gives an operator that works as a
standalone iteration and breaks PCG's use of ``r'z`` as a squared norm.

``E`` is SPD provided ``Z`` has full column rank, which the truncated SVD in
:mod:`diffusion` guarantees by construction -- it is the reason the rank is
truncated to the numerical rank of the snapshot matrix rather than to whatever
was requested. A small Tikhonov term is still added, because a coarse space
built from snapshots at large diffusion times can be numerically close to
rank-deficient even after truncation.

The regularization is not free, and :meth:`TwoLevelPreconditioner.
deflation_residual` is how you see its price. The identity ``Z' K P' = 0`` holds
exactly for ``E = Z'KZ``, and adding ``lambda I`` perturbs it by
``I - E E_reg^-1``. Measured on a contrast-1e6 diffusion problem with an IC(0)
baseline and rank 40:

    strategy   cond(E)   residual at lambda=0   at lambda=1e-12
    A               88               5.0e-12            2.6e-10
    B              456               8.5e-08            9.4e-06

So read that residual against the regularization rather than against zero. None
of these values harms convergence -- both strategies cut IC(0) from 84
iterations to 25 -- but a residual far larger than ``lambda`` times
``cond(E)`` means the basis and ``KZ`` disagree, which is the failure mode the
Strategy A coordinate mapping below can cause.

Cost per application
--------------------
One baseline application, plus: two products with the dense ``n x r`` blocks
``Z`` and ``KZ``, one sparse product with ``K``, and two ``r x r`` triangular
solves. ``KZ`` and the Cholesky factor of ``E`` are formed once at setup. So the
overhead is ``O(n r)`` per solve, which is why ``r`` in the tens is the useful
range -- and why a cheap baseline like Jacobi can end up *slower* per iteration
than it was unpreconditioned, even while taking far fewer iterations.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg
from scipy import sparse

from simulacrum.solvers.preconditioners import Preconditioner

__all__ = ["TwoLevelPreconditioner"]


class TwoLevelPreconditioner(Preconditioner):
    r"""``T^-1 = P' B^-1 P + Z E^-1 Z'`` around any baseline.

    Parameters
    ----------
    coarse : a :class:`~diffusion.CoarseSpace`, or a raw ``(n, r)`` array. When
        a ``CoarseSpace`` from Strategy A is given, its ``D^-1/2`` scaling is
        applied automatically -- that basis lives in scaled coordinates and
        using it directly in K-space is the easiest way to get a subtly wrong
        answer that still converges.
    baseline : the ``B^-1`` to wrap. Any :class:`~simulacrum.solvers.
        preconditioners.Preconditioner`, or a callable.
    regularization : Tikhonov term added to ``E``, relative to its trace.

    Notes
    -----
    ``is_spd`` is inherited from the baseline: the two-level wrapper preserves
    symmetry but cannot create it. Wrapping ILU gives a non-SPD two-level
    operator, which is what the paper means by needing a flexible Krylov method
    there.
    """

    name = "two-level"

    def __init__(self, coarse, baseline, regularization: float = 1e-12):
        super().__init__()
        self.baseline = baseline
        self.regularization = float(regularization)

        Z = getattr(coarse, "Z", coarse)
        scaling = getattr(coarse, "scaling", None)
        self.strategy = getattr(coarse, "strategy", "?")
        # Strategy A builds its basis on S = D^-1/2 K D^-1/2, so it must be
        # mapped back before the Galerkin operator is formed in K-space (Eq. 52)
        self._Z = np.asfortranarray(
            Z if scaling is None else scaling[:, None] * np.asarray(Z))
        self.rank = int(self._Z.shape[1])

        self._apply_baseline = (baseline.apply if hasattr(baseline, "apply")
                                else baseline)
        self.is_spd = bool(getattr(baseline, "is_spd", True))

    # ------------------------------------------------------------------
    def _setup(self, A: sparse.csr_matrix) -> None:
        if self._Z.shape[0] != A.shape[0]:
            raise ValueError(
                f"the coarse basis has {self._Z.shape[0]} rows but the matrix "
                f"is {A.shape[0]}x{A.shape[0]}")

        if hasattr(self.baseline, "setup") and self.baseline.n_unknowns == 0:
            self.baseline.setup(A)

        self._A = A
        started = time.perf_counter()

        self._KZ = np.asfortranarray(A @ self._Z)          # Eq. 56
        E = self._Z.T @ self._KZ
        if self.regularization > 0.0:
            E = E + self.regularization * np.trace(E) / max(1, self.rank) \
                * np.eye(self.rank)

        try:
            self._chol = scipy.linalg.cho_factor(E, lower=True)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"the coarse operator E = Z'KZ is not positive definite, so the "
                f"basis is rank-deficient in K-space. Reduce the requested rank "
                f"or raise `regularization`. ({exc})") from exc

        self._E = E
        self.coarse_setup_time = time.perf_counter() - started
        self._Zt = np.ascontiguousarray(self._Z.T)

    # ------------------------------------------------------------------
    def apply(self, v: np.ndarray) -> np.ndarray:
        """Eqs. 57-60, in the order the paper gives them."""
        v = np.asarray(v, dtype=float).ravel()

        # 1. coarse solve
        c = scipy.linalg.cho_solve(self._chol, self._Zt @ v)
        coarse = self._Z @ c

        # 2. deflation: remove what the coarse solve already handled
        deflated = v - self._KZ @ c

        # 3. fine solve with the baseline
        fine = np.asarray(self._apply_baseline(deflated), dtype=float).ravel()

        # 4. symmetric projection, which is what keeps T^-1 symmetric
        correction = self._Z @ scipy.linalg.cho_solve(
            self._chol, self._Zt @ (self._A @ fine))
        return coarse + fine - correction

    # ------------------------------------------------------------------
    @property
    def coarse_condition(self) -> float:
        """``cond(E)``. A huge value means the basis is nearly rank-deficient in
        K-space and the coarse solve is amplifying round-off, which shows up as
        PCG stalling at a residual floor rather than diverging."""
        return float(np.linalg.cond(self._E))

    def deflation_residual(self, n_vectors: int = 5,
                           seed: int | None = 0) -> float:
        r"""``max ||Z' K P x|| / ||x||`` over random ``x``.

        The identity ``Z' K P = 0`` holds exactly, by construction: ``P``
        deflates precisely ``range(Z)`` in the ``K`` inner product. It is the
        sharpest available check that ``Z``, ``KZ`` and ``E`` are mutually
        consistent -- a mismatched scaling between them (the Strategy A trap)
        breaks it immediately while still leaving a preconditioner that
        converges.
        """
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_vectors):
            x = rng.standard_normal(self._Z.shape[0])
            c = scipy.linalg.cho_solve(self._chol, self._Zt @ (self._A @ x))
            Px = x - self._Z @ c
            worst = max(worst, float(np.linalg.norm(self._Zt @ (self._A @ Px))
                                     / np.linalg.norm(x)))
        return worst

    def __repr__(self) -> str:
        baseline = getattr(self.baseline, "name", "callable")
        return (f"TwoLevelPreconditioner(strategy {self.strategy}, rank "
                f"{self.rank}, baseline '{baseline}')")
