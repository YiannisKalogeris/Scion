r"""
Diffusion-filtered coarse spaces (Kalogeris, Sections 3.1.1-3.1.3).

The idea in one line: slow PCG convergence is driven by a low-dimensional set of
near-null modes, and the heat semigroup is a spectral low-pass filter that
isolates them. For an SPD operator ``S`` with eigenpairs ``(lambda_i, q_i)``,

    exp(-t S) q_i = exp(-t lambda_i) q_i,

so applying ``exp(-tS)`` to an arbitrary probe damps the large-eigenvalue
content exponentially and leaves the small-eigenvalue content nearly untouched.
Filter a handful of random probes at a few diffusion times, stack the responses,
and the leading left singular vectors of that stack span the slow modes. No
geometry, no mesh, no training -- only matrix-vector products.

Two strategies, differing in *which* operator is filtered
--------------------------------------------------------
**Strategy A** filters the Jacobi-scaled operator ``S = D^-1/2 K D^-1/2``, which
is SPD and has unit diagonal. The resulting basis targets low-energy modes of
``K`` itself, so it is independent of whatever preconditioner will be used --
"physics-driven". Baseline information enters only through *mixed probes*: half
the probes are passed through ``B^-1`` first, biasing the snapshots towards
components the baseline handles badly. Because the filtering happens on an SPD
operator, Strategy A works with a non-SPD baseline such as ILU.

**Strategy B** filters the preconditioned operator ``M = B^-1 K`` directly. That
is the operator PCG actually sees, so the basis is solver-aware by construction
and targets exactly the modes the baseline fails to damp. It requires ``B`` to be
SPD -- otherwise ``M`` has no real positive spectrum and ``exp(-tM)`` is not a
low-pass filter at all -- and each Krylov step costs one application of ``B^-1``,
which for AMG is the dominant expense.

Why the semigroup reuse matters
-------------------------------
The naive implementation builds one Krylov basis per (probe, time) pair, costing
``O(m p k)`` generator applications. But for sorted times,

    exp(-t_j M) = exp(-(t_j - t_{j-1}) M) exp(-t_{j-1} M),

so a *single* Krylov basis per probe suffices: step the small projected vector
forward through the time grid instead. That is ``O(m k)`` generator applications,
independent of the number of diffusion times ``p`` -- a factor of ``p`` saved on
the expensive part, which is the whole reason Strategy B is affordable with an
AMG baseline. Measured here on nos7 with 4 times: 4.0x fewer operator
applications, and identical snapshots to 1e-12.

The dense work is negligible by comparison: one ``k x k`` matrix exponential per
time, with ``k`` in the tens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
from scipy import sparse

__all__ = [
    "CoarseSpace",
    "generate_probes",
    "lanczos",
    "arnoldi",
    "semigroup_snapshots",
    "build_coarse_space",
    "DEFAULT_TIMES",
]

#: Diffusion times from the paper's experiments. Larger ``t`` pushes the
#: snapshots further towards the smallest-eigenvalue subspace; using several
#: improves coverage of the low end rather than concentrating on its very
#: bottom.
DEFAULT_TIMES = (32.0, 64.0, 128.0, 256.0)


@dataclass
class CoarseSpace:
    """An orthonormal basis for the slow modes, plus how it was obtained."""

    Z: np.ndarray                    #: ``(n, r)``, orthonormal, in K-coordinates
    strategy: str
    singular_values: np.ndarray = field(default_factory=lambda: np.empty(0))
    numerical_rank: int = 0
    n_operator_applications: int = 0
    setup_time: float = 0.0
    #: ``D^-1/2`` for Strategy A, ``None`` for Strategy B. Kept because the
    #: basis is built in scaled coordinates and must be mapped back.
    scaling: np.ndarray | None = None
    info: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.Z.shape[1])

    @property
    def spectral_decay(self) -> float:
        r"""``sigma_r / sigma_1``: how much genuine information the last kept
        basis vector carries.

        The signal that the requested rank is larger than the filter found, and
        it separates the two strategies sharply. Measured on a contrast-1e6
        diffusion problem at rank 40 with an IC(0) baseline: Strategy A gives
        ``9.7e-4``, Strategy B ``3.3e-7``. Strategy B's snapshots are far more
        redundant, which is the expected consequence of filtering the
        *preconditioned* operator -- the baseline has already removed most of
        the spectrum, so the slow subspace that remains is genuinely
        lower-dimensional. Read it as "Strategy B needs a smaller rank", not as
        a defect: on the same problem with an AMG baseline it truncated itself
        to 24 and still beat Strategy A at rank 40.
        """
        s = self.singular_values
        if s.size == 0 or s[0] == 0.0:
            return 0.0
        return float(s[min(self.rank, s.size) - 1] / s[0])

    def orthonormality_error(self) -> float:
        """``max |Z'Z - I|``. The SVD should make this round-off; a large value
        means the snapshot matrix was rank-deficient in a way the truncation
        missed."""
        gram = self.Z.T @ self.Z
        return float(np.abs(gram - np.eye(gram.shape[0])).max())

    def __repr__(self) -> str:
        return (f"CoarseSpace({self.strategy}, rank {self.rank}"
                f"/{self.numerical_rank}, "
                f"{self.n_operator_applications} operator applications, "
                f"{self.setup_time:.2f}s)")


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def generate_probes(n: int, n_probes: int, rng: np.random.Generator,
                    localized_fraction: float = 0.5) -> np.ndarray:
    """``(n_probes, n)`` probes: half Gaussian, half locally supported.

    Gaussian probes have, with high probability, a component along every
    eigenvector, so the filter sees the whole low end of the spectrum at once --
    the randomised-sketching argument. Locally supported probes are the
    complement: they carry high-frequency content that resolves modes localised
    in a small region, which a global random vector represents only weakly.
    Mixing them is what makes the coarse space cover both.
    """
    if n_probes < 1:
        raise ValueError("at least one probe is needed")
    n_local = int(round(localized_fraction * n_probes))

    probes = np.empty((n_probes, n))
    width = max(5, n // 300)
    for k in range(n_probes):
        if k < n_probes - n_local:
            g = rng.standard_normal(n)
        else:
            g = np.zeros(n)
            centre = int(rng.integers(0, n))
            lo, hi = max(0, centre - width), min(n, centre + width)
            g[lo:hi] = 1.0
        probes[k] = g / (np.linalg.norm(g) + 1e-30)
    return probes


# ---------------------------------------------------------------------------
# Krylov bases
# ---------------------------------------------------------------------------

def lanczos(apply_S, v: np.ndarray, k: int, reorthogonalize: bool = True):
    r"""Symmetric Lanczos: ``S V_k ~ V_k T_k`` with ``T_k`` tridiagonal.

    Full reorthogonalisation is on by default and is not optional in practice.
    The three-term recurrence loses orthogonality after a handful of steps once
    an eigenvalue has converged, and a basis that is not orthonormal makes
    ``V exp(-tT) e1`` a projection onto nothing in particular. It costs
    ``O(nk^2)``, which for ``k`` in the tens is far below the sparse matvecs.

    Returns ``(V, T, n_applications)``, with ``V`` possibly narrower than ``k``
    if an invariant subspace was reached -- which is a success, not a failure:
    the exponential is then exact on that subspace.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    V = np.zeros((n, k + 1), order="F")
    alpha = np.zeros(k)
    beta = np.zeros(k + 1)

    beta[0] = np.linalg.norm(v)
    if beta[0] < 1e-30:
        return np.zeros((n, 0)), np.zeros((0, 0)), 0
    V[:, 0] = v / beta[0]

    applications = 0
    for j in range(k):
        w = np.asarray(apply_S(V[:, j]), dtype=float).ravel()
        applications += 1
        if j > 0:
            w -= beta[j] * V[:, j - 1]
        alpha[j] = float(np.dot(V[:, j], w))
        w -= alpha[j] * V[:, j]

        if reorthogonalize:
            # twice: one pass leaves a component of size O(eps * ||w||_before)
            for _ in range(2):
                w -= V[:, :j + 1] @ (V[:, :j + 1].T @ w)

        beta[j + 1] = float(np.linalg.norm(w))
        if beta[j + 1] < 1e-12 * (1.0 + abs(alpha[j])):
            return (V[:, :j + 1].copy(),
                    _tridiagonal(alpha[:j + 1], beta[1:j + 1]), applications)
        V[:, j + 1] = w / beta[j + 1]

    return V[:, :k].copy(), _tridiagonal(alpha, beta[1:k]), applications


def _tridiagonal(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    T = np.diag(np.asarray(alpha, dtype=float))
    m = T.shape[0]
    if m > 1 and len(beta) >= m - 1:
        off = np.asarray(beta[:m - 1], dtype=float)
        T += np.diag(off, 1) + np.diag(off, -1)
    return T


def arnoldi(apply_M, v: np.ndarray, k: int):
    r"""Arnoldi: ``M V_k ~ V_k H_k`` with ``H_k`` upper Hessenberg.

    For Strategy B, where ``M = B^-1 K`` is self-adjoint in the ``B`` inner
    product but not in the Euclidean one, so Lanczos does not apply. Each step
    costs one ``B^-1`` application, which is the dominant expense when the
    baseline is AMG -- hence the semigroup reuse in
    :func:`semigroup_snapshots`.

    Modified Gram-Schmidt with a second pass, for the same reason Lanczos
    reorthogonalises.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    V = np.zeros((n, k + 1), order="F")
    H = np.zeros((k + 1, k))

    norm = np.linalg.norm(v)
    if norm < 1e-30:
        return np.zeros((n, 0)), np.zeros((0, 0)), 0
    V[:, 0] = v / norm

    applications = 0
    for j in range(k):
        w = np.asarray(apply_M(V[:, j]), dtype=float).ravel()
        applications += 1
        for i in range(j + 1):
            H[i, j] = float(np.dot(V[:, i], w))
            w -= H[i, j] * V[:, i]
        correction = V[:, :j + 1].T @ w          # second pass
        H[:j + 1, j] += correction
        w -= V[:, :j + 1] @ correction

        H[j + 1, j] = float(np.linalg.norm(w))
        if H[j + 1, j] < 1e-12 * (1.0 + abs(H[j, j])):
            return V[:, :j + 1].copy(), H[:j + 1, :j + 1].copy(), applications
        V[:, j + 1] = w / H[j + 1, j]

    return V[:, :k].copy(), H[:k, :k].copy(), applications


def semigroup_snapshots(V: np.ndarray, H: np.ndarray, times,
                        norm: float = 1.0) -> list:
    r"""Snapshots at every time from **one** Krylov basis (Eqs. 47-51).

    ``exp(-t_j H) e1`` is obtained by stepping the previous result forward by
    ``t_j - t_{j-1}``, which is legitimate because the semigroup composes. The
    saving is a factor of ``len(times)`` on generator applications -- the
    expensive part -- at the cost of one small dense exponential per time.
    """
    times = np.sort(np.asarray(times, dtype=float))
    m = H.shape[0]
    if m == 0:
        return [np.zeros(V.shape[0]) for _ in times]

    xi = np.zeros(m)
    xi[0] = 1.0
    snapshots, previous = [], 0.0
    for t in times:
        step = t - previous
        if step > 0.0:
            xi = scipy.linalg.expm(-step * H) @ xi
        snapshots.append(norm * (V @ xi))
        previous = t
    return snapshots


# ---------------------------------------------------------------------------
# The coarse space
# ---------------------------------------------------------------------------

def build_coarse_space(K, strategy: str = "A", rank: int = 60,
                       n_probes: int = 16, times=DEFAULT_TIMES,
                       krylov_dim: int = 50, apply_Binv=None,
                       raw_fraction: float = 0.5, svd_rcond: float = 1e-12,
                       seed: int | None = None) -> CoarseSpace:
    """Build the diffusion-filtered coarse space.

    Parameters
    ----------
    K : the SPD system matrix.
    strategy : ``"A"`` filters ``D^-1/2 K D^-1/2``; ``"B"`` filters ``B^-1 K``
        and therefore requires ``apply_Binv``.
    rank : requested basis size. Truncated automatically to the numerical rank
        of the snapshot matrix, so asking for more than the filter found is
        harmless rather than a source of near-singular coarse operators.
    n_probes, times, krylov_dim : the filter. Cost is
        ``n_probes * krylov_dim`` operator applications, independent of
        ``len(times)``.
    apply_Binv : the baseline ``B^-1``. Optional for Strategy A, where it
        supplies the mixed probes; required for Strategy B.
    raw_fraction : fraction of probes left unpreconditioned. ``1.0`` disables
        mixed probes entirely.

    Notes
    -----
    Strategy B needs ``B`` **SPD**. With a non-SPD baseline, ``B^-1 K`` has no
    real positive spectrum, ``exp(-tM)`` is not a low-pass filter, and the
    basis is meaningless rather than merely worse -- so it is refused. Use
    Strategy A there, whose mixed probes still inject baseline information while
    filtering an operator that is guaranteed SPD.
    """
    import time as _time

    strategy = strategy.upper()
    if strategy not in ("A", "B"):
        raise ValueError(f"strategy must be 'A' or 'B', got {strategy!r}")

    K = sparse.csr_matrix(K, dtype=float)
    n = K.shape[0]
    rng = np.random.default_rng(seed)
    times = np.sort(np.asarray(times, dtype=float))
    started = _time.perf_counter()

    if strategy == "B" and apply_Binv is None:
        raise ValueError(
            "Strategy B filters B^-1 K, so it needs apply_Binv. Strategy A "
            "works without a baseline, and with a non-SPD one.")

    probes = generate_probes(n, n_probes, rng)
    n_raw = max(0, min(n_probes, int(round(raw_fraction * n_probes))))

    scaling = None
    if strategy == "A":
        diagonal = K.diagonal().copy()
        positive = diagonal[diagonal > 0]
        diagonal[diagonal <= 0] = positive.min() if positive.size else 1.0
        d_inv_sqrt = 1.0 / np.sqrt(diagonal)
        d_sqrt = np.sqrt(diagonal)
        scaling = d_inv_sqrt

        def generator(x):
            return d_inv_sqrt * (K @ (d_inv_sqrt * x))
    else:
        def generator(x):
            return np.asarray(apply_Binv(K @ x), dtype=float).ravel()

    columns, applications = [], 0
    for index, g in enumerate(probes):
        mixed = apply_Binv is not None and index >= n_raw
        if strategy == "A":
            source = g
            if mixed:
                candidate = np.asarray(apply_Binv(g), dtype=float).ravel()
                if candidate.size == n and np.linalg.norm(candidate) > 1e-30:
                    source = candidate
            v = d_sqrt * source                     # into scaled coordinates
        else:
            v = g
            if mixed:
                candidate = np.asarray(apply_Binv(g), dtype=float).ravel()
                if candidate.size == n and np.linalg.norm(candidate) > 1e-30:
                    v = candidate

        v = v / (np.linalg.norm(v) + 1e-30)

        if strategy == "A":
            V, H, used = lanczos(generator, v, krylov_dim)
        else:
            V, H, used = arnoldi(generator, v, krylov_dim)
        applications += used

        columns.extend(semigroup_snapshots(V, H, times))

    Y = np.column_stack(columns)
    U, singular, _ = np.linalg.svd(Y, full_matrices=False)

    threshold = svd_rcond * singular[0] if singular[0] > 0 else 0.0
    numerical_rank = max(1, int(np.count_nonzero(singular > threshold)))
    effective = min(rank, numerical_rank, U.shape[1])

    return CoarseSpace(
        Z=np.ascontiguousarray(U[:, :effective]),
        strategy=strategy,
        singular_values=singular,
        numerical_rank=numerical_rank,
        n_operator_applications=applications,
        setup_time=_time.perf_counter() - started,
        scaling=scaling,
        info={"n_probes": n_probes, "times": times, "krylov_dim": krylov_dim,
              "n_snapshots": Y.shape[1], "requested_rank": rank,
              "mixed_probes": apply_Binv is not None and raw_fraction < 1.0})
