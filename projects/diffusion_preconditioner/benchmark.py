#!/usr/bin/env python3
r"""
Two-level diffusion-filtered preconditioning against its baselines.

Reproduces the structure of the comparison in Kalogeris, *Two-Level
Preconditioning with Diffusion-Filtered Coarse Spaces for Ill-Conditioned
Finite-Element SPD Systems*: for each baseline preconditioner, iteration counts
and wall-clock times with and without the two-level correction, over multiple
right-hand sides.

Two things are reported that a single-solve table cannot show
------------------------------------------------------------
**Amortisation.** The coarse space costs a one-off setup and is then reused for
every right-hand side. On one solve that setup can dominate; over fifty it is
noise. The break-even count is printed, because it is the number that decides
whether the method is worth using on a given problem.

**Iterations and time separately.** They do not agree. A coarse correction that
removes 70% of the iterations can still lose on wall-clock against a cheap
baseline, because the two-level apply costs ``O(n r)`` on top of the baseline
and Jacobi's baseline cost is almost nothing. Reporting only iterations flatters
the method; reporting only time hides where the gain comes from.

What is *not* claimed here
--------------------------
These numbers do not reproduce the paper's tables. The paper used pyamg for AMG
and PETSc for ICC; this runs Simulacrum's own smoothed aggregation and IC(0),
whose coarsening and dropping differ in detail. The qualitative findings do
reproduce -- large iteration reductions for Jacobi, solid gains for AMG, and
iteration reductions for ICC that do not always pay for themselves in time --
and where a number differs from the paper that is a difference in the baseline,
not in the method.

Run
---
    python -m projects.diffusion_preconditioner.benchmark
    python -m projects.diffusion_preconditioner.benchmark --quick
    python -m projects.diffusion_preconditioner.benchmark --suitesparse
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from projects.diffusion_preconditioner.diffusion import build_coarse_space
from projects.diffusion_preconditioner.two_level import TwoLevelPreconditioner
from simulacrum.core.logging import quiet
from simulacrum.solvers import get_preconditioner, pcg

HERE = Path(__file__).resolve().parent
MATRICES = HERE / "matrices"

BASELINES = ("jacobi", "icc", "amg")
TOL = 1e-8
MAX_ITER = 20000

FULL = dict(n=40, contrast=1e6, rank=40, n_probes=16, krylov=50, n_rhs=8)
QUICK = dict(n=24, contrast=1e5, rank=30, n_probes=10, krylov=35, n_rhs=3)


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def heterogeneous_diffusion(n: int, contrast: float = 1e6,
                            seed: int = 0) -> sparse.csr_matrix:
    """Cell-wise random conductivity over ``contrast`` orders of magnitude.

    A finite-volume 5-point discretisation with harmonic-free averaging at the
    faces. The coefficient jumps are what make it ill-conditioned in the way the
    method targets: the near-null space is no longer the constant vector, so
    diagonal scaling cannot reach it.
    """
    rng = np.random.default_rng(seed)
    k = 10.0 ** rng.uniform(0.0, np.log10(contrast), n * n)
    h = 1.0 / (n + 1)
    index = lambda i, j: i * n + j
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            centre = index(i, j)
            total = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                inside = 0 <= ii < n and 0 <= jj < n
                face = (0.5 * (k[centre] + k[index(ii, jj)]) if inside
                        else k[centre])
                total += face
                if inside:
                    rows.append(centre)
                    cols.append(index(ii, jj))
                    vals.append(-face / h ** 2)
            rows.append(centre)
            cols.append(centre)
            vals.append(total / h ** 2)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))


def load_suitesparse(path: Path):
    """Load a SuiteSparse ``.mat`` file, returning the SPD matrix.

    The files in ``matrices/`` are the small benchmark set from the paper
    (Library I). The larger sets are not committed -- point ``--suitesparse``
    at your own directory for those.
    """
    from scipy.io import loadmat

    data = loadmat(str(path))
    if "Problem" in data:                       # SuiteSparse struct format
        return sparse.csr_matrix(data["Problem"][0, 0]["A"])
    for key, value in data.items():
        if not key.startswith("__") and sparse.issparse(value):
            return sparse.csr_matrix(value)
    raise ValueError(f"no sparse matrix found in {path.name}")


def condition_estimate(A) -> float:
    """Extreme eigenvalues by Lanczos, for reporting only."""
    from scipy.sparse.linalg import eigsh
    try:
        big = float(eigsh(A, k=1, which="LM", return_eigenvectors=False,
                          maxiter=5000)[0])
        small = float(eigsh(A, k=1, which="SM", return_eigenvectors=False,
                            maxiter=5000, tol=1e-4)[0])
        return abs(big / small) if small else float("inf")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

def multi_rhs(A, preconditioner, rhs_list) -> dict:
    """Solve every right-hand side, returning mean iterations and total time."""
    iterations, total, converged = [], 0.0, 0
    for b in rhs_list:
        result = pcg(A, b, preconditioner, tol=TOL, max_iter=MAX_ITER)
        iterations.append(result.iterations)
        total += result.time
        converged += int(result.converged)
    return {"iterations": float(np.mean(iterations)),
            "solve_time": total,
            "converged": converged == len(rhs_list)}


def compare(A, label: str, size: dict, seed: int = 0) -> dict:
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    rhs_list = [rng.standard_normal(n) for _ in range(size["n_rhs"])]

    print(f"\n{'=' * 86}")
    print(f"  {label}:  n = {n},  nnz = {A.nnz},  {size['n_rhs']} right-hand "
          f"sides")
    print("=" * 86)
    print(f"\n  {'baseline':<10}{'variant':<10}{'iters':>8}{'setup':>9}"
          f"{'solve':>9}{'total':>9}{'iters x':>9}{'break-even':>12}")

    rows = {}
    for name in BASELINES:
        baseline = get_preconditioner(name, A)
        plain = multi_rhs(A, baseline, rhs_list)
        base_total = baseline.setup_time + plain["solve_time"]
        print(f"  {name:<10}{'plain':<10}{plain['iterations']:>8.1f}"
              f"{baseline.setup_time:>9.3f}{plain['solve_time']:>9.3f}"
              f"{base_total:>9.3f}{'-':>9}{'-':>12}")
        rows[name] = {"plain": plain, "setup": baseline.setup_time}

        for strategy in ("A", "B"):
            started = time.perf_counter()
            coarse = build_coarse_space(
                A, strategy=strategy, rank=size["rank"],
                n_probes=size["n_probes"], krylov_dim=size["krylov"],
                apply_Binv=baseline.apply, seed=seed)
            two_level = TwoLevelPreconditioner(coarse, baseline).setup(A)
            setup = time.perf_counter() - started

            enhanced = multi_rhs(A, two_level, rhs_list)
            gain = plain["iterations"] / max(enhanced["iterations"], 1e-9)

            # per-solve saving in the iteration phase, against the extra setup
            per_solve = ((plain["solve_time"] - enhanced["solve_time"])
                         / size["n_rhs"])
            break_even = (setup / per_solve if per_solve > 0 else float("inf"))

            total = baseline.setup_time + setup + enhanced["solve_time"]
            print(f"  {'':<10}{'+ diff ' + strategy:<10}"
                  f"{enhanced['iterations']:>8.1f}{setup:>9.3f}"
                  f"{enhanced['solve_time']:>9.3f}{total:>9.3f}"
                  f"{gain:>8.2f}x"
                  f"{(f'{break_even:.0f} rhs' if np.isfinite(break_even) else 'never'):>12}")
            rows[name][strategy] = {
                **enhanced, "setup": setup, "gain": gain,
                "break_even": break_even, "rank": coarse.rank,
                "decay": coarse.spectral_decay,
                "applications": coarse.n_operator_applications}
    return rows


def report(rows: dict) -> None:
    print(f"\n{'=' * 86}")
    print("  Reading the table")
    print("=" * 86)

    print("\n  Iterations fall for every baseline. Whether that pays depends "
          "on\n  what the baseline cost per iteration in the first place:")
    for name in BASELINES:
        entry = rows[name]
        best = min(("A", "B"), key=lambda s: entry[s]["iterations"])
        gain = entry[best]["gain"]
        wins = entry[best]["solve_time"] < entry["plain"]["solve_time"]
        print(f"    {name:<8} {gain:.2f}x fewer iterations (strategy {best}); "
              f"iteration phase {'faster' if wins else 'SLOWER'} in wall-clock")

    print("\n  Coarse-space setup, and the rank each strategy actually used:")
    for name in BASELINES:
        for strategy in ("A", "B"):
            entry = rows[name][strategy]
            print(f"    {name:<8} {strategy}: {entry['setup']:.2f}s, "
                  f"rank {entry['rank']}, "
                  f"{entry['applications']} operator applications, "
                  f"sigma_r/sigma_1 = {entry['decay']:.1e}")

    print("\n  A small sigma_r/sigma_1 means the requested rank exceeded what")
    print("  the filter found. Strategy B shows this much more strongly, which")
    print("  is expected: it filters the *preconditioned* operator, so the")
    print("  baseline has already removed most of the spectrum and the slow")
    print("  subspace that remains is genuinely lower-dimensional.")


def semigroup_saving(A, size) -> None:
    """The claimed factor-of-p saving from Eq. 47, measured."""
    print(f"\n{'=' * 86}")
    print("  Semigroup reuse (Eq. 47): the saving that makes Strategy B "
          "affordable")
    print("=" * 86)

    times = (32.0, 64.0, 128.0, 256.0)
    with_reuse = build_coarse_space(A, strategy="A", rank=size["rank"],
                                    n_probes=6, krylov_dim=size["krylov"],
                                    times=times, seed=0)
    one_time = build_coarse_space(A, strategy="A", rank=size["rank"],
                                  n_probes=6, krylov_dim=size["krylov"],
                                  times=(32.0,), seed=0)
    naive = one_time.n_operator_applications * len(times)

    print(f"\n    {len(times)} diffusion times, 6 probes")
    print(f"      one Krylov basis per (probe, time):  {naive:6d} operator "
          f"applications")
    print(f"      one basis per probe, stepped:        "
          f"{with_reuse.n_operator_applications:6d}")
    print(f"      saving:                              "
          f"{naive / with_reuse.n_operator_applications:6.2f}x  "
          f"(theory {len(times)}x)")
    print("\n    Independent of the number of times, which is why several can "
          "be\n    used for spectral coverage at no extra cost in matvecs.")


# ---------------------------------------------------------------------------

def main() -> None:
    quick = "--quick" in sys.argv
    size = QUICK if quick else FULL

    print("Two-level diffusion-filtered preconditioning")
    print("  baselines from simulacrum.solvers; coarse space from this project")

    A = heterogeneous_diffusion(size["n"], size["contrast"])
    kappa = condition_estimate(A)
    rows = compare(A, f"heterogeneous diffusion, contrast {size['contrast']:.0e}"
                      f" (cond ~ {kappa:.1e})", size)
    report(rows)
    semigroup_saving(A, size)

    if "--suitesparse" in sys.argv:
        paths = sorted(MATRICES.glob("*.mat"))
        if not paths:
            print(f"\n  no .mat files in {MATRICES}")
            return
        for path in paths[:4 if quick else len(paths)]:
            try:
                M = load_suitesparse(path)
            except Exception as exc:
                print(f"\n  {path.name}: could not load ({exc})")
                continue
            if M.shape[0] > 40000:
                print(f"\n  {path.name}: {M.shape[0]} unknowns, skipping")
                continue
            compare(M, path.stem, size)


if __name__ == "__main__":
    with quiet():
        main()
