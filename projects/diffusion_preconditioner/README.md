# Two-level preconditioning with diffusion-filtered coarse spaces

Implementation of Kalogeris, *Two-Level Preconditioning with Diffusion-Filtered
Coarse Spaces for Ill-Conditioned Finite-Element SPD Systems* (`ΜΑΙΝ.pdf`).

## What is where

The **baselines** the method is compared against are library code, in
`simulacrum.solvers` — they are general-purpose and are tested as such:

| module | contents |
|---|---|
| `solvers/krylov.py` | `conjugate_gradient`, `pcg` (Algorithm 1), `KrylovResult` |
| `solvers/nonsymmetric.py` | `gmres(m)`, `fgmres`, `bicg`, `bicgstab` |
| `solvers/preconditioners.py` | `Jacobi`, `IncompleteCholesky` (IC(0) and threshold ICT), `IncompleteLU`, `AlgebraicMultigrid` |
| `solvers/amg.py` | smoothed-aggregation hierarchy and symmetric V-cycle |

The **contribution** is here:

| module | paper section |
|---|---|
| `diffusion.py` | 3.1.1–3.1.3 — Lanczos/Arnoldi filtering, semigroup reuse, truncated-SVD compression, Strategies A and B |
| `two_level.py` | 3.2 — `T⁻¹ = PᵗB⁻¹P + ZE⁻¹Zᵗ` |
| `benchmark.py` | 4 — the comparison |
| `matrices/` | SuiteSparse Library I, the paper's small benchmark set |

```bash
python -m projects.diffusion_preconditioner.benchmark            # generated FE problem
python -m projects.diffusion_preconditioner.benchmark --quick
python -m projects.diffusion_preconditioner.benchmark --suitesparse
```

## Measured results

PCG iterations to `‖r‖/‖b‖ < 1e-8`, mean over 5 right-hand sides, IC(0)
baseline, coarse rank 60, 16 probes, Krylov dimension 50, times
`(32, 64, 128, 256)`. `shift` is the diagonal shift IC(0) needed.

| matrix | n | shift | IC(0) | + Strategy A | + Strategy B | gain A | gain B |
|---|---|---|---|---|---|---|---|
| bcsstk06 | 420 | 0.102 | 114 | 16 | 16 | 7.14× | 7.14× |
| bcsstk08 | 1074 | 0 | 33 | 22 | 18 | 1.48× | 1.77× |
| bcsstk11 | 1473 | 0.026 | 924 | 824 | 362 | 1.12× | 2.55× |
| msc01050 | 1050 | 0.205 | 1445 | 698 | 253 | 2.07× | 5.70× |
| msc01440 | 1440 | 2e-4 | 247 | 188 | 57 | 1.32× | 4.36× |
| nasa1824 | 1824 | 0.051 | 278 | 173 | 37 | 1.61× | 7.55× |
| nos1 | 237 | 0.41 | 268 | 21 | 19 | 12.88× | 14.40× |
| nos3 | 960 | 0 | 49 | 11 | 12 | 4.44× | 4.07× |
| nos7 | 729 | 0 | 30 | 11 | 12 | 2.76× | 2.53× |
| plbuckle | 1282 | 4e-4 | 67 | 20 | 22 | 3.36× | 3.05× |

**Strategy B wins where the problem is hardest.** On the structural matrices with
the worst baseline counts — bcsstk11, msc01050, msc01440, nasa1824 — B is 2 to 5
times better than A. That is what "solver-aware" buys: A targets low-energy modes
of `K`, B targets the modes `B⁻¹K` actually fails to damp, and those diverge most
when the baseline is struggling. On the easy matrices (nos3, nos7, plbuckle) the
two are within a couple of iterations and A is marginally cheaper to build.

**Semigroup reuse delivers exactly its theoretical factor.** 4 diffusion times,
6 probes: 840 operator applications naively, 210 with reuse — **4.00× against a
theoretical 4×**, and independent of the number of times.

## Two caveats, measured

**These are not the paper's numbers.** The paper used pyamg for AMG and PETSc for
ICC; this uses Simulacrum's own smoothed aggregation and IC(0), whose coarsening
and dropping differ. The qualitative findings reproduce — large reductions for
Jacobi, solid gains for AMG, ICC reductions that do not always pay in wall-clock.
Where a number differs from the paper, that is the baseline differing, not the
method.

**At fixed rank the gain shrinks as the problem grows.** On a generated
heterogeneous-diffusion problem with rank fixed at 40:

| unknowns | IC(0) | + Strategy A | gain |
|---|---|---|---|
| 576 | 65 | 20 | 3.29× |
| 1600 | 142 | 70 | 2.03× |
| 6400 | 284 | 216 | 1.32× |
| 14400 | 427 | 364 | 1.17× |

The rank has to grow with the problem, because the number of slow modes does. At
n = 14400, rank 40 → 120 → 300 gives 1.17× → 1.47× → 2.39×. Since the two-level
apply costs `O(n·r)`, that erodes the wall-clock case on a problem with a *cheap*
baseline — on this generated problem the iteration phase never broke even against
IC(0) at any size tested, while the iteration counts fell throughout. It is the
amortised, expensive-baseline, fixed-size regime the method is for, which is the
regime the paper's benchmark set represents.

## Notes on the implementation

- **Strategy A's basis lives in scaled coordinates.** It is built on
  `S = D^-1/2 K D^-1/2` and must be mapped back by `D^-1/2` before the Galerkin
  operator is formed (Eq. 52). `TwoLevelPreconditioner` does this automatically
  from `CoarseSpace.scaling`; doing it by hand and forgetting produces a
  preconditioner that still converges, just worse. `deflation_residual()` is the
  check that catches it.
- **`spectral_decay` tells you the rank is too large.** σ_r/σ_1 came out 9.7e-4
  for A and 3.3e-7 for B on the same problem: B's snapshots are far more
  redundant, because the baseline has already removed most of the spectrum. Read
  it as "B needs a smaller rank" — on one AMG run it truncated itself to 24 and
  still beat A at 40.
- **A non-SPD baseline needs a flexible Krylov method**, as §3.2 notes. Wrapping
  ILU gives a two-level operator with `is_spd = False`, and `fgmres` converges on
  it where PCG's assumptions fail — tested. The coarse space also accelerates
  GMRES as readily as PCG, since nothing in its construction knows which solver
  will consume it.
- **Strategy B requires an SPD baseline** and refuses without one. With a
  non-SPD `B`, `B⁻¹K` has no real positive spectrum and `exp(-tM)` is not a
  low-pass filter at all. Use Strategy A there; its mixed probes still inject
  baseline information while filtering an operator guaranteed SPD.
- IC(0) needed a diagonal shift on 7 of the 10 matrices, up to 0.41 for `nos1`.
  With the shift budget capped lower, `nos1` and `msc01050` fail outright.

Tests: `tests/test_solvers/test_krylov.py` (baselines),
`tests/test_projects/test_diffusion_preconditioner.py` (this method).
