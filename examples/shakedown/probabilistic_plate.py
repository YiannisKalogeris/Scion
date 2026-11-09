r"""
Example 2: a plate with a hole and a random yield-stress field.

The first example here whose randomness is a *field* rather than a handful of
scalars, and the first whose transition loads cost a finite element solve per
realization. The model is :mod:`examples.shakedown.plate_model`; this script
adds the Karhunen--Loeve yield field, the transition loads, and the direct
probability integration of the regime bands.

The random inputs
-----------------
``s = n_modes + 2`` standard inputs:

* the Karhunen--Loeve coefficients of a lognormal yield field, mean 355 MPa,
  coefficient of variation 0.10, squared-exponential correlation at ``l_c/L``
  of 0.25, 0.5 or 1, truncated at 99% of the domain-integrated variance --
  31, 12 and 6 modes, so ``s`` is 33, 14 and 8. See :func:`expansion` for why
  the kernel is smooth and not exponential;
* the ratio of the two edge tractions, uniform on ``[0.2, 1]``;
* Young's modulus, lognormal at a coefficient of variation of 0.05, which
  **cannot** affect either transition load and is carried as an exactness
  check. Under prescribed tractions and homogeneous displacement conditions
  the elastic stress field is independent of ``E`` -- the displacement scales
  as ``1/E`` and the stress does not -- so a dependence would be a bug, not a
  modelling choice. Checked rather than asserted; see ``--checks``.

Which transition loads this example reports, and why not all three
------------------------------------------------------------------
``lambda_el`` and ``lambda_sd`` are computed for every realization.
``lambda_rr`` is not, and the reason is a cost asymmetry worth stating rather
than hiding.

``lambda_sd`` is a **first passage**: shaking down at a load factor implies
shaking down below it, so a bisection on "does this cycle shake down" is
exact up to its own resolution, and each test is one constrained solve.
``lambda_rr`` is not a first passage of anything the constrained solve returns
cheaply. Just above ``lambda_sd`` the asymptotic cycle is *alternating*, and
on this plate it converges slowly -- 311 to 355 cycles at ``1.02`` and
``1.15 lambda_sd``, and not within 2000 cycles at some factors between. A
solve that has not become periodic raises, and reading that refusal as
ratcheting is wrong: measured here at ``1.15 lambda_sd`` the state is
alternating with ``r`` at ``2.0e-8`` of the strain scale against a plastic
strain range of ``2.9e-1``. The instrument that locates ``lambda_rr`` without
needing convergence is the growth of the constant Fourier term, which costs
two RSDM runs per load factor, an order of magnitude more than one constrained
solve. It is therefore measured on a subsample and reported with its cost,
and the bands integrated over the full point sets are ``P_E``, ``P_S`` and
their complement.

That complement still sums to one exactly, for the same reason the four bands
do: it is a difference of cumulative distribution functions of the same
transition loads.

**An earlier reading of this example recorded an empty alternating band and
was wrong.** With a 250-cycle budget every load factor above ``lambda_sd``
refused, which reads as ratcheting; with 2000 cycles the same states come back
alternating. The refusal is a budget, not a regime, whenever ``r`` is
available to contradict it.

Run
---
    python -m examples.shakedown.probabilistic_plate --checks   # verification
    python -m examples.shakedown.probabilistic_plate --quick    # minutes
    python -m examples.shakedown.probabilistic_plate            # hours
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scion.materials.plasticity.yield_criteria import VonMises
from scion.shakedown import (CyclicProgram, LoadDomain, cyclic_steady_state,
                             elastic_factor)
from scion.uq.density import RepresentativePoints, gf_points
from scion.uq.forward import (JointIndependent, KarhunenLoeve, LogNormal,
                              LogNormalField, Normal, Uniform)

from examples.shakedown.plate_model import DOMAINS, E_REF, YIELD, build
from examples.shakedown.probabilistic_bree import regime_bands

RESULTS = Path(__file__).parent / "results"
SAFE = ("elastic", "shakedown")

YIELD_COV = 0.10
E_COV = 0.05
RATIO_RANGE = (0.2, 1.0)
CORRELATION_LENGTHS = (0.25, 0.5, 1.0)
VARIANCE_TARGET = 0.99
KERNEL = "squared_exponential"

#: Bisection steps for a transition load. Seven on a bracket of
#: 0.25 ``lambda_el`` resolves ``lambda_sd`` to 2e-3 ``lambda_el``, two orders
#: below the spread of ``lambda_el`` itself, so the resolution does not enter
#: the probability comparison. The bracket is guarded, so a realization
#: outside it is widened rather than mis-resolved.
BISECT_STEPS = 7


# ---------------------------------------------------------------------------
# the random inputs
# ---------------------------------------------------------------------------

def expansion(model: dict, correlation_length: float) -> KarhunenLoeve:
    """The yield field's expansion on this mesh's quadrature points.

    Built on the quadrature points and their weights, not the nodes: the
    weights enter the eigenproblem, and the field is only ever evaluated where
    the yield condition is checked.

    **The kernel is squared exponential, and an exponential one was tried
    first and rejected on measurement.** An exponential covariance is not
    differentiable, so its eigenvalues decay like ``k^-2`` and a variance
    target of any useful size retains modes at the *mesh* scale: at 95% of the
    variance this plate needs 44 modes on 144 points and 62 on 256, and the
    16th eigenvalue disagrees by 23% between those two meshes at every
    correlation length tried. The expansion would then be a property of the
    discretization rather than of the field. A squared-exponential kernel
    decays geometrically and is resolved: 12 modes at ``l_c/L = 0.5`` on both
    144 and 576 points, with the retained eigenvalues agreeing to 2.9e-2 and
    the pointwise variance minimum at 0.94.
    """
    return KarhunenLoeve.from_points(
        model["xq"], model["weights"], correlation_length,
        variance_target=VARIANCE_TARGET, kernel=KERNEL)


def distribution(n_modes: int) -> JointIndependent:
    """``n_modes`` standard normals, the traction ratio, then ``E``."""
    return JointIndependent(
        [Normal(0.0, 1.0) for _ in range(n_modes)]
        + [Uniform(*RATIO_RANGE),
           LogNormal.from_moments(E_REF, E_COV * E_REF)])


def unpack(theta: np.ndarray, n_modes: int):
    """``(xi, ratio, E)`` from the input vector, one row per realization."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    return theta[:, :n_modes], theta[:, n_modes], theta[:, n_modes + 1]


# ---------------------------------------------------------------------------
# the mechanics, one realization at a time
# ---------------------------------------------------------------------------

def _history(model: dict, name: str, ratio: float, per_leg: int = 1):
    """The cycle for a load domain whose second traction is scaled.

    The elastic cases are solved once; a realization only re-weights them, so
    the traction ratio costs no solve.
    """
    base = DOMAINS[name]
    lower = np.asarray(base.lower, dtype=float).copy()
    upper = np.asarray(base.upper, dtype=float).copy()
    lower[1] *= ratio
    upper[1] *= ratio
    dom = LoadDomain.box(lower, upper)
    program = CyclicProgram.through_vertices(dom, per_leg=per_leg)
    return program.elastic_history(model["envelope"]), dom


def _bisect(unsafe, lo: float, hi: float, steps: int = BISECT_STEPS) -> float:
    """First passage of ``unsafe``, which is monotone in the load factor.

    Shakedown at ``lambda`` implies shakedown below it, so the crossing is
    unique and a bisection is exact up to its own resolution rather than a
    search that might find a different root.
    """
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if unsafe(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def transition_loads(model: dict, sigma_y: np.ndarray, ratio: float,
                     name: str = "box", max_cycles: int = 20,
                     bracket=(1.33, 1.58)) -> SimpleNamespace:
    """``lambda_el``, ``lambda_sd`` and ``lambda_rr`` for one realization.

    ``lambda_el`` is the elastic solution and costs nothing. ``lambda_sd`` is
    bisected on the constrained cyclic solve, whose refusal to equilibrate
    past the limit load *is* the answer rather than a solver failure.

    **``max_cycles`` is 20 rather than the solver's own default, and that is a
    measurement rather than a guess.** The bisection does not need the
    asymptotic cycle, only the *regime*, and the two sides are separated by
    how fast the cycle settles rather than by where it settles: below
    ``lambda_sd`` it is periodic within about three cycles, above it there is
    no periodic state at any budget. So every budget that clears the
    shakedown side classifies identically. Measured over six realizations,
    ``lambda_sd`` is the same to all seven printed digits at ``max_cycles`` of
    120, 30, 15 and 8, while the cost falls from 3.4--8.8 s to 0.4--1.0 s. The
    default keeps a margin of six cycles over the observed convergence and
    still runs about four times faster than 120.

    A budget too small for the shakedown side would misread a slow-converging
    periodic state as unsafe and bias ``lambda_sd`` *down*, which is
    conservative but wrong; that is the direction to suspect if this is cut
    further.

    ``lambda_rr`` is **not** computed here; see the module docstring. The
    returned ``ratchet`` is ``inf``, which makes the alternating band run to
    the top of the load range and keeps the band identity exact, rather than
    equal to ``lambda_sd``, which would silently assert an empty alternating
    band this plate does not have.
    """
    history, domain = _history(model, name, ratio)
    lam_el = elastic_factor(model["envelope"].vertex_stresses(domain),
                            VonMises(), sigma_y)

    def unsafe(lam: float) -> bool:
        try:
            state = cyclic_steady_state(lam * history, model["tables"],
                                        model["free"], model["C"], sigma_y,
                                        max_cycles=max_cycles)
            return state.regime() not in SAFE
        except Exception:
            # past the limit load of the discrete model nothing equilibrates,
            # and a cycle that never becomes periodic is ratcheting: both are
            # "not shaking down", which is what the bisection asks
            return True

    # the bracket is tight because lambda_sd/lambda_el is nearly constant for
    # this configuration -- measured 1.42 to 1.50 over realizations -- which
    # buys three bisection steps. It is *guarded* rather than trusted: a
    # realization whose ratio falls outside is widened until it is bracketed,
    # so the saving cannot turn into a silently mis-resolved transition load.
    lo, hi = bracket[0] * lam_el, bracket[1] * lam_el
    guard = 0
    # downwards, halving the excess over lambda_el each time: the limit is
    # lambda_el itself, which is always safe because rho = 0 is admissible, so
    # this terminates rather than merely running out of attempts
    while guard < 8 and unsafe(lo):
        hi = lo
        lo = lam_el * (1.0 + 0.5 * (lo / lam_el - 1.0))
        guard += 1
    while guard < 16 and not unsafe(hi):
        lo, hi, guard = hi, 2.0 * hi, guard + 1
    lam_sd = _bisect(unsafe, lo, hi)
    return SimpleNamespace(elastic=lam_el, shakedown=lam_sd,
                           ratchet=np.inf)


def sweep(model: dict, kl: KarhunenLoeve, theta: np.ndarray,
          name: str = "box", verbose: bool = False) -> SimpleNamespace:
    """The three transition loads for every row of ``theta``."""
    n_modes = kl.n_modes
    xi, ratio, _ = unpack(theta, n_modes)
    field = LogNormalField(kl, mean=YIELD, cov=YIELD_COV)
    el = np.empty(len(xi))
    sd = np.empty(len(xi))
    t0 = time.time()
    for i in range(len(xi)):
        loads = transition_loads(model, field.realize(xi[i]), float(ratio[i]),
                                 name=name)
        el[i], sd[i] = loads.elastic, loads.shakedown
        if verbose and (i + 1) % 25 == 0:
            done = i + 1
            print(f"      {done}/{len(xi)}  {time.time() - t0:.0f}s elapsed, "
                  f"{(time.time() - t0) / done * (len(xi) - done):.0f}s left",
                  flush=True)
    # ratchet is infinite, as in transition_loads: this example does not run
    # the growth-ratio instrument, so everything above lambda_sd is reported
    # as one band rather than split on a quantity that was not measured
    return SimpleNamespace(elastic=el, shakedown=sd,
                           ratchet=np.full(len(sd), np.inf),
                           seconds=time.time() - t0)


def plastic_activity(model: dict, kl: KarhunenLoeve, theta: np.ndarray,
                     lam: float, eta: float = 1.0e-12,
                     verbose: bool = False) -> SimpleNamespace:
    """Per-point probability of having yielded on the way to the steady cycle.

    One converged solve per realization at a *fixed* load factor, not a
    bisection, so this costs about a tenth of a transition load.

    **The quantity is the accumulated plastic strain, not its range over the
    cycle, and that distinction is the whole definition.** Shaking down means
    the asymptotic cycle is elastic, so the plastic strain *range* is
    identically zero below the local shakedown limit -- a map of it would be
    empty everywhere the structure is safe and undefined where it is not,
    which is no map at all. What is non-trivial, and what a designer would
    want, is where the plate yielded *while building its residual stress
    field*: the total ``eps^p`` the converged state carries is nonzero exactly
    at those points and stays nonzero once the cycle has become elastic.

    Evaluated below the bulk of the ``lambda_sd`` distribution, so nearly
    every realization has a periodic asymptote. Those that do not are
    reported through ``nonperiodic`` rather than hidden: the map is
    conditional on a periodic state, and at a load factor where many
    realizations ratchet that condition would bias it.
    """
    xi, ratio, _ = unpack(theta, kl.n_modes)
    field = LogNormalField(kl, mean=YIELD, cov=YIELD_COV)
    active = np.zeros(model["tables"].n_points)
    counted = 0
    nonperiodic = 0
    t0 = time.time()
    for i in range(len(xi)):
        history, _ = _history(model, "box", float(ratio[i]))
        try:
            state = cyclic_steady_state(lam * history, model["tables"],
                                        model["free"], model["C"],
                                        field.realize(xi[i]), max_cycles=400)
        except Exception:
            nonperiodic += 1
            continue
        # the accumulated plastic strain, at the end of the asymptotic cycle:
        # nonzero wherever the point ever yielded, including in the transient
        magnitude = np.sqrt((state.plastic_strain[-1] ** 2).sum(axis=-1))
        active += magnitude > eta * state.plastic_strain_scale
        counted += 1
        if verbose and (i + 1) % 50 == 0:
            print(f"      activity {i + 1}/{len(xi)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return SimpleNamespace(probability=active / max(counted, 1),
                           counted=counted, nonperiodic=nonperiodic,
                           seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# probability integration
# ---------------------------------------------------------------------------

def reference_samples(model: dict, kl: KarhunenLoeve, correlation_length: float,
                      n_target: int, mesh: int, batch: int = 200,
                      verbose: bool = True) -> SimpleNamespace:
    """Monte Carlo transition loads, cached on disk and extensible in place.

    The reference is the expensive half of this example and the part that has
    to grow: its own standard error at a probability near one half is
    ``1/(2 sqrt(n))``, so a comparison against it cannot resolve a DPIM error
    below that however many representative points are used. Enlarging it must
    therefore not mean recomputing it.

    Realizations are drawn in independent batches keyed by ``(seed, ell, k)``,
    so batch ``k`` is reproducible on its own and extending the file adds
    batches rather than redrawing the set. Each batch is written as it
    finishes, which also makes a long run survive being interrupted -- the
    first version of this saved only at the end, and four hours of reference
    went with the session that was killed.
    """
    path = RESULTS / f"plate_ref_lc{correlation_length:g}_m{mesh}.npz"
    el = np.empty(0)
    sd = np.empty(0)
    seconds = 0.0
    if path.exists():
        with np.load(path) as data:
            el, sd = data["elastic"], data["shakedown"]
            seconds = float(data["seconds"])
        if verbose:
            print(f"    reference: {el.size} cached in {path.name}", flush=True)

    dist = distribution(kl.n_modes)
    while el.size < n_target:
        k = el.size // batch
        rng = np.random.default_rng(
            [20_260_101, int(round(correlation_length * 1000)), k])
        if verbose:
            print(f"    reference batch {k}: {batch} realizations "
                  f"({el.size} -> {el.size + batch} of {n_target})", flush=True)
        loads = sweep(model, kl, dist.sample(batch, rng=rng), verbose=verbose)
        el = np.concatenate([el, loads.elastic])
        sd = np.concatenate([sd, loads.shakedown])
        seconds += loads.seconds
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, elastic=el, shakedown=sd, seconds=seconds,
                            correlation_length=correlation_length, mesh=mesh)
    return SimpleNamespace(elastic=el, shakedown=sd,
                           ratchet=np.full(el.size, np.inf), seconds=seconds,
                           standard_error=0.5 / np.sqrt(el.size))


def curves(points: RepresentativePoints, lam, loads) -> dict:
    """The regime bands, through Example 1's single implementation."""
    return regime_bands(points, lam, loads)


def monte_carlo(loads, lam) -> dict:
    """Reference curves by counting, from an independent sample."""
    lam = np.asarray(lam, dtype=float).reshape(-1, 1)
    return {
        "E": (loads.elastic >= lam).mean(axis=1),
        "S": ((loads.shakedown >= lam) & (loads.elastic < lam)).mean(axis=1),
        "AP": ((loads.ratchet >= lam) & (loads.shakedown < lam)).mean(axis=1),
        "R": (loads.ratchet < lam).mean(axis=1),
    }


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def checks(quick: bool = False) -> int:
    """Everything about the model that can be checked without probabilities.

    Printed rather than asserted, because the numbers are what the manuscript
    quotes and a silent pass would not supply them.
    """
    from scion.shakedown import linear_matching
    from examples.shakedown.plate_model import (E_REF, TRACTION, build, hoop,
                                                quarter_plate)
    from scion.discretization.fem.tables import QuadratureTables
    from scion.shakedown import ElasticEnvelope

    print("1. the elastic field against Kirsch (external, closed form)")
    print("     n   pts   area err   r_min/R   uniaxial   equibiaxial hoop")
    for n in ((4, 8, 12) if quick else (4, 8, 12, 16, 24)):
        m = build(n, n)
        s = m["envelope"].stresses / TRACTION
        r = np.hypot(m["xq"][:, 0], m["xq"][:, 1])
        ring = r < r.min() * 1.02
        h = hoop(s[0] + s[1], m["xq"])
        exact_area = 1.0 - np.pi * 0.1 ** 2 / 4.0
        print("   %3d %5d  %.2e    %.4f    %.4f     %.4f"
              % (n, m["tables"].n_points,
                 abs(m["weights"].sum() - exact_area), r.min() / 0.1,
                 s[0][:, 0].max(), h[ring].max()))
    print("     Kirsch at the hole surface: uniaxial 3.0, hoop 2.0;")
    print("     the quadrature points sit inside the elements, so a coarse")
    print("     mesh reads low rather than wrong\n")

    m = build(8, 8)
    print("2. a fully reversed domain has its shakedown limit at first yield")
    rev = LoadDomain.box([-1.0, -1.0], [1.0, 1.0])
    program = CyclicProgram.through_vertices(rev, per_leg=1)
    hist = program.elastic_history(m["envelope"])
    lam_el = elastic_factor(m["envelope"].vertex_stresses(rev), VonMises(),
                            YIELD)

    def unsafe_rev(lam):
        try:
            st = cyclic_steady_state(lam * hist, m["tables"], m["free"],
                                     m["C"], YIELD, max_cycles=150)
            return st.regime() not in SAFE
        except Exception:
            return True

    lam_sd = _bisect(unsafe_rev, 0.8 * lam_el, 2.0 * lam_el, 14)
    print("     lam_el %.6f   lam_sd %.6f   ratio %.6f" %
          (lam_el, lam_sd, lam_sd / lam_el))
    print("     an identity, not a benchmark: a domain symmetric about zero")
    print("     builds no mean residual stress, so there is nothing for a")
    print("     residual field to buy. The residual is the bisection's\n"
          "     resolution.\n")

    print("3. neither transition load may depend on E")
    print("     under prescribed tractions the displacement scales as 1/E and")
    print("     the stress does not, so this is exact")
    for factor in (0.5, 1.0, 2.0):
        mm = build(6, 6)
        mm["C"] = mm["C"] * factor
        loads = transition_loads(mm, np.full(mm["tables"].n_points, YIELD),
                                 1.0)
        print("     E/E_ref %.1f   lam_el %.9f   lam_sd %.9f"
              % (factor, loads.elastic, loads.shakedown))
    print()

    print("4. the cycle needs only the vertices of the load domain")
    print("     Melan's condition sees the extremes and nothing between")
    for per_leg in (1, 2, 4):
        mm = build(8, 8, per_leg=per_leg)
        loads = transition_loads(mm, np.full(mm["tables"].n_points, YIELD),
                                 1.0)
        print("     per_leg %d  n_time %2d   lam_sd %.6f"
              % (per_leg, mm["history"]["box"].shape[0], loads.shakedown))
    print()

    print("5. the Koiter bound of the same mesh, at matched integration")
    print("     LMM needs b-bar and so does the constrained solve, so these")
    print("     two are like for like; a Melan bound would need full")
    print("     integration and the locking difference would swamp the")
    print("     comparison")
    for n in ((6, 8) if quick else (6, 8, 12)):
        mm = build(n, n)
        vertex = mm["envelope"].vertex_stresses(DOMAINS["box"])
        bound = linear_matching(vertex, mm["tables"], mm["free"], YIELD)
        loads = transition_loads(mm, np.full(mm["tables"].n_points, YIELD),
                                 1.0, bracket=(1.02, 2.0))
        print("     n %2d  Koiter %.5f   constrained %.5f   gap %+.2f%%  "
              "monotone %s"
              % (n, bound.factor, loads.shakedown,
                 100.0 * (loads.shakedown / bound.factor - 1.0),
                 bound.monotone))
    print()

    print("6. the alternating band is NOT empty on this domain")
    mm = build(6, 6)
    hist_box = mm["history"]["box"]
    sy = np.full(mm["tables"].n_points, YIELD)
    sd = transition_loads(mm, sy, 1.0).shakedown
    print("     lam_sd %.5f;  r and deps^p read off the asymptotic cycle" % sd)
    for f in (1.02, 1.15, 1.30):
        try:
            st = cyclic_steady_state(f * sd * hist_box, mm["tables"],
                                     mm["free"], mm["C"], sy,
                                     max_cycles=3000)
            scale = st.plastic_strain_scale
            print("     %.2f lam_sd  %-12s deps^p/scale %.3e  r/scale %.3e  "
                  "cycles %4d" % (f, st.regime(),
                                  st.plastic_strain_range / scale,
                                  st.ratchet_rate / scale, st.cycles))
        except Exception as exc:
            print("     %.2f lam_sd  no periodic state within 3000 cycles "
                  "(%s)" % (f, str(exc)[:40]))
    print("     a refusal is a cycle budget, not a regime, wherever r can be")
    print("     measured to contradict it\n")
    return 0


# ---------------------------------------------------------------------------
# the probabilistic study
# ---------------------------------------------------------------------------

def _load_grid(el, sd, n: int = 61) -> np.ndarray:
    """A load range spanning both transition loads with room either side."""
    lo = 0.85 * float(np.min(el))
    hi = 1.15 * float(np.max(sd[np.isfinite(sd)]))
    return np.linspace(lo, hi, n)


def study(n_points=(256, 512, 1024), n_reference: int = 2000,
          correlation_lengths=CORRELATION_LENGTHS, mesh: int = 6,
          n_activity: int = 400, verbose: bool = True) -> dict:
    """The point-count study and the regime curves, saved as one archive."""
    model = build(mesh, mesh)
    out = {"mesh": mesh, "points": model["tables"].n_points,
           "xq": model["xq"], "weights": model["weights"]}

    for ell in correlation_lengths:
        kl = expansion(model, ell)
        dist = distribution(kl.n_modes)
        tag = f"lc{ell:g}"
        if verbose:
            print(f"\n  l_c/L = {ell:g}:  {kl.n_modes} modes, "
                  f"{kl.variance_captured:.4f} of the variance, pointwise "
                  f"minimum {kl.pointwise_variance.min():.4f}, s = {dist.dim}",
                  flush=True)
        out[f"{tag}_modes"] = kl.n_modes
        out[f"{tag}_captured"] = kl.variance_captured
        out[f"{tag}_pointwise_min"] = kl.pointwise_variance.min()

        ref = reference_samples(model, kl, ell, n_reference, mesh,
                                verbose=verbose)
        out[f"{tag}_ref_el"] = ref.elastic
        out[f"{tag}_ref_sd"] = ref.shakedown
        out[f"{tag}_ref_seconds"] = ref.seconds
        out[f"{tag}_ref_stderr"] = ref.standard_error
        if verbose:
            print(f"    reference: {ref.elastic.size} realizations, standard "
                  f"error {ref.standard_error:.4f} at a probability of one "
                  f"half", flush=True)

        lam = _load_grid(ref.elastic, ref.shakedown)
        out[f"{tag}_lam"] = lam
        mc = monte_carlo(ref, lam)
        for key, value in mc.items():
            out[f"{tag}_mc_{key}"] = value

        for n in n_points:
            pts = gf_points(dist, n)
            if verbose:
                print(f"    N = {n}", flush=True)
            loads = sweep(model, kl, pts.z, verbose=verbose)
            bands = curves(pts, lam, loads)
            for key, value in bands.items():
                out[f"{tag}_n{n}_{key}"] = value
            out[f"{tag}_n{n}_el"] = loads.elastic
            out[f"{tag}_n{n}_sd"] = loads.shakedown
            out[f"{tag}_n{n}_weights"] = pts.probability
            out[f"{tag}_n{n}_seconds"] = loads.seconds
            stack = sum(bands[k] for k in ("E", "S", "AP", "R"))
            out[f"{tag}_n{n}_band_residual"] = np.abs(stack - 1.0).max()
            if verbose:
                worst = max(np.abs(bands[k] - mc[k]).max()
                            for k in ("E", "S"))
                print(f"      {loads.seconds:.0f}s, band residual "
                      f"{np.abs(stack - 1.0).max():.1e}, worst gap in "
                      f"P_E/P_S against the reference {worst:.4f} "
                      f"(reference noise {ref.standard_error:.4f})",
                      flush=True)
        # written per correlation length: a long run must not lose a finished
        # block to an interruption
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RESULTS / "plate_partial.npz", **out)

    # the activity map, at the mean shakedown factor of the middle
    # correlation length -- one fixed load factor, so one solve per
    # realization rather than a bisection
    tag = f"lc{correlation_lengths[len(correlation_lengths) // 2]:g}"
    kl = expansion(model, correlation_lengths[len(correlation_lengths) // 2])
    dist = distribution(kl.n_modes)
    # the tenth percentile of lambda_sd, so about nine realizations in ten
    # have a periodic asymptote and the map is not conditioned on a selected
    # minority. At the *mean* shakedown factor roughly half would ratchet.
    lam = float(np.percentile(out[f"{tag}_ref_sd"], 10.0))
    rng = np.random.default_rng(99)
    if verbose:
        print(f"\n  activity map at lambda = {lam:.4f} "
              f"({n_activity} realizations)", flush=True)
    act = plastic_activity(model, kl, dist.sample(n_activity, rng=rng), lam,
                           verbose=verbose)
    out["activity"] = act.probability
    out["activity_lambda"] = lam
    out["activity_counted"] = act.counted
    out["activity_nonperiodic"] = act.nonperiodic
    out["activity_seconds"] = act.seconds
    if verbose:
        print(f"    {act.counted} periodic, {act.nonperiodic} not, "
              f"{act.seconds:.0f}s; active at some point in "
              f"{float((act.probability > 0).mean()) * 100:.1f}% of the plate",
              flush=True)
    return out


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Example 2: plate with a hole, random yield field")
    parser.add_argument("--checks", action="store_true",
                        help="the verification pass, no probabilities")
    parser.add_argument("--quick", action="store_true",
                        help="minutes rather than hours; not publishable")
    parser.add_argument("--mesh", type=int, default=6)
    parser.add_argument("--reference", type=int, default=3000,
                        help="Monte Carlo realizations per correlation length")
    parser.add_argument("--counts", type=int, nargs="+",
                        default=[128, 256, 512])
    parser.add_argument("--activity", type=int, default=300)
    parser.add_argument("--lengths", type=float, nargs="+",
                        default=list(CORRELATION_LENGTHS))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.checks:
        return checks(quick=args.quick)

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Example 2: plate with a hole, random yield-stress field")
    if args.quick:
        print("  QUICK RUN -- not publishable")
        out = study(n_points=(32, 64), n_reference=64,
                    correlation_lengths=(0.5,), mesh=4, n_activity=24)
        name = "plate_quick.npz"
    else:
        out = study(n_points=tuple(args.counts), n_reference=args.reference,
                    correlation_lengths=tuple(args.lengths), mesh=args.mesh,
                    n_activity=args.activity)
        name = "plate.npz"
    path = RESULTS / name
    np.savez_compressed(path, **out)
    print(f"\n  saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
