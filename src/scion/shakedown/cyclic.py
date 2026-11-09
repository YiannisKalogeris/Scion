r"""
The asymptotic cycle with yield enforced exactly, and the plastic strain that
only that gives.

:mod:`~scion.shakedown.rsdm` answers the regime question cheaply and correctly,
and its indicators ``A`` and ``r`` are validated against the Bree closed form
to 0.02%. What it cannot give is a plastic strain *magnitude*, and the reason
is structural rather than a tolerance: the 2012 iteration drives the residual
stress with the radial excess itself,

    eps_dot^p = c D^-1 sigma_p,     c = 1

which is a Perzyna regularisation with a relaxation time of one cycle. Its
fixed point leaves the stress *outside* the yield surface by an amount
proportional to ``1/c``, and the converged state is exactly proportional to
``c`` in the stable region -- measured on the Bree strip, a residual stress
range of 0.100 at ``c = 1`` and 0.199 at ``c = 2`` against an exact 1.0. The
physical answer needs ``c`` of about ten and the explicit fixed point diverges
above ``c ~ 8``, so there is no value of the penalty that is both stable and
right. ``sigma_p = 0`` is scale free, which is why the *regime* boundaries are
untouched by any of this and only the magnitude is lost.

What replaces it
----------------
Rate-independent plasticity, integrated. At each cycle point the stress is
required to lie *on or inside* the surface and the plastic strain increment
comes from the consistency condition, which is the ordinary backward-Euler
return map and has no stability limit at all. Writing the total stress as the
elastic one plus a residual, ``sigma = sigma^E + rho`` with
``rho = D (B r - eps^p)`` for a residual displacement ``r``, the step at cycle
point ``i`` is a Newton solve for ``r_i``:

    eps~_i = D^-1 sigma^E_i + B r_i                 (an auxiliary total strain)
    sigma_i, eps^p_i, C_alg = return_map(eps~_i, eps^p_{i-1})
    G(r_i) = INT B^T (sigma_i - sigma^E_i) dV = 0        on the free dofs

``G = 0`` is self-equilibrium of the residual field -- the *first* half of
Melan's condition, arriving here as the equilibrium equation of the incremental
problem rather than as a constraint to search under, while the second half,
admissibility at every cycle point, is what the return map enforces. The
tangent is the consistent elastoplastic stiffness. The cycle is then repeated until the
*stress* history over it stops changing. That is the right convergence test and
not the plastic strain: a steadily ratchetting structure has a periodic stress
cycle with a plastic strain that grows without bound, so a test on the strain
would never pass there.

``eps~`` is an auxiliary variable and not the physical strain, which is worth
being explicit about: ``D^-1 sigma^E`` carries the elastic solution's thermal
eigenstrain with the opposite sign, so in plane strain its out-of-plane
component is exactly ``-alpha T`` and the constraint ``eps_33 = 0`` comes out
of the formulation rather than having to be imposed. The identity it rests on
is ``sigma = sigma^E + D (B r - eps^p)``, which needs ``sigma^E`` as a full
six-component stress -- the in-plane reduction would silently drop the
constraint.

B-bar is required here and must not be used with Melan
-----------------------------------------------------
This is a displacement-based incremental analysis with plastic flow that is
exactly incompressible, so it locks, and the locking is not a small error.
Measured on the thick cylinder at 8 cycle points, bisecting for the load factor
at which a cycle point stops equilibrating at all:

    tables                 refuses at      independent Koiter bound (lmm.py)
    b-bar                  1.00747 p_lim            1.00730 p_lim
    fully integrated       1.17933 p_lim            does not run

**With b-bar it lands on the Koiter bound of the same mesh to 0.02%**, which is
the first agreement here between a kinematic bounding method and an incremental
elastoplastic analysis, and the two are as independent as two answers to one
question get. There is no bound to compare the fully integrated column against,
because LMM needs b-bar and returns ``nan`` without it -- so that row is read
against the b-bar one, which is legitimate only as a measure of the locking.
Fully integrated it carries **17% more**, and nothing in the run looks wrong: it converges, the residual field is self-equilibrated, yield is
satisfied at round-off. The locked model genuinely does carry that load -- it
is simply not the structure anyone asked about.

Note the asymmetry with the rest of this package, which is easy to get
backwards. :mod:`~scion.shakedown.melan` must **not** have b-bar, because a
field self-equilibrated against ``B_bar`` is not self-equilibrated against
``B``. :mod:`~scion.shakedown.lmm` must have it, because its matched material
is incompressible. :mod:`~scion.shakedown.rsdm` does not care either way, since
its iteration never forms a plastic tangent -- the elastic operator is the real
one. This module is in the second group. Use two table objects on one mesh.

What it costs, and when to use which
------------------------------------
An incremental analysis gives up the thing RSDM exists for -- one factorisation
for the whole run -- and pays a tangent assembly and factorisation per Newton
iteration. Measured anyway, at 0.9 ``p_lim`` with 8 cycle points, against RSDM
to a relative change of 1e-8:

    n      points    this (s)   cycles  Newton    rsdm (s)   iters    ratio
    5         100      0.026         3      32       1.433     593     54.2x
    8         256      0.038         3      32       2.479     619     65.8x
    12        576      0.085         3      32       3.662     575     43.0x
    16       1024      0.136         3      32       5.732     578     42.3x
    22       1936      0.238         3      32      10.019     586     42.1x

**This is 42 to 66 times faster, and that is not the algorithmic verdict it
looks like.** The Newton count is *constant* -- 3 cycles, 8 points, ~1.3
iterations each -- while RSDM needs ~590 linear iterations, and RSDM's
iteration assembles the internal force at *every cycle point*. Counted:
**4745 assembly calls against 40**. So the ratio is a statement about assembly
cost in this implementation and not about the two methods, and quoting it as
one would be wrong.

What survives the correction: this is affordable at a load factor, and RSDM's
real advantage is the *sweep*, where one factorisation and a warm start serve
every load factor and every grid point. So use RSDM for the sweep and this for
the magnitude at a factor that sweep has already identified -- which is what
the two are for, rather than a cheaper and a dearer route to one answer.

What is verified
----------------
The sharp check is that the strip reproduces Bree's closed-form plastic strain
range **exactly** rather than approximately -- relative error at or below
9.3e-16 on eleven alternating-plasticity rays, against the penalty form's
0.084261 for an exact 1.0. Three structural properties come with it and none was built in:

* **the per-cycle plastic strain increment is uniform through the thickness**
  to 1e-14 in the ratchetting regime, which is what a periodic stress cycle
  forces and is not imposed anywhere;
* **the ratchet rate vanishes exactly at** ``lambda_rr`` on both branches --
  0.0 at and below, 4.0e-4 one part in ten thousand above -- which validates
  ``x y = 1``, the one boundary the Melan derivation in
  :mod:`~scion.shakedown.bree` cannot reach, and from the strain side rather
  than by bisection;
* **the ratchet rate itself has a closed form on the R2 branch**,
  ``r = 2 (lambda^2 x y - 1) sigma_Y / E'``, derived here and reproduced to
  1.1e-6 or better over fourteen rays with ``lambda^2 x y`` from 1.125 to 4.5
  and ``y/4x`` from 1.1 to 40 -- everywhere except ``y = 4x`` exactly, which is
  the branch boundary and is excluded by measurement rather than by caution.

See :func:`~scion.shakedown.bree.cyclic_strip` for the strip and
:func:`cyclic_steady_state` for the finite element path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import linalg as spla

from ..materials.notation import (
    from_mandel,
    mandel_to_voigt_stress,
    mandel_to_voigt_tangent,
    voigt_strain_to_mandel,
)
from ..materials.plasticity import PlasticState, get_criterion, get_isotropic, return_map
from .rsdm import _frobenius, _strain_range

__all__ = ["CyclicState", "cyclic_steady_state"]


class CyclicConvergenceError(RuntimeError):
    """The cycle did not become periodic, or a step did not equilibrate.

    Raised rather than returned, because the indicators are read off the last
    cycle and a non-periodic one makes them meaningless -- unlike RSDM, where
    a truncated run still carries a usable growth ratio.
    """


@dataclass
class CyclicState:
    """The asymptotic cycle with yield satisfied, and its indicators.

    Carries the same indicator names as :class:`~scion.shakedown.RSDMResult`
    and computes them from the same code, so the two are directly comparable:
    what differs is only how the state they describe was obtained.

    ``n_c`` is the number of stress components -- six in Mandel for the finite
    element path, one for the uniaxial strip, which is why nothing here
    hard-codes a notation.
    """

    elastic: np.ndarray = field(repr=False)        # (n_time, n_points, n_c)
    stress: np.ndarray = field(repr=False)         # (n_time, n_points, n_c)
    plastic_strain: np.ndarray = field(repr=False)  # (n_time, n_points, n_c)
    net_plastic_strain: np.ndarray = field(repr=False)   # (n_points, n_c)
    effective: np.ndarray = field(repr=False)      # (n_time, n_points)
    yield_stress: np.ndarray = field(repr=False)   # (n_points,)
    compliance: np.ndarray = field(repr=False)     # (n_c, n_c)
    weights: np.ndarray = field(repr=False)        # (n_points,)
    sigma_0: float = 1.0
    cycles: int = 0
    converged: bool = False
    newton_iterations: int = 0

    # -- the state ------------------------------------------------------

    @property
    def residual(self) -> np.ndarray:
        """``rho = sigma - sigma^E``, self-equilibrated at every cycle point."""
        return self.stress - self.elastic

    @property
    def constant_residual(self) -> np.ndarray:
        """The cycle mean of the residual field, and the whole of it at
        shakedown.

        The counterpart of :attr:`~scion.shakedown.RSDMResult.constant_residual`
        (``a_0/2``), reached by averaging rather than by closing a Fourier
        series.

        **The two do NOT agree, and expecting them to was a mistake worth
        recording.** With ``sigma_p = 0`` both methods are solving the same
        *problem* below ``lambda_sd``, so the fields looked like they had to
        match -- but the shakedown residual field is **not unique**. Any
        self-equilibrated field keeping every cycle point admissible will do,
        and the two methods reach different members of that set: measured on
        the thick cylinder at 0.9 ``p_lim``, ``|rho|_inf`` of 143.055 against
        RSDM's 143.129, differing by **2.73%**, and stable to every digit as
        RSDM's budget goes 1000 / 2000 / 4000 / 8000 iterations, so it is not
        an unconverged run.

        What *is* checkable, and is the right cross-validation, is that both
        members are admissible and that their difference is a self-equilibrated
        field carrying no yield violation of its own: yield violation 6.8e-16
        here and 1.2e-13 for RSDM, equilibrium residual 5.5e-11 and 1.1e-11,
        and the difference self-equilibrated to 5.7e-11. Two valid answers,
        not one answer and one error.

        The reason RSDM's is path dependent: once ``sigma_p`` reaches zero its
        update leaves ``a_0`` wherever the iteration had got to, so the field
        is a property of the iteration path. This one is a property of the load
        path, which is the physical answer -- but only for *that* load path.
        """
        return self.stress.mean(axis=0) - self.elastic.mean(axis=0)

    # -- verification ---------------------------------------------------

    @property
    def max_yield_violation(self) -> float:
        """``max (sigma_eff - sigma_Y) / sigma_Y`` over the cycle, clipped at
        zero.

        **This is the quantity that distinguishes the method from RSDM and the
        one to assert on.** The penalty iteration leaves it finite by
        construction -- that residue *is* its plastic strain rate -- while here
        it is a return-map tolerance, measured at round-off. A nonzero value
        means a step did not equilibrate, not that the material yielded.
        """
        over = (self.effective - self.yield_stress[None, :]) / self.sigma_0
        return float(max(0.0, over.max()))

    # -- indicators -----------------------------------------------------

    @property
    def plastic_strain_range(self) -> float:
        r"""``Delta eps^p``: the largest plastic strain range per cycle.

        The low-cycle-fatigue input, and **a magnitude here rather than an
        indicator**. Measured against
        :func:`~scion.shakedown.bree.plastic_strain_range_exact` on eleven
        alternating-plasticity rays: relative error at or below 9.3e-16, and
        independent of the cycle sampling (0.4000000000 at every ``n_t`` from
        8 to 96) and of the waveform, both of which the penalty form is not.

        A range, not an amplitude: halve it before handing it to anything in
        :mod:`scion.fatigue`, which is the factor of two that module is built
        around.
        """
        return _strain_range(self.plastic_strain)

    @property
    def ratchet_rate(self) -> float:
        r"""``r``: the net plastic strain accumulated over one asymptotic cycle.

        Exactly zero in the elastic, shakedown and alternating-plasticity
        regimes, and positive above ``lambda_rr``. Not thresholded: on the
        strip it reads 0.0 at ``lambda_rr`` and 4.0e-4 at
        ``1.0001 lambda_rr``.
        """
        return float(_frobenius(self.net_plastic_strain).max())

    @property
    def plastic_strain_scale(self) -> float:
        """``sigma_0 / (2 mu)``, a scale to read the two strains against.

        The last diagonal entry of the compliance is ``1/(2 mu)`` in Mandel --
        all three shear slots carry it -- and ``1/E'`` for the one-component
        strip, which is why it is indexed from the end.
        """
        return float(self.sigma_0 * self.compliance[-1, -1])

    @property
    def cyclic_amplitude(self) -> float:
        r"""``A``: the normalised amplitude of the cyclic residual stress.

        Defined on the Fourier coefficients in
        :attr:`~scion.shakedown.RSDMResult.cyclic_amplitude` as
        ``(1/sigma_0) (sum_k ||a_k||_W^2 + ||b_k||_W^2)^{1/2}``, and computed
        here by Parseval from the history itself,

            A = (1/sigma_0) ( 2 mean_t || rho(t) - rho_bar ||_W^2 )^{1/2}

        which is the *same* quantity summed over every harmonic rather than
        over the truncation. So the two are comparable, and a disagreement
        between them at shakedown would be a real one.
        """
        rho = self.residual
        centred = rho - rho.mean(axis=0, keepdims=True)
        w = self.weights[None, :, None]
        return np.sqrt(2.0 * float((w * centred ** 2).sum())
                       / centred.shape[0]) / self.sigma_0

    def regime(self, eta_p: float = 1.0e-9, eta_r: float = 1.0e-6) -> str:
        """Which of the four regimes.

        **Keyed on the plastic strain range, where**
        :meth:`~scion.shakedown.RSDMResult.regime` **is keyed on** ``A``, and
        that is a deliberate difference rather than an oversight: with yield
        enforced, ``Delta eps^p`` is *identically* zero below ``lambda_sd``
        rather than merely small, so the threshold is only there to absorb
        round-off. Measured across the boundary at ``x, y = 0.1, 2.0``:

            lambda/lambda_sd   0.99    0.9999   1.0    1.0001   1.001
            Delta eps^p/scale   0.0      0.0    0.0    2.0e-4   2.0e-3
            A                 2.4e-16 2.8e-16 2.8e-16 2.0e-5   2.0e-4

        Exactly zero on three rows of the first, round-off on three rows of
        the second, and a response ten times sharper above the boundary. ``A``
        remains available and remains comparable with RSDM's; it is simply not
        the better test when the better one exists. Both thresholds are
        relative to :attr:`plastic_strain_scale`, since both are absolute
        strains.

        **The two defaults differ by three orders, and that is measured rather
        than cautious.** ``Delta eps^p`` is identically zero below
        ``lambda_sd``, so ``eta_p`` only has to clear round-off. ``r`` is zero
        in alternating plasticity by a *cancellation* over the cycle, and how
        exactly that cancels depends on the problem: on the uniaxial strip the
        antisymmetry is exact and ``r`` lands at 1e-16 of the strain scale,
        while on a finite element mesh the same state leaves
        ``r/scale = 3.6e-8`` -- measured on the thin annulus of a Bree-type
        ray, where ``Delta eps^p = 2.2e-4`` and the state is plainly
        alternating. An ``eta_r`` of 1e-9 therefore reports ratchetting for a
        cycle that does not ratchet, which is the unsafe direction, so the
        default is 1e-6 as in :meth:`~scion.shakedown.RSDMResult.regime`.

        ``elastic`` and ``shakedown`` are separated by whether a residual field
        developed at all, since both carry no plastic strain in the asymptotic
        cycle.
        """
        scale = self.plastic_strain_scale
        if self.ratchet_rate > eta_r * scale:
            return "ratchetting"
        if self.plastic_strain_range > eta_p * scale:
            return "alternating"
        if np.abs(self.constant_residual).max() <= 1e-12 * self.sigma_0:
            return "elastic"
        return "shakedown"

    def __repr__(self) -> str:
        return (f"CyclicState(regime={self.regime()}, "
                f"dEps_p={self.plastic_strain_range:.4g}, "
                f"r={self.ratchet_rate:.4g}, cycles={self.cycles}, "
                f"converged={self.converged})")


# ---------------------------------------------------------------------------

def cyclic_steady_state(elastic_history: np.ndarray, tables, free_dofs,
                        C_mandel, sigma_y, *, criterion=None, isotropic=None,
                        max_cycles: int = 50, tol: float = 1.0e-8,
                        newton_tol: float = 1.0e-9, newton_max: int = 30,
                        verbose: bool = False) -> CyclicState:
    """The asymptotic cycle of a prescribed load history, yield enforced.

    Takes exactly the arguments :func:`~scion.shakedown.rsdm` takes, so it is a
    drop-in alternative on the same problem -- which is what makes comparing
    them a measurement rather than a port.

    Parameters
    ----------
    elastic_history : ``(n_time, n_points, 6)`` Mandel elastic stress over the
        cycle, already scaled by the load factor. Must be the *full* six
        components; see the module docstring on why the in-plane reduction
        loses the out-of-plane constraint.
    tables, free_dofs, C_mandel : the mesh, its unconstrained degrees of
        freedom and the elastic stiffness in Mandel form.
    sigma_y : scalar or ``(n_points,)`` yield stress.
    criterion, isotropic : default to von Mises and perfect plasticity.
        Another criterion or a hardening law is accepted and **neither is
        validated here** -- every measured result in this module is von Mises
        and elastic-perfectly-plastic, as is shakedown theory everywhere else
        in this package. The yield measure behind
        :attr:`CyclicState.max_yield_violation` does follow whichever criterion
        is passed, so at least that diagnostic will not lie about a
        Drucker-Prager run.
    max_cycles : cycles to repeat before giving up. Three is typical and
        fifteen is a lot; a case that wants more is usually sitting within
        round-off of a regime boundary.
    tol : relative change in the *stress* history between successive cycles.
    newton_tol : relative residual force at one cycle point.

    Raises
    ------
    CyclicConvergenceError
        if a step does not equilibrate or the cycle does not become periodic.
        Returning an unconverged state would be worse than failing: the
        indicators are read off the last cycle and mean nothing if it is not
        the asymptotic one.
    """
    history = np.asarray(elastic_history, dtype=float)
    if history.ndim != 3 or history.shape[-1] != 6:
        raise ValueError(
            "elastic_history must be (n_time, n_points, 6) Mandel stress; "
            f"got {history.shape}")
    n_time, n_points, _ = history.shape
    yield_stress = np.broadcast_to(
        np.asarray(sigma_y, dtype=float), (n_points,)).copy()
    free = np.asarray(free_dofs, dtype=np.int64)
    n_dof = tables.vector_dofs.n_total
    nv = tables.n_voigt
    C = np.asarray(C_mandel, dtype=float)
    D_inv = np.linalg.inv(C)

    crit = criterion if criterion is not None else get_criterion("j2")
    # A point-varying yield stress cannot be passed through the hardening
    # interface, which carries one scalar kappa and is asked for it on the
    # *plastic subset* of the points, so an array would not even line up. It
    # enters instead as a pointwise rescaling of the strain, which for a
    # criterion positively homogeneous of degree one in (sigma, kappa) -- J2
    # and Drucker-Prager both are -- is exact rather than approximate: the
    # closest-point projection commutes with a positive scaling, so
    #
    #     sigma(eps; sy) = sy * sigma(eps / sy; 1)
    #
    # with eps^p scaling the same way and the algorithmic tangent, being of
    # degree zero, unchanged. The whole solve therefore runs in the scaled
    # variables and only the reported fields are scaled back. Verified: a
    # uniform field reproduces the unscaled path bit for bit.
    uniform = float(np.mean(yield_stress))
    sy = yield_stress[:, None]                             # (n_points, 1)
    varying = not np.allclose(yield_stress, yield_stress[0])
    if varying and isotropic is not None:
        raise NotImplementedError(
            "a point-varying yield stress is handled by rescaling the strain, "
            "which is exact only because perfect plasticity has no internal "
            "length in kappa. With a hardening law, kappa(alpha) is not "
            "homogeneous in the scaling, so the two do not commute. Run one "
            "call per material instead.")
    hard = (isotropic if isotropic is not None
            else get_isotropic("perfect", kappa_0=1.0))
    kappa_unit = 1.0

    # eps~ = D^-1 sigma^E + B r, so the load enters the return map as a strain
    eps_elastic = (history @ D_inv.T) / sy[None]           # (n_time, n_p, 6)
    kappa = hard.kappa(np.zeros(n_points))

    # the load enters only as INT B^T sigma^E dV, which is the nodal force the
    # elastic stress equilibrates. The *largest* of those over the cycle is the
    # reference the Newton residual is measured against: a per-point one is
    # useless at a cycle point carrying no load, which a program starting from
    # zero always has, and a floor built from the yield stress and the volume
    # is not a force at all -- it came out 3.2x the real scale on the cylinder.
    external = np.array([_internal_force(tables, history[t], n_dof, nv)
                         for t in range(n_time)])
    reference = max(float(np.abs(external[:, free]).max()), 1e-300)

    state = PlasticState.zeros(n_points)
    disp = np.zeros(n_dof)
    sigma_cycle = np.zeros((n_time, n_points, 6))
    eps_p_cycle = np.zeros((n_time, n_points, 6))
    effective = np.zeros((n_time, n_points))
    previous = None
    newton_total = 0
    converged = False
    cycle = 0
    change = np.inf

    for cycle in range(1, max_cycles + 1):
        eps_p_start = state.eps_p.copy()
        for t in range(n_time):
            trial_state = state
            f_ext = external[t]
            for it in range(1, newton_max + 1):
                eps = eps_elastic[t] + voigt_strain_to_mandel(
                    tables.strains_voigt(disp), tables.dim) / sy
                rm = return_map(eps, trial_state, crit, hard, C)
                # back to true stress before assembly; the tangent is of
                # degree zero in the scaling and needs no correction
                K, f_int = tables.stiffness_and_force(
                    mandel_to_voigt_stress(sy * rm.sigma, tables.dim),
                    mandel_to_voigt_tangent(rm.C_alg, tables.dim), n_dof)
                G = (f_int - f_ext)[free]
                newton_total += 1
                if np.abs(G).max() <= newton_tol * reference:
                    break
                step = spla.spsolve(K.tocsr()[free][:, free].tocsc(), -G)
                disp[free] += step
            else:
                raise CyclicConvergenceError(
                    f"cycle {cycle}, cycle point {t}: the step did not "
                    f"equilibrate in {newton_max} Newton iterations, "
                    f"|G|/|f| = {np.abs(G).max() / reference:.2e}. The usual "
                    f"cause is not the solver: past the limit load of the "
                    f"discrete model no admissible stress field equilibrates "
                    f"this cycle point at all, and that *is* the answer -- the "
                    f"structure collapses rather than reaching an asymptotic "
                    f"cycle. On the thick cylinder with b-bar tables the "
                    f"refusal lands at 1.00747 p_lim against an independent "
                    f"Koiter bound of 1.0073 on the same mesh. If the tables "
                    f"are fully integrated, suspect volumetric locking first: "
                    f"see this module's docstring.")
            state = rm.state
            sigma_cycle[t] = sy * rm.sigma
            eps_p_cycle[t] = sy * rm.state.eps_p
            # the criterion's own equivalent stress, not a hard-coded von
            # Mises: ``f = sigma_eq - kappa`` for J2, so adding kappa back
            # recovers it, and a caller who passes Drucker-Prager gets its
            # measure rather than a silently wrong one. Evaluated on the
            # scaled stress against a unit kappa and scaled back, which is the
            # same number by homogeneity and carries the local yield stress.
            effective[t] = yield_stress * (
                kappa_unit + crit.f(from_mandel(rm.sigma), kappa))

        if previous is not None:
            # pointwise against the local yield stress, which reduces to the
            # old global measure exactly when the field is uniform
            change = np.abs((sigma_cycle - previous) / sy[None]).max()
            if verbose:
                print(f"      cycle {cycle:3d}: |dsigma|/sigma_Y = "
                      f"{change:.3e}")
            if change <= tol:
                converged = True
                break
        previous = sigma_cycle.copy()

    if not converged:
        raise CyclicConvergenceError(
            f"the stress cycle was still moving after {max_cycles} cycles, by "
            f"{change:.2e} of the yield stress against a tolerance of "
            f"{tol:.0e}. "
            f"Either the state is not yet asymptotic -- raise max_cycles -- or "
            f"the load factor is past the limit load, where no periodic state "
            f"exists at all and the displacement grows without bound.")

    return CyclicState(
        elastic=history, stress=sigma_cycle, plastic_strain=eps_p_cycle,
        net_plastic_strain=sy * (state.eps_p - eps_p_start), effective=effective,
        yield_stress=yield_stress, compliance=D_inv,
        weights=np.asarray(tables.w, dtype=float).ravel(),
        sigma_0=uniform, cycles=cycle, converged=True,
        newton_iterations=newton_total)


def _internal_force(tables, sigma_mandel, n_dof, nv):
    """``INT B^T sigma dV``: the load that the elastic stress equilibrates."""
    _, f = tables.stiffness_and_force(
        mandel_to_voigt_stress(sigma_mandel, tables.dim),
        np.zeros((sigma_mandel.shape[0], nv, nv)), n_dof)
    return f

