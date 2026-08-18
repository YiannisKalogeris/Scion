r"""
The Linear Matching Method: the shakedown upper bound.

Koiter's kinematic theorem bounds the shakedown factor from above by any
admissible cyclic mechanism:

    lambda_UB = sum_n INT sigma_y ebar^n dV  /  sum_n INT sigma^E(t_n) : eps^n dV

The difficulty is that minimising it over mechanisms is a nonlinear programming
problem. Ponter and Carter's insight, generalised by Ponter and Engelhardt, is
that the minimisation can be carried out by a sequence of *linear* problems:
at each iteration a linear material is chosen whose response at the current
strain rate lands exactly on the yield surface, the linear problem is solved,
and the resulting strain rate becomes the next iterate. Each pass gives a valid
upper bound and the sequence decreases monotonically.

The formulation implemented here is Chen (2010), section 3.1 and equation 16,
which is Ponter and Engelhardt's global minimisation for the shakedown limit.

The iteration
-------------
With effective strain rates ``ebar^n`` known at each of the ``N`` load vertices:

    matching       mu^n   = sigma_y / (3 ebar^n)                          (6)
    harmonic mean  1/mubar = sum_n 1/mu^n                                 (9b)
    initial stress sigma_init = mubar sum_n sigma^E'(t_n) / mu^n          (9b)
    linear solve   find du with  INT B^T [2 mubar eps(du) - sigma_init] dV = 0
    residual       rhobar' = 2 mubar eps'(du) - sigma_init                (9a)
    increments     deps'^n = (1/(2 mu^n)) [sigma^E'(t_n) + rhobar']       (8)
    update         ebar^n = sqrt(2/3 deps'^n : deps'^n)

Everything is deviatoric, and the compatible strain increment summed over the
cycle is what ties the vertices together: one displacement field ``du`` has to
serve all of them, which is exactly the kinematic admissibility the theorem
requires.

The linear problem is incompressible, and that is not negotiable
----------------------------------------------------------------
Equations 8 and 9 both carry ``eps_kk = 0`` explicitly. The matched material has
a shear modulus and no bulk modulus at all, so every linear solve in the
iteration is an exactly incompressible one -- the regime where a bilinear
quadrilateral with full integration locks hard enough to be useless. Measured on
a cantilever, the tip deflection at ``nu = 0.4999`` comes out at 1.9% of its
value at ``nu = 0.3``; with the mean-dilatation treatment it settles at 82%,
which is the correct plane-strain answer.

So :class:`~scion.discretization.fem.tables.QuadratureTables` must be built
with ``bbar=True``, and this module refuses tables that are not unless told
explicitly to proceed. A locked mesh does not fail visibly here: it produces a
stiffer mechanism, a larger dissipation integral, and an upper bound that is too
high while still decreasing monotonically and looking converged.

Incompressibility is imposed by a penalty bulk modulus rather than a multiplier
field, which the mean-dilatation B matrix makes stable.

Which way the errors point
--------------------------
The opposite way from Melan, and this is the reason to have both. Every
restriction on the *mechanism* -- a coarse mesh, an iteration stopped early --
raises the upper bound, so this side is safe when it is pessimistic and the
bound is only as good as the mesh. Running the two together brackets the true
factor from both sides on the same discretisation, which is worth more than
either number alone.

The residual field is reusable
------------------------------
:attr:`LMMResult.residual_stress` is self-equilibrated by construction, and it
is the field the iteration converged onto -- a far better direction than any
eigenstrain guess. Feeding it to
:class:`~scion.shakedown.basis.ResidualBasis` is exactly the
"basis vectors from the differences of intermediate stresses during equilibrium
iteration" the lower-bound literature recommends, and it makes the two methods
share work rather than duplicate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import linalg as spla

from ..materials.notation import mandel_to_voigt_stress, mandel_to_voigt_tangent
from ..materials.notation import voigt_strain_to_mandel

__all__ = ["LMMResult", "linear_matching"]

#: Mandel image of the identity: the trace direction.
_TRACE = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _deviator(a: np.ndarray) -> np.ndarray:
    """Deviatoric part of a batch of Mandel vectors."""
    trace = a[:, :3].sum(axis=1)
    return a - (trace / 3.0)[:, None] * _TRACE


def _effective(strain: np.ndarray) -> np.ndarray:
    """von Mises effective strain ``sqrt(2/3 e:e)`` of a deviatoric Mandel field."""
    return np.sqrt(2.0 / 3.0 * np.einsum("pi,pi->p", strain, strain))


@dataclass
class LMMResult:
    """Outcome of a linear matching run."""

    factor: float                       # the final upper bound
    history: list = field(repr=False)   # the monotone sequence
    iterations: int
    converged: bool
    monotone: bool
    residual_stress: np.ndarray = field(repr=False)   # (n_points, 6) Mandel
    effective_strain: np.ndarray = field(repr=False)  # (n_vertices, n_points)

    @property
    def reduction(self) -> float:
        """How far the bound fell from the first iterate. A run that barely
        moved either started at the answer or never got going."""
        return (self.history[0] / self.factor) if self.history else float("nan")

    def __repr__(self) -> str:
        return (f"LMMResult(factor={self.factor:.6g}, "
                f"iterations={self.iterations}, "
                f"converged={self.converged}, monotone={self.monotone}, "
                f"reduction={self.reduction:.3f}x)")


def linear_matching(vertex_stresses: np.ndarray, tables, free_dofs,
                    sigma_y, max_iter: int = 100, tol: float = 1.0e-6,
                    penalty: float = 1.0e3, strain_floor: float = 1.0e-9,
                    allow_locking: bool = False,
                    verbose: bool = False) -> LMMResult:
    """Upper bound on the shakedown factor by linear matching.

    Parameters
    ----------
    vertex_stresses : ``(n_vertices, n_points, 6)`` Mandel elastic stresses, from
        :meth:`~scion.shakedown.elastic.ElasticEnvelope.vertex_stresses`.
        A single vertex gives limit analysis, exactly as for Melan.
    tables : quadrature tables built with ``bbar=True``; see the module
        docstring for why that is required rather than advised.
    free_dofs : the unconstrained displacement degrees of freedom. The mechanism
        satisfies the same homogeneous boundary conditions as the structure.
    sigma_y : yield stress, scalar or one value per point.
    penalty : bulk modulus as a multiple of the local shear modulus. Large
        enough to impose incompressibility, small enough not to wreck the
        conditioning; 1000 is about ``nu = 0.4995``.
    strain_floor : effective strains below this fraction of the mean are floored
        before the matching divides by them. Rigid regions have zero strain rate
        and would otherwise demand an infinite modulus. It is a numerical
        device, not part of the method -- if the answer moves when it changes,
        the mechanism is not resolved.

    Returns an :class:`LMMResult`. The sequence should decrease monotonically;
    :attr:`LMMResult.monotone` reports whether it did, because a rise means the
    floor is active somewhere that matters or the linear solves are not
    converging, and either way the bound is no longer trustworthy.
    """
    if not getattr(tables, "bbar", False) and not allow_locking:
        raise ValueError(
            "linear matching solves an exactly incompressible linear problem at "
            "every iteration, and full integration locks there -- the bound "
            "comes out too high while still looking converged. Build the "
            "tables with QuadratureTables(..., bbar=True), or pass "
            "allow_locking=True if you are deliberately measuring the effect.")

    vertex_stresses = np.asarray(vertex_stresses, dtype=float)
    n_vertices, n_points, _ = vertex_stresses.shape
    yield_stress = np.broadcast_to(np.asarray(sigma_y, dtype=float),
                                   (n_points,))
    free = np.asarray(free_dofs, dtype=np.int64)
    n_dof = tables.vector_dofs.n_total
    nv = tables.n_voigt
    weights = tables.w.ravel()

    deviatoric = np.stack([_deviator(s) for s in vertex_stresses])

    # start from a uniform modulus: one elastic solve with the same stiffness
    # everywhere, which is what an unweighted first guess amounts to
    effective = np.ones((n_vertices, n_points))

    history: list[float] = []
    residual = np.zeros((n_points, 6))
    converged = False

    for iteration in range(1, max_iter + 1):
        floor = strain_floor * max(float(effective.mean()), 1e-300)
        mu = yield_stress[None, :] / (3.0 * np.maximum(effective, floor))

        mu_bar = 1.0 / np.sum(1.0 / mu, axis=0)                    # (n_points,)
        sigma_init = mu_bar[:, None] * np.einsum(
            "vpi,vp->pi", deviatoric, 1.0 / mu)

        # linear solve with the matched, near-incompressible material
        bulk = penalty * mu_bar
        C = (2.0 * mu_bar[:, None, None] * np.eye(6)
             + (bulk - 2.0 * mu_bar / 3.0)[:, None, None]
             * np.outer(_TRACE, _TRACE))
        C_v = mandel_to_voigt_tangent(C, tables.dim)
        K, _ = tables.stiffness_and_force(np.zeros((n_points, nv)), C_v, n_dof)
        _, f_init = tables.stiffness_and_force(
            mandel_to_voigt_stress(sigma_init, tables.dim),
            np.zeros((n_points, nv, nv)), n_dof)

        du = np.zeros(n_dof)
        du[free] = spla.spsolve(K.tocsr()[free][:, free].tocsc(), f_init[free])

        eps = voigt_strain_to_mandel(tables.strains_voigt(du), tables.dim)

        # The field the linear solve made self-equilibrated is C:eps(du) minus
        # the initial stress, *including* the volumetric part. Taking the
        # deviator of it is not the same thing: the penalty bulk modulus is
        # large by design, so (K - 2mu/3) tr(eps) is the pressure that enforces
        # incompressibility, not a small remainder. Dropping it leaves a field
        # that is out of equilibrium by order one -- which matters because this
        # is the field the lower bound reuses.
        residual = np.einsum("pij,pj->pi", C, eps) - sigma_init
        residual_dev = _deviator(residual)

        increments = ((deviatoric + residual_dev[None, :, :])
                      / (2.0 * mu)[:, :, None])
        effective = np.stack([_effective(e) for e in increments])

        dissipation = float(np.sum(weights * (yield_stress * effective).sum(0)))
        work = float(np.sum(weights * np.einsum(
            "vpi,vpi->p", deviatoric, increments)))
        if work <= 0.0:
            raise RuntimeError(
                f"the mechanism does no work against the elastic stress at "
                f"iteration {iteration}; the load domain may be entirely "
                f"self-equilibrated, or the mechanism has collapsed to nothing")

        bound = dissipation / work
        history.append(bound)
        if verbose:
            print(f"    iteration {iteration:3d}: upper bound = {bound:.8g}")

        if iteration > 1 and abs(history[-2] - bound) <= tol * bound:
            converged = True
            break

    monotone = all(b <= a * (1.0 + 1e-9) for a, b in zip(history, history[1:]))
    return LMMResult(factor=history[-1], history=history,
                     iterations=len(history), converged=converged,
                     monotone=monotone, residual_stress=residual,
                     effective_strain=effective)
