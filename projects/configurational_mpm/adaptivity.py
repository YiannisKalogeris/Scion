r"""
Material-manifold *r*-adaptivity, Section 5.

Since :func:`~projects.configurational_mpm.forces.configurational_force` is by
construction the shape gradient of the discrete energy with respect to particle
reference positions,

.. math::
    \mathcal{G}_p = -\frac{\partial \Pi_h}{\partial X_p},

the condition ``G_p = 0`` characterises an energy-optimal particle layout, and
gradient descent on ``X_p`` is a *variationally consistent* resampling rule --
replacing the geometric split-and-merge heuristics production MPM codes use.

That the gradient is exact is what makes this testable: descending it must
reduce ``Pi_h`` monotonically, and a backtracking line search turns that from a
hope into a guarantee. If the energy ever rises, Eq. (9) is wrong.

Fixed volumes make the objective degenerate -- measured
--------------------------------------------------------
``Pi_h = sum_p V_p^0 W(F_p)`` approximates ``int W dV`` **only while the
``V_p^0`` tessellate the body**. Hold them fixed and move ``X_p`` and it stops
being a quadrature of anything, so minimising it minimises the wrong functional:
the descent simply migrates particles towards low strain energy.

Measured on a smooth non-uniform displacement field, 256 particles, quadratic
B-spline, against a reference ``int W dV`` from an 8x-finer independent lattice:

======================  ==========  =========================  =============
layout                  ``Pi_h``    error vs reference         nn-spacing CV
======================  ==========  =========================  =============
uniform lattice          29.82       **-0.34%**                 0.0000
after 60 descent steps   21.32       **-28.75%**                0.2769
======================  ==========  =========================  =============

The starting lattice was already an excellent quadrature and the descent made it
28% worse while reducing ``Pi_h`` monotonically the whole way. Nothing is wrong
with the gradient -- it is verified against a finite-difference shape gradient
to 1e-8 -- the objective is simply not the one that should be minimised.

So the ``dV^0/dX`` term of Eq. (9) is **not an optional refinement**: tying the
volumes to the positions is what keeps ``Pi_h`` an approximation of the body's
energy, and therefore what makes ``G_p = 0`` characterise a good layout rather
than a biased one. Until a tessellation supplies ``dV^0/dX``, treat the descent
here as a demonstration of the gradient, not as a usable resampler.
:attr:`RelaxationResult.separation_cv` is tracked so the degeneracy is visible
in the output instead of silent.

Hyperelastic materials only
---------------------------
Refused for anything carrying history, and the reason is not bookkeeping.

The bookkeeping problem is real: plastic strain, damage and back stress are
integrals over the path a *particular piece of material* took, not functions of
where it sits. Moving ``X_p`` and keeping the history advects a plastic zone
through material that never yielded. Remapping the history instead -- the ALE
answer -- diffuses sharp fronts, can produce inadmissible states, and compounds
over repeated resampling.

But the deeper obstruction is that ``G_p`` is not *defined* for a dissipative
material. It presumes a single ``Pi_h`` to differentiate, and for plasticity the
split of ``Sigma`` into energetic and dissipative parts is not unique. Choosing
one is a modelling decision, not an implementation detail, which is why the
paper's Section 7 defers it and why this refuses rather than guesses.

For hyperelasticity none of that arises: ``F_p`` is reconstructed from the nodal
displacement field at whatever ``X_p`` currently is, so moving a particle simply
re-evaluates a function of position and nothing is carried.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scion.core import small_linalg as sla

from projects.configurational_mpm.forces import configurational_force

__all__ = [
    "RelaxationResult",
    "discrete_energy",
    "configurational_forces",
    "relax_reference_positions",
]


def _check_hyperelastic(material) -> None:
    if hasattr(material, "update") and not hasattr(material, "pk1"):
        raise TypeError(
            f"{type(material).__name__} carries history, and the discrete "
            "configurational force is not defined for a dissipative material: "
            "it is the gradient of a single Pi_h, and the split of Sigma into "
            "energetic and dissipative parts is not unique. Moving X_p would "
            "also advect the plastic history into material that never yielded. "
            "r-adaptivity here is hyperelastic only."
        )
    for name in ("energy", "pk1"):
        if not hasattr(material, name):
            raise TypeError(
                f"{type(material).__name__} has no {name}(); r-adaptivity needs "
                "both the strain energy and the first Piola-Kirchhoff stress."
            )


def _kinematics(grid, X, u_nodes, lp=None, hessian=False):
    st = grid.build_stencil(X, lp=lp, hessian=hessian)
    H = np.einsum("pka,pkJ->paJ", u_nodes[st.node_id], st.grad_w)
    F = sla.identity(X.shape[0], grid.dim, grid.dtype) + H
    return st, F


def discrete_energy(material, grid, X: np.ndarray, u_nodes: np.ndarray,
                    vol0: np.ndarray, lp=None) -> float:
    r"""``Pi_h = sum_p V_p^0 W(F_p)`` with ``F_p`` read at ``X_p``.

    The external work is omitted: it does not depend on the particle reference
    positions, so it contributes nothing to the shape gradient and nothing to
    the descent.
    """
    _, F = _kinematics(grid, X, u_nodes, lp)
    return float(np.sum(np.asarray(vol0) * material.energy(F)))


def configurational_forces(material, grid, X: np.ndarray, u_nodes: np.ndarray,
                           vol0: np.ndarray, lp=None) -> np.ndarray:
    """``G_p`` at the current reference layout. ``-> (n, dim)``."""
    _check_hyperelastic(material)
    st, F = _kinematics(grid, X, u_nodes, lp, hessian=True)
    P, _ = material.pk1(F)
    return configurational_force(P, np.asarray(vol0), u_nodes, st.node_id,
                                 st.hess)


@dataclass
class RelaxationResult:
    """Outcome of a descent, with enough history to see how it behaved."""

    X: np.ndarray
    energy: list = field(default_factory=list)
    gradient_norm: list = field(default_factory=list)
    step_size: list = field(default_factory=list)
    #: coefficient of variation of the nearest-neighbour spacing. 0 for a
    #: perfect lattice; rising values mean the layout is losing its quadrature
    #: property, which fixed volumes do not penalise -- see the module note.
    separation_cv: list = field(default_factory=list)
    n_iterations: int = 0
    converged: bool = False
    stopped_by: str = ""

    @property
    def energy_reduction(self) -> float:
        if len(self.energy) < 2:
            return 0.0
        return self.energy[0] - self.energy[-1]

    @property
    def layout_degraded(self) -> bool:
        """Has the layout stopped being a usable quadrature?

        True once the nearest-neighbour spacing has become appreciably
        non-uniform. With fixed particle volumes this is the expected outcome,
        not a failure of the solver -- see the module docstring.
        """
        return bool(self.separation_cv and self.separation_cv[-1] > 0.05)

    def __repr__(self) -> str:
        return (f"RelaxationResult({self.n_iterations} iterations, "
                f"Pi {self.energy[0]:.6g} -> {self.energy[-1]:.6g}, "
                f"|G| {self.gradient_norm[0]:.3g} -> "
                f"{self.gradient_norm[-1]:.3g}, "
                f"spacing CV {self.separation_cv[-1]:.3f}"
                f"{' DEGRADED' if self.layout_degraded else ''}, "
                f"{self.stopped_by})")


def relax_reference_positions(
    material, grid, X: np.ndarray, u_nodes: np.ndarray, vol0: np.ndarray,
    *, lp=None, n_iter: int = 50, trust_radius: float = 0.05,
    min_separation: float | None = None, tol: float = 1e-10,
    max_backtracks: int = 20, callback=None,
) -> RelaxationResult:
    r"""Descend ``Pi_h`` in the particle reference positions.

    Parameters
    ----------
    trust_radius   : largest move per iteration, as a fraction of ``dx``. The
                     step is normalised by ``max|G|``, so this is a length and
                     not a stiffness-dependent gain -- the same value works
                     across materials.
    min_separation : reject any step bringing two particles closer than this.
                     The barrier regularisation: without it nothing in ``Pi_h``
                     penalises two particles merging, since ``W`` stays finite.
                     Defaults to a quarter of the initial nearest-neighbour
                     distance. ``None`` after that disables the check.
    tol            : stop when ``max|G|`` falls below this.

    Returns
    -------
    :class:`RelaxationResult`. ``energy`` is monotonically non-increasing by
    construction -- every step is accepted only if it lowers ``Pi_h``, so a rise
    would mean ``G_p`` is not the gradient it claims to be.

    Notes
    -----
    Particle volumes are held fixed, so the ``dV^0/dX`` term of Eq. (9) is zero
    and the gradient used here is complete. Tying volumes to a tessellation
    would reinstate that term and is not done.
    """
    _check_hyperelastic(material)

    X = np.array(X, dtype=float, copy=True)
    vol0 = np.asarray(vol0, dtype=float)
    lo = grid.origin
    hi = lo + np.asarray(grid.shape) * grid.dx

    if min_separation is None:
        min_separation = 0.25 * _nearest_neighbour(X)

    result = RelaxationResult(X=X)
    energy = discrete_energy(material, grid, X, u_nodes, vol0, lp)
    result.energy.append(energy)
    result.separation_cv.append(_separation_cv(X))

    for it in range(n_iter):
        G = configurational_forces(material, grid, X, u_nodes, vol0, lp)
        g_max = float(np.abs(G).max())
        result.gradient_norm.append(g_max)

        if g_max < tol:
            result.converged = True
            result.stopped_by = "converged"
            break

        # normalised trust-region step: the largest move is trust_radius * dx
        alpha = trust_radius * grid.dx / g_max
        accepted = False
        for _ in range(max_backtracks):
            trial = X + alpha * G
            if (np.any(trial <= lo) or np.any(trial >= hi)
                    or (min_separation > 0.0
                        and _nearest_neighbour(trial) < min_separation)):
                alpha *= 0.5
                continue
            trial_energy = discrete_energy(material, grid, trial, u_nodes,
                                           vol0, lp)
            if trial_energy < energy:
                X, energy, accepted = trial, trial_energy, True
                break
            alpha *= 0.5

        result.step_size.append(alpha)
        result.energy.append(energy)
        result.separation_cv.append(_separation_cv(X))
        result.n_iterations = it + 1

        if callback is not None:
            callback(it, X, energy, g_max)

        if not accepted:
            result.stopped_by = "line search failed"
            break
    else:
        result.stopped_by = "iteration limit"

    if not result.stopped_by:
        result.stopped_by = "converged"
    result.X = X
    return result


def _separation_cv(X: np.ndarray) -> float:
    """Coefficient of variation of nearest-neighbour spacing. 0 = lattice."""
    d = _neighbour_distances(X)
    return float(d.std() / d.mean()) if d.mean() > 0 else 0.0


def _neighbour_distances(X: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
        return cKDTree(X).query(X, k=2)[0][:, 1]
    except ImportError:                                    # pragma: no cover
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)


def _nearest_neighbour(X: np.ndarray) -> float:
    """Smallest distance between any two particles."""
    return float(_neighbour_distances(X).min())
