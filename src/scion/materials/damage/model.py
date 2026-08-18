r"""
The damage model: the three axes behind one interface.

    model = DamageModel(E=30e3, nu=0.2,
                        equivalent="modified-von-mises",
                        equivalent_kwargs={"k": 10.0},
                        evolution=ExponentialSoftening(1e-4, alpha=0.99,
                                                       beta=500.0),
                        split="spectral")

Same boundary as :class:`~scion.materials.plasticity.model.PlasticityModel`
and the same interface -- :class:`InelasticMaterial`, so ``update`` returns
``(sigma_voigt, C_voigt, result)`` and the model drops straight into
:class:`~scion.discretization.fem.nonlinear_assembler.ElastoplasticAssembler`
with no new assembly code. Inside it is full 3-D tensors; outside it is Voigt in
the element's own dimension.

Damage is not plasticity
------------------------
The stress here is an *explicit* function of the current strain and the history
variable ``kappa``, with no local Newton solve: there is no return map, because
there is nothing to project. What makes the model path-dependent is only
``kappa = max over history of eps_eq``, which is why the whole state is one
scalar per point rather than a tensor and a scalar.

The consequence is visible in unloading. A damaged point unloads along the
secant to the origin -- reduced stiffness, no permanent strain -- where a
plastic point unloads on the original elastic slope and keeps one. Any test that
confuses the two will pass for the wrong reason on the loading branch and fail
on the first reversal.

The tangent, and why it is unsymmetric
--------------------------------------
    ``sigma = (1 - D) sigma_plus + sigma_minus``

    ``C = (1 - D) C_plus + C_minus  -  (dD/dkappa) sigma_plus (x) deps_eq/deps``

The rank-one term appears only while loading (``eps_eq > kappa``) and vanishes
on unloading, leaving the secant. It is also generally **unsymmetric**, because
``sigma_plus`` and ``deps_eq/deps`` are different tensors -- the model has a
non-associated structure in the same sense as non-associative plastic flow.
:attr:`DamageModel.symmetric_tangent` reports when it happens to be symmetric,
which is exactly the Simo-Ju energy measure with no split, where the two
tensors are parallel.

Dropping the rank-one term gives the secant method: still convergent, linearly,
and the only symptom is the iteration count. Check it with
:func:`~scion.solvers.tangent_check.check_tangent`.

Driving damage from somewhere else
----------------------------------
``update(..., driver=...)`` replaces the local ``eps_eq`` in the history update
with an externally supplied field, which is what
:class:`~scion.multiphysics.gradient_damage.GradientDamageProblem` passes
for the nonlocal equivalent strain. The rank-one term then disappears -- at
fixed ``driver`` the damage does not depend on the local strain at all -- so the
momentum problem becomes *linear* and symmetric. That is the whole reason the
regularised problem is cheaper per iteration than the local one it fixes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..notation import voigt_strain_to_tensor
from ..plasticity.model import InelasticMaterial
from ..splits import NoSplit, Split, get_split, voigt_stress, voigt_tangent
from .equivalent import EquivalentStrain, get_equivalent_strain
from .evolution import DamageLaw, get_damage_law

__all__ = ["DamageState", "DamageResult", "DamageModel", "GradientDamageModel"]

#: rows of the 3-D Voigt ordering kept in plane strain, [s11, s22, s12]
_PLANE_STRAIN = [0, 1, 5]


@dataclass
class DamageState:
    """The committed history variable at a set of quadrature points.

    One scalar per point. ``kappa`` starts at the damage threshold rather than
    at zero so that the loading function ``f = eps_eq - kappa`` is non-positive
    in the virgin state; starting from zero would make every point report
    itself as loading on the first increment.
    """

    kappa: np.ndarray

    @classmethod
    def zeros(cls, n: int, kappa_0: float) -> "DamageState":
        return cls(np.full(int(n), float(kappa_0)))

    def copy(self) -> "DamageState":
        return DamageState(self.kappa.copy())

    def __repr__(self) -> str:
        return (f"DamageState({self.kappa.size} points, "
                f"kappa max={self.kappa.max():.4g})")


@dataclass
class DamageResult:
    """Outcome of a stress update."""

    state: DamageState
    damage: np.ndarray           # (n,) D
    eps_eq: np.ndarray           # (n,) the local equivalent strain
    loading: np.ndarray          # (n,) bool, damage grew this increment

    #: The assembler reads these two off any inelastic result. ``plastic`` is
    #: its generic "this point is inelastic right now" mask -- yielding for a
    #: plasticity model, damaging for this one -- and there is no analogue of a
    #: cone apex here.
    @property
    def plastic(self) -> np.ndarray:
        return self.loading

    @property
    def apex(self) -> np.ndarray:
        return np.zeros(self.loading.shape, dtype=bool)

    #: No local solve, so nothing can fail to converge and the count is zero.
    iterations: int = 0
    converged: bool = True

    def __repr__(self) -> str:
        n = self.damage.size
        return (f"DamageResult(points={n}, "
                f"loading={int(self.loading.sum())}, "
                f"D max={self.damage.max():.4f}, "
                f"broken={int((self.damage > 0.99).sum())})")


# ---------------------------------------------------------------------------

class DamageModel(InelasticMaterial):
    """Isotropic scalar continuum damage for small strains.

    Parameters
    ----------
    E, nu : elastic constants.
    equivalent : name or :class:`~.equivalent.EquivalentStrain`. The model binds
        its own ``E`` and ``nu`` into the measure and refuses a contradiction.
    evolution : name or :class:`~.evolution.DamageLaw`. Required -- there is no
        sensible default threshold.
    split : name or :class:`~scion.materials.splits.Split`. ``None`` gives
        no split, which degrades the material identically in tension and
        compression. Fine for a metal; wrong for anything that closes cracks.
    equivalent_kwargs, evolution_kwargs : passed through when the corresponding
        argument is a name.

    Plane stress is refused for the same reason as in plasticity: it constrains
    ``sigma_33 = 0`` with ``eps_33`` unknown, so the out-of-plane strain would
    have to be solved for inside the update, and the equivalent strain -- built
    from the full 3-D tensor -- would be wrong without it.
    """

    name = "damage"

    def __init__(self, E: float, nu: float,
                 evolution: str | DamageLaw,
                 equivalent: str | EquivalentStrain = "mazars",
                 split: str | Split | None = None,
                 equivalent_kwargs: dict | None = None,
                 evolution_kwargs: dict | None = None,
                 plane_strain: bool = True):
        self.E, self.nu = float(E), float(nu)
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.mu = self.E / (2.0 * (1.0 + self.nu))

        self.equivalent = (
            get_equivalent_strain(equivalent, **(equivalent_kwargs or {}))
            if isinstance(equivalent, str) else equivalent)
        self.equivalent.bind(self.E, self.nu)

        self.evolution = (get_damage_law(evolution, **(evolution_kwargs or {}))
                          if isinstance(evolution, str) else evolution)
        self.split = (get_split(split) if isinstance(split, str)
                      else (split or NoSplit()))

        if not plane_strain:
            raise NotImplementedError(
                "plane stress leaves eps_33 unknown, and the equivalent strain "
                "is built from the full 3-D tensor; use plane strain or 3-D")
        self.plane_strain = True

    # ------------------------------------------------------------------
    @property
    def symmetric_tangent(self) -> bool:
        """Whether the consistent tangent may be treated as symmetric.

        Only when the rank-one term is ``sigma (x) sigma`` up to a scalar, which
        needs the equivalent strain's derivative parallel to the degraded
        stress. That is the energy-release measure with no split, and nothing
        else. Getting this wrong the optimistic way makes a Cholesky
        factorisation of an unsymmetric matrix return nonsense.
        """
        return (getattr(self.equivalent, "symmetric_rank_one", False)
                and isinstance(self.split, NoSplit))

    @property
    def linear_momentum(self) -> bool:
        """Whether the stress is linear in the strain at fixed damage.

        True without a split, and false with one: a spectral or
        volumetric-deviatoric decomposition is positively homogeneous in the
        strain but not linear in it, so the momentum equation needs Newton even
        when the damage is frozen. Exactly the trade the phase-field module
        makes between its isotropic and anisotropic couplings.
        """
        return isinstance(self.split, NoSplit)

    def initial_state(self, n_points: int) -> DamageState:
        return DamageState.zeros(n_points, self.evolution.kappa_0)

    # ------------------------------------------------------------------
    def update(self, eps_voigt: np.ndarray, state: DamageState,
               dim: int = 3, driver: np.ndarray | None = None):
        """Integrate one strain increment for a batch of quadrature points.

        ``eps_voigt`` is ``(n, 3)`` in 2-D (``[e11, e22, gamma12]``, plane
        strain) or ``(n, 6)`` in 3-D, with engineering shears.

        ``driver`` optionally replaces the local equivalent strain in the
        history update -- the nonlocal field, for gradient damage. The returned
        tangent is then the secant, which is correct: at fixed ``driver`` the
        damage does not respond to the local strain.

        ``state`` is the committed history and is never modified; the new one is
        ``result.state``. A Newton iterate that is later discarded must not
        advance ``kappa``, exactly as a plastic strain must not.
        """
        eps = voigt_strain_to_tensor(np.asarray(eps_voigt, dtype=float), dim)
        eps_eq = self.equivalent.value(eps)

        drive = eps_eq if driver is None else np.asarray(driver, dtype=float)
        kappa = np.maximum(state.kappa, drive)
        loading = drive > state.kappa
        D = self.evolution.D(kappa)

        s_p, s_m, C_p, C_m = self.split.stress_tangent(eps, self.lam, self.mu)
        g = (1.0 - D)[:, None, None]
        sigma = g * s_p + s_m
        C = g[:, :, :, None, None] * C_p + C_m

        if driver is None:
            dD = self.evolution.dD_dkappa(kappa) * loading
            deq = self.equivalent.dvalue_deps(eps)
            C = C - dD[:, None, None, None, None] * np.einsum(
                "nij,nkl->nijkl", s_p, deq)

        sigma_v = voigt_stress(sigma)
        C_v = voigt_tangent(C)
        if dim == 2:
            # exact, not a reduction: eps_33 = gamma_13 = gamma_23 = 0 in plane
            # strain, so the dropped columns contribute nothing to the kept rows
            sigma_v = sigma_v[:, _PLANE_STRAIN]
            C_v = C_v[:, _PLANE_STRAIN][:, :, _PLANE_STRAIN]

        return sigma_v, C_v, DamageResult(
            state=DamageState(kappa), damage=D, eps_eq=eps_eq, loading=loading)

    # ------------------------------------------------------------------
    def equivalent_strain(self, eps_voigt: np.ndarray,
                          dim: int = 3) -> np.ndarray:
        """``eps_eq`` alone, ``(n,)``.

        The right-hand side of the nonlocal Helmholtz equation, and the cheapest
        field to look at when a run localises somewhere unexpected.
        """
        eps = voigt_strain_to_tensor(np.asarray(eps_voigt, dtype=float), dim)
        return self.equivalent.value(eps)

    def elastic_energy(self, eps_voigt: np.ndarray, damage: np.ndarray,
                       dim: int = 3) -> np.ndarray:
        """Stored energy density ``(1-D) psi_plus + psi_minus``, ``(n,)``."""
        eps = voigt_strain_to_tensor(np.asarray(eps_voigt, dtype=float), dim)
        psi_p, psi_m = self.split.energies(eps, self.lam, self.mu)
        return (1.0 - np.asarray(damage, dtype=float)) * psi_p + psi_m

    def elastic_tangent_voigt(self, dim: int = 3) -> np.ndarray:
        """The undamaged tangent in the same convention, for a first iterate."""
        eps = np.zeros((1, 3, 3))
        _, _, C_p, C_m = self.split.stress_tangent(eps, self.lam, self.mu)
        C_v = voigt_tangent(C_p + C_m)
        if dim == 2:
            C_v = C_v[:, _PLANE_STRAIN][:, :, _PLANE_STRAIN]
        return C_v[0]

    def __repr__(self) -> str:
        return (f"DamageModel(E={self.E:g}, nu={self.nu:g}, "
                f"equivalent={self.equivalent.name}, "
                f"evolution={self.evolution!r}, split={self.split.name}, "
                f"symmetric={self.symmetric_tangent})")


# ---------------------------------------------------------------------------

class GradientDamageModel(DamageModel):
    r"""Local damage plus the length scale that makes it well posed.

    Adds nothing to the constitutive update -- it is the same model, driven by a
    nonlocal equivalent strain instead of the local one. What it carries is the
    internal length ``ell`` and the coefficient of the Helmholtz equation the
    driver solves,

        ``ebar - ell^2 laplacian(ebar) = eps_eq``,   ``grad(ebar) . n = 0``,

    which is the implicit gradient formulation of Peerlings et al. (1996).

    Why a length scale is not optional
    ----------------------------------
    Every law in :mod:`~scion.materials.damage.evolution` softens, and a
    softening local continuum loses ellipticity. The localisation band then has
    no width of its own and collapses onto whatever the mesh provides: refine
    and the band narrows, the dissipated energy falls with it, and in the limit
    the structure fails at the peak load having absorbed no energy at all. The
    computed answer is a property of the discretisation, not of the material.

    The gradient term restores a width. ``ell`` sets it, the band converges
    under refinement, and the results become mesh-independent -- provided the
    mesh can resolve ``ell``, which is what :meth:`check_mesh_size` is for.

    Choosing ell
    ------------
    It is a material parameter, not a numerical one, and it is not free: with a
    given evolution law it fixes the fracture energy per unit area, roughly
    ``g_f * ell`` for a band of that width. Calibrate ``ell`` and the law
    together against a measured ``G_c`` and a measured strength, or the model
    will match one and miss the other.
    """

    name = "gradient-damage"

    def __init__(self, E: float, nu: float, evolution: str | DamageLaw,
                 ell: float, **kwargs):
        super().__init__(E, nu, evolution, **kwargs)
        if ell <= 0.0:
            raise ValueError("the internal length ell must be positive; use "
                             "DamageModel for the unregularised model")
        self.ell = float(ell)

    def gradient_coefficient(self) -> float:
        """``c = ell^2``, the coefficient in the Helmholtz equation."""
        return self.ell ** 2

    def recommended_mesh_size(self, safety: float = 3.0) -> float:
        """Largest element that resolves the band, ``ell / safety``.

        Three elements across the internal length is the usual minimum. Fewer
        and the discrete band is set by the mesh again, which reproduces the
        pathology the length scale was added to remove -- while looking
        regularised, because the damage field is now smooth.
        """
        return self.ell / float(safety)

    def check_mesh_size(self, h: float, safety: float = 3.0) -> None:
        limit = self.recommended_mesh_size(safety)
        if h > limit:
            raise ValueError(
                f"element size h = {h:.4g} cannot resolve the internal length "
                f"ell = {self.ell:.4g}; need h <= {limit:.4g} "
                f"({safety:g} elements across ell). A coarser mesh gives a "
                f"mesh-dependent band that still looks smooth.")

    def __repr__(self) -> str:
        return (f"GradientDamageModel(E={self.E:g}, nu={self.nu:g}, "
                f"ell={self.ell:g}, equivalent={self.equivalent.name}, "
                f"evolution={self.evolution!r}, split={self.split.name})")
