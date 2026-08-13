r"""
The plasticity model: composition of the three axes behind one interface.

    model = PlasticityModel(E=210e3, nu=0.3,
                            criterion="j2",
                            isotropic=LinearIsotropic(250.0, 2000.0),
                            flow=Associative(),
                            kinematic=NoKinematic())

This is the boundary between the constitutive layer and the finite element
layer. Inside, everything is Mandel and full 3-D; outside, everything is Voigt
and the element's own dimension. Putting the conversion here rather than in the
assembler means an element loop never sees a ``sqrt(2)``, and a second
discretisation -- meshfree, MPM, spectral -- gets the same model for free.

Why a separate base class
-------------------------
:class:`~simulacrum.materials.base.Material` declares ``stress(strain, dim)``,
a pure function of the current strain. No such function exists for a plastic
material: the stress at a given strain depends on how that strain was reached.
What a path-dependent model can offer is a *stress update*, which needs the
committed state as an argument and returns the new one -- a different signature,
so a sibling base rather than a bent one. The same reasoning already produced
``BatchedMaterial`` for the hyperelastic models.

State
-----
The committed state is owned by the caller, normally in a
:class:`~simulacrum.discretization.fem.quadrature_state.QuadratureState`, and
is never mutated here. A Newton iterate that is later discarded must not
advance the plastic strain -- the same discipline the phase-field history field
needs, and the reason ``QuadratureState`` has ``commit`` and ``restore``.
"""

from __future__ import annotations

import numpy as np

from ..notation import (
    mandel_to_voigt_stress,
    mandel_to_voigt_tangent,
    voigt_strain_to_mandel,
)
from .flow_rules import Associative, FlowRule, get_flow_rule
from .hardening import (
    IsotropicHardening,
    KinematicHardening,
    NoKinematic,
    get_isotropic,
    get_kinematic,
)
from .return_mapping import PlasticState, return_map
from .yield_criteria import YieldCriterion, get_criterion

__all__ = ["InelasticMaterial", "PlasticityModel"]


class InelasticMaterial:
    """Base for path-dependent materials.

    Subclasses provide :meth:`initial_state` and :meth:`update`. Deliberately
    not an ABC inheriting from ``Material``: the two interfaces are different
    in kind, and a class that can only raise ``NotImplementedError`` for half
    its parent's methods is worse than one that does not claim the parent.
    """

    name: str = "inelastic"

    #: False when the consistent tangent is unsymmetric, so the assembler must
    #: not choose a symmetric storage format or a Cholesky solver.
    symmetric_tangent: bool = True

    def initial_state(self, n_points: int):
        raise NotImplementedError

    def update(self, eps_voigt, state, dim: int = 3):
        """``(sigma_voigt, C_voigt, new_state)`` for a batch of points."""
        raise NotImplementedError


class PlasticityModel(InelasticMaterial):
    """Rate-independent plasticity for small strains.

    Parameters
    ----------
    E, nu : elastic constants.
    criterion : name or :class:`~.yield_criteria.YieldCriterion`.
    isotropic : name or :class:`~.hardening.IsotropicHardening`.
    flow : defaults to associative.
    kinematic : defaults to none.
    criterion_kwargs, isotropic_kwargs : passed through when the corresponding
        argument is given as a name.

    Only plane strain and full 3-D are supported. Plane stress imposes
    ``sigma_33 = 0`` with ``eps_33`` unknown, which turns the return map into a
    nested solve; it is refused rather than approximated, because the usual
    shortcut -- running the plane-strain update and discarding ``sigma_33`` --
    silently violates the constraint and gives the wrong yield point.
    """

    name = "plasticity"

    def __init__(self, E: float, nu: float,
                 criterion: str | YieldCriterion = "j2",
                 isotropic: str | IsotropicHardening = "perfect",
                 flow: str | FlowRule | None = None,
                 kinematic: str | KinematicHardening | None = None,
                 criterion_kwargs: dict | None = None,
                 isotropic_kwargs: dict | None = None,
                 plane_strain: bool = True):
        from ..linear_elastic import isotropic_stiffness_mandel

        self.E, self.nu = float(E), float(nu)
        self.C_el = isotropic_stiffness_mandel(E, nu)

        self.criterion = (get_criterion(criterion, **(criterion_kwargs or {}))
                          if isinstance(criterion, str) else criterion)
        self.isotropic = (get_isotropic(isotropic, **(isotropic_kwargs or {}))
                          if isinstance(isotropic, str) else isotropic)
        self.flow = (get_flow_rule(flow) if isinstance(flow, str)
                     else (flow or Associative()))
        self.kinematic = (get_kinematic(kinematic) if isinstance(kinematic, str)
                          else (kinematic or NoKinematic()))

        if not plane_strain:
            raise NotImplementedError(
                "plane stress requires solving for eps_33 inside the return "
                "map; use plane strain or a full 3-D model")
        self.plane_strain = True

        if not self.criterion.smooth:
            raise NotImplementedError(
                f"{self.criterion.name} has corners and needs a multi-surface "
                f"return map")

    # ------------------------------------------------------------------
    @property
    def symmetric_tangent(self) -> bool:
        return self.flow.symmetric_tangent

    def initial_state(self, n_points: int) -> PlasticState:
        return PlasticState.zeros(n_points)

    # ------------------------------------------------------------------
    def update(self, eps_voigt: np.ndarray, state: PlasticState,
               dim: int = 3, **kw):
        """Integrate one strain increment for a batch of quadrature points.

        ``eps_voigt`` is ``(n, 3)`` in 2-D (``[e11, e22, gamma12]``) or
        ``(n, 6)`` in 3-D, with engineering shears. The returned tangent pairs
        with the same convention, so it drops straight into ``B.T @ C @ B``.

        Returns ``(sigma_voigt, C_voigt, result)``; the updated state is
        ``result.state``, and ``result`` also carries the plastic and apex
        masks and the iteration count.
        """
        eps_m = voigt_strain_to_mandel(np.asarray(eps_voigt, dtype=float), dim)
        result = return_map(eps_m, state, self.criterion, self.isotropic,
                            self.C_el, self.flow, self.kinematic, **kw)
        return (mandel_to_voigt_stress(result.sigma, dim),
                mandel_to_voigt_tangent(result.C_alg, dim),
                result)

    # ------------------------------------------------------------------
    def elastic_tangent_voigt(self, dim: int = 3) -> np.ndarray:
        """The elastic tangent in the same convention, for a first iteration."""
        return mandel_to_voigt_tangent(self.C_el[None], dim)[0]

    def __repr__(self) -> str:
        return (f"PlasticityModel(E={self.E:g}, nu={self.nu:g}, "
                f"criterion={self.criterion.name}, "
                f"isotropic={self.isotropic.name}, "
                f"flow={self.flow.name}, kinematic={self.kinematic.name})")
