r"""
Direct methods for shakedown.

A structure under a *varying* load does not have to collapse to fail. It can
ratchet -- accumulate a little plastic strain every cycle until the geometry is
unusable -- or it can alternate plastically in place until it cracks. Or it can
**shake down**: yield for a while, build up a residual stress field, and settle
into a response that is elastic forever after. Which of the three happens
depends on the size of the load domain, and the shakedown factor is the largest
domain for which the good outcome is guaranteed.

Direct methods answer that without integrating a single cycle. They ask instead
whether a field with the right properties *exists*, which turns a path-dependent
simulation into a convex optimisation problem posed on the elastic solution
alone. The whole load history collapses onto the vertices of the load domain.

    from scion.shakedown import (LoadDomain, ElasticEnvelope,
                                      SelfEquilibrium, melan_lower_bound)

    domain = LoadDomain.box(lower=[0.0, 0.0], upper=[1.0, 1.0])
    envelope = ElasticEnvelope.from_elastic_constants(tables, E, nu)
    envelope.add_cases(displacements_for_each_basic_case)

    result = melan_lower_bound(envelope.vertex_stresses(domain),
                               SelfEquilibrium(tables, free_dofs),
                               VonMises(), sigma_y)
    result.factor                       # certified lower bound
    result.verify(...)                  # check it independently

Which theorem gives which side
------------------------------
**Melan's static theorem** (implemented) gives a **lower** bound: find one
self-equilibrated residual field that keeps every vertex admissible. Every
approximation -- a coarse mesh, a restricted residual field, a loose tolerance
-- shrinks the feasible set and lowers the answer, so the result is safe by
construction and a coarse mesh is still usable.

**Koiter's kinematic theorem** gives an **upper** bound from an assumed
mechanism, and there the errors point the other way. The Linear Matching Method
and RSDM are the practical routes to it and are not implemented yet; when they
are, running them alongside Melan brackets the true factor from both sides on
the same mesh, which is worth more than either alone.

Limit analysis for free
-----------------------
A load domain with no variation has one vertex, and Melan's condition collapses
onto the static theorem of limit analysis. So ``LoadDomain.fixed()`` turns this
machinery into a plastic collapse solver, which is how the implementation is
checked: plastic collapse has closed-form answers that shakedown does not.

Deliberate omissions
--------------------
* **Upper bounds.** See above; LMM and RSDM are the next pieces.
* **Hardening.** Everything here is elastic-perfectly-plastic. Melan's theorem
  extends to limited kinematic hardening by bounding the back stress, which
  changes the constraint set and therefore the optimisation, not just a
  parameter.
* **Second-order effects.** The elastic fields are computed on the undeformed
  geometry and the theorems assume small displacements. A structure whose
  shakedown limit is near its buckling load is outside all of this.
* **Temperature-dependent yield.** ``kappa`` may vary from point to point but
  not between load vertices, so a thermal load domain in which the yield stress
  itself varies with the temperature is not covered.
"""

from .basis import ResidualBasis
from .equilibrium_elements import (
    EquilibriumTriangles,
    LimitAnalysisResult,
    lower_bound_limit,
)
from .elastic import ElasticEnvelope, solve_elastic
from .equilibrium import SelfEquilibrium
from .lmm import LMMResult, linear_matching
from .load_domain import LoadDomain
from .rsdm import CyclicProgram, RSDMResult, rsdm
from .melan import MelanResult, elastic_factor, melan_lower_bound

__all__ = [
    "LoadDomain",
    "ElasticEnvelope",
    "solve_elastic",
    "SelfEquilibrium",
    "ResidualBasis",
    "melan_lower_bound",
    "MelanResult",
    "linear_matching",
    "LMMResult",
    "EquilibriumTriangles",
    "lower_bound_limit",
    "LimitAnalysisResult",
    "CyclicProgram",
    "rsdm",
    "RSDMResult",
    "elastic_factor",
]
