"""
Configurational forces in the Material Point Method.

Implementation of Kalogeris, Spitas, Atzarakis, Stavroulakis & Papadopoulos,
*Configurational Forces in the Material Point Method: Discrete Noether
Structure, Spurious Pseudomomentum, and Material-Manifold Adaptivity*.

What is where
-------------
The generic machinery is library code, in ``scion``, because it is older than
this paper and has consumers outside it:

===================================  ====================================
``materials/eshelby.py``             Eshelby stress, pseudomomentum,
                                     nodal material force (Eq. 4, 10, 12)
``discretization/mpm/basis.py``      ``second_derivatives_1d`` on each basis
``discretization/mpm/``              periodic grids, ``build_stencil(
``background_grid.py``               hessian=True)``
===================================  ====================================

The contribution is here: the discrete configurational force conjugate to a
particle's *reference* position, Eq. (9), which is the object the paper defines
and which has no other user.
"""

from projects.configurational_mpm.forces import (
    configurational_force,
    displacement_gradient,
    second_displacement_gradient,
)
from projects.configurational_mpm.adaptivity import (
    RelaxationResult,
    configurational_forces,
    discrete_energy,
    relax_reference_positions,
)

__all__ = [
    "configurational_force",
    "displacement_gradient",
    "second_displacement_gradient",
    "discrete_energy",
    "configurational_forces",
    "relax_reference_positions",
    "RelaxationResult",
]
