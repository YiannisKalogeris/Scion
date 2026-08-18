"""Two-level preconditioning with diffusion-filtered coarse spaces.

Kalogeris, *Two-Level Preconditioning with Diffusion-Filtered Coarse Spaces for
Ill-Conditioned Finite-Element SPD Systems*. The baselines it is compared
against (CG, PCG, Jacobi, IC(0), ILU, AMG) live in
:mod:`simulacrum.solvers`; this package is the contribution.
"""

from .diffusion import CoarseSpace, build_coarse_space, DEFAULT_TIMES
from .two_level import TwoLevelPreconditioner

__all__ = ["CoarseSpace", "build_coarse_space", "TwoLevelPreconditioner",
           "DEFAULT_TIMES"]
