"""
Smoothed particle hydrodynamics: kernels, neighbours and discrete operators.

The pieces are deliberately separate -- a neighbour list that knows nothing
about physics, operators that take one, and a solver that owns the state --
for the same reason ``solvers/`` knows no physics: the diffusion solver here
is forty lines on top of operators that an SPH momentum equation would reuse
unchanged.

Conventions that differ between SPH codes and are fixed here:

* ``q = r/h`` with support ``2h`` (SPHinXsys's convention, not the one where
  the support is ``h``);
* a pair is stored once, with ``dx = x_i - x_j`` already minimum-imaged;
* the gradient correction matrix is the inverse of the renormalization
  tensor, so the corrected gradient is exact for a linear field.
"""

from .diffusion import (DiffusionSolver, DirectionalDiffusivity,
                        IsotropicDiffusivity)
from .fluid import FluidSolver, WeaklyCompressibleFluid, lattice_number_density
from .kernels import KERNELS, CubicSpline, SPHKernel, WendlandC2, get_kernel
from .neighbors import NeighborList, brute_force_neighbors, build_neighbors
from .operators import (kernel_pair_values, laplacian_rate,
                        linear_gradient_correction, number_density,
                        pair_conductance, shepard_interpolate, sph_gradient)
from .particles import lattice

__all__ = ["DiffusionSolver", "DirectionalDiffusivity", "IsotropicDiffusivity",
           "FluidSolver", "WeaklyCompressibleFluid", "lattice_number_density",
           "KERNELS", "CubicSpline", "SPHKernel", "WendlandC2", "get_kernel",
           "NeighborList", "brute_force_neighbors", "build_neighbors",
           "kernel_pair_values", "laplacian_rate", "linear_gradient_correction",
           "number_density", "pair_conductance", "shepard_interpolate",
           "sph_gradient", "lattice"]
