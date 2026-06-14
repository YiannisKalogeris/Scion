"""
Optimization subpackage.

Submodules
----------
gradient        Gradient-based optimizers (L-BFGS, projected gradient)
evolutionary    Derivative-free methods (CMA-ES, genetic algorithms)
bayesian_opt    Bayesian optimisation using GP surrogates
oed             Optimal Experimental Design
"""

from simulacrum.optimization.gradient import ScipyOptimizer
from simulacrum.optimization.evolutionary import CMAES
from simulacrum.optimization.bayesian_opt import BayesianOptimizer
from simulacrum.optimization.oed import OptimalExperimentalDesign

__all__ = [
    "ScipyOptimizer",
    "CMAES",
    "BayesianOptimizer",
    "OptimalExperimentalDesign",
]
