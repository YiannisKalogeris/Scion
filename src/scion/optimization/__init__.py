"""
Optimization subpackage.

Submodules
----------
gradient        Gradient-based optimizers (L-BFGS, projected gradient)
evolutionary    Derivative-free methods (CMA-ES, genetic algorithms)
bayesian_opt    Bayesian optimisation using GP surrogates
oed             Optimal Experimental Design
"""

from scion.optimization.gradient import ScipyOptimizer
from scion.optimization.evolutionary import CMAES
from scion.optimization.bayesian_opt import BayesianOptimizer
from scion.optimization.oed import OptimalExperimentalDesign

__all__ = [
    "ScipyOptimizer",
    "CMAES",
    "BayesianOptimizer",
    "OptimalExperimentalDesign",
]
