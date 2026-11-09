"""Density methods: the full PDF of a response and, in time, its evolution.

``uq`` is organised by the question asked. ``forward`` pushes a distribution
through a model and returns moments, sensitivities or a failure probability;
``bayesian`` recovers a distribution from measurements. This subpackage asks a
third question -- what is the *whole* density of the response, and how does it
evolve -- which is the question Li & Chen's probability density evolution
method and Chen & Yang's direct probability integral method were built to
answer, and which a second-order statistic cannot be read off.

Implemented from:

- J. Li, J. B. Chen, *Probability density evolution method for dynamic
  response analysis of structures with uncertain parameters*, Computational
  Mechanics 34 (2004) 400-409.
- G. Chen, D. Yang, *Direct probability integral method for stochastic
  response analysis of static and dynamic structural systems*, CMAME 357
  (2019) 112612.

Both are implemented from their published formulations; no code was taken from
either.
"""
from scion.uq.density.dpim import (
    SIGMA_FACTOR,
    UNIFORMITY_LIMIT,
    SIGMA_RATIO,
    knn_bandwidth,
    paper_bandwidth,
    response_spacing,
    band_probability,
    exceedance_probability,
    silverman_bandwidth,
    solve_dpim,
    spacing_uniformity,
)
from scion.uq.density.mixed import (
    MixedDensity,
    mixed_density,
    reflected_density,
)
from scion.uq.density.gdee import (
    DensityEvolution,
    ProbabilityNotConserved,
    response_histories,
    solve_gdee,
)
from scion.uq.density.points import (
    PROBABILITY_TOL,
    RepresentativePoints,
    gf_discrepancy,
    gf_points,
    lattice,
)
from scion.uq.density.reference import (
    OMEGA_BOUNDS,
    X0,
    sdof_displacement_moments,
    sdof_displacement_pdf,
    sdof_response,
    vanderpol_coefficient,
    vanderpol_pdf,
    vanderpol_response,
)

__all__ = [
    # bounded supports and the atom a first-passage indicator carries
    "MixedDensity",
    "mixed_density",
    "reflected_density",
    "DensityEvolution",
    "OMEGA_BOUNDS",
    "PROBABILITY_TOL",
    "ProbabilityNotConserved",
    "RepresentativePoints",
    "SIGMA_FACTOR",
    "SIGMA_RATIO",
    "UNIFORMITY_LIMIT",
    "X0",
    "gf_discrepancy",
    "gf_points",
    "lattice",
    "response_histories",
    "sdof_displacement_moments",
    "sdof_displacement_pdf",
    "sdof_response",
    "vanderpol_coefficient",
    "vanderpol_pdf",
    "vanderpol_response",
    "knn_bandwidth",
    "paper_bandwidth",
    "response_spacing",
    "solve_dpim",
    "exceedance_probability",
    "band_probability",
    "silverman_bandwidth",
    "spacing_uniformity",
    "solve_gdee",
]
