"""The micropolar Eshelby stress on an *equilibrated* body.

This is the test implicit MPM was built for. A material-manifold balance law
only means anything on a state that satisfies equilibrium: ``Div Sigma = 0``
follows from material translation invariance **on solutions of the balance
laws**, and explicit dynamics never provides one -- inertia carries the
residual, so a material force computed on an explicit run is contaminated by it.

Measured here: with the body equilibrated to ~1e-14, the nodal material force at
fully-supported interior nodes falls

    dx = 0.30    1.20e-04
    dx = 0.20    8.64e-06      rate 6.5
    dx = 0.15    1.72e-06      rate 5.6

a factor of 70 over a 2x refinement, which is what distinguishes "zero" from
"small". The rate is far above the discretisation order of the scheme because
the quantity is *identically* zero in the continuum -- what is being measured is
how fast the quadrature error vanishes, not a truncation order.
"""

import numpy as np
import pytest

from scion.core.logging import quiet
from scion.discretization.mpm import (
    CartesianGrid,
    GridDirichlet,
    ParticleSet,
    get_basis,
)
from scion.discretization.mpm.micropolar import (
    MicropolarMPMStepper,
    attach_micropolar_state,
)
from scion.materials.eshelby import material_force, micropolar_eshelby_stress
from scion.materials.micropolar import MicropolarHooke

EXTENT = (2.4, 1.2, 0.6)
ORIGIN = (0.6, 0.6, 0.6)
ELASTIC = MicropolarHooke.from_technical(E=1000.0, nu=0.25, N=0.6,
                                         l_t=0.6, l_b=0.3, psi=1.2)


def equilibrate(dx, ppc=2, n_steps=6):
    """Solve a bending-type BVP and return the interior material force,
    normalised by the scale of the contributions that build it."""
    nan = np.nan
    sub, pad = dx / ppc, 3
    shape = tuple(int(round(EXTENT[k] / dx)) + 2 * pad for k in range(3))
    grid = CartesianGrid(shape=shape, dx=dx,
                         origin=np.array(ORIGIN) - pad * dx,
                         basis=get_basis("linear"))
    axes = [np.arange(ORIGIN[k] + sub / 2, ORIGIN[k] + EXTENT[k], sub)
            for k in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    particles = attach_micropolar_state(
        ParticleSet(np.stack([m.ravel() for m in mesh], axis=1),
                    vol0=sub ** 3, density=1.0))
    bcs = [GridDirichlet(lambda c: c[:, 0] < ORIGIN[0] + 1e-9, np.zeros(6)),
           GridDirichlet(lambda c: c[:, 0] > ORIGIN[0] + EXTENT[0] - 1e-9,
                         np.array([0.0, 0.10, 0.0, nan, nan, nan]))]
    stepper = MicropolarMPMStepper(grid, ELASTIC, dirichlet=bcs, max_iter=25)
    with quiet():
        stepper.run(particles, n_steps=n_steps)

    c = stepper.constitutive(particles)
    Q, F, Gam = c["Q"], c["F"], c["Ge"]
    E = np.einsum("pji,pjk->pik", Q, F) - np.eye(3)
    B, S = ELASTIC.stresses(E, Gam)
    P = np.einsum("pia,pab->pib", Q, B)
    Sigma = micropolar_eshelby_stress(ELASTIC.energy(E, Gam), F, P, Gam, S)

    # material gradients: the stepper is updated-Lagrangian, so its stencil
    # gradients are *spatial* and must be pulled back with F^T. Using them raw
    # computes a different object entirely.
    grad_X = np.einsum("prb,pkr->pkb", F, stepper._grad)

    support = np.zeros(grid.n_nodes)
    np.add.at(support, stepper._st.node_id,
              particles.vol0[:, None] * stepper._st.w)
    # "interior" by *support*, not by geometry: a node whose stencil is clipped
    # by the edge of the particle region carries a boundary term that looks
    # exactly like a divergence error, and a geometric box is the wrong shape
    # for catching it
    interior = support > 0.98 * support.max()

    contributions = material_force(Sigma, particles.vol0, grad_X)
    G = np.zeros((grid.n_nodes, 3))
    np.add.at(G, stepper._st.node_id, contributions)
    scale = np.abs(contributions).sum(axis=(0, 1)).max()
    converged = all(r.converged for r in stepper.history)
    return np.abs(G[interior]).max() / scale, int(interior.sum()), converged


def test_the_material_force_vanishes_under_refinement():
    """``Div Sigma = 0`` on an equilibrated homogeneous body.

    The convergence is the claim, not the level: a single mesh only shows the
    force is *small*, which any sufficiently smooth wrong tensor would also
    manage.
    """
    results = [equilibrate(dx) for dx in (0.3, 0.2, 0.15)]
    for _, n_interior, converged in results:
        assert converged
        assert n_interior > 10
    forces = [r[0] for r in results]
    assert all(b < a for a, b in zip(forces, forces[1:])), forces
    assert forces[-1] < 0.02 * forces[0]


def test_spatial_gradients_do_not_substitute_for_material_ones():
    """``material_force`` needs ``Grad_X w``. The updated-Lagrangian stepper has
    the *spatial* ones, and passing those computes a different object -- one
    that does not vanish on an equilibrated body. The error is silent in shape
    and in units, and only wrong in meaning.

    It has to be tested at real strain. Measured against the applied shear:

        max|F - I| = 0.055    spatial / material = 3.4
        max|F - I| = 0.221                        23.4
        max|F - I| = 0.443                        21.6

    At small strain ``F ~ I`` and the two gradients coincide, so a gentle test
    problem cannot tell them apart -- which is the more dangerous direction,
    since it means the mistake survives a small-strain check.
    """
    nan = np.nan
    dx, ppc, pad = 0.3, 2, 3
    sub = dx / ppc
    shape = tuple(int(round(EXTENT[k] / dx)) + 2 * pad for k in range(3))
    grid = CartesianGrid(shape=shape, dx=dx,
                         origin=np.array(ORIGIN) - pad * dx,
                         basis=get_basis("linear"))
    axes = [np.arange(ORIGIN[k] + sub / 2, ORIGIN[k] + EXTENT[k], sub)
            for k in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    particles = attach_micropolar_state(
        ParticleSet(np.stack([m.ravel() for m in mesh], axis=1),
                    vol0=sub ** 3, density=1.0))
    bcs = [GridDirichlet(lambda c: c[:, 0] < ORIGIN[0] + 1e-9, np.zeros(6)),
           GridDirichlet(lambda c: c[:, 0] > ORIGIN[0] + EXTENT[0] - 1e-9,
                         np.array([0.0, 0.40, 0.0, nan, nan, nan]))]
    stepper = MicropolarMPMStepper(grid, ELASTIC, dirichlet=bcs, max_iter=25)
    with quiet():
        stepper.run(particles, n_steps=8)

    c = stepper.constitutive(particles)
    Q, F, Gam = c["Q"], c["F"], c["Ge"]
    assert np.abs(F - np.eye(3)).max() > 0.15      # or the test has no power
    E = np.einsum("pji,pjk->pik", Q, F) - np.eye(3)
    B, S = ELASTIC.stresses(E, Gam)
    P = np.einsum("pia,pab->pib", Q, B)
    Sigma = micropolar_eshelby_stress(ELASTIC.energy(E, Gam), F, P, Gam, S)

    support = np.zeros(grid.n_nodes)
    np.add.at(support, stepper._st.node_id,
              particles.vol0[:, None] * stepper._st.w)
    interior = support > 0.98 * support.max()

    def interior_force(grad):
        contributions = material_force(Sigma, particles.vol0, grad)
        G = np.zeros((grid.n_nodes, 3))
        np.add.at(G, stepper._st.node_id, contributions)
        return (np.abs(G[interior]).max()
                / np.abs(contributions).sum(axis=(0, 1)).max())

    material = interior_force(np.einsum("prb,pkr->pkb", F, stepper._grad))
    spatial = interior_force(stepper._grad)
    assert spatial > 5.0 * material
