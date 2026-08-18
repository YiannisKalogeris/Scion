"""
Verification for the elastoplastic assembly.

The decisive test is the global tangent against finite differences: it exercises
the whole path -- B matrices, the Voigt/Mandel boundary, the return map and the
consistent tangent -- in the one place where an error in any of them shows up
as a number rather than as slow convergence.

python -m pytest tests/test_fem/test_elastoplastic_assembler.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import linalg as spla

from scion.core.domain import Domain
from scion.discretization.fem.nonlinear_assembler import (
    ElastoplasticAssembler,
)
from scion.materials.plasticity import (
    DruckerPrager,
    LinearIsotropic,
    LinearKinematic,
    NonAssociative,
    PerfectPlasticity,
    PlasticityModel,
    Swift,
    Voce,
)
from scion.mesh.mesh import ElementBlock, Mesh
from scion.solvers.tangent_check import check_tangent

E, NU, Y, H = 210e3, 0.3, 250.0, 2000.0


def grid(n=3, side=1.0):
    xs = np.linspace(0.0, side, n + 1)
    nodes = np.array([[x, y] for y in xs for x in xs])
    conn = np.array([[j * (n + 1) + i, j * (n + 1) + i + 1,
                      j * (n + 1) + i + n + 2, j * (n + 1) + i + n + 1]
                     for j in range(n) for i in range(n)])
    return Domain(Mesh(nodes, [ElementBlock("quad4", conn)]), dim=2)


def _setup(model, n=3):
    dom = grid(n)
    asm = ElastoplasticAssembler(dom, model)
    nd = dom.coordinates
    dm = asm.dof_map
    left = np.where(nd[:, 0] < 1e-12)[0]
    right = np.where(nd[:, 0] > 1 - 1e-12)[0]
    bot = np.where(nd[:, 1] < 1e-12)[0]
    fixed = np.concatenate([dm.element_dofs(left)[0::2],
                            dm.element_dofs(bot)[1::2],
                            dm.element_dofs(right)[0::2]])
    ctrl = dm.element_dofs(right)[0::2]
    free = np.setdiff1d(np.arange(asm.n_dof), fixed)
    return asm, ctrl, free


# ---------------------------------------------------------------------------
# Elastic limit
# ---------------------------------------------------------------------------

def test_below_yield_the_assembly_reproduces_linear_elasticity():
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()

    u = np.zeros(asm.n_dof)
    u[ctrl] = 1e-6
    res = asm.assemble(u, st)

    assert not res.plastic.any()
    assert np.abs(res.f_int - res.K @ u).max() < 1e-6 * np.abs(res.f_int).max()

    C_el = model.elastic_tangent_voigt(dim=2)
    assert np.abs(res.K.toarray() - _elastic_K(asm, C_el)).max() < 1e-6


def _elastic_K(asm, C_el):
    ke = np.einsum("eqab,bc,eqcd,eq->ead", np.swapaxes(asm.B, 2, 3), C_el,
                   asm.B, asm.w)
    K = np.zeros((asm.n_dof, asm.n_dof))
    for e, dofs in enumerate(asm.edofs):
        K[np.ix_(dofs, dofs)] += ke[e]
    return K


# ---------------------------------------------------------------------------
# The global consistent tangent
# ---------------------------------------------------------------------------

CASES = [
    ("j2-linear", "j2", LinearIsotropic(Y, H), None, None, 0.004),
    ("j2-perfect", "j2", PerfectPlasticity(Y), None, None, 0.004),
    ("j2-voce", "j2", Voce(Y, 400.0, 20.0), None, None, 0.004),
    ("j2-swift", "j2", Swift.from_yield(Y, 600.0, 0.25), None, None, 0.004),
    ("j2-kinematic", "j2", LinearIsotropic(Y, H), None,
     LinearKinematic(1500.0), 0.004),
    ("dp-associative", DruckerPrager.from_mohr_coulomb(30.0),
     LinearIsotropic(500.0, 500.0), None, None, 0.0015),
    ("dp-non-associative", DruckerPrager.from_mohr_coulomb(30.0),
     LinearIsotropic(500.0, 500.0), NonAssociative.drucker_prager(10.0), None,
     0.0015),
    ("dp-apex", DruckerPrager.from_mohr_coulomb(30.0),
     LinearIsotropic(50.0, 500.0), None, None, 0.004),
]


@pytest.mark.parametrize("label,crit,iso,flow,kin,disp", CASES,
                         ids=[c[0] for c in CASES])
def test_global_tangent_matches_finite_differences(label, crit, iso, flow, kin,
                                                   disp):
    """Exact for the algorithm as implemented, across every combination. A
    wrong tangent still converges -- linearly instead of quadratically -- so
    nothing but this test distinguishes the two."""
    model = PlasticityModel(E, NU, crit, iso, flow=flow, kinematic=kin)
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()

    u = np.zeros(asm.n_dof)
    u[ctrl] = disp

    result = asm.assemble(u, st)
    assert result.plastic.any(), "the test state must actually yield"
    assert result.converged

    residual, tangent = asm.residual_and_tangent(st)
    report = check_tangent(residual, tangent, u, free=free, rtol=2e-4)
    assert report.passed, f"{label}: {report}"


def test_tangent_symmetry_is_reported_and_real():
    """Non-associative flow makes the global tangent unsymmetric. The flag
    exists so the assembler can pick a solver instead of discovering it."""
    crit = DruckerPrager.from_mohr_coulomb(30.0)
    iso = LinearIsotropic(500.0, 500.0)

    assoc = PlasticityModel(E, NU, crit, iso)
    non = PlasticityModel(E, NU, crit, iso,
                          flow=NonAssociative.drucker_prager(10.0))
    assert assoc.symmetric_tangent and not non.symmetric_tangent

    for model, expect_sym in ((assoc, True), (non, False)):
        asm, ctrl, free = _setup(model)
        assert asm.symmetric is expect_sym
        u = np.zeros(asm.n_dof)
        u[ctrl] = 0.0015
        K = asm.assemble(u, asm.initial_state()).K.toarray()
        asym = np.linalg.norm(K - K.T) / np.linalg.norm(K)
        if expect_sym:
            assert asym < 1e-12
        else:
            assert asym > 1e-3


# ---------------------------------------------------------------------------
# State discipline
# ---------------------------------------------------------------------------

def test_assembly_never_mutates_the_committed_state():
    """A Newton iterate that is later discarded must not advance the plastic
    strain. Without this, a rejected load step corrupts the history."""
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()
    before = (st.eps_p.copy(), st.alpha.copy(), st.beta.copy())

    u = np.zeros(asm.n_dof)
    for trial in (0.002, 0.006, 0.001):
        u[ctrl] = trial
        asm.assemble(u, st)

    assert np.array_equal(st.eps_p, before[0])
    assert np.array_equal(st.alpha, before[1])
    assert np.array_equal(st.beta, before[2])


def test_repeated_assembly_at_the_same_u_is_reproducible():
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()
    u = np.zeros(asm.n_dof)
    u[ctrl] = 0.004

    a = asm.assemble(u, st)
    b = asm.assemble(u, st)
    assert np.abs(a.f_int - b.f_int).max() == 0.0
    assert np.abs((a.K - b.K).toarray()).max() == 0.0


# ---------------------------------------------------------------------------
# A load path
# ---------------------------------------------------------------------------

def test_newton_converges_quadratically_on_a_load_step():
    """The payoff of the consistent tangent, and the only externally visible
    difference between a right and a merely plausible one."""
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()

    u = np.zeros(asm.n_dof)
    u[ctrl] = 0.003                       # well past yield in one step

    norms = []
    for _ in range(6):
        r = asm.assemble(u, st)
        res = r.f_int[free]
        norms.append(np.linalg.norm(res))
        if norms[-1] < 1e-9 * max(norms[0], 1.0):
            break
        du = spla.spsolve(r.K.tocsc()[free][:, free], -res)
        u[free] += du

    assert len(norms) <= 8
    assert norms[-1] < 1e-8 * norms[0]
    # The hallmark of a consistent tangent: once in the asymptotic regime the
    # residual is bounded by the square of the previous one. A merely plausible
    # tangent still converges, but only linearly, and would fail this.
    assert norms[-2] < 1.0
    assert norms[-1] < norms[-2] ** 2


def test_load_path_lands_on_the_hardening_curve():
    """After a converged path the state must satisfy the yield condition
    exactly: ``q = kappa(alpha)``.

    Stated on the invariant rather than on ``sigma_11``, because the specimen
    is in *plane strain*: stretching in x with y free still leaves
    ``sigma_33 != 0``, so the uniaxial relation ``sigma_11 = Y + H eps_p`` does
    not apply and asserting it would be testing the wrong thing.
    """
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    asm, ctrl, free = _setup(model)
    st = asm.initial_state()

    u = np.zeros(asm.n_dof)
    for step in range(1, 21):
        u[ctrl] = 0.0002 * step
        for _ in range(12):
            r = asm.assemble(u, st)
            res = r.f_int[free]
            if np.linalg.norm(res) < 1e-9 * max(np.abs(r.f_int).max(), 1.0):
                break
            u[free] += spla.spsolve(r.K.tocsc()[free][:, free], -res)
        st = r.state

    from scion.materials.notation import from_mandel, voigt_strain_to_mandel
    from scion.materials.plasticity import von_mises_stress

    eps_m = voigt_strain_to_mandel(asm.strains(u), 2)
    sigma_m = (eps_m - st.eps_p) @ model.C_el.T
    q = von_mises_stress(from_mandel(sigma_m))
    kappa = model.isotropic.kappa(st.alpha)

    assert st.alpha.min() > 0.0                    # everything yielded
    assert np.abs(q - kappa).max() < 1e-6 * kappa.mean()
    assert np.abs(np.diff(st.alpha)).max() < 1e-8 * st.alpha.mean()  # uniform


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_plane_stress_is_refused_rather_than_approximated():
    """Running the plane-strain update and discarding sigma_33 violates the
    constraint and moves the yield point, so it is rejected."""
    with pytest.raises(NotImplementedError, match="plane stress"):
        PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H), plane_strain=False)


def test_multiple_element_blocks_are_refused():
    """The state array is indexed by a single quadrature-point numbering."""
    dom = grid(2)
    blk = dom.mesh.element_blocks[0]
    dom.mesh.element_blocks = [blk, blk]
    model = PlasticityModel(E, NU, "j2", LinearIsotropic(Y, H))
    with pytest.raises(NotImplementedError, match="single element block"):
        ElastoplasticAssembler(dom, model)
