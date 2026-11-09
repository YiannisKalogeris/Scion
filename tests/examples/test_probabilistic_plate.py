"""Example 2: the plate with a hole and a random yield-stress field.

Small meshes throughout -- these check identities and refusals, not accuracy,
and the accuracy claims are measured by the example's own ``--checks`` pass.
"""

import numpy as np
import pytest

from scion.materials.plasticity.yield_criteria import VonMises
from scion.shakedown import (CyclicProgram, LoadDomain, cyclic_steady_state,
                             elastic_factor)
from scion.uq.density import RepresentativePoints
from scion.uq.forward import LogNormalField

from examples.shakedown.plate_model import (DOMAINS, L_HALF, R_HOLE, TRACTION,
                                            YIELD, build, hoop)
from examples.shakedown.probabilistic_plate import (curves, distribution,
                                                    expansion, monte_carlo,
                                                    plastic_activity, sweep,
                                                    transition_loads, unpack)

SAFE = ("elastic", "shakedown")


@pytest.fixture(scope="module")
def model():
    return build(4, 4)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

def test_the_mesh_covers_the_quarter_plate_minus_the_hole(model):
    """The area is the one geometric property that is exact in closed form, so
    it catches a transfinite map that has folded or left a gap."""
    exact = L_HALF ** 2 - 0.25 * np.pi * R_HOLE ** 2
    assert model["weights"].sum() == pytest.approx(exact, rel=2.0e-3)
    assert (model["weights"] > 0.0).all()


def test_the_far_field_reproduces_the_applied_traction(model):
    """Each unit load case must give unit stress far from the hole; a wrong
    edge-force assembly shows up here and nowhere else."""
    s = model["envelope"].stresses / TRACTION
    far = np.abs(model["xq"][:, 0] - L_HALF) < 0.25 * L_HALF
    assert s[0][far, 0].mean() == pytest.approx(1.0, rel=0.05)
    far_y = np.abs(model["xq"][:, 1] - L_HALF) < 0.25 * L_HALF
    assert s[1][far_y, 1].mean() == pytest.approx(1.0, rel=0.05)


def test_the_hole_concentrates_the_stress(model):
    """Kirsch gives 3 at the hole surface under uniaxial tension. The
    quadrature points are inside the elements, so a coarse mesh must read
    *below* 3 and above the far field -- the direction is the check here, the
    value is the example's convergence table."""
    s = model["envelope"].stresses / TRACTION
    assert 1.5 < s[0][:, 0].max() < 3.0


def test_plane_strain_keeps_the_out_of_plane_stress(model):
    """``sigma_33`` carries the out-of-plane constraint and enters ``J2``;
    a zero there would mean the 2-D reduction had dropped it."""
    s = model["envelope"].stresses
    assert np.abs(s[0][:, 2]).max() > 0.1 * np.abs(s[0][:, 0]).max()


# ---------------------------------------------------------------------------
# the transition loads
# ---------------------------------------------------------------------------

def test_shakedown_exceeds_first_yield(model):
    """``rho = 0`` is admissible, so the shakedown factor is at least the
    elastic one. Below it would be a bug rather than a conservative answer."""
    sigma_y = np.full(model["tables"].n_points, YIELD)
    loads = transition_loads(model, sigma_y, 1.0)
    assert loads.shakedown > loads.elastic
    assert loads.shakedown / loads.elastic < 2.0


def test_the_transition_loads_do_not_depend_on_the_modulus(model):
    """Under prescribed tractions the displacement scales as ``1/E`` and the
    stress does not, so this is an exact independence and the example carries
    ``E`` as a random input precisely to check it."""
    sigma_y = np.full(model["tables"].n_points, YIELD)
    base = transition_loads(model, sigma_y, 1.0)
    stiff = dict(model)
    stiff["C"] = model["C"] * 3.0
    other = transition_loads(stiff, sigma_y, 1.0)
    assert other.elastic == pytest.approx(base.elastic, rel=1.0e-12)
    assert other.shakedown == pytest.approx(base.shakedown, rel=1.0e-9)


def test_the_cycle_needs_only_the_vertices_of_the_load_domain():
    """Melan's condition sees the extremes of the domain and nothing between,
    so refining the path cannot move the shakedown factor. This is a cost
    reduction of a factor of eight and it has to be checked, not assumed."""
    sigma_y = None
    factors = []
    for per_leg in (1, 2, 4):
        m = build(4, 4, per_leg=per_leg)
        if sigma_y is None:
            sigma_y = np.full(m["tables"].n_points, YIELD)
        factors.append(transition_loads(m, sigma_y, 1.0).shakedown)
    assert factors[0] == pytest.approx(factors[1], rel=1.0e-9)
    assert factors[0] == pytest.approx(factors[2], rel=1.0e-9)


def test_a_fully_reversed_domain_shakes_down_at_first_yield(model):
    """An identity rather than a benchmark: a load domain symmetric about zero
    builds no mean residual stress, so there is nothing for a residual field
    to buy and the two factors coincide."""
    reversed_domain = LoadDomain.box([-1.0, -1.0], [1.0, 1.0])
    history = CyclicProgram.through_vertices(
        reversed_domain, per_leg=1).elastic_history(model["envelope"])
    lam_el = elastic_factor(
        model["envelope"].vertex_stresses(reversed_domain), VonMises(), YIELD)

    def unsafe(lam):
        try:
            state = cyclic_steady_state(lam * history, model["tables"],
                                        model["free"], model["C"], YIELD,
                                        max_cycles=120)
            return state.regime() not in SAFE
        except Exception:
            return True

    lo, hi = 0.8 * lam_el, 1.6 * lam_el
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if unsafe(mid) else (mid, hi)
    assert 0.5 * (lo + hi) == pytest.approx(lam_el, rel=5.0e-3)


def test_the_ratchet_limit_is_reported_as_infinite_not_as_the_shakedown_one(model):
    """This example does not run the growth-ratio instrument, and returning
    ``lambda_rr = lambda_sd`` would silently assert an empty alternating band
    that the plate does not have -- measured alternating at
    ``1.15 lambda_sd``."""
    sigma_y = np.full(model["tables"].n_points, YIELD)
    assert not np.isfinite(transition_loads(model, sigma_y, 1.0).ratchet)


# ---------------------------------------------------------------------------
# the random field
# ---------------------------------------------------------------------------

def test_the_expansion_lives_on_the_quadrature_points(model):
    kl = expansion(model, 0.5)
    assert kl.n_points == model["tables"].n_points
    assert kl.variance_captured >= 0.95


def test_the_input_vector_unpacks_in_the_documented_order(model):
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    assert dist.dim == kl.n_modes + 2
    theta = np.arange(dist.dim, dtype=float)[None, :]
    xi, ratio, modulus = unpack(theta, kl.n_modes)
    assert xi.shape == (1, kl.n_modes)
    assert ratio[0] == kl.n_modes
    assert modulus[0] == kl.n_modes + 1


def test_a_weaker_field_gives_a_lower_shakedown_factor(model):
    """Monotone in the yield stress, which is the one qualitative property the
    whole study rests on."""
    kl = expansion(model, 0.5)
    field = LogNormalField(kl, mean=YIELD, cov=0.10)
    strong = transition_loads(model, 1.2 * field.realize(np.zeros(kl.n_modes)),
                              1.0)
    weak = transition_loads(model, 0.8 * field.realize(np.zeros(kl.n_modes)),
                            1.0)
    assert weak.shakedown < strong.shakedown
    assert weak.elastic < strong.elastic


def test_scaling_the_field_scales_both_transition_loads(model):
    """Degree one in the yield stress, exactly: the yield function is
    positively homogeneous and the admissible residual fields form a cone."""
    kl = expansion(model, 0.5)
    field = LogNormalField(kl, mean=YIELD, cov=0.10)
    sigma_y = field.realize(np.linspace(-1.0, 1.0, kl.n_modes))
    base = transition_loads(model, sigma_y, 1.0)
    for k in (0.6, 1.5):
        scaled = transition_loads(model, k * sigma_y, 1.0)
        assert scaled.elastic == pytest.approx(k * base.elastic, rel=1.0e-10)
        assert scaled.shakedown == pytest.approx(k * base.shakedown,
                                                 rel=1.0e-9)


# ---------------------------------------------------------------------------
# the probability integration
# ---------------------------------------------------------------------------

def test_the_bands_sum_to_one_exactly(model):
    """Structural, not a tolerance: each band is a difference of cumulative
    distribution functions of the same transition loads, so the four add to
    one whatever the point count or the load factor."""
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(0)
    theta = dist.sample(12, rng=rng)
    points = RepresentativePoints(theta, np.full(12, 1.0 / 12))
    loads = sweep(model, kl, theta)
    lam = np.linspace(0.5 * loads.elastic.min(), 1.5 * loads.shakedown.max(),
                      9)
    bands = curves(points, lam, loads)
    total = sum(bands[k] for k in ("E", "S", "AP", "R"))
    assert np.abs(total - 1.0).max() == 0.0


def test_every_band_is_a_probability(model):
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(1)
    theta = dist.sample(10, rng=rng)
    points = RepresentativePoints(theta, np.full(10, 0.1))
    loads = sweep(model, kl, theta)
    lam = np.linspace(0.5 * loads.elastic.min(), 1.5 * loads.shakedown.max(),
                      7)
    for value in curves(points, lam, loads).values():
        assert (value >= -1.0e-15).all() and (value <= 1.0 + 1.0e-15).all()


def test_ratcheting_is_empty_while_the_ratchet_limit_is_infinite(model):
    """The consequence of not running the growth ratio: all of "not shaking
    down" lands in the alternating band. Asserted so that a later change which
    starts computing ``lambda_rr`` cannot quietly leave the figures reading
    the old way."""
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(2)
    theta = dist.sample(8, rng=rng)
    points = RepresentativePoints(theta, np.full(8, 0.125))
    loads = sweep(model, kl, theta)
    lam = np.linspace(0.5 * loads.elastic.min(), 1.5 * loads.shakedown.max(),
                      5)
    assert np.abs(curves(points, lam, loads)["R"]).max() == 0.0


def test_the_reference_curves_agree_with_the_bands_on_the_same_points(model):
    """Counting and the Heaviside sum are the same operation on equal weights,
    so these must agree exactly -- which is what makes the Monte Carlo
    reference a reference rather than a second approximation."""
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(3)
    theta = dist.sample(16, rng=rng)
    points = RepresentativePoints(theta, np.full(16, 1.0 / 16))
    loads = sweep(model, kl, theta)
    lam = np.linspace(0.5 * loads.elastic.min(), 1.5 * loads.shakedown.max(),
                      11)
    bands = curves(points, lam, loads)
    ref = monte_carlo(loads, lam)
    for key in ("E", "S"):
        assert np.abs(bands[key] - ref[key]).max() < 1.0e-12


# ---------------------------------------------------------------------------
# the activity map
# ---------------------------------------------------------------------------

def test_the_activity_map_is_a_probability_per_point(model):
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(4)
    theta = dist.sample(4, rng=rng)
    loads = sweep(model, kl, theta)
    lam = float(np.mean(loads.shakedown))
    act = plastic_activity(model, kl, theta, lam)
    assert act.probability.shape == (model["tables"].n_points,)
    assert (act.probability >= 0.0).all() and (act.probability <= 1.0).all()
    assert act.counted + act.nonperiodic == 4


def test_nothing_is_active_well_below_the_elastic_limit(model):
    """With yield enforced as a constraint the plastic strain range is
    *identically* zero below the local shakedown limit, not merely small, so
    this is an exact emptiness rather than a thresholded one."""
    kl = expansion(model, 0.5)
    dist = distribution(kl.n_modes)
    rng = np.random.default_rng(5)
    theta = dist.sample(3, rng=rng)
    loads = sweep(model, kl, theta)
    act = plastic_activity(model, kl, theta, 0.5 * float(loads.elastic.min()))
    assert act.nonperiodic == 0
    assert act.probability.max() == 0.0
