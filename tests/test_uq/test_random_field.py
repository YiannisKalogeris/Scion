"""The truncated Karhunen-Loeve expansion.

The tests that matter here are the ones that separate the weighted
eigenproblem from the naive one, because both look right on a uniform mesh and
both reproduce the covariance exactly when every mode is kept.
"""

import numpy as np
import pytest

from scion.uq.forward.random_field import (KERNEL_REGISTRY, KarhunenLoeve,
                                           LogNormalField)


def _graded_line(n=100, power=2.5):
    """A 1-D mesh whose cell sizes span three orders of magnitude."""
    t = np.linspace(0.0, 1.0, n) ** power
    return t[:, None], np.gradient(t)


def _uniform_square(n=12):
    g = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([x.ravel(), y.ravel()], axis=1)
    return pts, np.full(pts.shape[0], 1.0 / pts.shape[0])


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

def test_modes_are_orthonormal_against_the_quadrature_weights():
    """``INT phi_k phi_l dV = delta_kl``, which is what the expansion's
    ``sqrt(lam)`` coefficients assume. On a graded mesh the unweighted
    orthonormality that ``eigh`` delivers is a different statement."""
    pts, w = _graded_line()
    kl = KarhunenLoeve.from_points(pts, w, 0.2, n_modes=pts.shape[0])
    gram = (kl.modes * w) @ kl.modes.T
    assert np.abs(gram - np.eye(kl.n_modes)).max() < 1.0e-12


def test_every_mode_retained_returns_the_covariance_diagonal():
    """``sum_k lam_k phi_k(x)^2 = C(x,x) = 1``. True of the naive
    eigendecomposition as well -- see the next test for what is not."""
    pts, w = _graded_line()
    kl = KarhunenLoeve.from_points(pts, w, 0.2, n_modes=pts.shape[0])
    assert np.abs(kl.pointwise_variance - 1.0).max() < 1.0e-10
    assert kl.variance_captured == pytest.approx(1.0, abs=1.0e-12)


def test_the_weighted_truncation_beats_the_naive_one_in_l2_of_the_domain():
    """The KL expansion's defining property is that it is the optimal
    ``k``-term approximation of the covariance in ``L2(D)``. Solving ``eigh(C)``
    instead gives up that optimality *and* reports a larger variance fraction,
    because it normalises by ``trace(C)`` rather than by the domain measure --
    flattering and less accurate at once, which is why it needs a test."""
    pts, w = _graded_line()
    ell = 0.2
    d = np.abs(pts - pts.T)
    cov = np.exp(-d / ell)

    for k in (4, 8, 16):
        kl = KarhunenLoeve.from_points(pts, w, ell, n_modes=k)
        good = kl.modes.T @ (kl.eigenvalues[:, None] * kl.modes)

        lam, vec = np.linalg.eigh(cov)
        order = np.argsort(lam)[::-1]
        lam_n = np.maximum(lam[order], 0.0)[:k]
        vec_n = vec[:, order][:, :k].T
        naive = vec_n.T @ (lam_n[:, None] * vec_n)

        wt = w[:, None] * w[None, :]
        err_good = np.sqrt(((cov - good) ** 2 * wt).sum())
        err_naive = np.sqrt(((cov - naive) ** 2 * wt).sum())
        assert err_good < err_naive
        # and the naive one claims to have captured more than it has
        assert lam_n.sum() / pts.shape[0] > kl.variance_captured


# ---------------------------------------------------------------------------
# truncation
# ---------------------------------------------------------------------------

def test_variance_target_keeps_the_fewest_modes_that_reach_it():
    pts, w = _uniform_square()
    kl = KarhunenLoeve.from_points(pts, w, 0.5, variance_target=0.95)
    assert kl.variance_captured >= 0.95
    fewer = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=kl.n_modes - 1)
    assert fewer.variance_captured < 0.95


def test_a_shorter_correlation_length_needs_more_modes():
    """The eigenvalue spectrum decays more slowly as the field decorrelates,
    which is the whole cost model of a random-field problem."""
    pts, w = _uniform_square()
    counts = [KarhunenLoeve.from_points(pts, w, ell, variance_target=0.9).n_modes
              for ell in (2.0, 1.0, 0.5, 0.25)]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_truncation_lowers_the_pointwise_variance_and_says_where():
    """``variance_captured`` is an integral, so it cannot report where the
    expansion is thin. Truncating makes realizations *less* extreme, which for
    a strength field is the unsafe direction."""
    pts, w = _uniform_square()
    kl = KarhunenLoeve.from_points(pts, w, 0.5, variance_target=0.9)
    pv = kl.pointwise_variance
    assert pv.min() < kl.variance_captured
    assert pv.max() <= 1.0 + 1.0e-10


def test_the_smoothed_kernel_decays_faster_than_the_exponential_one():
    pts, w = _uniform_square()
    n_exp = KarhunenLoeve.from_points(pts, w, 0.5, variance_target=0.95,
                                      kernel="exponential").n_modes
    n_sq = KarhunenLoeve.from_points(pts, w, 0.5, variance_target=0.95,
                                     kernel="squared_exponential").n_modes
    assert n_sq < n_exp


# ---------------------------------------------------------------------------
# realizations
# ---------------------------------------------------------------------------

def test_realizations_reproduce_the_covariance_by_sampling():
    """The end-to-end check: the sample covariance of many realizations must
    return the kernel it was built from, up to truncation."""
    pts, w = _uniform_square(8)
    kl = KarhunenLoeve.from_points(pts, w, 0.6, variance_target=0.995)
    rng = np.random.default_rng(0)
    g = kl.realize(rng.standard_normal((60000, kl.n_modes)))
    emp = np.cov(g, rowvar=False)
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    assert np.abs(emp - np.exp(-d / 0.6)).max() < 0.03


def test_realize_accepts_one_sample_or_many():
    pts, w = _uniform_square(6)
    kl = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=5)
    one = kl.realize(np.ones(5))
    many = kl.realize(np.ones((3, 5)))
    assert one.shape == (pts.shape[0],)
    assert many.shape == (3, pts.shape[0])
    # not bit-identical: a (3, k) matmul associates its sum differently from a
    # (k,) one, so this is round-off and not a broadcasting mistake
    assert np.abs(many - one[None]).max() < 1.0e-15


def test_a_zero_coefficient_vector_gives_the_mean_field():
    pts, w = _uniform_square(6)
    kl = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=5)
    assert np.abs(kl.realize(np.zeros(5))).max() == 0.0


# ---------------------------------------------------------------------------
# the lognormal wrapper
# ---------------------------------------------------------------------------

def test_lognormal_field_takes_the_quantitys_own_moments():
    """The same trap ``LogNormal`` carries: handing the target mean to the
    underlying normal inflates it by ``exp(s^2/2)``."""
    pts, w = _uniform_square(8)
    kl = KarhunenLoeve.from_points(pts, w, 0.6, variance_target=0.999)
    f = LogNormalField(kl, mean=355.0, cov=0.10)
    rng = np.random.default_rng(1)
    s = f.realize(rng.standard_normal((40000, kl.n_modes)))
    assert s.mean(axis=0).mean() == pytest.approx(355.0, rel=0.01)
    assert (s.std(axis=0) / s.mean(axis=0)).mean() == pytest.approx(0.10, rel=0.05)
    assert (s > 0.0).all()


def test_the_truncation_bias_in_the_mean_is_reported_and_correct():
    """Truncation leaves the pointwise variance below one, which biases the
    realized mean low by a factor the class predicts in closed form."""
    pts, w = _uniform_square(8)
    kl = KarhunenLoeve.from_points(pts, w, 0.3, variance_target=0.8)
    f = LogNormalField(kl, mean=355.0, cov=0.30)
    assert (f.mean_bias < 1.0).any()
    rng = np.random.default_rng(2)
    s = f.realize(rng.standard_normal((80000, kl.n_modes)))
    realized = s.mean(axis=0) / 355.0
    assert np.abs(realized - f.mean_bias).max() < 0.02


def test_lognormal_parameters_match_the_closed_form():
    pts, w = _uniform_square(4)
    kl = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=3)
    f = LogNormalField(kl, mean=2.0, cov=0.5)
    assert f.sigma == pytest.approx(np.sqrt(np.log(1.25)))
    assert f.mu == pytest.approx(np.log(2.0) - 0.5 * np.log(1.25))


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_the_constructor_refuses_an_ambiguous_truncation():
    pts, w = _uniform_square(4)
    with pytest.raises(ValueError, match="exactly one"):
        KarhunenLoeve.from_points(pts, w, 0.5)
    with pytest.raises(ValueError, match="exactly one"):
        KarhunenLoeve.from_points(pts, w, 0.5, n_modes=3, variance_target=0.9)


def test_nonpositive_weights_are_refused():
    """The symmetric form takes their square root, so a zero weight is not a
    degenerate case to tolerate but a point that cannot be normalised."""
    pts, w = _uniform_square(4)
    w = w.copy(); w[0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        KarhunenLoeve.from_points(pts, w, 0.5, n_modes=2)


def test_mismatched_weights_and_points_are_refused():
    pts, w = _uniform_square(4)
    with pytest.raises(ValueError, match="weights for"):
        KarhunenLoeve.from_points(pts, w[:-1], 0.5, n_modes=2)


def test_an_unknown_kernel_names_the_alternatives():
    pts, w = _uniform_square(4)
    with pytest.raises(ValueError, match="unknown kernel"):
        KarhunenLoeve.from_points(pts, w, 0.5, n_modes=2, kernel="matern")
    assert set(KERNEL_REGISTRY) == {"exponential", "squared_exponential"}


def test_too_many_modes_is_refused():
    pts, w = _uniform_square(4)
    with pytest.raises(ValueError, match="n_modes must be"):
        KarhunenLoeve.from_points(pts, w, 0.5, n_modes=pts.shape[0] + 1)


def test_a_nonpositive_mean_is_refused_for_a_lognormal_field():
    pts, w = _uniform_square(4)
    kl = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=2)
    with pytest.raises(ValueError, match="positive mean"):
        LogNormalField(kl, mean=0.0, cov=0.1)
    with pytest.raises(ValueError, match="cov must be positive"):
        LogNormalField(kl, mean=1.0, cov=0.0)


def test_a_wrong_coefficient_count_is_refused():
    pts, w = _uniform_square(4)
    kl = KarhunenLoeve.from_points(pts, w, 0.5, n_modes=3)
    with pytest.raises(ValueError, match="3 coefficients"):
        kl.realize(np.ones(4))
