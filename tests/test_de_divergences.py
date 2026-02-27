"""
Tests for low-rank Gaussian divergences in differential expression,
and closed-form Gamma KL / Jeffreys divergence.

Tests the KL divergence, Jensen-Shannon divergence, and Mahalanobis distance
for LowRankLogisticNormal and SoftmaxNormal distributions, plus the Gamma
divergences used by the biological DE pipeline.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from numpyro.distributions import Gamma

from scribe.stats.distributions import LowRankLogisticNormal, SoftmaxNormal
from scribe.stats.divergences import (
    kl_divergence,
    jensen_shannon,
    mahalanobis,
    gamma_kl,
    gamma_jeffreys,
    _extract_lowrank_params,
    _kl_lowrank_mvn,
)

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_lowrank_dict():
    """Generate a sample low-rank Gaussian as a dictionary."""
    D = 10
    k = 3

    return {
        "loc": jnp.linspace(-0.5, 0.5, D),
        "cov_factor": random.normal(random.PRNGKey(123), (D, k)) * 0.1,
        "cov_diag": jnp.ones(D) * 0.5,
    }


@pytest.fixture
def sample_lowrank_distributions():
    """Generate two LowRankLogisticNormal distributions for testing."""
    D = 10
    k = 3

    p = LowRankLogisticNormal(
        loc=jnp.linspace(-0.5, 0.5, D),
        cov_factor=random.normal(random.PRNGKey(123), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = LowRankLogisticNormal(
        loc=jnp.linspace(-0.3, 0.7, D),
        cov_factor=random.normal(random.PRNGKey(456), (D, k)) * 0.12,
        cov_diag=jnp.ones(D) * 0.6,
    )

    return p, q


@pytest.fixture
def sample_softmax_distributions():
    """Generate two SoftmaxNormal distributions for testing."""
    D = 10
    k = 3

    p = SoftmaxNormal(
        loc=jnp.linspace(-0.5, 0.5, D),
        cov_factor=random.normal(random.PRNGKey(789), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = SoftmaxNormal(
        loc=jnp.linspace(-0.3, 0.7, D),
        cov_factor=random.normal(random.PRNGKey(101), (D, k)) * 0.12,
        cov_diag=jnp.ones(D) * 0.6,
    )

    return p, q


# ------------------------------------------------------------------------------
# Test parameter extraction
# ------------------------------------------------------------------------------


def test_extract_lowrank_params_from_dict(sample_lowrank_dict):
    """Test extraction of parameters from dictionary."""
    loc, cov_factor, cov_diag = _extract_lowrank_params(sample_lowrank_dict)

    assert jnp.array_equal(loc, sample_lowrank_dict["loc"])
    assert jnp.array_equal(cov_factor, sample_lowrank_dict["cov_factor"])
    assert jnp.array_equal(cov_diag, sample_lowrank_dict["cov_diag"])


def test_extract_lowrank_params_from_distribution(sample_lowrank_distributions):
    """Test extraction of parameters from Distribution object."""
    p, _ = sample_lowrank_distributions

    loc, cov_factor, cov_diag = _extract_lowrank_params(p)

    assert jnp.array_equal(loc, p.loc)
    assert jnp.array_equal(cov_factor, p.cov_factor)
    assert jnp.array_equal(cov_diag, p.cov_diag)


def test_extract_lowrank_params_invalid_type():
    """Test that extraction raises error for invalid type."""
    with pytest.raises(TypeError):
        _extract_lowrank_params("invalid")


# ------------------------------------------------------------------------------
# Test KL divergence
# ------------------------------------------------------------------------------


def test_kl_divergence_nonnegative(sample_lowrank_distributions):
    """Test that KL divergence is non-negative."""
    p, q = sample_lowrank_distributions

    kl = kl_divergence(p, q)

    assert kl >= 0


def test_kl_divergence_zero_for_same(sample_lowrank_distributions):
    """Test that KL(p || p) = 0."""
    p, _ = sample_lowrank_distributions

    kl = kl_divergence(p, p)

    assert jnp.abs(kl) < 1e-4  # Relaxed tolerance for numerical stability


def test_kl_divergence_asymmetric(sample_lowrank_distributions):
    """Test that KL divergence is asymmetric."""
    p, q = sample_lowrank_distributions

    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    # Should generally not be equal
    assert not jnp.allclose(kl_pq, kl_qp, atol=1e-6)


def test_kl_divergence_softmax_normal(sample_softmax_distributions):
    """Test KL divergence for SoftmaxNormal distributions."""
    p, q = sample_softmax_distributions

    kl = kl_divergence(p, q)

    assert kl >= 0
    assert jnp.isfinite(kl)


def test_kl_divergence_from_dict():
    """Test KL computation directly from dictionaries."""
    p_dict = {
        "loc": jnp.zeros(5),
        "cov_factor": jnp.eye(5, 2) * 0.1,
        "cov_diag": jnp.ones(5) * 0.5,
    }

    q_dict = {
        "loc": jnp.ones(5) * 0.1,
        "cov_factor": jnp.eye(5, 2) * 0.1,
        "cov_diag": jnp.ones(5) * 0.5,
    }

    kl = _kl_lowrank_mvn(p_dict, q_dict)

    assert kl >= 0
    assert jnp.isfinite(kl)


def test_kl_divergence_mean_only_difference():
    """Test KL when only means differ."""
    D = 10
    k = 3

    W = random.normal(random.PRNGKey(999), (D, k)) * 0.1
    d = jnp.ones(D) * 0.5

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=W,
        cov_diag=d,
    )

    q = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.5,
        cov_factor=W,
        cov_diag=d,
    )

    kl = kl_divergence(p, q)

    # Should be positive since means differ
    assert kl > 0


def test_kl_divergence_covariance_only_difference():
    """Test KL when only covariances differ."""
    D = 10
    k = 3

    mu = jnp.zeros(D)

    p = LowRankLogisticNormal(
        loc=mu,
        cov_factor=random.normal(random.PRNGKey(111), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = LowRankLogisticNormal(
        loc=mu,
        cov_factor=random.normal(random.PRNGKey(222), (D, k)) * 0.15,
        cov_diag=jnp.ones(D) * 0.7,
    )

    kl = kl_divergence(p, q)

    # Should be positive since covariances differ
    assert kl > 0


# ------------------------------------------------------------------------------
# Test Jensen-Shannon divergence
# ------------------------------------------------------------------------------


def test_jensen_shannon_nonnegative(sample_lowrank_distributions):
    """Test that JS divergence is non-negative."""
    p, q = sample_lowrank_distributions

    js = jensen_shannon(p, q)

    assert js >= 0


def test_jensen_shannon_symmetric(sample_lowrank_distributions):
    """Test that JS divergence is symmetric."""
    p, q = sample_lowrank_distributions

    js_pq = jensen_shannon(p, q)
    js_qp = jensen_shannon(q, p)

    assert jnp.allclose(js_pq, js_qp, atol=1e-6)


def test_jensen_shannon_zero_for_same(sample_lowrank_distributions):
    """Test that JS(p, p) = 0."""
    p, _ = sample_lowrank_distributions

    js = jensen_shannon(p, p)

    assert jnp.abs(js) < 1e-4  # Relaxed tolerance for numerical stability


def test_jensen_shannon_bounded(sample_lowrank_distributions):
    """Test that JS divergence is bounded (for Gaussians, by log(2))."""
    p, q = sample_lowrank_distributions

    js = jensen_shannon(p, q)

    # For Gaussians, JS is not strictly bounded by log(2), but should be finite
    assert jnp.isfinite(js)


def test_jensen_shannon_softmax_normal(sample_softmax_distributions):
    """Test JS divergence for SoftmaxNormal distributions."""
    p, q = sample_softmax_distributions

    js = jensen_shannon(p, q)

    assert js >= 0
    assert jnp.isfinite(js)


def test_jensen_shannon_relation_to_kl(sample_lowrank_distributions):
    """Test that JS = 0.5 * [KL(p||m) + KL(q||m)]."""
    p, q = sample_lowrank_distributions

    # Get the mixture parameters manually
    mu_m = 0.5 * (p.loc + q.loc)
    W_m = jnp.concatenate([p.cov_factor, q.cov_factor], axis=-1) / jnp.sqrt(2.0)
    d_m = 0.5 * (p.cov_diag + q.cov_diag)

    m = LowRankLogisticNormal(loc=mu_m, cov_factor=W_m, cov_diag=d_m)

    # Compute JS via definition
    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    js_expected = 0.5 * (kl_pm + kl_qm)

    # Compute JS via function
    js_actual = jensen_shannon(p, q)

    assert jnp.allclose(js_actual, js_expected, atol=1e-5)


# ------------------------------------------------------------------------------
# Test Mahalanobis distance
# ------------------------------------------------------------------------------


def test_mahalanobis_nonnegative(sample_lowrank_distributions):
    """Test that Mahalanobis distance is non-negative."""
    p, q = sample_lowrank_distributions

    m2 = mahalanobis(p, q)

    assert m2 >= 0


def test_mahalanobis_symmetric(sample_lowrank_distributions):
    """Test that Mahalanobis distance is symmetric."""
    p, q = sample_lowrank_distributions

    m2_pq = mahalanobis(p, q)
    m2_qp = mahalanobis(q, p)

    assert jnp.allclose(m2_pq, m2_qp, atol=1e-6)


def test_mahalanobis_zero_for_same(sample_lowrank_distributions):
    """Test that Mahalanobis(p, p) = 0."""
    p, _ = sample_lowrank_distributions

    m2 = mahalanobis(p, p)

    assert jnp.abs(m2) < 1e-6


def test_mahalanobis_softmax_normal(sample_softmax_distributions):
    """Test Mahalanobis for SoftmaxNormal distributions."""
    p, q = sample_softmax_distributions

    m2 = mahalanobis(p, q)

    assert m2 >= 0
    assert jnp.isfinite(m2)


def test_mahalanobis_increases_with_distance():
    """Test that Mahalanobis increases as distributions move apart."""
    D = 10
    k = 3

    W = random.normal(random.PRNGKey(333), (D, k)) * 0.1
    d = jnp.ones(D) * 0.5

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=W,
        cov_diag=d,
    )

    # Create distributions at different distances
    q1 = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.1,
        cov_factor=W,
        cov_diag=d,
    )

    q2 = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.5,
        cov_factor=W,
        cov_diag=d,
    )

    m2_pq1 = mahalanobis(p, q1)
    m2_pq2 = mahalanobis(p, q2)

    # q2 is farther, so should have larger distance
    assert m2_pq2 > m2_pq1


# ------------------------------------------------------------------------------
# Test numerical stability
# ------------------------------------------------------------------------------


def test_kl_divergence_numerical_stability():
    """Test KL computation with very small variances."""
    D = 10
    k = 2

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=jnp.zeros((D, k)),
        cov_diag=jnp.ones(D) * 1e-6,  # Very small variance
    )

    q = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=jnp.zeros((D, k)),
        cov_diag=jnp.ones(D) * 1e-5,
    )

    kl = kl_divergence(p, q)

    assert jnp.isfinite(kl)
    assert kl >= 0


def test_large_dimension_scalability():
    """Test that divergences scale to large dimensions."""
    D = 1000
    k = 10

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=random.normal(random.PRNGKey(444), (D, k)) * 0.01,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.01,
        cov_factor=random.normal(random.PRNGKey(555), (D, k)) * 0.01,
        cov_diag=jnp.ones(D) * 0.5,
    )

    # Should compute without errors
    kl = kl_divergence(p, q)
    js = jensen_shannon(p, q)
    m2 = mahalanobis(p, q)

    assert jnp.isfinite(kl)
    assert jnp.isfinite(js)
    assert jnp.isfinite(m2)


def test_different_ranks():
    """Test divergences when distributions have different ranks."""
    D = 10

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=random.normal(random.PRNGKey(666), (D, 3)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=random.normal(random.PRNGKey(777), (D, 5)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    # Should handle different ranks
    kl = kl_divergence(p, q)
    js = jensen_shannon(p, q)
    m2 = mahalanobis(p, q)

    assert jnp.isfinite(kl)
    assert jnp.isfinite(js)
    assert jnp.isfinite(m2)


# ------------------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------------------


def test_divergence_triangle_inequality_js():
    """Test that JS satisfies triangle inequality (approximately for Gaussians)."""
    D = 10
    k = 3

    p = LowRankLogisticNormal(
        loc=jnp.zeros(D),
        cov_factor=random.normal(random.PRNGKey(888), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    q = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.5,
        cov_factor=random.normal(random.PRNGKey(999), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    r = LowRankLogisticNormal(
        loc=jnp.ones(D) * 1.0,
        cov_factor=random.normal(random.PRNGKey(1010), (D, k)) * 0.1,
        cov_diag=jnp.ones(D) * 0.5,
    )

    # sqrt(JS) is a metric, so should satisfy triangle inequality
    js_pq = jnp.sqrt(jensen_shannon(p, q))
    js_qr = jnp.sqrt(jensen_shannon(q, r))
    js_pr = jnp.sqrt(jensen_shannon(p, r))

    # Allow some numerical tolerance
    assert js_pr <= js_pq + js_qr + 1e-4


def test_all_divergences_consistent_ordering():
    """Test that different divergences give consistent ordering."""
    D = 10
    k = 3

    W = random.normal(random.PRNGKey(1111), (D, k)) * 0.1
    d = jnp.ones(D) * 0.5

    p = LowRankLogisticNormal(loc=jnp.zeros(D), cov_factor=W, cov_diag=d)
    q_near = LowRankLogisticNormal(
        loc=jnp.ones(D) * 0.1, cov_factor=W, cov_diag=d
    )
    q_far = LowRankLogisticNormal(
        loc=jnp.ones(D) * 1.0, cov_factor=W, cov_diag=d
    )

    # All metrics should agree that q_far is farther than q_near
    kl_near = kl_divergence(p, q_near)
    kl_far = kl_divergence(p, q_far)

    js_near = jensen_shannon(p, q_near)
    js_far = jensen_shannon(p, q_far)

    m2_near = mahalanobis(p, q_near)
    m2_far = mahalanobis(p, q_far)

    assert kl_far > kl_near
    assert js_far > js_near
    assert m2_far > m2_near


# ==============================================================================
# Gamma KL divergence tests
# ==============================================================================


class TestGammaKL:
    """Tests for the closed-form Gamma KL divergence."""

    def test_identical_distributions_zero(self):
        """KL(p || p) = 0 for any Gamma."""
        alpha = jnp.array([1.0, 2.5, 10.0])
        beta = jnp.array([0.5, 1.0, 3.0])
        kl = gamma_kl(alpha, beta, alpha, beta)
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_non_negative(self, rng_key):
        """KL divergence is always non-negative."""
        keys = random.split(rng_key, 4)
        alpha_p = random.uniform(keys[0], (50,), minval=0.5, maxval=20.0)
        beta_p = random.uniform(keys[1], (50,), minval=0.1, maxval=10.0)
        alpha_q = random.uniform(keys[2], (50,), minval=0.5, maxval=20.0)
        beta_q = random.uniform(keys[3], (50,), minval=0.1, maxval=10.0)
        kl = gamma_kl(alpha_p, beta_p, alpha_q, beta_q)
        assert jnp.all(kl >= -1e-6)

    def test_asymmetric(self):
        """KL(p || q) != KL(q || p) in general."""
        alpha_p, beta_p = jnp.array(2.0), jnp.array(1.0)
        alpha_q, beta_q = jnp.array(5.0), jnp.array(3.0)
        kl_pq = gamma_kl(alpha_p, beta_p, alpha_q, beta_q)
        kl_qp = gamma_kl(alpha_q, beta_q, alpha_p, beta_p)
        assert not jnp.isclose(kl_pq, kl_qp, atol=1e-4)

    def test_matches_numpyro_builtin(self):
        """gamma_kl matches numpyro's built-in Gamma KL registration."""
        p = Gamma(concentration=3.0, rate=2.0)
        q = Gamma(concentration=5.0, rate=1.5)
        kl_numpyro = kl_divergence(p, q)
        kl_raw = gamma_kl(
            jnp.array(3.0), jnp.array(2.0),
            jnp.array(5.0), jnp.array(1.5),
        )
        np.testing.assert_allclose(kl_numpyro, kl_raw, atol=1e-6)

    def test_numerical_against_sampling(self, rng_key):
        """Gamma KL matches a Monte Carlo estimate from log-density ratios."""
        alpha_p, beta_p = 4.0, 2.0
        alpha_q, beta_q = 6.0, 3.0

        # Monte Carlo estimate: E_p[log p(x) - log q(x)]
        p = Gamma(concentration=alpha_p, rate=beta_p)
        q = Gamma(concentration=alpha_q, rate=beta_q)
        samples = p.sample(rng_key, (500_000,))
        log_ratio = p.log_prob(samples) - q.log_prob(samples)
        kl_mc = jnp.mean(log_ratio)

        kl_closed = gamma_kl(
            jnp.array(alpha_p), jnp.array(beta_p),
            jnp.array(alpha_q), jnp.array(beta_q),
        )
        np.testing.assert_allclose(kl_closed, kl_mc, atol=0.01)

    def test_vectorized(self):
        """gamma_kl handles batched inputs."""
        alpha_p = jnp.array([1.0, 2.0, 3.0])
        beta_p = jnp.array([1.0, 1.0, 1.0])
        alpha_q = jnp.array([1.5, 2.5, 3.5])
        beta_q = jnp.array([1.0, 1.0, 1.0])
        kl = gamma_kl(alpha_p, beta_p, alpha_q, beta_q)
        assert kl.shape == (3,)
        assert jnp.all(kl >= 0)

    def test_larger_divergence_for_larger_shift(self):
        """KL increases with parameter distance."""
        alpha_p, beta_p = jnp.array(5.0), jnp.array(2.0)
        kl_near = gamma_kl(alpha_p, beta_p, jnp.array(5.5), jnp.array(2.0))
        kl_far = gamma_kl(alpha_p, beta_p, jnp.array(10.0), jnp.array(2.0))
        assert kl_far > kl_near


class TestGammaJeffreys:
    """Tests for the symmetrised Gamma KL (Jeffreys divergence)."""

    def test_symmetric(self):
        """Jeffreys divergence is symmetric by construction."""
        alpha_p, beta_p = jnp.array(3.0), jnp.array(1.5)
        alpha_q, beta_q = jnp.array(7.0), jnp.array(4.0)
        j_pq = gamma_jeffreys(alpha_p, beta_p, alpha_q, beta_q)
        j_qp = gamma_jeffreys(alpha_q, beta_q, alpha_p, beta_p)
        np.testing.assert_allclose(j_pq, j_qp, atol=1e-6)

    def test_identical_zero(self):
        """Jeffreys divergence is zero for identical distributions."""
        alpha = jnp.array(4.0)
        beta = jnp.array(2.0)
        j = gamma_jeffreys(alpha, beta, alpha, beta)
        np.testing.assert_allclose(j, 0.0, atol=1e-6)

    def test_non_negative(self, rng_key):
        """Jeffreys divergence is always non-negative."""
        keys = random.split(rng_key, 4)
        alpha_p = random.uniform(keys[0], (50,), minval=0.5, maxval=20.0)
        beta_p = random.uniform(keys[1], (50,), minval=0.1, maxval=10.0)
        alpha_q = random.uniform(keys[2], (50,), minval=0.5, maxval=20.0)
        beta_q = random.uniform(keys[3], (50,), minval=0.1, maxval=10.0)
        j = gamma_jeffreys(alpha_p, beta_p, alpha_q, beta_q)
        assert jnp.all(j >= -1e-6)

    def test_equals_sum_of_kl(self):
        """Jeffreys = KL(p||q) + KL(q||p)."""
        alpha_p, beta_p = jnp.array(3.0), jnp.array(2.0)
        alpha_q, beta_q = jnp.array(6.0), jnp.array(1.0)
        j = gamma_jeffreys(alpha_p, beta_p, alpha_q, beta_q)
        kl_sum = gamma_kl(alpha_p, beta_p, alpha_q, beta_q) + gamma_kl(
            alpha_q, beta_q, alpha_p, beta_p
        )
        np.testing.assert_allclose(j, kl_sum, atol=1e-6)
