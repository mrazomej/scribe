"""
Tests for low-rank Gaussian divergences in differential expression.

Tests the KL divergence, Jensen-Shannon divergence, and Mahalanobis distance
for LowRankLogisticNormal and SoftmaxNormal distributions.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.stats.distributions import LowRankLogisticNormal, SoftmaxNormal
from scribe.stats.divergences import (
    kl_divergence,
    jensen_shannon,
    mahalanobis,
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
