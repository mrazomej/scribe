"""Tests for dimension-aware parameter extraction (scribe.de._extract).

Validates that ``extract_alr_params`` correctly handles:
- D-dimensional embedded ALR dicts (from fit_logistic_normal_from_posterior)
- (D-1)-dimensional raw ALR dicts
- LowRankLogisticNormal distribution objects
- SoftmaxNormal distribution objects
"""

import pytest
import jax.numpy as jnp
from jax import random

from scribe.de import extract_alr_params
from scribe.stats.distributions import LowRankLogisticNormal, SoftmaxNormal


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def embedded_alr_dict():
    """Create a D-dimensional embedded ALR dict (as from
    fit_logistic_normal_from_posterior).

    D_alr = 4, so the dict has D=5 dimensional params with
    zeros appended for the reference component.
    """
    D_alr = 4
    k = 2
    # Raw ALR values
    mu_alr = jnp.array([0.1, 0.2, 0.3, 0.4])
    W_alr = random.normal(random.PRNGKey(42), (D_alr, k)) * 0.1
    d_alr = jnp.ones(D_alr) * 0.5

    # Embed: append zeros for reference
    mu_embedded = jnp.concatenate([mu_alr, jnp.array([0.0])])
    W_embedded = jnp.concatenate([W_alr, jnp.zeros((1, k))], axis=0)
    d_embedded = jnp.concatenate([d_alr, jnp.array([0.0])])

    return {
        "loc": mu_embedded,
        "cov_factor": W_embedded,
        "cov_diag": d_embedded,
    }, mu_alr, W_alr, d_alr


@pytest.fixture
def raw_alr_dict():
    """Create a (D-1)-dimensional raw ALR dict (no embedding)."""
    D_alr = 4
    k = 2
    mu = jnp.array([0.1, 0.2, 0.3, 0.4])
    W = random.normal(random.PRNGKey(42), (D_alr, k)) * 0.1
    d = jnp.ones(D_alr) * 0.5

    return {"loc": mu, "cov_factor": W, "cov_diag": d}, mu, W, d


# --------------------------------------------------------------------------
# Tests for embedded ALR dict
# --------------------------------------------------------------------------


def test_extract_embedded_alr_strips_reference(embedded_alr_dict):
    """Embedded ALR dict should be stripped to (D-1) dimensions."""
    model_dict, expected_mu, expected_W, expected_d = embedded_alr_dict

    mu, W, d = extract_alr_params(model_dict)

    # Should be (D-1)-dimensional
    assert mu.shape == (4,)
    assert W.shape == (4, 2)
    assert d.shape == (4,)

    # Values should match the raw ALR before embedding
    assert jnp.allclose(mu, expected_mu)
    assert jnp.allclose(W, expected_W)
    assert jnp.allclose(d, expected_d)


def test_extract_raw_alr_unchanged(raw_alr_dict):
    """Raw (D-1)-dimensional ALR dict should be returned unchanged."""
    model_dict, expected_mu, expected_W, expected_d = raw_alr_dict

    mu, W, d = extract_alr_params(model_dict)

    # Should remain (D-1)-dimensional
    assert mu.shape == (4,)
    assert W.shape == (4, 2)
    assert d.shape == (4,)

    # Values should be identical
    assert jnp.allclose(mu, expected_mu)
    assert jnp.allclose(W, expected_W)
    assert jnp.allclose(d, expected_d)


# --------------------------------------------------------------------------
# Tests for distribution objects
# --------------------------------------------------------------------------


def test_extract_lowrank_logistic_normal():
    """LowRankLogisticNormal params should be returned as-is (D-1 dim)."""
    D_alr = 4
    k = 2
    mu = jnp.zeros(D_alr)
    W = random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1
    d = jnp.ones(D_alr) * 0.5

    dist = LowRankLogisticNormal(loc=mu, cov_factor=W, cov_diag=d)
    extracted_mu, extracted_W, extracted_d = extract_alr_params(dist)

    assert extracted_mu.shape == (D_alr,)
    assert extracted_W.shape == (D_alr, k)
    assert extracted_d.shape == (D_alr,)

    # Values should be identical to what was passed
    assert jnp.allclose(extracted_mu, mu)
    assert jnp.allclose(extracted_W, W)
    assert jnp.allclose(extracted_d, d)


def test_extract_softmax_normal():
    """SoftmaxNormal params should be stripped from D to D-1 dimensions."""
    D = 5  # D-dimensional (embedded)
    k = 2
    mu = jnp.array([0.1, 0.2, 0.3, 0.4, 0.0])
    W = jnp.concatenate(
        [random.normal(random.PRNGKey(456), (D - 1, k)) * 0.1,
         jnp.zeros((1, k))],
        axis=0,
    )
    d = jnp.concatenate([jnp.ones(D - 1) * 0.5, jnp.array([0.0])])

    dist = SoftmaxNormal(loc=mu, cov_factor=W, cov_diag=d)
    extracted_mu, extracted_W, extracted_d = extract_alr_params(dist)

    # Should be stripped to (D-1) dimensions
    assert extracted_mu.shape == (D - 1,)
    assert extracted_W.shape == (D - 1, k)
    assert extracted_d.shape == (D - 1,)


# --------------------------------------------------------------------------
# Tests for error handling
# --------------------------------------------------------------------------


def test_extract_unsupported_type_raises():
    """Passing an unsupported type should raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported model type"):
        extract_alr_params("not_a_model")


def test_extract_unsupported_type_list_raises():
    """Passing a list should raise TypeError."""
    with pytest.raises(TypeError, match="Unsupported model type"):
        extract_alr_params([1, 2, 3])


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_extract_minimal_dimension():
    """Test extraction with minimal dimension (D_alr=1)."""
    # Embedded: D=2
    model = {
        "loc": jnp.array([0.5, 0.0]),
        "cov_factor": jnp.array([[0.1], [0.0]]),
        "cov_diag": jnp.array([0.5, 0.0]),
    }
    mu, W, d = extract_alr_params(model)

    assert mu.shape == (1,)
    assert W.shape == (1, 1)
    assert d.shape == (1,)


def test_extract_large_dimension():
    """Test extraction with large dimension."""
    D_alr = 999
    k = 10
    mu_alr = jnp.zeros(D_alr)
    W_alr = random.normal(random.PRNGKey(789), (D_alr, k)) * 0.01
    d_alr = jnp.ones(D_alr) * 0.5

    # Create embedded version
    mu_embedded = jnp.concatenate([mu_alr, jnp.array([0.0])])
    W_embedded = jnp.concatenate([W_alr, jnp.zeros((1, k))], axis=0)
    d_embedded = jnp.concatenate([d_alr, jnp.array([0.0])])

    model = {
        "loc": mu_embedded,
        "cov_factor": W_embedded,
        "cov_diag": d_embedded,
    }
    mu, W, d = extract_alr_params(model)

    assert mu.shape == (D_alr,)
    assert W.shape == (D_alr, k)
    assert d.shape == (D_alr,)


def test_extract_nonzero_last_diag_not_stripped():
    """Dict with non-zero last diagonal should NOT be stripped."""
    # This simulates a genuinely (D-1)-dimensional dict that happens to
    # have small but non-zero last elements
    D_alr = 4
    k = 2
    mu = jnp.array([0.1, 0.2, 0.3, 0.4])
    W = random.normal(random.PRNGKey(999), (D_alr, k)) * 0.1
    d = jnp.array([0.5, 0.5, 0.5, 0.3])  # Last element is not zero

    model = {"loc": mu, "cov_factor": W, "cov_diag": d}
    extracted_mu, extracted_W, extracted_d = extract_alr_params(model)

    # Should NOT be stripped -- returned as-is
    assert extracted_mu.shape == (4,)
    assert jnp.allclose(extracted_d, d)
