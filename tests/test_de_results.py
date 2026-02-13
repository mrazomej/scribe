"""Tests for ScribeDEResults dataclass and compare() factory.

Validates the structured results class and its methods for gene-level
analysis, gene-set testing, error control, and formatting.
"""

import pytest
import jax.numpy as jnp
from jax import random

from scribe.de import compare, ScribeDEResults


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def sample_models():
    """Create two sample (D-1)-dimensional ALR model dicts."""
    D_alr = 50
    k = 5

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }
    model_B = {
        "loc": random.normal(random.PRNGKey(456), (D_alr,)) * 0.3,
        "cov_factor": random.normal(random.PRNGKey(789), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }
    return model_A, model_B


@pytest.fixture
def de_results(sample_models):
    """Create a ScribeDEResults via compare()."""
    model_A, model_B = sample_models
    return compare(model_A, model_B, label_A="WT", label_B="KO")


# --------------------------------------------------------------------------
# Test compare() factory
# --------------------------------------------------------------------------


def test_compare_returns_correct_type(sample_models):
    """compare() should return a ScribeDEResults."""
    model_A, model_B = sample_models
    de = compare(model_A, model_B)
    assert isinstance(de, ScribeDEResults)


def test_compare_dimensions(de_results):
    """Dimensions should be consistent."""
    assert de_results.D_alr == 50
    assert de_results.D == 51


def test_compare_labels(de_results):
    """Labels should be stored correctly."""
    assert de_results.label_A == "WT"
    assert de_results.label_B == "KO"


def test_compare_auto_gene_names(sample_models):
    """Gene names should be auto-generated when not provided."""
    model_A, model_B = sample_models
    de = compare(model_A, model_B)
    assert len(de.gene_names) == de.D
    assert de.gene_names[0] == "gene_0"


def test_compare_custom_gene_names(sample_models):
    """Custom gene names should be stored."""
    model_A, model_B = sample_models
    D = model_A["loc"].shape[0] + 1
    names = [f"GENE{i}" for i in range(D)]
    de = compare(model_A, model_B, gene_names=names)
    assert de.gene_names == names


def test_compare_dimension_mismatch_raises():
    """compare() should raise if dimensions don't match."""
    model_A = {
        "loc": jnp.zeros(10),
        "cov_factor": jnp.ones((10, 2)),
        "cov_diag": jnp.ones(10),
    }
    model_B = {
        "loc": jnp.zeros(20),
        "cov_factor": jnp.ones((20, 2)),
        "cov_diag": jnp.ones(20),
    }
    with pytest.raises(ValueError, match="dimensions do not match"):
        compare(model_A, model_B)


def test_compare_wrong_length_gene_names_raises():
    """compare() should raise if gene_names has wrong length."""
    D_alr = 10
    model = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": jnp.ones((D_alr, 2)),
        "cov_diag": jnp.ones(D_alr),
    }
    # D = D_alr + 1 = 11, but we pass 5 names
    with pytest.raises(ValueError, match="gene_names has length 5"):
        compare(model, model, gene_names=[f"g{i}" for i in range(5)])


def test_compare_correct_length_gene_names_passes():
    """compare() should accept gene_names with correct length."""
    D_alr = 10
    model = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": jnp.ones((D_alr, 2)),
        "cov_diag": jnp.ones(D_alr),
    }
    D = D_alr + 1
    names = [f"g{i}" for i in range(D)]
    de = compare(model, model, gene_names=names)
    assert de.gene_names == names


def test_compare_with_embedded_alr():
    """compare() should handle D-dimensional embedded ALR dicts."""
    D_alr = 10
    k = 2

    # Create embedded ALR dicts (D-dimensional)
    mu_alr = jnp.zeros(D_alr)
    W_alr = random.normal(random.PRNGKey(42), (D_alr, k)) * 0.1
    d_alr = jnp.ones(D_alr) * 0.5

    model_A = {
        "loc": jnp.concatenate([mu_alr, jnp.array([0.0])]),
        "cov_factor": jnp.concatenate(
            [W_alr, jnp.zeros((1, k))], axis=0
        ),
        "cov_diag": jnp.concatenate([d_alr, jnp.array([0.0])]),
    }
    model_B = {
        "loc": jnp.concatenate(
            [jnp.ones(D_alr) * 0.2, jnp.array([0.0])]
        ),
        "cov_factor": jnp.concatenate(
            [W_alr * 1.1, jnp.zeros((1, k))], axis=0
        ),
        "cov_diag": jnp.concatenate([d_alr, jnp.array([0.0])]),
    }

    de = compare(model_A, model_B)

    # After extraction, D_alr should be 10 (stripped from 11)
    assert de.D_alr == D_alr
    assert de.D == D_alr + 1


# --------------------------------------------------------------------------
# Test gene_level()
# --------------------------------------------------------------------------


def test_gene_level_returns_dict(de_results):
    """gene_level() should return a dict with expected keys."""
    results = de_results.gene_level(tau=0.0)
    expected_keys = {
        "delta_mean", "delta_sd", "prob_positive",
        "prob_effect", "lfsr", "lfsr_tau", "gene_names",
    }
    assert set(results.keys()) == expected_keys


def test_gene_level_shapes(de_results):
    """gene_level() results should have shape (D,)."""
    results = de_results.gene_level(tau=0.0)
    D = de_results.D
    assert results["delta_mean"].shape == (D,)
    assert results["delta_sd"].shape == (D,)
    assert results["lfsr"].shape == (D,)


def test_gene_level_caching(de_results):
    """gene_level() should cache results."""
    r1 = de_results.gene_level(tau=0.0)
    # Accessing _gene_results directly should return the cached value
    assert de_results._gene_results is not None
    assert de_results._gene_results is r1


# --------------------------------------------------------------------------
# Test call_genes()
# --------------------------------------------------------------------------


def test_call_genes_returns_bool(de_results):
    """call_genes() should return a boolean array."""
    is_de = de_results.call_genes(tau=0.0)
    assert is_de.dtype == jnp.bool_
    assert is_de.shape == (de_results.D,)


# --------------------------------------------------------------------------
# Test gene-set methods
# --------------------------------------------------------------------------


def test_test_gene_set_via_results(de_results):
    """test_gene_set() should return dict with expected keys."""
    gene_set = jnp.array([0, 1, 2, 3, 4])
    result = de_results.test_gene_set(gene_set, tau=0.0)
    assert "delta_mean" in result
    assert "lfsr" in result
    assert 0 <= result["lfsr"] <= 0.5


def test_test_contrast_via_results(de_results):
    """test_contrast() should return dict with expected keys."""
    D = de_results.D
    contrast = jnp.zeros(D)
    contrast = contrast.at[0].set(1.0)
    contrast = contrast.at[-1].set(-1.0)

    result = de_results.test_contrast(contrast, tau=0.0)
    assert "delta_mean" in result
    assert "lfsr" in result


# --------------------------------------------------------------------------
# Test error-control methods
# --------------------------------------------------------------------------


def test_compute_pefp_via_results(de_results):
    """compute_pefp() should return a float."""
    de_results.gene_level(tau=0.0)
    pefp = de_results.compute_pefp(threshold=0.05)
    assert isinstance(pefp, float)
    assert 0 <= pefp <= 1


def test_find_threshold_via_results(de_results):
    """find_threshold() should return a non-negative float."""
    de_results.gene_level(tau=0.0)
    threshold = de_results.find_threshold(target_pefp=0.05)
    assert isinstance(threshold, float)
    assert threshold >= 0


# --------------------------------------------------------------------------
# Test summary / repr
# --------------------------------------------------------------------------


def test_summary_returns_string(de_results):
    """summary() should return a non-empty string."""
    s = de_results.summary(tau=0.0)
    assert isinstance(s, str)
    assert len(s) > 0


def test_repr(de_results):
    """__repr__ should contain key information."""
    r = repr(de_results)
    assert "ScribeDEResults" in r
    assert "D=" in r
    assert "WT" in r
    assert "KO" in r


# --------------------------------------------------------------------------
# Fix 1: stale tau cache invalidation
# --------------------------------------------------------------------------


def test_call_genes_different_tau_invalidates_cache(de_results):
    """call_genes with different tau values should produce different results.

    Regression test for the stale-cache bug: previously, call_genes(tau=X)
    would silently re-use results computed with a different tau.
    """
    # Compute with tau=0
    is_de_0 = de_results.call_genes(tau=0.0)
    assert de_results._cached_tau == 0.0

    # Compute with a large tau -- should recompute, not reuse cache
    is_de_large = de_results.call_genes(tau=5.0)
    assert de_results._cached_tau == 5.0

    # A large tau should yield fewer (or equal) DE genes
    assert int(is_de_large.sum()) <= int(is_de_0.sum())


def test_gene_level_caching_with_tau(de_results):
    """gene_level() should update cached_tau and recompute on change."""
    r1 = de_results.gene_level(tau=0.0)
    assert de_results._cached_tau == 0.0
    assert de_results._gene_results is r1

    r2 = de_results.gene_level(tau=1.0)
    assert de_results._cached_tau == 1.0
    assert de_results._gene_results is r2
    assert r2 is not r1  # should be a new dict


def test_compute_pefp_tau_propagation(de_results):
    """compute_pefp(tau=...) should recompute when tau differs."""
    # First call with tau=0
    pefp_0 = de_results.compute_pefp(threshold=0.05, tau=0.0)
    assert de_results._cached_tau == 0.0

    # Second call with a different tau
    pefp_1 = de_results.compute_pefp(threshold=0.05, tau=2.0)
    assert de_results._cached_tau == 2.0

    # Results may or may not differ, but the cache must have been refreshed
    assert isinstance(pefp_0, float)
    assert isinstance(pefp_1, float)


def test_find_threshold_tau_propagation(de_results):
    """find_threshold(tau=...) should recompute when tau differs."""
    t0 = de_results.find_threshold(target_pefp=0.05, tau=0.0)
    assert de_results._cached_tau == 0.0

    t1 = de_results.find_threshold(target_pefp=0.05, tau=2.0)
    assert de_results._cached_tau == 2.0

    assert isinstance(t0, float)
    assert isinstance(t1, float)


def test_summary_tau_propagation(de_results):
    """summary(tau=...) should recompute when tau differs."""
    s0 = de_results.summary(tau=0.0)
    assert de_results._cached_tau == 0.0

    s1 = de_results.summary(tau=2.0)
    assert de_results._cached_tau == 2.0

    # Both should be valid non-empty strings
    assert isinstance(s0, str) and len(s0) > 0
    assert isinstance(s1, str) and len(s1) > 0


# --------------------------------------------------------------------------
# Fix 2 (exposed): lfsr_tau via results class
# --------------------------------------------------------------------------


def test_compute_pefp_use_lfsr_tau(de_results):
    """compute_pefp(use_lfsr_tau=True) should use lfsr_tau from results."""
    # Standard lfsr-based PEFP
    pefp_std = de_results.compute_pefp(threshold=0.05, tau=0.1)

    # lfsr_tau-based PEFP
    pefp_tau = de_results.compute_pefp(
        threshold=0.5, tau=0.1, use_lfsr_tau=True
    )

    assert isinstance(pefp_std, float)
    assert isinstance(pefp_tau, float)
    assert 0 <= pefp_std <= 1
    assert 0 <= pefp_tau <= 1


def test_find_threshold_use_lfsr_tau(de_results):
    """find_threshold(use_lfsr_tau=True) should use lfsr_tau."""
    t_std = de_results.find_threshold(target_pefp=0.05, tau=0.0)
    t_tau = de_results.find_threshold(
        target_pefp=0.05, tau=0.0, use_lfsr_tau=True
    )

    # At tau=0 they should be identical (lfsr_tau == lfsr when tau=0)
    assert abs(t_std - t_tau) < 1e-6
