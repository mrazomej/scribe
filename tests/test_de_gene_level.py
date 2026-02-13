"""
Tests for gene-level differential expression analysis.

Tests the Bayesian differential expression functions for gene-level inference.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.de import (
    differential_expression,
    call_de_genes,
)

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_models():
    """Generate two sample models for differential expression."""
    D_alr = 50
    k = 5

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.2,  # Mean shift
        "cov_factor": random.normal(random.PRNGKey(456), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    return model_A, model_B


@pytest.fixture
def sample_de_results():
    """Generate sample DE results for testing call_de_genes."""
    n_genes = 100

    # Simulate some DE genes with low lfsr and high prob_effect
    delta_mean = random.normal(random.PRNGKey(789), (n_genes,)) * 0.5
    delta_sd = jnp.ones(n_genes) * 0.1

    # Compute posterior probabilities
    z_scores = delta_mean / delta_sd
    prob_positive = 1.0 / (1.0 + jnp.exp(-z_scores))  # Approximation

    # Simple prob_effect calculation
    prob_effect = jnp.abs(z_scores) / (1.0 + jnp.abs(z_scores))  # Simplified

    # lfsr
    lfsr = jnp.minimum(prob_positive, 1 - prob_positive)

    return {
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
    }


# ------------------------------------------------------------------------------
# Test differential_expression function
# ------------------------------------------------------------------------------


def test_differential_expression_output_keys(sample_models):
    """Test that differential_expression returns all expected keys."""
    model_A, model_B = sample_models

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    expected_keys = {
        "delta_mean",
        "delta_sd",
        "prob_positive",
        "prob_effect",
        "lfsr",
        "gene_names",
    }

    assert set(results.keys()) == expected_keys


def test_differential_expression_output_shapes(sample_models):
    """Test that differential_expression outputs have correct shapes."""
    model_A, model_B = sample_models
    D_alr = model_A["loc"].shape[0]
    D_clr = D_alr + 1

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    assert results["delta_mean"].shape == (D_clr,)
    assert results["delta_sd"].shape == (D_clr,)
    assert results["prob_positive"].shape == (D_clr,)
    assert results["prob_effect"].shape == (D_clr,)
    assert results["lfsr"].shape == (D_clr,)
    assert len(results["gene_names"]) == D_clr


def test_differential_expression_positive_sd(sample_models):
    """Test that standard deviations are positive."""
    model_A, model_B = sample_models

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    assert jnp.all(results["delta_sd"] > 0)


def test_differential_expression_probabilities_bounded(sample_models):
    """Test that probabilities are between 0 and 1."""
    model_A, model_B = sample_models

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    assert jnp.all(results["prob_positive"] >= 0)
    assert jnp.all(results["prob_positive"] <= 1)

    assert jnp.all(results["prob_effect"] >= 0)
    assert jnp.all(results["prob_effect"] <= 1)

    assert jnp.all(results["lfsr"] >= 0)
    assert jnp.all(results["lfsr"] <= 0.5)  # lfsr is min(p, 1-p)


def test_differential_expression_lfsr_symmetric(sample_models):
    """Test that lfsr is symmetric (doesn't depend on direction)."""
    model_A, model_B = sample_models

    results_AB = differential_expression(
        model_A,
        model_B,
        tau=0.0,
        coordinate="clr",
    )

    results_BA = differential_expression(
        model_B,
        model_A,
        tau=0.0,
        coordinate="clr",
    )

    # lfsr should be the same regardless of order
    assert jnp.allclose(results_AB["lfsr"], results_BA["lfsr"], atol=1e-6)


def test_differential_expression_with_gene_names(sample_models):
    """Test that custom gene names are used."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    gene_names = [f"GENE{i}" for i in range(D_clr)]

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
        gene_names=gene_names,
    )

    assert results["gene_names"] == gene_names


def test_differential_expression_no_difference():
    """Test DE when models are identical."""
    D_alr = 20
    k = 3

    model = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(999), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    results = differential_expression(
        model,
        model,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    # Mean difference should be zero
    assert jnp.allclose(results["delta_mean"], 0.0, atol=1e-6)

    # prob_positive should be ~0.5 (no directional bias)
    assert jnp.allclose(results["prob_positive"], 0.5, atol=1e-2)

    # lfsr should be ~0.5 (maximum uncertainty)
    assert jnp.allclose(results["lfsr"], 0.5, atol=1e-2)


def test_differential_expression_large_difference():
    """Test DE when models differ substantially."""
    D_alr = 20
    k = 3

    W = random.normal(random.PRNGKey(888), (D_alr, k)) * 0.1
    d = jnp.ones(D_alr) * 0.5

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": W,
        "cov_diag": d,
    }

    # Use a non-uniform shift so CLR centering does not cancel it out
    model_B = {
        "loc": random.normal(random.PRNGKey(777), (D_alr,)) * 5.0,
        "cov_factor": W,
        "cov_diag": d,
    }

    results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    # Most genes should have very small lfsr (high confidence)
    assert jnp.mean(results["lfsr"] < 0.05) > 0.7  # At least 70% significant

    # Most genes should have extreme prob_positive
    assert (
        jnp.mean(
            (results["prob_positive"] < 0.05)
            | (results["prob_positive"] > 0.95)
        )
        > 0.7
    )


def test_differential_expression_tau_effect():
    """Test that tau threshold affects prob_effect."""
    D_alr = 20
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(777), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.5,
        "cov_factor": random.normal(random.PRNGKey(888), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    results_small_tau = differential_expression(model_A, model_B, tau=0.0)
    results_large_tau = differential_expression(
        model_A, model_B, tau=jnp.log(2.0)
    )

    # Larger tau should give smaller prob_effect
    assert jnp.all(
        results_large_tau["prob_effect"] <= results_small_tau["prob_effect"]
    )


def test_differential_expression_invalid_coordinate():
    """Test that invalid coordinate raises error."""
    D_alr = 20
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(111), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    with pytest.raises(NotImplementedError, match="Coordinate|coordinate"):
        differential_expression(model_A, model_A, tau=0.0, coordinate="invalid")


# ------------------------------------------------------------------------------
# Test call_de_genes function
# ------------------------------------------------------------------------------


def test_call_de_genes_output_type(sample_de_results):
    """Test that call_de_genes returns boolean array."""
    is_de = call_de_genes(sample_de_results, lfsr_threshold=0.05)

    assert is_de.dtype == jnp.bool_


def test_call_de_genes_output_shape(sample_de_results):
    """Test that call_de_genes returns correct shape."""
    is_de = call_de_genes(sample_de_results, lfsr_threshold=0.05)

    n_genes = len(sample_de_results["delta_mean"])
    assert is_de.shape == (n_genes,)


def test_call_de_genes_strict_threshold():
    """Test that stricter thresholds give fewer DE genes."""
    # Create results with known properties
    n_genes = 100

    results = {
        "delta_mean": random.normal(random.PRNGKey(555), (n_genes,)),
        "delta_sd": jnp.ones(n_genes) * 0.1,
        "prob_positive": random.uniform(random.PRNGKey(666), (n_genes,)),
        "prob_effect": random.uniform(random.PRNGKey(777), (n_genes,)),
        "lfsr": random.uniform(random.PRNGKey(888), (n_genes,)) * 0.5,
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
    }

    is_de_loose = call_de_genes(
        results, lfsr_threshold=0.1, prob_effect_threshold=0.8
    )
    is_de_strict = call_de_genes(
        results, lfsr_threshold=0.05, prob_effect_threshold=0.95
    )

    # Stricter should give fewer or equal DE genes
    assert is_de_strict.sum() <= is_de_loose.sum()


def test_call_de_genes_both_conditions():
    """Test that both lfsr and prob_effect conditions must be met."""
    n_genes = 100

    # Create data where some genes meet only one condition
    lfsr = jnp.concatenate(
        [
            jnp.ones(25) * 0.01,  # Low lfsr
            jnp.ones(25) * 0.1,  # High lfsr
            jnp.ones(25) * 0.01,  # Low lfsr
            jnp.ones(25) * 0.1,  # High lfsr
        ]
    )

    prob_effect = jnp.concatenate(
        [
            jnp.ones(25) * 0.99,  # High prob_effect
            jnp.ones(25) * 0.99,  # High prob_effect
            jnp.ones(25) * 0.8,  # Low prob_effect
            jnp.ones(25) * 0.8,  # Low prob_effect
        ]
    )

    results = {
        "delta_mean": jnp.ones(n_genes),
        "delta_sd": jnp.ones(n_genes),
        "prob_positive": jnp.ones(n_genes) * 0.5,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
    }

    is_de = call_de_genes(
        results, lfsr_threshold=0.05, prob_effect_threshold=0.95
    )

    # Only first 25 meet both conditions
    assert is_de.sum() == 25
    assert jnp.all(is_de[:25])
    assert jnp.all(~is_de[25:])


def test_call_de_genes_no_genes_called():
    """Test when no genes meet thresholds."""
    n_genes = 50

    results = {
        "delta_mean": jnp.zeros(n_genes),
        "delta_sd": jnp.ones(n_genes),
        "prob_positive": jnp.ones(n_genes) * 0.5,
        "prob_effect": jnp.zeros(n_genes),  # No genes have effects
        "lfsr": jnp.ones(n_genes) * 0.5,  # Maximum uncertainty
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
    }

    is_de = call_de_genes(
        results, lfsr_threshold=0.05, prob_effect_threshold=0.95
    )

    assert is_de.sum() == 0


def test_call_de_genes_all_genes_called():
    """Test when all genes meet thresholds."""
    n_genes = 50

    results = {
        "delta_mean": jnp.ones(n_genes) * 10.0,  # Large effects
        "delta_sd": jnp.ones(n_genes) * 0.1,
        "prob_positive": jnp.ones(n_genes),  # All positive
        "prob_effect": jnp.ones(n_genes),  # All have effects
        "lfsr": jnp.zeros(n_genes),  # Zero uncertainty
        "gene_names": [f"gene_{i}" for i in range(n_genes)],
    }

    is_de = call_de_genes(
        results, lfsr_threshold=0.05, prob_effect_threshold=0.95
    )

    assert is_de.sum() == n_genes


# ------------------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------------------


def test_de_pipeline_integration(sample_models):
    """Test complete DE pipeline from models to called genes."""
    model_A, model_B = sample_models

    # Step 1: Compute DE
    de_results = differential_expression(
        model_A,
        model_B,
        tau=jnp.log(1.1),
        coordinate="clr",
    )

    # Step 2: Call DE genes
    is_de = call_de_genes(
        de_results,
        lfsr_threshold=0.05,
        prob_effect_threshold=0.95,
    )

    # Should return valid results
    assert isinstance(is_de, jnp.ndarray)
    assert is_de.dtype == jnp.bool_
    assert is_de.shape[0] == de_results["delta_mean"].shape[0]


def test_de_symmetric_comparison():
    """Test that swapping A and B gives symmetric results."""
    D_alr = 30
    k = 4

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.3,
        "cov_factor": random.normal(random.PRNGKey(456), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    results_AB = differential_expression(model_A, model_B, tau=0.0)
    results_BA = differential_expression(model_B, model_A, tau=0.0)

    # Mean differences should be negatives
    assert jnp.allclose(
        results_AB["delta_mean"], -results_BA["delta_mean"], atol=1e-5
    )

    # lfsr should be identical
    assert jnp.allclose(results_AB["lfsr"], results_BA["lfsr"], atol=1e-6)

    # Standard deviations should be identical
    assert jnp.allclose(
        results_AB["delta_sd"], results_BA["delta_sd"], atol=1e-6
    )


def test_de_with_different_ranks():
    """Test DE when models have different low-rank dimensions."""
    D_alr = 20

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(111), (D_alr, 3)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.3,
        "cov_factor": random.normal(random.PRNGKey(222), (D_alr, 5)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    # Should handle different ranks
    results = differential_expression(model_A, model_B, tau=jnp.log(1.1))

    assert results["delta_mean"].shape[0] == D_alr + 1


def test_de_large_scale():
    """Test DE scales to realistic single-cell dimensions."""
    D_alr = 999  # 1000 genes in CLR
    k = 10

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(333), (D_alr, k)) * 0.01,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": random.normal(random.PRNGKey(444), (D_alr,)) * 0.1,
        "cov_factor": random.normal(random.PRNGKey(555), (D_alr, k)) * 0.01,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    # Should compute without errors
    results = differential_expression(model_A, model_B, tau=jnp.log(1.1))
    is_de = call_de_genes(results, lfsr_threshold=0.05)

    assert results["delta_mean"].shape == (1000,)
    assert is_de.dtype == jnp.bool_
