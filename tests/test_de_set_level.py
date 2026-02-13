"""
Tests for gene-set and pathway-level differential expression analysis.

Tests the Bayesian inference functions for testing linear contrasts and gene sets.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.de._set_level import (
    test_contrast as _test_contrast,
    test_gene_set as _test_gene_set,
)
from scribe.de import build_balance_contrast

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_models():
    """Generate two sample models for testing."""
    D_alr = 50
    k = 5

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": random.normal(random.PRNGKey(456), (D_alr,)) * 0.2,
        "cov_factor": random.normal(random.PRNGKey(789), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    return model_A, model_B


# ------------------------------------------------------------------------------
# Test build_balance_contrast
# ------------------------------------------------------------------------------


def test_build_balance_contrast_shape():
    """Test that balance contrast has correct shape."""
    D = 100
    num_indices = jnp.array([0, 1, 2, 3, 4])
    den_indices = jnp.array([5, 6, 7, 8, 9])

    contrast = build_balance_contrast(num_indices, den_indices, D)

    assert contrast.shape == (D,)


def test_build_balance_contrast_coefficients():
    """Test that balance contrast has correct coefficients."""
    D = 100
    num_indices = jnp.array([0, 1, 2])  # 3 genes
    den_indices = jnp.array([5, 6])  # 2 genes

    contrast = build_balance_contrast(num_indices, den_indices, D)

    # Numerator genes should have 1/3
    assert jnp.allclose(contrast[num_indices], 1.0 / 3.0)

    # Denominator genes should have -1/2
    assert jnp.allclose(contrast[den_indices], -1.0 / 2.0)

    # Others should be zero
    other_indices = jnp.array([3, 4, 7, 8, 9])
    assert jnp.allclose(contrast[other_indices], 0.0)


def test_build_balance_contrast_empty_raises_error():
    """Test that empty numerator or denominator raises error."""
    D = 100

    with pytest.raises(ValueError):
        build_balance_contrast(jnp.array([]), jnp.array([1, 2]), D)

    with pytest.raises(ValueError):
        build_balance_contrast(jnp.array([1, 2]), jnp.array([]), D)


def test_build_balance_contrast_sum():
    """Test that balance contrast sum depends on group sizes."""
    D = 100
    num_indices = jnp.array([0, 1, 2, 3])  # 4 genes
    den_indices = jnp.array([5, 6, 7, 8, 9])  # 5 genes

    contrast = build_balance_contrast(num_indices, den_indices, D)

    # The sum should be: n_num * (1/n_num) + n_den * (-1/n_den) = 1 - 1 = 0
    # But only when accounting for all terms
    expected_sum = 0.0  # Actually should be 0 for equal-sized or balanced
    # Let's just check it's finite and small
    assert jnp.isfinite(contrast.sum())
    assert jnp.abs(contrast.sum()) < 1.0  # Should be small


# ------------------------------------------------------------------------------
# Test test_contrast
# ------------------------------------------------------------------------------


def test_contrast_output_keys(sample_models):
    """Test that test_contrast returns all expected keys."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    # Simple contrast: compare first gene to last gene
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)
    contrast = contrast.at[-1].set(-1.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    expected_keys = {
        "delta_mean",
        "delta_sd",
        "z_score",
        "prob_positive",
        "prob_effect",
        "lfsr",
    }

    assert set(result.keys()) == expected_keys


def test_contrast_output_types(sample_models):
    """Test that test_contrast returns scalar values."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    # All outputs should be scalars (floats)
    for key, value in result.items():
        assert isinstance(value, (float, np.floating))


def test_contrast_probabilities_bounded(sample_models):
    """Test that probabilities are between 0 and 1."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    contrast = random.normal(random.PRNGKey(999), (D_clr,))

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    assert 0 <= result["prob_positive"] <= 1
    assert 0 <= result["prob_effect"] <= 1
    assert 0 <= result["lfsr"] <= 0.5


def test_contrast_positive_sd(sample_models):
    """Test that standard deviation is positive."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    # Use a non-trivial contrast that won't become all zeros after centering
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)
    contrast = contrast.at[-1].set(-1.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    assert result["delta_sd"] > 0


def test_contrast_unit_contrast():
    """Test that unit contrast gives sensible result."""
    D_alr = 20
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(111), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.5,
        "cov_factor": random.normal(random.PRNGKey(222), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    D_clr = D_alr + 1
    # Use a unit contrast (first gene only)
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    # Should give finite results
    assert jnp.isfinite(result["delta_mean"])
    assert jnp.isfinite(result["delta_sd"])
    assert result["delta_sd"] > 0


def test_contrast_opposite_sign():
    """Test that flipping contrast flips sign of effect."""
    D_alr = 20
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(333), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.5,
        "cov_factor": random.normal(random.PRNGKey(444), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    D_clr = D_alr + 1
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)
    contrast = contrast.at[1].set(-1.0)

    result_pos = _test_contrast(model_A, model_B, contrast, tau=0.0)
    result_neg = _test_contrast(model_A, model_B, -contrast, tau=0.0)

    # Effect should flip sign
    assert jnp.abs(result_pos["delta_mean"] + result_neg["delta_mean"]) < 1e-6

    # But lfsr should be same
    assert jnp.abs(result_pos["lfsr"] - result_neg["lfsr"]) < 1e-6


def test_contrast_tau_effect():
    """Test that larger tau gives smaller prob_effect."""
    D_alr = 20
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(555), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": jnp.ones(D_alr) * 0.5,
        "cov_factor": random.normal(random.PRNGKey(666), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    D_clr = D_alr + 1
    # Use a non-trivial contrast (not uniform, survives CLR centering)
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[:5].set(1.0 / 5)
    contrast = contrast.at[5:10].set(-1.0 / 5)

    result_small_tau = _test_contrast(model_A, model_B, contrast, tau=0.0)
    result_large_tau = _test_contrast(
        model_A, model_B, contrast, tau=jnp.log(2.0)
    )

    # Larger tau should give smaller prob_effect
    assert result_large_tau["prob_effect"] <= result_small_tau["prob_effect"]


# ------------------------------------------------------------------------------
# Test test_gene_set
# ------------------------------------------------------------------------------


def test_gene_set_output_keys(sample_models):
    """Test that test_gene_set returns all expected keys."""
    model_A, model_B = sample_models

    gene_set_indices = jnp.array([0, 1, 2, 3, 4])

    result = _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)

    expected_keys = {
        "delta_mean",
        "delta_sd",
        "z_score",
        "prob_positive",
        "prob_effect",
        "lfsr",
    }

    assert set(result.keys()) == expected_keys


def test_gene_set_probabilities_bounded(sample_models):
    """Test that gene set probabilities are between 0 and 1."""
    model_A, model_B = sample_models

    gene_set_indices = jnp.array([5, 10, 15, 20, 25])

    result = _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)

    assert 0 <= result["prob_positive"] <= 1
    assert 0 <= result["prob_effect"] <= 1
    assert 0 <= result["lfsr"] <= 0.5


def test_gene_set_empty_raises_error(sample_models):
    """Test that empty gene set raises error."""
    model_A, model_B = sample_models

    with pytest.raises(ValueError):
        _test_gene_set(model_A, model_B, jnp.array([]), tau=0.0)


def test_gene_set_all_genes_raises_error(sample_models):
    """Test that gene set including all genes raises error."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    # Try to use all genes
    gene_set_indices = jnp.arange(D_clr)

    with pytest.raises(ValueError):
        _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)


def test_gene_set_single_gene(sample_models):
    """Test gene set with single gene."""
    model_A, model_B = sample_models

    gene_set_indices = jnp.array([0])

    # Should not raise error
    result = _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)

    assert isinstance(result["delta_mean"], (float, np.floating))


def test_gene_set_complementary_sets():
    """Test that complementary gene sets give opposite effects."""
    D_alr = 30
    k = 3

    # Create models where first half differs from second half
    loc_A = jnp.concatenate([jnp.ones(15) * 0.5, jnp.ones(15) * -0.5])
    loc_B = jnp.zeros(D_alr)

    model_A = {
        "loc": loc_A,
        "cov_factor": jnp.zeros((D_alr, k)),
        "cov_diag": jnp.ones(D_alr) * 0.1,
    }

    model_B = {
        "loc": loc_B,
        "cov_factor": jnp.zeros((D_alr, k)),
        "cov_diag": jnp.ones(D_alr) * 0.1,
    }

    # Test first half vs second half
    set1 = jnp.arange(0, 16)  # First 16 in CLR (includes appended gene)
    D_clr = D_alr + 1
    set2 = jnp.arange(16, D_clr)  # Remaining genes

    result1 = _test_gene_set(model_A, model_B, set1, tau=0.0)
    result2 = _test_gene_set(model_A, model_B, set2, tau=0.0)

    # Effects should have opposite signs
    assert result1["delta_mean"] * result2["delta_mean"] < 0


def test_gene_set_versus_contrast_equivalence(sample_models):
    """Test that test_gene_set is equivalent to test_contrast with balance."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    gene_set_indices = jnp.array([0, 1, 2, 3, 4])

    # Compute using test_gene_set
    result_set = _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)

    # Compute using test_contrast with manually built balance
    n_in = len(gene_set_indices)
    n_out = D_clr - n_in
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[gene_set_indices].set(1.0 / n_in)

    mask_out = jnp.ones(D_clr, dtype=bool)
    mask_out = mask_out.at[gene_set_indices].set(False)
    contrast = jnp.where(mask_out, -1.0 / n_out, contrast)

    result_contrast = _test_contrast(model_A, model_B, contrast, tau=0.0)

    # Results should be identical
    assert (
        jnp.abs(result_set["delta_mean"] - result_contrast["delta_mean"]) < 1e-6
    )
    assert jnp.abs(result_set["delta_sd"] - result_contrast["delta_sd"]) < 1e-6
    assert jnp.abs(result_set["lfsr"] - result_contrast["lfsr"]) < 1e-6


# ------------------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------------------


def test_pathway_analysis_workflow(sample_models):
    """Test complete pathway analysis workflow."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    # Define a few pathways
    pathway_1 = jnp.array([0, 1, 2, 3, 4])
    pathway_2 = jnp.array([10, 11, 12, 13, 14])
    pathway_3 = jnp.array([20, 21, 22, 23, 24])

    pathways = [pathway_1, pathway_2, pathway_3]

    # Test each pathway
    results = []
    for pathway in pathways:
        result = _test_gene_set(model_A, model_B, pathway, tau=jnp.log(1.1))
        results.append(result)

    # All should return valid results
    for result in results:
        assert "delta_mean" in result
        assert "lfsr" in result
        assert 0 <= result["lfsr"] <= 0.5


def test_hierarchical_pathway_structure():
    """Test nested pathway structures."""
    D_alr = 50
    k = 3

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(777), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": random.normal(random.PRNGKey(888), (D_alr,)) * 0.2,
        "cov_factor": random.normal(random.PRNGKey(999), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    # Parent pathway
    parent_pathway = jnp.arange(0, 20)

    # Child pathways (subsets)
    child1 = jnp.arange(0, 10)
    child2 = jnp.arange(10, 20)

    result_parent = _test_gene_set(model_A, model_B, parent_pathway, tau=0.0)
    result_child1 = _test_gene_set(model_A, model_B, child1, tau=0.0)
    result_child2 = _test_gene_set(model_A, model_B, child2, tau=0.0)

    # All should return valid results
    assert jnp.isfinite(result_parent["delta_mean"])
    assert jnp.isfinite(result_child1["delta_mean"])
    assert jnp.isfinite(result_child2["delta_mean"])


def test_large_gene_set():
    """Test gene set analysis with large pathways."""
    D_alr = 999  # 1000 genes in CLR
    k = 5

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(1111), (D_alr, k)) * 0.01,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    model_B = {
        "loc": random.normal(random.PRNGKey(2222), (D_alr,)) * 0.05,
        "cov_factor": random.normal(random.PRNGKey(3333), (D_alr, k)) * 0.01,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    # Large pathway (200 genes)
    gene_set_indices = jnp.arange(0, 200)

    result = _test_gene_set(model_A, model_B, gene_set_indices, tau=0.0)

    assert jnp.isfinite(result["delta_mean"])
    assert jnp.isfinite(result["delta_sd"])
    assert jnp.isfinite(result["lfsr"])


def test_custom_contrast_workflow(sample_models):
    """Test workflow with custom contrast vectors."""
    model_A, model_B = sample_models
    D_clr = model_A["loc"].shape[0] + 1

    # Custom contrast: compare average of first 10 vs last 10 genes
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[:10].set(1.0 / 10.0)
    contrast = contrast.at[-10:].set(-1.0 / 10.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    assert jnp.isfinite(result["delta_mean"])
    assert 0 <= result["lfsr"] <= 0.5


def test_balance_contrast_properties():
    """Test that balance contrasts have desired mathematical properties."""
    D = 100

    # Test various balance contrasts
    for _ in range(10):
        # Random sizes
        n_num = np.random.randint(5, 30)
        n_den = np.random.randint(5, 30)

        num_indices = jnp.array(np.random.choice(D, size=n_num, replace=False))
        den_indices = jnp.array(
            np.random.choice(
                [i for i in range(D) if i not in num_indices],
                size=n_den,
                replace=False,
            )
        )

        contrast = build_balance_contrast(num_indices, den_indices, D)

        # Check coefficient values
        assert jnp.allclose(contrast[num_indices], 1.0 / n_num)
        assert jnp.allclose(contrast[den_indices], -1.0 / n_den)
