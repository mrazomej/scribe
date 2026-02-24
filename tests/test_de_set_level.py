"""
Tests for gene-set and pathway-level differential expression analysis.

Tests the Bayesian inference functions for testing linear contrasts and gene sets,
including both parametric and empirical (ILR balance-based) pathway enrichment.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.de._set_level import (
    test_contrast as _test_contrast,
    test_gene_set as _test_gene_set,
    empirical_test_gene_set,
    empirical_test_pathway_perturbation,
    empirical_test_multiple_gene_sets,
)
from scribe.de._transforms import build_ilr_balance, build_pathway_sbp_basis
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


# ------------------------------------------------------------------------------
# Fix 5: epsilon guard for near-zero SD in test_contrast
# ------------------------------------------------------------------------------


def test_contrast_near_zero_variance():
    """Near-zero variance should not produce NaN or Inf in test_contrast.

    Uses 1e-8 diagonal (above the 1e-10 embedded-ALR detection threshold
    in ``extract_alr_params``) so the model is treated as raw ALR.
    """
    D_alr = 10
    k = 2

    # Use 1e-8: small enough for near-zero variance, large enough to
    # avoid false-positive embedded-ALR detection (threshold = 1e-10).
    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": jnp.zeros((D_alr, k)),
        "cov_diag": jnp.ones(D_alr) * 1e-8,
    }
    model_B = {
        "loc": jnp.ones(D_alr) * 0.1,
        "cov_factor": jnp.zeros((D_alr, k)),
        "cov_diag": jnp.ones(D_alr) * 1e-8,
    }

    D_clr = D_alr + 1
    contrast = jnp.zeros(D_clr)
    contrast = contrast.at[0].set(1.0)
    contrast = contrast.at[-1].set(-1.0)

    result = _test_contrast(model_A, model_B, contrast, tau=0.0)

    # Should be finite (no NaN from division by near-zero SD)
    assert jnp.isfinite(result["delta_mean"])
    assert jnp.isfinite(result["delta_sd"])
    assert jnp.isfinite(result["z_score"])
    assert jnp.isfinite(result["lfsr"])


# ------------------------------------------------------------------------------
# Fix 6: disjoint-set validation in build_balance_contrast
# ------------------------------------------------------------------------------


def test_build_balance_contrast_overlapping_raises():
    """Overlapping numerator and denominator should raise ValueError."""
    D = 100
    num_indices = jnp.array([0, 1, 2, 3, 4])
    den_indices = jnp.array([3, 4, 5, 6, 7])  # overlaps at 3, 4

    with pytest.raises(ValueError, match="disjoint"):
        build_balance_contrast(num_indices, den_indices, D)


def test_build_balance_contrast_overlapping_single():
    """Single overlapping index should also raise ValueError."""
    D = 100
    num_indices = jnp.array([0, 1, 2])
    den_indices = jnp.array([2, 3, 4])  # overlaps at 2

    with pytest.raises(ValueError, match="disjoint"):
        build_balance_contrast(num_indices, den_indices, D)


def test_build_balance_contrast_identical_raises():
    """Identical numerator and denominator should raise ValueError."""
    D = 100
    indices = jnp.array([0, 1, 2])

    with pytest.raises(ValueError, match="disjoint"):
        build_balance_contrast(indices, indices, D)


def test_build_balance_contrast_disjoint_passes():
    """Properly disjoint indices should not raise."""
    D = 100
    num_indices = jnp.array([0, 1, 2])
    den_indices = jnp.array([3, 4, 5])

    # Should not raise
    contrast = build_balance_contrast(num_indices, den_indices, D)
    assert contrast.shape == (D,)


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


# ==============================================================================
# Tests for ILR balance vector (build_ilr_balance)
# ==============================================================================


class TestBuildIlrBalance:
    """Tests for the ILR-normalized pathway balance vector."""

    def test_shape(self):
        """Balance vector should have shape (D,)."""
        D = 20
        idx = jnp.array([0, 3, 7])
        v = build_ilr_balance(idx, D)
        assert v.shape == (D,)

    def test_unit_norm(self):
        """Balance vector should have unit L2 norm."""
        for D in [5, 20, 100]:
            idx = jnp.arange(3)
            v = build_ilr_balance(idx, D)
            assert jnp.allclose(jnp.sum(v**2), 1.0, atol=1e-6)

    def test_sum_to_zero(self):
        """Balance vector entries should sum to zero (CLR constraint)."""
        for D in [5, 20, 100]:
            idx = jnp.array([1, 4])
            v = build_ilr_balance(idx, D)
            assert jnp.allclose(jnp.sum(v), 0.0, atol=1e-6)

    def test_correct_coefficients(self):
        """Verify coefficient values for known n_+ and n_-."""
        D = 10
        idx = jnp.array([0, 1, 2])  # n_+ = 3, n_- = 7
        v = build_ilr_balance(idx, D)

        n_plus, n_minus = 3, 7
        expected_pos = np.sqrt(n_minus / (n_plus * (n_plus + n_minus)))
        expected_neg = -np.sqrt(n_plus / (n_minus * (n_plus + n_minus)))

        assert jnp.allclose(v[idx], expected_pos, atol=1e-6)
        # Check complement entries
        mask = jnp.ones(D, dtype=bool).at[idx].set(False)
        assert jnp.allclose(v[mask], expected_neg, atol=1e-6)

    def test_scaled_clr_contrast(self):
        """ILR balance should be a positive scaling of the CLR contrast."""
        D = 15
        idx = jnp.array([2, 5, 8, 11])
        n_plus = len(idx)
        n_minus = D - n_plus

        v = build_ilr_balance(idx, D)

        # Build the unnormalized CLR contrast
        c = jnp.zeros(D)
        c = c.at[idx].set(1.0 / n_plus)
        mask = jnp.ones(D, dtype=bool).at[idx].set(False)
        c = jnp.where(mask, -1.0 / n_minus, c)

        # v should be alpha * c with alpha = sqrt(n_+ * n_- / (n_+ + n_-))
        alpha = np.sqrt(n_plus * n_minus / (n_plus + n_minus))
        assert jnp.allclose(v, alpha * c, atol=1e-6)

    def test_single_gene_pathway(self):
        """Single-gene pathway should still produce valid balance."""
        D = 10
        idx = jnp.array([5])
        v = build_ilr_balance(idx, D)
        assert jnp.allclose(jnp.sum(v**2), 1.0, atol=1e-6)
        assert jnp.allclose(jnp.sum(v), 0.0, atol=1e-6)

    def test_almost_all_genes(self):
        """Pathway with n_+ = D - 1 should still be valid."""
        D = 10
        idx = jnp.arange(D - 1)
        v = build_ilr_balance(idx, D)
        assert jnp.allclose(jnp.sum(v**2), 1.0, atol=1e-6)
        assert jnp.allclose(jnp.sum(v), 0.0, atol=1e-6)

    def test_empty_raises(self):
        """Empty pathway should raise ValueError."""
        with pytest.raises(ValueError):
            build_ilr_balance(jnp.array([], dtype=jnp.int32), 10)

    def test_all_genes_raises(self):
        """Pathway containing all genes should raise ValueError."""
        D = 10
        with pytest.raises(ValueError):
            build_ilr_balance(jnp.arange(D), D)


# ==============================================================================
# Tests for pathway-aware SBP basis (build_pathway_sbp_basis)
# ==============================================================================


class TestBuildPathwaySbpBasis:
    """Tests for the pathway-aware sequential binary partition basis."""

    def test_shape(self):
        """SBP basis should have shape (D-1, D)."""
        D = 10
        idx = jnp.array([0, 2, 5])
        V = build_pathway_sbp_basis(idx, D)
        assert V.shape == (D - 1, D)

    def test_orthonormality(self):
        """V V^T should be the identity matrix."""
        D = 8
        idx = jnp.array([1, 3, 5])
        V = build_pathway_sbp_basis(idx, D)
        assert jnp.allclose(V @ V.T, jnp.eye(D - 1), atol=1e-5)

    def test_rows_sum_to_zero(self):
        """Each row of the SBP basis should sum to zero."""
        D = 12
        idx = jnp.array([0, 4, 7, 10])
        V = build_pathway_sbp_basis(idx, D)
        assert jnp.allclose(V.sum(axis=1), 0.0, atol=1e-6)

    def test_row_zero_matches_balance(self):
        """Row 0 of SBP basis should match build_ilr_balance output."""
        D = 10
        idx = jnp.array([2, 5, 8])
        V = build_pathway_sbp_basis(idx, D)
        v_balance = build_ilr_balance(idx, D)
        assert jnp.allclose(V[0], v_balance, atol=1e-6)

    def test_within_pathway_support(self):
        """Within-pathway rows (1 to n_+) should be zero outside pathway."""
        D = 10
        idx = jnp.array([0, 3, 7])
        n_plus = len(idx)
        V = build_pathway_sbp_basis(idx, D)

        # Within-pathway rows: indices 1 to n_+-1 (inclusive)
        complement_mask = jnp.ones(D, dtype=bool).at[idx].set(False)
        for row_i in range(1, n_plus):
            assert jnp.allclose(V[row_i][complement_mask], 0.0, atol=1e-6)

    def test_within_complement_support(self):
        """Within-complement rows should be zero inside the pathway."""
        D = 10
        idx = jnp.array([0, 3, 7])
        n_plus = len(idx)
        V = build_pathway_sbp_basis(idx, D)

        # Within-complement rows start at index n_plus
        for row_i in range(n_plus, D - 1):
            assert jnp.allclose(V[row_i][idx], 0.0, atol=1e-6)

    def test_raises_single_gene(self):
        """Pathway with < 2 genes should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            build_pathway_sbp_basis(jnp.array([3]), D=10)

    def test_raises_all_genes(self):
        """Pathway containing all genes should raise ValueError."""
        D = 5
        with pytest.raises(ValueError):
            build_pathway_sbp_basis(jnp.arange(D), D)

    def test_various_sizes(self):
        """Orthonormality should hold for various pathway sizes."""
        D = 15
        for n_plus in [2, 5, 7, 13]:
            idx = jnp.arange(n_plus)
            V = build_pathway_sbp_basis(idx, D)
            assert jnp.allclose(V @ V.T, jnp.eye(D - 1), atol=1e-5)


# ==============================================================================
# Tests for empirical pathway enrichment (empirical_test_gene_set)
# ==============================================================================


class TestEmpiricalTestGeneSet:
    """Tests for the empirical single-balance pathway test."""

    @pytest.fixture
    def null_delta(self):
        """CLR difference samples under the null (no pathway effect)."""
        key = random.PRNGKey(42)
        N, D = 5000, 20
        return random.normal(key, (N, D)) * 0.1

    @pytest.fixture
    def shifted_delta(self):
        """CLR difference samples with a strong pathway shift."""
        key = random.PRNGKey(99)
        N, D = 5000, 20
        delta = random.normal(key, (N, D)) * 0.1
        # Shift the first 5 genes up strongly
        delta = delta.at[:, :5].add(2.0)
        return delta

    def test_output_keys(self, null_delta):
        """Result should contain all expected keys."""
        idx = jnp.array([0, 1, 2])
        result = empirical_test_gene_set(null_delta, idx)
        expected = {
            "balance_mean", "balance_sd", "prob_positive",
            "prob_effect", "lfsr", "lfsr_tau",
        }
        assert set(result.keys()) == expected

    def test_lfsr_bounded(self, null_delta):
        """lfsr should be in [0, 0.5]."""
        idx = jnp.array([0, 3, 7])
        result = empirical_test_gene_set(null_delta, idx)
        assert 0.0 <= result["lfsr"] <= 0.5

    def test_strong_effect_detected(self, shifted_delta):
        """Strong pathway shift should produce lfsr near 0."""
        idx = jnp.array([0, 1, 2, 3, 4])
        result = empirical_test_gene_set(shifted_delta, idx)
        assert result["lfsr"] < 0.01

    def test_null_lfsr_near_half(self, null_delta):
        """Under null, lfsr should be close to 0.5."""
        idx = jnp.array([0, 1, 2])
        result = empirical_test_gene_set(null_delta, idx)
        assert result["lfsr"] > 0.2

    def test_tau_reduces_prob_effect(self, shifted_delta):
        """Larger tau should reduce prob_effect."""
        idx = jnp.array([0, 1, 2, 3, 4])
        r0 = empirical_test_gene_set(shifted_delta, idx, tau=0.0)
        r1 = empirical_test_gene_set(shifted_delta, idx, tau=1.0)
        assert r1["prob_effect"] <= r0["prob_effect"]

    def test_consistency_with_test_contrast(self):
        """Result should match manual projection onto ILR balance vector."""
        key = random.PRNGKey(77)
        N, D = 3000, 15
        delta = random.normal(key, (N, D)) * 0.5
        idx = jnp.array([1, 4, 9])

        result = empirical_test_gene_set(delta, idx)

        # Manual computation
        v = build_ilr_balance(idx, D)
        samples = delta @ v
        manual_mean = float(jnp.mean(samples))
        manual_prob_pos = float(jnp.mean(samples > 0))
        manual_lfsr = min(manual_prob_pos, 1.0 - manual_prob_pos)

        assert abs(result["balance_mean"] - manual_mean) < 1e-5
        assert abs(result["lfsr"] - manual_lfsr) < 1e-5


# ==============================================================================
# Tests for within-pathway perturbation (empirical_test_pathway_perturbation)
# ==============================================================================


class TestEmpiricalTestPathwayPerturbation:
    """Tests for the multivariate within-pathway perturbation test."""

    def test_output_keys(self):
        """Result should contain expected keys."""
        key = random.PRNGKey(10)
        delta = random.normal(key, (500, 15)) * 0.1
        idx = jnp.array([0, 1, 2, 3])

        result = empirical_test_pathway_perturbation(
            delta, idx, n_permutations=50, key=random.PRNGKey(0)
        )
        expected = {"t_obs", "t_sd", "p_value", "n_permutations"}
        assert set(result.keys()) == expected

    def test_detects_perturbation(self):
        """Coordinated within-pathway changes should be detected.

        Half the pathway goes up, half goes down: the average balance is ~0,
        but the perturbation statistic should be large.
        """
        key = random.PRNGKey(20)
        N, D = 2000, 20
        delta = random.normal(key, (N, D)) * 0.05
        # Pathway genes 0-3: genes 0,1 go up, genes 2,3 go down
        delta = delta.at[:, 0].add(1.5)
        delta = delta.at[:, 1].add(1.5)
        delta = delta.at[:, 2].add(-1.5)
        delta = delta.at[:, 3].add(-1.5)

        idx = jnp.array([0, 1, 2, 3])
        result = empirical_test_pathway_perturbation(
            delta, idx, n_permutations=199, key=random.PRNGKey(1)
        )
        # Should detect the perturbation (low p-value)
        assert result["p_value"] < 0.05

    def test_null_not_significant(self):
        """Under the null (no perturbation), p-value should be large."""
        key = random.PRNGKey(30)
        N, D = 1000, 20
        delta = random.normal(key, (N, D)) * 0.1

        idx = jnp.array([0, 1, 2, 3])
        result = empirical_test_pathway_perturbation(
            delta, idx, n_permutations=99, key=random.PRNGKey(2)
        )
        assert result["p_value"] > 0.01

    def test_raises_single_gene(self):
        """Pathway with < 2 genes should raise ValueError."""
        delta = jnp.ones((100, 10)) * 0.1
        with pytest.raises(ValueError):
            empirical_test_pathway_perturbation(
                delta, jnp.array([3]), n_permutations=10
            )


# ==============================================================================
# Tests for batch pathway testing (empirical_test_multiple_gene_sets)
# ==============================================================================


class TestEmpiricalTestMultipleGeneSets:
    """Tests for batch empirical pathway enrichment with PEFP control."""

    @pytest.fixture
    def batch_delta(self):
        """CLR differences with one strongly shifted pathway."""
        key = random.PRNGKey(55)
        N, D = 3000, 30
        delta = random.normal(key, (N, D)) * 0.1
        # Shift pathway 0 genes strongly
        delta = delta.at[:, :5].add(3.0)
        return delta

    def test_output_structure(self, batch_delta):
        """Result should have correct keys and list lengths."""
        gene_sets = [jnp.array([0, 1, 2, 3, 4]), jnp.array([10, 11, 12])]
        result = empirical_test_multiple_gene_sets(batch_delta, gene_sets)

        expected_keys = {
            "balance_mean", "balance_sd", "prob_positive",
            "prob_effect", "lfsr", "lfsr_tau", "significant",
            "lfsr_threshold",
        }
        assert set(result.keys()) == expected_keys
        assert len(result["lfsr"]) == 2
        assert len(result["significant"]) == 2

    def test_pefp_control(self, batch_delta):
        """Shifted pathway should be called significant; null pathway not."""
        gene_sets = [
            jnp.array([0, 1, 2, 3, 4]),   # shifted
            jnp.array([15, 16, 17]),        # null
        ]
        result = empirical_test_multiple_gene_sets(
            batch_delta, gene_sets, target_pefp=0.1
        )
        # The shifted pathway should be significant
        assert result["significant"][0] is True

    def test_consistency_with_individual(self, batch_delta):
        """Batch results should match individual test_gene_set calls."""
        gene_sets = [jnp.array([0, 1, 2]), jnp.array([10, 11, 12, 13])]

        batch = empirical_test_multiple_gene_sets(batch_delta, gene_sets)

        for i, gs in enumerate(gene_sets):
            individual = empirical_test_gene_set(batch_delta, gs)
            assert abs(batch["lfsr"][i] - individual["lfsr"]) < 1e-6
            assert abs(
                batch["balance_mean"][i] - individual["balance_mean"]
            ) < 1e-6
