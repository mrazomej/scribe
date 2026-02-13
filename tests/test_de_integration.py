"""Integration tests for the scribe.de module.

End-to-end tests that exercise the full DE pipeline from model creation
through gene-level analysis, gene-set testing, and error control.
"""

import pytest
import jax.numpy as jnp
from jax import random

from scribe.de import (
    compare,
    ScribeDEResults,
    extract_alr_params,
    differential_expression,
    call_de_genes,
    build_balance_contrast,
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
    alr_to_clr,
    transform_gaussian_alr_to_clr,
    build_ilr_basis,
    clr_to_ilr,
    ilr_to_clr,
)
from scribe.de._set_level import (
    test_contrast as _test_contrast,
    test_gene_set as _test_gene_set,
)
from scribe.stats.distributions import LowRankLogisticNormal, SoftmaxNormal


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def models_with_known_difference():
    """Two models with a known non-uniform difference for testing."""
    D_alr = 30
    k = 4

    key_A = random.PRNGKey(100)
    key_B = random.PRNGKey(200)

    model_A = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(key_A, (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.3,
    }

    # Non-uniform shift so CLR centering doesn't cancel it
    model_B = {
        "loc": random.normal(key_B, (D_alr,)) * 2.0,
        "cov_factor": random.normal(
            random.PRNGKey(300), (D_alr, k)
        ) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.3,
    }

    return model_A, model_B


# --------------------------------------------------------------------------
# Full pipeline tests
# --------------------------------------------------------------------------


def test_full_pipeline_functional(models_with_known_difference):
    """Test full pipeline using functional API."""
    model_A, model_B = models_with_known_difference

    # 1. Gene-level DE
    results = differential_expression(model_A, model_B, tau=jnp.log(1.1))

    # Check all keys present
    assert "delta_mean" in results
    assert "lfsr" in results

    # 2. Call DE genes
    is_de = call_de_genes(results, lfsr_threshold=0.05)
    assert is_de.dtype == jnp.bool_

    # 3. Error control
    pefp = compute_pefp(results["lfsr"], threshold=0.05)
    assert 0 <= pefp <= 1

    threshold = find_lfsr_threshold(results["lfsr"], target_pefp=0.1)
    assert threshold >= 0

    # 4. Formatting
    table = format_de_table(results, sort_by="lfsr", top_n=10)
    assert isinstance(table, str)


def test_full_pipeline_class_based(models_with_known_difference):
    """Test full pipeline using ScribeDEResults class."""
    model_A, model_B = models_with_known_difference

    # 1. Create comparison
    de = compare(model_A, model_B, label_A="WT", label_B="KO")

    # Check type
    assert isinstance(de, ScribeDEResults)

    # 2. Gene-level analysis
    results = de.gene_level(tau=jnp.log(1.1))
    assert results["delta_mean"].shape == (de.D,)

    # 3. Call genes
    is_de = de.call_genes(lfsr_threshold=0.05)
    assert is_de.shape == (de.D,)

    # 4. Pathway analysis
    pathway = jnp.array([0, 1, 2, 3, 4])
    result = de.test_gene_set(pathway, tau=0.0)
    assert "lfsr" in result

    # 5. Error control
    pefp = de.compute_pefp(threshold=0.05)
    assert 0 <= pefp <= 1

    # 6. Summary
    summary = de.summary(top_n=10)
    assert len(summary) > 0


# --------------------------------------------------------------------------
# Embedded vs raw ALR consistency
# --------------------------------------------------------------------------


def test_embedded_vs_raw_alr_give_same_results():
    """Embedded and raw ALR inputs should give identical DE results."""
    D_alr = 20
    k = 3
    key = random.PRNGKey(42)

    # Raw ALR params
    mu_alr = random.normal(key, (D_alr,)) * 0.5
    W_alr = random.normal(random.PRNGKey(43), (D_alr, k)) * 0.1
    d_alr = jnp.ones(D_alr) * 0.5

    # Raw ALR dict
    raw_dict = {"loc": mu_alr, "cov_factor": W_alr, "cov_diag": d_alr}

    # Embedded ALR dict (as from fit_logistic_normal_from_posterior)
    embedded_dict = {
        "loc": jnp.concatenate([mu_alr, jnp.array([0.0])]),
        "cov_factor": jnp.concatenate(
            [W_alr, jnp.zeros((1, k))], axis=0
        ),
        "cov_diag": jnp.concatenate([d_alr, jnp.array([0.0])]),
    }

    # Both should extract the same ALR params
    mu_raw, W_raw, d_raw = extract_alr_params(raw_dict)
    mu_emb, W_emb, d_emb = extract_alr_params(embedded_dict)

    assert jnp.allclose(mu_raw, mu_emb)
    assert jnp.allclose(W_raw, W_emb)
    assert jnp.allclose(d_raw, d_emb)

    # DE results should be identical
    model_B = {
        "loc": jnp.zeros(D_alr),
        "cov_factor": random.normal(random.PRNGKey(44), (D_alr, k)) * 0.1,
        "cov_diag": jnp.ones(D_alr) * 0.5,
    }

    results_raw = differential_expression(raw_dict, model_B, tau=0.0)
    results_emb = differential_expression(embedded_dict, model_B, tau=0.0)

    assert jnp.allclose(
        results_raw["delta_mean"], results_emb["delta_mean"], atol=1e-5
    )
    assert jnp.allclose(
        results_raw["lfsr"], results_emb["lfsr"], atol=1e-5
    )


# --------------------------------------------------------------------------
# Distribution object integration
# --------------------------------------------------------------------------


def test_lowrank_logistic_normal_in_pipeline():
    """LowRankLogisticNormal objects should work in the DE pipeline."""
    D_alr = 20
    k = 3

    dist_A = LowRankLogisticNormal(
        loc=jnp.zeros(D_alr),
        cov_factor=random.normal(random.PRNGKey(10), (D_alr, k)) * 0.1,
        cov_diag=jnp.ones(D_alr) * 0.5,
    )
    dist_B = LowRankLogisticNormal(
        loc=jnp.ones(D_alr) * 0.3,
        cov_factor=random.normal(random.PRNGKey(20), (D_alr, k)) * 0.1,
        cov_diag=jnp.ones(D_alr) * 0.5,
    )

    # Should work with compare()
    de = compare(dist_A, dist_B)
    assert de.D_alr == D_alr

    # And with direct functional calls
    results = differential_expression(dist_A, dist_B, tau=0.0)
    assert results["delta_mean"].shape == (D_alr + 1,)


# --------------------------------------------------------------------------
# Transform integration
# --------------------------------------------------------------------------


def test_transform_pipeline_consistency():
    """ALR -> CLR -> ILR -> CLR should round-trip."""
    D_alr = 10
    D = D_alr + 1

    z_alr = random.normal(random.PRNGKey(55), (D_alr,))

    # ALR -> CLR
    z_clr = alr_to_clr(z_alr)
    assert z_clr.shape == (D,)
    assert jnp.abs(z_clr.sum()) < 1e-6

    # CLR -> ILR -> CLR roundtrip
    V = build_ilr_basis(D)
    z_ilr = clr_to_ilr(z_clr, V)
    z_clr_back = ilr_to_clr(z_ilr, V)

    assert jnp.allclose(z_clr, z_clr_back, atol=1e-5)


def test_gaussian_transform_then_de():
    """Transform Gaussian params and verify DE consistency."""
    D_alr = 15
    k = 3

    mu = random.normal(random.PRNGKey(66), (D_alr,))
    W = random.normal(random.PRNGKey(67), (D_alr, k)) * 0.1
    d = jnp.ones(D_alr) * 0.5

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu, W, d)

    # CLR mean should be centered
    assert jnp.abs(mu_clr.sum()) < 1e-6

    # CLR dimensions should be D = D_alr + 1
    assert mu_clr.shape == (D_alr + 1,)
    assert W_clr.shape == (D_alr + 1, k)
    assert d_clr.shape == (D_alr + 1,)

    # All variances should be positive
    total_var = jnp.sum(W_clr**2, axis=-1) + d_clr
    assert jnp.all(total_var > 0)


# --------------------------------------------------------------------------
# Module import test
# --------------------------------------------------------------------------


def test_all_public_api_importable():
    """All public API symbols should be importable from scribe.de."""
    from scribe.de import (
        ScribeDEResults,
        compare,
        extract_alr_params,
        alr_to_clr,
        transform_gaussian_alr_to_clr,
        build_ilr_basis,
        clr_to_ilr,
        ilr_to_clr,
        differential_expression,
        call_de_genes,
        compute_lfdr,
        compute_pefp,
        find_lfsr_threshold,
        format_de_table,
        build_balance_contrast,
    )
    # Just verify they're callable
    assert callable(compare)
    assert callable(differential_expression)


def test_top_level_imports():
    """ScribeDEResults and compare should be importable from scribe."""
    from scribe import ScribeDEResults, compare

    assert callable(compare)
    assert ScribeDEResults is not None
