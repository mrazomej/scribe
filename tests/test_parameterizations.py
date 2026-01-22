"""Tests for parameterization strategies.

This module tests the parameterization classes and their derived parameter
computations, particularly focusing on proper broadcasting for mixture models.
"""

import jax.numpy as jnp
import pytest

from scribe.models.parameterizations import (
    PARAMETERIZATIONS,
    CanonicalParameterization,
    MeanOddsParameterization,
    MeanProbParameterization,
    _broadcast_scalar_for_mixture,
    _compute_r_from_mu_p,
    _compute_r_from_mu_phi,
)


# ==============================================================================
# Test Parameterization Registry
# ==============================================================================


class TestParameterizationRegistry:
    """Test the PARAMETERIZATIONS registry."""

    def test_registry_contains_all_parameterizations(self):
        """Test registry has all expected parameterizations."""
        expected = {"canonical", "mean_prob", "mean_odds"}
        assert expected.issubset(set(PARAMETERIZATIONS.keys()))

    def test_registry_aliases(self):
        """Test backward compatibility aliases."""
        assert PARAMETERIZATIONS["standard"] is PARAMETERIZATIONS["canonical"]
        assert PARAMETERIZATIONS["linked"] is PARAMETERIZATIONS["mean_prob"]
        assert PARAMETERIZATIONS["odds_ratio"] is PARAMETERIZATIONS["mean_odds"]

    def test_canonical_core_parameters(self):
        """Test canonical parameterization core parameters."""
        param = PARAMETERIZATIONS["canonical"]
        assert param.core_parameters == ["p", "r"]

    def test_mean_prob_core_parameters(self):
        """Test mean_prob parameterization core parameters."""
        param = PARAMETERIZATIONS["mean_prob"]
        assert param.core_parameters == ["p", "mu"]

    def test_mean_odds_core_parameters(self):
        """Test mean_odds parameterization core parameters."""
        param = PARAMETERIZATIONS["mean_odds"]
        assert param.core_parameters == ["phi", "mu"]


# ==============================================================================
# Test Broadcasting Helper
# ==============================================================================


class TestBroadcastScalarForMixture:
    """Test the _broadcast_scalar_for_mixture helper function."""

    def test_scalar_no_expansion(self):
        """Test scalar param is not expanded."""
        scalar = jnp.array(0.5)
        gene = jnp.array([1.0, 2.0, 3.0])
        result = _broadcast_scalar_for_mixture(scalar, gene)
        assert result.shape == ()

    def test_1d_with_1d_no_expansion(self):
        """Test 1D params don't expand when gene param is also 1D."""
        scalar = jnp.array([0.5, 0.6])
        gene = jnp.array([1.0, 2.0, 3.0])
        result = _broadcast_scalar_for_mixture(scalar, gene)
        assert result.shape == (2,)

    def test_mixture_expansion(self):
        """Test mixture param is expanded for broadcasting with gene param."""
        # scalar: (n_components,) = (2,)
        # gene: (n_components, n_genes) = (2, 5)
        scalar = jnp.array([0.5, 0.6])
        gene = jnp.ones((2, 5))
        result = _broadcast_scalar_for_mixture(scalar, gene)
        assert result.shape == (2, 1)

    def test_already_2d_no_expansion(self):
        """Test 2D param is not further expanded."""
        scalar = jnp.ones((2, 5))
        gene = jnp.ones((2, 5))
        result = _broadcast_scalar_for_mixture(scalar, gene)
        assert result.shape == (2, 5)

    def test_mismatched_components_no_expansion(self):
        """Test no expansion when component counts don't match."""
        scalar = jnp.array([0.5, 0.6, 0.7])  # 3 components
        gene = jnp.ones((2, 5))  # 2 components
        result = _broadcast_scalar_for_mixture(scalar, gene)
        assert result.shape == (3,)  # No expansion


# ==============================================================================
# Test Mean Odds Derived Parameters
# ==============================================================================


class TestMeanOddsDerivedParams:
    """Test derived parameter computation for mean_odds parameterization."""

    def test_scalar_phi_gene_mu(self):
        """Test r = mu * phi with scalar phi and gene-specific mu."""
        phi = jnp.array(2.0)
        mu = jnp.array([1.0, 2.0, 3.0])
        r = _compute_r_from_mu_phi(phi, mu)
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(r, expected)

    def test_mixture_phi_gene_mu(self):
        """Test r = mu * phi with mixture-specific phi and gene-specific mu.

        This is the case that previously failed with broadcasting error.
        phi: (n_components,) = (2,)
        mu: (n_components, n_genes) = (2, 3)
        """
        phi = jnp.array([2.0, 3.0])  # 2 components
        mu = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        r = _compute_r_from_mu_phi(phi, mu)

        # Expected: r[k, g] = mu[k, g] * phi[k]
        # r[0, :] = [1, 2, 3] * 2 = [2, 4, 6]
        # r[1, :] = [4, 5, 6] * 3 = [12, 15, 18]
        expected = jnp.array([[2.0, 4.0, 6.0], [12.0, 15.0, 18.0]])
        assert r.shape == (2, 3)
        assert jnp.allclose(r, expected)

    def test_both_mixture_gene_specific(self):
        """Test r = mu * phi when both are mixture and gene specific."""
        phi = jnp.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])  # (2, 3)
        mu = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        r = _compute_r_from_mu_phi(phi, mu)
        expected = phi * mu
        assert r.shape == (2, 3)
        assert jnp.allclose(r, expected)

    def test_build_derived_params(self):
        """Test MeanOddsParameterization.build_derived_params returns correct specs."""
        param = MeanOddsParameterization()
        derived = param.build_derived_params()

        assert len(derived) == 2
        assert derived[0].name == "r"
        assert derived[0].deps == ["phi", "mu"]
        assert derived[1].name == "p"
        assert derived[1].deps == ["phi"]


# ==============================================================================
# Test Mean Prob Derived Parameters
# ==============================================================================


class TestMeanProbDerivedParams:
    """Test derived parameter computation for mean_prob parameterization."""

    def test_scalar_p_gene_mu(self):
        """Test r = mu * (1-p) / p with scalar p and gene-specific mu."""
        p = jnp.array(0.5)
        mu = jnp.array([1.0, 2.0, 3.0])
        r = _compute_r_from_mu_p(p, mu)
        # r = mu * (1 - 0.5) / 0.5 = mu * 1 = mu
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(r, expected)

    def test_mixture_p_gene_mu(self):
        """Test r = mu * (1-p) / p with mixture-specific p and gene-specific mu.

        This is the case that previously failed with broadcasting error.
        p: (n_components,) = (2,)
        mu: (n_components, n_genes) = (2, 3)
        """
        p = jnp.array([0.5, 0.25])  # 2 components
        mu = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        r = _compute_r_from_mu_p(p, mu)

        # Expected: r[k, g] = mu[k, g] * (1 - p[k]) / p[k]
        # r[0, :] = [1, 2, 3] * (1 - 0.5) / 0.5 = [1, 2, 3] * 1 = [1, 2, 3]
        # r[1, :] = [4, 5, 6] * (1 - 0.25) / 0.25 = [4, 5, 6] * 3 = [12, 15, 18]
        expected = jnp.array([[1.0, 2.0, 3.0], [12.0, 15.0, 18.0]])
        assert r.shape == (2, 3)
        assert jnp.allclose(r, expected)

    def test_both_mixture_gene_specific(self):
        """Test r = mu * (1-p) / p when both are mixture and gene specific."""
        p = jnp.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]])  # (2, 3)
        mu = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        r = _compute_r_from_mu_p(p, mu)
        expected = mu * (1 - p) / p
        assert r.shape == (2, 3)
        assert jnp.allclose(r, expected)

    def test_build_derived_params(self):
        """Test MeanProbParameterization.build_derived_params returns correct specs."""
        param = MeanProbParameterization()
        derived = param.build_derived_params()

        assert len(derived) == 1
        assert derived[0].name == "r"
        assert derived[0].deps == ["p", "mu"]


# ==============================================================================
# Test Canonical Parameterization (No Derived Params)
# ==============================================================================


class TestCanonicalParameterization:
    """Test canonical parameterization has no derived parameters."""

    def test_no_derived_params(self):
        """Test canonical parameterization has empty derived params."""
        param = CanonicalParameterization()
        derived = param.build_derived_params()
        assert derived == []


# ==============================================================================
# Integration Tests with Model Factory
# ==============================================================================


class TestMixtureBroadcastingIntegration:
    """Integration tests for mixture broadcasting through model factory."""

    @pytest.fixture
    def n_cells(self):
        return 50

    @pytest.fixture
    def n_genes(self):
        return 20

    def test_mean_odds_mixture_model_runs(self, n_cells, n_genes):
        """Test mean_odds mixture model with phi mixture-specific runs without error."""
        import numpyro

        from scribe.models.config import ModelConfigBuilder
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["mu", "phi"])
            .build()
        )
        model, guide = create_model(config, validate=False)

        # Run model in prior predictive mode
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=config,
                    counts=None,
                )

        # Verify shapes
        assert trace["phi"]["value"].shape == (2,)
        assert trace["mu"]["value"].shape == (2, n_genes)
        assert trace["r"]["value"].shape == (2, n_genes)

    def test_mean_prob_mixture_model_runs(self, n_cells, n_genes):
        """Test mean_prob mixture model with p mixture-specific runs without error."""
        import numpyro

        from scribe.models.config import ModelConfigBuilder
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_prob")
            .as_mixture(2, ["mu", "p"])
            .build()
        )
        model, guide = create_model(config, validate=False)

        # Run model in prior predictive mode
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=config,
                    counts=None,
                )

        # Verify shapes
        assert trace["p"]["value"].shape == (2,)
        assert trace["mu"]["value"].shape == (2, n_genes)
        assert trace["r"]["value"].shape == (2, n_genes)

    def test_mean_odds_only_mu_mixture(self, n_cells, n_genes):
        """Test mean_odds with only mu mixture-specific (phi shared)."""
        import numpyro

        from scribe.models.config import ModelConfigBuilder
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbdm")
            .with_parameterization("mean_odds")
            .as_mixture(2, ["mu"])  # Only mu is mixture-specific
            .build()
        )
        model, guide = create_model(config, validate=False)

        # Run model in prior predictive mode
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model(
                    n_cells=n_cells,
                    n_genes=n_genes,
                    model_config=config,
                    counts=None,
                )

        # phi is scalar (shared), mu and r are mixture+gene specific
        assert trace["phi"]["value"].shape == ()
        assert trace["mu"]["value"].shape == (2, n_genes)
        assert trace["r"]["value"].shape == (2, n_genes)
