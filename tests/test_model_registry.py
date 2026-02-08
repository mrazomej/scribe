"""
Test the decorator-based model registry system.

This test suite validates that the @register decorator correctly registers
all models and guides, and that get_model_and_guide retrieves them correctly.
"""

import pytest
from src.scribe.models import get_model_and_guide
from src.scribe.models.model_registry import (
    _MODEL_REGISTRY,
    _GUIDE_REGISTRY,
    SUPPORTED_PARAMETERIZATIONS,
    SUPPORTED_INFERENCE_METHODS,
    get_model_and_guide_legacy,
)
from src.scribe.inference.preset_builder import build_config_from_preset


class TestRegistryPopulation:
    """Test that the registry is properly populated."""

    def test_registry_not_empty(self):
        """Registry should contain registered models and guides."""
        assert len(_MODEL_REGISTRY) > 0, "Model registry is empty"
        assert len(_GUIDE_REGISTRY) > 0, "Guide registry is empty"

    def test_registry_counts(self):
        """Check expected number of registrations."""
        # We have 4 base model types (nbdm, zinb, nbvcp, zinbvcp)
        # Each with _mix variant = 8 model types
        # 3 parameterizations (standard, linked, odds_ratio)
        # 2 inference methods for non-VAE (svi, mcmc)
        # 2 variants (constrained, unconstrained)
        # Expected non-VAE models: 8 * 3 * 2 * 2 = 96 (mean_field only for models)

        non_vae_models = sum(
            1 for k in _MODEL_REGISTRY if k[2] in ["svi", "mcmc"]
        )
        assert (
            non_vae_models == 96
        ), f"Expected 96 non-VAE models, got {non_vae_models}"

        # VAE models: 2 base types (nbdm, nbvcp) * 3 param * 2 prior * 2 unconstrained = 24
        vae_models = sum(1 for k in _MODEL_REGISTRY if k[2] == "vae")
        assert vae_models == 24, f"Expected 24 VAE models, got {vae_models}"

    def test_all_parameterizations_registered(self):
        """All parameterizations should have registered models."""
        for param in SUPPORTED_PARAMETERIZATIONS:
            models_for_param = [k for k in _MODEL_REGISTRY if k[1] == param]
            assert len(models_for_param) > 0, f"No models for {param}"

    def test_all_inference_methods_registered(self):
        """All inference methods should have registered models."""
        for method in SUPPORTED_INFERENCE_METHODS:
            models_for_method = [k for k in _MODEL_REGISTRY if k[2] == method]
            assert len(models_for_method) > 0, f"No models for {method}"


class TestGetModelAndGuide:
    """Test the get_model_and_guide function."""

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    @pytest.mark.parametrize("inference_method", ["svi", "mcmc"])
    def test_basic_lookup(self, model_type, parameterization, inference_method):
        """Test basic model/guide lookup for non-VAE models."""
        config = build_config_from_preset(
            model=model_type,
            parameterization=parameterization,
            inference_method=inference_method,
        )
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None
        assert callable(model)
        assert callable(guide)

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    def test_mixture_models(self, model_type, parameterization):
        """Test mixture model lookup."""
        config = build_config_from_preset(
            model=model_type,
            parameterization=parameterization,
            inference_method="svi",
            n_components=2,  # Create mixture model
        )
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    def test_unconstrained_variants(self, model_type, parameterization):
        """Test unconstrained model/guide lookup."""
        config = build_config_from_preset(
            model=model_type,
            parameterization=parameterization,
            inference_method="svi",
            unconstrained=True,
        )
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None

    @pytest.mark.parametrize("model_type", ["nbdm", "zinb", "nbvcp", "zinbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    @pytest.mark.parametrize("guide_rank", [5, 10])
    def test_low_rank_guides(self, model_type, parameterization, guide_rank):
        """Test low-rank guide lookup."""
        config = build_config_from_preset(
            model=model_type,
            parameterization=parameterization,
            inference_method="svi",
            guide_rank=guide_rank,
        )
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None

    @pytest.mark.parametrize("model_type", ["nbdm", "nbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    @pytest.mark.parametrize("prior_type", ["standard", "decoupled"])
    def test_vae_models(self, model_type, parameterization, prior_type):
        """Test VAE model factory lookup."""
        # VAE models use the legacy API
        factory, guide = get_model_and_guide_legacy(
            model_type=model_type,
            parameterization=parameterization,
            inference_method="vae",
            prior_type=prior_type,
        )
        assert factory is not None
        assert callable(factory)
        assert guide is None  # VAE returns factory, not guide

    @pytest.mark.parametrize("model_type", ["nbdm", "nbvcp"])
    @pytest.mark.parametrize(
        "parameterization", ["standard", "linked", "odds_ratio"]
    )
    def test_vae_unconstrained(self, model_type, parameterization):
        """Test unconstrained VAE models."""
        # VAE models use the legacy API
        factory, guide = get_model_and_guide_legacy(
            model_type=model_type,
            parameterization=parameterization,
            inference_method="vae",
            prior_type="standard",
            unconstrained=True,
        )
        assert factory is not None
        assert callable(factory)


class TestRegistryKeyStructure:
    """Test the structure of registry keys."""

    def test_key_format(self):
        """Registry keys should be 6-tuples."""
        for key in _MODEL_REGISTRY.keys():
            assert isinstance(key, tuple)
            assert len(key) == 6
            (
                model_type,
                param,
                inf_method,
                prior_type,
                unconstrained,
                guide_variant,
            ) = key
            assert isinstance(model_type, str)
            assert isinstance(param, str)
            assert isinstance(inf_method, str)
            assert prior_type is None or isinstance(prior_type, str)
            assert isinstance(unconstrained, bool)
            assert isinstance(guide_variant, str)

    def test_non_vae_keys_have_null_prior(self):
        """Non-VAE models should have None for prior_type."""
        for key in _MODEL_REGISTRY.keys():
            _, _, inf_method, prior_type, _, _ = key
            if inf_method in ["svi", "mcmc"]:
                assert prior_type is None

    def test_vae_keys_have_prior_type(self):
        """VAE models should have a prior_type."""
        for key in _MODEL_REGISTRY.keys():
            _, _, inf_method, prior_type, _, _ = key
            if inf_method == "vae":
                assert prior_type in ["standard", "decoupled"]

    def test_models_use_mean_field_variant(self):
        """All models should be registered with mean_field guide_variant."""
        for key in _MODEL_REGISTRY.keys():
            _, _, _, _, _, guide_variant = key
            assert guide_variant == "mean_field"

    def test_guides_have_both_variants(self):
        """Guides should be registered for both mean_field and low_rank."""
        mean_field_guides = sum(
            1 for k in _GUIDE_REGISTRY if k[5] == "mean_field"
        )
        low_rank_guides = sum(1 for k in _GUIDE_REGISTRY if k[5] == "low_rank")

        assert mean_field_guides > 0
        assert low_rank_guides > 0

        # Non-VAE guides should have both variants
        non_vae_mean_field = sum(
            1
            for k in _GUIDE_REGISTRY
            if k[5] == "mean_field" and k[2] in ["svi", "mcmc"]
        )
        non_vae_low_rank = sum(
            1
            for k in _GUIDE_REGISTRY
            if k[5] == "low_rank" and k[2] in ["svi", "mcmc"]
        )
        assert (
            non_vae_mean_field == non_vae_low_rank
        ), f"Non-VAE guides should have equal mean_field ({non_vae_mean_field}) and low_rank ({non_vae_low_rank})"

        # VAE guides only have mean_field variant (no low_rank for VAE)
        vae_mean_field = sum(
            1 for k in _GUIDE_REGISTRY if k[5] == "mean_field" and k[2] == "vae"
        )
        vae_low_rank = sum(
            1 for k in _GUIDE_REGISTRY if k[5] == "low_rank" and k[2] == "vae"
        )
        assert vae_mean_field > 0, "Should have VAE mean_field guides"
        assert vae_low_rank == 0, "Should have no VAE low_rank guides"


class TestErrorHandling:
    """Test error handling in get_model_and_guide."""

    def test_invalid_parameterization(self):
        """Should raise error for invalid parameterization."""
        with pytest.raises(ValueError, match="Unknown parameterization"):
            build_config_from_preset("nbdm", parameterization="invalid_param")

    def test_nonexistent_model_type(self):
        """Should raise error for non-existent model type."""
        from scribe.models.config import ModelConfigBuilder

        config = (
            ModelConfigBuilder()
            .for_model("nonexistent_model")
            .with_parameterization("canonical")
            .with_inference("svi")
            .build()
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_model_and_guide(config)


class TestBackwardCompatibility:
    """Test that the registry maintains backward compatibility."""

    def test_default_parameters(self):
        """Test that default parameters work as before."""
        # Should default to canonical/svi
        config = build_config_from_preset("nbdm")
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None

    def test_mcmc_and_svi_share_models(self):
        """SVI and MCMC should retrieve the same model/guide functions."""
        config_svi = build_config_from_preset(
            "nbdm", parameterization="standard", inference_method="svi"
        )
        config_mcmc = build_config_from_preset(
            "nbdm", parameterization="standard", inference_method="mcmc"
        )
        model_svi, guide_svi, _ = get_model_and_guide(config_svi)
        model_mcmc, guide_mcmc, _ = get_model_and_guide(config_mcmc)

        # They should be the exact same function objects
        assert model_svi is model_mcmc
        assert guide_svi is guide_mcmc

    def test_low_rank_shares_model(self):
        """Low-rank guides should share models with mean-field."""
        config_mf = build_config_from_preset(
            "nbdm", parameterization="standard", inference_method="svi"
        )
        model_mf, _, _ = get_model_and_guide(config_mf)
        config_lr = build_config_from_preset(
            "nbdm",
            parameterization="standard",
            inference_method="svi",
            guide_rank=5,
        )
        model_lr, _, _ = get_model_and_guide(config_lr)

        # Should be the same model
        assert model_mf is model_lr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
