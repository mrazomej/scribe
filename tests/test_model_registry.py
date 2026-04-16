"""
Test the model registry system.

This test suite validates that ``get_model_and_guide`` retrieves model/guide
function pairs via the composable builder factory for all supported model
types, parameterizations, and inference methods.
"""

import pytest
from src.scribe.models import get_model_and_guide
from src.scribe.inference.preset_builder import build_config_from_preset


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
            n_components=2,
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


class TestErrorHandling:
    """Test error handling in get_model_and_guide."""

    def test_invalid_parameterization(self):
        """Should raise error for invalid parameterization."""
        with pytest.raises(ValueError, match="Unknown parameterization"):
            build_config_from_preset("nbdm", parameterization="invalid_param")

    def test_nonexistent_model_type(self):
        """Should raise error for non-existent model type."""
        from scribe.models.config import ModelConfigBuilder

        with pytest.raises(Exception, match="[Ii]nvalid model"):
            ModelConfigBuilder().for_model(
                "nonexistent_model"
            ).with_parameterization("canonical").with_inference("svi").build()


class TestBackwardCompatibility:
    """Test that the registry maintains backward compatibility."""

    def test_default_parameters(self):
        """Test that default parameters work as before."""
        config = build_config_from_preset("nbdm")
        model, guide, _ = get_model_and_guide(config)
        assert model is not None
        assert guide is not None

    def test_mcmc_and_svi_both_return_callables(self):
        """SVI and MCMC should both return callable model/guide pairs."""
        config_svi = build_config_from_preset(
            "nbdm", parameterization="standard", inference_method="svi"
        )
        config_mcmc = build_config_from_preset(
            "nbdm", parameterization="standard", inference_method="mcmc"
        )
        model_svi, guide_svi, _ = get_model_and_guide(config_svi)
        model_mcmc, guide_mcmc, _ = get_model_and_guide(config_mcmc)

        assert callable(model_svi)
        assert callable(guide_svi)
        assert callable(model_mcmc)
        assert callable(guide_mcmc)

    def test_low_rank_returns_callable_model(self):
        """Low-rank and mean-field configs should both produce callable models."""
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

        assert callable(model_mf)
        assert callable(model_lr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
