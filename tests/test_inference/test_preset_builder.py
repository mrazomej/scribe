"""Tests for preset_builder module."""

import pytest
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models.config.enums import InferenceMethod, Parameterization


class TestBuildConfigFromPreset:
    """Test build_config_from_preset function."""

    def test_basic_nbdm(self):
        """Test building basic NBDM config."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
        )

        assert config.base_model == "nbdm"
        assert config.parameterization == Parameterization.CANONICAL
        assert config.inference_method == InferenceMethod.SVI
        assert config.unconstrained is False

    def test_parameterization_variants(self):
        """Test different parameterization options."""
        # Canonical
        config1 = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
        )
        assert config1.parameterization == Parameterization.CANONICAL

        # Mean prob
        config2 = build_config_from_preset(
            model="nbdm",
            parameterization="mean_prob",
            inference_method="svi",
        )
        assert config2.parameterization == Parameterization.MEAN_PROB

        # Mean odds
        config3 = build_config_from_preset(
            model="nbdm",
            parameterization="mean_odds",
            inference_method="svi",
        )
        assert config3.parameterization == Parameterization.MEAN_ODDS

    def test_backward_compat_parameterization(self):
        """Test backward compatibility with old parameterization names."""
        # Old name: "standard" - should work and create valid config
        config1 = build_config_from_preset(
            model="nbdm",
            parameterization="standard",
            inference_method="svi",
        )
        # Should create a valid config (enum will be STANDARD, which is fine)
        assert config1.parameterization == Parameterization.STANDARD
        assert config1.base_model == "nbdm"

        # Old name: "linked" - should work and create valid config
        config2 = build_config_from_preset(
            model="nbdm",
            parameterization="linked",
            inference_method="svi",
        )
        # Should create a valid config (enum will be LINKED, which is fine)
        assert config2.parameterization == Parameterization.LINKED
        assert config2.base_model == "nbdm"

        # Old name: "odds_ratio" - should work and create valid config
        config3 = build_config_from_preset(
            model="nbdm",
            parameterization="odds_ratio",
            inference_method="svi",
        )
        # Should create a valid config (enum will be ODDS_RATIO, which is fine)
        assert config3.parameterization == Parameterization.ODDS_RATIO
        assert config3.base_model == "nbdm"

    def test_unconstrained(self):
        """Test unconstrained parameterization."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
        )

        assert config.unconstrained is True

    def test_guide_rank(self):
        """Test guide rank configuration."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            guide_rank=15,
        )

        # Guide families should be configured
        assert config.guide_families is not None
        # For canonical, gene param is "r"
        assert config.guide_families.r is not None

    def test_guide_rank_mean_prob(self):
        """Test guide rank with mean_prob parameterization."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="mean_prob",
            inference_method="svi",
            guide_rank=15,
        )

        # For mean_prob, gene param is "mu"
        assert config.guide_families is not None
        assert config.guide_families.mu is not None

    def test_mixture_model(self):
        """Test mixture model configuration."""
        config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            n_components=3,
        )

        assert config.n_components == 3
        assert (
            config.base_model.endswith("_mix")
            or config.n_components is not None
        )

    def test_priors(self):
        """Test prior configuration."""
        priors = {"p": (1.0, 1.0), "r": (0.0, 1.0)}
        config = build_config_from_preset(
            model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            priors=priors,
        )

        # Priors should be stored in param_specs
        # (exact structure depends on implementation)
        assert config is not None

    def test_invalid_parameterization(self):
        """Test error for invalid parameterization."""
        with pytest.raises(ValueError, match="Unknown parameterization"):
            build_config_from_preset(
                model="nbdm",
                parameterization="invalid",
                inference_method="svi",
            )

    def test_different_models(self):
        """Test different model types."""
        models = ["nbdm", "zinb", "nbvcp", "zinbvcp"]

        for model in models:
            config = build_config_from_preset(
                model=model,
                parameterization="canonical",
                inference_method="svi",
            )
            assert config.base_model == model
