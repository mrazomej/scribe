"""Tests for biology-informed capture probability prior.

Tests organism prior resolution, BiologyInformedCaptureSpec creation,
ModelConfig validation, shared_capture_scaling auto-promote, and
model dry-run with the biology-informed capture prior.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from scribe.models.config.organism_priors import (
    ORGANISM_PRIORS,
    resolve_organism_priors,
)
from scribe.models.config import ModelConfig, ModelConfigBuilder
from scribe.models.builders.parameter_specs import BiologyInformedCaptureSpec
from scribe.models.presets.registry import build_capture_spec
from scribe.models.config.groups import GuideFamilyConfig
from scribe.models.config.enums import Parameterization


# =============================================================================
# Organism prior resolution
# =============================================================================


class TestOrganismPriorResolution:
    """Test organism_priors.py lookup table and resolution."""

    def test_human_defaults(self):
        priors = resolve_organism_priors("human")
        assert priors["total_mrna_mean"] == 200_000
        assert priors["total_mrna_log_sigma"] == 0.5

    def test_mouse_defaults(self):
        priors = resolve_organism_priors("mouse")
        assert priors["total_mrna_mean"] == 200_000

    def test_yeast_defaults(self):
        priors = resolve_organism_priors("yeast")
        assert priors["total_mrna_mean"] == 60_000

    def test_ecoli_defaults(self):
        priors = resolve_organism_priors("ecoli")
        assert priors["total_mrna_mean"] == 3_000

    def test_case_insensitive(self):
        priors = resolve_organism_priors("Human")
        assert priors["total_mrna_mean"] == 200_000

    def test_alias_homo_sapiens(self):
        priors = resolve_organism_priors("homo_sapiens")
        assert priors["total_mrna_mean"] == 200_000

    def test_alias_mus_musculus(self):
        priors = resolve_organism_priors("mus_musculus")
        assert priors["total_mrna_mean"] == 200_000

    def test_unknown_organism_raises(self):
        with pytest.raises(ValueError, match="Unknown organism"):
            resolve_organism_priors("zebrafish")


# =============================================================================
# ModelConfig validation
# =============================================================================


class TestModelConfigCapturePrior:
    """Test ModelConfig validation for capture_prior fields."""

    def test_flat_prior_no_organism_needed(self):
        """Flat prior is the default and doesn't need organism."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .build()
        )
        assert config.capture_prior == "default"
        assert config.organism is None

    def test_biology_informed_requires_organism_or_M0(self):
        """biology_informed without organism or M_0 should raise."""
        with pytest.raises(ValueError, match="organism.*total_mrna_mean"):
            ModelConfigBuilder()._capture_prior = "biology_informed"
            builder = ModelConfigBuilder()
            builder._base_model = "nbvcp"
            builder._capture_prior = "biology_informed"
            builder.build()

    def test_biology_informed_with_organism(self):
        """biology_informed with organism should resolve M_0."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._capture_prior = "biology_informed"
        builder._organism = "human"
        config = builder.build()
        assert config.capture_prior == "biology_informed"
        assert config.total_mrna_mean == 200_000
        assert config.total_mrna_log_sigma == 0.5

    def test_biology_informed_manual_M0(self):
        """Explicit total_mrna_mean overrides organism default."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._capture_prior = "biology_informed"
        builder._total_mrna_mean = 100_000
        config = builder.build()
        assert config.total_mrna_mean == 100_000
        # sigma defaults to 0.5 when not specified
        assert config.total_mrna_log_sigma == 0.5

    def test_biology_informed_manual_override_organism(self):
        """Explicit M_0 takes precedence over organism default."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._capture_prior = "biology_informed"
        builder._organism = "human"
        builder._total_mrna_mean = 150_000
        config = builder.build()
        # Explicit value wins over organism default
        assert config.total_mrna_mean == 150_000

    def test_shared_capture_scaling_with_biology_informed(self):
        """shared_capture_scaling with biology_informed keeps the prior mode."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._capture_prior = "biology_informed"
        builder._shared_capture_scaling = True
        builder._organism = "human"
        config = builder.build()
        assert config.capture_prior == "biology_informed"
        assert config.shared_capture_scaling is True
        assert config.total_mrna_mean == 200_000

    def test_shared_capture_scaling_auto_promotes_default(self):
        """shared_capture_scaling with default auto-promotes to biology_informed."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._shared_capture_scaling = True
        config = builder.build()
        # Auto-promoted from default to biology_informed
        assert config.capture_prior == "biology_informed"
        assert config.shared_capture_scaling is True
        # Falls back to mammalian default
        assert config.total_mrna_mean == 200_000
        assert config.total_mrna_log_sigma == 0.5

    def test_shared_capture_scaling_auto_promote_with_organism(self):
        """Auto-promote uses organism M_0 as center when provided."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._shared_capture_scaling = True
        builder._organism = "yeast"
        config = builder.build()
        assert config.capture_prior == "biology_informed"
        assert config.total_mrna_mean == 60_000

    def test_shared_capture_scaling_requires_vcp(self):
        """shared_capture_scaling with non-VCP model should raise."""
        with pytest.raises(ValueError, match="VCP"):
            builder = ModelConfigBuilder()
            builder._base_model = "nbdm"
            builder._shared_capture_scaling = True
            builder.build()

    def test_capture_prior_requires_vcp(self):
        """Non-flat capture_prior with non-VCP model should raise."""
        with pytest.raises(ValueError, match="VCP"):
            builder = ModelConfigBuilder()
            builder._base_model = "nbdm"
            builder._capture_prior = "biology_informed"
            builder._organism = "human"
            builder.build()

    def test_invalid_capture_prior_mode(self):
        """Invalid capture_prior mode should raise."""
        with pytest.raises(ValueError, match="capture_prior"):
            builder = ModelConfigBuilder()
            builder._base_model = "nbvcp"
            builder._capture_prior = "invalid_mode"
            builder.build()

    def test_data_driven_is_no_longer_valid_mode(self):
        """'data_driven' was replaced by shared_capture_scaling flag."""
        with pytest.raises(ValueError, match="capture_prior"):
            builder = ModelConfigBuilder()
            builder._base_model = "nbvcp"
            builder._capture_prior = "data_driven"
            builder.build()


# =============================================================================
# BiologyInformedCaptureSpec creation
# =============================================================================


class TestBiologyInformedCaptureSpec:
    """Test the BiologyInformedCaptureSpec class."""

    def test_spec_creation_phi(self):
        """Spec for phi_capture (mean_odds)."""
        spec = BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=("n_cells",),
            default_params=(math.log(200_000), 0.5),
            is_cell_specific=True,
            log_M0=math.log(200_000),
            sigma_M=0.5,
            data_driven=False,
            use_phi_capture=True,
        )
        assert spec.use_phi_capture is True
        assert spec.data_driven is False
        assert spec.log_M0 == pytest.approx(math.log(200_000))

    def test_spec_creation_p(self):
        """Spec for p_capture (canonical/mean_prob)."""
        spec = BiologyInformedCaptureSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(math.log(200_000), 0.5),
            is_cell_specific=True,
            log_M0=math.log(200_000),
            sigma_M=0.5,
            data_driven=False,
            use_phi_capture=False,
        )
        assert spec.use_phi_capture is False

    def test_data_driven_spec(self):
        """Data-driven spec with learned mu_eta."""
        spec = BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=("n_cells",),
            default_params=(math.log(200_000), 0.5),
            is_cell_specific=True,
            log_M0=math.log(200_000),
            sigma_M=0.5,
            data_driven=True,
            sigma_mu=1.0,
            use_phi_capture=True,
        )
        assert spec.data_driven is True
        assert spec.sigma_mu == 1.0


# =============================================================================
# build_capture_spec integration
# =============================================================================


class TestBuildCaptureSpec:
    """Test build_capture_spec with biology-informed config."""

    def test_flat_returns_standard_spec(self):
        """Flat prior should return ExpNormalSpec for phi_capture."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .build()
        )
        from scribe.models.builders.parameter_specs import ExpNormalSpec
        from scribe.models.parameterizations import PARAMETERIZATIONS

        param_strategy = PARAMETERIZATIONS[config.parameterization]
        spec = build_capture_spec(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
            param_strategy=param_strategy,
            model_config=config,
        )
        assert isinstance(spec, ExpNormalSpec)

    def test_biology_informed_returns_bio_spec(self):
        """Biology-informed config should return BiologyInformedCaptureSpec."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._parameterization = Parameterization.MEAN_ODDS
        builder._capture_prior = "biology_informed"
        builder._organism = "human"
        config = builder.build()

        from scribe.models.parameterizations import PARAMETERIZATIONS

        param_strategy = PARAMETERIZATIONS[config.parameterization]
        spec = build_capture_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            param_strategy=param_strategy,
            model_config=config,
        )
        assert isinstance(spec, BiologyInformedCaptureSpec)
        assert spec.use_phi_capture is True
        assert spec.log_M0 == pytest.approx(math.log(200_000))

    def test_shared_scaling_returns_bio_spec_data_driven(self):
        """shared_capture_scaling should produce BiologyInformedCaptureSpec with data_driven=True."""
        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._parameterization = Parameterization.MEAN_ODDS
        builder._capture_prior = "biology_informed"
        builder._shared_capture_scaling = True
        builder._organism = "mouse"
        config = builder.build()

        from scribe.models.parameterizations import PARAMETERIZATIONS

        param_strategy = PARAMETERIZATIONS[config.parameterization]
        spec = build_capture_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            param_strategy=param_strategy,
            model_config=config,
        )
        assert isinstance(spec, BiologyInformedCaptureSpec)
        assert spec.data_driven is True


# =============================================================================
# Model dry run with biology-informed capture prior
# =============================================================================


class TestModelDryRun:
    """Test that model creation succeeds with biology-informed capture."""

    def test_nbvcp_biology_informed_mean_odds(self):
        """NBVCP with biology-informed prior in mean_odds should create."""
        from scribe.models.presets.factory import create_model

        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._parameterization = Parameterization.MEAN_ODDS
        builder._unconstrained = True
        builder._capture_prior = "biology_informed"
        builder._organism = "human"
        config = builder.build()

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None
        assert guide_fn is not None

        # Check that a BiologyInformedCaptureSpec was produced
        bio_specs = [
            s for s in param_specs
            if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].use_phi_capture is True

    def test_zinbvcp_biology_informed_canonical(self):
        """ZINBVCP with biology-informed prior in canonical should create."""
        from scribe.models.presets.factory import create_model

        builder = ModelConfigBuilder()
        builder._base_model = "zinbvcp"
        builder._parameterization = Parameterization.CANONICAL
        builder._capture_prior = "biology_informed"
        builder._organism = "yeast"
        config = builder.build()

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs
            if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].use_phi_capture is False
        assert bio_specs[0].log_M0 == pytest.approx(math.log(60_000))

    def test_nbvcp_shared_capture_scaling(self):
        """NBVCP with shared_capture_scaling should create with data_driven spec."""
        from scribe.models.presets.factory import create_model

        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._parameterization = Parameterization.MEAN_ODDS
        builder._unconstrained = True
        builder._capture_prior = "biology_informed"
        builder._shared_capture_scaling = True
        builder._organism = "human"
        config = builder.build()

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs
            if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].data_driven is True

    def test_nbvcp_auto_promote_shared_scaling(self):
        """Auto-promoted shared_capture_scaling should create successfully."""
        from scribe.models.presets.factory import create_model

        builder = ModelConfigBuilder()
        builder._base_model = "nbvcp"
        builder._parameterization = Parameterization.MEAN_ODDS
        builder._unconstrained = True
        builder._shared_capture_scaling = True
        config = builder.build()

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs
            if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].data_driven is True

    def test_builder_with_capture_prior_method(self):
        """Test the builder's with_capture_prior method."""
        builder = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_prior(
                mode="biology_informed",
                organism="human",
            )
        )
        config = builder.build()
        assert config.capture_prior == "biology_informed"
        assert config.total_mrna_mean == 200_000

    def test_builder_with_organism_method(self):
        """Test the builder's with_organism method."""
        builder = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_prior(mode="biology_informed")
            .with_organism("mouse")
        )
        config = builder.build()
        assert config.organism == "mouse"
        assert config.total_mrna_mean == 200_000
