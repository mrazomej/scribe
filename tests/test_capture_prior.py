"""Tests for biology-informed capture probability prior.

Tests organism prior resolution, BiologyInformedCaptureSpec creation,
ModelConfig validation, shared_capture_scaling, TruncatedNormal
enforcement, and model dry-run with the biology-informed capture prior.

The capture prior is configured entirely through the ``priors`` dict:
  - ``priors.organism``    — shortcut to resolve defaults
  - ``priors.eta_capture`` — explicit ``[log_M0, sigma_M]``
  - ``priors.mu_eta``      — explicit ``[center, sigma_mu]``
"""

import math

import jax.numpy as jnp
import numpyro.distributions as dist
import numpy as np
import pytest

from scribe.models.config.organism_priors import (
    ORGANISM_PRIORS,
    resolve_organism_priors,
)
from scribe.models.config import ModelConfig, ModelConfigBuilder
from scribe.models.builders.parameter_specs import BiologyInformedCaptureSpec
from scribe.models.presets.registry import build_capture_spec
from scribe.models.config.groups import GuideFamilyConfig, PriorOverrides
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
# ModelConfig validation — new priors-based API
# =============================================================================


class TestModelConfigCapturePrior:
    """Test ModelConfig validation for priors-based capture configuration."""

    def test_default_no_biology_informed(self):
        """Default config has no biology-informed capture."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .build()
        )
        assert config.uses_biology_informed_capture is False

    def test_organism_activates_biology_informed(self):
        """priors.organism should resolve eta_capture and activate bio path."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human")
            .build()
        )
        assert config.uses_biology_informed_capture is True
        extra = getattr(config.priors, "__pydantic_extra__", {})
        eta = extra.get("eta_capture")
        assert eta is not None
        assert eta[0] == pytest.approx(math.log(200_000))
        assert eta[1] == pytest.approx(0.5)

    def test_explicit_eta_capture(self):
        """Explicit priors.eta_capture overrides organism defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(eta_capture=(11.0, 0.3))
            .build()
        )
        assert config.uses_biology_informed_capture is True
        extra = getattr(config.priors, "__pydantic_extra__", {})
        assert extra["eta_capture"] == (11.0, 0.3)

    def test_eta_capture_overrides_organism(self):
        """Explicit eta_capture takes precedence over organism."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human", eta_capture=(10.0, 0.2))
            .build()
        )
        extra = getattr(config.priors, "__pydantic_extra__", {})
        # Explicit eta_capture wins
        assert extra["eta_capture"] == (10.0, 0.2)

    def test_shared_scaling_with_organism(self):
        """shared_capture_scaling=True + organism resolves mu_eta defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human", shared_capture_scaling=True)
            .build()
        )
        assert config.shared_capture_scaling is True
        extra = getattr(config.priors, "__pydantic_extra__", {})
        mu_eta = extra.get("mu_eta")
        assert mu_eta is not None
        # Center from eta_capture[0], sigma_mu defaults to 1.0 (anchored)
        assert mu_eta[0] == pytest.approx(math.log(200_000))
        assert mu_eta[1] == pytest.approx(1.0)

    def test_shared_scaling_explicit_mu_eta(self):
        """Explicit priors.mu_eta overrides defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(
                organism="human",
                mu_eta=(11.5, 0.5),
                shared_capture_scaling=True,
            )
            .build()
        )
        extra = getattr(config.priors, "__pydantic_extra__", {})
        assert extra["mu_eta"] == (11.5, 0.5)

    def test_shared_scaling_requires_vcp(self):
        """shared_capture_scaling with non-VCP model should raise."""
        with pytest.raises(ValueError, match="VCP"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_capture_priors(
                    organism="human", shared_capture_scaling=True
                )
                .build()
            )

    def test_eta_capture_requires_vcp(self):
        """priors.eta_capture with non-VCP model should raise."""
        with pytest.raises(ValueError, match="VCP"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_capture_priors(organism="human")
                .build()
            )

    def test_sigma_mu_default_anchored(self):
        """When shared scaling + anchor, sigma_mu defaults to 1.0."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(
                eta_capture=(11.5, 0.5), shared_capture_scaling=True
            )
            .build()
        )
        extra = getattr(config.priors, "__pydantic_extra__", {})
        assert extra["mu_eta"][1] == pytest.approx(1.0)

    def test_backward_compat_setstate(self):
        """Old pickled configs with capture_prior field should migrate."""
        # Simulate old __setstate__ dict
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .build()
        )
        old_state = {"__dict__": config.__dict__.copy()}
        old_dict = old_state["__dict__"]
        old_dict["capture_prior"] = "biology_informed"
        old_dict["organism"] = "human"
        old_dict["total_mrna_mean"] = 200_000
        old_dict["total_mrna_log_sigma"] = 0.5

        config2 = ModelConfig.__new__(ModelConfig)
        config2.__setstate__(old_state)
        extra = getattr(config2.priors, "__pydantic_extra__", {})
        assert "eta_capture" in extra
        assert extra["eta_capture"][0] == pytest.approx(math.log(200_000))
        assert extra.get("organism") == "human"


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
    """Test build_capture_spec with the new priors-based config."""

    def test_flat_returns_standard_spec(self):
        """
        Default (no priors) should return PositiveNormalSpec for phi_capture.
        """
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .build()
        )
        from scribe.models.builders.parameter_specs import PositiveNormalSpec
        from scribe.models.parameterizations import PARAMETERIZATIONS

        param_strategy = PARAMETERIZATIONS[config.parameterization]
        spec = build_capture_spec(
            unconstrained=True,
            guide_families=GuideFamilyConfig(),
            param_strategy=param_strategy,
            model_config=config,
        )
        assert isinstance(spec, PositiveNormalSpec)

    def test_organism_returns_bio_spec(self):
        """priors.organism should produce BiologyInformedCaptureSpec."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human")
            .build()
        )
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

    def test_shared_scaling_returns_data_driven(self):
        """shared_capture_scaling + organism should produce data_driven spec."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="mouse", shared_capture_scaling=True)
            .build()
        )
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

    def test_explicit_sigma_mu_propagates(self):
        """Explicit mu_eta sigma should propagate to spec."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(
                organism="human",
                mu_eta=(math.log(200_000), 0.5),
                shared_capture_scaling=True,
            )
            .build()
        )
        from scribe.models.parameterizations import PARAMETERIZATIONS

        param_strategy = PARAMETERIZATIONS[config.parameterization]
        spec = build_capture_spec(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            param_strategy=param_strategy,
            model_config=config,
        )
        assert isinstance(spec, BiologyInformedCaptureSpec)
        assert spec.sigma_mu == pytest.approx(0.5)


# =============================================================================
# Model dry run with biology-informed capture prior
# =============================================================================


class TestModelDryRun:
    """Test that model creation succeeds with biology-informed capture."""

    def test_nbvcp_biology_informed_mean_odds(self):
        """NBVCP with organism prior in mean_odds should create."""
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human")
            .build()
        )

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None
        assert guide_fn is not None

        bio_specs = [
            s for s in param_specs if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].use_phi_capture is True

    def test_zinbvcp_biology_informed_canonical(self):
        """ZINBVCP with organism=yeast in canonical should create."""
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("zinbvcp")
            .with_parameterization("canonical")
            .with_capture_priors(organism="yeast")
            .build()
        )

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].use_phi_capture is False
        assert bio_specs[0].log_M0 == pytest.approx(math.log(60_000))

    def test_nbvcp_shared_capture_scaling(self):
        """NBVCP with shared_capture_scaling should create data_driven spec."""
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human", shared_capture_scaling=True)
            .build()
        )

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].data_driven is True

    def test_shared_scaling_with_explicit_eta_and_mu(self):
        """Explicit eta_capture + mu_eta should propagate to spec."""
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(
                eta_capture=(11.5, 0.3),
                mu_eta=(11.5, 0.5),
                shared_capture_scaling=True,
            )
            .build()
        )

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        spec = bio_specs[0]
        assert spec.data_driven is True
        assert spec.log_M0 == pytest.approx(11.5)
        assert spec.sigma_M == pytest.approx(0.3)
        assert spec.sigma_mu == pytest.approx(0.5)

    def test_builder_with_capture_priors_method(self):
        """Test the builder's with_capture_priors method."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human")
            .build()
        )
        assert config.uses_biology_informed_capture is True
        extra = getattr(config.priors, "__pydantic_extra__", {})
        assert extra["eta_capture"][0] == pytest.approx(math.log(200_000))


# =============================================================================
# TruncatedNormal enforcement for eta_capture
# =============================================================================


class TestTruncatedNormalPrior:
    """Verify that the eta_capture prior and posterior use TruncatedNormal."""

    def test_model_prior_uses_truncated_normal(self):
        """_sample_capture_biology_informed should sample from TruncatedNormal."""
        import jax
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_capture_biology_informed,
        )

        log_lib_sizes = jnp.log(jnp.array([5_000.0, 20_000.0, 100_000.0]))

        def _model():
            _sample_capture_biology_informed(
                log_lib_sizes=log_lib_sizes,
                log_M0=math.log(200_000),
                sigma_M=0.5,
                use_phi_capture=True,
            )

        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(_model, rng_seed=0)
        ).get_trace()

        eta_site = trace["eta_capture"]
        eta_dist = eta_site["fn"]
        assert isinstance(
            eta_dist, dist.truncated.LeftTruncatedDistribution
        ), f"Expected TruncatedNormal, got {type(eta_dist)}"
        assert jnp.all(eta_site["value"] >= 0)

    def test_model_prior_samples_positive(self):
        """Draw many samples from the prior and verify all are non-negative."""
        import jax
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_capture_biology_informed,
        )

        log_lib_sizes = jnp.log(jnp.array([150_000.0]))

        def _model():
            _sample_capture_biology_informed(
                log_lib_sizes=log_lib_sizes,
                log_M0=math.log(200_000),
                sigma_M=0.5,
                use_phi_capture=False,
            )

        predictive = numpyro.infer.Predictive(_model, num_samples=1000)
        samples = predictive(jax.random.PRNGKey(42))
        eta_samples = samples["eta_capture"]
        assert jnp.all(
            eta_samples >= 0
        ), f"Found negative eta_capture: min={float(eta_samples.min())}"
        p_samples = jnp.exp(-eta_samples)
        assert jnp.all(p_samples <= 1.0)
        assert jnp.all(p_samples > 0.0)

    def test_posterior_uses_truncated_normal(self):
        """_build_biology_informed_capture_posterior returns TruncatedNormal."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        params = {
            "eta_capture_loc": jnp.array([1.0, 2.0, 0.3]),
            "eta_capture_scale": jnp.array([0.5, 0.4, 0.6]),
        }
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human")
            .build()
        )

        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        eta_dist = result["eta_capture"]
        assert isinstance(eta_dist, dist.truncated.LeftTruncatedDistribution)

        result_split = _build_biology_informed_capture_posterior(
            params, config, split=True
        )
        for d in result_split["eta_capture"]:
            assert isinstance(d, dist.truncated.LeftTruncatedDistribution)
