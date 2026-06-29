"""Tests for biology-informed capture probability prior.

Tests organism prior resolution, BiologyInformedCaptureSpec creation,
ModelConfig validation, capture_scaling_prior, TruncatedNormal enforcement,
and model dry-run with the biology-informed capture prior.

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

    def test_capture_scaling_prior_with_organism(self):
        """capture_scaling_prior='gaussian' + organism resolves mu_eta defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human", capture_scaling_prior="gaussian")
            .build()
        )
        assert config.capture_scaling_prior.value == "gaussian"
        extra = getattr(config.priors, "__pydantic_extra__", {})
        mu_eta = extra.get("mu_eta")
        assert mu_eta is not None
        # Center from eta_capture[0], sigma_mu defaults to 1.0 (anchored)
        assert mu_eta[0] == pytest.approx(math.log(200_000))
        assert mu_eta[1] == pytest.approx(1.0)

    def test_capture_scaling_prior_explicit_mu_eta(self):
        """Explicit priors.mu_eta overrides defaults."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(
                organism="human",
                mu_eta=(11.5, 0.5),
                capture_scaling_prior="gaussian",
            )
            .build()
        )
        extra = getattr(config.priors, "__pydantic_extra__", {})
        assert extra["mu_eta"] == (11.5, 0.5)

    def test_capture_scaling_prior_requires_vcp(self):
        """capture_scaling_prior with non-VCP model should raise."""
        with pytest.raises(ValueError, match="VCP"):
            (
                ModelConfigBuilder()
                .for_model("nbdm")
                .with_capture_priors(
                    organism="human", capture_scaling_prior="gaussian"
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
        """When capture_scaling_prior + anchor, sigma_mu defaults to 1.0."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(
                eta_capture=(11.5, 0.5), capture_scaling_prior="gaussian"
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
            use_phi_capture=False,
        )
        assert spec.use_phi_capture is False

    def test_data_driven_spec(self):
        """Data-driven spec with learned mu_eta (capture_scaling_prior='gaussian')."""
        spec = BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=("n_cells",),
            default_params=(math.log(200_000), 0.5),
            is_cell_specific=True,
            log_M0=math.log(200_000),
            sigma_M=0.5,
            mu_eta_prior="gaussian",
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

    def test_capture_scaling_prior_returns_data_driven(self):
        """capture_scaling_prior='gaussian' + organism should produce data_driven spec."""
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="mouse", capture_scaling_prior="gaussian")
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
                capture_scaling_prior="gaussian",
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

    def test_nbvcp_capture_scaling_prior(self):
        """NBVCP with capture_scaling_prior='gaussian' should create data_driven spec."""
        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human", capture_scaling_prior="gaussian")
            .build()
        )

        model_fn, guide_fn, param_specs = create_model(config)
        assert model_fn is not None

        bio_specs = [
            s for s in param_specs if isinstance(s, BiologyInformedCaptureSpec)
        ]
        assert len(bio_specs) == 1
        assert bio_specs[0].data_driven is True

    def test_capture_scaling_prior_with_explicit_eta_and_mu(self):
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
                capture_scaling_prior="gaussian",
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


# =============================================================================
# Hierarchical mu_eta model-side helpers
# =============================================================================


class TestHierarchicalMuEtaSamplers:
    """Test _sample_hierarchical_mu_eta_* model-side helper functions.

    Each prior type should produce per-dataset mu_eta values with the
    expected shape, finite values, and the correct set of sample sites.
    """

    N_DATASETS = 4
    LOG_M0 = math.log(200_000)
    SIGMA_MU = 1.0

    @pytest.fixture()
    def _trace_gaussian(self):
        """Trace the Gaussian hierarchical mu_eta sampler."""
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_hierarchical_mu_eta_gaussian,
        )

        def _model():
            _sample_hierarchical_mu_eta_gaussian(
                self.LOG_M0, self.SIGMA_MU, self.N_DATASETS
            )

        return numpyro.handlers.trace(
            numpyro.handlers.seed(_model, rng_seed=0)
        ).get_trace()

    @pytest.fixture()
    def _trace_horseshoe(self):
        """Trace the Horseshoe hierarchical mu_eta sampler."""
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_hierarchical_mu_eta_horseshoe,
        )

        def _model():
            _sample_hierarchical_mu_eta_horseshoe(
                self.LOG_M0, self.SIGMA_MU, self.N_DATASETS
            )

        return numpyro.handlers.trace(
            numpyro.handlers.seed(_model, rng_seed=0)
        ).get_trace()

    @pytest.fixture()
    def _trace_neg(self):
        """Trace the NEG hierarchical mu_eta sampler."""
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_hierarchical_mu_eta_neg,
        )

        def _model():
            _sample_hierarchical_mu_eta_neg(
                self.LOG_M0, self.SIGMA_MU, self.N_DATASETS
            )

        return numpyro.handlers.trace(
            numpyro.handlers.seed(_model, rng_seed=0)
        ).get_trace()

    # -- Gaussian -------------------------------------------------------------

    def test_gaussian_shape(self, _trace_gaussian):
        """Gaussian mu_eta should have shape (D,)."""
        mu_eta = _trace_gaussian["mu_eta"]["value"]
        assert mu_eta.shape == (self.N_DATASETS,)

    def test_gaussian_finite(self, _trace_gaussian):
        """All Gaussian mu_eta values should be finite."""
        mu_eta = _trace_gaussian["mu_eta"]["value"]
        assert jnp.all(jnp.isfinite(mu_eta))

    def test_gaussian_sites(self, _trace_gaussian):
        """Gaussian sampler should register mu_eta_pop, tau_eta, mu_eta_raw."""
        trace = _trace_gaussian
        assert "mu_eta_pop" in trace
        assert "tau_eta" in trace
        assert "mu_eta_raw" in trace
        assert "mu_eta" in trace

    # -- Horseshoe ------------------------------------------------------------

    def test_horseshoe_shape(self, _trace_horseshoe):
        """Horseshoe mu_eta should have shape (D,)."""
        mu_eta = _trace_horseshoe["mu_eta"]["value"]
        assert mu_eta.shape == (self.N_DATASETS,)

    def test_horseshoe_finite(self, _trace_horseshoe):
        """All Horseshoe mu_eta values should be finite."""
        mu_eta = _trace_horseshoe["mu_eta"]["value"]
        assert jnp.all(jnp.isfinite(mu_eta))

    def test_horseshoe_sites(self, _trace_horseshoe):
        """Horseshoe sampler should register the expected sample sites."""
        trace = _trace_horseshoe
        for name in (
            "mu_eta_pop",
            "tau_mu_eta",
            "lambda_mu_eta",
            "c_sq_mu_eta",
            "mu_eta_raw",
            "mu_eta",
        ):
            assert name in trace, f"Missing site: {name}"

    def test_horseshoe_lambda_shape(self, _trace_horseshoe):
        """lambda_mu_eta should be per-dataset, shape (D,)."""
        lam = _trace_horseshoe["lambda_mu_eta"]["value"]
        assert lam.shape == (self.N_DATASETS,)

    # -- NEG ------------------------------------------------------------------

    def test_neg_shape(self, _trace_neg):
        """NEG mu_eta should have shape (D,)."""
        mu_eta = _trace_neg["mu_eta"]["value"]
        assert mu_eta.shape == (self.N_DATASETS,)

    def test_neg_finite(self, _trace_neg):
        """All NEG mu_eta values should be finite."""
        mu_eta = _trace_neg["mu_eta"]["value"]
        assert jnp.all(jnp.isfinite(mu_eta))

    def test_neg_sites(self, _trace_neg):
        """NEG sampler should register zeta, psi, raw, pop, mu_eta."""
        trace = _trace_neg
        for name in (
            "mu_eta_pop",
            "zeta_mu_eta",
            "psi_mu_eta",
            "mu_eta_raw",
            "mu_eta",
        ):
            assert name in trace, f"Missing site: {name}"

    def test_neg_psi_shape(self, _trace_neg):
        """psi_mu_eta should be per-dataset, shape (D,)."""
        psi = _trace_neg["psi_mu_eta"]["value"]
        assert psi.shape == (self.N_DATASETS,)

    # -- Dispatcher -----------------------------------------------------------

    def test_dispatcher_gaussian(self):
        """_sample_hierarchical_mu_eta dispatches to Gaussian correctly."""
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_hierarchical_mu_eta,
        )

        spec = BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=(),
            default_params=(0.0, 1.0),
            log_M0=self.LOG_M0,
            sigma_M=0.5,
            mu_eta_prior="gaussian",
            sigma_mu=self.SIGMA_MU,
            use_phi_capture=True,
        )

        def _model():
            return _sample_hierarchical_mu_eta(spec, self.N_DATASETS)

        trace = numpyro.handlers.trace(
            numpyro.handlers.seed(_model, rng_seed=0)
        ).get_trace()
        assert "tau_eta" in trace
        assert trace["mu_eta"]["value"].shape == (self.N_DATASETS,)

    def test_dispatcher_unknown_raises(self):
        """_sample_hierarchical_mu_eta raises for unknown prior types."""
        import numpyro

        from scribe.models.components.likelihoods.base import (
            _sample_hierarchical_mu_eta,
        )

        spec = BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=(),
            default_params=(0.0, 1.0),
            log_M0=self.LOG_M0,
            sigma_M=0.5,
            mu_eta_prior="unknown_type",
            sigma_mu=self.SIGMA_MU,
            use_phi_capture=True,
        )
        with pytest.raises(ValueError, match="Unknown mu_eta_prior"):
            with numpyro.handlers.seed(rng_seed=0):
                _sample_hierarchical_mu_eta(spec, self.N_DATASETS)


# =============================================================================
# Hierarchical mu_eta guide-side helper
# =============================================================================


class TestGuideMuEtaHierarchy:
    """Test guide_mu_eta_hierarchy variational parameter registration.

    Verifies that the guide helper creates the correct numpyro params
    and sample sites for each prior type, and that the single-dataset
    fallback works.
    """

    N_DATASETS = 4
    LOG_M0 = math.log(200_000)

    def _make_spec(self, prior_type):
        return BiologyInformedCaptureSpec(
            name="phi_capture",
            shape_dims=(),
            default_params=(0.0, 1.0),
            log_M0=self.LOG_M0,
            sigma_M=0.5,
            mu_eta_prior=prior_type,
            sigma_mu=1.0,
            use_phi_capture=True,
        )

    def _trace_guide(self, prior_type, n_datasets=None):
        """Run guide_mu_eta_hierarchy under trace and return the trace."""
        import numpyro

        from scribe.models.builders._guide_cell_specific_mixin import (
            guide_mu_eta_hierarchy,
        )

        n_ds = n_datasets if n_datasets is not None else self.N_DATASETS
        spec = self._make_spec(prior_type)

        def _guide():
            guide_mu_eta_hierarchy(spec, n_ds)

        return numpyro.handlers.trace(
            numpyro.handlers.seed(_guide, rng_seed=0)
        ).get_trace()

    def test_gaussian_guide_sites(self):
        """Gaussian guide should register pop, tau_eta, and raw sites."""
        trace = self._trace_guide("gaussian")
        assert "mu_eta_pop" in trace
        assert "tau_eta" in trace
        assert "mu_eta_raw" in trace

    def test_horseshoe_guide_sites(self):
        """Horseshoe guide should register pop, tau, lambda, c_sq, and raw."""
        trace = self._trace_guide("horseshoe")
        for name in (
            "mu_eta_pop",
            "tau_mu_eta",
            "lambda_mu_eta",
            "c_sq_mu_eta",
            "mu_eta_raw",
        ):
            assert name in trace, f"Missing guide site: {name}"

    def test_neg_guide_sites(self):
        """NEG guide should register pop, zeta, psi, and raw."""
        trace = self._trace_guide("neg")
        for name in (
            "mu_eta_pop",
            "zeta_mu_eta",
            "psi_mu_eta",
            "mu_eta_raw",
        ):
            assert name in trace, f"Missing guide site: {name}"

    def test_single_dataset_fallback(self):
        """n_datasets=1 should use scalar mu_eta instead of hierarchy."""
        trace = self._trace_guide("gaussian", n_datasets=1)
        assert "mu_eta" in trace
        # Hierarchy sites should NOT be present
        assert "mu_eta_pop" not in trace
        assert "tau_eta" not in trace
        assert "mu_eta_raw" not in trace

    def test_single_dataset_scalar_shape(self):
        """Single-dataset mu_eta should be a scalar."""
        trace = self._trace_guide("gaussian", n_datasets=1)
        mu_eta_val = trace["mu_eta"]["value"]
        assert mu_eta_val.ndim == 0

    def test_guide_raw_shape(self):
        """mu_eta_raw guide should have shape (D,)."""
        trace = self._trace_guide("gaussian")
        raw = trace["mu_eta_raw"]["value"]
        assert raw.shape == (self.N_DATASETS,)


# =============================================================================
# Posterior extraction for hierarchical mu_eta
# =============================================================================


class TestHierarchicalMuEtaPosterior:
    """Test _build_biology_informed_capture_posterior with hierarchical params."""

    def _make_config(self, capture_scaling_prior="gaussian"):
        return (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human", capture_scaling_prior=capture_scaling_prior)
            .build()
        )

    def test_gaussian_posterior_sites(self):
        """Gaussian hierarchical posterior should extract pop, tau, raw."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        params = {
            "mu_eta_pop_loc": jnp.array(11.5),
            "mu_eta_pop_scale": jnp.array(0.1),
            "tau_eta_loc": jnp.array(-2.0),
            "tau_eta_scale": jnp.array(0.1),
            "mu_eta_raw_loc": jnp.array([0.0, 0.1, -0.1, 0.05]),
            "mu_eta_raw_scale": jnp.array([0.1, 0.1, 0.1, 0.1]),
            "eta_capture_loc": jnp.array([1.0, 2.0]),
            "eta_capture_scale": jnp.array([0.5, 0.5]),
        }
        config = self._make_config("gaussian")
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        assert "mu_eta_pop" in result
        assert "tau_eta" in result
        assert "mu_eta_raw" in result
        assert isinstance(result["mu_eta_pop"], dist.Normal)

    def test_horseshoe_posterior_sites(self):
        """Horseshoe hierarchical posterior should extract all auxiliary sites."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        n_ds = 3
        params = {
            "mu_eta_pop_loc": jnp.array(11.5),
            "mu_eta_pop_scale": jnp.array(0.1),
            "tau_mu_eta_loc": jnp.array(0.1),
            "tau_mu_eta_scale": jnp.array(0.1),
            "lambda_mu_eta_loc": jnp.ones(n_ds) * 0.1,
            "lambda_mu_eta_scale": jnp.ones(n_ds) * 0.1,
            "c_sq_mu_eta_loc": jnp.array(2.0),
            "c_sq_mu_eta_scale": jnp.array(0.1),
            "mu_eta_raw_loc": jnp.zeros(n_ds),
            "mu_eta_raw_scale": jnp.ones(n_ds) * 0.1,
            "eta_capture_loc": jnp.array([1.0, 2.0]),
            "eta_capture_scale": jnp.array([0.5, 0.5]),
        }
        config = self._make_config("horseshoe")
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        for name in (
            "mu_eta_pop",
            "tau_mu_eta",
            "lambda_mu_eta",
            "c_sq_mu_eta",
            "mu_eta_raw",
        ):
            assert name in result, f"Missing posterior site: {name}"

    def test_neg_posterior_sites(self):
        """NEG hierarchical posterior should extract zeta, psi, raw."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        n_ds = 3
        params = {
            "mu_eta_pop_loc": jnp.array(11.5),
            "mu_eta_pop_scale": jnp.array(0.1),
            "zeta_mu_eta_loc": jnp.ones(n_ds),
            "zeta_mu_eta_scale": jnp.ones(n_ds) * 0.1,
            "psi_mu_eta_loc": jnp.ones(n_ds),
            "psi_mu_eta_scale": jnp.ones(n_ds) * 0.1,
            "mu_eta_raw_loc": jnp.zeros(n_ds),
            "mu_eta_raw_scale": jnp.ones(n_ds) * 0.1,
            "eta_capture_loc": jnp.array([1.0, 2.0]),
            "eta_capture_scale": jnp.array([0.5, 0.5]),
        }
        config = self._make_config("neg")
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        for name in ("mu_eta_pop", "zeta_mu_eta", "psi_mu_eta", "mu_eta_raw"):
            assert name in result, f"Missing posterior site: {name}"

    def test_scalar_mu_eta_fallback(self):
        """Old/single-dataset params with mu_eta_loc should still work."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        params = {
            "mu_eta_loc": jnp.array(11.5),
            "mu_eta_scale": jnp.array(0.1),
            "eta_capture_loc": jnp.array([1.0, 2.0]),
            "eta_capture_scale": jnp.array([0.5, 0.5]),
        }
        config = self._make_config("gaussian")
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        assert "mu_eta" in result
        assert isinstance(result["mu_eta"], dist.Normal)
        # Hierarchy sites should NOT be present
        assert "mu_eta_pop" not in result

    def test_eta_capture_always_present(self):
        """eta_capture posterior should always be present when params exist."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        params = {
            "mu_eta_pop_loc": jnp.array(11.5),
            "mu_eta_pop_scale": jnp.array(0.1),
            "tau_eta_loc": jnp.array(-2.0),
            "tau_eta_scale": jnp.array(0.1),
            "mu_eta_raw_loc": jnp.zeros(3),
            "mu_eta_raw_scale": jnp.ones(3) * 0.1,
            "eta_capture_loc": jnp.array([1.0, 2.0, 0.3]),
            "eta_capture_scale": jnp.array([0.5, 0.4, 0.6]),
        }
        config = self._make_config("gaussian")
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        assert "eta_capture" in result
        assert isinstance(
            result["eta_capture"],
            dist.truncated.LeftTruncatedDistribution,
        )


# =============================================================================
# Stale shared_capture_scaling references
# =============================================================================


class TestSharedCaptureScalingRemoved:
    """Ensure shared_capture_scaling is fully removed from the public API."""

    def test_model_config_no_shared_capture_field(self):
        """ModelConfig should not have a shared_capture_scaling field."""
        assert not hasattr(ModelConfig, "shared_capture_scaling")
        config = ModelConfig(base_model="nbdm")
        assert not hasattr(config, "shared_capture_scaling")

    def test_model_config_builder_no_shared_capture(self):
        """ModelConfigBuilder.with_capture_priors should reject the kwarg."""
        with pytest.raises(TypeError):
            (
                ModelConfigBuilder()
                .for_model("nbvcp")
                .with_parameterization("mean_odds")
                .with_capture_priors(
                    organism="human", shared_capture_scaling=True
                )
                .build()
            )

    def test_pickle_compat_migrates_shared_capture(self):
        """Old pickles with shared_capture_scaling=True migrate to capture_scaling_prior."""
        import pickle

        config = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            capture_scaling_prior="none",
        )
        state = config.__getstate__()
        state["__dict__"]["shared_capture_scaling"] = True
        state["__dict__"].pop("capture_scaling_prior", None)
        restored = pickle.loads(pickle.dumps(config))
        restored.__setstate__(state)
        # After __setstate__ the field is a raw string (not yet re-validated);
        # it migrates to "gaussian" which is the enum's value.
        mu_eta_val = restored.__dict__.get("capture_scaling_prior", None)
        if hasattr(mu_eta_val, "value"):
            assert mu_eta_val.value == "gaussian"
        else:
            assert str(mu_eta_val) == "gaussian"


# =============================================================================
# Softplus-normal guide for eta_capture
# =============================================================================


class TestSoftplusNormalGuide:
    """Tests for the softplus-normal variational guide on eta_capture.

    The softplus-normal guide samples an unconstrained Normal (params
    ``eta_capture_raw_loc`` / ``eta_capture_raw_scale``) and maps
    through softplus to get ``eta_capture``. This induces a logit-normal
    on ``nu_c``, with smooth gradients and no truncation boundary.
    """

    def test_softplus_normal_guide_sites(self):
        """Guide should register eta_capture_raw_loc/scale params."""
        import numpyro

        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human")
            .build()
        )
        # Default eta_capture_guide is "softplus_normal"
        assert config.eta_capture_guide == "softplus_normal"

        _model_fn, guide_fn, _specs = create_model(config)

        n_cells, n_genes = 5, 10
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.int32)
        guide_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as guide_tr:
                guide_fn(**guide_kwargs)

        # Softplus-normal path should register raw params
        param_names = {
            k for k, v in guide_tr.items() if v.get("type") == "param"
        }
        assert "eta_capture_raw_loc" in param_names
        assert "eta_capture_raw_scale" in param_names
        # Legacy params should NOT be present
        assert "eta_capture_loc" not in param_names
        assert "eta_capture_scale" not in param_names

    def test_softplus_normal_guide_eta_positive(self):
        """eta_capture sampled via softplus should always be positive."""
        import jax
        import numpyro

        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human")
            .build()
        )

        _model_fn, guide_fn, _specs = create_model(config)

        n_cells, n_genes = 20, 10
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.int32)
        guide_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        # Draw many samples and verify all positive
        predictive = numpyro.infer.Predictive(guide_fn, num_samples=200)
        samples = predictive(jax.random.PRNGKey(42), **guide_kwargs)

        assert "eta_capture" in samples
        eta_vals = samples["eta_capture"]
        assert jnp.all(
            eta_vals > 0
        ), f"Found non-positive eta: min={float(eta_vals.min())}"

    def test_softplus_normal_posterior_reconstruction(self):
        """Posterior with eta_capture_raw_loc should use TransformedDist."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        # Softplus-normal params (new path)
        params = {
            "eta_capture_raw_loc": jnp.array([1.0, 2.0, 0.3]),
            "eta_capture_raw_scale": jnp.array([0.5, 0.4, 0.6]),
        }
        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .with_capture_priors(organism="human")
            .build()
        )

        # Non-split
        result = _build_biology_informed_capture_posterior(
            params, config, split=False
        )
        eta_dist = result["eta_capture"]
        assert isinstance(eta_dist, dist.TransformedDistribution)
        assert isinstance(
            eta_dist.transforms[-1], dist.transforms.SoftplusTransform
        )

        # Split per-cell
        result_split = _build_biology_informed_capture_posterior(
            params, config, split=True
        )
        for d in result_split["eta_capture"]:
            assert isinstance(d, dist.TransformedDistribution)

    def test_backward_compat_eta_capture_guide_defaults_truncated(self):
        """Old pickles without eta_capture_guide default to truncated_normal."""
        config = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
        )
        state = config.__getstate__()
        # Remove the new field to simulate an old pickle
        state["__dict__"].pop("eta_capture_guide", None)
        restored = ModelConfig.__new__(ModelConfig)
        restored.__setstate__(state)

        guide_val = restored.__dict__.get("eta_capture_guide", None)
        if hasattr(guide_val, "value"):
            assert guide_val.value == "truncated_normal"
        else:
            assert str(guide_val) == "truncated_normal"

    def test_truncated_normal_guide_still_works(self):
        """Explicit eta_capture_guide='truncated_normal' uses old path."""
        import numpyro

        from scribe.models.presets.factory import create_model

        config = (
            ModelConfigBuilder()
            .for_model("nbvcp")
            .with_parameterization("mean_odds")
            .unconstrained()
            .with_capture_priors(organism="human")
            .build()
        )
        # Override to legacy guide
        config = config.model_copy(
            update={"eta_capture_guide": "truncated_normal"}
        )
        assert config.eta_capture_guide == "truncated_normal"

        _model_fn, guide_fn, _specs = create_model(config)

        n_cells, n_genes = 5, 10
        counts = jnp.ones((n_cells, n_genes), dtype=jnp.int32)
        guide_kwargs = dict(
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as guide_tr:
                guide_fn(**guide_kwargs)

        param_names = {
            k for k, v in guide_tr.items() if v.get("type") == "param"
        }
        # Legacy path should use eta_capture_loc/scale
        assert "eta_capture_loc" in param_names
        assert "eta_capture_scale" in param_names
        # New params should NOT be present
        assert "eta_capture_raw_loc" not in param_names

    def test_legacy_posterior_still_reconstructs_truncated_normal(self):
        """Posterior with eta_capture_loc (no _raw) still uses TruncatedNormal."""
        from scribe.models.builders.posterior import (
            _build_biology_informed_capture_posterior,
        )

        params = {
            "eta_capture_loc": jnp.array([1.0, 2.0]),
            "eta_capture_scale": jnp.array([0.5, 0.4]),
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

    def test_invalid_eta_capture_guide_raises(self):
        """Invalid eta_capture_guide value should raise in validate_config."""
        with pytest.raises(ValueError, match="eta_capture_guide"):
            config = ModelConfig(
                base_model="nbvcp",
                parameterization="mean_odds",
                eta_capture_guide="invalid_value",
            )
            config.validate_config()
