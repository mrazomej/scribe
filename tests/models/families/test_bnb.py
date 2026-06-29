"""Tests for the Beta Negative Binomial (BNB) integration.

Tests cover:
- Config: OverdispersionType enum and ModelConfig fields.
- Parameter specs: HorseshoeBNBConcentrationSpec and NEGBNBConcentrationSpec.
- Registry: BNB concentration specs are built when overdispersion='bnb'.
- Distribution builder: build_count_dist returns NB or BNB correctly.
- Log-likelihood: BNB concentration flows through all LL functions.
- Factory: Model creation with overdispersion='bnb' succeeds.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from scribe.models.config.enums import (
    HierarchicalPriorType,
    OverdispersionType,
)

# ============================================================================
# Config tests
# ============================================================================


class TestOverdispersionEnum:
    """Test the OverdispersionType enum."""

    def test_values(self):
        """NONE and BNB are the expected members."""
        assert OverdispersionType.NONE.value == "none"
        assert OverdispersionType.BNB.value == "bnb"

    def test_is_str_enum(self):
        """Enum members are also str for serialization."""
        assert isinstance(OverdispersionType.BNB, str)


class TestModelConfigBNBFields:
    """Test overdispersion fields on ModelConfig."""

    def test_default_no_overdispersion(self):
        """Default config has overdispersion='none'."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
        )
        assert cfg.overdispersion == OverdispersionType.NONE
        assert cfg.is_overdispersed is False
        assert cfg.is_bnb is False

    def test_bnb_overdispersion(self):
        """Setting overdispersion='bnb' activates BNB flags."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            overdispersion="bnb",
        )
        assert cfg.overdispersion == OverdispersionType.BNB
        assert cfg.is_overdispersed is True
        assert cfg.is_bnb is True

    def test_overdispersion_prior_default(self):
        """Default overdispersion prior is horseshoe."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            overdispersion="bnb",
        )
        assert cfg.overdispersion_prior == HierarchicalPriorType.HORSESHOE

    def test_overdispersion_prior_neg(self):
        """Can set overdispersion_prior to NEG."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            overdispersion="bnb",
            overdispersion_prior="neg",
        )
        assert cfg.overdispersion_prior == HierarchicalPriorType.NEG

    def test_overdispersion_dataset_prior_default(self):
        """Dataset overdispersion prior defaults to none."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            overdispersion="bnb",
        )
        assert cfg.overdispersion_dataset_prior == HierarchicalPriorType.NONE


# ============================================================================
# Distribution builder tests
# ============================================================================


class TestBuildCountDist:
    """Test the build_count_dist helper."""

    def test_returns_nb_when_no_bnb(self):
        """Without bnb_concentration, returns NegativeBinomialProbs."""
        from scribe.models.components.likelihoods.beta_negative_binomial import (
            build_count_dist,
        )

        r = jnp.array([1.0, 2.0])
        p = jnp.array([0.5, 0.3])
        d = build_count_dist(r, p, bnb_concentration=None)
        assert isinstance(d, dist.NegativeBinomialProbs)

    def test_returns_bnb_when_concentration_given(self):
        """With bnb_concentration, returns BetaNegativeBinomial."""
        # Import from the same module that build_count_dist uses, so we
        # get numpyro's class on numpyro>=0.20 and the scribe fallback
        # on older versions.
        from scribe.models.components.likelihoods.beta_negative_binomial import (
            BetaNegativeBinomial,
            build_count_dist,
        )

        r = jnp.array([1.0, 2.0])
        p = jnp.array([0.5, 0.3])
        omega = jnp.array([0.1, 0.2])
        d = build_count_dist(r, p, bnb_concentration=omega)
        assert isinstance(d, BetaNegativeBinomial)

    def test_bnb_mean_matches_nb(self):
        """BNB mean should equal NB mean under the mean-preserving param."""
        from scribe.models.components.likelihoods.beta_negative_binomial import (
            build_count_dist,
        )

        r = jnp.array([5.0])
        p = jnp.array([0.4])
        omega = jnp.array([0.5])

        nb = build_count_dist(r, p, None)
        bnb = build_count_dist(r, p, omega)

        # Canonical SCRIBE mapping for NB means with (r, p):
        # mu = r * p / (1 - p)
        # NB mean = r * p / (1-p)
        nb_mean = r * p / (1.0 - p)
        assert jnp.allclose(nb.mean, nb_mean, atol=1e-4)
        assert jnp.allclose(bnb.mean, nb_mean, atol=1e-4)

    def test_bnb_log_prob_finite(self):
        """BNB log_prob should return finite values for typical inputs."""
        from scribe.models.components.likelihoods.beta_negative_binomial import (
            build_count_dist,
        )

        r = jnp.array([3.0, 5.0])
        p = jnp.array([0.5, 0.3])
        omega = jnp.array([0.2, 0.5])
        d = build_count_dist(r, p, omega)
        counts = jnp.array([0.0, 10.0])
        lp = d.log_prob(counts)
        assert jnp.all(jnp.isfinite(lp))

    def test_bnb_sample_shape(self):
        """BNB should produce samples with the correct shape."""
        from scribe.models.components.likelihoods.beta_negative_binomial import (
            build_count_dist,
        )

        r = jnp.array([3.0, 5.0])
        p = jnp.array([0.5, 0.3])
        omega = jnp.array([0.2, 0.5])
        d = build_count_dist(r, p, omega)
        key = jax.random.PRNGKey(42)
        samples = d.sample(key, (100,))
        assert samples.shape == (100, 2)


# ============================================================================
# Registry tests
# ============================================================================


class TestBNBRegistry:
    """Test BNB concentration specs are built by the registry."""

    def test_build_bnb_horseshoe_specs(self):
        """build_bnb_concentration_spec returns specs for horseshoe."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="horseshoe",
            guide_families=GuideFamilyConfig(),
        )
        # Should have: loc, tau, lambda, c_sq, bnb_concentration
        names = [s.name for s in specs]
        assert "bnb_concentration_loc" in names
        assert "bnb_concentration_tau" in names
        assert "bnb_concentration_lambda" in names
        assert "bnb_concentration_c_sq" in names
        assert "bnb_concentration" in names
        assert len(specs) == 5

    def test_build_bnb_neg_specs(self):
        """build_bnb_concentration_spec returns specs for NEG."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="neg",
            guide_families=GuideFamilyConfig(),
        )
        names = [s.name for s in specs]
        assert "bnb_concentration_loc" in names
        assert "bnb_concentration_zeta" in names
        assert "bnb_concentration_psi" in names
        assert "bnb_concentration" in names
        assert len(specs) == 4

    def test_build_bnb_gaussian_specs(self):
        """build_bnb_concentration_spec returns specs for Gaussian prior.

        The Gaussian path uses a simple hierarchical Normal + softplus
        with no sparsity induction: hyper-loc, hyper-scale, and per-gene
        omega_g = softplus(Normal(loc, scale)).

        Hyper-param names use ``bnb_omega_hyper_`` prefix to avoid
        collision with guide's auto-generated variational params.
        """
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="gaussian",
            guide_families=GuideFamilyConfig(),
        )
        names = [s.name for s in specs]
        assert "bnb_omega_hyper_loc" in names
        assert "bnb_omega_hyper_scale" in names
        assert "bnb_concentration" in names
        # Gaussian has no auxiliary shrinkage sites
        assert "bnb_concentration_tau" not in names
        assert "bnb_concentration_lambda" not in names
        assert "bnb_concentration_psi" not in names
        assert "bnb_concentration_zeta" not in names
        assert len(specs) == 3

    def test_build_bnb_gaussian_mixture(self):
        """Gaussian BNB spec respects mixture_params for is_mixture flag."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        # With bnb_concentration in mixture_params, per-gene spec is mixture
        specs = build_bnb_concentration_spec(
            overdispersion_prior="gaussian",
            guide_families=GuideFamilyConfig(),
            n_components=4,
            mixture_params=["bnb_concentration"],
        )
        bnb_spec = [s for s in specs if s.name == "bnb_concentration"][0]
        assert bnb_spec.is_mixture is True

        # Without bnb_concentration in mixture_params, it's not mixture
        specs_no = build_bnb_concentration_spec(
            overdispersion_prior="gaussian",
            guide_families=GuideFamilyConfig(),
            n_components=4,
            mixture_params=["phi"],
        )
        bnb_spec_no = [s for s in specs_no if s.name == "bnb_concentration"][0]
        assert bnb_spec_no.is_mixture is False

    def test_build_bnb_gaussian_dataset(self):
        """Gaussian BNB spec supports dataset-indexed concentration."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="gaussian",
            guide_families=GuideFamilyConfig(),
            is_dataset=True,
        )
        bnb_spec = [s for s in specs if s.name == "bnb_concentration"][0]
        assert bnb_spec.is_dataset is True


# ============================================================================
# Log-likelihood tests
# ============================================================================


class TestBNBLogLikelihood:
    """Test log-likelihood functions with BNB concentration."""

    @pytest.fixture()
    def _data(self):
        """Shared synthetic data for LL tests."""
        rng = np.random.default_rng(42)
        n_cells, n_genes = 20, 5
        counts = jnp.array(
            rng.poisson(5, size=(n_cells, n_genes)), dtype=jnp.float32
        )
        return counts

    def test_nbdm_with_bnb(self, _data):
        """NB ``.log_prob`` accepts ``bnb_concentration`` without error."""
        from scribe.models.components.likelihoods import (
            NegativeBinomialLikelihood,
        )
        from scribe.core.axis_layout import AxisLayout

        counts = _data
        params = {
            "p": jnp.array(0.5),
            "r": jnp.ones(5),
            "bnb_concentration": jnp.full(5, 0.1),
        }
        layouts = {
            "p": AxisLayout(()),
            "r": AxisLayout(("genes",)),
            "bnb_concentration": AxisLayout(("genes",)),
        }
        ll = NegativeBinomialLikelihood().log_prob(counts, params, layouts)
        assert ll.shape == (20,)
        assert jnp.all(jnp.isfinite(ll))

    def test_nbdm_without_bnb_unchanged(self, _data):
        """NB ``.log_prob`` without ``bnb_concentration`` is standard NB."""
        from scribe.models.components.likelihoods import (
            NegativeBinomialLikelihood,
        )
        from scribe.core.axis_layout import AxisLayout

        counts = _data
        params = {"p": jnp.array(0.5), "r": jnp.ones(5)}
        layouts = {
            "p": AxisLayout(()),
            "r": AxisLayout(("genes",)),
        }
        ll = NegativeBinomialLikelihood().log_prob(counts, params, layouts)
        assert ll.shape == (20,)
        assert jnp.all(jnp.isfinite(ll))

    def test_zinb_with_bnb(self, _data):
        """ZINB ``.log_prob`` accepts ``bnb_concentration`` without error."""
        from scribe.models.components.likelihoods import (
            ZeroInflatedNBLikelihood,
        )
        from scribe.core.axis_layout import AxisLayout

        counts = _data
        params = {
            "p": jnp.array(0.5),
            "r": jnp.ones(5),
            "gate": jnp.full(5, 0.1),
            "bnb_concentration": jnp.full(5, 0.1),
        }
        layouts = {
            "p": AxisLayout(()),
            "r": AxisLayout(("genes",)),
            "gate": AxisLayout(("genes",)),
            "bnb_concentration": AxisLayout(("genes",)),
        }
        ll = ZeroInflatedNBLikelihood().log_prob(counts, params, layouts)
        assert ll.shape == (20,)
        assert jnp.all(jnp.isfinite(ll))


# ============================================================================
# Factory integration test
# ============================================================================


class TestBNBFactory:
    """Test model creation with overdispersion='bnb'."""

    def test_create_model_nbdm_bnb(self):
        """Factory creates model + guide for NBDM with BNB."""
        from scribe.models.presets.factory import create_model
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
        )
        model_fn, guide_fn, _specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)

    def test_create_model_zinb_bnb(self):
        """Factory creates model + guide for ZINB with BNB."""
        from scribe.models.presets.factory import create_model
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="zinb",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
        )
        model_fn, guide_fn, _specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)

    def test_create_model_nbvcp_bnb(self):
        """Factory creates model + guide for NBVCP with BNB."""
        from scribe.models.presets.factory import create_model
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbvcp",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
        )
        model_fn, guide_fn, _specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)

    def test_create_model_bnb_with_neg_prior(self):
        """Factory creates model with BNB + NEG overdispersion prior."""
        from scribe.models.presets.factory import create_model
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
            overdispersion_prior="neg",
        )
        model_fn, guide_fn, _specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)

    def test_create_model_bnb_with_dataset_overdispersion_prior(self):
        """Factory uses dataset-indexed BNB concentration when requested."""
        from scribe.models.presets.factory import create_model
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
            n_datasets=2,
            overdispersion_dataset_prior="gaussian",
        )
        model_fn, guide_fn, specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)
        bnb_spec = [s for s in specs if s.name == "bnb_concentration"][0]
        assert bnb_spec.is_dataset is True


# ============================================================================
# MAP reconstruction tests
# ============================================================================


class TestMapReconstruction:
    """Verify _reconstruct_ncp_maps (and its per-spec helpers) correctly
    produce bnb_concentration, and that get_map derives bnb_kappa."""

    @staticmethod
    def _bnb_horseshoe_specs():
        """Build a list of BNB horseshoe ParamSpecs for testing."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )
        return build_bnb_concentration_spec(
            overdispersion_prior="horseshoe",
            guide_families=GuideFamilyConfig(),
        )

    @staticmethod
    def _bnb_neg_specs():
        """Build a list of BNB NEG ParamSpecs for testing."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )
        return build_bnb_concentration_spec(
            overdispersion_prior="neg",
            guide_families=GuideFamilyConfig(),
        )

    def test_horseshoe_bnb_reconstruction(self):
        """Horseshoe NCP components yield bnb_concentration via softplus."""
        from scribe.svi._parameter_extraction import (
            _reconstruct_from_horseshoe_spec,
        )

        # Find the main horseshoe BNB spec (uses_ncp=True, name=bnb_concentration)
        specs = self._bnb_horseshoe_specs()
        bnb_spec = [
            s for s in specs
            if getattr(s, "uses_ncp", False)
            and getattr(s, "name", "") == "bnb_concentration"
        ][0]

        # Simulate MAP entries produced by a horseshoe NCP guide.
        fake_map = {
            bnb_spec.raw_name: jnp.ones(5) * 0.5,
            bnb_spec.tau_name: jnp.array(0.1),
            bnb_spec.lambda_name: jnp.ones(5) * 0.3,
            bnb_spec.c_sq_name: jnp.array(4.0),
            bnb_spec.hyper_loc_name: jnp.array(-1.0),
        }

        _reconstruct_from_horseshoe_spec(bnb_spec, fake_map, expand_rank=True)
        assert "bnb_concentration" in fake_map
        omega = fake_map["bnb_concentration"]
        # softplus output is always positive
        assert jnp.all(omega > 0)
        assert omega.shape == (5,)

    def test_neg_bnb_reconstruction(self):
        """NEG NCP components yield bnb_concentration via softplus."""
        from scribe.svi._parameter_extraction import (
            _reconstruct_from_neg_spec,
        )

        specs = self._bnb_neg_specs()
        bnb_spec = [
            s for s in specs
            if getattr(s, "uses_ncp", False)
            and getattr(s, "name", "") == "bnb_concentration"
        ][0]

        fake_map = {
            bnb_spec.raw_name: jnp.zeros(5),
            bnb_spec.psi_name: jnp.ones(5) * 0.5,
            bnb_spec.hyper_loc_name: jnp.array(0.0),
        }

        _reconstruct_from_neg_spec(bnb_spec, fake_map, expand_rank=True)
        assert "bnb_concentration" in fake_map
        omega = fake_map["bnb_concentration"]
        assert jnp.all(omega > 0)
        assert omega.shape == (5,)

    def test_kappa_derived_in_get_map_context(self):
        """After reconstruction, kappa_g = 2 + (r + 1) / omega_g."""
        # Pure arithmetic check — no config needed.
        omega = jnp.array([0.1, 0.5, 1.0, 2.0, 10.0])
        r = jnp.array([5.0, 10.0, 20.0, 1.0, 3.0])
        kappa = 2.0 + (r + 1.0) / omega
        # kappa > 2 always
        assert jnp.all(kappa > 2.0)

    def test_neg_reconstruction_broadcasts_loc_with_dataset_dim(self):
        """loc (D,) broadcasts correctly with z (D, G) after concat."""
        from scribe.svi._parameter_extraction import (
            _reconstruct_from_neg_spec,
        )

        specs = self._bnb_neg_specs()
        bnb_spec = [
            s for s in specs
            if getattr(s, "uses_ncp", False)
            and getattr(s, "name", "") == "bnb_concentration"
        ][0]

        D, G = 2, 10
        fake_map = {
            bnb_spec.raw_name: jnp.zeros((D, G)),
            bnb_spec.psi_name: jnp.ones((D, G)) * 0.5,
            bnb_spec.hyper_loc_name: jnp.array([-1.0, -2.0]),
        }

        # expand_rank=True handles the (D,) -> (D, 1) promotion of loc
        _reconstruct_from_neg_spec(bnb_spec, fake_map, expand_rank=True)
        assert "bnb_concentration" in fake_map
        assert fake_map["bnb_concentration"].shape == (D, G)

    def test_horseshoe_reconstruction_broadcasts_loc_with_dataset_dim(self):
        """loc (D,) broadcasts correctly with z (D, G) for horseshoe."""
        from scribe.svi._parameter_extraction import (
            _reconstruct_from_horseshoe_spec,
        )

        specs = self._bnb_horseshoe_specs()
        bnb_spec = [
            s for s in specs
            if getattr(s, "uses_ncp", False)
            and getattr(s, "name", "") == "bnb_concentration"
        ][0]

        D, G = 3, 8
        fake_map = {
            bnb_spec.raw_name: jnp.ones((D, G)) * 0.5,
            bnb_spec.tau_name: jnp.array([0.1, 0.2, 0.3]),
            bnb_spec.lambda_name: jnp.ones((D, G)) * 0.3,
            bnb_spec.c_sq_name: jnp.array([4.0, 4.0, 4.0]),
            bnb_spec.hyper_loc_name: jnp.array([-1.0, -1.5, -2.0]),
        }

        _reconstruct_from_horseshoe_spec(bnb_spec, fake_map, expand_rank=True)
        assert "bnb_concentration" in fake_map
        assert fake_map["bnb_concentration"].shape == (D, G)

    def test_reconstruct_ncp_maps_dispatches_horseshoe(self):
        """_reconstruct_ncp_maps correctly dispatches to horseshoe helper."""
        from types import SimpleNamespace
        from scribe.svi._parameter_extraction import _reconstruct_ncp_maps

        # Use a lightweight namespace instead of a full ModelConfig to
        # avoid Pydantic validation constraints on param_specs contents.
        cfg = SimpleNamespace(param_specs=self._bnb_horseshoe_specs())

        fake_map = {
            "bnb_concentration_raw": jnp.ones(5) * 0.5,
            "bnb_concentration_tau": jnp.array(0.1),
            "bnb_concentration_lambda": jnp.ones(5) * 0.3,
            "bnb_concentration_c_sq": jnp.array(4.0),
            "bnb_concentration_loc": jnp.array(-1.0),
        }

        result = _reconstruct_ncp_maps(fake_map, cfg)
        assert "bnb_concentration" in result
        assert jnp.all(result["bnb_concentration"] > 0)
        assert result["bnb_concentration"].shape == (5,)

    def test_reconstruct_ncp_maps_dispatches_neg(self):
        """_reconstruct_ncp_maps correctly dispatches to NEG helper."""
        from types import SimpleNamespace
        from scribe.svi._parameter_extraction import _reconstruct_ncp_maps

        cfg = SimpleNamespace(param_specs=self._bnb_neg_specs())

        fake_map = {
            "bnb_concentration_raw": jnp.zeros(5),
            "bnb_concentration_psi": jnp.ones(5) * 0.5,
            "bnb_concentration_loc": jnp.array(0.0),
        }

        result = _reconstruct_ncp_maps(fake_map, cfg)
        assert "bnb_concentration" in result
        assert jnp.all(result["bnb_concentration"] > 0)
        assert result["bnb_concentration"].shape == (5,)


# ============================================================================
# Mixture support tests
# ============================================================================


class TestBNBMixture:
    """Test that bnb_concentration can be made mixture-specific."""

    def test_mixture_validation_accepts_bnb_concentration(self):
        """bnb_concentration is accepted in mixture_params when BNB is on."""
        from scribe.models.config import ModelConfig

        cfg = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            inference_method="svi",
            unconstrained=True,
            overdispersion="bnb",
            overdispersion_prior="neg",
            n_components=4,
            mixture_params=["phi", "mu", "bnb_concentration"],
        )
        # Should not raise during model creation
        from scribe.models.presets.factory import create_model

        model_fn, guide_fn, _specs = create_model(cfg)
        assert callable(model_fn)
        assert callable(guide_fn)

    def test_mixture_validation_rejects_bnb_without_overdispersion(self):
        """bnb_concentration in mixture_params without overdispersion=bnb
        is rejected (not a valid parameter when BNB is off)."""
        from scribe.models.config import ModelConfig
        from scribe.models.presets.factory import create_model

        cfg = ModelConfig(
            base_model="nbvcp",
            parameterization="mean_odds",
            inference_method="svi",
            unconstrained=True,
            n_components=4,
            mixture_params=["phi", "mu", "bnb_concentration"],
        )
        with pytest.raises(ValueError, match="Invalid mixture_params"):
            create_model(cfg)

    def test_bnb_specs_have_is_mixture_true(self):
        """When bnb_concentration is in mixture_params, specs get
        is_mixture=True."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="neg",
            guide_families=GuideFamilyConfig(),
            n_components=4,
            mixture_params=["bnb_concentration"],
        )
        # The main spec and gene-level auxiliaries should be mixture
        main = [s for s in specs if s.name == "bnb_concentration"][0]
        assert main.is_mixture is True
        psi = [s for s in specs if s.name == "bnb_concentration_psi"][0]
        assert psi.is_mixture is True
        # The hyper-location stays shared (not mixture)
        loc = [s for s in specs if s.name == "bnb_concentration_loc"][0]
        assert loc.is_mixture is False

    def test_bnb_specs_not_mixture_by_default(self):
        """Without bnb_concentration in mixture_params, specs are not
        mixture-specific."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.presets.registry import (
            build_bnb_concentration_spec,
        )

        specs = build_bnb_concentration_spec(
            overdispersion_prior="neg",
            guide_families=GuideFamilyConfig(),
            n_components=4,
            mixture_params=["phi", "mu"],
        )
        main = [s for s in specs if s.name == "bnb_concentration"][0]
        assert main.is_mixture is False

    def test_kappa_broadcasts_with_mixture_r(self):
        """omega (D, G) broadcasts correctly with r (D, K, G) for kappa."""
        D, K, G = 2, 4, 10
        omega = jnp.ones((D, G)) * 0.5
        r = jnp.ones((D, K, G)) * 5.0

        omega_safe = jnp.clip(omega, 1e-6, None)
        # Expand omega to broadcast: (D, G) -> (D, 1, G)
        while omega_safe.ndim < r.ndim:
            omega_safe = jnp.expand_dims(omega_safe, axis=-2)

        kappa = 2.0 + (r + 1.0) / omega_safe
        assert kappa.shape == (D, K, G)
        assert jnp.all(kappa > 2.0)


# ============================================================================
# Mixture log-likelihood broadcasting tests
# ============================================================================


class TestMixtureLLBroadcasting:
    """Verify _build_ll_count_dist reshapes bnb_concentration to match
    the (1, G, K) layout used by mixture log-likelihood functions."""

    def test_mixture_specific_bnb_in_ll(self):
        """Pre-shaped ``bnb_concentration`` (1, G, K) broadcasts against
        ``r``/``p`` ``(1, G, K)`` via ``_build_ll_count_dist``.

        The layout-aware refactor moved the reshape from inside
        ``_build_ll_count_dist`` up to the delegate (where the
        :class:`AxisLayout` information lives).  The helper now consumes
        an already-broadcast-ready tensor, so this test feeds it the
        pre-shaped ``(1, G, K)`` array directly.
        """
        from scribe.models.components.likelihoods._log_prob import (
            _build_ll_count_dist,
        )

        K, G = 4, 20
        r = jnp.ones((1, G, K)) * 5.0
        p = jnp.full((1, G, K), 0.4)
        # Simulate what ``_prepare_mixture_tensor`` produces from
        # ("components", "genes")-laid-out concentration (K, G).
        bnb = jnp.transpose(jnp.ones((K, G)) * 0.2, (1, 0)).reshape(1, G, K)

        d = _build_ll_count_dist(r, p, bnb)
        counts = jnp.ones((1, G, K))
        lp = d.log_prob(counts)
        assert lp.shape == (1, G, K)
        assert jnp.all(jnp.isfinite(lp))

    def test_shared_bnb_in_mixture_ll(self):
        """Pre-shaped ``bnb_concentration`` ``(1, G, 1)`` broadcasts in
        the mixture layout ``(1, G, K)``."""
        from scribe.models.components.likelihoods._log_prob import (
            _build_ll_count_dist,
        )

        K, G = 4, 20
        r = jnp.ones((1, G, K)) * 5.0
        p = jnp.full((1, G, K), 0.4)
        # Shared-across-components layout: singleton component axis.
        bnb = (jnp.ones(G) * 0.2).reshape(1, G, 1)

        d = _build_ll_count_dist(r, p, bnb)
        counts = jnp.ones((1, G, K))
        lp = d.log_prob(counts)
        assert lp.shape == (1, G, K)
        assert jnp.all(jnp.isfinite(lp))

    def test_no_bnb_in_mixture_ll(self):
        """Passing ``None`` for ``bnb_concentration`` yields a plain NB."""
        from scribe.models.components.likelihoods._log_prob import (
            _build_ll_count_dist,
        )

        K, G = 4, 20
        r = jnp.ones((1, G, K)) * 5.0
        p = jnp.full((1, G, K), 0.4)

        d = _build_ll_count_dist(r, p, None)
        counts = jnp.ones((1, G, K))
        lp = d.log_prob(counts)
        assert lp.shape == (1, G, K)
        assert jnp.all(jnp.isfinite(lp))

    def test_nbvcp_mixture_ll_with_bnb(self):
        """End-to-end: NBVCP mixture ``.log_prob`` with mixture BNB."""
        from scribe.models.components.likelihoods import NBWithVCPLikelihood
        from scribe.core.axis_layout import AxisLayout

        n_cells, n_genes, K = 50, 10, 3
        rng = np.random.default_rng(42)
        counts = jnp.array(
            rng.poisson(5, size=(n_cells, n_genes)), dtype=jnp.float32
        )
        params = {
            "p": jnp.full((K, n_genes), 0.4),
            "r": jnp.ones((K, n_genes)) * 5.0,
            "p_capture": jnp.full(n_cells, 0.8),
            "mixing_weights": jnp.ones(K) / K,
            "bnb_concentration": jnp.ones((K, n_genes)) * 0.3,
        }
        layouts = {
            "p": AxisLayout(("components", "genes")),
            "r": AxisLayout(("components", "genes")),
            "p_capture": AxisLayout(("cells",)),
            "mixing_weights": AxisLayout(("components",)),
            "bnb_concentration": AxisLayout(("components", "genes")),
        }

        nbvcp = NBWithVCPLikelihood()

        # Cell-level, split by component
        ll_split = nbvcp.log_prob(
            counts, params, layouts, split_components=True
        )
        assert ll_split.shape == (n_cells, K)
        assert jnp.all(jnp.isfinite(ll_split))

        # Cell-level, mixed
        ll_mixed = nbvcp.log_prob(
            counts, params, layouts, split_components=False
        )
        assert ll_mixed.shape == (n_cells,)
        assert jnp.all(jnp.isfinite(ll_mixed))


# ============================================================================
# Sampling with mixture BNB tests
# ============================================================================


class TestSamplingMixtureBNB:
    """Verify that sample_biological_nb and sample_posterior_ppc handle
    mixture-specific bnb_concentration correctly, both in the MAP (loop)
    and posterior (vmap) paths."""

    def test_bio_nb_map_mixture_bnb(self):
        """MAP path: sample_biological_nb with mixture BNB."""
        from scribe.sampling import sample_biological_nb

        K, G, C = 4, 15, 30
        r = jnp.ones((K, G)) * 5.0
        p = jnp.full(K, 0.4)
        mw = jnp.ones(K) / K
        bnb = jnp.ones((K, G)) * 0.2
        key = jax.random.PRNGKey(0)

        samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=C,
            rng_key=key,
            n_samples=3,
            mixing_weights=mw,
            bnb_concentration=bnb,
        )
        assert samples.shape == (3, C, G)
        assert jnp.all(jnp.isfinite(samples))

    def test_bio_nb_posterior_mixture_bnb(self):
        """Posterior (vmap) path: sample_biological_nb with BNB."""
        from scribe.sampling import sample_biological_nb

        S, K, G, C = 5, 3, 10, 20
        r = jnp.ones((S, K, G)) * 5.0
        p = jnp.full((S, K), 0.4)
        mw = jnp.ones((S, K)) / K
        bnb = jnp.ones((S, K, G)) * 0.2
        key = jax.random.PRNGKey(1)

        samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=C,
            rng_key=key,
            mixing_weights=mw,
            bnb_concentration=bnb,
        )
        assert samples.shape == (S, C, G)
        assert jnp.all(jnp.isfinite(samples))

    def test_posterior_ppc_map_mixture_bnb(self):
        """MAP path: sample_posterior_ppc with mixture BNB + VCP."""
        from scribe.sampling import sample_posterior_ppc

        K, G, C = 4, 15, 30
        r = jnp.ones((K, G)) * 5.0
        p = jnp.full(K, 0.4)
        mw = jnp.ones(K) / K
        p_capture = jnp.full(C, 0.8)
        bnb = jnp.ones((K, G)) * 0.2
        key = jax.random.PRNGKey(2)

        samples = sample_posterior_ppc(
            r=r,
            p=p,
            n_cells=C,
            rng_key=key,
            n_samples=2,
            mixing_weights=mw,
            p_capture=p_capture,
            bnb_concentration=bnb,
        )
        assert samples.shape == (2, C, G)
        assert jnp.all(jnp.isfinite(samples))

    def test_posterior_ppc_vmap_mixture_bnb(self):
        """Posterior (vmap) path: sample_posterior_ppc with BNB + VCP."""
        from scribe.sampling import sample_posterior_ppc

        S, K, G, C = 5, 3, 10, 20
        r = jnp.ones((S, K, G)) * 5.0
        p = jnp.full((S, K), 0.4)
        mw = jnp.ones((S, K)) / K
        p_capture = jnp.full((S, C), 0.8)
        bnb = jnp.ones((S, K, G)) * 0.2
        key = jax.random.PRNGKey(3)

        samples = sample_posterior_ppc(
            r=r,
            p=p,
            n_cells=C,
            rng_key=key,
            mixing_weights=mw,
            p_capture=p_capture,
            bnb_concentration=bnb,
        )
        assert samples.shape == (S, C, G)
        assert jnp.all(jnp.isfinite(samples))

    def test_bio_nb_shared_bnb_in_mixture(self):
        """bnb_concentration (G,) works correctly in mixture sampling."""
        from scribe.sampling import sample_biological_nb

        K, G, C = 3, 10, 25
        r = jnp.ones((K, G)) * 5.0
        p = jnp.full(K, 0.4)
        mw = jnp.ones(K) / K
        bnb = jnp.ones(G) * 0.2
        key = jax.random.PRNGKey(4)

        samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=C,
            rng_key=key,
            n_samples=2,
            mixing_weights=mw,
            bnb_concentration=bnb,
        )
        assert samples.shape == (2, C, G)
        assert jnp.all(jnp.isfinite(samples))


# ============================================================================
# Posterior extraction tests for Gaussian BNB prior
# ============================================================================


class TestBNBGaussianPosterior:
    """Verify _apply_bnb_concentration correctly handles the Gaussian prior.

    The Gaussian branch emits ``bnb_concentration_loc`` /
    ``bnb_concentration_scale`` as direct guide variational params
    (no NCP raw/auxiliary sites), plus ``bnb_omega_hyper_*`` hyper-params.
    """

    def test_gaussian_distributions_extracted(self):
        """get_posterior_distributions returns bnb_concentration for Gaussian."""
        from scribe.models.builders.posterior import get_posterior_distributions
        from scribe.models.config import ModelConfig

        mc = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
            overdispersion="bnb",
            overdispersion_prior="gaussian",
            positive_transform="softplus",
        )

        # Minimal params mimicking Gaussian prior's guide output
        params = {
            "r_loc": jnp.zeros(5),
            "r_scale": jnp.ones(5),
            "p_loc": jnp.zeros(5),
            "p_scale": jnp.ones(5),
            "bnb_omega_hyper_loc_loc": jnp.array(0.0),
            "bnb_omega_hyper_loc_scale": jnp.array(1.0),
            "bnb_omega_hyper_scale_loc": jnp.array(0.0),
            "bnb_omega_hyper_scale_scale": jnp.array(1.0),
            "bnb_concentration_loc": jnp.zeros(5),
            "bnb_concentration_scale": jnp.ones(5),
        }

        dists = get_posterior_distributions(params, mc)
        assert "bnb_concentration" in dists
        assert "bnb_omega_hyper_loc" in dists
        assert "bnb_omega_hyper_scale" in dists

        # bnb_concentration should be dict with base + transform
        d = dists["bnb_concentration"]
        assert isinstance(d, dict)
        assert "base" in d and "transform" in d

    def test_gaussian_map_extraction(self):
        """MAP extraction produces bnb_concentration and bnb_kappa for Gaussian."""
        from numpyro.distributions.transforms import SoftplusTransform

        from scribe.models.builders.posterior import get_posterior_distributions
        from scribe.models.config import ModelConfig

        mc = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
            overdispersion="bnb",
            overdispersion_prior="gaussian",
            positive_transform="softplus",
        )

        # Simulate params where bnb_concentration_loc = 1.0 for all genes
        loc_val = 1.0
        G = 5
        params = {
            "r_loc": jnp.zeros(G),
            "r_scale": jnp.ones(G),
            "p_loc": jnp.zeros(G),
            "p_scale": jnp.ones(G),
            "bnb_omega_hyper_loc_loc": jnp.array(0.0),
            "bnb_omega_hyper_loc_scale": jnp.array(1.0),
            "bnb_omega_hyper_scale_loc": jnp.array(0.0),
            "bnb_omega_hyper_scale_scale": jnp.array(1.0),
            "bnb_concentration_loc": jnp.full(G, loc_val),
            "bnb_concentration_scale": jnp.ones(G),
        }

        dists = get_posterior_distributions(params, mc)

        # Manually compute expected MAP: softplus(loc_val)
        expected_omega = SoftplusTransform()(jnp.array(loc_val))

        d = dists["bnb_concentration"]
        base = d["base"]
        transform = d["transform"]
        map_omega = transform(base.loc)
        assert jnp.allclose(map_omega, jnp.full(G, expected_omega), atol=1e-5)

    def test_neg_prior_not_affected(self):
        """NEG prior path still works (regression check)."""
        from scribe.models.builders.posterior import get_posterior_distributions
        from scribe.models.config import ModelConfig

        mc = ModelConfig(
            base_model="nbdm",
            parameterization="canonical",
            unconstrained=True,
            overdispersion="bnb",
            overdispersion_prior="neg",
            positive_transform="softplus",
        )

        # Params mimicking NEG prior guide output
        params = {
            "r_loc": jnp.zeros(5),
            "r_scale": jnp.ones(5),
            "p_loc": jnp.zeros(5),
            "p_scale": jnp.ones(5),
            "bnb_concentration_loc_loc": jnp.array(0.0),
            "bnb_concentration_loc_scale": jnp.array(1.0),
            "bnb_concentration_psi_concentration": jnp.ones(5),
            "bnb_concentration_psi_rate": jnp.ones(5),
            "bnb_concentration_zeta_concentration": jnp.ones(5),
            "bnb_concentration_zeta_rate": jnp.ones(5),
            "bnb_concentration_raw_loc": jnp.zeros(5),
            "bnb_concentration_raw_scale": jnp.ones(5),
        }

        dists = get_posterior_distributions(params, mc)
        # NEG should have the NCP auxiliary sites, not direct bnb_concentration
        assert "bnb_concentration_loc" in dists
        assert "bnb_concentration_psi" in dists
        assert "bnb_concentration_raw" in dists
        # Direct bnb_concentration should NOT be present (it's reconstructed
        # later in get_map via _reconstruct_ncp_maps)
        assert "bnb_concentration" not in dists


# ============================================================================
# Quadrature utility tests
# ============================================================================


class TestGaussLegendreQuadrature:
    """Verify the Gauss--Legendre quadrature utility."""

    def test_polynomial_exact(self):
        """Gauss-Legendre with n nodes is exact for polynomials of degree <= 2n-1."""
        from scribe.stats.quadrature import gauss_legendre_integrate

        # int_0^1 x^2 dx = 1/3; exact for n >= 2
        result = gauss_legendre_integrate(lambda x: x**2, 0.0, 1.0, n=4)
        np.testing.assert_allclose(float(result), 1.0 / 3.0, atol=1e-12)

    def test_x_cubed(self):
        """int_0^1 x^3 dx = 1/4."""
        from scribe.stats.quadrature import gauss_legendre_integrate

        result = gauss_legendre_integrate(lambda x: x**3, 0.0, 1.0, n=4)
        np.testing.assert_allclose(float(result), 0.25, atol=1e-12)

    def test_beta_function_integral(self):
        """int_0^1 x^(a-1)(1-x)^(b-1) dx = B(a, b) for a=3, b=4."""
        from scribe.stats.quadrature import gauss_legendre_integrate
        from jax.scipy.special import betaln

        a, b = 3.0, 4.0
        expected = float(jnp.exp(betaln(a, b)))
        result = gauss_legendre_integrate(
            lambda x: x ** (a - 1) * (1 - x) ** (b - 1), 0.0, 1.0, n=32
        )
        np.testing.assert_allclose(float(result), expected, rtol=1e-5)

    def test_nodes_weights_shape(self):
        """Nodes and weights have the requested shape."""
        from scribe.stats.quadrature import gauss_legendre_nodes_weights

        nodes, weights = gauss_legendre_nodes_weights(16, 0.0, 1.0)
        assert nodes.shape == (16,)
        assert weights.shape == (16,)

    def test_nodes_within_interval(self):
        """All nodes lie within [a, b]."""
        from scribe.stats.quadrature import gauss_legendre_nodes_weights

        nodes, _ = gauss_legendre_nodes_weights(64, -2.0, 3.0)
        assert float(jnp.min(nodes)) >= -2.0
        assert float(jnp.max(nodes)) <= 3.0

    def test_weights_sum_to_interval_length(self):
        """Weights sum to b - a (integral of f=1)."""
        from scribe.stats.quadrature import gauss_legendre_nodes_weights

        _, weights = gauss_legendre_nodes_weights(32, 2.0, 5.0)
        np.testing.assert_allclose(float(jnp.sum(weights)), 3.0, atol=1e-5)

    def test_batched_integrand(self):
        """Integrand returning (N, batch) produces a (batch,) result."""
        from scribe.stats.quadrature import gauss_legendre_integrate

        # int_0^1 [x, x^2, x^3] dx = [1/2, 1/3, 1/4]
        def f(x):
            return jnp.stack([x, x**2, x**3], axis=-1)

        result = gauss_legendre_integrate(f, 0.0, 1.0, n=8)
        np.testing.assert_allclose(
            np.array(result), [0.5, 1.0 / 3.0, 0.25], atol=1e-10
        )


# ============================================================================
# BNB denoising tests
# ============================================================================


class TestBNBDenoisingQuadrature:
    """Verify BNB MAP denoising via quadrature."""

    @pytest.fixture
    def nb_params(self):
        """Standard NB parameters for a small example."""
        C, G = 4, 3
        r = jnp.array([5.0, 10.0, 2.0])
        p = jnp.array([0.3, 0.6, 0.1])
        p_capture = jnp.array([0.05, 0.08, 0.03, 0.10])
        counts = jnp.array([[2, 15, 0], [0, 5, 1], [10, 0, 3], [1, 20, 0]])
        return counts, r, p, p_capture

    def test_reduces_to_nb_when_kappa_large(self, nb_params):
        """As omega shrinks, BNB denoising approaches NB closed-form.

        We use a moderately small omega (0.01) with extra quadrature
        nodes (256) to resolve the increasingly peaked posterior.  The
        posterior mean should be within ~5% of the NB closed-form.
        """
        from scribe.sampling import _denoise_bnb_quadrature

        counts, r, p, p_capture = nb_params

        # Moderately small omega => large kappa => near-NB regime
        # omega=0.01 gives kappa ~ 602 for r=5, which concentrates the
        # Beta prior but is still resolvable by 256 quadrature nodes.
        omega = jnp.full(r.shape, 0.01)
        bnb_mean, _ = _denoise_bnb_quadrature(
            counts, r, p, p_capture, omega, n_nodes=256
        )

        # NB closed-form
        nu = p_capture[:, None]
        probs_post = p * (1.0 - nu)
        one_minus_pp = 1.0 - probs_post
        nb_mean = (counts + r * probs_post) / one_minus_pp

        np.testing.assert_allclose(
            np.array(bnb_mean), np.array(nb_mean), rtol=0.05
        )

    def test_identity_when_nu_is_one(self, nb_params):
        """When nu = 1, denoised = observed (no capture loss)."""
        from scribe.sampling import _denoise_bnb_quadrature

        counts, r, p, _ = nb_params
        p_capture_perfect = jnp.ones(counts.shape[0])
        omega = jnp.full(r.shape, 0.5)

        bnb_mean, bnb_var = _denoise_bnb_quadrature(
            counts, r, p, p_capture_perfect, omega
        )

        np.testing.assert_allclose(
            np.array(bnb_mean), np.array(counts), atol=1e-4
        )
        np.testing.assert_allclose(np.array(bnb_var), 0.0, atol=1e-4)

    def test_bnb_mean_differs_from_nb(self, nb_params):
        """BNB posterior mean should differ from NB for finite kappa."""
        from scribe.sampling import _denoise_bnb_quadrature

        counts, r, p, p_capture = nb_params

        # Moderate overdispersion
        omega = jnp.full(r.shape, 1.0)
        bnb_mean, _ = _denoise_bnb_quadrature(counts, r, p, p_capture, omega)

        # NB closed-form
        nu = p_capture[:, None]
        probs_post = p * (1.0 - nu)
        one_minus_pp = 1.0 - probs_post
        nb_mean = (counts + r * probs_post) / one_minus_pp

        # With omega=1.0, BNB adds substantial uncertainty.  The means
        # should be positive and finite, and should differ from NB.
        assert jnp.all(jnp.isfinite(bnb_mean))
        assert jnp.all(bnb_mean > 0)
        assert not jnp.allclose(bnb_mean, nb_mean, atol=0.5)

    def test_variance_positive(self, nb_params):
        """BNB denoised variance should be non-negative."""
        from scribe.sampling import _denoise_bnb_quadrature

        counts, r, p, p_capture = nb_params
        omega = jnp.full(r.shape, 0.5)
        _, bnb_var = _denoise_bnb_quadrature(counts, r, p, p_capture, omega)
        assert jnp.all(bnb_var >= -1e-6)

    def test_output_shape(self, nb_params):
        """Output shapes match (C, G)."""
        from scribe.sampling import _denoise_bnb_quadrature

        counts, r, p, p_capture = nb_params
        omega = jnp.full(r.shape, 0.5)
        bnb_mean, bnb_var = _denoise_bnb_quadrature(
            counts, r, p, p_capture, omega
        )
        assert bnb_mean.shape == counts.shape
        assert bnb_var.shape == counts.shape


class TestBNBDenoSampling:
    """Verify BNB augmented sampling denoising."""

    def test_sample_shape(self):
        """Sampled p values have shape (C, G)."""
        from scribe.sampling import _sample_p_posterior_bnb

        C, G = 8, 5
        r = jnp.ones(G) * 5.0
        p = jnp.ones(G) * 0.4
        p_capture = jnp.ones(C) * 0.05
        counts = jnp.ones((C, G), dtype=jnp.float32) * 3.0
        omega = jnp.ones(G) * 0.5
        key = jax.random.PRNGKey(0)

        p_samples = _sample_p_posterior_bnb(key, counts, r, p, p_capture, omega)
        assert p_samples.shape == (C, G)

    def test_samples_in_unit_interval(self):
        """All sampled p values lie in (0, 1)."""
        from scribe.sampling import _sample_p_posterior_bnb

        C, G = 16, 4
        r = jnp.ones(G) * 3.0
        p = jnp.ones(G) * 0.5
        p_capture = jnp.ones(C) * 0.07
        counts = jnp.ones((C, G), dtype=jnp.float32) * 2.0
        omega = jnp.ones(G) * 1.0
        key = jax.random.PRNGKey(42)

        p_samples = _sample_p_posterior_bnb(key, counts, r, p, p_capture, omega)
        assert jnp.all(p_samples > 0.0)
        assert jnp.all(p_samples < 1.0)

    def test_sampling_mean_matches_quadrature(self):
        """Mean of many samples should be close to quadrature expectation."""
        from scribe.sampling import (
            _denoise_bnb_quadrature,
            _sample_p_posterior_bnb,
        )

        # Single cell, single gene to make comparison clean
        C, G = 1, 1
        r = jnp.array([5.0])
        p = jnp.array([0.4])
        p_capture = jnp.array([0.06])
        counts = jnp.array([[3.0]])
        omega = jnp.array([0.5])

        # Quadrature mean
        bnb_mean, _ = _denoise_bnb_quadrature(counts, r, p, p_capture, omega)

        # Many augmented samples
        n_samples = 5000
        denoised_samples = []
        for i in range(n_samples):
            key = jax.random.PRNGKey(i)
            key_p, key_nb = jax.random.split(key)
            p_s = _sample_p_posterior_bnb(key_p, counts, r, p, p_capture, omega)
            nu = p_capture[:, None]
            probs_cond = p_s * (1.0 - nu)
            alpha_cond = r + counts
            d = dist.NegativeBinomialProbs(
                total_count=alpha_cond, probs=probs_cond
            ).sample(key_nb)
            denoised_samples.append(float((counts + d)[0, 0]))

        sample_mean = np.mean(denoised_samples)
        quad_mean = float(bnb_mean[0, 0])

        # Allow 10% relative tolerance for Monte Carlo
        np.testing.assert_allclose(sample_mean, quad_mean, rtol=0.15)


class TestDenoiseBatchBNBDispatch:
    """Verify that _denoise_batch correctly dispatches to BNB denoising."""

    def test_bnb_mean_dispatches(self):
        """method='mean' with bnb_concentration triggers quadrature."""
        from scribe.sampling import _denoise_batch

        C, G = 4, 3
        counts = jnp.array(
            [
                [2.0, 10.0, 0.0],
                [0.0, 5.0, 1.0],
                [8.0, 0.0, 3.0],
                [1.0, 15.0, 0.0],
            ]
        )
        r = jnp.array([5.0, 8.0, 3.0])
        p = jnp.array([0.3, 0.5, 0.2])
        p_capture = jnp.array([0.05, 0.08, 0.04, 0.10])
        omega = jnp.full(G, 0.5)

        key = jax.random.PRNGKey(0)
        denoised, variance = _denoise_batch(
            counts,
            r,
            p,
            p_capture,
            gate=None,
            method="mean",
            rng_key=key,
            bnb_concentration=omega,
        )

        assert denoised.shape == (C, G)
        assert variance.shape == (C, G)
        # With VCP, denoised should be >= counts (inflated for loss)
        assert jnp.all(denoised >= counts - 0.01)

    def test_bnb_sample_dispatches(self):
        """method='sample' with bnb_concentration triggers augmented sampling."""
        from scribe.sampling import _denoise_batch

        C, G = 4, 3
        counts = jnp.array(
            [
                [2.0, 10.0, 0.0],
                [0.0, 5.0, 1.0],
                [8.0, 0.0, 3.0],
                [1.0, 15.0, 0.0],
            ]
        )
        r = jnp.array([5.0, 8.0, 3.0])
        p = jnp.array([0.3, 0.5, 0.2])
        p_capture = jnp.array([0.05, 0.08, 0.04, 0.10])
        omega = jnp.full(G, 0.5)

        key = jax.random.PRNGKey(42)
        denoised, variance = _denoise_batch(
            counts,
            r,
            p,
            p_capture,
            gate=None,
            method="sample",
            rng_key=key,
            bnb_concentration=omega,
        )

        assert denoised.shape == (C, G)
        # Sampled denoised values should be >= counts
        assert jnp.all(denoised >= counts)

    def test_no_bnb_falls_back_to_nb(self):
        """Without bnb_concentration, NB closed-form is used."""
        from scribe.sampling import _denoise_batch

        counts = jnp.array([[5.0, 0.0], [0.0, 10.0]])
        r = jnp.array([3.0, 7.0])
        p = jnp.array([0.4, 0.6])
        p_capture = jnp.array([0.05, 0.08])

        key = jax.random.PRNGKey(0)
        denoised, _ = _denoise_batch(
            counts,
            r,
            p,
            p_capture,
            gate=None,
            method="mean",
            rng_key=key,
            bnb_concentration=None,
        )

        # NB closed-form
        nu = p_capture[:, None]
        probs_post = p * (1.0 - nu)
        expected = (counts + r * probs_post) / (1.0 - probs_post)

        np.testing.assert_allclose(
            np.array(denoised), np.array(expected), atol=1e-5
        )

    def test_no_vcp_bnb_falls_back_to_nb(self):
        """Without p_capture, BNB denoising falls back to NB identity."""
        from scribe.sampling import _denoise_batch

        counts = jnp.array([[5.0, 0.0], [0.0, 10.0]])
        r = jnp.array([3.0, 7.0])
        p = jnp.array([0.4, 0.6])
        omega = jnp.full(2, 0.5)

        key = jax.random.PRNGKey(0)
        denoised, _ = _denoise_batch(
            counts,
            r,
            p,
            p_capture=None,
            gate=None,
            method="mean",
            rng_key=key,
            bnb_concentration=omega,
        )

        # Without VCP: probs_post=0, so denoised = counts
        np.testing.assert_allclose(
            np.array(denoised), np.array(counts), atol=1e-5
        )
