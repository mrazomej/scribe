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
        from scribe.stats.distributions import BetaNegativeBinomial

        from scribe.models.components.likelihoods.beta_negative_binomial import (
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

        # numpyro convention: probs = failure probability
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
        counts = jnp.array(rng.poisson(5, size=(n_cells, n_genes)), dtype=jnp.float32)
        return counts

    def test_nbdm_with_bnb(self, _data):
        """nbdm_log_likelihood accepts bnb_concentration without error."""
        from scribe.models.log_likelihood import nbdm_log_likelihood

        counts = _data
        params = {
            "p": jnp.array(0.5),
            "r": jnp.ones(5),
            "bnb_concentration": jnp.full(5, 0.1),
        }
        ll = nbdm_log_likelihood(counts, params)
        assert ll.shape == (20,)
        assert jnp.all(jnp.isfinite(ll))

    def test_nbdm_without_bnb_unchanged(self, _data):
        """nbdm_log_likelihood without bnb_concentration is same as before."""
        from scribe.models.log_likelihood import nbdm_log_likelihood

        counts = _data
        params = {"p": jnp.array(0.5), "r": jnp.ones(5)}
        ll = nbdm_log_likelihood(counts, params)
        assert ll.shape == (20,)
        assert jnp.all(jnp.isfinite(ll))

    def test_zinb_with_bnb(self, _data):
        """zinb_log_likelihood accepts bnb_concentration without error."""
        from scribe.models.log_likelihood import zinb_log_likelihood

        counts = _data
        params = {
            "p": jnp.array(0.5),
            "r": jnp.ones(5),
            "gate": jnp.full(5, 0.1),
            "bnb_concentration": jnp.full(5, 0.1),
        }
        ll = zinb_log_likelihood(counts, params)
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
