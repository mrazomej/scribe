"""
Tests for LatentSpec and GaussianLatentSpec.

Covers:
- LatentSpec base (NotImplementedError for make_guide_dist)
- GaussianLatentSpec.make_guide_dist shape and semantics
- sample_site
- Integration: encoder output + latent_spec -> guide dist -> sample z
"""

import pytest
import numpy.testing as npt
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from scribe.models.builders.parameter_specs import (
    LatentSpec,
    GaussianLatentSpec,
)
from scribe.models.components.vae_components import GaussianEncoder


# ---------------------------------------------------------------------------
# LatentSpec base
# ---------------------------------------------------------------------------


class TestLatentSpec:
    """Tests for the base LatentSpec."""

    def test_base_make_guide_dist_raises(self):
        """LatentSpec base should raise NotImplementedError for make_guide_dist."""
        spec = LatentSpec(sample_site="z")
        var_params = {"loc": jnp.zeros(3), "log_scale": jnp.zeros(3)}
        with pytest.raises(
            NotImplementedError, match="must implement make_guide_dist"
        ):
            spec.make_guide_dist(var_params)

    def test_base_sample_site_default(self):
        """Default sample_site is 'z'."""
        spec = LatentSpec()
        assert spec.sample_site == "z"

    def test_base_sample_site_custom(self):
        """sample_site can be overridden."""
        spec = LatentSpec(sample_site="latent")
        assert spec.sample_site == "latent"


# ---------------------------------------------------------------------------
# GaussianLatentSpec
# ---------------------------------------------------------------------------


class TestGaussianLatentSpec:
    """Tests for GaussianLatentSpec."""

    @pytest.fixture
    def latent_spec(self):
        return GaussianLatentSpec(latent_dim=5, sample_site="z")

    def test_make_guide_dist_returns_normal_to_event(self, latent_spec):
        """make_guide_dist returns a distribution with event_shape (latent_dim,)."""
        batch = 4
        loc = jnp.zeros((batch, 5))
        log_scale = jnp.full(
            (batch, 5), -1.0
        )  # log_var = -1 -> scale = exp(-0.5)
        var_params = {"loc": loc, "log_scale": log_scale}
        d = latent_spec.make_guide_dist(var_params)
        # .to_event(1) wraps in Independent; we care about shapes
        assert d.event_shape == (5,)
        assert d.batch_shape == (4,)

    def test_make_guide_dist_scale_from_log_variance(self, latent_spec):
        """Scale is exp(0.5 * log_scale) (log-variance convention). Check via sample std."""
        loc = jnp.zeros(5)
        log_scale = jnp.zeros(5)  # log_var = 0 -> scale = 1
        var_params = {"loc": loc, "log_scale": log_scale}
        d = latent_spec.make_guide_dist(var_params)
        key = jax.random.PRNGKey(0)
        samples = d.sample(key, sample_shape=(5000,))
        npt.assert_allclose(jnp.std(samples, axis=0), jnp.ones(5), atol=0.05)

        log_scale_2 = jnp.full(
            5, 2.0 * jnp.log(2.0)
        )  # log_var = 2*ln(2) -> scale = 2
        var_params_2 = {"loc": loc, "log_scale": log_scale_2}
        d2 = latent_spec.make_guide_dist(var_params_2)
        samples_2 = d2.sample(key, sample_shape=(5000,))
        npt.assert_allclose(
            jnp.std(samples_2, axis=0), jnp.full(5, 2.0), atol=0.1
        )

    def test_sample_site_default(self):
        """Default sample_site is 'z'."""
        spec = GaussianLatentSpec(latent_dim=3)
        assert spec.sample_site == "z"

    def test_sample_site_custom(self):
        """sample_site can be overridden."""
        spec = GaussianLatentSpec(latent_dim=3, sample_site="latent_z")
        assert spec.sample_site == "latent_z"

    def test_log_prob_and_sample_consistent(self, latent_spec):
        """Samples from the guide dist have finite log_prob."""
        batch = 2
        loc = jnp.zeros((batch, 5))
        log_scale = jnp.full((batch, 5), -0.5)
        var_params = {"loc": loc, "log_scale": log_scale}
        guide_dist = latent_spec.make_guide_dist(var_params)
        key = jax.random.PRNGKey(0)
        samples = guide_dist.sample(key)
        assert samples.shape == (2, 5)
        lp = guide_dist.log_prob(samples)
        assert lp.shape == (2,)
        assert jnp.all(jnp.isfinite(lp))


# ---------------------------------------------------------------------------
# Integration: encoder + LatentSpec
# ---------------------------------------------------------------------------


class TestEncoderLatentSpecIntegration:
    """Encoder output -> var_params -> latent_spec.make_guide_dist -> sample z."""

    def test_encoder_output_to_guide_dist(self):
        """GaussianEncoder (loc, log_scale) -> var_params -> make_guide_dist -> sample."""
        rng = jax.random.PRNGKey(42)
        input_dim, latent_dim = 20, 5
        encoder = GaussianEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=[32, 16],
        )
        params = encoder.init(rng, jnp.zeros(input_dim))
        counts = jax.random.poisson(rng, lam=5.0, shape=(3, input_dim)).astype(
            jnp.float32
        )
        loc, log_scale = encoder.apply(params, counts)
        assert loc.shape == (3, latent_dim)
        assert log_scale.shape == (3, latent_dim)

        latent_spec = GaussianLatentSpec(latent_dim=latent_dim)
        var_params = {"loc": loc, "log_scale": log_scale}
        guide_dist = latent_spec.make_guide_dist(var_params)
        z = guide_dist.sample(jax.random.PRNGKey(1))
        assert z.shape == (3, latent_dim)
        assert jnp.all(jnp.isfinite(z))
        assert jnp.all(jnp.isfinite(guide_dist.log_prob(z)))
