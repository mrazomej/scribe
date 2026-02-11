"""Unit tests for dispatched free functions in ``_latent_dispatch.py``.

Tests that each ``@dispatch`` implementation produces the correct output
structure independently of the ``LatentSpaceMixin`` or ``ScribeVAEResults``.

Device: use ``pytest --device cpu`` (default) or ``pytest --device gpu``.
"""

import os

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random

from scribe.models.builders.parameter_specs import GaussianLatentSpec
from scribe.models.components.vae_components import GaussianEncoder
from scribe.svi._latent_dispatch import (
    get_latent_embedding,
    run_encoder,
    sample_latent_posterior,
)

# ==============================================================================
# Constants
# ==============================================================================

LATENT_DIM = 5
N_GENES = 20
N_CELLS = 10
HIDDEN_DIMS = [16]


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="session", autouse=True)
def _set_device(request):
    """Use --device (default: cpu) so these tests match the rest of the suite."""
    device = request.config.getoption("--device", default="cpu")
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        jax.config.update("jax_platform_name", "cpu")
    else:
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]


@pytest.fixture
def gaussian_spec():
    """Create a GaussianLatentSpec for testing."""
    return GaussianLatentSpec(latent_dim=LATENT_DIM)


@pytest.fixture
def encoder():
    """Create a GaussianEncoder Linen module."""
    return GaussianEncoder(
        input_dim=N_GENES,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
    )


@pytest.fixture
def encoder_params(encoder):
    """Initialize encoder and return trained params.

    Uses ``encoder.init`` to get a valid parameter tree.
    """
    key = random.PRNGKey(0)
    dummy_input = jnp.ones((1, N_GENES))
    variables = encoder.init(key, dummy_input)
    return variables["params"]


@pytest.fixture
def counts():
    """Create a dummy count matrix."""
    return random.poisson(
        random.PRNGKey(1), lam=10.0, shape=(N_CELLS, N_GENES)
    ).astype(jnp.float32)


@pytest.fixture
def rng_key():
    """Create a PRNG key for sampling."""
    return random.PRNGKey(42)


# ==============================================================================
# Tests: run_encoder
# ==============================================================================


class TestRunEncoder:
    """Tests for the ``run_encoder`` dispatched function."""

    def test_gaussian_returns_correct_dict_keys(
        self, gaussian_spec, encoder, encoder_params, counts
    ):
        """Gaussian run_encoder should return dict with 'loc' and 'log_scale'.

        The output keys are specific to the GaussianEncoder — other encoder
        types would produce different keys (e.g. 'mu', 'kappa' for vMF).
        """
        result = run_encoder(gaussian_spec, encoder, encoder_params, counts)
        assert isinstance(result, dict)
        assert "loc" in result
        assert "log_scale" in result

    def test_gaussian_returns_correct_shapes(
        self, gaussian_spec, encoder, encoder_params, counts
    ):
        """Both loc and log_scale should have shape (n_cells, latent_dim).

        These are the variational parameters for the per-cell latent
        distributions.
        """
        result = run_encoder(gaussian_spec, encoder, encoder_params, counts)
        assert result["loc"].shape == (N_CELLS, LATENT_DIM)
        assert result["log_scale"].shape == (N_CELLS, LATENT_DIM)

    def test_gaussian_output_is_finite(
        self, gaussian_spec, encoder, encoder_params, counts
    ):
        """Encoder output should be finite for valid input.

        Non-finite values would indicate numerical issues in the
        encoder network.
        """
        result = run_encoder(gaussian_spec, encoder, encoder_params, counts)
        assert jnp.all(jnp.isfinite(result["loc"]))
        assert jnp.all(jnp.isfinite(result["log_scale"]))


# ==============================================================================
# Tests: sample_latent_posterior
# ==============================================================================


class TestSampleLatentPosterior:
    """Tests for the ``sample_latent_posterior`` dispatched function."""

    def test_gaussian_produces_correct_shape(self, gaussian_spec, rng_key):
        """Samples should have shape (n_samples, n_cells, latent_dim).

        This is the standard shape for posterior samples — the first axis
        indexes over samples, the second over cells.
        """
        n_samples = 7
        var_params = {
            "loc": jnp.zeros((N_CELLS, LATENT_DIM)),
            "log_scale": jnp.zeros((N_CELLS, LATENT_DIM)),
        }
        z = sample_latent_posterior(
            gaussian_spec, var_params, rng_key, n_samples
        )
        assert z.shape == (n_samples, N_CELLS, LATENT_DIM)

    def test_gaussian_samples_are_finite(self, gaussian_spec, rng_key):
        """All posterior samples should be finite.

        Non-finite samples would indicate overflow in exp(0.5 * log_scale).
        """
        var_params = {
            "loc": jnp.ones((N_CELLS, LATENT_DIM)),
            "log_scale": -jnp.ones((N_CELLS, LATENT_DIM)),
        }
        z = sample_latent_posterior(gaussian_spec, var_params, rng_key, 10)
        assert jnp.all(jnp.isfinite(z))

    def test_gaussian_reparameterization_deterministic(
        self, gaussian_spec
    ):
        """Same PRNG key should produce identical samples.

        This validates that the reparameterization trick is purely
        deterministic given the same random state.
        """
        var_params = {
            "loc": jnp.ones((N_CELLS, LATENT_DIM)) * 2.0,
            "log_scale": jnp.zeros((N_CELLS, LATENT_DIM)),
        }
        key = random.PRNGKey(123)
        z1 = sample_latent_posterior(gaussian_spec, var_params, key, 5)
        z2 = sample_latent_posterior(gaussian_spec, var_params, key, 5)
        np.testing.assert_array_equal(np.array(z1), np.array(z2))

    def test_gaussian_different_keys_produce_different_samples(
        self, gaussian_spec
    ):
        """Different PRNG keys should produce different samples.

        This ensures the sampling is actually stochastic across different
        random seeds.
        """
        var_params = {
            "loc": jnp.ones((N_CELLS, LATENT_DIM)),
            "log_scale": jnp.zeros((N_CELLS, LATENT_DIM)),
        }
        z1 = sample_latent_posterior(
            gaussian_spec, var_params, random.PRNGKey(0), 5
        )
        z2 = sample_latent_posterior(
            gaussian_spec, var_params, random.PRNGKey(1), 5
        )
        # Samples should differ
        assert not jnp.allclose(z1, z2)

    def test_gaussian_loc_shifts_samples(self, gaussian_spec, rng_key):
        """Shifting loc should shift the sample mean correspondingly.

        With large n_samples the sample mean should be close to loc.
        """
        loc_val = 5.0
        var_params = {
            "loc": jnp.full((N_CELLS, LATENT_DIM), loc_val),
            "log_scale": -2.0 * jnp.ones((N_CELLS, LATENT_DIM)),
        }
        z = sample_latent_posterior(gaussian_spec, var_params, rng_key, 1000)
        # Sample mean should be close to loc
        sample_mean = z.mean(axis=0)
        np.testing.assert_allclose(
            np.array(sample_mean),
            np.full((N_CELLS, LATENT_DIM), loc_val),
            atol=0.2,
        )


# ==============================================================================
# Tests: get_latent_embedding
# ==============================================================================


class TestGetLatentEmbedding:
    """Tests for the ``get_latent_embedding`` dispatched function."""

    def test_gaussian_returns_loc(self, gaussian_spec):
        """Gaussian embedding should be the posterior mean (loc).

        For Gaussian encoders, the natural point embedding is the
        posterior mean.  Other encoder types (e.g. vMF) would return
        a different quantity (e.g. the mode on the sphere).
        """
        loc = jnp.ones((N_CELLS, LATENT_DIM)) * 3.0
        var_params = {
            "loc": loc,
            "log_scale": jnp.zeros((N_CELLS, LATENT_DIM)),
        }
        embedding = get_latent_embedding(gaussian_spec, var_params)
        np.testing.assert_array_equal(np.array(embedding), np.array(loc))

    def test_gaussian_returns_correct_shape(self, gaussian_spec):
        """Embedding should have shape (n_cells, latent_dim)."""
        var_params = {
            "loc": jnp.zeros((N_CELLS, LATENT_DIM)),
            "log_scale": jnp.zeros((N_CELLS, LATENT_DIM)),
        }
        embedding = get_latent_embedding(gaussian_spec, var_params)
        assert embedding.shape == (N_CELLS, LATENT_DIM)
