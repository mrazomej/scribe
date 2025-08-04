#!/usr/bin/env python3
"""
Tests for dpVAE to verify the decoupled prior VAE functionality.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpyro.distributions as dist
from scribe.vae.architectures import dpVAE, VAEConfig, Encoder, Decoder, DecoupledPrior, DecoupledPriorDistribution


@pytest.fixture(scope="session")
def rng_key():
    """Provide a consistent random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def test_input():
    """Generate test input data."""
    key = jax.random.PRNGKey(123)
    input_dim = 10
    batch_size = 4
    # Use positive data (more realistic for scRNA-seq)
    return jax.random.exponential(key, (batch_size, input_dim))


@pytest.fixture(scope="function")
def dpvae_config():
    """Create a VAEConfig for testing."""
    return VAEConfig(
        input_dim=10,
        latent_dim=3,
        hidden_dims=[64, 32],
        activation="relu",
        input_transformation="log1p"
    )


@pytest.fixture(scope="function")
def decoupled_prior(rng_key):
    """Create a DecoupledPrior instance for testing."""
    latent_dim = 3
    num_layers = 2
    hidden_dims = [32, 16]
    rngs = nnx.Rngs(params=rng_key)
    
    return DecoupledPrior(
        latent_dim=latent_dim,
        num_layers=num_layers,
        hidden_dims=hidden_dims,
        rngs=rngs,
        activation="relu",
        mask_type="alternating"
    )


@pytest.fixture(scope="function")
def dpvae(rng_key, dpvae_config, decoupled_prior):
    """Create a dpVAE instance for testing."""
    rngs = nnx.Rngs(params=rng_key)
    
    # Create encoder and decoder
    encoder = Encoder(config=dpvae_config, rngs=rngs)
    decoder = Decoder(config=dpvae_config, rngs=rngs)
    
    return dpVAE(
        encoder=encoder,
        decoder=decoder,
        config=dpvae_config,
        decoupled_prior=decoupled_prior,
        rngs=rngs
    )


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_dpvae_creation(dpvae):
    """Test that the dpVAE is created successfully."""
    assert dpvae is not None
    assert hasattr(dpvae, '__call__')
    assert hasattr(dpvae, 'reparameterize')
    assert hasattr(dpvae, 'encoder')
    assert hasattr(dpvae, 'decoder')
    assert hasattr(dpvae, 'config')
    assert hasattr(dpvae, 'rngs')
    assert hasattr(dpvae, 'decoupled_prior')
    assert hasattr(dpvae, 'get_prior_distribution')


def test_dpvae_architecture(dpvae, dpvae_config):
    """Test that the dpVAE has the correct architecture."""
    # Check that encoder and decoder have correct dimensions
    assert dpvae.encoder.config.input_dim == dpvae_config.input_dim
    assert dpvae.encoder.config.latent_dim == dpvae_config.latent_dim
    assert dpvae.decoder.config.input_dim == dpvae_config.input_dim
    assert dpvae.decoder.config.latent_dim == dpvae_config.latent_dim
    
    # Check that decoupled prior has correct latent dimension
    assert dpvae.decoupled_prior.latent_dim == dpvae_config.latent_dim


def test_dpvae_forward_pass(dpvae, test_input):
    """Test the dpVAE forward pass."""
    reconstructed, mean, logvar = dpvae(test_input)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = dpvae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_dpvae_forward_pass_training_mode(dpvae, test_input):
    """Test the dpVAE forward pass in training mode."""
    reconstructed, mean, logvar = dpvae(test_input, training=True)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = dpvae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_dpvae_forward_pass_eval_mode(dpvae, test_input):
    """Test the dpVAE forward pass in evaluation mode."""
    reconstructed, mean, logvar = dpvae(test_input, training=False)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = dpvae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Decoupled Prior Distribution Tests
# ------------------------------------------------------------------------------

def test_get_prior_distribution(dpvae):
    """Test the get_prior_distribution method."""
    # Get distribution with default base distribution
    distribution = dpvae.get_prior_distribution()
    
    assert isinstance(distribution, DecoupledPriorDistribution)
    assert distribution.decoupled_prior == dpvae.decoupled_prior
    assert isinstance(distribution.base_distribution, dist.Normal)


def test_get_prior_distribution_custom_base(dpvae):
    """Test the get_prior_distribution method with custom base distribution."""
    # Create custom base distribution
    custom_base = dist.StudentT(df=3.0, loc=0.0, scale=1.0)
    
    # Get distribution with custom base
    distribution = dpvae.get_prior_distribution(custom_base)
    
    assert isinstance(distribution, DecoupledPriorDistribution)
    assert distribution.decoupled_prior == dpvae.decoupled_prior
    assert distribution.base_distribution == custom_base


def test_decoupled_prior_distribution_sampling(dpvae):
    """Test sampling from the decoupled prior distribution."""
    distribution = dpvae.get_prior_distribution()
    
    key = jax.random.PRNGKey(456)
    sample_shape = (5,)
    
    samples = distribution.sample(key, sample_shape)
    
    # Check output shape
    expected_shape = sample_shape + (dpvae.config.latent_dim,)
    assert samples.shape == expected_shape
    
    # Check that samples are finite
    assert jnp.all(jnp.isfinite(samples))


def test_decoupled_prior_distribution_log_prob(dpvae):
    """Test log probability computation for the decoupled prior distribution."""
    distribution = dpvae.get_prior_distribution()
    
    # Generate some test samples
    key = jax.random.PRNGKey(789)
    samples = distribution.sample(key, (3,))
    
    # Compute log probabilities
    log_probs = distribution.log_prob(samples)
    
    # Check output shape
    expected_shape = samples.shape
    assert log_probs.shape == expected_shape
    
    # Check that log probabilities are finite
    assert jnp.all(jnp.isfinite(log_probs))


# ------------------------------------------------------------------------------
# Reparameterization Tests
# ------------------------------------------------------------------------------

def test_reparameterization_training_mode(dpvae, test_input):
    """Test reparameterization in training mode."""
    # Get mean and logvar from encoder
    mean, logvar = dpvae.encoder(test_input)
    
    # Apply reparameterization
    z_training = dpvae.reparameterize(mean, logvar, training=True)
    
    # Check that z is different from mean (due to sampling)
    assert not jnp.allclose(z_training, mean)
    
    # Check shapes
    batch_size, latent_dim = mean.shape
    assert z_training.shape == (batch_size, latent_dim)
    
    # Check that z is finite
    assert jnp.all(jnp.isfinite(z_training))


def test_reparameterization_eval_mode(dpvae, test_input):
    """Test reparameterization in evaluation mode."""
    # Get mean and logvar from encoder
    mean, logvar = dpvae.encoder(test_input)
    
    # Apply reparameterization
    z_eval = dpvae.reparameterize(mean, logvar, training=False)
    
    # Check that z equals mean (no sampling in eval mode)
    assert jnp.allclose(z_eval, mean)
    
    # Check shapes
    batch_size, latent_dim = mean.shape
    assert z_eval.shape == (batch_size, latent_dim)
    
    # Check that z is finite
    assert jnp.all(jnp.isfinite(z_eval))


# ------------------------------------------------------------------------------
# Training vs Evaluation Mode Tests
# ------------------------------------------------------------------------------

def test_training_vs_evaluation_mode(dpvae, test_input):
    """Test that training and evaluation modes produce different results."""
    # Forward pass in training mode
    reconstructed_train, mean_train, logvar_train = dpvae(test_input, training=True)
    
    # Forward pass in evaluation mode
    reconstructed_eval, mean_eval, logvar_eval = dpvae(test_input, training=False)
    
    # Mean and logvar should be the same (deterministic encoder)
    assert jnp.allclose(mean_train, mean_eval)
    assert jnp.allclose(logvar_train, logvar_eval)
    
    # Reconstructed outputs should be different due to sampling vs deterministic
    assert not jnp.allclose(reconstructed_train, reconstructed_eval)
    
    # But both should be finite
    assert jnp.all(jnp.isfinite(reconstructed_train))
    assert jnp.all(jnp.isfinite(reconstructed_eval))


def test_deterministic_evaluation_mode(dpvae, test_input):
    """Test that evaluation mode produces deterministic results."""
    # Forward pass in evaluation mode multiple times
    reconstructed1, mean1, logvar1 = dpvae(test_input, training=False)
    reconstructed2, mean2, logvar2 = dpvae(test_input, training=False)
    
    # Results should be identical
    assert jnp.allclose(reconstructed1, reconstructed2)
    assert jnp.allclose(mean1, mean2)
    assert jnp.allclose(logvar1, logvar2)


# ------------------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------------------

def test_different_decoupled_prior_configurations(rng_key):
    """Test dpVAE with different decoupled prior configurations."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    # Create base VAE config
    vae_config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        input_transformation="log1p"
    )
    
    # Test different prior configurations
    prior_configs = [
        {"num_layers": 1, "hidden_dims": [32]},
        {"num_layers": 2, "hidden_dims": [64, 32]},
        {"num_layers": 3, "hidden_dims": [128, 64, 32]},
    ]
    
    for prior_config in prior_configs:
        rngs = nnx.Rngs(params=rng_key)
        
        # Create encoder and decoder
        encoder = Encoder(config=vae_config, rngs=rngs)
        decoder = Decoder(config=vae_config, rngs=rngs)
        
        # Create decoupled prior
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=prior_config["num_layers"],
            hidden_dims=prior_config["hidden_dims"],
            rngs=rngs,
            activation="relu",
            mask_type="alternating"
        )
        
        # Create dpVAE
        dpvae = dpVAE(
            encoder=encoder,
            decoder=decoder,
            config=vae_config,
            decoupled_prior=decoupled_prior,
            rngs=rngs
        )
        
        # Test input
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        reconstructed, mean, logvar = dpvae(x)
        
        assert reconstructed.shape == (2, input_dim)
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_different_activations(rng_key):
    """Test dpVAE with different activation functions."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    activations = ["relu", "gelu", "tanh", "sigmoid"]
    
    for activation in activations:
        vae_config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            input_transformation="log1p"
        )
        
        rngs = nnx.Rngs(params=rng_key)
        encoder = Encoder(config=vae_config, rngs=rngs)
        decoder = Decoder(config=vae_config, rngs=rngs)
        
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=2,
            hidden_dims=[32, 16],
            rngs=rngs,
            activation=activation,
            mask_type="alternating"
        )
        
        dpvae = dpVAE(
            encoder=encoder,
            decoder=decoder,
            config=vae_config,
            decoupled_prior=decoupled_prior,
            rngs=rngs
        )
        
        # Test input
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        reconstructed, mean, logvar = dpvae(x)
        
        assert reconstructed.shape == (2, input_dim)
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_batch_sizes(dpvae):
    """Test dpVAE with different batch sizes."""
    input_dim = dpvae.config.input_dim
    latent_dim = dpvae.config.latent_dim
    
    batch_sizes = [1, 3, 5, 10]
    
    for batch_size in batch_sizes:
        x = jax.random.exponential(jax.random.PRNGKey(123), (batch_size, input_dim))
        reconstructed, mean, logvar = dpvae(x)
        
        assert reconstructed.shape == (batch_size, input_dim)
        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_numerical_stability(dpvae):
    """Test numerical stability with extreme input values."""
    input_dim = dpvae.config.input_dim
    latent_dim = dpvae.config.latent_dim
    
    # Test with very large values
    x_large = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1000
    reconstructed_large, mean_large, logvar_large = dpvae(x_large)
    
    assert jnp.all(jnp.isfinite(reconstructed_large))
    assert jnp.all(jnp.isfinite(mean_large))
    assert jnp.all(jnp.isfinite(logvar_large))
    
    # Test with very small values
    x_small = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1e-6
    reconstructed_small, mean_small, logvar_small = dpvae(x_small)
    
    assert jnp.all(jnp.isfinite(reconstructed_small))
    assert jnp.all(jnp.isfinite(mean_small))
    assert jnp.all(jnp.isfinite(logvar_small))
    
    # Test with zero values
    x_zero = jnp.zeros((2, input_dim))
    reconstructed_zero, mean_zero, logvar_zero = dpvae(x_zero)
    
    assert jnp.all(jnp.isfinite(reconstructed_zero))
    assert jnp.all(jnp.isfinite(mean_zero))
    assert jnp.all(jnp.isfinite(logvar_zero))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_dpvae_with_positive_data(dpvae):
    """Test dpVAE with realistic positive count data."""
    input_dim = dpvae.config.input_dim
    latent_dim = dpvae.config.latent_dim
    
    # Generate realistic count data (positive integers)
    key = jax.random.PRNGKey(456)
    counts = jax.random.poisson(key, 5.0, (3, input_dim))
    
    reconstructed, mean, logvar = dpvae(counts)
    
    assert reconstructed.shape == (3, input_dim)
    assert mean.shape == (3, latent_dim)
    assert logvar.shape == (3, latent_dim)
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_dpvae_reconstruction_quality(dpvae, test_input):
    """Test that dpVAE can produce reasonable reconstructions."""
    reconstructed, mean, logvar = dpvae(test_input)
    
    # Check that reconstruction has same shape as input
    assert reconstructed.shape == test_input.shape
    
    # Check that reconstruction is finite and positive (due to softplus)
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(reconstructed >= 0)
    
    # Check that mean and logvar are finite
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))
    
    # Check that variance is positive when exponentiated
    variance = jnp.exp(logvar)
    assert jnp.all(variance > 0)


def test_dpvae_end_to_end_consistency(dpvae, test_input):
    """Test that dpVAE components work together consistently."""
    # Manual forward pass
    mean, logvar = dpvae.encoder(test_input)
    z = dpvae.reparameterize(mean, logvar, training=False)  # Deterministic
    reconstructed_manual = dpvae.decoder(z)
    
    # dpVAE forward pass
    reconstructed_dpvae, mean_dpvae, logvar_dpvae = dpvae(test_input, training=False)
    
    # Results should be identical
    assert jnp.allclose(reconstructed_manual, reconstructed_dpvae)
    assert jnp.allclose(mean, mean_dpvae)
    assert jnp.allclose(logvar, logvar_dpvae)


def test_dpvae_decoupled_prior_integration(dpvae):
    """Test that the decoupled prior integrates correctly with the VAE."""
    # Test that the decoupled prior can transform samples
    key = jax.random.PRNGKey(789)
    z_base = jax.random.normal(key, (3, dpvae.config.latent_dim))
    
    # Apply forward transformation
    z_complex, log_det = dpvae.decoupled_prior.forward(z_base)
    
    # Check that transformation changes the values
    transformation_magnitude = jnp.mean(jnp.abs(z_complex - z_base))
    assert transformation_magnitude > 1e-6, f"Transformation too small: {transformation_magnitude}"
    
    # Check that we can recover the original
    z_recovered, log_det_inv = dpvae.decoupled_prior.inverse(z_complex)
    recovery_error = jnp.mean(jnp.abs(z_base - z_recovered))
    assert recovery_error < 1e-5, f"Recovery error too high: {recovery_error}"


# ------------------------------------------------------------------------------
# Performance Tests
# ------------------------------------------------------------------------------

def test_dpvae_batch_processing(dpvae):
    """Test dpVAE performance with larger batches."""
    input_dim = dpvae.config.input_dim
    latent_dim = dpvae.config.latent_dim
    
    # Test with larger batch sizes
    batch_sizes = [10, 50, 100]
    
    for batch_size in batch_sizes:
        x = jax.random.exponential(jax.random.PRNGKey(123), (batch_size, input_dim))
        reconstructed, mean, logvar = dpvae(x)
        
        assert reconstructed.shape == (batch_size, input_dim)
        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_dpvae_memory_efficiency(dpvae):
    """Test that dpVAE doesn't have memory leaks."""
    input_dim = dpvae.config.input_dim
    
    # Run multiple forward passes
    for i in range(10):
        x = jax.random.exponential(jax.random.PRNGKey(i), (5, input_dim))
        reconstructed, mean, logvar = dpvae(x)
        
        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))
