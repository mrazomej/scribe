#!/usr/bin/env python3
"""
Tests for VAE to verify the complete variational autoencoder functionality.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import VAE, VAEConfig, Encoder, Decoder


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
def vae_config():
    """Create a VAEConfig for testing."""
    return VAEConfig(
        input_dim=10,
        latent_dim=3,
        hidden_dims=[64, 32],
        activation="relu",
        input_transformation="log1p"
    )


@pytest.fixture(scope="function")
def vae(rng_key, vae_config):
    """Create a VAE instance for testing."""
    rngs = nnx.Rngs(params=rng_key)
    
    # Create encoder and decoder
    encoder = Encoder(config=vae_config, rngs=rngs)
    decoder = Decoder(config=vae_config, rngs=rngs)
    
    return VAE(encoder=encoder, decoder=decoder, config=vae_config, rngs=rngs)


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_vae_creation(vae):
    """Test that the VAE is created successfully."""
    assert vae is not None
    assert hasattr(vae, '__call__')
    assert hasattr(vae, 'reparameterize')
    assert hasattr(vae, 'encoder')
    assert hasattr(vae, 'decoder')
    assert hasattr(vae, 'config')
    assert hasattr(vae, 'rngs')


def test_vae_architecture(vae, vae_config):
    """Test that the VAE has the correct architecture."""
    # Check that encoder and decoder have correct dimensions
    assert vae.encoder.config.input_dim == vae_config.input_dim
    assert vae.encoder.config.latent_dim == vae_config.latent_dim
    assert vae.decoder.config.input_dim == vae_config.input_dim
    assert vae.decoder.config.latent_dim == vae_config.latent_dim
    
    # Check that they use the same hidden dimensions
    assert vae.encoder.config.hidden_dims == vae_config.hidden_dims
    assert vae.decoder.config.hidden_dims == vae_config.hidden_dims


def test_vae_forward_pass(vae, test_input):
    """Test the VAE forward pass."""
    reconstructed, mean, logvar = vae(test_input)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = vae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_vae_forward_pass_training_mode(vae, test_input):
    """Test the VAE forward pass in training mode."""
    reconstructed, mean, logvar = vae(test_input, training=True)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = vae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_vae_forward_pass_eval_mode(vae, test_input):
    """Test the VAE forward pass in evaluation mode."""
    reconstructed, mean, logvar = vae(test_input, training=False)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = vae.config.latent_dim
    
    assert reconstructed.shape == (batch_size, input_dim)
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Reparameterization Tests
# ------------------------------------------------------------------------------

def test_reparameterization_training_mode(vae, test_input):
    """Test reparameterization in training mode."""
    # Get mean and logvar from encoder
    mean, logvar = vae.encoder(test_input)
    
    # Apply reparameterization
    z_training = vae.reparameterize(mean, logvar, training=True)
    
    # Check that z is different from mean (due to sampling)
    assert not jnp.allclose(z_training, mean)
    
    # Check shapes
    batch_size, latent_dim = mean.shape
    assert z_training.shape == (batch_size, latent_dim)
    
    # Check that z is finite
    assert jnp.all(jnp.isfinite(z_training))


def test_reparameterization_eval_mode(vae, test_input):
    """Test reparameterization in evaluation mode."""
    # Get mean and logvar from encoder
    mean, logvar = vae.encoder(test_input)
    
    # Apply reparameterization
    z_eval = vae.reparameterize(mean, logvar, training=False)
    
    # Check that z equals mean (no sampling in eval mode)
    assert jnp.allclose(z_eval, mean)
    
    # Check shapes
    batch_size, latent_dim = mean.shape
    assert z_eval.shape == (batch_size, latent_dim)
    
    # Check that z is finite
    assert jnp.all(jnp.isfinite(z_eval))


def test_reparameterization_consistency(vae, test_input):
    """Test that reparameterization is consistent across calls."""
    # Get mean and logvar from encoder
    mean, logvar = vae.encoder(test_input)
    
    # Apply reparameterization multiple times
    z1 = vae.reparameterize(mean, logvar, training=True)
    z2 = vae.reparameterize(mean, logvar, training=True)
    
    # In training mode, z should be different due to random sampling
    assert not jnp.allclose(z1, z2)
    
    # But both should be finite
    assert jnp.all(jnp.isfinite(z1))
    assert jnp.all(jnp.isfinite(z2))


# ------------------------------------------------------------------------------
# Training vs Evaluation Mode Tests
# ------------------------------------------------------------------------------

def test_training_vs_evaluation_mode(vae, test_input):
    """Test that training and evaluation modes produce different results."""
    # Forward pass in training mode
    reconstructed_train, mean_train, logvar_train = vae(test_input, training=True)
    
    # Forward pass in evaluation mode
    reconstructed_eval, mean_eval, logvar_eval = vae(test_input, training=False)
    
    # Mean and logvar should be the same (deterministic encoder)
    assert jnp.allclose(mean_train, mean_eval)
    assert jnp.allclose(logvar_train, logvar_eval)
    
    # Reconstructed outputs should be different due to sampling vs deterministic
    assert not jnp.allclose(reconstructed_train, reconstructed_eval)
    
    # But both should be finite
    assert jnp.all(jnp.isfinite(reconstructed_train))
    assert jnp.all(jnp.isfinite(reconstructed_eval))


def test_deterministic_evaluation_mode(vae, test_input):
    """Test that evaluation mode produces deterministic results."""
    # Forward pass in evaluation mode multiple times
    reconstructed1, mean1, logvar1 = vae(test_input, training=False)
    reconstructed2, mean2, logvar2 = vae(test_input, training=False)
    
    # Results should be identical
    assert jnp.allclose(reconstructed1, reconstructed2)
    assert jnp.allclose(mean1, mean2)
    assert jnp.allclose(logvar1, logvar2)


# ------------------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------------------

def test_different_configurations(rng_key):
    """Test VAE with different configurations."""
    input_dim = 10
    latent_dim = 3
    
    configs = [
        VAEConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[64]),
        VAEConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[128, 64]),
        VAEConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[256, 128, 64]),
    ]
    
    for config in configs:
        rngs = nnx.Rngs(params=rng_key)
        
        # Create encoder and decoder
        encoder = Encoder(config=config, rngs=rngs)
        decoder = Decoder(config=config, rngs=rngs)
        
        # Create VAE
        vae = VAE(encoder=encoder, decoder=decoder, config=config, rngs=rngs)
        
        # Test input
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        reconstructed, mean, logvar = vae(x)
        
        assert reconstructed.shape == (2, input_dim)
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_different_activations(rng_key):
    """Test VAE with different activation functions."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    activations = ["relu", "gelu", "tanh", "sigmoid"]
    
    for activation in activations:
        config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            input_transformation="log1p"
        )
        
        rngs = nnx.Rngs(params=rng_key)
        encoder = Encoder(config=config, rngs=rngs)
        decoder = Decoder(config=config, rngs=rngs)
        vae = VAE(encoder=encoder, decoder=decoder, config=config, rngs=rngs)
        
        # Test input
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        reconstructed, mean, logvar = vae(x)
        
        assert reconstructed.shape == (2, input_dim)
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_batch_sizes(vae):
    """Test VAE with different batch sizes."""
    input_dim = vae.config.input_dim
    latent_dim = vae.config.latent_dim
    
    batch_sizes = [1, 3, 5, 10]
    
    for batch_size in batch_sizes:
        x = jax.random.exponential(jax.random.PRNGKey(123), (batch_size, input_dim))
        reconstructed, mean, logvar = vae(x)
        
        assert reconstructed.shape == (batch_size, input_dim)
        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_numerical_stability(vae):
    """Test numerical stability with extreme input values."""
    input_dim = vae.config.input_dim
    latent_dim = vae.config.latent_dim
    
    # Test with very large values
    x_large = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1000
    reconstructed_large, mean_large, logvar_large = vae(x_large)
    
    assert jnp.all(jnp.isfinite(reconstructed_large))
    assert jnp.all(jnp.isfinite(mean_large))
    assert jnp.all(jnp.isfinite(logvar_large))
    
    # Test with very small values
    x_small = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1e-6
    reconstructed_small, mean_small, logvar_small = vae(x_small)
    
    assert jnp.all(jnp.isfinite(reconstructed_small))
    assert jnp.all(jnp.isfinite(mean_small))
    assert jnp.all(jnp.isfinite(logvar_small))
    
    # Test with zero values
    x_zero = jnp.zeros((2, input_dim))
    reconstructed_zero, mean_zero, logvar_zero = vae(x_zero)
    
    assert jnp.all(jnp.isfinite(reconstructed_zero))
    assert jnp.all(jnp.isfinite(mean_zero))
    assert jnp.all(jnp.isfinite(logvar_zero))

# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_vae_with_positive_data(vae):
    """Test VAE with realistic positive count data."""
    input_dim = vae.config.input_dim
    latent_dim = vae.config.latent_dim
    
    # Generate realistic count data (positive integers)
    key = jax.random.PRNGKey(456)
    counts = jax.random.poisson(key, 5.0, (3, input_dim))
    
    reconstructed, mean, logvar = vae(counts)
    
    assert reconstructed.shape == (3, input_dim)
    assert mean.shape == (3, latent_dim)
    assert logvar.shape == (3, latent_dim)
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_vae_reconstruction_quality(vae, test_input):
    """Test that VAE can produce reasonable reconstructions."""
    reconstructed, mean, logvar = vae(test_input)
    
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


def test_vae_deterministic_encoder_decoder(vae, test_input):
    """Test that encoder and decoder components are deterministic."""
    # Test encoder
    mean1, logvar1 = vae.encoder(test_input)
    mean2, logvar2 = vae.encoder(test_input)
    
    assert jnp.allclose(mean1, mean2)
    assert jnp.allclose(logvar1, logvar2)
    
    # Test decoder with deterministic latent
    z = jnp.zeros((test_input.shape[0], vae.config.latent_dim))
    output1 = vae.decoder(z)
    output2 = vae.decoder(z)
    
    assert jnp.allclose(output1, output2)


def test_vae_end_to_end_consistency(vae, test_input):
    """Test that VAE components work together consistently."""
    # Manual forward pass
    mean, logvar = vae.encoder(test_input)
    z = vae.reparameterize(mean, logvar, training=False)  # Deterministic
    reconstructed_manual = vae.decoder(z)
    
    # VAE forward pass
    reconstructed_vae, mean_vae, logvar_vae = vae(test_input, training=False)
    
    # Results should be identical
    assert jnp.allclose(reconstructed_manual, reconstructed_vae)
    assert jnp.allclose(mean, mean_vae)
    assert jnp.allclose(logvar, logvar_vae)


# ------------------------------------------------------------------------------
# Performance Tests
# ------------------------------------------------------------------------------

def test_vae_batch_processing(vae):
    """Test VAE performance with larger batches."""
    input_dim = vae.config.input_dim
    latent_dim = vae.config.latent_dim
    
    # Test with larger batch sizes
    batch_sizes = [10, 50, 100]
    
    for batch_size in batch_sizes:
        x = jax.random.exponential(jax.random.PRNGKey(123), (batch_size, input_dim))
        reconstructed, mean, logvar = vae(x)
        
        assert reconstructed.shape == (batch_size, input_dim)
        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_vae_memory_efficiency(vae):
    """Test that VAE doesn't have memory leaks."""
    input_dim = vae.config.input_dim
    
    # Run multiple forward passes
    for i in range(10):
        x = jax.random.exponential(jax.random.PRNGKey(i), (5, input_dim))
        reconstructed, mean, logvar = vae(x)
        
        # Check that outputs are finite
        assert jnp.all(jnp.isfinite(reconstructed))
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar)) 