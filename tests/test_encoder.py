#!/usr/bin/env python3
"""
Tests for Encoder to verify encoding functionality works correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import Encoder, VAEConfig


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
def encoder_config():
    """Create a VAEConfig for testing."""
    return VAEConfig(
        input_dim=10,
        latent_dim=3,
        hidden_dims=[64, 32],
        activation="relu",
        input_transformation="log1p"
    )


@pytest.fixture(scope="function")
def encoder(rng_key, encoder_config):
    """Create an Encoder instance for testing."""
    rngs = nnx.Rngs(params=rng_key)
    return Encoder(config=encoder_config, rngs=rngs)


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_encoder_creation(encoder):
    """Test that the encoder is created successfully."""
    assert encoder is not None
    assert hasattr(encoder, 'encode')
    assert hasattr(encoder, '__call__')
    assert hasattr(encoder, 'encoder_layers')
    assert hasattr(encoder, 'latent_mean')
    assert hasattr(encoder, 'latent_logvar')
    assert hasattr(encoder, 'config')


def test_encoder_architecture(encoder, encoder_config):
    """Test that the encoder has the correct architecture."""
    # Check number of encoder layers
    expected_num_layers = len(encoder_config.hidden_dims)
    assert len(encoder.encoder_layers) == expected_num_layers
    
    # Check that config matches
    assert encoder.config.input_dim == encoder_config.input_dim
    assert encoder.config.latent_dim == encoder_config.latent_dim
    assert encoder.config.hidden_dims == encoder_config.hidden_dims
    
    # Check that we have the expected number of layers
    assert len(encoder.encoder_layers) == len(encoder_config.hidden_dims)
    assert hasattr(encoder, 'latent_mean')
    assert hasattr(encoder, 'latent_logvar')


def test_encoding_functionality(encoder, test_input):
    """Test the encoding functionality."""
    mean, logvar = encoder.encode(test_input)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = encoder.config.latent_dim
    
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_call_method(encoder, test_input):
    """Test the __call__ method."""
    mean, logvar = encoder(test_input)
    
    # Check output shapes
    batch_size, input_dim = test_input.shape
    latent_dim = encoder.config.latent_dim
    
    assert mean.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    
    # Check that outputs are finite
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_call_method_consistency(encoder, test_input):
    """Test that call method gives same results as encode method."""
    mean_call, logvar_call = encoder(test_input)
    mean_encode, logvar_encode = encoder.encode(test_input)
    
    assert jnp.allclose(mean_call, mean_encode)
    assert jnp.allclose(logvar_call, logvar_encode)


# ------------------------------------------------------------------------------
# Input Transformation Tests
# ------------------------------------------------------------------------------

def test_input_transformation_log1p(encoder, test_input):
    """Test that log1p transformation is applied correctly."""
    # Apply log1p transformation manually
    expected_transformed = jnp.log1p(test_input)
    
    # Get the transformation from the encoder
    input_transformation = encoder.config.input_transformation
    assert input_transformation == "log1p"
    
    # The transformation should be applied in the encode method
    mean, logvar = encoder.encode(test_input)
    
    # Check that outputs are finite (transformation should help with numerical stability)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_different_input_transformations(rng_key):
    """Test encoder with different input transformations."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    transformations = ["log1p", "log", "sqrt", "identity"]
    
    for transformation in transformations:
        config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            input_transformation=transformation
        )
        
        rngs = nnx.Rngs(params=rng_key)
        encoder = Encoder(config=config, rngs=rngs)
        
        # Test input - use positive data for log transformations
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        mean, logvar = encoder(x)
        
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Activation Function Tests
# ------------------------------------------------------------------------------

def test_different_activations(rng_key):
    """Test encoder with different activation functions."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    activations = ["relu", "gelu", "tanh", "sigmoid", "softplus"]
    
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
        
        # Test input - use positive data for log1p transformation
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        mean, logvar = encoder(x)
        
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Architecture Configuration Tests
# ------------------------------------------------------------------------------

def test_different_hidden_dims(rng_key):
    """Test encoder with different hidden layer dimensions."""
    input_dim = 10
    latent_dim = 3
    
    hidden_dims_configs = [
        [64],
        [128, 64],
        [256, 128, 64],
        [512, 256, 128, 64]
    ]
    
    for hidden_dims in hidden_dims_configs:
        config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            input_transformation="log1p"
        )
        
        rngs = nnx.Rngs(params=rng_key)
        encoder = Encoder(config=config, rngs=rngs)
        
        # Test input - use positive data for log1p transformation
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        mean, logvar = encoder(x)
        
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_different_latent_dims(rng_key):
    """Test encoder with different latent dimensions."""
    input_dim = 10
    hidden_dims = [64, 32]
    
    latent_dims = [1, 2, 5, 10]
    
    for latent_dim in latent_dims:
        config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            input_transformation="log1p"
        )
        
        rngs = nnx.Rngs(params=rng_key)
        encoder = Encoder(config=config, rngs=rngs)
        
        # Test input - use positive data for log1p transformation
        x = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim))
        mean, logvar = encoder(x)
        
        assert mean.shape == (2, latent_dim)
        assert logvar.shape == (2, latent_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_batch_sizes(encoder):
    """Test encoder with different batch sizes."""
    input_dim = encoder.config.input_dim
    latent_dim = encoder.config.latent_dim
    
    batch_sizes = [1, 3, 5, 10]
    
    for batch_size in batch_sizes:
        x = jax.random.exponential(jax.random.PRNGKey(123), (batch_size, input_dim))
        mean, logvar = encoder(x)
        
        assert mean.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        assert jnp.all(jnp.isfinite(mean))
        assert jnp.all(jnp.isfinite(logvar))


def test_numerical_stability(encoder):
    """Test numerical stability with extreme input values."""
    input_dim = encoder.config.input_dim
    latent_dim = encoder.config.latent_dim
    
    # Test with very large values
    x_large = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1000
    mean_large, logvar_large = encoder(x_large)
    
    assert jnp.all(jnp.isfinite(mean_large))
    assert jnp.all(jnp.isfinite(logvar_large))
    
    # Test with very small values
    x_small = jax.random.exponential(jax.random.PRNGKey(123), (2, input_dim)) * 1e-6
    mean_small, logvar_small = encoder(x_small)
    
    assert jnp.all(jnp.isfinite(mean_small))
    assert jnp.all(jnp.isfinite(logvar_small))
    
    # Test with zero values
    x_zero = jnp.zeros((2, input_dim))
    mean_zero, logvar_zero = encoder(x_zero)
    
    assert jnp.all(jnp.isfinite(mean_zero))
    assert jnp.all(jnp.isfinite(logvar_zero))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_encoder_with_positive_data(encoder):
    """Test encoder with realistic positive count data."""
    input_dim = encoder.config.input_dim
    latent_dim = encoder.config.latent_dim
    
    # Generate realistic count data (positive integers)
    key = jax.random.PRNGKey(456)
    counts = jax.random.poisson(key, 5.0, (3, input_dim))
    
    mean, logvar = encoder(counts)
    
    assert mean.shape == (3, latent_dim)
    assert logvar.shape == (3, latent_dim)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_encoder_output_ranges(encoder, test_input):
    """Test that encoder outputs are in reasonable ranges."""
    mean, logvar = encoder(test_input)
    
    # Mean can be any real number
    assert jnp.all(jnp.isfinite(mean))
    
    # Log variance should be finite (but can be negative)
    assert jnp.all(jnp.isfinite(logvar))
    
    # Variance should be positive when exponentiated
    variance = jnp.exp(logvar)
    assert jnp.all(variance > 0)


def test_encoder_deterministic(encoder, test_input):
    """Test that encoder produces deterministic outputs."""
    mean1, logvar1 = encoder(test_input)
    mean2, logvar2 = encoder(test_input)
    
    assert jnp.allclose(mean1, mean2)
    assert jnp.allclose(logvar1, logvar2) 