#!/usr/bin/env python3
"""
Tests for Decoder to verify decoding functionality works correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import Decoder, VAEConfig, Encoder


@pytest.fixture(scope="session")
def rng_key():
    """Provide a consistent random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def test_input():
    """Generate test input data."""
    key = jax.random.PRNGKey(123)
    latent_dim = 3
    batch_size = 4
    # Use normal distribution for latent space (can be negative)
    return jax.random.normal(key, (batch_size, latent_dim))


@pytest.fixture(scope="function")
def decoder_config():
    """Create a VAEConfig for testing."""
    return VAEConfig(
        input_dim=10,
        latent_dim=3,
        hidden_dims=[64, 32],
        activation="relu",
        input_transformation="log1p"
    )


@pytest.fixture(scope="function")
def decoder(rng_key, decoder_config):
    """Create a Decoder instance for testing."""
    rngs = nnx.Rngs(params=rng_key)
    return Decoder(config=decoder_config, rngs=rngs)


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_decoder_creation(decoder):
    """Test that the decoder is created successfully."""
    assert decoder is not None
    assert hasattr(decoder, 'decode')
    assert hasattr(decoder, '__call__')
    assert hasattr(decoder, 'decoder_layers')
    assert hasattr(decoder, 'decoder_output')
    assert hasattr(decoder, 'config')


def test_decoder_architecture(decoder, decoder_config):
    """Test that the decoder has the correct architecture."""
    # Check number of decoder layers (reverse of encoder)
    expected_num_layers = len(decoder_config.hidden_dims)
    assert len(decoder.decoder_layers) == expected_num_layers
    
    # Check that config matches
    assert decoder.config.input_dim == decoder_config.input_dim
    assert decoder.config.latent_dim == decoder_config.latent_dim
    assert decoder.config.hidden_dims == decoder_config.hidden_dims
    
    # Check that we have the expected number of layers
    assert len(decoder.decoder_layers) == len(decoder_config.hidden_dims)
    assert hasattr(decoder, 'decoder_output')


def test_decoding_functionality(decoder, test_input):
    """Test the decoding functionality."""
    output = decoder.decode(test_input)
    
    # Check output shapes
    batch_size, latent_dim = test_input.shape
    input_dim = decoder.config.input_dim
    
    assert output.shape == (batch_size, input_dim)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(output))


def test_call_method(decoder, test_input):
    """Test the __call__ method."""
    output = decoder(test_input)
    
    # Check output shapes
    batch_size, latent_dim = test_input.shape
    input_dim = decoder.config.input_dim
    
    assert output.shape == (batch_size, input_dim)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(output))


def test_call_method_consistency(decoder, test_input):
    """Test that call method gives same results as decode method."""
    output_call = decoder(test_input)
    output_decode = decoder.decode(test_input)
    
    assert jnp.allclose(output_call, output_decode)


# ------------------------------------------------------------------------------
# Activation Function Tests
# ------------------------------------------------------------------------------

def test_different_activations(rng_key):
    """Test decoder with different activation functions."""
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
        decoder = Decoder(config=config, rngs=rngs)
        
        # Test input
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        output = decoder(z)
        
        assert output.shape == (2, input_dim)
        assert jnp.all(jnp.isfinite(output))


# ------------------------------------------------------------------------------
# Architecture Configuration Tests
# ------------------------------------------------------------------------------

def test_different_hidden_dims(rng_key):
    """Test decoder with different hidden layer dimensions."""
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
        decoder = Decoder(config=config, rngs=rngs)
        
        # Test input
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        output = decoder(z)
        
        assert output.shape == (2, input_dim)
        assert jnp.all(jnp.isfinite(output))


def test_different_latent_dims(rng_key):
    """Test decoder with different latent dimensions."""
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
        decoder = Decoder(config=config, rngs=rngs)
        
        # Test input
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        output = decoder(z)
        
        assert output.shape == (2, input_dim)
        assert jnp.all(jnp.isfinite(output))


def test_different_input_dims(rng_key):
    """Test decoder with different input dimensions."""
    latent_dim = 3
    hidden_dims = [64, 32]
    
    input_dims = [5, 10, 20, 50]
    
    for input_dim in input_dims:
        config = VAEConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            input_transformation="log1p"
        )
        
        rngs = nnx.Rngs(params=rng_key)
        decoder = Decoder(config=config, rngs=rngs)
        
        # Test input
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        output = decoder(z)
        
        assert output.shape == (2, input_dim)
        assert jnp.all(jnp.isfinite(output))


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_batch_sizes(decoder):
    """Test decoder with different batch sizes."""
    latent_dim = decoder.config.latent_dim
    input_dim = decoder.config.input_dim
    
    batch_sizes = [1, 3, 5, 10]
    
    for batch_size in batch_sizes:
        z = jax.random.normal(jax.random.PRNGKey(123), (batch_size, latent_dim))
        output = decoder(z)
        
        assert output.shape == (batch_size, input_dim)
        assert jnp.all(jnp.isfinite(output))


def test_numerical_stability(decoder):
    """Test numerical stability with extreme input values."""
    latent_dim = decoder.config.latent_dim
    input_dim = decoder.config.input_dim
    
    # Test with very large values
    z_large = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim)) * 100
    output_large = decoder(z_large)
    
    assert jnp.all(jnp.isfinite(output_large))
    
    # Test with very small values
    z_small = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim)) * 1e-6
    output_small = decoder(z_small)
    
    assert jnp.all(jnp.isfinite(output_small))
    
    # Test with zero values
    z_zero = jnp.zeros((2, latent_dim))
    output_zero = decoder(z_zero)
    
    assert jnp.all(jnp.isfinite(output_zero))


def test_extreme_latent_values(decoder):
    """Test decoder with extreme latent space values."""
    latent_dim = decoder.config.latent_dim
    input_dim = decoder.config.input_dim
    
    # Test with very large positive values
    z_large_pos = jnp.ones((2, latent_dim)) * 1000
    output_large_pos = decoder(z_large_pos)
    
    assert jnp.all(jnp.isfinite(output_large_pos))
    
    # Test with very large negative values
    z_large_neg = jnp.ones((2, latent_dim)) * -1000
    output_large_neg = decoder(z_large_neg)
    
    assert jnp.all(jnp.isfinite(output_large_neg))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_decoder_with_realistic_latent_data(decoder):
    """Test decoder with realistic latent space data."""
    latent_dim = decoder.config.latent_dim
    input_dim = decoder.config.input_dim
    
    # Generate realistic latent data (normal distribution)
    key = jax.random.PRNGKey(456)
    z = jax.random.normal(key, (3, latent_dim))
    
    output = decoder(z)
    
    assert output.shape == (3, input_dim)
    assert jnp.all(jnp.isfinite(output))


def test_decoder_output_ranges(decoder, test_input):
    """Test that decoder outputs are in reasonable ranges."""
    output = decoder(test_input)
    
    # Output should be finite
    assert jnp.all(jnp.isfinite(output))
    
    # With softplus activation, output should be positive
    assert jnp.all(output >= 0)


def test_decoder_deterministic(decoder, test_input):
    """Test that decoder produces deterministic outputs."""
    output1 = decoder(test_input)
    output2 = decoder(test_input)
    
    assert jnp.allclose(output1, output2)


def test_decoder_symmetry_with_encoder(rng_key):
    """Test that decoder architecture is symmetric with encoder."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [64, 32]
    
    # Create encoder and decoder with same config
    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        input_transformation="log1p"
    )
    
    rngs = nnx.Rngs(params=rng_key)
    encoder = Encoder(config=config, rngs=rngs)
    decoder = Decoder(config=config, rngs=rngs)
    
    # Test that they have symmetric architectures
    assert len(encoder.encoder_layers) == len(decoder.decoder_layers)
    
    # Test that dimensions are symmetric
    assert encoder.encoder_layers[0].in_features == decoder.decoder_output.out_features
    assert encoder.latent_mean.out_features == decoder.decoder_layers[0].in_features


# ------------------------------------------------------------------------------
# Performance Tests
# ------------------------------------------------------------------------------

def test_decoder_batch_processing(decoder):
    """Test decoder performance with larger batches."""
    latent_dim = decoder.config.latent_dim
    input_dim = decoder.config.input_dim
    
    # Test with larger batch sizes
    batch_sizes = [10, 50, 100]
    
    for batch_size in batch_sizes:
        z = jax.random.normal(jax.random.PRNGKey(123), (batch_size, latent_dim))
        output = decoder(z)
        
        assert output.shape == (batch_size, input_dim)
        assert jnp.all(jnp.isfinite(output))
        assert jnp.all(output >= 0) 