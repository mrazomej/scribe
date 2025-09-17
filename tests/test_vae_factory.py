#!/usr/bin/env python3
"""
Tests for VAE factory functions to verify they create correct architectures.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import (
    create_encoder, create_decoder, create_vae, create_dpvae,
    VAEConfig, Encoder, Decoder, VAE, dpVAE, DecoupledPrior
)


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


# ------------------------------------------------------------------------------
# Encoder Factory Tests
# ------------------------------------------------------------------------------

def test_create_encoder_basic():
    """Test basic encoder creation."""
    input_dim = 10
    latent_dim = 3
    
    encoder = create_encoder(input_dim=input_dim, latent_dim=latent_dim)
    
    assert isinstance(encoder, Encoder)
    assert encoder.config.input_dim == input_dim
    assert encoder.config.latent_dim == latent_dim
    assert encoder.config.hidden_dims == [256, 256]  # Default
    assert encoder.config.activation == "relu"  # Default
    assert encoder.config.input_transformation == "log1p"  # Default


def test_create_encoder_custom_config():
    """Test encoder creation with custom configuration."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64, 32]
    activation = "gelu"
    input_transformation = "sqrt"
    
    encoder = create_encoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation
    )
    
    assert isinstance(encoder, Encoder)
    assert encoder.config.input_dim == input_dim
    assert encoder.config.latent_dim == latent_dim
    assert encoder.config.hidden_dims == hidden_dims
    assert encoder.config.activation == activation
    assert encoder.config.input_transformation == input_transformation


def test_create_encoder_functionality(test_input):
    """Test that created encoder works correctly."""
    input_dim = test_input.shape[1]
    latent_dim = 3
    
    encoder = create_encoder(input_dim=input_dim, latent_dim=latent_dim)
    
    mean, logvar = encoder(test_input)
    
    assert mean.shape == (test_input.shape[0], latent_dim)
    assert logvar.shape == (test_input.shape[0], latent_dim)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_create_encoder_different_configurations():
    """Test encoder creation with different configurations."""
    input_dim = 10
    
    configs = [
        {"latent_dim": 1, "hidden_dims": [64]},
        {"latent_dim": 2, "hidden_dims": [128, 64]},
        {"latent_dim": 5, "hidden_dims": [256, 128, 64]},
        {"latent_dim": 10, "hidden_dims": [512, 256, 128, 64]},
    ]
    
    for config in configs:
        encoder = create_encoder(input_dim=input_dim, **config)
        
        assert isinstance(encoder, Encoder)
        assert encoder.config.input_dim == input_dim
        assert encoder.config.latent_dim == config["latent_dim"]
        assert encoder.config.hidden_dims == config["hidden_dims"]


# ------------------------------------------------------------------------------
# Decoder Factory Tests
# ------------------------------------------------------------------------------

def test_create_decoder_basic():
    """Test basic decoder creation."""
    input_dim = 10
    latent_dim = 3
    
    decoder = create_decoder(input_dim=input_dim, latent_dim=latent_dim)
    
    assert isinstance(decoder, Decoder)
    assert decoder.config.input_dim == input_dim
    assert decoder.config.latent_dim == latent_dim
    assert decoder.config.hidden_dims == [256, 256]  # Default
    assert decoder.config.activation == "relu"  # Default
    assert decoder.config.input_transformation == "log1p"  # Default


def test_create_decoder_custom_config():
    """Test decoder creation with custom configuration."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64, 32]
    activation = "gelu"
    input_transformation = "sqrt"
    
    decoder = create_decoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation
    )
    
    assert isinstance(decoder, Decoder)
    assert decoder.config.input_dim == input_dim
    assert decoder.config.latent_dim == latent_dim
    assert decoder.config.hidden_dims == hidden_dims
    assert decoder.config.activation == activation
    assert decoder.config.input_transformation == input_transformation


def test_create_decoder_functionality():
    """Test that created decoder works correctly."""
    input_dim = 10
    latent_dim = 3
    batch_size = 4
    
    decoder = create_decoder(input_dim=input_dim, latent_dim=latent_dim)
    
    z = jax.random.normal(jax.random.PRNGKey(123), (batch_size, latent_dim))
    output = decoder(z)
    
    assert output.shape == (batch_size, input_dim)
    assert jnp.all(jnp.isfinite(output))


def test_create_decoder_different_configurations():
    """Test decoder creation with different configurations."""
    input_dim = 10
    
    configs = [
        {"latent_dim": 1, "hidden_dims": [64]},
        {"latent_dim": 2, "hidden_dims": [128, 64]},
        {"latent_dim": 5, "hidden_dims": [256, 128, 64]},
        {"latent_dim": 10, "hidden_dims": [512, 256, 128, 64]},
    ]
    
    for config in configs:
        decoder = create_decoder(input_dim=input_dim, **config)
        
        assert isinstance(decoder, Decoder)
        assert decoder.config.input_dim == input_dim
        assert decoder.config.latent_dim == config["latent_dim"]
        assert decoder.config.hidden_dims == config["hidden_dims"]


# ------------------------------------------------------------------------------
# VAE Factory Tests
# ------------------------------------------------------------------------------

def test_create_vae_basic():
    """Test basic VAE creation."""
    input_dim = 10
    latent_dim = 3
    
    vae = create_vae(input_dim=input_dim, latent_dim=latent_dim)
    
    assert isinstance(vae, VAE)
    assert vae.config.input_dim == input_dim
    assert vae.config.latent_dim == latent_dim
    assert vae.config.hidden_dims == [256, 256]  # Default
    assert vae.config.activation == "relu"  # Default
    assert vae.config.input_transformation == "log1p"  # Default


def test_create_vae_custom_config():
    """Test VAE creation with custom configuration."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64, 32]
    activation = "gelu"
    input_transformation = "sqrt"
    
    vae = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation
    )
    
    assert isinstance(vae, VAE)
    assert vae.config.input_dim == input_dim
    assert vae.config.latent_dim == latent_dim
    assert vae.config.hidden_dims == hidden_dims
    assert vae.config.activation == activation
    assert vae.config.input_transformation == input_transformation


def test_create_vae_functionality(test_input):
    """Test that created VAE works correctly."""
    input_dim = test_input.shape[1]
    latent_dim = 3
    
    vae = create_vae(input_dim=input_dim, latent_dim=latent_dim)
    
    reconstructed, mean, logvar = vae(test_input)
    
    assert reconstructed.shape == test_input.shape
    assert mean.shape == (test_input.shape[0], latent_dim)
    assert logvar.shape == (test_input.shape[0], latent_dim)
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_create_vae_different_configurations():
    """Test VAE creation with different configurations."""
    input_dim = 10
    
    configs = [
        {"latent_dim": 1, "hidden_dims": [64]},
        {"latent_dim": 2, "hidden_dims": [128, 64]},
        {"latent_dim": 5, "hidden_dims": [256, 128, 64]},
        {"latent_dim": 10, "hidden_dims": [512, 256, 128, 64]},
    ]
    
    for config in configs:
        vae = create_vae(input_dim=input_dim, **config)
        
        assert isinstance(vae, VAE)
        assert vae.config.input_dim == input_dim
        assert vae.config.latent_dim == config["latent_dim"]
        assert vae.config.hidden_dims == config["hidden_dims"]


def test_create_vae_encoder_decoder_symmetry():
    """Test that VAE encoder and decoder are symmetric."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64]
    
    vae = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # Check that encoder and decoder have symmetric architectures
    assert len(vae.encoder.encoder_layers) == len(vae.decoder.decoder_layers)
    assert vae.encoder.config.hidden_dims == vae.decoder.config.hidden_dims
    
    # Check that configs are symmetric
    assert vae.encoder.config.input_dim == vae.decoder.config.input_dim
    assert vae.encoder.config.latent_dim == vae.decoder.config.latent_dim


# ------------------------------------------------------------------------------
# dpVAE Factory Tests
# ------------------------------------------------------------------------------

def test_create_dpvae_basic():
    """Test basic dpVAE creation."""
    input_dim = 10
    latent_dim = 3
    
    dpvae = create_dpvae(input_dim=input_dim, latent_dim=latent_dim)
    
    assert isinstance(dpvae, dpVAE)
    assert dpvae.config.input_dim == input_dim
    assert dpvae.config.latent_dim == latent_dim
    assert dpvae.config.hidden_dims == [256, 256]  # Default
    assert dpvae.config.activation == "relu"  # Default
    assert dpvae.config.input_transformation == "log1p"  # Default
    
    # Check decoupled prior
    assert isinstance(dpvae.decoupled_prior, DecoupledPrior)
    assert dpvae.decoupled_prior.latent_dim == latent_dim
    assert dpvae.decoupled_prior.num_layers == 2  # Default from hidden_dims length
    assert dpvae.decoupled_prior.hidden_dims == [64, 64]  # Default


def test_create_dpvae_custom_config():
    """Test dpVAE creation with custom configuration."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64, 32]
    activation = "gelu"
    input_transformation = "sqrt"
    prior_hidden_dims = [128, 64]
    prior_activation = "tanh"
    prior_mask_type = "checkerboard"
    
    dpvae = create_dpvae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        input_transformation=input_transformation,
        prior_hidden_dims=prior_hidden_dims,
        prior_activation=prior_activation,
        prior_mask_type=prior_mask_type
    )
    
    assert isinstance(dpvae, dpVAE)
    assert dpvae.config.input_dim == input_dim
    assert dpvae.config.latent_dim == latent_dim
    assert dpvae.config.hidden_dims == hidden_dims
    assert dpvae.config.activation == activation
    assert dpvae.config.input_transformation == input_transformation
    
    # Check decoupled prior
    assert dpvae.decoupled_prior.latent_dim == latent_dim
    assert dpvae.decoupled_prior.num_layers == len(prior_hidden_dims)
    assert dpvae.decoupled_prior.hidden_dims == prior_hidden_dims
    assert dpvae.decoupled_prior.activation == prior_activation
    assert dpvae.decoupled_prior.mask_type == prior_mask_type


def test_create_dpvae_functionality(test_input):
    """Test that created dpVAE works correctly."""
    input_dim = test_input.shape[1]
    latent_dim = 3
    
    dpvae = create_dpvae(input_dim=input_dim, latent_dim=latent_dim)
    
    reconstructed, mean, logvar = dpvae(test_input)
    
    assert reconstructed.shape == test_input.shape
    assert mean.shape == (test_input.shape[0], latent_dim)
    assert logvar.shape == (test_input.shape[0], latent_dim)
    assert jnp.all(jnp.isfinite(reconstructed))
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(logvar))


def test_create_dpvae_different_configurations():
    """Test dpVAE creation with different configurations."""
    input_dim = 10
    
    configs = [
        {"latent_dim": 1, "hidden_dims": [64], "prior_hidden_dims": [32]},
        {"latent_dim": 2, "hidden_dims": [128, 64], "prior_hidden_dims": [64, 32]},
        {"latent_dim": 5, "hidden_dims": [256, 128, 64], "prior_hidden_dims": [128, 64, 32]},
        {"latent_dim": 10, "hidden_dims": [512, 256, 128, 64], "prior_hidden_dims": [256, 128, 64, 32]},
    ]
    
    for config in configs:
        dpvae = create_dpvae(input_dim=input_dim, **config)
        
        assert isinstance(dpvae, dpVAE)
        assert dpvae.config.input_dim == input_dim
        assert dpvae.config.latent_dim == config["latent_dim"]
        assert dpvae.config.hidden_dims == config["hidden_dims"]
        assert dpvae.decoupled_prior.hidden_dims == config["prior_hidden_dims"]


def test_create_dpvae_encoder_decoder_symmetry():
    """Test that dpVAE encoder and decoder are symmetric."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64]
    
    dpvae = create_dpvae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # Check that encoder and decoder have symmetric architectures
    assert len(dpvae.encoder.encoder_layers) == len(dpvae.decoder.decoder_layers)
    assert dpvae.encoder.config.hidden_dims == dpvae.decoder.config.hidden_dims
    
    # Check that configs are symmetric
    assert dpvae.encoder.config.input_dim == dpvae.decoder.config.input_dim
    assert dpvae.encoder.config.latent_dim == dpvae.decoder.config.latent_dim


def test_create_dpvae_decoupled_prior_integration():
    """Test that dpVAE decoupled prior integrates correctly."""
    input_dim = 10
    latent_dim = 3
    
    dpvae = create_dpvae(input_dim=input_dim, latent_dim=latent_dim)
    
    # Test that the decoupled prior can transform samples
    key = jax.random.PRNGKey(789)
    z_base = jax.random.normal(key, (3, latent_dim))
    
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
# Factory Function Comparison Tests
# ------------------------------------------------------------------------------

def test_vae_vs_dpvae_creation():
    """Test that VAE and dpVAE factory functions create compatible architectures."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64]
    
    # Create both architectures
    vae = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    dpvae = create_dpvae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # Test that both have same basic functionality
    x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
    
    # Both should produce same shapes
    reconstructed_vae, mean_vae, logvar_vae = vae(x)
    reconstructed_dpvae, mean_dpvae, logvar_dpvae = dpvae(x)
    
    assert reconstructed_vae.shape == reconstructed_dpvae.shape
    assert mean_vae.shape == mean_dpvae.shape
    assert logvar_vae.shape == logvar_dpvae.shape


def test_factory_functions_consistency():
    """Test that factory functions are consistent with manual creation."""
    input_dim = 10
    latent_dim = 3
    hidden_dims = [128, 64]
    
    # Create using factory function
    vae_factory = create_vae(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    
    # Create manually
    config = VAEConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        activation="relu",
        input_transformation="log1p"
    )
    
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    encoder = Encoder(config=config, rngs=rngs)
    decoder = Decoder(config=config, rngs=rngs)
    vae_manual = VAE(encoder=encoder, decoder=decoder, config=config, rngs=rngs)
    
    # Test that both produce same results
    x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
    
    reconstructed_factory, mean_factory, logvar_factory = vae_factory(x)
    reconstructed_manual, mean_manual, logvar_manual = vae_manual(x)
    
    assert jnp.allclose(reconstructed_factory, reconstructed_manual)
    assert jnp.allclose(mean_factory, mean_manual)
    assert jnp.allclose(logvar_factory, logvar_manual)


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_factory_functions_edge_cases():
    """Test factory functions with edge cases."""
    input_dim = 1
    latent_dim = 1
    
    # Test with minimal dimensions
    vae = create_vae(input_dim=input_dim, latent_dim=latent_dim)
    dpvae = create_dpvae(input_dim=input_dim, latent_dim=latent_dim)
    
    assert vae.config.input_dim == input_dim
    assert vae.config.latent_dim == latent_dim
    assert dpvae.config.input_dim == input_dim
    assert dpvae.config.latent_dim == latent_dim
    
    # Test with large dimensions
    large_input_dim = 1000
    large_latent_dim = 100
    
    vae_large = create_vae(input_dim=large_input_dim, latent_dim=large_latent_dim)
    dpvae_large = create_dpvae(input_dim=large_input_dim, latent_dim=large_latent_dim)
    
    assert vae_large.config.input_dim == large_input_dim
    assert vae_large.config.latent_dim == large_latent_dim
    assert dpvae_large.config.input_dim == large_input_dim
    assert dpvae_large.config.latent_dim == large_latent_dim


def test_factory_functions_different_activations():
    """Test factory functions with different activation functions."""
    input_dim = 10
    latent_dim = 3
    
    activations = ["relu", "gelu", "tanh", "sigmoid", "softplus"]
    
    for activation in activations:
        vae = create_vae(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation
        )
        
        dpvae = create_dpvae(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation=activation
        )
        
        assert vae.config.activation == activation
        assert dpvae.config.activation == activation
        
        # Test that they work
        x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
        
        reconstructed_vae, mean_vae, logvar_vae = vae(x)
        reconstructed_dpvae, mean_dpvae, logvar_dpvae = dpvae(x)
        
        assert jnp.all(jnp.isfinite(reconstructed_vae))
        assert jnp.all(jnp.isfinite(reconstructed_dpvae)) 