#!/usr/bin/env python3
"""
Tests for DecoupledPrior to verify forward and inverse transformations work correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import DecoupledPrior


@pytest.fixture(scope="session")
def rng_key():
    """Provide a consistent random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def test_input():
    """Generate test input data."""
    key = jax.random.PRNGKey(123)
    latent_dim = 6
    return jax.random.normal(key, (2, latent_dim))


@pytest.fixture(scope="function")
def decoupled_prior(rng_key):
    """Create a DecoupledPrior instance for testing."""
    latent_dim = 6
    num_layers = 3
    hidden_dims = [64, 64]
    rngs = nnx.Rngs(params=rng_key)
    
    return DecoupledPrior(
        latent_dim=latent_dim,
        num_layers=num_layers,
        hidden_dims=hidden_dims,
        rngs=rngs,
        activation="relu",
        mask_type="alternating"
    )


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_decoupled_prior_creation(decoupled_prior):
    """Test that the decoupled prior is created successfully."""
    assert decoupled_prior is not None
    assert hasattr(decoupled_prior, 'forward')
    assert hasattr(decoupled_prior, 'inverse')
    assert hasattr(decoupled_prior, '__call__')
    assert hasattr(decoupled_prior, 'coupling_layers')
    assert len(decoupled_prior.coupling_layers) == decoupled_prior.num_layers


def test_forward_transformation(decoupled_prior, test_input):
    """Test the forward transformation."""
    z_transformed, log_det_forward = decoupled_prior.forward(test_input)
    
    # Check output shapes
    assert z_transformed.shape == test_input.shape
    assert log_det_forward.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(z_transformed))
    assert jnp.all(jnp.isfinite(log_det_forward))


def test_inverse_transformation(decoupled_prior, test_input):
    """Test the inverse transformation."""
    # First apply forward transformation
    z_transformed, _ = decoupled_prior.forward(test_input)
    
    # Then apply inverse transformation
    z_reconstructed, log_det_inverse = decoupled_prior.inverse(z_transformed)
    
    # Check output shapes
    assert z_reconstructed.shape == test_input.shape
    assert log_det_inverse.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(z_reconstructed))
    assert jnp.all(jnp.isfinite(log_det_inverse))


def test_reconstruction_accuracy(decoupled_prior, test_input):
    """Test that forward + inverse transformation reconstructs the input accurately."""
    # Apply forward transformation
    z_transformed, _ = decoupled_prior.forward(test_input)
    
    # Apply inverse transformation
    z_reconstructed, _ = decoupled_prior.inverse(z_transformed)
    
    # Check reconstruction error
    reconstruction_error = jnp.mean(jnp.abs(test_input - z_reconstructed))
    assert reconstruction_error < 1e-5, f"Reconstruction error too high: {reconstruction_error}"


def test_log_determinant_relationship(decoupled_prior, test_input):
    """Test that log determinants sum to approximately zero."""
    # Apply forward transformation
    _, log_det_forward = decoupled_prior.forward(test_input)
    
    # Apply inverse transformation
    z_transformed, _ = decoupled_prior.forward(test_input)
    _, log_det_inverse = decoupled_prior.inverse(z_transformed)
    
    # Check that log determinants sum to approximately zero
    log_det_sum = log_det_forward + log_det_inverse
    assert jnp.allclose(log_det_sum, 0.0, atol=1e-5), f"Log det sum not zero: {log_det_sum}"


def test_transformation_magnitude(decoupled_prior, test_input):
    """Test that the transformation actually changes the distribution."""
    # Apply forward transformation
    z_transformed, _ = decoupled_prior.forward(test_input)
    
    # Check that the transformation actually changes the values
    transformation_magnitude = jnp.mean(jnp.abs(z_transformed - test_input))
    assert transformation_magnitude > 1e-6, f"Transformation too small: {transformation_magnitude}"


# ------------------------------------------------------------------------------
# Call Method Tests
# ------------------------------------------------------------------------------

def test_call_method_forward(decoupled_prior, test_input):
    """Test the __call__ method for forward transformation."""
    z_transformed, log_det = decoupled_prior(test_input)
    
    # Check output shapes
    assert z_transformed.shape == test_input.shape
    assert log_det.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(z_transformed))
    assert jnp.all(jnp.isfinite(log_det))


def test_call_method_inverse(decoupled_prior, test_input):
    """Test the __call__ method for inverse transformation."""
    # First apply forward transformation
    z_transformed, _ = decoupled_prior.forward(test_input)
    
    # Then apply inverse using call method
    z_reconstructed, log_det = decoupled_prior(z_transformed, inverse=True)
    
    # Check output shapes
    assert z_reconstructed.shape == test_input.shape
    assert log_det.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(z_reconstructed))
    assert jnp.all(jnp.isfinite(log_det))


def test_call_method_consistency(decoupled_prior, test_input):
    """Test that call method gives same results as explicit forward/inverse methods."""
    # Test forward consistency
    z_call, log_det_call = decoupled_prior(test_input)
    z_forward, log_det_forward = decoupled_prior.forward(test_input)
    
    assert jnp.allclose(z_call, z_forward)
    assert jnp.allclose(log_det_call, log_det_forward)
    
    # Test inverse consistency
    z_inv_call, log_det_inv_call = decoupled_prior(z_call, inverse=True)
    z_inverse, log_det_inverse = decoupled_prior.inverse(z_call)
    
    assert jnp.allclose(z_inv_call, z_inverse)
    assert jnp.allclose(log_det_inv_call, log_det_inverse)


# ------------------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------------------

def test_different_activations(rng_key):
    """Test decoupled prior with different activation functions."""
    latent_dim = 6
    num_layers = 2
    hidden_dims = [32, 32]
    rngs = nnx.Rngs(params=rng_key)
    
    activations = ["relu", "tanh", "sigmoid"]
    
    for activation in activations:
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation=activation,
            mask_type="alternating"
        )
        
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        z_transformed, log_det = decoupled_prior(z)
        
        assert z_transformed.shape == z.shape
        assert jnp.all(jnp.isfinite(z_transformed))
        assert jnp.all(jnp.isfinite(log_det))


def test_different_mask_types(rng_key):
    """Test decoupled prior with different mask types."""
    latent_dim = 6
    num_layers = 2
    hidden_dims = [32, 32]
    rngs = nnx.Rngs(params=rng_key)
    
    # Only test supported mask types
    mask_types = ["alternating", "checkerboard"]
    
    for mask_type in mask_types:
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation="relu",
            mask_type=mask_type
        )
        
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        z_transformed, log_det = decoupled_prior(z)
        
        assert z_transformed.shape == z.shape
        assert jnp.all(jnp.isfinite(z_transformed))
        assert jnp.all(jnp.isfinite(log_det))


def test_different_hidden_dims(rng_key):
    """Test decoupled prior with different hidden dimensions."""
    latent_dim = 6
    num_layers = 2
    rngs = nnx.Rngs(params=rng_key)
    
    hidden_dims_configs = [
        [32],
        [64, 32],
        [128, 64, 32]
    ]
    
    for hidden_dims in hidden_dims_configs:
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation="relu",
            mask_type="alternating"
        )
        
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        z_transformed, log_det = decoupled_prior(z)
        
        assert z_transformed.shape == z.shape
        assert jnp.all(jnp.isfinite(z_transformed))
        assert jnp.all(jnp.isfinite(log_det))


def test_different_num_layers(rng_key):
    """Test decoupled prior with different numbers of layers."""
    latent_dim = 6
    hidden_dims = [32, 32]
    rngs = nnx.Rngs(params=rng_key)
    
    num_layers_list = [1, 2, 4]
    
    for num_layers in num_layers_list:
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation="relu",
            mask_type="alternating"
        )
        
        z = jax.random.normal(jax.random.PRNGKey(123), (2, latent_dim))
        z_transformed, log_det = decoupled_prior(z)
        
        assert z_transformed.shape == z.shape
        assert jnp.all(jnp.isfinite(z_transformed))
        assert jnp.all(jnp.isfinite(log_det))
        
        # Check that the number of coupling layers matches
        assert len(decoupled_prior.coupling_layers) == num_layers


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_input_shapes(decoupled_prior):
    """Test the decoupled prior with different input shapes."""
    key = jax.random.PRNGKey(456)
    latent_dim = decoupled_prior.latent_dim
    
    # Test with different batch sizes
    for batch_size in [1, 3, 5]:
        z = jax.random.normal(key, (batch_size, latent_dim))
        z_transformed, log_det = decoupled_prior(z)
        
        assert z_transformed.shape == (batch_size, latent_dim)
        assert log_det.shape == (batch_size,)


def test_numerical_stability(decoupled_prior):
    """Test numerical stability with reasonable input values."""
    key = jax.random.PRNGKey(789)
    latent_dim = decoupled_prior.latent_dim
    
    # Test with moderately large values (not extreme)
    z_large = jax.random.normal(key, (2, latent_dim)) * 10
    z_transformed_large, log_det_large = decoupled_prior(z_large)
    
    assert jnp.all(jnp.isfinite(z_transformed_large))
    assert jnp.all(jnp.isfinite(log_det_large))
    
    # Test with very small values
    z_small = jax.random.normal(key, (2, latent_dim)) * 1e-6
    z_transformed_small, log_det_small = decoupled_prior(z_small)
    
    assert jnp.all(jnp.isfinite(z_transformed_small))
    assert jnp.all(jnp.isfinite(log_det_small))
    
    # Test with zero values
    z_zero = jnp.zeros((2, latent_dim))
    z_transformed_zero, log_det_zero = decoupled_prior(z_zero)
    
    assert jnp.all(jnp.isfinite(z_transformed_zero))
    assert jnp.all(jnp.isfinite(log_det_zero))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_multiple_transformations(decoupled_prior, test_input):
    """Test applying multiple forward/inverse transformations."""
    # Apply multiple forward transformations
    z1, log_det1 = decoupled_prior.forward(test_input)
    z2, log_det2 = decoupled_prior.forward(z1)
    
    # Apply multiple inverse transformations
    z1_reconstructed, log_det_inv1 = decoupled_prior.inverse(z2)
    z0_reconstructed, log_det_inv0 = decoupled_prior.inverse(z1_reconstructed)
    
    # Check that we can reconstruct the original input
    reconstruction_error = jnp.mean(jnp.abs(test_input - z0_reconstructed))
    assert reconstruction_error < 1e-5, f"Reconstruction error too high: {reconstruction_error}"
    
    # Check that log determinants accumulate correctly
    total_log_det = log_det1 + log_det2 + log_det_inv1 + log_det_inv0
    assert jnp.allclose(total_log_det, 0.0, atol=1e-5), f"Total log det not zero: {total_log_det}"


def test_batch_processing(decoupled_prior):
    """Test processing multiple inputs in a batch."""
    key = jax.random.PRNGKey(999)
    batch_size = 10
    latent_dim = decoupled_prior.latent_dim
    
    z_batch = jax.random.normal(key, (batch_size, latent_dim))
    z_transformed_batch, log_det_batch = decoupled_prior(z_batch)
    
    assert z_transformed_batch.shape == (batch_size, latent_dim)
    assert log_det_batch.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(z_transformed_batch))
    assert jnp.all(jnp.isfinite(log_det_batch))
    
    # Test inverse on batch
    z_reconstructed, log_det_inv = decoupled_prior.inverse(z_transformed_batch)
    assert z_reconstructed.shape == (batch_size, latent_dim)
    assert log_det_inv.shape == (batch_size,)
    
    # Check reconstruction accuracy for batch
    reconstruction_error = jnp.mean(jnp.abs(z_batch - z_reconstructed))
    assert reconstruction_error < 1e-5, f"Batch reconstruction error too high: {reconstruction_error}"


def test_layer_by_layer_consistency(decoupled_prior, test_input):
    """Test that the full transformation matches layer-by-layer application."""
    # Apply full transformation
    z_full, log_det_full = decoupled_prior.forward(test_input)
    
    # Apply layer by layer
    z_current = test_input
    log_det_accumulated = 0.0
    
    for coupling_layer in decoupled_prior.coupling_layers:
        z_current, log_det = coupling_layer.forward(z_current)
        log_det_accumulated += log_det
    
    # Check that results match
    assert jnp.allclose(z_full, z_current)
    assert jnp.allclose(log_det_full, log_det_accumulated)


def test_inverse_layer_by_layer_consistency(decoupled_prior, test_input):
    """Test that the full inverse transformation matches layer-by-layer application."""
    # Apply forward transformation first
    z_transformed, _ = decoupled_prior.forward(test_input)
    
    # Apply full inverse transformation
    z_full_inverse, log_det_full_inverse = decoupled_prior.inverse(z_transformed)
    
    # Apply inverse layer by layer (in reverse order)
    z_current = z_transformed
    log_det_accumulated = 0.0
    
    for coupling_layer in reversed(decoupled_prior.coupling_layers):
        z_current, log_det = coupling_layer.inverse(z_current)
        log_det_accumulated += log_det
    
    # Check that results match
    assert jnp.allclose(z_full_inverse, z_current)
    assert jnp.allclose(log_det_full_inverse, log_det_accumulated)