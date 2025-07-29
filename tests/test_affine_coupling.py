#!/usr/bin/env python3
"""
Tests for AffineCouplingLayer to verify forward and inverse transformations work correctly.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from scribe.vae.architectures import AffineCouplingLayer


@pytest.fixture(scope="session")
def rng_key():
    """Provide a consistent random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def test_input():
    """Generate test input data."""
    key = jax.random.PRNGKey(123)
    input_dim = 6
    return jax.random.normal(key, (2, input_dim))


@pytest.fixture(scope="function")
def coupling_layer(rng_key):
    """Create an AffineCouplingLayer instance for testing."""
    input_dim = 6
    hidden_dims = [64, 64]
    rngs = nnx.Rngs(params=rng_key)
    
    return AffineCouplingLayer(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        rngs=rngs,
        activation="relu",
        mask_type="alternating"
    )


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_coupling_layer_creation(coupling_layer):
    """Test that the coupling layer is created successfully."""
    assert coupling_layer is not None
    assert hasattr(coupling_layer, 'forward')
    assert hasattr(coupling_layer, 'inverse')
    assert hasattr(coupling_layer, '__call__')


def test_forward_transformation(coupling_layer, test_input):
    """Test the forward transformation."""
    y, log_det_forward = coupling_layer.forward(test_input)
    
    # Check output shapes
    assert y.shape == test_input.shape
    assert log_det_forward.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(log_det_forward))


def test_inverse_transformation(coupling_layer, test_input):
    """Test the inverse transformation."""
    # First apply forward transformation
    y, _ = coupling_layer.forward(test_input)
    
    # Then apply inverse transformation
    x_reconstructed, log_det_inverse = coupling_layer.inverse(y)
    
    # Check output shapes
    assert x_reconstructed.shape == test_input.shape
    assert log_det_inverse.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(x_reconstructed))
    assert jnp.all(jnp.isfinite(log_det_inverse))


def test_reconstruction_accuracy(coupling_layer, test_input):
    """Test that forward + inverse transformation reconstructs the input accurately."""
    # Apply forward transformation
    y, _ = coupling_layer.forward(test_input)
    
    # Apply inverse transformation
    x_reconstructed, _ = coupling_layer.inverse(y)
    
    # Check reconstruction error
    reconstruction_error = jnp.mean(jnp.abs(test_input - x_reconstructed))
    assert reconstruction_error < 1e-5, f"Reconstruction error too high: {reconstruction_error}"


def test_log_determinant_relationship(coupling_layer, test_input):
    """Test that log determinants sum to approximately zero."""
    # Apply forward transformation
    _, log_det_forward = coupling_layer.forward(test_input)
    
    # Apply inverse transformation
    y, _ = coupling_layer.forward(test_input)
    _, log_det_inverse = coupling_layer.inverse(y)
    
    # Check that log determinants sum to approximately zero
    log_det_sum = log_det_forward + log_det_inverse
    assert jnp.allclose(log_det_sum, 0.0, atol=1e-5), f"Log det sum not zero: {log_det_sum}"


# ------------------------------------------------------------------------------
# Call Method Tests
# ------------------------------------------------------------------------------

def test_call_method_forward(coupling_layer, test_input):
    """Test the __call__ method for forward transformation."""
    y, log_det = coupling_layer(test_input)
    
    # Check output shapes
    assert y.shape == test_input.shape
    assert log_det.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(log_det))


def test_call_method_inverse(coupling_layer, test_input):
    """Test the __call__ method for inverse transformation."""
    # First apply forward transformation
    y, _ = coupling_layer.forward(test_input)
    
    # Then apply inverse using call method
    x_reconstructed, log_det = coupling_layer(y, inverse=True)
    
    # Check output shapes
    assert x_reconstructed.shape == test_input.shape
    assert log_det.shape == (test_input.shape[0],)
    
    # Check that output is finite
    assert jnp.all(jnp.isfinite(x_reconstructed))
    assert jnp.all(jnp.isfinite(log_det))


def test_call_method_consistency(coupling_layer, test_input):
    """Test that call method gives same results as explicit forward/inverse methods."""
    # Test forward consistency
    y_call, log_det_call = coupling_layer(test_input)
    y_forward, log_det_forward = coupling_layer.forward(test_input)
    
    assert jnp.allclose(y_call, y_forward)
    assert jnp.allclose(log_det_call, log_det_forward)
    
    # Test inverse consistency
    x_inv_call, log_det_inv_call = coupling_layer(y_call, inverse=True)
    x_inverse, log_det_inverse = coupling_layer.inverse(y_call)
    
    assert jnp.allclose(x_inv_call, x_inverse)
    assert jnp.allclose(log_det_inv_call, log_det_inverse)


# ------------------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------------------

def test_different_activations(rng_key):
    """Test coupling layer with different activation functions."""
    input_dim = 6
    hidden_dims = [32, 32]
    rngs = nnx.Rngs(params=rng_key)
    
    activations = ["relu", "tanh", "sigmoid"]
    
    for activation in activations:
        coupling_layer = AffineCouplingLayer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation=activation,
            mask_type="alternating"
        )
        
        x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
        y, log_det = coupling_layer(x)
        
        assert y.shape == x.shape
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(log_det))


def test_different_mask_types(rng_key):
    """Test coupling layer with different mask types."""
    input_dim = 6
    hidden_dims = [32, 32]
    rngs = nnx.Rngs(params=rng_key)
    
    # Only test supported mask types
    mask_types = ["alternating", "checkerboard"]
    
    for mask_type in mask_types:
        coupling_layer = AffineCouplingLayer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation="relu",
            mask_type=mask_type
        )
        
        x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
        y, log_det = coupling_layer(x)
        
        assert y.shape == x.shape
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(log_det))


def test_different_hidden_dims(rng_key):
    """Test coupling layer with different hidden dimensions."""
    input_dim = 6
    rngs = nnx.Rngs(params=rng_key)
    
    hidden_dims_configs = [
        [32],
        [64, 32],
        [128, 64, 32]
    ]
    
    for hidden_dims in hidden_dims_configs:
        coupling_layer = AffineCouplingLayer(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            rngs=rngs,
            activation="relu",
            mask_type="alternating"
        )
        
        x = jax.random.normal(jax.random.PRNGKey(123), (2, input_dim))
        y, log_det = coupling_layer(x)
        
        assert y.shape == x.shape
        assert jnp.all(jnp.isfinite(y))
        assert jnp.all(jnp.isfinite(log_det))


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_input_shapes(coupling_layer):
    """Test the coupling layer with different input shapes."""
    key = jax.random.PRNGKey(456)
    
    # Test with different batch sizes
    for batch_size in [1, 3, 5]:
        x = jax.random.normal(key, (batch_size, 6))
        y, log_det = coupling_layer(x)
        
        assert y.shape == (batch_size, 6)
        assert log_det.shape == (batch_size,)


def test_numerical_stability(coupling_layer):
    """Test numerical stability with reasonable input values."""
    key = jax.random.PRNGKey(789)
    
    # Test with moderately large values (not extreme)
    x_large = jax.random.normal(key, (2, 6)) * 10
    y_large, log_det_large = coupling_layer(x_large)
    
    assert jnp.all(jnp.isfinite(y_large))
    assert jnp.all(jnp.isfinite(log_det_large))
    
    # Test with very small values
    x_small = jax.random.normal(key, (2, 6)) * 1e-6
    y_small, log_det_small = coupling_layer(x_small)
    
    assert jnp.all(jnp.isfinite(y_small))
    assert jnp.all(jnp.isfinite(log_det_small))
    
    # Test with zero values
    x_zero = jnp.zeros((2, 6))
    y_zero, log_det_zero = coupling_layer(x_zero)
    
    assert jnp.all(jnp.isfinite(y_zero))
    assert jnp.all(jnp.isfinite(log_det_zero))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_multiple_transformations(coupling_layer, test_input):
    """Test applying multiple forward/inverse transformations."""
    # Apply multiple forward transformations
    y1, log_det1 = coupling_layer.forward(test_input)
    y2, log_det2 = coupling_layer.forward(y1)
    
    # Apply multiple inverse transformations
    x1, log_det_inv1 = coupling_layer.inverse(y2)
    x0, log_det_inv0 = coupling_layer.inverse(x1)
    
    # Check that we can reconstruct the original input
    reconstruction_error = jnp.mean(jnp.abs(test_input - x0))
    assert reconstruction_error < 1e-5, f"Reconstruction error too high: {reconstruction_error}"
    
    # Check that log determinants accumulate correctly
    total_log_det = log_det1 + log_det2 + log_det_inv1 + log_det_inv0
    assert jnp.allclose(total_log_det, 0.0, atol=1e-5), f"Total log det not zero: {total_log_det}"


def test_batch_processing(coupling_layer):
    """Test processing multiple inputs in a batch."""
    key = jax.random.PRNGKey(999)
    batch_size = 10
    input_dim = 6
    
    x_batch = jax.random.normal(key, (batch_size, input_dim))
    y_batch, log_det_batch = coupling_layer(x_batch)
    
    assert y_batch.shape == (batch_size, input_dim)
    assert log_det_batch.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(y_batch))
    assert jnp.all(jnp.isfinite(log_det_batch))
    
    # Test inverse on batch
    x_reconstructed, log_det_inv = coupling_layer.inverse(y_batch)
    assert x_reconstructed.shape == (batch_size, input_dim)
    assert log_det_inv.shape == (batch_size,)
    
    # Check reconstruction accuracy for batch
    reconstruction_error = jnp.mean(jnp.abs(x_batch - x_reconstructed))
    assert reconstruction_error < 1e-5, f"Batch reconstruction error too high: {reconstruction_error}"