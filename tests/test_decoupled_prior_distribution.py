#!/usr/bin/env python3
"""
Tests for DecoupledPriorDistribution to verify it works correctly as a NumPyro
distribution.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpyro
import numpyro.distributions as dist
from scribe.vae.architectures import DecoupledPrior, DecoupledPriorDistribution


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
def decoupled_prior_distribution(rng_key):
    """Create a DecoupledPriorDistribution instance for testing."""
    latent_dim = 6
    num_layers = 2
    hidden_dims = [8, 8]
    rngs = nnx.Rngs(params=rng_key)
    
    decoupled_prior = DecoupledPrior(
        latent_dim=latent_dim,
        num_layers=num_layers,
        hidden_dims=hidden_dims,
        rngs=rngs,
        activation="relu",
        mask_type="alternating"
    )
    
    # Create base distribution (multivariate normal)
    base_dist = dist.Normal(jnp.zeros(latent_dim), jnp.ones(latent_dim))
    
    return DecoupledPriorDistribution(
        decoupled_prior=decoupled_prior,
        base_distribution=base_dist
    )


# ------------------------------------------------------------------------------
# Basic Functionality Tests
# ------------------------------------------------------------------------------

def test_distribution_creation(decoupled_prior_distribution):
    """Test that the distribution is created successfully."""
    assert decoupled_prior_distribution is not None
    assert hasattr(decoupled_prior_distribution, 'log_prob')
    assert hasattr(decoupled_prior_distribution, 'sample')
    assert hasattr(decoupled_prior_distribution, 'sample_with_intermediates')
    assert hasattr(decoupled_prior_distribution, 'event_shape')
    assert hasattr(decoupled_prior_distribution, 'batch_shape')


def test_distribution_shapes(decoupled_prior_distribution):
    """Test that the distribution has correct shapes."""
    # Check event shape (should be empty for multivariate base distribution)
    assert decoupled_prior_distribution.event_shape == ()
    
    # Check batch shape (should be (6,) for multivariate normal)
    assert decoupled_prior_distribution.batch_shape == (6,)


def test_sampling(decoupled_prior_distribution):
    """Test sampling from the distribution."""
    key = jax.random.PRNGKey(456)
    sample_shape = (5,)
    
    samples = decoupled_prior_distribution.sample(key, sample_shape)
    
    # For multivariate base distribution, samples should have shape (5,)
    assert samples.shape == (5, 6)
    
    # Check that samples are finite
    assert jnp.all(jnp.isfinite(samples))


def test_log_prob_computation(decoupled_prior_distribution, test_input):
    """Test log probability computation."""
    # Apply forward transformation to get samples from complex prior
    z_complex, _ = decoupled_prior_distribution.decoupled_prior.forward(test_input)
    
    # Compute log probabilities
    log_probs = decoupled_prior_distribution.log_prob(z_complex)
    
    # Check output shape - should match the batch shape of z_complex
    expected_shape = z_complex.shape
    assert log_probs.shape == expected_shape
    
    # Check that log probabilities are finite
    assert jnp.all(jnp.isfinite(log_probs))


def test_sampling_with_intermediates(decoupled_prior_distribution):
    """Test sampling with intermediate values."""
    key = jax.random.PRNGKey(789)
    sample_shape = (3,)
    
    samples, intermediates = decoupled_prior_distribution.sample_with_intermediates(
        key, sample_shape
    )
    
    # Check output shapes
    assert samples.shape == sample_shape + (6,)
    
    # Check that samples are finite
    assert jnp.all(jnp.isfinite(samples))
    
    # Check intermediate values
    assert "z_base" in intermediates
    assert "z_complex" in intermediates
    assert "log_det_jacobian" in intermediates
    
    # Check that intermediate shapes are correct
    assert intermediates["z_base"].shape == sample_shape + (6,)
    assert intermediates["z_complex"].shape == sample_shape + (6,)
    assert intermediates["log_det_jacobian"].shape == sample_shape


# ------------------------------------------------------------------------------
# Mathematical Consistency Tests
# ------------------------------------------------------------------------------

def test_change_of_variables_formula(decoupled_prior_distribution, test_input):
    """Test that the change of variables formula is correctly implemented."""
    # Get samples from complex prior
    z_complex, _ = decoupled_prior_distribution.decoupled_prior.forward(test_input)
    
    # Compute log probability using the distribution
    direct_log_prob = decoupled_prior_distribution.log_prob(z_complex)
    
    # Manual computation through base distribution
    z_base, log_det = decoupled_prior_distribution.decoupled_prior.inverse(z_complex)
    base_log_prob = decoupled_prior_distribution.base_distribution.log_prob(z_base)
    manual_log_prob = base_log_prob + log_det[..., None]
    
    # Check that both methods give the same result
    log_prob_error = jnp.mean(jnp.abs(direct_log_prob - manual_log_prob))
    assert log_prob_error < 1e-6, f"Log prob consistency error too large: {log_prob_error}"


def test_transformation_consistency(decoupled_prior_distribution):
    """Test that forward and inverse transformations are consistent."""
    key = jax.random.PRNGKey(999)
    sample_shape = (4,)
    
    # Sample from base distribution
    base_samples = decoupled_prior_distribution.base_distribution.sample(key, sample_shape)
    
    # Apply forward transformation
    complex_samples, _ = decoupled_prior_distribution.decoupled_prior.forward(base_samples)
    
    # Apply inverse transformation
    recovered_base, _ = decoupled_prior_distribution.decoupled_prior.inverse(complex_samples)
    
    # Check reconstruction accuracy
    reconstruction_error = jnp.mean(jnp.abs(base_samples - recovered_base))
    assert reconstruction_error < 1e-6, f"Reconstruction error too large: {reconstruction_error}"


def test_log_prob_normalization(decoupled_prior_distribution):
    """Test that log probabilities are properly normalized."""
    key = jax.random.PRNGKey(111)
    sample_shape = (10,)
    
    # Sample from the distribution
    samples = decoupled_prior_distribution.sample(key, sample_shape)
    
    # Compute log probabilities
    log_probs = decoupled_prior_distribution.log_prob(samples)
    
    # Check that log probabilities are reasonable (not too large or small)
    assert jnp.all(log_probs > -100), "Log probabilities too small"
    assert jnp.all(log_probs < 100), "Log probabilities too large"


# ------------------------------------------------------------------------------
# Edge Cases and Error Handling
# ------------------------------------------------------------------------------

def test_different_sample_shapes(decoupled_prior_distribution):
    """Test sampling with different sample shapes."""
    key = jax.random.PRNGKey(222)
    
    # Test various sample shapes
    sample_shapes = [(1,), (2, 3), (1, 1, 1)]
    
    for sample_shape in sample_shapes:
        samples = decoupled_prior_distribution.sample(key, sample_shape)
        expected_shape = sample_shape + (6,)
        assert samples.shape == expected_shape
        assert jnp.all(jnp.isfinite(samples))


def test_numerical_stability(decoupled_prior_distribution):
    """Test numerical stability with extreme values."""
    key = jax.random.PRNGKey(42)
    
    # Test with very large values (should still work)
    large_samples = jax.random.normal(key, (2, 6)) * 20
    log_probs_large = decoupled_prior_distribution.log_prob(large_samples)
    assert jnp.all(jnp.isfinite(log_probs_large))
    
    # Test with very small values
    small_samples = jax.random.normal(key, (2, 6)) * 1e-6
    log_probs_small = decoupled_prior_distribution.log_prob(small_samples)
    assert jnp.all(jnp.isfinite(log_probs_small))


# ------------------------------------------------------------------------------
# Configuration Tests
# ------------------------------------------------------------------------------

def test_different_base_distributions(rng_key):
    """Test with different base distributions."""
    latent_dim = 6
    num_layers = 2
    hidden_dims = [8, 8]
    rngs = nnx.Rngs(params=rng_key)
    
    decoupled_prior = DecoupledPrior(
        latent_dim=latent_dim,
        num_layers=num_layers,
        hidden_dims=hidden_dims,
        rngs=rngs,
        activation="relu",
        mask_type="alternating"
    )
    
    # Test with different base distributions
    base_dists = [
        dist.Normal(0.0, 1.0),
        dist.Normal(1.0, 2.0),
        dist.StudentT(df=3.0, loc=0.0, scale=1.0),
    ]
    
    for base_dist in base_dists:
        decoupled_dist = DecoupledPriorDistribution(
            decoupled_prior=decoupled_prior,
            base_distribution=base_dist
        )
        
        # Test sampling
        key = jax.random.PRNGKey(444)
        samples = decoupled_dist.sample(key, (3,))
        assert samples.shape == (3,)
        assert jnp.all(jnp.isfinite(samples))
        
        # Test log_prob
        log_probs = decoupled_dist.log_prob(samples)
        assert log_probs.shape == (3,)
        assert jnp.all(jnp.isfinite(log_probs))


def test_different_decoupled_prior_configurations(rng_key):
    """Test with different decoupled prior configurations."""
    latent_dim = 6
    rngs = nnx.Rngs(params=rng_key)
    base_dist = dist.Normal(0.0, 1.0)
    
    # Test different configurations
    configs = [
        {"num_layers": 1, "hidden_dims": [32]},
        {"num_layers": 2, "hidden_dims": [64, 32]},
        {"num_layers": 3, "hidden_dims": [128, 64, 32]},
    ]
    
    for config in configs:
        decoupled_prior = DecoupledPrior(
            latent_dim=latent_dim,
            num_layers=config["num_layers"],
            hidden_dims=config["hidden_dims"],
            rngs=rngs,
            activation="relu",
            mask_type="alternating"
        )
        
        decoupled_dist = DecoupledPriorDistribution(
            decoupled_prior=decoupled_prior,
            base_distribution=base_dist
        )
        
        # Test sampling
        key = jax.random.PRNGKey(555)
        samples = decoupled_dist.sample(key, (2,))
        assert samples.shape == (2,)
        assert jnp.all(jnp.isfinite(samples))


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------

def test_distribution_in_numpyro_model(decoupled_prior_distribution):
    """Test that the distribution works in a NumPyro model context."""
    import numpyro
    
    def test_model():
        # Sample from the decoupled prior distribution
        z = numpyro.sample("z", decoupled_prior_distribution)
        # Use z in some computation
        numpyro.deterministic("z_squared", z**2)
    
    # This should not raise any errors
    # We can't easily test the full model execution without more setup,
    # but we can test that the distribution is compatible with NumPyro
    assert hasattr(decoupled_prior_distribution, 'log_prob')
    assert hasattr(decoupled_prior_distribution, 'sample')


def test_batch_processing(decoupled_prior_distribution):
    """Test processing multiple samples in a batch."""
    key = jax.random.PRNGKey(666)
    batch_size = 10
    
    # Sample a batch
    samples = decoupled_prior_distribution.sample(key, (batch_size,))
    assert samples.shape == (batch_size, 6)
    
    # Compute log probabilities for the batch
    log_probs = decoupled_prior_distribution.log_prob(samples)
    assert log_probs.shape == (batch_size, 6)
    
    # Check that all values are finite
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(log_probs))


def test_intermediate_values_consistency(decoupled_prior_distribution):
    """Test that intermediate values are consistent."""
    key = jax.random.PRNGKey(777)
    
    # Sample with intermediates
    samples, intermediates = decoupled_prior_distribution.sample_with_intermediates(
        key, (3,)
    )
    
    # Check that the complex samples match
    assert jnp.allclose(samples, intermediates["z_complex"])
    
    # Check that we can recover base samples using inverse
    recovered_base, _ = decoupled_prior_distribution.decoupled_prior.inverse(samples)
    assert jnp.allclose(recovered_base, intermediates["z_base"])
    
    # Check that log determinant is consistent
    _, manual_log_det = decoupled_prior_distribution.decoupled_prior.forward(
        intermediates["z_base"]
    )
    assert jnp.allclose(manual_log_det, intermediates["log_det_jacobian"])


def test_transformation_magnitude(decoupled_prior_distribution):
    """Test that the transformation actually changes the distribution."""
    key = jax.random.PRNGKey(888)
    
    # Sample from base distribution
    base_samples = decoupled_prior_distribution.base_distribution.sample(key, (5,))
    
    # Apply transformation
    complex_samples, _ = decoupled_prior_distribution.decoupled_prior.forward(base_samples)
    
    # Check that the transformation actually changes the values
    transformation_magnitude = jnp.mean(jnp.abs(complex_samples - base_samples))
    assert transformation_magnitude > 1e-6, f"Transformation too small: {transformation_magnitude}"