# tests/test_stats.py
"""
Tests for statistics functions in stats.py
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.stats import (
    compute_histogram_percentiles,
    compute_histogram_credible_regions,
    sample_dirichlet_from_parameters,
    fit_dirichlet_mle
)

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate some random samples
    n_samples = 1000
    n_points = 100
    samples = np.random.negative_binomial(n=5, p=0.3, size=(n_samples, n_points))
    
    return samples

@pytest.fixture
def dirichlet_samples():
    """Generate sample data for Dirichlet-related tests."""
    # Set random seed for reproducibility
    rng_key = random.PRNGKey(42)
    
    # Generate concentration parameters
    n_samples = 1000
    n_variables = 5
    alpha = jnp.ones((n_samples, n_variables)) * 2.0  # concentration parameters
    
    # Generate one sample from each set of concentration parameters
    samples = sample_dirichlet_from_parameters(
        parameter_samples=alpha,
        n_samples_dirichlet=1,
        rng_key=rng_key
    )
    
    return samples, alpha

# ------------------------------------------------------------------------------
# Test histogram functions
# ------------------------------------------------------------------------------

def test_compute_histogram_percentiles(sample_data):
    """Test computation of histogram percentiles."""
    percentiles = [5, 25, 50, 75, 95]
    bin_edges, hist_percentiles = compute_histogram_percentiles(
        sample_data,
        percentiles=percentiles
    )
    
    # Check output shapes
    assert len(bin_edges) > 1  # Should have at least 2 bin edges
    assert hist_percentiles.shape[0] == len(percentiles)
    assert hist_percentiles.shape[1] == len(bin_edges) - 1
    
    # Check that percentiles are in ascending order
    assert np.all(np.diff(hist_percentiles, axis=0) >= -1e-10)  # Allow small numerical errors
    
def test_compute_histogram_credible_regions(sample_data):
    """Test computation of histogram credible regions."""
    credible_regions = [95, 68, 50]
    results = compute_histogram_credible_regions(
        sample_data,
        credible_regions=credible_regions
    )
    
    # Check that all expected keys exist
    assert 'bin_edges' in results
    assert 'regions' in results
    for cr in credible_regions:
        assert cr in results['regions']
        
    # Check that each region has required components
    for cr in credible_regions:
        region = results['regions'][cr]
        assert 'lower' in region
        assert 'upper' in region
        assert 'median' in region
        
        # Check shapes
        n_bins = len(results['bin_edges']) - 1
        assert region['lower'].shape == (n_bins,)
        assert region['upper'].shape == (n_bins,)
        assert region['median'].shape == (n_bins,)
        
        # Check ordering
        assert np.all(region['lower'] <= region['median'])
        assert np.all(region['median'] <= region['upper'])

# ------------------------------------------------------------------------------
# Test Dirichlet-related functions
# ------------------------------------------------------------------------------

def test_sample_dirichlet_from_parameters(dirichlet_samples, rng_key):
    """Test sampling from Dirichlet distribution given parameters."""
    samples, alpha = dirichlet_samples
    
    # Test with single sample
    output = sample_dirichlet_from_parameters(
        jnp.array(alpha),
        n_samples_dirichlet=1,
        rng_key=rng_key
    )
    
    assert isinstance(output, jnp.ndarray)
    assert output.shape == alpha.shape
    assert jnp.allclose(output.sum(axis=1), 1.0)
    assert jnp.all(output >= 0)
    
    # Test with multiple samples
    n_samples = 100
    output = sample_dirichlet_from_parameters(
        jnp.array(alpha),
        n_samples_dirichlet=n_samples,
        rng_key=rng_key
    )
    
    assert output.shape == (*alpha.shape, n_samples)
    assert jnp.allclose(output.sum(axis=1), 1.0)
    assert jnp.all(output >= 0)

def test_fit_dirichlet_mle(dirichlet_samples):
    """Test fitting Dirichlet distribution using MLE."""
    samples, true_alpha = dirichlet_samples
    
    # Fit the model
    fitted_alpha = fit_dirichlet_mle(
        samples,
        max_iter=1000,
        tol=1e-7
    )
    
    # Check shape
    assert fitted_alpha.shape == true_alpha[0].shape
    
    # Check that parameters are positive
    assert jnp.all(fitted_alpha > 0)
    
    # Check that the fitted parameters are reasonably close to true parameters
    # Note: This is a probabilistic test, so we use a relatively loose tolerance
    assert jnp.allclose(fitted_alpha, true_alpha[0], rtol=0.2)