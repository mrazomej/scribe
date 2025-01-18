# tests/test_zinbvcp.py
"""
Tests for the Zero-Inflated Negative Binomial with Variable Capture Probability model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from scribe.models import zinbvcp_model, zinbvcp_guide
from scribe.svi import run_scribe, rerun_scribe
from scribe.sampling import (
    sample_variational_posterior,
    generate_predictive_samples,
    generate_ppc_samples
)
from scribe.results import ZINBVCPResults

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define number of steps for testing
N_STEPS = 3

@pytest.fixture
def example_zinbvcp_results(small_dataset, rng_key):
    """Generate example ZINBVCP results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        model_type="zinbvcp",
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )

# ------------------------------------------------------------------------------
# Test inference
# ------------------------------------------------------------------------------

def test_inference_run(small_dataset, rng_key):
    """Test that inference runs and produces expected results."""
    counts, _ = small_dataset
    n_cells, n_genes = counts.shape
    
    results = run_scribe(
        counts=counts,
        model_type="zinbvcp",
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )
    
    assert isinstance(results, ZINBVCPResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

def test_parameter_ranges(example_zinbvcp_results):
    """Test that inferred parameters are in valid ranges."""
    # All parameters should be positive
    assert jnp.all(example_zinbvcp_results.params['alpha_p'] > 0)
    assert jnp.all(example_zinbvcp_results.params['beta_p'] > 0)
    assert jnp.all(example_zinbvcp_results.params['alpha_r'] > 0)
    assert jnp.all(example_zinbvcp_results.params['beta_r'] > 0)
    assert jnp.all(example_zinbvcp_results.params['alpha_gate'] > 0)
    assert jnp.all(example_zinbvcp_results.params['beta_gate'] > 0)
    assert jnp.all(example_zinbvcp_results.params['alpha_p_capture'] > 0)
    assert jnp.all(example_zinbvcp_results.params['beta_p_capture'] > 0)

    # Check shapes for cell-specific parameters
    assert example_zinbvcp_results.params['alpha_p_capture'].shape == (example_zinbvcp_results.n_cells,)
    assert example_zinbvcp_results.params['beta_p_capture'].shape == (example_zinbvcp_results.n_cells,)

    # Check shapes for gene-specific parameters
    assert example_zinbvcp_results.params['alpha_gate'].shape == (example_zinbvcp_results.n_genes,)
    assert example_zinbvcp_results.params['beta_gate'].shape == (example_zinbvcp_results.n_genes,)

def test_continue_training(example_zinbvcp_results, small_dataset, rng_key):
    """Test that continuing training from a previous results object works."""
    counts, _ = small_dataset
    n_cells, n_genes = counts.shape
    # Run inference again
    results = rerun_scribe(
        results=example_zinbvcp_results,
        counts=counts,
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5,
    )

    assert isinstance(results, ZINBVCPResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS + N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_zinbvcp_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 100
    samples = example_zinbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'p' in samples
    assert 'r' in samples
    assert 'gate' in samples
    assert 'p_capture' in samples
    assert samples['p'].shape == (n_samples,)
    assert samples['r'].shape == (n_samples, example_zinbvcp_results.n_genes)
    assert samples['gate'].shape == (n_samples, example_zinbvcp_results.n_genes)
    assert samples['p_capture'].shape == (n_samples, example_zinbvcp_results.n_cells)
    
    # Test that samples are in valid ranges
    assert jnp.all(samples['p'] >= 0) and jnp.all(samples['p'] <= 1)
    assert jnp.all(samples['r'] > 0)
    assert jnp.all(samples['gate'] >= 0) and jnp.all(samples['gate'] <= 1)
    assert jnp.all(samples['p_capture'] >= 0) and jnp.all(samples['p_capture'] <= 1)

def test_predictive_sampling(example_zinbvcp_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 50
    
    # First get posterior samples
    posterior_samples = example_zinbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    # Generate predictive samples
    predictive_samples = example_zinbvcp_results.get_predictive_samples(
        rng_key=random.split(rng_key)[1]
    )
    
    expected_shape = (
        n_samples,
        example_zinbvcp_results.n_cells,
        example_zinbvcp_results.n_genes
    )
    assert predictive_samples.shape == expected_shape
    assert jnp.all(predictive_samples >= 0)  # Counts should be non-negative

def test_ppc_sampling(example_zinbvcp_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 50
    
    ppc_samples = example_zinbvcp_results.get_ppc_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'parameter_samples' in ppc_samples
    assert 'predictive_samples' in ppc_samples
    
    # Check parameter samples
    assert 'p' in ppc_samples['parameter_samples']
    assert 'r' in ppc_samples['parameter_samples']
    assert 'gate' in ppc_samples['parameter_samples']
    assert 'p_capture' in ppc_samples['parameter_samples']
    
    # Check predictive samples shape
    expected_shape = (
        n_samples,
        example_zinbvcp_results.n_cells,
        example_zinbvcp_results.n_genes
    )
    assert ppc_samples['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_distributions(example_zinbvcp_results):
    """Test getting variational distributions."""
    distributions = example_zinbvcp_results.get_distributions()
    
    assert 'p' in distributions
    assert 'r' in distributions
    assert 'gate' in distributions
    assert 'p_capture' in distributions

def test_indexing(example_zinbvcp_results):
    """Test indexing functionality."""
    # Single gene
    subset = example_zinbvcp_results[0]
    assert isinstance(subset, ZINBVCPResults)
    assert subset.n_genes == 1
    assert subset.n_cells == example_zinbvcp_results.n_cells
    
    # Multiple genes
    subset = example_zinbvcp_results[0:2]
    assert isinstance(subset, ZINBVCPResults)
    assert subset.n_genes == 2
    assert subset.n_cells == example_zinbvcp_results.n_cells
    
    # Boolean indexing
    mask = jnp.array([True, False, True] + [False] * (example_zinbvcp_results.n_genes - 3))
    subset = example_zinbvcp_results[mask]
    assert isinstance(subset, ZINBVCPResults)
    assert subset.n_genes == int(mask.sum())

def test_parameter_subsetting(example_zinbvcp_results):
    """Test that parameter subsetting works correctly."""
    subset = example_zinbvcp_results[0:2]
    
    # Check that gene-specific parameters are subset
    assert subset.params['alpha_r'].shape == (2,)
    assert subset.params['beta_r'].shape == (2,)
    assert subset.params['alpha_gate'].shape == (2,)
    assert subset.params['beta_gate'].shape == (2,)
    
    # Check that cell-specific parameters remain unchanged
    assert subset.params['alpha_p_capture'].shape == (example_zinbvcp_results.n_cells,)
    assert subset.params['beta_p_capture'].shape == (example_zinbvcp_results.n_cells,)
    
    # Check that shared parameters remain the same
    assert subset.params['alpha_p'].shape == ()
    assert subset.params['beta_p'].shape == ()