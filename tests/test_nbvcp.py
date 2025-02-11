# tests/test_nbvcp.py
"""
Tests for the Negative Binomial with Variable Capture Probability model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from scribe.models import nbvcp_model, nbvcp_guide
from scribe.svi import run_scribe, rerun_scribe
from scribe.sampling import (
    sample_variational_posterior,
    generate_predictive_samples,
    generate_ppc_samples
)
from scribe.results import NBVCPResults

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define number of steps for testing
N_STEPS = 3
# Define number of samples for testing
N_SAMPLES = 10

@pytest.fixture
def example_nbvcp_results(small_dataset, rng_key):
    """Generate example NBVCP results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        model_type="nbvcp",
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
        model_type="nbvcp",
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )
    
    assert isinstance(results, NBVCPResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

def test_parameter_ranges(example_nbvcp_results):
    """Test that inferred parameters are in valid ranges."""
    # All parameters should be positive
    assert jnp.all(example_nbvcp_results.params['alpha_p'] > 0)
    assert jnp.all(example_nbvcp_results.params['beta_p'] > 0)
    assert jnp.all(example_nbvcp_results.params['alpha_r'] > 0)
    assert jnp.all(example_nbvcp_results.params['beta_r'] > 0)
    assert jnp.all(example_nbvcp_results.params['alpha_p_capture'] > 0)
    assert jnp.all(example_nbvcp_results.params['beta_p_capture'] > 0)

    # Check shapes for cell-specific parameters
    assert example_nbvcp_results.params['alpha_p_capture'].shape == (example_nbvcp_results.n_cells,)
    assert example_nbvcp_results.params['beta_p_capture'].shape == (example_nbvcp_results.n_cells,)

def test_continue_training(example_nbvcp_results, small_dataset, rng_key):
    """Test that continuing training from a previous results object works."""
    counts, _ = small_dataset
    n_cells, n_genes = counts.shape
    # Run inference again
    results = rerun_scribe(
        results=example_nbvcp_results,
        counts=counts,
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5,
    )

    assert isinstance(results, NBVCPResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS + N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_nbvcp_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 100
    samples = example_nbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'p' in samples
    assert 'r' in samples
    assert 'p_capture' in samples
    assert samples['p'].shape == (n_samples,)
    assert samples['r'].shape == (n_samples, example_nbvcp_results.n_genes)
    assert samples['p_capture'].shape == (n_samples, example_nbvcp_results.n_cells)
    
    # Test that samples are in valid ranges
    assert jnp.all(samples['p'] >= 0) and jnp.all(samples['p'] <= 1)
    assert jnp.all(samples['r'] > 0)
    assert jnp.all(samples['p_capture'] >= 0) and jnp.all(samples['p_capture'] <= 1)

def test_predictive_sampling(example_nbvcp_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 50
    
    # First get posterior samples
    posterior_samples = example_nbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    # Generate predictive samples
    predictive_samples = example_nbvcp_results.get_predictive_samples(
        rng_key=random.split(rng_key)[1]
    )
    
    expected_shape = (
        n_samples,
        example_nbvcp_results.n_cells,
        example_nbvcp_results.n_genes
    )
    assert predictive_samples.shape == expected_shape
    assert jnp.all(predictive_samples >= 0)  # Counts should be non-negative

def test_ppc_sampling(example_nbvcp_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 50
    
    ppc_samples = example_nbvcp_results.get_ppc_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'parameter_samples' in ppc_samples
    assert 'predictive_samples' in ppc_samples
    
    # Check parameter samples
    assert 'p' in ppc_samples['parameter_samples']
    assert 'r' in ppc_samples['parameter_samples']
    assert 'p_capture' in ppc_samples['parameter_samples']
    
    # Check predictive samples shape
    expected_shape = (
        n_samples,
        example_nbvcp_results.n_cells,
        example_nbvcp_results.n_genes
    )
    assert ppc_samples['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_distributions(example_nbvcp_results):
    """Test getting variational distributions."""
    distributions = example_nbvcp_results.get_distributions()
    
    assert 'p' in distributions
    assert 'r' in distributions
    assert 'p_capture' in distributions

def test_indexing(example_nbvcp_results):
    """Test indexing functionality."""
    # Single gene
    subset = example_nbvcp_results[0]
    assert isinstance(subset, NBVCPResults)
    assert subset.n_genes == 1
    assert subset.n_cells == example_nbvcp_results.n_cells
    
    # Multiple genes
    subset = example_nbvcp_results[0:2]
    assert isinstance(subset, NBVCPResults)
    assert subset.n_genes == 2
    assert subset.n_cells == example_nbvcp_results.n_cells
    
    # Boolean indexing
    mask = jnp.array([True, False, True] + [False] * (example_nbvcp_results.n_genes - 3))
    subset = example_nbvcp_results[mask]
    assert isinstance(subset, NBVCPResults)
    assert subset.n_genes == int(mask.sum())

def test_parameter_subsetting(example_nbvcp_results):
    """Test that parameter subsetting works correctly."""
    subset = example_nbvcp_results[0:2]
    
    # Check that gene-specific parameters are subset
    assert subset.params['alpha_r'].shape == (2,)
    assert subset.params['beta_r'].shape == (2,)
    
    # Check that cell-specific parameters remain unchanged
    assert subset.params['alpha_p_capture'].shape == (example_nbvcp_results.n_cells,)
    assert subset.params['beta_p_capture'].shape == (example_nbvcp_results.n_cells,)
    
    # Check that shared parameters remain the same
    assert subset.params['alpha_p'].shape == ()
    assert subset.params['beta_p'].shape == ()

# ------------------------------------------------------------------------------
# Test log likelihood
# ------------------------------------------------------------------------------

def test_log_likelihood(example_nbvcp_results, small_dataset, rng_key):
    """Test evaluation of the log likelihood function."""
    # Get counts and total counts from dataset
    counts, _ = small_dataset
    total_counts = counts.sum(axis=1)
    
    # Get posterior samples 
    # - these should already be available from previous tests
    example_nbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES
    )
    
    # Test log likelihood evaluation - with cell axis
    cell_log_liks = example_nbvcp_results.compute_log_likelihood(
        counts=counts, 
        cells_axis=0,
        return_by='cell'
    )
    
    # Shape checks for cell-wise log likelihood
    assert cell_log_liks.shape[0] == N_SAMPLES  # n_samples
    assert cell_log_liks.shape[1] == example_nbvcp_results.n_cells  # n_cells
    
    # Test log likelihood evaluation - with gene axis
    gene_log_liks = example_nbvcp_results.compute_log_likelihood(
        counts=counts, 
        cells_axis=0,
        return_by='gene'
    )
    
    # Shape checks for gene-wise log likelihood
    assert gene_log_liks.shape[0] == N_SAMPLES  # n_samples
    assert gene_log_liks.shape[1] == example_nbvcp_results.n_genes  # n_genes
    
    # Basic sanity checks
    assert jnp.all(jnp.isfinite(cell_log_liks))  # No NaNs or infs
    assert jnp.all(jnp.isfinite(gene_log_liks))  # No NaNs or infs
    assert jnp.all(cell_log_liks <= 0)  # Log likelihoods should be <= 0
    assert jnp.all(gene_log_liks <= 0)  # Log likelihoods should be <= 0

def test_log_likelihood_batching(example_nbvcp_results, small_dataset, rng_key):
    """Test that batched and non-batched log likelihood give same results."""
    # Get counts and total counts from dataset
    counts, _ = small_dataset
    total_counts = counts.sum(axis=1)
    
    # Get posterior samples
    example_nbvcp_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES
    )
    
    # Compute log likelihood without batching
    full_log_liks = example_nbvcp_results.compute_log_likelihood(
        counts=counts, 
        cells_axis=0,
        return_by='cell'
    )
    
    # Compute log likelihood with batching
    batched_log_liks = example_nbvcp_results.compute_log_likelihood(
        counts=counts, 
        batch_size=5,  # Small batch size for testing
        cells_axis=0,
        return_by='cell'
    )
    
    # Check that results match
    assert jnp.allclose(full_log_liks, batched_log_liks, rtol=1e-5)