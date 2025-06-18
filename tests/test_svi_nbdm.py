# tests/test_nbdm.py
"""
Tests for the Negative Binomial-Dirichlet Multinomial model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from scribe.models import nbdm_model, nbdm_guide, nbdm_log_likelihood
from scribe.svi import run_scribe
from scribe.sampling import (
    sample_variational_posterior, 
    generate_predictive_samples,
    generate_ppc_samples
)
from scribe.model_config import ModelConfig
from scribe.model_registry import get_model_and_guide, get_log_likelihood_fn

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define number of steps for testing
N_STEPS = 3
# Define number of samples for testing
N_SAMPLES = 10

@pytest.fixture
def example_nbdm_results(small_dataset, rng_key):
    """Generate example NBDM results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        zero_inflated=False,
        variable_capture=False,
        mixture_model=False,
        n_steps=N_STEPS,
        batch_size=5,
        r_dist="gamma",
        r_prior=(2, 0.1),
        p_prior=(1, 1),
        seed=42
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
        zero_inflated=False,
        variable_capture=False,
        mixture_model=False,
        n_steps=N_STEPS,
        batch_size=5,
        seed=42
    )
    
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease
    assert results.model_type == "nbdm"
    assert isinstance(results.model_config, ModelConfig)

def test_parameter_ranges(example_nbdm_results):
    """Test that inferred parameters are in valid ranges."""
    # All parameters should be positive - look for r_distribution parameters
    r_params = [param for param in example_nbdm_results.params if param.startswith('r_')]
    for param in r_params:
        assert jnp.all(example_nbdm_results.params[param] > 0)
        
    # Check p parameters - should be positive and appropriate for Beta distribution
    p_params = [param for param in example_nbdm_results.params if param.startswith('p_')]
    for param in p_params:
        assert jnp.all(example_nbdm_results.params[param] > 0)

# ------------------------------------------------------------------------------
# Test model and guide functions
# ------------------------------------------------------------------------------

def test_get_model_and_guide():
    """Test that get_model_and_guide returns the correct functions."""
    model, guide = get_model_and_guide("nbdm")
    
    assert model == nbdm_model
    assert guide == nbdm_guide

def test_get_log_likelihood_fn():
    """Test that get_log_likelihood_fn returns the correct function."""
    log_lik_fn = get_log_likelihood_fn("nbdm")
    
    assert log_lik_fn == nbdm_log_likelihood

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_nbdm_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 10
    posterior_samples = example_nbdm_results.get_posterior_samples(
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        store_samples=True
    )
    
    # Check structure of posterior samples
    assert 'p' in posterior_samples
    assert 'r' in posterior_samples
    
    # Check dimensions of samples
    assert posterior_samples['p'].shape == (n_samples,)
    assert posterior_samples['r'].shape == (n_samples, example_nbdm_results.n_genes)
    
    # Test that samples are in valid ranges
    assert jnp.all(posterior_samples['p'] >= 0) 
    assert jnp.all(posterior_samples['p'] <= 1)
    assert jnp.all(posterior_samples['r'] > 0)

def test_predictive_sampling(example_nbdm_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 10
    
    # First get posterior samples
    example_nbdm_results.get_posterior_samples(
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        store_samples=True
    )
    
    # Then generate predictive samples
    pred_samples = example_nbdm_results.get_predictive_samples(
        rng_key=random.PRNGKey(43),
        store_samples=True
    )
    
    expected_shape = (
        n_samples, 
        example_nbdm_results.n_cells, 
        example_nbdm_results.n_genes
    )
    assert pred_samples.shape == expected_shape
    assert jnp.all(pred_samples >= 0)  # Counts should be non-negative

def test_ppc_sampling(example_nbdm_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 10
    
    ppc_results = example_nbdm_results.get_ppc_samples(
        rng_key=random.PRNGKey(44),
        n_samples=n_samples,
        store_samples=True
    )
    
    assert 'parameter_samples' in ppc_results
    assert 'predictive_samples' in ppc_results
    
    # Check predictive samples shape
    expected_shape = (n_samples, example_nbdm_results.n_cells, example_nbdm_results.n_genes)
    assert ppc_results['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_distributions(example_nbdm_results):
    """Test getting distribution objects from results."""
    # Get scipy distributions
    scipy_dists = example_nbdm_results.get_distributions(backend="scipy")
    assert 'p' in scipy_dists
    assert 'r' in scipy_dists
    
    # Get numpyro distributions
    numpyro_dists = example_nbdm_results.get_distributions(backend="numpyro")
    assert 'p' in numpyro_dists
    assert 'r' in numpyro_dists

def test_get_map(example_nbdm_results):
    """Test getting MAP estimates."""
    map_estimates = example_nbdm_results.get_map()
    
    assert 'p' in map_estimates
    assert 'r' in map_estimates
    assert map_estimates['p'].shape == ()  # scalar
    assert map_estimates['r'].shape == (example_nbdm_results.n_genes,)
    
def test_model_and_guide_retrieval(example_nbdm_results):
    """Test that model and guide functions can be retrieved from results."""
    model, guide = example_nbdm_results._model_and_guide()
    
    assert callable(model)
    assert callable(guide)

def test_log_likelihood_fn_retrieval(example_nbdm_results):
    """Test that log likelihood function can be retrieved from results."""
    log_lik_fn = example_nbdm_results._log_likelihood_fn()
    
    assert callable(log_lik_fn)

# ------------------------------------------------------------------------------
# Test indexing
# ------------------------------------------------------------------------------

def test_getitem_integer(example_nbdm_results):
    """Test indexing with an integer."""
    subset = example_nbdm_results[0]
    
    assert subset.n_genes == 1
    assert subset.n_cells == example_nbdm_results.n_cells
    assert subset.model_type == example_nbdm_results.model_type
    
    # Check r parameters are correctly subset
    for param in example_nbdm_results.params:
        if param.startswith('r_'):
            # Gene-specific parameters should be subset
            orig_shape = example_nbdm_results.params[param].shape
            if len(orig_shape) > 0 and orig_shape[0] == example_nbdm_results.n_genes:
                assert subset.params[param].shape == (1,)

def test_getitem_slice(example_nbdm_results):
    """Test indexing with a slice."""
    subset = example_nbdm_results[0:2]
    
    assert subset.n_genes == 2
    assert subset.n_cells == example_nbdm_results.n_cells
    assert subset.model_type == example_nbdm_results.model_type
    
    # Check r parameters are correctly subset
    for param in example_nbdm_results.params:
        if param.startswith('r_'):
            # Gene-specific parameters should be subset
            orig_shape = example_nbdm_results.params[param].shape
            if len(orig_shape) > 0 and orig_shape[0] == example_nbdm_results.n_genes:
                assert subset.params[param].shape == (2,)

def test_getitem_boolean(example_nbdm_results):
    """Test indexing with a boolean array."""
    mask = jnp.array([True, False, True] + [False] * (example_nbdm_results.n_genes - 3))
    subset = example_nbdm_results[mask]
    
    assert subset.n_genes == int(jnp.sum(mask))
    assert subset.n_cells == example_nbdm_results.n_cells
    
    # Check r parameters are correctly subset
    for param in example_nbdm_results.params:
        if param.startswith('r_'):
            # Gene-specific parameters should be subset
            orig_shape = example_nbdm_results.params[param].shape
            if len(orig_shape) > 0 and orig_shape[0] == example_nbdm_results.n_genes:
                assert subset.params[param].shape == (int(jnp.sum(mask)),)

def test_subset_with_posterior_samples(example_nbdm_results, rng_key):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    example_nbdm_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Subset results
    subset = example_nbdm_results[0:2]
    
    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None
    assert 'r' in subset.posterior_samples
    assert 'p' in subset.posterior_samples
    assert subset.posterior_samples['r'].shape == (N_SAMPLES, 2)
    assert subset.posterior_samples['p'].shape == (N_SAMPLES,)

# ------------------------------------------------------------------------------
# Test log likelihood
# ------------------------------------------------------------------------------

def test_compute_log_likelihood(example_nbdm_results, small_dataset, rng_key):
    """Test computing log likelihood with the model."""
    counts, _ = small_dataset
    
    # First generate posterior samples
    example_nbdm_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Compute per-cell log likelihood
    cell_ll = example_nbdm_results.log_likelihood(
        counts,
        return_by='cell'
    )
    
    # Check shape - should be (n_samples, n_cells)
    assert cell_ll.shape == (N_SAMPLES, example_nbdm_results.n_cells)
    assert jnp.all(jnp.isfinite(cell_ll))
    assert jnp.all(cell_ll <= 0)  # Log likelihoods should be negative
    
    # Compute per-gene log likelihood
    gene_ll = example_nbdm_results.log_likelihood(
        counts,
        return_by='gene'
    )
    
    # Check shape - should be (n_samples, n_genes)
    assert gene_ll.shape == (N_SAMPLES, example_nbdm_results.n_genes)
    assert jnp.all(jnp.isfinite(gene_ll))

def test_compute_log_likelihood_batched(example_nbdm_results, small_dataset, rng_key):
    """Test computing log likelihood with batching."""
    counts, _ = small_dataset
    
    # First generate posterior samples
    example_nbdm_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Compute without batching
    full_ll = example_nbdm_results.log_likelihood(
        counts,
        return_by='cell'
    )
    
    # Compute with batching
    batched_ll = example_nbdm_results.log_likelihood(
        counts,
        batch_size=3,
        return_by='cell'
    )
    
    # Results should match
    assert jnp.allclose(full_ll, batched_ll, rtol=1e-5, atol=1e-5)