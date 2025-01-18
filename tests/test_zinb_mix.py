# tests/test_zinb_mix.py
"""
Tests for the Zero-Inflated Negative Binomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from scribe.models_mix import zinb_mixture_model, zinb_mixture_guide
from scribe.svi import run_scribe, rerun_scribe
from scribe.sampling import (
    sample_variational_posterior,
    generate_predictive_samples,
    generate_ppc_samples
)
from scribe.results import ZINBMixtureResults

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define number of steps for testing
N_STEPS = 3
N_COMPONENTS = 2  # Number of mixture components for testing

@pytest.fixture
def example_zinb_mix_results(small_dataset, rng_key):
    """Generate example ZINB mixture model results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        model_type="zinb_mix",
        n_components=N_COMPONENTS,
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
        model_type="zinb_mix",
        n_components=N_COMPONENTS,
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )
    
    assert isinstance(results, ZINBMixtureResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert results.n_components == N_COMPONENTS
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

def test_parameter_ranges(example_zinb_mix_results):
    """Test that inferred parameters are in valid ranges."""
    # All parameters should be positive
    assert jnp.all(example_zinb_mix_results.params['alpha_p'] > 0)
    assert jnp.all(example_zinb_mix_results.params['beta_p'] > 0)
    assert jnp.all(example_zinb_mix_results.params['alpha_r'] > 0)
    assert jnp.all(example_zinb_mix_results.params['beta_r'] > 0)
    assert jnp.all(example_zinb_mix_results.params['alpha_gate'] > 0)
    assert jnp.all(example_zinb_mix_results.params['beta_gate'] > 0)
    assert jnp.all(example_zinb_mix_results.params['alpha_mixing'] > 0)

    # Check shapes of component-specific parameters
    assert example_zinb_mix_results.params['alpha_mixing'].shape == (N_COMPONENTS,)
    assert example_zinb_mix_results.params['alpha_r'].shape == (N_COMPONENTS, example_zinb_mix_results.n_genes)
    assert example_zinb_mix_results.params['beta_r'].shape == (N_COMPONENTS, example_zinb_mix_results.n_genes)
    
    # Check shapes of gene-specific parameters (gate parameters are per gene)
    assert example_zinb_mix_results.params['alpha_gate'].shape == (N_COMPONENTS, example_zinb_mix_results.n_genes)
    assert example_zinb_mix_results.params['beta_gate'].shape == (N_COMPONENTS, example_zinb_mix_results.n_genes)

def test_continue_training(example_zinb_mix_results, small_dataset, rng_key):
    """Test that continuing training from a previous results object works."""
    counts, _ = small_dataset
    n_cells, n_genes = counts.shape
    
    # Run inference again
    results = rerun_scribe(
        results=example_zinb_mix_results,
        counts=counts,
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5,
    )

    assert isinstance(results, ZINBMixtureResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert results.n_components == N_COMPONENTS
    assert len(results.loss_history) == N_STEPS + N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_zinb_mix_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 100
    samples = example_zinb_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    # Check that all parameters are present
    assert 'mixing_weights' in samples
    assert 'p' in samples
    assert 'r' in samples
    assert 'gate' in samples
    
    # Check shapes
    assert samples['mixing_weights'].shape == (n_samples, N_COMPONENTS)
    assert samples['p'].shape == (n_samples,)
    assert samples['r'].shape == (n_samples, N_COMPONENTS, example_zinb_mix_results.n_genes)
    assert samples['gate'].shape == (n_samples, N_COMPONENTS, example_zinb_mix_results.n_genes)
    
    # Test that samples are in valid ranges
    assert jnp.all(samples['mixing_weights'] >= 0) and jnp.all(samples['mixing_weights'] <= 1)
    assert jnp.allclose(samples['mixing_weights'].sum(axis=1), 1.0)  # Mixing weights sum to 1
    assert jnp.all(samples['p'] >= 0) and jnp.all(samples['p'] <= 1)
    assert jnp.all(samples['r'] > 0)
    assert jnp.all(samples['gate'] >= 0) and jnp.all(samples['gate'] <= 1)

def test_predictive_sampling(example_zinb_mix_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 50
    
    # First get posterior samples
    posterior_samples = example_zinb_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    # Generate predictive samples
    predictive_samples = example_zinb_mix_results.get_predictive_samples(
        rng_key=random.split(rng_key)[1]
    )
    
    expected_shape = (
        n_samples,
        example_zinb_mix_results.n_cells,
        example_zinb_mix_results.n_genes
    )
    assert predictive_samples.shape == expected_shape
    assert jnp.all(predictive_samples >= 0)  # Counts should be non-negative

def test_ppc_sampling(example_zinb_mix_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 50
    
    ppc_samples = example_zinb_mix_results.get_ppc_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'parameter_samples' in ppc_samples
    assert 'predictive_samples' in ppc_samples
    
    # Check parameter samples
    assert 'mixing_weights' in ppc_samples['parameter_samples']
    assert 'p' in ppc_samples['parameter_samples']
    assert 'r' in ppc_samples['parameter_samples']
    assert 'gate' in ppc_samples['parameter_samples']
    
    # Check predictive samples shape
    expected_shape = (
        n_samples,
        example_zinb_mix_results.n_cells,
        example_zinb_mix_results.n_genes
    )
    assert ppc_samples['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_distributions(example_zinb_mix_results):
    """Test getting variational distributions."""
    distributions = example_zinb_mix_results.get_distributions()
    
    assert 'mixing_weights' in distributions
    assert 'p' in distributions
    assert 'r' in distributions
    assert 'gate' in distributions

def test_indexing(example_zinb_mix_results):
    """Test indexing functionality."""
    # Single gene
    subset = example_zinb_mix_results[0]
    assert isinstance(subset, ZINBMixtureResults)
    assert subset.n_genes == 1
    assert subset.n_cells == example_zinb_mix_results.n_cells
    assert subset.n_components == N_COMPONENTS
    
    # Multiple genes
    subset = example_zinb_mix_results[0:2]
    assert isinstance(subset, ZINBMixtureResults)
    assert subset.n_genes == 2
    assert subset.n_cells == example_zinb_mix_results.n_cells
    assert subset.n_components == N_COMPONENTS
    
    # Boolean indexing
    mask = jnp.array([True, False, True] + [False] * (example_zinb_mix_results.n_genes - 3))
    subset = example_zinb_mix_results[mask]
    assert isinstance(subset, ZINBMixtureResults)
    assert subset.n_genes == int(mask.sum())
    assert subset.n_components == N_COMPONENTS

def test_parameter_subsetting(example_zinb_mix_results):
    """Test that parameter subsetting works correctly."""
    subset = example_zinb_mix_results[0:2]
    
    # Check that gene-specific parameters are subset correctly
    # r parameters should maintain component dimension but subset gene dimension
    assert subset.params['alpha_r'].shape == (N_COMPONENTS, 2)
    assert subset.params['beta_r'].shape == (N_COMPONENTS, 2)
    
    # Gate parameters should be subset (they're gene-specific)
    assert subset.params['alpha_gate'].shape == (N_COMPONENTS, 2)
    assert subset.params['beta_gate'].shape == (N_COMPONENTS, 2)
    
    # Check that component-specific parameters remain unchanged
    assert subset.params['alpha_mixing'].shape == (N_COMPONENTS,)
    
    # Check that shared parameters remain unchanged
    assert subset.params['alpha_p'].shape == ()
    assert subset.params['beta_p'].shape == ()