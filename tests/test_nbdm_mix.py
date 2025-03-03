# tests/test_nbdm_mix.py
"""
Tests for the Negative Binomial-Dirichlet Multinomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from scribe.models_mix import nbdm_mixture_model, nbdm_mixture_guide, nbdm_mixture_log_likelihood
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
N_COMPONENTS = 2  # Number of mixture components for testing
N_SAMPLES = 11  # Number of samples for posterior sampling

@pytest.fixture
def example_nbdm_mix_results(small_dataset, rng_key):
    """Generate example NBDM mixture model results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        zero_inflated=False,
        variable_capture=False,
        mixture_model=True,
        n_components=N_COMPONENTS,
        n_steps=N_STEPS,
        batch_size=5,
        r_dist="gamma",
        r_prior=(2, 0.1),
        p_prior=(1, 1),
        mixing_prior=jnp.ones(N_COMPONENTS),
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
        mixture_model=True,
        mixing_prior=jnp.ones(N_COMPONENTS),
        n_components=N_COMPONENTS,
        n_steps=N_STEPS,
        batch_size=5,
        seed=42
    )
    
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert results.n_components == N_COMPONENTS
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease
    assert results.model_type == "nbdm_mix"
    assert isinstance(results.model_config, ModelConfig)

def test_parameter_ranges(example_nbdm_mix_results):
    """Test that inferred parameters are in valid ranges."""
    # All parameters should be positive - look for distribution parameters
    r_params = [
        param for param in example_nbdm_mix_results.params 
        if param.startswith('r_')
    ]
    for param in r_params:
        assert jnp.all(example_nbdm_mix_results.params[param] > 0)
        assert example_nbdm_mix_results.params[param].shape == (N_COMPONENTS, example_nbdm_mix_results.n_genes)
        
    # Check p parameters - should be positive
    p_params = [
        param for param in example_nbdm_mix_results.params 
        if param.startswith('p_')
    ]
    for param in p_params:
        assert jnp.all(example_nbdm_mix_results.params[param] > 0)
        
    # Check mixing parameters - should be positive
    mixing_params = [
        param for param in example_nbdm_mix_results.params 
        if param.startswith('mixing_')
    ]
    for param in mixing_params:
        assert jnp.all(example_nbdm_mix_results.params[param] > 0)
        assert example_nbdm_mix_results.params[param].shape == (N_COMPONENTS,)

# ------------------------------------------------------------------------------
# Test model and guide functions
# ------------------------------------------------------------------------------

def test_get_model_and_guide():
    """Test that get_model_and_guide returns the correct functions."""
    model, guide = get_model_and_guide("nbdm_mix")
    
    assert model == nbdm_mixture_model
    assert guide == nbdm_mixture_guide

def test_get_log_likelihood_fn():
    """Test that get_log_likelihood_fn returns the correct function."""
    log_lik_fn = get_log_likelihood_fn("nbdm_mix")
    
    assert log_lik_fn == nbdm_mixture_log_likelihood

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_nbdm_mix_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 10
    posterior_samples = example_nbdm_mix_results.get_posterior_samples(
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        store_samples=True
    )
    
    # Check structure of posterior samples
    assert 'mixing_weights' in posterior_samples
    assert 'p' in posterior_samples
    assert 'r' in posterior_samples
    
    # Check dimensions of samples
    assert posterior_samples['mixing_weights'].shape == (n_samples, N_COMPONENTS)
    assert posterior_samples['p'].shape == (n_samples,)
    assert posterior_samples['r'].shape == (n_samples, N_COMPONENTS, example_nbdm_mix_results.n_genes)
    
    # Test that samples are in valid ranges
    assert jnp.all(posterior_samples['mixing_weights'] >= 0)
    assert jnp.all(posterior_samples['mixing_weights'] <= 1)
    assert jnp.allclose(jnp.sum(posterior_samples['mixing_weights'], axis=1), jnp.ones(n_samples))
    assert jnp.all(posterior_samples['p'] >= 0) 
    assert jnp.all(posterior_samples['p'] <= 1)
    assert jnp.all(posterior_samples['r'] > 0)

def test_predictive_sampling(example_nbdm_mix_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 10
    
    # First get posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=random.PRNGKey(42),
        n_samples=n_samples,
        store_samples=True
    )
    
    # Then generate predictive samples
    pred_samples = example_nbdm_mix_results.get_predictive_samples(
        rng_key=random.PRNGKey(43),
        store_samples=True
    )
    
    expected_shape = (n_samples, example_nbdm_mix_results.n_cells, example_nbdm_mix_results.n_genes)
    assert pred_samples.shape == expected_shape
    assert jnp.all(pred_samples >= 0)  # Counts should be non-negative

def test_ppc_sampling(example_nbdm_mix_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 10
    
    ppc_results = example_nbdm_mix_results.get_ppc_samples(
        rng_key=random.PRNGKey(44),
        n_samples=n_samples,
        store_samples=True
    )
    
    assert 'parameter_samples' in ppc_results
    assert 'predictive_samples' in ppc_results
    
    # Check predictive samples shape
    expected_shape = (n_samples, example_nbdm_mix_results.n_cells, example_nbdm_mix_results.n_genes)
    assert ppc_results['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_distributions(example_nbdm_mix_results):
    """Test getting distribution objects from results."""
    # Get scipy distributions
    scipy_dists = example_nbdm_mix_results.get_distributions(backend="scipy")
    assert 'p' in scipy_dists
    assert 'r' in scipy_dists
    assert 'mixing_weights' in scipy_dists
    
    # Get numpyro distributions
    numpyro_dists = example_nbdm_mix_results.get_distributions(backend="numpyro")
    assert 'p' in numpyro_dists
    assert 'r' in numpyro_dists
    assert 'mixing_weights' in numpyro_dists

def test_get_map(example_nbdm_mix_results):
    """Test getting MAP estimates."""
    map_estimates = example_nbdm_mix_results.get_map()
    
    assert 'p' in map_estimates
    assert 'r' in map_estimates
    assert 'mixing_weights' in map_estimates
    assert map_estimates['p'].shape == ()  # scalar
    assert map_estimates['r'].shape == (N_COMPONENTS, example_nbdm_mix_results.n_genes)
    assert map_estimates['mixing_weights'].shape == (N_COMPONENTS,)

def test_model_and_guide_retrieval(example_nbdm_mix_results):
    """Test that model and guide functions can be retrieved from results."""
    model, guide = example_nbdm_mix_results.get_model_and_guide()
    
    assert callable(model)
    assert callable(guide)

def test_log_likelihood_fn_retrieval(example_nbdm_mix_results):
    """Test that log likelihood function can be retrieved from results."""
    log_lik_fn = example_nbdm_mix_results.get_log_likelihood_fn()
    
    assert callable(log_lik_fn)

# ------------------------------------------------------------------------------
# Test indexing
# ------------------------------------------------------------------------------

def test_getitem_integer(example_nbdm_mix_results):
    """Test indexing with an integer."""
    subset = example_nbdm_mix_results[0]
    
    assert subset.n_genes == 1
    assert subset.n_cells == example_nbdm_mix_results.n_cells
    assert subset.model_type == example_nbdm_mix_results.model_type
    assert subset.n_components == N_COMPONENTS
    
    # Check component and gene-specific parameters are correctly subset
    for param in example_nbdm_mix_results.params:
        if param.startswith('r_'):
            # Parameters should maintain component dimension but subset gene dimension
            assert subset.params[param].shape == (N_COMPONENTS, 1)

def test_getitem_slice(example_nbdm_mix_results):
    """Test indexing with a slice."""
    subset = example_nbdm_mix_results[0:2]
    
    assert subset.n_genes == 2
    assert subset.n_cells == example_nbdm_mix_results.n_cells
    assert subset.model_type == example_nbdm_mix_results.model_type
    assert subset.n_components == N_COMPONENTS
    
    # Check component and gene-specific parameters are correctly subset
    for param in example_nbdm_mix_results.params:
        if param.startswith('r_'):
            # Parameters should maintain component dimension but subset gene dimension
            assert subset.params[param].shape == (N_COMPONENTS, 2)

def test_getitem_boolean(example_nbdm_mix_results):
    """Test indexing with a boolean array."""
    mask = jnp.array([True, False, True] + [False] * (example_nbdm_mix_results.n_genes - 3))
    subset = example_nbdm_mix_results[mask]
    
    assert subset.n_genes == int(jnp.sum(mask))
    assert subset.n_cells == example_nbdm_mix_results.n_cells
    assert subset.n_components == N_COMPONENTS
    
    # Check component and gene-specific parameters are correctly subset
    for param in example_nbdm_mix_results.params:
        if param.startswith('r_'):
            # Parameters should maintain component dimension but subset gene dimension
            assert subset.params[param].shape == (N_COMPONENTS, int(jnp.sum(mask)))

def test_subset_with_posterior_samples(example_nbdm_mix_results, rng_key):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Subset results by genes
    subset = example_nbdm_mix_results[0:2]
    
    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None
    assert 'p' in subset.posterior_samples
    assert 'r' in subset.posterior_samples
    assert 'mixing_weights' in subset.posterior_samples
    assert subset.posterior_samples['r'].shape == (N_SAMPLES, N_COMPONENTS, 2)
    assert subset.posterior_samples['p'].shape == (N_SAMPLES,)
    assert subset.posterior_samples['mixing_weights'].shape == (N_SAMPLES, N_COMPONENTS)

# ------------------------------------------------------------------------------
# Test component selection
# ------------------------------------------------------------------------------

def test_get_component(example_nbdm_mix_results):
    """Test selecting a specific component from a mixture model."""
    # Select the first component
    component = example_nbdm_mix_results.get_component(0)
    
    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert component.model_type == "nbdm"  # Model type should lose _mix suffix
    
    # Check that gene counts are preserved
    assert component.n_genes == example_nbdm_mix_results.n_genes
    assert component.n_cells == example_nbdm_mix_results.n_cells
    
    # Check that parameters are correctly subset
    for param in example_nbdm_mix_results.params:
        if param.startswith('r_'):
            # Component dimension should be removed, gene dimension preserved
            orig_shape = example_nbdm_mix_results.params[param].shape
            if len(orig_shape) > 1 and orig_shape[0] == N_COMPONENTS:
                assert component.params[param].shape == (example_nbdm_mix_results.n_genes,)
                # Check values match
                assert jnp.allclose(
                    component.params[param],
                    example_nbdm_mix_results.params[param][0]
                )

def test_get_component_with_posterior_samples(example_nbdm_mix_results, rng_key):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Select component
    component = example_nbdm_mix_results.get_component(1)
    
    # Check posterior samples are correctly subset
    assert component.posterior_samples is not None
    assert 'p' in component.posterior_samples
    assert 'r' in component.posterior_samples
    
    # Check shapes
    assert component.posterior_samples['p'].shape == (N_SAMPLES,)
    assert component.posterior_samples['r'].shape == (
        N_SAMPLES, example_nbdm_mix_results.n_genes
    )
    
    # Check values match the original component values
    assert jnp.allclose(
        component.posterior_samples['r'],
        example_nbdm_mix_results.posterior_samples['r'][:, 1, :]
    )

def test_component_then_gene_indexing(example_nbdm_mix_results):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = example_nbdm_mix_results.get_component(0)
    
    # Then select genes
    gene_subset = component[0:2]
    
    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == example_nbdm_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "nbdm"
    
    # Check parameters
    for param in component.params:
        if param.startswith('r_'):
            # Gene dimension should be subset
            orig_shape = component.params[param].shape
            if len(orig_shape) > 0 and orig_shape[0] == example_nbdm_mix_results.n_genes:
                assert gene_subset.params[param].shape == (2,)

# ------------------------------------------------------------------------------
# Test log likelihood
# ------------------------------------------------------------------------------

def test_compute_log_likelihood(example_nbdm_mix_results, small_dataset, rng_key):
    """Test computing log likelihood with the model."""
    counts, _ = small_dataset
    
    # First generate posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Compute per-cell log likelihood (marginal over components)
    cell_ll = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='cell',
        split_components=False
    )
    
    # Check shape - should be (n_samples, n_cells)
    assert cell_ll.shape == (N_SAMPLES, example_nbdm_mix_results.n_cells)
    assert jnp.all(jnp.isfinite(cell_ll))
    assert jnp.all(cell_ll <= 0)  # Log likelihoods should be negative
    
    # Compute per-gene log likelihood (marginal over components)
    gene_ll = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='gene',
        split_components=False
    )
    
    # Check shape - should be (n_samples, n_genes)
    assert gene_ll.shape == (N_SAMPLES, example_nbdm_mix_results.n_genes)
    assert jnp.all(jnp.isfinite(gene_ll))
    
    # Now test with split_components=True
    # Compute per-cell log likelihood by component
    cell_ll_by_comp = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='cell',
        split_components=True
    )
    
    # Check shape - should be (n_samples, n_cells, n_components)
    assert cell_ll_by_comp.shape == (N_SAMPLES, example_nbdm_mix_results.n_cells, N_COMPONENTS)
    assert jnp.all(jnp.isfinite(cell_ll_by_comp))
    
    # Compute per-gene log likelihood by component
    gene_ll_by_comp = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='gene',
        split_components=True
    )
    
    # Check shape - should be (n_samples, n_genes, n_components)
    assert gene_ll_by_comp.shape == (N_SAMPLES, example_nbdm_mix_results.n_genes, N_COMPONENTS)
    assert jnp.all(jnp.isfinite(gene_ll_by_comp))

def test_compute_log_likelihood_batched(example_nbdm_mix_results, small_dataset, rng_key):
    """Test computing log likelihood with batching."""
    counts, _ = small_dataset
    
    # First generate posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Compute without batching (marginal over components)
    full_ll = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='cell',
        split_components=False
    )
    
    # Compute with batching
    batched_ll = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        batch_size=3,
        return_by='cell',
        split_components=False
    )
    
    # Results should match
    assert jnp.allclose(full_ll, batched_ll, rtol=1e-5, atol=1e-5)
    
    # Now test with split_components=True
    # Compute without batching
    full_ll_split = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        return_by='cell',
        split_components=True
    )
    
    # Compute with batching
    batched_ll_split = example_nbdm_mix_results.compute_log_likelihood(
        counts,
        batch_size=3,
        return_by='cell',
        split_components=True
    )
    
    # Results should match
    assert jnp.allclose(full_ll_split, batched_ll_split, rtol=1e-5, atol=1e-5)

# ------------------------------------------------------------------------------
# Test cell type assignments
# ------------------------------------------------------------------------------

def test_compute_cell_type_assignments(example_nbdm_mix_results, small_dataset, rng_key):
    """Test computing cell type assignments."""
    counts, _ = small_dataset
    
    # First generate posterior samples
    example_nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=N_SAMPLES,
        store_samples=True
    )
    
    # Compute assignments
    assignments = example_nbdm_mix_results.compute_cell_type_assignments(
        counts,
        fit_distribution=True,
        verbose=False
    )
    
    # Check structure of results
    assert 'concentration' in assignments
    assert 'mean_probabilities' in assignments
    assert 'sample_probabilities' in assignments
    
    # Check shapes
    assert assignments['concentration'].shape == (example_nbdm_mix_results.n_cells, N_COMPONENTS)
    assert assignments['mean_probabilities'].shape == (example_nbdm_mix_results.n_cells, N_COMPONENTS)
    assert assignments['sample_probabilities'].shape == (N_SAMPLES, example_nbdm_mix_results.n_cells, N_COMPONENTS)
    
    # Test without fitting distribution
    assignments_no_fit = example_nbdm_mix_results.compute_cell_type_assignments(
        counts,
        fit_distribution=False,
        verbose=False
    )
    
    assert 'sample_probabilities' in assignments_no_fit
    assert 'concentration' not in assignments_no_fit
    assert 'mean_probabilities' not in assignments_no_fit
    assert assignments_no_fit['sample_probabilities'].shape == (N_SAMPLES, example_nbdm_mix_results.n_cells, N_COMPONENTS)

def test_compute_cell_type_assignments_map(example_nbdm_mix_results, small_dataset):
    """Test computing cell type assignments using MAP estimates."""
    counts, _ = small_dataset
    
    # Compute assignments using MAP
    assignments = example_nbdm_mix_results.compute_cell_type_assignments_map(
        counts,
        verbose=False
    )
    
    # Check structure of results
    assert 'probabilities' in assignments
    
    # Check shapes
    assert assignments['probabilities'].shape == (example_nbdm_mix_results.n_cells, N_COMPONENTS)