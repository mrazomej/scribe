# tests/test_mcmc_nbdm.py
"""
Tests for the Negative Binomial-Dirichlet Multinomial model using MCMC inference.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
from numpyro.infer import NUTS
from scribe.models import nbdm_model, nbdm_log_likelihood
from scribe.mcmc import run_scribe, create_mcmc_instance
from scribe.sampling import generate_predictive_samples
from scribe.model_config import ModelConfig
from scribe.model_registry import get_unconstrained_model, get_log_likelihood_fn
from scribe.results_mcmc import ScribeMCMCResults

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define minimal number of steps for testing - reduced to keep tests fast
NUM_WARMUP = 2
NUM_SAMPLES = 3
NUM_CHAINS = 1

@pytest.fixture
def example_nbdm_results(small_dataset, rng_key):
    """Generate example NBDM MCMC results for testing."""
    counts, _ = small_dataset
    return run_scribe(
        counts=counts,
        zero_inflated=False,
        variable_capture=False,
        mixture_model=False,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        r_prior=(0, 1),
        p_prior=(0, 1),
        seed=42
    )

# ------------------------------------------------------------------------------
# Test MCMC instance creation
# ------------------------------------------------------------------------------

def test_create_mcmc_instance():
    """Test that MCMC instance can be created with correct parameters."""
    mcmc_instance = create_mcmc_instance(
        model_type="nbdm",
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="parallel",
        kernel=NUTS,
        kernel_kwargs={"target_accept_prob": 0.9},
    )
    
    assert mcmc_instance.num_warmup == NUM_WARMUP
    assert mcmc_instance.num_samples == NUM_SAMPLES
    assert mcmc_instance.num_chains == NUM_CHAINS
    assert mcmc_instance.chain_method == "parallel"
    assert isinstance(mcmc_instance.sampler, NUTS)
    assert mcmc_instance.sampler._target_accept_prob == 0.9

# ------------------------------------------------------------------------------
# Test inference
# ------------------------------------------------------------------------------

def test_inference_run(small_dataset, rng_key):
    """Test that MCMC inference runs and produces expected results."""
    counts, _ = small_dataset
    n_cells, n_genes = counts.shape
    
    results = run_scribe(
        counts=counts,
        zero_inflated=False,
        variable_capture=False,
        mixture_model=False,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        seed=42
    )
    
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert results.model_type == "nbdm"
    assert isinstance(results.model_config, ModelConfig)
    assert isinstance(results, ScribeMCMCResults)
    
    # Check that samples were generated
    samples = results.get_samples()
    assert 'p' in samples
    assert 'r' in samples
    assert samples['p'].shape[0] == NUM_SAMPLES
    assert samples['r'].shape[0] == NUM_SAMPLES
    assert samples['r'].shape[1] == n_genes

def test_parameter_ranges(example_nbdm_results):
    """Test that inferred parameters are in valid ranges."""
    # Get posterior samples
    samples = example_nbdm_results.get_samples()
    
    # All r parameters should be positive
    assert jnp.all(samples['r'] > 0)
        
    # Check p parameters - should be between 0 and 1
    assert jnp.all(samples['p'] >= 0)
    assert jnp.all(samples['p'] <= 1)

# ------------------------------------------------------------------------------
# Test model function retrieval
# ------------------------------------------------------------------------------

def test_get_model_function():
    """Test that model functions can be retrieved correctly."""
    model = get_unconstrained_model("nbdm")
    
    assert callable(model)

def test_get_log_likelihood_fn():
    """Test that log likelihood function returns the correct function."""
    log_lik_fn = get_log_likelihood_fn("nbdm")
    
    assert log_lik_fn == nbdm_log_likelihood

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_samples(example_nbdm_results):
    """Test access to posterior samples."""
    samples = example_nbdm_results.get_posterior_samples()
    
    # Check structure of posterior samples
    assert 'p' in samples
    assert 'r' in samples
    
    # Check dimensions of samples
    assert samples['p'].shape == (NUM_SAMPLES,)
    assert samples['r'].shape == (NUM_SAMPLES, example_nbdm_results.n_genes)
    
    # Test that samples are in valid ranges
    assert jnp.all(samples['p'] >= 0) 
    assert jnp.all(samples['p'] <= 1)
    assert jnp.all(samples['r'] > 0)

def test_posterior_quantiles(example_nbdm_results):
    """Test computing quantiles from posterior samples."""
    # Test for p parameter
    p_quantiles = example_nbdm_results.get_posterior_quantiles('p')
    
    assert 0.025 in p_quantiles
    assert 0.5 in p_quantiles
    assert 0.975 in p_quantiles
    
    # Test for r parameter
    r_quantiles = example_nbdm_results.get_posterior_quantiles('r')
    
    assert 0.025 in r_quantiles
    assert 0.5 in r_quantiles
    assert 0.975 in r_quantiles

def test_predictive_sampling(example_nbdm_results, rng_key):
    """Test generating predictive samples."""
    # Generate predictive samples
    pred_samples = example_nbdm_results.get_ppc_samples(
        rng_key=random.PRNGKey(43),
        store_samples=True
    )
    
    expected_shape = (
        NUM_SAMPLES, 
        example_nbdm_results.n_cells, 
        example_nbdm_results.n_genes
    )
    assert pred_samples.shape == expected_shape
    assert jnp.all(pred_samples >= 0)  # Counts should be non-negative
    
    # Check stored samples
    assert example_nbdm_results.predictive_samples is not None
    assert example_nbdm_results.predictive_samples.shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_map(example_nbdm_results):
    """Test getting MAP estimates."""
    map_estimates = example_nbdm_results.get_map()
    
    assert 'p' in map_estimates
    assert 'r' in map_estimates
    assert map_estimates['p'].shape == ()  # scalar
    assert map_estimates['r'].shape == (example_nbdm_results.n_genes,)
    
def test_model_retrieval(example_nbdm_results):
    """Test that model function can be retrieved from results."""
    model = example_nbdm_results._model()
    
    assert callable(model)

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
    samples = example_nbdm_results.get_samples()
    subset_samples = subset.get_posterior_samples()
    
    assert subset_samples['r'].shape == (NUM_SAMPLES, 1)
    assert jnp.allclose(subset_samples['r'], samples['r'][:, 0:1])

def test_getitem_slice(example_nbdm_results):
    """Test indexing with a slice."""
    subset = example_nbdm_results[0:2]
    
    assert subset.n_genes == 2
    assert subset.n_cells == example_nbdm_results.n_cells
    assert subset.model_type == example_nbdm_results.model_type
    
    # Check r parameters are correctly subset
    samples = example_nbdm_results.get_samples()
    subset_samples = subset.get_posterior_samples()
    
    assert subset_samples['r'].shape == (NUM_SAMPLES, 2)
    assert jnp.allclose(subset_samples['r'], samples['r'][:, 0:2])

def test_getitem_boolean(example_nbdm_results):
    """Test indexing with a boolean array."""
    mask = jnp.array([True, False, True] + [False] * (example_nbdm_results.n_genes - 3))
    subset = example_nbdm_results[mask]
    
    assert subset.n_genes == int(jnp.sum(mask))
    assert subset.n_cells == example_nbdm_results.n_cells
    
    # Check r parameters are correctly subset
    samples = example_nbdm_results.get_samples()
    subset_samples = subset.get_posterior_samples()
    
    assert subset_samples['r'].shape == (NUM_SAMPLES, int(jnp.sum(mask)))
    assert jnp.allclose(subset_samples['r'], samples['r'][:, mask])

# ------------------------------------------------------------------------------
# Test log likelihood
# ------------------------------------------------------------------------------

def test_compute_log_likelihood(example_nbdm_results, small_dataset, rng_key):
    """Test computing log likelihood with the model."""
    counts, _ = small_dataset
    
    # Compute per-cell log likelihood
    cell_ll = example_nbdm_results.log_likelihood(
        counts,
        return_by='cell'
    )
    
    # Check shape - should be (n_samples, n_cells)
    assert cell_ll.shape == (NUM_SAMPLES, example_nbdm_results.n_cells)
    assert jnp.all(jnp.isfinite(cell_ll))
    assert jnp.all(cell_ll <= 0)  # Log likelihoods should be negative
    
    # Compute per-gene log likelihood
    gene_ll = example_nbdm_results.log_likelihood(
        counts,
        return_by='gene'
    )
    
    # Check shape - should be (n_samples, n_genes)
    assert gene_ll.shape == (NUM_SAMPLES, example_nbdm_results.n_genes)
    assert jnp.all(jnp.isfinite(gene_ll))

def test_compute_log_likelihood_batched(example_nbdm_results, small_dataset, rng_key):
    """Test computing log likelihood with batching."""
    counts, _ = small_dataset
    
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

# ------------------------------------------------------------------------------
# Test prior configuration
# ------------------------------------------------------------------------------

def test_model_with_custom_priors(small_dataset, rng_key):
    """Test that custom priors can be specified."""
    counts, _ = small_dataset
    
    # Custom priors
    r_prior = (1.0, 0.5)
    p_prior = (2.0, 0.5)
    
    results = run_scribe(
        counts=counts,
        zero_inflated=False,
        variable_capture=False,
        mixture_model=False,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        r_prior=r_prior,
        p_prior=p_prior,
        seed=42
    )
    
    # Check that priors were stored correctly
    assert results.prior_params['r_prior'] == r_prior
    assert results.prior_params['p_prior'] == p_prior
    
    # Check model config
    assert results.model_config.r_unconstrained_loc == r_prior[0]
    assert results.model_config.r_unconstrained_scale == r_prior[1]
    assert results.model_config.p_unconstrained_loc == p_prior[0]
    assert results.model_config.p_unconstrained_scale == p_prior[1] 