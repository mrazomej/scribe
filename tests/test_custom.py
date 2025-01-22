"""
Tests for the CustomResults class using a custom NBDM model with LogNormal guide.
"""
import pytest
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from scribe.svi import run_scribe, rerun_scribe
from scribe.results import CustomResults

# ------------------------------------------------------------------------------
# Define custom model for testing
# ------------------------------------------------------------------------------

def nbdm_lognormal_model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """Custom NBDM model with LogNormal prior on r."""
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample(
        "r", dist.LogNormal(r_prior[0], r_prior[1]).expand([n_genes]))

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts
                )

            with numpyro.plate("cells", n_cells):
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts
                )
        else:
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts[idx]),
                    obs=counts[idx]
                )
    else:
        with numpyro.plate("cells", n_cells):
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            numpyro.sample("counts", dist_nb)

# ------------------------------------------------------------------------------
# Define custom guide for testing
# ------------------------------------------------------------------------------

def nbdm_lognormal_guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (0, 1),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """Custom guide function using LogNormal for r."""
    # Register parameters for p (keep Beta distribution)
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[0]),
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[1]),
        constraint=constraints.positive
    )

    # Register parameters for r (using LogNormal)
    mu_r = numpyro.param(
        "mu_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=constraints.real
    )
    sigma_r = numpyro.param(
        "sigma_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample from the variational distributions
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.LogNormal(mu_r, sigma_r))

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------

# Define number of steps for testing
N_STEPS = 3

@pytest.fixture
def param_spec():
    """Parameter specification for custom model."""
    return {
        "alpha_p": {"type": "global"},
        "beta_p": {"type": "global"},
        "mu_r": {"type": "gene-specific"},
        "sigma_r": {"type": "gene-specific"}
    }

# ------------------------------------------------------------------------------
# Define example custom results for testing
# ------------------------------------------------------------------------------

@pytest.fixture
def example_custom_results(small_dataset, rng_key, param_spec):
    """Generate example custom model results for testing."""
    counts, total_counts = small_dataset
    return run_scribe(
        counts=counts,
        model_type="nbdm_lognormal",
        custom_model=nbdm_lognormal_model,
        custom_guide=nbdm_lognormal_guide,
        param_spec=param_spec,
        custom_args={"total_counts": total_counts},
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )

# ------------------------------------------------------------------------------
# Test inference
# ------------------------------------------------------------------------------

def test_inference_run(small_dataset, rng_key, param_spec):
    """Test that inference runs and produces expected results."""
    counts, total_counts = small_dataset
    n_cells, n_genes = counts.shape
    
    results = run_scribe(
        counts=counts,
        model_type="nbdm_lognormal",
        custom_model=nbdm_lognormal_model,
        custom_guide=nbdm_lognormal_guide,
        param_spec=param_spec,
        custom_args={"total_counts": total_counts},
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5
    )
    
    assert isinstance(results, CustomResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS
    assert results.loss_history[-1] < results.loss_history[0]  # Loss should decrease

# ------------------------------------------------------------------------------
# Test parameter validation
# ------------------------------------------------------------------------------

def test_parameter_validation(example_custom_results):
    """Test parameter validation."""
    # Test that parameter shapes match specifications
    for param_name, spec in example_custom_results.param_spec.items():
        param = example_custom_results.params[param_name]
        
        if spec['type'] == 'global':
            assert param.ndim == 0 or param.shape == ()
        elif spec['type'] == 'gene-specific':
            assert param.shape[-1] == example_custom_results.n_genes
        elif spec['type'] == 'cell-specific':
            assert param.shape[-1] == example_custom_results.n_cells

# ------------------------------------------------------------------------------
# Test continuing training
# ------------------------------------------------------------------------------

def test_continue_training(example_custom_results, small_dataset, rng_key):
    """Test that continuing training from a previous results object works."""
    counts, total_counts = small_dataset
    n_cells, n_genes = counts.shape
    
    # Run inference again
    results = rerun_scribe(
        results=example_custom_results,
        custom_args={"total_counts": total_counts},
        counts=counts,
        rng_key=rng_key,
        n_steps=N_STEPS,
        batch_size=5,
    )

    assert isinstance(results, CustomResults)
    assert results.n_cells == n_cells
    assert results.n_genes == n_genes
    assert len(results.loss_history) == N_STEPS + N_STEPS

# ------------------------------------------------------------------------------
# Test sampling
# ------------------------------------------------------------------------------

def test_posterior_sampling(example_custom_results, rng_key):
    """Test sampling from the variational posterior."""
    n_samples = 100
    samples = example_custom_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'p' in samples
    assert 'r' in samples
    assert samples['p'].shape == (n_samples,)
    assert samples['r'].shape == (n_samples, example_custom_results.n_genes)
    
    # Test that samples are in valid ranges
    assert jnp.all(samples['p'] >= 0) and jnp.all(samples['p'] <= 1)
    assert jnp.all(samples['r'] > 0)

# ------------------------------------------------------------------------------
# Test predictive sampling
# ------------------------------------------------------------------------------

def test_predictive_sampling(example_custom_results, rng_key):
    """Test generating predictive samples."""
    n_samples = 50
    
    # First get posterior samples
    posterior_samples = example_custom_results.get_posterior_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    # Generate predictive samples
    predictive_samples = example_custom_results.get_predictive_samples(
        rng_key=random.split(rng_key)[1]
    )
    
    expected_shape = (
        n_samples,
        example_custom_results.n_cells,
        example_custom_results.n_genes
    )
    assert predictive_samples.shape == expected_shape
    assert jnp.all(predictive_samples >= 0)  # Counts should be non-negative

# ------------------------------------------------------------------------------
# Test posterior predictive check sampling
# ------------------------------------------------------------------------------

def test_ppc_sampling(example_custom_results, rng_key):
    """Test posterior predictive check sampling."""
    n_samples = 50
    
    ppc_samples = example_custom_results.get_ppc_samples(
        rng_key=rng_key,
        n_samples=n_samples
    )
    
    assert 'parameter_samples' in ppc_samples
    assert 'predictive_samples' in ppc_samples
    
    # Check parameter samples
    assert 'p' in ppc_samples['parameter_samples']
    assert 'r' in ppc_samples['parameter_samples']
    
    # Check predictive samples shape
    expected_shape = (
        n_samples,
        example_custom_results.n_cells,
        example_custom_results.n_genes
    )
    assert ppc_samples['predictive_samples'].shape == expected_shape

# ------------------------------------------------------------------------------
# Test results methods
# ------------------------------------------------------------------------------

def test_get_model_args(example_custom_results):
    """Test getting model arguments."""
    model_args = example_custom_results.get_model_args()
    assert 'n_cells' in model_args
    assert 'n_genes' in model_args
    assert model_args['n_cells'] == example_custom_results.n_cells
    assert model_args['n_genes'] == example_custom_results.n_genes

def test_get_distributions(example_custom_results):
    """Test getting variational distributions when no get_distributions_fn provided."""
    distributions = example_custom_results.get_distributions()
    assert isinstance(distributions, dict)
    assert len(distributions) == 0  # Should be empty if no get_distributions_fn

def test_indexing(example_custom_results):
    """Test indexing functionality."""
    # Single gene
    subset = example_custom_results[0]
    assert isinstance(subset, CustomResults)
    assert subset.n_genes == 1
    assert subset.n_cells == example_custom_results.n_cells
    
    # Multiple genes
    subset = example_custom_results[0:2]
    assert isinstance(subset, CustomResults)
    assert subset.n_genes == 2
    assert subset.n_cells == example_custom_results.n_cells
    
    # Boolean indexing
    mask = jnp.array([True, False, True] + [False] * (example_custom_results.n_genes - 3))
    subset = example_custom_results[mask]
    assert isinstance(subset, CustomResults)
    assert subset.n_genes == int(mask.sum())

def test_parameter_subsetting(example_custom_results):
    """Test that parameter subsetting works correctly."""
    subset = example_custom_results[0:2]
    
    # Check that gene-specific parameters are subset
    assert subset.params['mu_r'].shape == (2,)
    assert subset.params['sigma_r'].shape == (2,)
    
    # Check that global parameters remain unchanged
    assert subset.params['alpha_p'].shape == ()
    assert subset.params['beta_p'].shape == ()