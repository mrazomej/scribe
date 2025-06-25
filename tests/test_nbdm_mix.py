# tests/test_nbdm_mix.py
"""
Tests for the Negative Binomial-Dirichlet Multinomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import os

ALL_METHODS = ["svi"]  # MCMC not yet implemented for mixture models
ALL_PARAMETERIZATIONS = ["standard", "linked", "odds_ratio", "unconstrained"]

# ------------------------------------------------------------------------------
# Dynamic matrix parametrization
# ------------------------------------------------------------------------------


def pytest_generate_tests(metafunc):
    """
    Dynamically generate test combinations for inference methods and
    parameterizations.

    Excludes incompatible combinations and handles command-line options for
    selective testing.
    """
    # Check if this test function uses the required fixtures
    if {"nbdm_mix_results", "inference_method", "parameterization"}.issubset(
        metafunc.fixturenames
    ):
        # Get command-line options for method and parameterization
        method_opt = metafunc.config.getoption("--method")
        param_opt = metafunc.config.getoption("--parameterization")

        # Determine which methods to test based on command-line option
        methods = ALL_METHODS if method_opt == "all" else [method_opt]

        # Determine which parameterizations to test based on command-line option
        params = ALL_PARAMETERIZATIONS if param_opt == "all" else [param_opt]

        # Generate all valid combinations
        combinations = [(m, p) for m in methods for p in params]

        # Parametrize the test with the generated combinations
        metafunc.parametrize(
            "inference_method,parameterization",
            combinations,
        )


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device_type():
    # Default to CPU for matrix tests; can override with env var if needed
    return os.environ.get("SCRIBE_TEST_DEVICE", "cpu")


# Global cache for results
_nbdm_mix_results_cache = {}


@pytest.fixture(scope="function")
def nbdm_mix_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _nbdm_mix_results_cache:
        return _nbdm_mix_results_cache[key]

    # Configure JAX device
    if device_type == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        import jax

        jax.config.update("jax_platform_name", "cpu")
    else:
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]

    counts, _ = small_dataset

    # Set up priors based on parameterization
    if parameterization == "standard":
        priors = {"r_prior": (2, 0.1), "p_prior": (1, 1)}
    elif parameterization == "linked":
        priors = {"p_prior": (1, 1), "mu_prior": (0, 1)}
    elif parameterization == "odds_ratio":
        priors = {"phi_prior": (3, 2), "mu_prior": (0, 1)}
    elif parameterization == "unconstrained":
        priors = {}  # Unconstrained uses defaults
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    from scribe import run_scribe

    result = run_scribe(
        counts=counts,
        inference_method="svi",
        zero_inflated=False,
        variable_capture=False,
        mixture_model=True,
        n_components=2,  # Test with 2 components
        parameterization=parameterization,
        n_steps=3,
        batch_size=5,
        seed=42,
        r_prior=priors.get("r_prior"),
        p_prior=priors.get("p_prior"),
        mu_prior=priors.get("mu_prior"),
        phi_prior=priors.get("phi_prior"),
        mixing_prior=jnp.ones(2),  # Uniform prior for 2 components
    )

    _nbdm_mix_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(nbdm_mix_results):
    assert nbdm_mix_results.n_cells > 0
    assert nbdm_mix_results.n_genes > 0
    assert nbdm_mix_results.model_type == "nbdm_mix"
    assert hasattr(nbdm_mix_results, "model_config")
    assert nbdm_mix_results.n_components == 2


# ------------------------------------------------------------------------------


def test_parameterization_config(nbdm_mix_results, parameterization):
    """Test that the correct parameterization is used."""
    assert nbdm_mix_results.model_config.parameterization == parameterization


# ------------------------------------------------------------------------------


def test_parameter_ranges(nbdm_mix_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # Get parameters from results
    params = nbdm_mix_results.params

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p and r (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "r" in params or any(k.startswith("r_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "linked":
        # Linked parameterization: p and mu (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi and mu (component-specific)
        assert "phi" in params or any(
            k.startswith("phi_") for k in params.keys()
        )
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "unconstrained":
        # Unconstrained parameterization: unconstrained parameters
        assert any(k.startswith("p_unconstrained_") for k in params.keys())
        assert any(k.startswith("r_unconstrained_") for k in params.keys())
        assert any(
            k.startswith("mixing_logits_unconstrained_") for k in params.keys()
        )


# ------------------------------------------------------------------------------


def test_posterior_sampling(nbdm_mix_results, rng_key):
    """Test sampling from the variational posterior."""
    samples = nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Check that we have the expected parameters based on parameterization
    parameterization = nbdm_mix_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples
        assert (
            samples["r"].shape[-2] == nbdm_mix_results.n_components
        )  # Component dimension
        assert samples["r"].shape[-1] == nbdm_mix_results.n_genes
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples
        assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
        assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples
        assert samples["mu"].shape[-2] == nbdm_mix_results.n_components
        assert samples["mu"].shape[-1] == nbdm_mix_results.n_genes

    # Check mixing weights
    assert "mixing_weights" in samples
    assert samples["mixing_weights"].shape[-1] == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_predictive_sampling(nbdm_mix_results, rng_key):
    """Test generating predictive samples."""
    # First get posterior samples
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Then generate predictive samples
    pred = nbdm_mix_results.get_predictive_samples(
        rng_key=rng_key, store_samples=True
    )

    assert pred.shape[-1] == nbdm_mix_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(nbdm_mix_results):
    map_est = nbdm_mix_results.get_map()
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = nbdm_mix_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est

    # Check mixing weights
    assert "mixing_weights" in map_est


# ------------------------------------------------------------------------------


def test_indexing_integer(nbdm_mix_results):
    """Test indexing with an integer."""
    subset = nbdm_mix_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_slice(nbdm_mix_results):
    """Test indexing with a slice."""
    end_idx = min(2, nbdm_mix_results.n_genes)
    subset = nbdm_mix_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_boolean(nbdm_mix_results):
    """Test indexing with a boolean array."""
    mask = jnp.zeros(nbdm_mix_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    subset = nbdm_mix_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_mix_results.n_cells
    assert subset.n_components == nbdm_mix_results.n_components


# ------------------------------------------------------------------------------


def test_log_likelihood(nbdm_mix_results, small_dataset, rng_key):
    counts, _ = small_dataset

    # Generate posterior samples first
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Test marginal log likelihood (across components)
    ll = nbdm_mix_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))

    # Test component-specific log likelihood
    ll_components = nbdm_mix_results.log_likelihood(
        counts, return_by="cell", split_components=True
    )
    assert ll_components.shape[-1] == nbdm_mix_results.n_components
    assert jnp.all(jnp.isfinite(ll_components))


# ------------------------------------------------------------------------------


def test_parameter_relationships(nbdm_mix_results, parameterization):
    """Test that parameter relationships are correctly maintained."""
    # Get parameters
    params = nbdm_mix_results.params

    if parameterization == "linked":
        # In linked parameterization, r should be computed as r = mu * p / (1 - p)
        if "p" in params and "mu" in params and "r" in params:
            p, mu, r = params["p"], params["mu"], params["r"]
            # p is component-specific, mu is component and gene-specific
            expected_r = mu * p[..., None] / (1 - p[..., None])
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # In odds_ratio parameterization:
        # p = 1 / (1 + phi)
        # r = mu * phi
        if (
            "phi" in params
            and "mu" in params
            and "p" in params
            and "r" in params
        ):
            phi, mu, p, r = (
                params["phi"],
                params["mu"],
                params["p"],
                params["r"],
            )
            expected_p = 1.0 / (1.0 + phi)
            # phi is component-specific, mu is component and gene-specific
            expected_r = mu * phi[..., None]
            assert jnp.allclose(p, expected_p, rtol=1e-5)
            assert jnp.allclose(r, expected_r, rtol=1e-5)


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(nbdm_mix_results, inference_method):
    if inference_method == "svi":
        assert hasattr(nbdm_mix_results, "loss_history")
        assert len(nbdm_mix_results.loss_history) > 0


# ------------------------------------------------------------------------------
# Mixture-specific Tests
# ------------------------------------------------------------------------------


def test_component_selection(nbdm_mix_results):
    """Test selecting a specific component from the mixture."""
    # Select the first component
    component = nbdm_mix_results.get_component(0)

    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert component.model_type == "nbdm"  # Model type should lose _mix suffix

    # Check that gene counts are preserved
    assert component.n_genes == nbdm_mix_results.n_genes
    assert component.n_cells == nbdm_mix_results.n_cells


# ------------------------------------------------------------------------------


def test_component_selection_with_posterior_samples(nbdm_mix_results, rng_key):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Select component
    component = nbdm_mix_results.get_component(1)

    # Check posterior samples are correctly subset
    assert component.posterior_samples is not None
    assert "p" in component.posterior_samples
    assert "r" in component.posterior_samples

    # Check shapes - component dimension should be removed
    assert component.posterior_samples["p"].shape == (3,)  # n_samples
    assert component.posterior_samples["r"].shape == (
        3,
        nbdm_mix_results.n_genes,
    )


# ------------------------------------------------------------------------------


def test_cell_type_assignments(nbdm_mix_results, small_dataset, rng_key):
    """Test computing cell type assignments."""
    counts, _ = small_dataset

    # Generate posterior samples
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Compute assignments
    assignments = nbdm_mix_results.cell_type_probabilities(
        counts, fit_distribution=True, verbose=False
    )

    # Check structure of results
    assert "concentration" in assignments
    assert "mean_probabilities" in assignments
    assert "sample_probabilities" in assignments

    # Check shapes
    assert assignments["concentration"].shape == (
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )
    assert assignments["mean_probabilities"].shape == (
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )
    assert assignments["sample_probabilities"].shape == (
        3,
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_cell_type_assignments_map(nbdm_mix_results, small_dataset):
    """Test computing cell type assignments using MAP estimates."""
    counts, _ = small_dataset

    # Compute assignments using MAP
    assignments = nbdm_mix_results.cell_type_probabilities_map(
        counts, verbose=False
    )

    # Check structure of results
    assert "probabilities" in assignments

    # Check shapes
    assert assignments["probabilities"].shape == (
        nbdm_mix_results.n_cells,
        nbdm_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_mixture_component_entropy(nbdm_mix_results, small_dataset, rng_key):
    """Test computing mixture component entropy."""
    counts, _ = small_dataset

    # Generate posterior samples
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Compute entropy
    entropy = nbdm_mix_results.mixture_component_entropy(
        counts, return_by="cell"
    )

    # Check shape and values
    assert entropy.shape == (3, nbdm_mix_results.n_cells)  # n_samples, n_cells
    assert jnp.all(entropy >= 0)  # Entropy should be non-negative


# ------------------------------------------------------------------------------


def test_hellinger_distance(nbdm_mix_results):
    """Test computing Hellinger distances between components."""
    distances = nbdm_mix_results.hellinger_distance()

    # Check that we have distances for each pair of components
    expected_pairs = ["0_1"]  # For 2 components
    assert all(pair in distances for pair in expected_pairs)

    # Check that distances are between 0 and 1
    for distance in distances.values():
        assert jnp.all(distance >= 0) and jnp.all(distance <= 1)


# ------------------------------------------------------------------------------


def test_kl_divergence(nbdm_mix_results):
    """Test computing KL divergences between components."""
    divergences = nbdm_mix_results.kl_divergence()

    # Check that we have divergences for each pair of components
    expected_pairs = ["0_1", "1_0"]  # For 2 components, both directions
    assert all(pair in divergences for pair in expected_pairs)

    # Check that divergences are non-negative
    for divergence in divergences.values():
        assert jnp.all(divergence >= 0)


# ------------------------------------------------------------------------------


def test_jensen_shannon_divergence(nbdm_mix_results):
    """Test computing Jensen-Shannon divergences between components."""
    divergences = nbdm_mix_results.jensen_shannon_divergence()

    # Check that we have divergences for each pair of components
    expected_pairs = ["0_1"]  # For 2 components, symmetric
    assert all(pair in divergences for pair in expected_pairs)

    # Check that divergences are between 0 and 1
    for divergence in divergences.values():
        assert jnp.all(divergence >= 0) and jnp.all(divergence <= 1)


# ------------------------------------------------------------------------------


def test_subset_with_posterior_samples(nbdm_mix_results, rng_key):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    nbdm_mix_results.get_posterior_samples(
        rng_key=rng_key, n_samples=3, store_samples=True
    )

    # Subset results by genes
    subset = nbdm_mix_results[0:2]

    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None
    assert "p" in subset.posterior_samples
    assert "r" in subset.posterior_samples
    assert "mixing_weights" in subset.posterior_samples
    assert subset.posterior_samples["r"].shape == (
        3,
        nbdm_mix_results.n_components,
        2,
    )
    assert subset.posterior_samples["p"].shape == (3,)
    assert subset.posterior_samples["mixing_weights"].shape == (
        3,
        nbdm_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_component_then_gene_indexing(nbdm_mix_results):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = nbdm_mix_results.get_component(0)

    # Then select genes
    gene_subset = component[0:2]

    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == nbdm_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "nbdm"
