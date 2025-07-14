# tests/test_zinb_mix.py
"""
Tests for the Zero-Inflated Negative Binomial Mixture Model.
"""
import pytest
import jax.numpy as jnp
from jax import random
import os

ALL_METHODS = [
    "svi",
    "mcmc",
]  # Both SVI and MCMC are supported for mixture models
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
    if {"zinb_mix_results", "inference_method", "parameterization"}.issubset(
        metafunc.fixturenames
    ):
        # Get command-line options for method and parameterization
        method_opt = metafunc.config.getoption("--method")
        param_opt = metafunc.config.getoption("--parameterization")

        # Determine which methods to test based on command-line option
        methods = ALL_METHODS if method_opt == "all" else [method_opt]

        # Determine which parameterizations to test based on command-line option
        params = ALL_PARAMETERIZATIONS if param_opt == "all" else [param_opt]

        # Generate all valid combinations, excluding SVI+unconstrained
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
_zinb_mix_results_cache = {}


@pytest.fixture(scope="function")
def zinb_mix_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _zinb_mix_results_cache:
        return _zinb_mix_results_cache[key]

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
        priors = {"r_prior": (2, 0.1), "p_prior": (1, 1), "gate_prior": (1, 1)}
    elif parameterization == "linked":
        priors = {"p_prior": (1, 1), "mu_prior": (0, 1), "gate_prior": (1, 1)}
    elif parameterization == "odds_ratio":
        priors = {"phi_prior": (3, 2), "mu_prior": (0, 1), "gate_prior": (1, 1)}
    elif parameterization == "unconstrained":
        priors = {}  # Unconstrained uses defaults
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    from scribe import run_scribe

    if inference_method == "svi":
        result = run_scribe(
            counts=counts,
            inference_method="svi",
            zero_inflated=True,
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
            gate_prior=priors.get("gate_prior"),
            mixing_prior=jnp.ones(2),  # Uniform prior for 2 components
        )
    else:
        result = run_scribe(
            counts=counts,
            inference_method="mcmc",
            zero_inflated=True,
            variable_capture=False,
            mixture_model=True,
            n_components=2,  # Test with 2 components
            parameterization=parameterization,
            n_warmup=2,
            n_samples=3,
            n_chains=1,
            seed=42,
            r_prior=priors.get("r_prior"),
            p_prior=priors.get("p_prior"),
            mu_prior=priors.get("mu_prior"),
            phi_prior=priors.get("phi_prior"),
            gate_prior=priors.get("gate_prior"),
            mixing_prior=jnp.ones(2),  # Uniform prior for 2 components
        )

    _zinb_mix_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(zinb_mix_results):
    assert zinb_mix_results.n_cells > 0
    assert zinb_mix_results.n_genes > 0
    assert zinb_mix_results.model_type == "zinb_mix"
    assert hasattr(zinb_mix_results, "model_config")
    assert zinb_mix_results.n_components == 2


# ------------------------------------------------------------------------------


def test_parameterization_config(zinb_mix_results, parameterization):
    """Test that the correct parameterization is used."""
    assert zinb_mix_results.model_config.parameterization == parameterization


# ------------------------------------------------------------------------------


def test_parameter_ranges(zinb_mix_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(zinb_mix_results, "params") and hasattr(
        zinb_mix_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = zinb_mix_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(zinb_mix_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = zinb_mix_results.params
    else:
        # Fallback: try to get samples
        samples = zinb_mix_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p, r, and gate (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "r" in params or any(k.startswith("r_") for k in params.keys())
        assert "gate" in params or any(
            k.startswith("gate_") for k in params.keys()
        )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "linked":
        # Linked parameterization: p, mu, and gate (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())
        assert "gate" in params or any(
            k.startswith("gate_") for k in params.keys()
        )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi, mu, and gate (component-specific)
        assert "phi" in params or any(
            k.startswith("phi_") for k in params.keys()
        )
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())
        assert "gate" in params or any(
            k.startswith("gate_") for k in params.keys()
        )

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "unconstrained":
        # Unconstrained parameterization: unconstrained parameters
        assert any(k.startswith("p_unconstrained") for k in params.keys())
        assert any(k.startswith("r_unconstrained") for k in params.keys())
        assert any(k.startswith("gate_unconstrained") for k in params.keys())
        assert any(
            k.startswith("mixing_logits_unconstrained") for k in params.keys()
        )


# ------------------------------------------------------------------------------


def test_posterior_sampling(zinb_mix_results, rng_key):
    """Test sampling from the variational posterior."""
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just call
    # get_posterior_samples without parameters
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            samples = zinb_mix_results.get_posterior_samples()
        else:  # SVI case
            samples = zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = zinb_mix_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    parameterization = zinb_mix_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples and "gate" in samples
        assert (
            samples["r"].shape[-2] == zinb_mix_results.n_components
        )  # Component dimension
        assert samples["r"].shape[-1] == zinb_mix_results.n_genes
        assert (
            samples["gate"].shape[-2] == zinb_mix_results.n_components
        )  # Component dimension
        assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples and "gate" in samples
        assert samples["mu"].shape[-2] == zinb_mix_results.n_components
        assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
        assert samples["gate"].shape[-2] == zinb_mix_results.n_components
        assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples and "gate" in samples
        assert samples["mu"].shape[-2] == zinb_mix_results.n_components
        assert samples["mu"].shape[-1] == zinb_mix_results.n_genes
        assert samples["gate"].shape[-2] == zinb_mix_results.n_components
        assert samples["gate"].shape[-1] == zinb_mix_results.n_genes
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in samples
        assert "r_unconstrained" in samples
        assert "gate_unconstrained" in samples
        assert (
            samples["r_unconstrained"].shape[-2]
            == zinb_mix_results.n_components
        )
        assert samples["r_unconstrained"].shape[-1] == zinb_mix_results.n_genes
        assert (
            samples["gate_unconstrained"].shape[-2]
            == zinb_mix_results.n_components
        )
        assert (
            samples["gate_unconstrained"].shape[-1] == zinb_mix_results.n_genes
        )

    # Check mixing weights
    assert (
        "mixing_weights" in samples or "mixing_logits_unconstrained" in samples
    )
    if "mixing_weights" in samples:
        assert (
            samples["mixing_weights"].shape[-1] == zinb_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in samples:
        assert (
            samples["mixing_logits_unconstrained"].shape[-1]
            == zinb_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_predictive_sampling(zinb_mix_results, rng_key):
    """Test generating predictive samples."""
    # For SVI, must generate posterior samples first
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = zinb_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )
    assert pred.shape[-1] == zinb_mix_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(zinb_mix_results):
    map_est = (
        zinb_mix_results.get_map()
        if hasattr(zinb_mix_results, "get_map")
        else None
    )
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = zinb_mix_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est and "gate" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est and "gate" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est and "gate" in map_est
    elif parameterization == "unconstrained":
        assert (
            "r_unconstrained" in map_est
            and "p_unconstrained" in map_est
            and "gate_unconstrained" in map_est
        )

    # Check mixing weights
    assert (
        "mixing_weights" in map_est or "mixing_logits_unconstrained" in map_est
    )
    if "mixing_weights" in map_est:
        assert (
            map_est["mixing_weights"].shape[-1] == zinb_mix_results.n_components
        )
    elif "mixing_logits_unconstrained" in map_est:
        assert (
            map_est["mixing_logits_unconstrained"].shape[-1]
            == zinb_mix_results.n_components
        )


# ------------------------------------------------------------------------------


def test_indexing_integer(zinb_mix_results):
    """Test indexing with an integer."""
    subset = zinb_mix_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_slice(zinb_mix_results):
    """Test indexing with a slice."""
    end_idx = min(2, zinb_mix_results.n_genes)
    subset = zinb_mix_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_indexing_boolean(zinb_mix_results):
    """Test indexing with a boolean array."""
    mask = jnp.zeros(zinb_mix_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    subset = zinb_mix_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_mix_results.n_cells
    assert subset.n_components == zinb_mix_results.n_components


# ------------------------------------------------------------------------------


def test_log_likelihood(zinb_mix_results, small_dataset, rng_key):
    counts, _ = small_dataset

    # For SVI, must generate posterior samples first
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True, canonical=True
            )

    # Test marginal log likelihood (across components)
    ll = zinb_mix_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))

    # Test component-specific log likelihood
    ll_components = zinb_mix_results.log_likelihood(
        counts, return_by="cell", split_components=True
    )
    assert ll_components.shape[-1] == zinb_mix_results.n_components
    assert jnp.all(jnp.isfinite(ll_components))


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(zinb_mix_results, inference_method):
    if inference_method == "svi":
        assert hasattr(zinb_mix_results, "loss_history")
        assert len(zinb_mix_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(zinb_mix_results, inference_method):
    if inference_method == "mcmc":
        samples = zinb_mix_results.get_posterior_samples()

        # Check parameters based on parameterization
        parameterization = zinb_mix_results.model_config.parameterization
        if parameterization == "standard":
            assert "r" in samples and "p" in samples and "gate" in samples
            assert samples["r"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["gate"].ndim >= 3  # n_samples, n_components, n_genes
        elif parameterization == "linked":
            assert "p" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["gate"].ndim >= 3  # n_samples, n_components, n_genes
        elif parameterization == "odds_ratio":
            assert "phi" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["gate"].ndim >= 3  # n_samples, n_components, n_genes

        # Check mixing weights
        assert "mixing_weights" in samples
        assert samples["mixing_weights"].ndim >= 2  # n_samples, n_components


# ------------------------------------------------------------------------------
# Mixture-specific Tests
# ------------------------------------------------------------------------------


def test_component_selection(zinb_mix_results):
    """Test selecting a specific component from the mixture."""
    # Select the first component
    component = zinb_mix_results.get_component(0)

    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert "_mix" not in component.model_type

    # Check that gene counts are preserved
    assert component.n_genes == zinb_mix_results.n_genes
    assert component.n_cells == zinb_mix_results.n_cells


# ------------------------------------------------------------------------------


def test_component_selection_with_posterior_samples(zinb_mix_results, rng_key):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Select component
    component = zinb_mix_results.get_component(1)

    # Check posterior samples are correctly subset
    assert component.posterior_samples is not None

    # Check parameters based on parameterization
    parameterization = zinb_mix_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in component.posterior_samples
        assert "r" in component.posterior_samples
        assert "gate" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p"].shape == (3,)  # n_samples
        assert component.posterior_samples["r"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
        assert component.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
    elif parameterization == "linked":
        assert "p" in component.posterior_samples
        assert "mu" in component.posterior_samples
        assert "gate" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p"].shape == (3,)  # n_samples
        assert component.posterior_samples["mu"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
        assert component.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
    elif parameterization == "odds_ratio":
        assert "phi" in component.posterior_samples
        assert "mu" in component.posterior_samples
        assert "gate" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["phi"].shape == (3,)  # n_samples
        assert component.posterior_samples["mu"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
        assert component.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in component.posterior_samples
        assert "r_unconstrained" in component.posterior_samples
        assert "gate_unconstrained" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p_unconstrained"].shape == (
            3,
        )  # n_samples
        assert component.posterior_samples["r_unconstrained"].shape == (
            3,
            zinb_mix_results.n_genes,
        )
        assert component.posterior_samples["gate_unconstrained"].shape == (
            3,
            zinb_mix_results.n_genes,
        )


# ------------------------------------------------------------------------------


def test_cell_type_assignments(zinb_mix_results, small_dataset, rng_key):
    """Test computing cell type assignments."""
    counts, _ = small_dataset

    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True
        )

    # Compute assignments
    assignments = zinb_mix_results.cell_type_probabilities(
        counts, fit_distribution=True, verbose=False
    )

    # Check structure of results
    assert "concentration" in assignments
    assert "mean_probabilities" in assignments
    assert "sample_probabilities" in assignments

    # Check shapes
    assert assignments["concentration"].shape == (
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )
    assert assignments["mean_probabilities"].shape == (
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )
    assert assignments["sample_probabilities"].shape == (
        3,
        zinb_mix_results.n_cells,
        zinb_mix_results.n_components,
    )


# ------------------------------------------------------------------------------


def test_subset_with_posterior_samples(zinb_mix_results, rng_key):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        zinb_mix_results.get_posterior_samples(
            rng_key=rng_key,
            n_samples=3,
            store_samples=True,
        )

    # Subset results by genes
    subset = zinb_mix_results[0:2]

    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None

    # Check parameters based on parameterization
    parameterization = zinb_mix_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in subset.posterior_samples
        assert "r" in subset.posterior_samples
        assert "gate" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["r"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p"].shape == (3,)
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            zinb_mix_results.n_components,
        )
    elif parameterization == "linked":
        assert "p" in subset.posterior_samples
        assert "mu" in subset.posterior_samples
        assert "gate" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["mu"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p"].shape == (3,)
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            zinb_mix_results.n_components,
        )
    elif parameterization == "odds_ratio":
        assert "phi" in subset.posterior_samples
        assert "mu" in subset.posterior_samples
        assert "gate" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["mu"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["gate"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["phi"].shape == (3,)
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            zinb_mix_results.n_components,
        )
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in subset.posterior_samples
        assert "r_unconstrained" in subset.posterior_samples
        assert "gate_unconstrained" in subset.posterior_samples
        assert "mixing_logits_unconstrained" in subset.posterior_samples
        assert subset.posterior_samples["r_unconstrained"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["gate_unconstrained"].shape == (
            3,
            zinb_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p_unconstrained"].shape == (3,)
        assert subset.posterior_samples[
            "mixing_logits_unconstrained"
        ].shape == (
            3,
            zinb_mix_results.n_components,
        )


# ------------------------------------------------------------------------------


def test_component_then_gene_indexing(zinb_mix_results):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = zinb_mix_results.get_component(0)

    # Then select genes
    gene_subset = component[0:2]

    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == zinb_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "zinb"


# ------------------------------------------------------------------------------
# ZINB-specific Tests
# ------------------------------------------------------------------------------


def test_gate_parameter_ranges(zinb_mix_results, parameterization):
    """Test that gate parameters are in valid probability range [0, 1]."""
    # Get parameters
    params = None
    if hasattr(zinb_mix_results, "params"):
        params = zinb_mix_results.params
    if params is None:
        samples = zinb_mix_results.get_posterior_samples()
        params = samples

    # Check gate parameters regardless of parameterization
    if "gate" in params:
        gate = params["gate"]
        assert jnp.all(
            (gate >= 0) & (gate <= 1)
        ), "Gate parameters must be in [0, 1]"

def test_zero_inflation_behavior(zinb_mix_results, rng_key):
    """Test that the model exhibits zero-inflation behavior."""
    # Generate predictive samples
    if hasattr(zinb_mix_results, "get_posterior_samples"):
        if hasattr(zinb_mix_results, "get_samples"):  # MCMC case
            pred = zinb_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )

    # Check that we have some zeros (zero-inflation)
    assert jnp.any(
        pred == 0
    ), "ZINB mixture model should produce some zero counts"

    # Check that we also have non-zero counts
    assert jnp.any(
        pred > 0
    ), "ZINB mixture model should also produce non-zero counts"
