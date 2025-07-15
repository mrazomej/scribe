# tests/test_nbvcp_mix.py
"""
Tests for the Negative Binomial Mixture Model with Variable Capture Probability.
"""
import pytest
import jax.numpy as jnp
from jax import random
import os

ALL_METHODS = ["svi", "mcmc"]
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
    if {"nbvcp_mix_results", "inference_method", "parameterization"}.issubset(
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
        combinations = [
            (m, p)
            for m in methods
            for p in params
        ]

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
_nbvcp_mix_results_cache = {}


@pytest.fixture(scope="function")
def nbvcp_mix_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _nbvcp_mix_results_cache:
        return _nbvcp_mix_results_cache[key]

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
        priors = {
            "r_prior": (2, 0.1),
            "p_prior": (1, 1),
            "p_capture_prior": (1, 1),
        }
    elif parameterization == "linked":
        priors = {
            "p_prior": (1, 1),
            "mu_prior": (0, 1),
            "p_capture_prior": (1, 1),
        }
    elif parameterization == "odds_ratio":
        priors = {
            "phi_prior": (3, 2),
            "mu_prior": (0, 1),
            "phi_capture_prior": (3, 2),
        }
    elif parameterization == "unconstrained":
        priors = {}  # Unconstrained uses defaults
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    from scribe import run_scribe

    if inference_method == "svi":
        result = run_scribe(
            counts=counts,
            inference_method="svi",
            zero_inflated=False,
            variable_capture=True,
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
            p_capture_prior=priors.get("p_capture_prior"),
            phi_capture_prior=priors.get("phi_capture_prior"),
            mixing_prior=jnp.ones(2),  # Uniform prior for 2 components
        )
    else:
        result = run_scribe(
            counts=counts,
            inference_method="mcmc",
            zero_inflated=False,
            variable_capture=True,
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
            p_capture_prior=priors.get("p_capture_prior"),
            phi_capture_prior=priors.get("phi_capture_prior"),
            mixing_prior=jnp.ones(2),  # Uniform prior for 2 components
        )

    _nbvcp_mix_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(nbvcp_mix_results):
    """Test that inference runs and produces expected results."""
    assert nbvcp_mix_results.n_cells > 0
    assert nbvcp_mix_results.n_genes > 0
    assert nbvcp_mix_results.model_type == "nbvcp_mix"
    assert hasattr(nbvcp_mix_results, "model_config")
    assert nbvcp_mix_results.n_components == 2


def test_parameterization_config(nbvcp_mix_results, parameterization):
    """Test that the correct parameterization is used."""
    assert nbvcp_mix_results.model_config.parameterization == parameterization


def test_parameter_ranges(nbvcp_mix_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(nbvcp_mix_results, "params") and hasattr(
        nbvcp_mix_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = nbvcp_mix_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(nbvcp_mix_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = nbvcp_mix_results.params
    else:
        # Fallback: try to get samples
        samples = nbvcp_mix_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p, r, and p_capture (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "r" in params or any(k.startswith("r_") for k in params.keys())
        assert "p_capture" in params or any(k.startswith("p_capture_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "linked":
        # Linked parameterization: p, mu, and p_capture (component-specific)
        assert "p" in params or any(k.startswith("p_") for k in params.keys())
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())
        assert "p_capture" in params or any(k.startswith("p_capture_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi, mu, and phi_capture (component-specific)
        assert "phi" in params or any(
            k.startswith("phi_") for k in params.keys()
        )
        assert "mu" in params or any(k.startswith("mu_") for k in params.keys())
        assert "phi_capture" in params or any(k.startswith("phi_capture_") for k in params.keys())

        # Check mixing weights
        assert "mixing" in params or any(
            k.startswith("mixing_") for k in params.keys()
        )

    elif parameterization == "unconstrained":
        # Unconstrained parameterization: unconstrained parameters
        assert any(k.startswith("p_unconstrained") for k in params.keys())
        assert any(k.startswith("r_unconstrained") for k in params.keys())
        assert any(k.startswith("p_capture_unconstrained") for k in params.keys())
        assert any(
            k.startswith("mixing_logits_unconstrained") for k in params.keys()
        )


def test_posterior_sampling(nbvcp_mix_results, rng_key):
    """Test sampling from the variational posterior."""
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just call
    # get_posterior_samples without parameters
    if hasattr(nbvcp_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
            samples = nbvcp_mix_results.get_posterior_samples()
        else:  # SVI case
            samples = nbvcp_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = nbvcp_mix_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    parameterization = nbvcp_mix_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples and "p_capture" in samples
        assert (
            samples["r"].shape[-2] == nbvcp_mix_results.n_components
        )  # Component dimension
        assert samples["r"].shape[-1] == nbvcp_mix_results.n_genes
        assert samples["p_capture"].shape[-1] == nbvcp_mix_results.n_cells
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples and "p_capture" in samples
        assert samples["mu"].shape[-2] == nbvcp_mix_results.n_components
        assert samples["mu"].shape[-1] == nbvcp_mix_results.n_genes
        assert samples["p_capture"].shape[-1] == nbvcp_mix_results.n_cells
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples and "phi_capture" in samples
        assert samples["mu"].shape[-2] == nbvcp_mix_results.n_components
        assert samples["mu"].shape[-1] == nbvcp_mix_results.n_genes
        assert samples["phi_capture"].shape[-1] == nbvcp_mix_results.n_cells
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in samples
        assert "r_unconstrained" in samples
        assert "p_capture_unconstrained" in samples
        assert samples["r_unconstrained"].shape[-2] == nbvcp_mix_results.n_components
        assert samples["r_unconstrained"].shape[-1] == nbvcp_mix_results.n_genes
        assert samples["p_capture_unconstrained"].shape[-1] == nbvcp_mix_results.n_cells

    # Check mixing weights
    assert "mixing_weights" in samples or "mixing_logits_unconstrained" in samples
    if "mixing_weights" in samples:
        assert samples["mixing_weights"].shape[-1] == nbvcp_mix_results.n_components
    elif "mixing_logits_unconstrained" in samples:
        assert samples["mixing_logits_unconstrained"].shape[-1] == nbvcp_mix_results.n_components


def test_predictive_sampling(nbvcp_mix_results, rng_key):
    """Test generating predictive samples."""
    # For SVI, must generate posterior samples first
    if hasattr(nbvcp_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = nbvcp_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbvcp_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbvcp_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbvcp_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )
    assert pred.shape[-1] == nbvcp_mix_results.n_genes
    assert jnp.all(pred >= 0)


def test_get_map(nbvcp_mix_results):
    """Test getting MAP estimates."""
    map_est = (
        nbvcp_mix_results.get_map()
        if hasattr(nbvcp_mix_results, "get_map")
        else None
    )
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = nbvcp_mix_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est and "p_capture" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est and "p_capture" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est and "phi_capture" in map_est

    # Check mixing weights
    assert "mixing_weights" in map_est


def test_indexing_integer(nbvcp_mix_results):
    """Test indexing with an integer."""
    subset = nbvcp_mix_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == nbvcp_mix_results.n_cells
    assert subset.n_components == nbvcp_mix_results.n_components


def test_indexing_slice(nbvcp_mix_results):
    """Test indexing with a slice."""
    end_idx = min(2, nbvcp_mix_results.n_genes)
    subset = nbvcp_mix_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == nbvcp_mix_results.n_cells
    assert subset.n_components == nbvcp_mix_results.n_components


def test_indexing_boolean(nbvcp_mix_results):
    """Test indexing with a boolean array."""
    mask = jnp.zeros(nbvcp_mix_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    subset = nbvcp_mix_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == nbvcp_mix_results.n_cells
    assert subset.n_components == nbvcp_mix_results.n_components


def test_log_likelihood(nbvcp_mix_results, small_dataset, rng_key):
    """Test computing log likelihood."""
    counts, _ = small_dataset

    # For SVI, must generate posterior samples first
    if hasattr(nbvcp_mix_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            nbvcp_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )

    # Test marginal log likelihood (across components)
    ll = nbvcp_mix_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))

    # Test component-specific log likelihood
    ll_components = nbvcp_mix_results.log_likelihood(
        counts, return_by="cell", split_components=True
    )
    assert ll_components.shape[-1] == nbvcp_mix_results.n_components
    assert jnp.all(jnp.isfinite(ll_components))


def test_parameter_relationships(nbvcp_mix_results, parameterization):
    """Test that parameter relationships are correctly maintained."""
    # Get parameters
    params = None
    if hasattr(nbvcp_mix_results, "params"):
        params = nbvcp_mix_results.params
    if params is None:
        samples = nbvcp_mix_results.get_posterior_samples()
        params = samples

    if parameterization == "linked":
        # In linked parameterization, r should be computed as
        # r = mu * (1 - p) / p
        if "p" in params and "mu" in params and "r" in params:
            p, mu, r = params["p"], params["mu"], params["r"]
            # Handle different shapes for SVI vs MCMC
            if p.ndim == 1 and mu.ndim == 2:
                # SVI case: p is (n_samples,), mu is (n_samples, n_components, n_genes)
                expected_r = mu * (1 - p[:, None, None]) / p[:, None, None]
            elif p.ndim == 0 and mu.ndim == 2:
                # MCMC case: p is scalar, mu is (n_components, n_genes)
                expected_r = mu * (1 - p) / p
            else:
                # Skip test if shapes are unexpected
                return
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # In odds_ratio parameterization:
        # p = 1 / (1 + phi)
        # r = mu * phi
        # p_capture = 1 / (1 + phi_capture)
        if (
            "phi" in params
            and "mu" in params
            and "p" in params
            and "r" in params
            and "phi_capture" in params
            and "p_capture" in params
        ):
            phi, mu, p, r, phi_capture, p_capture = (
                params["phi"],
                params["mu"],
                params["p"],
                params["r"],
                params["phi_capture"],
                params["p_capture"],
            )
            expected_p = 1.0 / (1.0 + phi)

            # Handle different shapes for SVI vs MCMC
            if phi.ndim == 1 and mu.ndim == 2:
                # SVI case: phi is (n_samples,), mu is (n_samples, n_components, n_genes)
                expected_r = mu * phi[:, None, None]
            elif phi.ndim == 0 and mu.ndim == 2:
                # MCMC case: phi is scalar, mu is (n_components, n_genes)
                expected_r = mu * phi
            else:
                # Skip r test if shapes are unexpected
                expected_r = None

            # phi_capture is cell-specific per sample
            expected_p_capture = 1.0 / (1.0 + phi_capture)

            assert jnp.allclose(p, expected_p, rtol=1e-5)
            if expected_r is not None:
                assert jnp.allclose(r, expected_r, rtol=1e-5)
            # Make shapes compatible for broadcasting
            pc1 = jnp.squeeze(p_capture)
            pc2 = jnp.squeeze(expected_p_capture)
            assert (
                pc1.shape == pc2.shape
            ), f"Shape mismatch after squeeze: {pc1.shape} vs {pc2.shape}"
            assert jnp.allclose(pc1, pc2, rtol=1e-5)


# ------------------------------------------------------------------------------
# NBVCP-specific Tests
# ------------------------------------------------------------------------------


def test_cell_specific_parameters(nbvcp_mix_results, parameterization):
    """Test that cell-specific parameters have correct shapes and ranges."""
    # Get parameters
    params = None
    if hasattr(nbvcp_mix_results, "params"):
        params = nbvcp_mix_results.params
    if params is None:
        samples = nbvcp_mix_results.get_posterior_samples()
        params = samples

    # Check cell-specific parameters regardless of parameterization
    if parameterization == "odds_ratio":
        if "phi_capture" in params:
            phi_capture = params["phi_capture"]
            assert (
                phi_capture.shape[-1] == nbvcp_mix_results.n_cells
            ), "phi_capture should be cell-specific"
            assert jnp.all(phi_capture > 0), "phi_capture should be positive"
    else:
        if "p_capture" in params:
            p_capture = params["p_capture"]
            assert (
                p_capture.shape[-1] == nbvcp_mix_results.n_cells
            ), "p_capture should be cell-specific"
            assert jnp.all(
                (p_capture >= 0) & (p_capture <= 1)
            ), "p_capture should be in [0, 1]"


def test_variable_capture_behavior(nbvcp_mix_results, rng_key):
    """Test that the model exhibits variable capture behavior."""
    # Generate predictive samples
    if hasattr(nbvcp_mix_results, "get_posterior_samples"):
        if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
            pred = nbvcp_mix_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbvcp_mix_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbvcp_mix_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbvcp_mix_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )

    # Check that we have non-negative counts
    assert jnp.all(pred >= 0), "NBVCP model should produce non-negative counts"

    # Check that we have some non-zero counts
    assert jnp.any(pred > 0), "NBVCP model should produce some non-zero counts"


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(nbvcp_mix_results, inference_method):
    if inference_method == "svi":
        assert hasattr(nbvcp_mix_results, "loss_history")
        assert len(nbvcp_mix_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(nbvcp_mix_results, inference_method):
    if inference_method == "mcmc":
        samples = nbvcp_mix_results.get_posterior_samples()

        # Check parameters based on parameterization
        parameterization = nbvcp_mix_results.model_config.parameterization
        if parameterization == "standard":
            assert "r" in samples and "p" in samples and "p_capture" in samples
            assert samples["r"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["p_capture"].ndim >= 2  # n_samples, n_cells
        elif parameterization == "linked":
            assert "p" in samples and "mu" in samples and "p_capture" in samples
            assert samples["mu"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["p_capture"].ndim >= 2  # n_samples, n_cells
        elif parameterization == "odds_ratio":
            assert (
                "phi" in samples
                and "mu" in samples
                and "phi_capture" in samples
            )
            assert samples["mu"].ndim >= 3  # n_samples, n_components, n_genes
            assert samples["phi_capture"].ndim >= 2  # n_samples, n_cells

        # Check mixing weights
        assert "mixing_weights" in samples
        assert samples["mixing_weights"].ndim >= 2  # n_samples, n_components


# ------------------------------------------------------------------------------
# Mixture-specific Tests
# ------------------------------------------------------------------------------


def test_component_selection(nbvcp_mix_results):
    """Test selecting a specific component from the mixture."""
    # Select the first component
    component = nbvcp_mix_results.get_component(0)

    # Check that it's no longer a mixture model
    assert component.n_components is None
    assert "_mix" not in component.model_type
    
    # Check that gene and cell counts are preserved
    assert component.n_genes == nbvcp_mix_results.n_genes
    assert component.n_cells == nbvcp_mix_results.n_cells


def test_component_selection_with_posterior_samples(nbvcp_mix_results, rng_key):
    """Test component selection with posterior samples."""
    # Generate posterior samples
    if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbvcp_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True,
        )

    # Select component
    component = nbvcp_mix_results.get_component(1)

    # Check posterior samples are correctly subset
    assert component.posterior_samples is not None
    
    # Check parameters based on parameterization
    parameterization = nbvcp_mix_results.model_config.parameterization
    
    if parameterization == "standard":
        assert "p" in component.posterior_samples
        assert "r" in component.posterior_samples
        assert "p_capture" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p"].shape == (3,)  # n_samples
        assert component.posterior_samples["r"].shape == (
            3,
            nbvcp_mix_results.n_genes,
        )
        assert component.posterior_samples["p_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
    elif parameterization == "linked":
        assert "p" in component.posterior_samples
        assert "mu" in component.posterior_samples
        assert "p_capture" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p"].shape == (3,)  # n_samples
        assert component.posterior_samples["mu"].shape == (
            3,
            nbvcp_mix_results.n_genes,
        )
        assert component.posterior_samples["p_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
    elif parameterization == "odds_ratio":
        assert "phi" in component.posterior_samples
        assert "mu" in component.posterior_samples
        assert "phi_capture" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["phi"].shape == (3,)  # n_samples
        assert component.posterior_samples["mu"].shape == (
            3,
            nbvcp_mix_results.n_genes,
        )
        assert component.posterior_samples["phi_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in component.posterior_samples
        assert "r_unconstrained" in component.posterior_samples
        assert "p_capture_unconstrained" in component.posterior_samples
        # Check shapes - component dimension should be removed
        assert component.posterior_samples["p_unconstrained"].shape == (3,)  # n_samples
        assert component.posterior_samples["r_unconstrained"].shape == (
            3,
            nbvcp_mix_results.n_genes,
        )
        assert component.posterior_samples["p_capture_unconstrained"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )


def test_cell_type_assignments(nbvcp_mix_results, small_dataset, rng_key):
    """Test computing cell type assignments."""
    counts, _ = small_dataset

    # Generate posterior samples
    if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbvcp_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True
        )

    # Compute assignments
    assignments = nbvcp_mix_results.cell_type_probabilities(
        counts, fit_distribution=True, verbose=False
    )

    # Check structure of results
    assert "concentration" in assignments
    assert "mean_probabilities" in assignments
    assert "sample_probabilities" in assignments

    # Check shapes
    assert assignments["concentration"].shape == (
        nbvcp_mix_results.n_cells,
        nbvcp_mix_results.n_components,
    )
    assert assignments["mean_probabilities"].shape == (
        nbvcp_mix_results.n_cells,
        nbvcp_mix_results.n_components,
    )
    assert assignments["sample_probabilities"].shape == (
        3,
        nbvcp_mix_results.n_cells,
        nbvcp_mix_results.n_components,
    )


def test_subset_with_posterior_samples(nbvcp_mix_results, rng_key):
    """Test that subsetting preserves posterior samples."""
    # Generate posterior samples
    if hasattr(nbvcp_mix_results, "get_samples"):  # MCMC case
        # MCMC already has samples
        pass
    else:  # SVI case
        nbvcp_mix_results.get_posterior_samples(
            rng_key=rng_key, n_samples=3, store_samples=True,
        )

    # Subset results by genes
    subset = nbvcp_mix_results[0:2]

    # Check posterior samples were preserved and correctly subset
    assert subset.posterior_samples is not None
    
    # Check parameters based on parameterization
    parameterization = nbvcp_mix_results.model_config.parameterization
    
    if parameterization == "standard":
        assert "p" in subset.posterior_samples
        assert "r" in subset.posterior_samples
        assert "p_capture" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["r"].shape == (
            3,
            nbvcp_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p"].shape == (3,)
        assert subset.posterior_samples["p_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            nbvcp_mix_results.n_components,
        )
    elif parameterization == "linked":
        assert "p" in subset.posterior_samples
        assert "mu" in subset.posterior_samples
        assert "p_capture" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["mu"].shape == (
            3,
            nbvcp_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p"].shape == (3,)
        assert subset.posterior_samples["p_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            nbvcp_mix_results.n_components,
        )
    elif parameterization == "odds_ratio":
        assert "phi" in subset.posterior_samples
        assert "mu" in subset.posterior_samples
        assert "phi_capture" in subset.posterior_samples
        assert "mixing_weights" in subset.posterior_samples
        assert subset.posterior_samples["mu"].shape == (
            3,
            nbvcp_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["phi"].shape == (3,)
        assert subset.posterior_samples["phi_capture"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
        assert subset.posterior_samples["mixing_weights"].shape == (
            3,
            nbvcp_mix_results.n_components,
        )
    elif parameterization == "unconstrained":
        assert "p_unconstrained" in subset.posterior_samples
        assert "r_unconstrained" in subset.posterior_samples
        assert "p_capture_unconstrained" in subset.posterior_samples
        assert "mixing_logits_unconstrained" in subset.posterior_samples
        assert subset.posterior_samples["r_unconstrained"].shape == (
            3,
            nbvcp_mix_results.n_components,
            2,
        )
        assert subset.posterior_samples["p_unconstrained"].shape == (3,)
        assert subset.posterior_samples["p_capture_unconstrained"].shape == (
            3,
            nbvcp_mix_results.n_cells,
        )
        assert subset.posterior_samples["mixing_logits_unconstrained"].shape == (
            3,
            nbvcp_mix_results.n_components,
        )


def test_component_then_gene_indexing(nbvcp_mix_results):
    """Test selecting a component and then indexing genes."""
    # First select a component
    component = nbvcp_mix_results.get_component(0)

    # Then select genes
    gene_subset = component[0:2]

    # Check dimensions
    assert gene_subset.n_genes == 2
    assert gene_subset.n_cells == nbvcp_mix_results.n_cells
    assert gene_subset.n_components is None
    assert gene_subset.model_type == "nbvcp"