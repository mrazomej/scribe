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

    Excludes incompatible combinations (SVI + unconstrained) and handles
    command-line options for selective testing.
    """
    # Check if this test function uses the required fixtures
    if {"nbdm_results", "inference_method", "parameterization"}.issubset(
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
        combinations = [
            (m, p)
            for m in methods
            for p in params
            if not (m == "svi" and p == "unconstrained")
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
_nbdm_results_cache = {}

@pytest.fixture(scope="function")
def nbdm_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _nbdm_results_cache:
        return _nbdm_results_cache[key]
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

    if inference_method == "svi":
        result = run_scribe(
            counts=counts,
            inference_method="svi",
            zero_inflated=False,
            variable_capture=False,
            mixture_model=False,
            parameterization=parameterization,
            n_steps=3,
            batch_size=5,
            seed=42,
            r_prior=priors.get("r_prior"),
            p_prior=priors.get("p_prior"),
            mu_prior=priors.get("mu_prior"),
            phi_prior=priors.get("phi_prior"),
        )
    else:
        result = run_scribe(
            counts=counts,
            inference_method="mcmc",
            zero_inflated=False,
            variable_capture=False,
            mixture_model=False,
            parameterization=parameterization,
            n_warmup=2,
            n_samples=3,
            n_chains=1,
            seed=42,
            r_prior=priors.get("r_prior"),
            p_prior=priors.get("p_prior"),
            mu_prior=priors.get("mu_prior"),
            phi_prior=priors.get("phi_prior"),
        )
    _nbdm_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests 
# ------------------------------------------------------------------------------

def test_inference_run(nbdm_results):
    assert nbdm_results.n_cells > 0
    assert nbdm_results.n_genes > 0
    assert nbdm_results.model_type == "nbdm"
    assert hasattr(nbdm_results, "model_config")

# ------------------------------------------------------------------------------

def test_parameterization_config(nbdm_results, parameterization):
    """Test that the correct parameterization is used."""
    assert nbdm_results.model_config.parameterization == parameterization

# ------------------------------------------------------------------------------

def test_parameter_ranges(nbdm_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(nbdm_results, "params") and hasattr(
        nbdm_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = nbdm_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(nbdm_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = nbdm_results.params
    else:
        # Fallback: try to get samples
        samples = nbdm_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p and r
        assert "p" in params
        assert "r" in params
        p, r = params["p"], params["r"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(r > 0)  # r is positive dispersion

    elif parameterization == "linked":
        # Linked parameterization: p and mu
        assert "p" in params
        assert "mu" in params
        p, mu = params["p"], params["mu"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(mu > 0)  # mu is positive mean

        # Check that r is computed correctly: r = mu * (1 - p) / p
        if "r" in params:
            r = params["r"]
            # p is scalar per sample, mu is gene-specific per sample
            # Need to broadcast p to match mu's gene dimension
            expected_r = mu * (1 - p[..., None]) / p[..., None]
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi and mu
        assert "phi" in params
        assert "mu" in params
        phi, mu = params["phi"], params["mu"]
        assert jnp.all(phi > 0)  # phi is positive
        assert jnp.all(mu > 0)  # mu is positive

        # Check that p and r are computed correctly
        if (
            "phi" in params
            and "mu" in params
            and "p" in params
            and "r" in params
        ):
            p, r = (
                params["p"],
                params["r"],
            )
            expected_p = 1.0 / (1.0 + phi)
            # phi is scalar per sample, mu is gene-specific per sample
            # Need to broadcast phi to match mu's gene dimension
            expected_r = mu * phi[..., None]
            assert jnp.allclose(p, expected_p, rtol=1e-5)
            assert jnp.allclose(r, expected_r, rtol=1e-5)

# ------------------------------------------------------------------------------

def test_posterior_sampling(nbdm_results, rng_key):
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just call
    # get_posterior_samples without parameters
    if hasattr(nbdm_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_results, "get_samples"):  # MCMC case
            samples = nbdm_results.get_posterior_samples()
        else:  # SVI case
            samples = nbdm_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = nbdm_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    parameterization = nbdm_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples
        assert samples["r"].shape[-1] == nbdm_results.n_genes
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples
        assert samples["mu"].shape[-1] == nbdm_results.n_genes
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples
        assert samples["mu"].shape[-1] == nbdm_results.n_genes

# ------------------------------------------------------------------------------

def test_predictive_sampling(nbdm_results, rng_key):
    # For SVI, must generate posterior samples first
    if hasattr(nbdm_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = nbdm_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbdm_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbdm_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbdm_results.get_ppc_samples(rng_key=rng_key, store_samples=True)
    assert pred.shape[-1] == nbdm_results.n_genes
    assert jnp.all(pred >= 0)

# ------------------------------------------------------------------------------

def test_get_map(nbdm_results):
    map_est = (
        nbdm_results.get_map() if hasattr(nbdm_results, "get_map") else None
    )
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = nbdm_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est

# ------------------------------------------------------------------------------

def test_indexing_integer(nbdm_results):
    """Test indexing with an integer."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbdm_results.n_genes = {nbdm_results.n_genes}")

    subset = nbdm_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_results.n_cells

# ------------------------------------------------------------------------------

def test_indexing_slice(nbdm_results):
    """Test indexing with a slice."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbdm_results.n_genes = {nbdm_results.n_genes}")

    # Use a slice that works regardless of the number of genes
    end_idx = min(2, nbdm_results.n_genes)
    subset = nbdm_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == nbdm_results.n_cells

# ------------------------------------------------------------------------------

def test_indexing_boolean(nbdm_results):
    """Test indexing with a boolean array."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbdm_results.n_genes = {nbdm_results.n_genes}")

    # Create a boolean mask that selects only the first gene
    mask = jnp.zeros(nbdm_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    # Verify the mask has exactly one True value
    assert jnp.sum(mask) == 1

    subset = nbdm_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == nbdm_results.n_cells

# ------------------------------------------------------------------------------

def test_log_likelihood(nbdm_results, small_dataset, rng_key):
    counts, _ = small_dataset
    # For SVI, must generate posterior samples first
    if hasattr(nbdm_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbdm_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            nbdm_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    ll = nbdm_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))

# ------------------------------------------------------------------------------

def test_parameter_relationships(nbdm_results, parameterization):
    """Test that parameter relationships are correctly maintained."""
    # Get parameters
    params = None
    if hasattr(nbdm_results, "params"):
        params = nbdm_results.params
    if params is None:
        samples = nbdm_results.get_posterior_samples()
        params = samples

    if parameterization == "linked":
        # In linked parameterization, r should be computed as r = mu * (1 - p) / p
        if "p" in params and "mu" in params and "r" in params:
            p, mu, r = params["p"], params["mu"], params["r"]
            # p is scalar per sample, mu is gene-specific per sample
            # Need to broadcast p to match mu's gene dimension
            expected_r = mu * (1 - p[..., None]) / p[..., None]
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # In odds_ratio parameterization:
        # p = 1 / (1 + phi)
        # r = mu * phi
        if (
            "phi" in params
            and "mu" in params
            and "r" in params
        ):
            phi, mu, p, r = (
                params["phi"],
                params["mu"],
                params["p"],
                params["r"],
            )
            expected_p = 1.0 / (1.0 + phi)
            # phi is scalar per sample, mu is gene-specific per sample
            # Need to broadcast phi to match mu's gene dimension
            expected_r = mu * phi[..., None]
            assert jnp.allclose(p, expected_p, rtol=1e-5)
            assert jnp.allclose(r, expected_r, rtol=1e-5)


# ------------------------------------------------------------------------------
# SVI-specific Tests 
# ------------------------------------------------------------------------------

def test_svi_loss_history(nbdm_results, inference_method):
    if inference_method == "svi":
        assert hasattr(nbdm_results, "loss_history")
        assert len(nbdm_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests 
# ------------------------------------------------------------------------------

def test_mcmc_samples_shape(nbdm_results, inference_method):
    if inference_method == "mcmc":
        samples = nbdm_results.get_posterior_samples()

        # Check parameters based on parameterization
        parameterization = nbdm_results.model_config.parameterization
        if parameterization == "standard":
            assert "r" in samples and "p" in samples
            assert samples["r"].ndim >= 2
        elif parameterization == "linked":
            assert "p" in samples and "mu" in samples
            assert samples["mu"].ndim >= 2
        elif parameterization == "odds_ratio":
            assert "phi" in samples and "mu" in samples
            assert samples["mu"].ndim >= 2
