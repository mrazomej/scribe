# tests/test_zinb.py
"""
Tests for the Zero-Inflated Negative Binomial model.
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

    Excludes incompatible combinations (SVI + unconstrained) and handles
    command-line options for selective testing.
    """
    # Check if this test function uses the required fixtures
    if {"zinb_results", "inference_method", "parameterization"}.issubset(
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
_zinb_results_cache = {}


@pytest.fixture(scope="function")
def zinb_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _zinb_results_cache:
        return _zinb_results_cache[key]
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
            mixture_model=False,
            parameterization=parameterization,
            n_steps=3,
            batch_size=5,
            seed=42,
            r_prior=priors.get("r_prior"),
            p_prior=priors.get("p_prior"),
            mu_prior=priors.get("mu_prior"),
            phi_prior=priors.get("phi_prior"),
            gate_prior=priors.get("gate_prior"),
        )
    else:
        result = run_scribe(
            counts=counts,
            inference_method="mcmc",
            zero_inflated=True,
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
            gate_prior=priors.get("gate_prior"),
        )
    _zinb_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(zinb_results):
    assert zinb_results.n_cells > 0
    assert zinb_results.n_genes > 0
    assert zinb_results.model_type == "zinb"
    assert hasattr(zinb_results, "model_config")


# ------------------------------------------------------------------------------


def test_parameterization_config(zinb_results, parameterization):
    """Test that the correct parameterization is used."""
    assert zinb_results.model_config.parameterization == parameterization


# ------------------------------------------------------------------------------


def test_parameter_ranges(zinb_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(zinb_results, "params") and hasattr(
        zinb_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = zinb_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(zinb_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = zinb_results.params
    else:
        # Fallback: try to get samples
        samples = zinb_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p, r, and gate
        assert "p" in params
        assert "r" in params
        assert "gate" in params
        p, r, gate = params["p"], params["r"], params["gate"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(r > 0)  # r is positive dispersion
        assert jnp.all((gate >= 0) & (gate <= 1))  # gate is probability

    elif parameterization == "linked":
        # Linked parameterization: p, mu, and gate
        assert "p" in params
        assert "mu" in params
        assert "gate" in params
        p, mu, gate = params["p"], params["mu"], params["gate"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(mu > 0)  # mu is positive mean
        assert jnp.all((gate >= 0) & (gate <= 1))  # gate is probability

        # Check that r is computed correctly: r = mu * p / (1 - p)
        if "r" in params:
            r = params["r"]
            # p is scalar per sample, mu is gene-specific per sample
            # Need to broadcast p to match mu's gene dimension
            expected_r = mu * p[..., None] / (1 - p[..., None])
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi, mu, and gate
        assert "phi" in params
        assert "mu" in params
        assert "gate" in params
        phi, mu, gate = params["phi"], params["mu"], params["gate"]
        assert jnp.all(phi > 0)  # phi is positive
        assert jnp.all(mu > 0)  # mu is positive
        assert jnp.all((gate >= 0) & (gate <= 1))  # gate is probability

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


def test_posterior_sampling(zinb_results, rng_key):
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just
    # call get_posterior_samples without parameters
    if hasattr(zinb_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_results, "get_samples"):  # MCMC case
            samples = zinb_results.get_posterior_samples()
        else:  # SVI case
            samples = zinb_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = zinb_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    parameterization = zinb_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples and "gate" in samples
        assert samples["r"].shape[-1] == zinb_results.n_genes
        assert samples["gate"].shape[-1] == zinb_results.n_genes
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples and "gate" in samples
        assert samples["mu"].shape[-1] == zinb_results.n_genes
        assert samples["gate"].shape[-1] == zinb_results.n_genes
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples and "gate" in samples
        assert samples["mu"].shape[-1] == zinb_results.n_genes
        assert samples["gate"].shape[-1] == zinb_results.n_genes


# ------------------------------------------------------------------------------


def test_predictive_sampling(zinb_results, rng_key):
    # For SVI, must generate posterior samples first
    if hasattr(zinb_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = zinb_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_results.get_ppc_samples(rng_key=rng_key, store_samples=True)
    assert pred.shape[-1] == zinb_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(zinb_results):
    map_est = (
        zinb_results.get_map() if hasattr(zinb_results, "get_map") else None
    )
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = zinb_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est and "gate" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est and "gate" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est and "gate" in map_est


# ------------------------------------------------------------------------------


def test_indexing_integer(zinb_results):
    """Test indexing with an integer."""
    # Debug: print the actual number of genes
    print(f"DEBUG: zinb_results.n_genes = {zinb_results.n_genes}")

    subset = zinb_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_results.n_cells


# ------------------------------------------------------------------------------


def test_indexing_slice(zinb_results):
    """Test indexing with a slice."""
    # Debug: print the actual number of genes
    print(f"DEBUG: zinb_results.n_genes = {zinb_results.n_genes}")

    # Use a slice that works regardless of the number of genes
    end_idx = min(2, zinb_results.n_genes)
    subset = zinb_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == zinb_results.n_cells


# ------------------------------------------------------------------------------


def test_indexing_boolean(zinb_results):
    """Test indexing with a boolean array."""
    # Debug: print the actual number of genes
    print(f"DEBUG: zinb_results.n_genes = {zinb_results.n_genes}")

    # Create a boolean mask that selects only the first gene
    mask = jnp.zeros(zinb_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    # Verify the mask has exactly one True value
    assert jnp.sum(mask) == 1

    subset = zinb_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == zinb_results.n_cells


# ------------------------------------------------------------------------------


def test_log_likelihood(zinb_results, small_dataset, rng_key):
    counts, _ = small_dataset
    # For SVI, must generate posterior samples first
    if hasattr(zinb_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(zinb_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            zinb_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    ll = zinb_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))


# ------------------------------------------------------------------------------


def test_parameter_relationships(zinb_results, parameterization):
    """Test that parameter relationships are correctly maintained."""
    # Get parameters
    params = None
    if hasattr(zinb_results, "params"):
        params = zinb_results.params
    if params is None:
        samples = zinb_results.get_posterior_samples()
        params = samples

    if parameterization == "linked":
        # In linked parameterization, r should be computed as r = mu * p / (1 - p)
        if "p" in params and "mu" in params and "r" in params:
            p, mu, r = params["p"], params["mu"], params["r"]
            # p is scalar per sample, mu is gene-specific per sample
            # Need to broadcast p to match mu's gene dimension
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
            # phi is scalar per sample, mu is gene-specific per sample
            # Need to broadcast phi to match mu's gene dimension
            expected_r = mu * phi[..., None]
            assert jnp.allclose(p, expected_p, rtol=1e-5)
            assert jnp.allclose(r, expected_r, rtol=1e-5)


# ------------------------------------------------------------------------------
# ZINB-specific Tests
# ------------------------------------------------------------------------------


def test_gate_parameter_ranges(zinb_results, parameterization):
    """Test that gate parameters are in valid probability range [0, 1]."""
    # Get parameters
    params = None
    if hasattr(zinb_results, "params"):
        params = zinb_results.params
    if params is None:
        samples = zinb_results.get_posterior_samples()
        params = samples

    # Check gate parameters regardless of parameterization
    if "gate" in params:
        gate = params["gate"]
        assert jnp.all(
            (gate >= 0) & (gate <= 1)
        ), "Gate parameters must be in [0, 1]"


def test_zero_inflation_behavior(zinb_results, rng_key):
    """Test that the model exhibits zero-inflation behavior."""
    # Generate predictive samples
    if hasattr(zinb_results, "get_posterior_samples"):
        if hasattr(zinb_results, "get_samples"):  # MCMC case
            pred = zinb_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            zinb_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = zinb_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = zinb_results.get_ppc_samples(rng_key=rng_key, store_samples=True)

    # Check that we have some zeros (zero-inflation)
    assert jnp.any(pred == 0), "ZINB model should produce some zero counts"

    # Check that we also have non-zero counts
    assert jnp.any(pred > 0), "ZINB model should also produce non-zero counts"


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(zinb_results, inference_method):
    if inference_method == "svi":
        assert hasattr(zinb_results, "loss_history")
        assert len(zinb_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(zinb_results, inference_method):
    if inference_method == "mcmc":
        samples = zinb_results.get_posterior_samples()

        # Check parameters based on parameterization
        parameterization = zinb_results.model_config.parameterization
        if parameterization == "standard":
            assert "r" in samples and "p" in samples and "gate" in samples
            assert samples["r"].ndim >= 2
            assert samples["gate"].ndim >= 2
        elif parameterization == "linked":
            assert "p" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].ndim >= 2
            assert samples["gate"].ndim >= 2
        elif parameterization == "odds_ratio":
            assert "phi" in samples and "mu" in samples and "gate" in samples
            assert samples["mu"].ndim >= 2
            assert samples["gate"].ndim >= 2
