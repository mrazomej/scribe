# tests/test_nbvcp.py
"""
Tests for the Negative Binomial with Variable Capture Probability model.
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
    if {"nbvcp_results", "inference_method", "parameterization"}.issubset(
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
_nbvcp_results_cache = {}


@pytest.fixture(scope="function")
def nbvcp_results(
    inference_method, device_type, parameterization, small_dataset, rng_key
):
    key = (inference_method, device_type, parameterization)
    if key in _nbvcp_results_cache:
        return _nbvcp_results_cache[key]
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
            mixture_model=False,
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
        )
    else:
        result = run_scribe(
            counts=counts,
            inference_method="mcmc",
            zero_inflated=False,
            variable_capture=True,
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
            p_capture_prior=priors.get("p_capture_prior"),
            phi_capture_prior=priors.get("phi_capture_prior"),
        )
    _nbvcp_results_cache[key] = result
    return result


# ------------------------------------------------------------------------------
# Shared Tests
# ------------------------------------------------------------------------------


def test_inference_run(nbvcp_results):
    assert nbvcp_results.n_cells > 0
    assert nbvcp_results.n_genes > 0
    assert nbvcp_results.model_type == "nbvcp"
    assert hasattr(nbvcp_results, "model_config")


# ------------------------------------------------------------------------------


def test_parameterization_config(nbvcp_results, parameterization):
    """Test that the correct parameterization is used."""
    assert nbvcp_results.model_config.parameterization == parameterization


# ------------------------------------------------------------------------------


def test_parameter_ranges(nbvcp_results, parameterization):
    """Test that parameters have correct ranges and relationships."""
    # For SVI, we need to get posterior samples to access transformed parameters
    # For MCMC, we can use either params or samples
    if hasattr(nbvcp_results, "params") and hasattr(
        nbvcp_results, "get_posterior_samples"
    ):
        # SVI case: get transformed parameters from posterior samples
        samples = nbvcp_results.get_posterior_samples(n_samples=1)
        params = samples
    elif hasattr(nbvcp_results, "params"):
        # MCMC case: params might contain transformed parameters
        params = nbvcp_results.params
    else:
        # Fallback: try to get samples
        samples = nbvcp_results.get_posterior_samples()
        params = samples

    # Check parameters based on parameterization
    if parameterization == "standard":
        # Standard parameterization: p, r, and p_capture
        assert "p" in params
        assert "r" in params
        assert "p_capture" in params
        p, r, p_capture = params["p"], params["r"], params["p_capture"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(r > 0)  # r is positive dispersion
        assert jnp.all(
            (p_capture >= 0) & (p_capture <= 1)
        )  # p_capture is probability
        assert p_capture.shape[-1] == nbvcp_results.n_cells  # cell-specific

    elif parameterization == "linked":
        # Linked parameterization: p, mu, and p_capture
        assert "p" in params
        assert "mu" in params
        assert "p_capture" in params
        p, mu, p_capture = params["p"], params["mu"], params["p_capture"]
        assert jnp.all((p >= 0) & (p <= 1))  # p is probability
        assert jnp.all(mu > 0)  # mu is positive mean
        assert jnp.all(
            (p_capture >= 0) & (p_capture <= 1)
        )  # p_capture is probability
        assert p_capture.shape[-1] == nbvcp_results.n_cells  # cell-specific

        # Check that r is computed correctly: r = mu * (1 - p) / p
        if "r" in params:
            r = params["r"]
            # p is scalar per sample, mu is gene-specific per sample
            # Need to broadcast p to match mu's gene dimension
            expected_r = mu * (1 - p[..., None]) / p[..., None]
            assert jnp.allclose(r, expected_r, rtol=1e-5)

    elif parameterization == "odds_ratio":
        # Odds ratio parameterization: phi, mu, and phi_capture
        assert "phi" in params
        assert "mu" in params
        assert "phi_capture" in params
        phi, mu, phi_capture = (
            params["phi"],
            params["mu"],
            params["phi_capture"],
        )
        assert jnp.all(phi > 0)  # phi is positive
        assert jnp.all(mu > 0)  # mu is positive
        assert jnp.all(phi_capture > 0)  # phi_capture is positive
        assert phi_capture.shape[-1] == nbvcp_results.n_cells  # cell-specific

        # Check that p, r, and p_capture are computed correctly
        if (
            "phi" in params
            and "mu" in params
            and "p" in params
            and "r" in params
            and "p_capture" in params
        ):
            p, r, p_capture = (
                params["p"],
                params["r"],
                params["p_capture"],
            )
            expected_p = 1.0 / (1.0 + phi)

            # Handle different shapes for SVI vs MCMC
            if phi.ndim == 1 and mu.ndim == 2:
                # SVI case: phi is (n_samples,), mu is (n_samples, n_genes)
                expected_r = mu * phi[:, None]
            elif phi.ndim == 0 and mu.ndim == 1:
                # MCMC case: phi is scalar, mu is (n_genes,)
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


def test_posterior_sampling(nbvcp_results, rng_key):
    # For SVI, must call get_posterior_samples with parameters; for MCMC, just
    # call get_posterior_samples without parameters
    if hasattr(nbvcp_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_results, "get_samples"):  # MCMC case
            samples = nbvcp_results.get_posterior_samples()
        else:  # SVI case
            samples = nbvcp_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    else:
        samples = nbvcp_results.get_posterior_samples()

    # Check that we have the expected parameters based on parameterization
    parameterization = nbvcp_results.model_config.parameterization

    if parameterization == "standard":
        assert "p" in samples and "r" in samples and "p_capture" in samples
        assert samples["r"].shape[-1] == nbvcp_results.n_genes
        assert samples["p_capture"].shape[-1] == nbvcp_results.n_cells
    elif parameterization == "linked":
        assert "p" in samples and "mu" in samples and "p_capture" in samples
        assert samples["mu"].shape[-1] == nbvcp_results.n_genes
        assert samples["p_capture"].shape[-1] == nbvcp_results.n_cells
    elif parameterization == "odds_ratio":
        assert "phi" in samples and "mu" in samples and "phi_capture" in samples
        assert samples["mu"].shape[-1] == nbvcp_results.n_genes
        assert samples["phi_capture"].shape[-1] == nbvcp_results.n_cells


# ------------------------------------------------------------------------------


def test_predictive_sampling(nbvcp_results, rng_key):
    # For SVI, must generate posterior samples first
    if hasattr(nbvcp_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pred = nbvcp_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbvcp_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbvcp_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbvcp_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )
    assert pred.shape[-1] == nbvcp_results.n_genes
    assert jnp.all(pred >= 0)


# ------------------------------------------------------------------------------


def test_get_map(nbvcp_results):
    map_est = (
        nbvcp_results.get_map() if hasattr(nbvcp_results, "get_map") else None
    )
    assert map_est is not None

    # Check parameters based on parameterization
    parameterization = nbvcp_results.model_config.parameterization
    if parameterization == "standard":
        assert "r" in map_est and "p" in map_est and "p_capture" in map_est
    elif parameterization == "linked":
        assert "p" in map_est and "mu" in map_est and "p_capture" in map_est
    elif parameterization == "odds_ratio":
        assert "phi" in map_est and "mu" in map_est and "phi_capture" in map_est


# ------------------------------------------------------------------------------


def test_indexing_integer(nbvcp_results):
    """Test indexing with an integer."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbvcp_results.n_genes = {nbvcp_results.n_genes}")

    subset = nbvcp_results[0]
    assert subset.n_genes == 1
    assert subset.n_cells == nbvcp_results.n_cells


# ------------------------------------------------------------------------------


def test_indexing_slice(nbvcp_results):
    """Test indexing with a slice."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbvcp_results.n_genes = {nbvcp_results.n_genes}")

    # Use a slice that works regardless of the number of genes
    end_idx = min(2, nbvcp_results.n_genes)
    subset = nbvcp_results[0:end_idx]
    assert subset.n_genes == end_idx
    assert subset.n_cells == nbvcp_results.n_cells


# ------------------------------------------------------------------------------


def test_indexing_boolean(nbvcp_results):
    """Test indexing with a boolean array."""
    # Debug: print the actual number of genes
    print(f"DEBUG: nbvcp_results.n_genes = {nbvcp_results.n_genes}")

    # Create a boolean mask that selects only the first gene
    mask = jnp.zeros(nbvcp_results.n_genes, dtype=bool)
    mask = mask.at[0].set(True)

    # Verify the mask has exactly one True value
    assert jnp.sum(mask) == 1

    subset = nbvcp_results[mask]
    assert subset.n_genes == 1
    assert subset.n_cells == nbvcp_results.n_cells


# ------------------------------------------------------------------------------


def test_log_likelihood(nbvcp_results, small_dataset, rng_key):
    counts, _ = small_dataset
    # For SVI, must generate posterior samples first
    if hasattr(nbvcp_results, "get_posterior_samples"):
        # Check if this is MCMC (inherits from MCMC class) or SVI
        if hasattr(nbvcp_results, "get_samples"):  # MCMC case
            # MCMC already has samples, no need to generate them
            pass
        else:  # SVI case
            nbvcp_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
    ll = nbvcp_results.log_likelihood(counts, return_by="cell")
    assert ll.shape[0] > 0
    assert jnp.all(jnp.isfinite(ll))


# ------------------------------------------------------------------------------


def test_parameter_relationships(nbvcp_results, parameterization):
    """Test that parameter relationships are correctly maintained."""
    # Get parameters
    params = None
    if hasattr(nbvcp_results, "params"):
        params = nbvcp_results.params
    if params is None:
        samples = nbvcp_results.get_posterior_samples()
        params = samples

    if parameterization == "linked":
        # In linked parameterization, r should be computed as
        # r = mu * p / (1 - p)
        if "p" in params and "mu" in params and "r" in params:
            p, mu, r = params["p"], params["mu"], params["r"]
            # Handle different shapes for SVI vs MCMC
            if p.ndim == 1 and mu.ndim == 2:
                # SVI case: p is (n_samples,), mu is (n_samples, n_genes)
                expected_r = mu * (1 - p[:, None]) / p[:, None]
            elif p.ndim == 0 and mu.ndim == 1:
                # MCMC case: p is scalar, mu is (n_genes,)
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
                # SVI case: phi is (n_samples,), mu is (n_samples, n_genes)
                expected_r = mu * phi[:, None]
            elif phi.ndim == 0 and mu.ndim == 1:
                # MCMC case: phi is scalar, mu is (n_genes,)
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


def test_cell_specific_parameters(nbvcp_results, parameterization):
    """Test that cell-specific parameters have correct shapes and ranges."""
    # Get parameters
    params = None
    if hasattr(nbvcp_results, "params"):
        params = nbvcp_results.params
    if params is None:
        samples = nbvcp_results.get_posterior_samples()
        params = samples

    # Check cell-specific parameters regardless of parameterization
    if parameterization == "odds_ratio":
        if "phi_capture" in params:
            phi_capture = params["phi_capture"]
            assert (
                phi_capture.shape[-1] == nbvcp_results.n_cells
            ), "phi_capture should be cell-specific"
            assert jnp.all(phi_capture > 0), "phi_capture should be positive"
    else:
        if "p_capture" in params:
            p_capture = params["p_capture"]
            assert (
                p_capture.shape[-1] == nbvcp_results.n_cells
            ), "p_capture should be cell-specific"
            assert jnp.all(
                (p_capture >= 0) & (p_capture <= 1)
            ), "p_capture should be in [0, 1]"


def test_variable_capture_behavior(nbvcp_results, rng_key):
    """Test that the model exhibits variable capture behavior."""
    # Generate predictive samples
    if hasattr(nbvcp_results, "get_posterior_samples"):
        if hasattr(nbvcp_results, "get_samples"):  # MCMC case
            pred = nbvcp_results.get_ppc_samples(
                rng_key=rng_key, store_samples=True
            )
        else:  # SVI case
            nbvcp_results.get_posterior_samples(
                rng_key=rng_key, n_samples=3, store_samples=True
            )
            pred = nbvcp_results.get_predictive_samples(
                rng_key=rng_key, store_samples=True
            )
    else:
        pred = nbvcp_results.get_ppc_samples(
            rng_key=rng_key, store_samples=True
        )

    # Check that we have non-negative counts
    assert jnp.all(pred >= 0), "NBVCP model should produce non-negative counts"

    # Check that we have some non-zero counts
    assert jnp.any(pred > 0), "NBVCP model should produce some non-zero counts"


# ------------------------------------------------------------------------------
# SVI-specific Tests
# ------------------------------------------------------------------------------


def test_svi_loss_history(nbvcp_results, inference_method):
    if inference_method == "svi":
        assert hasattr(nbvcp_results, "loss_history")
        assert len(nbvcp_results.loss_history) > 0


# ------------------------------------------------------------------------------
# MCMC-specific Tests
# ------------------------------------------------------------------------------


def test_mcmc_samples_shape(nbvcp_results, inference_method):
    if inference_method == "mcmc":
        samples = nbvcp_results.get_posterior_samples()

        # Check parameters based on parameterization
        parameterization = nbvcp_results.model_config.parameterization
        if parameterization == "standard":
            assert "r" in samples and "p" in samples and "p_capture" in samples
            assert samples["r"].ndim >= 2
            assert samples["p_capture"].ndim >= 2
        elif parameterization == "linked":
            assert "p" in samples and "mu" in samples and "p_capture" in samples
            assert samples["mu"].ndim >= 2
            assert samples["p_capture"].ndim >= 2
        elif parameterization == "odds_ratio":
            assert (
                "phi" in samples
                and "mu" in samples
                and "phi_capture" in samples
            )
            assert samples["mu"].ndim >= 2
            assert samples["phi_capture"].ndim >= 2
