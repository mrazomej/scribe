"""
Shared test fixtures and configuration for SCRIBE tests.
"""

import pytest
import numpy as np
import os


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to run tests on: cpu or gpu",
    )
    parser.addoption(
        "--method",
        default="all",
        choices=["svi", "mcmc", "all"],
        help="Inference method to test: svi, mcmc, or all",
    )
    parser.addoption(
        "--parameterization",
        default="all",
        choices=["standard", "linked", "odds_ratio", "all"],
        help=(
            "Model parameterization to test: standard, linked, odds_ratio, or all"
        ),
    )
    parser.addoption(
        "--unconstrained",
        default="all",
        choices=["false", "true", "all"],
        help="Whether to test unconstrained variants: false, true, or all",
    )
    parser.addoption(
        "--guide-rank",
        default="all",
        help="Guide rank to test: none (mean-field), integer (low-rank), or all",
    )


def pytest_configure(config):
    """Configure JAX device before any imports happen."""
    device = config.getoption("--device")
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    else:
        # Remove the environment variable to allow JAX to use GPU
        if "JAX_PLATFORM_NAME" in os.environ:
            del os.environ["JAX_PLATFORM_NAME"]


@pytest.fixture(scope="session")
def device_type(request):
    return request.config.getoption("--device")


@pytest.fixture(scope="session")
def inference_method(request):
    return request.config.getoption("--method")


@pytest.fixture(scope="session")
def parameterization(request):
    return request.config.getoption("--parameterization")


@pytest.fixture(scope="session")
def unconstrained(request):
    """Convert string option to boolean for unconstrained parameter."""
    opt = request.config.getoption("--unconstrained")
    if opt == "all":
        return "all"
    return opt == "true"


@pytest.fixture(scope="session")
def rng_key():
    """Provide a consistent random key for tests."""
    # Import JAX here to ensure environment is configured first
    from jax import random

    return random.PRNGKey(42)


@pytest.fixture(scope="session")
def small_dataset():
    """Generate a small test dataset."""
    # Import JAX here to ensure environment is configured first
    import jax.numpy as jnp

    n_cells = 10
    n_genes = 5

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random count data
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    total_counts = counts.sum(axis=1)

    return jnp.array(counts), jnp.array(total_counts)
