"""
Shared test fixtures and configuration for SCRIBE tests.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random


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
        choices=["standard", "linked", "odds_ratio", "unconstrained", "all"],
        help=(
            "Model parameterization to test: standard, linked, odds_ratio, "
            "unconstrained, or all"
        ),
    )


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
def rng_key():
    """Provide a consistent random key for tests."""
    return random.PRNGKey(42)


@pytest.fixture(scope="session")
def small_dataset():
    """Generate a small test dataset."""
    n_cells = 10
    n_genes = 5

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random count data
    counts = np.random.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes))
    total_counts = counts.sum(axis=1)

    return jnp.array(counts), jnp.array(total_counts)
