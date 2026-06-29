"""Shared fixtures for inference tests."""

import os
import pytest
import jax.numpy as jnp
from jax import random
from scribe.models.config import (
    ModelConfigBuilder,
    InferenceConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
)


# =============================================================================
# Pytest configuration for slow tests
# =============================================================================


def pytest_addoption(parser):
    """Add --run-slow option to pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow integration tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skip unless --run-slow)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is specified."""
    if config.getoption("--run-slow"):
        # --run-slow given: do not skip slow tests
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def sample_counts():
    """Create a small sample count matrix for testing."""
    rng_key = random.PRNGKey(42)
    # Small dataset: 50 cells, 100 genes
    counts = random.poisson(rng_key, lam=5.0, shape=(50, 100))
    return jnp.array(counts)


@pytest.fixture
def nbdm_model_config():
    """Create a basic NBDM model config."""
    return (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_parameterization("canonical")
        .with_inference("svi")
        .build()
    )


@pytest.fixture
def zinb_model_config():
    """Create a ZINB model config."""
    return (
        ModelConfigBuilder()
        .for_model("zinb")
        .with_parameterization("mean_prob")
        .with_inference("svi")
        .build()
    )


@pytest.fixture
def svi_inference_config():
    """Create a basic SVI inference config."""
    return InferenceConfig.from_svi(SVIConfig(n_steps=100, batch_size=None))


@pytest.fixture
def mcmc_inference_config():
    """Create a basic MCMC inference config."""
    return InferenceConfig.from_mcmc(MCMCConfig(n_samples=100, n_warmup=50))


@pytest.fixture
def data_config():
    """Create a basic data config."""
    return DataConfig(cells_axis=0, layer=None)
