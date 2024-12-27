"""
Shared test fixtures and configuration for SCRIBE tests.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

@pytest.fixture
def rng_key():
    """Provide a consistent random key for tests."""
    return random.PRNGKey(42)

@pytest.fixture
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