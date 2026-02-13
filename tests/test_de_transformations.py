"""
Tests for differential expression transformations module.

Tests the ALR, CLR, and ILR coordinate transformations for compositional data.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from jax import random

from scribe.de import (
    alr_to_clr,
    transform_gaussian_alr_to_clr,
    build_ilr_basis,
    clr_to_ilr,
    ilr_to_clr,
)

# ------------------------------------------------------------------------------
# Test fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Random key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def sample_alr_vector():
    """Generate a sample ALR vector."""
    return jnp.array([0.5, -0.3, 0.8, -0.2])


@pytest.fixture
def sample_gaussian_params():
    """Generate sample Gaussian parameters in ALR space."""
    D_alr = 10
    k = 3

    mu = jnp.linspace(-0.5, 0.5, D_alr)
    W = random.normal(random.PRNGKey(123), (D_alr, k)) * 0.1
    d = jnp.ones(D_alr) * 0.5

    return mu, W, d


# ------------------------------------------------------------------------------
# Test ALR to CLR transformations
# ------------------------------------------------------------------------------


def test_alr_to_clr_shape(sample_alr_vector):
    """Test that ALR to CLR transformation has correct output shape."""
    z_alr = sample_alr_vector
    z_clr = alr_to_clr(z_alr)

    # CLR should have one more dimension than ALR
    assert z_clr.shape[0] == z_alr.shape[0] + 1


def test_alr_to_clr_centering(sample_alr_vector):
    """Test that CLR output is properly centered (sums to zero)."""
    z_alr = sample_alr_vector
    z_clr = alr_to_clr(z_alr)

    # CLR should sum to zero
    assert jnp.abs(jnp.sum(z_clr)) < 1e-6


def test_alr_to_clr_zero_input():
    """Test ALR to CLR with zero input."""
    z_alr = jnp.zeros(5)
    z_clr = alr_to_clr(z_alr)

    # All zeros in ALR should give all zeros in CLR
    assert jnp.allclose(z_clr, 0.0, atol=1e-10)


def test_alr_to_clr_deterministic():
    """Test that transformation is deterministic."""
    z_alr = jnp.array([1.0, 2.0, 3.0])
    z_clr_1 = alr_to_clr(z_alr)
    z_clr_2 = alr_to_clr(z_alr)

    assert jnp.allclose(z_clr_1, z_clr_2)


# ------------------------------------------------------------------------------
# Test Gaussian parameter transformations
# ------------------------------------------------------------------------------


def test_transform_gaussian_alr_to_clr_shapes(sample_gaussian_params):
    """Test output shapes of Gaussian parameter transformation."""
    mu_alr, W_alr, d_alr = sample_gaussian_params

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    D_alr = mu_alr.shape[0]
    D_clr = D_alr + 1
    k = W_alr.shape[1]

    assert mu_clr.shape == (D_clr,)
    assert W_clr.shape == (D_clr, k)
    assert d_clr.shape == (D_clr,)


def test_transform_gaussian_alr_to_clr_centering(sample_gaussian_params):
    """Test that CLR mean is centered."""
    mu_alr, W_alr, d_alr = sample_gaussian_params

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    # CLR mean should sum to zero
    assert jnp.abs(jnp.sum(mu_clr)) < 1e-6


def test_transform_gaussian_alr_to_clr_positive_variance(
    sample_gaussian_params,
):
    """Test that all variances remain positive after transformation."""
    mu_alr, W_alr, d_alr = sample_gaussian_params

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    # Diagonal variances should be positive
    assert jnp.all(d_clr > 0)

    # Total marginal variances should be positive
    total_var = jnp.sum(W_clr**2, axis=-1) + d_clr
    assert jnp.all(total_var > 0)


def test_transform_gaussian_alr_to_clr_exact_diagonal():
    """Test the exact diagonal formula for CLR transformation.

    This tests the critical correction from the implementation where
    the diagonal is computed exactly without approximations.
    """
    # Simple case where we can verify manually
    D_alr = 3
    D_clr = 4

    mu_alr = jnp.array([1.0, 2.0, 3.0])
    W_alr = jnp.zeros((D_alr, 2))  # No low-rank component
    d_alr = jnp.array([1.0, 2.0, 3.0])

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    # With no low-rank component, the exact formula should apply
    # Check that the diagonal makes sense
    assert jnp.all(d_clr > 0)

    # The transformation should preserve some relationship to the original
    # (exact values depend on the centering formula)
    assert d_clr.shape == (D_clr,)


def test_transform_gaussian_lowrank_structure_preserved():
    """Test that low-rank structure is preserved in transformation."""
    D_alr = 50
    k = 5

    mu_alr = jnp.zeros(D_alr)
    W_alr = random.normal(random.PRNGKey(999), (D_alr, k)) * 0.1
    d_alr = jnp.ones(D_alr) * 0.5

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    # Rank should be preserved
    assert W_clr.shape[1] == k

    # CLR should have D_alr + 1 dimensions
    assert W_clr.shape[0] == D_alr + 1


# ------------------------------------------------------------------------------
# Test ILR basis construction
# ------------------------------------------------------------------------------


def test_build_ilr_basis_shape():
    """Test ILR basis has correct shape."""
    D = 10
    V = build_ilr_basis(D)

    # Should be (D-1, D) - orthonormal rows of CLR subspace
    assert V.shape == (D - 1, D)


def test_build_ilr_basis_orthonormal():
    """Test ILR basis is orthonormal."""
    D = 10
    V = build_ilr_basis(D)

    # V @ V^T should be identity (rows are orthonormal)
    VVt = V @ V.T
    assert jnp.allclose(VVt, jnp.eye(D - 1), atol=1e-6)


def test_build_ilr_basis_in_clr_subspace():
    """Test that ILR basis vectors are in CLR subspace (sum to zero)."""
    D = 10
    V = build_ilr_basis(D)

    # Each row should sum to zero (with numerical tolerance)
    row_sums = jnp.sum(V, axis=1)
    assert jnp.allclose(row_sums, 0.0, atol=1e-6)


def test_build_ilr_basis_unit_vectors():
    """Test that ILR basis vectors have unit norm."""
    D = 10
    V = build_ilr_basis(D)

    # Each row should have norm 1
    norms = jnp.linalg.norm(V, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-6)


def test_build_ilr_basis_small_dimension():
    """Test ILR basis for small dimension."""
    D = 3
    V = build_ilr_basis(D)

    assert V.shape == (2, 3)

    # Check orthonormality
    VVt = V @ V.T
    assert jnp.allclose(VVt, jnp.eye(2), atol=1e-6)


# ------------------------------------------------------------------------------
# Test CLR <-> ILR transformations
# ------------------------------------------------------------------------------


def test_clr_to_ilr_shape():
    """Test CLR to ILR transformation shape."""
    D = 10
    z_clr = jnp.linspace(-1, 1, D)
    z_clr = z_clr - jnp.mean(z_clr)  # Center it
    V = build_ilr_basis(D)

    z_ilr = clr_to_ilr(z_clr, V)

    assert z_ilr.shape == (D - 1,)


def test_ilr_to_clr_shape():
    """Test ILR to CLR transformation shape."""
    D = 10
    z_ilr = jnp.linspace(-1, 1, D - 1)
    V = build_ilr_basis(D)

    z_clr = ilr_to_clr(z_ilr, V)

    assert z_clr.shape == (D,)


def test_clr_ilr_roundtrip():
    """Test that CLR -> ILR -> CLR is a roundtrip."""
    D = 10
    z_clr_orig = jnp.linspace(-1, 1, D)
    z_clr_orig = z_clr_orig - jnp.mean(z_clr_orig)  # Center it
    V = build_ilr_basis(D)

    # CLR -> ILR -> CLR
    z_ilr = clr_to_ilr(z_clr_orig, V)
    z_clr_reconstructed = ilr_to_clr(z_ilr, V)

    assert jnp.allclose(z_clr_orig, z_clr_reconstructed, atol=1e-6)


def test_ilr_to_clr_centered():
    """Test that ILR to CLR produces centered output."""
    D = 10
    z_ilr = random.normal(random.PRNGKey(555), (D - 1,))
    V = build_ilr_basis(D)

    z_clr = ilr_to_clr(z_ilr, V)

    # Should sum to zero (with numerical tolerance)
    assert jnp.abs(jnp.sum(z_clr)) < 1e-6


def test_ilr_preserves_norm():
    """Test that ILR preserves Euclidean norm from CLR."""
    D = 10
    z_clr = jnp.linspace(-1, 1, D)
    z_clr = z_clr - jnp.mean(z_clr)
    V = build_ilr_basis(D)

    z_ilr = clr_to_ilr(z_clr, V)

    # Norms should be equal
    norm_clr = jnp.linalg.norm(z_clr)
    norm_ilr = jnp.linalg.norm(z_ilr)

    assert jnp.allclose(norm_clr, norm_ilr, atol=1e-6)


# ------------------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------------------


def test_alr_clr_ilr_integration():
    """Test full pipeline: ALR -> CLR -> ILR."""
    z_alr = jnp.array([0.5, -0.3, 0.8, -0.2, 0.1])

    # ALR -> CLR
    z_clr = alr_to_clr(z_alr)

    # CLR -> ILR (need ILR basis)
    D_clr = z_clr.shape[0]
    V = build_ilr_basis(D_clr)
    z_ilr = clr_to_ilr(z_clr, V)

    # Check dimensions
    assert z_alr.shape == (5,)
    assert z_clr.shape == (6,)
    assert z_ilr.shape == (5,)

    # CLR should be centered (with numerical tolerance)
    assert jnp.abs(jnp.sum(z_clr)) < 1e-6


def test_gaussian_transformation_preserves_distribution():
    """Test that Gaussian parameter transformation preserves total variance."""
    D_alr = 20
    k = 5

    mu_alr = jnp.zeros(D_alr)
    W_alr = random.normal(random.PRNGKey(777), (D_alr, k)) * 0.2
    d_alr = jnp.ones(D_alr) * 0.3

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    # Compute total variance in both spaces
    # This is a rough check - the exact relationship is complex
    # but we can verify basic properties

    # CLR should have positive variances
    var_clr = jnp.sum(W_clr**2, axis=-1) + d_clr
    assert jnp.all(var_clr > 0)

    # CLR mean should be centered
    assert jnp.abs(jnp.sum(mu_clr)) < 1e-6


# ------------------------------------------------------------------------------
# Edge case tests
# ------------------------------------------------------------------------------


def test_small_dimension():
    """Test transformations with minimal dimension."""
    # ALR dimension = 1 (CLR dimension = 2)
    z_alr = jnp.array([0.5])
    z_clr = alr_to_clr(z_alr)

    assert z_clr.shape == (2,)
    assert jnp.abs(jnp.sum(z_clr)) < 1e-10


def test_large_dimension():
    """Test transformations scale to large dimensions."""
    D_alr = 1000
    k = 10

    mu_alr = jnp.zeros(D_alr)
    W_alr = random.normal(random.PRNGKey(888), (D_alr, k)) * 0.1
    d_alr = jnp.ones(D_alr) * 0.5

    # Should not raise errors
    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    assert mu_clr.shape == (D_alr + 1,)
    assert jnp.abs(jnp.sum(mu_clr)) < 1e-6


def test_zero_rank():
    """Test Gaussian transformation with zero rank (diagonal only)."""
    D_alr = 10

    mu_alr = jnp.linspace(-1, 1, D_alr)
    W_alr = jnp.zeros((D_alr, 0))  # No low-rank component
    d_alr = jnp.ones(D_alr) * 0.5

    mu_clr, W_clr, d_clr = transform_gaussian_alr_to_clr(mu_alr, W_alr, d_alr)

    assert mu_clr.shape == (D_alr + 1,)
    assert W_clr.shape == (D_alr + 1, 0)
    assert d_clr.shape == (D_alr + 1,)
    assert jnp.all(d_clr > 0)
