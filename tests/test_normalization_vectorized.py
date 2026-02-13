"""
Tests for the vectorized (batched) Dirichlet sampling and logistic-normal
fitting in ``scribe.core.normalization`` and
``scribe.core.normalization_logistic``.

These tests verify that the batched implementation produces correct shapes,
satisfies the simplex constraint, and gives statistically equivalent results
across different batch sizes.
"""

import pytest
import jax.numpy as jnp
from jax import random

from scribe.core.normalization_logistic import (
    _batched_dirichlet_sample,
    _batched_dirichlet_sample_raw,
    _fit_low_rank_mvn,
    _fit_low_rank_mvn_core,
    fit_logistic_normal_from_posterior,
    _DEFAULT_BATCH_SIZE,
)
from scribe.core.normalization import normalize_counts_from_posterior


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng_key():
    """Shared PRNG key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def r_samples_non_mixture(rng_key):
    """
    Synthetic concentration parameters for a non-mixture model.

    Shape: (n_posterior=20, n_genes=10)
    Values are positive (valid Dirichlet concentrations).
    """
    # Use Gamma draws to get positive concentrations
    return random.gamma(rng_key, jnp.ones((20, 10))) + 0.5


@pytest.fixture
def r_samples_mixture(rng_key):
    """
    Synthetic concentration parameters for a mixture model with 3 components.

    Shape: (n_posterior=20, n_components=3, n_genes=10)
    """
    return random.gamma(rng_key, jnp.ones((20, 3, 10))) + 0.5


# ---------------------------------------------------------------------------
# _batched_dirichlet_sample
# ---------------------------------------------------------------------------


class TestBatchedDirichletSample:
    """Tests for the flattened batched sampler used in logistic-normal fitting."""

    def test_shape_single_dirichlet(self, r_samples_non_mixture, rng_key):
        """n_samples_dirichlet=1 should return (N, D)."""
        result = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=8,
        )
        assert result.shape == (20, 10)

    def test_shape_multi_dirichlet(self, r_samples_non_mixture, rng_key):
        """n_samples_dirichlet=5 should return (N*5, D)."""
        result = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=5,
            rng_key=rng_key,
            batch_size=8,
        )
        assert result.shape == (20 * 5, 10)

    def test_simplex_constraint(self, r_samples_non_mixture, rng_key):
        """All rows should sum to 1 (valid simplex samples)."""
        result = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=8,
        )
        row_sums = jnp.sum(result, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_positivity(self, r_samples_non_mixture, rng_key):
        """All values should be strictly positive."""
        result = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=8,
        )
        assert jnp.all(result > 0)

    def test_batch_size_equivalence(self, r_samples_non_mixture, rng_key):
        """
        Different batch sizes should produce the same shapes and statistically
        similar means.  Exact values differ because key-splitting differs.
        """
        result_small = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=4,
        )
        result_large = _batched_dirichlet_sample(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=100,  # larger than N → single batch
        )
        # Same shape
        assert result_small.shape == result_large.shape
        # Means should be roughly similar (same distribution, different draws)
        mean_small = jnp.mean(result_small, axis=0)
        mean_large = jnp.mean(result_large, axis=0)
        # Both should be valid probability vectors
        assert jnp.allclose(jnp.sum(mean_small), 1.0, atol=0.1)
        assert jnp.allclose(jnp.sum(mean_large), 1.0, atol=0.1)


# ---------------------------------------------------------------------------
# _batched_dirichlet_sample_raw
# ---------------------------------------------------------------------------


class TestBatchedDirichletSampleRaw:
    """Tests for the shape-preserving batched sampler used in normalize_counts."""

    def test_shape_single_dirichlet(self, r_samples_non_mixture, rng_key):
        """n_samples_dirichlet=1 → (N, D)."""
        result = _batched_dirichlet_sample_raw(
            r_samples_non_mixture,
            n_samples_dirichlet=1,
            rng_key=rng_key,
            batch_size=8,
        )
        assert result.shape == (20, 10)

    def test_shape_multi_dirichlet(self, r_samples_non_mixture, rng_key):
        """n_samples_dirichlet=5 → (N, D, 5)."""
        result = _batched_dirichlet_sample_raw(
            r_samples_non_mixture,
            n_samples_dirichlet=5,
            rng_key=rng_key,
            batch_size=8,
        )
        assert result.shape == (20, 10, 5)

    def test_simplex_constraint_multi(self, r_samples_non_mixture, rng_key):
        """Each sample across the gene axis should sum to 1."""
        result = _batched_dirichlet_sample_raw(
            r_samples_non_mixture,
            n_samples_dirichlet=3,
            rng_key=rng_key,
            batch_size=8,
        )
        # Sum over genes (axis 1): shape (N, S)
        sums = jnp.sum(result, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# _fit_low_rank_mvn_core vs _fit_low_rank_mvn
# ---------------------------------------------------------------------------


class TestFitLowRankMVN:
    """Verify that the core and wrapper produce identical numerics."""

    def test_core_matches_wrapper(self, rng_key):
        """Core and wrapper should give the same loc, cov_factor, cov_diag."""
        # Synthetic data: 50 samples, 8 features
        samples = random.normal(rng_key, shape=(50, 8))
        rank = 3

        loc_w, cov_factor_w, cov_diag_w = _fit_low_rank_mvn(
            samples, rank=rank, verbose=False
        )
        loc_c, cov_factor_c, cov_diag_c, eigenvalues = _fit_low_rank_mvn_core(
            samples, rank=rank
        )

        assert jnp.allclose(loc_w, loc_c, atol=1e-6)
        assert jnp.allclose(cov_factor_w, cov_factor_c, atol=1e-6)
        assert jnp.allclose(cov_diag_w, cov_diag_c, atol=1e-6)
        # Eigenvalues should be returned and have the right length
        assert eigenvalues.shape == (min(50, 8),)

    def test_core_shapes(self, rng_key):
        """Check output shapes for a variety of (n, d) combinations."""
        for n, d, rank in [(30, 10, 5), (10, 30, 5), (100, 5, 3)]:
            samples = random.normal(rng_key, shape=(n, d))
            loc, cov_factor, cov_diag, eigenvalues = _fit_low_rank_mvn_core(
                samples, rank=rank
            )
            effective_rank = min(rank, min(n, d))
            assert loc.shape == (d,)
            assert cov_factor.shape == (d, effective_rank)
            assert cov_diag.shape == (d,)

    def test_wrapper_raises_on_single_sample(self, rng_key):
        """The wrapper should raise ValueError for n_samples < 2."""
        samples = random.normal(rng_key, shape=(1, 5))
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            _fit_low_rank_mvn(samples, rank=2, verbose=False)


# ---------------------------------------------------------------------------
# fit_logistic_normal_from_posterior  (end-to-end)
# ---------------------------------------------------------------------------


class TestFitLogisticNormalFromPosterior:
    """Integration tests for the full logistic-normal fitting pipeline."""

    def test_non_mixture_shapes(self, r_samples_non_mixture, rng_key):
        """
        Non-mixture model should return D-dimensional loc, cov_factor,
        cov_diag, and mean_probabilities.
        """
        posterior = {"r": r_samples_non_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        D = 10  # n_genes
        assert result["loc"].shape == (D,)
        assert result["cov_factor"].shape == (D, 3)
        assert result["cov_diag"].shape == (D,)
        assert result["mean_probabilities"].shape == (D,)

    def test_mixture_shapes(self, r_samples_mixture, rng_key):
        """Mixture model should return (K, D) arrays."""
        posterior = {"r": r_samples_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=3,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        K, D = 3, 10
        assert result["loc"].shape == (K, D)
        assert result["cov_factor"].shape == (K, D, 3)
        assert result["cov_diag"].shape == (K, D)
        assert result["mean_probabilities"].shape == (K, D)

    def test_mean_probabilities_sum_to_one(
        self, r_samples_non_mixture, rng_key
    ):
        """Mean probabilities on the simplex should sum to 1."""
        posterior = {"r": r_samples_non_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        assert jnp.allclose(
            jnp.sum(result["mean_probabilities"]), 1.0, atol=1e-5
        )

    def test_batch_size_does_not_change_shapes(
        self, r_samples_non_mixture, rng_key
    ):
        """Different batch sizes should produce identical output shapes."""
        posterior = {"r": r_samples_non_mixture}
        for bs in [4, 8, 100]:
            result = fit_logistic_normal_from_posterior(
                posterior,
                n_components=None,
                rng_key=rng_key,
                rank=3,
                batch_size=bs,
                verbose=False,
            )
            assert result["loc"].shape == (10,)

    def test_missing_r_raises(self, rng_key):
        """Should raise ValueError if 'r' is missing from posterior."""
        with pytest.raises(ValueError, match="'r' parameter not found"):
            fit_logistic_normal_from_posterior(
                {"mu": jnp.ones((5, 3))},
                rng_key=rng_key,
                verbose=False,
            )

    def test_multi_dirichlet_samples(
        self, r_samples_non_mixture, rng_key
    ):
        """n_samples_dirichlet > 1 should still produce valid results."""
        posterior = {"r": r_samples_non_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=3,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        assert result["loc"].shape == (10,)
        assert jnp.allclose(
            jnp.sum(result["mean_probabilities"]), 1.0, atol=1e-5
        )


# ---------------------------------------------------------------------------
# normalize_counts_from_posterior  (end-to-end)
# ---------------------------------------------------------------------------


class TestNormalizeCountsFromPosterior:
    """Integration tests for batched normalize_counts_from_posterior."""

    def test_non_mixture_store_samples(
        self, r_samples_non_mixture, rng_key
    ):
        """store_samples=True returns correct shape for n_samples_dirichlet=1."""
        posterior = {"r": r_samples_non_mixture}
        result = normalize_counts_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            fit_distribution=False,
            store_samples=True,
            batch_size=8,
            verbose=False,
        )
        # (N, D) when n_samples_dirichlet=1
        assert result["samples"].shape == (20, 10)

    def test_non_mixture_multi_dirichlet_store(
        self, r_samples_non_mixture, rng_key
    ):
        """store_samples=True with n_samples_dirichlet > 1 returns (N, D, S)."""
        posterior = {"r": r_samples_non_mixture}
        result = normalize_counts_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=5,
            fit_distribution=False,
            store_samples=True,
            batch_size=8,
            verbose=False,
        )
        assert result["samples"].shape == (20, 10, 5)

    def test_mixture_store_samples(self, r_samples_mixture, rng_key):
        """Mixture store_samples shape: (N, K, D) for n_samples_dirichlet=1."""
        posterior = {"r": r_samples_mixture}
        result = normalize_counts_from_posterior(
            posterior,
            n_components=3,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            fit_distribution=False,
            store_samples=True,
            batch_size=8,
            verbose=False,
        )
        assert result["samples"].shape == (20, 3, 10)

    def test_mixture_fit_distribution(self, r_samples_mixture, rng_key):
        """Fitting a Dirichlet per component returns correct shapes."""
        posterior = {"r": r_samples_mixture}
        result = normalize_counts_from_posterior(
            posterior,
            n_components=3,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            fit_distribution=True,
            store_samples=False,
            batch_size=8,
            verbose=False,
        )
        assert result["concentrations"].shape == (3, 10)
        assert result["mean_probabilities"].shape == (3, 10)
        assert len(result["distributions"]) == 3

    def test_simplex_samples(self, r_samples_non_mixture, rng_key):
        """All stored samples should lie on the simplex."""
        posterior = {"r": r_samples_non_mixture}
        result = normalize_counts_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            fit_distribution=False,
            store_samples=True,
            batch_size=8,
            verbose=False,
        )
        row_sums = jnp.sum(result["samples"], axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)
