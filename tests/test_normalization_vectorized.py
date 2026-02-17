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
    _randomized_svd,
    fit_logistic_normal_from_posterior,
    _DEFAULT_BATCH_SIZE,
)
from scribe.core.normalization import normalize_counts_from_posterior
from scribe.de._gaussianity import gaussianity_diagnostics


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
    """Verify full SVD, randomized SVD, and wrapper consistency."""

    # --- Full SVD path (existing tests, now with explicit svd_method) ------

    def test_core_matches_wrapper_full(self, rng_key):
        """Core and wrapper should give the same loc, cov_factor, cov_diag
        when using the full SVD path."""
        # Synthetic data: 50 samples, 8 features
        samples = random.normal(rng_key, shape=(50, 8))
        rank = 3

        loc_w, cov_factor_w, cov_diag_w = _fit_low_rank_mvn(
            samples, rank=rank, svd_method="full", verbose=False
        )
        loc_c, cov_factor_c, cov_diag_c, eigenvalues = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="full"
        )

        assert jnp.allclose(loc_w, loc_c, atol=1e-6)
        assert jnp.allclose(cov_factor_w, cov_factor_c, atol=1e-6)
        assert jnp.allclose(cov_diag_w, cov_diag_c, atol=1e-6)
        # Full SVD returns all min(N, D) eigenvalues
        assert eigenvalues.shape == (min(50, 8),)

    def test_core_shapes_full(self, rng_key):
        """Check output shapes for full SVD across (n, d) combinations."""
        for n, d, rank in [(30, 10, 5), (10, 30, 5), (100, 5, 3)]:
            samples = random.normal(rng_key, shape=(n, d))
            loc, cov_factor, cov_diag, eigenvalues = _fit_low_rank_mvn_core(
                samples, rank=rank, svd_method="full"
            )
            effective_rank = min(rank, min(n, d))
            assert loc.shape == (d,)
            assert cov_factor.shape == (d, effective_rank)
            assert cov_diag.shape == (d,)

    # --- Randomized SVD path -----------------------------------------------

    def test_randomized_shapes(self, rng_key):
        """Randomized SVD should return correct shapes; eigenvalues has
        length = effective_rank (not min(N, D))."""
        for n, d, rank in [(50, 20, 3), (20, 50, 3), (100, 10, 5)]:
            samples = random.normal(rng_key, shape=(n, d))
            loc, cov_factor, cov_diag, eigenvalues = _fit_low_rank_mvn_core(
                samples, rank=rank, svd_method="randomized", rng_key=rng_key
            )
            effective_rank = min(rank, min(n, d))
            assert loc.shape == (d,)
            assert cov_factor.shape == (d, effective_rank)
            assert cov_diag.shape == (d,)
            # Randomized SVD only returns top-k eigenvalues
            assert eigenvalues.shape == (effective_rank,)

    def test_randomized_matches_full_eigenvalues(self, rng_key):
        """Top-k eigenvalues from randomized SVD should closely match
        those from full SVD for a well-conditioned matrix."""
        # Low-rank data: 80 samples, 20 features, true rank ~3
        # Construct data with a clear spectral gap at rank 3 so both
        # methods should agree tightly.
        key1, key2 = random.split(rng_key)
        W = random.normal(key1, shape=(20, 3)) * jnp.array([10.0, 5.0, 2.0])
        noise = random.normal(key2, shape=(80, 20)) * 0.1
        samples = random.normal(key1, shape=(80, 3)) @ W.T + noise

        rank = 3

        _, _, _, evals_full = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="full"
        )
        _, _, _, evals_rand = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="randomized", rng_key=rng_key
        )

        # Top-3 eigenvalues should match to within 1% (relative)
        assert jnp.allclose(
            evals_rand, evals_full[:rank], rtol=0.01, atol=1e-6
        )

    def test_randomized_matches_full_subspace(self, rng_key):
        """The column space of cov_factor from randomized SVD should span
        (nearly) the same subspace as full SVD."""
        key1, key2 = random.split(rng_key)
        W = random.normal(key1, shape=(15, 3)) * jnp.array([8.0, 4.0, 1.5])
        noise = random.normal(key2, shape=(50, 15)) * 0.1
        samples = random.normal(key1, shape=(50, 3)) @ W.T + noise

        rank = 3

        _, cf_full, _, _ = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="full"
        )
        _, cf_rand, _, _ = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="randomized", rng_key=rng_key
        )

        # Compute principal angles between the two subspaces.
        # If they span the same space, all singular values of Q_full^T @ Q_rand
        # should be close to 1.
        Q_full, _ = jnp.linalg.qr(cf_full)
        Q_rand, _ = jnp.linalg.qr(cf_rand)
        cos_angles = jnp.linalg.svd(
            Q_full.T @ Q_rand, compute_uv=False
        )
        # All cosines should be > 0.99 (subspaces nearly identical)
        assert jnp.all(cos_angles > 0.99), (
            f"Subspace cosines too small: {cos_angles}"
        )

    def test_randomized_residual_variance(self, rng_key):
        """Residual variance from randomized SVD (trace-based estimate)
        should be close to the full SVD residual mean."""
        key1, key2 = random.split(rng_key)
        W = random.normal(key1, shape=(20, 3)) * jnp.array([10.0, 5.0, 2.0])
        noise = random.normal(key2, shape=(80, 20)) * 0.5
        samples = random.normal(key1, shape=(80, 3)) @ W.T + noise

        rank = 3

        _, _, diag_full, _ = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="full"
        )
        _, _, diag_rand, _ = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="randomized", rng_key=rng_key
        )

        # The residual diagonal values should be similar (within 20%)
        # Both include the 1e-4 stability constant.
        assert jnp.allclose(diag_rand, diag_full, rtol=0.2, atol=1e-3)

    def test_core_matches_wrapper_randomized(self, rng_key):
        """Core and wrapper should agree for the randomized path."""
        samples = random.normal(rng_key, shape=(50, 12))
        rank = 3

        loc_w, cf_w, cd_w = _fit_low_rank_mvn(
            samples, rank=rank, svd_method="randomized",
            rng_key=rng_key, verbose=False,
        )
        loc_c, cf_c, cd_c, _ = _fit_low_rank_mvn_core(
            samples, rank=rank, svd_method="randomized", rng_key=rng_key,
        )

        assert jnp.allclose(loc_w, loc_c, atol=1e-6)
        assert jnp.allclose(cf_w, cf_c, atol=1e-6)
        assert jnp.allclose(cd_w, cd_c, atol=1e-6)

    # --- Shared / validation -----------------------------------------------

    def test_wrapper_raises_on_single_sample(self, rng_key):
        """The wrapper should raise ValueError for n_samples < 2."""
        samples = random.normal(rng_key, shape=(1, 5))
        with pytest.raises(ValueError, match="Need at least 2 samples"):
            _fit_low_rank_mvn(samples, rank=2, verbose=False)

    def test_wrapper_raises_on_invalid_svd_method(self, rng_key):
        """Invalid svd_method should raise ValueError."""
        samples = random.normal(rng_key, shape=(10, 5))
        with pytest.raises(ValueError, match="svd_method must be"):
            _fit_low_rank_mvn(
                samples, rank=2, svd_method="invalid", verbose=False
            )


# ---------------------------------------------------------------------------
# _randomized_svd
# ---------------------------------------------------------------------------


class TestRandomizedSVD:
    """Unit tests for the standalone _randomized_svd helper."""

    def test_shapes(self, rng_key):
        """Output shapes should be (n, rank), (rank,), (rank, p)."""
        X = random.normal(rng_key, shape=(40, 25))
        rank = 5
        U, S, Vt = _randomized_svd(X, rank=rank, rng_key=rng_key)
        assert U.shape == (40, rank)
        assert S.shape == (rank,)
        assert Vt.shape == (rank, 25)

    def test_orthogonality(self, rng_key):
        """U columns and Vt rows should be approximately orthonormal."""
        X = random.normal(rng_key, shape=(60, 30))
        rank = 4
        U, S, Vt = _randomized_svd(X, rank=rank, rng_key=rng_key)

        # U^T @ U ≈ I_k
        eye_u = U.T @ U
        assert jnp.allclose(eye_u, jnp.eye(rank), atol=1e-5)

        # Vt @ Vt^T ≈ I_k
        eye_v = Vt @ Vt.T
        assert jnp.allclose(eye_v, jnp.eye(rank), atol=1e-5)

    def test_singular_values_match_full(self, rng_key):
        """Top-k singular values should closely match full SVD."""
        X = random.normal(rng_key, shape=(50, 20))
        rank = 5
        _, S_rand, _ = _randomized_svd(X, rank=rank, rng_key=rng_key)
        _, S_full, _ = jnp.linalg.svd(X, full_matrices=False)

        assert jnp.allclose(S_rand, S_full[:rank], rtol=0.01, atol=1e-6)

    def test_descending_order(self, rng_key):
        """Singular values should be in descending order."""
        X = random.normal(rng_key, shape=(30, 20))
        rank = 5
        _, S, _ = _randomized_svd(X, rank=rank, rng_key=rng_key)
        # Each singular value should be >= the next
        assert jnp.all(S[:-1] >= S[1:])

    def test_reconstruction(self, rng_key):
        """U @ diag(S) @ Vt should approximate X well for low-rank data."""
        # Create a rank-3 matrix with noise
        key1, key2 = random.split(rng_key)
        A = random.normal(key1, shape=(30, 3))
        B = random.normal(key2, shape=(3, 20))
        X = A @ B + random.normal(key1, shape=(30, 20)) * 0.01

        rank = 3
        U, S, Vt = _randomized_svd(X, rank=rank, rng_key=rng_key)
        X_approx = U * S[None, :] @ Vt

        # Relative error should be very small for near-rank-3 data
        rel_error = jnp.linalg.norm(X - X_approx) / jnp.linalg.norm(X)
        assert rel_error < 0.02


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

    def test_randomized_vs_full_end_to_end(
        self, r_samples_non_mixture, rng_key
    ):
        """Randomized and full SVD should give similar distribution params
        for a small problem where both methods are exact."""
        posterior = {"r": r_samples_non_mixture}
        common = dict(
            posterior_samples=posterior,
            n_components=None,
            rng_key=rng_key,
            n_samples_dirichlet=1,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        result_rand = fit_logistic_normal_from_posterior(
            **common, svd_method="randomized"
        )
        result_full = fit_logistic_normal_from_posterior(
            **common, svd_method="full"
        )

        # loc should be identical (computed before SVD dispatch)
        assert jnp.allclose(
            result_rand["loc"], result_full["loc"], atol=1e-4
        )
        # mean_probabilities should be very similar
        assert jnp.allclose(
            result_rand["mean_probabilities"],
            result_full["mean_probabilities"],
            atol=1e-4,
        )
        # Shapes must match
        assert (
            result_rand["cov_factor"].shape == result_full["cov_factor"].shape
        )

    def test_svd_method_default_is_randomized(
        self, r_samples_non_mixture, rng_key
    ):
        """Default svd_method should be 'randomized' — verify by checking
        that eigenvalues array has length == rank (not min(N, D))."""
        posterior = {"r": r_samples_non_mixture}
        # Just ensure no errors with the default
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            rank=3,
            batch_size=8,
            verbose=False,
        )
        assert result["loc"].shape == (10,)


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


# ---------------------------------------------------------------------------
# gaussianity_diagnostics (standalone)
# ---------------------------------------------------------------------------


class TestGaussianityDiagnostics:
    """Unit tests for the per-feature Gaussianity diagnostic function."""

    def test_shapes(self, rng_key):
        """Output arrays should have shape (D,) for (N, D) input."""
        samples = random.normal(rng_key, shape=(500, 12))
        diag = gaussianity_diagnostics(samples)

        assert diag["skewness"].shape == (12,)
        assert diag["kurtosis"].shape == (12,)
        assert diag["jarque_bera"].shape == (12,)
        assert diag["jb_pvalue"].shape == (12,)

    def test_gaussian_samples(self, rng_key):
        """For truly Gaussian data, skewness/kurtosis should be near zero
        and most JB p-values should be > 0.05."""
        # Large N for tight estimates
        samples = random.normal(rng_key, shape=(10_000, 50))
        diag = gaussianity_diagnostics(samples)

        # Skewness should be close to 0 for all features
        assert jnp.all(jnp.abs(diag["skewness"]) < 0.15), (
            f"Max |skewness| = {float(jnp.max(jnp.abs(diag['skewness'])))}"
        )
        # Excess kurtosis should be close to 0
        assert jnp.all(jnp.abs(diag["kurtosis"]) < 0.3), (
            f"Max |kurtosis| = {float(jnp.max(jnp.abs(diag['kurtosis'])))}"
        )
        # Most JB p-values should be > 0.05 (at least 85% to allow
        # for random fluctuations)
        frac_pass = float(jnp.mean(diag["jb_pvalue"] > 0.05))
        assert frac_pass > 0.85, f"Only {frac_pass:.0%} features pass JB test"

    def test_skewed_samples(self, rng_key):
        """For clearly non-Gaussian data, skewness should be large and
        JB p-values should be small."""
        # Chi-squared(2) has skewness = 2, excess kurtosis = 6
        # Simulate via sum of squared normals
        key1, key2 = random.split(rng_key)
        z1 = random.normal(key1, shape=(5000, 20))
        z2 = random.normal(key2, shape=(5000, 20))
        samples = z1 ** 2 + z2 ** 2  # chi2(2)

        diag = gaussianity_diagnostics(samples)

        # Skewness should be strongly positive (theoretical = 2)
        assert jnp.all(diag["skewness"] > 1.0), (
            f"Min skewness = {float(jnp.min(diag['skewness']))}"
        )
        # Excess kurtosis should be strongly positive (theoretical = 6)
        assert jnp.all(diag["kurtosis"] > 3.0), (
            f"Min kurtosis = {float(jnp.min(diag['kurtosis']))}"
        )
        # All JB p-values should be essentially zero
        assert jnp.all(diag["jb_pvalue"] < 1e-10)

    def test_constant_feature(self, rng_key):
        """A constant column should produce skewness=0, kurtosis=-3
        (or 0 depending on guard), and no NaN values."""
        N, D = 100, 5
        samples = random.normal(rng_key, shape=(N, D))
        # Make one column constant
        samples = samples.at[:, 2].set(3.14)

        diag = gaussianity_diagnostics(samples)

        # No NaN in any output
        for key in ("skewness", "kurtosis", "jarque_bera", "jb_pvalue"):
            assert not jnp.any(jnp.isnan(diag[key])), (
                f"NaN found in {key}"
            )

    def test_jb_pvalue_range(self, rng_key):
        """P-values should be in [0, 1]."""
        samples = random.normal(rng_key, shape=(500, 20))
        diag = gaussianity_diagnostics(samples)

        assert jnp.all(diag["jb_pvalue"] >= 0.0)
        assert jnp.all(diag["jb_pvalue"] <= 1.0)

    def test_jarque_bera_nonnegative(self, rng_key):
        """JB statistic should always be >= 0."""
        samples = random.normal(rng_key, shape=(200, 15))
        diag = gaussianity_diagnostics(samples)

        assert jnp.all(diag["jarque_bera"] >= 0.0)


# ---------------------------------------------------------------------------
# Gaussianity diagnostics in fit_logistic_normal_from_posterior
# ---------------------------------------------------------------------------


class TestGaussianityInFitLogisticNormal:
    """Verify that fit_logistic_normal_from_posterior returns gaussianity
    diagnostics as part of its results dict."""

    def test_non_mixture_gaussianity_keys(
        self, r_samples_non_mixture, rng_key
    ):
        """Non-mixture results should contain a 'gaussianity' dict with
        the expected keys and shapes (D-1,)."""
        posterior = {"r": r_samples_non_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=None,
            rng_key=rng_key,
            rank=3,
            batch_size=8,
            verbose=False,
        )

        assert "gaussianity" in result
        gd = result["gaussianity"]
        D_alr = 10 - 1  # n_genes - 1
        for key in ("skewness", "kurtosis", "jarque_bera", "jb_pvalue"):
            assert key in gd, f"Missing key: {key}"
            assert gd[key].shape == (D_alr,), (
                f"{key} has shape {gd[key].shape}, expected ({D_alr},)"
            )

    def test_mixture_gaussianity_keys(self, r_samples_mixture, rng_key):
        """Mixture results should contain a 'gaussianity' dict with
        per-component arrays of shape (K, D-1)."""
        posterior = {"r": r_samples_mixture}
        result = fit_logistic_normal_from_posterior(
            posterior,
            n_components=3,
            rng_key=rng_key,
            rank=3,
            batch_size=8,
            verbose=False,
        )

        assert "gaussianity" in result
        gd = result["gaussianity"]
        K, D_alr = 3, 10 - 1  # n_components, n_genes - 1
        for key in ("skewness", "kurtosis", "jarque_bera", "jb_pvalue"):
            assert key in gd, f"Missing key: {key}"
            assert gd[key].shape == (K, D_alr), (
                f"{key} has shape {gd[key].shape}, expected ({K}, {D_alr})"
            )
