"""Tests for PLN data initialization, numerical stability, and preset wiring.

Mirrors ``test_lnm_stability.py`` for the PLN model: verifies data-driven
initializers, PCA-based W init, encoder standardization, and the preset
builder PLN branch.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from scribe.core.pln_data_init import (
    empirical_log_mean_from_counts,
    pca_loadings_init,
    inject_pln_vae_data_init,
)
from scribe.models.config import ModelConfigBuilder
from scribe.stats.distributions import LowRankPoissonLogNormal


# ==============================================================================
# empirical_log_mean_from_counts
# ==============================================================================


class TestEmpiricalLogMean:
    """Tests for ``empirical_log_mean_from_counts``."""

    def test_output_shape(self):
        """Output has shape ``(G,)`` matching the number of genes."""
        counts = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
        result = empirical_log_mean_from_counts(counts)
        assert result.shape == (3,)

    def test_output_values(self):
        """Verify output = mean(log(counts + pseudocount)) per gene.

        Updated from the older ``log(mean(u) + c)`` formula to
        ``mean(log(u + c))`` so the bias matches the centering
        reference of :func:`pca_loadings_init`. The two formulas
        coincide only when ``Var[log u]`` is zero (i.e. the gene's
        counts are constant across cells).
        """
        # All four entries are equal across cells per gene, so
        # ``mean(log(u + 1)) == log(mean(u) + 1)`` and the formula
        # change is observationally identical here. The value
        # comparison is therefore unchanged but the *formula* is
        # documented to be ``mean(log(u + c))``.
        counts = jnp.array([[2, 4], [2, 4]], dtype=jnp.float32)
        expected = jnp.log(jnp.array([2.0, 4.0]) + 1.0)
        result = empirical_log_mean_from_counts(counts, pseudocount=1.0)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_uses_mean_of_log_not_log_of_mean(self):
        """Regression: helper computes ``mean(log(u + c))``, not ``log(mean(u) + c)``.

        On data with ``Var[log u] > 0`` (i.e. variable counts across
        cells per gene) Jensen's inequality gives
        ``mean(log(u + c)) < log(mean(u) + c)``. Here the gene is
        either ``1`` or ``9`` with equal frequency:

            mean(u + 1)         = 6   ->   log(6) ≈ 1.7918
            mean(log(u + 1))    = (log 2 + log 10) / 2 ≈ 1.4979

        We assert the helper returns the second value -- the one that
        aligns with the PCA centering -- not the first.
        """
        counts = jnp.array([[1.0], [9.0]], dtype=jnp.float32)
        result = empirical_log_mean_from_counts(counts, pseudocount=1.0)
        # Mean-of-log answer (correct).
        expected = (jnp.log(2.0) + jnp.log(10.0)) / 2.0
        # Log-of-mean answer (incorrect; what older code returned).
        wrong = jnp.log(6.0)
        assert jnp.allclose(result, expected, atol=1e-5)
        assert not jnp.allclose(result, wrong, atol=1e-3)

    def test_custom_pseudocount(self):
        """Different pseudocounts change the result."""
        counts = jnp.array([[0, 0], [0, 0]], dtype=jnp.float32)
        result_1 = empirical_log_mean_from_counts(counts, pseudocount=1.0)
        result_05 = empirical_log_mean_from_counts(counts, pseudocount=0.5)
        # With all-zero counts, mean(log(0 + c)) == log(c).
        assert jnp.allclose(result_1, jnp.log(jnp.array([1.0, 1.0])))
        assert jnp.allclose(result_05, jnp.log(jnp.array([0.5, 0.5])))
        assert not jnp.allclose(result_1, result_05)

    def test_finite_for_sparse_counts(self):
        """Even with many zeros, the result is finite."""
        counts = jnp.zeros((100, 50), dtype=jnp.float32)
        result = empirical_log_mean_from_counts(counts)
        assert jnp.all(jnp.isfinite(result))


# ==============================================================================
# pca_loadings_init
# ==============================================================================


class TestPCALoadingsInit:
    """Tests for ``pca_loadings_init``."""

    def test_output_shape(self):
        """PCA loadings have shape ``(G, k)``."""
        n_cells, n_genes, k = 50, 10, 3
        counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        result = pca_loadings_init(counts, latent_dim=k)
        assert result.shape == (n_genes, k)

    def test_g_dimensional_not_g_minus_1(self):
        """PLN loadings are G-dimensional, unlike LNM's (G-1)."""
        n_cells, n_genes, k = 50, 10, 3
        counts = random.poisson(random.PRNGKey(1), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        result = pca_loadings_init(counts, latent_dim=k)
        # First dimension must be G (not G-1).
        assert result.shape[0] == n_genes

    def test_latent_dim_exceeds_rank_raises(self):
        """When ``latent_dim > min(n_cells, G) - 1``, ``ValueError`` is raised."""
        n_cells, n_genes = 5, 3
        counts = random.poisson(random.PRNGKey(2), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        with pytest.raises(ValueError, match="latent_dim"):
            pca_loadings_init(counts, latent_dim=10)

    def test_finite_values(self):
        """PCA loadings are finite for typical count data."""
        n_cells, n_genes, k = 100, 20, 5
        counts = random.poisson(random.PRNGKey(3), 10.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        result = pca_loadings_init(counts, latent_dim=k)
        assert jnp.all(jnp.isfinite(result))

    def test_custom_pseudocount(self):
        """Different pseudocounts change the loadings."""
        n_cells, n_genes, k = 50, 10, 3
        counts = random.poisson(random.PRNGKey(4), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        w1 = pca_loadings_init(counts, latent_dim=k, pseudocount=1.0)
        w2 = pca_loadings_init(counts, latent_dim=k, pseudocount=0.1)
        # Should be different (though sign ambiguity makes exact comparison tricky).
        assert not jnp.allclose(jnp.abs(w1), jnp.abs(w2), atol=1e-3)

    def test_subsampling_triggers_for_large_n(self):
        """When n_cells > max_cells, subsampling produces valid loadings."""
        n_cells, n_genes, k = 200, 10, 3
        counts = random.poisson(random.PRNGKey(5), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        # Force subsampling by setting max_cells well below n_cells.
        result = pca_loadings_init(counts, latent_dim=k, max_cells=50)
        assert result.shape == (n_genes, k)
        assert jnp.all(jnp.isfinite(result))

    def test_subsampling_disabled(self):
        """``max_cells=0`` disables subsampling and uses all cells."""
        n_cells, n_genes, k = 100, 10, 3
        counts = random.poisson(random.PRNGKey(6), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        result = pca_loadings_init(counts, latent_dim=k, max_cells=0)
        assert result.shape == (n_genes, k)
        assert jnp.all(jnp.isfinite(result))

    def test_reproducible_with_same_seed(self):
        """Same ``random_state`` produces identical loadings."""
        n_cells, n_genes, k = 200, 10, 3
        counts = random.poisson(random.PRNGKey(7), 5.0, shape=(n_cells, n_genes))
        counts = counts.astype(jnp.float32)
        w1 = pca_loadings_init(counts, latent_dim=k, max_cells=50, random_state=42)
        w2 = pca_loadings_init(counts, latent_dim=k, max_cells=50, random_state=42)
        assert jnp.allclose(w1, w2)


# ==============================================================================
# inject_pln_vae_data_init
# ==============================================================================


class TestInjectPLNVAEDataInit:
    """Tests for ``inject_pln_vae_data_init``."""

    def _base_config(self):
        """Minimal PLN config with VAE."""
        return (
            ModelConfigBuilder()
            .for_model("pln")
            .with_parameterization("poisson_lognormal")
            .with_inference("vae")
            .with_vae(
                latent_dim=3,
                encoder_hidden_dims=[32],
                decoder_hidden_dims=[32],
            )
            .build()
        )

    def test_injects_log_mean_bias(self):
        """After injection, ``vae.empirical_log_mean_bias_init`` is populated."""
        config = self._base_config()
        n_cells, n_genes = 50, 10
        counts = random.poisson(
            random.PRNGKey(0), 5.0, shape=(n_cells, n_genes)
        ).astype(jnp.float32)

        updated = inject_pln_vae_data_init(config, counts, latent_dim=3)
        assert updated.vae.empirical_log_mean_bias_init is not None
        assert updated.vae.empirical_log_mean_bias_init.shape == (n_genes,)

    def test_injects_pca_loadings(self):
        """After injection, ``vae.pca_loadings_init`` is populated."""
        config = self._base_config()
        n_cells, n_genes = 50, 10
        counts = random.poisson(
            random.PRNGKey(1), 5.0, shape=(n_cells, n_genes)
        ).astype(jnp.float32)

        updated = inject_pln_vae_data_init(config, counts, latent_dim=3)
        assert updated.vae.pca_loadings_init is not None
        assert updated.vae.pca_loadings_init.shape == (n_genes, 3)

    def test_injects_encoder_standardization_when_enabled(self):
        """When ``vae.standardize=True``, standardization stats are injected."""
        config = self._base_config()
        # Ensure standardize is True.
        updated_vae = config.vae.model_copy(update={"standardize": True})
        config = config.model_copy(update={"vae": updated_vae})

        n_cells, n_genes = 50, 10
        counts = random.poisson(
            random.PRNGKey(2), 5.0, shape=(n_cells, n_genes)
        ).astype(jnp.float32)

        updated = inject_pln_vae_data_init(config, counts, latent_dim=3)
        assert updated.vae.standardize_mean is not None
        assert updated.vae.standardize_std is not None

    def test_original_config_unchanged(self):
        """Injection returns a new config, leaving the original unmodified."""
        config = self._base_config()
        n_cells, n_genes = 50, 10
        counts = random.poisson(
            random.PRNGKey(3), 5.0, shape=(n_cells, n_genes)
        ).astype(jnp.float32)

        _updated = inject_pln_vae_data_init(config, counts, latent_dim=3)
        # Original config must not have been mutated.
        assert config.vae.empirical_log_mean_bias_init is None
        assert config.vae.pca_loadings_init is None


# ==============================================================================
# LowRankPoissonLogNormal distribution
# ==============================================================================


class TestLowRankPoissonLogNormal:
    """Tests for the ``LowRankPoissonLogNormal`` distribution."""

    def test_sample_shape(self):
        """Sampled counts have the expected shape."""
        g, k = 5, 2
        mu = jnp.zeros(g)
        W = jnp.ones((g, k)) * 0.1
        d = jnp.ones(g) * 0.01
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        samples = pln.sample(random.PRNGKey(0), (100,))
        assert samples.shape == (100, g)

    def test_sample_non_negative_integer(self):
        """Poisson counts are non-negative integers."""
        g, k = 5, 2
        mu = jnp.zeros(g)
        W = jnp.ones((g, k)) * 0.1
        d = jnp.ones(g) * 0.01
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        samples = pln.sample(random.PRNGKey(1), (200,))
        assert jnp.all(samples >= 0)
        assert jnp.allclose(samples, jnp.floor(samples))

    def test_log_prob_raises(self):
        """Marginal log_prob is intractable and must raise."""
        g, k = 3, 2
        mu = jnp.zeros(g)
        W = jnp.ones((g, k)) * 0.1
        d = jnp.ones(g) * 0.01
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        with pytest.raises(NotImplementedError, match="closed form"):
            pln.log_prob(jnp.array([1, 2, 3]))

    def test_importance_log_prob_returns_finite_scalar(self):
        """``importance_log_prob`` runs end-to-end with a Gaussian guide.

        Mirrors the typical post-fit usage: feed the encoder posterior
        as ``q_dist`` and request an IWAE-style estimate of
        ``log p(u)``.  We use a generic ``MultivariateNormal`` here in
        place of the encoder posterior since it implements the same
        ``Distribution`` interface (``sample``, ``log_prob``).
        """
        import numpyro.distributions as dist

        g, k = 4, 2
        mu = jnp.zeros(g)
        W = jnp.ones((g, k)) * 0.1
        d = jnp.ones(g) * 0.05
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)

        # Toy single-cell observation.
        u = jnp.array([1, 0, 3, 2])
        q = dist.MultivariateNormal(loc=mu, covariance_matrix=jnp.eye(g))

        lp = pln.importance_log_prob(
            u, q, n_samples=128, rng_key=random.PRNGKey(0)
        )
        # Single-cell estimator returns a scalar.
        assert lp.shape == ()
        assert jnp.isfinite(lp)

    def test_importance_log_prob_requires_rng_key(self):
        """The method refuses an implicit default rng_key.

        Reproducibility for any number that goes into a paper or
        diagnostic should be explicit. Verifies the safety guard.
        """
        import numpyro.distributions as dist

        g, k = 3, 2
        pln = LowRankPoissonLogNormal(
            loc=jnp.zeros(g),
            cov_factor=jnp.ones((g, k)) * 0.1,
            cov_diag=jnp.ones(g) * 0.01,
        )
        q = dist.MultivariateNormal(
            loc=jnp.zeros(g), covariance_matrix=jnp.eye(g)
        )
        with pytest.raises(ValueError, match="rng_key"):
            pln.importance_log_prob(
                jnp.zeros(g, dtype=jnp.int32), q, n_samples=4
            )

    def test_mean_closed_form(self):
        """Mean uses the log-normal moment formula."""
        g = 4
        mu = jnp.array([1.0, 2.0, 0.5, 0.0])
        W = jnp.zeros((g, 2))
        d = jnp.array([0.1, 0.2, 0.5, 1.0])
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        expected = jnp.exp(mu + d / 2.0)
        assert jnp.allclose(pln.mean, expected, rtol=1e-5)

    def test_variance_closed_form(self):
        """Variance uses the law of total variance: E[lambda] + Var[lambda].

        For Poisson-LogNormal counts ``u | lambda ~ Poisson(lambda)``
        with ``log lambda ~ Normal(mu, sigma^2)``:

            Var[u] = E[Var[u|lambda]] + Var[E[u|lambda]]
                   = E[lambda] + Var[lambda]
                   = exp(mu + sigma^2/2)
                     + exp(2*mu + sigma^2) * (exp(sigma^2) - 1).

        The earlier formula ``exp(2 mu + sigma^2) * (exp(sigma^2) - 1)``
        was the variance of the *rate* ``lambda``, not of the count
        ``u``, and was off by ``+ E[lambda]`` everywhere. Verified
        against an empirical Monte-Carlo Var[u] in
        ``test_variance_matches_empirical``.
        """
        g = 3
        mu = jnp.array([1.0, 2.0, 0.0])
        W = jnp.zeros((g, 2))
        d = jnp.array([0.1, 0.2, 0.5])
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        sigma_sq = d
        mean_lambda = jnp.exp(mu + sigma_sq / 2.0)
        var_lambda = jnp.exp(2 * mu + sigma_sq) * (jnp.exp(sigma_sq) - 1.0)
        expected = mean_lambda + var_lambda
        assert jnp.allclose(pln.variance, expected, rtol=1e-5)

    def test_variance_matches_empirical(self):
        """Property matches an empirical Monte-Carlo Var[u].

        Independent verification that the closed-form formula is the
        marginal variance of *counts* (what users want), not of the
        latent rate. Uses 50k samples to keep MC noise small enough
        for a 5% tolerance.
        """
        import jax.random as random

        g = 3
        mu = jnp.array([0.5, 1.5, -0.5])
        W = jnp.zeros((g, 2))
        d = jnp.array([0.3, 0.5, 0.8])
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)

        samples = pln.sample(random.PRNGKey(0), sample_shape=(50_000,))
        empirical_var = jnp.var(samples, axis=0)
        # 5% relative tolerance accommodates ~50k-sample MC noise on
        # variance estimates with ``sigma_x`` up to ~0.9.
        assert jnp.allclose(pln.variance, empirical_var, rtol=0.05)

    def test_mean_increases_with_mu(self):
        """Higher ``mu`` leads to higher expected counts."""
        g, k = 5, 2
        W = jnp.ones((g, k)) * 0.1
        d = jnp.ones(g) * 0.01
        pln_low = LowRankPoissonLogNormal(
            loc=jnp.zeros(g), cov_factor=W, cov_diag=d
        )
        pln_high = LowRankPoissonLogNormal(
            loc=jnp.ones(g) * 3.0, cov_factor=W, cov_diag=d
        )
        assert jnp.all(pln_high.mean > pln_low.mean)

    def test_sample_finite_for_extreme_mu(self):
        """Clamping prevents overflow even with large mu."""
        g, k = 3, 2
        # mu = 25 is large but within clamp range [-30, 30].
        mu = jnp.full(g, 25.0)
        W = jnp.zeros((g, k))
        d = jnp.ones(g) * 0.01
        pln = LowRankPoissonLogNormal(loc=mu, cov_factor=W, cov_diag=d)
        samples = pln.sample(random.PRNGKey(0), (10,))
        # Poisson with very high rate can produce large counts,
        # but they should still be finite.
        assert jnp.all(jnp.isfinite(samples.astype(jnp.float32)))


# ==============================================================================
# Mixture rejection
# ==============================================================================


class TestPLNMixtureRejection:
    """PLN v1 does not support mixtures -- verify the guard."""

    def test_mixture_n_components_raises(self):
        """``n_components > 1`` must raise ``NotImplementedError``."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.parameterizations import PoissonLogNormalParameterization

        pln = PoissonLogNormalParameterization()
        with pytest.raises(NotImplementedError, match="mixture"):
            pln.build_param_specs(
                unconstrained=False,
                guide_families=GuideFamilyConfig(),
                n_components=2,
            )

    def test_n_components_1_ok(self):
        """``n_components=1`` (or default) must succeed."""
        from scribe.models.config import GuideFamilyConfig
        from scribe.models.parameterizations import PoissonLogNormalParameterization

        pln = PoissonLogNormalParameterization()
        specs = pln.build_param_specs(
            unconstrained=False,
            guide_families=GuideFamilyConfig(),
            n_components=1,
        )
        assert specs == []
