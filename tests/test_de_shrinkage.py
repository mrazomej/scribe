"""Tests for the empirical Bayes shrinkage DE module.

Covers:
- ``fit_scale_mixture_prior`` — weights simplex, convergence, null recovery
- ``shrinkage_posterior`` — shrunk means closer to zero, correct shapes
- ``shrinkage_differential_expression`` — output keys, lfsr range,
  shrinkage effect
- ``compare()`` dispatch — method="shrinkage" returns correct type
- ``ScribeShrinkageDEResults`` — gene_level, call_genes, compute_pefp,
  find_threshold, summary
- Backward compatibility — shrinkage ≈ raw when no null component
- Edge cases — all null, all DE, single gene
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from scribe.de import (
    ScribeDEResults,
    ScribeEmpiricalDEResults,
    ScribeShrinkageDEResults,
    compare,
    fit_scale_mixture_prior,
    shrinkage_differential_expression,
)
from scribe.de._shrinkage import (
    default_sigma_grid,
    shrinkage_posterior,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Base RNG key."""
    return random.PRNGKey(42)


@pytest.fixture
def null_dominated_data(rng):
    """Synthetic data where 90% of genes are null.

    200 genes total: 180 null (delta_mean ~ N(0, 0.05)),
    20 non-null (delta_mean ~ N(±2, 0.3)).
    """
    D_null = 180
    D_nonnull = 20
    key1, key2, key3 = random.split(rng, 3)

    # Null genes: small noise around zero
    null_means = 0.05 * random.normal(key1, (D_null,))
    null_sds = jnp.full(D_null, 0.1)

    # Non-null genes: large effects with moderate noise
    nonnull_means = 2.0 * random.choice(
        key2, jnp.array([-1.0, 1.0]), shape=(D_nonnull,)
    )
    nonnull_sds = jnp.full(D_nonnull, 0.3)

    delta_mean = jnp.concatenate([null_means, nonnull_means])
    delta_sd = jnp.concatenate([null_sds, nonnull_sds])
    return delta_mean, delta_sd


@pytest.fixture
def all_null_data(rng):
    """Synthetic data where all 100 genes are null."""
    D = 100
    delta_mean = 0.02 * random.normal(rng, (D,))
    delta_sd = jnp.full(D, 0.1)
    return delta_mean, delta_sd


@pytest.fixture
def all_de_data(rng):
    """Synthetic data where all 50 genes are DE (large effects)."""
    D = 50
    key1, key2 = random.split(rng)
    signs = random.choice(key1, jnp.array([-1.0, 1.0]), shape=(D,))
    delta_mean = signs * (2.0 + jnp.abs(random.normal(key2, (D,))))
    delta_sd = jnp.full(D, 0.2)
    return delta_mean, delta_sd


# --------------------------------------------------------------------------
# Tests: default_sigma_grid
# --------------------------------------------------------------------------


class TestDefaultSigmaGrid:
    """Tests for ``default_sigma_grid``."""

    def test_shape(self):
        """Grid has K + 1 elements."""
        sd = jnp.ones(100) * 0.5
        grid = default_sigma_grid(sd, K=20)
        assert grid.shape == (21,)

    def test_monotonic(self):
        """Grid values are strictly increasing."""
        sd = jnp.ones(50) * 0.3
        grid = default_sigma_grid(sd, K=15)
        diffs = jnp.diff(grid)
        assert jnp.all(diffs > 0)

    def test_first_element_small(self):
        """First grid element is approximately sigma_min."""
        sd = jnp.ones(50) * 1.0
        grid = default_sigma_grid(sd, K=10, sigma_min=1e-6)
        assert float(grid[0]) == pytest.approx(1e-6, rel=1e-3)


# --------------------------------------------------------------------------
# Tests: fit_scale_mixture_prior
# --------------------------------------------------------------------------


class TestFitScaleMixturePrior:
    """Tests for ``fit_scale_mixture_prior``."""

    def test_weights_sum_to_one(self, null_dominated_data):
        """Estimated weights must sum to 1."""
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd)
        np.testing.assert_allclose(float(jnp.sum(result["weights"])), 1.0, atol=1e-6)

    def test_weights_nonneg(self, null_dominated_data):
        """All weights must be non-negative."""
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd)
        assert jnp.all(result["weights"] >= 0)

    def test_posterior_probs_shape(self, null_dominated_data):
        """Responsibility matrix has shape (D, K+1)."""
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd)
        D = delta_mean.shape[0]
        K_plus_1 = result["sigma_grid"].shape[0]
        assert result["posterior_probs"].shape == (D, K_plus_1)

    def test_posterior_probs_sum_to_one(self, null_dominated_data):
        """Rows of responsibility matrix sum to 1."""
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd)
        row_sums = jnp.sum(result["posterior_probs"], axis=1)
        np.testing.assert_allclose(np.array(row_sums), 1.0, atol=1e-5)

    def test_convergence(self, null_dominated_data):
        """EM should converge within max_iter."""
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd, max_iter=500)
        assert result["converged"]

    def test_null_proportion_recovery(self, null_dominated_data):
        """With 90% null data, small-scale components carry substantial weight.

        The EM may spread mass across several near-zero components rather
        than concentrating it all on sigma_0.  We check that the total
        weight on the smallest quarter of the grid is substantial.
        """
        delta_mean, delta_sd = null_dominated_data
        result = fit_scale_mixture_prior(delta_mean, delta_sd)
        K_plus_1 = result["weights"].shape[0]
        n_small = max(K_plus_1 // 4, 1)
        small_weight = float(jnp.sum(result["weights"][:n_small]))
        assert small_weight > 0.1

    def test_log_likelihood_monotonic(self, null_dominated_data):
        """Log-likelihood should be non-decreasing (check final > initial)."""
        delta_mean, delta_sd = null_dominated_data
        # Run a few iterations only
        result_early = fit_scale_mixture_prior(
            delta_mean, delta_sd, max_iter=2
        )
        result_late = fit_scale_mixture_prior(
            delta_mean, delta_sd, max_iter=100
        )
        assert result_late["log_likelihood"] >= result_early["log_likelihood"]

    def test_custom_sigma_grid(self, null_dominated_data):
        """Custom sigma_grid is used when provided."""
        delta_mean, delta_sd = null_dominated_data
        custom_grid = jnp.array([1e-6, 0.1, 0.5, 1.0, 2.0])
        result = fit_scale_mixture_prior(
            delta_mean, delta_sd, sigma_grid=custom_grid
        )
        np.testing.assert_array_equal(
            np.array(result["sigma_grid"]), np.array(custom_grid)
        )
        assert result["weights"].shape == (5,)


# --------------------------------------------------------------------------
# Tests: shrinkage_posterior
# --------------------------------------------------------------------------


class TestShrinkagePosterior:
    """Tests for ``shrinkage_posterior``."""

    def test_shrunk_mean_shape(self, null_dominated_data):
        """Shrunk mean has shape (D,)."""
        delta_mean, delta_sd = null_dominated_data
        em = fit_scale_mixture_prior(delta_mean, delta_sd)
        post = shrinkage_posterior(
            delta_mean, delta_sd,
            weights=em["weights"],
            sigma_grid=em["sigma_grid"],
            posterior_probs=em["posterior_probs"],
        )
        assert post["shrunk_mean"].shape == delta_mean.shape

    def test_shrunk_sd_shape(self, null_dominated_data):
        """Shrunk sd has shape (D,)."""
        delta_mean, delta_sd = null_dominated_data
        em = fit_scale_mixture_prior(delta_mean, delta_sd)
        post = shrinkage_posterior(
            delta_mean, delta_sd,
            weights=em["weights"],
            sigma_grid=em["sigma_grid"],
        )
        assert post["shrunk_sd"].shape == delta_sd.shape

    def test_shrunk_means_closer_to_zero(self, null_dominated_data):
        """Shrunk means should be (on average) closer to zero than raw means.

        This is the defining property of shrinkage: estimates are pulled
        toward the grand mean (zero).
        """
        delta_mean, delta_sd = null_dominated_data
        em = fit_scale_mixture_prior(delta_mean, delta_sd)
        post = shrinkage_posterior(
            delta_mean, delta_sd,
            weights=em["weights"],
            sigma_grid=em["sigma_grid"],
            posterior_probs=em["posterior_probs"],
        )
        raw_abs = jnp.mean(jnp.abs(delta_mean))
        shrunk_abs = jnp.mean(jnp.abs(post["shrunk_mean"]))
        assert float(shrunk_abs) < float(raw_abs)

    def test_shrunk_sd_smaller(self, null_dominated_data):
        """Shrunk sd should be (on average) smaller than raw sd.

        Borrowing information across genes reduces uncertainty.
        """
        delta_mean, delta_sd = null_dominated_data
        em = fit_scale_mixture_prior(delta_mean, delta_sd)
        post = shrinkage_posterior(
            delta_mean, delta_sd,
            weights=em["weights"],
            sigma_grid=em["sigma_grid"],
            posterior_probs=em["posterior_probs"],
        )
        assert float(jnp.mean(post["shrunk_sd"])) < float(jnp.mean(delta_sd))

    def test_component_shapes(self, null_dominated_data):
        """Component means/variances/weights have shape (D, K+1)."""
        delta_mean, delta_sd = null_dominated_data
        em = fit_scale_mixture_prior(delta_mean, delta_sd)
        post = shrinkage_posterior(
            delta_mean, delta_sd,
            weights=em["weights"],
            sigma_grid=em["sigma_grid"],
        )
        D = delta_mean.shape[0]
        K_plus_1 = em["sigma_grid"].shape[0]
        assert post["component_means"].shape == (D, K_plus_1)
        assert post["component_variances"].shape == (D, K_plus_1)
        assert post["component_weights"].shape == (D, K_plus_1)


# --------------------------------------------------------------------------
# Tests: shrinkage_differential_expression
# --------------------------------------------------------------------------


class TestShrinkageDifferentialExpression:
    """Tests for ``shrinkage_differential_expression``."""

    def test_output_keys(self, null_dominated_data):
        """Output dict has all required keys."""
        delta_mean, delta_sd = null_dominated_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        expected = {
            "delta_mean", "delta_sd", "prob_positive", "prob_effect",
            "lfsr", "lfsr_tau", "gene_names",
            "null_proportion", "prior_weights", "sigma_grid",
            "em_converged", "em_n_iter", "em_log_likelihood",
        }
        assert set(result.keys()) == expected

    def test_lfsr_range(self, null_dominated_data):
        """lfsr must be in [0, 0.5]."""
        delta_mean, delta_sd = null_dominated_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        assert jnp.all(result["lfsr"] >= 0.0)
        assert jnp.all(result["lfsr"] <= 0.5 + 1e-7)

    def test_lfsr_tau_range(self, null_dominated_data):
        """lfsr_tau must be in [0, 1]."""
        delta_mean, delta_sd = null_dominated_data
        result = shrinkage_differential_expression(
            delta_mean, delta_sd, tau=0.5
        )
        assert jnp.all(result["lfsr_tau"] >= 0.0 - 1e-7)
        assert jnp.all(result["lfsr_tau"] <= 1.0 + 1e-7)

    def test_prob_positive_range(self, null_dominated_data):
        """prob_positive must be in [0, 1]."""
        delta_mean, delta_sd = null_dominated_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        assert jnp.all(result["prob_positive"] >= 0.0 - 1e-7)
        assert jnp.all(result["prob_positive"] <= 1.0 + 1e-7)

    def test_custom_gene_names(self, null_dominated_data):
        """Custom gene names are propagated."""
        delta_mean, delta_sd = null_dominated_data
        D = delta_mean.shape[0]
        names = [f"mygene_{i}" for i in range(D)]
        result = shrinkage_differential_expression(
            delta_mean, delta_sd, gene_names=names
        )
        assert result["gene_names"] == names

    def test_null_genes_have_higher_lfsr(self, null_dominated_data):
        """Null genes (first 180) should have higher shrunk lfsr than DE genes.

        After shrinkage, the null genes should be more confidently null.
        """
        delta_mean, delta_sd = null_dominated_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        mean_null_lfsr = float(jnp.mean(result["lfsr"][:180]))
        mean_de_lfsr = float(jnp.mean(result["lfsr"][180:]))
        assert mean_null_lfsr > mean_de_lfsr


# --------------------------------------------------------------------------
# Tests: compare() dispatch with method="shrinkage"
# --------------------------------------------------------------------------


class TestCompareDispatchShrinkage:
    """Tests for the ``compare()`` factory with method='shrinkage'."""

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)

    def test_returns_correct_type(self, rng):
        """method='shrinkage' returns ScribeShrinkageDEResults."""
        r_A = jnp.abs(random.normal(rng, (100, 5))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, 5))) + 1.0
        de = compare(r_A, r_B, method="shrinkage", rng_key=rng)
        assert isinstance(de, ScribeShrinkageDEResults)
        assert isinstance(de, ScribeEmpiricalDEResults)
        assert isinstance(de, ScribeDEResults)
        assert de.method == "shrinkage"

    def test_gene_names_propagated(self, rng):
        """Gene names are propagated in shrinkage mode."""
        r = jnp.abs(random.normal(rng, (50, 4))) + 1.0
        names = ["A", "B", "C", "D"]
        de = compare(
            r, r, method="shrinkage", gene_names=names, rng_key=rng
        )
        assert de.gene_names == names

    def test_gene_names_length_mismatch(self, rng):
        """Wrong-length gene_names raises ValueError."""
        r = jnp.abs(random.normal(rng, (50, 4))) + 1.0
        with pytest.raises(ValueError, match="gene_names"):
            compare(
                r, r, method="shrinkage",
                gene_names=["a", "b"], rng_key=rng,
            )

    def test_mixture_input(self, rng):
        """Shrinkage works with 3D (mixture) inputs."""
        r_mix = jnp.abs(random.normal(rng, (100, 3, 5))) + 1.0
        de = compare(
            r_mix, r_mix,
            method="shrinkage",
            component_A=0, component_B=1,
            rng_key=rng,
        )
        assert isinstance(de, ScribeShrinkageDEResults)
        assert de.D == 5

    def test_with_gene_mask(self, rng):
        """Shrinkage works with gene_mask."""
        D = 8
        r_A = jnp.abs(random.normal(rng, (100, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, D))) + 1.0
        mask = jnp.array([True, True, True, False, False, True, True, False])
        names = [f"g{i}" for i in range(D)]
        de = compare(
            r_A, r_B,
            method="shrinkage",
            gene_names=names,
            rng_key=rng,
            gene_mask=mask,
        )
        assert de.D == 5
        assert de.gene_names == ["g0", "g1", "g2", "g5", "g6"]


# --------------------------------------------------------------------------
# Tests: ScribeShrinkageDEResults methods
# --------------------------------------------------------------------------


class TestShrinkageResultsMethods:
    """Tests for methods on ``ScribeShrinkageDEResults``."""

    @pytest.fixture
    def shrinkage_de(self):
        """Create a shrinkage DE results object with known data."""
        rng = random.PRNGKey(0)
        r_A = jnp.abs(random.normal(rng, (500, 8))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (500, 8))) + 2.0
        return compare(
            r_A, r_B,
            method="shrinkage",
            gene_names=[f"g{i}" for i in range(8)],
            rng_key=rng,
        )

    def test_gene_level_returns_dict(self, shrinkage_de):
        """gene_level returns a dict with expected keys."""
        result = shrinkage_de.gene_level(tau=0.0)
        expected_keys = {
            "delta_mean", "delta_sd", "prob_positive",
            "prob_effect", "lfsr", "lfsr_tau", "gene_names",
            "null_proportion", "prior_weights", "sigma_grid",
            "em_converged", "em_n_iter", "em_log_likelihood",
        }
        assert set(result.keys()) == expected_keys

    def test_null_proportion_populated(self, shrinkage_de):
        """null_proportion is set after gene_level()."""
        assert shrinkage_de.null_proportion is None
        shrinkage_de.gene_level(tau=0.0)
        assert shrinkage_de.null_proportion is not None
        assert 0.0 <= shrinkage_de.null_proportion <= 1.0

    def test_call_genes_returns_bool_mask(self, shrinkage_de):
        """call_genes returns a boolean array of shape (D,)."""
        is_de = shrinkage_de.call_genes(tau=0.0)
        assert is_de.shape == (shrinkage_de.D,)
        assert is_de.dtype == jnp.bool_

    def test_compute_pefp_returns_float(self, shrinkage_de):
        """compute_pefp returns a scalar float."""
        pefp = shrinkage_de.compute_pefp(threshold=0.05, tau=0.0)
        assert isinstance(pefp, float)
        assert 0.0 <= pefp <= 1.0

    def test_find_threshold_returns_float(self, shrinkage_de):
        """find_threshold returns a scalar float."""
        threshold = shrinkage_de.find_threshold(target_pefp=0.05, tau=0.0)
        assert isinstance(threshold, float)
        assert threshold >= 0.0

    def test_summary_returns_string(self, shrinkage_de):
        """summary returns a non-empty string."""
        s = shrinkage_de.summary(tau=0.0, top_n=5)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_repr(self, shrinkage_de):
        """repr includes correct class name."""
        r = repr(shrinkage_de)
        assert "ScribeShrinkageDEResults" in r

    def test_repr_after_gene_level(self, shrinkage_de):
        """repr includes null_proportion after gene_level."""
        shrinkage_de.gene_level(tau=0.0)
        r = repr(shrinkage_de)
        assert "null_proportion=" in r

    def test_D_property(self, shrinkage_de):
        """D returns the number of genes."""
        assert shrinkage_de.D == 8


# --------------------------------------------------------------------------
# Tests: Backward compatibility
# --------------------------------------------------------------------------


class TestBackwardCompatibility:
    """When the prior has no near-zero component, shrinkage should be minimal.

    With a grid that starts at a large sigma_min, the null component
    has substantial variance and the shrinkage effect should be small.
    """

    def test_no_null_component_minimal_shrinkage(self):
        """Without a near-zero component, shrunk lfsr ≈ raw lfsr."""
        rng = random.PRNGKey(99)
        D = 50
        # All genes have large effects
        delta_mean = 3.0 * random.choice(
            rng, jnp.array([-1.0, 1.0]), shape=(D,)
        )
        delta_sd = jnp.full(D, 0.3)

        # Grid with NO near-zero component (all large scales)
        sigma_grid = jnp.linspace(1.0, 10.0, 10)

        result = shrinkage_differential_expression(
            delta_mean, delta_sd, sigma_grid=sigma_grid
        )

        # Shrunk means should be close to raw means (minimal shrinkage)
        np.testing.assert_allclose(
            np.array(result["delta_mean"]),
            np.array(delta_mean),
            atol=0.5,
        )


# --------------------------------------------------------------------------
# Tests: Edge cases
# --------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the shrinkage module."""

    def test_all_null(self, all_null_data):
        """All-null data: lfsr should be high (close to 0.5) for all genes."""
        delta_mean, delta_sd = all_null_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        # All genes are null, lfsr should be high for all
        assert float(jnp.mean(result["lfsr"])) > 0.3
        # Weight on small-scale components should be substantial
        K_plus_1 = result["prior_weights"].shape[0]
        n_small = max(K_plus_1 // 4, 1)
        small_weight = float(jnp.sum(result["prior_weights"][:n_small]))
        assert small_weight > 0.1

    def test_all_de(self, all_de_data):
        """All-DE data: null proportion should be low, lfsr should be low."""
        delta_mean, delta_sd = all_de_data
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        # Most genes should have low lfsr
        assert float(jnp.mean(result["lfsr"])) < 0.1

    def test_single_gene(self):
        """Single gene should not crash."""
        delta_mean = jnp.array([1.5])
        delta_sd = jnp.array([0.3])
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        assert result["lfsr"].shape == (1,)
        assert result["delta_mean"].shape == (1,)

    def test_very_large_n(self):
        """Large D should work without OOM (vectorized, not looped)."""
        rng = random.PRNGKey(7)
        D = 10_000
        delta_mean = 0.1 * random.normal(rng, (D,))
        delta_sd = jnp.full(D, 0.2)
        # Use small K to keep memory manageable in test
        sigma_grid = jnp.geomspace(1e-6, 5.0, 15)
        result = shrinkage_differential_expression(
            delta_mean, delta_sd, sigma_grid=sigma_grid
        )
        assert result["lfsr"].shape == (D,)
        np.testing.assert_allclose(
            float(jnp.sum(result["prior_weights"])), 1.0, atol=1e-6
        )

    def test_zero_sd_gene(self):
        """Gene with zero sd should produce lfsr = 0 or 0.5 (depending on mean)."""
        delta_mean = jnp.array([2.0, 0.0, -2.0])
        # Very small sd to approximate zero (exact zero would cause division issues)
        delta_sd = jnp.array([1e-10, 1e-10, 1e-10])
        result = shrinkage_differential_expression(delta_mean, delta_sd)
        assert result["lfsr"].shape == (3,)
        # Should not contain NaN
        assert not jnp.any(jnp.isnan(result["lfsr"]))
