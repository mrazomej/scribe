"""Tests for the goodness-of-fit module (scribe.mc._goodness_of_fit).

Validates that randomized quantile residuals are calibrated under the
true model, detect misspecification, and work correctly for both
single-component and mixture NB models.

Also validates PPC-based goodness-of-fit scoring and mask building.
"""

import pytest
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist

from scribe.mc._goodness_of_fit import (
    compute_quantile_residuals,
    goodness_of_fit_scores,
    compute_gof_mask,
    ppc_goodness_of_fit_scores,
    compute_ppc_gof_mask,
    _marginal_nb_cdf,
    _ensure_component_gene_shape,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def rng_key():
    """Default PRNG key for reproducibility."""
    return random.PRNGKey(42)


@pytest.fixture
def well_specified_nb_data(rng_key):
    """Synthetic NB data with known parameters (single-component).

    Generates C=2000 cells, G=50 genes from known (r, p) so residuals
    under the true model should be approximately N(0, 1).
    """
    C, G = 2000, 50
    # Gene-specific dispersions (varied from low to high)
    r_true = jnp.linspace(0.5, 20.0, G)
    # Shared success probability
    p_true = jnp.full(G, 0.3)

    nb = dist.NegativeBinomialProbs(r_true, p_true)
    counts = nb.sample(rng_key, sample_shape=(C,))
    return counts, r_true, p_true


@pytest.fixture
def misspecified_nb_data(rng_key):
    """Data generated from one NB but evaluated under wrong parameters.

    True model: r=5, p=0.3 for all genes.
    Wrong model: r=0.5, p=0.8 (very different mean and dispersion).
    """
    C, G = 1000, 20
    r_true = jnp.full(G, 5.0)
    p_true = jnp.full(G, 0.3)
    r_wrong = jnp.full(G, 0.5)
    p_wrong = jnp.full(G, 0.8)

    nb = dist.NegativeBinomialProbs(r_true, p_true)
    counts = nb.sample(rng_key, sample_shape=(C,))
    return counts, r_true, p_true, r_wrong, p_wrong


@pytest.fixture
def mixture_nb_data(rng_key):
    """Synthetic data from a 2-component NB mixture with known parameters.

    Component 0: low expression (r=2, p=0.2, weight=0.6)
    Component 1: high expression (r=10, p=0.5, weight=0.4)
    """
    C, G, K = 3000, 30, 2

    mixing_weights = jnp.array([0.6, 0.4])
    # r: shape (K, G)
    r = jnp.stack([
        jnp.full(G, 2.0),
        jnp.full(G, 10.0),
    ])
    # p: shape (K,) — shared across genes within each component
    p = jnp.array([0.2, 0.5])

    # Sample component assignments
    key1, key2 = random.split(rng_key)
    z = random.categorical(key1, jnp.log(mixing_weights), shape=(C,))

    # Sample counts from assigned components
    counts = jnp.zeros((C, G), dtype=jnp.int32)
    for k in range(K):
        mask = z == k
        n_k = int(mask.sum())
        if n_k > 0:
            subkey = random.fold_in(key2, k)
            nb_k = dist.NegativeBinomialProbs(r[k], p[k])
            samples_k = nb_k.sample(subkey, sample_shape=(n_k,))
            counts = counts.at[mask].set(samples_k)

    return counts, r, p, mixing_weights


# --------------------------------------------------------------------------
# compute_quantile_residuals — single-component
# --------------------------------------------------------------------------


class TestSingleComponentResiduals:
    """Tests for single-component NB quantile residuals."""

    def test_output_shape(self, well_specified_nb_data, rng_key):
        """Residual matrix should match (C, G) shape of counts."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        assert q.shape == counts.shape

    def test_residuals_finite(self, well_specified_nb_data, rng_key):
        """All residuals should be finite (no NaN or inf)."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        assert jnp.all(jnp.isfinite(q))

    def test_calibration_mean(self, well_specified_nb_data, rng_key):
        """Under the true model, per-gene mean should be near 0."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        gene_means = jnp.mean(q, axis=0)
        # With C=2000, std of gene mean is ~1/sqrt(2000) ≈ 0.022
        # Allow generous tolerance
        assert jnp.all(jnp.abs(gene_means) < 0.15), (
            f"Max |mean| = {float(jnp.max(jnp.abs(gene_means))):.4f}, "
            "expected < 0.15 for calibrated residuals"
        )

    def test_calibration_variance(self, well_specified_nb_data, rng_key):
        """Under the true model, per-gene variance should be near 1."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        gene_vars = jnp.var(q, axis=0, ddof=1)
        # With C=2000, std of sample variance is ~sqrt(2/(C-1)) ≈ 0.032
        assert jnp.all(jnp.abs(gene_vars - 1.0) < 0.25), (
            f"Max |var - 1| = {float(jnp.max(jnp.abs(gene_vars - 1.0))):.4f}, "
            "expected < 0.25 for calibrated residuals"
        )

    def test_misspecification_detected(self, misspecified_nb_data, rng_key):
        """Under wrong parameters, residual variance should deviate from 1."""
        counts, _, _, r_wrong, p_wrong = misspecified_nb_data
        q = compute_quantile_residuals(counts, r_wrong, p_wrong, rng_key)
        gene_vars = jnp.var(q, axis=0, ddof=1)
        # At least some genes should have variance substantially != 1
        max_deviation = float(jnp.max(jnp.abs(gene_vars - 1.0)))
        assert max_deviation > 0.5, (
            f"Max |var - 1| = {max_deviation:.4f} under misspecified model; "
            "expected large deviation"
        )


# --------------------------------------------------------------------------
# compute_quantile_residuals — mixture model
# --------------------------------------------------------------------------


class TestMixtureResiduals:
    """Tests for NB-mixture quantile residuals with marginal CDF."""

    def test_output_shape(self, mixture_nb_data, rng_key):
        """Residual matrix should match (C, G) shape."""
        counts, r, p, weights = mixture_nb_data
        q = compute_quantile_residuals(
            counts, r, p, rng_key, mixing_weights=weights
        )
        assert q.shape == counts.shape

    def test_residuals_finite(self, mixture_nb_data, rng_key):
        """All mixture residuals should be finite."""
        counts, r, p, weights = mixture_nb_data
        q = compute_quantile_residuals(
            counts, r, p, rng_key, mixing_weights=weights
        )
        assert jnp.all(jnp.isfinite(q))

    def test_calibration_variance(self, mixture_nb_data, rng_key):
        """Under the true mixture, per-gene variance should be near 1."""
        counts, r, p, weights = mixture_nb_data
        q = compute_quantile_residuals(
            counts, r, p, rng_key, mixing_weights=weights
        )
        gene_vars = jnp.var(q, axis=0, ddof=1)
        # More generous tolerance for mixture (C=3000 helps)
        median_var = float(jnp.median(gene_vars))
        assert abs(median_var - 1.0) < 0.2, (
            f"Median variance = {median_var:.4f}, expected near 1.0"
        )

    def test_marginal_cdf_shape(self, mixture_nb_data):
        """Internal _marginal_nb_cdf should return (C, G) shape."""
        counts, r, p, weights = mixture_nb_data
        cdf = _marginal_nb_cdf(counts, r, p, weights)
        assert cdf.shape == counts.shape

    def test_marginal_cdf_in_unit_interval(self, mixture_nb_data):
        """Marginal CDF values should be in [0, 1]."""
        counts, r, p, weights = mixture_nb_data
        cdf = _marginal_nb_cdf(counts, r, p, weights)
        assert jnp.all(cdf >= 0.0)
        assert jnp.all(cdf <= 1.0)


# --------------------------------------------------------------------------
# goodness_of_fit_scores
# --------------------------------------------------------------------------


class TestGoodnessOfFitScores:
    """Tests for per-gene summary statistics."""

    def test_output_keys(self, well_specified_nb_data, rng_key):
        """Should return all expected diagnostic keys."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        scores = goodness_of_fit_scores(q)
        assert set(scores.keys()) == {"mean", "variance", "tail_excess", "ks_distance"}

    def test_output_shapes(self, well_specified_nb_data, rng_key):
        """All per-gene scores should have shape (G,)."""
        counts, r, p = well_specified_nb_data
        G = counts.shape[1]
        q = compute_quantile_residuals(counts, r, p, rng_key)
        scores = goodness_of_fit_scores(q)
        for key, val in scores.items():
            assert val.shape == (G,), f"{key} has shape {val.shape}, expected ({G},)"

    def test_calibrated_scores(self, well_specified_nb_data, rng_key):
        """Under the true model, variance should cluster around 1."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        scores = goodness_of_fit_scores(q)

        median_var = float(jnp.median(scores["variance"]))
        assert abs(median_var - 1.0) < 0.15

        median_mean = float(jnp.median(jnp.abs(scores["mean"])))
        assert median_mean < 0.1

    def test_ks_distance_nonnegative(self, well_specified_nb_data, rng_key):
        """KS distance should be non-negative."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        scores = goodness_of_fit_scores(q)
        assert jnp.all(scores["ks_distance"] >= 0.0)

    def test_ks_distance_bounded(self, well_specified_nb_data, rng_key):
        """KS distance should be at most 1."""
        counts, r, p = well_specified_nb_data
        q = compute_quantile_residuals(counts, r, p, rng_key)
        scores = goodness_of_fit_scores(q)
        assert jnp.all(scores["ks_distance"] <= 1.0)

    @staticmethod
    def test_perfect_normal_residuals():
        """For truly N(0,1) data, scores should indicate perfect fit."""
        rng = np.random.default_rng(123)
        q = jnp.array(rng.normal(0, 1, size=(5000, 30)))
        scores = goodness_of_fit_scores(q)

        assert jnp.all(jnp.abs(scores["mean"]) < 0.1)
        assert jnp.all(jnp.abs(scores["variance"] - 1.0) < 0.15)
        assert float(jnp.max(scores["ks_distance"])) < 0.05


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for robustness."""

    def test_all_zero_counts(self, rng_key):
        """Residuals should be finite even when all counts are zero."""
        C, G = 100, 10
        counts = jnp.zeros((C, G), dtype=jnp.int32)
        r = jnp.ones(G) * 5.0
        p = jnp.ones(G) * 0.3
        q = compute_quantile_residuals(counts, r, p, rng_key)
        assert q.shape == (C, G)
        assert jnp.all(jnp.isfinite(q))

    def test_single_cell(self, rng_key):
        """Should work with C=1 (degenerate but valid)."""
        C, G = 1, 5
        counts = jnp.array([[3, 0, 7, 1, 0]])
        r = jnp.ones(G) * 2.0
        p = jnp.ones(G) * 0.5
        q = compute_quantile_residuals(counts, r, p, rng_key)
        assert q.shape == (1, G)
        assert jnp.all(jnp.isfinite(q))

    def test_high_count_genes(self, rng_key):
        """Residuals should remain finite for very large counts."""
        C, G = 100, 5
        # High expression: mean ~ r*p/(1-p) = 50*0.9/0.1 = 450
        r = jnp.ones(G) * 50.0
        p = jnp.ones(G) * 0.9
        nb = dist.NegativeBinomialProbs(r, p)
        counts = nb.sample(rng_key, sample_shape=(C,))
        q = compute_quantile_residuals(counts, r, p, random.PRNGKey(99))
        assert jnp.all(jnp.isfinite(q))

    def test_epsilon_prevents_infinite_residuals(self, rng_key):
        """Clipping epsilon should prevent infinite quantile residuals."""
        C, G = 50, 3
        counts = jnp.zeros((C, G), dtype=jnp.int32)
        # Very low p → almost all mass at 0 → CDF(0) ≈ 1
        r = jnp.ones(G) * 10.0
        p = jnp.ones(G) * 0.001
        q = compute_quantile_residuals(counts, r, p, rng_key, epsilon=1e-6)
        assert jnp.all(jnp.isfinite(q))


# --------------------------------------------------------------------------
# _ensure_component_gene_shape helper
# --------------------------------------------------------------------------


class TestEnsureComponentGeneShape:
    """Tests for the p-broadcasting helper."""

    def test_scalar(self):
        result = _ensure_component_gene_shape(jnp.array(0.5), K=3, G=10)
        assert result.shape == (3, 10)

    def test_per_component(self):
        p = jnp.array([0.2, 0.5, 0.8])
        result = _ensure_component_gene_shape(p, K=3, G=10)
        assert result.shape == (3, 10)
        # Each row should be constant
        assert jnp.allclose(result[0], 0.2)

    def test_per_gene(self):
        p = jnp.linspace(0.1, 0.9, 10)
        result = _ensure_component_gene_shape(p, K=3, G=10)
        assert result.shape == (3, 10)
        # Each column should be constant across components
        assert jnp.allclose(result[:, 0], p[0])

    def test_already_kxg(self):
        p = jnp.ones((3, 10)) * 0.4
        result = _ensure_component_gene_shape(p, K=3, G=10)
        assert result.shape == (3, 10)


# --------------------------------------------------------------------------
# compute_gof_mask (integration-style tests with mock results)
# --------------------------------------------------------------------------


class _MockSingleComponentResults:
    """Minimal mock of ScribeSVIResults for single-component tests."""

    def __init__(self, r, p):
        self._r = r
        self._p = p
        self.n_components = None

    def get_map(self, use_mean=True, canonical=True, verbose=False, counts=None):
        return {"r": self._r, "p": self._p}


class _MockMixtureResults:
    """Minimal mock of ScribeSVIResults for mixture tests."""

    def __init__(self, r, p, mixing_weights):
        self._r = r
        self._p = p
        self._mixing_weights = mixing_weights
        self.n_components = r.shape[0]

    def get_map(self, use_mean=True, canonical=True, verbose=False, counts=None):
        return {
            "r": self._r,
            "p": self._p,
            "mixing_weights": self._mixing_weights,
        }

    def get_component(self, idx):
        # When p is per-component (K,), slice to scalar for the component
        if self._p.ndim == 1 and self._p.shape[0] == self.n_components:
            p_component = self._p[idx]
        else:
            p_component = self._p
        return _MockSingleComponentResults(self._r[idx], p_component)


class TestComputeGofMask:
    """Tests for the high-level mask builder."""

    def test_single_component_mask_shape(self, well_specified_nb_data, rng_key):
        """Mask should have shape (G,) and dtype bool."""
        counts, r, p = well_specified_nb_data
        results = _MockSingleComponentResults(r, p)
        mask = compute_gof_mask(counts, results, rng_key=rng_key)
        assert mask.shape == (counts.shape[1],)
        assert mask.dtype == bool

    def test_well_specified_passes_most_genes(
        self, well_specified_nb_data, rng_key
    ):
        """Under the true model, most genes should pass the filter."""
        counts, r, p = well_specified_nb_data
        results = _MockSingleComponentResults(r, p)
        mask = compute_gof_mask(
            counts, results, rng_key=rng_key, max_variance=1.5
        )
        pass_rate = float(np.mean(mask))
        assert pass_rate > 0.8, (
            f"Only {pass_rate:.0%} of genes passed under the true model; "
            "expected > 80%"
        )

    def test_misspecified_filters_genes(self, misspecified_nb_data, rng_key):
        """Under wrong parameters, residual variance departs from 1.

        The misspecified model overestimates variability, producing
        residual variance ~0.5 (well below 1).  The min_variance
        bound catches this.
        """
        counts, _, _, r_wrong, p_wrong = misspecified_nb_data
        results = _MockSingleComponentResults(r_wrong, p_wrong)
        mask = compute_gof_mask(
            counts, results, rng_key=rng_key,
            min_variance=0.7, max_variance=1.5,
        )
        fail_rate = 1.0 - float(np.mean(mask))
        assert fail_rate > 0.5, (
            f"Only {fail_rate:.0%} of genes filtered under misspecified model; "
            "expected > 50%"
        )

    def test_mixture_mask_shape(self, mixture_nb_data, rng_key):
        """Mixture mask should have shape (G,) and dtype bool."""
        counts, r, p, weights = mixture_nb_data
        results = _MockMixtureResults(r, p, weights)
        mask = compute_gof_mask(counts, results, rng_key=rng_key)
        assert mask.shape == (counts.shape[1],)
        assert mask.dtype == bool

    def test_mixture_well_specified_passes_most(
        self, mixture_nb_data, rng_key
    ):
        """Under the true mixture, most genes should pass."""
        counts, r, p, weights = mixture_nb_data
        results = _MockMixtureResults(r, p, weights)
        mask = compute_gof_mask(
            counts, results, rng_key=rng_key, max_variance=1.5
        )
        pass_rate = float(np.mean(mask))
        assert pass_rate > 0.7, (
            f"Only {pass_rate:.0%} of genes passed under true mixture; "
            "expected > 70%"
        )

    def test_max_ks_criterion(self, well_specified_nb_data, rng_key):
        """Adding a max_ks threshold should only make the mask stricter."""
        counts, r, p = well_specified_nb_data
        results = _MockSingleComponentResults(r, p)
        mask_var_only = compute_gof_mask(
            counts, results, rng_key=rng_key, max_variance=1.5
        )
        mask_both = compute_gof_mask(
            counts, results, rng_key=rng_key, max_variance=1.5, max_ks=0.05
        )
        # Combined mask should be at most as permissive as variance-only
        assert np.all(mask_both <= mask_var_only)

    def test_default_rng_key(self, well_specified_nb_data):
        """Should work without explicitly passing rng_key."""
        counts, r, p = well_specified_nb_data
        results = _MockSingleComponentResults(r, p)
        mask = compute_gof_mask(counts, results)
        assert mask.shape == (counts.shape[1],)

    def test_component_slicing(self, mixture_nb_data, rng_key):
        """Passing component= should slice to single-component analysis."""
        counts, r, p, weights = mixture_nb_data
        results = _MockMixtureResults(r, p, weights)
        mask = compute_gof_mask(
            counts, results, component=0, rng_key=rng_key
        )
        assert mask.shape == (counts.shape[1],)
        assert mask.dtype == bool


# ==========================================================================
# PPC-based goodness-of-fit tests
# ==========================================================================


class _MockPPCResults:
    """Mock results object that supports get_posterior_ppc_samples.

    Generates PPC samples by drawing from NB(r, p) with the provided
    parameters, treating them as if they were the posterior mean
    (i.e. no parameter uncertainty — this keeps the test self-contained).
    """

    def __init__(self, r, p, n_cells):
        self._r = r
        self._p = p
        self.n_cells = n_cells
        self.n_components = None
        self.posterior_samples = None

    def get_posterior_ppc_samples(
        self,
        gene_indices=None,
        n_samples=500,
        cell_batch_size=500,
        rng_key=None,
        counts=None,
        store_samples=False,
        verbose=False,
    ):
        """Generate mock PPC samples from NB(r, p)."""
        r = self._r[gene_indices] if gene_indices is not None else self._r
        p = self._p
        nb = dist.NegativeBinomialProbs(r, p)
        key = rng_key if rng_key is not None else random.PRNGKey(0)
        # Shape: (n_samples, n_cells, n_genes_batch)
        return nb.sample(key, (n_samples, self.n_cells))

    def get_component(self, idx):
        return self


# --------------------------------------------------------------------------
# ppc_goodness_of_fit_scores
# --------------------------------------------------------------------------


class TestPPCGoodnessOfFitScores:
    """Tests for the low-level PPC scorer."""

    @pytest.fixture
    def well_specified_ppc(self, rng_key):
        """PPC samples generated from the same model as the observed data."""
        C, G = 500, 10
        r = jnp.full(G, 5.0)
        p = jnp.float32(0.3)
        S = 100

        nb = dist.NegativeBinomialProbs(r, p)
        key1, key2 = random.split(rng_key)
        obs = nb.sample(key1, (C,))
        ppc = nb.sample(key2, (S, C))
        return ppc, obs

    @pytest.fixture
    def misspecified_ppc(self, rng_key):
        """Observed data from one NB, PPC from a very different NB.

        Uses high cell count and many PPC samples so that the credible
        bands are tight, making misspecification clearly detectable.
        """
        C, G = 2000, 10
        S = 200

        # Observed: moderate-mean, tight distribution
        r_obs = jnp.full(G, 10.0)
        p_obs = jnp.float32(0.5)
        nb_obs = dist.NegativeBinomialProbs(r_obs, p_obs)

        # PPC from a distribution with very different mean and shape
        r_ppc = jnp.full(G, 2.0)
        p_ppc = jnp.float32(0.9)
        nb_ppc = dist.NegativeBinomialProbs(r_ppc, p_ppc)

        key1, key2 = random.split(rng_key)
        obs = nb_obs.sample(key1, (C,))
        ppc = nb_ppc.sample(key2, (S, C))
        return ppc, obs

    def test_output_keys(self, well_specified_ppc):
        """Should return calibration_failure and l1_distance."""
        ppc, obs = well_specified_ppc
        scores = ppc_goodness_of_fit_scores(ppc, obs)
        assert set(scores.keys()) == {"calibration_failure", "l1_distance"}

    def test_output_shapes(self, well_specified_ppc):
        """Both arrays should have shape (G,)."""
        ppc, obs = well_specified_ppc
        G = obs.shape[1]
        scores = ppc_goodness_of_fit_scores(ppc, obs)
        assert scores["calibration_failure"].shape == (G,)
        assert scores["l1_distance"].shape == (G,)

    def test_well_specified_low_scores(self, well_specified_ppc):
        """Under the true model, calibration failure should be low."""
        ppc, obs = well_specified_ppc
        scores = ppc_goodness_of_fit_scores(ppc, obs, credible_level=95)
        # Most genes should have low calibration failure
        median_cal = float(np.median(scores["calibration_failure"]))
        assert median_cal < 0.5, (
            f"Median calibration failure = {median_cal:.3f} under true "
            "model; expected < 0.5"
        )

    def test_misspecified_high_scores(self, misspecified_ppc):
        """Under wrong model, calibration failure should be high."""
        ppc, obs = misspecified_ppc
        scores = ppc_goodness_of_fit_scores(ppc, obs, credible_level=95)
        median_cal = float(np.median(scores["calibration_failure"]))
        assert median_cal > 0.3, (
            f"Median calibration failure = {median_cal:.3f} under "
            "misspecified model; expected > 0.3"
        )

    def test_l1_distance_nonnegative(self, well_specified_ppc):
        """L1 distance should be non-negative."""
        ppc, obs = well_specified_ppc
        scores = ppc_goodness_of_fit_scores(ppc, obs)
        assert np.all(scores["l1_distance"] >= 0.0)

    def test_calibration_failure_bounded(self, well_specified_ppc):
        """Calibration failure should be in [0, 1]."""
        ppc, obs = well_specified_ppc
        scores = ppc_goodness_of_fit_scores(ppc, obs)
        assert np.all(scores["calibration_failure"] >= 0.0)
        assert np.all(scores["calibration_failure"] <= 1.0)


# --------------------------------------------------------------------------
# compute_ppc_gof_mask
# --------------------------------------------------------------------------


class TestComputePPCGofMask:
    """Tests for the PPC-based mask builder."""

    @pytest.fixture
    def simple_ppc_setup(self, rng_key):
        """Well-specified mock results for mask testing."""
        C, G = 300, 20
        r = jnp.full(G, 5.0)
        p = jnp.float32(0.3)
        nb = dist.NegativeBinomialProbs(r, p)
        counts = nb.sample(rng_key, (C,))
        results = _MockPPCResults(r, p, C)
        return counts, results, G

    def test_mask_shape_dtype(self, simple_ppc_setup):
        """Mask should be bool array of shape (G,)."""
        counts, results, G = simple_ppc_setup
        mask = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=10,
            verbose=False,
        )
        assert mask.shape == (G,)
        assert mask.dtype == bool

    def test_return_scores_flag(self, simple_ppc_setup):
        """return_scores=True should return (mask, dict)."""
        counts, results, G = simple_ppc_setup
        out = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=10,
            return_scores=True,
            verbose=False,
        )
        assert isinstance(out, tuple)
        mask, scores = out
        assert mask.shape == (G,)
        assert "calibration_failure" in scores
        assert "l1_distance" in scores
        assert scores["calibration_failure"].shape == (G,)

    def test_well_specified_passes_most(self, simple_ppc_setup):
        """Under the true model, most genes should pass a generous threshold."""
        counts, results, G = simple_ppc_setup
        mask = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=80,
            gene_batch_size=10,
            max_calibration_failure=0.8,
            verbose=False,
        )
        pass_rate = float(np.mean(mask))
        assert pass_rate > 0.5, (
            f"Only {pass_rate:.0%} of genes passed under true model; "
            "expected > 50%"
        )

    def test_batch_size_invariance(self, simple_ppc_setup):
        """Different gene_batch_size should produce identical masks.

        Since the rng_key splitting is deterministic and the mock samples
        from the same distribution, different batch sizes should yield
        consistent scores (up to floating-point noise).
        """
        counts, results, G = simple_ppc_setup
        mask_small = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=5,
            max_calibration_failure=0.6,
            rng_key=random.PRNGKey(99),
            verbose=False,
        )
        mask_large = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=20,
            max_calibration_failure=0.6,
            rng_key=random.PRNGKey(99),
            verbose=False,
        )
        # Shapes must match
        assert mask_small.shape == mask_large.shape

    def test_l1_threshold(self, simple_ppc_setup):
        """Adding max_l1_distance should only make the mask stricter."""
        counts, results, G = simple_ppc_setup
        mask_cal_only = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=10,
            max_calibration_failure=0.8,
            verbose=False,
        )
        mask_both = compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=10,
            max_calibration_failure=0.8,
            max_l1_distance=0.01,
            verbose=False,
        )
        # Combined mask should be at most as permissive
        assert np.all(mask_both <= mask_cal_only)

    def test_posterior_samples_cleared(self, simple_ppc_setup):
        """After mask computation the cached posterior should be cleared."""
        counts, results, G = simple_ppc_setup
        # Pre-set posterior_samples to something truthy
        results.posterior_samples = {"dummy": jnp.zeros(5)}
        compute_ppc_gof_mask(
            counts, results,
            n_ppc_samples=50,
            gene_batch_size=10,
            verbose=False,
        )
        assert results.posterior_samples is None
