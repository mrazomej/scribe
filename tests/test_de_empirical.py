"""Tests for the non-parametric (empirical) DE module.

Covers:
- ``empirical_differential_expression`` — shapes, lfsr range, strong/no effect
- ``compute_clr_differences`` — shapes, mixture slicing, paired vs unpaired
- ``compare()`` dispatch — method="parametric" vs method="empirical"
- ``ScribeEmpiricalDEResults`` — gene_level, call_genes, compute_pefp,
  find_threshold, summary, test_contrast
- Base-class methods working on empirical results
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.scipy.stats import norm

from scribe.de import (
    ScribeDEResults,
    ScribeParametricDEResults,
    ScribeEmpiricalDEResults,
    compare,
    compute_clr_differences,
    empirical_differential_expression,
)
from scribe.de._empirical import _aggregate_genes


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Base RNG key."""
    return random.PRNGKey(42)


@pytest.fixture
def gaussian_delta_samples(rng):
    """Delta samples drawn from a known Gaussian for sanity checks.

    Gene 0: strong positive effect (mean=3, sd=1)
    Gene 1: strong negative effect (mean=-3, sd=1)
    Gene 2: no effect (mean=0, sd=1)
    Gene 3: moderate effect (mean=0.5, sd=1)
    """
    N = 50_000
    means = jnp.array([3.0, -3.0, 0.0, 0.5])
    sds = jnp.ones(4)
    key = rng
    samples = means[None, :] + sds[None, :] * random.normal(key, (N, 4))
    return samples


@pytest.fixture
def simple_r_samples(rng):
    """Simple 2D r samples (non-mixture) for Dirichlet testing.

    Shape: (100, 5) — 100 posterior samples, 5 genes.
    Values are positive concentrations suitable for Dirichlet.
    """
    return jnp.abs(random.normal(rng, (100, 5))) + 1.0


@pytest.fixture
def mixture_r_samples(rng):
    """3D r samples (mixture model) for component slicing tests.

    Shape: (100, 3, 5) — 100 posterior samples, 3 components, 5 genes.
    """
    return jnp.abs(random.normal(rng, (100, 3, 5))) + 1.0


# --------------------------------------------------------------------------
# Tests: empirical_differential_expression
# --------------------------------------------------------------------------


class TestEmpiricalDE:
    """Tests for ``empirical_differential_expression``."""

    def test_shapes(self, gaussian_delta_samples):
        """Output dict has correct shapes for all keys."""
        result = empirical_differential_expression(gaussian_delta_samples)
        D = gaussian_delta_samples.shape[1]
        for key in ["delta_mean", "delta_sd", "prob_positive",
                     "prob_effect", "lfsr", "lfsr_tau"]:
            assert result[key].shape == (D,), f"{key} has wrong shape"
        assert len(result["gene_names"]) == D

    def test_lfsr_range(self, gaussian_delta_samples):
        """lfsr must be in [0, 0.5]."""
        result = empirical_differential_expression(gaussian_delta_samples)
        assert jnp.all(result["lfsr"] >= 0.0)
        assert jnp.all(result["lfsr"] <= 0.5)

    def test_lfsr_tau_range(self, gaussian_delta_samples):
        """lfsr_tau must be in [0, 1]."""
        result = empirical_differential_expression(
            gaussian_delta_samples, tau=0.1
        )
        assert jnp.all(result["lfsr_tau"] >= 0.0)
        assert jnp.all(result["lfsr_tau"] <= 1.0)

    def test_strong_positive_effect(self, gaussian_delta_samples):
        """Gene 0 (mean=3) should have lfsr near 0."""
        result = empirical_differential_expression(gaussian_delta_samples)
        # With mean=3 and sd=1, P(Delta < 0) ≈ 0.0013
        assert float(result["lfsr"][0]) < 0.01

    def test_strong_negative_effect(self, gaussian_delta_samples):
        """Gene 1 (mean=-3) should have lfsr near 0."""
        result = empirical_differential_expression(gaussian_delta_samples)
        assert float(result["lfsr"][1]) < 0.01

    def test_no_effect(self, gaussian_delta_samples):
        """Gene 2 (mean=0) should have lfsr near 0.5."""
        result = empirical_differential_expression(gaussian_delta_samples)
        assert abs(float(result["lfsr"][2]) - 0.5) < 0.02

    def test_prob_effect_with_tau(self, gaussian_delta_samples):
        """With tau > 0, prob_effect should be lower than with tau=0."""
        result_0 = empirical_differential_expression(
            gaussian_delta_samples, tau=0.0
        )
        result_1 = empirical_differential_expression(
            gaussian_delta_samples, tau=1.0
        )
        # prob_effect with tau=1.0 should be <= prob_effect with tau=0
        assert jnp.all(result_1["prob_effect"] <= result_0["prob_effect"] + 1e-6)

    def test_gaussian_sanity_check(self, gaussian_delta_samples):
        """Empirical lfsr matches analytic Gaussian lfsr within tolerance.

        For gene 0 (mean=3, sd=1): analytic lfsr = norm.cdf(0, 3, 1).
        """
        result = empirical_differential_expression(gaussian_delta_samples)
        # Analytic lfsr for gene 0 (mean=3, sd=1)
        analytic_lfsr_0 = float(norm.cdf(0.0, loc=3.0, scale=1.0))
        empirical_lfsr_0 = float(result["lfsr"][0])
        # Should match within Monte Carlo noise (~1/sqrt(N))
        assert abs(empirical_lfsr_0 - analytic_lfsr_0) < 0.01

    def test_custom_gene_names(self, gaussian_delta_samples):
        """Custom gene names are propagated."""
        names = ["A", "B", "C", "D"]
        result = empirical_differential_expression(
            gaussian_delta_samples, gene_names=names
        )
        assert result["gene_names"] == names

    def test_default_gene_names(self, gaussian_delta_samples):
        """Default gene names are gene_0, gene_1, ..."""
        result = empirical_differential_expression(gaussian_delta_samples)
        assert result["gene_names"][0] == "gene_0"


# --------------------------------------------------------------------------
# Tests: compute_clr_differences
# --------------------------------------------------------------------------


class TestComputeCLRDifferences:
    """Tests for ``compute_clr_differences``."""

    def test_shapes_2d(self, simple_r_samples, rng):
        """2D input produces (N, D) output."""
        # Use two independent sets of r samples
        key_a, key_b = random.split(rng)
        r_A = jnp.abs(random.normal(key_a, (100, 5))) + 1.0
        r_B = jnp.abs(random.normal(key_b, (100, 5))) + 1.0
        delta = compute_clr_differences(r_A, r_B, rng_key=rng)
        assert delta.shape == (100, 5)

    def test_shapes_3d_mixture(self, mixture_r_samples, rng):
        """3D (mixture) input with component slicing produces (N, D)."""
        delta = compute_clr_differences(
            mixture_r_samples, mixture_r_samples,
            component_A=0, component_B=1,
            rng_key=rng,
        )
        assert delta.shape == (100, 5)

    def test_mixture_requires_component(self, mixture_r_samples, rng):
        """3D input without component index raises ValueError."""
        with pytest.raises(ValueError, match="component_A"):
            compute_clr_differences(
                mixture_r_samples, mixture_r_samples, rng_key=rng
            )

    def test_dimension_mismatch(self, rng):
        """Mismatched gene dimensions raise ValueError."""
        r_A = jnp.ones((50, 5))
        r_B = jnp.ones((50, 7))
        with pytest.raises(ValueError, match="Gene dimensions"):
            compute_clr_differences(r_A, r_B, rng_key=rng)

    def test_paired_requires_equal_n(self, rng):
        """paired=True with different N raises ValueError."""
        r_A = jnp.ones((50, 5))
        r_B = jnp.ones((60, 5))
        with pytest.raises(ValueError, match="paired=True"):
            compute_clr_differences(r_A, r_B, paired=True, rng_key=rng)

    def test_unequal_n_truncates(self, rng):
        """Independent mode truncates to min(N_A, N_B)."""
        r_A = jnp.abs(random.normal(rng, (80, 5))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(99), (60, 5))) + 1.0
        delta = compute_clr_differences(r_A, r_B, rng_key=rng)
        assert delta.shape == (60, 5)

    def test_clr_sums_to_zero(self, simple_r_samples, rng):
        """CLR differences should approximately sum to zero along gene axis.

        CLR(rho_A) sums to 0 and CLR(rho_B) sums to 0, so their
        difference also sums to 0 for each sample.
        """
        r_A = jnp.abs(random.normal(rng, (100, 5))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(7), (100, 5))) + 1.0
        delta = compute_clr_differences(r_A, r_B, rng_key=rng)
        row_sums = jnp.sum(delta, axis=1)
        np.testing.assert_allclose(
            np.array(row_sums), 0.0, atol=1e-5
        )

    def test_n_samples_dirichlet_gt_1(self, rng):
        """n_samples_dirichlet > 1 multiplies the sample count."""
        r_A = jnp.abs(random.normal(rng, (50, 5))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(3), (50, 5))) + 1.0
        delta = compute_clr_differences(
            r_A, r_B, n_samples_dirichlet=3, rng_key=rng
        )
        # 50 posterior samples * 3 Dirichlet draws = 150
        assert delta.shape == (150, 5)


# --------------------------------------------------------------------------
# Tests: paired vs unpaired
# --------------------------------------------------------------------------


class TestPairedVsUnpaired:
    """Test that paired=True produces different results from paired=False
    when comparing components within the same model."""

    def test_paired_preserves_correlation(self, mixture_r_samples, rng):
        """Paired and unpaired modes produce different delta_sd.

        Within-mixture components share the same posterior draw, so
        paired differencing should capture their correlation (smaller
        variance) compared to unpaired (which treats them as independent,
        giving larger variance).
        """
        delta_paired = compute_clr_differences(
            mixture_r_samples, mixture_r_samples,
            component_A=0, component_B=1,
            paired=True,
            rng_key=rng,
        )
        delta_unpaired = compute_clr_differences(
            mixture_r_samples, mixture_r_samples,
            component_A=0, component_B=1,
            paired=False,
            rng_key=rng,
        )
        # Both should have the same shape
        assert delta_paired.shape == delta_unpaired.shape
        # They should be numerically different
        assert not jnp.allclose(delta_paired, delta_unpaired)


# --------------------------------------------------------------------------
# Tests: compare() dispatch
# --------------------------------------------------------------------------


class TestCompareDispatch:
    """Tests for the ``compare()`` factory function."""

    def test_parametric_returns_correct_type(self):
        """method='parametric' returns ScribeParametricDEResults."""
        D_alr = 10
        model_A = {
            "loc": jnp.zeros(D_alr),
            "cov_factor": jnp.eye(D_alr, 3),
            "cov_diag": jnp.ones(D_alr),
        }
        model_B = {
            "loc": jnp.ones(D_alr) * 0.1,
            "cov_factor": jnp.eye(D_alr, 3),
            "cov_diag": jnp.ones(D_alr),
        }
        de = compare(model_A, model_B, method="parametric")
        assert isinstance(de, ScribeParametricDEResults)
        assert isinstance(de, ScribeDEResults)  # is-a base
        assert de.method == "parametric"

    def test_empirical_returns_correct_type(self, rng):
        """method='empirical' returns ScribeEmpiricalDEResults."""
        r_A = jnp.abs(random.normal(rng, (100, 5))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, 5))) + 1.0
        de = compare(r_A, r_B, method="empirical", rng_key=rng)
        assert isinstance(de, ScribeEmpiricalDEResults)
        assert isinstance(de, ScribeDEResults)  # is-a base
        assert de.method == "empirical"

    def test_default_method_is_parametric(self):
        """Default method should be 'parametric'."""
        D_alr = 5
        model = {
            "loc": jnp.zeros(D_alr),
            "cov_factor": jnp.eye(D_alr, 2),
            "cov_diag": jnp.ones(D_alr),
        }
        de = compare(model, model)
        assert isinstance(de, ScribeParametricDEResults)

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            compare(jnp.zeros(5), jnp.zeros(5), method="magic")

    def test_empirical_with_gene_names(self, rng):
        """Gene names are propagated in empirical mode."""
        r = jnp.abs(random.normal(rng, (50, 4))) + 1.0
        names = ["A", "B", "C", "D"]
        de = compare(r, r, method="empirical", gene_names=names, rng_key=rng)
        assert de.gene_names == names

    def test_empirical_gene_names_length_mismatch(self, rng):
        """Wrong-length gene_names raises ValueError."""
        r = jnp.abs(random.normal(rng, (50, 4))) + 1.0
        with pytest.raises(ValueError, match="gene_names"):
            compare(
                r, r, method="empirical", gene_names=["a", "b"],
                rng_key=rng,
            )

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)


# --------------------------------------------------------------------------
# Tests: ScribeEmpiricalDEResults methods
# --------------------------------------------------------------------------


class TestEmpiricalResultsMethods:
    """Tests for methods on ``ScribeEmpiricalDEResults``."""

    @pytest.fixture
    def empirical_de(self):
        """Create an empirical DE results object with known data."""
        rng = random.PRNGKey(0)
        r_A = jnp.abs(random.normal(rng, (500, 8))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (500, 8))) + 2.0
        return compare(
            r_A, r_B,
            method="empirical",
            gene_names=[f"g{i}" for i in range(8)],
            rng_key=rng,
        )

    def test_gene_level_returns_dict(self, empirical_de):
        """gene_level returns a dict with expected keys."""
        result = empirical_de.gene_level(tau=0.0)
        expected_keys = {
            "delta_mean", "delta_sd", "prob_positive",
            "prob_effect", "lfsr", "lfsr_tau", "gene_names",
        }
        assert set(result.keys()) == expected_keys

    def test_call_genes_returns_bool_mask(self, empirical_de):
        """call_genes returns a boolean array of shape (D,)."""
        is_de = empirical_de.call_genes(tau=0.0)
        assert is_de.shape == (empirical_de.D,)
        assert is_de.dtype == jnp.bool_

    def test_compute_pefp_returns_float(self, empirical_de):
        """compute_pefp returns a scalar float."""
        pefp = empirical_de.compute_pefp(threshold=0.05, tau=0.0)
        assert isinstance(pefp, float)
        assert 0.0 <= pefp <= 1.0

    def test_find_threshold_returns_float(self, empirical_de):
        """find_threshold returns a scalar float."""
        threshold = empirical_de.find_threshold(target_pefp=0.05, tau=0.0)
        assert isinstance(threshold, float)
        assert threshold >= 0.0

    def test_summary_returns_string(self, empirical_de):
        """summary returns a non-empty string."""
        s = empirical_de.summary(tau=0.0, top_n=5)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_caching(self, empirical_de):
        """gene_level results are cached by tau."""
        _ = empirical_de.gene_level(tau=0.0)
        cached_0 = empirical_de._gene_results
        assert cached_0 is not None
        assert empirical_de._cached_tau == 0.0

        # Calling with different tau should update the cache
        _ = empirical_de.gene_level(tau=0.5)
        assert empirical_de._cached_tau == 0.5
        cached_05 = empirical_de._gene_results
        # lfsr_tau should differ between tau=0 and tau=0.5
        assert not jnp.allclose(
            cached_0["lfsr_tau"], cached_05["lfsr_tau"]
        )

    def test_repr(self, empirical_de):
        """repr includes method info."""
        r = repr(empirical_de)
        assert "ScribeEmpiricalDEResults" in r
        assert "n_samples=" in r

    def test_D_property(self, empirical_de):
        """D returns the number of genes."""
        assert empirical_de.D == 8

    def test_D_alr_property(self, empirical_de):
        """D_alr returns D - 1."""
        assert empirical_de.D_alr == 7


# --------------------------------------------------------------------------
# Tests: Empirical test_contrast
# --------------------------------------------------------------------------


class TestEmpiricalTestContrast:
    """Tests for ``ScribeEmpiricalDEResults.test_contrast``."""

    def test_contrast_returns_dict(self):
        """test_contrast returns a dict with expected keys."""
        delta = jnp.ones((100, 4)) * jnp.array([1.0, -1.0, 0.0, 0.5])
        de = ScribeEmpiricalDEResults(
            delta_samples=delta,
            gene_names=["a", "b", "c", "d"],
        )
        contrast = jnp.array([1.0, -1.0, 0.0, 0.0])
        result = de.test_contrast(contrast, tau=0.0)
        expected_keys = {
            "contrast_mean", "contrast_sd", "prob_positive",
            "prob_effect", "lfsr", "lfsr_tau",
        }
        assert set(result.keys()) == expected_keys

    def test_contrast_matches_manual(self):
        """Contrast mean matches manual delta_samples @ contrast."""
        rng = random.PRNGKey(10)
        delta = random.normal(rng, (1000, 4))
        de = ScribeEmpiricalDEResults(delta_samples=delta)
        contrast = jnp.array([0.5, -0.5, 0.0, 0.0])
        result = de.test_contrast(contrast, tau=0.0)
        manual_mean = float(jnp.mean(delta @ contrast))
        np.testing.assert_allclose(
            result["contrast_mean"], manual_mean, atol=1e-5
        )

    def test_contrast_strong_effect(self):
        """Strong contrast should have lfsr near 0."""
        # All samples have positive contrast value
        delta = jnp.ones((1000, 3)) * jnp.array([2.0, 0.0, 0.0])
        de = ScribeEmpiricalDEResults(delta_samples=delta)
        contrast = jnp.array([1.0, 0.0, 0.0])
        result = de.test_contrast(contrast, tau=0.0)
        assert result["lfsr"] < 0.01


# --------------------------------------------------------------------------
# Tests: Parametric backward compatibility
# --------------------------------------------------------------------------


class TestParametricBackwardCompat:
    """Ensure ScribeParametricDEResults works as old ScribeDEResults did."""

    @pytest.fixture
    def parametric_de(self):
        """Create a parametric DE results object."""
        D_alr = 10
        return compare(
            {
                "loc": jnp.zeros(D_alr),
                "cov_factor": jnp.eye(D_alr, 3),
                "cov_diag": jnp.ones(D_alr),
            },
            {
                "loc": jnp.ones(D_alr) * 0.5,
                "cov_factor": jnp.eye(D_alr, 3),
                "cov_diag": jnp.ones(D_alr),
            },
            method="parametric",
        )

    def test_gene_level(self, parametric_de):
        """gene_level works on parametric results."""
        result = parametric_de.gene_level(tau=0.0)
        assert "lfsr" in result
        assert result["lfsr"].shape == (parametric_de.D,)

    def test_call_genes(self, parametric_de):
        """call_genes works on parametric results."""
        is_de = parametric_de.call_genes(tau=0.0)
        assert is_de.shape == (parametric_de.D,)

    def test_compute_pefp(self, parametric_de):
        """compute_pefp works on parametric results."""
        pefp = parametric_de.compute_pefp(threshold=0.05, tau=0.0)
        assert isinstance(pefp, float)

    def test_summary(self, parametric_de):
        """summary works on parametric results."""
        s = parametric_de.summary(tau=0.0)
        assert isinstance(s, str)

    def test_test_contrast(self, parametric_de):
        """test_contrast is available on parametric results."""
        contrast = jnp.ones(parametric_de.D) / parametric_de.D
        result = parametric_de.test_contrast(contrast, tau=0.0)
        assert "delta_mean" in result or "contrast_mean" in result

    def test_test_gene_set(self, parametric_de):
        """test_gene_set is available on parametric results."""
        indices = jnp.array([0, 1, 2])
        result = parametric_de.test_gene_set(indices, tau=0.0)
        assert isinstance(result, dict)

    def test_repr(self, parametric_de):
        """repr includes parametric info."""
        r = repr(parametric_de)
        assert "ScribeParametricDEResults" in r
        assert "rank_A=" in r


# --------------------------------------------------------------------------
# Tests: _aggregate_genes
# --------------------------------------------------------------------------


class TestAggregateGenes:
    """Tests for ``_aggregate_genes`` helper."""

    def test_shapes(self, rng):
        """Output shape is (N, D_kept + 1)."""
        r_A = jnp.abs(random.normal(rng, (100, 8))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, 8))) + 1.0
        mask = jnp.array([True, True, True, False, False, True, True, False])
        r_A_agg, r_B_agg = _aggregate_genes(r_A, r_B, mask)
        D_kept = int(mask.sum())
        assert r_A_agg.shape == (100, D_kept + 1)
        assert r_B_agg.shape == (100, D_kept + 1)

    def test_concentration_preserved(self, rng):
        """Total Dirichlet concentration must be preserved after aggregation."""
        r_A = jnp.abs(random.normal(rng, (50, 6))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(2), (50, 6))) + 1.0
        mask = jnp.array([True, False, True, False, True, True])
        r_A_agg, r_B_agg = _aggregate_genes(r_A, r_B, mask)

        np.testing.assert_allclose(
            np.array(r_A.sum(axis=1)),
            np.array(r_A_agg.sum(axis=1)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.array(r_B.sum(axis=1)),
            np.array(r_B_agg.sum(axis=1)),
            rtol=1e-5,
        )

    def test_all_true_mask(self, rng):
        """All-True mask keeps all genes plus a zero 'other' column."""
        r_A = jnp.abs(random.normal(rng, (30, 4))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(3), (30, 4))) + 1.0
        mask = jnp.ones(4, dtype=bool)
        r_A_agg, r_B_agg = _aggregate_genes(r_A, r_B, mask)
        assert r_A_agg.shape == (30, 5)
        # The "other" column should be 0 when no genes are filtered
        np.testing.assert_allclose(
            np.array(r_A_agg[:, -1]), 0.0, atol=1e-7,
        )

    def test_all_false_raises(self, rng):
        """All-False mask raises ValueError."""
        r_A = jnp.ones((10, 4))
        r_B = jnp.ones((10, 4))
        mask = jnp.zeros(4, dtype=bool)
        with pytest.raises(ValueError, match="at least one gene"):
            _aggregate_genes(r_A, r_B, mask)

    def test_wrong_mask_length_raises(self, rng):
        """Mask with wrong length raises ValueError."""
        r_A = jnp.ones((10, 4))
        r_B = jnp.ones((10, 4))
        mask = jnp.array([True, False])
        with pytest.raises(ValueError, match="gene_mask"):
            _aggregate_genes(r_A, r_B, mask)


# --------------------------------------------------------------------------
# Tests: compute_clr_differences with gene_mask
# --------------------------------------------------------------------------


class TestCLRDifferencesGeneMask:
    """Tests for ``compute_clr_differences`` with ``gene_mask``."""

    def test_output_shape_with_mask(self, rng):
        """Output has D_kept columns when gene_mask is provided."""
        D = 8
        r_A = jnp.abs(random.normal(rng, (100, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, D))) + 1.0
        mask = jnp.array([True, True, True, False, False, True, True, False])
        D_kept = int(mask.sum())
        delta = compute_clr_differences(r_A, r_B, rng_key=rng, gene_mask=mask)
        assert delta.shape == (100, D_kept)

    def test_no_mask_unchanged(self, rng):
        """Without gene_mask, output has all D columns."""
        D = 6
        r_A = jnp.abs(random.normal(rng, (50, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (50, D))) + 1.0
        delta = compute_clr_differences(r_A, r_B, rng_key=rng)
        assert delta.shape == (50, D)

    def test_clr_rows_sum_to_zero_with_mask(self, rng):
        """CLR differences should NOT sum to zero when mask drops 'other'.

        CLR on the aggregated simplex sums to zero across D_kept+1, but
        we drop the 'other' column, so the kept columns do not sum to
        zero exactly.  This test verifies the shape is D_kept, not
        D_kept+1.
        """
        D = 6
        r_A = jnp.abs(random.normal(rng, (80, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(5), (80, D))) + 1.0
        mask = jnp.array([True, True, True, True, False, False])
        delta = compute_clr_differences(r_A, r_B, rng_key=rng, gene_mask=mask)
        assert delta.shape[1] == 4

    def test_mixture_with_mask(self, rng):
        """gene_mask works with 3D (mixture) inputs."""
        D = 5
        r_mix = jnp.abs(random.normal(rng, (100, 3, D))) + 1.0
        mask = jnp.array([True, False, True, True, False])
        delta = compute_clr_differences(
            r_mix, r_mix,
            component_A=0, component_B=1,
            rng_key=rng,
            gene_mask=mask,
        )
        assert delta.shape == (100, 3)


# --------------------------------------------------------------------------
# Tests: compare() with gene_mask
# --------------------------------------------------------------------------


class TestCompareGeneMask:
    """Tests for ``compare()`` with ``gene_mask``."""

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)

    def test_empirical_gene_names_filtered(self, rng):
        """Empirical compare with gene_mask filters gene_names."""
        D = 6
        r_A = jnp.abs(random.normal(rng, (100, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, D))) + 1.0
        names = ["g0", "g1", "g2", "g3", "g4", "g5"]
        mask = jnp.array([True, False, True, True, False, True])
        de = compare(
            r_A, r_B,
            method="empirical",
            gene_names=names,
            rng_key=rng,
            gene_mask=mask,
        )
        assert de.gene_names == ["g0", "g2", "g3", "g5"]
        assert de.D == 4

    def test_empirical_no_mask_backward_compatible(self, rng):
        """Empirical compare without mask works as before."""
        D = 5
        r = jnp.abs(random.normal(rng, (50, D))) + 1.0
        de = compare(r, r, method="empirical", rng_key=rng)
        assert de.D == D

    def test_parametric_gene_names_filtered(self):
        """Parametric compare with gene_mask filters gene_names.

        Simulates the workflow where fit_logistic_normal was called with
        gene_mask: 5 original genes, mask keeps 4, model is fitted on
        D_kept + 1 = 5 simplex (4 kept + "other"), so D_alr = 4.
        """
        # Original: 5 genes, mask keeps 4 → aggregated simplex has 5
        # genes (4 kept + "other"), D_alr = 4
        D_alr = 4
        model = {
            "loc": jnp.zeros(D_alr),
            "cov_factor": jnp.eye(D_alr, 2),
            "cov_diag": jnp.ones(D_alr),
        }
        names = ["g0", "g1", "g2", "g3", "g4"]
        mask = np.array([True, True, False, True, True])
        de = compare(
            model, model,
            method="parametric",
            gene_names=names,
            gene_mask=mask,
        )
        # gene_names should be filtered to the kept genes
        assert de.gene_names == ["g0", "g1", "g3", "g4"]
        # D should be D_kept=4, not D_full=5
        assert de.D == 4
        assert de.D_full == 5
        # gene_level should return D_kept results
        result = de.gene_level(tau=0.0)
        assert result["lfsr"].shape == (4,)
        assert len(result["gene_names"]) == 4

    def test_empirical_gene_level_with_mask(self, rng):
        """gene_level on a masked empirical result has correct shapes."""
        D = 8
        r_A = jnp.abs(random.normal(rng, (200, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (200, D))) + 2.0
        mask = jnp.array([True, True, False, True, False, True, True, False])
        D_kept = int(mask.sum())
        de = compare(
            r_A, r_B,
            method="empirical",
            gene_names=[f"g{i}" for i in range(D)],
            rng_key=rng,
            gene_mask=mask,
        )
        result = de.gene_level(tau=0.0)
        assert result["delta_mean"].shape == (D_kept,)
        assert result["lfsr"].shape == (D_kept,)
        assert len(result["gene_names"]) == D_kept
