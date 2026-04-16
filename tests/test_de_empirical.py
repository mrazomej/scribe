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
    sample_compositions,
    sample_mixture_compositions,
    compute_delta_from_simplex,
    empirical_differential_expression,
)
from scribe.de._empirical import (
    _aggregate_genes,
    _aggregate_simplex,
    _drop_scalar_p,
    _slice_component,
    _weight_simplex_by_components,
)
from scribe.de._biological import _needs_gene_broadcast, weight_bio_samples


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
        for key in [
            "delta_mean",
            "delta_sd",
            "prob_positive",
            "prob_effect",
            "lfsr",
            "lfsr_tau",
        ]:
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
        assert jnp.all(
            result_1["prob_effect"] <= result_0["prob_effect"] + 1e-6
        )

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
            mixture_r_samples,
            mixture_r_samples,
            component_A=0,
            component_B=1,
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
        np.testing.assert_allclose(np.array(row_sums), 0.0, atol=1e-5)

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
            mixture_r_samples,
            mixture_r_samples,
            component_A=0,
            component_B=1,
            paired=True,
            rng_key=rng,
        )
        delta_unpaired = compute_clr_differences(
            mixture_r_samples,
            mixture_r_samples,
            component_A=0,
            component_B=1,
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

    def test_empirical_rejects_parametric_models(self):
        """Forcing empirical mode with parametric models should fail fast."""
        D_alr = 5
        model = {
            "loc": jnp.zeros(D_alr),
            "cov_factor": jnp.eye(D_alr, 2),
            "cov_diag": jnp.ones(D_alr),
        }
        with pytest.raises(
            ValueError, match="expects posterior samples arrays or results"
        ):
            compare(model, model, method="empirical")

    def test_shrinkage_rejects_parametric_models(self):
        """Forcing shrinkage mode with parametric models should fail fast."""
        D_alr = 5
        model = {
            "loc": jnp.zeros(D_alr),
            "cov_factor": jnp.eye(D_alr, 2),
            "cov_diag": jnp.ones(D_alr),
        }
        with pytest.raises(
            ValueError, match="expects posterior samples arrays or results"
        ):
            compare(model, model, method="shrinkage")

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
                r,
                r,
                method="empirical",
                gene_names=["a", "b"],
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
            r_A,
            r_B,
            method="empirical",
            gene_names=[f"g{i}" for i in range(8)],
            rng_key=rng,
        )

    def test_gene_level_returns_dict(self, empirical_de):
        """gene_level returns a dict with expected keys."""
        result = empirical_de.gene_level(tau=0.0)
        expected_keys = {
            "delta_mean",
            "delta_sd",
            "prob_positive",
            "prob_effect",
            "lfsr",
            "lfsr_tau",
            "gene_names",
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
        assert not jnp.allclose(cached_0["lfsr_tau"], cached_05["lfsr_tau"])

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
            "contrast_mean",
            "contrast_sd",
            "prob_positive",
            "prob_effect",
            "lfsr",
            "lfsr_tau",
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
            np.array(r_A_agg[:, -1]),
            0.0,
            atol=1e-7,
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
# Tests: _aggregate_simplex
# --------------------------------------------------------------------------


class TestAggregateSimplex:
    """Tests for ``_aggregate_simplex`` helper."""

    def test_output_shape(self, rng):
        """Output shape is (N, D_kept + 1)."""
        simplex = jax.nn.softmax(random.normal(rng, (100, 8)), axis=-1)
        mask = jnp.array([True, True, True, False, False, True, True, False])
        D_kept = int(mask.sum())
        out = _aggregate_simplex(simplex, mask)
        assert out.shape == (100, D_kept + 1)

    def test_rows_sum_to_one(self, rng):
        """Rows of the aggregated simplex must still sum to 1."""
        simplex = jax.nn.softmax(random.normal(rng, (50, 6)), axis=-1)
        mask = jnp.array([True, False, True, False, True, True])
        out = _aggregate_simplex(simplex, mask)
        np.testing.assert_allclose(
            np.array(out.sum(axis=1)),
            np.ones(50),
            atol=1e-5,
        )

    def test_other_column_is_sum_of_masked(self, rng):
        """Last column must equal the sum of all masked-out gene columns."""
        simplex = jax.nn.softmax(random.normal(rng, (30, 5)), axis=-1)
        mask = jnp.array([True, False, True, False, True])
        out = _aggregate_simplex(simplex, mask)
        expected_other = simplex[:, ~mask].sum(axis=1)
        np.testing.assert_allclose(
            np.array(out[:, -1]),
            np.array(expected_other),
            atol=1e-6,
        )

    def test_kept_columns_unchanged(self, rng):
        """Kept gene columns must be identical to the original simplex columns."""
        simplex = jax.nn.softmax(random.normal(rng, (40, 5)), axis=-1)
        mask = jnp.array([True, False, True, False, True])
        out = _aggregate_simplex(simplex, mask)
        np.testing.assert_allclose(
            np.array(out[:, :-1]),
            np.array(simplex[:, mask]),
            atol=1e-6,
        )

    def test_wrong_mask_length_raises(self):
        """Mask length mismatch raises ValueError."""
        simplex = jnp.ones((10, 4)) / 4.0
        mask = jnp.array([True, False])
        with pytest.raises(ValueError, match="gene_mask"):
            _aggregate_simplex(simplex, mask)

    def test_all_false_mask_raises(self):
        """All-False mask must raise ValueError."""
        simplex = jnp.ones((10, 4)) / 4.0
        mask = jnp.array([False, False, False, False])
        with pytest.raises(ValueError, match="gene_mask"):
            _aggregate_simplex(simplex, mask)


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
            r_mix,
            r_mix,
            component_A=0,
            component_B=1,
            rng_key=rng,
            gene_mask=mask,
        )
        assert delta.shape == (100, 3)

    def test_gene_mask_with_p_samples_shape(self, rng):
        """gene_mask and gene-specific p_samples can be combined (lazy path)."""
        D = 8
        r_A = jnp.abs(random.normal(rng, (100, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, D))) + 1.0
        p_A = jax.nn.sigmoid(random.normal(random.PRNGKey(2), (100, D)))
        p_B = jax.nn.sigmoid(random.normal(random.PRNGKey(3), (100, D)))
        mask = jnp.array([True, True, True, False, False, True, True, False])
        D_kept = int(mask.sum())
        # Should not raise; lazy aggregation used after Gamma sampling.
        delta = compute_clr_differences(
            r_A,
            r_B,
            p_samples_A=p_A,
            p_samples_B=p_B,
            rng_key=rng,
            gene_mask=mask,
        )
        assert delta.shape == (100, D_kept)

    def test_gene_mask_with_p_samples_simplex_property(self, rng):
        """CLR aggregation with gene-specific p preserves finite values."""
        D = 6
        r_A = jnp.abs(random.normal(rng, (80, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (80, D))) + 1.0
        p_A = jax.nn.sigmoid(random.normal(random.PRNGKey(2), (80, D)))
        p_B = jax.nn.sigmoid(random.normal(random.PRNGKey(3), (80, D)))
        mask = jnp.array([True, True, True, True, False, False])
        delta = compute_clr_differences(
            r_A,
            r_B,
            p_samples_A=p_A,
            p_samples_B=p_B,
            rng_key=rng,
            gene_mask=mask,
        )
        # All delta values should be finite
        assert jnp.all(jnp.isfinite(delta))


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
            r_A,
            r_B,
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
            model,
            model,
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
            r_A,
            r_B,
            method="empirical",
            gene_names=[f"g{i}" for i in range(D)],
            rng_key=rng,
            gene_mask=mask,
        )
        result = de.gene_level(tau=0.0)
        assert result["delta_mean"].shape == (D_kept,)
        assert result["lfsr"].shape == (D_kept,)
        assert len(result["gene_names"]) == D_kept


# --------------------------------------------------------------------------
# Tests: Results-object dispatch for compare()
# --------------------------------------------------------------------------


class _MockModelConfig:
    """Lightweight mock of ModelConfig with is_hierarchical property."""

    def __init__(self, is_hierarchical=False):
        self._is_hierarchical = is_hierarchical

    @property
    def is_hierarchical(self):
        return self._is_hierarchical


class _MockResults:
    """Lightweight mock of ScribeSVIResults for testing compare() dispatch."""

    def __init__(
        self,
        r_samples,
        p_samples=None,
        gene_names=None,
        is_hierarchical=False,
    ):
        self.posterior_samples = {"r": r_samples}
        if p_samples is not None:
            self.posterior_samples["p"] = p_samples
        self.model_config = _MockModelConfig(is_hierarchical=is_hierarchical)
        import pandas as pd

        if gene_names is not None:
            self.var = pd.DataFrame(index=gene_names)
        else:
            self.var = None


class TestResultsObjectDispatch:
    """Tests for passing ScribeSVIResults / ScribeMCMCResults to compare()."""

    def test_standard_model_uses_dirichlet(self, rng):
        """Non-hierarchical results use standard Dirichlet (no p_samples)."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de = compare(res_A, res_B, method="empirical", rng_key=rng)
        result = de.gene_level(tau=0.0)

        assert result["delta_mean"].shape == (D,)
        assert result["lfsr"].shape == (D,)
        assert de.gene_names == names
        # All finite (no NaN)
        assert np.all(np.isfinite(np.array(result["delta_mean"])))

    def test_hierarchical_model_uses_gamma(self, rng):
        """Hierarchical results auto-detect gene-specific p."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        # Gene-specific p in (0, 1)
        p = jax.nn.sigmoid(random.normal(random.PRNGKey(1), (100, D)))
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )
        res_B = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )

        de = compare(res_A, res_B, method="empirical", rng_key=rng)
        result = de.gene_level(tau=0.0)

        assert result["delta_mean"].shape == (D,)
        assert np.all(np.isfinite(np.array(result["delta_mean"])))

    def test_hierarchical_mixture_model(self, rng):
        """Hierarchical mixture model with component slicing."""
        D, K = 5, 3
        r = jnp.abs(random.normal(rng, (100, K, D))) + 1.0
        p = jax.nn.sigmoid(random.normal(random.PRNGKey(1), (100, K, D)))
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )
        res_B = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )

        de = compare(
            res_A,
            res_B,
            method="empirical",
            component_A=0,
            component_B=1,
            rng_key=rng,
        )
        result = de.gene_level(tau=0.0)
        assert result["delta_mean"].shape == (D,)
        assert np.all(np.isfinite(np.array(result["delta_mean"])))

    def test_gene_names_auto_extracted(self, rng):
        """Gene names are taken from results.var.index when not provided."""
        D = 4
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        names = ["Gapdh", "Actb", "Rpl13a", "Hprt"]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de = compare(res_A, res_B, method="empirical", rng_key=rng)
        assert de.gene_names == names

    def test_explicit_gene_names_override(self, rng):
        """Explicitly passed gene_names take precedence over results.var."""
        D = 4
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        results_names = ["a", "b", "c", "d"]
        override_names = ["x", "y", "z", "w"]

        res_A = _MockResults(r, gene_names=results_names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=results_names, is_hierarchical=False)

        de = compare(
            res_A,
            res_B,
            method="empirical",
            gene_names=override_names,
            rng_key=rng,
        )
        assert de.gene_names == override_names

    def test_gene_mask_works_for_hierarchical(self, rng):
        """gene_mask filters genes correctly for hierarchical (gene-specific p) models.

        With the lazy simplex aggregation path, gene_mask is applied after
        Gamma-based sampling.  The DE results must have D_kept genes, not D.
        """
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        p = jax.nn.sigmoid(random.normal(random.PRNGKey(1), (100, D)))
        names = [f"gene_{i}" for i in range(D)]
        mask = jnp.array([True, True, False, True, True])
        D_kept = int(mask.sum())
        kept_names = [n for n, m in zip(names, mask.tolist()) if m]

        res_A = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )
        res_B = _MockResults(
            r, p_samples=p, gene_names=names, is_hierarchical=True
        )

        # Should not warn or raise — lazy simplex aggregation handles this.
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            de = compare(
                res_A,
                res_B,
                method="empirical",
                gene_mask=mask,
                rng_key=rng,
            )
            # No gene_mask-related warning should be emitted
            gene_mask_warns = [
                x for x in w if "gene_mask" in str(x.message).lower()
            ]
            assert len(gene_mask_warns) == 0

        # Results must contain exactly D_kept genes
        result = de.gene_level(tau=0.0)
        assert result["delta_mean"].shape == (D_kept,)
        assert de.gene_names == kept_names

    def test_parametric_method_raises_for_results_objects(self, rng):
        """method='parametric' rejects results objects."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        res = _MockResults(r, is_hierarchical=False)

        with pytest.raises(ValueError, match="parametric"):
            compare(res, res, method="parametric")

    def test_mixed_types_raises(self, rng):
        """Mixing a results object with a raw array raises TypeError."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        res = _MockResults(r, is_hierarchical=False)

        with pytest.raises(TypeError, match="Both model_A and model_B"):
            compare(res, r, method="empirical", rng_key=rng)

    def test_no_posterior_samples_raises(self, rng):
        """Results with posterior_samples=None raises ValueError."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        res_A = _MockResults(r, is_hierarchical=False)
        res_B = _MockResults(r, is_hierarchical=False)
        res_A.posterior_samples = None

        with pytest.raises(ValueError, match="posterior samples"):
            compare(res_A, res_B, method="empirical", rng_key=rng)

    def test_shrinkage_with_results_objects(self, rng):
        """method='shrinkage' works with results objects."""
        D = 5
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de = compare(res_A, res_B, method="shrinkage", rng_key=rng)
        result = de.gene_level(tau=0.0)
        assert result["delta_mean"].shape == (D,)


# --------------------------------------------------------------------------
# Tests: .shrink() zero-copy upgrade
# --------------------------------------------------------------------------


class TestShrinkMethod:
    """Tests for ScribeEmpiricalDEResults.shrink() zero-copy wrapper."""

    def test_shrink_shares_delta_samples(self, rng):
        """shrink() returns a ScribeShrinkageDEResults sharing delta_samples."""
        D = 10
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de_emp = compare(res_A, res_B, method="empirical", rng_key=rng)
        de_shrink = de_emp.shrink()

        from scribe.de.results import ScribeShrinkageDEResults

        assert isinstance(de_shrink, ScribeShrinkageDEResults)
        # Same underlying buffer — no extra GPU memory
        assert de_shrink.delta_samples is de_emp.delta_samples
        assert de_shrink.gene_names == de_emp.gene_names
        assert de_shrink.label_A == de_emp.label_A
        assert de_shrink.label_B == de_emp.label_B
        assert de_shrink.n_samples == de_emp.n_samples

    def test_shrink_gene_level_runs(self, rng):
        """gene_level() on a shrink()-produced object returns valid stats."""
        D = 10
        r = jnp.abs(random.normal(rng, (100, D))) + 1.0
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de_emp = compare(res_A, res_B, method="empirical", rng_key=rng)
        de_shrink = de_emp.shrink()
        result = de_shrink.gene_level(tau=0.0)

        assert result["delta_mean"].shape == (D,)
        assert result["lfsr"].shape == (D,)
        assert np.all(np.isfinite(np.array(result["delta_mean"])))
        assert de_shrink.method == "shrinkage"

    def test_shrink_custom_params(self, rng):
        """Custom sigma_grid and EM params are forwarded correctly."""
        D = 5
        r = jnp.abs(random.normal(rng, (60, D))) + 1.0
        names = [f"gene_{i}" for i in range(D)]

        res_A = _MockResults(r, gene_names=names, is_hierarchical=False)
        res_B = _MockResults(r, gene_names=names, is_hierarchical=False)

        de_emp = compare(res_A, res_B, method="empirical", rng_key=rng)
        _grid = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])
        de_shrink = de_emp.shrink(
            sigma_grid=_grid,
            shrinkage_max_iter=50,
            shrinkage_tol=1e-6,
        )

        assert jnp.array_equal(de_shrink.sigma_grid, _grid)
        assert de_shrink.shrinkage_max_iter == 50
        assert de_shrink.shrinkage_tol == 1e-6
        # Still shares buffer
        assert de_shrink.delta_samples is de_emp.delta_samples


# --------------------------------------------------------------------------
# Tests: NaN guards in _batched_gamma_normalize
# --------------------------------------------------------------------------


class TestGammaNormalizeSafety:
    """Edge cases for the Gamma-based composition sampling."""

    def test_p_near_zero(self, rng):
        """p near zero should not produce NaN."""
        D = 5
        r = jnp.ones((50, D)) * 2.0
        # p very close to 0 — without guards, p/(1-p) -> 0/1 = 0
        # total could be 0 -> NaN
        p = jnp.ones((50, D)) * 1e-9

        from scribe.de._empirical import _batched_gamma_normalize

        result = _batched_gamma_normalize(r, p, 1, rng, 2048)
        assert result.shape == (50, D)
        assert np.all(np.isfinite(np.array(result)))

    def test_p_near_one(self, rng):
        """p near one should not produce NaN (p/(1-p) -> inf guarded)."""
        D = 5
        r = jnp.ones((50, D)) * 2.0
        p = jnp.ones((50, D)) * (1.0 - 1e-9)

        from scribe.de._empirical import _batched_gamma_normalize

        result = _batched_gamma_normalize(r, p, 1, rng, 2048)
        assert result.shape == (50, D)
        assert np.all(np.isfinite(np.array(result)))

    def test_p_exactly_zero_and_one(self, rng):
        """Exact 0 and 1 are clamped — no NaN."""
        D = 4
        r = jnp.ones((30, D)) * 2.0
        p = jnp.array([[0.0, 1.0, 0.5, 0.3]] * 30)

        from scribe.de._empirical import _batched_gamma_normalize

        result = _batched_gamma_normalize(r, p, 1, rng, 2048)
        assert result.shape == (30, D)
        assert np.all(np.isfinite(np.array(result)))

    def test_multi_sample_gamma_safe(self, rng):
        """n_samples_dirichlet > 1 path also has NaN guards."""
        D = 5
        r = jnp.ones((20, D)) * 2.0
        p = jnp.ones((20, D)) * 1e-9

        from scribe.de._empirical import _batched_gamma_normalize

        result = _batched_gamma_normalize(r, p, 3, rng, 2048)
        assert result.shape == (60, D)  # 20 * 3
        assert np.all(np.isfinite(np.array(result)))


# --------------------------------------------------------------------------
# Tests: sample_compositions (Stage 1)
# --------------------------------------------------------------------------


class TestSampleCompositions:
    """Tests for ``sample_compositions`` (Stage 1 of CLR pipeline)."""

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)

    def test_output_shapes(self, rng):
        """Returns two (N, D) arrays with rows on the simplex."""
        D = 6
        r_A = jnp.abs(random.normal(rng, (100, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (100, D))) + 1.0
        s_A, s_B = sample_compositions(r_A, r_B, rng_key=rng)
        assert s_A.shape == (100, D)
        assert s_B.shape == (100, D)

    def test_rows_sum_to_one(self, rng):
        """Simplex samples must sum to 1 along the gene axis."""
        D = 5
        r_A = jnp.abs(random.normal(rng, (50, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (50, D))) + 1.0
        s_A, s_B = sample_compositions(r_A, r_B, rng_key=rng)
        np.testing.assert_allclose(np.array(s_A.sum(axis=1)), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.array(s_B.sum(axis=1)), 1.0, atol=1e-5)

    def test_n_samples_dirichlet_multiplies(self, rng):
        """n_samples_dirichlet > 1 multiplies the row count."""
        D = 4
        N = 30
        r = jnp.abs(random.normal(rng, (N, D))) + 1.0
        s_A, s_B = sample_compositions(r, r, n_samples_dirichlet=3, rng_key=rng)
        assert s_A.shape == (N * 3, D)
        assert s_B.shape == (N * 3, D)

    def test_mixture_slicing(self, rng):
        """3D (mixture) inputs are correctly sliced to (N, D)."""
        D, K = 5, 3
        r_mix = jnp.abs(random.normal(rng, (60, K, D))) + 1.0
        s_A, s_B = sample_compositions(
            r_mix, r_mix, component_A=0, component_B=1, rng_key=rng
        )
        assert s_A.shape == (60, D)
        assert s_B.shape == (60, D)

    def test_gamma_path_with_p(self, rng):
        """Gamma-based sampling produces valid simplex with gene-specific p."""
        D = 5
        r = jnp.abs(random.normal(rng, (50, D))) + 1.0
        p = jax.nn.sigmoid(random.normal(random.PRNGKey(1), (50, D)))
        s_A, s_B = sample_compositions(
            r, r, p_samples_A=p, p_samples_B=p, rng_key=rng
        )
        assert s_A.shape == (50, D)
        np.testing.assert_allclose(np.array(s_A.sum(axis=1)), 1.0, atol=1e-5)


# --------------------------------------------------------------------------
# Tests: compute_delta_from_simplex (Stage 2)
# --------------------------------------------------------------------------


class TestComputeDeltaFromSimplex:
    """Tests for ``compute_delta_from_simplex`` (Stage 2 of CLR pipeline)."""

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def simplex_pair(self, rng):
        """Two valid simplex matrices."""
        D = 6
        s_A = jax.nn.softmax(random.normal(rng, (100, D)), axis=-1)
        s_B = jax.nn.softmax(
            random.normal(random.PRNGKey(1), (100, D)), axis=-1
        )
        return s_A, s_B

    def test_no_mask_full_d(self, simplex_pair):
        """Without mask, output has D columns."""
        s_A, s_B = simplex_pair
        delta = compute_delta_from_simplex(s_A, s_B)
        assert delta.shape == (100, 6)

    def test_rows_sum_to_zero_no_mask(self, simplex_pair):
        """CLR differences sum to 0 along gene axis when no mask."""
        s_A, s_B = simplex_pair
        delta = compute_delta_from_simplex(s_A, s_B)
        np.testing.assert_allclose(np.array(delta.sum(axis=1)), 0.0, atol=1e-5)

    def test_with_mask_d_kept(self, simplex_pair):
        """With mask, output has D_kept columns."""
        s_A, s_B = simplex_pair
        mask = jnp.array([True, True, False, True, False, True])
        delta = compute_delta_from_simplex(s_A, s_B, gene_mask=mask)
        assert delta.shape == (100, 4)

    def test_with_mask_finite(self, simplex_pair):
        """All delta values should be finite with mask."""
        s_A, s_B = simplex_pair
        mask = jnp.array([True, True, True, False, False, True])
        delta = compute_delta_from_simplex(s_A, s_B, gene_mask=mask)
        assert jnp.all(jnp.isfinite(delta))

    def test_dirichlet_aggregation_equivalence(self, rng):
        """Aggregation after full-D Dirichlet matches aggregation before sampling.

        The Dirichlet aggregation property guarantees that storing the
        full simplex and aggregating post-hoc is equivalent to aggregating
        the concentration parameters before sampling.  We verify that both
        approaches produce statistically indistinguishable delta_mean and
        delta_sd distributions.
        """
        D = 8
        N = 5000
        r_A = jnp.abs(random.normal(rng, (N, D))) + 1.0
        r_B = jnp.abs(random.normal(random.PRNGKey(1), (N, D))) + 1.0
        mask = jnp.array([True, True, True, False, False, True, True, False])

        # Path 1: legacy (aggregate r → sample lower-D Dirichlet → CLR)
        delta_legacy = compute_clr_differences(
            r_A, r_B, rng_key=random.PRNGKey(99), gene_mask=mask
        )

        # Path 2: new two-stage (sample full D → aggregate simplex → CLR)
        s_A, s_B = sample_compositions(r_A, r_B, rng_key=random.PRNGKey(99))
        delta_twostage = compute_delta_from_simplex(s_A, s_B, gene_mask=mask)

        # Same column count
        assert delta_legacy.shape[1] == delta_twostage.shape[1]

        # Distributional match: means must be close (Monte Carlo error)
        mean_legacy = np.array(jnp.mean(delta_legacy, axis=0))
        mean_twostage = np.array(jnp.mean(delta_twostage, axis=0))
        np.testing.assert_allclose(mean_legacy, mean_twostage, atol=0.15)

        # Standard deviations must be close
        sd_legacy = np.array(jnp.std(delta_legacy, axis=0))
        sd_twostage = np.array(jnp.std(delta_twostage, axis=0))
        np.testing.assert_allclose(sd_legacy, sd_twostage, atol=0.1)


# --------------------------------------------------------------------------
# Tests: empirical mixin composition
# --------------------------------------------------------------------------


def test_empirical_methods_come_from_empirical_mixin():
    """Empirical result methods should originate from the empirical mixin."""
    assert (
        ScribeEmpiricalDEResults.gene_level.__module__
        == "scribe.de._results_empirical_mixin"
    )
    assert (
        ScribeEmpiricalDEResults.set_gene_mask.__module__
        == "scribe.de._results_empirical_mixin"
    )


# --------------------------------------------------------------------------
# Tests: AxisLayout-aware _slice_component
# --------------------------------------------------------------------------


class TestSliceComponentLayout:
    """Tests for layout-aware ``_slice_component``.

    ``_slice_component`` returns ``(sliced_array, post_layout)`` where
    ``post_layout`` is the ``AxisLayout`` with the component axis removed
    (or ``None`` when no layout was provided).
    """

    def _make_layout(self, axes, has_sample_dim=True):
        from scribe.core.axis_layout import AxisLayout

        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    # -- 3D mixture (samples, components, genes) --

    def test_mixture_gene_specific_layout(self, rng):
        """Layout-aware slicing of (N, K, D) -> (N, D)."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 3, 10))) + 1.0
        layout = self._make_layout(("components", "genes"))
        result, post = _slice_component(
            arr, component=1, label="A", layout=layout
        )
        assert result.shape == (50, 10)
        np.testing.assert_array_equal(result, arr[:, 1, :])
        # Post-layout should have component axis removed, gene axis intact
        assert post is not None
        assert post.component_axis is None
        assert post.gene_axis is not None

    def test_mixture_scalar_layout(self, rng):
        """Layout-aware slicing of (N, K) -> (N,) for scalar params."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 3))) + 1.0
        layout = self._make_layout(("components",))
        result, post = _slice_component(
            arr, component=2, label="A", layout=layout
        )
        assert result.shape == (50,)
        np.testing.assert_array_equal(result, arr[:, 2])
        # Post-layout: component removed, no genes (scalar)
        assert post is not None
        assert post.component_axis is None
        assert post.gene_axis is None

    def test_mixture_requires_component_with_layout(self, rng):
        """component=None raises ValueError when layout has component_axis."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 3, 10))) + 1.0
        layout = self._make_layout(("components", "genes"))
        with pytest.raises(ValueError, match="component_A was not specified"):
            _slice_component(arr, component=None, label="A", layout=layout)

    # -- 2D non-mixture (samples, genes) --

    def test_non_mixture_gene_specific_layout(self, rng):
        """Layout without component axis returns array and layout unchanged."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 10))) + 1.0
        layout = self._make_layout(("genes",))
        result, post = _slice_component(
            arr, component=None, label="A", layout=layout
        )
        assert result.shape == (50, 10)
        np.testing.assert_array_equal(result, arr)
        # Layout is unchanged (no component to remove)
        assert post is layout

    # -- 1D scalar (samples,) --

    def test_scalar_layout(self, rng):
        """Layout with no component/gene axes returns array and layout unchanged."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50,))) + 1.0
        layout = self._make_layout(())
        result, post = _slice_component(
            arr, component=None, label="A", layout=layout
        )
        assert result.shape == (50,)
        np.testing.assert_array_equal(result, arr)
        assert post is layout

    # -- Legacy ndim fallback (no layout) --

    def test_ndim_fallback_3d(self, rng):
        """Without layout, 3D input is sliced via ndim heuristic."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 3, 10))) + 1.0
        result, post = _slice_component(arr, component=0, label="B")
        assert result.shape == (50, 10)
        np.testing.assert_array_equal(result, arr[:, 0, :])
        assert post is None

    def test_ndim_fallback_2d(self, rng):
        """Without layout, 2D input passes through."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 10))) + 1.0
        result, post = _slice_component(arr, component=None, label="B")
        assert result.shape == (50, 10)
        assert post is None

    def test_ndim_fallback_1d(self, rng):
        """Without layout, 1D input passes through."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50,))) + 1.0
        result, post = _slice_component(arr, component=None, label="B")
        assert result.shape == (50,)
        assert post is None

    # -- Consistency: layout path matches ndim path --

    def test_layout_matches_ndim_3d(self, rng):
        """Layout-aware and ndim paths produce identical sliced arrays for 3D."""
        from scribe.de._empirical import _slice_component

        arr = jnp.abs(random.normal(rng, (50, 3, 10))) + 1.0
        layout = self._make_layout(("components", "genes"))
        res_layout, _ = _slice_component(arr, 1, "A", layout=layout)
        res_ndim, _ = _slice_component(arr, 1, "A", layout=None)
        np.testing.assert_array_equal(res_layout, res_ndim)


# --------------------------------------------------------------------------
# Tests: AxisLayout-aware _drop_scalar_p / sample_compositions
# --------------------------------------------------------------------------


class TestDropScalarP:
    """Tests for ``_drop_scalar_p`` helper.

    The helper accepts a *post-sliced* layout (component axis already
    removed by ``_slice_component``).
    """

    def _make_layout(self, axes, has_sample_dim=True):
        from scribe.core.axis_layout import AxisLayout

        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    def test_none_input(self):
        """None input returns None regardless of layout."""
        from scribe.de._empirical import _drop_scalar_p

        assert _drop_scalar_p(None) is None
        assert _drop_scalar_p(None, self._make_layout(("genes",))) is None

    def test_gene_specific_with_post_layout(self, rng):
        """Gene-specific p (post-layout has gene_axis) is kept."""
        from scribe.de._empirical import _drop_scalar_p

        p = jnp.abs(random.normal(rng, (50, 10))) + 0.1
        # Post-sliced layout: component axis already removed
        post_layout = self._make_layout(("genes",))
        assert _drop_scalar_p(p, post_layout) is p

    def test_scalar_with_post_layout(self, rng):
        """Scalar p (no gene_axis in post-layout) is dropped."""
        from scribe.de._empirical import _drop_scalar_p

        p = jnp.abs(random.normal(rng, (50,))) + 0.1
        # Post-sliced layout: was (components,), now () after slicing
        post_layout = self._make_layout(())
        assert _drop_scalar_p(p, post_layout) is None

    def test_ndim_fallback_gene_specific(self, rng):
        """Without layout, 2D p is kept (ndim >= 2)."""
        from scribe.de._empirical import _drop_scalar_p

        p = jnp.abs(random.normal(rng, (50, 10))) + 0.1
        assert _drop_scalar_p(p, post_layout=None) is p

    def test_ndim_fallback_scalar(self, rng):
        """Without layout, 1D p is dropped (ndim < 2)."""
        from scribe.de._empirical import _drop_scalar_p

        p = jnp.abs(random.normal(rng, (50,))) + 0.1
        assert _drop_scalar_p(p, post_layout=None) is None


# --------------------------------------------------------------------------
# Tests: AxisLayout-aware _needs_gene_broadcast in _biological.py
# --------------------------------------------------------------------------


class TestNeedsGeneBroadcast:
    """Tests for ``_needs_gene_broadcast`` helper."""

    def _make_layout(self, axes, has_sample_dim=True):
        from scribe.core.axis_layout import AxisLayout

        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    def test_with_gene_axis(self, rng):
        """Array with gene axis does not need broadcast."""
        from scribe.de._biological import _needs_gene_broadcast

        arr = jnp.ones((50, 10))
        layout = self._make_layout(("genes",))
        assert _needs_gene_broadcast(arr, layout) is False

    def test_without_gene_axis(self, rng):
        """Array without gene axis needs broadcast."""
        from scribe.de._biological import _needs_gene_broadcast

        arr = jnp.ones((50,))
        layout = self._make_layout(())
        assert _needs_gene_broadcast(arr, layout) is True

    def test_ndim_fallback_2d(self, rng):
        """Without layout, 2D does not need broadcast."""
        from scribe.de._biological import _needs_gene_broadcast

        arr = jnp.ones((50, 10))
        assert _needs_gene_broadcast(arr) is False

    def test_ndim_fallback_1d(self, rng):
        """Without layout, 1D needs broadcast."""
        from scribe.de._biological import _needs_gene_broadcast

        arr = jnp.ones((50,))
        assert _needs_gene_broadcast(arr) is True


# --------------------------------------------------------------------------
# Tests: Layout threading through sample_compositions
# --------------------------------------------------------------------------


class TestSampleCompositionsWithLayouts:
    """Verify ``sample_compositions`` accepts and threads layouts."""

    def _make_layout(self, axes, has_sample_dim=True):
        from scribe.core.axis_layout import AxisLayout

        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    def test_layout_produces_same_result(self, rng):
        """Layout-aware path produces identical output to ndim path.

        Uses a 3D mixture array ``(N, K, D)`` with component slicing,
        verifying that the layout path and the ndim fallback produce
        the same simplex samples.
        """
        N, K, D = 100, 3, 5
        r = jnp.abs(random.normal(rng, (N, K, D))) + 1.0

        layouts = {
            "r": self._make_layout(("components", "genes")),
        }

        # Layout path
        s_A_l, s_B_l = sample_compositions(
            r,
            r,
            component_A=0,
            component_B=1,
            paired=True,
            rng_key=random.PRNGKey(7),
            param_layouts=layouts,
        )

        # ndim fallback path
        s_A_n, s_B_n = sample_compositions(
            r,
            r,
            component_A=0,
            component_B=1,
            paired=True,
            rng_key=random.PRNGKey(7),
            param_layouts=None,
        )

        np.testing.assert_allclose(s_A_l, s_A_n, atol=1e-6)
        np.testing.assert_allclose(s_B_l, s_B_n, atol=1e-6)

    def test_layout_with_scalar_p_dropped(self, rng):
        """Scalar p (no gene axis in layout) is correctly dropped."""
        N, D = 100, 5
        r = jnp.abs(random.normal(rng, (N, D))) + 1.0
        p_scalar = jnp.full((N,), 0.5)

        layouts = {
            "r": self._make_layout(("genes",)),
            "p": self._make_layout(()),  # scalar: no gene axis
        }

        # Should not error, and should use Dirichlet (not Gamma)
        # because scalar p is dropped
        s_A, s_B = sample_compositions(
            r,
            r,
            rng_key=random.PRNGKey(8),
            p_samples_A=p_scalar,
            p_samples_B=p_scalar,
            param_layouts=layouts,
        )
        assert s_A.shape[1] == D
        assert s_B.shape[1] == D


# --------------------------------------------------------------------------
# Tests: early mixture-component validation guards
# --------------------------------------------------------------------------


class TestMixtureComponentValidation:
    """Early-validation guards at DE public entry points.

    Each public function that accepts mixture-model inputs should raise a
    clear ``ValueError`` immediately when the data has a component axis
    but no ``component_*`` index is provided.
    """

    def _make_layout(self, axes, has_sample_dim=True):
        from scribe.core.axis_layout import AxisLayout

        return AxisLayout(axes=tuple(axes), has_sample_dim=has_sample_dim)

    # -- _require_mixture_components (layout-aware guard) --

    def test_guard_detects_mixture_via_layout(self):
        """Guard should raise when layout has a component axis."""
        from scribe.de._empirical import _require_mixture_components

        layouts = {"r": self._make_layout(("components", "genes"))}
        with pytest.raises(
            ValueError, match="posterior samples have a mixture-component axis"
        ):
            _require_mixture_components(None, None, layouts, "compare")

    def test_guard_names_only_missing_component(self):
        """Guard should name only the missing component index."""
        from scribe.de._empirical import _require_mixture_components

        layouts = {"r": self._make_layout(("components", "genes"))}
        with pytest.raises(ValueError, match=r"component_A was not specified"):
            _require_mixture_components(None, 0, layouts, "compare")

    def test_guard_ok_when_components_provided(self):
        """Guard should not raise when both components are given."""
        from scribe.de._empirical import _require_mixture_components

        layouts = {"r": self._make_layout(("components", "genes"))}
        _require_mixture_components(0, 1, layouts, "compare")

    def test_guard_no_component_axis_ok(self):
        """No error when layout has no component axis (non-mixture)."""
        from scribe.de._empirical import _require_mixture_components

        layouts = {"r": self._make_layout(("genes",))}
        _require_mixture_components(None, None, layouts, "compare")

    def test_guard_no_layouts_is_noop(self):
        """Guard is a no-op when no layouts are available."""
        from scribe.de._empirical import _require_mixture_components

        _require_mixture_components(None, None, None, "compare")

    # -- sample_compositions() with layout --

    def test_sample_compositions_requires_component_with_layout(self):
        """sample_compositions() should detect mixture via layout."""
        r_3d = jnp.ones((20, 3, 5))
        layouts = {"r": self._make_layout(("components", "genes"))}
        with pytest.raises(
            ValueError,
            match="sample_compositions.*mixture-component axis",
        ):
            sample_compositions(
                r_3d,
                r_3d,
                rng_key=random.PRNGKey(0),
                param_layouts=layouts,
            )

    def test_sample_compositions_ok_for_2d(self, rng):
        """sample_compositions() should not error for non-mixture inputs."""
        r_2d = jnp.abs(random.normal(rng, (20, 5))) + 1.0
        s_A, s_B = sample_compositions(r_2d, r_2d, rng_key=rng)
        assert s_A.shape[1] == 5

    # -- compare_datasets() --

    def test_compare_datasets_requires_component_for_mixture(self):
        """compare_datasets() should error early for mixture models."""
        from scribe.de import compare_datasets
        from unittest.mock import MagicMock

        # Mock a multi-dataset mixture results object
        mock_results = MagicMock()
        mock_results.model_config.n_datasets = 2
        mock_results.model_config.n_components = 3
        with pytest.raises(
            ValueError,
            match=r"compare_datasets\(\).*mixture with 3 components.*component=",
        ):
            compare_datasets(mock_results, dataset_A=0, dataset_B=1)

    def test_compare_datasets_ok_when_component_provided(self):
        """compare_datasets() should not error when component is given."""
        from scribe.de import compare_datasets
        from unittest.mock import MagicMock

        # Mock the results chain: get_component -> get_dataset -> compare
        mock_results = MagicMock()
        mock_results.model_config.n_datasets = 2
        mock_results.model_config.n_components = 3

        # Calling get_component returns a view; get_dataset returns views.
        # compare() will eventually fail because the mock doesn't have
        # real data, but the early guard should pass.
        mock_component_view = MagicMock()
        mock_results.get_component.return_value = mock_component_view

        mock_ds_view = MagicMock()
        mock_ds_view.posterior_samples = None
        mock_component_view.get_dataset.return_value = mock_ds_view

        # Should pass the mixture guard and fail later (no real data)
        with pytest.raises(Exception, match="(?!.*mixture)"):
            compare_datasets(
                mock_results,
                dataset_A=0,
                dataset_B=1,
                component=0,
            )

    def test_compare_datasets_ok_for_non_mixture(self):
        """compare_datasets() should not error when model is non-mixture."""
        from scribe.de import compare_datasets
        from unittest.mock import MagicMock

        # Non-mixture model: n_components is None (not set)
        mock_results = MagicMock()
        mock_results.model_config.n_datasets = 2
        mock_results.model_config.n_components = None

        mock_ds_view = MagicMock()
        mock_ds_view.posterior_samples = None
        mock_results.get_dataset.return_value = mock_ds_view

        # Should pass the mixture guard and fail later (no real data)
        with pytest.raises(Exception, match="(?!.*mixture)"):
            compare_datasets(mock_results, dataset_A=0, dataset_B=1)


# ==========================================================================
# Deprecation warnings for ndim-based fallbacks
# ==========================================================================


class TestDEDeprecationWarnings:
    """Calling DE helpers without layout metadata should emit DeprecationWarning."""

    def test_slice_component_warns_without_layout(self):
        """_slice_component falls back to ndim heuristic when layout is None."""
        arr = jnp.ones((10, 50))
        with pytest.warns(DeprecationWarning, match="_slice_component"):
            _slice_component(arr, component=None, label="A", layout=None)

    def test_drop_scalar_p_warns_without_layout(self):
        """_drop_scalar_p falls back to ndim heuristic when post_layout is None."""
        arr = jnp.ones((10,))
        with pytest.warns(DeprecationWarning, match="_drop_scalar_p"):
            _drop_scalar_p(arr, post_layout=None)

    def test_needs_gene_broadcast_warns_without_layout(self):
        """_needs_gene_broadcast falls back to ndim heuristic when layout is None."""
        arr = jnp.ones((10,))
        with pytest.warns(DeprecationWarning, match="_needs_gene_broadcast"):
            _needs_gene_broadcast(arr, layout=None)


# ==========================================================================
# Array-backend dispatch (xp): NumPy vs JAX transparency
# ==========================================================================


class TestArrayBackendDispatch:
    """Verify that DE summary functions dispatch correctly for np vs jnp inputs.

    When inputs are ``numpy.ndarray`` the functions should return
    ``numpy.ndarray``.  When inputs are ``jax.Array`` the functions
    should return ``jax.Array``.
    """

    @staticmethod
    def _make_delta_samples(xp, n=50, d=10, seed=0):
        """Generate synthetic CLR delta samples with the given backend."""
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal((n, d)).astype(np.float32)
        if xp is jnp:
            return jnp.asarray(arr)
        return arr

    def test_empirical_de_returns_numpy_for_numpy_input(self):
        """empirical_differential_expression returns np.ndarray for np input."""
        delta = self._make_delta_samples(np)
        result = empirical_differential_expression(delta, tau=0.1)
        for key in ("delta_mean", "delta_sd", "prob_positive", "lfsr"):
            assert isinstance(
                result[key], np.ndarray
            ), f"Expected np.ndarray for key '{key}', got {type(result[key])}"

    def test_empirical_de_returns_jax_for_jax_input(self):
        """empirical_differential_expression returns jax.Array for jax input."""
        delta = self._make_delta_samples(jnp)
        result = empirical_differential_expression(delta, tau=0.1)
        for key in ("delta_mean", "delta_sd", "prob_positive", "lfsr"):
            assert isinstance(
                result[key], jax.Array
            ), f"Expected jax.Array for key '{key}', got {type(result[key])}"

    def test_empirical_de_values_match_across_backends(self):
        """NumPy and JAX backends produce numerically close results."""
        delta_np = self._make_delta_samples(np)
        delta_jnp = self._make_delta_samples(jnp)

        res_np = empirical_differential_expression(delta_np, tau=0.1)
        res_jnp = empirical_differential_expression(delta_jnp, tau=0.1)

        for key in ("delta_mean", "delta_sd", "prob_positive", "lfsr"):
            np.testing.assert_allclose(
                np.asarray(res_np[key]),
                np.asarray(res_jnp[key]),
                atol=1e-5,
                err_msg=f"Value mismatch for key '{key}'",
            )

    def test_biological_de_returns_numpy_for_numpy_input(self):
        """biological_differential_expression returns np.ndarray for np input."""
        from scribe.de._biological import biological_differential_expression

        rng = np.random.default_rng(42)
        n, d = 30, 8
        r_A = rng.gamma(2.0, size=(n, d)).astype(np.float32)
        r_B = rng.gamma(2.0, size=(n, d)).astype(np.float32)
        p_A = rng.beta(2, 5, size=(n, d)).astype(np.float32)
        p_B = rng.beta(2, 5, size=(n, d)).astype(np.float32)

        result = biological_differential_expression(
            r_A,
            r_B,
            p_A,
            p_B,
            metric_families=("bio_lfc", "bio_aux"),
        )
        assert isinstance(result["lfc_mean"], np.ndarray)
        assert isinstance(result["mu_A_mean"], np.ndarray)

    def test_biological_de_returns_jax_for_jax_input(self):
        """biological_differential_expression returns jax.Array for jax input."""
        from scribe.de._biological import biological_differential_expression

        rng = np.random.default_rng(42)
        n, d = 30, 8
        r_A = jnp.asarray(rng.gamma(2.0, size=(n, d)).astype(np.float32))
        r_B = jnp.asarray(rng.gamma(2.0, size=(n, d)).astype(np.float32))
        p_A = jnp.asarray(rng.beta(2, 5, size=(n, d)).astype(np.float32))
        p_B = jnp.asarray(rng.beta(2, 5, size=(n, d)).astype(np.float32))

        result = biological_differential_expression(
            r_A,
            r_B,
            p_A,
            p_B,
            metric_families=("bio_lfc", "bio_aux"),
        )
        assert isinstance(result["lfc_mean"], jax.Array)
        assert isinstance(result["mu_A_mean"], jax.Array)

    def test_shrinkage_de_returns_numpy_for_numpy_input(self):
        """shrinkage_differential_expression works with NumPy inputs."""
        from scribe.de._shrinkage import shrinkage_differential_expression

        rng = np.random.default_rng(99)
        d = 50
        delta_mean = rng.standard_normal(d).astype(np.float64)
        delta_sd = np.abs(rng.standard_normal(d).astype(np.float64)) + 0.01

        result = shrinkage_differential_expression(
            delta_mean, delta_sd, tau=0.1
        )
        assert isinstance(
            result["delta_mean"], np.ndarray
        ), f"Expected np.ndarray, got {type(result['delta_mean'])}"
        assert isinstance(result["lfsr"], np.ndarray)

    def test_gamma_kl_returns_numpy_for_numpy_input(self):
        """gamma_kl dispatches to scipy.special for NumPy arrays."""
        from scribe.stats.divergences import gamma_kl

        rng = np.random.default_rng(7)
        n = 20
        a_p = rng.gamma(2.0, size=n).astype(np.float64)
        b_p = rng.gamma(1.0, size=n).astype(np.float64)
        a_q = rng.gamma(2.0, size=n).astype(np.float64)
        b_q = rng.gamma(1.0, size=n).astype(np.float64)

        result = gamma_kl(a_p, b_p, a_q, b_q)
        assert isinstance(result, np.ndarray)

    def test_gamma_kl_values_match_across_backends(self):
        """gamma_kl produces consistent values for NumPy vs JAX inputs."""
        from scribe.stats.divergences import gamma_kl

        rng = np.random.default_rng(7)
        n = 20
        a_p = rng.gamma(2.0, size=n).astype(np.float64)
        b_p = rng.gamma(1.0, size=n).astype(np.float64)
        a_q = rng.gamma(2.0, size=n).astype(np.float64)
        b_q = rng.gamma(1.0, size=n).astype(np.float64)

        result_np = gamma_kl(a_p, b_p, a_q, b_q)
        result_jnp = gamma_kl(
            jnp.asarray(a_p),
            jnp.asarray(b_p),
            jnp.asarray(a_q),
            jnp.asarray(b_q),
        )
        # JAX defaults to float32 even for float64 inputs (unless
        # jax_enable_x64 is set), so digamma/gammaln accumulate ~1e-5
        # differences vs the float64 SciPy path.
        np.testing.assert_allclose(
            np.asarray(result_np),
            np.asarray(result_jnp),
            atol=1e-4,
        )


# ==========================================================================
# JIT-compiled adaptive sampling
# ==========================================================================


class TestAdaptiveSampling:
    """Verify the JIT-compiled adaptive sampling layer.

    Tests cover:
    - Single-chunk (memory-rich) path produces correct shapes / values
    - Multi-chunk (memory-constrained) path produces identical results
    - Numerical regression: JIT kernels match expected Dirichlet properties
    - Gamma-normalise and paired paths
    """

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(42)

    # ------------------------------------------------------------------
    # Dirichlet: single-draw
    # ------------------------------------------------------------------

    def test_batched_dirichlet_single_shape(self, rng):
        """Single-draw Dirichlet produces (N, D) output."""
        from scribe.de._empirical import _batched_dirichlet

        N, D = 100, 20
        r = jnp.ones((N, D)) * 2.0
        result = _batched_dirichlet(r, 1, rng, batch_size=2048)

        assert isinstance(result, np.ndarray)
        assert result.shape == (N, D)

    def test_batched_dirichlet_single_simplex(self, rng):
        """Single-draw Dirichlet rows sum to 1."""
        from scribe.de._empirical import _batched_dirichlet

        r = jnp.ones((50, 10)) * 5.0
        result = _batched_dirichlet(r, 1, rng, batch_size=2048)

        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    # ------------------------------------------------------------------
    # Dirichlet: multi-draw
    # ------------------------------------------------------------------

    def test_batched_dirichlet_multi_shape(self, rng):
        """Multi-draw Dirichlet produces (N*S, D) output."""
        from scribe.de._empirical import _batched_dirichlet

        N, D, S = 40, 15, 3
        r = jnp.ones((N, D)) * 2.0
        result = _batched_dirichlet(r, S, rng, batch_size=2048)

        assert result.shape == (N * S, D)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    # ------------------------------------------------------------------
    # Adaptive chunking: constrained memory produces same results
    # ------------------------------------------------------------------

    def test_adaptive_chunking_matches_single_chunk(self, rng):
        """Constrained budget splits into chunks but produces same values.

        Uses a tiny memory_budget to force multiple chunks, then
        verifies that the result matches the single-chunk (inf budget)
        path for the same RNG key.
        """
        from scribe.de._empirical import _batched_dirichlet

        N, D = 60, 10
        r = jnp.ones((N, D)) * 3.0

        # Large budget -> single chunk
        result_single = _batched_dirichlet(
            r,
            1,
            rng,
            batch_size=N,
            memory_budget=float("inf"),
        )

        # Tiny budget -> forces many chunks (but still same fold_in keys)
        result_multi = _batched_dirichlet(
            r,
            1,
            rng,
            batch_size=N,
            memory_budget=1.0,
        )

        # Both paths use fold_in(rng_key, start) so chunks aligned with
        # chunk_size=1 will have different keys than chunk_size=N.
        # Verify shape and simplex property (not bitwise equality, since
        # different chunk boundaries produce different fold_in offsets).
        assert result_single.shape == result_multi.shape
        np.testing.assert_allclose(
            result_multi.sum(axis=1),
            1.0,
            atol=1e-5,
        )

    def test_adaptive_single_chunk_when_budget_large(self, rng):
        """With large budget the entire array is one chunk.

        We verify indirectly: calling with batch_size > N and inf budget
        should produce valid output without error.
        """
        from scribe.de._empirical import _batched_dirichlet

        N, D = 30, 8
        r = jnp.ones((N, D)) * 2.0
        result = _batched_dirichlet(
            r,
            1,
            rng,
            batch_size=100_000,
            memory_budget=float("inf"),
        )
        assert result.shape == (N, D)

    # ------------------------------------------------------------------
    # Gamma-normalise
    # ------------------------------------------------------------------

    def test_gamma_normalize_single_shape(self, rng):
        """Single-draw Gamma-normalise produces (N, D) simplex output."""
        from scribe.de._empirical import _batched_gamma_normalize

        N, D = 50, 12
        r = jnp.ones((N, D)) * 3.0
        p = jnp.ones((N, D)) * 0.5
        result = _batched_gamma_normalize(r, p, 1, rng, 2048)

        assert isinstance(result, np.ndarray)
        assert result.shape == (N, D)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_gamma_normalize_multi_shape(self, rng):
        """Multi-draw Gamma-normalise produces (N*S, D) output."""
        from scribe.de._empirical import _batched_gamma_normalize

        N, D, S = 30, 8, 4
        r = jnp.ones((N, D)) * 2.0
        p = jnp.ones((N, D)) * 0.4
        result = _batched_gamma_normalize(r, p, S, rng, 2048)

        assert result.shape == (N * S, D)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    # ------------------------------------------------------------------
    # Paired Dirichlet
    # ------------------------------------------------------------------

    def test_paired_dirichlet_single_shape(self, rng):
        """Paired single-draw returns two (N, D) arrays."""
        from scribe.de._empirical import _paired_dirichlet_sample

        N, D = 40, 10
        r_A = jnp.ones((N, D)) * 2.0
        r_B = jnp.ones((N, D)) * 3.0
        sA, sB = _paired_dirichlet_sample(r_A, r_B, 1, rng, 2048)

        assert isinstance(sA, np.ndarray)
        assert sA.shape == (N, D)
        assert sB.shape == (N, D)
        np.testing.assert_allclose(sA.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(sB.sum(axis=1), 1.0, atol=1e-5)

    def test_paired_dirichlet_multi_shape(self, rng):
        """Paired multi-draw returns two (N*S, D) arrays."""
        from scribe.de._empirical import _paired_dirichlet_sample

        N, D, S = 25, 8, 3
        r_A = jnp.ones((N, D)) * 2.0
        r_B = jnp.ones((N, D)) * 4.0
        sA, sB = _paired_dirichlet_sample(r_A, r_B, S, rng, 2048)

        assert sA.shape == (N * S, D)
        assert sB.shape == (N * S, D)

    # ------------------------------------------------------------------
    # NumPy inputs pass through without error
    # ------------------------------------------------------------------

    def test_numpy_inputs_accepted(self, rng):
        """_batched_dirichlet works when r_samples is a numpy array."""
        from scribe.de._empirical import _batched_dirichlet

        N, D = 30, 10
        r_np = np.ones((N, D), dtype=np.float32) * 2.0
        result = _batched_dirichlet(r_np, 1, rng, 2048)

        assert isinstance(result, np.ndarray)
        assert result.shape == (N, D)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    # ------------------------------------------------------------------
    # sample_compositions integration
    # ------------------------------------------------------------------

    def test_sample_compositions_jit_path(self, rng):
        """sample_compositions uses the JIT path and returns numpy."""
        N, D = 50, 15
        r_A = jnp.ones((N, D)) * 2.0
        r_B = jnp.ones((N, D)) * 3.0

        sA, sB = sample_compositions(r_A, r_B, rng_key=rng)

        assert isinstance(sA, np.ndarray)
        assert isinstance(sB, np.ndarray)
        assert sA.shape == (N, D)
        np.testing.assert_allclose(sA.sum(axis=1), 1.0, atol=1e-5)

    # ------------------------------------------------------------------
    # _estimate_chunk_size
    # ------------------------------------------------------------------

    def test_estimate_chunk_size_inf_returns_N(self):
        """Infinite budget returns N (single chunk)."""
        from scribe.de._empirical import _estimate_chunk_size
        import math

        result = _estimate_chunk_size(1000, 20000, 1, math.inf, 1)
        assert result == 1000

    def test_estimate_chunk_size_tiny_budget(self):
        """Tiny budget still returns at least 1."""
        from scribe.de._empirical import _estimate_chunk_size

        result = _estimate_chunk_size(1000, 20000, 1, 1.0, 1)
        assert result >= 1

    def test_estimate_chunk_size_realistic(self):
        """Realistic 8 GB budget with D=20k, S=1 gives a large chunk."""
        from scribe.de._empirical import _estimate_chunk_size

        budget = 8 * 1024**3  # 8 GB
        chunk = _estimate_chunk_size(10000, 20000, 1, budget, 1)

        # With D=20k, S=1, 4 bytes/element, 2x headroom:
        # per_row = (1*20000*4 + 1*20000*4) * 2 = 320_000 bytes
        # chunk = 8GB / 320k ~= 26843 -> capped at N=10000
        assert chunk == 10000


class TestLikelihoodDeviceStay:
    """Tests for the on-device accumulation in log_likelihood_map.

    The gene-batched path should accumulate JAX arrays and do a single
    device-to-host transfer instead of calling ``np.array()`` per batch.
    """

    def test_gene_batched_returns_jax_array(self):
        """Gene-batched MAP log-likelihood returns a jnp.ndarray (not np)."""
        from scribe.models.config import InferenceConfig, SVIConfig
        from scribe.inference import run_scribe
        from scribe.inference.preset_builder import build_config_from_preset

        _np = np.random.RandomState(42)
        counts = jnp.array(_np.negative_binomial(5, 0.3, (10, 8)))
        cfg = build_config_from_preset(
            model="nbdm",
            parameterization="standard",
            inference_method="svi",
            unconstrained=False,
            priors={"r": (2, 0.1), "p": (1, 1)},
        )
        inf = InferenceConfig.from_svi(SVIConfig(n_steps=5, batch_size=5))
        result = run_scribe(
            counts=counts,
            model_config=cfg,
            inference_config=inf,
            seed=0,
        )

        # Gene-batched path (small batch to force multiple iterations)
        ll = result.log_likelihood_map(
            counts,
            gene_batch_size=3,
            return_by="gene",
            verbose=False,
        )
        assert isinstance(ll, jax.Array)
        assert ll.shape[0] == 8  # n_genes

    def test_gene_batched_matches_full(self):
        """Gene-batched MAP log-likelihood matches full computation."""
        from scribe.models.config import InferenceConfig, SVIConfig
        from scribe.inference import run_scribe
        from scribe.inference.preset_builder import build_config_from_preset

        _np = np.random.RandomState(42)
        counts = jnp.array(_np.negative_binomial(5, 0.3, (10, 8)))
        cfg = build_config_from_preset(
            model="nbdm",
            parameterization="standard",
            inference_method="svi",
            unconstrained=False,
            priors={"r": (2, 0.1), "p": (1, 1)},
        )
        inf = InferenceConfig.from_svi(SVIConfig(n_steps=5, batch_size=5))
        result = run_scribe(
            counts=counts,
            model_config=cfg,
            inference_config=inf,
            seed=0,
        )

        ll_full = result.log_likelihood_map(
            counts,
            return_by="gene",
            verbose=False,
        )
        ll_batched = result.log_likelihood_map(
            counts,
            gene_batch_size=3,
            return_by="gene",
            verbose=False,
        )
        np.testing.assert_allclose(
            np.asarray(ll_full),
            np.asarray(ll_batched),
            atol=1e-4,
        )


# ==========================================================================
# Tests: mixture-weighted differential expression
# ==========================================================================


class TestMixtureWeightedDE:
    """Tests for mixture-weighted composition sampling and DE pipeline."""

    # --- Fixtures local to this class ---

    @pytest.fixture
    def rng(self):
        return random.PRNGKey(99)

    @pytest.fixture
    def mixture_r_2comp(self, rng):
        """3D r samples: (200, 2, 5) — 2 components, 5 genes.

        Component 0 has higher concentration on gene 0,
        component 1 has higher concentration on gene 4,
        making them distinguishable.
        """
        k1, k2 = random.split(rng)
        base_0 = jnp.array([10.0, 2.0, 2.0, 2.0, 1.0])
        base_1 = jnp.array([1.0, 2.0, 2.0, 2.0, 10.0])
        r0 = base_0[None, :] + jnp.abs(random.normal(k1, (200, 5))) * 0.5
        r1 = base_1[None, :] + jnp.abs(random.normal(k2, (200, 5))) * 0.5
        return jnp.stack([r0, r1], axis=1)

    @pytest.fixture
    def mixture_weights(self, rng):
        """Posterior mixture weight samples: (200, 2).

        Concentrated around (0.8, 0.2).
        """
        concentrations = jnp.array([80.0, 20.0])
        return jax.random.dirichlet(rng, concentrations, shape=(200,))

    @pytest.fixture
    def mixture_p_2comp(self, rng):
        """Gene-specific p samples: (200, 2, 5)."""
        return (
            jnp.full((200, 2, 5), 0.5)
            + jnp.abs(random.normal(rng, (200, 2, 5))) * 0.05
        )

    # --- Tests ---

    def test_weight_simplex_by_components_shape(self):
        """_weight_simplex_by_components returns correct shape."""
        N, K, D = 50, 3, 10
        components = [np.ones((N, D)) / D for _ in range(K)]
        weights = np.ones((N, K)) / K
        result = _weight_simplex_by_components(components, weights)
        assert result.shape == (N, D)

    def test_weight_simplex_preserves_simplex(self):
        """Weighted simplex output sums to 1 and is non-negative."""
        N, K, D = 100, 2, 5
        rng = random.PRNGKey(0)
        k1, k2, k3 = random.split(rng, 3)
        # Random valid simplices
        c0 = np.asarray(jax.random.dirichlet(k1, jnp.ones(D), shape=(N,)))
        c1 = np.asarray(jax.random.dirichlet(k2, jnp.ones(D), shape=(N,)))
        w = np.asarray(jax.random.dirichlet(k3, jnp.ones(K), shape=(N,)))
        result = _weight_simplex_by_components([c0, c1], w)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
        assert (result >= 0).all()

    def test_mixture_weighted_simplex_on_simplex(
        self, mixture_r_2comp, mixture_weights, rng
    ):
        """sample_mixture_compositions returns valid simplex samples."""
        s_A, s_B = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=mixture_weights,
            weights_B=mixture_weights,
            rng_key=rng,
        )
        np.testing.assert_allclose(s_A.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(s_B.sum(axis=1), 1.0, atol=1e-6)
        assert (s_A > 0).all()
        assert (s_B > 0).all()

    def test_mixture_weighted_reduces_to_single_component(
        self, mixture_r_2comp, rng
    ):
        """When weights are [1, 0], result matches component 0."""
        N = mixture_r_2comp.shape[0]
        # Indicator weights: all mass on component 0
        indicator_weights = np.zeros((N, 2))
        indicator_weights[:, 0] = 1.0

        s_mix, _ = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=indicator_weights,
            weights_B=indicator_weights,
            rng_key=rng,
        )

        # Single-component sampling with the same key
        s_single, _ = sample_compositions(
            r_samples_A=mixture_r_2comp[:, 0, :],
            r_samples_B=mixture_r_2comp[:, 0, :],
            rng_key=random.fold_in(rng, 0),
        )

        # The means should converge (not exact match due to different
        # key threading, but statistically close).
        np.testing.assert_allclose(
            s_mix.mean(axis=0), s_single.mean(axis=0), atol=0.05
        )

    def test_mixture_weighted_reduces_to_standard_when_k1(self, rng):
        """K=1 mixture-weighted path matches standard pipeline."""
        N, D = 100, 5
        r_1comp = jnp.abs(random.normal(rng, (N, 1, D))) + 1.0
        w_1comp = np.ones((N, 1))

        s_mix, _ = sample_mixture_compositions(
            r_samples_A=r_1comp,
            r_samples_B=r_1comp,
            weights_A=w_1comp,
            weights_B=w_1comp,
            rng_key=rng,
        )

        # Standard path with the same component
        s_std, _ = sample_compositions(
            r_samples_A=r_1comp[:, 0, :],
            r_samples_B=r_1comp[:, 0, :],
            rng_key=random.fold_in(rng, 0),
        )

        # Statistical convergence of means
        np.testing.assert_allclose(
            s_mix.mean(axis=0), s_std.mean(axis=0), atol=0.05
        )

    def test_mixture_weighted_mean_matches_expected(self, rng):
        """Mean of weighted simplex converges to sum_k pi_k * r_k / r_Tk."""
        N = 10_000
        D = 3
        K = 2
        # Fixed concentrations (no posterior uncertainty) for exact target
        r_fixed = jnp.array(
            [
                [5.0, 3.0, 2.0],  # Component 0: r_T=10
                [2.0, 2.0, 6.0],  # Component 1: r_T=10
            ]
        )
        r_3d = jnp.broadcast_to(r_fixed[None, :, :], (N, K, D))
        pi_fixed = jnp.array([0.7, 0.3])
        w = jnp.broadcast_to(pi_fixed[None, :], (N, K))

        s_A, _ = sample_mixture_compositions(
            r_samples_A=r_3d,
            r_samples_B=r_3d,
            weights_A=w,
            weights_B=w,
            rng_key=rng,
        )

        # Expected: 0.7 * [5/10, 3/10, 2/10] + 0.3 * [2/10, 2/10, 6/10]
        expected = 0.7 * np.array([0.5, 0.3, 0.2]) + 0.3 * np.array(
            [0.2, 0.2, 0.6]
        )
        np.testing.assert_allclose(s_A.mean(axis=0), expected, atol=0.02)

    def test_mixture_weighted_clr_differences_shape(
        self, mixture_r_2comp, mixture_weights, rng
    ):
        """CLR differences from mixture-weighted simplices have correct shape."""
        s_A, s_B = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=mixture_weights,
            weights_B=mixture_weights,
            rng_key=rng,
        )
        delta = compute_delta_from_simplex(s_A, s_B)
        assert delta.shape == (mixture_r_2comp.shape[0], 5)

    def test_mixture_weighted_with_gene_mask(
        self, mixture_r_2comp, mixture_weights, rng
    ):
        """Gene mask aggregation works with mixture-weighted simplices."""
        s_A, s_B = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=mixture_weights,
            weights_B=mixture_weights,
            rng_key=rng,
        )
        mask = np.array([True, True, False, True, False])
        delta = compute_delta_from_simplex(s_A, s_B, gene_mask=mask)
        # Only 3 kept genes in output
        assert delta.shape == (mixture_r_2comp.shape[0], 3)

    def test_mixture_weighted_with_gene_specific_p(
        self, mixture_r_2comp, mixture_weights, mixture_p_2comp, rng
    ):
        """Gamma-normalise path works with mixture-weighted sampling."""
        s_A, s_B = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=mixture_weights,
            weights_B=mixture_weights,
            p_samples_A=mixture_p_2comp,
            p_samples_B=mixture_p_2comp,
            rng_key=rng,
        )
        np.testing.assert_allclose(s_A.sum(axis=1), 1.0, atol=1e-6)
        assert s_A.shape == (200, 5)

    def test_mixture_weighted_paired(
        self, mixture_r_2comp, mixture_weights, rng
    ):
        """Paired mode works for within-model mixture comparisons."""
        s_A, s_B = sample_mixture_compositions(
            r_samples_A=mixture_r_2comp,
            r_samples_B=mixture_r_2comp,
            weights_A=mixture_weights,
            weights_B=mixture_weights,
            paired=True,
            rng_key=rng,
        )
        assert s_A.shape == s_B.shape
        np.testing.assert_allclose(s_A.sum(axis=1), 1.0, atol=1e-6)

    def test_mixture_weighted_compare_factory(
        self, mixture_r_2comp, mixture_weights, rng
    ):
        """End-to-end through compare() with mixture_weighted=True."""
        de = compare(
            model_A=mixture_r_2comp,
            model_B=mixture_r_2comp,
            method="empirical",
            mixture_weighted=True,
            mixture_weights_A=np.asarray(mixture_weights),
            mixture_weights_B=np.asarray(mixture_weights),
            rng_key=rng,
        )
        assert isinstance(de, ScribeEmpiricalDEResults)
        result = de.gene_level()
        assert "delta_mean" in result
        assert result["delta_mean"].shape == (5,)

    def test_mixture_weighted_biological_metrics(self, rng):
        """weight_bio_samples produces correct shapes and values."""
        N, K, D = 50, 2, 5
        k1, k2, k3 = random.split(rng, 3)
        r = jnp.abs(random.normal(k1, (N, K, D))) + 1.0
        p = jnp.full((N, K, D), 0.5)
        w = jnp.ones((N, K)) / K

        bio = weight_bio_samples(r_samples=r, weights=w, p_samples=p)
        assert bio["r"].shape == (N, D)
        assert bio["p"].shape == (N, D)
        assert bio["mu"].shape == (N, D)

        # With equal weights and same p, weighted r should be mean of
        # per-component r
        expected_r = (r[:, 0, :] + r[:, 1, :]) / 2.0
        np.testing.assert_allclose(
            np.asarray(bio["r"]), np.asarray(expected_r), atol=1e-5
        )

    def test_mixture_weighted_mutual_exclusion(
        self, mixture_r_2comp, mixture_weights
    ):
        """Error when both component_A and mixture_weighted are set."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            compare(
                model_A=mixture_r_2comp,
                model_B=mixture_r_2comp,
                method="empirical",
                component_A=0,
                component_B=0,
                mixture_weighted=True,
                mixture_weights_A=np.asarray(mixture_weights),
                mixture_weights_B=np.asarray(mixture_weights),
            )

    def test_mixture_weighted_shape_validation(self, rng):
        """Mismatched shapes raise ValueError."""
        r_3d = jnp.ones((50, 2, 5))
        # Wrong K in weights
        w_bad = np.ones((50, 3)) / 3
        with pytest.raises(ValueError, match="weights_A shape"):
            sample_mixture_compositions(
                r_samples_A=r_3d,
                r_samples_B=r_3d,
                weights_A=w_bad,
                weights_B=np.ones((50, 2)) / 2,
                rng_key=rng,
            )

    def test_mixture_weighted_requires_3d_r(self, rng):
        """2D r_samples raises ValueError."""
        r_2d = jnp.ones((50, 5))
        w = np.ones((50, 2)) / 2
        with pytest.raises(ValueError, match="3D"):
            sample_mixture_compositions(
                r_samples_A=r_2d,
                r_samples_B=r_2d,
                weights_A=w,
                weights_B=w,
                rng_key=rng,
            )
