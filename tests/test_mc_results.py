"""Tests for ScribeModelComparisonResults and the mc module.

Validates the results class methods (waic, psis_loo, rank, summary,
gene_level_comparison, stacking_weights) using synthetic log-likelihood
matrices, without requiring actual fitted SCRIBE models.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import pandas as pd

from scribe.mc import (
    ScribeModelComparisonResults,
    compute_stacking_weights,
    compute_psis_loo,
    waic,
    gene_level_comparison,
)
from scribe.mc._stacking import stacking_summary
from scribe.mc._gene_level import format_gene_comparison_table
from scribe.models.log_likelihood import nbdm_log_likelihood
from scribe.mcmc.results import _compute_log_likelihood
from scribe.svi._likelihood import LikelihoodMixin


# --------------------------------------------------------------------------
# Fixtures: synthetic log-likelihood matrices
# --------------------------------------------------------------------------


@pytest.fixture
def three_model_results():
    """Build a ScribeModelComparisonResults with three synthetic models.

    Model A: best   (log-lik ~ -2.0 ± 0.3)
    Model B: medium (log-lik ~ -2.5 ± 0.3)
    Model C: worst  (log-lik ~ -3.0 ± 0.3)
    """
    rng = np.random.default_rng(0)
    S, C, G = 300, 80, 20

    ll_A_cell = jnp.array(rng.normal(-2.0, 0.3, (S, C)), dtype=jnp.float32)
    ll_B_cell = jnp.array(rng.normal(-2.5, 0.3, (S, C)), dtype=jnp.float32)
    ll_C_cell = jnp.array(rng.normal(-3.0, 0.3, (S, C)), dtype=jnp.float32)

    ll_A_gene = jnp.array(rng.normal(-40.0, 2.0, (S, G)), dtype=jnp.float32)
    ll_B_gene = jnp.array(rng.normal(-42.0, 2.0, (S, G)), dtype=jnp.float32)
    ll_C_gene = jnp.array(rng.normal(-44.0, 2.0, (S, G)), dtype=jnp.float32)

    gene_names = [f"GENE{g}" for g in range(G)]

    return ScribeModelComparisonResults(
        model_names=["ModelA", "ModelB", "ModelC"],
        log_liks_cell=[ll_A_cell, ll_B_cell, ll_C_cell],
        log_liks_gene=[ll_A_gene, ll_B_gene, ll_C_gene],
        gene_names=gene_names,
        n_cells=C,
        n_genes=G,
    )


@pytest.fixture
def two_model_results():
    """Build a ScribeModelComparisonResults with two models (no gene liks)."""
    rng = np.random.default_rng(1)
    S, C = 200, 50
    ll_A = jnp.array(rng.normal(-2.0, 0.4, (S, C)), dtype=jnp.float32)
    ll_B = jnp.array(rng.normal(-2.8, 0.4, (S, C)), dtype=jnp.float32)
    return ScribeModelComparisonResults(
        model_names=["NBDM", "Hierarchical"],
        log_liks_cell=[ll_A, ll_B],
        n_cells=C,
        n_genes=10,
    )


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------


def test_K_property(three_model_results):
    """K should equal the number of models."""
    assert three_model_results.K == 3


def test_repr(three_model_results):
    """__repr__ should contain key info."""
    r = repr(three_model_results)
    assert "ScribeModelComparisonResults" in r
    assert "K=3" in r


# --------------------------------------------------------------------------
# waic()
# --------------------------------------------------------------------------


def test_waic_returns_list_for_all(three_model_results):
    """waic() without model_idx should return list of K dicts."""
    result = three_model_results.waic()
    assert isinstance(result, list)
    assert len(result) == 3


def test_waic_returns_dict_for_single(three_model_results):
    """waic(model_idx=0) should return a single dict."""
    result = three_model_results.waic(model_idx=0)
    assert isinstance(result, dict)
    assert "waic_2" in result


def test_waic_cached(three_model_results):
    """Second call to waic() should use cache (not recompute)."""
    r1 = three_model_results.waic()
    r2 = three_model_results.waic()
    assert r1 is r2


def test_waic_ordering(three_model_results):
    """Model A should have lower WAIC (better) than B and C."""
    waic_all = three_model_results.waic()
    assert waic_all[0]["waic_2"] < waic_all[1]["waic_2"]
    assert waic_all[1]["waic_2"] < waic_all[2]["waic_2"]


# --------------------------------------------------------------------------
# psis_loo()
# --------------------------------------------------------------------------


def test_psis_loo_returns_list(three_model_results):
    """psis_loo() should return list of K dicts."""
    result = three_model_results.psis_loo()
    assert isinstance(result, list) and len(result) == 3


def test_psis_loo_returns_dict_for_single(three_model_results):
    """psis_loo(model_idx=1) should return a single dict."""
    result = three_model_results.psis_loo(model_idx=1)
    assert isinstance(result, dict)
    assert "elpd_loo" in result


def test_psis_loo_elpd_ordering(three_model_results):
    """Model A should have higher elpd_loo than B and C."""
    loo_all = three_model_results.psis_loo()
    assert loo_all[0]["elpd_loo"] > loo_all[1]["elpd_loo"]
    assert loo_all[1]["elpd_loo"] > loo_all[2]["elpd_loo"]


def test_psis_loo_cached(three_model_results):
    """Second call to psis_loo() should use cache."""
    r1 = three_model_results.psis_loo()
    r2 = three_model_results.psis_loo()
    assert r1 is r2


# --------------------------------------------------------------------------
# rank()
# --------------------------------------------------------------------------


def test_rank_returns_dataframe(three_model_results):
    """rank() should return a pandas DataFrame."""
    df = three_model_results.rank()
    assert isinstance(df, pd.DataFrame)


def test_rank_has_required_columns(three_model_results):
    """rank() DataFrame should have required columns."""
    df = three_model_results.rank()
    required = {
        "model",
        "elpd",
        "p_eff",
        "elpd_diff",
        "elpd_diff_se",
        "z_score",
    }
    assert required.issubset(set(df.columns))


def test_rank_sorted_by_elpd(three_model_results):
    """rank() should sort models by elpd descending (best first)."""
    df = three_model_results.rank()
    elpd_vals = df["elpd"].values
    assert all(
        elpd_vals[i] >= elpd_vals[i + 1] for i in range(len(elpd_vals) - 1)
    )


def test_rank_best_model_has_diff_zero(three_model_results):
    """The best model should have elpd_diff = 0."""
    df = three_model_results.rank()
    assert abs(df["elpd_diff"].iloc[0]) < 1e-8


def test_rank_waic_criterion(three_model_results):
    """rank(criterion='waic_2') should also work."""
    df = three_model_results.rank(criterion="waic_2")
    assert "model" in df.columns
    assert df["elpd"].iloc[0] >= df["elpd"].iloc[-1]


def test_rank_unknown_criterion_raises(three_model_results):
    """rank() with unknown criterion should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown criterion"):
        three_model_results.rank(criterion="bad_criterion")


def test_rank_includes_stacking(three_model_results):
    """rank(include_stacking=True) should add weight_stacking column."""
    df = three_model_results.rank(include_stacking=True)
    assert "weight_stacking" in df.columns
    assert abs(df["weight_stacking"].sum() - 1.0) < 1e-3


# --------------------------------------------------------------------------
# summary()
# --------------------------------------------------------------------------


def test_summary_returns_string(three_model_results):
    """summary() should return a non-empty string."""
    s = three_model_results.summary()
    assert isinstance(s, str) and len(s) > 0


def test_summary_contains_model_names(three_model_results):
    """summary() should contain all model names."""
    s = three_model_results.summary()
    for name in three_model_results.model_names:
        assert name in s, f"Model name '{name}' not found in summary"


# --------------------------------------------------------------------------
# diagnostics()
# --------------------------------------------------------------------------


def test_diagnostics_returns_string(three_model_results):
    """diagnostics() should return a non-empty string."""
    s = three_model_results.diagnostics()
    assert isinstance(s, str) and len(s) > 0


def test_diagnostics_single_model(three_model_results):
    """diagnostics(model_idx=0) should refer to first model only."""
    s = three_model_results.diagnostics(model_idx=0)
    assert "ModelA" in s


# --------------------------------------------------------------------------
# stacking_weights()
# --------------------------------------------------------------------------


def test_stacking_weights_shape(three_model_results):
    """stacking_weights() should return shape (K,)."""
    w = three_model_results.stacking_weights()
    assert w.shape == (3,)


def test_stacking_weights_sum_to_one(three_model_results):
    """Stacking weights should sum to 1."""
    w = three_model_results.stacking_weights()
    assert abs(w.sum() - 1.0) < 1e-3


def test_stacking_weights_best_model_dominates(three_model_results):
    """The best model should receive the highest stacking weight."""
    df = three_model_results.rank(include_stacking=True)
    best_w = df["weight_stacking"].iloc[0]
    for i in range(1, len(df)):
        assert best_w >= df["weight_stacking"].iloc[i]


# --------------------------------------------------------------------------
# gene_level_comparison()
# --------------------------------------------------------------------------


def test_gene_level_comparison_returns_dataframe(three_model_results):
    """gene_level_comparison() should return a DataFrame."""
    df = three_model_results.gene_level_comparison("ModelA", "ModelC")
    assert isinstance(df, pd.DataFrame)


def test_gene_level_comparison_shape(three_model_results):
    """DataFrame should have G rows (one per gene)."""
    df = three_model_results.gene_level_comparison("ModelA", "ModelC")
    assert len(df) == three_model_results.n_genes


def test_gene_level_comparison_columns(three_model_results):
    """DataFrame should have required columns."""
    df = three_model_results.gene_level_comparison("ModelA", "ModelC")
    required = {"gene", "elpd_diff", "elpd_diff_se", "z_score", "favors"}
    assert required.issubset(set(df.columns))


def test_gene_level_comparison_elpd_sign(three_model_results):
    """Model A is better, so most genes should favor ModelA."""
    df = three_model_results.gene_level_comparison("ModelA", "ModelC")
    frac_a = (df["favors"] == "ModelA").mean()
    assert frac_a > 0.5, f"Expected ModelA to win most genes, got {frac_a:.2f}"


def test_gene_level_comparison_missing_gene_liks(two_model_results):
    """Should raise RuntimeError when gene liks are not available."""
    with pytest.raises(RuntimeError, match="compute_gene_liks"):
        two_model_results.gene_level_comparison("NBDM", "Hierarchical")


def test_gene_level_comparison_by_index(three_model_results):
    """Should also accept integer model indices."""
    df = three_model_results.gene_level_comparison(0, 2)
    assert isinstance(df, pd.DataFrame)


def test_gene_level_comparison_invalid_name_raises(three_model_results):
    """Invalid model name should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        three_model_results.gene_level_comparison(
            "NBDM_DOES_NOT_EXIST", "ModelC"
        )


# --------------------------------------------------------------------------
# format_gene_comparison_table
# --------------------------------------------------------------------------


def test_format_gene_comparison_table(three_model_results):
    """format_gene_comparison_table should return non-empty string."""
    df = three_model_results.gene_level_comparison("ModelA", "ModelB")
    s = format_gene_comparison_table(df, top_n=5)
    assert isinstance(s, str) and len(s) > 0


# --------------------------------------------------------------------------
# stacking_summary
# --------------------------------------------------------------------------


def test_stacking_summary_returns_string(three_model_results):
    """stacking_summary should return a non-empty string."""
    w = three_model_results.stacking_weights()
    s = stacking_summary(w, model_names=three_model_results.model_names)
    assert isinstance(s, str) and len(s) > 0


# --------------------------------------------------------------------------
# Standalone function tests
# --------------------------------------------------------------------------


def test_standalone_compute_stacking_weights():
    """compute_stacking_weights should work with list of numpy arrays."""
    rng = np.random.default_rng(99)
    K, n = 3, 150
    loo_list = [rng.normal(-k * 0.5, 0.2, n) for k in range(K)]
    w = compute_stacking_weights(loo_list, n_restarts=2, seed=0)
    assert w.shape == (K,)
    assert abs(w.sum() - 1.0) < 1e-3
    assert all(wi >= 0 for wi in w)


def test_standalone_gene_level_comparison():
    """gene_level_comparison should work as a standalone function."""
    rng = np.random.default_rng(5)
    S, G = 100, 30
    ll_A = rng.normal(-2.0, 0.5, (S, G))
    ll_B = rng.normal(-2.5, 0.5, (S, G))
    df = gene_level_comparison(ll_A, ll_B, label_A="A", label_B="B")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == G


def test_mc_imports():
    """All public names in scribe.mc should be importable."""
    from scribe.mc import (
        ScribeModelComparisonResults,
        compare_models,
        compute_waic_stats,
        waic,
        pseudo_bma_weights,
        compute_psis_loo,
        psis_loo_summary,
        gene_level_comparison,
        format_gene_comparison_table,
        compute_stacking_weights,
        stacking_summary,
    )

    # If all imports succeed, the module is properly set up
    assert ScribeModelComparisonResults is not None


def test_scribe_mc_accessible_from_top_level():
    """scribe.mc and scribe.compare_models should be importable from scribe."""
    import scribe

    assert hasattr(scribe, "mc")
    assert hasattr(scribe, "compare_models")
    assert hasattr(scribe, "ScribeModelComparisonResults")


# --------------------------------------------------------------------------
# Dead component pruning — _prune_dead_components()
# --------------------------------------------------------------------------


class MockMixtureResults:
    """Minimal mock of ScribeSVIResults for pruning tests.

    Provides the attributes and methods that ``_prune_dead_components``
    inspects: ``n_components``, ``posterior_samples``, and ``get_component``.
    """

    def __init__(self, mixing_weights: np.ndarray):
        """
        Parameters
        ----------
        mixing_weights : np.ndarray, shape ``(S, K)``
            Simulated posterior mixing weight samples.
        """
        S, K = mixing_weights.shape
        self.n_components = K
        self.posterior_samples = {"mixing_weights": jnp.array(mixing_weights)}
        self._mixing_weights = mixing_weights

    def get_component(self, mask, renormalize=True):
        """Return a new MockMixtureResults with only the selected components."""
        # Subset the mixing weights to the active components and renormalize
        active_idx = np.flatnonzero(np.asarray(mask))
        selected_mw = self._mixing_weights[:, active_idx]
        if renormalize:
            row_sums = selected_mw.sum(axis=1, keepdims=True)
            selected_mw = selected_mw / row_sums
        return MockMixtureResults(selected_mw)


class MockNonMixtureResults:
    """Minimal mock of a non-mixture ScribeSVIResults (n_components=None)."""

    def __init__(self):
        self.n_components = None
        self.posterior_samples = {}


@pytest.fixture
def mock_mixture_4comp():
    """Mock 4-component mixture where components 2 and 3 are near-zero.

    Posterior-mean mixing weights ≈ [0.45, 0.50, 0.003, 0.002].
    """
    rng = np.random.default_rng(7)
    S = 300
    # Draw from Dirichlet — heavily concentrated on components 0 and 1
    alpha = np.array([45.0, 50.0, 0.3, 0.2])
    mw = rng.dirichlet(alpha, size=S)  # shape (S, 4)
    return MockMixtureResults(mw)


@pytest.fixture
def mock_mixture_all_active():
    """Mock 3-component mixture where all components have substantial weight."""
    rng = np.random.default_rng(8)
    S = 200
    alpha = np.array([30.0, 35.0, 25.0])
    mw = rng.dirichlet(alpha, size=S)
    return MockMixtureResults(mw)


def test_prune_dead_components_reduces_k(mock_mixture_4comp):
    """With threshold=0.01, the two near-zero components should be pruned."""
    from scribe.mc.results import _prune_dead_components

    pruned, active = _prune_dead_components(mock_mixture_4comp, threshold=0.01)

    # active mask should exist and flag exactly 2 surviving components
    assert active is not None, "Pruning should have been applied"
    assert active.dtype == bool
    assert (
        int(active.sum()) == 2
    ), f"Expected 2 survivors, got {int(active.sum())}"

    # Pruned model should have n_components == 2
    assert pruned.n_components == 2

    # Mixing weights in pruned model should sum to 1 per sample
    mw_pruned = np.asarray(pruned.posterior_samples["mixing_weights"])
    np.testing.assert_allclose(mw_pruned.sum(axis=1), 1.0, atol=1e-5)


def test_prune_dead_components_non_mixture_unchanged():
    """Non-mixture model (n_components=None) should not be pruned."""
    from scribe.mc.results import _prune_dead_components

    results = MockNonMixtureResults()
    pruned, active = _prune_dead_components(results, threshold=0.05)

    assert pruned is results, "Non-mixture results must be returned unchanged"
    assert active is None


def test_prune_dead_components_zero_threshold_unchanged(mock_mixture_4comp):
    """threshold=0.0 should disable pruning entirely."""
    from scribe.mc.results import _prune_dead_components

    pruned, active = _prune_dead_components(mock_mixture_4comp, threshold=0.0)

    assert pruned is mock_mixture_4comp
    assert active is None


def test_prune_dead_components_all_active_unchanged(mock_mixture_all_active):
    """When all components exceed the threshold, nothing should be pruned."""
    from scribe.mc.results import _prune_dead_components

    pruned, active = _prune_dead_components(
        mock_mixture_all_active, threshold=0.01
    )

    # All three components have weight >> 0.01, so no pruning
    assert pruned is mock_mixture_all_active
    assert active is None


def test_prune_dead_components_aggressive_threshold_keeps_best(
    mock_mixture_4comp,
):
    """A threshold that kills all components should fall back to keeping the best."""
    from scribe.mc.results import _prune_dead_components
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pruned, active = _prune_dead_components(
            mock_mixture_4comp, threshold=0.99
        )

    # Should warn and keep exactly 1 component
    assert active is not None
    assert int(active.sum()) == 1
    assert pruned.n_components == 1
    # The kept component should be the one with the largest posterior-mean weight
    assert len(w) > 0  # a UserWarning was issued


def test_prune_dead_components_active_mask_correct_length(mock_mixture_4comp):
    """The active mask should have length K_original."""
    from scribe.mc.results import _prune_dead_components

    _, active = _prune_dead_components(mock_mixture_4comp, threshold=0.01)

    assert active is not None
    assert len(active) == mock_mixture_4comp.n_components  # 4


def test_prune_dead_components_missing_mixing_weights_warns():
    """If posterior_samples has no mixing_weights key, a UserWarning is issued."""
    from scribe.mc.results import _prune_dead_components
    import warnings

    # Create a mock with n_components > 1 but no mixing weights in samples
    class MockNoWeights:
        n_components = 3
        posterior_samples = {"some_other_param": jnp.ones((10, 3))}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pruned, active = _prune_dead_components(MockNoWeights(), threshold=0.05)

    assert active is None  # pruning skipped
    assert any("mixing_weights" in str(warning.message) for warning in w)


# --------------------------------------------------------------------------
# active_components stored in ScribeModelComparisonResults
# --------------------------------------------------------------------------


def test_active_components_none_when_no_pruning(three_model_results):
    """active_components should be None when no component_threshold is set."""
    # three_model_results is built without pruning
    assert three_model_results.active_components is None


def test_active_components_stored_on_pruned_results():
    """active_components should be populated when pruning occurs."""
    rng = np.random.default_rng(0)
    S, C = 200, 40
    # Build a results object with active_components set manually
    ll = jnp.array(rng.normal(-2.0, 0.3, (S, C)), dtype=jnp.float32)
    active_mask = np.array(
        [True, True, False, False]
    )  # 2 of 4 components active

    mc = ScribeModelComparisonResults(
        model_names=["MixtureModel"],
        log_liks_cell=[ll],
        n_cells=C,
        n_genes=10,
        active_components=[active_mask],
    )

    assert mc.active_components is not None
    assert mc.active_components[0] is active_mask


def test_repr_shows_pruning_arrow():
    """__repr__ should include the K_orig→K_pruned annotation when pruned."""
    rng = np.random.default_rng(0)
    S, C = 100, 30
    ll = jnp.array(rng.normal(-2.0, 0.3, (S, C)), dtype=jnp.float32)
    active_mask = np.array([True, True, False, False])  # 4 original, 2 kept

    mc = ScribeModelComparisonResults(
        model_names=["ZINBVCP"],
        log_liks_cell=[ll],
        n_cells=C,
        n_genes=5,
        active_components=[active_mask],
    )

    r = repr(mc)
    # Should show the 4→2 annotation with the unicode arrow
    assert "4" in r and "2" in r
    assert "→" in r or "->" in r or "\u2192" in r


def test_repr_no_arrow_when_no_pruning(two_model_results):
    """__repr__ should not contain any arrow when no pruning occurred."""
    r = repr(two_model_results)
    assert "→" not in r
    assert "\u2192" not in r


def test_svi_likelihood_chunking_matches_full_vmap():
    """Chunked SVI likelihood should match unchunked evaluation."""

    class _DummySVI(LikelihoodMixin):
        def __init__(self, posterior_samples):
            self.posterior_samples = posterior_samples
            self.n_components = None

        def _log_likelihood_fn(self):
            return nbdm_log_likelihood

    _rng = np.random.default_rng(42)
    _counts = jnp.array(_rng.poisson(2.0, (7, 4)), dtype=jnp.float32)
    _posterior = {
        "p": jnp.full((9,), 0.3, dtype=jnp.float32),
        "r": jnp.full((9, 4), 1.2, dtype=jnp.float32),
    }
    _results = _DummySVI(_posterior)

    _ll_full = _results.log_likelihood(
        _counts, return_by="cell", sample_chunk_size=None
    )
    _ll_chunk = _results.log_likelihood(
        _counts, return_by="cell", sample_chunk_size=3
    )

    np.testing.assert_allclose(
        np.asarray(_ll_full), np.asarray(_ll_chunk), rtol=1e-6
    )


def test_mcmc_likelihood_chunking_matches_full_vmap():
    """Chunked MCMC helper likelihood should match unchunked evaluation."""
    _rng = np.random.default_rng(7)
    _counts = jnp.array(_rng.poisson(2.5, (6, 3)), dtype=jnp.float32)
    _samples = {
        "p": jnp.full((8,), 0.25, dtype=jnp.float32),
        "r": jnp.full((8, 3), 1.4, dtype=jnp.float32),
    }

    _ll_full = _compute_log_likelihood(
        _samples,
        _counts,
        model_type="nbdm",
        sample_chunk_size=None,
        return_by="cell",
    )
    _ll_chunk = _compute_log_likelihood(
        _samples,
        _counts,
        model_type="nbdm",
        sample_chunk_size=2,
        return_by="cell",
    )

    np.testing.assert_allclose(
        np.asarray(_ll_full), np.asarray(_ll_chunk), rtol=1e-6
    )
