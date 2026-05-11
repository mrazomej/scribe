"""
Tests for corner-plot-style posterior predictive check visualization.

These tests verify the helper functions and grid layout logic in
``scribe.viz.corner_ppc`` without requiring a fitted model:

- Gene selection resolution (explicit indices, names, auto-select)
- Diagonal panel rendering with synthetic credible regions
- Off-diagonal panel rendering with synthetic 2-D data
- Corner grid structure (shape, upper-triangle hidden, diagonal/lower visible)
"""

import pytest
import numpy as np
import matplotlib
import types

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scribe.viz.corner_ppc as corner_ppc_module
from scribe.viz.corner_ppc import (
    plot_corner_ppc,
    _resolve_gene_indices,
    _render_diagonal_panel,
    _render_offdiag_panel,
    _select_genes_by_correlation_diversity,
    _get_correlation_matrix_for_selection,
    _empirical_correlation_residual,
)
from scribe.viz._interactive import _create_or_validate_grid_axes
from scribe.viz.ppc_rendering import get_ppc_render_options


# ==============================================================================
# Fixtures
# ==============================================================================


class _FakeVar:
    """Minimal stand-in for ``results.var`` with a pandas-like index."""

    def __init__(self, names):
        self.index = np.array(names)


class _FakeResults:
    """Minimal results-like object for gene-selection tests."""

    def __init__(self, n_genes, gene_names=None):
        self.n_genes = n_genes
        self.var = _FakeVar(gene_names) if gene_names is not None else None


@pytest.fixture
def synth_counts():
    """Synthetic count matrix (50 cells x 10 genes) with varying expression."""
    rng = np.random.default_rng(42)
    # Each gene column has a different NB mean so auto-select spreads them out
    means = np.linspace(1, 50, 10)
    _counts = np.column_stack(
        [rng.negative_binomial(n=5, p=5 / (5 + m), size=50) for m in means]
    )
    return _counts


@pytest.fixture
def fake_results():
    """Fake results object with 10 genes and names."""
    names = [f"Gene_{i}" for i in range(10)]
    return _FakeResults(n_genes=10, gene_names=names)


@pytest.fixture
def render_opts():
    """Default PPC render options."""
    return get_ppc_render_options(None)


# ==============================================================================
# Gene selection resolution
# ==============================================================================


class TestResolveGeneIndices:
    """Tests for the three-mode gene selection resolution logic."""

    def test_explicit_indices(self, fake_results, synth_counts):
        """Explicit gene_indices are returned as-is."""
        idx = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=[0, 3, 7],
            gene_names_list=None,
            n_genes=5,
        )
        np.testing.assert_array_equal(idx, [0, 3, 7])

    def test_gene_names_resolution(self, fake_results, synth_counts):
        """Gene names are resolved to column indices via results.var.index."""
        idx = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=None,
            gene_names_list=["Gene_2", "Gene_5", "Gene_9"],
            n_genes=5,
        )
        np.testing.assert_array_equal(idx, [2, 5, 9])

    def test_gene_names_missing_raises(self, fake_results, synth_counts):
        """Missing gene names raise a clear ValueError."""
        with pytest.raises(ValueError, match="not found"):
            _resolve_gene_indices(
                fake_results, synth_counts,
                gene_indices=None,
                gene_names_list=["Gene_0", "NONEXISTENT"],
                n_genes=5,
            )

    def test_gene_names_no_var_raises(self, synth_counts):
        """Passing gene_names_list without results.var raises ValueError."""
        _results_no_var = _FakeResults(n_genes=10, gene_names=None)
        with pytest.raises(ValueError, match="results.var is missing"):
            _resolve_gene_indices(
                _results_no_var, synth_counts,
                gene_indices=None,
                gene_names_list=["Gene_0"],
                n_genes=5,
            )

    def test_auto_select_count(self, fake_results, synth_counts):
        """Auto-selection returns exactly n_genes indices."""
        idx = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=None,
            gene_names_list=None,
            n_genes=4,
        )
        assert len(idx) == 4
        # All indices must be valid columns
        assert all(0 <= i < synth_counts.shape[1] for i in idx)

    def test_explicit_overrides_names(self, fake_results, synth_counts):
        """gene_indices takes priority even when gene_names_list is also given."""
        idx = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=[1, 2],
            gene_names_list=["Gene_5"],
            n_genes=5,
        )
        np.testing.assert_array_equal(idx, [1, 2])


# ==============================================================================
# Diagonal panel rendering
# ==============================================================================


class TestRenderDiagonalPanel:
    """Test the marginal PPC histogram rendering helper."""

    def test_renders_without_error(self, render_opts):
        """Diagonal panel runs without error on synthetic PPC data."""
        rng = np.random.default_rng(7)
        # Simulate PPC samples: (n_draws=20, n_cells=50)
        ppc_samples = rng.negative_binomial(n=5, p=0.4, size=(20, 50))
        true_counts = rng.negative_binomial(n=5, p=0.4, size=50)

        fig, ax = plt.subplots()
        _render_diagonal_panel(
            ax, ppc_samples, true_counts, render_opts, gene_label="TestGene"
        )
        # Should have a title containing the gene label
        assert "TestGene" in ax.get_title()
        plt.close(fig)

    def test_no_gene_label(self, render_opts):
        """Panel renders fine without a gene label."""
        rng = np.random.default_rng(8)
        ppc_samples = rng.negative_binomial(n=3, p=0.5, size=(10, 30))
        true_counts = rng.negative_binomial(n=3, p=0.5, size=30)

        fig, ax = plt.subplots()
        _render_diagonal_panel(ax, ppc_samples, true_counts, render_opts)
        # Title should contain mean expression but not a gene name
        assert "\\langle U \\rangle" in ax.get_title()
        plt.close(fig)


# ==============================================================================
# Off-diagonal panel rendering
# ==============================================================================


class TestRenderOffdiagPanel:
    """Test the bivariate PPC contour + scatter rendering helper."""

    def test_renders_without_error(self):
        """Off-diagonal panel runs without error on synthetic data."""
        rng = np.random.default_rng(10)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(15, 40))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(15, 40))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=40)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=40)

        fig, ax = plt.subplots()
        _render_offdiag_panel(ax, ppc_x, ppc_y, obs_x, obs_y)
        # At minimum, axis should contain collections (contourf) and scatter
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_log_scale(self):
        """Off-diagonal panel works with log_scale=True."""
        rng = np.random.default_rng(11)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        fig, ax = plt.subplots()
        _render_offdiag_panel(
            ax, ppc_x, ppc_y, obs_x, obs_y, log_scale=True
        )
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_degenerate_data_no_crash(self):
        """When all samples are identical the panel should not crash."""
        # Constant data → singular KDE covariance → should fall back
        ppc_x = np.ones((5, 20), dtype=int)
        ppc_y = np.ones((5, 20), dtype=int) * 3
        obs_x = np.ones(20, dtype=int)
        obs_y = np.ones(20, dtype=int) * 3

        fig, ax = plt.subplots()
        _render_offdiag_panel(ax, ppc_x, ppc_y, obs_x, obs_y)
        plt.close(fig)


# ==============================================================================
# Corner grid structure
# ==============================================================================


# ==============================================================================
# Correlation-diversity gene selection
# ==============================================================================


class TestCorrelationDiversitySelection:
    """Tests for the hybrid greedy-diversity gene selection algorithm."""

    def test_extreme_pairs_selected(self):
        """The most positively and negatively correlated pairs are selected."""
        # Build a synthetic 8x8 correlation matrix with known extreme pairs:
        # genes 0,1 highly positive (0.95), genes 4,5 highly negative (-0.90)
        rng = np.random.default_rng(99)
        n_g = 8
        corr = np.eye(n_g)
        # Weak background correlations
        for i in range(n_g):
            for j in range(i + 1, n_g):
                corr[i, j] = corr[j, i] = rng.uniform(-0.1, 0.1)
        # Inject known extremes
        corr[0, 1] = corr[1, 0] = 0.95
        corr[4, 5] = corr[5, 4] = -0.90
        np.fill_diagonal(corr, 1.0)

        # Synthetic counts (only used for median-expression sorting)
        counts = rng.negative_binomial(n=5, p=0.4, size=(50, n_g))

        selected = _select_genes_by_correlation_diversity(corr, 5, counts)
        assert len(selected) == 5
        # The extreme pair genes must appear in the selection
        assert 0 in selected and 1 in selected, (
            f"Positive pair (0,1) missing from {selected}"
        )
        assert 4 in selected and 5 in selected, (
            f"Negative pair (4,5) missing from {selected}"
        )

    def test_greedy_fill_adds_diverse_gene(self):
        """The 5th gene should not duplicate an already-selected cluster."""
        rng = np.random.default_rng(77)
        n_g = 6
        # Genes 0,1 strongly positive; genes 2,3 strongly negative;
        # gene 4 is strongly correlated with 0,1 (same cluster);
        # gene 5 is uncorrelated with everyone.
        corr = np.eye(n_g)
        corr[0, 1] = corr[1, 0] = 0.95
        corr[2, 3] = corr[3, 2] = -0.85
        corr[0, 4] = corr[4, 0] = 0.90
        corr[1, 4] = corr[4, 1] = 0.88
        # Gene 5 near-zero with everyone
        for i in range(5):
            corr[i, 5] = corr[5, i] = rng.uniform(-0.05, 0.05)

        counts = rng.negative_binomial(n=5, p=0.4, size=(50, n_g))
        selected = _select_genes_by_correlation_diversity(corr, 5, counts)
        assert len(selected) == 5
        # Gene 5 (the uncorrelated one) should be preferred over gene 4
        # (which duplicates the 0,1 cluster)
        assert 5 in selected, (
            f"Gene 5 (uncorrelated control) missing from {selected}"
        )

    def test_degenerate_identity_matrix(self):
        """Identity correlation (no structure) returns n_genes without error."""
        n_g = 10
        corr = np.eye(n_g)
        rng = np.random.default_rng(55)
        counts = rng.negative_binomial(n=5, p=0.4, size=(50, n_g))

        selected = _select_genes_by_correlation_diversity(corr, 5, counts)
        assert len(selected) == 5
        assert len(set(selected)) == 5

    def test_small_gene_count(self):
        """When total genes < n_genes, returns all available genes."""
        n_g = 3
        rng = np.random.default_rng(33)
        corr = np.corrcoef(rng.normal(size=(50, n_g)), rowvar=False)
        counts = rng.negative_binomial(n=5, p=0.4, size=(50, n_g))

        selected = _select_genes_by_correlation_diversity(corr, 5, counts)
        assert len(selected) == n_g
        np.testing.assert_array_equal(sorted(selected), list(range(n_g)))

    def test_alr_dimension_mismatch(self):
        """When corr has fewer rows than counts columns, excludes the last."""
        # Simulate LNM: corr is (G-1, G-1), counts is (n_cells, G)
        n_g = 8
        rng = np.random.default_rng(44)
        corr = np.corrcoef(rng.normal(size=(50, n_g - 1)), rowvar=False)
        counts = rng.negative_binomial(n=5, p=0.4, size=(50, n_g))

        selected = _select_genes_by_correlation_diversity(corr, 4, counts)
        assert len(selected) == 4
        # Last gene column (ALR reference) should never be selected
        assert (n_g - 1) not in selected

    def test_empirical_fallback(self, fake_results, synth_counts):
        """Non-Laplace/VAE results fall back to empirical Pearson."""
        corr, n_gc = _get_correlation_matrix_for_selection(
            fake_results, synth_counts
        )
        assert corr.shape == (synth_counts.shape[1], synth_counts.shape[1])
        assert n_gc == synth_counts.shape[1]
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)


# ==============================================================================
# Nuisance-direction removal (subtract_direction)
# ==============================================================================


class TestSubtractDirection:
    """Tests for PCA / library-size removal before gene selection."""

    def test_empirical_pc_removal_changes_corr(self, synth_counts):
        """Removing the top PC should change the correlation matrix."""
        corr_full = np.corrcoef(
            np.log1p(synth_counts.astype(float)), rowvar=False
        )
        corr_resid = _empirical_correlation_residual(
            synth_counts, method="pc", n_components=1
        )
        assert corr_resid.shape == corr_full.shape
        # The residual should differ from the original
        assert not np.allclose(corr_full, corr_resid, atol=1e-3)
        # Diagonal should still be close to 1.0 (it's a correlation matrix)
        np.testing.assert_allclose(np.diag(corr_resid), 1.0, atol=1e-6)

    def test_empirical_pc_removal_bounded(self, synth_counts):
        """Residual correlation values should stay in [-1, 1]."""
        corr_resid = _empirical_correlation_residual(
            synth_counts, method="pc", n_components=2
        )
        assert np.all(corr_resid >= -1.0 - 1e-10)
        assert np.all(corr_resid <= 1.0 + 1e-10)

    def test_dispatch_pc_empirical(self, fake_results, synth_counts):
        """Empirical path with subtract_direction='pc' returns valid corr."""
        corr, n_gc = _get_correlation_matrix_for_selection(
            fake_results, synth_counts,
            subtract_direction="pc", n_pcs_to_remove=1,
        )
        assert corr.shape == (synth_counts.shape[1], synth_counts.shape[1])
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_dispatch_library_size_falls_back_to_pc(
        self, fake_results, synth_counts
    ):
        """Empirical path: 'library_size' falls back to 'pc' gracefully."""
        corr, n_gc = _get_correlation_matrix_for_selection(
            fake_results, synth_counts,
            subtract_direction="library_size", n_pcs_to_remove=1,
        )
        assert corr.shape == (synth_counts.shape[1], synth_counts.shape[1])
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_selection_differs_with_subtract(self, fake_results, synth_counts):
        """Gene selection can change when nuisance direction is removed."""
        idx_full = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=None, gene_names_list=None, n_genes=4,
        )
        idx_resid = _resolve_gene_indices(
            fake_results, synth_counts,
            gene_indices=None, gene_names_list=None, n_genes=4,
            subtract_direction="pc", n_pcs_to_remove=1,
        )
        # Both should return valid indices
        assert len(idx_full) == 4
        assert len(idx_resid) == 4
        assert all(0 <= i < synth_counts.shape[1] for i in idx_resid)


# ==============================================================================
# Corner grid structure
# ==============================================================================


class TestCornerGridStructure:
    """Verify the N x N grid layout matches corner-plot conventions."""

    @pytest.mark.parametrize("n_genes", [3, 4, 5])
    def test_grid_shape(self, n_genes):
        """Grid has exactly n_genes x n_genes axes."""
        fig, axes_grid, axes_flat = _create_or_validate_grid_axes(
            n_rows=n_genes, n_cols=n_genes,
        )
        assert axes_grid.shape == (n_genes, n_genes)
        assert len(axes_flat) == n_genes * n_genes
        plt.close(fig)

    def test_upper_triangle_off(self):
        """Upper-triangle axes can be turned off without error."""
        n = 4
        fig, axes_grid, _ = _create_or_validate_grid_axes(
            n_rows=n, n_cols=n,
        )
        for row in range(n):
            for col in range(n):
                if row < col:
                    axes_grid[row, col].axis("off")

        # Verify upper-triangle axes have no visible frame
        for row in range(n):
            for col in range(row + 1, n):
                _ax = axes_grid[row, col]
                assert not _ax.axison
        plt.close(fig)

    def test_diagonal_and_lower_visible(self):
        """Diagonal and lower-triangle axes remain visible."""
        n = 4
        fig, axes_grid, _ = _create_or_validate_grid_axes(
            n_rows=n, n_cols=n,
        )
        # Turn off upper triangle as corner_ppc would do
        for row in range(n):
            for col in range(row + 1, n):
                axes_grid[row, col].axis("off")

        # Diagonal should be visible
        for i in range(n):
            assert axes_grid[i, i].axison

        # Lower triangle should be visible
        for row in range(n):
            for col in range(row):
                assert axes_grid[row, col].axison
        plt.close(fig)


def test_plot_corner_ppc_reuses_subset_sampling_object(monkeypatch):
    """Corner PPC should plot from the subset object that got sampled."""
    counts = np.arange(5 * 7, dtype=float).reshape(5, 7) + 1.0
    seen = {}

    class _FakeResultsForCorner:
        def __init__(self, n_genes):
            self.n_genes = int(n_genes)
            self.model_config = types.SimpleNamespace(inference_method="svi")
            self.predictive_samples = None
            self.var = None

        def __getitem__(self, index):
            # Return a fresh subset object, matching the behavior that exposed
            # the regression when predictive samples were attached elsewhere.
            subset = _FakeResultsForCorner(len(index))
            if self.predictive_samples is not None:
                idx = np.asarray(index, dtype=int)
                subset.predictive_samples = self.predictive_samples[:, :, idx]
            return subset

    def _fake_predictive(
        sampling_results, *, rng_key, n_samples, counts, store_samples,
        ppc_level=None,
    ):
        _ = rng_key, store_samples, ppc_level
        seen["sampling_results_id"] = id(sampling_results)
        sampling_results.predictive_samples = np.zeros(
            (n_samples, counts.shape[0], sampling_results.n_genes), dtype=float
        )
        return sampling_results.predictive_samples

    def _fake_diagonal(axis, ppc_samples, true_counts, render_opts, gene_label=None):
        _ = axis, true_counts, render_opts, gene_label
        seen["diag_called"] = True
        seen["diag_ppc_shape"] = tuple(ppc_samples.shape)

    monkeypatch.setattr(corner_ppc_module, "_coerce_counts", lambda x: np.asarray(x))
    monkeypatch.setattr(
        corner_ppc_module,
        "_coerce_and_align_counts_to_results",
        lambda raw, results, context=None: raw,
    )
    monkeypatch.setattr(
        corner_ppc_module,
        "_resolve_ppc_sampling_counts",
        lambda results, raw, aligned: aligned,
    )
    monkeypatch.setattr(
        corner_ppc_module,
        "_resolve_gene_indices",
        lambda *args, **kwargs: np.array([1, 4], dtype=int),
    )
    monkeypatch.setattr(
        corner_ppc_module, "_resolve_ppc_grid", lambda **kwargs: {"n_samples": 3}
    )
    monkeypatch.setattr(corner_ppc_module, "_get_gene_names", lambda results: None)
    monkeypatch.setattr(
        corner_ppc_module, "_get_predictive_samples_for_plot", _fake_predictive
    )
    monkeypatch.setattr(corner_ppc_module, "_render_diagonal_panel", _fake_diagonal)
    monkeypatch.setattr(
        corner_ppc_module, "_render_offdiag_panel", lambda *args, **kwargs: None
    )

    result = plot_corner_ppc(
        _FakeResultsForCorner(counts.shape[1]),
        counts,
        n_genes=2,
        n_samples=3,
        save=False,
        show=False,
    )

    # The diagonal renderer receives per-gene PPC slices from the sampled
    # subset object; shape (n_samples, n_cells) confirms predictive cache use.
    assert seen["diag_called"] is True
    assert seen["diag_ppc_shape"] == (3, counts.shape[0])
    plt.close(result.fig)
