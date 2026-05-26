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

    def test_abundance_strategy_uses_compositional_selector(
        self, fake_results, synth_counts, monkeypatch
    ):
        """`gene_selection='abundance'` should route through the abundance path."""
        # Patch the compositional selector so we can assert routing semantics
        # without depending on the concrete quantile output.
        import scribe.viz.compositional_ppc as compositional_ppc_module

        seen = {"called": False}

        def _fake_abundance_selector(counts, n_genes, min_mean_umi=5.0):
            _ = counts, n_genes, min_mean_umi
            seen["called"] = True
            return np.array([0, 2, 4], dtype=int)

        monkeypatch.setattr(
            compositional_ppc_module,
            "_select_compositional_genes",
            _fake_abundance_selector,
        )

        idx = _resolve_gene_indices(
            fake_results,
            synth_counts,
            gene_indices=None,
            gene_names_list=None,
            n_genes=3,
            gene_selection="abundance",
        )

        assert seen["called"] is True
        np.testing.assert_array_equal(idx, [0, 2, 4])

    def test_correlation_diverse_strategy_uses_diversity_selector(
        self, fake_results, synth_counts, monkeypatch
    ):
        """`gene_selection='correlation_diverse'` should use diversity path."""
        # Patch both correlation dispatch and selector so this test checks only
        # control-flow routing, not matrix algebra details.
        monkeypatch.setattr(
            corner_ppc_module,
            "_get_correlation_matrix_for_selection",
            lambda *args, **kwargs: (np.eye(synth_counts.shape[1]), synth_counts.shape[1]),
        )
        seen = {"called": False}

        def _fake_diverse_selector(
            corr_matrix,
            n_genes,
            counts,
            *,
            min_mean_umi_for_selection=None,
            exclude_idx=None,
        ):
            _ = corr_matrix, n_genes, counts, min_mean_umi_for_selection, exclude_idx
            seen["called"] = True
            return np.array([1, 3, 5], dtype=int)

        monkeypatch.setattr(
            corner_ppc_module,
            "_select_genes_by_correlation_diversity",
            _fake_diverse_selector,
        )

        idx = _resolve_gene_indices(
            fake_results,
            synth_counts,
            gene_indices=None,
            gene_names_list=None,
            n_genes=3,
            gene_selection="correlation_diverse",
        )

        assert seen["called"] is True
        np.testing.assert_array_equal(idx, [1, 3, 5])

    def test_invalid_gene_selection_raises(self, fake_results, synth_counts):
        """Unknown `gene_selection` should raise a clear ValueError."""
        with pytest.raises(ValueError, match="gene_selection must be"):
            _resolve_gene_indices(
                fake_results,
                synth_counts,
                gene_indices=None,
                gene_names_list=None,
                n_genes=3,
                gene_selection="unknown_strategy",
            )

    def test_explicit_indices_bypass_gene_selection_validation(
        self, fake_results, synth_counts
    ):
        """Explicit indices should short-circuit before strategy validation."""
        idx = _resolve_gene_indices(
            fake_results,
            synth_counts,
            gene_indices=[2, 6],
            gene_names_list=None,
            n_genes=3,
            gene_selection="unknown_strategy",
        )
        np.testing.assert_array_equal(idx, [2, 6])

    def test_gene_names_bypass_gene_selection_validation(
        self, fake_results, synth_counts
    ):
        """Name-based selection should short-circuit before strategy validation."""
        idx = _resolve_gene_indices(
            fake_results,
            synth_counts,
            gene_indices=None,
            gene_names_list=["Gene_1", "Gene_4"],
            n_genes=3,
            gene_selection="unknown_strategy",
        )
        np.testing.assert_array_equal(idx, [1, 4])


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

    def test_contours_are_drawn_above_scatter(self):
        """Contour layers should use higher z-order than scatter points."""
        rng = np.random.default_rng(123)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(12, 35))
        ppc_y = rng.negative_binomial(n=7, p=0.4, size=(12, 35))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=35)
        obs_y = rng.negative_binomial(n=7, p=0.4, size=35)

        fig, ax = plt.subplots()
        # Render one off-diagonal panel, then inspect collection layering.
        _render_offdiag_panel(ax, ppc_x, ppc_y, obs_x, obs_y)
        zorders = {float(coll.get_zorder()) for coll in ax.collections}

        # Scatter at zorder=1 and contour fill at zorder=2.
        assert 1.0 in zorders
        assert 2.0 in zorders
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

    def test_hist2d_default_does_not_call_kde(self, monkeypatch):
        """Default off-diagonal rendering should use fast hist2d, not KDE."""
        rng = np.random.default_rng(111)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        # If KDE is called in default mode this test should fail immediately.
        monkeypatch.setattr(
            corner_ppc_module,
            "gaussian_kde",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("gaussian_kde should not be called")
            ),
        )

        fig, ax = plt.subplots()
        _render_offdiag_panel(ax, ppc_x, ppc_y, obs_x, obs_y)
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_kde_mode_uses_kde_path(self, monkeypatch):
        """KDE mode should execute gaussian_kde and still render."""
        rng = np.random.default_rng(222)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)
        seen = {"called": False}

        # Build a minimal KDE stub that matches the callable contract.
        class _FakeKDE:
            def __call__(self, positions):
                _n = positions.shape[1]
                return np.ones(_n, dtype=float)

        def _fake_kde(_stacked):
            seen["called"] = True
            return _FakeKDE()

        monkeypatch.setattr(corner_ppc_module, "gaussian_kde", _fake_kde)

        fig, ax = plt.subplots()
        _render_offdiag_panel(
            ax, ppc_x, ppc_y, obs_x, obs_y, density_method="kde"
        )
        assert seen["called"] is True
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_invalid_density_method_raises(self):
        """Invalid density method should raise a clear ValueError."""
        rng = np.random.default_rng(333)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="density_method"):
            _render_offdiag_panel(
                ax,
                ppc_x,
                ppc_y,
                obs_x,
                obs_y,
                density_method="not_a_method",
            )
        plt.close(fig)

    def test_invalid_contour_mass_levels_raise(self):
        """HPD contour mass levels must lie strictly in (0, 1)."""
        rng = np.random.default_rng(444)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="contour_mass_levels"):
            _render_offdiag_panel(
                ax,
                ppc_x,
                ppc_y,
                obs_x,
                obs_y,
                contour_mass_levels=(0.5, 1.2),
            )
        plt.close(fig)

    def test_hpd_contours_render_in_hist2d_mode(self):
        """Default HPD mass contours render without error in hist2d mode."""
        rng = np.random.default_rng(555)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        fig, ax = plt.subplots()
        _render_offdiag_panel(
            ax,
            ppc_x,
            ppc_y,
            obs_x,
            obs_y,
            density_method="hist2d",
            contour_mass_levels=(0.5, 0.68, 0.95, 0.99),
        )
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_contour_edges_drawn_on_continuous_panel_by_default(self):
        """Default rendering should draw edges on smooth continuous panels."""
        rng = np.random.default_rng(556)
        ppc_x = rng.normal(loc=5.0, scale=1.5, size=(10, 30))
        ppc_y = rng.normal(loc=8.0, scale=2.0, size=(10, 30))
        obs_x = rng.normal(loc=5.0, scale=1.5, size=30)
        obs_y = rng.normal(loc=8.0, scale=2.0, size=30)

        class _RecordingAxis:
            """Minimal axis recorder for contour-edge call assertions."""

            def __init__(self):
                self.contour_calls = []

            def scatter(self, *args, **kwargs):
                _ = args, kwargs

            def contourf(self, *args, **kwargs):
                _ = args, kwargs

            def contour(self, *args, **kwargs):
                self.contour_calls.append((args, kwargs))

        axis = _RecordingAxis()
        _render_offdiag_panel(axis, ppc_x, ppc_y, obs_x, obs_y)
        assert len(axis.contour_calls) > 0
        _, kwargs = axis.contour_calls[-1]
        assert kwargs["colors"] == "gray"

    def test_contour_edges_suppressed_for_discrete_panels_by_default(self):
        """Discrete low-count panels should auto-suppress contour edges."""
        rng = np.random.default_rng(558)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        class _RecordingAxis:
            """Minimal axis recorder for contour-edge call assertions."""

            def __init__(self):
                self.contour_calls = []

            def scatter(self, *args, **kwargs):
                _ = args, kwargs

            def contourf(self, *args, **kwargs):
                _ = args, kwargs

            def contour(self, *args, **kwargs):
                self.contour_calls.append((args, kwargs))

        axis = _RecordingAxis()
        _render_offdiag_panel(axis, ppc_x, ppc_y, obs_x, obs_y)
        assert len(axis.contour_calls) == 0

    def test_contour_edges_can_be_disabled(self):
        """Contour-line outlines should be optional."""
        rng = np.random.default_rng(557)
        ppc_x = rng.negative_binomial(n=5, p=0.3, size=(10, 30))
        ppc_y = rng.negative_binomial(n=8, p=0.4, size=(10, 30))
        obs_x = rng.negative_binomial(n=5, p=0.3, size=30)
        obs_y = rng.negative_binomial(n=8, p=0.4, size=30)

        class _RecordingAxis:
            """Minimal axis recorder for contour-edge call assertions."""

            def __init__(self):
                self.contour_calls = []

            def scatter(self, *args, **kwargs):
                _ = args, kwargs

            def contourf(self, *args, **kwargs):
                _ = args, kwargs

            def contour(self, *args, **kwargs):
                self.contour_calls.append((args, kwargs))

        axis = _RecordingAxis()
        _render_offdiag_panel(
            axis,
            ppc_x,
            ppc_y,
            obs_x,
            obs_y,
            draw_contour_edges=False,
        )
        assert len(axis.contour_calls) == 0

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

    def test_min_mean_umi_threshold_filters_candidates(self):
        """Low-mean genes should be excluded from auto-selection candidates."""
        # Build a simple correlation matrix where all genes are otherwise valid.
        corr = np.eye(4, dtype=float)
        # Genes 0 and 1 stay below threshold; genes 2 and 3 exceed it.
        counts = np.array(
            [
                [1.0, 2.0, 8.0, 10.0],
                [2.0, 3.0, 7.0, 9.0],
                [1.0, 2.0, 9.0, 8.0],
            ]
        )
        selected = _select_genes_by_correlation_diversity(
            corr, 4, counts, min_mean_umi_for_selection=5.0
        )
        assert np.all(np.isin(selected, [2, 3]))
        assert len(selected) == 2

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

    def test_explicit_gene_indices_ignore_mean_umi_threshold(
        self, fake_results, synth_counts
    ):
        """Explicit indices should bypass the mean-UMI auto-selection filter."""
        idx = _resolve_gene_indices(
            fake_results,
            synth_counts,
            gene_indices=[0, 1],
            gene_names_list=None,
            n_genes=2,
            min_mean_umi_for_selection=1e6,
        )
        np.testing.assert_array_equal(idx, np.array([0, 1], dtype=int))

    def test_raises_when_no_genes_pass_mean_umi_threshold(
        self, fake_results, synth_counts
    ):
        """Auto-selection should fail clearly when threshold excludes all genes."""
        with pytest.raises(ValueError, match="No genes passed auto-selection"):
            _resolve_gene_indices(
                fake_results,
                synth_counts,
                gene_indices=None,
                gene_names_list=None,
                n_genes=4,
                min_mean_umi_for_selection=1e6,
            )


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


def test_plot_corner_ppc_matches_offdiag_limits_to_marginals_by_default(
    monkeypatch,
):
    """Off-diagonal limits should follow diagonal marginal limits by default."""
    # Build two-gene counts with clearly different scales so x/y matching is
    # easy to observe on the lower-triangle panel.
    counts = np.column_stack(
        [
            np.linspace(0, 15, 60),
            np.linspace(80, 400, 60),
        ]
    )

    class _FakeResultsForLimits:
        """Minimal subsettable results object with predictive cache."""

        def __init__(self, n_genes):
            self.n_genes = int(n_genes)
            self.model_config = types.SimpleNamespace(inference_method="svi")
            self.predictive_samples = None
            self.var = None

        def __getitem__(self, index):
            subset = _FakeResultsForLimits(len(index))
            if self.predictive_samples is not None:
                idx = np.asarray(index, dtype=int)
                subset.predictive_samples = self.predictive_samples[:, :, idx]
            return subset

    def _fake_predictive(
        sampling_results, *, rng_key, n_samples, counts, store_samples,
        ppc_level=None,
    ):
        _ = rng_key, store_samples, ppc_level
        # Populate predictive samples on the same count scale so diagonal
        # histogram limits are meaningful and deterministic in this test.
        base = np.broadcast_to(
            counts[None, :, :],
            (n_samples, counts.shape[0], counts.shape[1]),
        ).astype(float)
        sampling_results.predictive_samples = base
        return sampling_results.predictive_samples

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
        lambda *args, **kwargs: np.array([0, 1], dtype=int),
    )
    monkeypatch.setattr(
        corner_ppc_module, "_resolve_ppc_grid", lambda **kwargs: {"n_samples": 4}
    )
    monkeypatch.setattr(corner_ppc_module, "_get_gene_names", lambda results: None)
    monkeypatch.setattr(
        corner_ppc_module, "_get_predictive_samples_for_plot", _fake_predictive
    )

    result = plot_corner_ppc(
        _FakeResultsForLimits(counts.shape[1]),
        counts,
        n_genes=2,
        n_samples=4,
        save=False,
        show=False,
    )

    # Validate that lower-triangle limits match the diagonal marginals:
    # x-limit from column gene and y-limit from row gene.
    axes_grid = np.asarray(result.axes, dtype=object).reshape(2, 2)
    lower_ax = axes_grid[1, 0]
    assert np.allclose(lower_ax.get_xlim(), axes_grid[0, 0].get_xlim())
    assert np.allclose(lower_ax.get_ylim(), axes_grid[1, 1].get_xlim())
    plt.close(result.fig)


def test_plot_corner_ppc_uses_marginal_ranges_for_offdiag_density(monkeypatch):
    """Off-diagonal density ranges should be sourced from diagonal marginals."""
    counts = np.column_stack(
        [
            np.linspace(0, 20, 40),
            np.linspace(50, 250, 40),
        ]
    )
    seen = {}

    class _FakeResultsForRanges:
        """Subsettable fake results object used for panel-range wiring tests."""

        def __init__(self, n_genes):
            self.n_genes = int(n_genes)
            self.model_config = types.SimpleNamespace(inference_method="svi")
            self.predictive_samples = None
            self.var = None

        def __getitem__(self, index):
            subset = _FakeResultsForRanges(len(index))
            if self.predictive_samples is not None:
                idx = np.asarray(index, dtype=int)
                subset.predictive_samples = self.predictive_samples[:, :, idx]
            return subset

    def _fake_predictive(
        sampling_results, *, rng_key, n_samples, counts, store_samples,
        ppc_level=None,
    ):
        _ = rng_key, store_samples, ppc_level
        base = np.broadcast_to(
            counts[None, :, :],
            (n_samples, counts.shape[0], counts.shape[1]),
        ).astype(float)
        sampling_results.predictive_samples = base
        return sampling_results.predictive_samples

    def _fake_diag(axis, ppc_samples, true_counts, render_opts, gene_label=None):
        _ = ppc_samples, render_opts, gene_label
        # Set deterministic diagonal x-limits keyed by gene identity.
        if float(np.mean(true_counts)) < 40.0:
            axis.set_xlim(10.0, 20.0)
        else:
            axis.set_xlim(100.0, 200.0)

    def _fake_offdiag(
        axis, ppc_x, ppc_y, obs_x, obs_y, *, x_range=None, y_range=None, **kwargs
    ):
        _ = axis, ppc_x, ppc_y, obs_x, obs_y, kwargs
        seen["x_range"] = tuple(x_range)
        seen["y_range"] = tuple(y_range)

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
        lambda *args, **kwargs: np.array([0, 1], dtype=int),
    )
    monkeypatch.setattr(
        corner_ppc_module, "_resolve_ppc_grid", lambda **kwargs: {"n_samples": 3}
    )
    monkeypatch.setattr(corner_ppc_module, "_get_gene_names", lambda results: None)
    monkeypatch.setattr(
        corner_ppc_module, "_get_predictive_samples_for_plot", _fake_predictive
    )
    monkeypatch.setattr(corner_ppc_module, "_render_diagonal_panel", _fake_diag)
    monkeypatch.setattr(corner_ppc_module, "_render_offdiag_panel", _fake_offdiag)

    result = plot_corner_ppc(
        _FakeResultsForRanges(counts.shape[1]),
        counts,
        n_genes=2,
        n_samples=3,
        save=False,
        show=False,
    )

    assert seen["x_range"] == (10.0, 20.0)
    assert seen["y_range"] == (100.0, 200.0)
    plt.close(result.fig)
