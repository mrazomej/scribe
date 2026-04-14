"""Consolidated tests for the scribe.viz package refactor.

This module centralizes the coverage that previously lived in separate
viz-utils-focused test files. It validates cache helpers, dispatch helpers,
and package-root compatibility exports.
"""

from unittest.mock import MagicMock
import sys
import types

import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

from scribe.mcmc.results import ScribeMCMCResults
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models import get_model_and_guide
from scribe.models.config import ModelConfig
from scribe.svi.results import ScribeSVIResults
import scribe.viz.config as viz_config
from scribe.viz import (
    PlotResult,
    _build_umap_cache_path,
    _get_config_values,
    _get_predictive_samples_for_plot,
    _get_training_diagnostic_payload,
    plot_annotation_ppc,
    plot_bio_ppc,
    plot_capture_anchor,
    plot_p_capture_scaling,
    plot_correlation_heatmap,
    plot_ecdf,
    plot_loss,
    plot_mixture_composition,
    plot_mixture_ppc,
    plot_mixture_ppc_comparison,
    plot_mixture_ppc_components,
    plot_mixture_ppc_overview,
    plot_mu_pairwise,
    plot_ppc,
    plot_umap,
)
from scribe.viz.ppc_rendering import (
    compute_adaptive_max_bin,
    get_ppc_render_options,
    should_use_line_mode,
)
from scribe.viz.mixture_ppc import (
    _resolve_label_map_for_composition,
    _resolve_weight_fractions_for_composition,
)
from scribe.viz.annotation_ppc import (
    _resolve_label_map as _resolve_annotation_label_map,
)


def _make_mcmc_results_for_viz():
    """Create a compact MCMC results object for unit tests."""
    samples = {
        "p": jnp.array([0.3, 0.4, 0.5], dtype=jnp.float32),
        "r": jnp.array(
            [
                [1.0, 1.5],
                [1.1, 1.6],
                [1.2, 1.7],
            ],
            dtype=jnp.float32,
        ),
    }
    config = ModelConfig(base_model="nbdm", n_components=None)
    return ScribeMCMCResults(
        samples=samples,
        n_cells=3,
        n_genes=2,
        model_type="nbdm",
        model_config=config,
        prior_params={},
    )


def test_package_root_exports_expected_symbols():
    """Ensure refactor preserves key package-root imports and callables."""
    assert callable(plot_loss)
    assert callable(plot_ecdf)
    assert callable(plot_ppc)
    assert callable(plot_bio_ppc)
    assert callable(plot_umap)
    assert callable(plot_correlation_heatmap)
    assert callable(plot_mixture_ppc)
    assert callable(plot_mixture_ppc_overview)
    assert callable(plot_mixture_ppc_components)
    assert callable(plot_mixture_ppc_comparison)
    assert callable(plot_mixture_composition)
    assert callable(plot_annotation_ppc)
    assert callable(plot_capture_anchor)
    assert callable(plot_p_capture_scaling)
    assert callable(plot_mu_pairwise)
    assert callable(_get_config_values)
    assert callable(_get_predictive_samples_for_plot)
    assert callable(_get_training_diagnostic_payload)
    assert callable(_build_umap_cache_path)


def test_get_config_values_uses_mcmc_run_size_token():
    """MCMC config metadata should generate samples/warmup filename tokens."""
    cfg = OmegaConf.create(
        {
            "inference": {
                "method": "mcmc",
                "parameterization": "standard",
                "n_components": 1,
                "n_samples": 1200,
                "n_warmup": 300,
            },
            "model": "nbdm",
        }
    )
    results = _make_mcmc_results_for_viz()
    values = _get_config_values(cfg, results=results)
    assert values["run_size_token"] == "1200samples_300warmup"
    assert values["run_size_label"] == "samples"


def test_predictive_helper_truncates_mcmc_draws():
    """MCMC predictive helper should cap draw count for plotting."""
    results = _make_mcmc_results_for_viz()
    full_predictive = np.arange(30, dtype=np.int32).reshape(5, 3, 2)

    def _fake_get_ppc_samples(
        rng_key=None, batch_size=None, store_samples=True
    ):
        _ = rng_key
        _ = batch_size
        if store_samples:
            results.predictive_samples = full_predictive
        return full_predictive

    results.get_ppc_samples = _fake_get_ppc_samples

    selected = _get_predictive_samples_for_plot(
        results,
        rng_key=random.PRNGKey(0),
        n_samples=2,
        counts=np.zeros((3, 2), dtype=np.int32),
        batch_size=None,
        store_samples=True,
    )
    assert selected.shape == (2, 3, 2)
    # MCMC helper samples draws without replacement, so order/content are
    # deterministic for a fixed key but not guaranteed to be the first draws.
    assert len({draw.tobytes() for draw in selected}) == 2
    for draw in selected:
        assert any(
            np.array_equal(draw, candidate) for candidate in full_predictive
        )


def test_training_payload_includes_mcmc_diagnostics():
    """MCMC diagnostics payload should include trace and divergence info."""
    results = _make_mcmc_results_for_viz()
    mock_mcmc = MagicMock()
    mock_mcmc.get_extra_fields.return_value = {
        "potential_energy": jnp.array([5.0, 4.0, 3.0]),
        "diverging": jnp.array([0, 1, 0], dtype=jnp.int32),
    }
    mock_mcmc.get_samples.return_value = {
        "r": jnp.ones((2, 3, 2), dtype=jnp.float32)
    }
    results._mcmc = mock_mcmc

    payload = _get_training_diagnostic_payload(results)
    assert payload["plot_kind"] == "mcmc_diagnostics"
    assert payload["potential_energy"].shape == (3,)
    assert payload["diverging"].shape == (3,)
    assert payload["trace_by_chain"].shape == (2, 3)


def test_map_dispatch_forwards_targets_for_svi():
    """SVI map-dispatch helper should forward requested targets."""
    from scribe.viz.dispatch import _get_map_estimates_for_plot

    results = ScribeSVIResults(
        params={},
        loss_history=jnp.array([1.0], dtype=jnp.float32),
        n_cells=1,
        n_genes=1,
        model_type="nbdm",
        model_config=ModelConfig(base_model="nbdm"),
        prior_params={},
    )
    captured = {}

    def _fake_get_map(**kwargs):
        captured.update(kwargs)
        return {"r": np.array([1.0], dtype=float)}

    # Instance-level override lets the dispatch helper stay type-correct while
    # keeping the test lightweight and focused on forwarded kwargs.
    results.get_map = _fake_get_map

    out = _get_map_estimates_for_plot(
        results,
        counts=np.zeros((1, 1), dtype=float),
        use_mean=False,
        targets=["r", "p"],
    )
    assert set(out.keys()) == {"r"}
    assert captured["targets"] == ["r", "p"]
    assert captured["use_mean"] is False


def test_map_dispatch_ignores_targets_for_mcmc():
    """MCMC map-dispatch helper should ignore selective targets."""
    from scribe.viz.dispatch import _get_map_estimates_for_plot

    results = _make_mcmc_results_for_viz()
    captured = {}

    def _fake_get_map(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"r": np.array([1.0], dtype=float)}

    results.get_map = _fake_get_map

    out = _get_map_estimates_for_plot(
        results,
        counts=np.zeros((3, 2), dtype=float),
        use_mean=False,
        targets=["r"],
    )
    assert set(out.keys()) == {"r"}
    assert captured["args"] == ()
    assert captured["kwargs"] == {}


def test_umap_cache_path_without_subset_uses_full_suffix():
    """Unsplit runs should use a deterministic ``_full`` cache suffix."""
    cfg = OmegaConf.create({"data": {"path": "/tmp/example_dataset.h5ad"}})
    cache_path = _build_umap_cache_path(cfg=cfg, cache_umap=True)
    assert cache_path == "/tmp/example_dataset_umap_full.pkl"


def test_umap_cache_path_changes_across_subset_values():
    """Different split values should generate different cache paths."""
    cfg_a = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": "exp_condition",
                "subset_value": "bleo_d2",
            }
        }
    )
    cfg_b = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": "exp_condition",
                "subset_value": "bleo_d7",
            }
        }
    )
    cache_path_a = _build_umap_cache_path(cfg=cfg_a, cache_umap=True)
    cache_path_b = _build_umap_cache_path(cfg=cfg_b, cache_umap=True)
    assert cache_path_a != cache_path_b
    assert "exp_condition" in cache_path_a
    assert "bleo_d2" in cache_path_a
    assert "exp_condition" in cache_path_b
    assert "bleo_d7" in cache_path_b


def test_umap_cache_path_is_stable_for_same_subset():
    """Repeated calls with identical split config should be stable."""
    cfg = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": "exp_condition",
                "subset_value": "bleo_d2",
            }
        }
    )
    cache_path_1 = _build_umap_cache_path(cfg=cfg, cache_umap=True)
    cache_path_2 = _build_umap_cache_path(cfg=cfg, cache_umap=True)
    assert cache_path_1 == cache_path_2


def test_umap_cache_path_multi_column_differs_from_single_column():
    """A multi-column subset key must not collide with a single-column key."""
    cfg_single = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": "treatment",
                "subset_value": "drug",
            }
        }
    )
    # List-valued subset_column / subset_value (multi-column case)
    cfg_multi = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": ["treatment", "kit"],
                "subset_value": ["drug", "10x"],
            }
        }
    )
    path_single = _build_umap_cache_path(cfg=cfg_single, cache_umap=True)
    path_multi = _build_umap_cache_path(cfg=cfg_multi, cache_umap=True)
    assert path_single != path_multi


def test_umap_cache_path_multi_column_is_stable():
    """Repeated calls with identical multi-column split config should be stable."""
    cfg = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
                "subset_column": ["treatment", "kit"],
                "subset_value": ["drug", "10x"],
            }
        }
    )
    assert _build_umap_cache_path(cfg=cfg, cache_umap=True) == (
        _build_umap_cache_path(cfg=cfg, cache_umap=True)
    )


def test_umap_cache_path_multi_column_changes_across_combinations():
    """Different multi-column value combinations must produce different cache paths."""

    def _cfg(treatment, kit):
        return OmegaConf.create(
            {
                "data": {
                    "path": "/tmp/example_dataset.h5ad",
                    "subset_column": ["treatment", "kit"],
                    "subset_value": [treatment, kit],
                }
            }
        )

    path_a = _build_umap_cache_path(cfg=_cfg("drug", "10x"), cache_umap=True)
    path_b = _build_umap_cache_path(cfg=_cfg("ctrl", "10x"), cache_umap=True)
    path_c = _build_umap_cache_path(
        cfg=_cfg("drug", "dropseq"), cache_umap=True
    )
    assert len({path_a, path_b, path_c}) == 3


def test_plot_umap_subsets_ppc_to_hvg_genes(monkeypatch):
    """UMAP PPC sampling should use detected+HVG genes, not detected genes only."""
    import scribe.viz.umap as umap_module

    n_cells, n_genes = 5, 6
    counts = np.array(
        [
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0],
        ],
        dtype=float,
    )

    # Keep test deterministic and focused on indexing semantics.
    monkeypatch.setattr(umap_module, "_coerce_counts", lambda x: x)

    class _DummyReducer:
        def fit_transform(self, x):
            return np.stack([x[:, 0], x[:, 1]], axis=1)

        def transform(self, x):
            return np.stack([x[:, 0], x[:, 1]], axis=1)

    class _DummyUMAP:
        def __init__(self, **kwargs):
            _ = kwargs

        def fit_transform(self, x):
            return _DummyReducer().fit_transform(x)

        def transform(self, x):
            return _DummyReducer().transform(x)

    monkeypatch.setitem(
        sys.modules,
        "umap",
        types.SimpleNamespace(UMAP=_DummyUMAP),
    )

    class _DummyAnnData:
        def __init__(self, X):
            self.X = X
            self.var = {}

    def _dummy_hvg(adata_hvg, n_top_genes, flavor, inplace):
        _ = flavor, inplace
        mask = np.zeros(adata_hvg.X.shape[1], dtype=bool)
        mask[:n_top_genes] = True
        adata_hvg.var["highly_variable"] = mask

    monkeypatch.setitem(
        sys.modules,
        "anndata",
        types.SimpleNamespace(AnnData=_DummyAnnData),
    )
    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        types.SimpleNamespace(
            pp=types.SimpleNamespace(highly_variable_genes=_dummy_hvg)
        ),
    )

    class _DummyPCA:
        def __init__(self, n_components, random_state):
            self.n_components = int(n_components)
            _ = random_state

        def fit_transform(self, x):
            return x[:, : self.n_components]

        def transform(self, x):
            return x[:, : self.n_components]

    monkeypatch.setitem(
        sys.modules,
        "sklearn.decomposition",
        types.SimpleNamespace(PCA=_DummyPCA),
    )

    seen = {}

    def _fake_predictive(results_sub, **kwargs):
        # Track which gene count the UMAP path asks PPC sampling to use.
        seen["subset_n_genes"] = results_sub.n_genes
        n_cells_local = kwargs["counts"].shape[0]
        return np.zeros((1, n_cells_local, results_sub.n_genes), dtype=float)

    monkeypatch.setattr(
        umap_module, "_get_predictive_samples_for_plot", _fake_predictive
    )

    class _FakeResults:
        def __init__(self, n_genes):
            self.n_genes = int(n_genes)

        def __getitem__(self, index):
            return _FakeResults(len(index))

    # Detected genes with min_cells=3: indices [0, 1, 2, 4] => 4 genes.
    # HVG selection keeps n_top_genes=2 among detected => expected PPC subset=2.
    result = umap_module.plot_umap(
        _FakeResults(n_genes=n_genes),
        counts,
        save=False,
        viz_cfg=OmegaConf.create(
            {
                "umap_opts": {
                    "gene_filter_min_cells": 3,
                    "hvg_n_top_genes": 2,
                    "use_hvg": True,
                    "use_pca": False,
                    "cache_umap": False,
                    "n_ppc_samples": 1,
                }
            }
        ),
    )

    assert isinstance(result, PlotResult)
    assert seen["subset_n_genes"] == 2
    plt.close(result.fig)


def test_apply_scale_to_synth_scanpy_matches_manual():
    """Synthetic rows should match Scanpy-style (x - mean) / std clipping."""
    import scribe.viz.umap as umap_module

    rng = np.random.default_rng(0)
    synth_norm = rng.random(size=(4, 3))
    mean = np.array([0.1, 0.2, 0.3])
    std = np.array([0.5, 1.0, 2.0])
    scale_meta = {
        "kind": "scanpy",
        "mean": mean,
        "std": std,
        "max_value": 10.0,
    }
    out = umap_module._apply_scale_to_synth(synth_norm, scale_meta)
    expected = (synth_norm - mean) / std
    expected = np.clip(expected, -10.0, 10.0)
    np.testing.assert_allclose(out, expected)


def test_plot_bio_ppc_aligns_counts_subset_with_results_order(
    monkeypatch, tmp_path
):
    """Bio-PPC denoising must align count columns with subsetted result order.

    The ``results[selected_idx]`` path keeps genes in original-order semantics
    (boolean-mask subsetting), while ``selected_idx`` itself is log-bin order.
    This test ensures Bio-PPC denoising receives counts in original index order
    so denoised histograms align with the biological PPC parameter subset.
    """
    import scribe.viz.bio_ppc as bio_ppc_module

    # Keep selected_idx intentionally unsorted to exercise the alignment path.
    selected_idx = np.array([4, 1, 3], dtype=int)
    sorted_idx = np.array([1, 3, 4], dtype=int)
    n_cells = 5
    n_genes = 6
    counts = np.arange(n_cells * n_genes).reshape(n_cells, n_genes)

    class _FakeResults:
        """Minimal results stub supporting gene subsetting and config access."""

        model_type = "nbvcp"
        n_components = None

        def __getitem__(self, index):
            _ = index
            return self

    captured = {}

    def _fake_select_genes(_counts, _rows, _cols):
        means = np.median(_counts, axis=0)
        return selected_idx, means

    def _fake_bio_samples(
        _results, *, rng_key, n_samples, counts, batch_size, store_samples
    ):
        _ = rng_key, counts, batch_size, store_samples
        # Shape: (n_samples, n_cells, n_selected_genes)
        return np.zeros((n_samples, n_cells, selected_idx.size), dtype=float)

    def _fake_denoised(_results, *, counts, rng_key, method, cell_batch_size):
        _ = rng_key, method, cell_batch_size
        captured["counts_for_denoise"] = counts.copy()
        return np.zeros_like(counts, dtype=float)

    monkeypatch.setattr(bio_ppc_module, "_select_genes", _fake_select_genes)
    monkeypatch.setattr(
        bio_ppc_module,
        "_get_biological_ppc_samples_for_plot",
        _fake_bio_samples,
    )
    monkeypatch.setattr(
        bio_ppc_module, "_get_denoised_counts_for_plot", _fake_denoised
    )
    monkeypatch.setattr(
        bio_ppc_module.scribe.viz,
        "plot_histogram_credible_regions_stairs",
        lambda *args, **kwargs: None,
    )

    cfg = OmegaConf.create(
        {
            "inference": {
                "method": "svi",
                "parameterization": "mean_odds",
                "n_components": 1,
            },
            "model": "zinbvcp",
        }
    )
    viz_cfg = OmegaConf.create(
        {
            "ppc_opts": {"n_rows": 1, "n_cols": 3, "n_samples": 2},
            "bio_ppc_opts": {"denoise_cell_batch_size": 2},
            "format": "png",
        }
    )

    plot_bio_ppc(
        _FakeResults(),
        counts=counts,
        figs_dir=str(tmp_path),
        cfg=cfg,
        viz_cfg=viz_cfg,
    )

    expected = counts[:, sorted_idx]
    np.testing.assert_array_equal(captured["counts_for_denoise"], expected)


def test_plot_bio_ppc_joint_gate_results_no_gate_loc_error(tmp_path):
    """Bio-PPC should run for joint gate configs without gate_loc KeyErrors.

    This regression exercises both biological PPC sampling and MAP denoising
    with a fitted joint low-rank ZINBVCP model where ``gate`` is in
    ``joint_params``.
    """
    n_cells, n_genes = 24, 6
    key = random.PRNGKey(321)
    counts = random.poisson(key, lam=4.0, shape=(n_cells, n_genes))

    config = build_config_from_preset(
        model="zinbvcp",
        parameterization="mean_odds",
        unconstrained=True,
        n_components=2,
        prob_prior="gaussian",
        zero_inflation_prior="gaussian",
        guide_rank=2,
        joint_params=["mu", "phi", "gate"],
        priors={"eta_capture": (11.51, 0.01)},
    )
    model_fn, guide_fn, config = get_model_and_guide(config)
    model_kwargs = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": config,
        "counts": counts,
    }

    # Fit briefly to construct realistic variational parameters for plotting.
    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(322), **model_kwargs)
    for _ in range(3):
        svi_state, _ = svi.update(svi_state, **model_kwargs)

    results = ScribeSVIResults(
        params=svi.get_params(svi_state),
        loss_history=jnp.array([1.0, 0.5]),
        n_cells=n_cells,
        n_genes=n_genes,
        model_type="zinbvcp",
        model_config=config,
        prior_params={},
    )

    cfg = OmegaConf.create(
        {
            "inference": {"method": "svi", "n_steps": 100},
            "parameterization": "mean_odds",
            "model": "zinbvcp",
            "n_components": 2,
        }
    )
    viz_cfg = OmegaConf.create(
        {
            "ppc_opts": {"n_rows": 1, "n_cols": 2, "n_samples": 2},
            "bio_ppc_opts": {"denoise_cell_batch_size": 8},
            "format": "png",
        }
    )

    plot_bio_ppc(
        results=results,
        counts=counts,
        figs_dir=str(tmp_path),
        cfg=cfg,
        viz_cfg=viz_cfg,
    )
    assert any(
        path.name.endswith("_bio_ppc.png") for path in tmp_path.iterdir()
    )


def test_plot_capture_anchor_saves_output(monkeypatch, tmp_path):
    """Capture-anchor plot should write an output file with expected suffix.

    This test stubs map extraction and filename metadata so it can verify
    plotting behavior without requiring a full fitted results object.
    """
    import scribe.viz.capture_anchor as capture_anchor_module

    class _FakeResults:
        """Minimal stub used to satisfy result-object access in plotting."""

        model_type = "zinbvcp"
        n_components = 1

    # Provide deterministic eta values with one value per cell.
    eta_values = np.array([0.2, 0.4, 0.5, 0.1], dtype=float)
    counts = np.array(
        [
            [10, 0, 1],
            [4, 2, 0],
            [7, 1, 3],
            [2, 0, 0],
        ],
        dtype=float,
    )

    # Mock map extraction so the plot function receives eta values directly.
    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        assert targets == ["eta_capture"]
        return {"eta_capture": eta_values}

    monkeypatch.setattr(
        capture_anchor_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    # Mock filename metadata to keep assertion stable and focused.
    import scribe.viz.config as viz_config_module

    monkeypatch.setattr(
        viz_config_module,
        "_get_config_values",
        lambda *_args, **_kwargs: {
            "method": "svi",
            "parameterization": "mean_odds",
            "model_type": "zinbvcp",
            "n_components": 1,
            "run_size_token": "100steps",
        },
    )

    cfg = OmegaConf.create({"priors": {"eta_capture": [12.2, 1e-5]}})
    viz_cfg = OmegaConf.create(
        {
            "format": "png",
            "capture_anchor_opts": {
                "n_bins": 12,
                "scatter_size": 5,
                "scatter_alpha": 0.25,
            },
        }
    )

    result = plot_capture_anchor(
        _FakeResults(),
        counts=counts,
        figs_dir=str(tmp_path),
        cfg=cfg,
        viz_cfg=viz_cfg,
    )
    assert isinstance(result, PlotResult)
    assert result.output_path is not None
    assert result.output_path.endswith("_capture_anchor.png")


def test_plot_p_capture_scaling_saves_output(monkeypatch, tmp_path):
    """p-capture scaling plot should write an output file with expected suffix.

    This test stubs MAP extraction and assignment helpers so plotting can run
    with a lightweight fake results object.
    """
    import scribe.viz.capture_anchor as capture_anchor_module

    class _FakeResults:
        """Minimal result stub used by p-capture scaling plotting."""

        model_type = "zinbvcp"
        n_components = 2

    counts = np.array(
        [
            [10, 0, 1],
            [4, 2, 0],
            [7, 1, 3],
            [2, 0, 0],
            [6, 1, 0],
            [9, 2, 2],
        ],
        dtype=float,
    )
    p_capture = np.array([0.3, 0.5, 0.4, 0.2, 0.6, 0.55], dtype=float)

    # Stub map extraction to provide deterministic p_capture values.
    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        assert targets == ["p_capture"]
        return {"p_capture": p_capture}

    monkeypatch.setattr(
        capture_anchor_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    # Stub component assignment probabilities for mixture split plotting.
    monkeypatch.setattr(
        capture_anchor_module,
        "_get_cell_assignment_probabilities_for_plot",
        lambda *_args, **_kwargs: np.array(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.3, 0.7],
            ]
        ),
    )
    # Stub filename config to keep assertion stable.
    import scribe.viz.config as viz_config_module

    monkeypatch.setattr(
        viz_config_module,
        "_get_config_values",
        lambda *_args, **_kwargs: {
            "method": "svi",
            "parameterization": "mean_odds",
            "model_type": "zinbvcp",
            "n_components": 2,
            "run_size_token": "100steps",
        },
    )

    cfg = OmegaConf.create({})
    viz_cfg = OmegaConf.create(
        {
            "format": "png",
            "p_capture_scaling_opts": {
                "n_bins": 8,
                "min_cells_per_bin": 1,
                "assignment_batch_size": 4,
            },
        }
    )

    result = plot_p_capture_scaling(
        _FakeResults(),
        counts=counts,
        figs_dir=str(tmp_path),
        cfg=cfg,
        viz_cfg=viz_cfg,
        is_mixture=True,
        is_multi_dataset=True,
        dataset_codes=np.array([0, 0, 1, 1, 0, 1]),
        dataset_names=["A", "B"],
    )
    assert isinstance(result, PlotResult)
    assert result.output_path is not None
    assert result.output_path.endswith("_p_capture_scaling.png")


def test_ppc_render_options_defaults_are_stable():
    """PPC render helpers should expose stable high-bin defaults."""
    viz_cfg = OmegaConf.create({"ppc_opts": {}})
    opts = get_ppc_render_options(viz_cfg)
    assert opts["hist_max_bin_quantile"] == 0.99
    assert opts["hist_max_bin_floor"] == 10
    assert opts["render_auto_line_bin_threshold"] == 1000
    assert opts["render_line_target_points"] == 200
    assert opts["render_line_interpolate"] is True


def test_compute_adaptive_max_bin_uses_quantile_and_floor():
    """Adaptive max-bin helper should enforce configured floor."""
    counts = np.array([0, 0, 1, 2, 100], dtype=float)
    opts = {
        "hist_max_bin_quantile": 0.50,
        "hist_max_bin_floor": 10,
    }
    assert compute_adaptive_max_bin(counts, opts) == 10


def test_should_use_line_mode_obeys_threshold():
    """Large bin counts should trigger adaptive line rendering."""
    opts = {"render_auto_line_bin_threshold": 1000}
    assert should_use_line_mode(1001, opts) is True
    assert should_use_line_mode(1000, opts) is False


def test_plot_mu_pairwise_saves_output_for_multi_dataset(monkeypatch, tmp_path):
    """Mu pairwise plot should save output when multi-dataset mu is available.

    The test stubs MAP extraction and filename metadata to keep behavior
    deterministic and independent of heavy model objects.
    """
    import scribe.viz.mu_pairwise as mu_pairwise_module

    class _FakeResults:
        """Minimal result stub that exposes multi-dataset model metadata."""

        model_type = "zinb"
        n_components = 1

        def __init__(self):
            self.model_config = MagicMock(
                n_datasets=2,
                uses_variable_capture=False,
                is_bnb=False,
            )

    # Provide two dataset-specific mu vectors over shared genes.
    fake_mu = np.array(
        [
            [1.0, 2.0, 4.0, 8.0],
            [1.5, 3.0, 6.0, 12.0],
        ],
        dtype=float,
    )
    seen_targets = []

    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        seen_targets.append(targets)
        if targets == ["mu"]:
            return {"mu": fake_mu}
        if targets == ["mixing_weights"]:
            return {}
        raise AssertionError(f"Unexpected targets: {targets}")

    monkeypatch.setattr(
        mu_pairwise_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    # Provide AxisLayout metadata matching the fake mu shape (D=2, G=4).
    monkeypatch.setattr(
        mu_pairwise_module,
        "_get_layouts_for_plot",
        lambda _results: {"mu": AxisLayout(axes=("datasets", "genes"))},
    )
    monkeypatch.setattr(
        viz_config,
        "_get_config_values",
        lambda *_args, **_kwargs: {
            "method": "svi",
            "parameterization": "mean_odds",
            "model_type": "zinb",
            "n_components": 1,
            "run_size_token": "100steps",
        },
    )

    result = plot_mu_pairwise(
        results=_FakeResults(),
        counts=np.zeros((5, 4), dtype=float),
        figs_dir=str(tmp_path),
        cfg=OmegaConf.create({}),
        viz_cfg=OmegaConf.create({"format": "png"}),
        dataset_names=["A", "B"],
    )

    assert isinstance(result, PlotResult)
    assert result.output_path is not None
    assert result.output_path.endswith("_mean_pairwise.png")
    assert seen_targets == [["mu"], ["mixing_weights"]]


def test_plot_mean_calibration_requests_targeted_map(monkeypatch, tmp_path):
    """Mean calibration should request only the MAP keys it consumes."""
    import scribe.viz.mean_calibration as mean_calibration_module

    class _FakeResults:
        """Minimal result stub for mean-calibration plotting."""

        model_type = "nbdm"
        n_components = 1
        model_config = MagicMock(
            n_datasets=None,
            uses_variable_capture=False,
            is_bnb=False,
        )

    seen_targets = []

    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        seen_targets.append(targets)
        if targets == ["r", "p"]:
            return {"r": np.array([3.0, 4.0]), "p": np.array([0.4, 0.5])}
        raise AssertionError(f"Unexpected targets: {targets}")

    monkeypatch.setattr(
        mean_calibration_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    # Provide AxisLayout metadata matching the fake r, p shapes.
    monkeypatch.setattr(
        mean_calibration_module,
        "_get_layouts_for_plot",
        lambda _results: {
            "r": AxisLayout(axes=("genes",)),
            "p": AxisLayout(axes=("genes",)),
        },
    )
    monkeypatch.setattr(
        viz_config,
        "_get_config_values",
        lambda *_args, **_kwargs: {
            "method": "svi",
            "parameterization": "canonical",
            "model_type": "nbdm",
            "n_components": 1,
            "run_size_token": "100steps",
        },
    )

    result = mean_calibration_module.plot_mean_calibration(
        _FakeResults(),
        counts=np.array([[1.0, 2.0], [3.0, 1.0]], dtype=float),
        figs_dir=str(tmp_path),
        cfg=OmegaConf.create({}),
        viz_cfg=OmegaConf.create({"format": "png"}),
    )
    assert isinstance(result, PlotResult)
    assert result.output_path is not None
    assert result.output_path.endswith("_mean_calibration.png")
    assert seen_targets == [["r", "p"]]


def test_select_divergent_genes_requests_dynamic_target(monkeypatch):
    """Mixture PPC gene selection should request either ``mu`` or ``r`` only."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal result stub exposing parameterization metadata."""

        model_config = MagicMock(parameterization="mean_odds")

    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        assert targets == ["mu"]
        # 2 components, 3 genes
        return {"mu": jnp.array([[2.0, 3.0, 4.0], [3.0, 2.0, 5.0]])}

    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    selected, lfc = mixture_ppc_module._select_divergent_genes(
        _FakeResults(),
        counts=np.array([[1.0, 0.0, 2.0], [2.0, 1.0, 0.0]], dtype=float),
        n_rows=1,
        n_cols=2,
    )
    assert selected.shape[0] <= 2
    assert lfc.shape[0] == selected.shape[0]


def test_plot_mixture_ppc_interactive_default_renders_overview_only(
    monkeypatch,
):
    """Interactive default should emit only the overview figure."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for figure selection tests."""

        n_components = 2

    def _fake_prepare(*_args, **_kwargs):
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": np.array([0, 1], dtype=int),
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    calls = []

    def _fake_plot_ppc_figure(**kwargs):
        calls.append(kwargs["title"])
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    def _fake_plot_comparison(**_kwargs):
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(mixture_ppc_module, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_figure", _fake_plot_ppc_figure
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_comparison_figure", _fake_plot_comparison
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=None,
    )

    assert len(out) == 1
    assert calls == ["Mixture PPC (High CV Genes)"]


def test_plot_mixture_ppc_explicit_all_keeps_full_output(monkeypatch):
    """Explicit ``plots='all'`` should preserve legacy full output."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for figure selection tests."""

        n_components = 2

    def _fake_prepare(*_args, **_kwargs):
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": np.array([0, 1], dtype=int),
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    ppc_calls = []
    comparison_calls = []

    def _fake_plot_ppc_figure(**kwargs):
        ppc_calls.append(kwargs["title"])
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    def _fake_plot_comparison(**_kwargs):
        comparison_calls.append("comparison")
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(mixture_ppc_module, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_figure", _fake_plot_ppc_figure
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_comparison_figure", _fake_plot_comparison
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=None,
        plots="all",
    )

    assert len(out) == 4
    assert len(ppc_calls) == 3
    assert len(comparison_calls) == 1


def test_plot_mixture_ppc_supports_component_subset_selection(monkeypatch):
    """Component selectors should render only requested component plots."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for figure selection tests."""

        n_components = 3

    def _fake_prepare(*_args, **_kwargs):
        return {
            "n_components": 3,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 3, 1), dtype=float),
            "assignments": np.array([0, 1, 2], dtype=int),
            "component_samples_list": [
                np.ones((2, 3, 1), dtype=float),
                np.ones((2, 3, 1), dtype=float),
                np.ones((2, 3, 1), dtype=float),
            ],
            "render_opts": {},
        }

    titles = []

    def _fake_plot_ppc_figure(**kwargs):
        titles.append(kwargs["title"])
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(mixture_ppc_module, "_is_interactive_session", lambda: True)
    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_figure", _fake_plot_ppc_figure
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_plot_ppc_comparison_figure", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0], [3.0]], dtype=float),
        viz_cfg=None,
        plots=["component:2"],
    )

    assert len(out) == 1
    assert titles[0].startswith("Component 2 PPC")


def test_mixture_ppc_wrappers_forward_expected_plots(monkeypatch):
    """Convenience wrappers should forward stable selector values."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    seen = []

    def _fake_plot_mixture_ppc(*_args, **kwargs):
        seen.append(kwargs.get("plots"))
        return "ok"

    monkeypatch.setattr(mixture_ppc_module, "plot_mixture_ppc", _fake_plot_mixture_ppc)

    _ = mixture_ppc_module.plot_mixture_ppc_overview("r", "c")
    _ = mixture_ppc_module.plot_mixture_ppc_components(
        "r", "c", component_indices=[1, 3]
    )
    _ = mixture_ppc_module.plot_mixture_ppc_comparison("r", "c")

    assert seen == ["mixture", ["component:1", "component:3"], "comparison"]


def test_plot_mixture_ppc_comparison_requests_assignments_by_default(
    monkeypatch,
):
    """Comparison mode should request assignments for observed overlays."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for comparison wiring tests."""

        n_components = 2

    prepared_kwargs = {}
    comparison_kwargs = {}

    def _fake_prepare(*_args, **kwargs):
        prepared_kwargs.update(kwargs)
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": np.array([0, 1], dtype=int),
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    def _fake_plot_comparison(**kwargs):
        comparison_kwargs.update(kwargs)
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_plot_ppc_comparison_figure",
        _fake_plot_comparison,
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=None,
        plots="comparison",
    )

    assert len(out) == 1
    assert prepared_kwargs["need_assignments"] is True
    assert prepared_kwargs["need_mixture_samples"] is False
    assert comparison_kwargs["show_component_observed"] is True
    assert comparison_kwargs["show_mixture_overlay"] is False
    assert comparison_kwargs["assignments"] is not None


def test_plot_mixture_ppc_comparison_can_disable_component_observed_overlay(
    monkeypatch,
):
    """Comparison overlay toggle should disable assignment dependency."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for comparison wiring tests."""

        n_components = 2

    prepared_kwargs = {}
    comparison_kwargs = {}

    def _fake_prepare(*_args, **kwargs):
        prepared_kwargs.update(kwargs)
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": None,
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    def _fake_plot_comparison(**kwargs):
        comparison_kwargs.update(kwargs)
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_plot_ppc_comparison_figure",
        _fake_plot_comparison,
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=OmegaConf.create(
            {"mixture_ppc_opts": {"comparison_show_component_observed": False}}
        ),
        plots="comparison",
    )

    assert len(out) == 1
    assert prepared_kwargs["need_assignments"] is False
    assert prepared_kwargs["need_mixture_samples"] is False
    assert comparison_kwargs["show_component_observed"] is False
    assert comparison_kwargs["show_mixture_overlay"] is False
    assert comparison_kwargs["assignments"] is None


def test_plot_mixture_ppc_comparison_can_enable_mixture_overlay(monkeypatch):
    """Comparison overlay flag should request and forward mixture samples."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for comparison wiring tests."""

        n_components = 2

    prepared_kwargs = {}
    comparison_kwargs = {}

    def _fake_prepare(*_args, **kwargs):
        prepared_kwargs.update(kwargs)
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": np.array([0, 1], dtype=int),
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    def _fake_plot_comparison(**kwargs):
        comparison_kwargs.update(kwargs)
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_plot_ppc_comparison_figure",
        _fake_plot_comparison,
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=OmegaConf.create(
            {"mixture_ppc_opts": {"comparison_include_mixture_overlay": True}}
        ),
        plots="comparison",
    )

    assert len(out) == 1
    assert prepared_kwargs["need_mixture_samples"] is True
    assert comparison_kwargs["show_mixture_overlay"] is True


def test_plot_mixture_ppc_comparison_overlay_kwarg_overrides_config(
    monkeypatch,
):
    """Direct kwarg should override config for comparison mixture overlay."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub for comparison wiring tests."""

        n_components = 2

    prepared_kwargs = {}
    comparison_kwargs = {}

    def _fake_prepare(*_args, **kwargs):
        prepared_kwargs.update(kwargs)
        return {
            "n_components": 2,
            "n_rows": 1,
            "n_cols": 1,
            "n_samples": 8,
            "top_gene_indices": np.array([0], dtype=int),
            "top_lfc": np.array([1.0], dtype=float),
            "mixture_samples_np": np.ones((2, 2, 1), dtype=float),
            "assignments": np.array([0, 1], dtype=int),
            "component_samples_list": [
                np.ones((2, 2, 1), dtype=float),
                np.ones((2, 2, 1), dtype=float),
            ],
            "render_opts": {},
        }

    def _fake_plot_comparison(**kwargs):
        comparison_kwargs.update(kwargs)
        fig, ax = plt.subplots(1, 1)
        return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=None)

    monkeypatch.setattr(
        mixture_ppc_module, "_prepare_mixture_ppc_data", _fake_prepare
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_plot_ppc_comparison_figure",
        _fake_plot_comparison,
    )
    monkeypatch.setattr(
        mixture_ppc_module, "_get_gene_names", lambda _results: ["g0"]
    )

    out = mixture_ppc_module.plot_mixture_ppc(
        _FakeResults(),
        counts=np.array([[1.0], [2.0]], dtype=float),
        viz_cfg=OmegaConf.create(
            {"mixture_ppc_opts": {"comparison_include_mixture_overlay": False}}
        ),
        plots="comparison",
        comparison_include_mixture_overlay=True,
    )

    assert len(out) == 1
    assert prepared_kwargs["need_mixture_samples"] is True
    assert comparison_kwargs["show_mixture_overlay"] is True


def test_prepare_mixture_ppc_data_skips_unneeded_heavy_steps(monkeypatch):
    """Overview-only prep should skip assignments and component sampling."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal result stub supporting gene-subset indexing."""

        n_components = 2
        n_cells = 3
        model_config = MagicMock()

        def __getitem__(self, _idx):
            return self

    def _fake_select_divergent(*_args, **_kwargs):
        return np.array([0], dtype=int), np.array([1.0], dtype=float)

    def _fake_predictive(*_args, **_kwargs):
        return np.ones((2, 3, 1), dtype=float)

    monkeypatch.setattr(
        mixture_ppc_module, "_select_divergent_genes", _fake_select_divergent
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_map_like_predictive_samples_for_plot",
        _fake_predictive,
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_cell_assignment_probabilities_for_plot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Assignments should not be computed.")
        ),
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_component_ppc_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Component samples should not be computed.")
        ),
    )

    out = mixture_ppc_module._prepare_mixture_ppc_data(
        _FakeResults(),
        counts=np.array([[1.0], [2.0], [3.0]], dtype=float),
        viz_cfg=None,
        n_rows=1,
        n_cols=1,
        n_samples=2,
        need_mixture_samples=True,
        need_component_samples=False,
        need_assignments=False,
    )

    assert out["assignments"] is None
    assert out["component_samples_list"] == []


def test_prepare_mixture_ppc_data_skips_mixture_samples_for_components_only(
    monkeypatch,
):
    """Component-only prep should skip mixture-level predictive sampling."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal result stub supporting gene-subset indexing."""

        n_components = 2
        n_cells = 3
        model_config = MagicMock()

        def __getitem__(self, _idx):
            return self

    def _fake_select_divergent(*_args, **_kwargs):
        return np.array([0], dtype=int), np.array([1.0], dtype=float)

    monkeypatch.setattr(
        mixture_ppc_module, "_select_divergent_genes", _fake_select_divergent
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_map_like_predictive_samples_for_plot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Mixture samples should not be computed.")
        ),
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_cell_assignment_probabilities_for_plot",
        lambda *_args, **_kwargs: np.array(
            [[0.9, 0.1], [0.1, 0.9], [0.6, 0.4]], dtype=float
        ),
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_component_ppc_samples",
        lambda *_args, **_kwargs: np.ones((2, 3, 1), dtype=float),
    )

    out = mixture_ppc_module._prepare_mixture_ppc_data(
        _FakeResults(),
        counts=np.array([[1.0], [2.0], [3.0]], dtype=float),
        viz_cfg=None,
        n_rows=1,
        n_cols=1,
        n_samples=2,
        need_mixture_samples=False,
        need_component_samples=True,
        need_assignments=True,
    )

    assert out["mixture_samples_np"] is None
    assert out["assignments"] is not None
    assert len(out["component_samples_list"]) == 2


def test_mixture_composition_requests_mixing_weights_only(
    monkeypatch, tmp_path
):
    """Mixture composition should request only mixing-weight MAP values."""
    import scribe.viz.mixture_ppc as mixture_ppc_module

    class _FakeResults:
        """Minimal mixture result stub."""

        model_type = "nbdm"
        n_components = 2
        n_cells = 2
        _dataset_indices = None

    def _fake_get_map_estimates(
        _results, *, counts, use_mean=True, targets=None
    ):
        _ = _results
        _ = counts
        _ = use_mean
        assert targets == ["mixing_weights"]
        return {"mixing_weights": np.array([0.7, 0.3], dtype=float)}

    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_map_estimates_for_plot",
        _fake_get_map_estimates,
    )
    # Provide AxisLayout metadata matching the fake mixing_weights shape.
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_layouts_for_plot",
        lambda _results: {
            "mixing_weights": AxisLayout(axes=("components",)),
        },
    )
    monkeypatch.setattr(
        mixture_ppc_module,
        "_get_config_values",
        lambda *_args, **_kwargs: {
            "method": "svi",
            "parameterization": "canonical",
            "model_type": "nbdm",
            "n_components": 2,
            "run_size_token": "100steps",
        },
    )

    out = mixture_ppc_module.plot_mixture_composition(
        _FakeResults(),
        counts=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        figs_dir=str(tmp_path),
        cfg=OmegaConf.create({}),
        viz_cfg=OmegaConf.create({"format": "png"}),
        cell_labels=None,
    )
    _ = out
    assert any(
        path.name.endswith("_mixture_composition.png")
        for path in tmp_path.iterdir()
    )


def test_mixture_composition_uses_trained_label_map_before_alphabetical_fallback():
    """Composition mapping should reuse the fit-time label map when present."""

    class _FakeResults:
        """Expose a non-alphabetical fit-time label map for regression coverage."""

        _label_map = {"Endothelial": 0, "Epithelial": 1}

    annotations = np.array(["Epithelial", "Endothelial", "Epithelial"])
    resolved = _resolve_label_map_for_composition(
        results=_FakeResults(),
        cell_labels=annotations,
        cfg={},
    )
    assert resolved["Endothelial"] == 0
    assert resolved["Epithelial"] == 1


def test_mixture_composition_aggregates_dataset_specific_weights():
    """Dataset-specific (D, K) weights should aggregate by dataset cell fractions."""

    # Dataset 0 has two cells and dataset 1 has one cell.
    dataset_indices = np.array([0, 0, 1], dtype=int)
    # Use clearly distinct per-dataset mixtures to ensure weighted averaging.
    mixing_weights = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    layouts = {
        "mixing_weights": AxisLayout(axes=("datasets", "components")),
    }
    resolved = _resolve_weight_fractions_for_composition(
        mixing_weights=mixing_weights,
        n_components=2,
        dataset_indices=dataset_indices,
        layouts=layouts,
    )
    np.testing.assert_allclose(resolved, np.array([2.0 / 3.0, 1.0 / 3.0]))


def test_mixture_composition_fallback_map_ignores_missing_labels():
    """Fallback composition mapping should exclude NaN labels."""
    annotations = np.array(
        ["Epithelial", np.nan, "Endothelial", None], dtype=object
    )
    resolved = _resolve_label_map_for_composition(
        results=object(),
        cell_labels=annotations,
        cfg={},
    )
    assert "nan" not in resolved
    assert "None" not in resolved
    assert set(resolved.keys()) == {"Endothelial", "Epithelial"}


def test_annotation_ppc_uses_trained_label_map_before_fallback():
    """Annotation PPC mapping should prioritize fit-time metadata."""

    class _FakeResults:
        """Expose fit-time label mapping with non-alphabetical indices."""

        _label_map = {"Endothelial": 0, "Epithelial": 1}

    annotations = np.array(["Epithelial", "Endothelial", "Epithelial"])
    resolved = _resolve_annotation_label_map(
        results=_FakeResults(),
        cell_labels=annotations,
        component_order=None,
    )
    assert resolved["Endothelial"] == 0
    assert resolved["Epithelial"] == 1


def test_plot_loss_interactive_with_fig_returns_plot_result(monkeypatch):
    """Loss plotting should return a PlotResult with 2-panel layout when ``fig`` is provided."""
    import scribe.viz.loss as loss_module

    # Patch dispatch to avoid constructing heavyweight result objects.
    monkeypatch.setattr(
        loss_module,
        "_get_training_diagnostic_payload",
        lambda _results: {
            "plot_kind": "loss",
            "loss_history": np.array([3.0, 2.0, 1.0]),
        },
    )

    fig = plt.figure(figsize=(7.0, 3.0))
    result = loss_module.plot_loss(
        results=object(),
        figs_dir=None,
        cfg=None,
        viz_cfg=OmegaConf.create({"format": "png"}),
        fig=fig,
        save=False,
    )

    assert isinstance(result, PlotResult)
    assert result.fig is fig
    assert result.n_panels == 2
    assert len(result.axes) == 2
    assert result.output_path is None
    plt.close(fig)


def test_plot_ecdf_accepts_single_axis_interactive_and_optional_show(
    monkeypatch,
):
    """ECDF should render on caller axis and return a PlotResult."""
    import scribe.viz.ecdf as ecdf_module

    # Keep gene selection deterministic so axis assertions stay stable.
    monkeypatch.setattr(
        ecdf_module,
        "_select_genes_simple",
        lambda counts, n_genes: (
            np.array([0], dtype=int),
            np.mean(counts, axis=0),
        ),
    )
    import scribe.viz._interactive as _int_mod

    show_calls = {"count": 0}
    monkeypatch.setattr(
        _int_mod.plt,
        "show",
        lambda: show_calls.__setitem__("count", show_calls["count"] + 1),
    )

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    result = ecdf_module.plot_ecdf(
        counts=np.array([[1.0, 0.0], [2.0, 1.0], [4.0, 0.0]], dtype=float),
        figs_dir=None,
        cfg=None,
        viz_cfg=OmegaConf.create(
            {"ecdf_opts": {"n_genes": 1}, "format": "png"}
        ),
        ax=ax,
        save=False,
        show=True,
    )

    assert isinstance(result, PlotResult)
    assert result.fig is fig
    assert result.axes == (ax,)
    assert result.output_path is None
    assert show_calls["count"] == 1
    plt.close(fig)


def test_plot_ppc_rejects_single_axis_for_multi_panel_layout():
    """PPC should fail fast when callers pass a single axis."""
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    with np.testing.assert_raises(ValueError):
        plot_ppc(
            results=object(),
            counts=np.array([[1.0]], dtype=float),
            figs_dir=None,
            cfg=None,
            viz_cfg=OmegaConf.create(
                {"ppc_opts": {"n_rows": 1, "n_cols": 1, "n_samples": 1}}
            ),
            ax=ax,
            save=False,
        )
    plt.close(fig)


def test_plot_capture_anchor_validates_axis_count(monkeypatch):
    """Capture-anchor should explain required panel count for explicit axes."""
    import scribe.viz.capture_anchor as capture_anchor_module

    monkeypatch.setattr(
        capture_anchor_module,
        "_resolve_expected_log_m0",
        lambda _cfg: 12.0,
    )
    monkeypatch.setattr(
        capture_anchor_module,
        "_get_map_estimates_for_plot",
        lambda *_args, **_kwargs: {
            "eta_capture": np.array([0.2, 0.4], dtype=float)
        },
    )

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))
    with np.testing.assert_raises(ValueError):
        plot_capture_anchor(
            results=object(),
            counts=np.array([[1.0, 0.0], [2.0, 1.0]], dtype=float),
            figs_dir=None,
            cfg=OmegaConf.create({"priors": {"eta_capture": [12.0, 0.1]}}),
            viz_cfg=OmegaConf.create({"format": "png"}),
            axes=[ax],
            save=False,
        )
    plt.close(fig)


def test_finalize_figure_calls_plt_show_when_requested(monkeypatch):
    """_finalize_figure should invoke plt.show() when show=True."""
    import scribe.viz._interactive as interactive_module

    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.0))
    shown = {"count": 0}

    monkeypatch.setattr(
        interactive_module.plt,
        "show",
        lambda: shown.__setitem__("count", shown["count"] + 1),
    )

    result = interactive_module._finalize_figure(
        fig=fig,
        axes=[ax],
        n_panels=1,
        save=False,
        show=True,
        close=True,
        _fig_owned=False,
    )

    assert isinstance(result, PlotResult)
    assert shown["count"] == 1


def test_resolve_render_flags_defaults_show_false():
    """Show should default to False (display is handled by PlotResult repr)."""
    import scribe.viz._interactive as interactive_module

    save, show, close = interactive_module._resolve_render_flags(
        figs_dir=None,
        save=None,
        show=None,
        close=None,
    )

    assert save is False
    assert show is False
    assert close is False


def test_resolve_render_flags_enables_save_when_figs_dir_set():
    """Save should auto-enable when figs_dir is provided."""
    import scribe.viz._interactive as interactive_module

    save, show, close = interactive_module._resolve_render_flags(
        figs_dir="/tmp/figs",
        save=None,
        show=None,
        close=None,
    )

    assert save is True
    assert show is False
    # close defaults to match save
    assert close is True


def test_prepare_capture_anchor_data_returns_computed_quantities(monkeypatch):
    """_prepare_capture_anchor_data should compute eta + log(L) quantities."""
    from scribe.viz.capture_anchor import _prepare_capture_anchor_data
    import scribe.viz.capture_anchor as ca_module

    # Stub MAP extraction
    eta = np.array([0.1, 0.3, 0.5], dtype=float)
    monkeypatch.setattr(
        ca_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {"eta_capture": eta},
    )
    counts = np.array([[10, 5], [20, 10], [5, 2]], dtype=float)
    cfg = OmegaConf.create({"priors": {"eta_capture": [12.0, 0.1]}})
    viz_cfg = OmegaConf.create({"capture_anchor_opts": {"n_bins": 20}})

    data = _prepare_capture_anchor_data(object(), counts, cfg, viz_cfg)
    assert data is not None
    assert data["expected_log_m0"] == 12.0
    np.testing.assert_array_equal(data["eta_capture"], eta)
    assert data["n_bins"] == 20
    assert data["log_library_size"].shape == (3,)
    np.testing.assert_allclose(
        data["eta_plus_log_lib"],
        eta + np.log(np.maximum(counts.sum(axis=1), 1.0)),
    )


def test_prepare_capture_anchor_data_returns_none_without_prior(monkeypatch):
    """_prepare_capture_anchor_data returns None when anchor is unresolvable."""
    from scribe.viz.capture_anchor import _prepare_capture_anchor_data

    cfg = OmegaConf.create({})
    viz_cfg = OmegaConf.create({})
    result = _prepare_capture_anchor_data(
        object(), np.zeros((2, 2)), cfg, viz_cfg
    )
    assert result is None


def test_prepare_p_capture_data_returns_none_without_p_capture(monkeypatch):
    """_prepare_p_capture_data returns None when p_capture is missing from MAP."""
    from scribe.viz.capture_anchor import _prepare_p_capture_data
    import scribe.viz.capture_anchor as ca_module

    monkeypatch.setattr(
        ca_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {},
    )
    viz_cfg = OmegaConf.create({})
    data = _prepare_p_capture_data(object(), np.zeros((3, 2)), viz_cfg)
    assert data is None


def test_prepare_p_capture_data_builds_panel_specs(monkeypatch):
    """_prepare_p_capture_data should build correct panel specs for mixture + multi-dataset."""
    from scribe.viz.capture_anchor import _prepare_p_capture_data
    import scribe.viz.capture_anchor as ca_module

    p_cap = np.array([0.3, 0.4, 0.5, 0.6], dtype=float)
    monkeypatch.setattr(
        ca_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {"p_capture": p_cap},
    )
    # Stub assignment probabilities for mixture panel
    monkeypatch.setattr(
        ca_module,
        "_get_cell_assignment_probabilities_for_plot",
        lambda *_a, **_kw: np.array(
            [[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]]
        ),
    )

    viz_cfg = OmegaConf.create({"p_capture_scaling_opts": {"n_bins": 5}})
    data = _prepare_p_capture_data(
        object(),
        np.ones((4, 2)),
        viz_cfg,
        is_mixture=True,
        is_multi_dataset=True,
        dataset_codes=np.array([0, 0, 1, 1]),
        dataset_names=["A", "B"],
    )
    assert data is not None
    # global + component + dataset = 3 panels
    assert len(data["panel_specs"]) == 3
    np.testing.assert_array_equal(data["p_capture"], p_cap)
    assert "component_ids" in data


def test_prepare_calibration_data_single_dataset(monkeypatch):
    """_prepare_calibration_data returns obs/pred means for single-dataset mode."""
    from scribe.viz.mean_calibration import _prepare_calibration_data
    import scribe.viz.mean_calibration as mc_module

    class _Fake:
        model_config = MagicMock(uses_variable_capture=False, is_bnb=False)

    monkeypatch.setattr(
        mc_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {
            "r": np.array([3.0, 4.0]),
            "p": np.array([0.4, 0.5]),
        },
    )
    monkeypatch.setattr(
        mc_module,
        "_get_layouts_for_plot",
        lambda _r: {
            "r": AxisLayout(axes=("genes",)),
            "p": AxisLayout(axes=("genes",)),
        },
    )

    counts = np.array([[2.0, 3.0], [4.0, 1.0]], dtype=float)
    data = _prepare_calibration_data(_Fake(), counts)
    assert data is not None
    assert data["mode"] == "single"
    assert data["obs_mean"].shape == (2,)
    assert data["pred_mean"].shape == (2,)


def test_prepare_calibration_data_returns_none_when_r_missing(monkeypatch):
    """_prepare_calibration_data returns None when MAP r is unavailable."""
    from scribe.viz.mean_calibration import _prepare_calibration_data
    import scribe.viz.mean_calibration as mc_module

    class _Fake:
        model_config = MagicMock(uses_variable_capture=False)

    monkeypatch.setattr(
        mc_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {"p": np.array([0.5])},
    )
    monkeypatch.setattr(
        mc_module,
        "_get_layouts_for_plot",
        lambda _r: {"p": AxisLayout(axes=("genes",))},
    )

    data = _prepare_calibration_data(_Fake(), np.ones((2, 1)))
    assert data is None


def test_prepare_ppc_data_returns_sorted_genes_and_positions(monkeypatch):
    """_prepare_ppc_data should return sorted gene indices and position mapping."""
    from scribe.viz.ppc import _prepare_ppc_data
    import scribe.viz.ppc as ppc_module

    selected = np.array([3, 1, 4], dtype=int)
    monkeypatch.setattr(
        ppc_module,
        "_select_genes",
        lambda _c, _r, _co: (selected, np.mean(np.zeros((5, 6)), axis=0)),
    )
    # Stub predictive sampling
    monkeypatch.setattr(
        ppc_module,
        "_get_predictive_samples_for_plot",
        lambda *_a, **_kw: np.zeros((2, 5, 3)),
    )

    class _FakeResults:
        predictive_samples = np.zeros((2, 5, 3))

        def __getitem__(self, idx):
            return self

    viz_cfg = OmegaConf.create(
        {
            "ppc_opts": {"n_rows": 1, "n_cols": 3, "n_samples": 2},
        }
    )
    data = _prepare_ppc_data(
        _FakeResults(), np.zeros((5, 6)), viz_cfg,
        n_rows=1, n_cols=3, n_samples=2,
    )
    assert data is not None
    assert data["n_rows"] == 1
    assert data["n_cols"] == 3
    assert data["n_genes_selected"] == 3
    # subset_positions maps original gene index -> position in selected_idx
    for gene_idx in selected:
        assert int(gene_idx) in data["subset_positions"]


def test_prepare_mixture_composition_data_global_mode(monkeypatch):
    """_prepare_mixture_composition_data returns global fractions without labels."""
    from scribe.viz.mixture_ppc import _prepare_mixture_composition_data
    import scribe.viz.mixture_ppc as mp_module

    class _Fake:
        n_components = 2
        n_cells = 4
        _dataset_indices = None

    monkeypatch.setattr(
        mp_module,
        "_get_map_estimates_for_plot",
        lambda *_a, **_kw: {"mixing_weights": np.array([0.6, 0.4])},
    )
    monkeypatch.setattr(
        mp_module,
        "_get_layouts_for_plot",
        lambda _r: {
            "mixing_weights": AxisLayout(axes=("components",)),
        },
    )

    data = _prepare_mixture_composition_data(
        _Fake(),
        np.ones((4, 2)),
        cfg=OmegaConf.create({}),
        viz_cfg=OmegaConf.create({}),
        cell_labels=None,
    )
    assert data is not None
    assert data["mode"] == "global"
    assert data["n_components"] == 2
    np.testing.assert_allclose(data["component_fractions"], [0.6, 0.4])


def test_prepare_mixture_composition_data_returns_none_for_single_component():
    """_prepare_mixture_composition_data returns None for non-mixture models."""
    from scribe.viz.mixture_ppc import _prepare_mixture_composition_data

    class _Fake:
        n_components = 1

    data = _prepare_mixture_composition_data(
        _Fake(),
        np.ones((2, 2)),
        cfg=OmegaConf.create({}),
        viz_cfg=OmegaConf.create({}),
    )
    assert data is None


# ---------------------------------------------------------------------------
# PlotResultCollection tests
# ---------------------------------------------------------------------------


def _make_plot_result(label="test", output_path=None):
    """Helper: create a PlotResult with a tiny figure for testing."""
    import matplotlib.pyplot as plt
    from scribe.viz._interactive import PlotResult

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.set_title(label)
    plt.close(fig)
    return PlotResult(fig=fig, axes=(ax,), n_panels=1, output_path=output_path)


def test_plot_result_collection_length_and_indexing():
    """PlotResultCollection supports len(), indexing, and iteration."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("a", "/tmp/a.png")
    r2 = _make_plot_result("b", "/tmp/b.png")
    coll = PlotResultCollection([r1, r2])

    assert len(coll) == 2
    assert coll[0] is r1
    assert coll[1] is r2
    assert list(coll) == [r1, r2]


def test_plot_result_collection_first_result_convenience_accessors():
    """Collection forwards .fig/.axes/.output_path to first result."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("first", "/tmp/first.png")
    r2 = _make_plot_result("second", "/tmp/second.png")
    coll = PlotResultCollection([r1, r2])

    assert coll.fig is r1.fig
    assert coll.axes is r1.axes
    assert coll.n_panels == r1.n_panels
    assert coll.output_path == "/tmp/first.png"
    assert coll.output_paths == ["/tmp/first.png", "/tmp/second.png"]


def test_plot_result_collection_repr_html_includes_all_figures():
    """_repr_html_ should contain one <img> tag per result."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("a")
    r2 = _make_plot_result("b")
    r3 = _make_plot_result("c")
    coll = PlotResultCollection([r1, r2, r3])

    html = coll._repr_html_()
    assert html.count("<img") == 3
    assert "data:image/png;base64," in html


def test_plot_result_collection_repr_png_returns_first_figure():
    """_repr_png_ should return PNG bytes for the first result."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("a")
    r2 = _make_plot_result("b")
    coll = PlotResultCollection([r1, r2])

    png = coll._repr_png_()
    # PNG magic bytes
    assert png[:4] == b"\x89PNG"
    assert png == r1._repr_png_()


def test_plot_result_collection_repr_string():
    """__repr__ should mention figure count and saved paths."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("a", "/tmp/a.png")
    r2 = _make_plot_result("b")
    coll = PlotResultCollection([r1, r2])

    s = repr(coll)
    assert "n_figures=2" in s
    assert "/tmp/a.png" in s


def test_plot_result_collection_rejects_empty_list():
    """PlotResultCollection should raise on empty input."""
    from scribe.viz._interactive import PlotResultCollection
    import pytest

    with pytest.raises(ValueError, match="at least one"):
        PlotResultCollection([])


def test_plot_result_collection_pipeline_compat_with_none_paths():
    """Collection works when no result has an output_path (interactive mode)."""
    from scribe.viz._interactive import PlotResultCollection

    r1 = _make_plot_result("a")
    r2 = _make_plot_result("b")
    coll = PlotResultCollection([r1, r2])

    assert coll.output_path is None
    assert coll.output_paths == [None, None]


# ---------------------------------------------------------------------------
# @plot_function decorator tests
# ---------------------------------------------------------------------------


def test_plot_function_decorator_builds_correct_public_signature():
    """Decorated function should expose figs_dir/cfg/viz_cfg/save/show/close."""
    import inspect
    from scribe.viz._interactive import (
        _create_or_validate_single_axis,
        plot_function,
    )

    @plot_function(suffix="test_sig")
    def _demo(x, *, ctx, viz_cfg=None, fig=None, ax=None, axes=None):
        pass

    sig = inspect.signature(_demo)
    param_names = list(sig.parameters.keys())
    # Domain arg preserved, render args injected
    assert param_names == [
        "x",
        "figs_dir",
        "cfg",
        "viz_cfg",
        "fig",
        "ax",
        "axes",
        "save",
        "show",
        "close",
    ]
    # figs_dir/cfg/viz_cfg are positional-or-keyword
    for name in ("figs_dir", "cfg", "viz_cfg"):
        assert (
            sig.parameters[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
    # save/show/close are keyword-only
    for name in ("save", "show", "close"):
        assert sig.parameters[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_plot_function_decorator_auto_finalizes_tuple_return():
    """Decorated function returning (fig, axes, n) should yield PlotResult."""
    from scribe.viz._interactive import plot_function

    @plot_function(suffix="demo")
    def _demo(*, ctx, fig=None, ax=None, axes=None, viz_cfg=None):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        ax.plot([0, 1], [0, 1])
        return fig, [ax], 1

    result = _demo(save=False)
    assert isinstance(result, PlotResult)
    assert result.n_panels == 1
    assert result.output_path is None
    plt.close(result.fig)


def test_plot_function_decorator_passes_through_none():
    """Decorated function returning None should propagate None."""
    from scribe.viz._interactive import plot_function

    @plot_function(suffix="skip")
    def _demo(*, ctx, viz_cfg=None, fig=None, ax=None, axes=None):
        return None

    assert _demo(save=False) is None


def test_plot_function_decorator_passes_through_plot_result_collection():
    """Decorated function returning PlotResultCollection is not finalized."""
    from scribe.viz._interactive import plot_function, PlotResultCollection

    @plot_function()
    def _demo(*, ctx, viz_cfg=None, fig=None, ax=None, axes=None):
        r1 = _make_plot_result("a")
        r2 = _make_plot_result("b")
        return PlotResultCollection([r1, r2])

    result = _demo(save=False)
    assert isinstance(result, PlotResultCollection)
    assert len(result) == 2


def test_plot_function_decorator_handles_positional_calls():
    """Positional figs_dir/cfg/viz_cfg should be intercepted correctly."""
    from scribe.viz._interactive import plot_function

    @plot_function(suffix="pos")
    def _demo(data, *, ctx, viz_cfg=None, fig=None, ax=None, axes=None):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        return fig, [ax], 1

    # Positional call like the pipeline: _demo(data, figs_dir, cfg, viz_cfg)
    result = _demo([1, 2, 3], None, None, None, save=False)
    assert isinstance(result, PlotResult)
    plt.close(result.fig)


def test_plot_function_decorator_extra_dict_overrides_suffix():
    """Return tuple with extra dict should override default suffix."""
    from scribe.viz._interactive import plot_function

    @plot_function(suffix="default", save_label="default label")
    def _demo(*, ctx, viz_cfg=None, fig=None, ax=None, axes=None):
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        return fig, [ax], 1, {"suffix": "override", "save_label": "custom"}

    result = _demo(save=False)
    assert isinstance(result, PlotResult)
    plt.close(result.fig)


# -----------------------------------------------------------------------
# _resolve_ppc_grid tests
# -----------------------------------------------------------------------


def test_resolve_ppc_grid_defaults_without_viz_cfg():
    """_resolve_ppc_grid should return built-in defaults when viz_cfg=None."""
    from scribe.viz._interactive import _resolve_ppc_grid

    grid = _resolve_ppc_grid()
    assert grid == {"n_rows": 5, "n_cols": 5, "n_samples": 512}


def test_resolve_ppc_grid_explicit_kwargs_override_defaults():
    """Explicit keyword arguments should take priority over defaults."""
    from scribe.viz._interactive import _resolve_ppc_grid

    grid = _resolve_ppc_grid(n_rows=3, n_cols=4, n_samples=100)
    assert grid == {"n_rows": 3, "n_cols": 4, "n_samples": 100}


def test_resolve_ppc_grid_viz_cfg_fallback():
    """When viz_cfg is provided, values should be read from opts_key."""
    from scribe.viz._interactive import _resolve_ppc_grid

    viz_cfg = OmegaConf.create(
        {"ppc_opts": {"n_rows": 7, "n_cols": 8, "n_samples": 256}}
    )
    grid = _resolve_ppc_grid(viz_cfg=viz_cfg)
    assert grid == {"n_rows": 7, "n_cols": 8, "n_samples": 256}


def test_resolve_ppc_grid_explicit_overrides_viz_cfg():
    """Explicit kwargs should win over viz_cfg values."""
    from scribe.viz._interactive import _resolve_ppc_grid

    viz_cfg = OmegaConf.create(
        {"ppc_opts": {"n_rows": 7, "n_cols": 8, "n_samples": 256}}
    )
    grid = _resolve_ppc_grid(n_rows=3, viz_cfg=viz_cfg)
    assert grid["n_rows"] == 3
    assert grid["n_cols"] == 8
    assert grid["n_samples"] == 256


def test_resolve_ppc_grid_n_genes_derives_n_cols():
    """n_genes should derive n_cols = ceil(n_genes / n_rows)."""
    from scribe.viz._interactive import _resolve_ppc_grid

    grid = _resolve_ppc_grid(n_genes=16, n_rows=4)
    assert grid["n_cols"] == 4

    # Non-even division: ceil(15 / 4) = 4
    grid2 = _resolve_ppc_grid(n_genes=15, n_rows=4)
    assert grid2["n_cols"] == 4


def test_resolve_ppc_grid_n_genes_conflict_raises():
    """Inconsistent n_genes and n_cols should raise ValueError."""
    from scribe.viz._interactive import _resolve_ppc_grid
    import pytest

    with pytest.raises(ValueError, match="n_genes=16.*n_cols=4.*n_cols=3"):
        _resolve_ppc_grid(n_genes=16, n_rows=4, n_cols=3)


def test_resolve_ppc_grid_custom_opts_key():
    """Custom opts_key should read from the specified section."""
    from scribe.viz._interactive import _resolve_ppc_grid

    viz_cfg = OmegaConf.create(
        {"mixture_ppc_opts": {"n_rows": 10, "n_cols": 2, "n_samples": 50}}
    )
    grid = _resolve_ppc_grid(viz_cfg=viz_cfg, opts_key="mixture_ppc_opts")
    assert grid == {"n_rows": 10, "n_cols": 2, "n_samples": 50}


def test_resolve_ppc_grid_default_samples_override():
    """Per-call default_samples should override the built-in 512 default."""
    from scribe.viz._interactive import _resolve_ppc_grid

    grid = _resolve_ppc_grid(default_samples=1500)
    assert grid["n_samples"] == 1500


# -----------------------------------------------------------------------
# get_ppc_render_options viz_cfg=None safety
# -----------------------------------------------------------------------


def test_get_ppc_render_options_none_safety():
    """get_ppc_render_options(None) should return stable defaults."""
    from scribe.viz.ppc_rendering import get_ppc_render_options

    opts = get_ppc_render_options(None)
    assert isinstance(opts, dict)
    assert opts["hist_max_bin_quantile"] == 0.99
    assert opts["hist_max_bin_floor"] == 10


# -----------------------------------------------------------------------
# plot_ecdf n_genes kwarg and viz_cfg=None safety
# -----------------------------------------------------------------------


def test_plot_ecdf_n_genes_kwarg(monkeypatch):
    """plot_ecdf should honour n_genes kwarg without viz_cfg."""
    import scribe.viz.ecdf as ecdf_module
    from scribe.viz.ecdf import plot_ecdf

    # Capture what n_genes is passed to _select_genes_simple
    captured = {}

    def _fake_select(counts, n_genes):
        captured["n_genes"] = n_genes
        idx = np.arange(min(n_genes, counts.shape[1]))
        return idx, np.mean(counts, axis=0)

    monkeypatch.setattr(ecdf_module, "_select_genes_simple", _fake_select)

    counts = np.random.default_rng(0).poisson(5, size=(20, 30))
    result = plot_ecdf(counts, n_genes=10, save=False)
    assert captured["n_genes"] == 10
    assert isinstance(result, PlotResult)
    plt.close(result.fig)


def test_plot_ecdf_viz_cfg_none_uses_default(monkeypatch):
    """plot_ecdf with viz_cfg=None should use 25 as default n_genes."""
    import scribe.viz.ecdf as ecdf_module
    from scribe.viz.ecdf import plot_ecdf

    captured = {}

    def _fake_select(counts, n_genes):
        captured["n_genes"] = n_genes
        idx = np.arange(min(n_genes, counts.shape[1]))
        return idx, np.mean(counts, axis=0)

    monkeypatch.setattr(ecdf_module, "_select_genes_simple", _fake_select)

    counts = np.random.default_rng(0).poisson(5, size=(20, 30))
    result = plot_ecdf(counts, save=False)
    assert captured["n_genes"] == 25
    plt.close(result.fig)


# =========================================================================
# Phase D – AxisLayout-driven viz helper tests
# =========================================================================

from scribe.core.axis_layout import AxisLayout
from scribe.viz.mean_calibration import (
    _compute_predicted_mean,
    _compute_per_dataset_means,
)
from scribe.viz.mu_pairwise import (
    _collapse_mixture_axis,
    _get_dataset_count,
)


class TestGetLayoutsForPlot:
    """Verify _get_layouts_for_plot dispatch for SVI and MCMC results."""

    def test_mcmc_returns_layouts_without_sample_dim(self):
        """MCMC dispatch should return layouts with has_sample_dim=False."""
        from scribe.viz.dispatch import _get_layouts_for_plot

        results = _make_mcmc_results_for_viz()
        layouts = _get_layouts_for_plot(results)
        assert isinstance(layouts, dict)
        # r has shape (3, 2) = (samples, genes) so layout should know
        # about genes after stripping the sample dim.
        assert "r" in layouts
        assert not layouts["r"].has_sample_dim

    def test_svi_returns_canonical_layouts(self):
        """SVI dispatch should return layouts keyed by canonical names."""
        from scribe.viz.dispatch import _get_layouts_for_plot

        params = {
            "p": jnp.array(0.3, dtype=jnp.float32),
            "r": jnp.ones((5,), dtype=jnp.float32),
        }
        config = ModelConfig(base_model="nbdm", n_components=None)
        results = ScribeSVIResults(
            params=params,
            loss_history=jnp.zeros(10),
            n_cells=10,
            n_genes=5,
            model_type="nbdm",
            model_config=config,
            prior_params={},
        )
        layouts = _get_layouts_for_plot(results)
        assert isinstance(layouts, dict)
        assert "r" in layouts
        assert layouts["r"].gene_axis is not None


class TestComputePredictedMeanWithLayouts:
    """Verify _compute_predicted_mean layout-driven vs fallback paths agree."""

    def test_standard_model_with_layouts(self):
        """Standard (non-mixture) model should produce (G,) prediction."""
        r = np.array([1.0, 2.0, 3.0])
        p = np.array(0.4)
        layouts = {
            "r": AxisLayout(axes=("genes",)),
            "p": AxisLayout(axes=()),
        }
        pred = _compute_predicted_mean(r, p, layouts=layouts)
        # mu = r * p / (1-p) = r * 0.4/0.6
        expected = r * 0.4 / 0.6
        np.testing.assert_allclose(pred, expected, rtol=1e-6)

    def test_mixture_model_with_layouts(self):
        """Mixture model should collapse component axis to (G,)."""
        r = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = np.array([0.3, 0.5])
        w = np.array([0.6, 0.4])
        layouts = {
            "r": AxisLayout(axes=("components", "genes")),
            "p": AxisLayout(axes=("components",)),
        }
        pred = _compute_predicted_mean(r, p, w, layouts=layouts)
        assert pred.shape == (2,)
        assert pred[0] > 0

    def test_mixture_with_vcp(self):
        """Mixture + VCP should scale prediction by mean capture prob."""
        r = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = np.array([0.3, 0.5])
        w = np.array([0.6, 0.4])
        pc = np.array([0.8, 0.9, 0.7])
        layouts = {
            "r": AxisLayout(axes=("components", "genes")),
            "p": AxisLayout(axes=("components",)),
        }
        pred_vcp = _compute_predicted_mean(r, p, w, pc, layouts=layouts)
        pred_no_vcp = _compute_predicted_mean(r, p, w, layouts=layouts)
        # VCP scales by mean(pc) = 0.8; all predictions should shrink.
        np.testing.assert_allclose(pred_vcp, pred_no_vcp * np.mean(pc))


class TestPerDatasetMeansWithLayouts:
    """Verify _compute_per_dataset_means layout-driven vs fallback."""

    def test_dataset_slicing_with_layouts(self):
        """Per-dataset slicing should use layout.dataset_axis."""
        r = np.array([[1.0, 2.0], [3.0, 4.0]])  # (D=2, G=2)
        p = np.array([[0.3, 0.4], [0.5, 0.6]])   # (D=2, G=2)
        counts = np.ones((6, 2))
        ds_codes = np.array([0, 0, 0, 1, 1, 1])
        ds_names = ["A", "B"]
        layouts = {
            "r": AxisLayout(axes=("datasets", "genes")),
            "p": AxisLayout(axes=("datasets", "genes")),
        }
        results = _compute_per_dataset_means(
            counts, r, p, ds_codes, ds_names,
            n_datasets=2, layouts=layouts,
        )
        assert len(results) == 2
        assert results[0]["name"] == "A"
        assert results[1]["name"] == "B"
        # Dataset 0 uses r[0]=[1,2], p[0]=[0.3,0.4]; dataset 1 uses r[1], p[1].
        assert results[0]["pred_mean"].shape == (2,)
        assert results[1]["pred_mean"].shape == (2,)
        assert not np.allclose(results[0]["pred_mean"], results[1]["pred_mean"])


class TestCollapseMixtureAxisWithLayouts:
    """Verify _collapse_mixture_axis layout path vs fallback."""

    def test_no_mixture_1d(self):
        """Gene-only mu should become (1, G)."""
        mu = np.array([1.0, 2.0, 3.0])
        layouts = {"mu": AxisLayout(axes=("genes",))}
        # No component axis → skip collapsing, but must reshape to (1, G).
        result = _collapse_mixture_axis(mu, None, layouts=layouts)
        assert result.shape[0] == 1
        np.testing.assert_allclose(result[0], mu)

    def test_mixture_collapse_with_weights(self):
        """(K, G) mu with weights should collapse to (G,)."""
        mu = np.array([[1.0, 2.0], [3.0, 4.0]])  # (K=2, G=2)
        w = np.array([0.6, 0.4])
        layouts = {
            "mu": AxisLayout(axes=("components", "genes")),
            "mixing_weights": AxisLayout(axes=("components",)),
        }
        result = _collapse_mixture_axis(mu, w, layouts=layouts)
        # Weighted sum: 0.6*[1,2] + 0.4*[3,4] = [1.8, 2.8]
        np.testing.assert_allclose(result, np.array([1.8, 2.8]))

    def test_dataset_only_passthrough(self):
        """(D, G) mu without component axis should pass through."""
        mu = np.array([[1.0, 2.0], [3.0, 4.0]])
        layouts = {"mu": AxisLayout(axes=("datasets", "genes"))}
        result = _collapse_mixture_axis(mu, None, layouts=layouts)
        np.testing.assert_array_equal(result, mu)


class TestGetDatasetCountWithLayouts:
    """Verify _get_dataset_count layout path vs fallback."""

    def test_from_model_config(self):
        """Explicit n_datasets in model_config takes priority."""

        class _Fake:
            model_config = type("MC", (), {"n_datasets": 3})()

        assert _get_dataset_count(_Fake(), np.ones((3, 10))) == 3

    def test_from_layout(self):
        """Layout dataset_axis should be used when model_config lacks n_datasets."""

        class _Fake:
            model_config = type("MC", (), {"n_datasets": None})()

        mu = np.ones((2, 5))
        layouts = {"mu": AxisLayout(axes=("datasets", "genes"))}
        assert _get_dataset_count(_Fake(), mu, layouts=layouts) == 2

    def test_from_layout_no_dataset_axis(self):
        """Gene-only layout should return 1."""

        class _Fake:
            model_config = type("MC", (), {"n_datasets": None})()

        mu = np.ones((5,))
        layouts = {"mu": AxisLayout(axes=("genes",))}
        assert _get_dataset_count(_Fake(), mu, layouts=layouts) == 1


class TestNormalizeMixingFractionsWithLayouts:
    """Verify _resolve_weight_fractions_for_composition layout path."""

    def test_global_weights_1d(self):
        """1-D global weights (K,) should pass through regardless."""
        w = np.array([0.5, 0.3, 0.2])
        layouts = {"mixing_weights": AxisLayout(axes=("components",))}
        result = _resolve_weight_fractions_for_composition(
            w, n_components=3, layouts=layouts,
        )
        np.testing.assert_allclose(result, w / w.sum())

    def test_per_dataset_weights_with_layout(self):
        """(D, K) weights should average over dataset axis using layout."""
        w = np.array([[0.6, 0.4], [0.3, 0.7]])  # (D=2, K=2)
        layouts = {
            "mixing_weights": AxisLayout(axes=("datasets", "components")),
        }
        result = _resolve_weight_fractions_for_composition(
            w, n_components=2, layouts=layouts,
        )
        expected = np.mean(w, axis=0)
        expected = expected / expected.sum()
        np.testing.assert_allclose(result, expected)

    def test_per_dataset_weighted_aggregate(self):
        """(D, K) weights with dataset_indices should do cell-count weighting."""
        w = np.array([[0.6, 0.4], [0.3, 0.7]])
        ds_idx = np.array([0, 0, 0, 1])  # 3 cells in ds 0, 1 in ds 1
        layouts = {
            "mixing_weights": AxisLayout(axes=("datasets", "components")),
        }
        result = _resolve_weight_fractions_for_composition(
            w, n_components=2, dataset_indices=ds_idx, layouts=layouts,
        )
        # Weighted: ds_weights = [0.75, 0.25]
        # fractions = 0.75*[0.6,0.4] + 0.25*[0.3,0.7] = [0.525, 0.475]
        expected = np.array([0.525, 0.475])
        expected = expected / expected.sum()
        np.testing.assert_allclose(result, expected)
