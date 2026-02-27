"""Consolidated tests for the viz_utils package refactor.

This module centralizes the coverage that previously lived in separate
viz-utils-focused test files. It validates cache helpers, dispatch helpers,
and package-root compatibility exports.
"""

from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from scribe.mcmc.results import ScribeMCMCResults
from scribe.models.config import ModelConfig
from viz_utils import (
    _build_umap_cache_path,
    _get_config_values,
    _get_predictive_samples_for_plot,
    _get_training_diagnostic_payload,
    plot_annotation_ppc,
    plot_correlation_heatmap,
    plot_ecdf,
    plot_loss,
    plot_mixture_composition,
    plot_mixture_ppc,
    plot_ppc,
    plot_umap,
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
    assert callable(plot_umap)
    assert callable(plot_correlation_heatmap)
    assert callable(plot_mixture_ppc)
    assert callable(plot_mixture_composition)
    assert callable(plot_annotation_ppc)
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
        rng_key=None,
        n_samples=2,
        counts=np.zeros((3, 2), dtype=np.int32),
        batch_size=None,
        store_samples=True,
    )
    assert selected.shape == (2, 3, 2)
    assert np.array_equal(selected, full_predictive[:2])


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
    path_c = _build_umap_cache_path(cfg=_cfg("drug", "dropseq"), cache_umap=True)
    assert len({path_a, path_b, path_c}) == 3
