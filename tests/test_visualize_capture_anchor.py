"""Tests for capture-anchor wiring in ``visualize.py``.

These tests focus on config detection and branch gating so they remain
lightweight and deterministic without running full plotting pipelines.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from jax import random
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam

import visualize
from scribe.inference.preset_builder import build_config_from_preset
from scribe.models import get_model_and_guide
from scribe.svi.results import ScribeSVIResults
from viz_utils.capture_anchor import plot_p_capture_scaling


class _DummyResults:
    """Pickle-safe lightweight results object for visualize tests."""

    def __init__(self, n_datasets=None, n_components=None):
        # Keep model_config structure aligned with visualize expectations.
        self.model_config = SimpleNamespace(
            n_datasets=n_datasets,
            uses_variable_capture=False,
        )
        self.n_components = n_components

    def get_dataset(self, idx):
        """Return self for dataset subsetting in lightweight tests."""
        _ = idx
        return self


def _make_minimal_model_dir(
    tmp_path, cfg_data, *, n_datasets=None, n_components=None
):
    """Create a minimal run directory for ``_process_single_model_dir`` tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary pytest directory.
    cfg_data : dict
        Configuration dictionary to serialize into ``.hydra/config.yaml``.

    Returns
    -------
    str
        Absolute path to the synthetic model directory.
    """
    model_dir = tmp_path / "run"
    hydra_dir = model_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    # Create a placeholder data file because visualize validates path existence
    # before calling the loader.
    data_path = tmp_path / "dummy.h5ad"
    data_path.write_text("", encoding="utf-8")

    cfg = OmegaConf.create(cfg_data)
    cfg.data.path = str(data_path)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    # Persist a lightweight, pickle-safe results object used by visualize.
    results = _DummyResults(n_datasets=n_datasets, n_components=n_components)
    with (model_dir / "scribe_results.pkl").open("wb") as handle:
        pickle.dump(results, handle)

    return str(model_dir)


def _make_minimal_viz_cfg(
    *,
    capture_anchor=True,
    p_capture_scaling=False,
    mu_pairwise=False,
):
    """Build a compact viz config that enables only capture-anchor plotting.

    Returns
    -------
    OmegaConf
        Visualization config compatible with ``_process_single_model_dir``.
    """
    return OmegaConf.create(
        {
            "loss": False,
            "ecdf": False,
            "ppc": False,
            "bio_ppc": False,
            "umap": False,
            "heatmap": False,
            "mixture_ppc": False,
            "mixture_composition": False,
            "annotation_ppc": False,
            "capture_anchor": capture_anchor,
            "p_capture_scaling": p_capture_scaling,
            "mean_calibration": False,
            "mu_pairwise": mu_pairwise,
            "format": "png",
            "capture_anchor_opts": {
                "n_bins": 10,
                "scatter_size": 4,
                "scatter_alpha": 0.3,
            },
            "p_capture_scaling_opts": {
                "n_bins": 10,
                "min_cells_per_bin": 2,
                "assignment_batch_size": 8,
            },
            "mu_pairwise_opts": {
                "hist_bins": 20,
                "point_alpha": 0.3,
                "point_size": 4.0,
                "pseudocount": 1.0,
            },
        }
    )


def test_has_biology_informed_capture_prior_detects_prior_keys():
    """Detection helper should reflect presence/absence of prior keys."""
    cfg_enabled = OmegaConf.create({"priors": {"eta_capture": [12.2, 1e-5]}})
    cfg_disabled = OmegaConf.create({"priors": {"eta_capture": None}})

    assert visualize._has_biology_informed_capture_prior(cfg_enabled) is True
    assert visualize._has_biology_informed_capture_prior(cfg_disabled) is False


def test_process_single_model_dir_runs_capture_anchor_when_prior_present(
    monkeypatch, tmp_path
):
    """Capture-anchor branch should execute when toggle and prior are active."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": [12.2, 1e-5]},
            "data": {"name": "toy"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinbvcp",
        },
    )

    # Return simple AnnData-like object that exposes the fields visualize uses.
    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0]]),
        layers={},
        obs=pd.DataFrame(index=[0, 1]),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    calls = {"count": 0}

    # Track capture-anchor invocation without running heavy plotting.
    def _fake_plot_capture_anchor(*args, **kwargs):
        calls["count"] += 1
        return str(tmp_path / "capture_anchor.png")

    monkeypatch.setattr(
        visualize, "plot_capture_anchor", _fake_plot_capture_anchor
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(capture_anchor=True),
        overwrite=True,
    )
    assert ok is True
    assert calls["count"] == 1


def test_process_single_model_dir_skips_capture_anchor_without_prior(
    monkeypatch, tmp_path
):
    """Capture-anchor branch should skip when biology-informed prior is absent."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": None},
            "data": {"name": "toy"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinbvcp",
        },
    )

    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0]]),
        layers={},
        obs=pd.DataFrame(index=[0, 1]),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    calls = {"count": 0}
    monkeypatch.setattr(
        visualize,
        "plot_capture_anchor",
        lambda *args, **kwargs: calls.__setitem__("count", calls["count"] + 1),
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(capture_anchor=True),
        overwrite=True,
    )
    assert ok is True
    assert calls["count"] == 0


def test_process_single_model_dir_runs_p_capture_scaling_when_vcp(
    monkeypatch, tmp_path
):
    """p-capture scaling should run when enabled and model is VCP."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": None},
            "data": {"name": "toy"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinbvcp",
        },
        n_datasets=None,
        n_components=1,
    )

    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0]]),
        layers={},
        obs=pd.DataFrame(index=[0, 1]),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    calls = {"count": 0}

    def _fake_plot_p_capture_scaling(*args, **kwargs):
        calls["count"] += 1
        return str(tmp_path / "p_capture_scaling.png")

    monkeypatch.setattr(
        visualize,
        "plot_p_capture_scaling",
        _fake_plot_p_capture_scaling,
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            capture_anchor=False, p_capture_scaling=True
        ),
        overwrite=True,
    )
    assert ok is True
    assert calls["count"] == 1


def test_process_single_model_dir_skips_p_capture_scaling_non_vcp(
    monkeypatch, tmp_path
):
    """p-capture scaling should skip when enabled on non-VCP models."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": None},
            "data": {"name": "toy"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinb",
        },
        n_datasets=None,
        n_components=1,
    )

    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0]]),
        layers={},
        obs=pd.DataFrame(index=[0, 1]),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    calls = {"count": 0}
    monkeypatch.setattr(
        visualize,
        "plot_p_capture_scaling",
        lambda *args, **kwargs: calls.__setitem__("count", calls["count"] + 1),
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            capture_anchor=False, p_capture_scaling=True
        ),
        overwrite=True,
    )
    assert ok is True
    assert calls["count"] == 0


def test_p_capture_scaling_receives_both_split_groupings(monkeypatch, tmp_path):
    """When mixture and multi-dataset, scaling plot should receive both splits."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": None},
            "data": {"name": "toy", "dataset_key": "batch"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinbvcp",
            "n_components": 2,
        },
        n_datasets=2,
        n_components=2,
    )

    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0], [4.0, 1.0], [1.0, 0.0]]),
        layers={},
        obs=pd.DataFrame({"batch": ["a", "a", "b", "b"]}),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    captured = {}

    def _fake_plot_p_capture_scaling(*args, **kwargs):
        captured["is_mixture"] = kwargs.get("is_mixture")
        captured["is_multi_dataset"] = kwargs.get("is_multi_dataset")
        captured["dataset_names"] = kwargs.get("dataset_names")
        return str(tmp_path / "p_capture_scaling.png")

    monkeypatch.setattr(
        visualize,
        "plot_p_capture_scaling",
        _fake_plot_p_capture_scaling,
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            capture_anchor=False, p_capture_scaling=True
        ),
        overwrite=True,
    )
    assert ok is True
    assert captured["is_mixture"] is True
    assert captured["is_multi_dataset"] is True
    assert list(captured["dataset_names"]) == ["a", "b"]


def test_process_single_model_dir_runs_mu_pairwise_for_multi_dataset(
    monkeypatch, tmp_path
):
    """Mu pairwise branch should run only when multi-dataset is detected."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "priors": {"eta_capture": None},
            "data": {"name": "toy", "dataset_key": "batch"},
            "inference": {"method": "svi"},
            "parameterization": "mean_odds",
            "model": "zinb",
        },
        n_datasets=2,
        n_components=1,
    )

    fake_adata = SimpleNamespace(
        X=np.array([[3.0, 1.0], [0.0, 2.0], [4.0, 1.0], [1.0, 0.0]]),
        layers={},
        obs=pd.DataFrame({"batch": ["a", "a", "b", "b"]}),
    )
    monkeypatch.setattr(
        visualize.scribe.data_loader,
        "load_and_preprocess_anndata",
        lambda *args, **kwargs: fake_adata,
    )
    monkeypatch.setattr(visualize, "cleanup_plot_memory", lambda **kwargs: None)

    captured = {"count": 0}

    def _fake_plot_mu_pairwise(*args, **kwargs):
        captured["count"] += 1
        captured["dataset_names"] = kwargs.get("dataset_names")
        return str(tmp_path / "mu_pairwise.png")

    monkeypatch.setattr(visualize, "plot_mu_pairwise", _fake_plot_mu_pairwise)

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            capture_anchor=False,
            p_capture_scaling=False,
            mu_pairwise=True,
        ),
        overwrite=True,
    )
    assert ok is True
    assert captured["count"] == 1
    assert list(captured["dataset_names"]) == ["a", "b"]


def test_plot_p_capture_scaling_joint_gate_results_no_gate_loc_error(
    monkeypatch, tmp_path
):
    """Joint gate configs should render p-capture scaling without gate_loc keys.

    This integration regression exercises MAP extraction plus component
    assignment (`cell_type_probabilities_map`) with a fitted joint low-rank
    model where gate variational parameters are stored as
    ``joint_*_gate_{loc,W,raw_diag}``.
    """
    n_cells, n_genes = 24, 6
    key = random.PRNGKey(123)
    counts = random.poisson(key, lam=4.0, shape=(n_cells, n_genes))

    config = build_config_from_preset(
        model="zinbvcp",
        parameterization="mean_odds",
        unconstrained=True,
        n_components=2,
        p_prior="gaussian",
        gate_prior="gaussian",
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

    # Use a few optimization steps to instantiate realistic joint guide params.
    optimizer = Adam(1e-3)
    svi = SVI(model_fn, guide_fn, optimizer, loss=Trace_ELBO())
    svi_state = svi.init(random.PRNGKey(124), **model_kwargs)
    for _ in range(3):
        svi_state, _ = svi.update(svi_state, **model_kwargs)

    results = ScribeSVIResults(
        params=svi.get_params(svi_state),
        loss_history=np.array([1.0, 0.5]),
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
            "format": "png",
            "p_capture_scaling_opts": {
                "n_bins": 6,
                "min_cells_per_bin": 1,
                "assignment_batch_size": 8,
            },
        }
    )

    # Keep assignment lightweight while still exercising MAP extraction used by
    # component assignment workflows.
    def _fake_component_probs(_results, *, counts, batch_size, use_mean):
        _ = batch_size
        map_estimates = _results.get_map(
            use_mean=use_mean, canonical=True, verbose=False, counts=counts
        )
        assert "gate" in map_estimates
        return np.tile(np.array([[0.6, 0.4]], dtype=float), (counts.shape[0], 1))

    monkeypatch.setattr(
        "viz_utils.capture_anchor._get_cell_assignment_probabilities_for_plot",
        _fake_component_probs,
    )

    output_path = plot_p_capture_scaling(
        results=results,
        counts=counts,
        figs_dir=str(tmp_path),
        cfg=cfg,
        viz_cfg=viz_cfg,
        is_mixture=True,
        is_multi_dataset=False,
    )

    assert output_path is not None
    assert output_path.endswith("_p_capture_scaling.png")
