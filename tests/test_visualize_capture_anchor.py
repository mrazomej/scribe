"""Tests for capture-anchor wiring in ``visualize.py``.

These tests focus on config detection and branch gating so they remain
lightweight and deterministic without running full plotting pipelines.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import visualize


def _make_minimal_model_dir(tmp_path, cfg_data):
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

    # Persist a minimal, pickle-safe results object with model_config.n_datasets.
    results = SimpleNamespace(model_config=SimpleNamespace(n_datasets=None))
    with (model_dir / "scribe_results.pkl").open("wb") as handle:
        pickle.dump(results, handle)

    return str(model_dir)


def _make_minimal_viz_cfg():
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
            "capture_anchor": True,
            "format": "png",
            "capture_anchor_opts": {
                "n_bins": 10,
                "scatter_size": 4,
                "scatter_alpha": 0.3,
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

    monkeypatch.setattr(visualize, "plot_capture_anchor", _fake_plot_capture_anchor)

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(),
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
        viz_cfg=_make_minimal_viz_cfg(),
        overwrite=True,
    )
    assert ok is True
    assert calls["count"] == 0
