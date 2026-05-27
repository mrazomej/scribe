"""Tests for compositional and W-shrinkage wiring in ``scribe.viz.pipeline``.

These tests focus on config detection and branch gating so they remain
lightweight and deterministic without running full plotting pipelines.
"""

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

from scribe.viz import pipeline as visualize


class _DummyResults:
    """Pickle-safe lightweight results object for visualize tests."""

    def __init__(
        self,
        *,
        n_datasets=None,
        n_components=None,
        w_prior_diagnostics=None,
    ):
        self.model_config = SimpleNamespace(
            n_datasets=n_datasets,
            uses_variable_capture=False,
        )
        self.n_components = n_components
        self.w_prior_diagnostics = w_prior_diagnostics

    def get_dataset(self, idx):
        """Return self for dataset subsetting in lightweight tests."""
        _ = idx
        return self


class _CompositionalDummyResults(_DummyResults):
    """Results stub exposing ``get_compositional_samples``."""

    def get_compositional_samples(self, **kwargs):
        """Return a tiny placeholder compositional sample matrix."""
        _ = kwargs
        return np.zeros((2, 2))


def _make_minimal_model_dir(tmp_path, cfg_data, *, results=None):
    """Create a minimal run directory for ``_process_single_model_dir`` tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary pytest directory.
    cfg_data : dict
        Configuration dictionary to serialize into ``.hydra/config.yaml``.
    results : object or None
        Optional results object to pickle. Defaults to ``_DummyResults()``.

    Returns
    -------
    str
        Absolute path to the synthetic model directory.
    """
    model_dir = tmp_path / "run"
    hydra_dir = model_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    data_path = tmp_path / "dummy.h5ad"
    data_path.write_text("", encoding="utf-8")

    cfg = OmegaConf.create(cfg_data)
    cfg.data.path = str(data_path)
    OmegaConf.save(cfg, hydra_dir / "config.yaml")

    if results is None:
        results = _DummyResults()
    with (model_dir / "scribe_results.pkl").open("wb") as handle:
        pickle.dump(results, handle)

    return str(model_dir)


def _make_minimal_viz_cfg(
    *,
    corner_ppc=False,
    compositional_ppc=False,
    compositional_corner_ppc=False,
    w_shrinkage=False,
):
    """Build a compact viz config for compositional / W-shrinkage tests.

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
            "capture_anchor": False,
            "p_capture_scaling": False,
            "mean_calibration": False,
            "mu_pairwise": False,
            "corner_ppc": corner_ppc,
            "compositional_ppc": compositional_ppc,
            "compositional_corner_ppc": compositional_corner_ppc,
            "w_shrinkage": w_shrinkage,
            "format": "png",
            "ppc_opts": {
                "n_rows": 2,
                "n_cols": 2,
                "n_samples": 128,
            },
            "corner_ppc_opts": {
                "n_genes": 3,
                "gene_selection": "correlation_diverse",
                "min_mean_umi": 5.0,
            },
            "w_shrinkage_opts": {
                "show_sigma_k": True,
                "show_threshold": True,
                "threshold_fraction": 0.05,
            },
        }
    )


def _patch_visualize_data_loader(monkeypatch, tmp_path):
    """Monkeypatch AnnData loading with a tiny synthetic count matrix."""
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


def test_supports_compositional_ppc_detects_method():
    """Compositional support helper should reflect method availability."""
    assert (
        visualize._supports_compositional_ppc(_CompositionalDummyResults())
        is True
    )
    assert visualize._supports_compositional_ppc(_DummyResults()) is False


def test_has_w_shrinkage_diagnostics_detects_field():
    """W-shrinkage helper should reflect diagnostics availability."""
    assert (
        visualize._has_w_shrinkage_diagnostics(
            _DummyResults(w_prior_diagnostics={"column_frobenius_compositional": [1.0]})
        )
        is True
    )
    assert visualize._has_w_shrinkage_diagnostics(_DummyResults()) is False


def test_process_single_model_dir_skips_compositional_without_method(
    monkeypatch, tmp_path
):
    """Compositional branches should skip when sampling API is missing."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "data": {"name": "toy"},
            "inference": {"method": "laplace"},
            "model": "nbln",
        },
    )
    _patch_visualize_data_loader(monkeypatch, tmp_path)

    calls = {
        "compositional_ppc": 0,
        "compositional_corner_ppc": 0,
    }

    def _fake_compositional_ppc(*args, **kwargs):
        calls["compositional_ppc"] += 1

    def _fake_compositional_corner_ppc(*args, **kwargs):
        calls["compositional_corner_ppc"] += 1

    monkeypatch.setattr(
        visualize, "plot_compositional_ppc", _fake_compositional_ppc
    )
    monkeypatch.setattr(
        visualize,
        "plot_compositional_corner_ppc",
        _fake_compositional_corner_ppc,
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            compositional_ppc=True,
            compositional_corner_ppc=True,
        ),
        overwrite=True,
    )

    assert ok is True
    assert calls["compositional_ppc"] == 0
    assert calls["compositional_corner_ppc"] == 0


def test_process_single_model_dir_runs_compositional_when_supported(
    monkeypatch, tmp_path
):
    """Compositional branches should execute when sampling API is present."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "data": {"name": "toy"},
            "inference": {"method": "laplace"},
            "model": "nbln",
        },
        results=_CompositionalDummyResults(),
    )
    _patch_visualize_data_loader(monkeypatch, tmp_path)

    calls = {
        "compositional_ppc": 0,
        "compositional_corner_ppc": 0,
    }

    def _fake_compositional_ppc(*args, **kwargs):
        calls["compositional_ppc"] += 1

    def _fake_compositional_corner_ppc(*args, **kwargs):
        calls["compositional_corner_ppc"] += 1

    monkeypatch.setattr(
        visualize, "plot_compositional_ppc", _fake_compositional_ppc
    )
    monkeypatch.setattr(
        visualize,
        "plot_compositional_corner_ppc",
        _fake_compositional_corner_ppc,
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(
            compositional_ppc=True,
            compositional_corner_ppc=True,
        ),
        overwrite=True,
    )

    assert ok is True
    assert calls["compositional_ppc"] == 1
    assert calls["compositional_corner_ppc"] == 1


def test_process_single_model_dir_skips_w_shrinkage_without_diagnostics(
    monkeypatch, tmp_path
):
    """W-shrinkage branch should skip when diagnostics are unavailable."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "data": {"name": "toy"},
            "inference": {"method": "laplace"},
            "model": "nbln",
        },
    )
    _patch_visualize_data_loader(monkeypatch, tmp_path)

    calls = {"w_shrinkage": 0}

    def _fake_w_shrinkage(*args, **kwargs):
        calls["w_shrinkage"] += 1

    monkeypatch.setattr(
        visualize, "plot_w_shrinkage_spectrum", _fake_w_shrinkage
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(w_shrinkage=True),
        overwrite=True,
    )

    assert ok is True
    assert calls["w_shrinkage"] == 0


def test_process_single_model_dir_runs_w_shrinkage_with_diagnostics(
    monkeypatch, tmp_path
):
    """W-shrinkage branch should execute when diagnostics are present."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "data": {"name": "toy"},
            "inference": {"method": "laplace"},
            "model": "nbln",
        },
        results=_DummyResults(
            w_prior_diagnostics={"column_frobenius_compositional": [1.0, 0.1]},
        ),
    )
    _patch_visualize_data_loader(monkeypatch, tmp_path)

    calls = {"w_shrinkage": 0}

    def _fake_w_shrinkage(*args, **kwargs):
        calls["w_shrinkage"] += 1

    monkeypatch.setattr(
        visualize, "plot_w_shrinkage_spectrum", _fake_w_shrinkage
    )

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(w_shrinkage=True),
        overwrite=True,
    )

    assert ok is True
    assert calls["w_shrinkage"] == 1


def test_process_single_model_dir_runs_corner_ppc(monkeypatch, tmp_path):
    """Corner PPC branch should execute without compositional API gating."""
    model_dir = _make_minimal_model_dir(
        tmp_path,
        cfg_data={
            "data": {"name": "toy"},
            "inference": {"method": "laplace"},
            "model": "nbln",
        },
    )
    _patch_visualize_data_loader(monkeypatch, tmp_path)

    calls = {"corner_ppc": 0}

    def _fake_corner_ppc(*args, **kwargs):
        calls["corner_ppc"] += 1

    monkeypatch.setattr(visualize, "plot_corner_ppc", _fake_corner_ppc)

    ok = visualize._process_single_model_dir(
        model_dir=model_dir,
        viz_cfg=_make_minimal_viz_cfg(corner_ppc=True),
        overwrite=True,
    )

    assert ok is True
    assert calls["corner_ppc"] == 1
