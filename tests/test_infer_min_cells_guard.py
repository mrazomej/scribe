"""Tests for per-dataset minimum-cell guard in infer runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import scribe.cli.infer_runner as infer


def _build_cfg(
    *,
    min_cells_per_dataset: int | None,
    output_root: Path,
) -> object:
    """Build a minimal Hydra-like config for ``infer.main.__wrapped__`` tests.

    Parameters
    ----------
    min_cells_per_dataset : int or None
        Optional threshold for the per-dataset minimum cell guard.
    output_root : Path
        Temporary path used for output path placeholders.

    Returns
    -------
    object
        OmegaConf config object with required infer runner keys.
    """
    return OmegaConf.create(
        {
            "data": {
                "name": "toy_dataset",
                "path": str(output_root / "unused.h5ad"),
                "subset_column": None,
                "subset_value": None,
                "gpu_id": None,
                "layer": None,
                "filter_obs": {},
                "preprocessing": None,
                "dataset_key": "condition",
                "min_cells_per_dataset": min_cells_per_dataset,
            },
            "paths": {"outputs_dir": str(output_root / "unused_outputs_root")},
            "inference": {
                "method": "svi",
                "n_steps": 1,
                "batch_size": None,
                "stable_update": True,
                "log_progress_lines": False,
                "empirical_mixing": False,
                "early_stopping": None,
                "enable_x64": False,
            },
            "dirname_aliases": {"aliases": {}},
            "model": "nbdm",
            "parameterization": "canonical",
            "unconstrained": False,
            "expression_prior": "none",
            "prob_prior": "none",
            "zero_inflation_prior": "none",
            "n_datasets": None,
            "dataset_key": None,
            "dataset_params": None,
            "dataset_mixing": None,
            "expression_dataset_prior": "none",
            "prob_dataset_prior": "none",
            "prob_dataset_mode": "gene_specific",
            "zero_inflation_dataset_prior": "none",
            "overdispersion_dataset_prior": "none",
            "horseshoe_tau0": 1.0,
            "horseshoe_slab_df": 4,
            "horseshoe_slab_scale": 2.0,
            "neg_u": 1.0,
            "neg_a": 1.0,
            "neg_tau": 1.0,
            "capture_scaling_prior": "none",
            "expression_anchor": False,
            "expression_anchor_sigma": 0.3,
            "overdispersion": "none",
            "overdispersion_prior": "horseshoe",
            "n_components": None,
            "mixture_params": "all",
            "guide_rank": None,
            "joint_params": None,
            "dense_params": None,
            "guide_flow": None,
            "guide_flow_num_layers": 4,
            "guide_flow_hidden_dims": [64, 64],
            "guide_flow_activation": "relu",
            "guide_flow_n_bins": 8,
            "guide_flow_mixture_strategy": "independent",
            "guide_flow_zero_init": True,
            "guide_flow_layer_norm": True,
            "guide_flow_residual": True,
            "guide_flow_soft_clamp": True,
            "guide_flow_loft": True,
            "guide_flow_log_det_f64": False,
            "priors": {},
            "amortization": {"capture": {"enabled": False}},
            "annotation_key": None,
            "annotation_confidence": 3.0,
            "annotation_component_order": None,
            "annotation_min_cells": None,
            "svi_init": None,
            "resume_from": None,
            "cells_axis": 0,
            "layer": None,
            "seed": 42,
            "viz": False,
            "zero_inflation": False,
            "variable_capture": False,
            "output_dir": None,
        }
    )


def _make_condition_adata() -> anndata.AnnData:
    """Create a tiny AnnData with imbalanced per-condition cell counts."""
    rng = np.random.default_rng(123)
    counts = rng.poisson(3, (5, 4)).astype(np.float32)
    obs = pd.DataFrame(
        {"condition": ["A", "A", "A", "A", "B"]},
        index=[f"cell_{idx}" for idx in range(5)],
    )
    return anndata.AnnData(X=counts, obs=obs)


def test_min_cells_guard_fails_fast_and_writes_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail early and emit ``FAILED_MIN_CELLS.json`` when threshold is violated."""
    output_dir = tmp_path / "run_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = _make_condition_adata()

    # Keep Hydra runtime metadata deterministic in tests.
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(output_dir))
        ),
    )
    # Skip file IO by injecting the already-built AnnData object.
    monkeypatch.setattr(
        infer,
        "load_and_preprocess_anndata",
        lambda *_args, **_kwargs: adata.copy(),
    )

    fit_called = {"value": False}

    def _unexpected_fit(*_args, **_kwargs):
        fit_called["value"] = True
        raise AssertionError("scribe.fit should not run on min-cells failure")

    monkeypatch.setattr(infer.scribe, "fit", _unexpected_fit)
    cfg = _build_cfg(min_cells_per_dataset=2, output_root=tmp_path)

    with pytest.raises(
        infer.MinCellsPerDatasetError,
        match="min_cells_per_dataset=2",
    ):
        infer.main.__wrapped__(cfg)

    assert fit_called["value"] is False
    marker_path = output_dir / "FAILED_MIN_CELLS.json"
    assert marker_path.exists()
    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_payload["reason"] == "min_cells_per_dataset"
    assert marker_payload["dataset_key"] == "condition"
    assert marker_payload["min_cells_per_dataset"] == 2
    assert marker_payload["dataset_counts"] == {"A": 4, "B": 1}


def test_min_cells_guard_disabled_preserves_existing_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When threshold is unset, inference proceeds and no failure marker is written."""
    output_dir = tmp_path / "run_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = _make_condition_adata()

    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(output_dir))
        ),
    )
    monkeypatch.setattr(
        infer,
        "load_and_preprocess_anndata",
        lambda *_args, **_kwargs: adata.copy(),
    )

    class _DummyResults:
        """Pickle-friendly stand-in for infer runner result serialization."""

        n_components = None

    fit_called = {"value": False}

    def _fake_fit(*_args, **_kwargs):
        fit_called["value"] = True
        return _DummyResults()

    monkeypatch.setattr(infer.scribe, "fit", _fake_fit)
    cfg = _build_cfg(min_cells_per_dataset=None, output_root=tmp_path)
    infer.main.__wrapped__(cfg)

    assert fit_called["value"] is True
    assert (output_dir / "scribe_results.pkl").exists()
    assert not (output_dir / "FAILED_MIN_CELLS.json").exists()
