"""Regression tests for infer.py downgrade behavior with Hydra-like config."""

from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

import infer


def _make_single_survivor_h5ad(path: Path) -> None:
    """Create a tiny dataset where annotation_min_cells leaves one label.

    Parameters
    ----------
    path : Path
        Output file path for the generated ``.h5ad`` file.
    """
    rng = np.random.default_rng(42)
    labels = ["A"] * 10 + ["B"] * 1
    counts = rng.poisson(5, (11, 5)).astype(np.float32)
    adata = anndata.AnnData(X=counts, obs=pd.DataFrame({"ct": labels}))
    adata.write_h5ad(path)


def test_infer_main_single_survivor_downgrades_component_only_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """infer.py should auto-clear mu_prior after inferred mixture collapse.

    This test exercises the same code path used by Hydra/Submitit launches:
    ``infer.main`` receives a DictConfig, assembles ``scribe.fit`` kwargs,
    and runs the full pipeline. The annotation filter intentionally leaves
    one surviving class so auto-downgrade is triggered.
    """
    # Build a tiny AnnData file that reproduces the single-survivor condition.
    data_path = tmp_path / "single_survivor.h5ad"
    _make_single_survivor_h5ad(data_path)

    # Pre-create the output directory and patch Hydra runtime metadata used by
    # infer.py for output/checkpoint paths.
    output_dir = tmp_path / "run_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(output_dir))
        ),
    )

    # Construct a minimal Hydra-like config for infer.main.__wrapped__.
    # We set mu_prior=gaussian with unconstrained=True so the test verifies
    # the auto-downgrade now clears this component-only prior safely.
    cfg = OmegaConf.create(
        {
            "data": {
                "name": "toy_single_survivor",
                "output_prefix": "tests",
                "path": str(data_path),
                "subset_column": None,
                "subset_value": None,
                "gpu_id": None,
                "layer": None,
                "filter_obs": {},
                "preprocessing": None,
                "dataset_key": None,
            },
            "paths": {"outputs_dir": str(tmp_path / "unused_outputs_root")},
            "inference": {
                "method": "svi",
                "n_steps": 3,
                "batch_size": 11,
                "stable_update": True,
                "log_progress_lines": False,
                "empirical_mixing": False,
                "early_stopping": None,
                "enable_x64": False,
            },
            "dirname_aliases": {"aliases": {}},
            "model": "nbdm",
            "parameterization": "canonical",
            "unconstrained": True,
            "mu_prior": "gaussian",
            "p_prior": "none",
            "gate_prior": "none",
            "n_datasets": None,
            "dataset_key": None,
            "dataset_params": None,
            "mu_dataset_prior": "none",
            "p_dataset_prior": "none",
            "p_dataset_mode": "gene_specific",
            "gate_dataset_prior": "none",
            "horseshoe_tau0": 1.0,
            "horseshoe_slab_df": 4,
            "horseshoe_slab_scale": 2.0,
            "neg_u": 1.0,
            "neg_a": 1.0,
            "neg_tau": 1.0,
            "shared_capture_scaling": False,
            "n_components": None,
            "mixture_params": None,
            "guide_rank": None,
            "joint_params": None,
            "priors": {},
            "amortization": {"capture": {"enabled": False}},
            "annotation_key": "ct",
            "annotation_confidence": 3.0,
            "annotation_component_order": None,
            "annotation_min_cells": 5,
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

    # The key assertion: infer.py path now downgrades safely instead of
    # surfacing a ModelConfig validation error for mu_prior in non-mixture mode.
    with pytest.warns(UserWarning, match=r"mu_prior='gaussian' -> 'none'"):
        infer.main.__wrapped__(cfg)

    assert (output_dir / "scribe_results.pkl").exists()
