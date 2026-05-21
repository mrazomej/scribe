"""Regression tests for infer runner downgrade behavior."""

from pathlib import Path
from types import SimpleNamespace

import anndata
import numpy as np
import pandas as pd
import pytest

# Hydra CLI tests depend on omegaconf (optional extra [hydra] in
# pyproject.toml). Skip cleanly when absent.
pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
from omegaconf import OmegaConf  # noqa: E402

import scribe.cli.infer_runner as infer  # noqa: E402


@pytest.mark.parametrize(
    ("zero_inflation", "variable_capture", "expected_model"),
    [
        (False, False, "nbdm"),
        (True, False, "zinb"),
        (False, True, "nbvcp"),
        (True, True, "zinbvcp"),
    ],
)
def test_resolve_model_type_feature_flag_matrix(
    zero_inflation: bool, variable_capture: bool, expected_model: str
) -> None:
    """Resolve DM-family defaults from zero_inflation/variable_capture flags."""
    cfg = OmegaConf.create(
        {
            "zero_inflation": zero_inflation,
            "variable_capture": variable_capture,
            "model": None,
        }
    )
    assert infer._resolve_model_type(cfg) == expected_model


def test_resolve_model_type_explicit_override_wins() -> None:
    """Keep explicit model override instead of deriving from feature flags."""
    cfg = OmegaConf.create(
        {
            "model": "twostate_ln_rate",
            "zero_inflation": False,
            "variable_capture": False,
        }
    )
    assert infer._resolve_model_type(cfg) == "twostate_ln_rate"


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scribe_caplog
):
    """infer.py should auto-clear expression_prior after inferred mixture collapse.

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
    # We set expression_prior=gaussian with unconstrained=True so the test verifies
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
            "expression_prior": "gaussian",
            "prob_prior": "none",
            "zero_inflation_prior": "none",
            "n_datasets": None,
            "dataset_key": None,
            "dataset_params": None,
            "expression_dataset_prior": "none",
            "prob_dataset_prior": "none",
            "prob_dataset_mode": "gene_specific",
            "zero_inflation_dataset_prior": "none",
            "horseshoe_tau0": 1.0,
            "horseshoe_slab_df": 4,
            "horseshoe_slab_scale": 2.0,
            "neg_u": 1.0,
            "neg_a": 1.0,
            "neg_tau": 1.0,
            "capture_scaling_prior": "none",
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
    # surfacing a ModelConfig validation error for expression_prior in non-mixture mode.
    import logging

    scribe_caplog.set_level(logging.WARNING, logger="scribe")
    infer.main.__wrapped__(cfg)
    assert any(
        "expression_prior='gaussian' -> 'none'" in record.message
        for record in scribe_caplog.records
    )

    assert (output_dir / "scribe_results.pkl").exists()


def test_infer_main_forwards_laplace_and_cascade_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forward Laplace/cascade fields and pack Newton keys into laplace_config."""
    data_path = tmp_path / "toy.h5ad"
    _make_single_survivor_h5ad(data_path)
    output_dir = tmp_path / "laplace_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "hydra.core.hydra_config.HydraConfig.get",
        lambda: SimpleNamespace(
            runtime=SimpleNamespace(output_dir=str(output_dir))
        ),
    )

    captured_fit_kwargs: dict = {}
    sentinel_informative = object()

    # Capture kwargs passed into scribe.fit() without running expensive inference.
    def _fake_fit(*, counts, **kwargs):
        assert counts.shape == (11, 5)
        captured_fit_kwargs.update(kwargs)
        return SimpleNamespace(n_components=None)

    # Simulate loading informative-prior source while keeping svi_init unset.
    def _fake_load_svi_init(path, _console=None):
        if path is None:
            return None
        if str(path).endswith("cascade_source.pkl"):
            return sentinel_informative
        raise AssertionError(f"unexpected SVI source path: {path}")

    # Return a deterministic in-memory counts matrix to avoid I/O concerns.
    def _fake_load_and_preprocess(*_args, **_kwargs):
        return np.ones((11, 5), dtype=np.float32)

    monkeypatch.setattr("scribe.fit", _fake_fit)
    monkeypatch.setattr(infer, "_load_svi_init", _fake_load_svi_init)
    monkeypatch.setattr(
        infer, "load_and_preprocess_anndata", _fake_load_and_preprocess
    )

    cfg = OmegaConf.create(
        {
            "data": {
                "name": "toy_laplace",
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
                "method": "laplace",
                "n_steps": 11,
                "batch_size": 7,
                "optimizer_config": None,
                "log_progress_lines": False,
                "restore_best": False,
                "empirical_mixing": False,
                "early_stopping": None,
                "n_newton_steps": 9,
                "newton_tolerance": 1e-5,
                "damping": 0.05,
                "newton_max_step": 3.0,
                "convergence_action": "warn",
                "enable_x64": False,
            },
            "dirname_aliases": {"aliases": {}},
            "model": "nbln",
            "parameterization": "count_lognormal",
            "unconstrained": False,
            "positive_transform": "softplus",
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
            "latent_dim": 6,
            "d_mode": "learned",
            "alr_reference_idx": 2,
            "n_components": None,
            "mixture_params": "all",
            "guide_rank": None,
            "joint_params": None,
            "dense_params": None,
            "guide_flow": None,
            "guide_flow_num_layers": 4,
            "guide_flow_hidden_dims": [32, 32],
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
            "informative_priors_from": "cascade_source.pkl",
            "informative_priors_tau": 2.5,
            "informative_priors_freeze": ["r", "eta"],
            "cascade_map_method": "transform",
            "resume_from": None,
            "cells_axis": 0,
            "layer": None,
            "gene_coverage": 0.1,
            "kl_annealing_warmup": 1234,
            "seed": 42,
            "viz": False,
            "zero_inflation": False,
            "variable_capture": False,
            "output_dir": None,
        }
    )

    infer.main.__wrapped__(cfg)

    assert captured_fit_kwargs["inference_method"] == "laplace"
    assert captured_fit_kwargs["laplace_config"] == {
        "n_newton_steps": 9,
        "newton_tolerance": 1e-5,
        "damping": 0.05,
        "newton_max_step": 3.0,
        "convergence_action": "warn",
    }
    assert captured_fit_kwargs["n_steps"] == 11
    assert captured_fit_kwargs["batch_size"] == 7
    assert captured_fit_kwargs["positive_transform"] == "softplus"
    assert captured_fit_kwargs["latent_dim"] == 6
    assert captured_fit_kwargs["d_mode"] == "learned"
    assert captured_fit_kwargs["alr_reference_idx"] == 2
    assert captured_fit_kwargs["gene_coverage"] == 0.1
    assert captured_fit_kwargs["kl_annealing_warmup"] == 1234
    assert (
        captured_fit_kwargs["informative_priors_from"] is sentinel_informative
    )
    assert captured_fit_kwargs["informative_priors_tau"] == 2.5
    assert captured_fit_kwargs["informative_priors_freeze"] == ("r", "eta")
    assert captured_fit_kwargs["cascade_map_method"] == "transform"
