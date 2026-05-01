"""Tests for pre-fit gene coverage filtering utilities and integration.

This module validates:

1. Core coverage-mask and aggregation helpers in ``scribe.core.gene_coverage``.
2. Multi-dataset union behavior for coverage masks.
3. ``scribe.fit(..., gene_coverage=...)`` metadata wiring using a mocked
   inference dispatcher.
4. Automatic exclusion of pooled "other" in DE compare for pre-filtered
   empirical results objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from scribe.api import fit
from scribe.core.gene_coverage import (
    aggregate_counts_by_mask,
    build_filtered_gene_names,
    compute_empirical_gene_coverage_mask,
    compute_gene_coverage_rank,
)
from scribe.de import compare
from scribe.models.config import ModelConfigBuilder
from scribe.svi.results import ScribeSVIResults


def test_compute_empirical_gene_coverage_mask_single_dataset():
    """Single-dataset coverage should keep dominant genes first."""
    counts = jnp.array(
        [
            [100.0, 10.0, 1.0, 1.0],
            [80.0, 8.0, 1.0, 1.0],
        ]
    )
    mask = compute_empirical_gene_coverage_mask(counts, coverage=0.9)
    assert mask.dtype == bool
    assert mask.shape == (4,)
    assert bool(mask[0])
    assert not bool(mask[2])


def test_compute_empirical_gene_coverage_mask_multi_dataset_union():
    """Multi-dataset mask should be union of per-dataset masks."""
    # Dataset 0 dominates gene 0; dataset 1 dominates gene 3.
    counts = jnp.array(
        [
            [100.0, 1.0, 1.0, 1.0],
            [90.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 120.0],
            [1.0, 1.0, 1.0, 110.0],
        ]
    )
    dataset_indices = jnp.array([0, 0, 1, 1], dtype=jnp.int32)

    union_mask = compute_empirical_gene_coverage_mask(
        counts, coverage=0.8, dataset_indices=dataset_indices
    )
    assert bool(union_mask[0])
    assert bool(union_mask[3])


def test_aggregate_counts_by_mask_preserves_row_sums():
    """Pooling excluded genes should preserve per-row total counts."""
    counts = jnp.array([[5.0, 2.0, 1.0], [3.0, 1.0, 4.0]])
    mask = np.array([True, False, True], dtype=bool)
    aggregated = aggregate_counts_by_mask(counts, mask)
    np.testing.assert_allclose(
        np.asarray(aggregated).sum(axis=1), np.asarray(counts).sum(axis=1)
    )
    assert aggregated.shape == (2, 3)


def test_build_filtered_gene_names_appends_other():
    """Filtered name list should include trailing _other when genes excluded."""
    names = ["g0", "g1", "g2"]
    mask = np.array([True, False, True], dtype=bool)
    kept, excluded = build_filtered_gene_names(names, mask)
    assert kept == ["g0", "g2", "_other"]
    assert excluded == ["g1"]


def test_compute_gene_coverage_rank_descending():
    """Coverage ranks should assign rank 1 to highest-abundance gene."""
    counts = jnp.array([[10.0, 5.0, 2.0], [8.0, 5.0, 1.0]])
    ranks = compute_gene_coverage_rank(counts)
    assert ranks.shape == (3,)
    assert int(ranks[0]) == 1


def test_fit_gene_coverage_attaches_metadata(monkeypatch):
    """fit() should store coverage metadata on results and AnnData var."""
    anndata = pytest.importorskip("anndata")

    counts = np.array(
        [
            [40.0, 5.0, 1.0, 1.0],
            [35.0, 4.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 20.0],
            [2.0, 2.0, 2.0, 18.0],
        ]
    )
    adata = anndata.AnnData(
        X=counts,
        obs=pd.DataFrame({"dataset": ["a", "a", "b", "b"]}),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )

    def _fake_run_inference(method, **kwargs):
        model_config = kwargs["model_config"]
        return ScribeSVIResults(
            params={},
            loss_history=jnp.array([0.0]),
            n_cells=kwargs["n_cells"],
            n_genes=kwargs["n_genes"],
            model_type=model_config.base_model,
            model_config=model_config,
            prior_params={},
        )

    # Mock the heavy inference step so this test exercises fit() preprocessing only.
    monkeypatch.setattr("scribe.api._run_inference", _fake_run_inference)

    result = fit(
        adata,
        model="nbdm",
        inference_method="svi",
        n_steps=1,
        dataset_key="dataset",
        gene_coverage=0.8,
    )

    assert result.gene_coverage == pytest.approx(0.8)
    assert result.gene_coverage_mask is not None
    assert hasattr(result, "_excluded_gene_names")
    assert "scribe_gene_coverage_included" in adata.var.columns
    assert "scribe_gene_coverage_rank" in adata.var.columns


@dataclass
class _DummyEmpiricalResults:
    """Minimal results-like object for compare() empirical path tests."""

    posterior_samples: dict
    model_config: object
    var: pd.DataFrame
    _gene_coverage_mask: np.ndarray

    @property
    def layouts(self):
        """Return no layout metadata to trigger fallback behavior."""
        return {}


def test_compare_empirical_prefilter_auto_drops_other():
    """compare() should auto-drop trailing pooled gene for pre-filtered results."""
    cfg = (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_inference("svi")
        .build()
    )

    # 3 genes in model-space: g0, g1, _other
    r_a = np.abs(np.random.default_rng(0).normal(size=(100, 3))) + 1.0
    r_b = np.abs(np.random.default_rng(1).normal(size=(100, 3))) + 1.0
    var_df = pd.DataFrame(index=["g0", "g1", "_other"])
    original_mask = np.array([True, True, False, False], dtype=bool)

    res_a = _DummyEmpiricalResults(
        posterior_samples={"r": jnp.asarray(r_a)},
        model_config=cfg,
        var=var_df,
        _gene_coverage_mask=original_mask,
    )
    res_b = _DummyEmpiricalResults(
        posterior_samples={"r": jnp.asarray(r_b)},
        model_config=cfg,
        var=var_df,
        _gene_coverage_mask=original_mask,
    )

    de = compare(res_a, res_b, method="empirical")
    assert de.D == 2
    assert de.gene_names == ["g0", "g1"]
