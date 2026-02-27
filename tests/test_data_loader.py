"""Tests for load_and_preprocess_anndata in src/scribe/data_loader.py.

Covers single-column and multi-column subsetting paths without requiring
real h5ad files on disk (AnnData objects are built in-memory).
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scribe.data_loader import load_and_preprocess_anndata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_h5ad(tmp_path, obs_df: pd.DataFrame) -> str:
    """Write a minimal h5ad file with the given obs metadata and return its path."""
    n_cells = len(obs_df)
    n_genes = 5
    # Use integer counts so the loader can convert to JAX array cleanly
    X = np.ones((n_cells, n_genes), dtype=np.float32)
    adata = AnnData(X=X, obs=obs_df)
    path = str(tmp_path / "test.h5ad")
    adata.write_h5ad(path)
    return path


# ---------------------------------------------------------------------------
# Single-column subsetting (existing behaviour, regression tests)
# ---------------------------------------------------------------------------


def test_single_column_subset_retains_matching_cells(tmp_path):
    """Single-column subset should keep only cells matching the target value."""
    obs = pd.DataFrame(
        {"treatment": ["ctrl", "ctrl", "drug", "drug"]},
        index=[f"cell{i}" for i in range(4)],
    )
    path = _make_h5ad(tmp_path, obs)

    result = load_and_preprocess_anndata(
        path,
        return_jax=False,
        subset_column="treatment",
        subset_value="drug",
    )
    assert result.shape[0] == 2
    assert (result.obs["treatment"] == "drug").all()


def test_single_column_subset_raises_on_missing_column(tmp_path):
    """Passing a non-existent column name should raise ValueError."""
    obs = pd.DataFrame(
        {"treatment": ["ctrl", "drug"]},
        index=["c0", "c1"],
    )
    path = _make_h5ad(tmp_path, obs)

    with pytest.raises(ValueError, match="not found in adata.obs"):
        load_and_preprocess_anndata(
            path,
            return_jax=False,
            subset_column="nonexistent",
            subset_value="ctrl",
        )


def test_single_column_subset_raises_when_no_cells_match(tmp_path):
    """A value present in no cells should raise ValueError."""
    obs = pd.DataFrame(
        {"treatment": ["ctrl", "ctrl"]},
        index=["c0", "c1"],
    )
    path = _make_h5ad(tmp_path, obs)

    with pytest.raises(ValueError, match="No observations found"):
        load_and_preprocess_anndata(
            path,
            return_jax=False,
            subset_column="treatment",
            subset_value="drug",
        )


# ---------------------------------------------------------------------------
# Multi-column subsetting (new behaviour)
# ---------------------------------------------------------------------------


def test_multi_column_subset_retains_matching_combination(tmp_path):
    """Multi-column subset should keep only cells matching all column conditions."""
    obs = pd.DataFrame(
        {
            "treatment": ["ctrl", "ctrl", "drug", "drug"],
            "kit": ["10x", "dropseq", "10x", "dropseq"],
        },
        index=[f"cell{i}" for i in range(4)],
    )
    path = _make_h5ad(tmp_path, obs)

    result = load_and_preprocess_anndata(
        path,
        return_jax=False,
        subset_column=["treatment", "kit"],
        subset_value=["drug", "10x"],
    )
    # Only cell2 matches treatment=drug AND kit=10x
    assert result.shape[0] == 1
    assert result.obs["treatment"].iloc[0] == "drug"
    assert result.obs["kit"].iloc[0] == "10x"


def test_multi_column_subset_returns_multiple_matching_cells(tmp_path):
    """Multi-column subset works when multiple cells satisfy all conditions."""
    obs = pd.DataFrame(
        {
            "treatment": ["ctrl", "ctrl", "drug", "drug", "drug"],
            "kit": ["10x", "10x", "10x", "10x", "dropseq"],
        },
        index=[f"cell{i}" for i in range(5)],
    )
    path = _make_h5ad(tmp_path, obs)

    result = load_and_preprocess_anndata(
        path,
        return_jax=False,
        subset_column=["treatment", "kit"],
        subset_value=["drug", "10x"],
    )
    # cell2 and cell3 both match
    assert result.shape[0] == 2


def test_multi_column_subset_raises_on_missing_column(tmp_path):
    """Passing a non-existent column in the list should raise ValueError."""
    obs = pd.DataFrame(
        {"treatment": ["ctrl", "drug"]},
        index=["c0", "c1"],
    )
    path = _make_h5ad(tmp_path, obs)

    with pytest.raises(ValueError, match="not found in adata.obs"):
        load_and_preprocess_anndata(
            path,
            return_jax=False,
            subset_column=["treatment", "nonexistent"],
            subset_value=["ctrl", "val"],
        )


def test_multi_column_subset_raises_when_no_cells_match(tmp_path):
    """A combination that exists in no cells should raise ValueError."""
    obs = pd.DataFrame(
        {
            "treatment": ["ctrl", "drug"],
            "kit": ["10x", "dropseq"],
        },
        index=["c0", "c1"],
    )
    path = _make_h5ad(tmp_path, obs)

    # ctrl + dropseq combination does not exist
    with pytest.raises(ValueError, match="No observations found"):
        load_and_preprocess_anndata(
            path,
            return_jax=False,
            subset_column=["treatment", "kit"],
            subset_value=["ctrl", "dropseq"],
        )


def test_no_subset_returns_all_cells(tmp_path):
    """When no subset arguments are given all cells should be returned."""
    obs = pd.DataFrame(
        {"treatment": ["ctrl", "drug", "drug"]},
        index=["c0", "c1", "c2"],
    )
    path = _make_h5ad(tmp_path, obs)

    result = load_and_preprocess_anndata(path, return_jax=False)
    assert result.shape[0] == 3
