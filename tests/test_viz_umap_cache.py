"""Tests for split-aware UMAP cache naming in visualization utilities."""

from omegaconf import OmegaConf

from viz_utils import _build_umap_cache_path


def test_umap_cache_path_without_subset_uses_full_suffix():
    """Ensure unsplit runs use a stable ``_full`` UMAP cache filename."""
    # Build a minimal config with no subset metadata.
    cfg = OmegaConf.create(
        {
            "data": {
                "path": "/tmp/example_dataset.h5ad",
            }
        }
    )

    cache_path = _build_umap_cache_path(cfg=cfg, cache_umap=True)

    # Verify the filename is deterministic for unsplit data runs.
    assert cache_path == "/tmp/example_dataset_umap_full.pkl"


def test_umap_cache_path_changes_across_subset_values():
    """Ensure different split values generate different cache file paths."""
    # Build two configs that point to the same dataset but different subset
    # values. These must not share a cache path.
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

    # Confirm split-specific naming and collision resistance.
    assert cache_path_a != cache_path_b
    assert "exp_condition" in cache_path_a
    assert "bleo_d2" in cache_path_a
    assert "exp_condition" in cache_path_b
    assert "bleo_d7" in cache_path_b


def test_umap_cache_path_is_stable_for_same_subset():
    """Ensure repeated calls with same split config return the same path."""
    # Build one split config and call the helper twice to verify deterministic
    # output for stable cache reuse.
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

    # The same config should always resolve to the same cache path.
    assert cache_path_1 == cache_path_2
