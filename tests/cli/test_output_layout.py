"""Tests for output directory layout helpers."""

from omegaconf import OmegaConf

from scribe.cli.output_layout import (
    derive_output_prefix,
    format_nested_output_prefix,
    resolve_nested_output_prefix,
)


def test_derive_output_prefix_uses_parent_path_of_data_key():
    """Nested data keys should map to their parent folder path."""
    prefix = derive_output_prefix(
        "panfibrosis/cell_type_genecorr/CKD/GSE140023__sham__from__foo"
    )
    assert prefix == "panfibrosis/cell_type_genecorr/CKD"


def test_derive_output_prefix_returns_empty_for_flat_key():
    """Flat data keys should not introduce a nested prefix."""
    assert derive_output_prefix("my_dataset") == ""


def test_derive_output_prefix_ignores_empty_path_segments():
    """Extra slashes in the data key should not create empty path segments."""
    assert derive_output_prefix("panfibrosis//CKD/foo") == "panfibrosis/CKD"


def test_format_nested_output_prefix_adds_trailing_slash():
    """Non-empty prefixes should include a trailing slash for path templates."""
    assert format_nested_output_prefix("panfibrosis/CKD") == "panfibrosis/CKD/"
    assert format_nested_output_prefix("") == ""


def test_resolve_nested_output_prefix_prefers_explicit_yaml_value():
    """Explicit output_prefix in the data config should win over derivation."""
    data_cfg = OmegaConf.create({"output_prefix": "custom/prefix"})
    resolved = resolve_nested_output_prefix(
        data_cfg=data_cfg,
        data_choice="panfibrosis/CKD/foo",
    )
    assert resolved == "custom/prefix/"


def test_resolve_nested_output_prefix_derives_from_data_choice():
    """Missing output_prefix should derive from the selected data config key."""
    data_cfg = OmegaConf.create({"name": "my_dataset"})
    resolved = resolve_nested_output_prefix(
        data_cfg=data_cfg,
        data_choice="panfibrosis/cell_type_genecorr/CKD/foo",
    )
    assert resolved == "panfibrosis/cell_type_genecorr/CKD/"


def test_resolve_nested_output_prefix_returns_empty_for_flat_choice():
    """Flat data choices should resolve to an empty prefix segment."""
    data_cfg = OmegaConf.create({"name": "my_dataset"})
    resolved = resolve_nested_output_prefix(
        data_cfg=data_cfg,
        data_choice="my_dataset",
    )
    assert resolved == ""


def test_resolve_nested_output_prefix_skips_tmp_split_keys():
    """Temporary split config keys must not produce misleading prefixes."""
    data_cfg = OmegaConf.create({"name": "split_leaf"})
    resolved = resolve_nested_output_prefix(
        data_cfg=data_cfg,
        data_choice="_tmp_split_123/source__split_leaf",
    )
    assert resolved == ""
