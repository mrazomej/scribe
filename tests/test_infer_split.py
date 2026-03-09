"""Tests for infer_split orchestration helpers."""

from pathlib import Path

from infer_split import _derive_output_prefix, _generate_tmp_yamls


def test_derive_output_prefix_uses_parent_path_of_data_key():
    """Extract nested prefix from a data config key."""
    prefix = _derive_output_prefix(
        "panfibrosis/CKD/GSE140023_filter-none_split-disease"
    )
    assert prefix == "panfibrosis/CKD"


def test_generate_tmp_yamls_prefixes_data_name_for_nested_outputs(
    tmp_path: Path,
):
    """Generated split configs should preserve nested output grouping."""
    data_cfg = {
        "name": "panfibrosis_ckd_gse140023",
        "path": "/tmp/mock.h5ad",
        "filter_obs": {},
    }
    tmp_dir = tmp_path / "_tmp_split_case"

    names = _generate_tmp_yamls(
        data_cfg=data_cfg,
        data_name="panfibrosis/CKD/GSE140023_filter-none_split-disease",
        split_by="condition",
        covariate_values=["treated"],
        gpu_ids=["0"],
        tmp_dir=tmp_dir,
    )

    assert len(names) == 1
    yaml_path = tmp_dir / f"{names[0]}.yaml"
    assert yaml_path.exists()
    text = yaml_path.read_text()
    assert "name: panfibrosis/CKD/panfibrosis_ckd_gse140023_treated" in text
