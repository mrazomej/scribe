"""Tests for infer_split orchestration helpers."""

from pathlib import Path

from infer_split import (
    _build_submitit_multirun_command,
    _derive_output_prefix,
    _generate_tmp_yamls,
    _parse_args,
)


def test_derive_output_prefix_uses_parent_path_of_data_key():
    """Extract nested prefix from a data config key."""
    prefix = _derive_output_prefix(
        "panfibrosis/CKD/GSE140023_filter-none_split-disease"
    )
    assert prefix == "panfibrosis/CKD"


def test_generate_tmp_yamls_keeps_name_slash_safe_and_sets_prefix(
    tmp_path: Path,
):
    """Generated split configs should keep safe names plus output prefix."""
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
    assert "name: panfibrosis_ckd_gse140023_treated" in text
    assert "output_prefix: panfibrosis/CKD" in text


def test_parse_args_supports_multiple_data_configs():
    """Comma-separated data= should parse into multiple config keys."""
    data_names, data_overrides, split_overrides, forwarded = _parse_args(
        ["data=a/b,c/d", "data.n_jobs=4", "split.launcher=submitit_slurm", "model=zinbvcp"]
    )
    assert data_names == ["a/b", "c/d"]
    assert data_overrides == {"n_jobs": "4"}
    assert split_overrides == {"launcher": "submitit_slurm"}
    assert forwarded == ["model=zinbvcp"]


def test_submitit_command_uses_nested_output_subdir_override():
    """Hydra subdir override should encode nested grouping from output_prefix."""
    cmd = _build_submitit_multirun_command(
        data_list="_tmp_split_x/a,_tmp_split_x/b",
        n_jobs=4,
        split_overrides={},
        forwarded_args=[],
    )
    assert (
        "hydra.sweep.subdir='${data.output_prefix}/${data.name}/${model}/${inference.method}/${sanitize_dirname:${hydra:job.override_dirname}}'"
        in cmd
    )
