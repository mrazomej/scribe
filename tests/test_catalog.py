"""Tests for catalog parsing and filter matching behavior.

These tests focus on edge-cases where comma-delimited values appear in
override directory names or filter dictionaries.
"""

from pathlib import Path

import pytest

from scribe.catalog import (
    ExperimentCatalog,
    ExperimentRun,
    _expand_catalog_root_paths,
)


def test_expand_catalog_root_paths_non_glob_resolves_directory(tmp_path):
    """Literal paths resolve to a single existing directory.

    Returns
    -------
    None
        Asserts expander returns one resolved Path for a normal directory.
    """
    base = tmp_path / "only_root"
    base.mkdir()

    roots = _expand_catalog_root_paths(str(base))

    assert roots == [base.resolve()]


def test_expand_catalog_root_paths_glob_collects_multiple_directories(tmp_path):
    """Glob patterns expand to every matching directory.

    Returns
    -------
    None
        Asserts metacharacters select multiple roots and skip files.
    """
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_c.txt").write_text("not-a-dir", encoding="utf-8")

    roots = _expand_catalog_root_paths(str(tmp_path / "run_*"))

    assert [p.resolve() for p in roots] == sorted(
        [(tmp_path / "run_a").resolve(), (tmp_path / "run_b").resolve()]
    )


def test_expand_catalog_root_paths_recursive_glob(tmp_path):
    """``**`` in the pattern enables recursive directory discovery.

    Returns
    -------
    None
        Asserts nested directories match when the pattern uses ``**/``.
    """
    nested = tmp_path / "outer" / "inner" / "leaf"
    nested.mkdir(parents=True)
    (tmp_path / "outer" / "skip.txt").write_text("x", encoding="utf-8")

    pattern = str(tmp_path / "outer" / "**" / "leaf")
    roots = _expand_catalog_root_paths(pattern)

    assert roots == [nested.resolve()]


def test_expand_catalog_root_paths_empty_glob_raises(tmp_path):
    """A glob that matches no directories raises FileNotFoundError.

    Returns
    -------
    None
        Asserts missing matches surface as a clear error.
    """
    (tmp_path / "nothing_here").mkdir()

    with pytest.raises(FileNotFoundError, match="matched no directories"):
        _expand_catalog_root_paths(str(tmp_path / "no_such_*"))


def test_expand_catalog_root_paths_nonexistent_literal_raises(tmp_path):
    """A literal path that does not exist raises FileNotFoundError.

    Returns
    -------
    None
        Asserts the same validation as pre-glob expansion.
    """
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _expand_catalog_root_paths(str(missing))


def test_expand_catalog_root_paths_file_literal_raises_not_a_directory(tmp_path):
    """A literal path that is a file raises NotADirectoryError.

    Returns
    -------
    None
        Asserts catalog roots must be directories.
    """
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="must be a directory"):
        _expand_catalog_root_paths(str(file_path))


def test_experiment_catalog_init_dedupes_glob_roots(monkeypatch, tmp_path):
    """Overlapping glob results and list entries are only scanned once.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub scanning so the test does not require Hydra layouts.

    Returns
    -------
    None
        Asserts ``base_dirs`` contains unique paths when inputs overlap.
    """
    shared = tmp_path / "shared"
    shared.mkdir()

    monkeypatch.setattr(
        ExperimentCatalog,
        "_scan_experiments",
        lambda self: [],
    )

    catalog = ExperimentCatalog([str(shared), str(tmp_path / "sha*")])

    assert catalog.base_dirs == [shared.resolve()]
    assert catalog.base_dir == shared.resolve()


def _build_catalog_with_experiments(experiments: list[ExperimentRun]) -> ExperimentCatalog:
    """Create a catalog instance without triggering filesystem scanning.

    Parameters
    ----------
    experiments : list[ExperimentRun]
        In-memory experiment runs used by test cases.

    Returns
    -------
    ExperimentCatalog
        Catalog instance populated with the provided experiments.
    """
    # Bypass __init__ so tests stay unit-level and avoid disk traversal.
    catalog = ExperimentCatalog.__new__(ExperimentCatalog)
    catalog.experiments = experiments
    return catalog


def _sample_experiments() -> list[ExperimentRun]:
    """Create representative in-memory experiments for filtering tests.

    Returns
    -------
    list[ExperimentRun]
        Two experiments with distinct path names and metadata values.
    """
    # Keep path names expressive so lambda predicates can target run names.
    return [
        ExperimentRun(
            path=(
                "/tmp/bleo/annotation_key=cell-class,"
                "mixture_params=phi,mu,gate,model=zinbvcp"
            ),
            metadata={
                "model": "zinbvcp",
                "mixture_params": ["phi", "mu", "gate"],
                "annotation_key": "cell-class",
                "inference": {"batch_size": 4096},
            },
        ),
        ExperimentRun(
            path="/tmp/bleo/model=nbdm,annotation_key=cell-class",
            metadata={
                "model": "nbdm",
                "mixture_params": ["phi", "mu"],
                "annotation_key": "cell-class",
                "inference": {"batch_size": 1024},
            },
        ),
    ]


def test_parse_override_dirname_preserves_comma_delimited_values():
    """Parse values with internal commas as a single key-value pair.

    Returns
    -------
    None
        Asserts that ``mixture_params=phi,mu,gate`` stays attached to
        ``mixture_params`` instead of being split into separate parameters.
    """
    # Use __new__ to bypass filesystem scanning in ExperimentCatalog.__init__.
    catalog = ExperimentCatalog.__new__(ExperimentCatalog)
    parsed = catalog._parse_override_dirname(
        "model=zinbvcp,mixture_params=phi,mu,gate,annotation_key=cell-class"
    )

    assert parsed["model"] == "zinbvcp"
    assert parsed["mixture_params"] == "phi,mu,gate"
    assert parsed["annotation_key"] == "cell-class"


def test_parse_override_dirname_parses_nested_and_typed_values():
    """Parse nested keys and scalar types from override directory names.

    Returns
    -------
    None
        Asserts typed conversion and dot-key nesting behavior.
    """
    # Use __new__ to bypass filesystem scanning in ExperimentCatalog.__init__.
    catalog = ExperimentCatalog.__new__(ExperimentCatalog)
    parsed = catalog._parse_override_dirname(
        "inference.batch_size=4096,guide_rank=32,variable_capture=false"
    )

    assert parsed["inference"]["batch_size"] == 4096
    assert parsed["guide_rank"] == 32
    assert parsed["variable_capture"] is False


def test_parse_override_dirname_supports_leading_bare_boolean_tokens():
    """Parse compact dirname tokens that encode booleans as bare keys.

    Returns
    -------
    None
        Asserts that key-only boolean tokens are parsed as ``True`` while
        comma-delimited values in subsequent key-value entries remain intact.
    """
    # Use __new__ to bypass filesystem scanning in ExperimentCatalog.__init__.
    catalog = ExperimentCatalog.__new__(ExperimentCatalog)
    parsed = catalog._parse_override_dirname(
        "expression_dataset_prior=gaussian,guide_rank=256,mixture_params=phi,mu,gate"
    )

    assert parsed["expression_dataset_prior"] == "gaussian"
    assert parsed["guide_rank"] == 256
    assert parsed["mixture_params"] == "phi,mu,gate"


def test_find_matches_singleton_comma_list_filter_to_metadata_list():
    """Match list metadata against singleton comma-delimited list filters.

    Returns
    -------
    None
        Asserts that a filter value like ``["phi,mu,gate"]`` matches metadata
        stored as ``["phi", "mu", "gate"]``.
    """
    experiment = _sample_experiments()[0]
    catalog = _build_catalog_with_experiments([experiment])

    matches = catalog.find(
        mixture_params=["phi,mu,gate"], annotation_key="cell-class"
    )

    assert matches == [experiment]


def test_find_matches_comma_string_filter_to_metadata_list():
    """Match list metadata against comma-delimited string filters.

    Returns
    -------
    None
        Asserts that a filter value like ``"phi,mu,gate"`` matches metadata
        stored as ``["phi", "mu", "gate"]``.
    """
    experiment = _sample_experiments()[0]
    catalog = _build_catalog_with_experiments([experiment])

    matches = catalog.find(
        mixture_params="phi,mu,gate", annotation_key="cell-class"
    )

    assert matches == [experiment]


def test_find_matches_direct_list_filter_to_metadata_list():
    """Match list metadata against direct list-valued filters.

    Returns
    -------
    None
        Asserts that exact list filters are still supported.
    """
    experiment = _sample_experiments()[0]
    catalog = _build_catalog_with_experiments([experiment])

    matches = catalog.find(
        mixture_params=["phi", "mu", "gate"], annotation_key="cell-class"
    )

    assert matches == [experiment]


def test_find_supports_nested_dot_key_filters():
    """Resolve dot-key filters against nested metadata dictionaries.

    Returns
    -------
    None
        Asserts nested lookup works for keys like ``inference.batch_size``.
    """
    experiments = _sample_experiments()
    catalog = _build_catalog_with_experiments(experiments)

    matches = catalog.find(model="zinbvcp", **{"inference.batch_size": 4096})

    assert matches == [experiments[0]]


def test_find_supports_flattened_dot_key_filters():
    """Resolve dot-key filters against flattened metadata dictionaries.

    Returns
    -------
    None
        Asserts flattened keys like ``inference.enable_x64`` are matched
        directly without requiring nested dictionary traversal.
    """
    experiment = ExperimentRun(
        path="/tmp/bleo/model=nbvcp,inference.enable_x64=True",
        metadata={
            "model": "nbvcp",
            "inference.enable_x64": True,
        },
    )
    catalog = _build_catalog_with_experiments([experiment])

    matches = catalog.find(model="nbvcp", **{"inference.enable_x64": True})

    assert matches == [experiment]


def test_filter_with_lambda_over_run_name():
    """Filter experiments with a lambda operating on run path names.

    Returns
    -------
    None
        Asserts lambda predicates can target path-based run names.
    """
    experiments = _sample_experiments()
    catalog = _build_catalog_with_experiments(experiments)

    # Path(exp.path).name isolates the run directory name string.
    filtered = catalog.filter(
        lambda exp: "mixture_params=phi,mu,gate"
        in Path(exp.path).name
    )

    assert filtered == [experiments[0]]


def test_filter_accepts_pre_filtered_subsets():
    """Support chaining with metadata find + callable filtering.

    Returns
    -------
    None
        Asserts ``catalog.filter`` can refine the output from ``catalog.find``.
    """
    experiments = _sample_experiments()
    catalog = _build_catalog_with_experiments(experiments)

    nbdm_subset = catalog.find(model="nbdm")
    filtered = catalog.filter(
        lambda exp: exp.metadata["inference"]["batch_size"] < 2048,
        experiments=nbdm_subset,
    )

    assert filtered == [experiments[1]]


def test_filter_raises_type_error_for_non_callable_predicate():
    """Reject non-callable predicate inputs for catalog.filter.

    Returns
    -------
    None
        Asserts helpful error behavior for invalid predicate arguments.
    """
    catalog = _build_catalog_with_experiments(_sample_experiments())

    with pytest.raises(TypeError, match="predicate must be callable"):
        catalog.filter("not-a-callable")  # type: ignore[arg-type]


def test_experiment_run_load_data_replays_pipeline_by_default(monkeypatch):
    """Replay configured data pipeline by default in ExperimentRun.load_data.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace config loading and data loader calls.

    Returns
    -------
    None
        Asserts that preprocessing, subsetting, and filter_obs config fields are
        forwarded to ``load_and_preprocess_anndata`` when preprocessing is left
        at the default ``True``.
    """
    # Build a run with in-memory config so this stays unit-level.
    run = ExperimentRun(path="/tmp/exp", metadata={})
    mock_config = {
        "data": {
            "path": "relative/data.h5ad",
            "preprocessing": {"filter_cells": {"min_counts": 1000}},
            "subset_column": "condition",
            "subset_value": "Sham",
            "filter_obs": {"batch": ["A"]},
        }
    }

    # Avoid filesystem-dependent path resolution in the unit test.
    monkeypatch.setattr(run, "load_config", lambda: mock_config)
    monkeypatch.setattr(
        "scribe.catalog._resolve_run_data_path",
        lambda _run_path, p: f"/abs/{p}",
    )

    # Capture loader kwargs to verify the forwarding behavior.
    captured: dict = {}

    def _mock_loader(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(
        "scribe.data_loader.load_and_preprocess_anndata",
        _mock_loader,
    )

    result = run.load_data(return_jax=False)

    assert result == "sentinel"
    assert captured["path"] == "/abs/relative/data.h5ad"
    assert captured["kwargs"]["prep_config"] == {
        "filter_cells": {"min_counts": 1000}
    }
    assert captured["kwargs"]["return_jax"] is False
    assert captured["kwargs"]["subset_column"] == "condition"
    assert captured["kwargs"]["subset_value"] == "Sham"
    assert captured["kwargs"]["filter_obs"] == {"batch": ["A"]}


def test_experiment_run_load_data_skips_pipeline_when_disabled(monkeypatch):
    """Skip configured pipeline when ``preprocessing=False`` is provided.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace config loading and data loader calls.

    Returns
    -------
    None
        Asserts pipeline-specific kwargs are set to ``None`` for raw loading.
    """
    # Reuse the same configuration shape as production Hydra configs.
    run = ExperimentRun(path="/tmp/exp", metadata={})
    mock_config = {
        "data": {
            "path": "relative/data.h5ad",
            "preprocessing": {"filter_cells": {"min_counts": 1000}},
            "subset_column": "condition",
            "subset_value": "Sham",
            "filter_obs": {"batch": ["A"]},
        }
    }

    monkeypatch.setattr(run, "load_config", lambda: mock_config)
    monkeypatch.setattr(
        "scribe.catalog._resolve_run_data_path",
        lambda _run_path, p: f"/abs/{p}",
    )

    captured: dict = {}

    def _mock_loader(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return "sentinel"

    monkeypatch.setattr(
        "scribe.data_loader.load_and_preprocess_anndata",
        _mock_loader,
    )

    result = run.load_data(return_jax=True, preprocessing=False)

    assert result == "sentinel"
    assert captured["path"] == "/abs/relative/data.h5ad"
    assert captured["kwargs"]["return_jax"] is True
    assert captured["kwargs"]["prep_config"] is None
    assert captured["kwargs"]["subset_column"] is None
    assert captured["kwargs"]["subset_value"] is None
    assert captured["kwargs"]["filter_obs"] is None


def test_catalog_load_data_forwards_preprocessing_flag():
    """Forward preprocessing argument through ExperimentCatalog.load_data.

    Returns
    -------
    None
        Asserts ``ExperimentCatalog.load_data`` passes ``preprocessing`` and
        ``return_jax`` through to the selected run.
    """
    # Build a catalog with one in-memory run to avoid filesystem scans.
    run = ExperimentRun(path="/tmp/exp", metadata={"model": "zinbvcp"})
    catalog = _build_catalog_with_experiments([run])

    # Monkeypatch the bound method directly on this run instance.
    captured: dict = {}

    def _mock_run_load_data(return_jax, preprocessing):
        captured["return_jax"] = return_jax
        captured["preprocessing"] = preprocessing
        return "sentinel"

    run.load_data = _mock_run_load_data  # type: ignore[method-assign]

    result = catalog.load_data(
        return_jax=False, preprocessing=False, model="zinbvcp"
    )

    assert result == "sentinel"
    assert captured["return_jax"] is False
    assert captured["preprocessing"] is False
