"""Tests for catalog parsing and filter matching behavior.

These tests focus on edge-cases where comma-delimited values appear in
override directory names or filter dictionaries.
"""

from pathlib import Path

import pytest

from scribe.catalog import ExperimentCatalog, ExperimentRun


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
