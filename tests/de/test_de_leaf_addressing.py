"""Tests for leaf addressing + labels in compare_datasets (M1).

Covers the pure helpers ``_resolve_leaf_index`` / ``_leaf_label`` that let
``compare_datasets`` accept integer or ``{factor: level}`` dict leaf addresses
and report human-readable labels instead of ``dataset_<i>``.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from scribe.models.config.grouping import normalize_grouping
from scribe.de._results_factory import _resolve_leaf_index, _leaf_label


@pytest.fixture
def stub_results():
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D1", "D2", "D2", "D3"],
            "treatment": ["control", "drug", "control", "drug", "control"],
        }
    )
    spec, _ = normalize_grouping(
        dataset_key=["treatment", "sample"],
        hierarchy=None,
        interactions=None,
        obs=obs,
        dataset_priors={
            t: "none"
            for t in ("expression", "prob", "zero_inflation", "overdispersion", "regime")
        },
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=spec, n_datasets=spec.n_leaves)
    )


def test_int_passthrough(stub_results):
    assert _resolve_leaf_index(stub_results, 3) == 3


def test_dict_full_match(stub_results):
    # leaves are sorted: control|D1=0, control|D2=1, control|D3=2, drug|D1=3, drug|D2=4
    idx = _resolve_leaf_index(
        stub_results, {"treatment": "drug", "sample": "D1"}
    )
    assert idx == 3


def test_dict_ambiguous_partial(stub_results):
    # Partial selector matching >1 leaf must error.
    with pytest.raises(ValueError, match="matched 3 leaves"):
        _resolve_leaf_index(stub_results, {"treatment": "control"})


def test_dict_no_match(stub_results):
    with pytest.raises(ValueError, match="matched 0 leaves"):
        _resolve_leaf_index(stub_results, {"treatment": "drug", "sample": "D3"})


def test_dict_without_grouping_spec():
    res = SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=None, n_datasets=3)
    )
    with pytest.raises(ValueError, match="requires a multi-factor grouping_spec"):
        _resolve_leaf_index(res, {"sample": "D1"})


def test_leaf_label(stub_results):
    assert _leaf_label(stub_results, 3) == "drug | D1"


def test_leaf_label_fallback():
    res = SimpleNamespace(
        model_config=SimpleNamespace(grouping_spec=None, n_datasets=3)
    )
    assert _leaf_label(res, 1) == "dataset_1"
