"""Tests for the leaf-grouping accessor: get_group / iter_groups / GroupView.

A multi-factor fit flattens present factor combinations into a leaf axis;
``get_group`` slices that grid by fixed factor level(s) and hands back the
matching leaves as single-dataset views (via the parent's ``get_dataset``).
These tests use a real ``GroupingSpec`` and a stub ``get_dataset`` that echoes
the leaf index, so the leaf->view wiring is verified without a fit.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from scribe.core.grouping_view import GroupView, get_group, group_levels, iter_groups
from scribe.models.config.grouping import normalize_grouping

_PRIORS = {
    t: "none"
    for t in ("expression", "prob", "zero_inflation", "overdispersion", "regime")
}


def _spec(columns, keys):
    spec, _ = normalize_grouping(
        dataset_key=list(keys),
        hierarchy=None,
        interactions=None,
        obs=pd.DataFrame(columns),
        dataset_priors=_PRIORS,
    )
    return spec


def _results(spec):
    """Fake results: grouping_spec + a get_dataset that echoes the leaf index."""
    r = SimpleNamespace(model_config=SimpleNamespace(grouping_spec=spec))
    r.get_dataset = lambda i: ("view", i)
    return r


@pytest.fixture
def crossed():
    """condition (2) x sample (3), fully crossed."""
    spec = _spec(
        {
            "condition": ["control", "drug"] * 3,
            "sample": ["D1", "D1", "D2", "D2", "D3", "D3"],
        },
        keys=["condition", "sample"],
    )
    return _results(spec), spec


# ------------------------------------------------------------------------------
# get_group: single fixed factor
# ------------------------------------------------------------------------------


def test_get_group_fixes_one_factor(crossed):
    results, spec = crossed
    g = get_group(results, sample="D3")
    coords = spec.leaf_coords()
    expected = [i for i, c in enumerate(coords) if c["sample"] == "D3"]

    assert isinstance(g, GroupView)
    assert g.leaves == expected
    assert g.free_factors == ["condition"]
    assert set(g.keys()) == {"control", "drug"}
    assert len(g) == 2


def test_get_group_indexing_materializes_correct_leaf(crossed):
    results, spec = crossed
    g = get_group(results, sample="D3")
    coords = spec.leaf_coords()
    ctrl_leaf = next(
        i
        for i, c in enumerate(coords)
        if c["sample"] == "D3" and c["condition"] == "control"
    )
    # Stub get_dataset echoes ("view", leaf_index).
    assert g["control"] == ("view", ctrl_leaf)
    assert "control" in g
    assert "drug" in g


def test_get_group_labels_and_views(crossed):
    results, spec = crossed
    g = get_group(results, sample="D3")
    assert all("D3" in lbl for lbl in g.labels)
    views = g.views()
    assert set(views) == {"control", "drug"}
    # items() yields (key, view) consistent with views().
    assert dict(g.items()) == views


# ------------------------------------------------------------------------------
# iter_groups / group_levels
# ------------------------------------------------------------------------------


def test_group_levels_present_order(crossed):
    results, _ = crossed
    assert group_levels(results, "sample") == ["D1", "D2", "D3"]
    assert group_levels(results, "condition") == ["control", "drug"]


def test_iter_groups_enumerates_each_level(crossed):
    results, _ = crossed
    groups = dict(iter_groups(results, "sample"))
    assert list(groups) == ["D1", "D2", "D3"]
    for level, g in groups.items():
        assert g.fixed == {"sample": level}
        assert len(g) == 2  # control + drug


# ------------------------------------------------------------------------------
# Multi-free-factor keying + fully-specified single leaf
# ------------------------------------------------------------------------------


def test_get_group_multi_free_factor_keys_are_tuples():
    spec = _spec(
        {
            "condition": ["control", "drug", "control", "drug"],
            "sample": ["D1", "D1", "D2", "D2"],
            "batch": ["b1", "b1", "b2", "b2"],
        },
        keys=["condition", "sample", "batch"],
    )
    results = _results(spec)
    g = get_group(results, condition="control")
    assert g.free_factors == ["sample", "batch"]
    # Two control leaves -> keyed by (sample, batch) tuples.
    assert all(isinstance(k, tuple) and len(k) == 2 for k in g.keys())
    key = g.keys()[0]
    assert g[key] == ("view", g.leaf_for(key))


def test_get_group_fully_specified_single_leaf(crossed):
    results, spec = crossed
    g = get_group(results, condition="control", sample="D2")
    assert len(g) == 1
    assert g.free_factors == []
    # .dataset / view() resolve the sole leaf without a key.
    assert g.dataset == g.view()
    assert g.dataset == ("view", g.leaves[0])


# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------


def test_get_group_unknown_factor(crossed):
    results, _ = crossed
    with pytest.raises(ValueError, match="unknown factor"):
        get_group(results, donor="D1")


def test_get_group_no_match(crossed):
    results, _ = crossed
    with pytest.raises(ValueError, match="no present leaves match"):
        get_group(results, sample="D9")


def test_get_group_requires_filter(crossed):
    results, _ = crossed
    with pytest.raises(ValueError, match="at least one factor=level"):
        get_group(results)


def test_get_group_requires_grouping_spec():
    results = SimpleNamespace(model_config=SimpleNamespace(grouping_spec=None))
    with pytest.raises(ValueError, match="require a multi-factor fit"):
        get_group(results, sample="D1")


def test_group_view_missing_key_raises(crossed):
    results, _ = crossed
    g = get_group(results, sample="D3")
    with pytest.raises(KeyError, match="no leaf with"):
        g["nonexistent"]
