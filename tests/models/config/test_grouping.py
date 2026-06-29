"""Tests for multi-factor grouping normalization (Milestone 1).

Covers ``scribe.models.config.grouping``: the user-facing ``GroupLevel``, the
canonical ``GroupingSpec``/``Factor`` objects, prior-dict resolution, and the
``normalize_grouping`` builder that turns the three user spellings into a single
present-only leaf representation.
"""

import numpy as np
import pandas as pd
import pytest

from scribe.models.config.grouping import (
    GroupLevel,
    GroupingSpec,
    normalize_grouping,
    resolve_dataset_prior_dict,
    _reduce_leaf_axis_family,
    TARGET_NAMES,
)


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def obs_crossed():
    """A donor x treatment design with one missing combination (D3, drug)."""
    sample = ["D1", "D1", "D1", "D2", "D2", "D2", "D3"]
    treatment = [
        "control",
        "control",
        "drug",
        "control",
        "drug",
        "drug",
        "control",
    ]
    return pd.DataFrame({"sample": sample, "treatment": treatment})


def _no_priors():
    return {t: "none" for t in TARGET_NAMES}


# ------------------------------------------------------------------------------
# GroupLevel validation
# ------------------------------------------------------------------------------


def test_grouplevel_defaults():
    gl = GroupLevel(name="sample")
    assert gl.name == "sample"
    assert gl.nested_in is None
    assert gl.effect_type == "random"
    assert gl.fixed_scale is None


def test_grouplevel_bad_effect_type():
    with pytest.raises(ValueError):
        GroupLevel(name="t", effect_type="mixed")


def test_grouplevel_fixed_scale_positive():
    GroupLevel(name="t", effect_type="fixed", fixed_scale=0.5)
    with pytest.raises(ValueError):
        GroupLevel(name="t", effect_type="fixed", fixed_scale=0.0)
    with pytest.raises(ValueError):
        GroupLevel(name="t", effect_type="fixed", fixed_scale=-1.0)


def test_grouplevel_positional_and_keyword_name():
    # Positional ergonomic form.
    assert GroupLevel("treatment").name == "treatment"
    gl = GroupLevel("treatment", effect_type="fixed", fixed_scale=0.5)
    assert gl.name == "treatment" and gl.effect_type == "fixed"
    # Keyword form still works.
    assert GroupLevel(name="sample").name == "sample"


# ------------------------------------------------------------------------------
# resolve_dataset_prior_dict
# ------------------------------------------------------------------------------


def test_resolve_prior_broadcast():
    out = resolve_dataset_prior_dict("gaussian", ("a", "b"))
    assert {k: v.type for k, v in out.items()} == {
        "a": "gaussian",
        "b": "gaussian",
    }


def test_resolve_prior_dict_partial_defaults_none():
    out = resolve_dataset_prior_dict({"a": "horseshoe"}, ("a", "b"))
    assert {k: v.type for k, v in out.items()} == {
        "a": "horseshoe",
        "b": "none",
    }


def test_resolve_prior_unknown_key():
    with pytest.raises(ValueError, match="not a declared factor"):
        resolve_dataset_prior_dict({"zzz": "gaussian"}, ("a", "b"))


def test_resolve_prior_unknown_family():
    with pytest.raises(ValueError, match="Unknown prior family"):
        resolve_dataset_prior_dict("banana", ("a",))
    with pytest.raises(ValueError, match="Unknown prior family"):
        resolve_dataset_prior_dict({"a": "banana"}, ("a",))


def test_resolve_prior_bad_type():
    with pytest.raises(ValueError):
        resolve_dataset_prior_dict(5, ("a",))


# ------------------------------------------------------------------------------
# Declaration validation (via normalize_grouping)
# ------------------------------------------------------------------------------


def test_mutual_exclusivity(obs_crossed):
    with pytest.raises(ValueError, match="not both"):
        normalize_grouping(
            dataset_key="sample",
            hierarchy=[GroupLevel(name="sample")],
            interactions=None,
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


def test_no_grouping_returns_none(obs_crossed):
    out = normalize_grouping(
        dataset_key=None,
        hierarchy=None,
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    assert out is None


def test_missing_column(obs_crossed):
    with pytest.raises(ValueError, match="not found in adata.obs"):
        normalize_grouping(
            dataset_key=["sample", "nope"],
            hierarchy=None,
            interactions=None,
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


def test_duplicate_factor(obs_crossed):
    with pytest.raises(ValueError, match="Duplicate factor"):
        normalize_grouping(
            dataset_key=["sample", "sample"],
            hierarchy=None,
            interactions=None,
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


def test_nested_in_forward_ref(obs_crossed):
    with pytest.raises(ValueError, match="not a factor declared before"):
        normalize_grouping(
            dataset_key=None,
            hierarchy=[
                GroupLevel(name="sample", nested_in="treatment"),
                GroupLevel(name="treatment"),
            ],
            interactions=None,
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


def test_interactions_need_two_factors(obs_crossed):
    with pytest.raises(ValueError, match=">= 2 grouping factors"):
        normalize_grouping(
            dataset_key="sample",
            hierarchy=None,
            interactions=[("sample", "treatment")],
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


def test_interaction_unknown_operand(obs_crossed):
    with pytest.raises(ValueError, match="not a declared factor"):
        normalize_grouping(
            dataset_key=["sample", "treatment"],
            hierarchy=None,
            interactions=[("sample", "zzz")],
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )


# ------------------------------------------------------------------------------
# Single-factor normalization
# ------------------------------------------------------------------------------


def test_single_factor(obs_crossed):
    spec, leaf_index = normalize_grouping(
        dataset_key="sample",
        hierarchy=None,
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    assert isinstance(spec, GroupingSpec)
    assert spec.n_leaves == 3
    assert spec.factor_names == ("sample",)
    f = spec.factors[0]
    assert f.levels == ("D1", "D2", "D3")
    assert f.leaf_to_level == (0, 1, 2)
    assert spec.leaf_labels == ("D1", "D2", "D3")
    # Per-cell leaf index == categorical codes.
    np.testing.assert_array_equal(leaf_index, [0, 0, 0, 1, 1, 1, 2])
    assert leaf_index.dtype == np.int32


# ------------------------------------------------------------------------------
# Crossed factors: present-only leaves
# ------------------------------------------------------------------------------


def test_crossed_present_only(obs_crossed):
    spec, leaf_index = normalize_grouping(
        dataset_key=["treatment", "sample"],
        hierarchy=None,
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    # 5 present leaves out of 2 x 3 = 6 (missing D3/drug).
    assert spec.n_leaves == 5
    assert spec.factor_names == ("treatment", "sample")

    treatment, sample = spec.factors
    assert treatment.levels == ("control", "drug")
    assert sample.levels == ("D1", "D2", "D3")

    # numpy.unique lexicographic order over (treatment_code, sample_code):
    # (0,0)(0,1)(0,2)(1,0)(1,1)
    assert treatment.leaf_to_level == (0, 0, 0, 1, 1)
    assert sample.leaf_to_level == (0, 1, 2, 0, 1)
    assert spec.leaf_labels == (
        "control | D1",
        "control | D2",
        "control | D3",
        "drug | D1",
        "drug | D2",
    )

    # Per-cell leaf index for cells D1c,D1c,D1d,D2c,D2d,D2d,D3c
    np.testing.assert_array_equal(leaf_index, [0, 0, 3, 1, 4, 4, 2])

    # leaf_coords round-trips to the original combination.
    coords = spec.leaf_coords()
    assert coords[3] == {"treatment": "drug", "sample": "D1"}


def test_hierarchy_equivalent_to_list(obs_crossed):
    spec_list, idx_list = normalize_grouping(
        dataset_key=["treatment", "sample"],
        hierarchy=None,
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    spec_h, idx_h = normalize_grouping(
        dataset_key=None,
        hierarchy=[GroupLevel(name="treatment"), GroupLevel(name="sample")],
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    assert spec_list.leaf_labels == spec_h.leaf_labels
    np.testing.assert_array_equal(idx_list, idx_h)


# ------------------------------------------------------------------------------
# Interactions
# ------------------------------------------------------------------------------


def test_interaction_factor(obs_crossed):
    spec, _ = normalize_grouping(
        dataset_key=["treatment", "sample"],
        hierarchy=None,
        interactions=[("treatment", "sample")],
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    assert spec.factor_names == ("treatment", "sample", "treatment:sample")
    inter = spec.factors[2]
    assert inter.kind == "interaction"
    # Each present leaf is its own interaction level here (5 leaves).
    assert inter.n_levels == 5
    assert inter.leaf_to_level == (0, 1, 2, 3, 4)
    assert inter.levels[3] == "drug x D1"


# ------------------------------------------------------------------------------
# Prior families
# ------------------------------------------------------------------------------


def test_per_factor_priors_and_reduction(obs_crossed):
    priors = _no_priors()
    priors["expression"] = {"sample": "horseshoe", "treatment": "gaussian"}
    priors["prob"] = "gaussian"
    spec, _ = normalize_grouping(
        dataset_key=["treatment", "sample"],
        hierarchy=None,
        interactions=[("treatment", "sample")],
        obs=obs_crossed,
        dataset_priors=priors,
    )
    treatment, sample, inter = spec.factors
    _types = lambda fac: {t: s.type for t, s in fac.priors.items()}
    assert _types(treatment) == {"expression": "gaussian", "prob": "gaussian"}
    assert _types(sample) == {"expression": "horseshoe", "prob": "gaussian"}
    # Interaction absent from the expression dict -> none there, but prob broadcasts.
    assert _types(inter) == {"prob": "gaussian"}

    # Leaf-axis reduction: first non-none family in factor order.
    assert _reduce_leaf_axis_family(spec, "expression") == "gaussian"
    assert _reduce_leaf_axis_family(spec, "prob") == "gaussian"
    assert _reduce_leaf_axis_family(spec, "zero_inflation") == "none"


def test_grouping_spec_is_frozen(obs_crossed):
    spec, _ = normalize_grouping(
        dataset_key="sample",
        hierarchy=None,
        interactions=None,
        obs=obs_crossed,
        dataset_priors=_no_priors(),
    )
    with pytest.raises(Exception):
        spec.n_leaves = 99  # frozen


# ------------------------------------------------------------------------------
# Audit-flagged edge cases
# ------------------------------------------------------------------------------


def test_missing_grouping_value_rejected():
    # A missing value in a grouping factor would otherwise get code -1 and be
    # silently mislabeled as the last category, creating wrong/duplicate leaves.
    obs = pd.DataFrame(
        {
            "sample": ["D1", "D1", None, "D2"],
            "treatment": ["control", "drug", "drug", "control"],
        }
    )
    with pytest.raises(ValueError, match="missing/NaN"):
        normalize_grouping(
            dataset_key=["treatment", "sample"],
            hierarchy=None,
            interactions=None,
            obs=obs,
            dataset_priors=_no_priors(),
        )


def test_interactions_without_grouping_rejected(obs_crossed):
    with pytest.raises(ValueError, match="interactions require grouping"):
        normalize_grouping(
            dataset_key=None,
            hierarchy=None,
            interactions=[("sample", "treatment")],
            obs=obs_crossed,
            dataset_priors=_no_priors(),
        )
