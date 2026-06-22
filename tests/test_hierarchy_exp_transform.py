"""The additive expression hierarchy forces the exp link on the expression mean.

A cross-dataset / multi-factor expression hierarchy decomposes the unconstrained
mean accumulator additively and maps it to a positive mean via
``positive_transform``. Only ``exp`` makes that additive structure log-mean (so
the per-factor effects are log-fold-changes); softplus would silently place the
effects in softplus-inverse space. ``_force_exp_for_expression_hierarchy``
enforces exp (warning on override) — these tests pin that behaviour.
"""

import warnings

import pytest

from scribe.api.stages.model_config_build import (
    _force_exp_for_expression_hierarchy,
)
from scribe.inference.preset_builder import build_config_from_preset


def _cfg(**kw):
    base = dict(model="nbvcp", parameterization="mean_odds", unconstrained=True)
    base.update(kw)
    return build_config_from_preset(**base)


def test_expression_hierarchy_forces_exp_and_warns():
    mc = _cfg(
        positive_transform="softplus",
        n_datasets=2,
        expression_dataset_prior="gaussian",
        prob_dataset_prior="gaussian",
    )
    assert mc.resolve_positive_transform("mu") == "softplus"  # default
    with pytest.warns(UserWarning, match="log-additive"):
        forced = _force_exp_for_expression_hierarchy(mc)
    assert forced.resolve_positive_transform("mu") == "exp"


def test_explicit_exp_not_warned():
    mc = _cfg(
        positive_transform={"mean_expression": "exp"},
        n_datasets=2,
        expression_dataset_prior="gaussian",
        prob_dataset_prior="gaussian",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forced = _force_exp_for_expression_hierarchy(mc)
    assert forced.resolve_positive_transform("mu") == "exp"
    assert not any("log-additive" in str(w.message) for w in caught)


def test_no_expression_hierarchy_leaves_softplus():
    # No dataset hierarchy on the expression mean -> default link untouched.
    mc = _cfg(positive_transform="softplus", prob_prior="gaussian")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forced = _force_exp_for_expression_hierarchy(mc)
    assert forced.resolve_positive_transform("mu") == "softplus"
    assert not any("log-additive" in str(w.message) for w in caught)


def test_canonical_targets_r():
    # Under canonical, the expression mean is r, not mu.
    mc = _cfg(
        parameterization="canonical",
        positive_transform="softplus",
        n_datasets=2,
        expression_dataset_prior="gaussian",
        prob_dataset_prior="gaussian",
    )
    with pytest.warns(UserWarning, match="log-additive"):
        forced = _force_exp_for_expression_hierarchy(mc)
    assert forced.resolve_positive_transform("r") == "exp"
