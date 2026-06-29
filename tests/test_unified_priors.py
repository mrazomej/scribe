"""Golden tests for the unified ``priors`` routing-contract parser.

Exercises every valid value form of :func:`normalize_unified_priors` and its
rejections, independently of the factory/fit plumbing (Commit 2b).
"""

import pytest

from scribe.models.config.grouping import (
    normalize_unified_priors,
    PriorFamilySpec,
)

LEVELS = ("perturbation", "sample")


# ------------------------------------------------------------------------------
# Routing contract: each value form -> the right bucket
# ------------------------------------------------------------------------------


def test_empty_priors_yields_empty_buckets():
    base, gene, hier = normalize_unified_priors(None, LEVELS)
    assert base == {} and gene == {} and hier == {}
    base, gene, hier = normalize_unified_priors({}, LEVELS)
    assert base == {} and gene == {} and hier == {}


def test_tuple_is_base_hyperparameter_override():
    base, gene, hier = normalize_unified_priors(
        {"dispersion": (1.0, 1.0)}, LEVELS
    )
    # Base entries keep their ORIGINAL key; with_priors resolves them later.
    assert base == {"dispersion": (1.0, 1.0)}
    assert gene == {} and hier == {}


def test_loadings_strategy_dict_is_base():
    spec = {"type": "horseshoe_columnwise", "tau0": 0.5}
    base, gene, hier = normalize_unified_priors({"loadings": spec}, LEVELS)
    # W-strategy specs are stored raw (their 'type' vocabulary differs from the
    # hierarchical family vocabulary) and never coerced to PriorFamilySpec.
    assert base == {"loadings": spec}
    assert gene == {} and hier == {}


def test_unknown_base_key_passes_through():
    # Raw hyperprior-override keys / unrecognized base specs pass through to
    # base unchanged (validated downstream by with_priors), NOT rejected.
    base, gene, hier = normalize_unified_priors(
        {"logit_p_loc": (0.0, 1.0)}, LEVELS
    )
    assert base == {"logit_p_loc": (0.0, 1.0)}
    assert gene == {} and hier == {}


def test_bare_family_string_is_gene_level_selector():
    base, gene, hier = normalize_unified_priors(
        {"probability": "horseshoe"}, LEVELS
    )
    assert base == {} and hier == {}
    assert gene == {"p": PriorFamilySpec(type="horseshoe")}


def test_family_spec_dict_is_gene_level_with_hyperparams():
    base, gene, hier = normalize_unified_priors(
        {"probability": {"type": "horseshoe", "tau0": 1.0, "mode": "gene_specific"}},
        LEVELS,
    )
    assert gene == {
        "p": PriorFamilySpec(type="horseshoe", tau0=1.0, mode="gene_specific")
    }
    assert base == {} and hier == {}


def test_level_mapping_is_dataset_factor_hierarchy():
    # The v2 feature: condition-specific dispersion r.
    base, gene, hier = normalize_unified_priors(
        {"dispersion": {"perturbation": "gaussian"}}, LEVELS
    )
    assert hier == {"dispersion": {"perturbation": PriorFamilySpec(type="gaussian")}}
    assert base == {} and gene == {}


def test_base_key_in_level_mapping_routes_to_gene_level():
    base, gene, hier = normalize_unified_priors(
        {
            "mean_expression": {
                "base": "gaussian",
                "perturbation": "gaussian",
                "sample": "horseshoe",
            }
        },
        LEVELS,
    )
    assert gene == {"mu": PriorFamilySpec(type="gaussian")}
    assert hier == {
        "expression": {
            "perturbation": PriorFamilySpec(type="gaussian"),
            "sample": PriorFamilySpec(type="horseshoe"),
        }
    }
    assert base == {}


def test_internal_site_names_accepted():
    # Power users may pass raw site names.
    _, gene, hier = normalize_unified_priors(
        {"r": "gaussian", "mu": {"perturbation": "horseshoe"}}, LEVELS
    )
    assert gene == {"r": PriorFamilySpec(type="gaussian")}
    assert hier == {"expression": {"perturbation": PriorFamilySpec(type="horseshoe")}}


def test_canonical_names_resolve_to_sites():
    _, gene, _ = normalize_unified_priors(
        {
            "mean_expression": "gaussian",
            "dispersion": "gaussian",
            "odds_ratio": "gaussian",
            "zero_inflation": "gaussian",
            "overdispersion": "gaussian",
        },
        LEVELS,
    )
    assert set(gene) == {"mu", "r", "phi", "gate", "bnb_concentration"}


# ------------------------------------------------------------------------------
# Rejections
# ------------------------------------------------------------------------------


def test_unknown_target_name_rejected():
    with pytest.raises(ValueError, match="Unknown prior target"):
        normalize_unified_priors({"banana": "gaussian"}, LEVELS)


def test_unknown_level_rejected():
    with pytest.raises(ValueError, match="not a declared grouping level"):
        normalize_unified_priors({"dispersion": {"nonsense": "gaussian"}}, LEVELS)


def test_non_family_value_passes_through_to_base():
    # A non-tuple/str/dict value is not a family or hierarchy; it passes
    # through to base with its original key (downstream validates it).
    base, gene, hier = normalize_unified_priors({"dispersion": 5}, LEVELS)
    assert base == {"dispersion": 5}
    assert gene == {} and hier == {}


def test_invalid_family_rejected():
    with pytest.raises(ValueError, match="Unknown prior family"):
        normalize_unified_priors({"dispersion": "banana"}, LEVELS)
