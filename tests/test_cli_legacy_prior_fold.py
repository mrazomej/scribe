"""Regression tests for the CLI legacy-prior fold.

``scribe.fit()`` accepts prior families/hierarchies ONLY through the unified
``priors`` dict. The CLI still accepts the legacy flat config keys
(``expression_prior``, ``*_dataset_prior``, ``horseshoe_*`` / ``neg_*``, ...)
as a backward-compatible YAML surface and folds them into ``priors`` via
``_fold_legacy_prior_cfg`` before calling ``fit``. These tests pin that
translation (see also ``docs/guide/priors.md``).
"""

from scribe.cli.infer_runner import _fold_legacy_prior_cfg


def _fold(cfg, dataset_key=None):
    return _fold_legacy_prior_cfg(cfg, cfg.get("priors"), dataset_key)


def test_gene_level_selectors_become_bare_families():
    cfg = {"prob_prior": "gaussian", "expression_prior": "neg"}
    assert _fold(cfg) == {"probability": "gaussian", "mean_expression": "neg"}


def test_dataset_bare_family_keys_on_dataset_key_factor():
    cfg = {"expression_dataset_prior": "horseshoe"}
    assert _fold(cfg, dataset_key="batch") == {
        "mean_expression": {"batch": "horseshoe"}
    }


def test_dataset_dict_passes_through():
    cfg = {
        "expression_dataset_prior": {
            "perturbation": "gaussian",
            "sample": "horseshoe",
        }
    }
    assert _fold(cfg, dataset_key=["perturbation", "sample"]) == {
        "mean_expression": {"perturbation": "gaussian", "sample": "horseshoe"}
    }


def test_regime_dataset_prior_routes_to_regime_key():
    cfg = {
        "expression_dataset_prior": "horseshoe",
        "regime_dataset_prior": "horseshoe",
    }
    assert _fold(cfg, dataset_key="condition") == {
        "mean_expression": {"condition": "horseshoe"},
        "regime": {"condition": "horseshoe"},
    }


def test_horseshoe_hyperparameters_fold_into_family_spec():
    cfg = {
        "prob_prior": "horseshoe",
        "horseshoe_tau0": 0.5,
        "horseshoe_slab_df": 6,
    }
    assert _fold(cfg) == {
        "probability": {"type": "horseshoe", "tau0": 0.5, "slab_df": 6}
    }


def test_neg_hyperparameters_fold_into_family_spec():
    cfg = {"zero_inflation_prior": "neg", "neg_u": 2.0, "neg_tau": 0.3}
    assert _fold(cfg) == {
        "zero_inflation": {"type": "neg", "u": 2.0, "tau": 0.3}
    }


def test_overdispersion_gene_prior_only_when_overdispersion_enabled():
    # Default overdispersion="none": the gene-level prior is not folded.
    assert _fold({"overdispersion": "none", "overdispersion_prior": "horseshoe"}) is None
    # Enabled: the family is folded under the canonical name.
    assert _fold({"overdispersion": "bnb", "overdispersion_prior": "neg"}) == {
        "overdispersion": "neg"
    }


def test_explicit_priors_block_is_not_overwritten():
    cfg = {"priors": {"probability": "neg"}, "prob_prior": "gaussian"}
    assert _fold(cfg) == {"probability": "neg"}


def test_no_legacy_keys_returns_priors_unchanged():
    assert _fold({}) is None
    assert _fold({"priors": {"mean_expression": (0.0, 1.0)}}) == {
        "mean_expression": (0.0, 1.0)
    }


def test_dataset_bare_family_without_dataset_key_is_dropped():
    # No declared factor -> nothing to key on; the dataset_key validation guard
    # raises a clear error before fit() instead.
    assert _fold({"expression_dataset_prior": "horseshoe"}, dataset_key=None) is None
