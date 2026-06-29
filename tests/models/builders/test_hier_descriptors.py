"""Unit tests for the hierarchical-prior descriptors.

These pin down the exact site names, spec classes, and target-resolution rules
that the descriptors must reproduce so the generic NCP cores
(``_horseshoe_ncp`` / ``_neg_ncp``) stay byte-identical to the per-parameter
builders they replaced. They guard the traps called out in the dedup plan:

* ``raw`` site naming is non-uniform across levels (dataset ``p``/``gate``/regime
  use ``{t}_raw_dataset``; everything else uses ``{t}_raw``).
* the gene-level expression rule is *narrower* than the dataset-level rule —
  they diverge on the two-state family.
* the descriptor's ``expression_target_is_mu`` must agree with the factory's.
* regime ``is_sigmoid`` is per-coordinate (only ``inv_concentration``).
"""

import pytest

from scribe.models.builders import parameter_specs as ps
from scribe.models.builders.hier_descriptors import (
    HierParam,
    dataset_hier_param,
    expression_target_is_mu,
    gene_hier_param,
    regime_dataset_hier_param,
)
from scribe.models.config.enums import Parameterization as P


# ------------------------------------------------------------------------------
# expression_target_is_mu parity with the factory's private copy
# ------------------------------------------------------------------------------


def test_expression_target_is_mu_matches_factory():
    """The copied target rule must agree with factory._expression_target_is_mu."""
    from scribe.models.presets.factory import _expression_target_is_mu

    keys = [
        "canonical",
        "mean_prob",
        "mean_odds",
        "mean_disp",
        "two_state_natural",
        "two_state_ratio",
        "two_state_mean_fano",
        "two_state_moment_delta",
        "logistic_normal_canonical",
        "logistic_normal_mean_odds",
    ]
    for k in keys:
        assert expression_target_is_mu(k) == _expression_target_is_mu(k), k


# ------------------------------------------------------------------------------
# Gene-level descriptors
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param_key,target,loc,scale",
    [
        ("canonical", "r", "log_r_loc", "log_r_scale"),
        ("mean_prob", "mu", "log_mu_loc", "log_mu_scale"),
        ("mean_odds", "mu", "log_mu_loc", "log_mu_scale"),
        ("mean_disp", "mu", "log_mu_loc", "log_mu_scale"),
    ],
)
def test_gene_expression(param_key, target, loc, scale):
    d = gene_hier_param("expression", param_key)
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        target, loc, scale, target, f"{target}_raw",
    )
    assert d.is_sigmoid is False
    assert d.hier_cls is ps.HierarchicalPositiveNormalSpec
    assert d.hs_cls is ps.HorseshoeHierarchicalPositiveNormalSpec
    assert d.neg_cls is ps.NEGHierarchicalPositiveNormalSpec


def test_gene_expression_two_state_uses_narrow_rule():
    """Gene-level expression targets ``r`` for two-state (narrow rule)."""
    d = gene_hier_param("expression", "two_state_natural")
    assert d.target == "r"  # NOT mu — gene rule excludes two_state


def test_gene_prob_phi_positive():
    d = gene_hier_param("prob", "mean_odds")
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "phi", "log_phi_loc", "log_phi_scale", "phi", "phi_raw",
    )
    assert d.is_sigmoid is False
    assert d.hier_cls is ps.HierarchicalPositiveNormalSpec


@pytest.mark.parametrize("param_key", ["canonical", "mean_prob"])
def test_gene_prob_p_sigmoid(param_key):
    d = gene_hier_param("prob", param_key)
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "p", "logit_p_loc", "logit_p_scale", "p", "p_raw",
    )
    assert d.is_sigmoid is True
    assert d.hier_cls is ps.HierarchicalSigmoidNormalSpec
    assert d.hs_cls is ps.HorseshoeHierarchicalSigmoidNormalSpec
    assert d.neg_cls is ps.NEGHierarchicalSigmoidNormalSpec


def test_gene_gate_sigmoid():
    d = gene_hier_param("gate", "canonical")
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "gate", "logit_gate_loc", "logit_gate_scale", "gate", "gate_raw",
    )
    assert d.is_sigmoid is True
    assert d.hier_cls is ps.HierarchicalSigmoidNormalSpec


# ------------------------------------------------------------------------------
# Dataset-level descriptors
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param_key,target",
    [("canonical", "r"), ("mean_prob", "mu"), ("mean_odds", "mu"),
     ("mean_disp", "mu"), ("two_state_natural", "mu")],
)
def test_dataset_expression_target_and_raw(param_key, target):
    d = dataset_hier_param("expression", param_key)
    assert d.target == target
    assert d.prefix == f"{target}_dataset"
    assert d.loc == f"log_{target}_dataset_loc"
    assert d.scale == f"log_{target}_dataset_scale"
    # mu/r dataset raw is {t}_raw, NOT {t}_raw_dataset (the trap).
    assert d.raw == f"{target}_raw"
    assert d.is_sigmoid is False
    assert d.hier_cls is ps.DatasetHierarchicalPositiveNormalSpec
    assert d.hs_cls is ps.HorseshoeDatasetPositiveNormalSpec
    assert d.neg_cls is ps.NEGDatasetPositiveNormalSpec


def test_dataset_prob_phi_positive():
    d = dataset_hier_param("prob", "mean_odds")
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "phi", "log_phi_dataset_loc", "log_phi_dataset_scale",
        "phi_dataset", "phi_raw_dataset",
    )
    assert d.is_sigmoid is False
    assert d.hier_cls is ps.DatasetHierarchicalPositiveNormalSpec


@pytest.mark.parametrize("param_key", ["canonical", "mean_prob"])
def test_dataset_prob_p_sigmoid(param_key):
    d = dataset_hier_param("prob", param_key)
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "p", "logit_p_dataset_loc", "logit_p_dataset_scale",
        "p_dataset", "p_raw_dataset",  # p dataset raw IS _dataset.
    )
    assert d.is_sigmoid is True
    assert d.hier_cls is ps.DatasetHierarchicalSigmoidNormalSpec


def test_dataset_gate_sigmoid():
    d = dataset_hier_param("gate", "canonical")
    assert (d.target, d.loc, d.scale, d.prefix, d.raw) == (
        "gate", "logit_gate_dataset_loc", "logit_gate_dataset_scale",
        "gate_dataset", "gate_raw_dataset",
    )
    assert d.is_sigmoid is True
    assert d.hier_cls is ps.DatasetHierarchicalSigmoidNormalSpec


def test_gene_vs_dataset_expression_diverge_on_two_state():
    """The key trap: gene and dataset expression targets differ for two-state."""
    assert gene_hier_param("expression", "two_state_natural").target == "r"
    assert dataset_hier_param("expression", "two_state_natural").target == "mu"


def test_bad_role_raises():
    with pytest.raises(ValueError, match="unknown hierarchy role"):
        gene_hier_param("nonsense", "canonical")
    with pytest.raises(ValueError, match="unknown hierarchy role"):
        dataset_hier_param("nonsense", "canonical")


# ------------------------------------------------------------------------------
# Regime descriptors (dynamic coordinate, per-coordinate is_sigmoid)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param,coord,is_sigmoid,link",
    [
        (P.TWO_STATE_NATURAL, "k_off", False, "log"),
        (P.TWO_STATE_RATIO, "switching_ratio", False, "log"),
        (P.TWO_STATE_MEAN_FANO, "concentration", False, "log"),
        (P.TWO_STATE_MOMENT_DELTA, "inv_concentration", True, "logit"),
    ],
)
def test_regime_descriptor(param, coord, is_sigmoid, link):
    d = regime_dataset_hier_param(param)
    assert isinstance(d, HierParam)
    assert d.target == coord
    assert d.is_sigmoid is is_sigmoid
    assert d.loc == f"{link}_{coord}_dataset_loc"
    assert d.scale == f"{link}_{coord}_dataset_scale"
    assert d.prefix == f"{coord}_dataset"
    assert d.raw == f"{coord}_raw_dataset"
    expected_hier = (
        ps.DatasetHierarchicalSigmoidNormalSpec
        if is_sigmoid
        else ps.DatasetHierarchicalPositiveNormalSpec
    )
    assert d.hier_cls is expected_hier


def test_regime_target_override():
    d = regime_dataset_hier_param(P.TWO_STATE_NATURAL, target_override="k_off")
    assert d.target == "k_off"


def test_regime_none_for_non_two_state():
    """A parameterization with no regime coordinate yields None (no-op)."""
    assert regime_dataset_hier_param(P.CANONICAL) is None
