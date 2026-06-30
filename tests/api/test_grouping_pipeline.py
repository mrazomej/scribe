"""Stage-level tests for multi-factor grouping in the fit pipeline (M1).

Exercises ``process_data_and_datasets`` directly (no inference): leaf
construction, the preserved legacy single-``str`` path (bit-identical, incl.
unused categories), per-factor prior reduction, label propagation onto
``ctx.grouping_spec``, and the single-leaf downgrade.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scribe.api.context import FitContext
from scribe.api.stages.data_processing import process_data_and_datasets


def _make_adata(sample, treatment=None, n_genes=4, seed=0):
    n = len(sample)
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 10, size=(n, n_genes)).astype(float)
    obs = {"sample": sample}
    if treatment is not None:
        obs["treatment"] = treatment
    return ad.AnnData(X=X, obs=pd.DataFrame(obs))


def _run(adata, **kw):
    ctx = FitContext(counts=adata, kwargs=dict(kw))
    process_data_and_datasets(ctx)
    return ctx


# ------------------------------------------------------------------------------
# Legacy single-string path: must stay bit-identical
# ------------------------------------------------------------------------------


def test_legacy_str_bit_identical():
    sample = ["D1", "D1", "D2", "D2", "D3", "D3"]
    adata = _make_adata(sample, treatment=["control"] * 6)
    ctx = _run(adata, dataset_key="sample")

    codes = adata.obs["sample"].astype("category").cat.codes.values
    np.testing.assert_array_equal(np.asarray(ctx.dataset_indices), codes)
    assert ctx.n_datasets == 3
    assert ctx.grouping_spec.n_leaves == 3
    assert ctx.grouping_spec.leaf_labels == ("D1", "D2", "D3")


def test_legacy_str_unused_category_preserved():
    # An unused category still counts (legacy len(categories) semantics),
    # which the present-only multi-factor path would NOT reproduce.
    cat = pd.Categorical(["D1", "D1", "D2", "D2"], categories=["D1", "D2", "D3"])
    adata = ad.AnnData(X=np.ones((4, 3)), obs=pd.DataFrame({"sample": cat}))
    ctx = _run(adata, dataset_key="sample")

    assert ctx.n_datasets == 3  # includes the unused D3
    assert ctx.grouping_spec.n_leaves == 3
    codes = adata.obs["sample"].astype("category").cat.codes.values
    np.testing.assert_array_equal(np.asarray(ctx.dataset_indices), codes)


# ------------------------------------------------------------------------------
# Multi-factor path
# ------------------------------------------------------------------------------


def test_multifactor_crossed_present_only():
    sample = ["D1", "D1", "D2", "D2", "D3"]
    treatment = ["control", "drug", "control", "drug", "control"]
    adata = _make_adata(sample, treatment)
    ctx = _run(
        adata,
        dataset_key=["treatment", "sample"],
        expression_dataset_prior="gaussian",
    )
    assert ctx.n_datasets == 5  # present-only (missing D3/drug)
    assert ctx.grouping_spec.factor_names == ("treatment", "sample")
    np.testing.assert_array_equal(
        np.asarray(ctx.dataset_indices), [0, 3, 1, 4, 2]
    )
    # The reduced leaf-axis family is written back to kwargs as a plain string.
    assert ctx.kwargs["expression_dataset_prior"] == "gaussian"


def test_multifactor_prior_dict_reduction():
    sample = ["D1", "D1", "D2", "D2"]
    treatment = ["control", "drug", "control", "drug"]
    adata = _make_adata(sample, treatment)
    ctx = _run(
        adata,
        dataset_key=["treatment", "sample"],
        expression_dataset_prior={"sample": "horseshoe", "treatment": "gaussian"},
    )
    # Leaf-axis reduction picks the first non-"none" family in factor order.
    assert ctx.kwargs["expression_dataset_prior"] == "gaussian"
    sample_factor = next(
        f for f in ctx.grouping_spec.factors if f.name == "sample"
    )
    assert {t: s.type for t, s in sample_factor.priors.items()} == {
        "expression": "horseshoe"
    }


def test_multifactor_hierarchy_and_interaction():
    sample = ["D1", "D1", "D2", "D2"]
    treatment = ["control", "drug", "control", "drug"]
    adata = _make_adata(sample, treatment)
    from scribe import GroupLevel

    ctx = _run(
        adata,
        hierarchy=[GroupLevel(name="treatment"), GroupLevel(name="sample")],
        interactions=[("treatment", "sample")],
        prob_dataset_prior="gaussian",
    )
    assert ctx.grouping_spec.factor_names == (
        "treatment",
        "sample",
        "treatment:sample",
    )
    assert ctx.kwargs["prob_dataset_prior"] == "gaussian"


# ------------------------------------------------------------------------------
# Single-leaf downgrade
# ------------------------------------------------------------------------------


def test_single_leaf_downgrade():
    adata = _make_adata(["D1", "D1", "D1"], treatment=["control"] * 3)
    ctx = _run(adata, dataset_key="sample", expression_dataset_prior="gaussian")
    assert ctx.n_datasets is None
    assert ctx.grouping_spec is None
    assert ctx.kwargs["expression_dataset_prior"] == "none"


def test_grouping_requires_anndata():
    with pytest.raises(ValueError, match="AnnData"):
        ctx = FitContext(
            counts=np.ones((4, 3)),
            kwargs={"dataset_key": ["treatment", "sample"]},
        )
        process_data_and_datasets(ctx)


# ------------------------------------------------------------------------------
# Unified `priors` -> internal plumbing routing
# ------------------------------------------------------------------------------


def test_unified_priors_route_into_flat_plumbing():
    sample = ["D1", "D1", "D2", "D2"]
    treatment = ["control", "drug", "control", "drug"]
    adata = _make_adata(sample, treatment)
    ctx = FitContext(
        counts=adata,
        kwargs=dict(dataset_key=["treatment", "sample"]),
        priors={
            # dataset/factor hierarchy on the mean
            "mean_expression": {"treatment": "gaussian", "sample": "horseshoe"},
            # gene-level family selector
            "probability": "horseshoe",
            # base hyperparameter override (kept on ctx.priors)
            "dispersion": (2.0, 0.5),
        },
    )
    process_data_and_datasets(ctx)

    # Gene-level selector -> the *_prior field.
    assert ctx.kwargs["prob_prior"] == "horseshoe"
    # Dataset/factor hierarchy -> grouping_spec per-factor families.
    by_name = {f.name: f for f in ctx.grouping_spec.factors}
    assert by_name["treatment"].family("expression") == "gaussian"
    assert by_name["sample"].family("expression") == "horseshoe"
    # Base hyperparameters survive on ctx.priors (original key; with_priors
    # resolves it downstream).
    assert ctx.priors == {"dispersion": (2.0, 0.5)}


def test_unified_priors_interaction_slope_routes_into_grouping():
    # Regression guard for the random-slope path: the unified `priors` dict must
    # accept an INTERACTION key (the ":"-joined operands) and route its family
    # onto the interaction factor. Before interaction names were registered as
    # declared levels, this raised "not a declared grouping level", so a random
    # slope on the mean (perturbation:sample) was unreachable via the public API.
    from scribe import GroupLevel

    sample = ["D1", "D1", "D2", "D2"]
    treatment = ["control", "drug", "control", "drug"]
    adata = _make_adata(sample, treatment)
    ctx = FitContext(
        counts=adata,
        kwargs=dict(
            hierarchy=[
                GroupLevel(name="treatment", effect_type="fixed"),
                GroupLevel(name="sample"),
            ],
            interactions=[("treatment", "sample")],
        ),
        priors={
            "mean_expression": {
                "treatment": "gaussian",
                "sample": "horseshoe",
                "treatment:sample": "horseshoe",  # per-donor random slope
            },
        },
    )
    process_data_and_datasets(ctx)

    by_name = {f.name: f for f in ctx.grouping_spec.factors}
    # The interaction factor exists and carries the slope's mean-expression
    # family (so the factory builds the horseshoe-shrunk mu interaction effect).
    assert "treatment:sample" in by_name
    inter = by_name["treatment:sample"]
    assert inter.kind == "interaction"
    assert inter.family("expression") == "horseshoe"
    # Base factors still resolve unchanged (no regression).
    assert by_name["treatment"].family("expression") == "gaussian"
    assert by_name["sample"].family("expression") == "horseshoe"


def test_unified_priors_dispersion_routes_into_grouping():
    sample = ["D1", "D1", "D2", "D2"]
    treatment = ["control", "drug", "control", "drug"]
    adata = _make_adata(sample, treatment)
    ctx = FitContext(
        counts=adata,
        kwargs=dict(dataset_key=["treatment", "sample"]),
        priors={"dispersion": {"treatment": "gaussian"}},
    )
    process_data_and_datasets(ctx)
    by_name = {f.name: f for f in ctx.grouping_spec.factors}
    # Condition factor carries the dispersion family; others do not.
    assert by_name["treatment"].family("dispersion") == "gaussian"
    assert by_name["sample"].family("dispersion") == "none"


def test_unified_priors_capture_scaling_routes_to_field():
    """The mu_eta hierarchy FAMILY routes from priors[capture_scaling] to the
    internal ``capture_scaling_prior`` kwarg; the (center, sigma_mu) tuple stays
    a base hyperparameter. Replaces the removed ``capture_scaling_prior`` kwarg.
    """
    adata = _make_adata(["D1", "D1", "D2", "D2"])

    # (a) bare family string -> capture_scaling_prior; entry stripped, the
    #     anchor (eta_capture) stays a base hyperparameter.
    ctx = FitContext(
        counts=adata,
        kwargs={},
        priors={"eta_capture": (12.0, 1.0), "capture_scaling": "horseshoe"},
    )
    process_data_and_datasets(ctx)
    assert ctx.kwargs["capture_scaling_prior"] == "horseshoe"
    assert "capture_scaling" not in (ctx.priors or {})
    assert ctx.priors["eta_capture"] == (12.0, 1.0)

    # (b) spec dict -> family + (center, sigma_mu) tuple under the key.
    ctx = FitContext(
        counts=adata,
        kwargs={},
        priors={
            "capture_scaling": {"type": "neg", "center": 12.0, "sigma_mu": 0.5}
        },
    )
    process_data_and_datasets(ctx)
    assert ctx.kwargs["capture_scaling_prior"] == "neg"
    assert ctx.priors["capture_scaling"] == (12.0, 0.5)

    # (c) bare tuple -> base hyperparameter only, no family (existing behavior).
    ctx = FitContext(
        counts=adata, kwargs={}, priors={"capture_scaling": (12.0, 1.0)}
    )
    process_data_and_datasets(ctx)
    assert ctx.kwargs.get("capture_scaling_prior", "none") == "none"
    assert ctx.priors["capture_scaling"] == (12.0, 1.0)

    # (d) the mu_eta alias works identically to capture_scaling.
    ctx = FitContext(
        counts=adata, kwargs={}, priors={"mu_eta": "gaussian"}
    )
    process_data_and_datasets(ctx)
    assert ctx.kwargs["capture_scaling_prior"] == "gaussian"

    # (e) a partial spec dict (center without sigma_mu) is a clear error.
    ctx = FitContext(
        counts=adata,
        kwargs={},
        priors={"capture_scaling": {"type": "gaussian", "center": 12.0}},
    )
    with pytest.raises(ValueError, match="both 'center' and 'sigma_mu'"):
        process_data_and_datasets(ctx)
