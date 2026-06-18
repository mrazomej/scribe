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
    assert sample_factor.priors == {"expression": "horseshoe"}


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
