"""Tests for the FACTORS axis in the layout system (M2).

The additive multi-factor hierarchy exposes per-factor effect/raw sites whose
level axis (size L_f) is distinct from the leaf/dataset axis. These carry the
new ``FACTORS`` axis so the results/DE layers can slice them.
"""

import pytest

from scribe.core.axis_layout import (
    AxisLayout,
    COMPONENTS,
    DATASETS,
    FACTORS,
    GENES,
    layout_from_param_spec,
    subset_layouts,
)
from scribe.models.builders.parameter_specs import (
    GroupingFactorSpec,
    LayoutOnlySpec,
    MultiFactorPositiveNormalSpec,
)


def test_factor_axis_property():
    lay = AxisLayout((FACTORS, GENES))
    assert lay.factor_axis == 0
    assert lay.gene_axis == 1
    assert lay.dataset_axis is None


def test_factor_axis_with_components_and_sample_dim():
    lay = AxisLayout((COMPONENTS, FACTORS, GENES), has_sample_dim=True)
    assert lay.component_axis == 1
    assert lay.factor_axis == 2
    assert lay.gene_axis == 3


def test_layout_from_factor_spec():
    spec = LayoutOnlySpec(
        name="mu_eff", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        is_factor=True, is_gene_specific=True,
    )
    lay = layout_from_param_spec(spec)
    assert lay.axes == (FACTORS, GENES)


def test_layout_from_factor_spec_mixture():
    spec = LayoutOnlySpec(
        name="mu_eff", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        is_factor=True, is_mixture=True, is_gene_specific=True,
    )
    lay = layout_from_param_spec(spec)
    assert lay.axes == (COMPONENTS, FACTORS, GENES)


def test_leaf_spec_uses_dataset_not_factor():
    # The leaf parameter (mu) carries DATASETS, never FACTORS.
    f = GroupingFactorSpec(
        name="sample", n_levels=2, leaf_to_level=(0, 1), prior="gaussian",
        scale_name="s", raw_name="mu_raw", effect_name="mu_eff",
    )
    mu = MultiFactorPositiveNormalSpec(
        name="mu", shape_dims=("n_genes",), default_params=(0.0, 1.0),
        hyper_loc_name="loc", is_gene_specific=True, is_dataset=True,
        factors=(f,),
    )
    lay = layout_from_param_spec(mu)
    assert lay.axes == (DATASETS, GENES)
    assert lay.factor_axis is None

    # Its companion specs carry FACTORS.
    comp_layouts = {c.name: layout_from_param_spec(c) for c in mu.companion_specs}
    assert comp_layouts["mu_eff"].axes == (FACTORS, GENES)
    assert comp_layouts["mu_raw"].axes == (FACTORS, GENES)


def test_subset_layouts_drops_factors_only_from_factor_sites():
    layouts = {
        "mu": AxisLayout((DATASETS, GENES)),       # leaf
        "mu_eff": AxisLayout((FACTORS, GENES)),    # factor effect
    }
    out = subset_layouts(layouts, FACTORS)
    # Leaf untouched (no FACTORS axis); factor effect loses FACTORS.
    assert out["mu"].axes == (DATASETS, GENES)
    assert out["mu_eff"].axes == (GENES,)
