"""
Unit tests for the AxisLayout metadata system.

Tests the core abstraction in isolation: AxisLayout dataclass, factory
functions (``layout_from_param_spec``, ``infer_layout``), broadcasting
helpers, and the bulk builders.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from scribe.core.axis_layout import (
    AxisLayout,
    layout_from_param_spec,
    infer_layout,
    build_param_layouts,
    build_sample_layouts,
    gene_axes_from_layouts,
    reconstruct_param_layouts,
    align_to_layout,
    merge_layouts,
    broadcast_param_to_layout,
    _strip_param_key,
    GENES,
    COMPONENTS,
    DATASETS,
    CELLS,
)
from scribe.models.builders.parameter_specs import (
    BetaSpec,
    LogNormalSpec,
    DirichletSpec,
)


# =========================================================================
# AxisLayout construction and property tests
# =========================================================================


class TestAxisLayoutProperties:
    """Test axis-index properties for various layout configurations."""

    def test_scalar_layout(self):
        layout = AxisLayout(axes=())
        assert layout.gene_axis is None
        assert layout.component_axis is None
        assert layout.dataset_axis is None
        assert layout.cell_axis is None
        assert layout.rank == 0

    def test_gene_only(self):
        layout = AxisLayout(axes=(GENES,))
        assert layout.gene_axis == 0
        assert layout.component_axis is None
        assert layout.dataset_axis is None
        assert layout.rank == 1

    def test_component_gene(self):
        layout = AxisLayout(axes=(COMPONENTS, GENES))
        assert layout.component_axis == 0
        assert layout.gene_axis == 1
        assert layout.dataset_axis is None
        assert layout.rank == 2

    def test_component_dataset_gene(self):
        layout = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        assert layout.component_axis == 0
        assert layout.dataset_axis == 1
        assert layout.gene_axis == 2
        assert layout.rank == 3

    def test_cell_only(self):
        layout = AxisLayout(axes=(CELLS,))
        assert layout.cell_axis == 0
        assert layout.gene_axis is None
        assert layout.rank == 1

    def test_component_only(self):
        layout = AxisLayout(axes=(COMPONENTS,))
        assert layout.component_axis == 0
        assert layout.gene_axis is None
        assert layout.rank == 1

    def test_dataset_gene(self):
        layout = AxisLayout(axes=(DATASETS, GENES))
        assert layout.dataset_axis == 0
        assert layout.gene_axis == 1
        assert layout.rank == 2

    def test_absent_axes_return_none(self):
        layout = AxisLayout(axes=(GENES,))
        assert layout.component_axis is None
        assert layout.dataset_axis is None
        assert layout.cell_axis is None


class TestAxisLayoutSampleDim:
    """Test that has_sample_dim offsets all axis indices by 1."""

    def test_gene_with_sample_dim(self):
        layout = AxisLayout(axes=(GENES,), has_sample_dim=True)
        assert layout.gene_axis == 1
        assert layout.rank == 2

    def test_component_gene_with_sample_dim(self):
        layout = AxisLayout(axes=(COMPONENTS, GENES), has_sample_dim=True)
        assert layout.component_axis == 1
        assert layout.gene_axis == 2
        assert layout.rank == 3

    def test_full_with_sample_dim(self):
        layout = AxisLayout(
            axes=(COMPONENTS, DATASETS, GENES), has_sample_dim=True
        )
        assert layout.component_axis == 1
        assert layout.dataset_axis == 2
        assert layout.gene_axis == 3
        assert layout.rank == 4

    def test_scalar_with_sample_dim(self):
        layout = AxisLayout(axes=(), has_sample_dim=True)
        assert layout.gene_axis is None
        assert layout.rank == 1


class TestAxisLayoutTransformations:
    """Test with_sample_dim, without_sample_dim, and subset_axis."""

    def test_with_sample_dim(self):
        base = AxisLayout(axes=(COMPONENTS, GENES))
        with_s = base.with_sample_dim()
        assert with_s.has_sample_dim is True
        assert with_s.axes == (COMPONENTS, GENES)
        assert with_s.gene_axis == 2

    def test_with_sample_dim_idempotent(self):
        base = AxisLayout(axes=(GENES,), has_sample_dim=True)
        assert base.with_sample_dim() is base

    def test_without_sample_dim(self):
        base = AxisLayout(axes=(GENES,), has_sample_dim=True)
        without = base.without_sample_dim()
        assert without.has_sample_dim is False
        assert without.gene_axis == 0

    def test_without_sample_dim_idempotent(self):
        base = AxisLayout(axes=(GENES,))
        assert base.without_sample_dim() is base

    def test_subset_axis_removes_datasets(self):
        layout = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        sub = layout.subset_axis(DATASETS)
        assert sub.axes == (COMPONENTS, GENES)
        assert sub.dataset_axis is None
        assert sub.gene_axis == 1

    def test_subset_axis_removes_components(self):
        layout = AxisLayout(axes=(COMPONENTS, GENES))
        sub = layout.subset_axis(COMPONENTS)
        assert sub.axes == (GENES,)
        assert sub.component_axis is None
        assert sub.gene_axis == 0

    def test_subset_axis_preserves_sample_dim(self):
        layout = AxisLayout(
            axes=(COMPONENTS, DATASETS, GENES), has_sample_dim=True
        )
        sub = layout.subset_axis(DATASETS)
        assert sub.has_sample_dim is True
        assert sub.axes == (COMPONENTS, GENES)

    def test_subset_axis_raises_on_missing(self):
        layout = AxisLayout(axes=(GENES,))
        with pytest.raises(ValueError, match="not in layout"):
            layout.subset_axis(DATASETS)


# =========================================================================
# broadcast_to tests
# =========================================================================


class TestBroadcastTo:
    """Test the broadcast_to method for singleton insertion."""

    def test_scalar_to_component_gene(self):
        src = AxisLayout(axes=())
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (None, None)

    def test_component_to_component_gene(self):
        src = AxisLayout(axes=(COMPONENTS,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (slice(None), None)

    def test_gene_to_component_gene(self):
        src = AxisLayout(axes=(GENES,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (None, slice(None))

    def test_dataset_gene_to_component_dataset_gene(self):
        src = AxisLayout(axes=(DATASETS, GENES))
        tgt = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (None, slice(None), slice(None))

    def test_component_gene_to_component_dataset_gene(self):
        src = AxisLayout(axes=(COMPONENTS, GENES))
        tgt = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (slice(None), None, slice(None))

    def test_identity_broadcast(self):
        src = AxisLayout(axes=(COMPONENTS, GENES))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        assert idx == (slice(None), slice(None))

    def test_broadcast_with_sample_dim(self):
        src = AxisLayout(axes=(COMPONENTS,), has_sample_dim=True)
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        # sample dim: slice(None), then components: slice, genes: None
        assert idx == (slice(None), slice(None), None)

    def test_broadcast_numerical_equivalence_scalar_to_kg(self):
        """Verify broadcast produces correct shapes via jnp indexing."""
        src = AxisLayout(axes=())
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        tensor = jnp.array(0.5)
        result = tensor[idx]
        assert result.shape == (1, 1)

    def test_broadcast_numerical_equivalence_k_to_kg(self):
        src = AxisLayout(axes=(COMPONENTS,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        tensor = jnp.ones(3)
        result = tensor[idx]
        assert result.shape == (3, 1)

    def test_broadcast_numerical_gene_to_kg(self):
        src = AxisLayout(axes=(GENES,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        idx = src.broadcast_to(tgt)
        tensor = jnp.ones(100)
        result = tensor[idx]
        assert result.shape == (1, 100)


# =========================================================================
# align_to_layout tests
# =========================================================================


class TestAlignToLayout:
    """Test the align_to_layout helper for numerical correctness."""

    def test_component_scalar_to_component_gene(self):
        p = jnp.array([0.3, 0.4, 0.5])  # (K=3,)
        src = AxisLayout(axes=(COMPONENTS,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        result = align_to_layout(p, src, tgt)
        assert result.shape == (3, 1)
        assert jnp.allclose(result[:, 0], p)

    def test_gene_to_component_gene(self):
        r = jnp.ones(100)  # (G=100,)
        src = AxisLayout(axes=(GENES,))
        tgt = AxisLayout(axes=(COMPONENTS, GENES))
        result = align_to_layout(r, src, tgt)
        assert result.shape == (1, 100)

    def test_dataset_gene_to_component_dataset_gene(self):
        mu = jnp.ones((2, 100))  # (D=2, G=100)
        src = AxisLayout(axes=(DATASETS, GENES))
        tgt = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        result = align_to_layout(mu, src, tgt)
        assert result.shape == (1, 2, 100)

    def test_scalar_to_component_dataset_gene(self):
        p = jnp.array(0.5)
        src = AxisLayout(axes=())
        tgt = AxisLayout(axes=(COMPONENTS, DATASETS, GENES))
        result = align_to_layout(p, src, tgt)
        assert result.shape == (1, 1, 1)


# =========================================================================
# layout_from_param_spec factory tests
# =========================================================================


class TestLayoutFromParamSpec:
    """Test building layouts from ParamSpec objects."""

    def test_scalar_beta(self):
        spec = BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0))
        layout = layout_from_param_spec(spec)
        assert layout.axes == ()
        assert layout.has_sample_dim is False

    def test_gene_specific_lognormal(self):
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (GENES,)
        assert layout.gene_axis == 0

    def test_cell_specific(self):
        spec = BetaSpec(
            name="p_capture",
            shape_dims=("n_cells",),
            default_params=(1.0, 1.0),
            is_cell_specific=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (CELLS,)
        assert layout.cell_axis == 0

    def test_mixture_scalar(self):
        spec = BetaSpec(
            name="p",
            shape_dims=(),
            default_params=(1.0, 1.0),
            is_mixture=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (COMPONENTS,)

    def test_mixture_gene(self):
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (COMPONENTS, GENES)

    def test_dataset_gene(self):
        spec = LogNormalSpec(
            name="mu",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_gene_specific=True,
            is_dataset=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (DATASETS, GENES)

    def test_mixture_dataset_gene(self):
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
            is_dataset=True,
        )
        layout = layout_from_param_spec(spec)
        assert layout.axes == (COMPONENTS, DATASETS, GENES)

    def test_with_sample_dim(self):
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
        )
        layout = layout_from_param_spec(spec, has_sample_dim=True)
        assert layout.axes == (COMPONENTS, GENES)
        assert layout.has_sample_dim is True
        assert layout.component_axis == 1
        assert layout.gene_axis == 2


# =========================================================================
# infer_layout backward-compat tests
# =========================================================================


class TestInferLayout:
    """Test layout reconstruction from tensor shape + config metadata."""

    def test_scalar_p(self):
        layout = infer_layout("p_loc", jnp.array(0.5), n_genes=100)
        assert layout.axes == ()

    def test_gene_specific_r(self):
        layout = infer_layout("r_loc", jnp.ones(100), n_genes=100)
        assert layout.axes == (GENES,)

    def test_mixture_r(self):
        layout = infer_layout(
            "r_loc",
            jnp.ones((3, 100)),
            n_genes=100,
            n_components=3,
        )
        assert layout.axes == (COMPONENTS, GENES)

    def test_mixture_scalar_p(self):
        """p with shape (K,) and n_components=K => (components,)."""
        layout = infer_layout(
            "p_loc",
            jnp.ones(3),
            n_genes=100,
            n_components=3,
        )
        assert layout.axes == (COMPONENTS,)

    def test_cell_specific_p_capture(self):
        layout = infer_layout(
            "p_capture_loc",
            jnp.ones(50),
            n_genes=100,
            n_cells=50,
        )
        assert layout.axes == (CELLS,)

    def test_mixing_weights(self):
        layout = infer_layout(
            "mixing_weights_loc",
            jnp.ones(3),
            n_genes=100,
            n_components=3,
        )
        assert layout.axes == (COMPONENTS,)

    def test_component_dataset_gene(self):
        layout = infer_layout(
            "r_loc",
            jnp.ones((3, 2, 100)),
            n_genes=100,
            n_components=3,
            n_datasets=2,
            dataset_params=["r"],
        )
        assert layout.axes == (COMPONENTS, DATASETS, GENES)

    def test_dataset_gene(self):
        layout = infer_layout(
            "mu_loc",
            jnp.ones((2, 100)),
            n_genes=100,
            n_datasets=2,
            dataset_params=["mu"],
            n_components=None,
        )
        assert layout.axes == (DATASETS, GENES)

    def test_gene_axis_hint_resolves_ambiguity(self):
        """When n_components == n_genes, gene_axis_hint disambiguates."""
        layout = infer_layout(
            "r_loc",
            jnp.ones((5, 5)),
            n_genes=5,
            n_components=5,
            gene_axis_hint=1,
        )
        assert layout.axes == (COMPONENTS, GENES)
        assert layout.gene_axis == 1

    def test_has_sample_dim(self):
        """Posterior arrays with a leading sample dim."""
        layout = infer_layout(
            "r",
            jnp.ones((200, 100)),
            n_genes=100,
            has_sample_dim=True,
        )
        assert layout.axes == (GENES,)
        assert layout.has_sample_dim is True
        assert layout.gene_axis == 1

    def test_mixture_params_constrains_component_axis(self):
        """When mixture_params=['r'], p should not get a component axis."""
        layout = infer_layout(
            "p_loc",
            jnp.ones(3),
            n_genes=100,
            n_components=3,
            mixture_params=["r"],
        )
        # p is not in mixture_params, so 3 does not match n_components
        assert COMPONENTS not in layout.axes

    def test_flow_key_skipped(self):
        """Flax nested-dict keys are treated as opaque."""
        layout = infer_layout(
            "flow_p$params",
            jnp.ones((4, 5)),
            n_genes=5,
        )
        assert layout.axes == ()

    def test_agrees_with_layout_from_param_spec(self):
        """Cross-check: infer_layout should agree with layout_from_param_spec
        for a standard gene-specific mixture parameter."""
        spec = LogNormalSpec(
            name="r",
            shape_dims=("n_genes",),
            default_params=(0.0, 1.0),
            is_mixture=True,
            is_gene_specific=True,
        )
        from_spec = layout_from_param_spec(spec)
        from_infer = infer_layout(
            "r_loc",
            jnp.ones((3, 100)),
            n_genes=100,
            n_components=3,
        )
        assert from_spec.axes == from_infer.axes


# =========================================================================
# expand_membership_from_derived tests
# =========================================================================


class TestExpandMembershipFromDerived:
    """Test axis-membership propagation through DerivedParam dep chains."""

    def test_no_derived_params(self):
        """Empty derived list returns original set unchanged."""
        from scribe.core.axis_layout import expand_membership_from_derived

        result = expand_membership_from_derived({"phi", "mu"}, [])
        assert result == {"phi", "mu"}

    def test_single_derived_added(self):
        """Derived param whose dep is a member gets added."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        # r depends on phi and mu — both are members, so r is added
        derived = [
            DerivedParam("r", lambda phi, mu: mu * phi, ["phi", "mu"]),
        ]
        result = expand_membership_from_derived({"phi", "mu"}, derived)
        assert "r" in result
        assert result == {"phi", "mu", "r"}

    def test_multiple_derived_added(self):
        """Multiple derived params from same deps are all added."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        # mean_odds: r from [phi, mu], p from [phi]
        derived = [
            DerivedParam("r", lambda phi, mu: mu * phi, ["phi", "mu"]),
            DerivedParam("p", lambda phi: 1.0 / (1.0 + phi), ["phi"]),
        ]
        result = expand_membership_from_derived({"phi", "mu"}, derived)
        assert result == {"phi", "mu", "r", "p"}

    def test_transitive_expansion(self):
        """Derived param depending on another derived param gets added."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        # p depends on phi (direct), q depends on p (transitive)
        derived = [
            DerivedParam("p", lambda phi: phi, ["phi"]),
            DerivedParam("q", lambda p: p, ["p"]),
        ]
        result = expand_membership_from_derived({"phi"}, derived)
        assert result == {"phi", "p", "q"}

    def test_unrelated_derived_not_added(self):
        """Derived param whose deps are disjoint from members is not added."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        derived = [
            DerivedParam("gate", lambda z: z, ["z"]),
        ]
        result = expand_membership_from_derived({"phi", "mu"}, derived)
        assert "gate" not in result
        assert result == {"phi", "mu"}

    def test_partial_dep_overlap_adds(self):
        """A derived param is added if ANY dep is a member, not all."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        # r depends on [phi, mu], but only phi is a member
        derived = [
            DerivedParam("r", lambda phi, mu: mu * phi, ["phi", "mu"]),
        ]
        result = expand_membership_from_derived({"phi"}, derived)
        assert "r" in result

    def test_dataset_params_expansion(self):
        """Same function works for dataset membership propagation."""
        from scribe.core.axis_layout import expand_membership_from_derived
        from scribe.models.builders.parameter_specs import DerivedParam

        # dataset_params contains "mu", derived r depends on mu
        derived = [
            DerivedParam("r", lambda p, mu: mu, ["p", "mu"]),
        ]
        result = expand_membership_from_derived({"mu"}, derived)
        assert result == {"mu", "r"}


# =========================================================================
# _strip_param_key tests
# =========================================================================


class TestStripParamKey:
    """Test the variational-key to base-name stripping."""

    def test_loc_suffix(self):
        assert _strip_param_key("r_loc") == "r"

    def test_scale_suffix(self):
        assert _strip_param_key("r_scale") == "r"

    def test_log_prefix(self):
        assert _strip_param_key("log_mu_loc") == "mu"

    def test_logit_prefix(self):
        assert _strip_param_key("logit_p_loc") == "p"

    def test_base_loc_suffix(self):
        assert _strip_param_key("r_base_loc") == "r"

    def test_mixing_weights(self):
        assert _strip_param_key("mixing_weights_loc") == "mixing_weights"

    def test_flow_key_unchanged(self):
        assert _strip_param_key("flow_p$params") == "flow_p$params"

    def test_bare_name(self):
        assert _strip_param_key("p") == "p"


# =========================================================================
# build_param_layouts tests
# =========================================================================


class TestBuildParamLayouts:
    """Test bulk layout construction from param_specs."""

    def test_empty_specs_returns_empty(self):
        result = build_param_layouts([], {"r_loc": jnp.ones(5)})
        assert result == {}

    def test_standard_model(self):
        specs = [
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0)),
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
            ),
        ]
        params = {
            "p_loc": jnp.array(0.5),
            "p_scale": jnp.array(0.1),
            "r_loc": jnp.ones(100),
            "r_scale": jnp.ones(100),
        }
        layouts = build_param_layouts(specs, params)
        assert layouts["p_loc"].axes == ()
        assert layouts["r_loc"].axes == (GENES,)
        assert layouts["r_scale"].axes == (GENES,)

    def test_mixture_model(self):
        specs = [
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                is_mixture=True,
            ),
            LogNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_mixture=True,
                is_gene_specific=True,
            ),
        ]
        params = {
            "p_loc": jnp.ones(3),
            "r_loc": jnp.ones((3, 100)),
        }
        layouts = build_param_layouts(specs, params)
        assert layouts["p_loc"].axes == (COMPONENTS,)
        assert layouts["r_loc"].axes == (COMPONENTS, GENES)

    def test_skips_flow_keys(self):
        specs = [
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0)),
        ]
        params = {
            "p_loc": jnp.array(0.5),
            "flow_p$params": {"dense": jnp.ones((4, 5))},
        }
        layouts = build_param_layouts(specs, params)
        assert "flow_p$params" not in layouts

    def test_unmatched_key_gets_empty_layout(self):
        specs = [
            BetaSpec(name="p", shape_dims=(), default_params=(1.0, 1.0)),
        ]
        params = {
            "p_loc": jnp.array(0.5),
            "unknown_param_loc": jnp.ones(10),
        }
        layouts = build_param_layouts(specs, params)
        assert layouts["unknown_param_loc"].axes == ()


# =========================================================================
# reconstruct_param_layouts tests
# =========================================================================


class TestReconstructParamLayouts:
    """Test backward-compat bulk reconstruction."""

    def test_standard_model(self):
        params = {
            "p_loc": jnp.array(0.5),
            "r_loc": jnp.ones(100),
        }
        layouts = reconstruct_param_layouts(
            params, n_genes=100, n_cells=50
        )
        assert layouts["p_loc"].axes == ()
        assert layouts["r_loc"].axes == (GENES,)

    def test_mixture_model(self):
        params = {
            "p_loc": jnp.ones(3),
            "r_loc": jnp.ones((3, 100)),
            "mixing_weights_loc": jnp.ones(3),
        }
        layouts = reconstruct_param_layouts(
            params, n_genes=100, n_components=3
        )
        assert layouts["r_loc"].axes == (COMPONENTS, GENES)
        assert layouts["mixing_weights_loc"].axes == (COMPONENTS,)

    def test_with_gene_axis_hints(self):
        params = {
            "r_loc": jnp.ones((5, 5)),
        }
        layouts = reconstruct_param_layouts(
            params,
            n_genes=5,
            n_components=5,
            gene_axis_by_key={"r_loc": 1},
        )
        assert layouts["r_loc"].axes == (COMPONENTS, GENES)

    def test_skips_non_array_values(self):
        params = {
            "r_loc": jnp.ones(100),
            "metadata": "some_string",
        }
        layouts = reconstruct_param_layouts(params, n_genes=100)
        assert "metadata" not in layouts
        assert "r_loc" in layouts


# ========================================================================
# Tests for build_sample_layouts
# ========================================================================


class TestBuildSampleLayouts:
    """``build_sample_layouts``: spec-backed keys vs ``infer_layout`` fallback.

    Covers the hybrid path used by SVI dataset/component subsetting when
    posterior dicts mix declared parameters with derived tensors.
    """

    def test_spec_matched_keys_use_spec(self):
        """Keys matching a ParamSpec get their layout from the spec."""
        spec = LogNormalSpec(
            name="r", shape_dims=("n_genes",),
            default_params=(0.0, 1.0), is_mixture=True,
        )
        samples = {
            "r": jnp.ones((10, 3, 100)),
        }
        layouts = build_sample_layouts(
            [spec], samples,
            n_genes=100, n_components=3,
            has_sample_dim=True,
        )
        assert "r" in layouts
        assert layouts["r"].axes == ("components", "genes")
        assert layouts["r"].gene_axis == 2
        assert layouts["r"].component_axis == 1

    def test_unrecognised_keys_use_heuristic(self):
        """Derived keys not in specs fall back to infer_layout."""
        spec = LogNormalSpec(
            name="phi", shape_dims=("n_genes",),
            default_params=(0.0, 1.0), is_mixture=True,
        )
        samples = {
            "phi": jnp.ones((10, 3, 100)),
            "p": jnp.ones((10, 3, 100)),
        }
        layouts = build_sample_layouts(
            [spec], samples,
            n_genes=100, n_components=3,
            has_sample_dim=True,
        )
        assert layouts["phi"].axes == ("components", "genes")
        # "p" is not in specs -> infer_layout recognises it as a known
        # gene param and infers components + genes from shape.
        assert "p" in layouts
        assert layouts["p"].component_axis is not None
        assert layouts["p"].gene_axis is not None

    def test_empty_specs_all_heuristic(self):
        """When param_specs is empty, every key gets infer_layout."""
        samples = {
            "r": jnp.ones((10, 3, 100)),
        }
        layouts = build_sample_layouts(
            [], samples,
            n_genes=100, n_components=3,
            has_sample_dim=True,
        )
        assert "r" in layouts
        assert layouts["r"].component_axis is not None

    def test_skips_non_array_and_flow_keys(self):
        samples = {
            "r": jnp.ones((10, 100)),
            "flow_p$params": {"nested": True},
            "metadata": "text",
        }
        layouts = build_sample_layouts(
            [], samples, n_genes=100, has_sample_dim=True,
        )
        assert "r" in layouts
        assert "flow_p$params" not in layouts
        assert "metadata" not in layouts


# ========================================================================
# Tests for gene_axes_from_layouts
# ========================================================================


class TestGeneAxesFromLayouts:
    """``gene_axes_from_layouts``: dict comprehension over ``layout.gene_axis``."""

    def test_extracts_gene_axes(self):
        layouts = {
            "r_loc": AxisLayout(("components", "genes")),
            "mixing_weights_loc": AxisLayout(("components",)),
            "p_capture_loc": AxisLayout(("cells",)),
        }
        result = gene_axes_from_layouts(layouts)
        assert result == {"r_loc": 1}

    def test_empty_when_no_gene_axes(self):
        layouts = {
            "mixing_weights_loc": AxisLayout(("components",)),
        }
        result = gene_axes_from_layouts(layouts)
        assert result == {}

    def test_with_sample_dim(self):
        layouts = {
            "r": AxisLayout(("components", "genes"), has_sample_dim=True),
            "scalar": AxisLayout((), has_sample_dim=True),
        }
        result = gene_axes_from_layouts(layouts)
        assert result == {"r": 2}

    def test_empty_dict_input(self):
        assert gene_axes_from_layouts({}) == {}


# ========================================================================
# Tests for merge_layouts
# ========================================================================


class TestMergeLayouts:
    """``merge_layouts``: union of axes in canonical order."""

    def test_identical_layouts_unchanged(self):
        """Merging identical layouts returns the same axes."""
        a = AxisLayout((COMPONENTS, GENES))
        result = merge_layouts(a, a)
        assert result.axes == (COMPONENTS, GENES)

    def test_subset_to_superset(self):
        """Merging a subset with its superset returns the superset."""
        a = AxisLayout((COMPONENTS,))
        b = AxisLayout((COMPONENTS, GENES))
        result = merge_layouts(a, b)
        assert result.axes == (COMPONENTS, GENES)

    def test_disjoint_axes(self):
        """Merging layouts with no common axes returns all axes ordered."""
        a = AxisLayout((GENES,))
        b = AxisLayout((COMPONENTS,))
        result = merge_layouts(a, b)
        assert result.axes == (COMPONENTS, GENES)

    def test_three_layouts_full_union(self):
        """Merging three layouts returns all distinct axes."""
        a = AxisLayout((GENES,))
        b = AxisLayout((COMPONENTS,))
        c = AxisLayout((DATASETS,))
        result = merge_layouts(a, b, c)
        assert result.axes == (COMPONENTS, DATASETS, GENES)

    def test_component_gene_and_dataset_gene(self):
        """(K,G) merged with (D,G) => (K,D,G)."""
        a = AxisLayout((COMPONENTS, GENES))
        b = AxisLayout((DATASETS, GENES))
        result = merge_layouts(a, b)
        assert result.axes == (COMPONENTS, DATASETS, GENES)

    def test_scalar_layout_with_full(self):
        """Merging scalar layout with full layout returns the full."""
        a = AxisLayout(())
        b = AxisLayout((COMPONENTS, DATASETS, GENES))
        result = merge_layouts(a, b)
        assert result.axes == (COMPONENTS, DATASETS, GENES)

    def test_single_layout_returns_copy(self):
        """Merging a single layout returns an equivalent layout."""
        a = AxisLayout((GENES,))
        result = merge_layouts(a)
        assert result.axes == (GENES,)


# ========================================================================
# Tests for broadcast_param_to_layout
# ========================================================================


class TestBroadcastParamToLayout:
    """``broadcast_param_to_layout``: batch-aware broadcasting."""

    # ------------------------------------------------------------------
    # Non-batched cases (should match align_to_layout)
    # ------------------------------------------------------------------

    def test_scalar_to_component_gene(self):
        """Scalar p -> (1, 1) when target is (K, G)."""
        p = jnp.array(0.5)
        result = broadcast_param_to_layout(
            p, AxisLayout(()), AxisLayout((COMPONENTS, GENES))
        )
        assert result.shape == (1, 1)

    def test_component_to_component_gene(self):
        """(K,) p -> (K, 1) when target is (K, G)."""
        p = jnp.ones(3)
        result = broadcast_param_to_layout(
            p, AxisLayout((COMPONENTS,)), AxisLayout((COMPONENTS, GENES))
        )
        assert result.shape == (3, 1)

    def test_gene_to_component_gene(self):
        """(G,) p -> (1, G) when target is (K, G)."""
        p = jnp.ones(100)
        result = broadcast_param_to_layout(
            p, AxisLayout((GENES,)), AxisLayout((COMPONENTS, GENES))
        )
        assert result.shape == (1, 100)

    def test_identity_no_change(self):
        """(K, G) -> (K, G) when layouts already match."""
        p = jnp.ones((3, 100))
        result = broadcast_param_to_layout(
            p,
            AxisLayout((COMPONENTS, GENES)),
            AxisLayout((COMPONENTS, GENES)),
        )
        assert result.shape == (3, 100)

    def test_dataset_gene_to_component_dataset_gene(self):
        """(D, G) -> (1, D, G) when target is (K, D, G)."""
        mu = jnp.ones((2, 100))
        result = broadcast_param_to_layout(
            mu,
            AxisLayout((DATASETS, GENES)),
            AxisLayout((COMPONENTS, DATASETS, GENES)),
        )
        assert result.shape == (1, 2, 100)

    def test_component_gene_to_component_dataset_gene(self):
        """(K, G) -> (K, 1, G) when target is (K, D, G)."""
        r = jnp.ones((3, 100))
        result = broadcast_param_to_layout(
            r,
            AxisLayout((COMPONENTS, GENES)),
            AxisLayout((COMPONENTS, DATASETS, GENES)),
        )
        assert result.shape == (3, 1, 100)

    # ------------------------------------------------------------------
    # Batched cases (leading cell/batch dim not in layout)
    # ------------------------------------------------------------------

    def test_batched_gene_to_component_gene(self):
        """(batch, G) -> (batch, 1, G) when target is (K, G)."""
        p = jnp.ones((32, 100))
        result = broadcast_param_to_layout(
            p, AxisLayout((GENES,)), AxisLayout((COMPONENTS, GENES))
        )
        assert result.shape == (32, 1, 100)

    def test_batched_scalar_to_component_gene(self):
        """(batch,) -> (batch, 1, 1) when target is (K, G).

        A scalar param that picked up a batch dim after dataset indexing.
        """
        p = jnp.ones(32)
        result = broadcast_param_to_layout(
            p, AxisLayout(()), AxisLayout((COMPONENTS, GENES))
        )
        assert result.shape == (32, 1, 1)

    def test_batched_component_gene_to_component_gene(self):
        """(batch, K, G) -> (batch, K, G) — already matching."""
        p = jnp.ones((32, 3, 100))
        result = broadcast_param_to_layout(
            p,
            AxisLayout((COMPONENTS, GENES)),
            AxisLayout((COMPONENTS, GENES)),
        )
        assert result.shape == (32, 3, 100)

    def test_batched_component_to_component_gene(self):
        """(batch, K) -> (batch, K, 1) when target is (K, G)."""
        p = jnp.ones((32, 3))
        result = broadcast_param_to_layout(
            p,
            AxisLayout((COMPONENTS,)),
            AxisLayout((COMPONENTS, GENES)),
        )
        assert result.shape == (32, 3, 1)

    def test_non_batched_stays_non_batched(self):
        """Confirm that no spurious batch dim is introduced."""
        p = jnp.ones(3)
        result = broadcast_param_to_layout(
            p,
            AxisLayout((COMPONENTS,)),
            AxisLayout((COMPONENTS, GENES)),
        )
        # (K,) -> (K, 1), no leading batch dim
        assert result.shape == (3, 1)


# ==============================================================================
# Tests for _slice_gene_axis (sampling helper)
# ==============================================================================


class TestDeriveAxisMembership:
    """Test the unified ``derive_axis_membership`` cascade.

    Verifies every priority level (explicit config, ParamSpec flags,
    HierarchicalPriorType flags, concat shape scan) and the DerivedParam
    expansion step.
    """

    # ------------------------------------------------------------------
    # Helper to build lightweight config stubs
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg(**kwargs):
        """Return a SimpleNamespace masquerading as a ModelConfig."""
        from types import SimpleNamespace

        defaults = dict(
            parameterization="linked",
            mixture_params=None,
            dataset_params=None,
            param_specs=[],
            n_datasets=None,
            expression_dataset_prior=None,
            prob_dataset_prior=None,
            zero_inflation_dataset_prior=None,
            overdispersion_dataset_prior=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    # ------------------------------------------------------------------
    # Explicit config fields are returned as-is (but expanded)
    # ------------------------------------------------------------------

    def test_explicit_mixture_params_returned(self):
        """Explicit mixture_params on config are returned (with expansion)."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="mean_odds",
            mixture_params=["phi", "mu"],
        )
        mp, dp = derive_axis_membership(cfg)
        # phi and mu are explicitly set; derived "r" and "p" should be
        # added by expansion through mean_odds DerivedParam graph.
        assert "phi" in mp
        assert "mu" in mp
        assert "r" in mp
        assert "p" in mp
        assert dp is None

    def test_explicit_dataset_params_returned(self):
        """Explicit dataset_params on config are returned (with expansion)."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            dataset_params=["mu"],
        )
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        # "mu" is explicit; "r" derives from ["p", "mu"] in linked,
        # and mu is a member so "r" is added.
        assert "mu" in dp
        assert "r" in dp

    # ------------------------------------------------------------------
    # ParamSpec flags
    # ------------------------------------------------------------------

    def test_param_spec_is_mixture_flag(self):
        """When no explicit lists, ParamSpec.is_mixture is used."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            param_specs=[
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_mixture=True,
                    is_gene_specific=True,
                ),
            ],
        )
        mp, dp = derive_axis_membership(cfg)
        # "mu" detected from spec flag
        assert "mu" in mp
        # "r" expanded (linked: r depends on ["p", "mu"])
        assert "r" in mp
        assert dp is None

    def test_param_spec_is_dataset_flag(self):
        """When no explicit lists, ParamSpec.is_dataset is used."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            param_specs=[
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    is_dataset=True,
                ),
            ],
        )
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        assert "mu" in dp
        assert "r" in dp

    # ------------------------------------------------------------------
    # HierarchicalPriorType flags
    # ------------------------------------------------------------------

    def test_expression_dataset_prior_linked(self):
        """expression_dataset_prior under linked → mu."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="linked",
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        assert "mu" in dp
        # Derived expansion: r depends on mu in linked
        assert "r" in dp

    def test_expression_dataset_prior_canonical(self):
        """expression_dataset_prior under canonical → r."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="canonical",
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        assert "r" in dp

    def test_prob_dataset_prior_mean_odds(self):
        """prob_dataset_prior under mean_odds → phi (+ derived p)."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="mean_odds",
            prob_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        assert "phi" in dp
        # mean_odds: p = 1/(1+phi), so p is derived from phi
        assert "p" in dp

    def test_prob_dataset_prior_linked(self):
        """prob_dataset_prior under linked → p."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="linked",
            prob_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        mp, dp = derive_axis_membership(cfg)
        assert "p" in dp
        # linked: r depends on ["p", "mu"], and p is a member
        assert "r" in dp

    def test_zero_inflation_dataset_prior(self):
        """zero_inflation_dataset_prior → gate."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="linked",
            zero_inflation_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        _, dp = derive_axis_membership(cfg)
        assert "gate" in dp

    def test_overdispersion_dataset_prior(self):
        """overdispersion_dataset_prior → bnb_concentration."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="linked",
            overdispersion_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        _, dp = derive_axis_membership(cfg)
        assert "bnb_concentration" in dp

    # ------------------------------------------------------------------
    # Parameterization-dependent canonical-name mapping
    # ------------------------------------------------------------------

    def test_canonical_parameterization_maps_to_r_and_mu(self):
        """canonical: expression_dataset_prior → 'r', plus derived 'mu'."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="canonical",
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        _, dp = derive_axis_membership(cfg)
        assert "r" in dp
        # mu is derived from r and p in canonical, so it inherits dataset membership
        assert "mu" in dp

    def test_standard_alias_maps_like_canonical(self):
        """'standard' is an alias for canonical; mu derived from r."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="standard",
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        _, dp = derive_axis_membership(cfg)
        assert "r" in dp
        assert "mu" in dp

    def test_odds_ratio_alias_maps_like_mean_odds(self):
        """'odds_ratio' is an alias for mean_odds; prob_dataset → phi."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="odds_ratio",
            prob_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        _, dp = derive_axis_membership(cfg)
        assert "phi" in dp
        assert "p" in dp  # derived from phi

    # ------------------------------------------------------------------
    # Derived expansion through DerivedParam graph
    # ------------------------------------------------------------------

    def test_mean_odds_mixture_expansion(self):
        """mixture_params=['phi', 'mu'] under mean_odds → includes r, p."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="mean_odds",
            mixture_params=["phi", "mu"],
        )
        mp, _ = derive_axis_membership(cfg)
        assert set(mp) >= {"phi", "mu", "r", "p"}

    def test_linked_dataset_expansion(self):
        """dataset_params=['p', 'mu'] under linked → includes r."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            dataset_params=["p", "mu"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "r" in dp

    def test_canonical_expands_mu_from_r_and_p(self):
        """canonical declares mu as derived from r and p, so mu is expanded."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            mixture_params=["r", "p"],
        )
        mp, _ = derive_axis_membership(cfg)
        assert sorted(mp) == ["mu", "p", "r"]

    # ------------------------------------------------------------------
    # Concat multi-dataset shape scan
    # ------------------------------------------------------------------

    def test_concat_shape_scan_detects_dataset_keys(self):
        """Shape scan finds keys with leading dim == n_datasets."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(n_datasets=3, parameterization="canonical")
        samples = {
            "r": jnp.ones((3, 100)),
            "p": jnp.ones((3, 100)),
            "mixing_weights": jnp.ones(3),
        }
        _, dp = derive_axis_membership(cfg, samples=samples)
        assert "r" in dp
        assert "p" in dp
        # mixing_weights also has shape[0] == 3, but is not a cell param
        # so it should be detected too
        assert "mixing_weights" in dp

    def test_concat_shape_scan_excludes_cell_params(self):
        """Cell-specific params (p_capture, etc.) are excluded from scan."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(n_datasets=2, parameterization="canonical")
        samples = {
            "r": jnp.ones((2, 100)),
            "p_capture": jnp.ones((2,)),
        }
        _, dp = derive_axis_membership(cfg, samples=samples)
        assert "r" in dp
        # p_capture is a known cell param and should be excluded
        assert "p_capture" not in dp

    def test_concat_shape_scan_with_sample_dim(self):
        """has_sample_dim shifts dataset axis to position 1."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(n_datasets=2, parameterization="canonical")
        # With sample dim: shape is (n_samples, n_datasets, ...)
        samples = {
            "r": jnp.ones((10, 2, 100)),
            "p": jnp.ones((10, 2, 100)),
        }
        _, dp = derive_axis_membership(
            cfg, samples=samples, has_sample_dim=True,
        )
        assert "r" in dp
        assert "p" in dp

    def test_concat_scan_skipped_when_priors_active(self):
        """If HierarchicalPriorType flags resolve dataset_params,
        the concat scan is never reached."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="linked",
            n_datasets=2,
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        # Samples with leading dim 2 for ALL keys — but the prior flag
        # should resolve before the scan runs.
        samples = {
            "r": jnp.ones((2, 100)),
            "p": jnp.ones((2, 100)),
            "mu": jnp.ones((2, 100)),
        }
        _, dp = derive_axis_membership(cfg, samples=samples)
        # "mu" from prior flag (linked), "r" from derived expansion —
        # but "p" should NOT be added because the scan didn't run.
        assert "mu" in dp
        assert "r" in dp
        assert "p" not in dp

    # ------------------------------------------------------------------
    # No signals at all → (None, None)
    # ------------------------------------------------------------------

    def test_no_signals_returns_none_none(self):
        """When there are no signals, both should be None."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg()
        mp, dp = derive_axis_membership(cfg)
        assert mp is None
        assert dp is None

    def test_no_signals_single_dataset(self):
        """Single dataset (n_datasets=1) never triggers concat scan."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(n_datasets=1)
        samples = {"r": jnp.ones((1, 100))}
        mp, dp = derive_axis_membership(cfg, samples=samples)
        assert mp is None
        assert dp is None

    # ------------------------------------------------------------------
    # Combined: both mixture and dataset
    # ------------------------------------------------------------------

    def test_combined_mixture_and_dataset(self):
        """Both mixture_params and dataset_params derived simultaneously."""
        from scribe.core.axis_layout import derive_axis_membership
        from scribe.models.config.enums import HierarchicalPriorType

        cfg = self._cfg(
            parameterization="mean_odds",
            mixture_params=["phi", "mu"],
            expression_dataset_prior=HierarchicalPriorType.GAUSSIAN,
        )
        mp, dp = derive_axis_membership(cfg)
        # Mixture: phi, mu + derived r, p
        assert set(mp) >= {"phi", "mu", "r", "p"}
        # Dataset: mean_odds + expression → mu + derived r
        assert "mu" in dp
        assert "r" in dp


class TestSliceGeneAxis:
    """Tests for ``_slice_gene_axis`` — subsets the gene dimension of a tensor
    using a known axis index from an ``AxisLayout``.
    """

    def test_slices_standard_gene_array(self):
        """Standard r of shape (S, G) should be sliced on axis=1."""
        from scribe.sampling import _slice_gene_axis

        r = jnp.arange(60).reshape(3, 20)
        gene_idx = jnp.array([0, 5, 10])
        result = _slice_gene_axis(r, gene_axis=1, gene_indices=gene_idx)
        assert result.shape == (3, 3)
        # Verify correct values were selected
        np.testing.assert_array_equal(result, r[:, gene_idx])

    def test_slices_mixture_gene_array(self):
        """Mixture r of shape (S, K, G) should be sliced on axis=2."""
        from scribe.sampling import _slice_gene_axis

        r = jnp.arange(180).reshape(3, 3, 20)
        gene_idx = jnp.array([1, 2, 3])
        result = _slice_gene_axis(r, gene_axis=2, gene_indices=gene_idx)
        assert result.shape == (3, 3, 3)
        np.testing.assert_array_equal(result, r[:, :, gene_idx])

    def test_none_array_passes_through(self):
        """None input should be returned as None."""
        from scribe.sampling import _slice_gene_axis

        result = _slice_gene_axis(None, gene_axis=1, gene_indices=jnp.array([0]))
        assert result is None

    def test_none_gene_axis_passes_through(self):
        """When gene_axis is None the array should be returned unchanged."""
        from scribe.sampling import _slice_gene_axis

        arr = jnp.ones((5,))
        result = _slice_gene_axis(arr, gene_axis=None, gene_indices=jnp.array([0]))
        assert result is arr


class TestSubsetGeneDimSamplesWithLayouts:
    """Tests for layout-enhanced ``_subset_gene_dim_samples`` which
    uses ``AxisLayout.gene_axis`` when layouts are available and falls
    back to shape scanning otherwise.
    """

    def test_layout_path_selects_correct_axis(self):
        """Gene axis from layout should be used for slicing."""
        from scribe.svi._sampling_posterior_predictive import (
            _subset_gene_dim_samples,
        )

        n_genes = 20
        samples = {
            # (S, G) — gene axis is 1 with sample dim
            "r": jnp.arange(60).reshape(3, n_genes),
            # scalar per sample — no gene axis
            "p": jnp.ones((3,)),
        }
        layouts = {
            "r": AxisLayout((GENES,), has_sample_dim=True),
            "p": AxisLayout((), has_sample_dim=True),
        }
        gene_idx = np.array([0, 5, 10])

        result = _subset_gene_dim_samples(
            samples, gene_idx, n_genes, layouts=layouts,
        )

        # r should be sliced on axis 1 (gene axis with sample offset)
        assert result["r"].shape == (3, 3)
        np.testing.assert_array_equal(result["r"], samples["r"][:, gene_idx])
        # p has no gene axis — should pass through unchanged
        assert result["p"].shape == (3,)

    def test_fallback_when_no_layouts(self):
        """Without layouts, shape-scanning fallback should work."""
        from scribe.svi._sampling_posterior_predictive import (
            _subset_gene_dim_samples,
        )

        n_genes = 20
        samples = {"r": jnp.ones((3, n_genes)), "p": jnp.array(0.5)}
        gene_idx = np.array([0, 5])

        result = _subset_gene_dim_samples(
            samples, gene_idx, n_genes, layouts=None,
        )

        assert result["r"].shape == (3, 2)
        assert result["p"].shape == ()

    def test_unknown_key_falls_back_to_shape_scan(self):
        """Keys not in layouts should fall back to shape scanning."""
        from scribe.svi._sampling_posterior_predictive import (
            _subset_gene_dim_samples,
        )

        n_genes = 20
        samples = {
            "r": jnp.ones((3, n_genes)),
            # key not in layouts — should fall back to shape scan
            "some_flow_param": jnp.ones((3, n_genes)),
        }
        layouts = {
            "r": AxisLayout((GENES,), has_sample_dim=True),
        }
        gene_idx = np.array([0, 5])

        result = _subset_gene_dim_samples(
            samples, gene_idx, n_genes, layouts=layouts,
        )

        # Both should be sliced correctly
        assert result["r"].shape == (3, 2)
        assert result["some_flow_param"].shape == (3, 2)


# ==============================================================================
# Dataset membership propagation through DerivedParam for all parameterizations
# ==============================================================================


class TestDerivedDatasetMembershipAllParameterizations:
    """Verify that dataset membership propagates through the DerivedParam graph
    for every parameterization, ensuring ``get_dataset()`` will correctly slice
    all derived posterior keys.

    The core invariant: if any *source* parameter of a DerivedParam is
    dataset-specific, the derived param must also become dataset-specific.
    """

    @staticmethod
    def _cfg(**kwargs):
        from types import SimpleNamespace

        defaults = dict(
            parameterization="canonical",
            mixture_params=None,
            dataset_params=None,
            param_specs=[],
            n_datasets=None,
            expression_dataset_prior=None,
            prob_dataset_prior=None,
            zero_inflation_dataset_prior=None,
            overdispersion_dataset_prior=None,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    # ------------------------------------------------------------------
    # Canonical: r is dataset → mu must become dataset
    # ------------------------------------------------------------------

    def test_canonical_r_dataset_propagates_to_mu(self):
        """When r is dataset-specific, derived mu inherits dataset membership."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            dataset_params=["r"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "r" in dp
        assert "mu" in dp, (
            "mu is derived from r in canonical; must inherit dataset membership"
        )

    def test_canonical_p_dataset_propagates_to_mu(self):
        """When p is dataset-specific, derived mu inherits dataset membership."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            dataset_params=["p"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "p" in dp
        assert "mu" in dp

    def test_canonical_both_r_and_p_dataset(self):
        """When both r and p are dataset, mu is also dataset."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            dataset_params=["r", "p"],
        )
        _, dp = derive_axis_membership(cfg)
        assert set(dp) >= {"r", "p", "mu"}

    def test_canonical_no_dataset_params_no_mu_expansion(self):
        """Without any dataset params, mu is not incorrectly added."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(parameterization="canonical")
        _, dp = derive_axis_membership(cfg)
        assert dp is None

    # ------------------------------------------------------------------
    # Mean prob (linked): mu is dataset → r must become dataset
    # ------------------------------------------------------------------

    def test_linked_mu_dataset_propagates_to_r(self):
        """linked/mean_prob: mu dataset → r inherits (derived from p, mu)."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            dataset_params=["mu"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "mu" in dp
        assert "r" in dp

    def test_linked_p_dataset_propagates_to_r(self):
        """linked/mean_prob: p dataset → r inherits."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="linked",
            dataset_params=["p"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "p" in dp
        assert "r" in dp

    # ------------------------------------------------------------------
    # Mean odds (odds_ratio): phi/mu dataset → r, p must become dataset
    # ------------------------------------------------------------------

    def test_mean_odds_mu_dataset_propagates_to_r(self):
        """mean_odds: mu dataset → r inherits (r depends on phi, mu)."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="mean_odds",
            dataset_params=["mu"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "mu" in dp
        assert "r" in dp

    def test_mean_odds_phi_dataset_propagates_to_r_and_p(self):
        """mean_odds: phi dataset → both r and p inherit."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="mean_odds",
            dataset_params=["phi"],
        )
        _, dp = derive_axis_membership(cfg)
        assert "phi" in dp
        assert "r" in dp, "r depends on [phi, mu]; phi is dataset"
        assert "p" in dp, "p depends on [phi]; phi is dataset"

    # ------------------------------------------------------------------
    # Mixture membership mirrors dataset propagation
    # ------------------------------------------------------------------

    def test_canonical_mixture_r_propagates_to_mu(self):
        """Mixture axis: if r is mixture, derived mu inherits."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            mixture_params=["r"],
        )
        mp, _ = derive_axis_membership(cfg)
        assert "r" in mp
        assert "mu" in mp

    def test_mean_odds_mixture_phi_propagates_to_r_and_p(self):
        """Mixture axis under mean_odds: phi → r, p."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="mean_odds",
            mixture_params=["phi"],
        )
        mp, _ = derive_axis_membership(cfg)
        assert set(mp) >= {"phi", "r", "p"}

    # ------------------------------------------------------------------
    # Combined mixture + dataset
    # ------------------------------------------------------------------

    def test_canonical_mixture_and_dataset_both_propagate(self):
        """Mixture r + dataset r → mu appears in both lists."""
        from scribe.core.axis_layout import derive_axis_membership

        cfg = self._cfg(
            parameterization="canonical",
            mixture_params=["r", "p"],
            dataset_params=["r"],
        )
        mp, dp = derive_axis_membership(cfg)
        assert "mu" in mp, "mu derived from r (mixture member)"
        assert "mu" in dp, "mu derived from r (dataset member)"
