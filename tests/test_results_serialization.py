"""Roundtrip pickle tests for SCRIBE results classes.

This module validates that all major results classes can be serialized and
deserialized with ``pickle`` without raising errors.
"""

import pickle

import jax.numpy as jnp
import numpy as np

from scribe.de.results import (
    ScribeParametricDEResults,
    ScribeEmpiricalDEResults,
    ScribeShrinkageDEResults,
)
from scribe.mc.results import ScribeModelComparisonResults
from scribe.mcmc.results import ScribeMCMCResults
from scribe.models.config import GuideFamilyConfig, ModelConfigBuilder
from scribe.models.components import (
    JointLowRankGuide,
    LowRankGuide,
    NormalizingFlowGuide,
)
from scribe.svi.results import ScribeSVIResults
from scribe.svi.vae_results import ScribeVAEResults
from scribe.svi._latent_space import _ENCODER_KEY, _DECODER_KEY


class _DummyLatentSpec:
    """Minimal latent spec object with required ``flow`` attribute."""

    def __init__(self):
        self.flow = None


def _roundtrip(obj):
    """Serialize and deserialize object with pickle."""
    return pickle.loads(pickle.dumps(obj))


def test_svi_results_pickle_roundtrip():
    """SVI results should survive pickle roundtrip."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
    results = ScribeSVIResults(
        params={"p_loc": jnp.array(0.0)},
        loss_history=jnp.array([5.0, 3.0, 2.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )
    restored = _roundtrip(results)
    assert isinstance(restored, ScribeSVIResults)
    assert restored.n_cells == 4
    assert restored.model_type == "nbdm"


def test_svi_results_repr_mean_field_summary():
    """SVI repr should be compact and report mean-field defaults."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
    results = ScribeSVIResults(
        params={"p_loc": jnp.array(0.0), "r_loc": jnp.ones(3)},
        loss_history=jnp.array([5.0, 3.0, 2.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )

    rendered = repr(results)
    assert rendered.startswith("ScribeSVIResults(")
    assert "model='nbdm'" in rendered
    assert "n_cells=4" in rendered
    assert "n_genes=3" in rendered
    assert "n_steps=3" in rendered
    assert "guide='mean_field'" in rendered
    assert "params={" not in rendered


def test_svi_results_repr_low_rank_summary():
    """SVI repr should summarize single-family low-rank guides."""
    cfg = (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_inference("svi")
        .with_guide_families(GuideFamilyConfig(r=LowRankGuide(rank=16)))
        .build()
    )
    results = ScribeSVIResults(
        params={"r_loc": jnp.ones(3)},
        loss_history=jnp.array([2.0, 1.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )

    rendered = repr(results)
    assert "guide='low_rank(k=16) on r'" in rendered


def test_svi_results_repr_joint_low_rank_summary():
    """SVI repr should summarize joint low-rank guides with metadata."""
    joint = JointLowRankGuide(rank=8, group="joint")
    cfg = (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_inference("svi")
        .with_joint_params(["p", "r"])
        .with_dense_params(["r"])
        .with_guide_families(GuideFamilyConfig(p=joint, r=joint))
        .build()
    )
    results = ScribeSVIResults(
        params={"joint_p_loc": jnp.array(0.0)},
        loss_history=jnp.array([2.0, 1.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )

    rendered = repr(results)
    assert (
        "guide='joint_low_rank(k=8,group=joint,params=p,r,dense=r)'" in rendered
    )


def test_svi_results_repr_mixed_summary():
    """SVI repr should summarize mixed guide-family overrides."""
    cfg = (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_inference("svi")
        .with_guide_families(
            GuideFamilyConfig(
                r=LowRankGuide(rank=12),
                p=NormalizingFlowGuide(flow_type="maf", num_layers=2),
            )
        )
        .build()
    )
    results = ScribeSVIResults(
        params={"r_loc": jnp.ones(3)},
        loss_history=jnp.array([2.0, 1.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )

    rendered = repr(results)
    assert "guide='mixed[" in rendered
    assert "p=flow(type=maf,layers=2)" in rendered
    assert "r=low_rank(k=12)" in rendered


def test_svi_results_repr_html_compact_summary():
    """HTML repr should be compact and include the same summary fields."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
    results = ScribeSVIResults(
        params={"p_loc": jnp.array(0.0), "r_loc": jnp.ones(3)},
        loss_history=jnp.array([5.0, 3.0, 2.0]),
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )

    rendered = results._repr_html_()
    assert rendered.startswith("<div>")
    assert "ScribeSVIResults" in rendered
    assert "<td>model</td><td>nbdm</td>" in rendered
    assert "<td>n_cells</td><td>4</td>" in rendered
    assert "<td>n_genes</td><td>3</td>" in rendered
    assert "<td>n_steps</td><td>3</td>" in rendered
    assert "<td>guide</td><td>mean_field</td>" in rendered
    assert "params={" not in rendered


def test_mcmc_results_pickle_roundtrip_drops_unpicklable_mcmc():
    """MCMC results should roundtrip even when `_mcmc` is unpicklable."""

    class _UnpicklableMCMC:
        def __init__(self):
            # Lambdas are not picklable with stdlib pickle.
            self.bad = lambda x: x

    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("mcmc").build()
    results = ScribeMCMCResults(
        samples={
            "p": jnp.array([0.2, 0.3, 0.4]),
            "r": jnp.ones((3, 3), dtype=jnp.float32),
        },
        n_cells=4,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
        _mcmc=_UnpicklableMCMC(),
    )
    restored = _roundtrip(results)
    assert isinstance(restored, ScribeMCMCResults)
    assert restored._mcmc is None
    assert "r" in restored.samples


def test_composable_vae_results_pickle_roundtrip():
    """Composable SVI VAE results should survive pickle roundtrip."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("vae").build()
    results = ScribeVAEResults(
        params={_ENCODER_KEY: {}, _DECODER_KEY: {}},
        loss_history=jnp.array([4.0, 2.0]),
        n_cells=2,
        n_genes=3,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
        _encoder=object(),
        _decoder=object(),
        _latent_spec=_DummyLatentSpec(),
    )
    restored = _roundtrip(results)
    assert isinstance(restored, ScribeVAEResults)
    assert _ENCODER_KEY in restored.params
    assert _DECODER_KEY in restored.params


def test_de_results_pickle_roundtrip():
    """DE result subclasses should be pickle-safe."""
    parametric = ScribeParametricDEResults(
        mu_A=jnp.array([0.1, 0.2]),
        W_A=jnp.zeros((2, 1)),
        d_A=jnp.ones(2),
        mu_B=jnp.array([0.0, 0.1]),
        W_B=jnp.zeros((2, 1)),
        d_B=jnp.ones(2),
        gene_names=["g1", "g2", "g3"],
        label_A="A",
        label_B="B",
    )
    empirical = ScribeEmpiricalDEResults(
        delta_samples=jnp.array([[0.1, -0.2, 0.0], [0.2, -0.1, 0.1]]),
        gene_names=["g1", "g2", "g3"],
        label_A="A",
        label_B="B",
    )
    shrinkage = ScribeShrinkageDEResults(
        delta_samples=jnp.array([[0.1, -0.2, 0.0], [0.2, -0.1, 0.1]]),
        gene_names=["g1", "g2", "g3"],
        label_A="A",
        label_B="B",
    )

    restored_parametric = _roundtrip(parametric)
    restored_empirical = _roundtrip(empirical)
    restored_shrinkage = _roundtrip(shrinkage)
    assert isinstance(restored_parametric, ScribeParametricDEResults)
    assert isinstance(restored_empirical, ScribeEmpiricalDEResults)
    assert isinstance(restored_shrinkage, ScribeShrinkageDEResults)


def test_model_comparison_results_pickle_roundtrip():
    """Model-comparison results should survive pickle roundtrip."""
    ll_a = jnp.array(np.random.default_rng(0).normal(-2.0, 0.2, (5, 4)))
    ll_b = jnp.array(np.random.default_rng(1).normal(-2.4, 0.2, (5, 4)))
    results = ScribeModelComparisonResults(
        model_names=["A", "B"],
        log_liks_cell=[ll_a, ll_b],
        n_cells=4,
        n_genes=3,
    )
    restored = _roundtrip(results)
    assert isinstance(restored, ScribeModelComparisonResults)
    assert restored.K == 2


# =========================================================================
# AxisLayout backward compatibility tests
# =========================================================================

from scribe.core.axis_layout import AxisLayout, GENES, COMPONENTS


def test_svi_results_with_param_layouts_roundtrip():
    """New SVI results with param_layouts should survive pickle roundtrip."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
    layouts = {
        "p_loc": AxisLayout(axes=()),
        "r_loc": AxisLayout(axes=(GENES,)),
    }
    results = ScribeSVIResults(
        params={"p_loc": jnp.array(0.5), "r_loc": jnp.ones(100)},
        loss_history=jnp.array([5.0, 3.0]),
        n_cells=50,
        n_genes=100,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
        param_layouts=layouts,
    )
    restored = _roundtrip(results)
    assert restored.param_layouts is not None
    assert restored.param_layouts["r_loc"].axes == (GENES,)
    assert restored.param_layouts["p_loc"].axes == ()


def test_mcmc_results_with_param_layouts_roundtrip():
    """New MCMC results with param_layouts should survive pickle roundtrip."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("mcmc").build()
    layouts = {
        "r": AxisLayout(axes=(GENES,), has_sample_dim=True),
    }
    results = ScribeMCMCResults(
        samples={"r": jnp.ones((200, 100))},
        n_cells=50,
        n_genes=100,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
        param_layouts=layouts,
    )
    restored = _roundtrip(results)
    assert restored.param_layouts is not None
    assert restored.param_layouts["r"].axes == (GENES,)
    assert restored.param_layouts["r"].has_sample_dim is True


def test_old_pickle_without_param_layouts_falls_back():
    """Old pickles without param_layouts should still provide .layouts."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("svi").build()
    results = ScribeSVIResults(
        params={"p_loc": jnp.array(0.5), "r_loc": jnp.ones(100)},
        loss_history=jnp.array([5.0, 3.0]),
        n_cells=50,
        n_genes=100,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )
    # Simulate old pickle by stripping param_layouts from state
    state = results.__getstate__()
    state.pop("param_layouts", None)
    restored = ScribeSVIResults.__new__(ScribeSVIResults)
    restored.__setstate__(state)

    # param_layouts is None but .layouts property still works
    assert restored.param_layouts is None
    layouts = restored.layouts
    assert "r_loc" in layouts
    assert layouts["r_loc"].gene_axis == 0

    # After the first access the reconstructed layouts are cached on
    # param_layouts so repeated calls do not recompute.
    assert restored.param_layouts is not None
    assert restored.param_layouts is layouts


def test_old_mcmc_pickle_without_param_layouts_falls_back():
    """Old MCMC pickles without param_layouts should still provide .layouts."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("mcmc").build()
    results = ScribeMCMCResults(
        samples={
            "p": jnp.array([0.2, 0.3, 0.4]),
            "r": jnp.ones((3, 100)),
        },
        n_cells=50,
        n_genes=100,
        model_type="nbdm",
        model_config=cfg,
        prior_params={},
    )
    state = results.__getstate__()
    state.pop("param_layouts", None)
    restored = ScribeMCMCResults.__new__(ScribeMCMCResults)
    restored.__setstate__(state)

    assert restored.param_layouts is None
    layouts = restored.layouts
    assert "r" in layouts
    # MCMC samples have a leading sample dim
    assert layouts["r"].has_sample_dim is True
    assert layouts["r"].gene_axis == 1

    # After the first access the reconstructed layouts are cached on
    # param_layouts so repeated calls do not recompute.
    assert restored.param_layouts is not None
    assert restored.param_layouts is layouts


def test_mixture_svi_old_pickle_reconstructs_layouts():
    """Mixture model old pickle should reconstruct component+gene layouts."""
    cfg = (
        ModelConfigBuilder()
        .for_model("nbdm")
        .with_inference("svi")
        .as_mixture(n_components=3)
        .build()
    )
    results = ScribeSVIResults(
        params={
            "p_loc": jnp.ones(3),
            "r_loc": jnp.ones((3, 100)),
            "mixing_weights_loc": jnp.ones(3),
        },
        loss_history=jnp.array([5.0, 3.0]),
        n_cells=50,
        n_genes=100,
        model_type="nbdm_mix",
        model_config=cfg,
        prior_params={},
        n_components=3,
    )
    state = results.__getstate__()
    state.pop("param_layouts", None)
    restored = ScribeSVIResults.__new__(ScribeSVIResults)
    restored.__setstate__(state)

    layouts = restored.layouts
    assert layouts["r_loc"].axes == (COMPONENTS, GENES)
    assert layouts["mixing_weights_loc"].axes == (COMPONENTS,)
