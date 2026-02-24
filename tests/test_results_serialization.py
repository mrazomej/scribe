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
from scribe.models.config import ModelConfigBuilder
from scribe.svi.results import ScribeSVIResults
from scribe.svi.vae_results import (
    ScribeVAEResults as ScribeComposableVAEResults,
)
from scribe.svi._latent_space import _ENCODER_KEY, _DECODER_KEY
from scribe.vae.results import ScribeVAEResults as ScribeLegacyVAEResults


class _DummyLatentSpec:
    """Minimal latent spec object with required ``flow`` attribute."""

    def __init__(self):
        self.flow = None


class _DummyLegacyVAEConfig:
    """Minimal legacy VAE config shape expected by old results class."""

    def __init__(self):
        self.inference_method = "vae"
        self.vae_prior_type = "standard"
        self.n_components = None


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


def test_legacy_vae_results_pickle_roundtrip():
    """Legacy VAE results class should be pickle-safe."""
    results = ScribeLegacyVAEResults(
        params={},
        loss_history=jnp.array([4.0, 2.0]),
        n_cells=2,
        n_genes=3,
        model_type="nbdm",
        model_config=_DummyLegacyVAEConfig(),
        prior_params={},
        _vae_model=None,
    )
    restored = _roundtrip(results)
    assert isinstance(restored, ScribeLegacyVAEResults)
    assert restored.model_config.inference_method == "vae"


def test_composable_vae_results_pickle_roundtrip():
    """Composable SVI VAE results should survive pickle roundtrip."""
    cfg = ModelConfigBuilder().for_model("nbdm").with_inference("vae").build()
    results = ScribeComposableVAEResults(
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
    assert isinstance(restored, ScribeComposableVAEResults)
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
