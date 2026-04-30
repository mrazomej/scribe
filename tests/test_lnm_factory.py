"""End-to-end smoke tests for LNM model creation and fitting."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import pytest
from jax import random
from numpyro.infer import SVI, TraceMeanField_ELBO

from scribe.api import VALID_MODELS
from scribe.models.config import ModelConfigBuilder, VAEConfig
from scribe.models.config.enums import ModelType
from scribe.models.presets.factory import create_model


def _lnm_config(*, d_mode: str = "low_rank"):
    """Build a tiny ``lnm`` :class:`ModelConfig` for smoke tests."""
    built = (
        ModelConfigBuilder()
        .for_model("lnm")
        .with_parameterization("logistic_normal")
        .with_inference("vae")
        .with_vae(
            latent_dim=2,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .build()
    )
    if d_mode != built.d_mode:
        return built.model_copy(update={"d_mode": d_mode})
    return built


def test_create_model_lnm_low_rank():
    """Factory builds a runnable low-rank LNM model and guide."""
    n_cells, n_genes = 8, 10
    config = _lnm_config(d_mode="low_rank")
    model, guide, param_specs = create_model(
        config, n_genes=n_genes, validate=False
    )
    assert param_specs
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(0)
    numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    numpyro.handlers.trace(numpyro.handlers.seed(guide, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )


def test_create_model_lnm_learned():
    """Factory builds a runnable learned-diagonal LNM model and guide."""
    n_cells, n_genes = 8, 10
    config = _lnm_config(d_mode="learned")
    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(1), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(1)
    numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    numpyro.handlers.trace(numpyro.handlers.seed(guide, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )


def test_svi_smoke_fit_low_rank():
    """Run a few SVI steps to verify the LNM model trains without crashing."""
    n_cells, n_genes, k = 16, 10, 3
    key = random.PRNGKey(0)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    config = (
        ModelConfigBuilder()
        .for_model("lnm")
        .with_parameterization("logistic_normal")
        .with_inference("vae")
        .with_vae(
            latent_dim=k,
            encoder_hidden_dims=[32],
            decoder_hidden_dims=[32],
        )
        .build()
    )

    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)

    optimizer = numpyro.optim.Adam(1e-3)
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    svi_state = svi.init(
        key,
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    for _ in range(5):
        svi_state, loss = svi.update(
            svi_state,
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )
    assert jnp.isfinite(loss)


def test_model_type_enum():
    """``ModelType`` includes the LNM-backed NBDM variant."""
    assert ModelType.LNM.value == "lnm"


def test_valid_models_includes_lnm():
    """Public API lists ``lnm`` as a supported model string."""
    assert "lnm" in VALID_MODELS


# ==============================================================================
# LNMVCP tests
# ==============================================================================


def _lnmvcp_config(*, d_mode: str = "low_rank"):
    """Build a tiny ``lnmvcp`` :class:`ModelConfig` for smoke tests."""
    built = (
        ModelConfigBuilder()
        .for_model("lnmvcp")
        .with_parameterization("logistic_normal")
        .with_inference("vae")
        .with_vae(
            latent_dim=2,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .build()
    )
    if d_mode != built.d_mode:
        return built.model_copy(update={"d_mode": d_mode})
    return built


def test_create_model_lnmvcp_low_rank():
    """Factory builds a runnable low-rank LNMVCP model and guide."""
    n_cells, n_genes = 8, 10
    config = _lnmvcp_config(d_mode="low_rank")
    model, guide, param_specs = create_model(
        config, n_genes=n_genes, validate=False
    )
    assert param_specs

    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(0)

    # Model trace: p_capture site must appear.
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, key)
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    assert (
        "p_capture" in model_trace
    ), "LNMVCP model trace must contain 'p_capture' site"
    assert "u_T" in model_trace
    assert "counts" in model_trace


def test_create_model_lnmvcp_learned():
    """Factory builds a runnable learned-diagonal LNMVCP model and guide."""
    n_cells, n_genes = 8, 10
    config = _lnmvcp_config(d_mode="learned")
    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(random.PRNGKey(1), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(1)

    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, key)
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    assert "p_capture" in model_trace
    assert "lnm_eps" in model_trace


def test_svi_smoke_fit_lnmvcp():
    """Run a few SVI steps to verify the LNMVCP model trains without crashing."""
    n_cells, n_genes, k = 16, 10, 3
    key = random.PRNGKey(0)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    config = (
        ModelConfigBuilder()
        .for_model("lnmvcp")
        .with_parameterization("logistic_normal")
        .with_inference("vae")
        .with_vae(
            latent_dim=k,
            encoder_hidden_dims=[32],
            decoder_hidden_dims=[32],
        )
        .build()
    )

    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)

    optimizer = numpyro.optim.Adam(1e-3)
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    svi_state = svi.init(
        key,
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    for _ in range(5):
        svi_state, loss = svi.update(
            svi_state,
            n_cells=n_cells,
            n_genes=n_genes,
            model_config=config,
            counts=counts,
        )
    assert jnp.isfinite(loss)


def test_lnmvcp_enum():
    """``ModelType`` includes ``LNMVCP``."""
    assert ModelType.LNMVCP.value == "lnmvcp"


def test_valid_models_includes_lnmvcp():
    """Public API lists ``lnmvcp`` as a supported model string."""
    assert "lnmvcp" in VALID_MODELS
