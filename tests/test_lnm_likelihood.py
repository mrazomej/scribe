"""Tests for LogisticNormalMultinomialLikelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random

from scribe.core.axis_layout import AxisLayout
from scribe.models.components.likelihoods.lnm import (
    LogisticNormalMultinomialLikelihood,
)
from scribe.models.config import ModelConfigBuilder, VAEConfig


def _make_mock_vae_cell_fn(
    n_genes: int, n_cells: int, batch_size: int | None, *, latent_dim: int = 3
):
    """Create a mock ``vae_cell_fn`` that returns fake decoder output.

    Parameters
    ----------
    n_genes
        Number of genes ``G`` (decoder outputs ``G - 1`` ALR coordinates).
    n_cells
        Dataset size when the cell plate is dense (``batch_size is None``).
    batch_size
        Subsample size when using a mini-batch plate; ``None`` for full plate.
    latent_dim
        Latent dimension for the fake ``z`` site.
    """

    def vae_cell_fn(idx):
        if batch_size is not None:
            batch = int(batch_size)
        else:
            batch = int(n_cells)
        z = numpyro.sample(
            "z",
            dist.Normal(0.0, 1.0).expand([latent_dim]).to_event(1),
        )
        assert z.shape[0] == batch
        y_alr = jnp.zeros((z.shape[0], n_genes - 1))
        return {"y_alr": y_alr}

    return vae_cell_fn


@pytest.fixture
def lnm_log_prob_inputs():
    """Counts, LNM parameters, and layouts for :meth:`log_prob` shape tests."""
    n_cells, n_genes = 20, 5
    key = random.PRNGKey(42)
    counts = random.poisson(key, 10.0, shape=(n_cells, n_genes))
    params = {
        "r_T": jnp.array(100.0),
        "p": jnp.array(0.5),
        "y_alr": random.normal(key, (n_cells, n_genes - 1)),
    }
    layouts = {
        "r_T": AxisLayout(()),
        "p": AxisLayout(()),
        "y_alr": AxisLayout(()),
    }
    return counts, params, layouts


@pytest.fixture
def tiny_lnm_model_config():
    """Minimal valid ``lnm`` config for likelihood ``sample`` tests."""
    return (
        ModelConfigBuilder()
        .for_model("lnm")
        .with_parameterization("logistic_normal")
        .with_inference("vae")
        .with_vae(
            latent_dim=3,
            encoder_hidden_dims=[32],
            decoder_hidden_dims=[32],
        )
        .build()
    )


def test_alr_inverse_produces_valid_simplex():
    """Random ALR vectors map to strictly positive simplex weights summing to one."""
    rng = random.PRNGKey(0)
    likelihood = LogisticNormalMultinomialLikelihood()
    y = random.normal(rng, (4, 7))
    rho = likelihood._alr_inverse(y)
    assert rho.shape == (4, 8)
    assert jnp.all(rho > 0)
    assert jnp.allclose(rho.sum(axis=-1), 1.0)


def test_alr_inverse_shape():
    """Batch shape ``(B, G - 1)`` maps to ``(B, G)`` on the simplex."""
    likelihood = LogisticNormalMultinomialLikelihood()
    b, g = 6, 10
    y = jnp.zeros((b, g - 1))
    rho = likelihood._alr_inverse(y)
    assert rho.shape == (b, g)


def test_alr_inverse_reference_at_zero():
    """ALR coordinates equal to zero give a uniform composition (``1 / G``)."""
    likelihood = LogisticNormalMultinomialLikelihood()
    g = 8
    y = jnp.zeros((3, g - 1))
    rho = likelihood._alr_inverse(y)
    expected = jnp.full((g,), 1.0 / float(g), dtype=rho.dtype)
    assert jnp.allclose(rho, expected[None, :])


@pytest.mark.parametrize("d_mode", ["low_rank", "learned"])
def test_sample_traces_expected_sites(d_mode: str, tiny_lnm_model_config):
    """LNM ``sample`` registers the expected NumPyro sites for each ``d_mode``."""
    n_cells, n_genes, latent_dim = 10, 5, 3
    batch_size = 8
    likelihood = LogisticNormalMultinomialLikelihood(d_mode=d_mode)  # type: ignore[arg-type]
    key = random.PRNGKey(1)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    param_values: dict = {
        "r_T": jnp.array(20.0),
        "p": jnp.array(0.4),
    }
    if d_mode == "learned":
        param_values["d_lnm"] = jnp.full((n_genes - 1,), 0.25)

    def model():
        likelihood.sample(
            param_values=dict(param_values),
            cell_specs=[],
            counts=counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=batch_size,
            model_config=tiny_lnm_model_config,
            vae_cell_fn=_make_mock_vae_cell_fn(
                n_genes, n_cells, batch_size, latent_dim=latent_dim
            ),
        )

    tr = numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace()
    assert "u_T" in tr
    assert "z" in tr
    assert "counts" in tr
    if d_mode == "low_rank":
        assert "lnm_eps" not in tr
    else:
        assert "lnm_eps" in tr


def test_sample_requires_vae_cell_fn(tiny_lnm_model_config):
    """Sampling without a VAE cell function is rejected."""
    likelihood = LogisticNormalMultinomialLikelihood()
    n_cells, n_genes = 8, 5
    counts = jnp.ones((n_cells, n_genes))

    with pytest.raises(ValueError, match="requires vae_cell_fn"):
        likelihood.sample(
            param_values={"r_T": jnp.array(1.0), "p": jnp.array(0.5)},
            cell_specs=[],
            counts=counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=None,
            model_config=tiny_lnm_model_config,
            vae_cell_fn=None,
        )


def test_log_prob_cell_shape(lnm_log_prob_inputs):
    """``return_by='cell'`` yields one log-probability per cell."""
    counts, params, layouts = lnm_log_prob_inputs
    likelihood = LogisticNormalMultinomialLikelihood()
    lp = likelihood.log_prob(counts, params, layouts, return_by="cell")
    assert lp.shape == (counts.shape[0],)


def test_log_prob_gene_raises(lnm_log_prob_inputs):
    """Per-gene LNM log-probabilities are not defined."""
    counts, params, layouts = lnm_log_prob_inputs
    likelihood = LogisticNormalMultinomialLikelihood()
    with pytest.raises(NotImplementedError, match="return_by='gene'"):
        likelihood.log_prob(counts, params, layouts, return_by="gene")


def test_constructor_validates_d_mode():
    """Invalid ``d_mode`` values raise ``ValueError``."""
    with pytest.raises(ValueError, match="d_mode"):
        LogisticNormalMultinomialLikelihood(d_mode="scalar")  # type: ignore[arg-type]
