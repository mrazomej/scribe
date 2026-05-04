"""Tests for PoissonLogNormalLikelihood."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest
from jax import random

from scribe.core.axis_layout import AxisLayout
from scribe.models.components.likelihoods.pln import (
    PoissonLogNormalLikelihood,
    _LOG_RATE_MAX,
    _LOG_RATE_MIN,
)
from scribe.models.config import ModelConfigBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_vae_cell_fn(
    n_genes: int, n_cells: int, batch_size: int | None, *, latent_dim: int = 3
):
    """Create a mock ``vae_cell_fn`` that returns fake decoder output.

    Returns a G-dimensional ``y_log_rate`` (not G-1 like LNM's y_alr).
    """

    def vae_cell_fn(idx):
        if batch_size is not None:
            batch = int(batch_size)
        else:
            batch = int(n_cells)
        # Sample latent z (encoder output).
        z = numpyro.sample(
            "z",
            dist.Normal(0.0, 1.0).expand([latent_dim]).to_event(1),
        )
        assert z.shape[0] == batch
        # Return G-dimensional log-rates (identity decoder head).
        y_log_rate = jnp.zeros((z.shape[0], n_genes))
        return {"y_log_rate": y_log_rate}

    return vae_cell_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pln_log_prob_inputs():
    """Counts, PLN parameters, and layouts for :meth:`log_prob` shape tests."""
    n_cells, n_genes = 20, 5
    key = random.PRNGKey(42)
    counts = random.poisson(key, 10.0, shape=(n_cells, n_genes))
    params = {
        "y_log_rate": random.normal(key, (n_cells, n_genes)),
    }
    layouts = {
        "y_log_rate": AxisLayout(()),
    }
    return counts, params, layouts


@pytest.fixture
def tiny_pln_model_config():
    """Minimal valid ``pln`` config for likelihood ``sample`` tests."""
    return (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=3,
            encoder_hidden_dims=[32],
            decoder_hidden_dims=[32],
        )
        .build()
    )


# ---------------------------------------------------------------------------
# Log-rate clamping
# ---------------------------------------------------------------------------


def test_clamp_log_rate_clips_extreme_values():
    """Verify ``_clamp_log_rate`` clips values to the expected range."""
    likelihood = PoissonLogNormalLikelihood()
    y = jnp.array([-100.0, -30.0, 0.0, 30.0, 100.0])
    clamped = likelihood._clamp_log_rate(y)
    expected = jnp.array([_LOG_RATE_MIN, _LOG_RATE_MIN, 0.0, _LOG_RATE_MAX, _LOG_RATE_MAX])
    assert jnp.allclose(clamped, expected)


def test_clamp_log_rate_preserves_safe_values():
    """Values within bounds pass through unchanged."""
    likelihood = PoissonLogNormalLikelihood()
    y = jnp.array([-5.0, 0.0, 5.0, 10.0])
    clamped = likelihood._clamp_log_rate(y)
    assert jnp.allclose(clamped, y)


# ---------------------------------------------------------------------------
# sample() trace sites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d_mode", ["low_rank", "learned"])
def test_sample_traces_expected_sites(d_mode: str, tiny_pln_model_config):
    """PLN ``sample`` registers the expected NumPyro sites for each ``d_mode``."""
    n_cells, n_genes, latent_dim = 10, 5, 3
    batch_size = 8
    likelihood = PoissonLogNormalLikelihood(d_mode=d_mode)
    key = random.PRNGKey(1)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    param_values: dict = {}
    if d_mode == "learned":
        # d_pln is G-dimensional (not G-1 like LNM's d_lnm).
        param_values["d_pln"] = jnp.full((n_genes,), 0.25)

    def model():
        likelihood.sample(
            param_values=dict(param_values),
            cell_specs=[],
            counts=counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=batch_size,
            model_config=tiny_pln_model_config,
            vae_cell_fn=_make_mock_vae_cell_fn(
                n_genes, n_cells, batch_size, latent_dim=latent_dim
            ),
        )

    tr = numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace()
    # PLN has no u_T (total counts) -- counts site directly.
    assert "u_T" not in tr
    assert "z" in tr
    assert "counts" in tr
    # pln_eps is blocked from the trace (same pattern as lnm_eps).
    assert "pln_eps" not in tr


def test_sample_without_batch_size(tiny_pln_model_config):
    """Full plate mode (``batch_size=None``) runs without error."""
    n_cells, n_genes, latent_dim = 8, 5, 3
    likelihood = PoissonLogNormalLikelihood()
    key = random.PRNGKey(2)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    def model():
        likelihood.sample(
            param_values={},
            cell_specs=[],
            counts=counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=None,
            model_config=tiny_pln_model_config,
            vae_cell_fn=_make_mock_vae_cell_fn(
                n_genes, n_cells, None, latent_dim=latent_dim
            ),
        )

    tr = numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace()
    assert "counts" in tr
    assert "z" in tr


def test_sample_prior_predictive(tiny_pln_model_config):
    """Prior predictive mode (``counts=None``) generates samples."""
    n_cells, n_genes, latent_dim = 8, 5, 3
    likelihood = PoissonLogNormalLikelihood()
    key = random.PRNGKey(3)

    def model():
        likelihood.sample(
            param_values={},
            cell_specs=[],
            counts=None,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=None,
            model_config=tiny_pln_model_config,
            vae_cell_fn=_make_mock_vae_cell_fn(
                n_genes, n_cells, None, latent_dim=latent_dim
            ),
        )

    tr = numpyro.handlers.trace(numpyro.handlers.seed(model, key)).get_trace()
    assert "counts" in tr
    # Prior predictive: counts should be sampled (not observed).
    assert not tr["counts"]["is_observed"]


def test_sample_requires_vae_cell_fn(tiny_pln_model_config):
    """Sampling without a VAE cell function is rejected."""
    likelihood = PoissonLogNormalLikelihood()
    n_cells, n_genes = 8, 5
    counts = jnp.ones((n_cells, n_genes))

    with pytest.raises(ValueError, match="requires vae_cell_fn"):
        likelihood.sample(
            param_values={},
            cell_specs=[],
            counts=counts,
            dims={"n_cells": n_cells, "n_genes": n_genes},
            batch_size=None,
            model_config=tiny_pln_model_config,
            vae_cell_fn=None,
        )


# ---------------------------------------------------------------------------
# log_prob
# ---------------------------------------------------------------------------


def test_log_prob_cell_shape(pln_log_prob_inputs):
    """``return_by='cell'`` yields one log-probability per cell."""
    counts, params, layouts = pln_log_prob_inputs
    likelihood = PoissonLogNormalLikelihood()
    lp = likelihood.log_prob(counts, params, layouts, return_by="cell")
    assert lp.shape == (counts.shape[0],)


def test_log_prob_gene_shape(pln_log_prob_inputs):
    """``return_by='gene'`` yields one log-probability per gene."""
    counts, params, layouts = pln_log_prob_inputs
    likelihood = PoissonLogNormalLikelihood()
    lp = likelihood.log_prob(counts, params, layouts, return_by="gene")
    assert lp.shape == (counts.shape[1],)


def test_log_prob_finite(pln_log_prob_inputs):
    """Log-probabilities are finite for reasonable inputs."""
    counts, params, layouts = pln_log_prob_inputs
    likelihood = PoissonLogNormalLikelihood()
    lp = likelihood.log_prob(counts, params, layouts, return_by="cell")
    assert jnp.all(jnp.isfinite(lp))


def test_log_prob_clamping_prevents_overflow():
    """Extreme log-rates don't cause inf/nan in log_prob."""
    likelihood = PoissonLogNormalLikelihood()
    n_cells, n_genes = 5, 3
    counts = jnp.ones((n_cells, n_genes))
    # Extreme positive log-rates that would overflow exp() without clamping.
    params = {"y_log_rate": jnp.full((n_cells, n_genes), 100.0)}
    layouts = {"y_log_rate": AxisLayout(())}
    lp = likelihood.log_prob(counts, params, layouts, return_by="cell")
    assert jnp.all(jnp.isfinite(lp))


def test_log_prob_split_components_raises(pln_log_prob_inputs):
    """PLN does not support split_components."""
    counts, params, layouts = pln_log_prob_inputs
    likelihood = PoissonLogNormalLikelihood()
    with pytest.raises(NotImplementedError, match="split_components"):
        likelihood.log_prob(
            counts, params, layouts, split_components=True
        )


def test_log_prob_weighted_raises(pln_log_prob_inputs):
    """PLN does not support weighted log_prob."""
    counts, params, layouts = pln_log_prob_inputs
    likelihood = PoissonLogNormalLikelihood()
    with pytest.raises(NotImplementedError, match="weighted"):
        likelihood.log_prob(
            counts, params, layouts, weights=jnp.ones(5)
        )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_validates_d_mode():
    """Invalid ``d_mode`` values raise ``ValueError``."""
    with pytest.raises(ValueError, match="d_mode"):
        PoissonLogNormalLikelihood(d_mode="scalar")


def test_constructor_capture_anchor_requires_spec():
    """``capture_anchor=True`` without ``biology_informed_spec`` raises."""
    with pytest.raises(ValueError, match="biology_informed_spec"):
        PoissonLogNormalLikelihood(capture_anchor=True)
