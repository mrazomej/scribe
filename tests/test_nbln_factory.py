"""End-to-end smoke tests for NB-LogNormal model creation and fitting.

Critically, this module includes a *regression test* for the bug where
the factory's ``is_poisson_lognormal_family`` branch hardcoded
``PoissonLogNormalLikelihood`` instead of dispatching on ``base_model``.
That bug caused ``model="nbln"`` to silently fit a PLN model with an
unused ``r_g`` parameter.  See ``test_nbln_factory_uses_nbln_likelihood``
below.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import pytest
from jax import random

from scribe.models.components.likelihoods import (
    NBLogNormalLikelihood,
    PoissonLogNormalLikelihood,
)
from scribe.models.config import ModelConfigBuilder
from scribe.models.presets.factory import create_model
from scribe.models.presets.registry import LIKELIHOOD_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nbln_config(*, d_mode: str = "low_rank"):
    """Build a tiny ``nbln`` :class:`ModelConfig` for smoke tests."""
    built = (
        ModelConfigBuilder()
        .for_model("nbln")
        .with_parameterization("poisson_lognormal")
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


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


def test_likelihood_registry_routes_nbln():
    """``LIKELIHOOD_REGISTRY['nbln']`` must point to ``NBLogNormalLikelihood``.

    A regression test against the original bug where the registry was
    plumbed but the factory short-circuited it for the
    ``poisson_lognormal`` family and used ``PoissonLogNormalLikelihood``.
    """
    assert LIKELIHOOD_REGISTRY["nbln"] is NBLogNormalLikelihood


def test_nbln_in_api_valid_models():
    """``"nbln"`` must be in :data:`scribe.api.constants.VALID_MODELS`.

    Regression test for a real user-reported bug: the API layer's
    ``validate_inputs`` stage rejects unknown model strings against
    ``VALID_MODELS``, and a previous wiring pass added NBLN to the
    factory and registry but missed this set -- so
    ``scribe.fit(model="nbln", ...)`` blew up with "Unknown model:
    'nbln'" before ever reaching the model factory.

    Factory tests that call ``create_model`` directly (like the ones
    above) bypass this validation, which is exactly why the bug
    slipped through.
    """
    from scribe.api.constants import VALID_MODELS

    assert "nbln" in VALID_MODELS, (
        "nbln must be in VALID_MODELS so scribe.fit(model='nbln', ...) "
        "passes the API entry-point validation. Without this, the bug "
        "regression test above doesn't help -- users never reach the "
        "factory."
    )


# ---------------------------------------------------------------------------
# Factory: create_model
# ---------------------------------------------------------------------------


def test_create_model_nbln_low_rank():
    """Factory builds a runnable low-rank NBLN model and guide."""
    n_cells, n_genes = 8, 10
    config = _nbln_config(d_mode="low_rank")
    model, guide, param_specs = create_model(
        config, n_genes=n_genes, validate=False
    )
    assert param_specs is not None
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(0)

    # Model trace should have z, r (gene dispersion), and counts.
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, key)
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    assert "z" in model_trace
    assert "counts" in model_trace
    # The ``r`` site is what distinguishes NBLN from PLN. Its absence
    # would mean the factory built a PLN likelihood by mistake.
    assert "r" in model_trace, (
        "NBLN model must sample the gene dispersion 'r'. Its absence "
        "indicates the factory's poisson_lognormal branch silently "
        "fell through to PoissonLogNormalLikelihood."
    )
    # NBLN does not have a totals submodel.
    assert "u_T" not in model_trace

    # Guide trace.
    numpyro.handlers.trace(numpyro.handlers.seed(guide, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )


def test_create_model_nbln_learned():
    """Factory builds a runnable learned-diagonal NBLN model and guide."""
    n_cells, n_genes = 8, 10
    config = _nbln_config(d_mode="learned")
    model, guide, param_specs = create_model(
        config, n_genes=n_genes, validate=False
    )
    assert param_specs is not None
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(0)

    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, key)
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    assert "z" in model_trace
    assert "r" in model_trace
    assert "counts" in model_trace
    # Learned d_mode samples a gene-specific log-rate diagonal ``d_nbln``
    # outside the cell plate. The per-cell ``nbln_eps`` standard normal
    # is sampled inside ``numpyro.handlers.block()`` so it does NOT
    # appear in the trace.
    assert "d_nbln" in model_trace
    # The ``d_pln`` site must NOT appear -- that would mean the factory
    # leaked the PLN d-name into NBLN.
    assert "d_pln" not in model_trace


# ---------------------------------------------------------------------------
# Regression test for the High-2 audit finding.
# ---------------------------------------------------------------------------


def test_nbln_factory_uses_nbln_likelihood():
    """Regression test: ``model="nbln"`` must construct ``NBLogNormalLikelihood``.

    The original implementation registered ``NBLogNormalLikelihood`` in
    ``LIKELIHOOD_REGISTRY["nbln"]`` but the factory's
    ``is_poisson_lognormal_family`` branch ignored the registry and
    hardcoded ``PoissonLogNormalLikelihood``. The bug was silent: NBLN
    fits would actually run PLN. Asserting that the model trace samples
    the ``r`` site (which PLN never does) is the simplest way to catch
    a recurrence.
    """
    n_cells, n_genes = 4, 6
    config = _nbln_config(d_mode="low_rank")
    model, _, _ = create_model(config, n_genes=n_genes, validate=False)
    counts = random.poisson(
        random.PRNGKey(0), 5.0, shape=(n_cells, n_genes)
    )

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, random.PRNGKey(0))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )

    # The ``r`` site is the unmistakable NBLN signature.
    assert "r" in trace, (
        "Bug regression: NBLN model trace is missing 'r'. The factory "
        "fell through to PoissonLogNormalLikelihood — exactly the "
        "silent bug the audit caught."
    )
    # Gene dispersion has shape (n_genes,).
    r_value = trace["r"]["value"]
    assert r_value.shape == (n_genes,), (
        f"Expected r_g shape ({n_genes},), got {r_value.shape}"
    )
    # All sampled r should be strictly positive.
    assert jnp.all(r_value > 0), "r_g must be positive"


def test_nbln_distinct_from_pln_in_trace():
    """The NBLN trace must include 'r'; the PLN trace must not."""
    n_cells, n_genes = 4, 6

    # NBLN
    cfg_nbln = _nbln_config()
    m_nbln, _, _ = create_model(cfg_nbln, n_genes=n_genes, validate=False)
    counts = random.poisson(
        random.PRNGKey(0), 5.0, shape=(n_cells, n_genes)
    )
    trace_nbln = numpyro.handlers.trace(
        numpyro.handlers.seed(m_nbln, random.PRNGKey(1))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=cfg_nbln,
        counts=counts,
    )

    # PLN
    cfg_pln = (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=2,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .build()
    )
    m_pln, _, _ = create_model(cfg_pln, n_genes=n_genes, validate=False)
    trace_pln = numpyro.handlers.trace(
        numpyro.handlers.seed(m_pln, random.PRNGKey(1))
    ).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=cfg_pln,
        counts=counts,
    )

    assert "r" in trace_nbln
    assert "r" not in trace_pln, (
        "PLN must not sample 'r' — if it does, the dispatch is "
        "leaking NBLN parameters into PLN."
    )
