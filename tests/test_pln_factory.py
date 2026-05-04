"""End-to-end smoke tests for PLN model creation and fitting."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro
import pytest
from jax import random
from numpyro.infer import SVI, TraceMeanField_ELBO

from scribe.api import VALID_MODELS
from scribe.models.config import ModelConfigBuilder
from scribe.models.config.enums import ModelType
from scribe.models.presets.factory import create_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pln_config(*, d_mode: str = "low_rank"):
    """Build a tiny ``pln`` :class:`ModelConfig` for smoke tests."""
    built = (
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
    if d_mode != built.d_mode:
        return built.model_copy(update={"d_mode": d_mode})
    return built


# ---------------------------------------------------------------------------
# Factory: create_model
# ---------------------------------------------------------------------------


def test_create_model_pln_low_rank():
    """Factory builds a runnable low-rank PLN model and guide."""
    n_cells, n_genes = 8, 10
    config = _pln_config(d_mode="low_rank")
    model, guide, param_specs = create_model(
        config, n_genes=n_genes, validate=False
    )
    assert param_specs is not None
    counts = random.poisson(random.PRNGKey(0), 5.0, shape=(n_cells, n_genes))
    key = random.PRNGKey(0)

    # Model trace should have z and counts but not u_T.
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
    assert "u_T" not in model_trace

    # Guide trace.
    numpyro.handlers.trace(numpyro.handlers.seed(guide, key)).get_trace(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )


def test_create_model_pln_learned():
    """Factory builds a runnable learned-diagonal PLN model and guide."""
    n_cells, n_genes = 8, 10
    config = _pln_config(d_mode="learned")
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
    assert "counts" in model_trace
    # pln_eps is blocked from the trace (q = p, KL = 0 by construction).
    assert "pln_eps" not in model_trace


# ---------------------------------------------------------------------------
# SVI smoke test
# ---------------------------------------------------------------------------


def test_svi_smoke_fit_pln():
    """Run a few SVI steps to verify the PLN model trains without crashing."""
    n_cells, n_genes, k = 16, 10, 3
    key = random.PRNGKey(0)
    counts = random.poisson(key, 5.0, shape=(n_cells, n_genes))

    config = (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
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


# ---------------------------------------------------------------------------
# Enum / registry
# ---------------------------------------------------------------------------


def test_model_type_enum():
    """``ModelType`` includes PLN."""
    assert ModelType.PLN.value == "pln"


def test_valid_models_includes_pln():
    """Public API lists ``pln`` as a supported model string."""
    assert "pln" in VALID_MODELS


# ---------------------------------------------------------------------------
# Factory wiring: data-init flows into the decoder
# ---------------------------------------------------------------------------


def _walk_for_head_kernel(params, head_name="head_y_log_rate"):
    """Recursively find ``{kernel, bias}`` under ``head_name`` in a PyTree.

    Flax stores Dense parameters either flat (``{kernel, bias}``) or
    one level deeper (``{Dense_0: {kernel, bias}}``). This helper
    handles both layouts.
    """
    found = {"kernel": None, "bias": None}

    def _walk(node):
        if isinstance(node, dict):
            if head_name in node:
                head = node[head_name]
                inner = head if "kernel" in head else head.get("Dense_0", {})
                if isinstance(inner, dict):
                    if "kernel" in inner:
                        found["kernel"] = jnp.asarray(inner["kernel"])
                    if "bias" in inner:
                        found["bias"] = jnp.asarray(inner["bias"])
            for v in node.values():
                _walk(v)

    _walk(params)
    return found


def _init_pln_svi_params(config, n_cells, n_genes, counts, seed=0):
    """Initialize SVI for a PLN config and return the variational params dict.

    Encapsulates the boilerplate required to materialize flax decoder
    weights from a freshly-built PLN model so tests can introspect
    them (linear-decoder shape, PCA-init kernel, log-mean bias, etc.).
    """
    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)
    optimizer = numpyro.optim.Adam(1e-3)
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    svi_state = svi.init(
        random.PRNGKey(seed),
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )
    return svi.get_params(svi_state)


def test_pln_factory_uses_linear_decoder():
    """Factory must build a *linear* decoder for PLN, not an MLP.

    PLN's biophysics derivation requires ``y_log_rate = mu + W @ z``
    (linear in the latent) so that the kernel of the decoder dense
    layer *is* the loadings matrix ``W`` of the generative
    ``Sigma = W W^T + diag(d)``. With a hidden-MLP decoder this
    correspondence is lost (the kernel maps a *hidden representation*
    to log-rates, not the latent itself).

    We assert this structurally by reading the decoder kernel shape
    after SVI.init. With a linear decoder, kernel for
    ``head_y_log_rate`` is ``(latent_dim, n_genes)``. With a hidden
    MLP, the final dense's in-features would be ``hidden_dims[-1]``,
    not ``latent_dim``. The user's ``decoder_hidden_dims`` argument
    here is *intentionally non-empty* -- the linear-decoder override
    must *defeat* it for PLN.
    """
    n_cells, n_genes, k = 8, 10, 3
    config = (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=k,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[64, 32],  # would-be MLP hidden layers
        )
        .build()
    )
    counts = random.poisson(
        random.PRNGKey(0), 3.0, shape=(n_cells, n_genes)
    )
    params = _init_pln_svi_params(config, n_cells, n_genes, counts)
    head = _walk_for_head_kernel(params)
    assert head["kernel"] is not None, (
        "Could not locate head_y_log_rate kernel in SVI params."
    )
    assert head["kernel"].shape == (k, n_genes), (
        "Expected linear-decoder kernel shape (latent_dim, n_genes) = "
        f"({k}, {n_genes}); got {head['kernel'].shape}. The PLN branch "
        "in factory.py is not overriding decoder_hidden_dims to ()."
    )


def test_pln_factory_consumes_pca_and_log_mean_data_init():
    """When the user injects data-init, the decoder weights match it.

    Mirrors what ``api.fit`` does: compute
    ``empirical_log_mean_bias_init`` and ``pca_loadings_init`` on a
    count matrix, stash them on ``model_config.vae``, then build the
    model. We initialize SVI to materialize the decoder weights and
    assert they equal the injected initializers (up to the
    ``(G, k) -> (k, G)`` transpose convention shift the factory
    applies between the generative model and flax's Dense storage).
    If the factory forgets to wire either of these through, the
    decoder weights would be at flax's defaults instead.
    """
    import numpy as np
    from scribe.core.pln_data_init import inject_pln_vae_data_init

    n_cells, n_genes, k = 24, 8, 3
    counts = np.asarray(
        random.poisson(random.PRNGKey(7), 5.0, shape=(n_cells, n_genes))
    ).astype(np.float32)

    config = _pln_config().model_copy(
        update={
            "vae": _pln_config().vae.model_copy(
                update={"latent_dim": k}
            )
        }
    )
    config = inject_pln_vae_data_init(config, counts, latent_dim=k)
    expected_bias = jnp.asarray(config.vae.empirical_log_mean_bias_init)
    expected_kernel = jnp.asarray(config.vae.pca_loadings_init).T  # (k, G)

    params = _init_pln_svi_params(
        config, n_cells, n_genes, jnp.asarray(counts)
    )
    head = _walk_for_head_kernel(params)
    assert head["bias"] is not None, (
        "Decoder bias for y_log_rate was not initialized from "
        "PLN data-init."
    )
    assert head["kernel"] is not None, (
        "Decoder kernel for y_log_rate was not initialized from "
        "PLN data-init."
    )
    # Bias matches empirical_log_mean exactly (constant initializer).
    assert jnp.allclose(head["bias"], expected_bias, atol=1e-5)
    # Kernel matches PCA loadings (transposed) exactly.
    assert jnp.allclose(head["kernel"], expected_kernel, atol=1e-5)


# ---------------------------------------------------------------------------
# Capture-anchor wiring: end-to-end through the factory
# ---------------------------------------------------------------------------


def test_pln_factory_wires_capture_anchor_when_prior_supplied():
    """Supplying ``priors={"capture_efficiency": ...}`` activates capture
    anchor on the PLN likelihood and registers an ``eta_capture`` site
    in *both* the model and guide traces.

    Without the ``param_specs.append(pln_capture_spec)`` in the factory,
    the guide would lack ``eta_capture`` and ``TraceMeanField_ELBO`` would
    raise ``KeyError: 'eta_capture'`` during KL computation.
    """
    import numpy as np

    n_cells, n_genes, k = 16, 8, 2
    key = random.PRNGKey(11)
    counts = random.poisson(key, 4.0, shape=(n_cells, n_genes))

    config = (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=k,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .with_priors(capture_efficiency=(float(np.log(1e5)), 0.5))
        .build()
    )

    model, guide, _ = create_model(config, n_genes=n_genes, validate=False)

    trace_kwargs = dict(
        n_cells=n_cells,
        n_genes=n_genes,
        model_config=config,
        counts=counts,
    )

    # Model trace: eta_capture must be present.
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, key)
    ).get_trace(**trace_kwargs)

    assert "eta_capture" in model_trace, (
        "Capture-anchor did not activate; eta_capture site missing. "
        "Check factory wiring for is_poisson_lognormal_family."
    )
    assert "counts" in model_trace
    # PLN has no totals NB submodel.
    assert "u_T" not in model_trace

    # Guide trace: must also contain eta_capture so ELBO can compute KL.
    guide_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(guide, key)
    ).get_trace(**trace_kwargs)

    assert "eta_capture" in guide_trace, (
        "Guide is missing eta_capture site. The BiologyInformedCaptureSpec "
        "must be appended to param_specs so GuideBuilder emits a matching "
        "variational distribution."
    )


def test_svi_smoke_fit_pln_with_capture():
    """Run a few SVI steps with PLN + capture anchor to confirm training
    does not crash with ``TraceMeanField_ELBO``.
    """
    import numpy as np

    n_cells, n_genes, k = 16, 8, 2
    key = random.PRNGKey(12)
    counts = random.poisson(key, 4.0, shape=(n_cells, n_genes))

    config = (
        ModelConfigBuilder()
        .for_model("pln")
        .with_parameterization("poisson_lognormal")
        .with_inference("vae")
        .with_vae(
            latent_dim=k,
            encoder_hidden_dims=[16],
            decoder_hidden_dims=[16],
        )
        .with_priors(capture_efficiency=(float(np.log(1e5)), 0.5))
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
    assert jnp.isfinite(loss), (
        f"PLN + capture SVI loss diverged to {loss}."
    )


# ---------------------------------------------------------------------------
# Public API: end-to-end fit("pln", ...)
# ---------------------------------------------------------------------------


def test_api_fit_pln_end_to_end_low_rank():
    """``scribe.fit(model='pln', ...)`` runs end-to-end and produces results.

    Exercises the path the user actually calls in production: data-init
    injection, factory wiring, SVI loop, post-fit results object. A
    short fit (50 steps) is enough to confirm the path works without
    runtime errors and that ``get_pln_*`` extraction methods return
    well-shaped arrays.
    """
    import anndata as ad
    import numpy as np

    import scribe

    n_cells, n_genes, k = 32, 12, 3
    counts = np.asarray(
        random.poisson(random.PRNGKey(3), 5.0, shape=(n_cells, n_genes))
    ).astype(np.float32)
    adata = ad.AnnData(counts)

    results = scribe.fit(
        adata,
        model="pln",
        vae_latent_dim=k,
        vae_encoder_hidden_dims=[16],
        n_steps=50,
        seed=0,
    )
    # PLN extraction methods produce the right shapes.
    mu = results.get_pln_mu()
    W = results.get_pln_W()
    assert mu.shape == (n_genes,)
    assert W.shape == (n_genes, k)


def test_api_fit_pln_end_to_end_learned_d():
    """``d_mode='learned'`` exercises the d_pln spec wiring at fit time.

    Without the ``pln_d_specs`` block in the factory this fit would
    fail at runtime when the likelihood reads ``param_values['d_pln']``
    and finds nothing. Catching that regression is the entire point of
    this test.
    """
    import anndata as ad
    import numpy as np

    import scribe

    n_cells, n_genes, k = 32, 12, 3
    counts = np.asarray(
        random.poisson(random.PRNGKey(4), 5.0, shape=(n_cells, n_genes))
    ).astype(np.float32)
    adata = ad.AnnData(counts)

    results = scribe.fit(
        adata,
        model="pln",
        vae_latent_dim=k,
        vae_encoder_hidden_dims=[16],
        d_mode="learned",
        n_steps=50,
        seed=0,
    )
    d = results.get_pln_d()
    assert d is not None, "d_pln must be present when d_mode='learned'."
    assert d.shape == (n_genes,)
    # All entries strictly positive after the LogNormal->guide path.
    assert jnp.all(d > 0)


def test_api_fit_pln_end_to_end_with_capture():
    """``scribe.fit(model='pln', priors={'capture_efficiency': ...})``
    runs end-to-end through the full inference pipeline.

    This exercises the complete path that was previously hitting
    ``KeyError: 'eta_capture'`` because the guide lacked the
    matching variational site.
    """
    import anndata as ad
    import numpy as np

    import scribe

    n_cells, n_genes, k = 32, 12, 3
    counts = np.asarray(
        random.poisson(random.PRNGKey(5), 5.0, shape=(n_cells, n_genes))
    ).astype(np.float32)
    adata = ad.AnnData(counts)

    results = scribe.fit(
        adata,
        model="pln",
        vae_latent_dim=k,
        vae_encoder_hidden_dims=[16],
        priors={"capture_efficiency": (float(np.log(1e5)), 0.5)},
        n_steps=50,
        seed=0,
    )
    mu = results.get_pln_mu()
    W = results.get_pln_W()
    assert mu.shape == (n_genes,)
    assert W.shape == (n_genes, k)


def test_api_fit_pln_with_variable_capture_and_priors():
    """``model='pln', variable_capture=True`` with capture priors works
    end-to-end without errors or warnings.
    """
    import anndata as ad
    import numpy as np
    import warnings

    import scribe

    n_cells, n_genes, k = 32, 12, 3
    counts = np.asarray(
        random.poisson(random.PRNGKey(6), 5.0, shape=(n_cells, n_genes))
    ).astype(np.float32)
    adata = ad.AnnData(counts)

    # No warning should be emitted because capture priors are provided.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = scribe.fit(
            adata,
            model="pln",
            variable_capture=True,
            vae_latent_dim=k,
            vae_encoder_hidden_dims=[16],
            priors={"capture_efficiency": (float(np.log(1e5)), 0.5)},
            n_steps=50,
            seed=0,
        )
    mu = results.get_pln_mu()
    assert mu.shape == (n_genes,)
