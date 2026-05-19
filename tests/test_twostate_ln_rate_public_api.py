"""Public-API smoke tests for TSLN-Rate Laplace.

Hits the user-facing entry points (``scribe.fit`` and the internal
``_run_laplace_inference`` bridge) end-to-end to verify the
``twostate_ln_rate`` base model is wired correctly through every
stage of the pipeline:

- The bridge guard accepts ``base_model="twostate_ln_rate"``.
- ``LaplaceInferenceEngine.run_inference`` dispatches to the new
  observation model.
- ``_format_laplace_results`` builds a complete ``ScribeLaplaceResults``
  with the TSLN-specific fields (``gene_mean``, ``burst_size``,
  ``k_off``, ``alpha``, ``beta``, ``r_hat``) and the clamp diagnostics.
- ``self.mu`` carries the latent log-rate prior center (= log(r_hat))
  per the round-5 convention, distinct from the user-facing
  ``self.gene_mean`` (positive TwoState mean).
- ``get_map`` / ``get_distributions`` return well-formed dicts.
- ``model_flags`` rejects ``variable_capture`` / ``zero_inflation``
  for ``twostate_ln_rate``.

These tests do NOT exercise the high-level ``scribe.fit`` wrapper
directly (that pulls in the SVI side, AnnData adapters, and a stack
of config-builders that aren't strictly needed for the math
verification — see test_twostate_ln_rate_cascade.py for the SVI-side
smoke).  They use ``_run_laplace_inference`` directly with a
hand-built ``ModelConfig`` to keep the surface narrow.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _build_cfg(positive_transform: str = "softplus", latent_dim: int = 2):
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.groups import VAEConfig
    from scribe.models.config.enums import (
        InferenceMethod,
        Parameterization,
    )

    return ModelConfig(
        base_model="twostate_ln_rate",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform=positive_transform,
        vae=VAEConfig(latent_dim=latent_dim),
    )


# ---------------------------------------------------------------------
# Test 1: end-to-end via the bridge → ScribeLaplaceResults
# ---------------------------------------------------------------------


def test_end_to_end_no_capture():
    """Fit succeeds, all expected fields populated, semantics correct."""
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.groups import LaplaceConfig, DataConfig

    cfg = _build_cfg()
    rng = np.random.default_rng(0)
    C, G = 24, 15
    counts = jnp.asarray(rng.integers(0, 6, size=(C, G)).astype(np.float32))

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(
            n_steps=15, n_newton_steps=4, batch_size=8
        ),
        data_config=DataConfig(),
        seed=42,
    )

    # base_model is correctly preserved.
    assert result.model_config.base_model == "twostate_ln_rate"

    # Field shapes and types.
    assert result.mu.shape == (G,)
    assert result.gene_mean.shape == (G,)
    assert result.burst_size.shape == (G,)
    assert result.k_off.shape == (G,)
    assert result.alpha.shape == (G,)
    assert result.beta.shape == (G,)
    assert result.r_hat.shape == (G,)
    assert result.x_loc.shape == (C, G)
    assert result.W.shape == (G, 2)
    assert result.d.shape == (G,)

    # Round-5 convention: ``self.mu`` is the latent log-rate prior
    # center == log(r_hat), NOT the positive TwoState mu.
    np.testing.assert_allclose(
        np.asarray(result.mu),
        np.asarray(jnp.log(result.r_hat)),
        atol=1e-5,
        err_msg="result.mu must be log(r_hat) for TSLN-Rate (round-5 convention)",
    )

    # NBLN-specific fields are None.
    assert result.r is None
    assert result.r_loc is None
    assert result.r_scale is None
    assert result.mu_loc is None
    assert result.mu_scale is None

    # TSLN-Rate uncertainty (compute_global_uncertainty populated).
    assert result.gene_mean_loc is not None
    assert result.gene_mean_scale is not None
    assert result.burst_size_loc is not None
    assert result.burst_size_scale is not None
    assert result.k_off_loc is not None
    assert result.k_off_scale is not None
    assert jnp.all(jnp.isfinite(result.gene_mean_scale))
    assert jnp.all(result.gene_mean_scale > 0)
    assert jnp.all(jnp.isfinite(result.burst_size_scale))
    assert jnp.all(result.burst_size_scale > 0)
    assert jnp.all(jnp.isfinite(result.k_off_scale))
    assert jnp.all(result.k_off_scale > 0)

    # Clamp diagnostics populated.
    assert result.a_raw_min is not None
    assert result.a_clamp_fraction is not None
    assert result.a_clamp_per_gene is not None
    assert result.a_clamp_per_gene.shape == (G,)

    # Loss is bounded — should not have diverged (10x growth would
    # indicate optimizer instability).  Mini-batch noise on a short
    # (15-step) fit can produce small step-to-step increases, so
    # don't insist on monotone decrease.
    assert float(result.losses[-1]) < 10.0 * float(result.losses[0])
    assert jnp.all(jnp.isfinite(result.losses))

    # Newton converged on the per-cell MAPs.
    assert float(result.final_grad_norms.max()) < 1e-2


# ---------------------------------------------------------------------
# Test 2: get_map / get_distributions surface the right keys
# ---------------------------------------------------------------------


def test_get_map_and_get_distributions_keys():
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.groups import LaplaceConfig, DataConfig

    cfg = _build_cfg()
    rng = np.random.default_rng(1)
    C, G = 20, 12
    counts = jnp.asarray(rng.integers(0, 6, size=(C, G)).astype(np.float32))

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=10, n_newton_steps=4),
        data_config=DataConfig(),
        seed=0,
    )

    m = result.get_map()
    expected_subset = {
        "mu", "W", "d_tsln", "y_log_rate",
        "gene_mean", "burst_size", "k_off",
        "alpha", "beta", "r_hat",
        "mu_frozen", "burst_size_frozen", "k_off_frozen", "eta_frozen",
    }
    assert expected_subset <= set(m.keys()), (
        f"Missing keys: {expected_subset - set(m.keys())}"
    )
    # Convention: m["mu"] is the latent log-rate (NBLN-style), m["gene_mean"]
    # is the positive TwoState mean.
    np.testing.assert_allclose(
        np.asarray(m["mu"]), np.asarray(jnp.log(m["r_hat"])), atol=1e-5
    )

    dists = result.get_distributions()
    assert "y_log_rate" in dists
    assert "gene_mean" in dists
    # Latent log-rate distribution loc = log(r_hat).
    np.testing.assert_allclose(
        np.asarray(dists["y_log_rate"].loc),
        np.asarray(jnp.log(result.r_hat)),
        atol=1e-5,
    )


# ---------------------------------------------------------------------
# Test 3: model_flags rejects variable_capture / zero_inflation
# ---------------------------------------------------------------------


def test_model_flags_rejects_variable_capture():
    from scribe.api.context import FitContext
    from scribe.api.stages.model_flags import resolve_model_flags

    ctx = FitContext(
        model="twostate_ln_rate",
        kwargs={"variable_capture": True},
        priors=None,
    )
    with pytest.raises(ValueError, match="does not accept variable_capture"):
        resolve_model_flags(ctx)

    ctx2 = FitContext(
        model="twostate_ln_rate",
        kwargs={"zero_inflation": True},
        priors=None,
    )
    with pytest.raises(ValueError, match="does not accept variable_capture"):
        resolve_model_flags(ctx2)


# ---------------------------------------------------------------------
# Test 4: gene subsetting preserves TSLN-Rate fields
# ---------------------------------------------------------------------


def test_gene_subsetting():
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.groups import LaplaceConfig, DataConfig

    cfg = _build_cfg()
    rng = np.random.default_rng(2)
    C, G = 16, 10
    counts = jnp.asarray(rng.integers(0, 6, size=(C, G)).astype(np.float32))

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=10, n_newton_steps=4),
        data_config=DataConfig(),
        seed=0,
    )

    # Subset to genes 0, 2, 5
    sub = result[[0, 2, 5]]
    assert sub.n_genes == 3
    assert sub.mu.shape == (3,)
    assert sub.gene_mean.shape == (3,)
    assert sub.burst_size.shape == (3,)
    assert sub.k_off.shape == (3,)
    assert sub.alpha.shape == (3,)
    assert sub.beta.shape == (3,)
    assert sub.r_hat.shape == (3,)
    assert sub.a_clamp_per_gene.shape == (3,)
    assert sub.gene_mean_scale.shape == (3,)
    assert sub.x_loc.shape == (C, 3)

    # Values match the original at those indices.
    np.testing.assert_allclose(
        np.asarray(sub.gene_mean),
        np.asarray(result.gene_mean[jnp.asarray([0, 2, 5])]),
    )
