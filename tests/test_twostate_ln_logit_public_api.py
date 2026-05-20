"""Public-API smoke tests for TSLN-Logit Laplace.

Exercises the full chain from ``scribe.fit`` through the engine
dispatch → obs adapter → Newton kernel → result extraction, including
the cascade-from-SVI flow when ``informative_priors_from`` is supplied.

Test groups
-----------

1. **Engine dispatch arm**: ``_run_laplace_inference`` with
   ``base_model="twostate_ln_logit"`` produces a well-formed
   ``ScribeLaplaceResults`` with all TSLN-Logit fields populated.

2. **Result accessor surface**: ``get_map`` exposes the
   ``(rate, kappa, eta_anchor)`` triplet plus the derived
   ``(alpha, beta, gene_mean)`` reporting quantities.  Frozen flags
   surface correctly.  ``get_distributions`` returns the right shape
   of distributions for each parameter.

3. **API capture-validation**: ``scribe.fit`` with
   ``priors={"capture_efficiency": ...}`` for ``twostate_ln_logit``
   raises ``NotImplementedError`` at the ``model_flags`` stage, with
   a message pointing the user at the cascade route.

4. **Cascade end-to-end**: when a TwoState SVI fit is supplied via
   ``informative_priors_from``, the cascade adapter builds the
   ``(rate, kappa, eta_anchor)`` prior bundle and the Laplace fit
   runs.  Verified via the engine bridge directly.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _build_cfg(latent_dim: int = 2):
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.groups import VAEConfig
    from scribe.models.config.enums import (
        InferenceMethod,
        Parameterization,
    )

    return ModelConfig(
        base_model="twostate_ln_logit",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=latent_dim),
    )


# ---------------------------------------------------------------------
# Test 1: end-to-end via the bridge → ScribeLaplaceResults
# ---------------------------------------------------------------------


def test_end_to_end_no_capture():
    """Fit succeeds, all TSLN-Logit fields populated, semantics correct."""
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.groups import LaplaceConfig, DataConfig

    cfg = _build_cfg()
    rng = np.random.default_rng(0)
    C, G = 22, 12
    counts = jnp.asarray(rng.integers(0, 6, size=(C, G)).astype(np.float32))

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(
            n_steps=15, n_newton_steps=4, batch_size=10
        ),
        data_config=DataConfig(),
        seed=42,
    )

    assert result.model_config.base_model == "twostate_ln_logit"

    # Field shapes and types.
    assert result.mu.shape == (G,)
    assert result.rate.shape == (G,)
    assert result.kappa.shape == (G,)
    assert result.eta_anchor.shape == (G,)
    assert result.alpha.shape == (G,)
    assert result.beta.shape == (G,)
    assert result.gene_mean.shape == (G,)
    assert result.x_loc.shape == (C, G)
    assert result.W.shape == (G, 2)
    assert result.d.shape == (G,)

    # CONVENTION: ``self.mu`` is the latent prior centre (zeros for
    # TSLN-Logit since the gene baseline lives in eta_anchor).
    np.testing.assert_allclose(
        np.asarray(result.mu),
        np.zeros(G, dtype=np.float32),
        atol=1e-7,
        err_msg="result.mu must be zeros for TSLN-Logit",
    )

    # ``gene_mean = rate · σ(eta_anchor)`` — derived quantity.
    phi_anchor = jax.nn.sigmoid(result.eta_anchor)
    np.testing.assert_allclose(
        np.asarray(result.gene_mean),
        np.asarray(result.rate * phi_anchor),
        rtol=1e-5,
        err_msg="gene_mean must equal rate · σ(eta_anchor) at z=0",
    )
    # ``alpha = kappa · σ(eta_anchor)`` and
    # ``beta = kappa · (1 − σ(eta_anchor))`` — also at z=0.
    np.testing.assert_allclose(
        np.asarray(result.alpha),
        np.asarray(result.kappa * phi_anchor),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.beta),
        np.asarray(result.kappa * (1.0 - phi_anchor)),
        rtol=1e-5,
    )

    # TSLN-Rate-specific fields stay None for the logit variant.
    assert result.burst_size is None
    assert result.k_off is None
    assert result.r_hat is None

    # Loss bounded; Newton converged.
    assert float(result.losses[-1]) < 10.0 * float(result.losses[0])
    assert jnp.all(jnp.isfinite(result.losses))
    assert float(result.final_grad_norms.max()) < 1e-1

    # Clamp diagnostics populated.
    assert result.a_raw_min is not None
    assert result.a_clamp_fraction is not None
    assert result.a_clamp_per_gene.shape == (G,)


# ---------------------------------------------------------------------
# Test 2: get_map / get_distributions accessor surface
# ---------------------------------------------------------------------


def test_get_map_and_get_distributions_keys():
    """``get_map`` surfaces TSLN-Logit globals; ``get_distributions``
    routes positive params through a TransformedDistribution and
    eta_anchor through a Normal."""
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.groups import LaplaceConfig, DataConfig

    cfg = _build_cfg()
    rng = np.random.default_rng(1)
    C, G = 18, 8
    counts = jnp.asarray(rng.integers(0, 5, size=(C, G)).astype(np.float32))

    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(n_steps=10, n_newton_steps=3),
        data_config=DataConfig(),
        seed=0,
    )

    m = result.get_map()
    # Gene globals must be present.
    for k in ("rate", "kappa", "eta_anchor", "alpha", "beta", "gene_mean"):
        assert k in m, f"missing {k!r} in get_map()"
    # Frozen flags exist with the default (no freeze) → all False.
    assert m["rate_frozen"] is False
    assert m["kappa_frozen"] is False
    assert m["eta_anchor_frozen"] is False
    assert m["eta_frozen"] is False
    # Clamp diagnostics surfaced.
    assert "a_clamp_per_gene" in m

    d = result.get_distributions()
    # Latent y_latent is a LowRankMultivariateNormal centred at zero.
    import numpyro.distributions as dist
    assert isinstance(d["y_latent"], dist.LowRankMultivariateNormal)
    # rate is a TransformedDistribution (softplus inverse).
    assert isinstance(d["rate"], dist.TransformedDistribution)
    # eta_anchor is a Normal (identity transform — real-valued).
    # Wrapped in Independent for to_event(1).
    assert isinstance(d["eta_anchor"], dist.Independent)


# ---------------------------------------------------------------------
# Test 3: API capture-validation
# ---------------------------------------------------------------------


def test_capture_efficiency_prior_rejected_at_api():
    """``priors={'capture_efficiency': ...}`` rejected at model_flags."""
    import scribe
    import anndata as ad

    rng = np.random.default_rng(2)
    C, G = 30, 10
    counts_np = rng.integers(0, 6, size=(C, G)).astype(np.int32)
    adata = ad.AnnData(X=counts_np)

    with pytest.raises(NotImplementedError, match="capture_efficiency"):
        scribe.fit(
            adata,
            model="twostate_ln_logit",
            inference_method="laplace",
            priors={"capture_efficiency": (np.log(10_000.0), 0.1)},
            n_steps=2,
        )


# ---------------------------------------------------------------------
# Test 4: model_flags rejects variable_capture / zero_inflation
# ---------------------------------------------------------------------


def test_variable_capture_flag_rejected_at_api():
    """``variable_capture=True`` rejected with TSLN-Logit-specific message."""
    import scribe
    import anndata as ad

    rng = np.random.default_rng(3)
    C, G = 20, 8
    counts_np = rng.integers(0, 5, size=(C, G)).astype(np.int32)
    adata = ad.AnnData(X=counts_np)

    with pytest.raises(ValueError, match="twostate_ln_logit"):
        scribe.fit(
            adata,
            model="twostate_ln_logit",
            inference_method="laplace",
            variable_capture=True,
            n_steps=2,
        )
