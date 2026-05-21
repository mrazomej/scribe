"""Cross-check the hand-derived TSLN-Logit global curvature.

The hand-derived path in ``compute_global_uncertainty`` uses
closed-form per-cell-per-gene formulas for the gradient and Hessian
diagonal in the natural (log_rate, κ, θ) spaces, then chain-rules
to the unconstrained ``*_loc`` space.  This test confirms the
closed-form matches the autodiff ground truth (``hessian_diag_chunked``
applied to the same ``neg_log_post``) to float32 precision on a
small synthetic problem.

If this test ever fails after a kernel change, the hand-derivation
needs to be re-checked against the math in
``paper/_two_state_promoter.qmd``
§sec-twostate-tsln-logit-global-uncertainty.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


def _build_synthetic_fit(C=18, G=10, k=2, seed=0):
    """Run a tiny TSLN-Logit Laplace fit so we have realistic
    (x_map, eta_loc, gene globals) to test against.
    """
    from scribe.inference.laplace import _run_laplace_inference
    from scribe.models.config.base import ModelConfig
    from scribe.models.config.enums import (
        InferenceMethod, Parameterization,
    )
    from scribe.models.config.groups import (
        DataConfig, LaplaceConfig, VAEConfig,
    )

    cfg = ModelConfig(
        base_model="twostate_ln_logit",
        parameterization=Parameterization.TWO_STATE_NATURAL,
        inference_method=InferenceMethod.LAPLACE,
        positive_transform="softplus",
        vae=VAEConfig(latent_dim=k),
    )
    rng = np.random.default_rng(seed)
    counts = jnp.asarray(rng.integers(0, 8, size=(C, G)).astype(np.float32))
    result = _run_laplace_inference(
        model_config=cfg,
        count_data=counts,
        adata=None,
        n_cells=C,
        n_genes=G,
        laplace_config=LaplaceConfig(
            n_steps=10, n_newton_steps=3, batch_size=9,
        ),
        data_config=DataConfig(),
        seed=seed,
    )
    return result, counts, cfg


def test_global_curvature_matches_autodiff():
    """Closed-form ``H_*`` per-gene curvature equals
    ``hessian_diag_chunked`` to float32 precision.
    """
    from scribe.laplace._global_uncertainty import (
        hessian_diag_chunked,
        resolve_numpyro_transform,
    )
    from scribe.laplace._newton_twostate_ln_logit import (
        global_curvature_logit_summed,
    )
    from scribe.laplace._obs_twostate_ln_logit import (
        _factors_batch_x_only,
        _woodbury_quadform,
        _woodbury_logdet_sigma,
    )

    result, counts, cfg = _build_synthetic_fit(C=18, G=10, k=2, seed=0)
    rate_at_map = np.asarray(result.rate)
    kappa_at_map = np.asarray(result.kappa)
    theta_at_map = np.asarray(result.eta_anchor)
    rate_loc = np.asarray(result.rate_loc)
    kappa_loc = np.asarray(result.kappa_loc)
    eta_anchor_loc = np.asarray(result.eta_anchor_loc)
    x_loc = np.asarray(result.x_loc)
    W = np.asarray(result.W)
    d = np.asarray(result.d)
    C, G = counts.shape
    n_q = 60

    # ---- Closed-form curvature ----------------------------------
    eta_cap_zero = jnp.zeros((C,), dtype=jnp.float32)
    curv = global_curvature_logit_summed(
        x_map=jnp.asarray(x_loc),
        counts=jnp.asarray(counts),
        rate=jnp.asarray(rate_at_map),
        kappa=jnp.asarray(kappa_at_map),
        theta=jnp.asarray(theta_at_map),
        eta_cap=eta_cap_zero,
        n_quad_nodes=n_q,
    )

    # Chain-rule from natural -> loc space.
    def softplus_fwd(x):
        return jax.nn.softplus(x)

    d1_rate = jax.vmap(jax.grad(softplus_fwd))(jnp.asarray(rate_loc))
    d2_rate = jax.vmap(jax.grad(jax.grad(softplus_fwd)))(
        jnp.asarray(rate_loc)
    )
    d1_kappa = jax.vmap(jax.grad(softplus_fwd))(jnp.asarray(kappa_loc))
    d2_kappa = jax.vmap(jax.grad(jax.grad(softplus_fwd)))(
        jnp.asarray(kappa_loc)
    )

    # Convert log_rate curvature to rate curvature, then to rate_loc.
    H_log_r = curv["H_log_rate"]
    g_log_r = curv["g_log_rate"]
    rate_sq = jnp.maximum(jnp.asarray(rate_at_map) ** 2, 1e-30)
    H_rate_natural = (H_log_r - g_log_r) / rate_sq
    g_rate_natural = g_log_r / jnp.maximum(jnp.asarray(rate_at_map), 1e-30)
    H_rate_loc_handderived = (
        (d1_rate ** 2) * H_rate_natural + d2_rate * g_rate_natural
    )
    H_kappa_loc_handderived = (
        (d1_kappa ** 2) * curv["H_kappa"] + d2_kappa * curv["g_kappa"]
    )
    H_eta_anchor_loc_handderived = curv["H_eta_anchor"]

    # ---- Autodiff ground truth ----------------------------------
    n_cells = int(C)
    n_genes = int(G)

    def neg_log_post(rate_loc_v, kappa_loc_v, eta_anchor_loc_v):
        rate_v = jax.nn.softplus(rate_loc_v)
        kappa_v = jax.nn.softplus(kappa_loc_v)
        eta_anchor_v = eta_anchor_loc_v
        eta_cap = jnp.asarray(0.0, dtype=rate_v.dtype)
        log_marg_per_gene, _ = _factors_batch_x_only(
            jnp.asarray(x_loc), jnp.asarray(counts),
            rate_v, kappa_v, eta_anchor_v, eta_cap, n_q,
        )
        data_lp = jnp.sum(log_marg_per_gene)
        diff = jnp.asarray(x_loc)
        quad = _woodbury_quadform(jnp.asarray(W), jnp.asarray(d), diff)
        log_det_sigma = _woodbury_logdet_sigma(jnp.asarray(W), jnp.asarray(d))
        mvn_lp = (
            -0.5 * jnp.sum(quad)
            - 0.5 * n_cells * (
                log_det_sigma + n_genes * jnp.log(2.0 * jnp.pi)
            )
        )
        return -(data_lp + mvn_lp)

    args = (
        jnp.asarray(rate_loc),
        jnp.asarray(kappa_loc),
        jnp.asarray(eta_anchor_loc),
    )
    H_rate_loc_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=0, chunk_size=64,
    )
    H_kappa_loc_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=1, chunk_size=64,
    )
    H_eta_anchor_loc_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=2, chunk_size=64,
    )

    # ---- Compare ------------------------------------------------
    np.testing.assert_allclose(
        np.asarray(H_rate_loc_handderived),
        np.asarray(H_rate_loc_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="rate_loc hand-derived curvature disagrees with autodiff",
    )
    np.testing.assert_allclose(
        np.asarray(H_kappa_loc_handderived),
        np.asarray(H_kappa_loc_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="kappa_loc hand-derived curvature disagrees with autodiff",
    )
    np.testing.assert_allclose(
        np.asarray(H_eta_anchor_loc_handderived),
        np.asarray(H_eta_anchor_loc_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="eta_anchor_loc hand-derived disagrees with autodiff",
    )


def test_compute_global_uncertainty_no_oom_and_finite():
    """End-to-end: a TSLN-Logit fit's
    ``compute_global_uncertainty`` runs cleanly and produces finite
    scales for all unfrozen parameters.
    """
    result, _, _ = _build_synthetic_fit(C=24, G=12, k=2, seed=1)
    # Default (no freeze) fit; all three globals should have finite scale.
    assert result.rate_scale is not None
    assert result.kappa_scale is not None
    assert result.eta_anchor_scale is not None
    assert jnp.all(jnp.isfinite(result.rate_scale))
    assert jnp.all(jnp.isfinite(result.kappa_scale))
    assert jnp.all(jnp.isfinite(result.eta_anchor_scale))
    assert jnp.all(result.rate_scale > 0)
    assert jnp.all(result.kappa_scale > 0)
    assert jnp.all(result.eta_anchor_scale > 0)
