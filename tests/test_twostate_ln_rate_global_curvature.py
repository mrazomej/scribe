"""Cross-check the hand-derived TSLN-Rate global curvature.

The hand-derived path in ``compute_global_uncertainty`` uses
closed-form per-cell-per-gene formulas for the gradient and Hessian
in the (α, β) data-side basis (from the log-marginal identity), adds
the MVN prior contribution to the (log r, log r) diagonal, and
Faà-di-Bruno-chain-rules through ``_twostate_reparam`` into the
unconstrained ``(mu_loc, burst_size_loc, k_off_loc)`` space.  This
test confirms the closed-form matches the autodiff ground truth
(``hessian_diag_chunked`` applied to the same ``neg_log_post``) to
float32 precision on a small synthetic problem.

If this test ever fails after a kernel change, the hand-derivation
needs to be re-checked against the math in
``paper/_two_state_promoter.qmd``
§sec-twostate-tsln-rate-global-uncertainty.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


def _build_synthetic_fit(C=18, G=10, k=2, seed=0):
    """Run a tiny TSLN-Rate Laplace fit so we have realistic
    (x_map, mu/burst_size/k_off, W, d) to test against.
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
        base_model="twostate_ln_rate",
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
    """Closed-form ``H_*_loc`` per-gene curvature equals
    ``hessian_diag_chunked`` to float32 precision.
    """
    from scribe.laplace._global_uncertainty import (
        hessian_diag_chunked,
        woodbury_inv_diag,
        woodbury_apply_inv,
    )
    from scribe.laplace._newton_twostate_ln_rate import (
        _twostate_ln_rate_factors,
        global_curvature_rate_summed,
    )
    from scribe.laplace._obs_twostate_ln_rate import (
        _factors_batch,
        _woodbury_quadform,
        _woodbury_logdet_sigma,
    )
    from scribe.models.components.likelihoods.two_state import (
        _twostate_reparam,
    )

    result, counts, cfg = _build_synthetic_fit(C=18, G=10, k=2, seed=0)
    mu_at_map = np.asarray(result.gene_mean)
    burst_size_at_map = np.asarray(result.burst_size)
    k_off_at_map = np.asarray(result.k_off)
    mu_loc = np.asarray(result.gene_mean_loc)
    burst_size_loc = np.asarray(result.burst_size_loc)
    k_off_loc = np.asarray(result.k_off_loc)
    x_loc = np.asarray(result.x_loc)
    W = np.asarray(result.W)
    d = np.asarray(result.d)
    eta_loc = result.eta_loc
    C, G = counts.shape
    n_q = 60

    # ---- Closed-form curvature path -----------------------------
    alpha_at_map, beta_at_map, rate_at_map, _ = _twostate_reparam(
        jnp.asarray(mu_at_map),
        jnp.asarray(burst_size_at_map),
        jnp.asarray(k_off_at_map),
    )
    log_rate_at_map = jnp.log(jnp.maximum(rate_at_map, 1e-30))

    if eta_loc is None:
        eta_cap_for_curv = jnp.zeros((C,), dtype=jnp.float32)
    else:
        eta_cap_for_curv = jnp.asarray(eta_loc)

    curv = global_curvature_rate_summed(
        x_map=jnp.asarray(x_loc),
        counts=jnp.asarray(counts),
        alpha=alpha_at_map,
        beta=beta_at_map,
        eta_cap=eta_cap_for_curv,
        n_quad_nodes=n_q,
    )

    # MVN block in (log r) — diagonal-only data path.
    sigma_inv_diag = woodbury_inv_diag(jnp.asarray(W), jnp.asarray(d))
    H_logr_logr_mvn = float(C) * sigma_inv_diag
    sum_diff = (
        jnp.sum(jnp.asarray(x_loc), axis=0) - float(C) * log_rate_at_map
    )
    g_logr_mvn = -woodbury_apply_inv(
        jnp.asarray(W), jnp.asarray(d), sum_diff,
    )

    # Build (G, 3) gradient and (G, 3, 3) Hessian in (α, β, log r).
    g_phi = jnp.stack(
        [curv["g_alpha"], curv["g_beta"], g_logr_mvn], axis=-1,
    )
    zero_g = jnp.zeros_like(curv["g_alpha"])
    H_phi = jnp.stack(
        [
            jnp.stack([curv["H_aa"], curv["H_ab"], zero_g], axis=-1),
            jnp.stack([curv["H_ab"], curv["H_bb"], zero_g], axis=-1),
            jnp.stack([zero_g,        zero_g,        H_logr_logr_mvn],
                      axis=-1),
        ],
        axis=-2,
    )
    phi_at_map_stack = jnp.stack(
        [alpha_at_map, beta_at_map, log_rate_at_map], axis=-1,
    )

    def softplus_fwd(x):
        return jax.nn.softplus(x)

    def per_gene_diag(loc_arr, phi_map_g, g_phi_g, H_phi_g):
        def f_local(loc_v):
            mu_v = softplus_fwd(loc_v[0])
            bs_v = softplus_fwd(loc_v[1])
            ko_v = softplus_fwd(loc_v[2])
            a_v, b_v, r_v, _ = _twostate_reparam(mu_v, bs_v, ko_v)
            phi_v = jnp.stack(
                [a_v, b_v, jnp.log(jnp.maximum(r_v, 1e-30))]
            )
            delta = phi_v - phi_map_g
            return jnp.dot(g_phi_g, delta) + 0.5 * jnp.dot(
                delta, jnp.dot(H_phi_g, delta),
            )
        return jnp.diag(jax.hessian(f_local)(loc_arr))

    loc_at_map = jnp.stack(
        [jnp.asarray(mu_loc), jnp.asarray(burst_size_loc),
         jnp.asarray(k_off_loc)], axis=-1,
    )
    H_loc_diag_handderived = jax.vmap(per_gene_diag)(
        loc_at_map, phi_at_map_stack, g_phi, H_phi,
    )

    # ---- Autodiff ground truth ----------------------------------
    n_cells = int(C)
    n_genes = int(G)

    log_rate_at_cell = jnp.asarray(x_loc) - eta_cap_for_curv[:, None]

    def neg_log_post(mu_loc_v, bs_loc_v, ko_loc_v):
        mu_v = softplus_fwd(mu_loc_v)
        bs_v = softplus_fwd(bs_loc_v)
        ko_v = softplus_fwd(ko_loc_v)
        a_v, b_v, r_v, _ = _twostate_reparam(mu_v, bs_v, ko_v)
        mu_x_v = jnp.log(jnp.maximum(r_v, 1e-30))
        log_marg_per_gene, _ = _factors_batch(
            log_rate_at_cell, jnp.asarray(counts), a_v, b_v, n_q,
        )
        data_lp = jnp.sum(log_marg_per_gene)
        diff = jnp.asarray(x_loc) - mu_x_v[None, :]
        quad = _woodbury_quadform(jnp.asarray(W), jnp.asarray(d), diff)
        log_det_sigma = _woodbury_logdet_sigma(
            jnp.asarray(W), jnp.asarray(d)
        )
        mvn_lp = (
            -0.5 * jnp.sum(quad)
            - 0.5 * n_cells * (
                log_det_sigma + n_genes * jnp.log(2.0 * jnp.pi)
            )
        )
        return -(data_lp + mvn_lp)

    args = (
        jnp.asarray(mu_loc),
        jnp.asarray(burst_size_loc),
        jnp.asarray(k_off_loc),
    )
    H_mu_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=0, chunk_size=64,
    )
    H_bs_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=1, chunk_size=64,
    )
    H_ko_ad = hessian_diag_chunked(
        neg_log_post, args, argnum=2, chunk_size=64,
    )

    # ---- Compare ------------------------------------------------
    # Tolerances mirror TSLN-Logit: rtol=5e-3, atol=5e-3 in float32.
    # The MVN block introduces some catastrophic cancellation at G=10
    # so we also accept an absolute floor.
    np.testing.assert_allclose(
        np.asarray(H_loc_diag_handderived[:, 0]),
        np.asarray(H_mu_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="mu_loc hand-derived curvature disagrees with autodiff",
    )
    np.testing.assert_allclose(
        np.asarray(H_loc_diag_handderived[:, 1]),
        np.asarray(H_bs_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="burst_size_loc hand-derived disagrees with autodiff",
    )
    np.testing.assert_allclose(
        np.asarray(H_loc_diag_handderived[:, 2]),
        np.asarray(H_ko_ad),
        rtol=5e-3, atol=5e-3,
        err_msg="k_off_loc hand-derived disagrees with autodiff",
    )


def test_compute_global_uncertainty_no_oom_and_finite():
    """End-to-end: a TSLN-Rate fit's ``compute_global_uncertainty``
    runs cleanly and produces finite scales for all unfrozen
    parameters.
    """
    result, _, _ = _build_synthetic_fit(C=24, G=12, k=2, seed=1)
    assert result.gene_mean_scale is not None
    assert result.burst_size_scale is not None
    assert result.k_off_scale is not None
    assert jnp.all(jnp.isfinite(result.gene_mean_scale))
    assert jnp.all(jnp.isfinite(result.burst_size_scale))
    assert jnp.all(jnp.isfinite(result.k_off_scale))
    assert jnp.all(result.gene_mean_scale > 0)
    assert jnp.all(result.burst_size_scale > 0)
    assert jnp.all(result.k_off_scale > 0)
