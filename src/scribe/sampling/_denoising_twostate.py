"""Two-State (Poisson–Beta compound) denoising helpers.

Provides quadrature-based posterior mean/variance and grid-based
inverse-CDF sampling for the latent ON-fraction p_{gc} under the
Two-State promoter model.  The generative model is:

    p_{gc} ~ Beta(α_g, β_g)
    m_{gc} | p_{gc} ~ Poisson(r̂_g · p_{gc})            (true count)
    u_{gc} | m_{gc}, ν_c ~ Binomial(m_{gc}, ν_c)       (observed UMI)

By Poisson thinning, given p_{gc} and ν_c, the observed and dropped
counts are conditionally independent Poissons:

    u_{gc} | p_{gc}, ν_c  ~  Poisson(ν_c · r̂_g · p_{gc})
    d_{gc} | p_{gc}, ν_c  ~  Poisson((1 − ν_c) · r̂_g · p_{gc})

Denoising amounts to computing  ⟨m_{gc} | u_{gc}⟩ = u_{gc} + ⟨d_{gc} | u_{gc}⟩
or drawing posterior samples of m_{gc}.

See ``paper/_two_state_promoter.qmd``, §sec-twostate-denoising for the
full mathematical derivation.
"""

from typing import Optional, Tuple

from jax import random, vmap
from jax import scipy as jsp
import jax.numpy as jnp

from scribe.stats.quadrature import gauss_legendre_nodes_weights

# ──────────────────────────────────────────────────────────────────────
# Numerical guard — shared by all functions in this module.
# ──────────────────────────────────────────────────────────────────────
_TS_DENOISE_EPS = 1e-12


# ======================================================================
# Log-posterior kernel of the latent ON-fraction
# ======================================================================

def _twostate_p_log_posterior_unnorm(
    p_grid: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    rate: jnp.ndarray,
    counts: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    r"""Log unnormalized posterior of the latent ON-fraction p_{gc}.

    The posterior density (up to a constant) is

        π(p | u, α, β, r̂, ν) ∝ Beta(p | α, β) · Poisson(u | ν r̂ p)

    Expanding in log-space:

        log π̃ = (α − 1) log p + (β − 1) log(1 − p) − log B(α, β)
               + u · log(ν r̂ p) − ν r̂ p − log Γ(u + 1)

    The constant terms (log B, log Γ(u+1)) cancel in every ratio we
    form downstream, but we include them here so the absolute scale is
    meaningful for debugging and the grid CDF in the sampling path.

    All inputs broadcast: ``p_grid`` carries a leading quadrature/grid
    axis while all other tensors have cell/gene axes.

    Parameters
    ----------
    p_grid : jnp.ndarray
        Quadrature nodes or grid points in (0, 1), shape ``(N, 1, 1)``.
    alpha : jnp.ndarray
        Beta first shape parameter (k⁺_g), shape ``(G,)`` or ``(C, G)``.
    beta : jnp.ndarray
        Beta second shape parameter (k⁻_g), shape ``(G,)`` or ``(C, G)``.
    rate : jnp.ndarray
        Poisson rate scale r̂_g, shape ``(G,)`` or ``(C, G)``.
    counts : jnp.ndarray
        Observed UMI counts, shape ``(C, G)``.
    nu : jnp.ndarray
        Per-cell capture probability, shape ``(C, 1)`` or scalar.

    Returns
    -------
    jnp.ndarray
        Log unnormalized posterior, shape ``(N, C, G)``.
    """
    # Clamp p away from 0 and 1 to avoid log(0) / division-by-zero.
    pg = jnp.clip(p_grid, _TS_DENOISE_EPS, 1.0 - _TS_DENOISE_EPS)
    log_pg = jnp.log(pg)
    log_1mp = jnp.log(1.0 - pg)

    # ── Beta log-density (unnormalized by B(α,β) — included for
    #    absolute-scale fidelity) ──────────────────────────────────
    log_beta_prior = (
        (alpha - 1.0) * log_pg
        + (beta - 1.0) * log_1mp
        - jsp.special.betaln(alpha, beta)
    )

    # ── Poisson log-likelihood at each node ──────────────────────
    # λ_k = ν · r̂ · p_k   (effective observed-count rate)
    # pg is (N, 1, 1), nu is (C, 1) or (), rate is (G,) or (C, G),
    # so log_lambda broadcasts to (N, C, G).
    log_lambda = jnp.log(jnp.clip(nu * rate, min=_TS_DENOISE_EPS)) + log_pg

    # Stable Poisson kernel with an explicit value=0 branch to avoid
    # the IEEE-754 trap  0 · log(tiny) → NaN.
    # counts is (C, G); JAX broadcasting against (N, C, G) promotes it
    # to (1, C, G) automatically — no manual expansion needed.
    safe_u = jnp.where(counts > 0, counts, 1.0)
    log_poiss_nonzero = (
        safe_u * log_lambda
        - jnp.exp(log_lambda)
        - jsp.special.gammaln(counts + 1.0)
    )
    log_poiss_zero = -jnp.exp(log_lambda)
    log_poiss = jnp.where(counts > 0, log_poiss_nonzero, log_poiss_zero)

    return log_beta_prior + log_poiss


# ======================================================================
# Quadrature-based posterior mean and variance
# ======================================================================

def _denoise_twostate_quadrature(
    counts: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    rate: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    n_nodes: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Two-State MAP denoising via Gauss–Legendre quadrature.

    Computes the posterior mean and variance of the denoised (true)
    count m_{gc} = u_{gc} + d_{gc} under the Two-State model by
    numerically integrating over the posterior of the latent
    ON-fraction p_{gc}.

    The key identities are:

        ⟨m | u⟩ = u + (1 − ν) r̂ ⟨p | u⟩

        Var(m | u) = (1 − ν) r̂ ⟨p | u⟩
                   + ((1 − ν) r̂)² · Var(p | u)

    where ⟨p | u⟩ and Var(p | u) are the posterior mean and variance
    of the latent ON-fraction, computed via quadrature.

    See ``paper/_two_state_promoter.qmd``, §sec-twostate-denoising.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI counts, shape ``(C, G)``.
    alpha : jnp.ndarray
        Beta first shape parameter α_g, shape ``(G,)`` or ``(C, G)``.
    beta : jnp.ndarray
        Beta second shape parameter β_g, shape ``(G,)`` or ``(C, G)``.
    rate : jnp.ndarray
        Poisson rate scale r̂_g, shape ``(G,)`` or ``(C, G)``.
    p_capture : jnp.ndarray or None
        Per-cell capture probability ν_c, shape ``(C,)`` or ``None``.
        ``None`` is treated as ν_c = 1 (perfect capture → identity
        denoising, since no molecules are dropped).
    n_nodes : int, optional
        Number of Gauss–Legendre quadrature nodes.  Default 64.

    Returns
    -------
    denoised_mean : jnp.ndarray
        ⟨m_{gc} | u_{gc}⟩, shape ``(C, G)``.
    denoised_var : jnp.ndarray
        Var(m_{gc} | u_{gc}), shape ``(C, G)``.
    """
    # ── Capture probability → (C, 1) for broadcasting with (C, G) ──
    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # ── Gauss–Legendre nodes and weights on [0, 1] ────────────────
    nodes, weights = gauss_legendre_nodes_weights(n_nodes, 0.0, 1.0)

    # Reshape to (N, 1, 1) so they broadcast against (C, G) tensors.
    pg = nodes[:, None, None]
    w_gl = weights[:, None, None]

    # ── Evaluate log unnormalized posterior at every node ─────────
    log_post = _twostate_p_log_posterior_unnorm(
        pg, alpha, beta, rate, counts, nu
    )

    # Stabilize via log-sum-exp trick before exponentiating.
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (N, C, G)

    # ── Normalizing constant  Z = Σ_k w_k π̃(p_k | u) ────────────
    Z = jnp.sum(w_gl * post_unnorm, axis=0)  # (C, G)
    Z = jnp.clip(Z, _TS_DENOISE_EPS, None)

    # ── Posterior moments of p under the discrete quadrature ──────
    # ⟨p | u⟩ = Σ_k w_k π̃_k p_k  /  Z
    E_p = jnp.sum(w_gl * post_unnorm * pg, axis=0) / Z

    # ⟨p² | u⟩ for Var(p | u) = ⟨p²⟩ − ⟨p⟩²
    E_p2 = jnp.sum(w_gl * post_unnorm * pg ** 2, axis=0) / Z
    Var_p = E_p2 - E_p ** 2

    # ── Dropped-count rate scale: (1 − ν) r̂ ─────────────────────
    drop_scale = (1.0 - nu) * rate  # (C, G) or broadcast

    # ── Posterior mean:  ⟨m | u⟩ = u + (1−ν) r̂ ⟨p | u⟩ ──────────
    denoised_mean = counts + drop_scale * E_p

    # ── Posterior variance via law of total variance ──────────────
    # d | p ~ Poisson((1−ν) r̂ p), so
    #   Var(d | u) = ⟨Var(d | u, p)⟩ + Var(⟨d | u, p⟩)
    #             = (1−ν) r̂ ⟨p | u⟩  +  ((1−ν) r̂)² Var(p | u)
    # Since u is fixed, Var(m | u) = Var(d | u).
    denoised_var = drop_scale * E_p + drop_scale ** 2 * Var_p

    return denoised_mean, denoised_var


# ======================================================================
# Grid-based inverse-CDF sampling of the latent ON-fraction
# ======================================================================

def _sample_p_posterior_twostate(
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    rate: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    n_grid: int = 1000,
) -> jnp.ndarray:
    r"""Sample the latent ON-fraction from its posterior via grid inversion.

    For each (cell, gene) pair, evaluates the unnormalized posterior
    π̃(p | u, α, β, r̂, ν) on a fine uniform grid over (0, 1),
    normalizes to a PMF, computes the CDF, draws a uniform variate,
    and inverts via ``jnp.searchsorted``.

    Parameters
    ----------
    rng_key : random.PRNGKey
        JAX PRNG key for the uniform draw.
    counts : jnp.ndarray
        Observed UMI counts, shape ``(C, G)``.
    alpha : jnp.ndarray
        Beta first shape parameter α_g, shape ``(G,)`` or ``(C, G)``.
    beta : jnp.ndarray
        Beta second shape parameter β_g, shape ``(G,)`` or ``(C, G)``.
    rate : jnp.ndarray
        Poisson rate scale r̂_g, shape ``(G,)`` or ``(C, G)``.
    p_capture : jnp.ndarray or None
        Per-cell capture probability ν_c, shape ``(C,)`` or ``None``.
    n_grid : int, optional
        Number of grid points.  Default 1000.

    Returns
    -------
    p_samples : jnp.ndarray
        Sampled ON-fraction values, shape ``(C, G)``, each in (0, 1).
    """
    # ── Capture probability ───────────────────────────────────────
    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # ── Fine uniform grid on (ε, 1−ε) ────────────────────────────
    grid = jnp.linspace(_TS_DENOISE_EPS, 1.0 - _TS_DENOISE_EPS, n_grid)
    pg = grid[:, None, None]  # (M, 1, 1)

    # ── Log unnormalized posterior on the grid ────────────────────
    log_post = _twostate_p_log_posterior_unnorm(
        pg, alpha, beta, rate, counts, nu
    )

    # Stabilize and exponentiate.
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (M, C, G)

    # ── Normalize to a PMF along the grid axis ────────────────────
    pmf = post_unnorm / jnp.clip(
        jnp.sum(post_unnorm, axis=0, keepdims=True), _TS_DENOISE_EPS, None
    )

    # ── Cumulative distribution function along grid axis ──────────
    cdf = jnp.cumsum(pmf, axis=0)  # (M, C, G)

    # ── Draw uniform samples and invert the CDF ──────────────────
    u_samples = random.uniform(rng_key, shape=counts.shape)  # (C, G)

    # searchsorted works on 1-D arrays; flatten (C, G) → C*G columns,
    # vmap over each column, then reshape back.
    C, G = counts.shape
    cdf_flat = cdf.reshape(n_grid, -1)  # (M, C*G)
    u_flat = u_samples.reshape(-1)      # (C*G,)

    def _search_one(cdf_col, u_val):
        return jnp.searchsorted(cdf_col, u_val, side="right")

    indices = vmap(_search_one, in_axes=(1, 0), out_axes=0)(
        cdf_flat, u_flat
    )  # (C*G,)

    # Clamp indices to valid range and look up grid values.
    indices = jnp.clip(indices, 0, n_grid - 1)
    p_samples = grid[indices].reshape(C, G)

    return p_samples
