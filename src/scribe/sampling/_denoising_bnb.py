"""BNB-specific quadrature and grid-sampling helpers for denoising."""

from typing import Optional, Tuple

from jax import random, vmap
import jax.numpy as jnp
import numpyro.distributions as dist

from scribe.stats.quadrature import gauss_legendre_nodes_weights


# Minimum epsilon to prevent log(0) / division-by-zero in BNB denoising.
_BNB_DENOISE_EPS = 1e-6


def _bnb_omega_to_alpha_kappa(
    r: jnp.ndarray,
    p: jnp.ndarray,
    omega: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert ``omega`` (excess dispersion fraction) to ``(alpha, kappa)``.

    Mirrors the reparameterization in ``build_bnb_dist`` but returns
    the intermediate parameters instead of a distribution object.

    Parameters
    ----------
    r : jnp.ndarray
        NB dispersion (>0).
    p : jnp.ndarray
        NB success probability in numpyro convention (>0, <1).
    omega : jnp.ndarray
        Per-gene excess dispersion fraction (>0).

    Returns
    -------
    alpha : jnp.ndarray
        First Beta shape parameter (mean-preserving).
    kappa : jnp.ndarray
        BNB concentration (second Beta shape parameter, >2).
    """
    omega = jnp.clip(omega, _BNB_DENOISE_EPS, None)
    kappa = 2.0 + (r + 1.0) / omega
    kappa = jnp.clip(kappa, 2.0 + _BNB_DENOISE_EPS, None)
    # Mean-preserving alpha: NB mean = r*p/(1-p) = BNB mean = r*alpha/(kappa-1)
    alpha = p * (kappa - 1.0) / (1.0 - p)
    return alpha, kappa


def _bnb_p_log_posterior_unnorm(
    p_grid: jnp.ndarray,
    r: jnp.ndarray,
    alpha: jnp.ndarray,
    kappa: jnp.ndarray,
    counts: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    """Log of the unnormalized posterior of the latent mixing variable.

    Uses the **numpyro convention** where ``p`` is the observation
    probability (NB mean = r·p/(1−p)).  The unnormalized posterior is:

        (u + α − 1)·log p + (r + κ − 1)·log(1−p) − (r + u)·log[1 − p·(1−ν)]

    This follows from:

    - Likelihood: ∝ (1−p)^r · p^u / [1 − p·(1−ν)]^(r+u)
    - Prior: p^(α−1) · (1−p)^(κ−1)

    All inputs are expected to broadcast: ``p_grid`` has a leading
    node/grid axis while all other arrays have cell/gene axes.

    Parameters
    ----------
    p_grid : jnp.ndarray
        Grid points in (0, 1), shape ``(N, 1, 1)`` or ``(N,)``.
    r, alpha, kappa : jnp.ndarray
        Gene-level parameters, shape ``(..., G)``.
    counts : jnp.ndarray
        Observed counts, shape ``(C, G)``.
    nu : jnp.ndarray
        Capture probability, shape ``(C, 1)`` or scalar.

    Returns
    -------
    jnp.ndarray
        Log unnormalized posterior, shape ``(N, C, G)``.
    """
    pg = jnp.clip(p_grid, _BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS)
    log_p = jnp.log(pg)
    log_1mp = jnp.log(1.0 - pg)
    # Denominator: 1 - p*(1-nu) = effective posterior success prob p'
    denom = 1.0 - pg * (1.0 - nu)
    log_denom = jnp.log(jnp.clip(denom, _BNB_DENOISE_EPS, None))

    return (
        (counts + alpha - 1.0) * log_p
        + (r + kappa - 1.0) * log_1mp
        - (r + counts) * log_denom
    )


def _denoise_bnb_quadrature(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    bnb_concentration: jnp.ndarray,
    n_nodes: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """BNB MAP denoising via Gauss-Legendre quadrature.

    Computes the posterior mean and variance of the denoised count
    mg = ug + dg under the BNB model by numerically integrating over
    the latent Beta mixing variable's posterior.

    See ``paper/_beta_negative_binomial.qmd``, @sec-bnb-denoising for
    the full derivation.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI counts, shape ``(C, G)``.
    r : jnp.ndarray
        NB dispersion, shape ``(G,)`` or ``(C, G)``.
    p : jnp.ndarray
        NB success probability (numpyro convention), scalar or ``(G,)``
        or ``(C, 1)``.
    p_capture : jnp.ndarray or None
        Capture probability, shape ``(C,)`` or ``None``.
        When ``None``, treated as νc = 1 (identity denoising).
    bnb_concentration : jnp.ndarray
        Excess dispersion ωg, shape ``(G,)`` or ``(C, G)``.
    n_nodes : int, optional
        Number of Gauss-Legendre quadrature nodes.  Default: 64.

    Returns
    -------
    denoised_mean : jnp.ndarray
        Posterior mean of the denoised count, shape ``(C, G)``.
    denoised_var : jnp.ndarray
        Posterior variance of the denoised count, shape ``(C, G)``.
    """
    alpha, kappa = _bnb_omega_to_alpha_kappa(r, p, bnb_concentration)

    # Capture probability: (C, 1) for broadcasting with (C, G)
    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # Gauss-Legendre nodes/weights on [0, 1], shape (N,)
    nodes, weights = gauss_legendre_nodes_weights(n_nodes, 0.0, 1.0)

    # Reshape nodes to (N, 1, 1) for broadcasting with (C, G)
    pg = nodes[:, None, None]
    w_gl = weights[:, None, None]

    # Log of unnormalized posterior at each quadrature node
    log_post = _bnb_p_log_posterior_unnorm(pg, r, alpha, kappa, counts, nu)

    # Subtract maximum for numerical stability (log-sum-exp trick)
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (N, C, G)

    # f(p) = p*(1-nu) / [1-p*(1-nu)] — conditional mean factor
    # (numpyro convention: E[d|u,p] = (r+u)*f(p))
    pg_safe = jnp.clip(pg, _BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS)
    pprime = 1.0 - pg_safe * (1.0 - nu)  # p' = 1 - p*(1-nu)
    f_p = pg_safe * (1.0 - nu) / jnp.clip(pprime, _BNB_DENOISE_EPS, None)

    # Numerator integral: sum_i w_i * post(p_i) * f(p_i)
    Z = jnp.sum(w_gl * post_unnorm, axis=0)  # (C, G) — normalizer
    Z = jnp.clip(Z, _BNB_DENOISE_EPS, None)

    E_f = jnp.sum(w_gl * post_unnorm * f_p, axis=0) / Z  # E[f(p)]

    # Posterior mean: u + (r+u) * E[f(p)]
    denoised_mean = counts + (r + counts) * E_f

    # Variance via law of total variance:
    # Var(d|u) = E_p[Var(d|u,p)] + Var_p[E(d|u,p)]

    # g(p) = p*(1-nu) / [1-p*(1-nu)]^2 — conditional variance factor
    # (numpyro convention: Var(d|u,p) = (r+u)*g(p))
    g_p = pg_safe * (1.0 - nu) / jnp.clip(pprime**2, _BNB_DENOISE_EPS, None)

    # E[g(p)] — average conditional variance factor
    E_g = jnp.sum(w_gl * post_unnorm * g_p, axis=0) / Z

    # E[f(p)^2] for the variance-of-means term
    E_f2 = jnp.sum(w_gl * post_unnorm * f_p**2, axis=0) / Z

    # Total variance: (r+u)*E[g] + (r+u)^2 * (E[f^2] - E[f]^2)
    rpu = r + counts
    denoised_var = rpu * E_g + rpu**2 * (E_f2 - E_f**2)

    return denoised_mean, denoised_var


def _sample_p_posterior_bnb(
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    bnb_concentration: jnp.ndarray,
    n_grid: int = 1000,
) -> jnp.ndarray:
    r"""Sample the latent Beta mixing variable from its posterior.

    Uses grid-based inverse CDF sampling: evaluate the unnormalized
    posterior on a fine uniform grid over (0, 1), normalize to a PMF,
    compute the CDF, and invert via ``searchsorted``.

    Parameters
    ----------
    rng_key : random.PRNGKey
        PRNG key for the uniform draw.
    counts : jnp.ndarray
        Observed counts, shape ``(C, G)``.
    r : jnp.ndarray
        NB dispersion, shape ``(G,)`` or ``(C, G)``.
    p : jnp.ndarray
        NB success probability (numpyro convention).
    p_capture : jnp.ndarray or None
        Capture probability, shape ``(C,)`` or ``None``.
    bnb_concentration : jnp.ndarray
        Excess dispersion ωg, shape ``(G,)`` or ``(C, G)``.
    n_grid : int, optional
        Number of grid points for inverse CDF sampling.  Default: 1000.

    Returns
    -------
    p_samples : jnp.ndarray
        Sampled latent mixing variable values, shape ``(C, G)``.
    """
    alpha, kappa = _bnb_omega_to_alpha_kappa(r, p, bnb_concentration)

    if p_capture is not None:
        nu = p_capture[:, None]
    else:
        nu = jnp.ones(())

    # Fine uniform grid on (0, 1), shape (M,)
    grid = jnp.linspace(_BNB_DENOISE_EPS, 1.0 - _BNB_DENOISE_EPS, n_grid)
    pg = grid[:, None, None]  # (M, 1, 1)

    # Log unnormalized posterior on the grid
    log_post = _bnb_p_log_posterior_unnorm(pg, r, alpha, kappa, counts, nu)

    # Stabilize and exponentiate
    log_post_max = jnp.max(log_post, axis=0, keepdims=True)
    post_unnorm = jnp.exp(log_post - log_post_max)  # (M, C, G)

    # Normalize to PMF along grid axis
    pmf = post_unnorm / jnp.clip(
        jnp.sum(post_unnorm, axis=0, keepdims=True), _BNB_DENOISE_EPS, None
    )

    # CDF along grid axis
    cdf = jnp.cumsum(pmf, axis=0)  # (M, C, G)

    # Draw uniform samples and invert CDF
    u_samples = random.uniform(rng_key, shape=counts.shape)  # (C, G)

    # For each (c, g) pair, find the grid index where CDF >= u
    # searchsorted works along the first axis; we need to transpose
    # to work per-(c,g) pair.  Flatten cell/gene dims, searchsorted
    # on each, then reshape.
    C, G = counts.shape
    cdf_flat = cdf.reshape(n_grid, -1)  # (M, C*G)
    u_flat = u_samples.reshape(-1)  # (C*G,)

    # vmap searchsorted over the C*G dimension
    def _search_one(cdf_col, u_val):
        return jnp.searchsorted(cdf_col, u_val, side="right")

    indices = vmap(_search_one, in_axes=(1, 0), out_axes=0)(
        cdf_flat, u_flat
    )  # (C*G,)

    # Clamp indices to valid range and look up grid values
    indices = jnp.clip(indices, 0, n_grid - 1)
    p_samples = grid[indices].reshape(C, G)

    return p_samples
