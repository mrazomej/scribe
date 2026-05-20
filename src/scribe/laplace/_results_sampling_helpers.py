"""Sampling helper backends for Laplace results mixins.

This module implements model-specific posterior-predictive kernels used by the
public Laplace sampling mixin.  The functions are intentionally module-private,
stateless, and explicit about tensor shapes so they can be tested in isolation
while keeping public APIs concise.

The helpers cover three conditioning regimes:

1. fully marginal population PPC (fresh imaginary cells),
2. library-anchored PPC (fresh composition with observed totals), and
3. per-cell predictive PPC (MAP-only or Laplace-uncertainty-aware).
"""

from __future__ import annotations

from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from ._results_shared import (
    _LOG_RATE_MAX,
    _LOG_RATE_MIN,
    _PPC_DEFAULT_SAMPLE_CHUNK,
)


def _ppc_pln_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Draw fully marginal PLN posterior predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key for latent and observation draws.
    n_samples : int
        Number of synthetic cells ``S`` to sample.
    mu : jnp.ndarray
        Decoder bias ``μ`` with shape ``(G,)``.
    W : jnp.ndarray
        Low-rank loading matrix ``W`` with shape ``(G, K)``.
    d : jnp.ndarray
        Residual diagonal variance ``d`` with shape ``(G,)``.
    eta_loc : jnp.ndarray, optional
        Optional empirical capture offsets ``η`` used as a bootstrap source.
        Expected shape ``(C,)``.

    Returns
    -------
    jnp.ndarray
        Integer counts with shape ``(S, G)``.

    Notes
    -----
    The sampler draws:

    - ``z_s ~ N(0, I_K)``
    - ``ε_s ~ N(0, I_G)``
    - ``x_s = μ + z_s Wᵀ + sqrt(d) ⊙ ε_s``

    and optionally subtracts a bootstrapped capture offset ``η_s`` before
    mapping to Poisson rates ``λ_sg = exp(clip(x_sg - η_s, ...))``.
    """
    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    eps = jax.random.normal(k2, (n_samples, g_genes), dtype=mu.dtype)
    x = mu[None, :] + z @ W.T + jnp.sqrt(d)[None, :] * eps
    if eta_loc is not None:
        eta_loc_arr = jnp.asarray(eta_loc).reshape(-1)
        idx = jax.random.randint(k3, (n_samples,), 0, eta_loc_arr.shape[0])
        eta_sample = eta_loc_arr[idx]
        log_rate = x - eta_sample[:, None]
    else:
        log_rate = x
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    rate = jnp.exp(log_rate)
    return jax.random.poisson(k4, rate)


def _ppc_pln_library_anchored(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    counts: jnp.ndarray,
    # Phase-2 R5-2: per-draw mu override for NBLN frozen-mu cascade fits.
    # Shape (S, G).  When provided, replaces the broadcast point ``mu``.
    # PLN never sets this (no cascade-freeze on PLN).
    mu_samples: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Draw PLN library-anchored predictive samples.

    This kernel evaluates compositional fit independently of total-count and
    capture submodels by pairing fresh latent composition draws with observed
    per-cell totals.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of predictive draws per observed cell.
    mu, W, d : jnp.ndarray
        Fitted PLN globals.
    counts : jnp.ndarray
        Observed counts used only to derive per-cell totals.

    Returns
    -------
    np.ndarray
        Predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    Let ``N_c = Σ_g u_cg`` be observed per-cell library sizes.
    For each sample ``s`` and cell ``c``, this kernel draws latent logits
    ``x_sc·`` from the fitted PLN latent Gaussian, converts them to
    probabilities ``p_sc· = softmax(x_sc·)``, then draws

    ``ũ_sc· ~ Multinomial(N_c, p_sc·)``.

    This isolates compositional uncertainty while preserving empirical total
    counts.
    """
    library_sizes = (
        jnp.asarray(counts, dtype=jnp.float32).sum(axis=-1).astype(jnp.int32)
    )
    n_cells = int(library_sizes.shape[0])
    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])

    mu_samples_arr = (
        jnp.asarray(mu_samples) if mu_samples is not None else None
    )

    def _mu_term(start: int, size: int) -> jnp.ndarray:
        # Returns shape (size, 1, G) for broadcast against (size, n_cells, G).
        if mu_samples_arr is not None:
            return mu_samples_arr[start : start + size, None, :]
        return mu[None, None, :]

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k1, k2, k3 = jax.random.split(rng_key, 3)
        z = jax.random.normal(k1, (size, n_cells, k_factors), dtype=mu.dtype)
        eps = jax.random.normal(k2, (size, n_cells, g_genes), dtype=mu.dtype)
        x = _mu_term(0, size) + z @ W.T + jnp.sqrt(d)[None, None, :] * eps
        p = jax.nn.softmax(x, axis=-1)
        n_b = jnp.broadcast_to(library_sizes, (size,) + library_sizes.shape)
        return np.asarray(_multinomial_sample(k3, n_b, p))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k1, k2, k3 = jax.random.split(chunk_keys[i], 3)
        z = jax.random.normal(k1, (size, n_cells, k_factors), dtype=mu.dtype)
        eps = jax.random.normal(k2, (size, n_cells, g_genes), dtype=mu.dtype)
        x = _mu_term(start, size) + z @ W.T + jnp.sqrt(d)[None, None, :] * eps
        p = jax.nn.softmax(x, axis=-1)
        n_b = jnp.broadcast_to(library_sizes, (size,) + library_sizes.shape)
        pieces.append(np.asarray(_multinomial_sample(k3, n_b, p)))
    return np.concatenate(pieces, axis=0)


def _ppc_pln_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Draw PLN per-cell MAP-only predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of draws per cell.
    x_loc : jnp.ndarray
        Per-cell MAP log-rates, shape ``(n_cells, G)``.
    eta_loc : jnp.ndarray, optional
        Optional per-cell capture offsets, shape ``(n_cells,)``.

    Returns
    -------
    np.ndarray
        MAP-only predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    This kernel keeps per-cell latent states fixed at MAP values and samples
    only observation noise:

    ``ũ_scg ~ Poisson(exp(clip(x̂_cg - η_c, ...)))``.

    It does **not** propagate posterior uncertainty in ``x``.
    """
    if eta_loc is not None:
        eff_log_rate = x_loc - eta_loc[:, None]
    else:
        eff_log_rate = x_loc
    eff_log_rate = jnp.clip(eff_log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)
    rate = jnp.exp(eff_log_rate)

    # Chunk to bound output/intermediate memory on large matrices.
    def _sample_chunk(chunk_key: jax.Array, size: int) -> jnp.ndarray:
        return jax.random.poisson(
            chunk_key, jnp.broadcast_to(rate, (size,) + rate.shape)
        )

    return _batched_sample_concat(rng_key, n_samples, _sample_chunk)


def _ppc_pln_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    W: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    """Draw PLN per-cell predictive samples with Laplace latent uncertainty.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of predictive draws per cell.
    x_loc : jnp.ndarray
        Per-cell MAP log-rates, shape ``(n_cells, G)``.
    eta_loc : jnp.ndarray, optional
        Per-cell capture offsets, shape ``(n_cells,)``.
    W : jnp.ndarray
        Decoder loadings used by the posterior sampler.
    d : jnp.ndarray
        Decoder diagonal residual variances used by the posterior sampler.

    Returns
    -------
    np.ndarray
        Predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    For each cell ``c``, this sampler first draws latent log-rates from a
    local Laplace approximation around the MAP point ``x̂_c`` using Newton
    posterior samplers, then draws Poisson observations:

    - ``x̃_sc· ~ q_Laplace(x_c | x̂_c, H_c⁻¹)``
    - ``ũ_scg ~ Poisson(exp(clip(x̃_scg - η_c, ...)))``.

    Compared with :func:`_ppc_pln_per_cell`, this includes latent posterior
    variability in addition to observation noise.
    """
    from ._newton_pln import sample_x_posterior_batch

    n_cells = int(x_loc.shape[0])
    eta_arr = (
        jnp.zeros(n_cells, dtype=x_loc.dtype)
        if eta_loc is None
        else jnp.asarray(eta_loc)
    )

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k_x, k_p = jax.random.split(rng_key)
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = jnp.clip(
            x_samples - eta_arr[None, :, None], _LOG_RATE_MIN, _LOG_RATE_MAX
        )
        return np.asarray(jax.random.poisson(k_p, jnp.exp(log_rate)))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k_x, k_p = jax.random.split(chunk_keys[i])
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = jnp.clip(
            x_samples - eta_arr[None, :, None], _LOG_RATE_MIN, _LOG_RATE_MAX
        )
        pieces.append(np.asarray(jax.random.poisson(k_p, jnp.exp(log_rate))))
    return np.concatenate(pieces, axis=0)


def _alr_to_softmax(
    y_alr: jnp.ndarray, alr_reference_idx: int, n_genes: int
) -> jnp.ndarray:
    """Map ALR logits to simplex probabilities.

    Parameters
    ----------
    y_alr : jnp.ndarray
        ALR logits with trailing dimension ``G-1``.
    alr_reference_idx : int
        Reference gene index in full-gene coordinates.
    n_genes : int
        Full gene count ``G``.

    Returns
    -------
    jnp.ndarray
        Simplex probabilities with trailing dimension ``G``.

    Notes
    -----
    This helper applies the composite map:

    ``y_alr -> y_full (insert 0 at reference index) -> softmax(y_full)``.

    The operation is vectorized over any leading sample/cell axes.
    """
    ref = int(alr_reference_idx)
    full = jnp.zeros(y_alr.shape[:-1] + (n_genes,), dtype=y_alr.dtype)
    other = list(range(n_genes))
    other.remove(ref)
    full = full.at[..., jnp.asarray(other)].set(y_alr)
    return jax.nn.softmax(full, axis=-1)


# =====================================================================
# NB-LogNormal predictive samplers
# =====================================================================
#
# Same scaffolding as the PLN samplers above; the only change is that
# the count layer draws from ``LogMeanNegativeBinomial`` (with the
# fitted gene dispersion ``r_g``) instead of ``Poisson``.  This is the
# difference between PLN and NBLN: PLN uses Poisson shot noise on
# ``exp(x - η)``; NBLN uses a Gamma-Poisson compound, equivalent to
# ``NB(mean=exp(x - η), concentration=r_g)``.
#
# ``_ppc_pln_library_anchored`` is a composition-only sampler
# (softmax of latent log-rates → Multinomial draws against observed
# library size) and is *identical* under NBLN, so the dispatch reuses
# it directly without a separate ``_ppc_nbln_library_anchored``.


def _ppc_nbln_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray] = None,
    r_loc: Optional[jnp.ndarray] = None,
    r_scale: Optional[jnp.ndarray] = None,
    pos_forward=None,
    # Phase-2 R5-2: pre-resolved per-draw arrays override the in-helper
    # construction.  Used by the cascade-aware dispatcher in `_sampling.py`
    # to inject SVI-cascade samples for frozen parameters and/or Laplace
    # ``Normal(mu_loc, mu_scale)`` samples for non-frozen ``mu``.
    r_samples: Optional[jnp.ndarray] = None,
    mu_samples: Optional[jnp.ndarray] = None,
    eta_samples: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Fully marginal NBLN posterior predictive samples.

    Mirrors :func:`_ppc_pln_marginal` but emits NB counts.

    Per-draw uncertainty sources (in priority order):

    - ``r``: ``r_samples`` if provided (already constrained, shape
      ``(S, G)``); else ``Normal(r_loc, r_scale)`` mapped through
      ``pos_forward`` if available; else the point ``r``.
    - ``mu``: ``mu_samples`` if provided (shape ``(S, G)``, log-rate
      space — i.e. NBLN's target coord); else the point ``mu``.
    - ``eta``: ``eta_samples`` if provided (shape ``(S,)``, constrained
      ``[0, ∞)``); else legacy uniform-pick from ``eta_loc``.
    """
    from ..stats.distributions import LogMeanNegativeBinomial

    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    eps = jax.random.normal(k2, (n_samples, g_genes), dtype=mu.dtype)

    # mu: per-draw if provided, else broadcast point.
    if mu_samples is not None:
        mu_b = jnp.asarray(mu_samples)  # (S, G)
    else:
        mu_b = mu[None, :]  # broadcasts over S
    x = mu_b + z @ W.T + jnp.sqrt(d)[None, :] * eps

    # eta: per-draw if provided; else legacy uniform-pick from eta_loc; else 0.
    if eta_samples is not None:
        eta_per_draw = jnp.asarray(eta_samples).reshape(-1)  # (S,)
        log_mean = x - eta_per_draw[:, None]
    elif eta_loc is not None:
        eta_loc_arr = jnp.asarray(eta_loc).reshape(-1)
        idx = jax.random.randint(k3, (n_samples,), 0, eta_loc_arr.shape[0])
        eta_pick = eta_loc_arr[idx]
        log_mean = x - eta_pick[:, None]
    else:
        log_mean = x
    log_mean = jnp.clip(log_mean, _LOG_RATE_MIN, _LOG_RATE_MAX)

    # r: pre-resolved samples > Normal posterior > point.
    if r_samples is not None:
        r_draw = jnp.asarray(r_samples)  # (S, G), constrained
    elif r_loc is not None and r_scale is not None and pos_forward is not None:
        r_unconstrained = (
            r_loc[None, :]
            + r_scale[None, :] * jax.random.normal(k5, (n_samples, g_genes))
        )
        r_draw = pos_forward(r_unconstrained)
    else:
        r_draw = r[None, :]

    return LogMeanNegativeBinomial(
        log_mean=log_mean, concentration=r_draw
    ).sample(k4)


def _ppc_nbln_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    r: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell MAP-only NBLN predictive samples.

    Mirrors :func:`_ppc_pln_per_cell` but emits NB counts conditional
    on the per-cell MAP latent ``(x_loc, eta_loc)``.
    """
    from ..stats.distributions import LogMeanNegativeBinomial

    log_mean = (
        x_loc - eta_loc[:, None] if eta_loc is not None else x_loc
    )
    log_mean = jnp.clip(log_mean, _LOG_RATE_MIN, _LOG_RATE_MAX)
    n_cells, g_genes = log_mean.shape
    log_mean_b = jnp.broadcast_to(
        log_mean, (n_samples, n_cells, g_genes)
    )
    return LogMeanNegativeBinomial(
        log_mean=log_mean_b, concentration=r[None, None, :]
    ).sample(rng_key)


def _ppc_nbln_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    W: jnp.ndarray,
    d: jnp.ndarray,
    r: jnp.ndarray,
    r_loc: Optional[jnp.ndarray] = None,
    r_scale: Optional[jnp.ndarray] = None,
    pos_forward=None,
    # Phase-2 R5-2: pre-resolved per-draw arrays from cascade or Laplace.
    # See ``_ppc_nbln_marginal`` for the semantics.  In the per-cell path:
    # ``r_samples`` (S, G) is broadcast across cells; ``eta_samples`` (S, N)
    # gives a per-cell eta per draw; ``mu_samples`` is NOT used here because
    # the per-cell path conditions on the cell-specific MAP ``x_loc`` rather
    # than redrawing ``x`` from its prior.
    r_samples: Optional[jnp.ndarray] = None,
    eta_samples: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-cell Laplace-perturbed NBLN predictive samples.

    Adds Gaussian noise from the prior covariance ``Sigma = W W^T +
    diag(d)`` to the per-cell MAP log-rate ``x_loc`` before drawing
    NB counts.  Per-draw uncertainty sources (priority order):

    - ``r``: ``r_samples`` (S, G) if provided (constrained); else
      ``Normal(r_loc, r_scale)`` via ``pos_forward``; else the point ``r``.
    - ``eta``: ``eta_samples`` (S, N) if provided (constrained, one entry
      per cell per draw — full SVI posterior fidelity for frozen-eta
      cascade fits); else legacy point ``eta_loc[None, :, None]``.
    """
    from ..stats.distributions import LogMeanNegativeBinomial

    n_cells, g_genes = x_loc.shape
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(
        k1, (n_samples, n_cells, k_factors), dtype=x_loc.dtype
    )
    eps = jax.random.normal(
        k2, (n_samples, n_cells, g_genes), dtype=x_loc.dtype
    )
    x = x_loc[None, :, :] + z @ W.T + jnp.sqrt(d)[None, None, :] * eps

    # eta: per-draw per-cell if provided; else legacy point.
    if eta_samples is not None:
        eta_arr = jnp.asarray(eta_samples)  # (S, N)
        log_mean = x - eta_arr[:, :, None]
    elif eta_loc is not None:
        log_mean = x - eta_loc[None, :, None]
    else:
        log_mean = x
    log_mean = jnp.clip(log_mean, _LOG_RATE_MIN, _LOG_RATE_MAX)

    # r: pre-resolved samples > Normal posterior > point.  Same shared-
    # across-cells semantic as before; the SVI ``r`` is a per-gene global.
    if r_samples is not None:
        r_draw = jnp.asarray(r_samples)[:, None, :]  # (S, 1, G)
    elif r_loc is not None and r_scale is not None and pos_forward is not None:
        r_unconstrained = (
            r_loc[None, :]
            + r_scale[None, :] * jax.random.normal(k4, (n_samples, g_genes))
        )
        r_draw = pos_forward(r_unconstrained)[:, None, :]
    else:
        r_draw = r[None, None, :]

    return LogMeanNegativeBinomial(
        log_mean=log_mean, concentration=r_draw
    ).sample(k3)


def _ppc_lnm_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    p_capture_loc: Optional[jnp.ndarray] = None,
    total_counts: Optional[Union[int, jnp.ndarray]] = None,
    totals_cov: Optional[jnp.ndarray] = None,
    mu_T_loc: Optional[jnp.ndarray] = None,
    r_T_loc: Optional[jnp.ndarray] = None,
    pos_forward=None,
    **kwargs,
) -> jnp.ndarray:
    """Draw fully marginal LNM/LNMVCP predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of synthetic cells.
    mu, W, d : jnp.ndarray
        Fitted LNM-family globals in ALR space.
    alr_reference_idx : int, optional
        ALR reference gene index.
    mu_T, r_T : jnp.ndarray, optional
        Optional NB-on-totals parameters.
    p_capture_loc : jnp.ndarray, optional
        Optional empirical capture probabilities for LNMVCP bootstrap.
    total_counts : int or jnp.ndarray, optional
        Fallback totals when NB-on-totals parameters are absent.
    **kwargs
        Optional helper options (for example ``chunk_size``).

    Returns
    -------
    np.ndarray
        Marginal predictive counts with shape ``(n_samples, G)``.

    Notes
    -----
    The composition branch samples ALR logits from the fitted low-rank
    Gaussian and maps them to simplex probabilities:

    - ``y_s ~ N(μ, W Wᵀ + diag(d))`` in ALR space
    - ``p_s· = softmax(augment_ref(y_s))``.

    Total-count branch precedence is:

    1. fitted NB-on-totals (optionally capture-scaled for LNMVCP),
    2. provided ``total_counts``,
    3. fixed fallback total ``1000``.

    Final draws follow ``ũ_s· ~ Multinomial(N_s, p_s·)``.
    """
    if alr_reference_idx is None:
        raise ValueError("LNM PPC requires alr_reference_idx.")
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1
    mvn = dist.LowRankMultivariateNormal(loc=mu, cov_factor=W, cov_diag=d)
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    y_alr = mvn.sample(k1, sample_shape=(n_samples,))
    p = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)

    if mu_T is not None and r_T is not None:
        # Sample totals with global uncertainty when available.
        k_glob = k4
        if (
            totals_cov is not None
            and mu_T_loc is not None
            and r_T_loc is not None
            and pos_forward is not None
        ):
            # Draw unconstrained (mu_T, r_T) from the joint 2D posterior.
            k_glob, k4 = jax.random.split(k4)
            totals_loc_vec = jnp.stack([mu_T_loc, r_T_loc])
            totals_unconstrained = dist.MultivariateNormal(
                loc=totals_loc_vec,
                covariance_matrix=totals_cov,
            ).sample(k_glob, sample_shape=(n_samples,))
            mu_T_draw = pos_forward(totals_unconstrained[:, 0])
            r_T_draw = pos_forward(totals_unconstrained[:, 1])
        else:
            mu_T_draw = jnp.broadcast_to(jnp.asarray(mu_T), (n_samples,))
            r_T_draw = jnp.broadcast_to(jnp.asarray(r_T), (n_samples,))

        if p_capture_loc is not None:
            p_capture_arr = jnp.asarray(p_capture_loc).reshape(-1)
            idx = jax.random.randint(
                k2, (n_samples,), 0, p_capture_arr.shape[0]
            )
            mu_t_eff = mu_T_draw * p_capture_arr[idx]
        else:
            mu_t_eff = mu_T_draw
        nb = dist.NegativeBinomial2(
            mean=mu_t_eff, concentration=r_T_draw
        )
        n_arr = nb.sample(k4, sample_shape=()).astype(jnp.int32)
    else:
        fallback = 1000 if total_counts is None else total_counts
        n_arr = jnp.broadcast_to(
            jnp.asarray(fallback, dtype=jnp.int32), (n_samples,)
        )

    chunk_size = kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    if chunk_size is None or chunk_size >= n_samples:
        return _multinomial_sample(k3, n_arr, p)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(k3, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        p_chunk = jax.lax.dynamic_slice_in_dim(p, start, end - start, axis=0)
        n_chunk = jax.lax.dynamic_slice_in_dim(
            n_arr, start, end - start, axis=0
        )
        pieces.append(
            np.asarray(_multinomial_sample(chunk_keys[i], n_chunk, p_chunk))
        )
    return np.concatenate(pieces, axis=0)


def _ppc_lnm_library_anchored(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alr_reference_idx: Optional[int],
    counts: jnp.ndarray,
) -> jnp.ndarray:
    """Draw LNM-family library-anchored predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of draws per observed cell.
    mu, W, d : jnp.ndarray
        Fitted LNM-family globals in ALR space.
    alr_reference_idx : int, optional
        ALR reference gene index.
    counts : jnp.ndarray
        Observed counts used to derive per-cell totals.

    Returns
    -------
    np.ndarray
        Predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    With observed totals ``N_c = Σ_g u_cg``, this sampler draws fresh ALR
    logits for each ``(sample, cell)`` pair from the fitted latent Gaussian,
    maps to simplex probabilities, and draws:

    ``ũ_sc· ~ Multinomial(N_c, p_sc·)``.

    This mirrors the PLN library-anchored regime while respecting ALR
    parameterization.
    """
    if alr_reference_idx is None:
        raise ValueError("LNM library-anchored PPC requires alr_reference_idx.")

    library_sizes = (
        jnp.asarray(counts, dtype=jnp.float32).sum(axis=-1).astype(jnp.int32)
    )
    n_cells = int(library_sizes.shape[0])
    g_minus1 = int(mu.shape[0])
    n_genes = g_minus1 + 1
    k_factors = int(W.shape[1])

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k1, k2, k3 = jax.random.split(rng_key, 3)
        z = jax.random.normal(k1, (size, n_cells, k_factors), dtype=mu.dtype)
        eps = jax.random.normal(k2, (size, n_cells, g_minus1), dtype=mu.dtype)
        y_alr = mu[None, None, :] + z @ W.T + jnp.sqrt(d)[None, None, :] * eps
        p = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)
        n_b = jnp.broadcast_to(library_sizes, (size,) + library_sizes.shape)
        return np.asarray(_multinomial_sample(k3, n_b, p))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k1, k2, k3 = jax.random.split(chunk_keys[i], 3)
        z = jax.random.normal(k1, (size, n_cells, k_factors), dtype=mu.dtype)
        eps = jax.random.normal(k2, (size, n_cells, g_minus1), dtype=mu.dtype)
        y_alr = mu[None, None, :] + z @ W.T + jnp.sqrt(d)[None, None, :] * eps
        p = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)
        n_b = jnp.broadcast_to(library_sizes, (size,) + library_sizes.shape)
        pieces.append(np.asarray(_multinomial_sample(k3, n_b, p)))
    return np.concatenate(pieces, axis=0)


def _ppc_lnm_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    p_capture_loc: Optional[jnp.ndarray] = None,
    counts: Optional[jnp.ndarray] = None,
    total_counts: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
    """Draw per-cell LNM-family MAP-only predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of draws per cell.
    mu, W, d : jnp.ndarray
        Fitted LNM-family globals in ALR space.
    z_loc, y_alr_loc : jnp.ndarray, optional
        Per-cell latent MAP state (exactly one branch should be present).
    alr_reference_idx : int, optional
        ALR reference gene index.
    mu_T, r_T : jnp.ndarray, optional
        Optional NB-on-totals parameters.
    p_capture_loc : jnp.ndarray, optional
        Optional per-cell capture probabilities for LNMVCP.
    counts : jnp.ndarray, optional
        Optional observed counts to condition on cell totals.
    total_counts : jnp.ndarray, optional
        Optional explicit per-cell totals.
    **kwargs
        Optional helper options (for example ``chunk_size``).

    Returns
    -------
    np.ndarray
        Predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    Latent branch selection:

    - if ``y_alr_loc`` is provided, use direct ALR MAP logits ``ŷ_c``;
    - else use low-rank reconstruction ``ŷ_c = μ + ẑ_c Wᵀ``.

    Total-count precedence:

    1. observed ``counts`` totals,
    2. explicit ``total_counts``,
    3. fitted NB-on-totals (with optional capture scaling),
    4. fallback constant total ``1000``.

    Once ``p_c· = softmax(augment_ref(ŷ_c))`` is fixed, draws are
    multinomial and include no latent posterior uncertainty.
    """
    if alr_reference_idx is None:
        raise ValueError("LNM PPC requires alr_reference_idx.")
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1
    if y_alr_loc is not None:
        y_alr = y_alr_loc
    elif z_loc is not None:
        y_alr = mu[None, :] + z_loc @ W.T
    else:
        raise ValueError("LNM per-cell PPC requires either z_loc or y_alr_loc.")

    p_per_cell = _alr_to_softmax(y_alr, alr_reference_idx, n_genes)
    n_cells = p_per_cell.shape[0]

    if counts is not None:
        n_arr_cells = jnp.asarray(counts).sum(axis=-1).astype(jnp.int32)
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)
    elif total_counts is not None:
        n_arr_cells = jnp.asarray(total_counts, dtype=jnp.int32)
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)
    elif mu_T is not None and r_T is not None:
        mu_t_per_cell = (
            jnp.asarray(mu_T) * jnp.asarray(p_capture_loc)
            if p_capture_loc is not None
            else jnp.broadcast_to(jnp.asarray(mu_T), (n_cells,))
        )
        nb = dist.NegativeBinomial2(
            mean=mu_t_per_cell, concentration=jnp.asarray(r_T)
        )
        k_nb, k_pred = jax.random.split(rng_key)
        n_b = nb.sample(k_nb, sample_shape=(n_samples,)).astype(jnp.int32)
    else:
        n_arr_cells = jnp.full((n_cells,), 1000, dtype=jnp.int32)
        k_pred = rng_key
        n_b = jnp.broadcast_to(n_arr_cells, (n_samples,) + n_arr_cells.shape)

    chunk_size = kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    is_per_sample_totals = hasattr(n_b, "shape") and n_b.ndim == 2
    n_b_static = jnp.asarray(n_b if is_per_sample_totals else n_arr_cells)

    if chunk_size is None or chunk_size >= n_samples:
        n_b_full = (
            n_b_static
            if is_per_sample_totals
            else jnp.broadcast_to(
                n_b_static, (int(n_samples),) + n_b_static.shape
            )
        )
        p_b = jnp.broadcast_to(p_per_cell, (int(n_samples),) + p_per_cell.shape)
        return np.asarray(_multinomial_sample(k_pred, n_b_full, p_b))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(k_pred, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        if is_per_sample_totals:
            n_b_chunk = jax.lax.dynamic_slice_in_dim(
                n_b_static, start, size, axis=0
            )
        else:
            n_b_chunk = jnp.broadcast_to(n_b_static, (size,) + n_b_static.shape)
        p_b_chunk = jnp.broadcast_to(p_per_cell, (size,) + p_per_cell.shape)
        pieces.append(
            np.asarray(_multinomial_sample(chunk_keys[i], n_b_chunk, p_b_chunk))
        )
    return np.concatenate(pieces, axis=0)


def _ppc_lnm_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    z_loc: Optional[jnp.ndarray],
    y_alr_loc: Optional[jnp.ndarray],
    alr_reference_idx: Optional[int],
    mu_T: Optional[jnp.ndarray] = None,
    r_T: Optional[jnp.ndarray] = None,
    p_capture_loc: Optional[jnp.ndarray] = None,
    counts: Optional[jnp.ndarray] = None,
    total_counts: Optional[jnp.ndarray] = None,
    totals_cov: Optional[jnp.ndarray] = None,
    mu_T_loc: Optional[jnp.ndarray] = None,
    r_T_loc: Optional[jnp.ndarray] = None,
    pos_forward=None,
    **kwargs,
) -> jnp.ndarray:
    """Draw per-cell LNM-family predictive samples with Laplace uncertainty.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of draws per cell.
    mu, W, d : jnp.ndarray
        Fitted LNM-family globals in ALR space.
    z_loc, y_alr_loc : jnp.ndarray, optional
        Per-cell latent MAP state. ``z_loc`` selects low-rank posterior
        sampling; ``y_alr_loc`` selects learned-diagonal posterior sampling.
    alr_reference_idx : int, optional
        ALR reference gene index.
    mu_T, r_T : jnp.ndarray, optional
        Optional NB-on-totals parameters.
    p_capture_loc : jnp.ndarray, optional
        Optional per-cell capture probabilities for LNMVCP.
    counts : jnp.ndarray, optional
        Optional observed counts used for conditional totals and ALR-count
        extraction.
    total_counts : jnp.ndarray, optional
        Optional explicit per-cell totals.
    **kwargs
        Optional helper options (for example ``chunk_size``).

    Returns
    -------
    np.ndarray
        Predictive counts with shape ``(n_samples, n_cells, G)``.

    Notes
    -----
    This is the uncertainty-aware counterpart of
    :func:`_ppc_lnm_per_cell`. It samples latent states from per-cell Laplace
    approximations and then samples observations.

    Branches:

    - **Low-rank branch** (``z_loc``): sample ``z̃_sc`` from a Newton-derived
      Laplace posterior and form ``ỹ_sc = μ + z̃_sc Wᵀ``.
    - **Learned-diagonal branch** (``y_alr_loc``): sample ``ỹ_sc`` directly
      from a Laplace posterior in ALR space.

    For each draw, compute ``p_sc· = softmax(augment_ref(ỹ_sc))`` and sample
    ``ũ_sc· ~ Multinomial(N_sc, p_sc·)``, where totals ``N_sc`` come from
    observed/explicit/NB/fallback logic.
    """
    if alr_reference_idx is None:
        raise ValueError("LNM PPC requires alr_reference_idx.")
    import numpyro.distributions as dist

    g_minus1 = mu.shape[0]
    n_genes = g_minus1 + 1
    if z_loc is not None and y_alr_loc is None:
        mode = "z"
        n_cells = int(z_loc.shape[0])
    elif y_alr_loc is not None:
        mode = "y_alr"
        n_cells = int(y_alr_loc.shape[0])
    else:
        raise ValueError(
            "LNM Laplace per-cell PPC requires either z_loc or y_alr_loc."
        )

    # Whether to draw per-sample global totals with uncertainty.
    _has_totals_uncertainty = (
        totals_cov is not None
        and mu_T_loc is not None
        and r_T_loc is not None
        and pos_forward is not None
    )

    if counts is not None:
        n_arr_cells_static = jnp.asarray(counts).sum(axis=-1).astype(jnp.int32)
        nb_fitted = False
    elif total_counts is not None:
        n_arr_cells_static = jnp.asarray(total_counts, dtype=jnp.int32)
        nb_fitted = False
    elif mu_T is not None and r_T is not None:
        n_arr_cells_static = None
        nb_fitted = True
        if not _has_totals_uncertainty:
            mu_t_per_cell = (
                jnp.asarray(mu_T) * jnp.asarray(p_capture_loc)
                if p_capture_loc is not None
                else jnp.broadcast_to(jnp.asarray(mu_T), (n_cells,))
            )
            nb_dist = dist.NegativeBinomial2(
                mean=mu_t_per_cell, concentration=jnp.asarray(r_T)
            )
    else:
        n_arr_cells_static = jnp.full((n_cells,), 1000, dtype=jnp.int32)
        nb_fitted = False

    if mode == "z":
        from ._newton_lnm import sample_z_posterior_batch
    else:
        from ._newton_lnm import sample_y_alr_posterior_batch

    if counts is not None:
        ref_idx = int(alr_reference_idx)
        all_idx = list(range(n_genes))
        all_idx.remove(ref_idx)
        u_alr_per_cell = jnp.asarray(counts, dtype=jnp.float32)[
            :, jnp.asarray(all_idx)
        ]
    else:
        u_alr_per_cell = jnp.zeros((n_cells, g_minus1), dtype=mu.dtype)

    if n_arr_cells_static is not None:
        n_total_per_cell = jnp.asarray(n_arr_cells_static, dtype=mu.dtype)
    elif not _has_totals_uncertainty:
        n_total_per_cell = jnp.asarray(mu_t_per_cell, dtype=mu.dtype)
    else:
        # Placeholder; per-cell totals for Newton sampling use MAP values.
        mu_t_map = (
            jnp.asarray(mu_T) * jnp.asarray(p_capture_loc)
            if p_capture_loc is not None
            else jnp.broadcast_to(jnp.asarray(mu_T), (n_cells,))
        )
        n_total_per_cell = jnp.asarray(mu_t_map, dtype=mu.dtype)

    chunk_size = kwargs.get("chunk_size", _PPC_DEFAULT_SAMPLE_CHUNK)
    if chunk_size is None or chunk_size >= n_samples:
        n_chunks = 1
        chunk_sizes = [int(n_samples)]
    else:
        n_chunks = (n_samples + chunk_size - 1) // chunk_size
        chunk_sizes = [
            min(chunk_size, n_samples - i * chunk_size) for i in range(n_chunks)
        ]

    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i, size in enumerate(chunk_sizes):
        k_lat, k_nb, k_mn = jax.random.split(chunk_keys[i], 3)
        cell_keys = jax.random.split(k_lat, n_cells)

        if mode == "z":
            z_samples = sample_z_posterior_batch(
                cell_keys,
                z_loc,
                u_alr_per_cell,
                n_total_per_cell,
                mu,
                W,
                alr_reference_idx,
                n_genes,
                size,
                0.0,
            )
            y_alr_samples = (
                mu[None, None, :] + jnp.transpose(z_samples, (1, 0, 2)) @ W.T
            )
        else:
            y_samples = sample_y_alr_posterior_batch(
                cell_keys,
                y_alr_loc,
                u_alr_per_cell,
                n_total_per_cell,
                mu,
                W,
                d,
                alr_reference_idx,
                n_genes,
                size,
                0.0,
            )
            y_alr_samples = jnp.transpose(y_samples, (1, 0, 2))

        p_per_sample = _alr_to_softmax(
            y_alr_samples, alr_reference_idx, n_genes
        )
        if nb_fitted:
            if _has_totals_uncertainty:
                # Draw per-sample (mu_T, r_T) from the joint 2D posterior.
                k_glob, k_nb2 = jax.random.split(k_nb)
                totals_loc_vec = jnp.stack([mu_T_loc, r_T_loc])
                totals_unc = dist.MultivariateNormal(
                    loc=totals_loc_vec,
                    covariance_matrix=totals_cov,
                ).sample(k_glob, sample_shape=(size,))
                mu_T_s = pos_forward(totals_unc[:, 0])
                r_T_s = pos_forward(totals_unc[:, 1])
                if p_capture_loc is not None:
                    mu_t_eff_s = mu_T_s[:, None] * jnp.asarray(
                        p_capture_loc
                    )[None, :]
                else:
                    mu_t_eff_s = jnp.broadcast_to(
                        mu_T_s[:, None], (size, n_cells)
                    )
                r_T_s_bc = jnp.broadcast_to(
                    r_T_s[:, None], (size, n_cells)
                )
                n_b_chunk = dist.NegativeBinomial2(
                    mean=mu_t_eff_s, concentration=r_T_s_bc
                ).sample(k_nb2).astype(jnp.int32)
            else:
                n_b_chunk = nb_dist.sample(
                    k_nb, sample_shape=(size,)
                ).astype(jnp.int32)
        else:
            n_b_chunk = jnp.broadcast_to(
                jnp.asarray(n_arr_cells_static, dtype=jnp.int32),
                (size, n_cells),
            )
        pieces.append(
            np.asarray(_multinomial_sample(k_mn, n_b_chunk, p_per_sample))
        )

    return np.concatenate(pieces, axis=0)


def _multinomial_sample(
    rng_key: jax.Array,
    n: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Sample multinomial counts with broadcasted leading dimensions.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n : jnp.ndarray
        Total counts broadcast-compatible with ``p`` leading dimensions.
    p : jnp.ndarray
        Category probabilities with trailing axis for genes/categories.

    Returns
    -------
    jnp.ndarray
        Multinomial draws with the broadcasted leading shape of ``n`` and
        ``p``.

    Notes
    -----
    If ``p`` has shape ``(..., G)`` and ``n`` broadcasts to ``...``, the
    result has shape ``(..., G)`` and each trailing vector sums to ``n``.
    """
    import numpyro.distributions as dist

    return dist.MultinomialProbs(probs=p, total_count=n).sample(rng_key)


def _batched_sample_concat(
    rng_key: jax.Array,
    n_samples: int,
    sampler_fn,
    chunk_size: Optional[int] = _PPC_DEFAULT_SAMPLE_CHUNK,
) -> np.ndarray:
    """Evaluate a sampler in chunks and concatenate on host memory.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Total number of requested draws.
    sampler_fn : Callable[[jax.Array, int], jnp.ndarray]
        Function receiving a chunk key and chunk size and returning a
        ``(chunk_size, ...)`` tensor.
    chunk_size : int, optional
        Chunk size for streaming device allocations.

    Returns
    -------
    np.ndarray
        Concatenated draws with shape ``(n_samples, ...)``.

    Notes
    -----
    Chunking limits peak device memory by materializing at most one
    ``(chunk_size, ...)`` block at a time before transferring to host.
    """
    if chunk_size is None or chunk_size >= n_samples:
        return np.asarray(sampler_fn(rng_key, int(n_samples)))

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        pieces.append(np.asarray(sampler_fn(chunk_keys[i], size)))
    return np.concatenate(pieces, axis=0)


# =====================================================================
# TSLN-Rate predictive samplers
# =====================================================================
#
# Mirror the PLN / NBLN scaffolding above; the count layer is the
# Poisson-Beta compound (Peccoud-Ycart 1995):
#
#     p_gc ~ Beta(α_g, β_g)         (gene-level α, β; independent per (c, g))
#     u_gc ~ Poisson(rate · p_gc)   (rate = exp(log_rate) on the latent axis)
#
# Library-anchored PPC is composition-only (softmax of the log-rate
# latent → Multinomial against observed totals) and is *identical* to
# PLN's ``_ppc_pln_library_anchored`` — no separate helper is needed;
# the dispatch reuses it.


def _ppc_twostate_ln_rate_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Fully marginal TSLN-Rate posterior predictive samples.

    Mirrors :func:`_ppc_pln_marginal` but replaces the Poisson count
    layer with the Poisson-Beta compound from
    :class:`scribe.stats.distributions.PoissonBetaCompound`.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of synthetic cells ``S``.
    mu, W, d : jnp.ndarray
        TSLN-Rate latent globals.  ``mu`` is ``log(r_hat)`` (the
        gene-level log on-production rate prior centre).
    alpha, beta : jnp.ndarray
        Gene-level Beta concentration / depletion parameters from the
        TwoState reparameterization.  Shape ``(G,)``.
    eta_loc : jnp.ndarray, optional
        Empirical per-cell capture offsets ``η_c = −log ν_c`` used as a
        bootstrap source.  Shape ``(C,)``.

    Returns
    -------
    np.ndarray
        Counts with shape ``(S, G)``.
    """
    from ..stats.distributions import PoissonBetaCompound

    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    eps = jax.random.normal(k2, (n_samples, g_genes), dtype=mu.dtype)
    x = mu[None, :] + z @ W.T + jnp.sqrt(d)[None, :] * eps
    if eta_loc is not None:
        eta_loc_arr = jnp.asarray(eta_loc).reshape(-1)
        idx = jax.random.randint(k3, (n_samples,), 0, eta_loc_arr.shape[0])
        log_rate = x - eta_loc_arr[idx][:, None]
    else:
        log_rate = x
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)

    return np.asarray(
        PoissonBetaCompound(
            alpha=alpha, beta=beta, log_rate=log_rate
        ).sample(k4)
    )


def _ppc_twostate_ln_rate_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell MAP-only TSLN-Rate predictive samples.

    Keeps per-cell latent log-rates fixed at MAP values
    (``x_loc - η_loc``) and samples observation noise only.

    Returns shape ``(S, C, G)``.
    """
    from ..stats.distributions import PoissonBetaCompound

    if eta_loc is not None:
        log_rate = x_loc - eta_loc[:, None]
    else:
        log_rate = x_loc
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)

    def _sample_chunk(chunk_key: jax.Array, size: int) -> jnp.ndarray:
        return PoissonBetaCompound(
            alpha=alpha, beta=beta, log_rate=log_rate
        ).sample(chunk_key, sample_shape=(size,))

    return _batched_sample_concat(rng_key, n_samples, _sample_chunk)


def _ppc_twostate_ln_rate_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    W: jnp.ndarray,
    d: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell Laplace-perturbed TSLN-Rate predictive samples.

    Adds Gaussian noise from the prior covariance
    ``Σ = W W^T + diag(d)`` to the per-cell MAP log-rate ``x_loc``
    before drawing Poisson-Beta counts.

    The latent-perturbation step uses the PLN per-cell posterior
    sampler — the prior covariance is identical (NBLN/PLN/TSLN-Rate
    all share ``Σ = WW^T + diag(d)``) and the per-cell data-side
    Hessian factor ``a_g`` is absorbed into the legacy implementation
    via the ``W, d`` arguments.  See ``_ppc_pln_per_cell_laplace`` for
    the underlying mechanics.

    Returns shape ``(S, C, G)``.
    """
    from ._newton_pln import sample_x_posterior_batch
    from ..stats.distributions import PoissonBetaCompound

    n_cells = int(x_loc.shape[0])
    eta_arr = (
        jnp.zeros(n_cells, dtype=x_loc.dtype)
        if eta_loc is None
        else jnp.asarray(eta_loc)
    )

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k_x, k_p = jax.random.split(rng_key)
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = jnp.clip(
            x_samples - eta_arr[None, :, None],
            _LOG_RATE_MIN,
            _LOG_RATE_MAX,
        )
        return np.asarray(
            PoissonBetaCompound(
                alpha=alpha, beta=beta, log_rate=log_rate
            ).sample(k_p)
        )

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k_x, k_p = jax.random.split(chunk_keys[i])
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        log_rate = jnp.clip(
            x_samples - eta_arr[None, :, None],
            _LOG_RATE_MIN,
            _LOG_RATE_MAX,
        )
        pieces.append(
            np.asarray(
                PoissonBetaCompound(
                    alpha=alpha, beta=beta, log_rate=log_rate
                ).sample(k_p)
            )
        )
    return np.concatenate(pieces, axis=0)


# =====================================================================
# TSLN-Logit predictive samplers
# =====================================================================
#
# Same scaffolding as the TSLN-Rate samplers above; the differences:
#
#   * Gene globals are ``(rate_g, kappa_g, eta_anchor_g = theta_g)``
#     rather than ``(alpha_g, beta_g, log_rate_g)``.
#   * Beta shape parameters depend on the latent: ``alpha_cg = kappa_g
#     · sigma(theta_g + z_cg)`` and ``beta_cg = kappa_g · (1 -
#     sigma(theta_g + z_cg))``.  This means alpha / beta are
#     ``(S, G)`` for marginal sampling and ``(S, C, G)`` for per-cell
#     sampling — not gene-rank like TSLN-Rate.
#   * The Poisson scale (rate) is gene-level and z-independent.
#
# ``PoissonBetaCompound.sample`` broadcasts alpha / beta / rate to the
# full batch shape internally; we just have to feed it
# correctly-shaped arrays.


def _ppc_twostate_ln_logit_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    eta_anchor: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Fully marginal TSLN-Logit posterior predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
    n_samples : int
        Number of synthetic cells ``S``.
    mu, W, d : jnp.ndarray
        TSLN-Logit latent globals.  ``mu`` is zeros (latent prior
        centre) and is passed for shape parity with TSLN-Rate.
    rate, kappa, eta_anchor : jnp.ndarray, shape ``(G,)``
        Gene-level globals.  ``eta_anchor`` is the per-gene
        activation log-odds ``θ_g``.
    eta_loc : jnp.ndarray, optional
        Empirical per-cell capture offsets ``η_c = -log ν_c``
        used as a bootstrap source.

    Returns
    -------
    np.ndarray, shape ``(S, G)``
    """
    from ..stats.distributions import PoissonBetaCompound

    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    eps = jax.random.normal(k2, (n_samples, g_genes), dtype=mu.dtype)
    x = mu[None, :] + z @ W.T + jnp.sqrt(d)[None, :] * eps

    # TSLN-Logit Beta shape parameters depend on (theta + x).
    eta_act = eta_anchor[None, :] + x                    # (S, G)
    phi = jax.nn.sigmoid(eta_act)
    alpha = kappa[None, :] * phi                         # (S, G)
    beta = kappa[None, :] * (1.0 - phi)                  # (S, G)

    # Poisson log-rate is gene-level (z-independent); capture is an
    # optional bootstrap-style additive offset.
    log_rate_g = jnp.log(jnp.maximum(rate, 1e-30))       # (G,)
    if eta_loc is not None:
        eta_loc_arr = jnp.asarray(eta_loc).reshape(-1)
        idx = jax.random.randint(k3, (n_samples,), 0, eta_loc_arr.shape[0])
        eta_sample = eta_loc_arr[idx]                    # (S,)
        log_rate = log_rate_g[None, :] - eta_sample[:, None]
    else:
        log_rate = jnp.broadcast_to(
            log_rate_g[None, :], (n_samples, g_genes)
        )
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)

    return np.asarray(
        PoissonBetaCompound(
            alpha=alpha, beta=beta, log_rate=log_rate
        ).sample(k4)
    )


def _ppc_twostate_ln_logit_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    eta_anchor: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell MAP-only TSLN-Logit predictive samples.

    Keeps per-cell latent ``x`` fixed at its MAP value and draws
    only observation noise.

    Returns shape ``(S, C, G)``.
    """
    from ..stats.distributions import PoissonBetaCompound

    # Beta shape parameters at the cell-specific MAP.
    eta_act = eta_anchor[None, :] + x_loc                # (C, G)
    phi = jax.nn.sigmoid(eta_act)
    alpha = kappa[None, :] * phi                         # (C, G)
    beta = kappa[None, :] * (1.0 - phi)                  # (C, G)

    log_rate_g = jnp.log(jnp.maximum(rate, 1e-30))       # (G,)
    if eta_loc is not None:
        log_rate = log_rate_g[None, :] - eta_loc[:, None]  # (C, G)
    else:
        log_rate = jnp.broadcast_to(
            log_rate_g[None, :], x_loc.shape
        )
    log_rate = jnp.clip(log_rate, _LOG_RATE_MIN, _LOG_RATE_MAX)

    def _sample_chunk(chunk_key: jax.Array, size: int) -> jnp.ndarray:
        return PoissonBetaCompound(
            alpha=alpha, beta=beta, log_rate=log_rate
        ).sample(chunk_key, sample_shape=(size,))

    return _batched_sample_concat(rng_key, n_samples, _sample_chunk)


def _ppc_twostate_ln_logit_per_cell_laplace(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    W: jnp.ndarray,
    d: jnp.ndarray,
    rate: jnp.ndarray,
    kappa: jnp.ndarray,
    eta_anchor: jnp.ndarray,
) -> jnp.ndarray:
    """Per-cell Laplace-perturbed TSLN-Logit predictive samples.

    Adds Gaussian noise from the prior covariance ``Σ = W W^T +
    diag(d)`` to the per-cell MAP latent ``x_loc`` before drawing
    Poisson-Beta counts.  The latent-perturbation step reuses the
    PLN per-cell posterior sampler (same Σ structure across PLN /
    NBLN / TSLN-Rate / TSLN-Logit); only the count-noise layer
    differs.

    Returns shape ``(S, C, G)``.
    """
    from ._newton_pln import sample_x_posterior_batch
    from ..stats.distributions import PoissonBetaCompound

    n_cells = int(x_loc.shape[0])
    eta_arr = (
        jnp.zeros(n_cells, dtype=x_loc.dtype)
        if eta_loc is None
        else jnp.asarray(eta_loc)
    )
    log_rate_g = jnp.log(jnp.maximum(rate, 1e-30))       # (G,)

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k_x, k_p = jax.random.split(rng_key)
        cell_keys = jax.random.split(k_x, n_cells)
        # Note: ``sample_x_posterior_batch`` uses the PLN log-rate
        # parameterisation in its internal coupling, but here we
        # only need it to draw from the latent prior covariance
        # around the MAP — and Σ is the same across all PLN-family
        # variants.  Pass eta_arr as the cell offset; the PLN
        # sampler treats it as a stop_gradient constant.
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))  # (S, C, G)

        eta_act = eta_anchor[None, None, :] + x_samples  # (S, C, G)
        phi = jax.nn.sigmoid(eta_act)
        alpha = kappa[None, None, :] * phi               # (S, C, G)
        beta = kappa[None, None, :] * (1.0 - phi)
        log_rate = jnp.clip(
            log_rate_g[None, None, :] - eta_arr[None, :, None],
            _LOG_RATE_MIN, _LOG_RATE_MAX,
        )
        return np.asarray(
            PoissonBetaCompound(
                alpha=alpha, beta=beta, log_rate=log_rate
            ).sample(k_p)
        )

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    chunk_keys = jax.random.split(rng_key, n_chunks)
    pieces: List[np.ndarray] = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n_samples)
        size = end - start
        k_x, k_p = jax.random.split(chunk_keys[i])
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))
        eta_act = eta_anchor[None, None, :] + x_samples
        phi = jax.nn.sigmoid(eta_act)
        alpha = kappa[None, None, :] * phi
        beta = kappa[None, None, :] * (1.0 - phi)
        log_rate = jnp.clip(
            log_rate_g[None, None, :] - eta_arr[None, :, None],
            _LOG_RATE_MIN, _LOG_RATE_MAX,
        )
        pieces.append(
            np.asarray(
                PoissonBetaCompound(
                    alpha=alpha, beta=beta, log_rate=log_rate
                ).sample(k_p)
            )
        )
    return np.concatenate(pieces, axis=0)
