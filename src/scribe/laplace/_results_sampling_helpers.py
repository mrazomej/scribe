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


# =====================================================================
# Decoupled-layout scatter helpers
# =====================================================================
#
# Under ``correlate_other_column=False`` with a pooled trailing
# ``_other`` column, the PLN-family latent ``W`` / ``d`` live on the
# kept axis (G_kept) while per-gene quantities like ``μ`` and ``r``
# stay on the observation axis (G_obs).  Every PPC kernel that draws
# fresh latent from the prior must therefore:
#
#   1. sample ``x_dev = z @ Wᵀ + √d ⊙ ε`` on G_kept,
#   2. scatter onto G_obs: kept positions get ``μ + x_dev``; the
#      single ``other_idx`` keeps ``μ`` only (deterministic, no
#      per-draw z modulation — matches the math contract).
#
# This is the same pattern used by ``get_compositional_samples`` and
# the deviation-form Newton kernels in ``_newton_nbln``.


def _scatter_x_dev_into_full(
    mu_b: jnp.ndarray,
    x_dev: jnp.ndarray,
    kept_idx: jnp.ndarray,
    target_shape: tuple,
) -> jnp.ndarray:
    """Build full G_obs log-rate by scattering kept-axis ``x_dev``.

    Parameters
    ----------
    mu_b : jnp.ndarray
        Broadcast-compatible μ tensor whose last axis is ``G_obs`` —
        either shape ``(G_obs,)``, ``(1, G_obs)``, ``(S, G_obs)``, or
        higher rank.  Provides the per-gene baseline (kept positions
        get ``μ + x_dev``; ``_other`` keeps ``μ``).
    x_dev : jnp.ndarray
        Per-draw kept-axis deviation tensor with the same leading
        batch axes as ``target_shape`` and trailing axis ``G_kept``.
    kept_idx : jnp.ndarray, shape (G_kept,) int
        Positions of kept genes in the G_obs axis.
    target_shape : tuple
        Desired output shape (must end in G_obs).

    Returns
    -------
    jnp.ndarray of shape ``target_shape``
        ``x[..., kept_idx[k]] = μ[..., kept_idx[k]] + x_dev[..., k]``
        ``x[..., other_idx]   = μ[..., other_idx]``  (no x_dev term)
    """
    base = jnp.broadcast_to(mu_b, target_shape)
    return base.at[..., kept_idx].add(x_dev)


def _reconstruct_full_log_rate_from_kept(
    x_loc_kept: jnp.ndarray,
    mu: jnp.ndarray,
    kept_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Build the full ``(N, G_obs)`` per-cell log-rate from kept x_dev.

    Used by per-cell PPC kernels (MAP-only and Laplace) when
    ``x_loc_kept`` is the converged ``x_dev`` on the kept axis.
    """
    n_cells = int(x_loc_kept.shape[0])
    G_obs = int(mu.shape[0])
    base = jnp.broadcast_to(mu[None, :], (n_cells, G_obs))
    return base.at[:, kept_idx].add(x_loc_kept)


def _ppc_pln_marginal(
    rng_key: jax.Array,
    n_samples: int,
    mu: jnp.ndarray,
    W: jnp.ndarray,
    d: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray] = None,
    # Commit 5b: under ``correlate_other_column=False``, ``W`` / ``d``
    # live on G_kept; the fresh ``x`` is built by sampling ``x_dev`` on
    # G_kept and scattering ``μ + x_dev`` at kept positions, ``μ`` at
    # ``_other`` (deterministic).  Same pattern as NBLN's PPC fix.
    kept_idx: Optional[jnp.ndarray] = None,
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
    g_obs = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    if kept_idx is not None:
        # Decoupled: latent lives on G_kept; scatter onto G_obs.
        g_kept = int(W.shape[0])
        eps = jax.random.normal(k2, (n_samples, g_kept), dtype=mu.dtype)
        x_dev = z @ W.T + jnp.sqrt(d)[None, :] * eps
        x = _scatter_x_dev_into_full(
            mu[None, :], x_dev, kept_idx, (n_samples, g_obs)
        )
    else:
        eps = jax.random.normal(k2, (n_samples, g_obs), dtype=mu.dtype)
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
    # Commit 2b: when ``kept_idx`` is provided (decoupled NBLN layout),
    # the latent is built by sampling ``x_dev`` on G_kept and
    # scattering ``μ + x_dev`` at kept positions; ``_other`` stays at
    # ``μ`` (deterministic).  Under PLN / legacy NBLN (``kept_idx is
    # None``), the kernel runs unchanged.
    kept_idx: Optional[jnp.ndarray] = None,
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
    g_obs = int(mu.shape[0])
    g_kept = int(W.shape[0])  # G_kept under decoupled; == G_obs under legacy
    k_factors = int(W.shape[1])
    # Per-draw sampling axis size for ``eps`` / ``x_dev``.  Under
    # decoupling this is G_kept; under legacy it's G_obs (same value).
    g_eff = g_kept if kept_idx is not None else g_obs

    mu_samples_arr = (
        jnp.asarray(mu_samples) if mu_samples is not None else None
    )

    def _mu_term(start: int, size: int) -> jnp.ndarray:
        # Returns shape (size, 1, G_obs) for broadcast against (size, n_cells, G_obs).
        if mu_samples_arr is not None:
            return mu_samples_arr[start : start + size, None, :]
        return mu[None, None, :]

    def _build_x(
        size: int,
        k1: jax.Array,
        k2: jax.Array,
        start: int,
    ) -> jnp.ndarray:
        """Per-chunk latent log-rate of shape ``(size, n_cells, G_obs)``."""
        z = jax.random.normal(k1, (size, n_cells, k_factors), dtype=mu.dtype)
        eps = jax.random.normal(k2, (size, n_cells, g_eff), dtype=mu.dtype)
        mu_term = _mu_term(start, size)  # (size, 1, G_obs)
        if kept_idx is not None:
            # Decoupled: build kept-axis x_dev, scatter onto G_obs.
            x_dev = z @ W.T + jnp.sqrt(d)[None, None, :] * eps
            base = jnp.broadcast_to(mu_term, (size, n_cells, g_obs))
            return base.at[..., kept_idx].add(x_dev)
        return mu_term + z @ W.T + jnp.sqrt(d)[None, None, :] * eps

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k1, k2, k3 = jax.random.split(rng_key, 3)
        x = _build_x(size, k1, k2, start=0)
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
        x = _build_x(size, k1, k2, start=start)
        p = jax.nn.softmax(x, axis=-1)
        n_b = jnp.broadcast_to(library_sizes, (size,) + library_sizes.shape)
        pieces.append(np.asarray(_multinomial_sample(k3, n_b, p)))
    return np.concatenate(pieces, axis=0)


def _ppc_pln_per_cell(
    rng_key: jax.Array,
    n_samples: int,
    x_loc: jnp.ndarray,
    eta_loc: Optional[jnp.ndarray],
    # Commit 5b: under decoupled PLN, ``x_loc`` is ``x_dev`` on
    # ``(N, G_kept)``.  Need ``mu`` + ``kept_idx`` to reconstruct
    # the full ``(N, G_obs)`` log-rate (``μ + x_dev`` at kept,
    # ``μ`` at ``_other``).
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Draw PLN per-cell MAP-only predictive samples.

    Parameters
    ----------
    rng_key : jax.Array
        PRNG key.
    n_samples : int
        Number of draws per cell.
    x_loc : jnp.ndarray
        Per-cell MAP log-rates, shape ``(n_cells, G)``.  Under
        legacy this is the full G_obs log-rate; under decoupling
        (``kept_idx is not None``) it is ``x_dev`` on G_kept.
    eta_loc : jnp.ndarray, optional
        Optional per-cell capture offsets, shape ``(n_cells,)``.
    mu, kept_idx : optional
        Required together under decoupled PLN to reconstruct the
        full G_obs log-rate.

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
    if kept_idx is not None:
        if mu is None:
            raise ValueError(
                "_ppc_pln_per_cell requires ``mu`` when ``kept_idx`` "
                "is provided (decoupled layout)."
            )
        full_log_rate = _reconstruct_full_log_rate_from_kept(
            x_loc, mu, kept_idx
        )
        eff_log_rate = (
            full_log_rate - eta_loc[:, None]
            if eta_loc is not None
            else full_log_rate
        )
    else:
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
    # Commit 5b: under decoupled PLN, ``x_loc`` is ``x_dev`` on
    # ``(N, G_kept)``; W/d are on G_kept too.  The per-cell Laplace
    # sampler runs on G_kept and we scatter ``μ + x_dev_perturbed``
    # onto the full G_obs axis before computing Poisson rates.
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
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

    if kept_idx is not None and mu is None:
        raise ValueError(
            "_ppc_pln_per_cell_laplace requires ``mu`` when ``kept_idx`` "
            "is provided (decoupled layout)."
        )

    def _build_log_rate(x_samples: jnp.ndarray) -> jnp.ndarray:
        """Reduce ``(S, C, G_eff)`` posterior draws → ``(S, C, G_obs)``
        per-cell log-rate (μ + x_dev at kept, μ at ``_other`` under
        decoupling; ``x_samples − η`` under legacy)."""
        if kept_idx is not None:
            n_s = int(x_samples.shape[0])
            n_c = int(x_samples.shape[1])
            g_obs = int(mu.shape[0])
            base = jnp.broadcast_to(
                mu[None, None, :] - eta_arr[None, :, None],
                (n_s, n_c, g_obs),
            )
            return base.at[..., kept_idx].add(x_samples)
        return x_samples - eta_arr[None, :, None]

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
            _build_log_rate(x_samples), _LOG_RATE_MIN, _LOG_RATE_MAX,
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
            _build_log_rate(x_samples), _LOG_RATE_MIN, _LOG_RATE_MAX,
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
    # Commit 2b: under ``correlate_other_column=False``, the latent
    # ``W`` / ``d`` live on G_kept; the per-draw x_dev gets scattered
    # onto the full G_obs axis (``_other`` deterministic at μ_other).
    kept_idx: Optional[jnp.ndarray] = None,
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

    When ``kept_idx`` is provided (decoupled layout, Commit 2b), the
    fresh ``x`` is built by sampling ``x_dev`` on G_kept and scattering
    ``μ + x_dev`` at kept positions, ``μ`` at ``_other`` (deterministic).
    Under the legacy / trivial layout (``kept_idx is None``), the
    sampler runs unchanged.
    """
    from ..stats.distributions import LogMeanNegativeBinomial

    g_genes = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)

    # mu: per-draw if provided, else broadcast point.
    if mu_samples is not None:
        mu_b = jnp.asarray(mu_samples)  # (S, G_obs)
    else:
        mu_b = mu[None, :]  # broadcasts over S

    if kept_idx is not None:
        # Decoupled: latent lives on G_kept; scatter onto G_obs.
        g_kept = int(W.shape[0])
        eps = jax.random.normal(k2, (n_samples, g_kept), dtype=mu.dtype)
        x_dev = z @ W.T + jnp.sqrt(d)[None, :] * eps
        x = _scatter_x_dev_into_full(
            mu_b, x_dev, kept_idx, (n_samples, g_genes)
        )
    else:
        # Legacy: latent lives on G_obs directly.
        eps = jax.random.normal(k2, (n_samples, g_genes), dtype=mu.dtype)
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
    # Commit 2b: under decoupled layout, ``x_loc`` is the kept-axis
    # deviation ``x_dev`` (shape ``(N, G_kept)``).  Need ``mu`` and
    # ``kept_idx`` to reconstruct the full ``(N, G_obs)`` log-rate.
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-cell MAP-only NBLN predictive samples.

    Mirrors :func:`_ppc_pln_per_cell` but emits NB counts conditional
    on the per-cell MAP latent ``(x_loc, eta_loc)``.

    When ``kept_idx`` is provided (decoupled layout, Commit 2b),
    ``x_loc`` carries the kept-axis ``x_dev``; ``mu`` is required so
    the full per-gene log-rate can be reconstructed as
    ``μ_g + x_dev[c, k_g]`` for kept genes and ``μ_g`` for ``_other``.
    """
    from ..stats.distributions import LogMeanNegativeBinomial

    if kept_idx is not None:
        if mu is None:
            raise ValueError(
                "_ppc_nbln_per_cell requires ``mu`` when ``kept_idx`` "
                "is provided (decoupled layout)."
            )
        full_log_rate = _reconstruct_full_log_rate_from_kept(
            x_loc, mu, kept_idx
        )
        log_mean = (
            full_log_rate - eta_loc[:, None]
            if eta_loc is not None
            else full_log_rate
        )
    else:
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
    # Commit 2b: decoupled-layout reconstruction.  When ``kept_idx`` is
    # provided, ``x_loc`` is ``x_dev`` on G_kept (W, d also on G_kept);
    # the full per-gene log-rate is reconstructed via
    # ``μ + x_dev`` (kept) ``μ`` (other), and the per-cell Laplace
    # noise is sampled on G_kept and scattered to G_obs.
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
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

    n_cells = int(x_loc.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)

    if kept_idx is not None:
        # Decoupled: ``x_loc`` is ``x_dev`` on G_kept; W/d are on G_kept;
        # per-cell Laplace noise lives on G_kept too.  Build full
        # G_obs log-rate by scattering kept latent with μ at all
        # genes (so ``_other`` gets only μ + the per-cell deviation
        # from the MAP, not a per-draw kept-latent draw).
        if mu is None:
            raise ValueError(
                "_ppc_nbln_per_cell_laplace requires ``mu`` when "
                "``kept_idx`` is provided (decoupled layout)."
            )
        g_obs = int(mu.shape[0])
        g_kept = int(W.shape[0])
        z = jax.random.normal(
            k1, (n_samples, n_cells, k_factors), dtype=x_loc.dtype
        )
        eps = jax.random.normal(
            k2, (n_samples, n_cells, g_kept), dtype=x_loc.dtype
        )
        # Per-cell, per-draw x_dev on kept axis: MAP shift + Laplace noise.
        x_dev_full = (
            x_loc[None, :, :] + z @ W.T + jnp.sqrt(d)[None, None, :] * eps
        )  # (S, N, G_kept)
        # Scatter onto G_obs: ``μ + x_dev`` at kept, ``μ`` at ``_other``.
        mu_full = jnp.broadcast_to(
            mu[None, None, :], (n_samples, n_cells, g_obs)
        )
        x = mu_full.at[..., kept_idx].add(x_dev_full)
        g_genes = g_obs
    else:
        # Legacy: ``x_loc`` already on G_obs.
        n_cells, g_genes = x_loc.shape
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
    # Commit 3b: under ``correlate_other_column=False``, ``W`` / ``d``
    # live on G_kept; the fresh ``x`` is built by sampling ``x_dev`` on
    # G_kept and scattering ``μ + x_dev`` at kept positions, ``μ`` at
    # ``_other`` (deterministic).  Mirrors NBLN's PPC fix in rev-11.
    kept_idx: Optional[jnp.ndarray] = None,
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

    g_obs = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    if kept_idx is not None:
        # Decoupled: latent lives on G_kept; scatter onto G_obs.
        g_kept = int(W.shape[0])
        eps = jax.random.normal(k2, (n_samples, g_kept), dtype=mu.dtype)
        x_dev = z @ W.T + jnp.sqrt(d)[None, :] * eps
        x = _scatter_x_dev_into_full(
            mu[None, :], x_dev, kept_idx, (n_samples, g_obs)
        )
    else:
        eps = jax.random.normal(k2, (n_samples, g_obs), dtype=mu.dtype)
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
    # Commit 3b: under decoupling, ``x_loc`` is ``x_dev`` on
    # ``(N, G_kept)``.  Need ``mu`` and ``kept_idx`` to reconstruct
    # the full ``(N, G_obs)`` log-rate.
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-cell MAP-only TSLN-Rate predictive samples.

    Keeps per-cell latent log-rates fixed at MAP values
    (``x_loc - η_loc``) and samples observation noise only.

    Returns shape ``(S, C, G)``.

    When ``kept_idx`` is provided (decoupled layout, Commit 3b),
    ``x_loc`` carries the kept-axis ``x_dev``; ``mu`` is required so
    the full per-gene log-rate can be reconstructed as
    ``μ_g + x_dev[c, k_g]`` for kept genes and ``μ_g`` for ``_other``.
    """
    from ..stats.distributions import PoissonBetaCompound

    if kept_idx is not None:
        if mu is None:
            raise ValueError(
                "_ppc_twostate_ln_rate_per_cell requires ``mu`` when "
                "``kept_idx`` is provided (decoupled layout)."
            )
        full_log_rate = _reconstruct_full_log_rate_from_kept(
            x_loc, mu, kept_idx
        )
        log_rate = (
            full_log_rate - eta_loc[:, None]
            if eta_loc is not None
            else full_log_rate
        )
    else:
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
    # Commit 3b: decoupled-layout reconstruction.  When ``kept_idx`` is
    # provided, ``x_loc`` is ``x_dev`` on G_kept (W, d also on G_kept);
    # ``sample_x_posterior_batch`` runs on the kept axis and we
    # scatter ``μ + x_dev_perturbed`` onto the full G_obs axis before
    # Poisson-Beta sampling.
    mu: Optional[jnp.ndarray] = None,
    kept_idx: Optional[jnp.ndarray] = None,
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

    if kept_idx is not None and mu is None:
        raise ValueError(
            "_ppc_twostate_ln_rate_per_cell_laplace requires ``mu`` when "
            "``kept_idx`` is provided (decoupled layout)."
        )

    def _build_log_rate(x_samples: jnp.ndarray) -> jnp.ndarray:
        """Reduce ``(S, C, G_eff)`` posterior draws → ``(S, C, G_obs)``
        per-gene log-rate.

        Under legacy, ``x_samples`` already lives on G_obs and
        ``log_rate = x_samples − η`` (the standard reduction).  Under
        decoupling, ``x_samples`` is ``x_dev`` on G_kept; we build
        ``log_rate[c, g] = μ_g + x_dev[s, c, k_g] − η_c`` for kept
        genes and ``μ_g − η_c`` for ``_other``.
        """
        if kept_idx is not None:
            n_s = int(x_samples.shape[0])
            n_c = int(x_samples.shape[1])
            g_obs = int(mu.shape[0])
            base = jnp.broadcast_to(
                mu[None, None, :] - eta_arr[None, :, None],
                (n_s, n_c, g_obs),
            )
            return base.at[..., kept_idx].add(x_samples)
        return x_samples - eta_arr[None, :, None]

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
            _build_log_rate(x_samples), _LOG_RATE_MIN, _LOG_RATE_MAX,
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
            _build_log_rate(x_samples), _LOG_RATE_MIN, _LOG_RATE_MAX,
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
    # Commit 4b: under ``correlate_other_column=False``, ``W`` / ``d``
    # live on G_kept.  Sample ``z_kept`` on G_kept and scatter onto
    # G_obs zero-base so ``η_act = θ + x_full`` reduces to ``θ`` at
    # ``_other`` (no z modulation).
    kept_idx: Optional[jnp.ndarray] = None,
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

    g_obs = int(mu.shape[0])
    k_factors = int(W.shape[1])
    k1, k2, k3, k4 = jax.random.split(rng_key, 4)
    z = jax.random.normal(k1, (n_samples, k_factors), dtype=mu.dtype)
    if kept_idx is not None:
        # Decoupled: latent on G_kept; scatter onto G_obs with zeros at
        # ``_other``.  ``mu`` is zeros for TSLN-Logit so we skip the
        # ``mu +`` addition entirely.
        g_kept = int(W.shape[0])
        eps = jax.random.normal(k2, (n_samples, g_kept), dtype=mu.dtype)
        z_kept_full = z @ W.T + jnp.sqrt(d)[None, :] * eps  # (S, G_kept)
        x = jnp.zeros((n_samples, g_obs), dtype=mu.dtype)
        x = x.at[:, kept_idx].add(z_kept_full)
    else:
        eps = jax.random.normal(k2, (n_samples, g_obs), dtype=mu.dtype)
        x = mu[None, :] + z @ W.T + jnp.sqrt(d)[None, :] * eps
    g_genes = g_obs  # for downstream broadcast / log_rate construction

    # TSLN-Logit Beta shape parameters depend on (theta + x).
    eta_act = eta_anchor[None, :] + x                    # (S, G_obs)
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
    # Commit 4b: under decoupling, ``x_loc`` is ``z_kept`` on
    # ``(N, G_kept)``.  Scatter onto G_obs (zeros at ``_other``)
    # before forming the activation log-odds ``θ + x_full``.
    kept_idx: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Per-cell MAP-only TSLN-Logit predictive samples.

    Keeps per-cell latent ``x`` fixed at its MAP value and draws
    only observation noise.

    Returns shape ``(S, C, G)``.

    When ``kept_idx`` is provided (decoupled layout, Commit 4b),
    ``x_loc`` carries ``z_kept`` on G_kept; the activation log-odds
    is reconstructed on G_obs as ``θ + z_kept[k_g]`` for kept genes
    and ``θ`` for ``_other`` (no z modulation).
    """
    from ..stats.distributions import PoissonBetaCompound

    n_cells = int(x_loc.shape[0])
    g_obs = int(eta_anchor.shape[0])

    if kept_idx is not None:
        # Scatter ``z_kept`` onto G_obs zero-base.
        x_full = jnp.zeros((n_cells, g_obs), dtype=x_loc.dtype)
        x_full = x_full.at[:, kept_idx].add(x_loc)
    else:
        x_full = x_loc

    # Beta shape parameters at the cell-specific activation log-odds.
    eta_act = eta_anchor[None, :] + x_full               # (C, G_obs)
    phi = jax.nn.sigmoid(eta_act)
    alpha = kappa[None, :] * phi                         # (C, G_obs)
    beta = kappa[None, :] * (1.0 - phi)                  # (C, G_obs)

    log_rate_g = jnp.log(jnp.maximum(rate, 1e-30))       # (G_obs,)
    if eta_loc is not None:
        log_rate = log_rate_g[None, :] - eta_loc[:, None]  # (C, G_obs)
    else:
        log_rate = jnp.broadcast_to(
            log_rate_g[None, :], (n_cells, g_obs)
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
    # Commit 4b: under decoupling, ``x_loc`` is ``z_kept`` on
    # ``(N, G_kept)`` and W/d are on G_kept too.  The Laplace per-cell
    # sampler runs on the kept axis; we scatter onto G_obs (zeros at
    # ``_other``) before building activation log-odds.
    kept_idx: Optional[jnp.ndarray] = None,
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
    g_obs = int(eta_anchor.shape[0])
    eta_arr = (
        jnp.zeros(n_cells, dtype=x_loc.dtype)
        if eta_loc is None
        else jnp.asarray(eta_loc)
    )
    log_rate_g = jnp.log(jnp.maximum(rate, 1e-30))       # (G_obs,)

    def _build_eta_act(x_samples: jnp.ndarray) -> jnp.ndarray:
        """Reduce ``(S, C, G_eff)`` posterior draws → ``(S, C, G_obs)``
        activation log-odds ``θ + x_full``.

        Under legacy, ``x_samples`` already lives on G_obs.  Under
        decoupling, ``x_samples`` is ``z_kept`` on G_kept; scatter
        onto a G_obs zero-base so ``η_act`` at ``_other`` reduces to
        ``θ_other`` (no z modulation).
        """
        if kept_idx is not None:
            n_s = int(x_samples.shape[0])
            n_c = int(x_samples.shape[1])
            base = jnp.zeros((n_s, n_c, g_obs), dtype=x_samples.dtype)
            x_full = base.at[..., kept_idx].add(x_samples)
        else:
            x_full = x_samples
        return eta_anchor[None, None, :] + x_full

    chunk_size = _PPC_DEFAULT_SAMPLE_CHUNK
    if chunk_size is None or chunk_size >= n_samples:
        size = int(n_samples)
        k_x, k_p = jax.random.split(rng_key)
        cell_keys = jax.random.split(k_x, n_cells)
        x_samples = sample_x_posterior_batch(
            cell_keys, x_loc, eta_arr, W, d, size, 0.0
        )
        x_samples = jnp.transpose(x_samples, (1, 0, 2))  # (S, C, G_eff)

        eta_act = _build_eta_act(x_samples)              # (S, C, G_obs)
        phi = jax.nn.sigmoid(eta_act)
        alpha = kappa[None, None, :] * phi               # (S, C, G_obs)
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
        eta_act = _build_eta_act(x_samples)
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
