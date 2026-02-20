"""Widely Applicable Information Criterion (WAIC) for Bayesian model comparison.

This module provides JAX-accelerated functions for computing WAIC statistics
from a matrix of posterior log-likelihoods.  All core routines are JIT-compiled
for efficiency on large single-cell RNA-seq datasets.

Mathematical background
-----------------------
Given S posterior samples and n observations (cells), define the (S × n) matrix
of log-likelihoods:

    L[s, i] = log p(y_i | theta^s)

The key quantities are:

    lppd       = sum_i log (1/S sum_s exp(L[s,i]))
    p_waic_1   = 2 * sum_i [log-avg_i - avg-log_i]    (version 1)
    p_waic_2   = sum_i var_s(L[s,i])                  (version 2, preferred)
    WAIC       = -2 * (lppd - p_waic)

See paper/_model_comparison.qmd for full derivations.

References
----------
Watanabe (2010), "Asymptotic Equivalence of Bayes Cross Validation and
    Widely Applicable Information Criterion in Singular Learning Theory."
Gelman, Hwang, Vehtari (2014), "Understanding predictive information criteria
    for Bayesian models."
"""

from __future__ import annotations

from functools import partial
from typing import Union, Optional

import jax.numpy as jnp
from jax import jit


# ---------------------------------------------------------------------------
# Private JIT-compiled building blocks
# ---------------------------------------------------------------------------


@partial(jit, static_argnames=["aggregate", "dtype"])
def _lppd(
    log_liks: jnp.ndarray,
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Union[float, jnp.ndarray]:
    """JIT-compiled log pointwise predictive density.

    Computes

        lppd_i = log (1/S sum_s exp(L[s,i]))

    using the log-sum-exp trick for numerical stability.

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, n)``
        Log-likelihoods for each posterior sample ``s`` and observation ``i``.
    aggregate : bool, default=True
        If ``True`` return the scalar ``sum_i lppd_i``; otherwise return the
        vector ``lppd_i`` of shape ``(n,)``.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision.

    Returns
    -------
    float or jnp.ndarray
        Scalar total lppd when ``aggregate=True``, per-observation vector
        otherwise.
    """
    # Promote to requested dtype
    log_liks = log_liks.astype(dtype)

    # Log-sum-exp trick: subtract per-observation maximum before exponentiation
    lse_max = jnp.max(log_liks, axis=0)  # shape (n,)
    log_mean_exp = lse_max + jnp.log(jnp.mean(jnp.exp(log_liks - lse_max), axis=0))

    if aggregate:
        return jnp.sum(log_mean_exp)
    return log_mean_exp


@partial(jit, static_argnames=["aggregate", "dtype"])
def _p_waic_1(
    log_liks: jnp.ndarray,
    lppd_pointwise: Optional[jnp.ndarray] = None,
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Union[float, jnp.ndarray]:
    """JIT-compiled effective parameter count, version 1 (bias-corrected).

    Computes

        p_waic_1 = 2 * sum_i (lppd_i - mean_s(L[s,i]))

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, n)``
        Log-likelihood matrix.
    lppd_pointwise : jnp.ndarray, shape ``(n,)``, optional
        Pre-computed per-observation lppd.  If ``None``, recomputed internally.
    aggregate : bool, default=True
        If ``True`` return scalar; otherwise return per-observation vector.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision.

    Returns
    -------
    float or jnp.ndarray
    """
    log_liks = log_liks.astype(dtype)

    # Per-observation lppd (pointwise)
    if lppd_pointwise is None:
        lppd_pointwise = _lppd(log_liks, aggregate=False, dtype=dtype)

    # Per-observation mean of log-likelihoods
    mean_log = jnp.mean(log_liks, axis=0)  # shape (n,)

    # p_waic_1 per observation
    pw1 = 2.0 * (lppd_pointwise - mean_log)

    if aggregate:
        return jnp.sum(pw1)
    return pw1


@partial(jit, static_argnames=["aggregate", "dtype"])
def _p_waic_2(
    log_liks: jnp.ndarray,
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Union[float, jnp.ndarray]:
    """JIT-compiled effective parameter count, version 2 (variance-based).

    Computes

        p_waic_2 = sum_i var_s(L[s,i])

    This is the preferred version (Gelman, Hwang, Vehtari 2014) because it
    is more numerically stable and has better theoretical properties.

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, n)``
        Log-likelihood matrix.
    aggregate : bool, default=True
        If ``True`` return scalar; otherwise return per-observation vector.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision.

    Returns
    -------
    float or jnp.ndarray
    """
    log_liks = log_liks.astype(dtype)

    # Variance of log-likelihoods over posterior samples
    var_log = jnp.var(log_liks, axis=0)  # shape (n,)

    if aggregate:
        return jnp.sum(var_log)
    return var_log


@partial(jit, static_argnames=["aggregate", "dtype"])
def compute_waic_stats(
    log_liks: jnp.ndarray,
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> dict:
    """JIT-compiled computation of all WAIC statistics.

    Computes lppd, both versions of the effective parameter count, and both
    WAIC variants from a log-likelihood matrix in a single forward pass.

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, n)``
        Log-likelihoods for each posterior sample ``s`` and observation ``i``.
        ``S`` is the number of posterior samples, ``n`` is the number of
        observations (cells or genes depending on context).
    aggregate : bool, default=True
        If ``True`` return scalar totals; if ``False`` return per-observation
        arrays of shape ``(n,)``.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision for all computations.

    Returns
    -------
    dict
        Dictionary with keys:

        ``lppd``
            Log pointwise predictive density.
        ``p_waic_1``
            Effective parameter count (bias-corrected version).
        ``p_waic_2``
            Effective parameter count (variance-based, preferred version).
        ``elppd_waic_1``
            Expected log pointwise predictive density under WAIC1.
        ``elppd_waic_2``
            Expected log pointwise predictive density under WAIC2.
        ``waic_1``
            WAIC on deviance scale, using p_waic_1.
        ``waic_2``
            WAIC on deviance scale, using p_waic_2 (recommended).
    """
    # Compute per-observation lppd once; reuse for p_waic_1
    lppd_pw = _lppd(log_liks, aggregate=False, dtype=dtype)

    # Aggregate lppd
    lppd = jnp.sum(lppd_pw) if aggregate else lppd_pw

    # Effective parameter counts
    pw1 = _p_waic_1(log_liks, lppd_pointwise=lppd_pw, aggregate=aggregate, dtype=dtype)
    pw2 = _p_waic_2(log_liks, aggregate=aggregate, dtype=dtype)

    # elpd and WAIC
    elppd_1 = lppd - pw1
    elppd_2 = lppd - pw2
    waic_1 = -2.0 * elppd_1
    waic_2 = -2.0 * elppd_2

    return {
        "lppd": lppd,
        "p_waic_1": pw1,
        "p_waic_2": pw2,
        "elppd_waic_1": elppd_1,
        "elppd_waic_2": elppd_2,
        "waic_1": waic_1,
        "waic_2": waic_2,
    }


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def waic(
    log_liks: jnp.ndarray,
    aggregate: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> dict:
    """Compute WAIC statistics from a posterior log-likelihood matrix.

    This is the public entry point.  All JIT-compiled computation is delegated
    to :func:`compute_waic_stats`.

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, n)``
        Matrix of log-likelihoods: rows are posterior samples, columns are
        observations (cells when ``return_by="cell"``, genes when
        ``return_by="gene"``).
    aggregate : bool, default=True
        If ``True`` return scalar totals; if ``False`` return per-observation
        vectors useful for gene-level or cell-level comparisons.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision.

    Returns
    -------
    dict
        Keys: ``lppd``, ``p_waic_1``, ``p_waic_2``, ``elppd_waic_1``,
        ``elppd_waic_2``, ``waic_1``, ``waic_2``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scribe.mc._waic import waic
    >>> log_liks = jnp.ones((500, 1000)) * -2.0   # (S=500, n=1000)
    >>> stats = waic(log_liks)
    >>> print(stats["waic_2"])
    4000.0
    """
    return compute_waic_stats(log_liks, aggregate=aggregate, dtype=dtype)


def pseudo_bma_weights(
    waic_values: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute pseudo-BMA model weights from an array of WAIC values.

    The pseudo-BMA weight for model k is

        w_k  ∝  exp(-0.5 * (WAIC_k - min_k WAIC_k))

    which mimics the AIC weight formula and provides a simple summary of
    relative model quality.

    Parameters
    ----------
    waic_values : jnp.ndarray, shape ``(K,)``
        WAIC values (on deviance scale, lower is better) for K models.
    dtype : jnp.dtype, default=jnp.float32
        Floating-point precision.

    Returns
    -------
    jnp.ndarray, shape ``(K,)``
        Normalized model weights summing to 1.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scribe.mc._waic import pseudo_bma_weights
    >>> w = pseudo_bma_weights(jnp.array([200.0, 210.0, 215.0]))
    >>> print(w.sum())
    1.0
    """
    waic_values = jnp.asarray(waic_values, dtype=dtype)
    # Subtract minimum for numerical stability before exponentiating
    delta = waic_values - jnp.min(waic_values)
    raw = jnp.exp(-0.5 * delta)
    return raw / jnp.sum(raw)
