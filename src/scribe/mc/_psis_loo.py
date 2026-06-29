"""Pareto-Smoothed Importance Sampling Leave-One-Out (PSIS-LOO) cross-validation.

This module implements PSIS-LOO @vehtari2017 for scalable Bayesian model
comparison.  The algorithm approximates exact LOO-CV from a single posterior
inference by re-weighting posterior samples using importance ratios and
stabilizing heavy tails via a fitted generalized Pareto distribution (GPD).

Algorithm outline (per observation i)
--------------------------------------
1.  Raw log IS weights: log w_s = -log p(y_i | theta^s)
2.  Identify M = min(S//5, ceil(3*sqrt(S))) tail samples.
3.  Fit a GPD to the M largest weights using the Zhang-Stephens estimator.
4.  Replace top-M weights with GPD quantiles.
5.  Truncate at log(S^0.75).
6.  Compute LOO contribution via a stabilized importance-weighted average.
7.  Record k_hat (Pareto shape) as a per-observation reliability diagnostic.

Diagnostic thresholds for k_hat
---------------------------------
- k < 0.5   : excellent, IS weights have finite variance.
- 0.5 <= k < 0.7 : acceptable but worth monitoring.
- k >= 0.7  : unreliable; LOO approximation may be poor.

References
----------
Vehtari, Gelman, Gabry (2017), "Practical Bayesian model evaluation using
    leave-one-out cross-validation and WAIC." Statistics and Computing.
Zhang, Stephens (2009), "A new and efficient estimation method for the
    generalized Pareto distribution." Technometrics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _n_tail(n_samples: int) -> int:
    """Compute the number of tail samples M for Pareto smoothing.

    M = min(floor(S/5), ceil(3*sqrt(S))), clamped to at least 5.

    Parameters
    ----------
    n_samples : int
        Total number of posterior samples S.

    Returns
    -------
    int
        Number of tail samples to fit the GPD on.
    """
    return max(5, min(n_samples // 5, int(np.ceil(3.0 * np.sqrt(n_samples)))))


def _fit_gpd(tail_values: np.ndarray) -> Tuple[float, float]:
    """Fit a generalized Pareto distribution to tail values.

    Uses scipy's MLE estimator with ``loc`` fixed at 0 (excess distribution).
    Falls back to a moment-matching estimator if the MLE fails.

    Parameters
    ----------
    tail_values : np.ndarray, shape ``(M,)``
        Positive tail values (already shifted so that minimum is near 0).
        These are excesses: ``tail_values[j] = w_{(S-M+j)} - w_{(S-M)}``.

    Returns
    -------
    k_hat : float
        Pareto shape parameter (diagnostic k̂).
    sigma_hat : float
        Pareto scale parameter.
    """
    # Discard any near-zero values that would confuse the GPD fit
    tail_values = tail_values[tail_values > 0]
    if len(tail_values) < 5:
        # Degenerate case: constant or near-constant input
        std_val = float(np.std(tail_values)) if len(tail_values) > 1 else 1e-8
        return 0.0, max(std_val, 1e-8)

    try:
        # scipy parameterization: genpareto(c, loc, scale) where c = k
        k_hat, _, sigma_hat = stats.genpareto.fit(tail_values, floc=0.0)
        # Clamp k to prevent extreme values from destabilizing smoothing
        k_hat = float(np.clip(k_hat, -2.0, 2.0))
        sigma_hat = float(max(sigma_hat, 1e-8))
        return k_hat, sigma_hat
    except Exception:
        # Method-of-moments fallback for GPD with loc=0:
        # mean = sigma/(1-k), var = sigma^2/((1-k)^2*(1-2k))
        mu = float(np.mean(tail_values))
        s2 = float(np.var(tail_values, ddof=1))
        if s2 < 1e-10 or mu < 1e-10:
            return 0.0, mu + 1e-8
        k_hat = 0.5 * (1.0 - mu**2 / s2)
        k_hat = float(np.clip(k_hat, -2.0, 0.49))
        sigma_hat = max(mu * (1.0 - k_hat), 1e-8)
        return k_hat, sigma_hat


def _pareto_smooth_single(log_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply PSIS smoothing to the log importance weights of one observation.

    Parameters
    ----------
    log_weights : np.ndarray, shape ``(S,)``
        Raw log IS weights for a single observation:
        ``log_w_s = -log p(y_i | theta^s)``.

    Returns
    -------
    smoothed_log_weights : np.ndarray, shape ``(S,)``
        Pareto-smoothed log IS weights (same shape, same index order).
    k_hat : float
        Estimated Pareto shape parameter (reliability diagnostic).
    """
    S = len(log_weights)
    M = _n_tail(S)

    # Early exit: if all weights are equal, no smoothing is needed or possible
    if np.ptp(log_weights) < 1e-10:
        return log_weights.copy(), 0.0

    # Sorted indices (ascending), so the last M entries are the M largest
    sort_idx = np.argsort(log_weights)
    sorted_lw = log_weights[sort_idx]

    # Tail log-weights (the M largest, in ascending order)
    tail_lw = sorted_lw[S - M :]

    # Normalize by the MAXIMUM of the tail so that exp(tail_lw - tail_norm)
    # is always in (0, 1].  Normalizing by the threshold (cutoff_lw) was
    # the original approach but it fails when the tail spans more than ~709
    # log-units above the cutoff — a common occurrence for outlier cells in
    # single-cell data, where IS weights are exp(-log_lik) and log_liks can
    # be deeply negative under some posterior samples.
    tail_norm = tail_lw[-1]  # max of sorted tail → maps to exp(0) = 1
    tail_w = np.exp(tail_lw - tail_norm)  # values in (0, 1], overflow-safe
    tail_excess = tail_w - tail_w[0]

    # Fit GPD to the excess values above the minimum tail weight
    k_hat, sigma_hat = _fit_gpd(tail_excess)

    # Replace tail with GPD quantiles at (j - 0.5) / M for j = 1, ..., M
    probs = (np.arange(1, M + 1) - 0.5) / M

    # Handle degenerate GPD (k=0 is Exponential, k→0 limit)
    if abs(k_hat) < 1e-6:
        # Exponential quantile: -sigma * log(1 - p)
        smooth_tail_w = tail_w[0] + sigma_hat * (-np.log(1.0 - probs))
    else:
        smooth_tail_w = tail_w[0] + stats.genpareto.ppf(
            probs, k_hat, loc=0.0, scale=sigma_hat
        )

    # Convert smoothed tail back to log scale using the same shift (tail_norm)
    smooth_tail_lw = tail_norm + np.log(np.maximum(smooth_tail_w, 1e-300))

    # Truncate at log(S^0.75) to prevent runaway weights
    max_lw = min(sorted_lw[-1], 0.75 * np.log(S))
    smooth_tail_lw = np.minimum(smooth_tail_lw, max_lw)

    # Assemble: only the M tail positions are replaced
    smoothed_lw = log_weights.copy()
    smoothed_lw[sort_idx[S - M :]] = smooth_tail_lw

    return smoothed_lw, float(k_hat)


# ---------------------------------------------------------------------------
# Vectorized JAX backend (GPU-accelerated)
# ---------------------------------------------------------------------------


def _gpdfit_jax(
    x: jnp.ndarray,
    prior_bs: float = 3.0,
    prior_k: float = 10.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Fit a generalized Pareto distribution to many tails simultaneously.

    Implements the closed-form Zhang--Stephens (2009) estimator used by the
    canonical PSIS implementation (ArviZ / the ``loo`` package).  Unlike an
    iterative MLE (e.g. ``scipy.stats.genpareto.fit``), this estimator is a
    fixed sequence of array operations, so it vectorizes across all ``N``
    observations at once and runs on the GPU under JAX.

    For each observation the GPD is fit to the tail *exceedances* (the tail
    importance weights shifted so the smallest is zero), following the same
    convention as :func:`_pareto_smooth_single`.

    Parameters
    ----------
    x : jnp.ndarray, shape ``(M, N)``
        Per-observation tail exceedances, ascending along axis 0 with
        ``x[0] == 0``.  ``M`` is the number of tail samples, ``N`` the number
        of observations (columns).  Must be non-negative.
    prior_bs : float, default=3.0
        Weakly-informative prior strength on the grid location (the ``b``
        candidates), matching the Zhang--Stephens reference.
    prior_k : float, default=10.0
        Weakly-informative prior pulling the shape estimate ``k`` toward 0.5,
        regularizing small-sample tails (Vehtari et al.).

    Returns
    -------
    k : jnp.ndarray, shape ``(N,)``
        Per-observation Pareto shape estimate (the k-hat diagnostic).
    sigma : jnp.ndarray, shape ``(N,)``
        Per-observation Pareto scale estimate (strictly positive).
    """
    M = x.shape[0]
    # Grid size and quartile index are static functions of M (Python ints), so
    # they do not trigger retracing and keep tensor shapes fixed.
    m_grid = 30 + int(M**0.5)
    q_idx = max(int(M / 4 + 0.5) - 1, 0)

    # Guard the quartile and max exceedances away from zero (the smallest
    # exceedance is exactly 0 by construction; a degenerate/constant tail is
    # handled by the caller's mask).
    x_quartile = jnp.maximum(x[q_idx, :], 1e-30)  # (N,)
    x_max = jnp.maximum(x[-1, :], 1e-30)  # (N,)

    # Candidate grid of the GPD location parameter b (one row per grid point).
    jj = jnp.arange(1, m_grid + 1, dtype=x.dtype)  # (m_grid,)
    base = 1.0 - jnp.sqrt(m_grid / (jj - 0.5))  # (m_grid,)
    b = (1.0 / x_max)[None, :] + base[:, None] / (prior_bs * x_quartile)[
        None, :
    ]  # (m_grid, N)

    # Profile shape per candidate: k(b) = mean_m log1p(-b * x_m).
    # The (m_grid, M, N) broadcast is the peak intermediate; M is small
    # (~tens) so this stays modest even for large N.
    k_cand = jnp.mean(
        jnp.log1p(-b[:, None, :] * x[None, :, :]), axis=1
    )  # (m_grid, N)

    # Profile log-likelihood per candidate, then softmax weights over the grid.
    # ``-b / k_cand`` is positive in the valid region; clip its argument to
    # keep the log finite for stray candidates (they get ~zero weight anyway).
    log_arg = jnp.clip(-b / k_cand, 1e-30, None)
    # Zhang--Stephens profile log-likelihood: M * (log(-b/k) - k - 1).
    profile_ll = M * (jnp.log(log_arg) - k_cand - 1.0)  # (m_grid, N)
    weights = jax.nn.softmax(profile_ll, axis=0)  # (m_grid, N)

    # Posterior-mean b, then the implied k and sigma.
    b_post = jnp.sum(b * weights, axis=0)  # (N,)
    k = jnp.mean(jnp.log1p(-b_post[None, :] * x), axis=0)  # (N,)
    sigma = -k / b_post  # (N,)

    # Weakly-informative prior on the shape (pull toward 0.5), as in PSIS.
    k = (M * k + prior_k * 0.5) / (M + prior_k)
    return k, jnp.maximum(sigma, 1e-30)


def _gpd_inverse_cdf(
    probs: jnp.ndarray,
    k: jnp.ndarray,
    sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Closed-form generalized-Pareto inverse CDF (quantile function).

    Computes ``F^{-1}(p) = (sigma / k) * ((1 - p)^{-k} - 1)`` for ``k != 0``
    and the exponential limit ``-sigma * log(1 - p)`` for ``k -> 0``, fully
    vectorized.  Used to replace each observation's tail importance weights
    with smooth GPD quantiles.

    Parameters
    ----------
    probs : jnp.ndarray, shape ``(M,)``
        Evaluation probabilities ``(j - 0.5) / M`` for ``j = 1..M``.
    k : jnp.ndarray, shape ``(N,)``
        Per-observation GPD shape parameter.
    sigma : jnp.ndarray, shape ``(N,)``
        Per-observation GPD scale parameter (positive).

    Returns
    -------
    jnp.ndarray, shape ``(M, N)``
        GPD quantiles for each (probability, observation) pair.
    """
    p = probs[:, None]  # (M, 1)
    k_b = k[None, :]  # (1, N)
    sig_b = sigma[None, :]  # (1, N)
    # Use a non-zero ``k`` in the generic branch to avoid 0/0, then select the
    # exponential limit wherever ``k`` is numerically zero (``jnp.where``
    # evaluates both branches, so the guard must keep each finite).
    k_safe = jnp.where(jnp.abs(k_b) < 1e-8, 1.0, k_b)
    generic = (sig_b / k_safe) * (jnp.power(1.0 - p, -k_safe) - 1.0)
    exponential = -sig_b * jnp.log1p(-p)
    return jnp.where(jnp.abs(k_b) < 1e-8, exponential, generic)


def _psis_loo_jax(log_liks: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Vectorized PSIS-LOO over all observations (GPU-accelerated).

    Mirrors the per-observation algorithm of :func:`_pareto_smooth_single`
    plus the importance-weighted LOO average of :func:`compute_psis_loo`, but
    evaluates every observation at once as batched ``jnp`` operations (no
    Python loop), so the whole computation runs on the accelerator.  The
    only algorithmic change from the scalar path is the GPD estimator
    (closed-form Zhang--Stephens instead of an iterative scipy MLE), which is
    what makes the fit vectorizable.

    Because the LOO contribution is computed via order-invariant log-sum-exp
    reductions over the sample axis, the smoothing is done in *sorted* order
    and never scattered back to the original sample order.

    Parameters
    ----------
    log_liks : jnp.ndarray, shape ``(S, N)``
        Posterior log-likelihood matrix: ``S`` posterior draws by ``N``
        observations (cells).

    Returns
    -------
    dict of str to jnp.ndarray
        ``elpd_loo_i`` (shape ``(N,)``), ``k_hat`` (shape ``(N,)``), and
        ``lppd_i`` (shape ``(N,)``, the in-sample log pointwise predictive
        density per observation).  Scalar aggregates are assembled by the
        public :func:`compute_psis_loo` wrapper.
    """
    S = log_liks.shape[0]
    M = _n_tail(S)

    # Raw log importance weights w_s = -log p(y_i | theta^s), sorted ascending
    # along the sample axis (largest weights last = the tail to smooth).
    raw_lw = -log_liks  # (S, N)
    order = jnp.argsort(raw_lw, axis=0)  # (S, N)
    sorted_lw = jnp.take_along_axis(raw_lw, order, axis=0)  # (S, N)
    # Log-likelihoods reordered to match (so the LOO average pairs correctly).
    sorted_ll = jnp.take_along_axis(log_liks, order, axis=0)  # (S, N)

    # Tail: the M largest log-weights, overflow-safely normalized by their max.
    tail_lw = sorted_lw[S - M :, :]  # (M, N)
    tail_norm = tail_lw[-1, :]  # (N,)
    tail_w = jnp.exp(tail_lw - tail_norm[None, :])  # (M, N) in (0, 1]
    tail_w0 = tail_w[0, :]  # (N,)
    tail_excess = tail_w - tail_w0[None, :]  # (M, N), [0] == 0

    # Fit the GPD to every tail at once and build smoothed tail quantiles.
    k_hat, sigma_hat = _gpdfit_jax(tail_excess)  # (N,), (N,)
    probs = (jnp.arange(1, M + 1, dtype=log_liks.dtype) - 0.5) / M  # (M,)
    smooth_tail_w = tail_w0[None, :] + _gpd_inverse_cdf(
        probs, k_hat, sigma_hat
    )  # (M, N)
    smooth_tail_lw = tail_norm[None, :] + jnp.log(
        jnp.maximum(smooth_tail_w, 1e-300)
    )  # (M, N)

    # Truncate at log(S^0.75) (and never above the largest raw weight).
    max_lw = jnp.minimum(sorted_lw[-1, :], 0.75 * jnp.log(S))  # (N,)
    smooth_tail_lw = jnp.minimum(smooth_tail_lw, max_lw[None, :])

    # Degenerate (constant / near-constant) tails: skip smoothing, k_hat -> 0.
    # ``ptp`` of the sorted weights detects a flat column.
    is_constant = (sorted_lw[-1, :] - sorted_lw[0, :]) < 1e-10  # (N,)
    smooth_tail_lw = jnp.where(is_constant[None, :], tail_lw, smooth_tail_lw)
    k_hat = jnp.where(is_constant, 0.0, k_hat)

    # Smoothed weights in sorted order: bulk unchanged, tail replaced.
    smoothed_sorted = jnp.concatenate(
        [sorted_lw[: S - M, :], smooth_tail_lw], axis=0
    )  # (S, N)

    # LOO contribution: log E_w[p(y_i|theta)] / E_w[1]
    #   = logsumexp(smoothed_lw + log_lik) - logsumexp(smoothed_lw)
    # (order-invariant, so sorted order is fine).
    from jax.scipy.special import logsumexp

    log_num = logsumexp(smoothed_sorted + sorted_ll, axis=0)  # (N,)
    log_den = logsumexp(smoothed_sorted, axis=0)  # (N,)
    elpd_loo_i = log_num - log_den  # (N,)

    # In-sample lppd per observation (same definition as WAIC).
    lppd_i = logsumexp(log_liks, axis=0) - jnp.log(S)  # (N,)

    return {"elpd_loo_i": elpd_loo_i, "k_hat": k_hat, "lppd_i": lppd_i}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_psis_loo(
    log_liks: np.ndarray,
    dtype: type = np.float64,
    backend: str = "jax",
) -> Dict[str, np.ndarray]:
    """Compute PSIS-LOO statistics from a posterior log-likelihood matrix.

    For each observation i, applies Pareto-smoothed importance sampling to
    approximate the LOO predictive density without refitting the model.  The
    Pareto shape parameter k̂_i serves as a per-observation reliability
    diagnostic.

    Parameters
    ----------
    log_liks : array-like, shape ``(S, n)``
        Log-likelihood matrix: rows are posterior samples, columns are
        observations (cells).  Can be a JAX or NumPy array; internally
        converted to NumPy for Pareto fitting.
    dtype : numpy dtype, default=np.float64
        Numerical precision.  Double precision is recommended for PSIS-LOO
        because the Pareto fitting can be sensitive to precision.
    backend : {"jax", "scipy"}, default="jax"
        Which implementation to use.

        - ``"jax"`` (default): vectorized, GPU-accelerated path. Fits all tails
          at once with the closed-form Zhang--Stephens estimator
          (:func:`_gpdfit_jax`) and computes every observation's LOO
          contribution as batched ``jnp`` reductions. Scales to large cell
          counts; the only algorithmic difference from ``"scipy"`` is the GPD
          estimator (closed-form vs iterative MLE).
        - ``"scipy"``: the reference per-observation loop using
          ``scipy.stats.genpareto.fit``. Slower (Python loop over
          observations) but kept as a numerical reference and for environments
          without a usable JAX device.

    Returns
    -------
    dict
        Keys:

        ``elpd_loo`` : float
            Total estimated expected log predictive density.
        ``p_loo`` : float
            Effective number of parameters:
            ``p_loo = lppd - elpd_loo``.
        ``looic`` : float
            LOO information criterion on deviance scale:
            ``looic = -2 * elpd_loo``.
        ``elpd_loo_i`` : np.ndarray, shape ``(n,)``
            Per-observation LOO log predictive density.
        ``k_hat`` : np.ndarray, shape ``(n,)``
            Per-observation Pareto shape diagnostic.
        ``lppd`` : float
            In-sample log pointwise predictive density (same definition as
            in WAIC, provided for convenience).
        ``n_bad`` : int
            Number of observations with k̂ ≥ 0.7 (unreliable LOO
            contributions).

    Examples
    --------
    >>> import numpy as np
    >>> from scribe.mc._psis_loo import compute_psis_loo
    >>> rng = np.random.default_rng(0)
    >>> log_liks = rng.normal(-3.0, 0.5, size=(500, 200))
    >>> result = compute_psis_loo(log_liks)
    >>> print(result["k_hat"].max())
    """
    # Convert to double-precision NumPy for stable Pareto fitting
    log_liks = np.asarray(log_liks, dtype=dtype)
    S, n = log_liks.shape

    if backend == "jax":
        # Vectorized, GPU-accelerated path: fit every tail and evaluate every
        # observation's LOO contribution in a single batched computation.
        jax_result = _psis_loo_jax(jnp.asarray(log_liks))
        elpd_loo_i = np.asarray(jax_result["elpd_loo_i"], dtype=dtype)
        k_hat = np.asarray(jax_result["k_hat"], dtype=dtype)
        lppd_i = np.asarray(jax_result["lppd_i"], dtype=dtype)
    elif backend == "scipy":
        # Reference path: process each observation independently in Python.
        k_hat = np.zeros(n, dtype=dtype)
        elpd_loo_i = np.zeros(n, dtype=dtype)
        for i in range(n):
            # Raw log IS weights: log w_s = -log p(y_i | theta^s)
            raw_lw = -log_liks[:, i]

            # Pareto-smooth the weights
            smooth_lw, ki = _pareto_smooth_single(raw_lw)
            k_hat[i] = ki

            # Numerically stable log of the IS-weighted average:
            # log p_loo(y_i | y_{-i}) ≈
            #   log(sum_s exp(smooth_lw_s + log_lik_s))
            #   - log(sum_s exp(smooth_lw_s))
            log_lik_i = log_liks[:, i]

            # Numerator: log sum_s exp(smooth_lw_s + log_lik_s)
            log_num_terms = smooth_lw + log_lik_i
            log_num_max = np.max(log_num_terms)
            log_numerator = log_num_max + np.log(
                np.sum(np.exp(log_num_terms - log_num_max))
            )

            # Denominator: log sum_s exp(smooth_lw_s)
            log_den_max = np.max(smooth_lw)
            log_denominator = log_den_max + np.log(
                np.sum(np.exp(smooth_lw - log_den_max))
            )

            elpd_loo_i[i] = log_numerator - log_denominator

        # In-sample lppd (same formula as WAIC, for reference)
        lse_max = np.max(log_liks, axis=0)  # shape (n,)
        lppd_i = lse_max + np.log(np.mean(np.exp(log_liks - lse_max), axis=0))
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'jax' or 'scipy'."
        )

    # Aggregate (shared across backends)
    lppd = float(np.sum(lppd_i))
    elpd_loo = float(np.sum(elpd_loo_i))
    p_loo = lppd - elpd_loo
    looic = -2.0 * elpd_loo

    return {
        "elpd_loo": elpd_loo,
        "p_loo": p_loo,
        "looic": looic,
        "elpd_loo_i": elpd_loo_i,
        "k_hat": k_hat,
        "lppd": lppd,
        "n_bad": int(np.sum(k_hat >= 0.7)),
    }


def psis_loo_summary(result: dict) -> str:
    """Format a human-readable summary of PSIS-LOO diagnostics.

    Parameters
    ----------
    result : dict
        Output of :func:`compute_psis_loo`.

    Returns
    -------
    str
        A multi-line summary string.

    Examples
    --------
    >>> print(psis_loo_summary(result))
    """
    k = result["k_hat"]
    n = len(k)
    n_ok = int(np.sum(k < 0.5))
    n_ok2 = int(np.sum((k >= 0.5) & (k < 0.7)))
    n_bad = int(np.sum(k >= 0.7))

    lines = [
        "PSIS-LOO Summary",
        "=" * 40,
        f"  elpd_loo : {result['elpd_loo']:.2f}",
        f"  p_loo    : {result['p_loo']:.2f}",
        f"  LOO-IC   : {result['looic']:.2f}",
        "",
        f"  Pareto k̂ diagnostics (n={n} observations):",
        f"    k̂ < 0.5   (excellent)    : {n_ok:5d}  ({100*n_ok/n:5.1f}%)",
        f"    0.5 ≤ k̂ < 0.7 (OK)      : {n_ok2:5d}  ({100*n_ok2/n:5.1f}%)",
        f"    k̂ ≥ 0.7   (problematic) : {n_bad:5d}  ({100*n_bad/n:5.1f}%)",
    ]
    if n_bad > 0:
        lines.append(
            f"\n  WARNING: {n_bad} observations have k̂ ≥ 0.7."
            " LOO estimates may be unreliable for these cells."
        )
    return "\n".join(lines)
