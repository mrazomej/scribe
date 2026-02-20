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

    # Threshold: the (S-M)-th largest value (boundary of the tail)
    cutoff_lw = sorted_lw[S - M - 1] if S > M else sorted_lw[0]

    # Tail values on the original (exponentiated) scale, shifted to excess
    tail_lw = sorted_lw[S - M:]
    # Exponentiate relative to the cutoff for numerical stability
    tail_w = np.exp(tail_lw - cutoff_lw)
    tail_excess = tail_w - tail_w[0]

    # Fit GPD to excess values
    k_hat, sigma_hat = _fit_gpd(tail_excess)

    # Replace tail with GPD quantiles at (j - 0.5) / M for j = 1, ..., M
    probs = (np.arange(1, M + 1) - 0.5) / M

    # Handle degenerate GPD (k=0 is Exponential, k→0 limit)
    if abs(k_hat) < 1e-6:
        # Exponential quantile: -sigma * log(1 - p)
        smooth_tail_w = tail_w[0] + sigma_hat * (-np.log(1.0 - probs))
    else:
        smooth_tail_w = tail_w[0] + stats.genpareto.ppf(probs, k_hat, loc=0.0, scale=sigma_hat)

    # Convert smoothed tail back to log scale
    smooth_tail_lw = cutoff_lw + np.log(np.maximum(smooth_tail_w, 1e-300))

    # Truncate at log(S^0.75) to prevent runaway weights
    max_lw = min(sorted_lw[-1], 0.75 * np.log(S))
    smooth_tail_lw = np.minimum(smooth_tail_lw, max_lw)

    # Assemble: only the M tail positions are replaced
    smoothed_lw = log_weights.copy()
    smoothed_lw[sort_idx[S - M:]] = smooth_tail_lw

    return smoothed_lw, float(k_hat)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_psis_loo(
    log_liks: np.ndarray,
    dtype: type = np.float64,
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

    # Storage for smoothed weights and diagnostics
    k_hat = np.zeros(n, dtype=dtype)
    elpd_loo_i = np.zeros(n, dtype=dtype)

    # Process each observation independently
    for i in range(n):
        # Raw log IS weights: log w_s = -log p(y_i | theta^s)
        raw_lw = -log_liks[:, i]

        # Pareto-smooth the weights
        smooth_lw, ki = _pareto_smooth_single(raw_lw)
        k_hat[i] = ki

        # Numerically stable log of the IS-weighted average:
        # log p_loo(y_i | y_{-i}) ≈
        #   log(sum_s exp(smooth_lw_s + log_lik_s)) - log(sum_s exp(smooth_lw_s))
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
    lse_max = np.max(log_liks, axis=0)          # shape (n,)
    lppd_i = lse_max + np.log(np.mean(np.exp(log_liks - lse_max), axis=0))
    lppd = float(np.sum(lppd_i))

    # Aggregate
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
