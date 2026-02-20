"""Model stacking: optimal predictive ensemble weights.

This module implements **model stacking** @yao2018, which optimizes a
convex combination of K models to maximize the leave-one-out log predictive
score of the ensemble.  Stacking weights outperform simple model selection
(winner-take-all) and pseudo-BMA (AIC-like) weights when models are
misspecified or closely related.

The optimization problem
------------------------
Given K models and n observations, the stacking weights solve:

    w* = argmax_{w in Delta^{K-1}} sum_i log sum_k w_k * p_loo(y_i | y_{-i}, M_k)

where Delta^{K-1} = {w in R^K : w_k >= 0, sum_k w_k = 1} is the K-simplex
and p_loo(y_i | y_{-i}, M_k) is the LOO predictive density from model k.

This is a concave maximization (equivalently, a convex minimization) over a
convex set, so it has a unique solution found efficiently by standard
first-order methods.

References
----------
Yao, Vehtari, Simpson, Gelman (2018), "Using Stacking to Average Bayesian
    Predictive Distributions." Bayesian Analysis.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_mix_loo(weights: np.ndarray, log_loo_i: np.ndarray) -> float:
    """Compute the negative stacking log-score objective.

    The stacking objective is:

        L(w) = sum_i log sum_k w_k * exp(loo_log_i_k)
             = sum_i logsumexp(loo_log_i_k + log(w_k))

    We minimize -L(w) (negative log-score).

    Parameters
    ----------
    weights : np.ndarray, shape ``(K,)``
        Current weight vector (must be on the simplex).  scipy passes this
        as the first argument when calling ``fun(x, *args)``.
    log_loo_i : np.ndarray, shape ``(n, K)``
        Per-observation LOO log predictive densities for each model.

    Returns
    -------
    float
        Negative stacking log-score (to be minimized).
    """
    log_w = np.log(np.maximum(weights, 1e-300))
    # log sum_k w_k * exp(loo_log_i_k) = logsumexp(loo_log_i_k + log_w_k, axis=1)
    log_mix = logsumexp(log_loo_i + log_w[np.newaxis, :], axis=1)
    return -float(np.sum(log_mix))


def _grad_log_mix_loo(weights: np.ndarray, log_loo_i: np.ndarray) -> np.ndarray:
    """Gradient of the negative stacking log-score with respect to weights.

    Parameters
    ----------
    weights : np.ndarray, shape ``(K,)``
        Current weight vector.  scipy passes this as the first argument when
        calling ``jac(x, *args)``.
    log_loo_i : np.ndarray, shape ``(n, K)``
        Per-observation LOO log predictive densities.

    Returns
    -------
    np.ndarray, shape ``(K,)``
        Gradient of -L(w) with respect to weights.
    """
    log_w = np.log(np.maximum(weights, 1e-300))
    log_mix = logsumexp(log_loo_i + log_w[np.newaxis, :], axis=1)  # shape (n,)
    # Gradient: d(-L)/d(w_k) = -sum_i exp(loo_log_i_k + log_w_k) / mix_i / w_k
    # = -sum_i exp(loo_log_i_k) / (sum_j w_j exp(loo_log_i_j)) / 1
    mix = np.exp(log_mix)  # shape (n,)
    # softmax-style per-obs contribution: shape (n, K)
    contributions = np.exp(log_loo_i + log_w[np.newaxis, :]) / mix[:, np.newaxis]
    # Gradient of L w.r.t. w_k: sum_i contributions[i, k] / w_k
    grad_L = np.sum(contributions / weights[np.newaxis, :], axis=0)
    return -grad_L  # gradient of -L


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_stacking_weights(
    loo_log_densities: List[np.ndarray],
    n_restarts: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute optimal model stacking weights via convex optimization.

    Solves the stacking problem:

        w* = argmax_{w in Delta^{K-1}} sum_i log sum_k w_k * exp(loo_i_k)

    using scipy's SLSQP solver with multiple random restarts to guard against
    local solutions (though the problem is strictly convex so local = global).

    Parameters
    ----------
    loo_log_densities : list of np.ndarray
        List of K arrays, each of shape ``(n,)``, containing the per-observation
        LOO log predictive densities ``log p_loo(y_i | y_{-i}, M_k)`` for model k.
        These are the ``elpd_loo_i`` arrays from :func:`~scribe.mc._psis_loo.compute_psis_loo`.
    n_restarts : int, default=5
        Number of random initializations.  The best solution across restarts
        is returned.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape ``(K,)``
        Optimal stacking weights, summing to 1.  A weight near 0 means the
        corresponding model contributes negligibly to the optimal ensemble.

    Examples
    --------
    >>> import numpy as np
    >>> from scribe.mc._stacking import compute_stacking_weights
    >>> rng = np.random.default_rng(0)
    >>> K, n = 3, 200
    >>> # Model 1 is better; it has higher LOO densities
    >>> loo1 = rng.normal(-2.0, 0.3, n)
    >>> loo2 = rng.normal(-2.5, 0.3, n)
    >>> loo3 = rng.normal(-3.0, 0.3, n)
    >>> w = compute_stacking_weights([loo1, loo2, loo3])
    >>> print(w)  # Should be concentrated on model 1
    """
    # Stack into (n, K) matrix
    log_loo_i = np.column_stack([np.asarray(l, dtype=np.float64) for l in loo_log_densities])
    K = log_loo_i.shape[1]

    rng = np.random.default_rng(seed)

    # Simplex constraints
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(1e-6, 1.0)] * K

    best_val = np.inf
    best_w = np.ones(K) / K  # uniform fallback

    for _ in range(n_restarts):
        # Random initialization on the simplex
        w0 = rng.dirichlet(np.ones(K))
        # scipy calls fun(x, *args) and jac(x, *args), so log_loo_i is
        # passed as the second positional argument to both functions.
        result = minimize(
            fun=_log_mix_loo,
            x0=w0,
            args=(log_loo_i,),
            jac=_grad_log_mix_loo,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_w = result.x

    # Project back to simplex (clip negatives from numerical noise)
    best_w = np.clip(best_w, 0.0, 1.0)
    best_w /= best_w.sum()
    return best_w


def stacking_summary(
    weights: np.ndarray,
    model_names: Optional[List[str]] = None,
) -> str:
    """Format a human-readable summary of stacking weights.

    Parameters
    ----------
    weights : np.ndarray, shape ``(K,)``
        Stacking weights.
    model_names : list of str, optional
        Names for the K models.

    Returns
    -------
    str
        Formatted summary string.
    """
    K = len(weights)
    if model_names is None:
        model_names = [f"Model {k}" for k in range(K)]

    # Sort by weight descending
    order = np.argsort(weights)[::-1]
    lines = ["Stacking Weights", "=" * 30]
    for k in order:
        lines.append(f"  {model_names[k]:30s}  {weights[k]:.4f}")
    return "\n".join(lines)
