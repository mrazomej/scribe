"""Empirical Bayes shrinkage for differential expression.

This module implements a scale-mixture-of-normals empirical Bayes procedure that
uses the *genome-wide* distribution of effects to improve per-gene inference.
The key idea is that most genes are not differentially expressed, and this
global sparsity structure can be learned from the data and fed back into each
gene's posterior.

The pipeline operates on the summary statistics produced by the empirical DE
path (``delta_mean``, ``delta_sd`` per gene) and outputs shrinkage-adjusted lfsr
values that are fully compatible with the existing PEFP error-control machinery.

Main functions
--------------
- ``fit_scale_mixture_prior`` : EM algorithm for estimating mixture
  weights on a fixed grid of prior scales.
- ``shrinkage_posterior`` : Compute the per-gene shrinkage posterior
  (mixture of Gaussians) given estimated prior.
- ``shrinkage_differential_expression`` : End-to-end convenience
  wrapper that fits the prior and computes all shrunk DE statistics.
- ``default_sigma_grid`` : Build a sensible default grid of prior
  scales from the data.

Mathematical details are in Section 10 of the paper
(``paper/_diffexp10.qmd``).
"""

from typing import Optional, Dict, Any

import jax.numpy as jnp
from jax.scipy.stats import norm


# --------------------------------------------------------------------------
# Default scale grid construction
# --------------------------------------------------------------------------


def default_sigma_grid(
    delta_sd: jnp.ndarray,
    K: int = 25,
    sigma_min: float = 1e-6,
    sigma_max_mult: float = 3.0,
) -> jnp.ndarray:
    """Build a geometric grid of prior scales from the observed data.

    The grid spans from ``sigma_min`` (approximating a point mass at
    zero for the null component) to ``sigma_max_mult`` times the
    maximum observed absolute effect, in ``K + 1`` geometrically
    spaced points.

    Parameters
    ----------
    delta_sd : jnp.ndarray, shape ``(D,)``
        Posterior standard deviations per gene (used to set the upper
        bound of the grid).
    K : int, default=25
        Number of non-null grid points.  The total grid has ``K + 1``
        points (including the near-zero null component).
    sigma_min : float, default=1e-6
        Smallest scale (approximates the null point mass).
    sigma_max_mult : float, default=3.0
        The maximum scale is ``sigma_max_mult`` times the maximum
        observed ``delta_sd``.

    Returns
    -------
    jnp.ndarray, shape ``(K + 1,)``
        Geometric grid of prior standard deviations.
    """
    sigma_max = sigma_max_mult * float(jnp.max(delta_sd))
    # Geometric sequence from sigma_min to sigma_max
    return jnp.geomspace(sigma_min, sigma_max, K + 1)


# --------------------------------------------------------------------------
# EM algorithm for fitting the scale mixture prior
# --------------------------------------------------------------------------


def fit_scale_mixture_prior(
    delta_mean: jnp.ndarray,
    delta_sd: jnp.ndarray,
    sigma_grid: Optional[jnp.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Estimate scale-mixture-of-normals prior via EM.

    Fits the model::

        beta_g ~ sum_k w_k N(0, sigma_k^2)       (prior)
        hat{Delta}_g | beta_g ~ N(beta_g, s_g^2)  (observation)

    by maximising the marginal log-likelihood over the mixture weights
    ``w`` with the EM algorithm.

    Parameters
    ----------
    delta_mean : jnp.ndarray, shape ``(D,)``
        Posterior mean CLR difference per gene.
    delta_sd : jnp.ndarray, shape ``(D,)``
        Posterior standard deviation per gene.
    sigma_grid : jnp.ndarray, shape ``(K+1,)``, optional
        Grid of prior standard deviations.  If ``None``, uses
        ``default_sigma_grid(delta_sd)``.
    max_iter : int, default=200
        Maximum number of EM iterations.
    tol : float, default=1e-8
        Convergence tolerance on the change in log-likelihood.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - **weights** : jnp.ndarray, shape ``(K+1,)``
            Estimated mixture weights.
        - **sigma_grid** : jnp.ndarray, shape ``(K+1,)``
            The prior scale grid used.
        - **posterior_probs** : jnp.ndarray, shape ``(D, K+1)``
            Responsibility matrix ``gamma_{gk}`` at convergence.
        - **null_proportion** : float
            Estimated null fraction ``w_0``.
        - **n_iter** : int
            Number of EM iterations to convergence.
        - **log_likelihood** : float
            Final marginal log-likelihood.
        - **converged** : bool
            Whether the algorithm converged before ``max_iter``.
    """
    delta_mean = jnp.asarray(delta_mean)
    delta_sd = jnp.asarray(delta_sd)

    if sigma_grid is None:
        sigma_grid = default_sigma_grid(delta_sd)
    sigma_grid = jnp.asarray(sigma_grid)

    D = delta_mean.shape[0]
    K_plus_1 = sigma_grid.shape[0]

    # Marginal variances: sigma_k^2 + s_g^2, shape (D, K+1)
    marginal_var = sigma_grid[None, :] ** 2 + delta_sd[:, None] ** 2

    # Precompute the (D, K+1) matrix of Gaussian log-densities
    # log N(hat{Delta}_g | 0, sigma_k^2 + s_g^2)
    log_densities = norm.logpdf(
        delta_mean[:, None],
        loc=0.0,
        scale=jnp.sqrt(marginal_var),
    )

    # Initialise with uniform weights
    weights = jnp.ones(K_plus_1) / K_plus_1
    prev_ll = -jnp.inf
    converged = False

    for iteration in range(max_iter):
        # --- E-step ---
        # log(w_k * N(...)) = log(w_k) + log_densities[g, k]
        log_numerator = jnp.log(weights[None, :]) + log_densities
        # Log-sum-exp for numerical stability
        log_denominator = jnp.max(
            log_numerator, axis=1, keepdims=True
        ) + jnp.log(
            jnp.sum(
                jnp.exp(
                    log_numerator
                    - jnp.max(log_numerator, axis=1, keepdims=True)
                ),
                axis=1,
                keepdims=True,
            )
        )
        log_gamma = log_numerator - log_denominator
        gamma = jnp.exp(log_gamma)

        # --- Log-likelihood ---
        ll = float(jnp.sum(log_denominator))
        if jnp.abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

        # --- M-step ---
        weights = jnp.mean(gamma, axis=0)
        # Guard against zero weights (add tiny floor and renormalize)
        weights = jnp.maximum(weights, 1e-15)
        weights = weights / jnp.sum(weights)

    return {
        "weights": weights,
        "sigma_grid": sigma_grid,
        "posterior_probs": gamma,
        "null_proportion": float(weights[0]),
        "n_iter": iteration + 1,
        "log_likelihood": float(ll),
        "converged": converged,
    }


# --------------------------------------------------------------------------
# Shrinkage posterior computation
# --------------------------------------------------------------------------


def shrinkage_posterior(
    delta_mean: jnp.ndarray,
    delta_sd: jnp.ndarray,
    weights: jnp.ndarray,
    sigma_grid: jnp.ndarray,
    posterior_probs: Optional[jnp.ndarray] = None,
) -> Dict[str, Any]:
    """Compute per-gene shrinkage posterior from the fitted prior.

    Given the estimated mixture weights and grid, computes the
    posterior mean, standard deviation, and component parameters for
    each gene.

    Parameters
    ----------
    delta_mean : jnp.ndarray, shape ``(D,)``
        Posterior mean CLR difference per gene.
    delta_sd : jnp.ndarray, shape ``(D,)``
        Posterior standard deviation per gene.
    weights : jnp.ndarray, shape ``(K+1,)``
        Estimated mixture weights from ``fit_scale_mixture_prior``.
    sigma_grid : jnp.ndarray, shape ``(K+1,)``
        Prior scale grid.
    posterior_probs : jnp.ndarray, shape ``(D, K+1)``, optional
        Responsibility matrix.  If ``None``, recomputed from the
        weights and data.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - **shrunk_mean** : jnp.ndarray, shape ``(D,)``
            Posterior mean of the true effect (shrunk estimate).
        - **shrunk_sd** : jnp.ndarray, shape ``(D,)``
            Posterior standard deviation of the true effect.
        - **component_means** : jnp.ndarray, shape ``(D, K+1)``
            Per-component posterior means ``m_{gk}``.
        - **component_variances** : jnp.ndarray, shape ``(D, K+1)``
            Per-component posterior variances ``v_{gk}``.
        - **component_weights** : jnp.ndarray, shape ``(D, K+1)``
            Responsibility matrix ``gamma_{gk}``.
    """
    delta_mean = jnp.asarray(delta_mean)
    delta_sd = jnp.asarray(delta_sd)
    weights = jnp.asarray(weights)
    sigma_grid = jnp.asarray(sigma_grid)

    sigma_sq = sigma_grid[None, :] ** 2  # (1, K+1)
    s_sq = delta_sd[:, None] ** 2  # (D, 1)

    # Per-component posterior mean:
    # m_{gk} = hat{Delta}_g * sigma_k^2 / (sigma_k^2 + s_g^2)
    denom = sigma_sq + s_sq  # (D, K+1)
    component_means = delta_mean[:, None] * sigma_sq / denom  # (D, K+1)

    # Per-component posterior variance:
    # v_{gk} = sigma_k^2 * s_g^2 / (sigma_k^2 + s_g^2)
    component_variances = sigma_sq * s_sq / denom  # (D, K+1)

    # Recompute responsibilities if not provided
    if posterior_probs is None:
        marginal_var = sigma_grid[None, :] ** 2 + delta_sd[:, None] ** 2
        log_densities = norm.logpdf(
            delta_mean[:, None], loc=0.0, scale=jnp.sqrt(marginal_var)
        )
        log_numerator = jnp.log(weights[None, :]) + log_densities
        log_max = jnp.max(log_numerator, axis=1, keepdims=True)
        log_gamma = (
            log_numerator
            - log_max
            - jnp.log(
                jnp.sum(jnp.exp(log_numerator - log_max), axis=1, keepdims=True)
            )
        )
        posterior_probs = jnp.exp(log_gamma)

    gamma = posterior_probs

    # Marginal posterior mean: E[beta_g | data] = sum_k gamma_{gk} * m_{gk}
    shrunk_mean = jnp.sum(gamma * component_means, axis=1)

    # Marginal posterior variance via law of total variance:
    # Var = E[V_k] + E[M_k^2] - (E[M_k])^2
    within_var = jnp.sum(gamma * component_variances, axis=1)
    mean_of_sq = jnp.sum(gamma * component_means**2, axis=1)
    shrunk_var = within_var + mean_of_sq - shrunk_mean**2
    shrunk_sd = jnp.sqrt(jnp.maximum(shrunk_var, 0.0))

    return {
        "shrunk_mean": shrunk_mean,
        "shrunk_sd": shrunk_sd,
        "component_means": component_means,
        "component_variances": component_variances,
        "component_weights": gamma,
    }


# --------------------------------------------------------------------------
# Shrunk DE statistics
# --------------------------------------------------------------------------


def shrinkage_differential_expression(
    delta_mean: jnp.ndarray,
    delta_sd: jnp.ndarray,
    tau: float = 0.0,
    gene_names=None,
    sigma_grid: Optional[jnp.ndarray] = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, Any]:
    """Compute shrinkage-adjusted DE statistics.

    End-to-end convenience function that fits the scale mixture prior
    via EM, computes the shrinkage posterior for each gene, and derives
    all gene-level DE statistics (shrunk lfsr, prob_positive, etc.).

    Parameters
    ----------
    delta_mean : jnp.ndarray, shape ``(D,)``
        Posterior mean CLR difference per gene.
    delta_sd : jnp.ndarray, shape ``(D,)``
        Posterior standard deviation per gene.
    tau : float, default=0.0
        Practical significance threshold (log-scale).
    gene_names : list of str, optional
        Gene names.  If ``None``, generic names are generated.
    sigma_grid : jnp.ndarray, shape ``(K+1,)``, optional
        Prior scale grid.  If ``None``, uses ``default_sigma_grid``.
    max_iter : int, default=200
        Maximum EM iterations.
    tol : float, default=1e-8
        EM convergence tolerance.

    Returns
    -------
    dict
        Dictionary with all standard DE keys (``delta_mean``,
        ``delta_sd``, ``prob_positive``, ``prob_effect``, ``lfsr``,
        ``lfsr_tau``, ``gene_names``) plus shrinkage-specific extras:

        - **null_proportion** : float
            Estimated fraction of truly null genes.
        - **prior_weights** : jnp.ndarray, shape ``(K+1,)``
            Estimated mixture weights.
        - **sigma_grid** : jnp.ndarray, shape ``(K+1,)``
            Prior scale grid used.
        - **em_converged** : bool
            Whether EM converged.
        - **em_n_iter** : int
            Number of EM iterations.
        - **em_log_likelihood** : float
            Final marginal log-likelihood.
    """
    delta_mean = jnp.asarray(delta_mean)
    delta_sd = jnp.asarray(delta_sd)

    # Step 1: Fit the prior
    em_result = fit_scale_mixture_prior(
        delta_mean,
        delta_sd,
        sigma_grid=sigma_grid,
        max_iter=max_iter,
        tol=tol,
    )

    # Step 2: Compute the shrinkage posterior
    post = shrinkage_posterior(
        delta_mean,
        delta_sd,
        weights=em_result["weights"],
        sigma_grid=em_result["sigma_grid"],
        posterior_probs=em_result["posterior_probs"],
    )

    gamma = post["component_weights"]  # (D, K+1)
    m = post["component_means"]  # (D, K+1)
    v = post["component_variances"]  # (D, K+1)
    sqrt_v = jnp.sqrt(jnp.maximum(v, 1e-30))

    # Step 3: Shrunk probability of positive effect
    # P(beta_g > 0) = sum_k gamma_{gk} Phi(m_{gk} / sqrt(v_{gk}))
    prob_positive = jnp.sum(gamma * norm.cdf(m / sqrt_v), axis=1)

    # Step 4: Shrunk lfsr
    lfsr = jnp.minimum(prob_positive, 1.0 - prob_positive)

    # Step 5: Practical significance
    # P(beta_g > tau) = sum_k gamma_{gk} Phi((m_{gk} - tau) / sqrt(v_{gk}))
    prob_up = jnp.sum(gamma * norm.cdf((m - tau) / sqrt_v), axis=1)
    # P(beta_g < -tau) = sum_k gamma_{gk} Phi(-(m_{gk} + tau) / sqrt(v_{gk}))
    prob_down = jnp.sum(gamma * norm.cdf(-(m + tau) / sqrt_v), axis=1)
    prob_effect = prob_up + prob_down
    lfsr_tau = 1.0 - jnp.maximum(prob_up, prob_down)

    # Gene names
    if gene_names is None:
        D = delta_mean.shape[0]
        gene_names = [f"gene_{i}" for i in range(D)]

    return {
        # Standard DE keys (using shrunk estimates)
        "delta_mean": post["shrunk_mean"],
        "delta_sd": post["shrunk_sd"],
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "lfsr_tau": lfsr_tau,
        "gene_names": gene_names,
        # Shrinkage-specific extras
        "null_proportion": em_result["null_proportion"],
        "prior_weights": em_result["weights"],
        "sigma_grid": em_result["sigma_grid"],
        "em_converged": em_result["converged"],
        "em_n_iter": em_result["n_iter"],
        "em_log_likelihood": em_result["log_likelihood"],
    }
