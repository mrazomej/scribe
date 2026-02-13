"""Gene-level differential expression analysis.

This module provides Bayesian differential expression analysis at the gene
level, computing exact analytic posteriors under the Gaussian assumption in
CLR space.  The key function is ``differential_expression``, which takes two
fitted logistic-normal models and returns per-gene posterior summaries
including effect sizes, posterior probabilities, and local false sign rates.
"""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.stats import norm

from ._transforms import transform_gaussian_alr_to_clr
from ._extract import extract_alr_params


# --------------------------------------------------------------------------
# Differential expression at the gene level
# --------------------------------------------------------------------------


def differential_expression(
    model_A,
    model_B,
    tau: float = 0.0,
    coordinate: str = "clr",
    gene_names: Optional[list] = None,
) -> dict:
    """Compute gene-level differential expression in a Bayesian framework.

    Under the fitted logistic-normal models, we have::

        z_A ~ N(mu_A, Sigma_A)   in CLR space
        z_B ~ N(mu_B, Sigma_B)   in CLR space

    The difference ``Delta = z_A - z_B ~ N(mu_A - mu_B, Sigma_A + Sigma_B)``
    is also Gaussian, providing exact analytic posteriors for each gene's
    effect size.

    This is a fully Bayesian approach that:

    - Works in CLR space (reference-invariant).
    - Accounts for correlation structure via low-rank covariance.
    - Provides exact posterior probabilities (not p-values).
    - Controls Bayesian error rates (lfsr, not FDR).

    Parameters
    ----------
    model_A : dict or LowRankLogisticNormal or SoftmaxNormal
        Fitted logistic-normal model for condition A.  Either a dict with
        ``'loc'``, ``'cov_factor'``, ``'cov_diag'`` keys, or a Distribution
        object with these attributes.  Parameters may be (D-1)-dim raw ALR
        or D-dim embedded ALR -- ``extract_alr_params`` handles both.
    model_B : dict or LowRankLogisticNormal or SoftmaxNormal
        Fitted logistic-normal model for condition B.
    tau : float, default=0.0
        Practical significance threshold in log-scale.
        For example, ``log(1.1) ~ 0.095`` for 10 % fold-change.
    coordinate : str, default='clr'
        Coordinate system for results.  Currently only ``'clr'`` is
        supported.
    gene_names : list of str, optional
        Gene names for output.  If ``None``, uses generic names.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - **delta_mean** : ndarray, shape ``(D,)``
            Posterior mean effect per gene (log-fold-change in CLR).
        - **delta_sd** : ndarray, shape ``(D,)``
            Posterior standard deviation per gene.
        - **prob_positive** : ndarray, shape ``(D,)``
            ``P(Delta > 0 | data)`` -- posterior probability of positive
            effect.
        - **prob_effect** : ndarray, shape ``(D,)``
            ``P(|Delta| > tau | data)`` -- posterior probability of
            practical effect.
        - **lfsr** : ndarray, shape ``(D,)``
            Local false sign rate = ``min(P(Delta <= 0), P(Delta >= 0))``.
            This is the Bayesian error rate (NOT FDR).
        - **lfsr_tau** : ndarray, shape ``(D,)``
            Modified local false sign rate incorporating practical
            significance, as defined in the paper::

                lfsr_g(tau) = 1 - max(P(Delta_g > tau), P(Delta_g < -tau))

            When ``tau=0`` this equals ``lfsr``.  Use this with
            ``compute_pefp(use_lfsr_tau=True)`` for the paper's
            practical-significance-aware error control.
        - **gene_names** : list of str
            Gene names (input or generated).

    Notes
    -----
    The returned probabilities are exact under the Gaussian assumption.
    They are true posterior probabilities, not frequentist p-values.

    The local false sign rate (lfsr) is the posterior probability of
    having the wrong sign -- a Bayesian analogue of the two-sided p-value.

    Examples
    --------
    >>> model_a = {'loc': jnp.zeros(100), 'cov_factor': jnp.eye(100, 10),
    ...            'cov_diag': jnp.ones(100)}
    >>> model_b = {'loc': jnp.ones(100) * 0.2,
    ...            'cov_factor': jnp.eye(100, 10),
    ...            'cov_diag': jnp.ones(100)}
    >>> results = differential_expression(
    ...     model_a, model_b, tau=jnp.log(1.1)
    ... )
    """
    # Validate the requested coordinate system
    if coordinate != "clr":
        raise NotImplementedError(
            f"Coordinate system '{coordinate}' not yet implemented. "
            f"Currently only 'clr' is supported."
        )

    # 1. Extract (D-1)-dimensional ALR parameters (handles embedded ALR)
    mu_A, W_A, d_A = extract_alr_params(model_A)
    mu_B, W_B, d_B = extract_alr_params(model_B)

    # 2. Transform to CLR space (exact transformation with proper diagonal)
    mu_A_clr, W_A_clr, d_A_clr = transform_gaussian_alr_to_clr(mu_A, W_A, d_A)
    mu_B_clr, W_B_clr, d_B_clr = transform_gaussian_alr_to_clr(mu_B, W_B, d_B)

    # 3. Compute difference distribution:
    # Delta ~ N(mu_A - mu_B, Sigma_A + Sigma_B)
    delta_mean = mu_A_clr - mu_B_clr

    # 4. Compute marginal variances efficiently (diagonal of Sigma_A + Sigma_B)
    # For low-rank Sigma = W W^T + diag(d), the diagonal is:
    #   diag(Sigma) = ||W[i,:]||^2 + d[i]
    var_A = jnp.sum(W_A_clr**2, axis=-1) + d_A_clr
    var_B = jnp.sum(W_B_clr**2, axis=-1) + d_B_clr
    delta_sd = jnp.sqrt(var_A + var_B)

    # Guard against near-zero SD (can happen when both models have
    # negligible variance for a gene, e.g. near-constant expression).
    # Without this, z_scores and probabilities become inf/NaN.
    delta_sd = jnp.maximum(delta_sd, 1e-30)

    # 5. Compute posterior probabilities (exact under Gaussian assumption)
    z_scores = delta_mean / delta_sd
    # P(Delta > 0 | data) via standard normal CDF
    prob_positive = norm.cdf(z_scores)

    # P(Delta > tau | data) and P(Delta < -tau | data)
    prob_up = 1 - norm.cdf(tau, loc=delta_mean, scale=delta_sd)
    prob_down = norm.cdf(-tau, loc=delta_mean, scale=delta_sd)

    # P(|Delta| > tau | data) = P(Delta > tau) + P(Delta < -tau)
    prob_effect = prob_up + prob_down

    # Local false sign rate (Bayesian error rate)
    # lfsr = posterior probability of wrong sign
    lfsr = jnp.minimum(prob_positive, 1 - prob_positive)

    # Modified local false sign rate incorporating practical significance
    # (paper Eq. from _diffexp06.qmd, lines 333-394):
    #   lfsr_g(tau) = 1 - max(P(Delta_g > tau | data), P(Delta_g < -tau | data))
    # When tau=0 this reduces to standard lfsr.
    lfsr_tau = 1.0 - jnp.maximum(prob_up, prob_down)

    # Generate gene names if not provided
    if gene_names is None:
        D = len(delta_mean)
        gene_names = [f"gene_{i}" for i in range(D)]

    return {
        "delta_mean": delta_mean,
        "delta_sd": delta_sd,
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "lfsr_tau": lfsr_tau,
        "gene_names": gene_names,
    }


# --------------------------------------------------------------------------
# Call DE genes using Bayesian decision rules
# --------------------------------------------------------------------------


def call_de_genes(
    de_results: dict,
    lfsr_threshold: float = 0.05,
    prob_effect_threshold: float = 0.95,
) -> jnp.ndarray:
    """Call DE genes using Bayesian decision rules.

    A gene is called differentially expressed if:

    1. ``lfsr < lfsr_threshold`` (confident about sign direction).
    2. ``prob_effect > prob_effect_threshold`` (practically significant).

    This is a fully Bayesian approach that controls posterior error rates,
    not frequentist error rates like FDR.

    Parameters
    ----------
    de_results : dict
        Output from ``differential_expression()``.
    lfsr_threshold : float, default=0.05
        Maximum acceptable local false sign rate.
    prob_effect_threshold : float, default=0.95
        Minimum posterior probability of practical effect.

    Returns
    -------
    ndarray of bool, shape ``(D,)``
        Boolean mask indicating which genes are called as DE.

    Examples
    --------
    >>> de_results = differential_expression(model_a, model_b,
    ...                                       tau=jnp.log(1.1))
    >>> is_de = call_de_genes(de_results, lfsr_threshold=0.05,
    ...                        prob_effect_threshold=0.95)
    >>> n_de = is_de.sum()
    """
    return (de_results["lfsr"] < lfsr_threshold) & (
        de_results["prob_effect"] > prob_effect_threshold
    )
