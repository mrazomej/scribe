"""
Gene-set and pathway-level differential expression analysis.

This module provides tools for testing linear contrasts and gene sets,
enabling pathway-level inference using compositional balances.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm

from .transformations import transform_gaussian_alr_to_clr
from ...stats.divergences import _extract_lowrank_params

# ------------------------------------------------------------------------------
# Test linear contrasts (Bayesian inference)
# ------------------------------------------------------------------------------


def test_contrast(
    model_A,
    model_B,
    contrast: jnp.ndarray,
    tau: float = 0.0,
) -> dict:
    """
    Test a linear contrast cᵀΔ (Bayesian inference).

    For a contrast vector c, we compute the posterior distribution of
    cᵀ(z_A - z_B) which is N(cᵀ(μ_A - μ_B), cᵀ(Σ_A + Σ_B)c).

    This is useful for:
        - Testing custom gene combinations
        - Pathway analysis via balances
        - Module-level effects

    Parameters
    ----------
    model_A : dict or Distribution
        Fitted logistic-normal model for condition A
    model_B : dict or Distribution
        Fitted logistic-normal model for condition B
    contrast : ndarray, shape (D,)
        Contrast vector in CLR space. For balances, this should
        compare one group of genes vs another (e.g., pathway vs rest).
    tau : float, default=0.0
        Practical significance threshold (log-scale)

    Returns
    -------
    dict
        Dictionary with the following keys:

        - delta_mean : float
            Posterior mean of cᵀΔ
        - delta_sd : float
            Posterior standard deviation of cᵀΔ
        - z_score : float
            Standardized effect size
        - prob_positive : float
            P(cᵀΔ > 0 | data)
        - prob_effect : float
            P(|cᵀΔ| > tau | data)
        - lfsr : float
            Local false sign rate for the contrast

    Notes
    -----
    The contrast should be in CLR space. For comparing a gene set
    vs its complement, use build_balance_contrast() or test_gene_set().

    The posterior variance is computed efficiently using the low-rank
    structure: cᵀ(W·Wᵀ + D)c = ||Wᵀc||² + cᵀDc

    Examples
    --------
    >>> # Test if first 10 genes have different mean than rest
    >>> D = 100
    >>> contrast = jnp.zeros(D)
    >>> contrast = contrast.at[:10].set(1.0 / 10)
    >>> contrast = contrast.at[10:].set(-1.0 / 90)
    >>> result = test_contrast(model_a, model_b, contrast)
    >>> print(f"Effect: {result['delta_mean']:.3f} ± {result['delta_sd']:.3f}")
    >>> print(f"P(positive): {result['prob_positive']:.3f}")
    """
    # Validate and center contrast to ensure CLR validity (sum to zero)
    contrast_sum = jnp.sum(contrast)
    if jnp.abs(contrast_sum) > 1e-6:
        # Center the contrast to make it CLR-valid
        contrast = contrast - jnp.mean(contrast)

    # Guard against all-zero contrasts
    if jnp.allclose(contrast, 0.0, atol=1e-10):
        raise ValueError(
            "Contrast vector is all zeros after centering. "
            "Please provide a non-trivial contrast."
        )

    # Transform to CLR
    mu_A, W_A, d_A = _extract_lowrank_params(model_A)
    mu_B, W_B, d_B = _extract_lowrank_params(model_B)
    mu_A_clr, W_A_clr, d_A_clr = transform_gaussian_alr_to_clr(mu_A, W_A, d_A)
    mu_B_clr, W_B_clr, d_B_clr = transform_gaussian_alr_to_clr(mu_B, W_B, d_B)

    # Posterior mean: cᵀ(μ_A - μ_B)
    delta_mean = contrast @ (mu_A_clr - mu_B_clr)

    # Posterior variance: cᵀ(Σ_A + Σ_B)c
    # For low-rank Σ = W·Wᵀ + diag(d):
    # cᵀΣc = cᵀWWᵀc + cᵀDc = ||Wᵀc||² + Σᵢ c²ᵢdᵢ
    def quadratic_form(c, W, d):
        """Compute cᵀ(W·Wᵀ + diag(d))c efficiently."""
        return jnp.sum((W.T @ c) ** 2) + jnp.sum(c**2 * d)

    var_delta = quadratic_form(contrast, W_A_clr, d_A_clr) + quadratic_form(
        contrast, W_B_clr, d_B_clr
    )
    delta_sd = jnp.sqrt(var_delta)

    # Posterior probabilities
    z_score = delta_mean / delta_sd
    prob_positive = norm.cdf(z_score)
    prob_effect = (
        1 - norm.cdf(tau, loc=delta_mean, scale=delta_sd)
    ) + norm.cdf(-tau, loc=delta_mean, scale=delta_sd)
    lfsr = float(jnp.minimum(prob_positive, 1 - prob_positive))

    return {
        "delta_mean": float(delta_mean),
        "delta_sd": float(delta_sd),
        "z_score": float(z_score),
        "prob_positive": float(prob_positive),
        "prob_effect": float(prob_effect),
        "lfsr": lfsr,
    }


# ------------------------------------------------------------------------------
# Test enrichment of a gene set using a compositional balance
# ------------------------------------------------------------------------------


def test_gene_set(
    model_A,
    model_B,
    gene_set_indices: jnp.ndarray,
    tau: float = 0.0,
) -> dict:
    """
    Test enrichment of a gene set using a compositional balance.

    Constructs a balance contrast that compares the geometric mean of
    genes in the set vs the geometric mean of genes outside the set.

    This is the compositionally correct way to test pathway enrichment,
    accounting for the simplex constraint.

    Parameters
    ----------
    model_A : dict or Distribution
        Fitted logistic-normal model for condition A
    model_B : dict or Distribution
        Fitted logistic-normal model for condition B
    gene_set_indices : ndarray of int
        Indices of genes in the set (0-indexed)
    tau : float, default=0.0
        Practical significance threshold (log-scale)

    Returns
    -------
    dict
        Same as test_contrast() - posterior inference for the gene set balance

    Notes
    -----
    The balance contrast is constructed as:
    - +1/n_in for genes in the set
    - -1/n_out for genes outside the set

    This tests whether the geometric mean of expression in the pathway
    differs between conditions, properly accounting for compositionality.

    Examples
    --------
    >>> # Test a pathway with 20 genes
    >>> pathway_genes = jnp.array([5, 12, 18, 23, ...])  # 20 indices
    >>> result = test_gene_set(model_a, model_b, pathway_genes, tau=jnp.log(1.1))
    >>> print(f"Pathway effect: {result['delta_mean']:.3f} ± {result['delta_sd']:.3f}")
    >>> if result['prob_effect'] > 0.95:
    ...     print("Pathway is significantly enriched")
    """
    mu_A, W_A, d_A = _extract_lowrank_params(model_A)
    D = mu_A.shape[-1] + 1  # ALR dimension + 1 = CLR dimension

    # Build balance contrast: compare set vs complement
    n_in = len(gene_set_indices)
    n_out = D - n_in

    if n_in == 0 or n_out == 0:
        raise ValueError(
            f"Gene set must have at least 1 gene and at least 1 gene outside. "
            f"Got {n_in} in set, {n_out} outside."
        )

    # CLR balance: +1/n_in for genes in set, -1/n_out for genes outside
    contrast = jnp.zeros(D)
    contrast = contrast.at[gene_set_indices].set(1.0 / n_in)

    # Set complement indices
    mask_out = jnp.ones(D, dtype=bool)
    mask_out = mask_out.at[gene_set_indices].set(False)

    contrast = jnp.where(mask_out, -1.0 / n_out, contrast)

    return test_contrast(model_A, model_B, contrast, tau)


# ------------------------------------------------------------------------------
# Build a compositional balance contrast
# ------------------------------------------------------------------------------


def build_balance_contrast(
    numerator_indices: jnp.ndarray,
    denominator_indices: jnp.ndarray,
    D: int,
) -> jnp.ndarray:
    """Build a compositional balance contrast.

    A balance compares the geometric mean of one group of components
    (numerator) vs another group (denominator).

    Parameters
    ----------
    numerator_indices : ndarray of int
        Indices for the numerator group
    denominator_indices : ndarray of int
        Indices for the denominator group
    D : int
        Total number of components (genes)

    Returns
    -------
    ndarray, shape (D,)
        Balance contrast in CLR space

    Notes
    -----
    The balance is:
    - +1/n_num for numerator genes
    - -1/n_den for denominator genes
    - 0 for other genes

    Examples
    --------
    >>> # Compare genes 0-9 vs genes 10-19
    >>> contrast = build_balance_contrast(
    ...     jnp.arange(10), jnp.arange(10, 20), D=100
    ... )
    >>> assert contrast.shape == (100,)
    >>> assert jnp.allclose(contrast.sum(), 0.0)  # Balances sum to 0
    """
    n_num = len(numerator_indices)
    n_den = len(denominator_indices)

    if n_num == 0 or n_den == 0:
        raise ValueError("Both numerator and denominator must be non-empty")

    contrast = jnp.zeros(D)
    contrast = contrast.at[numerator_indices].set(1.0 / n_num)
    contrast = contrast.at[denominator_indices].set(-1.0 / n_den)

    return contrast
