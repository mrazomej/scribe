"""Gene-set and pathway-level differential expression analysis.

This module provides tools for testing linear contrasts and gene sets,
enabling pathway-level inference using compositional balances.

- **Parametric tests** (``test_contrast``, ``test_gene_set``) return exact
  posterior probabilities under the Gaussian (logistic-normal) assumption.
- **Empirical tests** (``empirical_test_gene_set``,
  ``empirical_test_pathway_perturbation``, ``empirical_test_multiple_gene_sets``)
  compute statistics by counting over Monte Carlo CLR difference samples,
  requiring no distributional assumptions.
"""

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from ._transforms import (
    build_ilr_balance,
    build_pathway_sbp_basis,
    transform_gaussian_alr_to_clr,
)
from ._extract import extract_alr_params


# --------------------------------------------------------------------------
# Test linear contrasts (Bayesian inference)
# --------------------------------------------------------------------------


def test_contrast(
    model_A,
    model_B,
    contrast: jnp.ndarray,
    tau: float = 0.0,
) -> dict:
    """Test a linear contrast ``c^T Delta`` (Bayesian inference).

    For a contrast vector ``c``, we compute the posterior distribution of
    ``c^T (z_A - z_B)`` which is::

        N(c^T (mu_A - mu_B),  c^T (Sigma_A + Sigma_B) c)

    This is useful for:

    - Testing custom gene combinations.
    - Pathway analysis via balances.
    - Module-level effects.

    Parameters
    ----------
    model_A : dict or Distribution
        Fitted logistic-normal model for condition A.
    model_B : dict or Distribution
        Fitted logistic-normal model for condition B.
    contrast : ndarray, shape ``(D,)``
        Contrast vector in CLR space.  For balances, this should compare
        one group of genes vs another (e.g., pathway vs rest).
    tau : float, default=0.0
        Practical significance threshold (log-scale).

    Returns
    -------
    dict
        Dictionary with keys:

        - **delta_mean** : float -- Posterior mean of ``c^T Delta``.
        - **delta_sd** : float -- Posterior standard deviation.
        - **z_score** : float -- Standardized effect size.
        - **prob_positive** : float -- ``P(c^T Delta > 0 | data)``.
        - **prob_effect** : float -- ``P(|c^T Delta| > tau | data)``.
        - **lfsr** : float -- Local false sign rate for the contrast.

    Notes
    -----
    The posterior variance is computed efficiently using the low-rank
    structure: ``c^T (W W^T + D) c = ||W^T c||^2 + c^T D c``.

    Examples
    --------
    >>> D = 100
    >>> contrast = jnp.zeros(D)
    >>> contrast = contrast.at[:10].set(1.0 / 10)
    >>> contrast = contrast.at[10:].set(-1.0 / 90)
    >>> result = test_contrast(model_a, model_b, contrast)
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

    # Extract (D-1)-dim ALR params and transform to CLR
    mu_A, W_A, d_A = extract_alr_params(model_A)
    mu_B, W_B, d_B = extract_alr_params(model_B)
    mu_A_clr, W_A_clr, d_A_clr = transform_gaussian_alr_to_clr(
        mu_A, W_A, d_A
    )
    mu_B_clr, W_B_clr, d_B_clr = transform_gaussian_alr_to_clr(
        mu_B, W_B, d_B
    )

    # Posterior mean: c^T (mu_A - mu_B)
    delta_mean = contrast @ (mu_A_clr - mu_B_clr)

    # Posterior variance: c^T (Sigma_A + Sigma_B) c
    # For low-rank Sigma = W W^T + diag(d):
    #   c^T Sigma c = c^T W W^T c + c^T D c = ||W^T c||^2 + sum_i c_i^2 d_i
    def quadratic_form(c, W, d):
        """Compute c^T (W W^T + diag(d)) c efficiently."""
        return jnp.sum((W.T @ c) ** 2) + jnp.sum(c**2 * d)

    # Sum quadratic forms from both models
    var_delta = quadratic_form(
        contrast, W_A_clr, d_A_clr
    ) + quadratic_form(contrast, W_B_clr, d_B_clr)
    delta_sd = jnp.sqrt(var_delta)

    # Guard against near-zero SD to avoid inf/NaN in z_score
    delta_sd = jnp.maximum(delta_sd, 1e-30)

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


# --------------------------------------------------------------------------
# Test enrichment of a gene set using a compositional balance
# --------------------------------------------------------------------------


def test_gene_set(
    model_A,
    model_B,
    gene_set_indices: jnp.ndarray,
    tau: float = 0.0,
) -> dict:
    """Test enrichment of a gene set using a compositional balance.

    Constructs a balance contrast that compares the geometric mean of genes
    in the set vs the geometric mean of genes outside the set.

    This is the compositionally correct way to test pathway enrichment,
    accounting for the simplex constraint.

    Parameters
    ----------
    model_A : dict or Distribution
        Fitted logistic-normal model for condition A.
    model_B : dict or Distribution
        Fitted logistic-normal model for condition B.
    gene_set_indices : ndarray of int
        Indices of genes in the set (0-indexed, in CLR space).
    tau : float, default=0.0
        Practical significance threshold (log-scale).

    Returns
    -------
    dict
        Same as ``test_contrast()`` -- posterior inference for the gene set
        balance.

    Notes
    -----
    The balance contrast is constructed as:

    - ``+1/n_in`` for genes in the set.
    - ``-1/n_out`` for genes outside the set.

    Examples
    --------
    >>> pathway_genes = jnp.array([5, 12, 18, 23])
    >>> result = test_gene_set(model_a, model_b, pathway_genes,
    ...                         tau=jnp.log(1.1))
    """
    # Extract ALR params to determine CLR dimension
    mu_A, W_A, d_A = extract_alr_params(model_A)
    # D_alr = mu_A.shape[-1], so CLR dimension = D_alr + 1
    D = mu_A.shape[-1] + 1

    # Build balance contrast: compare set vs complement
    n_in = len(gene_set_indices)
    n_out = D - n_in

    # Validate sizes
    if n_in == 0 or n_out == 0:
        raise ValueError(
            f"Gene set must have at least 1 gene and at least 1 gene "
            f"outside.  Got {n_in} in set, {n_out} outside."
        )

    # CLR balance: +1/n_in for genes in set, -1/n_out for genes outside
    contrast = jnp.zeros(D)
    contrast = contrast.at[gene_set_indices].set(1.0 / n_in)

    # Set complement indices
    mask_out = jnp.ones(D, dtype=bool)
    mask_out = mask_out.at[gene_set_indices].set(False)
    contrast = jnp.where(mask_out, -1.0 / n_out, contrast)

    return test_contrast(model_A, model_B, contrast, tau)


# --------------------------------------------------------------------------
# Build a compositional balance contrast
# --------------------------------------------------------------------------


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
        Indices for the numerator group.
    denominator_indices : ndarray of int
        Indices for the denominator group.
    D : int
        Total number of components (genes, in CLR space).

    Returns
    -------
    ndarray, shape ``(D,)``
        Balance contrast in CLR space.

    Notes
    -----
    The balance is:

    - ``+1/n_num`` for numerator genes.
    - ``-1/n_den`` for denominator genes.
    - ``0`` for other genes.

    Examples
    --------
    >>> contrast = build_balance_contrast(
    ...     jnp.arange(10), jnp.arange(10, 20), D=100
    ... )
    >>> assert contrast.shape == (100,)
    >>> assert jnp.allclose(contrast.sum(), 0.0)
    """
    n_num = len(numerator_indices)
    n_den = len(denominator_indices)

    # Both groups must be non-empty
    if n_num == 0 or n_den == 0:
        raise ValueError("Both numerator and denominator must be non-empty")

    # Validate that numerator and denominator are disjoint -- overlapping
    # indices would silently produce incorrect coefficients since the
    # second ``at[].set`` would overwrite the first.
    num_set = set(int(i) for i in numerator_indices)
    den_set = set(int(i) for i in denominator_indices)
    overlap = num_set & den_set
    if overlap:
        raise ValueError(
            f"Numerator and denominator must be disjoint. "
            f"Found {len(overlap)} overlapping indices."
        )

    contrast = jnp.zeros(D)
    contrast = contrast.at[jnp.asarray(numerator_indices)].set(1.0 / n_num)
    contrast = contrast.at[jnp.asarray(denominator_indices)].set(-1.0 / n_den)

    return contrast


# --------------------------------------------------------------------------
# Empirical (sample-based) pathway enrichment via ILR balances
# --------------------------------------------------------------------------


def empirical_test_gene_set(
    delta_samples: jnp.ndarray,
    gene_set_indices: jnp.ndarray,
    tau: float = 0.0,
) -> dict:
    """Test pathway enrichment using the ILR balance and posterior samples.

    Builds the ILR-normalized balance vector for the pathway-vs-complement
    partition, projects each CLR difference sample onto it, and computes
    empirical statistics by counting.  No distributional assumption is needed.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        CLR-space posterior difference samples.  Each row is a paired
        ``CLR(rho_A) - CLR(rho_B)`` draw.
    gene_set_indices : jnp.ndarray
        Integer indices of genes in the pathway, shape ``(n_+,)``.
    tau : float, default=0.0
        Practical significance threshold (log-scale).

    Returns
    -------
    dict
        Dictionary with keys:

        - **balance_mean** : float -- Posterior mean of the ILR balance.
        - **balance_sd** : float -- Posterior standard deviation.
        - **prob_positive** : float -- ``P(balance > 0 | data)``.
        - **prob_effect** : float -- ``P(|balance| > tau | data)``.
        - **lfsr** : float -- Local false sign rate.
        - **lfsr_tau** : float -- lfsr with practical significance threshold.

    Notes
    -----
    Because the ILR balance is a positive scalar multiple of the unnormalized
    CLR contrast, the lfsr is identical regardless of which normalization is
    used.  The ILR normalization is preferred for its geometric
    interpretation (unit-norm vector on the simplex).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> delta = jnp.ones((500, 10)) * 0.01
    >>> result = empirical_test_gene_set(delta, jnp.array([0, 1, 2]))
    """
    gene_set_indices = jnp.asarray(gene_set_indices, dtype=jnp.int32)
    D = delta_samples.shape[1]

    # Build unit-norm ILR balance vector and project samples
    v = build_ilr_balance(gene_set_indices, D)
    balance_samples = delta_samples @ v  # (N,)

    # Posterior mean and standard deviation
    balance_mean = float(jnp.mean(balance_samples))
    balance_sd = float(jnp.std(balance_samples, ddof=1))

    # Posterior probabilities by counting
    prob_positive = float(jnp.mean(balance_samples > 0))
    prob_up = float(jnp.mean(balance_samples > tau))
    prob_down = float(jnp.mean(balance_samples < -tau))
    prob_effect = prob_up + prob_down

    # Local false sign rate
    lfsr = min(prob_positive, 1.0 - prob_positive)
    lfsr_tau = 1.0 - max(prob_up, prob_down)

    return {
        "balance_mean": balance_mean,
        "balance_sd": balance_sd,
        "prob_positive": prob_positive,
        "prob_effect": prob_effect,
        "lfsr": lfsr,
        "lfsr_tau": lfsr_tau,
    }


def empirical_test_pathway_perturbation(
    delta_samples: jnp.ndarray,
    gene_set_indices: jnp.ndarray,
    n_permutations: int = 999,
    key: Optional[jax.random.PRNGKey] = None,
) -> dict:
    """Test within-pathway compositional perturbation via ILR subspace.

    Constructs a pathway-aware sequential binary partition (SBP) basis,
    extracts the within-pathway ILR subspace, and computes a quadratic
    perturbation statistic that measures coordinated rearrangement among
    pathway genes.  Statistical significance is assessed by comparing
    against a null distribution obtained by permuting gene labels.

    This test is complementary to the single-balance test
    (``empirical_test_gene_set``): the balance test detects a net directional
    shift, while the perturbation test detects internal rearrangement even
    when the average balance is near zero.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        CLR-space posterior difference samples.
    gene_set_indices : jnp.ndarray
        Integer indices of genes in the pathway, shape ``(n_+,)``.
        Must contain at least 2 genes.
    n_permutations : int, default=999
        Number of permutations for null calibration.
    key : jax.random.PRNGKey, optional
        PRNG key for reproducibility.  If ``None``, uses ``jax.random.PRNGKey(0)``.

    Returns
    -------
    dict
        Dictionary with keys:

        - **t_obs** : float -- Observed mean within-pathway perturbation
          statistic ``mean_s ||V_within @ Delta^(s)||^2``.
        - **t_sd** : float -- Posterior standard deviation of T across samples.
        - **p_value** : float -- Empirical p-value from the permutation null.
        - **n_permutations** : int -- Number of permutations used.

    Raises
    ------
    ValueError
        If the gene set has fewer than 2 genes.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> delta = jnp.ones((500, 10)) * 0.01
    >>> result = empirical_test_pathway_perturbation(
    ...     delta, jnp.array([0, 1, 2, 3])
    ... )
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    gene_set_indices = jnp.asarray(gene_set_indices, dtype=jnp.int32)
    N, D = delta_samples.shape
    n_plus = gene_set_indices.shape[0]

    if n_plus < 2:
        raise ValueError(
            f"Pathway must contain at least 2 genes for a perturbation "
            f"test, got n_+={n_plus}."
        )

    # Build pathway-aware SBP basis and extract within-pathway rows
    V_sbp = build_pathway_sbp_basis(gene_set_indices, D)
    V_within = V_sbp[1:n_plus]  # (n_+-1, D)

    # Compute within-pathway ILR differences: (N, n_+-1) = (N, D) @ (D, n_+-1)
    b_within = delta_samples @ V_within.T
    # Quadratic statistic per sample: sum of squared within-pathway ILR coords
    T_per_sample = jnp.sum(b_within**2, axis=1)  # (N,)
    t_obs = float(jnp.mean(T_per_sample))
    t_sd = float(jnp.std(T_per_sample, ddof=1))

    # Null calibration via gene-label permutation
    all_indices = jnp.arange(D)
    t_null = jnp.zeros(n_permutations)
    keys = jax.random.split(key, n_permutations)

    for r in range(n_permutations):
        # Random subset of size n_plus
        perm = jax.random.permutation(keys[r], all_indices)
        perm_indices = perm[:n_plus]

        # Build the permuted within-pathway basis
        V_perm = build_pathway_sbp_basis(perm_indices, D)
        V_within_perm = V_perm[1:n_plus]

        # Permuted perturbation statistic
        b_perm = delta_samples @ V_within_perm.T
        T_perm = jnp.sum(b_perm**2, axis=1)
        t_null = t_null.at[r].set(jnp.mean(T_perm))

    # Empirical p-value (fraction of null values >= observed)
    p_value = float(jnp.mean(t_null >= t_obs))

    return {
        "t_obs": t_obs,
        "t_sd": t_sd,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def empirical_test_multiple_gene_sets(
    delta_samples: jnp.ndarray,
    gene_sets: List[jnp.ndarray],
    tau: float = 0.0,
    target_pefp: float = 0.05,
) -> Dict[str, list]:
    """Batch empirical enrichment test for multiple pathways with PEFP control.

    Applies ``empirical_test_gene_set`` to each pathway, collects the
    per-pathway lfsr values, and applies the PEFP threshold-finding algorithm
    to identify significantly enriched pathways at the desired error level.

    Parameters
    ----------
    delta_samples : jnp.ndarray, shape ``(N, D)``
        CLR-space posterior difference samples.
    gene_sets : list of jnp.ndarray
        Each element is an integer array of gene indices for one pathway.
    tau : float, default=0.0
        Practical significance threshold (log-scale).
    target_pefp : float, default=0.05
        Target PEFP level for multiple testing correction.

    Returns
    -------
    dict
        Dictionary with keys (each is a list of length ``M``):

        - **balance_mean** : per-pathway posterior mean balance.
        - **balance_sd** : per-pathway posterior standard deviation.
        - **prob_positive** : per-pathway P(balance > 0).
        - **prob_effect** : per-pathway P(|balance| > tau).
        - **lfsr** : per-pathway local false sign rate.
        - **lfsr_tau** : per-pathway lfsr with practical significance.
        - **significant** : bool list, True if pathway passes PEFP threshold.
        - **lfsr_threshold** : float, the lfsr cutoff for PEFP control.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> delta = jnp.ones((500, 10)) * 0.01
    >>> gene_sets = [jnp.array([0, 1]), jnp.array([3, 4, 5])]
    >>> result = empirical_test_multiple_gene_sets(delta, gene_sets)
    """
    from ._error_control import find_lfsr_threshold

    # Run single-balance test for each pathway
    results_list = [
        empirical_test_gene_set(delta_samples, gs, tau=tau)
        for gs in gene_sets
    ]

    # Collect per-pathway statistics
    balance_means = [r["balance_mean"] for r in results_list]
    balance_sds = [r["balance_sd"] for r in results_list]
    prob_positives = [r["prob_positive"] for r in results_list]
    prob_effects = [r["prob_effect"] for r in results_list]
    lfsrs = [r["lfsr"] for r in results_list]
    lfsrs_tau = [r["lfsr_tau"] for r in results_list]

    # PEFP control over the pathway-level lfsr vector
    lfsr_array = jnp.array(lfsrs)
    threshold = find_lfsr_threshold(lfsr_array, target_pefp=target_pefp)
    significant = [float(l) < threshold for l in lfsrs]

    return {
        "balance_mean": balance_means,
        "balance_sd": balance_sds,
        "prob_positive": prob_positives,
        "prob_effect": prob_effects,
        "lfsr": lfsrs,
        "lfsr_tau": lfsrs_tau,
        "significant": significant,
        "lfsr_threshold": float(threshold),
    }
