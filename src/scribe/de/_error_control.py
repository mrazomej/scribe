"""Bayesian error control and result formatting for DE analysis.

This module provides Bayesian error-control methods (lfdr, PEFP, threshold
finding) and result-formatting utilities for differential expression
analysis.

Key concepts
------------
- **lfsr** (local false sign rate): Exact posterior probability of having the
  wrong sign.  Computed directly in ``differential_expression()``.
- **lfdr** (local false discovery rate): Empirical Bayes approximation of the
  posterior probability of being null.
- **PEFP** (Posterior Expected False Discovery Proportion): Bayesian analogue
  of FDR.
"""

from typing import Optional

import jax.numpy as jnp
from jax.scipy.stats import norm


# --------------------------------------------------------------------------
# Compute local false discovery rate (lfdr)
# --------------------------------------------------------------------------


def compute_lfdr(
    delta_mean: jnp.ndarray,
    delta_sd: jnp.ndarray,
    prior_null_prob: float = 0.5,
) -> jnp.ndarray:
    """Compute local false discovery rate (lfdr) for each gene.

    The lfdr is the posterior probability that a gene is null (Delta = 0)
    given the observed data.  This uses an empirical Bayes approach.

    **WARNING**: This is a rough empirical Bayes approximation.  For
    primary differential-expression decisions, prefer **lfsr** (local false
    sign rate) and **PEFP** (posterior expected false discovery proportion),
    which are exact under the Gaussian assumption.

    Parameters
    ----------
    delta_mean : ndarray
        Posterior mean effects for each gene.
    delta_sd : ndarray
        Posterior standard deviations for each gene.
    prior_null_prob : float, default=0.5
        Prior probability that a gene is null.

    Returns
    -------
    ndarray
        lfdr: local false discovery rate for each gene (between 0 and 1).

    Notes
    -----
    ``lfdr_g = pi_0 f_0(z_g) / f(z_g)`` where:

    - ``pi_0`` = prior probability of null.
    - ``f_0(z) = N(0, s_0^2)`` is null density.
    - ``f(z)`` = mixture of null and alternative.

    Examples
    --------
    >>> delta_mean = jnp.array([0.1, 2.0, 0.05, 1.5])
    >>> delta_sd = jnp.ones(4)
    >>> lfdr = compute_lfdr(delta_mean, delta_sd)
    """
    # Observed z-scores
    z_scores = delta_mean / delta_sd

    # Estimate null sd (standard normal under null)
    s0 = 1.0

    # Null density
    f0 = norm.pdf(z_scores, loc=0, scale=s0)

    # Marginal density (approximate via mixture)
    # Simple approach: mixture of null and observed
    f_alt = norm.pdf(z_scores, loc=delta_mean, scale=delta_sd)
    f = prior_null_prob * f0 + (1 - prior_null_prob) * f_alt

    # Compute lfdr and clip to [0, 1]
    lfdr = prior_null_prob * f0 / jnp.maximum(f, 1e-10)
    lfdr = jnp.clip(lfdr, 0, 1)

    return lfdr


# --------------------------------------------------------------------------
# Compute posterior expected false discovery proportion (PEFP)
# --------------------------------------------------------------------------


def compute_pefp(
    lfsr: jnp.ndarray,
    threshold: float = 0.05,
) -> float:
    """Compute posterior expected false discovery proportion (PEFP).

    Given a set of genes called DE (based on ``lfsr < threshold``), the PEFP
    is the expected proportion of false discoveries::

        PEFP = E[# false discoveries / # discoveries | data]
             = sum(lfsr for called genes) / (# called genes)

    This is a Bayesian analogue of FDR.

    Parameters
    ----------
    lfsr : ndarray
        Local false sign rates for each gene.
    threshold : float, default=0.05
        Threshold for calling genes DE.

    Returns
    -------
    float
        Expected false discovery proportion.

    Notes
    -----
    Unlike frequentist FDR, PEFP is a posterior quantity that directly uses
    the data to compute expected error rates.

    Examples
    --------
    >>> lfsr = jnp.array([0.01, 0.03, 0.5, 0.02, 0.8])
    >>> pefp = compute_pefp(lfsr, threshold=0.05)
    """
    # Identify called genes
    called = lfsr < threshold
    n_called = jnp.sum(called)

    # No discoveries => PEFP is 0
    if n_called == 0:
        return 0.0

    # Sum of lfsr for called genes = expected number of false discoveries
    expected_false = jnp.sum(jnp.where(called, lfsr, 0.0))

    return float(expected_false / n_called)


# --------------------------------------------------------------------------
# Find lfsr threshold that controls PEFP at target level
# --------------------------------------------------------------------------


def find_lfsr_threshold(
    lfsr: jnp.ndarray,
    target_pefp: float = 0.05,
) -> float:
    """Find lfsr threshold that controls PEFP at target level.

    This is the Bayesian version of FDR control: find the threshold such
    that the posterior expected FDP is at most the target.

    Parameters
    ----------
    lfsr : ndarray
        Local false sign rates for each gene.
    target_pefp : float, default=0.05
        Target posterior expected false discovery proportion.

    Returns
    -------
    float
        Threshold value for lfsr that controls PEFP at ``target_pefp``.

    Notes
    -----
    This finds the largest threshold such that PEFP <= target.
    If no valid threshold exists, returns ``0.0``.

    Examples
    --------
    >>> lfsr = jnp.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.5])
    >>> threshold = find_lfsr_threshold(lfsr, target_pefp=0.05)
    """
    # Sort lfsr values
    sorted_lfsr = jnp.sort(lfsr)

    # Compute PEFP for each possible threshold
    pefps = jnp.array(
        [compute_pefp(lfsr, threshold=float(t)) for t in sorted_lfsr]
    )

    # Find largest threshold with PEFP <= target
    valid = pefps <= target_pefp
    if jnp.any(valid):
        # Get the maximum threshold among valid ones
        valid_thresholds = sorted_lfsr[valid]
        return float(jnp.max(valid_thresholds))
    else:
        return 0.0  # No valid threshold


# --------------------------------------------------------------------------
# Format DE results as a readable table
# --------------------------------------------------------------------------


def format_de_table(
    de_results: dict,
    sort_by: str = "lfsr",
    top_n: Optional[int] = None,
) -> str:
    """Format DE results as a readable table.

    Parameters
    ----------
    de_results : dict
        Output from ``differential_expression()``.
    sort_by : str, default='lfsr'
        Column to sort by.  Options: ``'lfsr'``, ``'delta_mean'``,
        ``'prob_effect'``.
    top_n : int, optional
        Number of top genes to display.  If ``None``, shows all genes.

    Returns
    -------
    str
        Formatted table string.

    Examples
    --------
    >>> results = differential_expression(model_a, model_b)
    >>> table = format_de_table(results, sort_by='lfsr', top_n=20)
    >>> print(table)
    """
    try:
        import pandas as pd
    except ImportError:
        # Fallback if pandas not available
        return _format_de_table_simple(de_results, sort_by, top_n)

    # Create dataframe from results
    df = pd.DataFrame(
        {
            "gene": de_results.get(
                "gene_names",
                [
                    f"gene_{i}"
                    for i in range(len(de_results["delta_mean"]))
                ],
            ),
            "log_fc": de_results["delta_mean"],
            "sd": de_results["delta_sd"],
            "prob_positive": de_results["prob_positive"],
            "prob_effect": de_results["prob_effect"],
            "lfsr": de_results["lfsr"],
        }
    )

    # Sort by requested column
    ascending = sort_by == "lfsr"  # Lower lfsr is better
    df = df.sort_values(sort_by, ascending=ascending)

    # Limit number of rows if requested
    if top_n is not None:
        df = df.head(top_n)

    return df.to_string(index=False)


# --------------------------------------------------------------------------
# Simple fallback table formatting without pandas
# --------------------------------------------------------------------------


def _format_de_table_simple(
    de_results: dict,
    sort_by: str = "lfsr",
    top_n: Optional[int] = None,
) -> str:
    """Simple fallback table formatting without pandas.

    Parameters
    ----------
    de_results : dict
        Output from ``differential_expression()``.
    sort_by : str
        Column to sort by.
    top_n : int, optional
        Number of top genes to display.

    Returns
    -------
    str
        Formatted table string.
    """
    gene_names = de_results.get(
        "gene_names",
        [f"gene_{i}" for i in range(len(de_results["delta_mean"]))],
    )

    # Create list of tuples for sorting
    data = list(
        zip(
            gene_names,
            de_results["delta_mean"],
            de_results["delta_sd"],
            de_results["prob_positive"],
            de_results["prob_effect"],
            de_results["lfsr"],
        )
    )

    # Sort by requested column
    sort_idx = {"delta_mean": 1, "prob_effect": 4, "lfsr": 5}[sort_by]
    ascending = sort_by == "lfsr"
    data.sort(key=lambda x: x[sort_idx], reverse=not ascending)

    # Limit rows
    if top_n is not None:
        data = data[:top_n]

    # Format as tab-separated table
    lines = ["gene\tlog_fc\tsd\tprob_positive\tprob_effect\tlfsr"]
    for row in data:
        lines.append(
            f"{row[0]}\t{row[1]:.3f}\t{row[2]:.3f}\t"
            f"{row[3]:.3f}\t{row[4]:.3f}\t{row[5]:.3f}"
        )

    return "\n".join(lines)
