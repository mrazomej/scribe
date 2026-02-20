"""Per-gene model comparison utilities.

This module provides functions for assessing model fit differences at the
gene level: given gene-level log-likelihoods (summed over all cells) from
two models, it computes per-gene elpd differences, their standard errors,
and associated z-scores.

The "gene-level log-likelihood" for gene g under sample s is:

    L_g^(s) = sum_c log p(u_{gc} | theta^s)

which is the total log-probability of all UMI counts for gene g across all
C cells.  When passed to WAIC or PSIS-LOO with ``return_by="gene"``, the
resulting elpd differences characterize which genes benefit most from
the additional flexibility of one model over another.

Intended use
------------
Call ``gene_level_comparison`` with two (S, G)-shaped log-likelihood matrices
(one per model) to obtain a summary DataFrame ready for inspection or
visualization.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from ._waic import compute_waic_stats


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def gene_level_comparison(
    log_liks_A: np.ndarray,
    log_liks_B: np.ndarray,
    gene_names: Optional[List[str]] = None,
    label_A: str = "A",
    label_B: str = "B",
    criterion: str = "waic_2",
) -> pd.DataFrame:
    """Compute per-gene model comparison statistics between two models.

    For each gene g, computes the pointwise elpd difference between model A
    and model B.  The difference is positive when model A provides better
    predictions than model B for gene g.

    The standard error of the total elpd difference follows from the CLT
    applied to the pointwise gene-level differences (see @eq-mc-se-delta-elpd
    in the paper):

        SE(delta_elpd) = sqrt(sum_g (d_g - d_bar)^2)

    where d_g = elpd_g(A) - elpd_g(B) is the per-gene difference.

    Parameters
    ----------
    log_liks_A : array-like, shape ``(S, G)``
        Gene-level log-likelihoods for model A.  Each entry is the total
        log p(all counts for gene g | theta^s) = sum_c log p(u_{gc}|theta^s).
        Rows are posterior samples, columns are genes.
    log_liks_B : array-like, shape ``(S, G)``
        Gene-level log-likelihoods for model B.  Must match ``log_liks_A``
        in shape.
    gene_names : list of str, optional
        Names for the G genes.  If ``None``, generic names ``gene_0, ...`` are
        generated.
    label_A : str, default='A'
        Human-readable label for model A.
    label_B : str, default='B'
        Human-readable label for model B.
    criterion : str, default='waic_2'
        Which WAIC variant to use for pointwise elpd values.  Must be one of
        ``'waic_1'``, ``'waic_2'``, ``'elppd_waic_1'``, ``'elppd_waic_2'``.
        ``'waic_2'`` (variance-based penalty) is recommended.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per gene and columns:

        ``gene``
            Gene name.
        ``elpd_A``
            Per-gene elpd for model A (negative half of WAIC).
        ``elpd_B``
            Per-gene elpd for model B.
        ``elpd_diff``
            Per-gene elpd difference (A - B); positive means A is better.
        ``elpd_diff_se``
            Standard error of the per-gene elpd difference (assumes each gene
            is an independent observation; see note below).
        ``z_score``
            z-score: ``elpd_diff / elpd_diff_se``.  Values |z| > 2 indicate
            a practically significant difference.
        ``p_waic_A``
            Effective parameter count per gene for model A.
        ``p_waic_B``
            Effective parameter count per gene for model B.
        ``favors``
            Which model is favored: ``label_A`` if ``elpd_diff > 0``, else
            ``label_B``.

    Notes
    -----
    The per-gene SE is computed from the cell-level pointwise differences
    within each gene.  However, since ``log_liks_A/B`` are already summed
    over cells (shape ``(S, G)``), the only variability captured here is
    across posterior samples.  The reported ``elpd_diff_se`` is therefore
    the posterior standard deviation of the per-gene elpd difference, not
    a frequentist SE.  It correctly reflects model uncertainty but not
    sampling variability across cells.
    """
    import jax.numpy as jnp

    log_liks_A = np.asarray(log_liks_A, dtype=np.float64)
    log_liks_B = np.asarray(log_liks_B, dtype=np.float64)

    S, G = log_liks_A.shape
    if log_liks_B.shape != (S, G):
        raise ValueError(
            f"Shape mismatch: log_liks_A has shape {log_liks_A.shape} but "
            f"log_liks_B has shape {log_liks_B.shape}."
        )

    # Generate gene names if not provided
    if gene_names is None:
        gene_names = [f"gene_{g}" for g in range(G)]
    elif len(gene_names) != G:
        raise ValueError(
            f"gene_names has length {len(gene_names)} but log_liks has G={G} genes."
        )

    # Compute per-gene WAIC stats (aggregate=False returns per-gene arrays)
    stats_A = compute_waic_stats(jnp.array(log_liks_A), aggregate=False)
    stats_B = compute_waic_stats(jnp.array(log_liks_B), aggregate=False)

    # Per-gene elpd (using the requested criterion)
    # The elppd_waic_X keys give per-gene elpd directly (with aggregate=False)
    elppd_key = criterion.replace("waic_", "elppd_waic_")
    if elppd_key not in stats_A:
        # If user asked for 'waic_2', map to 'elppd_waic_2'
        # If user asked for 'elppd_waic_2', use directly
        if "elppd" not in elppd_key:
            elppd_key = f"elppd_{criterion}"
    # Fallback to elppd_waic_2
    if elppd_key not in stats_A:
        elppd_key = "elppd_waic_2"

    elpd_A = np.asarray(stats_A[elppd_key])
    elpd_B = np.asarray(stats_B[elppd_key])
    p_waic_A = np.asarray(stats_A["p_waic_2"])
    p_waic_B = np.asarray(stats_B["p_waic_2"])

    # Pointwise difference per gene
    elpd_diff = elpd_A - elpd_B

    # Per-gene SE: posterior std-dev of (elpd_A_s - elpd_B_s) over samples
    # We compute sample-by-sample lppd contributions and take their difference
    # as a proxy for per-gene comparison uncertainty.
    # lppd contribution for sample s and gene g is just log p(gene g | theta^s)
    # (already in log_liks_A[:, g]).  The pointwise elpd difference is:
    #   d_g^(s) = log_liks_A[s, g] - log_liks_B[s, g]
    # SE = std over samples of d_g^(s)
    d_samples = log_liks_A - log_liks_B  # shape (S, G)
    elpd_diff_se = np.std(d_samples, axis=0, ddof=1)  # shape (G,)

    # z-score: how many SEs away from zero?
    # Guard against zero SE (all samples identical for some genes)
    z_score = np.where(elpd_diff_se > 0, elpd_diff / elpd_diff_se, 0.0)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names,
            f"elpd_{label_A}": elpd_A,
            f"elpd_{label_B}": elpd_B,
            "elpd_diff": elpd_diff,
            "elpd_diff_se": elpd_diff_se,
            "z_score": z_score,
            f"p_waic_{label_A}": p_waic_A,
            f"p_waic_{label_B}": p_waic_B,
            "favors": np.where(elpd_diff > 0, label_A, label_B),
        }
    )

    # Sort by absolute z-score descending (most decisive genes first)
    df = df.sort_values("z_score", key=np.abs, ascending=False).reset_index(drop=True)
    return df


def format_gene_comparison_table(
    df: pd.DataFrame,
    top_n: Optional[int] = 20,
    sort_by: str = "z_score",
) -> str:
    """Format a gene-level comparison DataFrame as a human-readable table.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`gene_level_comparison`.
    top_n : int, optional
        Number of top genes to display.  Displays all genes if ``None``.
    sort_by : str, default='z_score'
        Column to sort by (descending by absolute value for ``z_score``,
        descending for all other columns).

    Returns
    -------
    str
        Formatted table string.
    """
    display_df = df.copy()

    # Sort
    if sort_by == "z_score":
        display_df = display_df.sort_values(
            "z_score", key=np.abs, ascending=False
        )
    else:
        display_df = display_df.sort_values(sort_by, ascending=False)

    # Truncate
    if top_n is not None:
        display_df = display_df.head(top_n)

    # Select core columns for display
    core_cols = [
        "gene", "elpd_diff", "elpd_diff_se", "z_score", "favors"
    ]
    # Only keep columns that exist
    cols = [c for c in core_cols if c in display_df.columns]
    display_df = display_df[cols]

    # Format floats
    float_cols = ["elpd_diff", "elpd_diff_se", "z_score"]
    for col in float_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

    header = f"Gene-level model comparison (top {top_n} genes by |z-score|)\n"
    table = display_df.to_string(index=False)
    return header + table
