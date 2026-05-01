"""Gene-coverage pre-filtering utilities for model fitting.

This module provides helper functions for selecting a subset of genes based on
their empirical contribution to total transcriptome abundance. The selected
genes are retained explicitly, while all excluded genes are pooled into a
single trailing "other" pseudo-gene to preserve compositional closure.

The selection rule mirrors the DE coverage helper logic:

1. Sum counts across cells to estimate empirical gene abundance.
2. Convert abundances to proportions that sum to one.
3. Sort genes by descending proportion.
4. Keep the smallest prefix whose cumulative mass reaches the requested
   coverage threshold.

For multi-dataset fits, masks can be computed per dataset and then unioned so
that a gene abundant in any dataset is retained globally.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np


def _validate_coverage(coverage: float) -> float:
    """Validate and normalize a cumulative coverage threshold.

    Parameters
    ----------
    coverage : float
        Target cumulative abundance coverage in the open-closed interval
        ``(0, 1]``.

    Returns
    -------
    float
        Validated coverage value as a Python float.

    Raises
    ------
    ValueError
        If ``coverage`` is outside ``(0, 1]``.
    """
    cov = float(coverage)
    if not (0.0 < cov <= 1.0):
        raise ValueError(
            f"coverage must be in the interval (0, 1], got {coverage!r}."
        )
    return cov


def _coverage_mask_from_counts(
    counts: jnp.ndarray, coverage: float
) -> np.ndarray:
    """Build a per-group mask from cumulative empirical abundance coverage.

    Parameters
    ----------
    counts : jax.Array or numpy.ndarray, shape ``(n_cells, n_genes)``
        Count matrix for one group of cells (single dataset or single subset).
    coverage : float
        Target cumulative abundance coverage in ``(0, 1]``.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n_genes,)``
        Boolean keep mask where ``True`` denotes retained genes.

    Raises
    ------
    ValueError
        If ``counts`` is not a 2D matrix, has zero genes, or has zero total
        abundance.
    """
    cov = _validate_coverage(coverage)

    counts_arr = np.asarray(counts)
    if counts_arr.ndim != 2:
        raise ValueError(
            "counts must be a 2D array with shape (n_cells, n_genes)."
        )
    if counts_arr.shape[1] == 0:
        raise ValueError("counts must contain at least one gene.")

    # Sum over cells to estimate empirical gene abundance.
    gene_totals = np.asarray(counts_arr.sum(axis=0), dtype=float).ravel()
    total = float(gene_totals.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "counts must contain positive total abundance to compute coverage."
        )

    # Convert to compositional proportions and retain the smallest heavy-tail
    # prefix that reaches the requested cumulative coverage.
    proportions = gene_totals / total
    order = np.argsort(-proportions)
    cumulative = np.cumsum(proportions[order])
    n_keep = int(np.searchsorted(cumulative, cov, side="left")) + 1
    n_keep = min(max(n_keep, 1), proportions.shape[0])

    mask = np.zeros(proportions.shape[0], dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def compute_empirical_gene_coverage_mask(
    counts: jnp.ndarray,
    coverage: float = 0.95,
    dataset_indices: Optional[jnp.ndarray] = None,
) -> np.ndarray:
    """Compute a gene keep-mask from empirical abundance coverage.

    Parameters
    ----------
    counts : jax.Array or numpy.ndarray, shape ``(n_cells, n_genes)``
        Full count matrix used for model fitting.
    coverage : float, default=0.95
        Target cumulative abundance coverage in ``(0, 1]``.
    dataset_indices : jax.Array or numpy.ndarray, optional, shape ``(n_cells,)``
        Optional per-cell dataset IDs. When provided, a per-dataset mask is
        computed independently and the final mask is the union across datasets.
        This ensures that genes abundant in any dataset are retained.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n_genes,)``
        Boolean keep mask where ``True`` denotes retained genes.

    Raises
    ------
    ValueError
        If ``dataset_indices`` is not 1D or does not align with the number of
        cells in ``counts``.
    """
    if dataset_indices is None:
        return _coverage_mask_from_counts(counts, coverage=coverage)

    counts_arr = np.asarray(counts)
    ds = np.asarray(dataset_indices).ravel()
    if ds.ndim != 1:
        raise ValueError("dataset_indices must be a 1D array.")
    if counts_arr.shape[0] != ds.shape[0]:
        raise ValueError(
            "dataset_indices length must match the number of cells in counts."
        )

    combined = np.zeros(counts_arr.shape[1], dtype=bool)
    for dataset_id in np.unique(ds):
        dataset_mask = ds == dataset_id
        if int(np.sum(dataset_mask)) == 0:
            continue
        combined |= _coverage_mask_from_counts(
            counts_arr[dataset_mask, :], coverage=coverage
        )
    return combined


def aggregate_counts_by_mask(
    counts: jnp.ndarray, mask: Sequence[bool]
) -> jnp.ndarray:
    """Aggregate excluded genes into a trailing "other" count column.

    Parameters
    ----------
    counts : jax.Array or numpy.ndarray, shape ``(n_cells, n_genes)``
        Input count matrix.
    mask : array-like of bool, shape ``(n_genes,)``
        Keep mask where ``True`` genes remain explicit and ``False`` genes are
        pooled into the "other" column.

    Returns
    -------
    jax.Array, shape ``(n_cells, n_kept + 1)``
        Aggregated count matrix with a trailing "other" column.

    Raises
    ------
    ValueError
        If mask length does not match the gene dimension, or if all mask values
        are ``True``/``False`` in unsupported configurations.
    """
    counts_arr = jnp.asarray(counts)
    keep = np.asarray(mask, dtype=bool).ravel()

    if counts_arr.ndim != 2:
        raise ValueError(
            "counts must be a 2D array with shape (n_cells, n_genes)."
        )
    if keep.shape != (counts_arr.shape[1],):
        raise ValueError(
            f"mask must have shape ({counts_arr.shape[1]},), got {keep.shape}."
        )
    if int(keep.sum()) == 0:
        raise ValueError("mask must keep at least one gene.")

    # Keep all genes unchanged when no pooling is needed.
    if bool(np.all(keep)):
        return counts_arr

    kept_counts = counts_arr[:, keep]
    other_counts = counts_arr[:, ~keep].sum(axis=1, keepdims=True)
    return jnp.concatenate([kept_counts, other_counts], axis=1)


def compute_gene_coverage_rank(counts: jnp.ndarray) -> np.ndarray:
    """Compute one-based abundance ranks for genes.

    Parameters
    ----------
    counts : jax.Array or numpy.ndarray, shape ``(n_cells, n_genes)``
        Input count matrix.

    Returns
    -------
    numpy.ndarray of int, shape ``(n_genes,)``
        One-based abundance rank for each gene (1 = highest abundance).
    """
    totals = np.asarray(np.asarray(counts).sum(axis=0), dtype=float).ravel()
    order = np.argsort(-totals)
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, totals.shape[0] + 1, dtype=int)
    return ranks


def build_filtered_gene_names(
    gene_names: Sequence[str],
    mask: Sequence[bool],
    other_name: str = "_other",
) -> Tuple[List[str], List[str]]:
    """Build retained and excluded gene-name lists from a keep mask.

    Parameters
    ----------
    gene_names : sequence of str
        Full gene names aligned to the original gene axis.
    mask : array-like of bool
        Boolean keep mask aligned to ``gene_names``.
    other_name : str, default="_other"
        Label to append for the pooled "other" pseudo-gene when any gene is
        excluded.

    Returns
    -------
    kept_names : list of str
        Retained gene names. Includes ``other_name`` as a trailing entry when
        excluded genes exist.
    excluded_names : list of str
        Excluded gene names that were pooled into "other".
    """
    keep = np.asarray(mask, dtype=bool).ravel()
    if len(gene_names) != keep.shape[0]:
        raise ValueError(
            "gene_names length must match mask length for name construction."
        )

    kept = [name for name, keep_flag in zip(gene_names, keep) if keep_flag]
    excluded = [
        name for name, keep_flag in zip(gene_names, keep) if not keep_flag
    ]
    if excluded:
        kept = [*kept, other_name]
    return kept, excluded
