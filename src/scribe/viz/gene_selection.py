"""Gene selection utilities for ECDF and PPC plots."""

from typing import Optional

import numpy as np
import logging

_log = logging.getLogger(__name__)


def _resolve_pooled_other_idx(results) -> Optional[int]:
    """Position-in-model-gene-space of the pooled ``_other`` aggregate.

    Returns the index of the trailing pooled ``_other`` column for fits
    where ``gene_coverage<1.0`` was applied, regardless of the
    ``correlate_other_column`` setting.  The pooled aggregate is not a
    real gene and should not be selected for per-gene viz panels.

    Priority order:

    1. ``axis_layout.other_idx`` — present only under the decoupled
       layout (``correlate_other_column=False``).
    2. ``_excluded_gene_names`` non-empty — gene-coverage filtering ran,
       and the trailing model-space column is the pooled aggregate
       even under the legacy ``correlate_other_column=True`` layout.
    3. ``var.index[-1] == "_other"`` — manually-pre-filtered AnnData;
       last-resort literal-name match.

    Returns ``None`` when no pooled ``_other`` column is present.

    LNM exception
    -------------
    For LNM / LNMVCP fits, returns ``None`` regardless of the above.
    LNM treats the trailing column as the ALR reference gene, and the
    result's ``__getitem__`` subset operation REQUIRES the reference
    to be included in any subset.  Excluding ``_other`` from gene
    selection would produce a subset that fails the
    ``_subset_lnm`` reference-in-subset check.  LNM's predictive
    distribution for ``_other`` is also well-defined (recovered from
    the inverse-ALR), so keeping it in the gene-selection candidate
    pool is harmless.
    """
    model_config = getattr(results, "model_config", None)
    base_model = getattr(model_config, "base_model", None)
    base_model_str = (
        str(getattr(base_model, "value", base_model) or "").lower()
    )
    if base_model_str in ("lnm", "lnmvcp"):
        return None
    layout = getattr(results, "axis_layout", None)
    if layout is not None and getattr(layout, "other_idx", None) is not None:
        return int(layout.other_idx)
    excluded = getattr(results, "_excluded_gene_names", None)
    if excluded:
        n_genes = int(getattr(results, "n_genes", 0))
        if n_genes > 0:
            return n_genes - 1
    var = getattr(results, "var", None)
    if var is not None and hasattr(var, "index") and len(var.index) > 0:
        if str(var.index[-1]) == "_other":
            return int(len(var.index)) - 1
    return None


def _get_gene_names(results):
    """Extract gene names from a results object when available.

    Parameters
    ----------
    results : object
        Fitted model results.  Gene names are read from
        ``results.var.index`` when the ``var`` DataFrame is present.

    Returns
    -------
    pandas.Index or None
        Gene names index, or ``None`` when unavailable.
    """
    var = getattr(results, "var", None)
    if var is not None and hasattr(var, "index"):
        return var.index
    return None


def _resolve_explicit_genes(genes, results, counts=None):
    """Resolve an explicit gene selection to integer indices.

    Maps a caller-supplied list of gene identifiers to integer indices
    in the results' gene space, **preserving the caller-specified
    order** so that repeated calls with the same ``genes`` list line up
    panel-for-panel (e.g. comparing two models on the same genes).

    Parameters
    ----------
    genes : sequence
        Gene identifiers to display.  Each element is either an integer
        index into the results' gene axis, or a gene-name string matched
        against ``results.var.index``.
    results : object
        Fitted SCRIBE results object (provides gene names and
        ``n_genes``).
    counts : array-like, optional
        Observed counts, used only to infer the gene count when the
        results object does not expose ``n_genes``.

    Returns
    -------
    numpy.ndarray of int
        Integer gene indices in caller order.

    Raises
    ------
    ValueError
        If a name is not found, an integer index is out of range, a name
        is requested when the results object carries no gene names, or
        ``genes`` is empty.
    """
    gene_names = _get_gene_names(results)
    name_to_idx = (
        {str(name): i for i, name in enumerate(gene_names)}
        if gene_names is not None
        else None
    )
    n_genes = int(getattr(results, "n_genes", 0) or 0)
    if n_genes <= 0 and counts is not None:
        n_genes = int(_coerce_counts(counts).shape[1])

    resolved = []
    for g in genes:
        if isinstance(g, (int, np.integer)) and not isinstance(g, bool):
            idx = int(g)
            if n_genes and not (0 <= idx < n_genes):
                raise ValueError(
                    f"Gene index {idx} is out of range "
                    f"[0, {n_genes - 1}]."
                )
            resolved.append(idx)
        else:
            key = str(g)
            if name_to_idx is None:
                raise ValueError(
                    "Gene names were requested, but the results object "
                    "carries no gene names; pass integer indices instead."
                )
            if key not in name_to_idx:
                raise ValueError(
                    f"Gene name {key!r} not found in the results gene "
                    "space (it may have been excluded by gene_coverage "
                    "filtering and pooled into '_other')."
                )
            resolved.append(name_to_idx[key])

    if not resolved:
        raise ValueError("`genes` must contain at least one gene.")
    return np.asarray(resolved, dtype=int)


def _coerce_counts(counts):
    """Coerce counts to a dense 2-D numpy array.

    Handles AnnData objects (extracts ``.X``), scipy sparse matrices
    (``.toarray()``), and JAX arrays transparently.
    """
    if hasattr(counts, "X"):
        counts = counts.X
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    return np.asarray(counts)


def _coerce_and_align_counts_to_results(counts, results, *, context="viz"):
    """Coerce counts and align them to the fitted results gene space.

    Parameters
    ----------
    counts : array-like or AnnData
        Observed count matrix in either original or model-space gene axes.
    results : object
        Fitted SCRIBE results object.
    context : str, default="viz"
        Caller context used to make mismatch errors easier to diagnose.

    Returns
    -------
    numpy.ndarray
        Dense count matrix aligned to ``results.n_genes``.

    Raises
    ------
    ValueError
        If counts cannot be aligned to the model gene-space.
    """
    counts_arr = _coerce_counts(counts)
    n_genes_results = int(getattr(results, "n_genes", counts_arr.shape[1]))
    if int(counts_arr.shape[1]) == n_genes_results:
        return counts_arr

    mask = getattr(results, "gene_coverage_mask", None)
    if mask is None:
        mask = getattr(results, "_gene_coverage_mask", None)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool).ravel()
        if int(mask_arr.shape[0]) == int(counts_arr.shape[1]):
            from ..core.gene_coverage import aggregate_counts_by_mask

            aligned = np.asarray(
                aggregate_counts_by_mask(counts_arr, mask_arr)
            )
            if int(aligned.shape[1]) == n_genes_results:
                _log.info(
                    f"[{context}] Auto-aligned counts from original gene "
                    "space to model gene space using gene_coverage_mask."
                )
                return aligned

    raise ValueError(
        f"[{context}] counts gene dimension ({counts_arr.shape[1]}) does not "
        f"match results.n_genes ({n_genes_results}) and could not be aligned "
        "with results.gene_coverage_mask."
    )


def _select_genes_simple(counts, n_genes, *, exclude_idx=None):
    """Simple gene selection for ECDF plots (linear spacing).

    Parameters
    ----------
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    n_genes : int
        Number of genes to select.
    exclude_idx : int, optional
        Gene-axis position to exclude from selection (e.g., the pooled
        ``_other`` aggregate column).  Pass
        :func:`_resolve_pooled_other_idx` from the call site.
    """
    counts = _coerce_counts(counts)
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]
    if exclude_idx is not None:
        nonzero_idx = nonzero_idx[nonzero_idx != int(exclude_idx)]
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    spaced_indices = np.linspace(0, len(sorted_idx) - 1, num=n_genes, dtype=int)
    selected_idx = sorted_idx[spaced_indices]
    return selected_idx, mean_counts


def _select_genes(counts, n_rows, n_cols, *, exclude_idx=None):
    """Select genes for plotting using log-spaced binning.

    Parameters
    ----------
    counts : array-like
        Observed count matrix ``(n_cells, n_genes)``.
    n_rows, n_cols : int
        Target grid dimensions (selection produces ``n_rows * n_cols``
        genes).
    exclude_idx : int, optional
        Gene-axis position to exclude from selection (e.g., the pooled
        ``_other`` aggregate column).  Pass
        :func:`_resolve_pooled_other_idx` from the call site.
    """
    counts = _coerce_counts(counts)
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]
    if exclude_idx is not None:
        nonzero_idx = nonzero_idx[nonzero_idx != int(exclude_idx)]

    if len(nonzero_idx) == 0:
        return np.array([], dtype=int), mean_counts

    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    sorted_means = mean_counts[sorted_idx]
    min_expr = sorted_means[0]
    max_expr = sorted_means[-1]
    min_expr_safe = max(min_expr, 0.1)
    bin_edges = np.logspace(
        np.log10(min_expr_safe), np.log10(max_expr), num=n_rows + 1
    )
    bin_edges[0] = min_expr

    selected_set = set()
    selected_by_bin = []

    for i in range(n_rows):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        if i == n_rows - 1:
            in_bin = (sorted_means >= bin_start) & (sorted_means <= bin_end)
        else:
            in_bin = (sorted_means >= bin_start) & (sorted_means < bin_end)

        bin_indices = np.where(in_bin)[0]
        bin_selected = []

        if len(bin_indices) > 0:
            if len(bin_indices) <= n_cols:
                bin_selected = list(bin_indices)
            else:
                bin_means = sorted_means[bin_indices]
                bin_min = bin_means[0]
                bin_max = bin_means[-1]
                bin_min_safe = max(bin_min, 0.1)
                log_targets = np.logspace(
                    np.log10(bin_min_safe), np.log10(bin_max), num=n_cols
                )
                log_targets[0] = bin_min
                for target in log_targets:
                    closest_idx = np.argmin(np.abs(bin_means - target))
                    bin_selected.append(bin_indices[closest_idx])
                bin_selected = list(np.unique(bin_selected))

        selected_by_bin.append(bin_selected)
        selected_set.update(bin_selected)

    all_indices = set(range(len(sorted_idx)))
    unselected_indices = sorted(list(all_indices - selected_set))
    unselected_means = sorted_means[unselected_indices]

    final_selected = []
    for i in range(n_rows):
        bin_selected = selected_by_bin[i]
        needed = n_cols - len(bin_selected)

        if needed > 0:
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = np.sqrt(bin_start * bin_end)
            candidates = []
            for idx in unselected_indices:
                expr = sorted_means[idx]
                if expr <= bin_end:
                    distance = abs(expr - bin_center)
                    candidates.append((distance, idx))
            candidates.sort(key=lambda x: x[0])
            backfill_indices = [idx for _, idx in candidates[:needed]]
            bin_selected.extend(backfill_indices)
            for idx in backfill_indices:
                unselected_indices.remove(idx)

        final_selected.extend(
            [sorted_idx[idx] for idx in bin_selected[:n_cols]]
        )

    selected_idx = np.array(final_selected, dtype=int)
    return selected_idx, mean_counts
