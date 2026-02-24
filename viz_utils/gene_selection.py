"""Gene selection utilities for ECDF and PPC plots."""

import numpy as np


def _select_genes_simple(counts, n_genes):
    """Simple gene selection for ECDF plots (linear spacing)."""
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]
    sorted_idx = nonzero_idx[np.argsort(mean_counts[nonzero_idx])]
    spaced_indices = np.linspace(0, len(sorted_idx) - 1, num=n_genes, dtype=int)
    selected_idx = sorted_idx[spaced_indices]
    return selected_idx, mean_counts


def _select_genes(counts, n_rows, n_cols):
    """Select genes for plotting using log-spaced binning."""
    mean_counts = np.median(counts, axis=0)
    nonzero_idx = np.where(mean_counts > 0)[0]

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
