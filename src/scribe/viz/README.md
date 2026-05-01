# SCRIBE Viz

## Auto-Alignment to Model Gene Space

Visualization entry points now accept either:

- counts already in the fitted model-space gene axis, or
- counts in the original pre-filter gene axis when `gene_coverage` prefiltering
  was used during `scribe.fit()`.

When a fitted results object exposes `gene_coverage_mask`, the viz layer
automatically aligns original-space counts to model-space counts by:

1. keeping genes where the mask is `True`, in original order, and
2. aggregating all excluded genes (`False`) into a trailing pooled
   "other" column.

This preserves compositional closure and avoids index mismatch errors when
plotting from raw `adata` counts against filtered model results.

## Current Policy

- **Default behavior:** auto-align when possible.
- **No-op behavior:** if `counts.shape[1] == results.n_genes`, counts are used
  as-is.
- **Error behavior:** if counts cannot be aligned to `results.n_genes` and the
  mask is unavailable or incompatible, plotting raises a clear `ValueError`.

## Notes for Developers

- Do not manually re-implement pooled-gene aggregation in each plotting module.
  Use `scribe.viz.gene_selection._coerce_and_align_counts_to_results(...)`.
- When a plotting path subsets `results` to specific genes, pass a matching
  column-subset of counts into predictive/MAP calls to keep array shapes
  consistent.
