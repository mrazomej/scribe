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
- Distinguish **plotting counts** from **sampling counts**:
  - plotting counts must match `results.n_genes` for gene selection/panel
    histograms,
  - sampling counts may need the original full-gene matrix for amortized
    subset results.
- For VAE PPC paths, generate predictive samples in full result-space first,
  then subset samples for plotting panels. This avoids rebuilding VAE
  model/guide callables with inconsistent decoder head widths.
- `plot_mean_calibration` auto-detects LNM models and computes the predicted
  mean as `rho * E[u_T]` (compositional probability times expected total
  count) instead of the NB formula `r * p / (1-p)`. The ALR reference index
  is read from `model_config.alr_reference_idx`.
- `plot_mean_calibration` also auto-detects PLN models and computes the
  predicted mean as `exp(y_log_rate)` from the MAP decoder log-rate head.
  This keeps the diagnostic in the model's native log-rate space.

## PLN Compatibility Notes

The viz module supports PLN runs with model-aware behavior:

- **Supported**
  - `plot_mean_calibration` (PLN branch via `y_log_rate`)
  - `plot_ppc` and `plot_ecdf` (generic count-space diagnostics)
  - `plot_capture_anchor` / `plot_p_capture_scaling` when capture priors are active
- **Gracefully skipped for PLN**
  - `plot_correlation_heatmap` (expects NB-family posterior samples such as
    `r` / `mu`; use `get_pln_correlation()` / `get_pln_sigma()` instead)
  - `plot_bio_ppc` (defined as NB `r,p` biological bands)
  - `plot_mixture_ppc` (PLN mixtures are not supported in v1)
