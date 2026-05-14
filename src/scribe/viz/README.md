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
- For subset-sampling PPC paths (non-VAE / non-amortized), reuse the exact
  subset results object passed into predictive sampling when plotting panels.
  Reconstructing a fresh `results[selected_idx]` view can drop freshly cached
  `predictive_samples` when the parent results object has no predictive cache.
- `plot_mean_calibration` auto-detects LNM models and computes the predicted
  mean as `rho * E[u_T]` (compositional probability times expected total
  count) instead of the NB formula `r * p / (1-p)`. The ALR reference index
  is read from `model_config.alr_reference_idx`.
- `plot_mean_calibration` also auto-detects PLN models and computes the
  predicted mean as `exp(y_log_rate)` from the MAP decoder log-rate head.
  This keeps the diagnostic in the model's native log-rate space.

## Corner PPC (`plot_corner_ppc`)

`corner_ppc.py` renders an **N x N triangular grid** for a small set of genes
(typically 4-5), combining marginal and bivariate posterior predictive checks in
a single figure:

- **Diagonal panels** — marginal PPC histograms identical to `plot_ppc`: shaded
  credible-region bands from posterior predictive draws with the observed count
  histogram overlaid.
- **Lower-triangle panels** — bivariate PPC panels: 2-D density contour
  levels computed via
  pooled posterior predictive samples overlaid. By default, these densities are
  computed with a fast `numpy.histogram2d` path (`density_method="hist2d"`).
  For smoother (but slower) contours, `density_method="kde"` uses
  `scipy.stats.gaussian_kde`. Scatter points default to `scatter_color="black"`
  and are rendered in the background under contours.
  Contours use HPD-style mass levels by default:
  `contour_mass_levels=(0.5, 0.68, 0.95, 0.99)`.
  Gray contour-line edges are drawn on smoother panels by default
  (`draw_contour_edges=True`, `contour_edgecolor="gray"`), but are
  automatically suppressed on strongly discrete low-count panels to avoid
  dense stripe artifacts (`suppress_contour_edges_for_discrete=True`).
  Low-density background is left unfilled so panel facecolor continues to follow
  the active Matplotlib style.
- **Upper triangle** — hidden.

By default, off-diagonal panel limits are matched to their corresponding
diagonal marginals (`match_offdiag_limits_to_marginals=True`), so each bivariate
panel uses the same x/y ranges as its gene-specific marginals.

### Gene selection

Three modes (checked in priority order):

1. `gene_indices` — explicit column indices.
2. `gene_names_list` — gene name strings resolved against `results.var.index`.
3. **Auto (correlation-diversity)** — selects genes that span the correlation
   spectrum so the corner grid contains panels with strong positive, strong
   negative, and near-zero correlations:
   - **Expression floor**: candidates are filtered to genes with mean UMI
     `>= min_mean_umi_for_selection` (default `5`) before correlation-diversity
     selection.
   - **Seed**: find the most positively correlated pair and the most negatively
     correlated pair (up to 4 unique genes).
   - **Greedy fill**: add genes that maximise pairwise diversity (the gene
     whose minimum absolute correlation with all already-selected genes is
     largest).
   - **Sort**: final selection is ordered by median expression (ascending).
   - **Correlation source**: model-aware dispatch — Laplace fits use the
     analytic `W W^T + diag(d)` correlation, VAE PLN fits use
     `get_pln_correlation()`, everything else falls back to empirical Pearson
     on `log1p(counts)`.
   - **Nuisance removal** (`subtract_direction`): optionally project out
     library-size or top-PC directions before gene selection, mirroring the
     same option in `plot_correlation_heatmap`.  For Laplace/VAE PLN fits this
     uses the analytic `get_correlation_residual()`.  For the empirical path,
     `"library_size"` falls back to `"pc"` (eigendecomposition of the empirical
     covariance).

### Performance

The default `hist2d` estimator is designed for speed on large pooled PPC
samples.  When `density_method="kde"`, pooled samples are subsampled to at most
50 000 points and tiny jitter is added to break integer ties for KDE stability.

## Compositional PPCs (`plot_compositional_ppc`, `plot_compositional_corner_ppc`)

`plot_compositional_ppc` and `plot_compositional_corner_ppc` render
**compositional** posterior predictive checks: they consume
`results.get_compositional_samples(...)` (population-level simplex draws,
*pre-observation-noise*) and compare against two empirical comparators:

1. **Per-cell empirical compositions** — the cloud of `u_c / N_c` across
   cells. Carries per-cell Multinomial sampling noise on top of the true
   compositions.
2. **Dataset-level pseudobulk** — single point `sum_c u_c / sum_c N_c`.
   Noise averages away, leaving an unbiased estimator of the population
   mean composition.

Layout per panel:

- 1-D (diagonals + `plot_compositional_ppc`):
  - shaded model histogram (population compositional distribution),
  - step empirical histogram (per-cell `u_c,g / N_c`),
  - dashed vertical line at the pseudobulk value.
- 2-D (lower-triangle of `plot_compositional_corner_ppc`):
  - filled HPD contours from the pooled model samples,
  - per-cell empirical scatter,
  - large pseudobulk marker (`X`).

These are the cleanest available diagnostics for the **gauge-invariant
compositional structure** (Theorem 1 in
`paper/_diffexp_nbln_robustness.qmd`). Unlike `plot_ppc(level="library_anchored")`,
which adds Multinomial sampling noise to the model output, the
compositional PPC exposes the model's simplex distribution directly —
making it the right tool for inspecting cross-gene relationships
regardless of how the freeze pinned the gauge.

**Caveat — low-abundance noise floor.** Per-cell empirical compositions
have Multinomial sampling variance `p_g(1-p_g)/N_c`. For genes with
`p_g ≪ 1/N`, the empirical histogram is noise-dominated and the model
distribution will be *narrower* than the empirical one. That is a
correct prediction, not a misfit. The default `min_mean_umi=5.0`
filters auto-selected genes above this noise floor; lower it explicitly
only when the dataset has uniformly low expression and you understand
the caveat.

## W-shrinkage spectrum (`plot_w_shrinkage_spectrum`)

Renders the per-factor compositional-loading spectrum from a Phase-3
Laplace fit configured with a `w_prior` strategy. Companion diagnostic
to `plot_compositional_corner_ppc`: a clean elbow here correlates
with collapsed cross-block diagonals in the compositional corner panels.

```python
result = scribe.fit(
    adata, model="nbln", inference_method="laplace",
    priors={
        "loadings": {"type": "horseshoe_columnwise", "tau_scale": 1.0},
    },
    latent_dim=16,
)
scribe.viz.plot_w_shrinkage_spectrum(result, figsize=(5, 3.5))
```

What's drawn:

- **Primary (solid)**: `||W_⟂[:, k]||` sorted descending — the
  data-supported, gauge-invariant per-factor norm. Pulled from
  `result.w_prior_diagnostics["column_frobenius_compositional"]`. This
  is the spectrum that drives `column_norm_effective_rank` (the
  headline rank diagnostic, also exposed as `effective_rank`).
- **Secondary (dashed, when `show_sigma_k=True`)**: the strategy's
  aux MAP scales `σ_k`. Provided for comparison so users can see how
  the prior's aux scales relate to the data-supported column norms.
  Gracefully omitted for strategies without aux scales (`gaussian`).
- **Threshold (gray dotted)**: 5%-of-max line marking the
  `column_norm_effective_rank` cutoff.

Requires `result.w_prior_diagnostics is not None` — raises
`ValueError` for LNM-family results (which don't run the W-prior
integration in v1) and for fits without a `w_prior` configured. See
[`src/scribe/laplace/README.md`](../laplace/README.md) for the strategy
catalog and calibration workflow.

## `plot_ppc` conditioning levels

`plot_ppc(results, counts, ppc_level=...)` selects how much observed
data enters each predictive draw:

| `ppc_level` | Conditioning | Use case |
|---|---|---|
| `"marginal"` (default) | Fully unconditional — `x`, `η`, `N_c` all drawn from the model | Honest "does the generative story match the data?" |
| `"library_anchored"` | Fresh composition from prior; **observed per-cell totals** pin the Multinomial | Compositional fit test, isolated from totals/capture |
| `"per_cell"` | Per-cell MAP latents + Laplace noise; observed totals | Per-cell predictive — most conditioned |

For NBLN Laplace fits using the Phase-2 cascade freeze, **marginal**
and **per_cell** PPCs route frozen `r`/`mu`/`eta` through the embedded
`cascade_source` SVI guide (preserves full SVI fidelity).
**library_anchored** is composition-only — `r` and `eta` don't enter,
so the cascade-routing is a no-op there. The mean-calibration plot
(`plot_mean_calibration`) uses `x_loc` directly and is *not* a PPC; it
will exhibit Jensen-inequality inflation for very-low-expression genes
on heavy-tailed sparse data, which is a diagnostic artifact rather than
a model misfit. For population-level correlation analysis, use
`results.get_W_compositional()` — gauge-invariant under the
rigid-translation symmetry of NBLN. See
[`src/scribe/laplace/README.md`](../laplace/README.md) for the full
cascade-aware PPC routing details.

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
