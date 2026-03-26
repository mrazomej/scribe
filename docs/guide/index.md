# User Guide

This section covers the practical workflow of using SCRIBE---from selecting a
model through downstream analysis.

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } **The `scribe.fit()` Interface**

    ---

    The single entry point for all SCRIBE inference, with every parameter group
    explained, code examples, and links to deeper pages

    [The `scribe.fit()` Interface](fit.md)

-   :material-view-grid:{ .lg .middle } **Model Selection**

    ---

    Choosing the right model: NB base, zero inflation, variable capture, BNB
    overdispersion, mixture components, and parameterizations

    [Model Selection](model-selection.md)

-   :material-palette-swatch:{ .lg .middle } **Parameter Reference**

    ---

    Color-coded cheatsheet mapping every internal parameter name to its symbol,
    equation context, biological meaning, and parameterization

    [Parameter Reference](parameters.md)

-   :material-chart-bell-curve:{ .lg .middle } **Variational Guide Families**

    ---

    Mean-field, low-rank, joint low-rank, normalizing flows, amortized, and VAE
    latent guides: what they capture and when to use each

    [Variational Guide Families](guide-families.md)

-   :material-cog-play:{ .lg .middle } **Inference Methods**

    ---

    Choosing between SVI, MCMC, and VAE, key parameters, early stopping, and
    the SVI-to-MCMC warm-start workflow

    [Inference Methods](inference.md)

-   :material-package-variant:{ .lg .middle } **Results Class**

    ---

    Understanding and using `ScribeResults` for posterior analysis, sampling,
    denoising, and normalization

    [Results Class](results.md)

-   :material-compare-horizontal:{ .lg .middle } **Differential Expression**

    ---

    Bayesian DE with three methods, error control via lfsr and PEFP,
    biological-level metrics, gene masking, and pathway analysis

    [Differential Expression](differential-expression.md)

-   :material-scale-balance:{ .lg .middle } **Model Comparison**

    ---

    WAIC, PSIS-LOO, stacking weights, per-gene goodness-of-fit diagnostics, and
    integration with the DE pipeline

    [Model Comparison](model-comparison.md)

-   :material-console:{ .lg .middle } **`scribe-infer` CLI**

    ---

    Reproducible, config-driven inference via Hydra with SLURM integration and
    automatic covariate-split orchestration

    [`scribe-infer` CLI](cli_infer.md)

-   :material-chart-box-outline:{ .lg .middle } **`scribe-visualize` CLI**

    ---

    Post-inference diagnostic plots: loss curves, ECDF, PPC grids, UMAP
    overlays, heatmaps, and more --- with recursive and SLURM support

    [`scribe-visualize` CLI](cli_visualize.md)

</div>
