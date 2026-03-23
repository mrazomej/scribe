# User Guide

This section covers the practical workflow of using SCRIBE---from selecting a
model through downstream analysis.

- [**Model Selection**](model-selection.md) — choosing the right model for your
  data: NB base, zero inflation, variable capture, BNB overdispersion, mixture
  components, parameterizations, and hierarchical priors

- [**Variational Guide Families**](guide-families.md) — mean-field, low-rank,
  joint low-rank, amortized, and VAE latent guides: what they capture, when to
  use each, and how to configure them

- [**Inference Methods**](inference.md) — choosing between SVI, MCMC, and VAE,
  key parameters, early stopping, and the SVI-to-MCMC warm-start workflow

- [**Results Class**](results.md) — understanding and using the `ScribeResults`
  object for posterior analysis, sampling, denoising, and normalization

- [**Differential Expression**](differential-expression.md) — Bayesian DE with
  three methods (parametric, empirical, shrinkage), error control via lfsr and
  PEFP, biological-level metrics, gene masking, and pathway analysis

- [**Model Comparison**](model-comparison.md) — comparing models with WAIC and
  PSIS-LOO, stacking weights, per-gene goodness-of-fit diagnostics, and
  integration with the DE pipeline

- [**Custom Models**](custom-model.md) — defining your own probabilistic models
  and guides within the SCRIBE framework
