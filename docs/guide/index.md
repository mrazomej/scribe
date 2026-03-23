# User Guide

This section covers the practical workflow of using SCRIBE -- from choosing an
inference method through downstream analysis.

- [**Inference Methods**](inference.md) — choosing between SVI, MCMC, and VAE,
  key parameters, guide families, early stopping, and the SVI-to-MCMC
  warm-start workflow

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
