# Theory

SCRIBE is built on a rigorous mathematical foundation rooted in Bayesian
statistics and probabilistic modeling. This section provides an accessible
overview of the core theoretical results that underpin the package, presenting
the key ideas and their implications without exhaustive algebraic derivations.

!!! tip "For practitioners"
    You do not need to read these pages to use SCRIBE effectively. The [Model
    Selection](../guide/model-selection.md) guide provides all the practical
    information needed to choose and run models. The theory pages are for users
    who want to understand *why* the models work the way they do.

---

## Foundations: the Negative Binomial core

The default SCRIBE models treat genes as independent Negative Binomial draws
whose parameters arise from the biophysics of bursty transcription and mRNA
capture. This family covers the vast majority of use cases and forms the
backbone of the inference, denoising, and differential expression pipelines.

- :material-dice-multiple:{ .middle } [**Dirichlet-Multinomial Model**](dirichlet-multinomial.md) — Derives how
  independent negative binomial counts with a shared success probability
  factorize into a negative binomial for totals and a Dirichlet-Multinomial
  for compositions, providing a principled normalization scheme for scRNA-seq
  data.

- :material-sitemap:{ .middle } [**Hierarchical Gene-Specific p**](hierarchical-p.md) — Relaxes the
  shared-p assumption by placing a hierarchical prior on gene-specific
  success probabilities, with a generalized composition sampling procedure
  that strictly extends the Dirichlet model.

- :material-chart-bell-curve-cumulative:{ .middle } [**Beta Negative Binomial**](beta-negative-binomial.md) — Extends the
  NB with per-gene power-law tails via a mean-preserving Beta compound,
  with a biophysical interpretation as extrinsic noise in burst size and
  a sparsity-inducing hierarchical prior that defaults to NB behaviour.

- :material-anchor:{ .middle } [**Anchoring Priors**](anchoring-priors.md) — Resolves two layers of
  practical non-identifiability in the variable capture model: the
  capture-expression degeneracy (via biology-informed capture prior) and
  the mean-overdispersion degeneracy (via data-informed mean anchoring),
  each justified by a law-of-large-numbers concentration argument.

- :material-layers-triple:{ .middle } [**Hierarchical Priors**](hierarchical-priors.md) — Introduces the
  three prior families (Gaussian, Horseshoe, NEG) used for adaptive
  shrinkage across genes, mixture components, and datasets, with
  applications to gene-specific p, mu, the zero-inflation gate,
  and multi-dataset hierarchical models.

---

## Gene regulatory network models

When genes interact through regulatory networks, the independent-gene
assumption no longer holds. Under the linear noise approximation, the
steady-state distribution of log-abundances becomes a multivariate Gaussian
whose covariance encodes the regulatory structure. These models learn that
covariance explicitly, enabling recovery of gene-gene correlation programs.

- :material-dna:{ .middle } [**GRN Biophysics**](grn-biophysics.md) — Derives that the
  steady-state distribution of gene expression in a gene regulatory network
  under the linear noise approximation is a multivariate Gaussian on
  log-abundances, providing the shared biophysical foundation for both the
  PLN and LNM observation models.

- :material-chart-scatter-plot:{ .middle } [**Poisson Log-Normal Model**](poisson-lognormal.md) — Develops
  the PLN observation model where each gene's count is a Poisson draw from
  the exponentiated log-abundance, yielding log-concave posteriors, natural
  total-count coupling, and efficient Laplace inference via Woodbury Newton.

- :material-vector-polyline:{ .middle } [**Logistic-Normal Multinomial Model**](logistic-normal-multinomial.md) — Replaces
  the Dirichlet prior on compositions with a logistic-normal distribution,
  enabling arbitrary cross-gene correlations in log-ratio space with inference
  via Laplace approximation or VAE.

---

## Analysis framework

Once any model is fit---whether from the NB family or the log-normal
family---SCRIBE provides a unified set of tools for extracting biological
conclusions and validating the fit. These methods operate on the posterior
regardless of which generative model produced it.

- :material-filter-variant:{ .middle } [**Bayesian Denoising**](denoising.md) — Derives a closed-form posterior
  for the true transcript counts given observed UMIs, exploiting
  Poisson-Gamma conjugacy to recover a shifted negative binomial denoised
  distribution, with extensions for zero-inflated models and cross-gene
  correlations.

- :material-compare-horizontal:{ .middle } [**Differential Expression**](differential-expression.md) — Develops a
  fully Bayesian DE framework in compositional (CLR) space with three
  inference methods (parametric, empirical, shrinkage), complemented by
  biological-level metrics (LFC, log-variance ratio, Jeffreys divergence)
  that are free of compositional closure.

- :material-scale-balance:{ .middle } [**Model Comparison**](model-comparison.md) — Develops WAIC and
  PSIS-LOO criteria for ranking models by out-of-sample predictive
  accuracy, with pairwise uncertainty quantification, per-gene elpd
  decomposition, and optimal model stacking.

- :material-check-decagram:{ .middle } [**Goodness-of-Fit Diagnostics**](goodness-of-fit.md) — Provides
  expression-scale-invariant per-gene diagnostics via randomized quantile
  residuals (RQR) and posterior predictive checks (PPC), enabling
  principled gene filtering before downstream inference.
