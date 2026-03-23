# Theory

SCRIBE is built on a rigorous mathematical foundation rooted in Bayesian
statistics and probabilistic modeling. This section provides an accessible
overview of the core theoretical results that underpin the package, presenting
the key ideas and their implications without exhaustive algebraic derivations.

The probabilistic framework of SCRIBE can be understood through nine
theoretical contributions:

- [**Dirichlet-Multinomial Model**](dirichlet-multinomial.md) — Derives how
  independent negative binomial counts with a shared success probability
  factorize into a negative binomial for totals and a Dirichlet-Multinomial
  for compositions, providing a principled normalization scheme for scRNA-seq
  data.

- [**Hierarchical Gene-Specific \(p\)**](hierarchical-p.md) — Relaxes the
  shared-\(p\) assumption by placing a hierarchical prior on gene-specific
  success probabilities, with a generalized composition sampling procedure
  that strictly extends the Dirichlet model.

- [**Bayesian Denoising**](denoising.md) — Derives a closed-form posterior
  for the true transcript counts given observed UMIs, exploiting
  Poisson-Gamma conjugacy to recover a shifted negative binomial denoised
  distribution, with extensions for zero-inflated models and cross-gene
  correlations.

- [**Anchoring Priors**](anchoring-priors.md) — Resolves two layers of
  practical non-identifiability in the variable capture model: the
  capture-expression degeneracy (via biology-informed capture prior) and
  the mean-overdispersion degeneracy (via data-informed mean anchoring),
  each justified by a law-of-large-numbers concentration argument.

- [**Beta Negative Binomial**](beta-negative-binomial.md) — Extends the
  NB with per-gene power-law tails via a mean-preserving Beta compound,
  with a biophysical interpretation as extrinsic noise in burst size and
  a sparsity-inducing hierarchical prior that defaults to NB behaviour.

- [**Hierarchical Priors**](hierarchical-priors.md) — Introduces the
  three prior families (Gaussian, Horseshoe, NEG) used for adaptive
  shrinkage across genes, mixture components, and datasets, with
  applications to gene-specific \(p\), \(\mu\), the zero-inflation gate,
  and multi-dataset hierarchical models.

- [**Differential Expression**](differential-expression.md) — Develops a
  fully Bayesian DE framework in compositional (CLR) space with three
  inference methods (parametric, empirical, shrinkage), complemented by
  biological-level metrics (LFC, log-variance ratio, Jeffreys divergence)
  that are free of compositional closure.

- [**Model Comparison**](model-comparison.md) — Develops WAIC and
  PSIS-LOO criteria for ranking models by out-of-sample predictive
  accuracy, with pairwise uncertainty quantification, per-gene elpd
  decomposition, and optimal model stacking.

- [**Goodness-of-Fit Diagnostics**](goodness-of-fit.md) — Provides
  expression-scale-invariant per-gene diagnostics via randomized quantile
  residuals (RQR) and posterior predictive checks (PPC), enabling
  principled gene filtering before downstream inference.

!!! tip "For practitioners"
    You do not need to read these pages to use SCRIBE effectively. The [Model
    Selection](../guide/model-selection.md) guide provides all the practical
    information needed to choose and run models. The theory pages are for users
    who want to understand *why* the models work the way they do.
