# Theory

SCRIBE is built on a rigorous mathematical foundation rooted in Bayesian
statistics and probabilistic modeling. This section provides an accessible
overview of the core theoretical results that underpin the package, presenting
the key ideas and their implications without exhaustive algebraic derivations.

The probabilistic framework of SCRIBE can be understood through three
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

!!! tip "For practitioners"
    You do not need to read these pages to use SCRIBE effectively. The
    [Models](../models/index.md) section provides all the practical information
    needed to choose and run models. The theory pages are for users who want to
    understand *why* the models work the way they do.
