"""Bayesian differential expression analysis for compositional data.

This top-level module provides a comprehensive framework for differential
expression analysis that is:

- **Compositional**: Works in CLR/ILR space (reference-invariant).
- **Correlation-aware**: Uses low-rank covariance structure.
- **Fully Bayesian**: Provides exact posterior probabilities, not p-values.
- **Assumption-free** (empirical path): Monte Carlo estimation from
  posterior samples with no Gaussian assumption.

Quick start
-----------
Results-object interface (recommended for empirical / shrinkage):

>>> from scribe.de import compare
>>> de = compare(
...     results_bleo, results_ctrl,
...     method="empirical",
...     component_A=0, component_B=0,
... )
>>> results = de.gene_level(tau=jnp.log(1.1))
>>> is_de = de.call_genes(lfsr_threshold=0.05)

Parametric (analytic Gaussian, requires pre-fitted logistic-normal):

>>> de = compare(model_A, model_B, gene_names=gene_names)
>>> results = de.gene_level(tau=jnp.log(1.1))

Class hierarchy
---------------
- ``ScribeDEResults`` — abstract base with shared methods
  (``call_genes``, ``compute_pefp``, ``find_threshold``, ``summary``).
- ``ScribeParametricDEResults`` — analytic Gaussian path from low-rank
  ALR parameters.
- ``ScribeEmpiricalDEResults`` — Monte Carlo path from posterior CLR
  difference samples.
- ``ScribeShrinkageDEResults`` — empirical Bayes shrinkage on top of
  the Monte Carlo path (scale mixture of normals prior).

Other components
----------------
- Gene-level differential expression.
- Gene-set / pathway analysis via compositional balances.
- Bayesian error control (lfsr, PEFP).
- Coordinate transformations (ALR, CLR, ILR).
- Gaussianity diagnostics (skewness, kurtosis, Jarque-Bera).
- Empirical DE from posterior samples (non-parametric).
- Empirical Bayes shrinkage for improved per-gene inference.
"""

# Results class hierarchy and factory
from .results import (
    ScribeDEResults,
    ScribeParametricDEResults,
    ScribeEmpiricalDEResults,
    ScribeShrinkageDEResults,
    compare,
)

# Parameter extraction
from ._extract import extract_alr_params

# Coordinate transformations
from ._transforms import (
    alr_to_clr,
    transform_gaussian_alr_to_clr,
    build_ilr_basis,
    build_ilr_balance,
    build_pathway_sbp_basis,
    clr_to_ilr,
    ilr_to_clr,
)

# Gene-level DE
from ._gene_level import (
    differential_expression,
    call_de_genes,
)

# Set-level analysis
from ._set_level import (
    test_contrast,
    test_gene_set,
    build_balance_contrast,
    empirical_test_gene_set,
    empirical_test_pathway_perturbation,
    empirical_test_multiple_gene_sets,
)

# Bayesian error control and utilities
from ._error_control import (
    compute_lfdr,
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
)

# Gaussianity diagnostics
from ._gaussianity import gaussianity_diagnostics

# Empirical (non-parametric) DE
from ._empirical import (
    compute_clr_differences,
    empirical_differential_expression,
    compute_expression_mask,
)

# Empirical Bayes shrinkage
from ._shrinkage import (
    fit_scale_mixture_prior,
    shrinkage_differential_expression,
)

__all__ = [
    # Results hierarchy
    "ScribeDEResults",
    "ScribeParametricDEResults",
    "ScribeEmpiricalDEResults",
    "ScribeShrinkageDEResults",
    "compare",
    # Extraction
    "extract_alr_params",
    # Transformations
    "alr_to_clr",
    "transform_gaussian_alr_to_clr",
    "build_ilr_basis",
    "build_ilr_balance",
    "build_pathway_sbp_basis",
    "clr_to_ilr",
    "ilr_to_clr",
    # Gene-level
    "differential_expression",
    "call_de_genes",
    # Set-level
    "test_contrast",
    "test_gene_set",
    "build_balance_contrast",
    "empirical_test_gene_set",
    "empirical_test_pathway_perturbation",
    "empirical_test_multiple_gene_sets",
    # Bayesian error control
    "compute_lfdr",
    "compute_pefp",
    "find_lfsr_threshold",
    "format_de_table",
    # Diagnostics
    "gaussianity_diagnostics",
    # Empirical (non-parametric) DE
    "compute_clr_differences",
    "empirical_differential_expression",
    "compute_expression_mask",
    # Empirical Bayes shrinkage
    "fit_scale_mixture_prior",
    "shrinkage_differential_expression",
]
