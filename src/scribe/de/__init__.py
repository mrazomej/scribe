"""Bayesian differential expression analysis for compositional data.

This top-level module provides a comprehensive framework for differential
expression analysis that is:

- **Compositional**: Works in CLR/ILR space (reference-invariant).
- **Correlation-aware**: Uses low-rank covariance structure.
- **Fully Bayesian**: Provides exact posterior probabilities, not p-values.

Quick start
-----------
>>> from scribe.de import compare
>>> de = compare(model_A, model_B, gene_names=gene_names)
>>> results = de.gene_level(tau=jnp.log(1.1))
>>> is_de = de.call_genes(lfsr_threshold=0.05)
>>> print(de.summary())

Main components
---------------
- ``ScribeDEResults`` / ``compare()``: Structured pairwise comparison.
- Gene-level differential expression.
- Gene-set / pathway analysis via compositional balances.
- Bayesian error control (lfsr, PEFP).
- Coordinate transformations (ALR, CLR, ILR).
- Gaussianity diagnostics (skewness, kurtosis, Jarque-Bera).
"""

# Results class and factory
from .results import ScribeDEResults, compare

# Parameter extraction
from ._extract import extract_alr_params

# Coordinate transformations
from ._transforms import (
    alr_to_clr,
    transform_gaussian_alr_to_clr,
    build_ilr_basis,
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

__all__ = [
    # Results
    "ScribeDEResults",
    "compare",
    # Extraction
    "extract_alr_params",
    # Transformations
    "alr_to_clr",
    "transform_gaussian_alr_to_clr",
    "build_ilr_basis",
    "clr_to_ilr",
    "ilr_to_clr",
    # Gene-level
    "differential_expression",
    "call_de_genes",
    # Set-level
    "test_contrast",
    "test_gene_set",
    "build_balance_contrast",
    # Bayesian error control
    "compute_lfdr",
    "compute_pefp",
    "find_lfsr_threshold",
    "format_de_table",
    # Diagnostics
    "gaussianity_diagnostics",
]
