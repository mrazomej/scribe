"""
Bayesian differential expression analysis for compositional data.

This module provides a comprehensive framework for differential expression
analysis that is:
- Compositional: Works in CLR/ILR space (reference-invariant)
- Correlation-aware: Uses low-rank covariance structure
- Fully Bayesian: Provides exact posterior probabilities, not p-values

Main components:
- Global divergence metrics (KL, JS, Mahalanobis)
- Gene-level differential expression
- Gene-set/pathway analysis via balances
- Bayesian error control (lfsr, PEFP)
"""

# Import divergence metrics from stats module
# These are registered via multipledispatch and accessed through the global
# namespace
from ...stats.divergences import (
    kl_divergence,
    jensen_shannon,
    mahalanobis,
    _extract_lowrank_params,
)

# Transformations
from .transformations import (
    alr_to_clr,
    transform_gaussian_alr_to_clr,
    build_ilr_basis,
    clr_to_ilr,
    ilr_to_clr,
)

# Gene-level DE
from .gene_level import (
    differential_expression,
    call_de_genes,
)

# Set-level analysis
from .set_level import (
    test_contrast,
    test_gene_set,
    build_balance_contrast,
)

# Bayesian error control and utilities
from .utils import (
    compute_lfdr,
    compute_pefp,
    find_lfsr_threshold,
    format_de_table,
)

__all__ = [
    # Divergences (imported from stats.divergences, used via dispatch)
    "kl_divergence",
    "jensen_shannon",
    "mahalanobis",
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
]
