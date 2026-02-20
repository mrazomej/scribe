"""Bayesian model comparison for SCRIBE.

This module provides scalable, fully Bayesian model comparison tools based on
out-of-sample predictive accuracy.  It implements two complementary criteria:

- **WAIC** (Widely Applicable Information Criterion): a fast, analytical
  approximation to LOO-CV computed entirely from the posterior samples already
  available after fitting.
- **PSIS-LOO** (Pareto-Smoothed Importance Sampling LOO): a more reliable
  criterion that applies Pareto smoothing to the raw IS weights, with a
  per-observation diagnostic k̂.

In addition, the module provides:

- **Gene-level comparison**: per-gene elpd differences between two models,
  with standard errors and z-scores.
- **Model stacking**: optimal predictive ensemble weights via convex
  optimization of the LOO log-score.

Quick start
-----------

>>> from scribe.mc import compare_models
>>> mc = compare_models(
...     [results_nbdm, results_hierarchical],
...     counts=counts,
...     model_names=["NBDM", "Hierarchical"],
...     gene_names=gene_names,
...     compute_gene_liks=True,
... )
>>> print(mc.summary())           # ranked comparison table
>>> print(mc.diagnostics())       # PSIS k̂ diagnostics
>>> mc.rank()                     # pandas DataFrame
>>> mc.gene_level_comparison("NBDM", "Hierarchical")  # per-gene DataFrame

Class hierarchy
---------------
- ``ScribeModelComparisonResults`` — stores raw log-likelihood matrices and
  provides lazy-computed WAIC, PSIS-LOO, stacking, and gene-level methods.

Factory
-------
- ``compare_models()`` — accepts a list of fitted results objects, computes
  log-likelihoods for each model, and returns a
  ``ScribeModelComparisonResults``.

Low-level functions
-------------------
- ``waic()`` / ``compute_waic_stats()`` — JAX-accelerated WAIC.
- ``compute_psis_loo()`` — NumPy/SciPy PSIS-LOO with Pareto fitting.
- ``gene_level_comparison()`` — per-gene elpd differences.
- ``compute_stacking_weights()`` — stacking weight optimization.

See ``paper/_model_comparison.qmd`` for full mathematical derivations.
"""

# Results class and factory
from .results import ScribeModelComparisonResults, compare_models

# WAIC functions
from ._waic import (
    compute_waic_stats,
    waic,
    pseudo_bma_weights,
)

# PSIS-LOO functions
from ._psis_loo import (
    compute_psis_loo,
    psis_loo_summary,
)

# Gene-level comparison
from ._gene_level import (
    gene_level_comparison,
    format_gene_comparison_table,
)

# Model stacking
from ._stacking import (
    compute_stacking_weights,
    stacking_summary,
)

__all__ = [
    # Results class and factory
    "ScribeModelComparisonResults",
    "compare_models",
    # WAIC
    "compute_waic_stats",
    "waic",
    "pseudo_bma_weights",
    # PSIS-LOO
    "compute_psis_loo",
    "psis_loo_summary",
    # Gene-level comparison
    "gene_level_comparison",
    "format_gene_comparison_table",
    # Stacking
    "compute_stacking_weights",
    "stacking_summary",
]
