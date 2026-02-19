"""
Composable builders for constructing NumPyro models and guides.

This module provides a builder pattern for constructing NumPyro model and guide
functions from reusable parameter specifications. It handles the complexity of
plate management, batch sampling, and derived parameter computation using
multiple dispatch.

Architecture
------------
The building process follows this flow:

    ParamSpec (defines WHAT to sample)
        │
        ▼
    ModelBuilder / GuideBuilder (composes HOW to sample)
        │
        ▼
    model_fn / guide_fn (NumPyro callables)

Parameter Specifications
------------------------
Parameter specs define the distribution and metadata for each parameter:

| Spec Type | Distribution | Constraint | Example Params |
|-----------|--------------|------------|----------------|
| BetaSpec | Beta(α, β) | (0, 1) | p, gate, p_capture |
| LogNormalSpec | LogNormal(μ, σ) | (0, ∞) | r, mu |
| BetaPrimeSpec | BetaPrime(α, β) | (0, ∞) | phi |
| SigmoidNormalSpec | Normal → sigmoid | (0, 1) | p_unconstrained |
| ExpNormalSpec | Normal → exp | (0, ∞) | r_unconstrained |

Examples
--------
>>> from scribe.models.builders import ModelBuilder, BetaSpec, LogNormalSpec
>>> from scribe.models.components import NegativeBinomialLikelihood
>>>
>>> model = (ModelBuilder()
...     .add_param(BetaSpec("p", (), (1.0, 1.0)))
...     .add_param(LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
...     .with_likelihood(NegativeBinomialLikelihood())
...     .build())

See Also
--------
scribe.models.components : Reusable building blocks.
scribe.models.presets : Pre-configured model combinations.
"""

from .parameter_specs import (
    # Base class
    ParamSpec,
    # Constrained specs
    BetaSpec,
    LogNormalSpec,
    BetaPrimeSpec,
    DirichletSpec,
    # Unconstrained specs
    NormalWithTransformSpec,
    SigmoidNormalSpec,
    ExpNormalSpec,
    SoftplusNormalSpec,
    # Hierarchical specs (gene-specific with learned hyperprior)
    HierarchicalNormalWithTransformSpec,
    HierarchicalSigmoidNormalSpec,
    HierarchicalExpNormalSpec,
    # Latent specs (VAE z)
    LatentSpec,
    GaussianLatentSpec,
    # Derived parameters
    DerivedParam,
    # Functions
    sample_prior,
    resolve_shape,
)
from .model_builder import (
    ModelBuilder,
)
from .guide_builder import (
    GuideBuilder,
    setup_guide,
    setup_cell_specific_guide,
)
from .posterior import get_posterior_distributions

__all__ = [
    # Parameter specs
    "ParamSpec",
    "BetaSpec",
    "LogNormalSpec",
    "BetaPrimeSpec",
    "DirichletSpec",
    "NormalWithTransformSpec",
    "SigmoidNormalSpec",
    "ExpNormalSpec",
    "SoftplusNormalSpec",
    # Hierarchical specs
    "HierarchicalNormalWithTransformSpec",
    "HierarchicalSigmoidNormalSpec",
    "HierarchicalExpNormalSpec",
    "LatentSpec",
    "GaussianLatentSpec",
    # Spec functions
    "sample_prior",
    "resolve_shape",
    # Builders
    "DerivedParam",
    "ModelBuilder",
    "GuideBuilder",
    # Guide functions
    "setup_guide",
    "setup_cell_specific_guide",
    # Posterior extraction
    "get_posterior_distributions",
]
