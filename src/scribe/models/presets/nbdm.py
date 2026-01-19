"""NBDM (Negative Binomial Dropout Model) presets.

This module provides factory functions for creating NBDM model and guide
functions with various configurations.

The NBDM model assumes:
    counts ~ NegativeBinomialProbs(r, p)

where:
    - p ∈ (0, 1) is the success probability (shared across genes)
    - r > 0 is the dispersion parameter (one per gene)

Functions
---------
create_nbdm
    Create NBDM model and guide functions.

Examples
--------
>>> from scribe.models.presets import create_nbdm
>>> from scribe.models.config import GuideFamilyConfig
>>> from scribe.models.components import LowRankGuide
>>>
>>> # Standard parameterization with mean-field guide (default)
>>> model, guide = create_nbdm()
>>>
>>> # Linked parameterization with low-rank guide for mu
>>> model, guide = create_nbdm(
...     parameterization="linked",
...     guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=10)),
... )
"""

from typing import Callable, List, Optional, Tuple, Dict

from ..builders import (
    GuideBuilder,
    ModelBuilder,
)
from ..components import (
    NegativeBinomialLikelihood,
)
from ..config import GuideFamilyConfig
from ..parameterizations import PARAMETERIZATIONS

# ------------------------------------------------------------------------------
# NBDM Model and Guide Creation
# ------------------------------------------------------------------------------


def create_nbdm(
    parameterization: str = "canonical",
    unconstrained: bool = False,
    guide_families: Optional[GuideFamilyConfig] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Tuple[Callable, Callable]:
    """Create NBDM model and guide functions.

    This function creates model and guide functions for the Negative Binomial
    Dropout Model (NBDM) with configurable parameterization and guide families.

    Parameters
    ----------
    parameterization : str, default="canonical"
        Parameterization scheme:
        - "canonical" (or "standard"): Sample p ~ Beta, r ~ LogNormal directly
        - "mean_prob" (or "linked"): Sample p ~ Beta, mu ~ LogNormal,
                                     derive r = mu*(1-p)/p
        - "mean_odds" (or "odds_ratio"): Sample phi ~ BetaPrime, mu ~ LogNormal,
                                         derive r = mu*phi, p = 1/(1+phi)
    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
        - p sampled as sigmoid(Normal) instead of Beta
        - r/mu sampled as exp(Normal) instead of LogNormal
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration. Unspecified parameters
        default to MeanFieldGuide(). Example:
        ``GuideFamilyConfig(r=LowRankGuide(rank=10))``
    n_components : Optional[int], default=None
        Number of mixture components. If provided, creates a mixture model
        where parameters in mixture_params are component-specific.
        If None, creates a single-component model.
    mixture_params : Optional[List[str]], default=None
        List of parameter names to make mixture-specific. Only used if
        n_components is provided.
        - If None and n_components is set, defaults to all gene-specific
          parameters (e.g., ["r"] for canonical, ["mu"] for mean_prob/mean_odds).
        - Example: ["r"] makes r component-specific while p is shared.

    Returns
    -------
    model : Callable
        NumPyro model function with signature:
        model(n_cells, n_genes, model_config, counts=None, batch_size=None)
    guide : Callable
        NumPyro guide function with the same signature.

    Raises
    ------
    ValueError
        If parameterization is not recognized.

    Examples
    --------
    >>> # Standard mean-field (all defaults)
    >>> model, guide = create_nbdm()
    >>>
    >>> # Linked with low-rank for mu
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide
    >>> model, guide = create_nbdm(
    ...     parameterization="linked",
    ...     guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=15)),
    ... )
    >>>
    >>> # Unconstrained odds-ratio
    >>> model, guide = create_nbdm(
    ...     parameterization="odds_ratio",
    ...     unconstrained=True,
    ... )
    """
    # ========================================================================
    # Validate inputs and get parameterization strategy
    # ========================================================================
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )

    param_strategy = PARAMETERIZATIONS[parameterization]

    # ========================================================================
    # Resolve guide families from config (defaults to MeanFieldGuide)
    # ========================================================================
    if guide_families is None:
        guide_families = GuideFamilyConfig()

    # ========================================================================
    # Build parameter specs using parameterization strategy
    # ========================================================================
    param_specs = param_strategy.build_param_specs(
        unconstrained,
        guide_families,
        n_components=n_components,
        mixture_params=mixture_params,
    )

    # Update param_specs with prior/guide values if provided
    if priors is not None or guides is not None:
        updated_specs = []
        for spec in param_specs:
            updates = {}
            if priors is not None and spec.name in priors:
                updates["prior"] = priors[spec.name]
            if guides is not None and spec.name in guides:
                updates["guide"] = guides[spec.name]
            if updates:
                updated_spec = spec.model_copy(update=updates)
                updated_specs.append(updated_spec)
            else:
                updated_specs.append(spec)
        param_specs = updated_specs

    derived_params = param_strategy.build_derived_params()

    # ========================================================================
    # Build model
    # ========================================================================
    model_builder = ModelBuilder()
    for spec in param_specs:
        model_builder.add_param(spec)
    for d_param in derived_params:
        model_builder.add_derived(d_param.name, d_param.compute, d_param.deps)
    model_builder.with_likelihood(NegativeBinomialLikelihood())
    model = model_builder.build()

    # ========================================================================
    # Build guide (uses guide_family from each spec)
    # ========================================================================
    guide = GuideBuilder().from_specs(param_specs).build()

    return model, guide
