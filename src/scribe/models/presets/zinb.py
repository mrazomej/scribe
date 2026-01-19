"""ZINB (Zero-Inflated Negative Binomial) presets.

This module provides factory functions for creating ZINB model and guide
functions with various configurations.

The ZINB model assumes:
    counts ~ ZeroInflatedNegativeBinomial(gate, r, p)

where:
    - gate ∈ (0, 1) is the zero-inflation probability (one per gene)
    - p ∈ (0, 1) is the success probability (shared across genes)
    - r > 0 is the dispersion parameter (one per gene)

Functions
---------
create_zinb
    Create ZINB model and guide functions.

Examples
--------
>>> from scribe.models.presets import create_zinb
>>> from scribe.models.config import GuideFamilyConfig
>>> from scribe.models.components import LowRankGuide
>>>
>>> # Standard parameterization with mean-field guide (default)
>>> model, guide = create_zinb()
>>>
>>> # Linked parameterization with low-rank guide for mu
>>> model, guide = create_zinb(
...     parameterization="linked",
...     guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=10)),
... )
"""

from typing import Callable, Dict, List, Optional, Tuple

from ..builders import (
    BetaSpec,
    GuideBuilder,
    ModelBuilder,
    SigmoidNormalSpec,
)
from ..components import (
    ZeroInflatedNBLikelihood,
)
from ..config import GuideFamilyConfig
from ..parameterizations import PARAMETERIZATIONS


def create_zinb(
    parameterization: str = "canonical",
    unconstrained: bool = False,
    guide_families: Optional[GuideFamilyConfig] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Tuple[Callable, Callable]:
    """Create ZINB model and guide functions.

    This function creates model and guide functions for the Zero-Inflated
    Negative Binomial (ZINB) model with configurable parameterization and
    guide families.

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
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration. Unspecified parameters
        default to MeanFieldGuide(). Example:
        ``GuideFamilyConfig(r=LowRankGuide(rank=10), gate=MeanFieldGuide())``

    Returns
    -------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.

    Raises
    ------
    ValueError
        If parameterization is not recognized.

    Examples
    --------
    >>> # Standard mean-field (all defaults)
    >>> model, guide = create_zinb()
    >>>
    >>> # Linked with low-rank for mu
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide
    >>> model, guide = create_zinb(
    ...     parameterization="linked",
    ...     guide_families=GuideFamilyConfig(mu=LowRankGuide(rank=15)),
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

    gate_family = guide_families.get("gate")

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
    # Add gate parameter (model-specific, not parameterization-specific)
    # Gate is always gene-specific (one zero-inflation prob per gene)
    # ========================================================================
    # Determine if gate should be mixture-specific
    is_gate_mixture = False
    if n_components is not None:
        if mixture_params is None:
            # Default: make all gene-specific params mixture-specific
            is_gate_mixture = True
        else:
            is_gate_mixture = "gate" in mixture_params

    if unconstrained:
        param_specs.append(
            SigmoidNormalSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(-2.0, 1.0),  # Default to low zero-inflation
                is_gene_specific=True,
                guide_family=gate_family,
                is_mixture=is_gate_mixture,
            )
        )
    else:
        param_specs.append(
            BetaSpec(
                name="gate",
                shape_dims=("n_genes",),
                default_params=(1.0, 9.0),  # Default to low zero-inflation (mean ~0.1)
                is_gene_specific=True,
                guide_family=gate_family,
                is_mixture=is_gate_mixture,
            )
        )

    # ========================================================================
    # Build model
    # ========================================================================
    model_builder = ModelBuilder()
    for spec in param_specs:
        model_builder.add_param(spec)
    for d_param in derived_params:
        model_builder.add_derived(d_param.name, d_param.compute, d_param.deps)
    model_builder.with_likelihood(ZeroInflatedNBLikelihood())
    model = model_builder.build()

    # ========================================================================
    # Build guide
    # ========================================================================
    guide = GuideBuilder().from_specs(param_specs).build()

    return model, guide
