"""NBVCP (Negative Binomial with Variable Capture Probability) presets.

This module provides factory functions for creating NBVCP model and guide
functions with various configurations, including amortized inference for
the cell-specific p_capture parameter.

The NBVCP model assumes:
    p_eff = p * p_capture
    counts ~ NegativeBinomialProbs(r, p_eff)

where:
    - p ∈ (0, 1) is the base success probability (shared across genes)
    - r > 0 is the dispersion parameter (one per gene)
    - p_capture ∈ (0, 1) is the capture probability (one per cell)

Functions
---------
create_nbvcp
    Create NBVCP model and guide functions.

Examples
--------
>>> from scribe.models.presets import create_nbvcp
>>> from scribe.models.config import GuideFamilyConfig
>>> from scribe.models.components import LowRankGuide, AmortizedGuide
>>>
>>> # Standard with mean-field guide (default)
>>> model, guide = create_nbvcp()
>>>
>>> # With amortized p_capture
>>> model, guide = create_nbvcp(
...     guide_families=GuideFamilyConfig(p_capture=AmortizedGuide(amortizer=my_net)),
... )
>>>
>>> # Low-rank mu, amortized p_capture
>>> model, guide = create_nbvcp(
...     parameterization="linked",
...     guide_families=GuideFamilyConfig(
...         mu=LowRankGuide(rank=10),
...         p_capture=AmortizedGuide(amortizer=my_net),
...     ),
... )
"""

from typing import Callable, Dict, List, Optional, Tuple

from ..builders import (
    BetaPrimeSpec,
    BetaSpec,
    GuideBuilder,
    ModelBuilder,
    SigmoidNormalSpec,
)
from ..components import (
    NBWithVCPLikelihood,
)
from ..config import GuideFamilyConfig
from ..parameterizations import PARAMETERIZATIONS


def create_nbvcp(
    parameterization: str = "canonical",
    unconstrained: bool = False,
    guide_families: Optional[GuideFamilyConfig] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    priors: Optional[Dict[str, Tuple[float, ...]]] = None,
    guides: Optional[Dict[str, Tuple[float, ...]]] = None,
) -> Tuple[Callable, Callable]:
    """Create NBVCP model and guide functions.

    This function creates model and guide functions for the Negative Binomial
    with Variable Capture Probability (NBVCP) model. It supports both
    mean-field and amortized inference for the cell-specific p_capture
    parameter.

    Replaces ALL of these files:
    - standard.py: nbvcp_model, nbvcp_guide
    - standard_unconstrained.py: nbvcp_model, nbvcp_guide
    - standard_low_rank.py: nbvcp_guide
    - linked.py, linked_unconstrained.py, etc.
    - odds_ratio.py, odds_ratio_unconstrained.py, etc.

    Parameters
    ----------
    parameterization : str, default="canonical"
        Parameterization scheme:
        - "canonical" (or "standard"): Sample p ~ Beta, r ~ LogNormal directly
        - "mean_prob" (or "linked"): Sample p ~ Beta, mu ~ LogNormal,
                                     derive r = mu*(1-p)/p
        - "mean_odds" (or "odds_ratio"): Sample phi ~ BetaPrime, mu ~ LogNormal,
                                         derive r = mu*phi, p = 1/(1+phi)

        Note: For "mean_odds", p_capture is transformed to phi_capture.
    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration. Unspecified parameters
        default to MeanFieldGuide(). Example:
        ``GuideFamilyConfig(mu=LowRankGuide(rank=10), p_capture=AmortizedGuide(...))``

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
    >>> model, guide = create_nbvcp()
    >>>
    >>> # Low-rank for mu, amortized for p_capture
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide, AmortizedGuide, Amortizer, TOTAL_COUNT
    >>> amortizer = Amortizer(
    ...     sufficient_statistic=TOTAL_COUNT,
    ...     hidden_dims=[128, 64],
    ...     output_params=["log_alpha", "log_beta"],
    ... )
    >>> model, guide = create_nbvcp(
    ...     parameterization="linked",
    ...     guide_families=GuideFamilyConfig(
    ...         mu=LowRankGuide(rank=10),
    ...         p_capture=AmortizedGuide(amortizer=amortizer),
    ...     ),
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
    # Add capture parameter (cell-specific)
    # Parameter name is transformed based on parameterization:
    # - canonical/mean_prob: p_capture
    # - mean_odds: phi_capture
    # ========================================================================
    capture_param_name = param_strategy.transform_model_param("p_capture")
    capture_family = guide_families.get(capture_param_name)

    if unconstrained:
        param_specs.append(
            SigmoidNormalSpec(
                capture_param_name,
                ("n_cells",),
                (0.0, 1.0),
                is_cell_specific=True,
                guide_family=capture_family,
            )
        )
    else:
        # Use BetaPrime for phi_capture (mean_odds), Beta for p_capture (others)
        if capture_param_name == "phi_capture":
            param_specs.append(
                BetaPrimeSpec(
                    capture_param_name,
                    ("n_cells",),
                    (1.0, 1.0),  # Uniform prior on capture odds ratio
                    is_cell_specific=True,
                    guide_family=capture_family,
                )
            )
        else:
            param_specs.append(
                BetaSpec(
                    capture_param_name,
                    ("n_cells",),
                    (1.0, 1.0),  # Uniform prior on capture probability
                    is_cell_specific=True,
                    guide_family=capture_family,
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
    model_builder.with_likelihood(NBWithVCPLikelihood())
    model = model_builder.build()

    # ========================================================================
    # Build guide
    # ========================================================================
    guide = GuideBuilder().from_specs(param_specs).build()

    return model, guide
