"""Pre-configured model/guide factories for common use cases.

This module provides factory functions that create ready-to-use model and
guide functions for the common model types in SCRIBE. These presets hide
the complexity of the builder pattern while still allowing customization.

Available Presets
-----------------
create_nbdm
    Negative Binomial Dropout Model
create_zinb
    Zero-Inflated Negative Binomial
create_nbvcp
    NB with Variable Capture Probability
create_zinbvcp
    ZINB with Variable Capture Probability

Configuration Options
---------------------
All presets support:

- **parameterization**: "standard", "linked", "odds_ratio"
- **unconstrained**: Use Normal+transform instead of constrained distributions
- **r_guide** / **mu_guide**: "mean_field" or "low_rank" for gene-specific params
- **guide_rank**: Rank for low-rank guide

VCP presets additionally support:

- **p_capture_guide**: "mean_field" or "amortized"
- **capture_amortizer**: Custom Amortizer instance

Examples
--------
>>> from scribe.models.presets import create_nbdm, create_nbvcp
>>>
>>> # Basic usage
>>> model, guide = create_nbdm()
>>>
>>> # With low-rank guide
>>> model, guide = create_nbdm(
...     parameterization="linked",
...     r_guide="low_rank",
...     guide_rank=15,
... )
>>>
>>> # With amortized p_capture
>>> model, guide = create_nbvcp(
...     p_capture_guide="amortized",
... )

See Also
--------
scribe.models.builders : Low-level building blocks.
scribe.models.components : Reusable components.
"""

from .nbdm import create_nbdm
from .zinb import create_zinb
from .nbvcp import create_nbvcp
from .zinbvcp import create_zinbvcp

__all__ = [
    "create_nbdm",
    "create_zinb",
    "create_nbvcp",
    "create_zinbvcp",
]
