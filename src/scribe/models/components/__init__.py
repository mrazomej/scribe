"""
Reusable building blocks for constructing probabilistic models.

This module contains the atomic components used by the builders:

- **likelihoods**: Likelihood functions (NB, ZINB, VCP variants)
- **guide_families**: Variational family implementations (MeanField, LowRank,
  Amortized)
- **amortizers**: Neural network amortizers for variational parameters

All components handle three plate modes:

| Mode | counts | batch_size | Use Case |
|------|--------|------------|----------|
| Prior Predictive | None | - | Generate synthetic data |
| Full Sampling | provided | None | MCMC, small datasets |
| Batch Sampling | provided | specified | SVI on large datasets |

Examples
--------
>>> from scribe.models.components import (
...     NegativeBinomialLikelihood,
...     MeanFieldGuide,
...     LowRankGuide,
...     Amortizer,
...     TOTAL_COUNT,
... )

See Also
--------
scribe.models.builders : Uses components to build complete models.
scribe.models.presets : Pre-configured model combinations.
"""

from .likelihoods import (
    Likelihood,
    NegativeBinomialLikelihood,
    ZeroInflatedNBLikelihood,
    NBWithVCPLikelihood,
    ZINBWithVCPLikelihood,
)
from .guide_families import (
    GuideFamily,
    MeanFieldGuide,
    LowRankGuide,
    AmortizedGuide,
    GroupedAmortizedGuide,
)
from .amortizers import (
    SufficientStatistic,
    TOTAL_COUNT,
    Amortizer,
)

__all__ = [
    # Likelihoods
    "Likelihood",
    "NegativeBinomialLikelihood",
    "ZeroInflatedNBLikelihood",
    "NBWithVCPLikelihood",
    "ZINBWithVCPLikelihood",
    # Guide families
    "GuideFamily",
    "MeanFieldGuide",
    "LowRankGuide",
    "AmortizedGuide",
    "GroupedAmortizedGuide",
    # Amortizers
    "SufficientStatistic",
    "TOTAL_COUNT",
    "Amortizer",
]
