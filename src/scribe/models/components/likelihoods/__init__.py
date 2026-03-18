"""Likelihood components for count data models.

This module provides likelihood classes that handle the three plate modes:

- **Prior predictive** (counts=None): Sample synthetic counts from the prior
- **Full sampling** (counts provided, no batching): Condition on all cells
- **Batch sampling** (counts provided, with batch_size): Subsample for SVI

Each likelihood handles cell-specific parameter sampling and observation
sampling within the cell plate.

NB-based likelihoods live in ``negative_binomial.py``, ``zero_inflated.py``,
and ``vcp.py``.  Their BNB counterparts (which override only the base
distribution) live in ``beta_negative_binomial.py``.

Classes
-------
Likelihood
    Abstract base class for likelihood components.
NegativeBinomialLikelihood
    Standard Negative Binomial likelihood for count data.
ZeroInflatedNBLikelihood
    Zero-Inflated Negative Binomial likelihood.
NBWithVCPLikelihood
    Negative Binomial with Variable Capture Probability.
ZINBWithVCPLikelihood
    Zero-Inflated NB with Variable Capture Probability.
BetaNegativeBinomialLikelihood
    BNB counterpart of NegativeBinomialLikelihood.
ZeroInflatedBNBLikelihood
    BNB counterpart of ZeroInflatedNBLikelihood.
BNBWithVCPLikelihood
    BNB counterpart of NBWithVCPLikelihood.
ZIBNBWithVCPLikelihood
    BNB counterpart of ZINBWithVCPLikelihood.

Examples
--------
>>> from scribe.models.components import NegativeBinomialLikelihood
>>> likelihood = NegativeBinomialLikelihood()
>>> # Use in ModelBuilder
>>> builder.with_likelihood(likelihood)

See Also
--------
scribe.models.builders.model_builder : Uses likelihoods to build models.
"""

from .base import Likelihood, compute_cell_specific_mixing
from .negative_binomial import NegativeBinomialLikelihood
from .zero_inflated import ZeroInflatedNBLikelihood
from .vcp import NBWithVCPLikelihood, ZINBWithVCPLikelihood
from .beta_negative_binomial import (
    BetaNegativeBinomialLikelihood,
    ZeroInflatedBNBLikelihood,
    BNBWithVCPLikelihood,
    ZIBNBWithVCPLikelihood,
)

__all__ = [
    "Likelihood",
    "compute_cell_specific_mixing",
    # NB family
    "NegativeBinomialLikelihood",
    "ZeroInflatedNBLikelihood",
    "NBWithVCPLikelihood",
    "ZINBWithVCPLikelihood",
    # BNB family
    "BetaNegativeBinomialLikelihood",
    "ZeroInflatedBNBLikelihood",
    "BNBWithVCPLikelihood",
    "ZIBNBWithVCPLikelihood",
]
