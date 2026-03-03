"""
Sampling mixin compatibility layer for SVI results.

This module keeps the historical public import path
(``from scribe.svi._sampling import SamplingMixin``) stable while composing
focused internal sub-mixins.
"""

from ._sampling_anndata import DenoisedAnnDataMixin
from ._sampling_biological import BiologicalSamplingMixin
from ._sampling_denoising import (
    DenoisingSamplingMixin,
    _denoise_per_dataset,
    _slice_param_for_dataset,
)
from ._sampling_map_predictive import MapPredictiveSamplingMixin
from ._sampling_posterior_predictive import PosteriorPredictiveSamplingMixin


class SamplingMixin(
    PosteriorPredictiveSamplingMixin,
    MapPredictiveSamplingMixin,
    BiologicalSamplingMixin,
    DenoisingSamplingMixin,
    DenoisedAnnDataMixin,
):
    """Public sampling mixin composed from focused internal sub-mixins."""


__all__ = [
    "SamplingMixin",
    "_slice_param_for_dataset",
    "_denoise_per_dataset",
]
