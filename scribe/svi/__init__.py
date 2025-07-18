"""
Stochastic Variational Inference (SVI) module for single-cell RNA sequencing
data analysis.

This module implements SVI inference for SCRIBE models using Numpyro's SVI.
"""

from .inference_engine import SVIInferenceEngine
from .results_factory import SVIResultsFactory
from .results import ScribeSVIResults

__all__ = [
    "SVIInferenceEngine",
    "SVIResultsFactory",
    "ScribeSVIResults",
]
