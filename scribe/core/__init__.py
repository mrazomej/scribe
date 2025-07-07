"""
Core shared components for SCRIBE inference.

This module contains shared functionality used by both SVI and MCMC inference methods.
"""

from .input_processor import InputProcessor
from .prior_factory import PriorConfigFactory
from .config_factory import ModelConfigFactory
from .normalization import normalize_counts_from_posterior

__all__ = [
    "InputProcessor",
    "PriorConfigFactory",
    "ModelConfigFactory",
    "normalize_counts_from_posterior",
]
