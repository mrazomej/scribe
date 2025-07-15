"""
Core shared components for SCRIBE inference.

This module contains shared functionality used by both SVI and MCMC inference methods.
"""

from .input_processor import InputProcessor
from .normalization import normalize_counts_from_posterior

__all__ = [
    "InputProcessor",
    "Normalization",
]
