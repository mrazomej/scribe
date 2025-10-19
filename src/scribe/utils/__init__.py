"""
Utility modules for SCRIBE.

This package contains various utility classes and functions used throughout
the SCRIBE codebase for parameter collection, data processing, and other
common operations.
"""

from .parameter_collector import ParameterCollector
from .core import numpyro_to_scipy, git_root, use_cpu

__all__ = [
    "ParameterCollector",
    "numpyro_to_scipy",
    "git_root",
    "use_cpu",
]
