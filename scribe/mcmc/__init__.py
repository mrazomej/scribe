"""
Markov Chain Monte Carlo (MCMC) module for single-cell RNA sequencing
data analysis.

This module implements MCMC inference for SCRIBE models using Numpyro's NUTS.
"""

from .mcmc import run_scribe, run_inference
from .inference_engine import MCMCInferenceEngine
from .results_factory import MCMCResultsFactory

__all__ = [
    "run_scribe",
    "run_inference", 
    "MCMCInferenceEngine",
    "MCMCResultsFactory",
] 