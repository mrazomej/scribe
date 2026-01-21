"""
Stochastic Variational Inference (SVI) module for single-cell RNA sequencing
data analysis.

This module implements SVI inference for SCRIBE models using Numpyro's SVI.
Supports early stopping based on loss convergence and Orbax checkpointing.
"""

from .inference_engine import SVIInferenceEngine, SVIRunResult
from .results_factory import SVIResultsFactory
from .results import ScribeSVIResults
from .checkpoint import (
    checkpoint_exists,
    save_svi_checkpoint,
    load_svi_checkpoint,
    remove_checkpoint,
    CheckpointMetadata,
)

__all__ = [
    "SVIInferenceEngine",
    "SVIRunResult",
    "SVIResultsFactory",
    "ScribeSVIResults",
    # Checkpoint utilities
    "checkpoint_exists",
    "save_svi_checkpoint",
    "load_svi_checkpoint",
    "remove_checkpoint",
    "CheckpointMetadata",
]
