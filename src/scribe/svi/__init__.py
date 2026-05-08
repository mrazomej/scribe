"""
Stochastic Variational Inference (SVI) module for single-cell RNA sequencing
data analysis.

This module implements SVI inference for SCRIBE models using Numpyro's SVI.
Supports early stopping based on loss convergence and Orbax checkpointing.

Laplace-mode inference (``inference_method="laplace"``) lives in its own
submodule at :mod:`scribe.laplace` — it does not use NumPyro's SVI
machinery and so was deliberately separated.
"""

from .inference_engine import SVIInferenceEngine, SVIRunResult
from .results_factory import SVIResultsFactory
from .results import ScribeSVIResults
from .vae_results import ScribeVAEResults
from .variational_results_base import ScribeVariationalResults
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
    "ScribeVariationalResults",
    "ScribeSVIResults",
    "ScribeVAEResults",
    # SVI checkpoint utilities
    "checkpoint_exists",
    "save_svi_checkpoint",
    "load_svi_checkpoint",
    "remove_checkpoint",
    "CheckpointMetadata",
]
