"""
Simplified API for SCRIBE inference.

This package provides a user-friendly entry point for SCRIBE inference
with sensible defaults and flat kwargs instead of nested configuration
objects.

The public surface is intentionally kept identical to the original
monolithic ``scribe.api`` module so that all downstream imports and
``unittest.mock.patch`` targets continue to resolve correctly.

Re-exported symbols
-------------------
fit : callable
    Main entry point for SCRIBE inference.
VALID_MODELS, VALID_PARAMETERIZATIONS, VALID_INFERENCE_METHODS : set
    Validation sets for user-facing selectors.
ScribeResults : type alias
    Union of all result types that ``fit`` may return.
_run_inference : callable
    Inference dispatcher (re-exported for monkeypatch compatibility;
    14 test sites patch ``scribe.api._run_inference``).
build_config_from_preset : callable
    Preset-based model config builder (re-exported for import compat).
process_counts_data : callable
    Count data processor (re-exported for import compat).
"""

# -- Canonical exports (used by scribe.__init__ and user code) ----------------
from .fit import fit
from .constants import (
    VALID_MODELS,
    VALID_PARAMETERIZATIONS,
    VALID_INFERENCE_METHODS,
)
from .types import ScribeResults

# -- Patch-target re-exports --------------------------------------------------
# Tests monkeypatch these via ``patch("scribe.api._run_inference")`` etc.
# The binding must live on *this* module object so ``patch`` can find it.
from ..inference.dispatcher import _run_inference
from ..inference.preset_builder import build_config_from_preset
from ..inference.utils import process_counts_data

__all__ = [
    "fit",
    "VALID_MODELS",
    "VALID_PARAMETERIZATIONS",
    "VALID_INFERENCE_METHODS",
    "ScribeResults",
    "_run_inference",
    "build_config_from_preset",
    "process_counts_data",
]
