"""
Laplace-approximation inference for SCRIBE.

This submodule houses everything specific to
``inference_method="laplace"``:

* :class:`LaplaceInferenceEngine` — outer SVI on globals + inner
  Newton on per-cell latents (variational EM in the spirit of
  ``PLNmodels``).
* :class:`ScribeLaplaceResults` — single results class shared
  across all Laplace-supported models. Methods that depend on the
  generative model dispatch internally on
  ``model_config.base_model``.
* Orbax checkpointing helpers for resumable training.
* Pure-JAX Newton kernels per model (currently PLN; LNM in
  progress, see ``~/.claude/plans/lnm-laplace-and-laplace-submodule.md``).

The submodule deliberately does **not** depend on
``scribe.svi.inference_engine`` or NumPyro's SVI machinery — the
Laplace path is a parallel inference mode with its own enum
(``InferenceMethod.LAPLACE``), engine, results class, and
checkpoint format. The only cross-submodule dependency is
``scribe.svi._progress_backend`` (shared progress-bar
infrastructure used by both inference modes).

Public API
----------
The :mod:`scribe.laplace` package re-exports the canonical surface:

>>> from scribe.laplace import (
...     LaplaceInferenceEngine,
...     LaplaceRunResult,
...     ScribeLaplaceResults,
...     save_laplace_checkpoint,
...     load_laplace_checkpoint,
...     laplace_checkpoint_exists,
...     remove_laplace_checkpoint,
... )

Internal helpers (Newton kernels, ELBO components) live under
underscore-prefixed module names and are not part of the public
API.
"""

from .engine import LaplaceInferenceEngine, LaplaceRunResult
from .results import ScribeLaplaceResults
from .checkpoint import (
    LaplaceCheckpointMetadata,
    laplace_checkpoint_exists,
    load_laplace_checkpoint,
    remove_laplace_checkpoint,
    save_laplace_checkpoint,
)

__all__ = [
    # Engine + run-result container
    "LaplaceInferenceEngine",
    "LaplaceRunResult",
    # Results class (model-agnostic; dispatches on base_model)
    "ScribeLaplaceResults",
    # Orbax checkpoint helpers
    "LaplaceCheckpointMetadata",
    "laplace_checkpoint_exists",
    "save_laplace_checkpoint",
    "load_laplace_checkpoint",
    "remove_laplace_checkpoint",
]
