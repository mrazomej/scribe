"""
Inference subpackage for SCRIBE.

The inference machinery lives in submodules:

* :mod:`scribe.inference.preset_builder` -- build a ``ModelConfig`` from preset
  arguments (model string, parameterization, inference method).
* :mod:`scribe.inference.inference_config` -- default ``InferenceConfig``
  factories per inference method.
* :mod:`scribe.inference.dispatcher` -- multiple-dispatch routing to the
  SVI / MCMC / Laplace engines.
* :mod:`scribe.inference.svi`, ``.mcmc``, ``.laplace`` -- per-method engines.
* :mod:`scribe.inference.utils` -- data processing and config validation.

.. note::
   The former ``run_scribe`` entry point was removed. Use :func:`scribe.fit`
   (``scribe.api.fit``) -- the higher-level orchestrated entry point that wraps
   the same inference machinery -- instead.
"""

__all__ = []
