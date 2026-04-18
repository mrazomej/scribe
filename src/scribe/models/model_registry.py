"""Model registry for SCRIBE models.

This module provides functions for retrieving model and guide functions based on
the model type.

The primary API is ``get_model_and_guide()`` which uses the composable builder
system via preset factories. This provides:

- Per-parameter guide families (mean-field, low-rank, amortized, flow, VAE)
- Flexible configuration via GuideFamilyConfig
- Clean, composable architecture

Examples
--------
>>> from scribe.models.config import GuideFamilyConfig
>>> from scribe.models.components import LowRankGuide, AmortizedGuide
>>>
>>> # Simple usage (all mean-field)
>>> model, guide = get_model_and_guide("nbdm")
>>>
>>> # With per-parameter guide families
>>> model, guide = get_model_and_guide(
...     "nbvcp",
...     parameterization="linked",
...     guide_families=GuideFamilyConfig(
...         mu=LowRankGuide(rank=15),
...         p_capture=AmortizedGuide(amortizer=my_amortizer),
...     ),
... )
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple

if TYPE_CHECKING:
    from .config import ModelConfig, GuideFamilyConfig


# ------------------------------------------------------------------------------
# Model log likelihood functions
# ------------------------------------------------------------------------------


# Mapping from ``model_type`` strings to the concrete :class:`Likelihood`
# subclass whose ``.log_prob`` implements its count distribution.  Mixture
# variants (``<base>_mix``) share the same class as their non-mixture
# counterpart because ``Likelihood.log_prob`` dispatches on the presence of
# ``"mixing_weights"`` in the params dictionary.
#
# The BNB family is not represented here as a separate ``model_type`` string:
# BNB is encoded via ``OverdispersionType.BNB`` in :class:`ModelConfig` and
# activated at inference time by placing ``"bnb_concentration"`` into the
# parameter dictionary.  The same NB ``.log_prob`` routes to a Beta-NB
# distribution inside :func:`_build_ll_count_dist` whenever that key is
# present, so the four entries below cover all eight NB / BNB variants.
_LIKELIHOOD_CLASS_BY_MODEL_TYPE: dict = {}


def _likelihood_class_for(model_type: str):
    """Look up the :class:`Likelihood` subclass for a ``model_type`` string.

    The lookup table is populated lazily on first call to keep
    ``scribe.models`` importable without eagerly loading every likelihood
    module (which in turn imports numpyro distributions and JAX).

    Parameters
    ----------
    model_type : str
        Model-type key (e.g. ``"nbdm"``, ``"zinb_mix"``).

    Returns
    -------
    type
        Concrete :class:`Likelihood` subclass whose ``.log_prob`` method
        evaluates the model's count distribution.

    Raises
    ------
    ValueError
        If ``model_type`` is not one of the known NB / BNB variants.
    """
    global _LIKELIHOOD_CLASS_BY_MODEL_TYPE
    if not _LIKELIHOOD_CLASS_BY_MODEL_TYPE:
        # Defer the import until first use to avoid triggering the full
        # likelihood dependency chain at package import time.
        from .components.likelihoods import (
            NegativeBinomialLikelihood,
            ZeroInflatedNBLikelihood,
            NBWithVCPLikelihood,
            ZINBWithVCPLikelihood,
        )

        _LIKELIHOOD_CLASS_BY_MODEL_TYPE = {
            # Non-mixture NB / BNB variants.
            "nbdm": NegativeBinomialLikelihood,
            "zinb": ZeroInflatedNBLikelihood,
            "nbvcp": NBWithVCPLikelihood,
            "zinbvcp": ZINBWithVCPLikelihood,
            # Mixture variants reuse the same classes (dispatch on
            # "mixing_weights" at log_prob time).
            "nbdm_mix": NegativeBinomialLikelihood,
            "zinb_mix": ZeroInflatedNBLikelihood,
            "nbvcp_mix": NBWithVCPLikelihood,
            "zinbvcp_mix": ZINBWithVCPLikelihood,
        }

    cls = _LIKELIHOOD_CLASS_BY_MODEL_TYPE.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Expected one of "
            f"{sorted(_LIKELIHOOD_CLASS_BY_MODEL_TYPE.keys())}."
        )
    return cls


def get_log_likelihood_fn(model_type: str) -> Callable:
    """Return a bound ``Likelihood.log_prob`` method for *model_type*.

    This is the public dispatch entry point used by the SVI and MCMC
    post-inference likelihood mixins.  The returned callable has the
    signature

        fn(counts, params, *, return_by, cells_axis, split_components,
           weights, weight_type, r_floor, p_floor, dtype)

    matching :meth:`Likelihood.log_prob`.  The legacy ``batch_size``
    argument is gone; use ``sample_chunk_size`` (for posterior-sample
    memory) or ``gene_batch_size`` (for gene-wise evaluation) instead.

    Parameters
    ----------
    model_type : str
        Model-type key.  One of
        ``{"nbdm", "zinb", "nbvcp", "zinbvcp",
           "nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"}``.

        BNB-family models reuse the NB-family ``model_type`` strings; the
        BNB base distribution is activated by the presence of
        ``"bnb_concentration"`` in the ``params`` dictionary passed to the
        returned callable (see :func:`_build_ll_count_dist`).

    Returns
    -------
    Callable
        Bound ``.log_prob`` method of a fresh instance of the matching
        :class:`Likelihood` subclass.

    Raises
    ------
    ValueError
        If ``model_type`` is not recognised.
    """
    # Instantiate once per call; the bound method keeps a reference to the
    # instance alive.  ``Likelihood`` subclasses are effectively stateless
    # so the instantiation cost is negligible.
    likelihood_cls = _likelihood_class_for(model_type)
    return likelihood_cls().log_prob


# ------------------------------------------------------------------------------
# Main API: Composable builder system
# ------------------------------------------------------------------------------


def get_model_and_guide(
    model_config: "ModelConfig",
    unconstrained: Optional[bool] = None,
    guide_families: Optional["GuideFamilyConfig"] = None,
    n_genes: Optional[int] = None,
) -> Tuple[Callable, Callable, "ModelConfig"]:
    """Create model and guide functions using the unified factory.

    This is the primary API for getting model/guide functions. It uses the
    unified `create_model()` factory which provides a single entry point for
    all model types, eliminating code duplication.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration containing all model parameters:
            - base_model: Type of model ("nbdm", "zinb", "nbvcp", "zinbvcp")
            - parameterization: Parameterization scheme
            - unconstrained: Whether to use unconstrained parameterization
            - guide_families: Per-parameter guide family configuration
            - n_components: Number of mixture components (if mixture model)
            - mixture_params: List of mixture-specific parameter names
            - param_specs: Optional user-provided prior/guide overrides
    unconstrained : bool, optional
        Override the unconstrained setting from model_config. If None, uses
        model_config.unconstrained. Useful for special cases like predictive
        sampling where a constrained model is needed.
    guide_families : GuideFamilyConfig, optional
        Override the guide families from model_config. If None, uses
        model_config.guide_families. Useful for special cases where the guide
        is not needed (e.g., predictive sampling).

    Returns
    -------
    model : Callable
        NumPyro model function.
    guide : Callable
        NumPyro guide function.
    model_config_for_results : ModelConfig
        Config with param_specs set (for use when constructing results).
        Use this when creating ScribeSVIResults so subsetting has metadata.

    Raises
    ------
    ValueError
        If model_config.base_model is not recognized.

    Examples
    --------
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import LowRankGuide, AmortizedGuide
    >>> from scribe.inference.preset_builder import build_config_from_preset
    >>>
    >>> # Basic NBDM (all mean-field)
    >>> model_config = build_config_from_preset("nbdm")
    >>> model, guide, model_config_for_results = get_model_and_guide(model_config)
    >>>
    >>> # NBVCP with low-rank for mu and amortized p_capture
    >>> model_config = build_config_from_preset(
    ...     "nbvcp",
    ...     parameterization="linked",
    ...     guide_families=GuideFamilyConfig(
    ...         mu=LowRankGuide(rank=15),
    ...         p_capture=AmortizedGuide(amortizer=my_amortizer),
    ...     ),
    ... )
    >>> model, guide, model_config_for_results = get_model_and_guide(model_config)

    See Also
    --------
    scribe.models.presets.factory.create_model : Unified model factory.
    scribe.inference.preset_builder : Build ModelConfig from presets.
    GuideFamilyConfig : Per-parameter guide family configuration.
    """
    # Import the unified factory
    from .presets.factory import create_model

    # Handle overrides by creating a modified config if needed
    if unconstrained is not None or guide_families is not None:
        # Create a modified config with overrides
        updates = {}
        if unconstrained is not None:
            updates["unconstrained"] = unconstrained
        if guide_families is not None:
            updates["guide_families"] = guide_families

        # Use model_copy for immutable update
        effective_config = model_config.model_copy(update=updates)
    else:
        effective_config = model_config

    # Use the unified factory (n_genes required for VAE)
    model, guide, param_specs = create_model(
        effective_config, n_genes=n_genes
    )
    model_config_for_results = (
        effective_config.model_copy(update={"param_specs": param_specs})
        if param_specs
        else effective_config
    )
    return model, guide, model_config_for_results
