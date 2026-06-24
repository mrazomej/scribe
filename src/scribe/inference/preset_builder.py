"""
Preset-based model configuration builder.

This module provides functionality to build ModelConfig objects from simple
preset parameters (model name, parameterization, etc.). The actual model
and guide construction is handled by the unified factory in
`scribe.models.presets.factory`.

Functions
---------
build_config_from_preset
    Build a ModelConfig from simple parameters.

Examples
--------
>>> from scribe.inference.preset_builder import build_config_from_preset
>>>
>>> # Basic usage
>>> config = build_config_from_preset(model="nbdm")
>>>
>>> # With options
>>> config = build_config_from_preset(
...     model="zinb",
...     parameterization="linked",
...     n_components=3,
...     guide_rank=15,
... )

See Also
--------
scribe.models.presets.factory.create_model : Creates model/guide from config.
scribe.models.config.ModelConfigBuilder : Builder for ModelConfig objects.
"""

from typing import Any, Dict, List, Optional, Union

from ..models.config import (
    AmortizationConfig,
    GuideFamilyConfig,
    ModelConfig,
    ModelConfigBuilder,
)
from ..models.config.parameter_mapping import resolve_param_shorthand
from ..models.parameterizations import PARAMETERIZATIONS


# ==============================================================================
# Build ModelConfig from preset parameters
# ==============================================================================


def build_config_from_preset(
    model: str,
    parameterization: str = "canonical",
    inference_method: str = "svi",
    unconstrained: bool = False,
    # ``None`` triggers a model-aware default: ``{"mu": "exp"}`` for
    # the TwoState family (gene mean on multiplicative SVI geometry,
    # other positive parameters keep softplus), ``"softplus"``
    # otherwise.  Any explicit value (string or dict) is honoured.
    positive_transform: Optional[Union[str, Dict[str, str]]] = None,
    expression_prior: str = "none",
    prob_prior: str = "none",
    zero_inflation_prior: str = "none",
    n_datasets: Optional[int] = None,
    grouping_spec: Optional["GroupingSpec"] = None,
    dataset_params: Optional[List[str]] = None,
    dataset_mixing: Optional[bool] = None,
    expression_dataset_prior: str = "none",
    prob_dataset_prior: str = "none",
    prob_dataset_mode: str = "gene_specific",
    zero_inflation_dataset_prior: str = "none",
    overdispersion_dataset_prior: str = "none",
    regime_dataset_prior: str = "none",
    regime_dataset_target: Optional[str] = None,
    overdispersion_dataset_independent: bool = True,
    horseshoe_tau0: float = 1.0,
    horseshoe_slab_df: int = 4,
    horseshoe_slab_scale: float = 2.0,
    neg_u: float = 1.0,
    neg_a: float = 1.0,
    neg_tau: float = 1.0,
    capture_scaling_prior: str = "none",
    expression_anchor: bool = False,
    expression_anchor_sigma: float = 0.3,
    overdispersion: str = "none",
    overdispersion_prior: str = "horseshoe",
    # LNM diagonal mode (``lnm`` / ``lnmvcp`` only; see ``ModelConfig.d_mode``)
    d_mode: str = "low_rank",
    alr_reference_idx: int = -1,
    # Gauss-Legendre node count for the two-state Poisson-Beta
    # likelihood. ``None`` keeps the PoissonBetaCompound default (60).
    n_quad_nodes: Optional[int] = None,
    # Whether the trailing aggregated '_other' column participates in
    # the latent low-rank covariance.  Current default ``True`` is
    # legacy (held at True for the Commit 2 release; flips to ``False``
    # when Commit 2b lands the decoupled-math path).  See
    # ``scribe.models.config.base.ModelConfig.correlate_other_column``
    # for the per-model wiring status and the default-flip schedule.
    correlate_other_column: bool = True,
    guide_rank: Optional[int] = None,
    joint_params: Optional[Union[str, List[str]]] = None,
    dense_params: Optional[Union[str, List[str]]] = None,
    # Flow-based guide (mutually exclusive with guide_rank)
    guide_flow: Optional[str] = None,
    guide_flow_num_layers: int = 4,
    guide_flow_hidden_dims: Optional[List[int]] = None,
    guide_flow_activation: str = "relu",
    guide_flow_n_bins: int = 8,
    guide_flow_mixture_strategy: str = "independent",
    guide_flow_zero_init: bool = True,
    guide_flow_layer_norm: bool = True,
    guide_flow_residual: bool = True,
    guide_flow_soft_clamp: bool = True,
    guide_flow_loft: bool = True,
    guide_flow_log_det_f64: bool = False,
    n_components: Optional[int] = None,
    mixture_params: Optional[Union[str, List[str]]] = "all",
    priors: Optional[Dict[str, Any]] = None,
    # VAE architecture options (when inference_method="vae")
    vae_latent_dim: int = 10,
    vae_encoder_hidden_dims: Optional[List[int]] = None,
    vae_decoder_hidden_dims: Optional[List[int]] = None,
    vae_activation: Optional[str] = None,
    vae_input_transform: str = "log1p",
    # ``None`` (the default) is a sentinel meaning "auto-pick based on
    # model": ``True`` for ``lnm`` / ``lnmvcp`` and ``False`` everywhere
    # else. Pass an explicit boolean to opt in or out unconditionally.
    vae_standardize: Optional[bool] = None,
    vae_decoder_transforms: Optional[Dict[str, str]] = None,
    vae_flow_type: str = "none",
    vae_flow_num_layers: int = 4,
    vae_flow_hidden_dims: Optional[List[int]] = None,
    # Amortization options
    amortize_capture: bool = False,
    capture_hidden_dims: Optional[List[int]] = None,
    capture_activation: str = "leaky_relu",
    capture_output_transform: str = "softplus",
    capture_clamp_min: Optional[float] = 0.1,
    capture_clamp_max: Optional[float] = 50.0,
    capture_amortization: Optional[
        Union[AmortizationConfig, Dict[str, Any]]
    ] = None,
) -> ModelConfig:
    """Build ModelConfig from simple preset parameters.

    This function creates a ModelConfig by translating simple parameters
    into the appropriate configuration. It handles:

    - Model type validation
    - Parameterization validation and aliasing
    - Guide family configuration from guide_rank or guide_flow
    - Mixture model configuration

    The actual model/guide construction is done by `create_model()` in
    the factory module.

    Parameters
    ----------
    model : str
        Model type: ``"nbdm"``, ``"lnm"``, ``"lnmvcp"``, ``"zinb"``,
        ``"nbvcp"``, or ``"zinbvcp"``.  ``"lnm"`` and ``"lnmvcp"`` force
        parameterization ``"logistic_normal"`` and upgrade default
        ``inference_method="svi"`` to ``"vae"``.
    parameterization : str, default="canonical"
        Parameterization scheme: "canonical", "mean_prob", "mean_odds"
        (backward compat: "standard", "linked", "odds_ratio"), or
        ``"logistic_normal"`` for ``lnm`` / ``lnmvcp``.
    inference_method : str, default="svi"
        Inference method: "svi", "mcmc", or "vae".
    unconstrained : bool, default=False
        Whether to use unconstrained parameterization (Normal+transform).
    d_mode : str, default="low_rank"
        For ``model="lnm"`` / ``"lnmvcp"`` only: ``"low_rank"`` (decoder-only
        ALR mean) or ``"learned"`` (adds ``d_lnm`` and IID Gaussian noise in
        ALR space). Ignored for other models.
    alr_reference_idx : int, default=-1
        For ``model="lnm"`` / ``"lnmvcp"`` only: zero-based index of the ALR
        reference gene. ``-1`` selects the last gene (legacy default). Ignored
        for other models.
    guide_rank : Optional[int], default=None
        Rank for low-rank guide on gene-specific parameter. If provided,
        creates a LowRankGuide for the appropriate parameter (r or mu).
        Mutually exclusive with ``guide_flow``.  When ``joint_params`` is
        given but ``guide_rank`` is ``None`` (and no ``guide_flow``), the
        group uses the linear-coupling-only joint guide: diagonal marginals
        plus a per-gene linear regression among the listed params, with no
        low-rank factor ``W`` (a ``JointLowRankGuide(rank=0,
        dense_params=[])`` marker).
    guide_flow : Optional[str], default=None
        Normalizing-flow type for the variational guide. Mutually exclusive
        with ``guide_rank``. When set, creates a NormalizingFlowGuide
        (per-parameter) or JointNormalizingFlowGuide (when combined with
        ``joint_params``). Supported: "spline_coupling", "affine_coupling",
        "maf", "iaf".
    guide_flow_num_layers : int, default=4
        Number of flow layers in the normalizing-flow guide.
    guide_flow_hidden_dims : Optional[List[int]], default=None
        Hidden dimensions for the conditioner network. Default is [64, 64].
    guide_flow_activation : str, default="relu"
        Activation function for flow conditioner MLPs.
    guide_flow_n_bins : int, default=8
        Number of spline bins (only for ``guide_flow="spline_coupling"``).
    guide_flow_mixture_strategy : str, default="independent"
        How to handle mixture components and dataset indices in flow
        guides.  ``"independent"`` creates separate ``FlowChain``
        per component; ``"shared"`` uses one flow conditioned on a
        one-hot index vector.
    n_components : Optional[int], default=None
        Number of mixture components. If provided, creates a mixture model.
    mixture_params : Optional[List[str]], default=None
        List of parameter names to make mixture-specific. If None and
        n_components is set, defaults to all sampled core parameters for the
        selected parameterization.
    priors : Optional[Dict[str, Any]], default=None
        Dictionary of prior parameters keyed by parameter name.
        Values should be tuples of prior hyperparameters.
        Example: {"p": (2.0, 2.0), "r": (1.0, 0.5)}
    amortize_capture : bool, default=False
        Whether to use amortized inference for capture probability.
        Only applies to VCP models (nbvcp, zinbvcp). When True, uses a
        neural network to predict variational parameters from total UMI.
    capture_hidden_dims : Optional[List[int]], default=None
        Hidden layer dimensions for the capture amortizer MLP.
        Default is [64, 32]. Only used if amortize_capture=True.
    capture_activation : str, default="leaky_relu"
        Activation function for the capture amortizer MLP.
        Only used if amortize_capture=True.
    capture_output_transform : str, default="softplus"
        Transform for positive output parameters in constrained mode.
        "softplus" (default) or "exp". Only used if amortize_capture=True.
    capture_clamp_min : float or None, default=0.1
        Minimum clamp for amortizer positive outputs. Only used if
        amortize_capture=True and unconstrained=False.
    capture_clamp_max : float or None, default=50.0
        Maximum clamp for amortizer positive outputs. Only used if
        amortize_capture=True and unconstrained=False.
    capture_amortization : AmortizationConfig or dict, optional
        Single config object for capture amortization. When provided, it
        is used directly in guide_families and overrides the six
        capture_* parameters above. Can be an AmortizationConfig or a dict
        (converted to AmortizationConfig). When None and amortize_capture=True,
        an AmortizationConfig is built from the six capture_* parameters
        (backward compatible).

    Returns
    -------
    ModelConfig
        Configured model configuration object ready for use with
        `create_model()` or inference functions.

    Raises
    ------
    ValueError
        If model type is not recognized or parameterization is invalid.

    Examples
    --------
    Basic usage:

    >>> config = build_config_from_preset(model="nbdm")

    With low-rank guide:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="mean_prob",
    ...     guide_rank=15,
    ... )

    Mixture model:

    >>> config = build_config_from_preset(
    ...     model="zinb",
    ...     n_components=3,
    ... )

    With custom priors:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     priors={"p": (2.0, 2.0), "r": (1.0, 0.5)},
    ... )

    With normalizing-flow guide:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="mean_odds",
    ...     guide_flow="spline_coupling",
    ... )

    Joint normalizing-flow guide:

    >>> config = build_config_from_preset(
    ...     model="nbdm",
    ...     parameterization="mean_odds",
    ...     unconstrained=True,
    ...     prob_prior="gaussian",
    ...     guide_flow="spline_coupling",
    ...     joint_params=["mu", "phi"],
    ... )

    Notes
    -----
    - Parameterization names support both new ("canonical", "mean_prob",
      "mean_odds") and old ("standard", "linked", "odds_ratio") for
      backward compatibility.
    - Guide rank is applied to the gene-specific parameter (r for canonical,
      mu for mean_prob/mean_odds). Guide flow follows the same convention.
    - ``guide_rank`` and ``guide_flow`` are mutually exclusive.
    - The returned ModelConfig can be passed to `create_model()` to get
      the actual model and guide functions.

    See Also
    --------
    scribe.models.presets.factory.create_model : Creates model/guide from config.
    """
    # --- LNM family: VAE or Laplace (LNMVCP-Laplace adds a scalar
    # Newton on per-cell eta_capture decoupled from the composition
    # block via the block-diagonal Hessian).
    model_lower = model.lower()
    if model_lower in ("lnm", "lnmvcp"):
        from ..models.parameterizations import (
            resolve_user_parameterization_for_model,
        )
        parameterization = resolve_user_parameterization_for_model(
            model_lower, parameterization
        )
        if inference_method.lower() == "svi":
            inference_method = "vae"

    # --- PLN / NBLN family: single parameterization (POISSON_LOGNORMAL),
    # VAE-or-Laplace.  NBLN shares this branch with PLN -- both use the
    # same y_log_rate decoder; NBLN swaps the observation channel from
    # Poisson to NB and adds gene dispersion ``r_g`` as a global extra
    # parameter (handled by MODEL_EXTRA_PARAMS["nbln"]=["r"]).
    if model_lower in ("pln", "nbln"):
        from ..models.parameterizations import (
            resolve_user_parameterization_for_model,
        )
        parameterization = resolve_user_parameterization_for_model(
            model_lower, parameterization
        )
        # Auto-promote ``svi`` to ``vae`` (the encoder-based path)
        # only — ``laplace`` is a deliberate user choice that uses a
        # different engine entirely and must NOT be coerced.
        if inference_method.lower() == "svi":
            inference_method = "vae"
        # Validate Laplace is requested only on supported models.
    elif inference_method.lower() == "laplace":
        # Single source of truth: ``api.constants.LAPLACE_SUPPORTED_BASE_MODELS``.
        from ..api.constants import LAPLACE_SUPPORTED_BASE_MODELS

        if model_lower not in LAPLACE_SUPPORTED_BASE_MODELS:
            raise ValueError(
                "inference_method='laplace' is supported for "
                f"{sorted(LAPLACE_SUPPORTED_BASE_MODELS)}. "
                f"Use 'svi'/'vae'/'mcmc' for model={model_lower!r}."
            )
    if d_mode not in ("low_rank", "learned"):
        raise ValueError(
            f"d_mode must be 'low_rank' or 'learned', got {d_mode!r}."
        )

    # ==========================================================================
    # Validate parameterization
    # ==========================================================================
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown parameterization: {parameterization}. "
            f"Supported: {list(PARAMETERIZATIONS.keys())}"
        )

    # ==========================================================================
    # Build guide families configuration
    # ==========================================================================
    guide_family_kwargs = {}

    # guide_rank and guide_flow are mutually exclusive
    if guide_rank is not None and guide_flow is not None:
        raise ValueError(
            "guide_rank and guide_flow are mutually exclusive — "
            "use guide_rank for low-rank guides or guide_flow for "
            "normalizing-flow guides, not both"
        )

    # dense_params selects a subset for the cross-gene low-rank (Woodbury)
    # block, which only exists when a rank is given.  Requesting dense params
    # without a rank is contradictory.  Note: joint_params *without* a rank or
    # flow is valid — it selects linear-coupling-only mode (diagonal marginals
    # + per-gene linear regression, no low-rank W); see the guide-family block
    # below.
    if dense_params is not None and guide_rank is None:
        raise ValueError(
            "dense_params requires guide_rank to be set: the dense block is a "
            "cross-gene low-rank multivariate normal.  For per-gene linear "
            "coupling without any low-rank block, pass joint_params alone and "
            "omit dense_params."
        )

    # Resolve parameterization strategy (needed by both low-rank and flow)
    param_strategy = PARAMETERIZATIONS[parameterization]
    gene_param_name = param_strategy.gene_param_name  # "r" or "mu"

    # Resolve semantic shorthands ("all", "biological", "mean", etc.) to
    # concrete lists of internal parameter names before they are consumed
    # by the guide-family and builder logic below.
    mixture_params = resolve_param_shorthand(
        mixture_params, param_strategy, model
    )
    joint_params = resolve_param_shorthand(joint_params, param_strategy, model)
    dense_params = resolve_param_shorthand(dense_params, param_strategy, model)

    # Handle low-rank guide for parameters.  joint_params may include
    # both gene-specific and scalar parameters (heterogeneous dims).
    if guide_rank is not None:
        from ..models.components import JointLowRankGuide, LowRankGuide

        if joint_params is not None:
            # Joint low-rank: all listed params share a single covariance.
            # Supports heterogeneous dimensions (scalar + gene-specific).
            # When dense_params is a strict subset, the guide uses a
            # structured block where only dense params get cross-gene
            # low-rank factors and non-dense params get gene-local coupling.
            _effective_dense = dense_params
            if dense_params is not None and set(dense_params) == set(
                joint_params
            ):
                _effective_dense = None
            joint_guide = JointLowRankGuide(
                rank=guide_rank,
                group="joint",
                dense_params=_effective_dense,
            )
            for pname in joint_params:
                guide_family_kwargs[pname] = joint_guide
            # If the gene param is not in joint_params, give it an
            # individual LowRankGuide so it still gets low-rank treatment
            if gene_param_name not in joint_params:
                guide_family_kwargs[gene_param_name] = LowRankGuide(
                    rank=guide_rank
                )
        else:
            guide_family_kwargs[gene_param_name] = LowRankGuide(rank=guide_rank)

    # Linear-coupling-only joint guide: joint_params with neither a rank nor a
    # flow.  Every joint parameter gets a diagonal marginal plus a per-gene
    # linear regression on the earlier joint parameters at the same gene
    # (the non-dense block of the structured joint guide).  No cross-gene
    # low-rank factor W is built — the empty dense_params list routes the
    # group to setup_structured_joint_guide with an empty dense set, and the
    # rank-0 marker is never consumed.  This makes two (or more) gene-specific
    # parameters linearly correlated per gene without invoking the low-rank
    # multivariate Gaussian.  (guide_flow joint guides are handled below.)
    elif joint_params is not None and guide_flow is None:
        from ..models.components import JointLowRankGuide

        joint_guide = JointLowRankGuide(
            rank=0,
            group="joint",
            dense_params=[],
        )
        for pname in joint_params:
            guide_family_kwargs[pname] = joint_guide

    # Handle normalizing-flow guide (parallels the low-rank block above)
    if guide_flow is not None:
        from ..models.components import (
            JointNormalizingFlowGuide,
            NormalizingFlowGuide,
        )

        flow_kwargs = dict(
            flow_type=guide_flow,
            num_layers=guide_flow_num_layers,
            hidden_dims=tuple(guide_flow_hidden_dims or [64, 64]),
            activation=guide_flow_activation,
            n_bins=guide_flow_n_bins,
            mixture_strategy=guide_flow_mixture_strategy,
            zero_init_output=guide_flow_zero_init,
            use_layer_norm=guide_flow_layer_norm,
            use_residual=guide_flow_residual,
            soft_clamp=guide_flow_soft_clamp,
            use_loft=guide_flow_loft,
            log_det_f64=guide_flow_log_det_f64,
        )

        if joint_params is not None:
            _effective_dense = dense_params
            if dense_params is not None and set(dense_params) == set(
                joint_params
            ):
                _effective_dense = None
            joint_guide = JointNormalizingFlowGuide(
                group="joint",
                dense_params=_effective_dense,
                **flow_kwargs,
            )
            for pname in joint_params:
                guide_family_kwargs[pname] = joint_guide
            # If the gene param is not in joint_params, give it an
            # individual NormalizingFlowGuide
            if gene_param_name not in joint_params:
                guide_family_kwargs[gene_param_name] = NormalizingFlowGuide(
                    **flow_kwargs
                )
        else:
            guide_family_kwargs[gene_param_name] = NormalizingFlowGuide(
                **flow_kwargs
            )

    # Handle amortized inference for capture probability (VCP models only)
    if capture_amortization is not None:
        if model not in ("nbvcp", "zinbvcp"):
            raise ValueError(
                "capture_amortization is only valid for VCP models "
                f"(nbvcp, zinbvcp), not '{model}'"
            )
        effective = (
            AmortizationConfig(**capture_amortization)
            if isinstance(capture_amortization, dict)
            else capture_amortization
        )
        guide_family_kwargs["capture_amortization"] = effective
    elif amortize_capture:
        if model not in ("nbvcp", "zinbvcp"):
            raise ValueError(
                f"amortize_capture=True is only valid for VCP models "
                f"(nbvcp, zinbvcp), not '{model}'"
            )
        guide_family_kwargs["capture_amortization"] = AmortizationConfig(
            enabled=True,
            hidden_dims=capture_hidden_dims or [64, 32],
            activation=capture_activation,
            output_transform=capture_output_transform,
            output_clamp_min=capture_clamp_min,
            output_clamp_max=capture_clamp_max,
        )

    guide_families = (
        GuideFamilyConfig(**guide_family_kwargs)
        if guide_family_kwargs
        else None
    )

    # ==========================================================================
    # Build ModelConfig using builder
    # ==========================================================================
    builder = (
        ModelConfigBuilder()
        .for_model(model_lower)
        .with_parameterization(parameterization)
        .with_inference(inference_method)
    )

    # Build the VAE config for both ``vae`` and ``laplace`` inference
    # methods. The generative model is the same in both cases (linear
    # decoder, low-rank-plus-diagonal covariance); only the *inference*
    # procedure differs (encoder vs Newton). The Laplace engine reads
    # ``model_config.vae.latent_dim`` to size the decoder kernel, so
    # we must build VAEConfig here even for Laplace.
    if inference_method in ("vae", "laplace"):
        # Encoder input transform defaults to ``log1p_prop``
        # (compositional) for LNM and for PLN-with-capture-anchor; it
        # stays at ``log1p`` (raw counts on log scale) for
        # PLN-without-capture-anchor. The asymmetry is structural:
        #
        # * LNM's multinomial likelihood is intrinsically
        #   scale-invariant (only the simplex ``ρ`` matters), so the
        #   encoder being compositional is always fine.
        # * PLN's likelihood is in absolute log-rate space. With a
        #   capture anchor, the per-cell scale parameter
        #   ``eta_capture`` carries library-size variation, so a
        #   compositional encoder *helps* (it keeps scale
        #   information out of the latent ``z``, eliminating the
        #   identifiability ridge between ``z`` and
        #   ``eta_capture``).
        # * PLN *without* a capture anchor has no per-cell scale
        #   parameter at all -- the only place library size can be
        #   encoded is in the latent ``z`` via the decoder. A
        #   compositional encoder would strip that signal, forcing
        #   every cell to predict the same total counts. So the
        #   default falls back to ``log1p`` in that regime.
        #
        # The capture-anchor signal is the presence of any
        # capture-related prior key (canonical or alias) in
        # ``priors``. We check the small canonical set plus the
        # registered aliases ``capture_efficiency``,
        # ``capture_scaling``, ``capture_anchor``, ``organism``.
        resolved_vae_input_transform = vae_input_transform

        _CAPTURE_ANCHOR_PRIOR_KEYS = frozenset({
            "eta_capture",
            "mu_eta",
            "organism",
            "capture_efficiency",
            "capture_scaling",
            "capture_anchor",
        })
        _has_capture_anchor = bool(priors) and any(
            k in _CAPTURE_ANCHOR_PRIOR_KEYS for k in priors.keys()
        )

        _wants_compositional_input = model_lower in ("lnm", "lnmvcp") or (
            model_lower in ("pln", "nbln") and _has_capture_anchor
        )
        if _wants_compositional_input and vae_input_transform == "log1p":
            resolved_vae_input_transform = "log1p_prop"

        # Resolve the ``vae_standardize`` sentinel. ``None`` means "pick a
        # sensible per-model default": LNM, PLN, and NBLN benefit from
        # standardization, while every other VAE model preserves its
        # historical default of ``False``.
        if vae_standardize is None:
            resolved_vae_standardize = model_lower in (
                "lnm", "lnmvcp", "pln", "nbln"
            )
        else:
            resolved_vae_standardize = bool(vae_standardize)

        vae_kwargs = {
            "latent_dim": vae_latent_dim,
            "input_transform": resolved_vae_input_transform,
            "standardize": resolved_vae_standardize,
            "decoder_transforms": vae_decoder_transforms,
            "flow_type": vae_flow_type,
            "flow_num_layers": vae_flow_num_layers,
        }
        if vae_encoder_hidden_dims is not None:
            vae_kwargs["encoder_hidden_dims"] = vae_encoder_hidden_dims
        if vae_decoder_hidden_dims is not None:
            vae_kwargs["decoder_hidden_dims"] = vae_decoder_hidden_dims
        if vae_activation is not None:
            vae_kwargs["activation"] = vae_activation
        if vae_flow_hidden_dims is not None:
            vae_kwargs["flow_hidden_dims"] = vae_flow_hidden_dims
        builder.with_vae(**vae_kwargs)
        # ``with_vae`` force-sets ``_inference_method = VAE`` to keep
        # encoder-mode fits coherent. For Laplace mode we need the
        # VAEConfig (linear decoder, latent_dim, input transform, etc.)
        # but the *inference method* must remain LAPLACE so the
        # dispatcher routes to the Laplace handler. Restore it here.
        if inference_method.lower() == "laplace":
            from ..models.config.enums import InferenceMethod as _IM

            builder._inference_method = _IM.LAPLACE

    if unconstrained:
        builder.unconstrained()

    # Joint guides operate in unconstrained space and expect transform-aware
    # parameter specs. Promote to unconstrained automatically when users request
    # joint_params so preset calls do not fail at guide dry-run time.
    if joint_params is not None:
        builder.unconstrained()

    # Gene-level priors
    if expression_prior != "none":
        builder._expression_prior = expression_prior
        builder._unconstrained = True
    if prob_prior != "none":
        builder._prob_prior = prob_prior
        builder._unconstrained = True
    if zero_inflation_prior != "none":
        builder._zero_inflation_prior = zero_inflation_prior
        builder._unconstrained = True

    # Multi-dataset configuration: set builder fields directly
    if n_datasets is not None:
        builder._n_datasets = n_datasets
        builder._dataset_params = dataset_params
        builder._dataset_mixing = dataset_mixing
        builder._expression_dataset_prior = expression_dataset_prior
        builder._prob_dataset_prior = prob_dataset_prior
        builder._prob_dataset_mode = prob_dataset_mode
        builder._zero_inflation_dataset_prior = zero_inflation_dataset_prior
        builder._overdispersion_dataset_prior = overdispersion_dataset_prior
        # Two-state dataset-level regime hierarchy + free overdispersion.
        builder._regime_dataset_prior = regime_dataset_prior
        builder._regime_dataset_target = regime_dataset_target
        builder._overdispersion_dataset_independent = (
            overdispersion_dataset_independent
        )
        if (
            expression_dataset_prior != "none"
            or prob_dataset_prior != "none"
            or zero_inflation_dataset_prior != "none"
            or overdispersion_dataset_prior != "none"
            or regime_dataset_prior != "none"
        ):
            builder._unconstrained = True
    else:
        # Keep user intent explicit even when currently not multi-dataset.
        builder._dataset_mixing = dataset_mixing

    # Multi-factor grouping descriptor (None for single-factor / non-grouped).
    builder._grouping_spec = grouping_spec

    # Horseshoe hyperparameters
    builder._horseshoe_tau0 = horseshoe_tau0
    builder._horseshoe_slab_df = horseshoe_slab_df
    builder._horseshoe_slab_scale = horseshoe_slab_scale

    # NEG hyperparameters
    builder._neg_u = neg_u
    builder._neg_a = neg_a
    builder._neg_tau = neg_tau

    # Hierarchical prior for per-dataset mu_eta (capture scaling)
    builder._capture_scaling_prior = capture_scaling_prior

    # Data-informed mean anchoring prior
    builder._expression_anchor = expression_anchor
    builder._expression_anchor_sigma = expression_anchor_sigma

    # Gene-specific overdispersion (e.g. BNB)
    builder._overdispersion = overdispersion
    builder._overdispersion_prior = overdispersion_prior
    builder._overdispersion_dataset_prior = overdispersion_dataset_prior

    if n_components is not None:
        builder.as_mixture(n_components, mixture_params)

    if joint_params is not None:
        builder.with_joint_params(joint_params)

    if dense_params is not None:
        builder.with_dense_params(dense_params)

    if guide_families is not None:
        builder.with_guide_families(guide_families)

    if priors:
        builder.with_priors(**priors)

    if model_lower in ("lnm", "lnmvcp"):
        builder._d_mode = d_mode
        builder._alr_reference_idx = alr_reference_idx

    if model_lower == "pln":
        builder._d_mode = d_mode

    # `correlate_other_column` applies to PLN/NBLN/TSLN-Rate/TSLN-Logit
    # (informational for LNM — see ModelConfig.correlate_other_column
    # docstring).  Forward unconditionally; non-applicable model
    # families simply ignore the flag.
    builder._correlate_other_column = bool(correlate_other_column)

    # Forward positive_transform (string or per-parameter dict) to the
    # underlying ``ModelConfig``.  The dict form is normalized to
    # internal parameter names inside ``ModelConfig`` via its model
    # validator, so descriptive aliases (``mean_expression`` →
    # ``mu``, ``capture_prob`` → ``p_capture``, ...) work
    # transparently when the user routes through ``scribe.fit``.
    #
    # Model-aware default: when the caller passes ``positive_transform=None``
    # (the user did not set it explicitly), default to ``{"mu": "exp"}``
    # for the TwoState family — exp on the gene mean is essentially
    # mandatory for datasets spanning 3+ decades of expression because
    # the softplus Jacobian saturates to 1 in the large-loc regime,
    # leaving SVI with additive-step geometry on a multiplicative
    # quantity.  Other positive parameters (``burst_size``, ``k_off``,
    # ``p_capture``) keep softplus.  Other model families default to
    # ``"softplus"`` (legacy behavior).  This selection happens AFTER
    # ``model_lower`` resolution so the ``variable_capture`` upgrade of
    # ``"twostate"`` → ``"twostatevcp"`` is handled consistently.
    if positive_transform is None:
        if model_lower in ("twostate", "twostatevcp"):
            builder._positive_transform = {"mu": "exp"}
        else:
            builder._positive_transform = "softplus"
    else:
        builder._positive_transform = positive_transform

    # Gauss-Legendre node count for the two-state Poisson-Beta likelihood.
    # ``None`` keeps the PoissonBetaCompound default (60); ignored by
    # non-two-state model families.
    builder._n_quad_nodes = n_quad_nodes

    return builder.build()


# ==============================================================================
# Export
# ==============================================================================

__all__ = ["build_config_from_preset"]
