"""
Simplified API for SCRIBE inference.

This module provides a user-friendly entry point for SCRIBE inference with
sensible defaults and flat kwargs instead of nested configuration objects.

Functions
---------
fit
    Main entry point for SCRIBE inference with simplified API.

Examples
--------
>>> import scribe
>>>
>>> # Simplest usage - just data and model name
>>> results = scribe.fit(adata, model="nbdm")
>>>
>>> # With customization via flat kwargs
>>> results = scribe.fit(
...     adata,
...     model="zinb",
...     parameterization="linked",
...     n_components=3,
...     n_steps=100000,
...     batch_size=512,
... )
>>>
>>> # Power users can still pass explicit config objects
>>> from scribe.models.config import ModelConfigBuilder, InferenceConfig, SVIConfig
>>> model_config = ModelConfigBuilder().for_model("nbdm").build()
>>> inference_config = InferenceConfig.from_svi(SVIConfig(n_steps=75000))
>>> results = scribe.fit(
...     adata,
...     model_config=model_config,
...     inference_config=inference_config,
... )
"""

import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from .models.config import (
    AmortizationConfig,
    DataConfig,
    EarlyStoppingConfig,
    InferenceConfig,
    MCMCConfig,
    ModelConfig,
    SVIConfig,
)
from .models.config.enums import InferenceMethod
from .inference.utils import (
    process_counts_data,
    validate_inference_config_match,
)
from .inference.preset_builder import build_config_from_preset
from .inference.dispatcher import _run_inference
from .core.annotation_prior import (
    build_annotation_prior_logits,
    validate_annotation_prior_logits,
)

# Import result types for type annotations
from .svi.results import ScribeSVIResults
from .mcmc.results import ScribeMCMCResults
from .vae.results import ScribeVAEResults
from .models.parameterizations import PARAMETERIZATIONS

# Type alias for return type
ScribeResults = Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]

# Valid model types
VALID_MODELS = {"nbdm", "zinb", "nbvcp", "zinbvcp"}

# Derive valid parameterizations from the single source of truth
VALID_PARAMETERIZATIONS = set(PARAMETERIZATIONS.keys())

# Valid inference methods
VALID_INFERENCE_METHODS = {"svi", "mcmc", "vae"}


# ==============================================================================
# Internal helpers
# ==============================================================================


def _count_unique_labels(
    adata: "AnnData",
    annotation_key: Union[str, List[str]],
    min_cells: int = 0,
) -> int:
    """
    Count the number of unique non-null annotation labels.

    When ``annotation_key`` is a list of column names, composite labels
    are formed (identical to the logic in
    :func:`build_annotation_prior_logits`) and the unique composites are
    counted.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    annotation_key : str or list of str
        Column name(s) in ``adata.obs``.
    min_cells : int, optional
        Minimum number of cells for a label to be counted.  Labels with
        fewer than ``min_cells`` cells are excluded from the count.
        Default is ``0`` (no filtering).

    Returns
    -------
    int
        Number of unique non-null labels (or composite labels) that meet
        the ``min_cells`` threshold.
    """
    import pandas as pd

    if isinstance(annotation_key, str):
        obs_keys = [annotation_key]
    else:
        obs_keys = list(annotation_key)

    if len(obs_keys) == 1:
        col = adata.obs[obs_keys[0]]
        if hasattr(col, "cat"):
            col = col.astype(object)
        non_null = col.dropna()
    else:
        from .core.annotation_prior import _resolve_composite_annotations

        composite = _resolve_composite_annotations(adata, obs_keys)
        non_null = composite.dropna()

    if min_cells > 0:
        counts = non_null.value_counts()
        return int(len(counts[counts >= min_cells]))
    return int(len(non_null.unique()))


# ------------------------------------------------------------------------------


def _coerce_batch_size_for_dataset(
    batch_size: Optional[int], n_cells: int
) -> Optional[int]:
    """
    Normalize mini-batch size against dataset size.

    This helper preserves user-specified mini-batch sizes when they are valid
    and automatically switches to full-batch mode when the requested
    ``batch_size`` exceeds the number of cells in the current dataset.
    Full-batch mode is represented by ``None`` in SCRIBE's SVI/VAE configs.

    Parameters
    ----------
    batch_size : int or None
        Requested mini-batch size. ``None`` means full-batch mode.
    n_cells : int
        Number of cells available for inference after data processing.

    Returns
    -------
    int or None
        Effective batch size. Returns:

        - ``None`` when ``batch_size`` is ``None``.
        - The original ``batch_size`` when it is less than or equal to
          ``n_cells``.
        - ``None`` when ``batch_size`` is greater than ``n_cells``.

    Warns
    -----
    UserWarning
        Emitted when ``batch_size`` exceeds ``n_cells`` and is coerced to
        ``None``.
    """
    if batch_size is None:
        return None
    if batch_size > n_cells:
        warnings.warn(
            f"batch_size={batch_size} exceeds n_cells={n_cells}; "
            "using full-batch mode (batch_size=None).",
            UserWarning,
            stacklevel=2,
        )
        return None
    return batch_size


# ==============================================================================
# Public API
# ==============================================================================


def fit(
    counts: Union[jnp.ndarray, "AnnData"],
    model: str = "nbdm",
    # Model options
    parameterization: str = "canonical",
    unconstrained: bool = False,
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    guide_rank: Optional[int] = None,
    priors: Optional[Dict[str, Any]] = None,
    # VAE architecture options (when inference_method="vae")
    vae_latent_dim: int = 10,
    vae_encoder_hidden_dims: Optional[List[int]] = None,
    vae_decoder_hidden_dims: Optional[List[int]] = None,
    vae_activation: Optional[str] = None,
    vae_input_transform: str = "log1p",
    vae_standardize: bool = False,
    vae_decoder_transforms: Optional[Dict[str, str]] = None,
    vae_flow_type: str = "none",
    vae_flow_num_layers: int = 4,
    vae_flow_hidden_dims: Optional[List[int]] = None,
    # Amortization options (for VCP models)
    amortize_capture: bool = False,
    capture_hidden_dims: Optional[List[int]] = None,
    capture_activation: str = "leaky_relu",
    capture_output_transform: str = "softplus",
    capture_clamp_min: Optional[float] = 0.1,
    capture_clamp_max: Optional[float] = 50.0,
    capture_amortization: Optional[
        Union[AmortizationConfig, Dict[str, Any]]
    ] = None,
    # Inference options
    inference_method: str = "svi",
    n_steps: int = 50_000,
    batch_size: Optional[int] = None,
    stable_update: bool = True,
    log_progress_lines: bool = False,
    n_samples: int = 2_000,
    n_warmup: int = 1_000,
    n_chains: int = 1,
    # Early stopping options (for SVI/VAE)
    early_stopping: Optional[Union[EarlyStoppingConfig, Dict[str, Any]]] = None,
    # Data options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    seed: int = 42,
    # Annotation prior options (for mixture models)
    annotation_key: Optional[Union[str, List[str]]] = None,
    annotation_confidence: float = 3.0,
    annotation_component_order: Optional[List[str]] = None,
    annotation_min_cells: Optional[int] = None,
    # Power user: explicit configs override above
    model_config: Optional[ModelConfig] = None,
    inference_config: Optional[InferenceConfig] = None,
) -> ScribeResults:
    """
    Simplified entry point for SCRIBE inference.

    This function provides a user-friendly interface for running SCRIBE
    inference with sensible defaults. All options can be specified as flat
    keyword arguments instead of constructing nested configuration objects.

    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts.
        Shape should be (n_cells, n_genes) if cells_axis=0.

    model : str, default="nbdm"
        Model type to use:
            - "nbdm": Negative Binomial Dropout Model (simplest)
            - "zinb": Zero-Inflated Negative Binomial (handles excess zeros)
            - "nbvcp": NB with Variable Capture Probability (models technical
              dropout)
            - "zinbvcp": ZINB with Variable Capture Probability (most
              comprehensive)

    parameterization : str, default="canonical"
        Parameterization scheme:
            - "canonical" (or "standard"): Sample p ~ Beta, r ~ LogNormal
              directly
            - "linked" (or "mean_prob"): Sample p ~ Beta, mu ~ LogNormal, derive
              r
            - "odds_ratio" (or "mean_odds"): Sample phi ~ BetaPrime, mu ~
              LogNormal

    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained distributions.
        This can help with optimization in some cases.

    n_components : int, optional
        Number of mixture components for cell type discovery.
        If None (default), uses a single-component model.
        Must be >= 2 if specified.  When ``annotation_key`` is provided
        and ``n_components`` is omitted, the number of components is
        automatically inferred from the number of unique non-null
        annotation labels.

    mixture_params : List[str], optional
        Which parameters should be component-specific in mixture models. If None
        and n_components is set, defaults to all sampled core parameters for
        the chosen parameterization.
        Example: ["r"] makes only r component-specific while p is shared.

    guide_rank : int, optional
        Rank for low-rank variational guide on gene-specific parameters. If None
        (default), uses mean-field guide (fully factorized). Low-rank guides can
        capture gene correlations but use more memory.

    priors : Dict[str, Any], optional
        Dictionary of prior hyperparameters keyed by parameter name. Values
        should be tuples of prior hyperparameters. Example: {"p": (1.0, 1.0),
        "r": (0.0, 1.0)}

    amortize_capture : bool, default=False
        Whether to use amortized inference for capture probability. When True,
        a neural network predicts variational parameters for p_capture (or
        phi_capture for mean_odds parameterization) from total UMI count.
        This reduces the number of parameters from O(n_cells) to O(1).
        Only applies to VCP models (nbvcp, zinbvcp).

    capture_hidden_dims : List[int], optional
        Hidden layer dimensions for the capture amortizer MLP. Default is
        [64, 32]. Only used if amortize_capture=True.

    capture_activation : str, default="leaky_relu"
        Activation function for the capture amortizer MLP. Options include
        "relu", "gelu", "silu", "tanh", etc. Only used if amortize_capture=True.

    capture_output_transform : str, default="softplus"
        Transform for positive output parameters in constrained mode.
        "softplus" (default): softplus(x) + 0.5, numerically stable.
        "exp": exponential (original behavior, can produce extreme values).
        Only used if amortize_capture=True and unconstrained=False.

    capture_clamp_min : float or None, default=0.1
        Minimum clamp for amortizer positive outputs (alpha, beta) in
        constrained mode. Prevents extreme BetaPrime/Beta shape parameters.
        Set to None to disable. Only used if amortize_capture=True.

    capture_clamp_max : float or None, default=50.0
        Maximum clamp for amortizer positive outputs in constrained mode.
        Set to None to disable. Only used if amortize_capture=True.

    capture_amortization : AmortizationConfig or dict, optional
        Single config object for capture amortization. When provided, it
        overrides the six individual capture_* parameters above. Can be an
        AmortizationConfig instance or a dict (converted to AmortizationConfig).
        When None and amortize_capture=True, an AmortizationConfig is built
        from the six capture_* parameters (backward compatible).

    inference_method : str, default="svi"
        Inference method to use:
            - "svi": Stochastic Variational Inference (fast, scalable)
            - "mcmc": Markov Chain Monte Carlo (exact, slower)
            - "vae": Variational Autoencoder (for representation learning)

    n_steps : int, default=50_000
        Number of optimization steps for SVI/VAE inference.
        Increase for complex models or large datasets.

    batch_size : int, optional
        Mini-batch size for SVI/VAE. If ``None``, uses full-batch inference.
        If provided but larger than the dataset size, it is automatically
        coerced to ``None`` (full-batch mode). Recommended for large datasets
        (>10K cells).

    stable_update : bool, default=True
        Use numerically stable parameter updates in SVI.

    log_progress_lines : bool, default=False
        Whether to emit periodic plain-text progress lines during SVI/VAE
        inference. When enabled, the SVI engine logs approximately 20 updates
        over a run (every ``max(1, n_steps // 20)`` steps). This is useful for
        non-interactive logs such as SLURM ``.out`` files.

    n_samples : int, default=2_000
        Number of MCMC samples to draw (only for inference_method="mcmc").

    n_warmup : int, default=1_000
        Number of MCMC warmup samples (only for inference_method="mcmc").

    n_chains : int, default=1
        Number of MCMC chains to run in parallel (only for
        inference_method="mcmc").

    early_stopping : Union[EarlyStoppingConfig, Dict[str, Any]], optional
        Early stopping configuration for SVI/VAE inference. Can be:
        - EarlyStoppingConfig object
        - Dict with keys: enabled, patience, min_delta, check_every,
          smoothing_window, restore_best
        - None (default): no early stopping, runs for full n_steps
        Only applies to SVI and VAE inference methods.

    cells_axis : int, default=0
        Axis for cells in count matrix. 0 means cells are rows (n_cells,
        n_genes).

    layer : str, optional
        Layer in AnnData to use for counts. If None, uses .X.

    seed : int, default=42
        Random seed for reproducibility.

    annotation_key : str or list of str, optional
        Column name(s) in ``adata.obs`` containing categorical cell-type
        annotations.  When provided, the annotations are used as soft
        priors on per-cell mixture component assignments.  Requires
        ``counts`` to be an AnnData object.

        If ``n_components`` is **not** specified, it is automatically
        inferred from the number of unique non-null annotation labels.

        When a **list** of column names is given, composite labels are
        formed for each cell by joining the per-column values with
        ``"__"`` (double underscore).  For example, columns
        ``["cell_type", "treatment"]`` with values ``"T"`` and ``"ctrl"``
        produce the composite label ``"T__ctrl"``.  A cell is considered
        unlabeled (receives zero logits) if *any* of the specified
        columns has a missing value.

    annotation_confidence : float, default=3.0
        Strength of the annotation prior (kappa).  Controls how strongly
        the annotation influences the component assignment:

        * ``0`` — annotations are ignored (standard model).
        * ``3`` (default) — annotated component gets ~20x prior boost.
        * Large values — approaches hard assignment.

    annotation_component_order : list of str, optional
        Explicit mapping from annotation labels to component indices.
        The *i*-th element is assigned to component *i*.  If ``None``,
        unique labels are sorted alphabetically.  When using multiple
        ``annotation_key`` columns, these should be the composite labels
        using ``"__"`` as separator (e.g.
        ``["T__ctrl", "T__stim", "B__ctrl", "B__stim"]``).

    annotation_min_cells : int, optional
        Minimum number of cells required for an annotation label to be
        used as a component prior.  Labels with fewer than this many
        cells are treated as unlabeled — their logit rows are set to
        zero (no bias toward any component) and the labels are excluded
        from the component mapping.  When ``n_components`` is inferred
        automatically, only labels meeting this threshold are counted.
        If ``None`` (default), no filtering is applied.

    model_config : ModelConfig, optional
        Fully configured model configuration object.
        If provided, overrides model, parameterization, unconstrained,
        n_components, mixture_params, guide_rank, and priors.

    inference_config : InferenceConfig, optional
        Fully configured inference configuration object.
        If provided, overrides inference_method, n_steps, batch_size,
        stable_update, log_progress_lines, n_samples, n_warmup, and n_chains.

    Returns
    -------
    Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]
        Results object containing:
        - Posterior samples or variational parameters
        - Loss history (for SVI/VAE)
        - Diagnostic information
        - Methods for analysis (log_likelihood, posterior_samples, etc.)

    Raises
    ------
    ValueError
        If model, parameterization, or inference_method is not recognized.
        If configuration is invalid.

    Examples
    --------
    Basic usage with NBDM model:

    >>> results = scribe.fit(adata, model="nbdm")

    Zero-inflated model with mixture components:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="zinb",
    ...     n_components=3,
    ...     n_steps=100000,
    ... )

    Linked parameterization with low-rank guide:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="nbdm",
    ...     parameterization="linked",
    ...     guide_rank=15,
    ... )

    MCMC inference for small datasets:

    >>> results = scribe.fit(
    ...     adata,
    ...     model="nbdm",
    ...     inference_method="mcmc",
    ...     n_samples=5000,
    ...     n_chains=4,
    ... )

    See Also
    --------
    run_scribe : Lower-level inference function with more options.
    ModelConfigBuilder : Builder for creating ModelConfig objects.
    InferenceConfig : Unified inference configuration class.
    """
    # ==========================================================================
    # Step 1: Validate inputs
    # ==========================================================================
    if model_config is None:
        # Validate model type
        model_lower = model.lower()
        if model_lower not in VALID_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Valid models are: {', '.join(sorted(VALID_MODELS))}"
            )

        # Validate parameterization
        param_lower = parameterization.lower()
        if param_lower not in VALID_PARAMETERIZATIONS:
            raise ValueError(
                f"Unknown parameterization: '{parameterization}'. "
                f"Valid parameterizations are: "
                f"{', '.join(sorted(VALID_PARAMETERIZATIONS))}"
            )

    if inference_config is None:
        # Validate inference method
        method_lower = inference_method.lower()
        if method_lower not in VALID_INFERENCE_METHODS:
            raise ValueError(
                f"Unknown inference_method: '{inference_method}'. "
                f"Valid methods are: "
                f"{', '.join(sorted(VALID_INFERENCE_METHODS))}"
            )

    # ==========================================================================
    # Step 2: Process data
    # ==========================================================================
    data_config = DataConfig(cells_axis=cells_axis, layer=layer)
    count_data, adata, n_cells, n_genes = process_counts_data(
        counts, data_config
    )

    # ==========================================================================
    # Step 2b: Build annotation prior logits (if requested)
    # ==========================================================================
    annotation_prior_logits = None
    if annotation_key is not None:
        if adata is None:
            raise ValueError(
                "annotation_key requires counts to be an AnnData object "
                "(not a raw array), so that adata.obs can be read."
            )
        # Resolve n_components — may come from explicit kwarg, from a
        # user-supplied model_config, or be inferred from annotations.
        _n_comp = n_components
        if _n_comp is None and model_config is not None:
            _n_comp = model_config.n_components
        _min_cells = annotation_min_cells or 0
        if _n_comp is None:
            # Infer from the number of unique non-null annotation labels
            _n_comp = _count_unique_labels(
                adata, annotation_key, min_cells=_min_cells
            )
            n_components = _n_comp  # propagate so ModelConfig picks it up
        annotation_prior_logits, _label_map = build_annotation_prior_logits(
            adata=adata,
            obs_key=annotation_key,
            n_components=_n_comp,
            confidence=annotation_confidence,
            component_order=annotation_component_order,
            min_cells=_min_cells,
        )
        validate_annotation_prior_logits(
            annotation_prior_logits, n_cells, _n_comp
        )

    # ==========================================================================
    # Step 3: Build or use ModelConfig
    # ==========================================================================
    if model_config is None:
        # Single config object: prefer capture_amortization; else build from 6
        # params
        effective_capture_amortization = None
        if capture_amortization is not None:
            effective_capture_amortization = (
                AmortizationConfig(**capture_amortization)
                if isinstance(capture_amortization, dict)
                else capture_amortization
            )
        elif amortize_capture:
            effective_capture_amortization = AmortizationConfig(
                enabled=True,
                hidden_dims=capture_hidden_dims or [64, 32],
                activation=capture_activation,
                output_transform=capture_output_transform,
                output_clamp_min=capture_clamp_min,
                output_clamp_max=capture_clamp_max,
            )
        model_config = build_config_from_preset(
            model=model.lower(),
            parameterization=parameterization.lower(),
            inference_method=inference_method.lower(),
            unconstrained=unconstrained,
            guide_rank=guide_rank,
            n_components=n_components,
            mixture_params=mixture_params,
            priors=priors,
            vae_latent_dim=vae_latent_dim,
            vae_encoder_hidden_dims=vae_encoder_hidden_dims,
            vae_decoder_hidden_dims=vae_decoder_hidden_dims,
            vae_activation=vae_activation,
            vae_input_transform=vae_input_transform,
            vae_standardize=vae_standardize,
            vae_decoder_transforms=vae_decoder_transforms,
            vae_flow_type=vae_flow_type,
            vae_flow_num_layers=vae_flow_num_layers,
            vae_flow_hidden_dims=vae_flow_hidden_dims,
            amortize_capture=amortize_capture,
            capture_hidden_dims=capture_hidden_dims,
            capture_activation=capture_activation,
            capture_output_transform=capture_output_transform,
            capture_clamp_min=capture_clamp_min,
            capture_clamp_max=capture_clamp_max,
            capture_amortization=effective_capture_amortization,
        )

    # ==========================================================================
    # Step 4: Build or use InferenceConfig
    # ==========================================================================
    if inference_config is None:
        # Determine inference method from model_config or parameter
        method = model_config.inference_method
        effective_batch_size = _coerce_batch_size_for_dataset(
            batch_size=batch_size, n_cells=n_cells
        )

        # Process early_stopping configuration
        early_stop_config = None
        if early_stopping is not None:
            if isinstance(early_stopping, EarlyStoppingConfig):
                early_stop_config = early_stopping
            elif isinstance(early_stopping, dict):
                # Convert dict to EarlyStoppingConfig
                early_stop_config = EarlyStoppingConfig(**early_stopping)
            else:
                raise ValueError(
                    f"early_stopping must be EarlyStoppingConfig or dict, "
                    f"got {type(early_stopping)}"
                )

        if method == InferenceMethod.SVI:
            svi_config = SVIConfig(
                n_steps=n_steps,
                batch_size=effective_batch_size,
                stable_update=stable_update,
                log_progress_lines=log_progress_lines,
                early_stopping=early_stop_config,
            )
            inference_config = InferenceConfig.from_svi(svi_config)
        elif method == InferenceMethod.MCMC:
            if early_stopping is not None:
                warnings.warn(
                    "early_stopping is only supported for SVI and VAE "
                    "inference methods. Ignoring for MCMC.",
                    UserWarning,
                )
            mcmc_config = MCMCConfig(
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_chains=n_chains,
            )
            inference_config = InferenceConfig.from_mcmc(mcmc_config)
        elif method == InferenceMethod.VAE:
            # VAE uses SVI config
            svi_config = SVIConfig(
                n_steps=n_steps,
                batch_size=effective_batch_size,
                stable_update=stable_update,
                log_progress_lines=log_progress_lines,
                early_stopping=early_stop_config,
            )
            inference_config = InferenceConfig.from_vae(svi_config)
        else:
            raise ValueError(f"Unknown inference method: {method}")
    else:
        # Validate that inference_config matches model_config
        validate_inference_config_match(model_config, inference_config)

    # ==========================================================================
    # Step 5: Run inference
    # ==========================================================================
    return _run_inference(
        inference_config.method,
        model_config=model_config,
        count_data=count_data,
        inference_config=inference_config,
        adata=adata,
        n_cells=n_cells,
        n_genes=n_genes,
        data_config=data_config,
        seed=seed,
        annotation_prior_logits=annotation_prior_logits,
    )
