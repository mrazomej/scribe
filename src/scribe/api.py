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
from .models.config.enums import HierarchicalPriorType, InferenceMethod
from .inference.utils import (
    process_counts_data,
    validate_inference_config_match,
)
from .inference.preset_builder import build_config_from_preset
from .inference.dispatcher import _run_inference
from .core.annotation_prior import (
    build_annotation_prior_logits,
    build_component_mapping,
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


# ------------------------------------------------------------------------------


def _normalize_prior_type_name(prior: Any) -> str:
    """Normalize a hierarchical prior selector to a lowercase string value.

    Parameters
    ----------
    prior : Any
        Prior selector coming from ``fit`` kwargs. This may be either a
        ``HierarchicalPriorType`` enum instance or a plain string.

    Returns
    -------
    str
        Lowercase prior name (e.g. ``"none"``, ``"gaussian"``).
    """
    if isinstance(prior, str):
        return prior.lower()
    return str(getattr(prior, "value", prior)).lower()


# ==============================================================================
# Public API
# ==============================================================================


def fit(
    counts: Union[jnp.ndarray, "AnnData"],
    model: str = "nbdm",
    # Model options
    parameterization: str = "canonical",
    unconstrained: bool = False,
    mu_prior: str = "none",
    p_prior: str = "none",
    gate_prior: str = "none",
    # Multi-dataset hierarchy options
    n_datasets: Optional[int] = None,
    dataset_key: Optional[str] = None,
    dataset_params: Optional[List[str]] = None,
    dataset_mixing: Optional[bool] = None,
    mu_dataset_prior: str = "none",
    p_dataset_prior: str = "none",
    p_dataset_mode: str = "gene_specific",
    gate_dataset_prior: str = "none",
    overdispersion_dataset_prior: str = "none",
    auto_downgrade_single_dataset_hierarchy: bool = True,
    # Horseshoe hyperparameters
    horseshoe_tau0: float = 1.0,
    horseshoe_slab_df: int = 4,
    horseshoe_slab_scale: float = 2.0,
    # NEG (Normal-Exponential-Gamma) hyperparameters
    neg_u: float = 1.0,
    neg_a: float = 1.0,
    neg_tau: float = 1.0,
    # Hierarchical prior for per-dataset mu_eta (capture scaling)
    mu_eta_prior: str = "none",
    # Data-informed mean anchoring prior
    mu_mean_anchor: bool = False,
    mu_mean_anchor_sigma: float = 0.3,
    # Gene-specific overdispersion beyond the NB family
    overdispersion: str = "none",
    overdispersion_prior: str = "horseshoe",
    n_components: Optional[int] = None,
    mixture_params: Optional[List[str]] = None,
    guide_rank: Optional[int] = None,
    joint_params: Optional[List[str]] = None,
    dense_params: Optional[List[str]] = None,
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
    optimizer_config: Optional[Dict[str, Any]] = None,
    stable_update: bool = True,
    log_progress_lines: bool = False,
    n_samples: int = 2_000,
    n_warmup: int = 1_000,
    n_chains: int = 1,
    # Early stopping options (for SVI/VAE)
    early_stopping: Optional[Union[EarlyStoppingConfig, Dict[str, Any]]] = None,
    restore_best: bool = False,
    # Data options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    seed: int = 42,
    # Annotation prior options (for mixture models)
    annotation_key: Optional[Union[str, List[str]]] = None,
    annotation_confidence: float = 3.0,
    annotation_component_order: Optional[List[str]] = None,
    annotation_min_cells: Optional[int] = None,
    # SVI-to-MCMC initialization
    svi_init: Optional[ScribeSVIResults] = None,
    # Float64 precision — defaults to True for MCMC, False for SVI/VAE
    enable_x64: Optional[bool] = None,
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

    mu_prior : str, default="none"
        Gene-level hierarchical prior for mu (or r) across mixture
        components.  Per-component means are drawn from a shared
        gene-level population distribution per gene, providing adaptive
        shrinkage: most genes share similar expression across cell
        types, with only some deviating.  Each gene has its own
        hyperprior because expression magnitudes vary by orders of
        magnitude across genes.  Requires ``unconstrained=True`` and
        ``n_components >= 2``.  Accepted values: ``"none"``,
        ``"gaussian"``, ``"horseshoe"``, ``"neg"``.

    mu_mean_anchor : bool, default=False
        Enable data-informed anchoring prior on the biological mean
        ``mu_g``.  When True, per-gene prior centers are computed from
        the observed sample means and average capture probability,
        anchoring ``log(mu_g) ~ N(log(u_bar_g / nu_bar), sigma^2)``.
        This resolves the mu-phi degeneracy in the negative binomial.
        Automatically enables ``unconstrained=True``.  For VCP models,
        ``priors.eta_capture`` or ``priors.organism`` must be set so
        that ``nu_bar`` can be estimated from library sizes and
        ``M_0``.  Non-VCP models use ``nu_bar=1`` by default.

    mu_mean_anchor_sigma : float, default=0.3
        Log-scale standard deviation for the mean anchoring prior.
        Smaller values (0.1--0.2) give tight anchoring; moderate
        values (0.3--0.5) are recommended; large values (>1) give
        weak anchoring.

    overdispersion : str, default="none"
        Gene-specific overdispersion model.  ``"none"`` uses the
        standard Negative Binomial.  ``"bnb"`` uses the Beta Negative
        Binomial, adding a per-gene concentration parameter that
        allows heavier-than-NB tails.

    overdispersion_prior : str, default="horseshoe"
        Hierarchical prior for the BNB concentration parameter
        (``kappa_g``).  Controls shrinkage toward the NB limit.
        Only used when ``overdispersion`` is not ``"none"``.
        Accepted values: ``"horseshoe"``, ``"neg"``.
    overdispersion_dataset_prior : str, default="none"
        Dataset-level hierarchical prior for BNB concentration
        (``kappa_{d,g}``) in multi-dataset mode. Accepted values:
        ``"none"``, ``"gaussian"``, ``"horseshoe"``, ``"neg"``.
        Requires ``dataset_key``, ``n_datasets>=2``,
        ``unconstrained=True``, and ``overdispersion="bnb"``.

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

    joint_params : List[str], optional
        List of gene-specific parameter names to model jointly via a
        JointLowRankGuide (e.g., ``["mu", "phi"]``). Parameters in this
        list share a single low-rank covariance structure that captures
        cross-parameter correlations. Requires ``guide_rank`` to be set.
        Typically used with ``p_prior='gaussian'`` where multiple
        parameters become gene-specific.

    dense_params : List[str], optional
        Subset of ``joint_params`` that receive full cross-gene low-rank
        coupling. Non-dense joint params get only gene-local conditioning
        on dense params (per-gene regression + per-gene Cholesky among
        non-dense params). When ``None`` or equal to ``joint_params``,
        the standard fully-dense JointLowRankGuide is used.
        Example: ``joint_params=["mu", "phi", "gate"],
        dense_params=["mu"]`` gives ``mu`` cross-gene correlations while
        ``phi`` and ``gate`` only couple to ``mu`` at the same gene.

    guide_flow : str, optional
        Normalizing-flow type for the variational guide. Mutually exclusive
        with ``guide_rank``. When set, uses a ``NormalizingFlowGuide``
        (per-parameter) or ``JointNormalizingFlowGuide`` (when combined
        with ``joint_params``). Supported types: ``"spline_coupling"``,
        ``"affine_coupling"``, ``"maf"``, ``"iaf"``.

    guide_flow_num_layers : int, default=4
        Number of flow layers in the normalizing-flow guide.

    guide_flow_hidden_dims : List[int], optional
        Hidden dimensions for the conditioner network in each flow layer.
        Default is ``[64, 64]``.

    guide_flow_activation : str, default="relu"
        Activation function for flow conditioner MLPs. Supported values:
        ``"relu"``, ``"gelu"``, ``"silu"``, ``"swish"``, ``"tanh"``,
        ``"elu"``, ``"leaky_relu"``, ``"softplus"``.

    guide_flow_n_bins : int, default=8
        Number of spline bins (only used when ``guide_flow="spline_coupling"``).

    guide_flow_mixture_strategy : str, default="independent"
        Strategy for handling mixture components (and datasets) in
        flow guides.  ``"independent"`` creates a separate FlowChain
        per component — most expressive.  ``"shared"`` uses a single
        FlowChain conditioned on a one-hot component index — more
        parameter-efficient.  Ignored when no mixture / dataset axes.

    guide_flow_zero_init : bool, default=True
        Zero-initialize the conditioner output layer so the flow starts as an
        identity transform.  Prevents log-determinant overflow at init in
        high-dimensional flows.

    guide_flow_layer_norm : bool, default=True
        Apply ``nn.LayerNorm`` after each hidden Dense layer in the conditioner
        MLP.  Stabilizes activations when input fan-in is large.

    guide_flow_residual : bool, default=True
        Add residual (skip) connections between consecutive hidden layers of the
        same width in the conditioner MLP.

    guide_flow_soft_clamp : bool, default=True
        Use a smooth asymmetric ``arctan``-based clamp on the affine coupling
        log-scale (Andrade 2024) instead of hard ``jnp.clip``.  Preserves
        gradients at the boundary and tightly bounds per-layer expansion.

    guide_flow_loft : bool, default=True
        Apply a LOFT (Log Soft Extension) layer and a trainable final affine
        after all coupling layers.  LOFT compresses extreme sample magnitudes
        logarithmically while preserving identity near zero; the final affine
        re-expands the range to match the target posterior's scale.

    priors : Dict[str, Any], optional
        Dictionary of prior hyperparameters keyed by parameter name. Values
        should be tuples of prior hyperparameters. Example: {"p": (1.0, 1.0),
        "r": (0.0, 1.0)}.  For ``"mixing"``, a single scalar is broadcast
        to all ``n_components`` (symmetric Dirichlet), e.g.
        ``{"mixing": 5.0}`` is equivalent to ``{"mixing": (5.0, 5.0, 5.0)}``
        for a 3-component model.

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

    optimizer_config : Dict[str, Any], optional
        Serializable optimizer specification for SVI/VAE. This is passed to
        ``SVIConfig.optimizer_config`` and used to build a NumPyro optimizer
        at runtime. Minimal form:
        ``{"name": "adam", "step_size": 1e-3}``.
        Supported names: ``"adam"``, ``"clipped_adam"``, ``"adagrad"``,
        ``"rmsprop"``, ``"sgd"``, ``"momentum"``.
        If ``None``, engine defaults are used unless ``inference_config``
        provides an explicit optimizer object.

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

    restore_best : bool, default=False
        Track the best (lowest smoothed loss) variational parameters during
        training and restore them at the end, regardless of whether early
        stopping is configured or triggered.  When True and no
        ``early_stopping`` config is provided, a minimal internal config is
        created to enable best-state tracking.

    cells_axis : int, default=0
        Axis for cells in count matrix. 0 means cells are rows (n_cells,
        n_genes).

    layer : str, optional
        Layer in AnnData to use for counts. If None, uses .X.

    auto_downgrade_single_dataset_hierarchy : bool, default=True
        Whether to automatically downgrade dataset-level hierarchical flags
        when ``dataset_key`` resolves to a single dataset.
        When enabled and ``n_datasets == 1``:
        - ``mu_dataset_prior`` is downgraded to ``'none'``.
        - ``p_dataset_prior`` with ``p_dataset_mode='scalar'`` is downgraded
          to ``'none'``.
        - ``p_dataset_prior`` with ``p_dataset_mode`` in
          ``{'gene_specific','two_level'}`` is promoted to
          ``p_prior`` (gene-level).
        - ``gate_dataset_prior`` is promoted to ``gate_prior``
          (gene-level).
        - ``overdispersion_dataset_prior`` is downgraded to ``'none'``
          because the dataset axis collapses in single-dataset mode.
        A ``UserWarning`` is emitted when any downgrade is applied.

    dataset_mixing : bool, optional
        Whether to use dataset-specific mixture weights in multi-dataset
        mixture models. If ``None`` (default), SCRIBE enables dataset-specific
        mixing automatically when ``n_datasets >= 2`` and uses global mixing
        otherwise. Set to ``False`` to opt out and keep one global mixing
        vector shared by all datasets.

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

    svi_init : ScribeSVIResults, optional
        SVI results to use for initializing MCMC chains.  When provided,
        MAP estimates are extracted via ``get_map(use_mean=True,
        canonical=True)`` and converted to the target parameterization
        using ``compute_init_values``.  The resulting values are injected
        as ``init_to_value`` into the NUTS kernel, so all chains start
        near the SVI optimum.  Only valid when
        ``inference_method="mcmc"``.

        Cross-parameterization initialization is fully supported: for
        example, SVI run with ``parameterization="linked"`` can initialize
        MCMC with ``parameterization="odds_ratio"``.

    enable_x64 : bool, optional
        Whether to run inference in float64 (double) precision.  When
        ``None`` (the default), the effective value is determined by the
        inference method:

        - **MCMC** → ``True`` — Hamiltonian dynamics in NUTS benefit from
          double precision for numerical stability during leapfrog
          integration and mass-matrix adaptation.
        - **SVI / VAE** → ``False`` — float32 is sufficient and faster.

        Pass an explicit ``True`` or ``False`` to override the default
        for any method.  The setting is implemented via a
        ``jax.enable_x64()`` context manager so it does not permanently
        alter the JAX global configuration.

    model_config : ModelConfig, optional
        Fully configured model configuration object.
        If provided, overrides model, parameterization, unconstrained,
        n_components, mixture_params, guide_rank, and priors.

    inference_config : InferenceConfig, optional
        Fully configured inference configuration object.
        If provided, overrides inference_method, n_steps, batch_size,
        optimizer_config, stable_update, log_progress_lines, n_samples,
        n_warmup, and n_chains.

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

    Initialize MCMC from SVI results (same or different parameterization):

    >>> svi_results = scribe.fit(adata, model="nbdm", parameterization="linked")
    >>> mcmc_results = scribe.fit(
    ...     adata,
    ...     model="nbdm",
    ...     parameterization="odds_ratio",
    ...     inference_method="mcmc",
    ...     svi_init=svi_results,
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

    # Validate svi_init: only allowed with MCMC
    if svi_init is not None:
        _effective_method = (
            inference_method.lower()
            if inference_config is None
            else inference_config.method.value
        )
        if _effective_method != "mcmc":
            raise ValueError(
                f"svi_init is only supported with inference_method='mcmc', "
                f"got '{_effective_method}'."
            )

    # ==========================================================================
    # Step 2: Process data
    # ==========================================================================
    data_config = DataConfig(cells_axis=cells_axis, layer=layer)
    count_data, adata, n_cells, n_genes = process_counts_data(
        counts, data_config
    )

    # ==========================================================================
    # Step 2b: Build dataset indices (if multi-dataset model)
    # ==========================================================================
    dataset_indices = None
    if dataset_key is not None:
        if adata is None:
            raise ValueError(
                "dataset_key requires counts to be an AnnData object "
                "(not a raw array), so that adata.obs can be read."
            )
        import numpy as np

        ds_col = adata.obs[dataset_key]
        # Convert to categorical integer codes
        ds_cat = ds_col.astype("category")
        ds_codes = ds_cat.cat.codes.values
        _inferred_n_datasets = len(ds_cat.cat.categories)
        if n_datasets is not None and n_datasets != _inferred_n_datasets:
            raise ValueError(
                f"n_datasets={n_datasets} but dataset_key "
                f"'{dataset_key}' has {_inferred_n_datasets} unique "
                f"values: {list(ds_cat.cat.categories)}"
            )
        n_datasets = _inferred_n_datasets
        dataset_indices = jnp.asarray(np.asarray(ds_codes, dtype=np.int32))

    # Normalize dataset-level hierarchical flags for the single-dataset edge
    # case so callers can safely pass dataset-level options in mixed cohorts.
    if (
        auto_downgrade_single_dataset_hierarchy
        and n_datasets == 1
        and dataset_indices is not None
    ):
        downgrade_messages: List[str] = []

        # Dataset-level mu has no meaningful single-dataset hierarchy, so
        # disable it explicitly.
        if mu_dataset_prior != "none":
            mu_dataset_prior = "none"
            downgrade_messages.append("mu_dataset_prior -> 'none'")

        # Map dataset-level p modes to their single-dataset equivalents:
        # scalar -> shared p/phi; gene_specific/two_level -> gene-level hierarchy.
        if p_dataset_prior != "none":
            if p_dataset_mode == "scalar":
                p_dataset_prior = "none"
                downgrade_messages.append(
                    "p_dataset_prior='scalar' mode -> 'none'"
                )
            else:
                # gene_specific/two_level → promote to gene-level prior
                p_prior = p_dataset_prior
                p_dataset_prior = "none"
                downgrade_messages.append(
                    f"p_dataset_prior -> p_prior='{p_prior}'"
                )

        # Dataset-level gate also collapses to the gene-level hierarchy in
        # the single-dataset setting.
        if gate_dataset_prior != "none":
            gate_prior = gate_dataset_prior
            gate_dataset_prior = "none"
            downgrade_messages.append(
                f"gate_dataset_prior -> gate_prior='{gate_prior}'"
            )
        # Dataset-level overdispersion has no meaningful single-dataset
        # hierarchy, so disable it explicitly.
        if overdispersion_dataset_prior != "none":
            overdispersion_dataset_prior = "none"
            downgrade_messages.append("overdispersion_dataset_prior -> 'none'")

        if downgrade_messages:
            # Collapse back to single-dataset mode in ModelConfig, which uses
            # n_datasets=None as the canonical non-multi-dataset state.
            n_datasets = None
            # Single-dataset mode has no dataset axis for mixing.
            if dataset_mixing is None:
                dataset_mixing = False
            warnings.warn(
                "Detected a single dataset from dataset_key "
                f"'{dataset_key}'. Applied automatic hierarchy downgrade: "
                + "; ".join(downgrade_messages),
                UserWarning,
                stacklevel=2,
            )

    # Enforce that dataset-level hierarchical options are only used when
    # explicit cell-to-dataset mapping is available for indexing.
    uses_dataset_level_hierarchy = (
        mu_dataset_prior != "none"
        or p_dataset_prior != "none"
        or gate_dataset_prior != "none"
        or overdispersion_dataset_prior != "none"
    )
    if uses_dataset_level_hierarchy and dataset_indices is None:
        raise ValueError(
            "Dataset-level hierarchical priors "
            "(mu_dataset_prior, p_dataset_prior, "
            "gate_dataset_prior, overdispersion_dataset_prior) "
            "require dataset_key so cells can "
            "be mapped to datasets. Provide dataset_key as an adata.obs "
            "column when using dataset-level hierarchical priors."
        )

    # ==========================================================================
    # Step 2c: Build annotation prior logits (if requested)
    # ==========================================================================
    annotation_prior_logits = None
    effective_mixture_params = mixture_params
    _label_map = None
    _component_mapping = None
    if annotation_key is not None:
        if adata is None:
            raise ValueError(
                "annotation_key requires counts to be an AnnData object "
                "(not a raw array), so that adata.obs can be read."
            )
        # Resolve n_components — may come from explicit kwarg, from a
        # user-supplied model_config, or be inferred from annotations.
        _n_comp = n_components
        _n_comp_inferred = False
        if _n_comp is None and model_config is not None:
            _n_comp = model_config.n_components
        _min_cells = annotation_min_cells or 0
        if _n_comp is None:
            # Infer from the number of unique non-null annotation labels
            _n_comp = _count_unique_labels(
                adata, annotation_key, min_cells=_min_cells
            )
            _n_comp_inferred = True

        # If annotation-driven inference collapses to <=1 surviving class
        # after min_cells filtering, downgrade to the canonical non-mixture
        # path instead of raising from mixture validation.
        if _n_comp_inferred and _n_comp <= 1:
            # When the annotation filter leaves <=1 class, we must force
            # non-mixture mode and clear any component-only prior settings.
            # This keeps auto-downgrade behavior compatible with enum-style
            # prior flags (e.g. mu_prior="gaussian"), which otherwise fail
            # ModelConfig validation in non-mixture mode.
            downgraded_messages = []
            n_components = None
            effective_mixture_params = None
            if (
                _normalize_prior_type_name(mu_prior)
                != HierarchicalPriorType.NONE.value
            ):
                _mu_prior_old = _normalize_prior_type_name(mu_prior)
                mu_prior = HierarchicalPriorType.NONE.value
                downgraded_messages.append(
                    f"mu_prior='{_mu_prior_old}' -> 'none'"
                )

            _downgrade_suffix = (
                f"; {'; '.join(downgraded_messages)}"
                if downgraded_messages
                else ""
            )
            warnings.warn(
                "annotation_key/annotation_min_cells left <=1 surviving "
                "annotation class after filtering. "
                "Auto-downgrading to non-mixture mode "
                "(n_components=None, mixture_params ignored"
                f"{_downgrade_suffix}).",
                UserWarning,
                stacklevel=2,
            )
        else:
            n_components = _n_comp  # propagate so ModelConfig picks it up

            # When both annotation_key and dataset_key are present, build
            # a ComponentMapping that identifies shared vs exclusive
            # components across datasets.  The mapping's component_order
            # is passed to build_annotation_prior_logits for consistent
            # label → index alignment.
            _component_mapping = None
            _effective_component_order = annotation_component_order
            _shared_comp_override = None
            if model_config is not None:
                _shared_comp_override = getattr(
                    model_config, "shared_components", None
                )

            if dataset_key is not None:
                _component_mapping = build_component_mapping(
                    adata=adata,
                    annotation_key=annotation_key,
                    dataset_key=dataset_key,
                    min_cells=_min_cells,
                    shared_components=_shared_comp_override,
                )
                # Use the mapping's ordering for the annotation prior so
                # component indices are consistent.
                if _effective_component_order is None:
                    _effective_component_order = (
                        _component_mapping.component_order
                    )
                # The union may differ from the original _n_comp count;
                # update n_components if it was inferred.
                if _n_comp_inferred:
                    n_components = _component_mapping.n_components
                    _n_comp = n_components

            annotation_prior_logits, _label_map = build_annotation_prior_logits(
                adata=adata,
                obs_key=annotation_key,
                n_components=_n_comp,
                confidence=annotation_confidence,
                component_order=_effective_component_order,
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
            mu_prior=mu_prior,
            p_prior=p_prior,
            gate_prior=gate_prior,
            n_datasets=n_datasets,
            dataset_params=dataset_params,
            dataset_mixing=dataset_mixing,
            mu_dataset_prior=mu_dataset_prior,
            p_dataset_prior=p_dataset_prior,
            p_dataset_mode=p_dataset_mode,
            gate_dataset_prior=gate_dataset_prior,
            overdispersion_dataset_prior=overdispersion_dataset_prior,
            horseshoe_tau0=horseshoe_tau0,
            horseshoe_slab_df=horseshoe_slab_df,
            horseshoe_slab_scale=horseshoe_slab_scale,
            neg_u=neg_u,
            neg_a=neg_a,
            neg_tau=neg_tau,
            mu_eta_prior=mu_eta_prior,
            mu_mean_anchor=mu_mean_anchor,
            mu_mean_anchor_sigma=mu_mean_anchor_sigma,
            overdispersion=overdispersion,
            overdispersion_prior=overdispersion_prior,
            guide_rank=guide_rank,
            joint_params=joint_params,
            dense_params=dense_params,
            guide_flow=guide_flow,
            guide_flow_num_layers=guide_flow_num_layers,
            guide_flow_hidden_dims=guide_flow_hidden_dims,
            guide_flow_activation=guide_flow_activation,
            guide_flow_n_bins=guide_flow_n_bins,
            guide_flow_mixture_strategy=guide_flow_mixture_strategy,
            guide_flow_zero_init=guide_flow_zero_init,
            guide_flow_layer_norm=guide_flow_layer_norm,
            guide_flow_residual=guide_flow_residual,
            guide_flow_soft_clamp=guide_flow_soft_clamp,
            guide_flow_loft=guide_flow_loft,
            n_components=n_components,
            mixture_params=effective_mixture_params,
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
    # Step 3b: Inject shared_component_indices into model_config
    # ==========================================================================
    # When a ComponentMapping was built (annotation_key + dataset_key),
    # attach the shared component indices so the factory can build
    # per-component scale masking in the dataset hierarchical specs.
    if _component_mapping is not None:
        model_config = model_config.model_copy(
            update={
                "shared_component_indices": _component_mapping.shared_indices,
            }
        )

    # ==========================================================================
    # Step 3c: Compute data-informed mean anchor (if enabled)
    # ==========================================================================
    # When mu_mean_anchor is True, compute per-gene log-anchor centers
    # from the observed count matrix and store them in the priors dict.
    # The factory reads these to build AnchoredNormalSpec for log_mu_loc.
    if model_config.mu_mean_anchor:
        from .models.model_utils import compute_mu_anchor

        import numpy as _np

        _counts_np = _np.asarray(count_data)
        _lib_sizes = _counts_np.sum(axis=1)

        # Extract M_0 from capture prior if available (VCP models)
        _extra = getattr(model_config.priors, "__pydantic_extra__", None) or {}
        _eta_capture = _extra.get("eta_capture")
        _total_mrna = None
        if _eta_capture is not None:
            import math as _math

            _total_mrna = _math.exp(_eta_capture[0])

        _log_anchors = compute_mu_anchor(
            counts=_counts_np,
            library_sizes=_lib_sizes,
            total_mrna_mean=_total_mrna,
            epsilon=1e-3,
        )

        # Inject into priors as mu_anchor_centers (keep as numpy array
        # to avoid expensive Python-tuple round-trips for large gene counts)
        _updated_priors = dict(_extra)
        _updated_priors["mu_anchor_centers"] = _log_anchors
        from .models.config.groups import PriorOverrides

        model_config = model_config.model_copy(
            update={"priors": PriorOverrides(**_updated_priors)}
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
                optimizer_config=optimizer_config,
                stable_update=stable_update,
                log_progress_lines=log_progress_lines,
                early_stopping=early_stop_config,
                restore_best=restore_best,
            )
            inference_config = InferenceConfig.from_svi(svi_config)
        elif method == InferenceMethod.MCMC:
            if early_stopping is not None:
                warnings.warn(
                    "early_stopping is only supported for SVI and VAE "
                    "inference methods. Ignoring for MCMC.",
                    UserWarning,
                )
            if optimizer_config is not None:
                warnings.warn(
                    "optimizer_config is only supported for SVI and VAE "
                    "inference methods. Ignoring for MCMC.",
                    UserWarning,
                )

            # Build mcmc_kwargs, optionally injecting SVI init strategy
            svi_init_kwargs: Dict[str, Any] = {}
            if svi_init is not None:
                from numpyro.infer.initialization import init_to_value

                from .mcmc._init_from_svi import (
                    clamp_init_values,
                    compute_init_values,
                )

                if svi_init.model_config.base_model != model_config.base_model:
                    warnings.warn(
                        f"SVI base model '{svi_init.model_config.base_model}' "
                        f"differs from MCMC target "
                        f"'{model_config.base_model}'.",
                        UserWarning,
                    )
                # When parameterizations match, pass native MAP values
                # directly to avoid a lossy canonical round-trip (float32
                # precision loss at distribution boundaries).
                same_param = (
                    svi_init.model_config.parameterization
                    == model_config.parameterization
                )
                svi_map = svi_init.get_map(
                    use_mean=True, canonical=not same_param
                )
                init_values = (
                    svi_map
                    if same_param
                    else compute_init_values(svi_map, model_config)
                )
                # Clamp to strict interior of support — float32 SVI MAP
                # estimates can sit exactly on boundaries (e.g. phi_capture=0)
                init_values = clamp_init_values(init_values)

                # Promote to float64 when MCMC will run under x64 (the
                # default).  SVI stores float32; NUTS tree-building requires
                # matching dtypes across all JAX cond branches.
                if enable_x64 is not False:
                    import jax

                    with jax.enable_x64(True):
                        init_values = {
                            k: jnp.asarray(v, dtype=jnp.float64)
                            for k, v in init_values.items()
                        }

                svi_init_kwargs["init_strategy"] = init_to_value(
                    values=init_values
                )

                # Free SVI arrays that are no longer needed — every bit
                # of GPU memory matters for high-dimensional MCMC.
                del svi_map, init_values, svi_init
                import gc

                gc.collect()

            mcmc_config = MCMCConfig(
                n_samples=n_samples,
                n_warmup=n_warmup,
                n_chains=n_chains,
                mcmc_kwargs=svi_init_kwargs or None,
            )
            inference_config = InferenceConfig.from_mcmc(mcmc_config)
        elif method == InferenceMethod.VAE:
            # VAE uses SVI config
            svi_config = SVIConfig(
                n_steps=n_steps,
                batch_size=effective_batch_size,
                optimizer_config=optimizer_config,
                stable_update=stable_update,
                log_progress_lines=log_progress_lines,
                early_stopping=early_stop_config,
                restore_best=restore_best,
            )
            inference_config = InferenceConfig.from_vae(svi_config)
        else:
            raise ValueError(f"Unknown inference method: {method}")
    else:
        # Validate that inference_config matches model_config
        validate_inference_config_match(model_config, inference_config)

    # ==========================================================================
    # Step 5: Resolve float64 precision default per inference method
    # ==========================================================================
    # MCMC (NUTS) benefits from float64 for Hamiltonian dynamics stability;
    # SVI/VAE run faster in float32 and rarely need double precision.
    if enable_x64 is None:
        effective_x64 = inference_config.method == InferenceMethod.MCMC
    else:
        effective_x64 = enable_x64

    # ==========================================================================
    # Step 6: Run inference
    # ==========================================================================
    results = _run_inference(
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
        dataset_indices=dataset_indices,
        enable_x64=effective_x64,
    )

    # Attach annotation metadata to the results for downstream use
    # (e.g. label-based component matching in DE).
    if _label_map is not None and hasattr(results, "_label_map"):
        object.__setattr__(results, "_label_map", _label_map)
    if _component_mapping is not None and hasattr(
        results, "_component_mapping"
    ):
        object.__setattr__(results, "_component_mapping", _component_mapping)

    return results
