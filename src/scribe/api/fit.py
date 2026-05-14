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
>>> # Simplest usage - default model is NBVCP (variable capture on)
>>> results = scribe.fit(adata)
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData

from ..models.config import (
    AmortizationConfig,
    EarlyStoppingConfig,
    InferenceConfig,
    KLAnnealingConfig,
    ModelConfig,
)

from ..svi.results import ScribeSVIResults

from .types import ScribeResults
from .context import FitContext

from .stages.model_flags import resolve_model_flags
from .stages.validation import validate_inputs
from .stages.data_processing import process_data_and_datasets
from .stages.gene_coverage import apply_gene_coverage_and_alr
from .stages.lnm_priors import resolve_lnm_auto_priors
from .stages.annotation_priors import build_annotation_priors
from .stages.model_config_build import build_model_config
from .stages.inference_config_build import build_inference_config
from .stages.run_inference import dispatch_inference
from .stages.result_postprocess import postprocess_results


def fit(
    counts: Union[jnp.ndarray, "AnnData"],
    model: str = "nbvcp",
    # Compositional model flags (override model when set)
    variable_capture: Optional[bool] = None,
    zero_inflation: Optional[bool] = None,
    # Model options
    parameterization: str = "canonical",
    unconstrained: bool = False,
    expression_prior: str = "none",
    prob_prior: str = "none",
    zero_inflation_prior: str = "none",
    # Multi-dataset hierarchy options
    n_datasets: Optional[int] = None,
    dataset_key: Optional[str] = None,
    dataset_params: Optional[List[str]] = None,
    dataset_mixing: Optional[bool] = None,
    expression_dataset_prior: str = "none",
    prob_dataset_prior: str = "none",
    prob_dataset_mode: str = "gene_specific",
    zero_inflation_dataset_prior: str = "none",
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
    capture_scaling_prior: str = "none",
    # Data-informed mean anchoring prior
    expression_anchor: bool = False,
    expression_anchor_sigma: float = 0.3,
    # Gene-specific overdispersion beyond the NB family
    overdispersion: str = "none",
    overdispersion_prior: str = "horseshoe",
    # Diagonal residual noise mode for the latent log-rate
    # decomposition (see ``ModelConfig.d_mode``).  When ``None``,
    # ``api.fit`` resolves a sensible per-model default:
    #   - PLN: ``"learned"`` so the generative covariance is
    #     ``W W' + diag(d)`` (matches the Laplace inference path,
    #     which has always learned ``d``).
    #   - LNM and others: ``"low_rank"`` (legacy behavior).
    # Pass an explicit string to override.
    d_mode: Optional[str] = None,
    alr_reference_idx: Optional[int] = None,
    n_components: Optional[int] = None,
    mixture_params: Optional[Union[str, List[str]]] = "all",
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
    priors: Optional[Dict[str, Any]] = None,
    # Latent factor dimension.  ``latent_dim`` is the rank of the
    # low-rank loadings matrix ``W ∈ R^{G x k}`` in the generative
    # model — the ``k`` in PLN / LNM / NBLN's
    # ``Σ = W W^T + diag(d)``.  Same quantity drives the VAE
    # encoder/decoder latent dimensionality.  When ``inference_method``
    # is ``"laplace"`` or ``"svi"`` (non-amortized), this is the model
    # rank; when ``"vae"``, it's both the model rank and the encoder
    # latent.  ``None`` falls back to ``vae_latent_dim`` (legacy alias,
    # default 10).  Passing both raises ValueError.
    latent_dim: Optional[int] = None,
    # VAE architecture options (when inference_method="vae")
    # Legacy alias for ``latent_dim``.  Retained for backward compat;
    # new code should use ``latent_dim`` directly.  ``None`` here means
    # "not explicitly set"; default falls back to ``10`` when neither
    # ``latent_dim`` nor ``vae_latent_dim`` is supplied.
    vae_latent_dim: Optional[int] = None,
    vae_encoder_hidden_dims: Optional[List[int]] = None,
    vae_decoder_hidden_dims: Optional[List[int]] = None,
    vae_activation: Optional[str] = None,
    vae_input_transform: str = "log1p",
    vae_standardize: Optional[bool] = None,
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
    # KL annealing options (for VAE only -- automatically defaulted ON
    # for VAE-mode fits; pass an explicit ``KLAnnealingConfig`` or set
    # ``kl_annealing_warmup`` to customise; pass
    # ``KLAnnealingConfig(enabled=False)`` to disable).
    kl_annealing: Optional[
        Union["KLAnnealingConfig", Dict[str, Any], bool]
    ] = None,
    kl_annealing_warmup: Optional[int] = None,
    # Laplace-specific overrides -- accepts a dict of ``LaplaceConfig``
    # field overrides (e.g. ``{"n_newton_steps": 15, "damping": 1e-3,
    # "newton_tolerance": 1e-3}``) or a fully-built ``LaplaceConfig``
    # instance.  Top-level kwargs (n_steps, batch_size, optimizer_config,
    # early_stopping, restore_best, log_progress_lines) populate the
    # corresponding fields of the resulting LaplaceConfig; anything in
    # ``laplace_config`` overrides those defaults.
    laplace_config: Optional[Union["LaplaceConfig", Dict[str, Any]]] = None,
    # Data options
    cells_axis: int = 0,
    layer: Optional[str] = None,
    gene_coverage: Optional[float] = None,
    seed: int = 42,
    # Annotation prior options (for mixture models)
    annotation_key: Optional[Union[str, List[str]]] = None,
    annotation_confidence: float = 3.0,
    annotation_component_order: Optional[List[str]] = None,
    annotation_min_cells: Optional[int] = None,
    # SVI-to-MCMC initialization
    svi_init: Optional[ScribeSVIResults] = None,
    # SVI-to-Laplace informative-prior cascade (NBLN only).  When set,
    # empirical Gaussian priors on ``r``, ``mu``, and (when capture is
    # supplied) ``eta`` are derived from posterior samples of the SVI
    # results object and used as informative priors during the NBLN
    # Laplace fit.  See :mod:`scribe.laplace.priors` for details and
    # caveats (gene-identity safeguards, amortized-capture handling,
    # capture-mode trichotomy).  Recommended ``tau`` defaults to 1.0
    # (trust SVI exactly); raise to 2-3 in noisy / sparse regimes.
    informative_priors_from: Optional[ScribeSVIResults] = None,
    informative_priors_tau: float = 1.0,
    informative_priors_n_samples: int = 1000,
    informative_priors_verbose: bool = True,
    # Phase-2 freeze: which NBLN globals are fixed at the SVI cascade's
    # MAP rather than refined during the M-step.  Default ("r", "eta")
    # eliminates the rigid-translation gauge degeneracy and yields the
    # cleanest cross-gene correlation structure in W.  Pass () to
    # disable freezing entirely (Phase-1 soft cascade only).
    #
    # Accepts either the internal short names ("r", "mu", "eta") or
    # their descriptive aliases:
    #   - "r"   <-> "dispersion"
    #   - "mu"  <-> "expression" or "mean_expression"
    #   - "eta" <-> "capture_efficiency"
    # Both forms work, e.g. ("dispersion", "capture_efficiency") is
    # equivalent to ("r", "eta").  Passing both an internal name and
    # its alias (e.g. ("r", "dispersion")) raises ValueError.  See
    # FREEZE_KEY_ALIASES in scribe.models.config.parameter_mapping for
    # the canonical mapping.
    #
    # See paper/_diffexp_nbln_robustness.qmd and the Cascade-parameter
    # freeze subsection in paper/_nb_lognormal.qmd for the rationale.
    informative_priors_freeze: tuple = ("r", "eta"),
    # DEPRECATED: shrinkage prior on the loadings matrix W.
    # The preferred API is to pass the strategy spec inside the
    # ``priors`` dict under the descriptive ``"loadings"`` key:
    #   priors = {"loadings": {"type": "horseshoe_columnwise",
    #                          "tau_scale": 1.0}}
    # This top-level kwarg is retained for backward compatibility and
    # emits a ``DeprecationWarning`` when used.  Passing both the dict
    # form and the kwarg raises ``ValueError``.
    w_prior: Optional[dict] = None,
    # Float64 precision -- defaults to True for MCMC, False for SVI/VAE
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

    model : str, default="nbvcp"
        Model type to use.  The default ``"nbvcp"`` includes
        cell-specific capture probability, which is appropriate for the
        vast majority of scRNA-seq datasets.  In most cases it is
        simpler to control composition with ``variable_capture`` and
        ``zero_inflation`` instead of setting this directly.  Accepted
        strings:

            - ``"nbdm"``: Negative Binomial (base, no capture channel)
            - ``"lnm"``: NB total counts x logistic-normal multinomial
              compositions (VAE-only; uses parameterization
              ``logistic_normal``)
            - ``"lnmvcp"``: Like ``"lnm"`` but with per-cell variable
              capture probability on the totals NB submodel
            - ``"zinb"``: Zero-Inflated NB
            - ``"nbvcp"``: NB with Variable Capture Probability
            - ``"zinbvcp"``: ZINB with Variable Capture Probability
            - ``"pln"``: Poisson Log-Normal

    variable_capture : bool or None, default=None
        Add cell-specific capture probability to the model.  When set,
        the ``model`` string is derived automatically (``True`` is
        implied when neither flag nor ``model`` is specified, since the
        default model is ``"nbvcp"``):

        For standard (NBDM-family) parameterizations:

            - ``variable_capture=False, zero_inflation=False`` ->
              ``"nbdm"``
            - ``variable_capture=True, zero_inflation=False`` ->
              ``"nbvcp"``
            - ``variable_capture=False, zero_inflation=True`` ->
              ``"zinb"``
            - ``variable_capture=True, zero_inflation=True`` ->
              ``"zinbvcp"``

        For the LNM family (``model="lnm"`` / ``"lnmvcp"``):

            - ``variable_capture=False`` -> ``"lnm"``
            - ``variable_capture=True``  -> ``"lnmvcp"``
            - ``zero_inflation=True`` raises ``ValueError``

        For the PLN family (``model="pln"``):

            - Capture is an internal flag, not a separate model string.
              If ``variable_capture=True`` and capture priors are
              provided (``priors={"capture_efficiency": ...}`` or
              ``priors={"organism": ...}``), capture is silently
              activated.
            - If ``variable_capture=True`` but *no* capture prior is
              provided, a warning is emitted and PLN runs without
              capture.
            - ``zero_inflation=True`` raises ``ValueError``.

        An explicit ``model=`` that conflicts with the flags raises
        ``ValueError``.

    zero_inflation : bool or None, default=None
        Add a per-gene zero-inflation gate to the model.  See
        ``variable_capture`` for the resolution table.

    Parameterization and core priors
    --------------------------------
    parameterization : str, default="canonical"
        Parameterization scheme:

            - ``"canonical"`` (or ``"standard"``): Sample p ~ Beta,
              r ~ LogNormal directly
            - ``"linked"`` (or ``"mean_prob"``): Sample p ~ Beta,
              mu ~ LogNormal, derive r
            - ``"odds_ratio"`` (or ``"mean_odds"``): Sample
              phi ~ BetaPrime, mu ~ LogNormal

    unconstrained : bool, default=False
        If True, use Normal+transform instead of constrained
        distributions.  This can help with optimization in some cases.

    expression_prior : str, default="none"
        Gene-level hierarchical prior for mu (or r) across mixture
        components.  Per-component means are drawn from a shared
        gene-level population distribution per gene, providing adaptive
        shrinkage.  Requires ``unconstrained=True`` and
        ``n_components >= 2``.  Accepted values: ``"none"``,
        ``"gaussian"``, ``"horseshoe"``, ``"neg"``.

    prob_prior : str, default="none"
        Gene-level hierarchical prior for the probability parameter
        (``p`` in canonical/linked parameterizations, ``phi`` in
        mean-odds).  Provides adaptive shrinkage across genes and
        requires ``unconstrained=True``.  Accepted values:
        ``"none"``, ``"gaussian"``, ``"horseshoe"``, ``"neg"``.

    zero_inflation_prior : str, default="none"
        Gene-level hierarchical prior for the zero-inflation gate.
        Only used for zero-inflated models and requires
        ``unconstrained=True``.  Accepted values:
        ``"none"``, ``"gaussian"``, ``"horseshoe"``, ``"neg"``.

    expression_anchor : bool, default=False
        Enable data-informed anchoring prior on the biological mean
        ``mu_g``.  When True, per-gene prior centers are computed from
        the observed sample means and average capture probability,
        anchoring ``log(mu_g) ~ N(log(u_bar_g / nu_bar), sigma^2)``.
        This resolves the mu-phi degeneracy in the negative binomial.
        Automatically enables ``unconstrained=True``.  For VCP models,
        ``priors.eta_capture`` or ``priors.organism`` must be set so
        that ``nu_bar`` can be estimated.

    expression_anchor_sigma : float, default=0.3
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

    d_mode : str or None, default=None
        Diagonal residual-noise mode for models that decompose the
        latent log-rate as ``y = mu + W z`` (LNM, LNMVCP) or
        ``x = mu + W z`` (PLN).  ``"low_rank"`` skips the diagonal
        residual; ``"learned"`` adds a learnable per-gene ``diag(d)``
        term so the covariance is ``W W' + diag(d)``.  When ``None``
        (the default), ``api.fit`` resolves to ``"learned"``.

    alr_reference_idx : int or None, default=None
        Only for ``model="lnm"`` / ``"lnmvcp"``: zero-based index of
        the ALR reference gene (denominator).  ``None`` selects
        automatically from the count matrix (minimum-variance
        criterion).  Pass an explicit integer to override; ``-1``
        keeps the legacy last-gene reference.

    Multi-dataset hierarchy
    -----------------------
    n_datasets : int, optional
        Number of datasets in multi-dataset mode.  When ``dataset_key``
        is provided, this value is inferred automatically if omitted.

    dataset_key : str, optional
        Column name in ``adata.obs`` that identifies dataset membership
        for each cell (e.g. ``"batch"`` or ``"donor"``).  Required for
        dataset-level hierarchical priors.

    dataset_params : list of str, optional
        Explicit list of model parameters that should carry a dataset
        axis.  When ``None``, resolved from the selected dataset-level
        hierarchy settings.

    dataset_mixing : bool, optional
        Whether to use dataset-specific mixture weights.  ``None``
        (default) enables automatically when ``n_datasets >= 2``.

    expression_dataset_prior : str, default="none"
        Dataset-level hierarchical prior for expression parameters.
        Accepted values: ``"none"``, ``"gaussian"``, ``"horseshoe"``,
        ``"neg"``.

    prob_dataset_prior : str, default="none"
        Dataset-level hierarchical prior for the probability parameter.
        Accepted values: ``"none"``, ``"gaussian"``, ``"horseshoe"``,
        ``"neg"``.

    prob_dataset_mode : str, default="gene_specific"
        Structure of dataset-level probability hierarchy:
        ``"scalar"``, ``"gene_specific"``, or ``"two_level"``.

    zero_inflation_dataset_prior : str, default="none"
        Dataset-level hierarchical prior for zero-inflation gate
        parameters.

    overdispersion_dataset_prior : str, default="none"
        Dataset-level hierarchical prior for BNB concentration
        (``kappa_{d,g}``) in multi-dataset mode.

    auto_downgrade_single_dataset_hierarchy : bool, default=True
        Automatically downgrade dataset-level hierarchical flags when
        ``dataset_key`` resolves to a single dataset.

    Sparsity prior hyperparameters
    ------------------------------
    horseshoe_tau0 : float, default=1.0
        Global shrinkage scale for regularized horseshoe priors.

    horseshoe_slab_df : int, default=4
        Degrees of freedom for the horseshoe slab component.

    horseshoe_slab_scale : float, default=2.0
        Scale of the horseshoe slab component.

    neg_u : float, default=1.0
        Inner Gamma shape parameter for NEG priors.

    neg_a : float, default=1.0
        Outer Gamma shape parameter for NEG priors.

    neg_tau : float, default=1.0
        Global rate/scale parameter for NEG priors.

    capture_scaling_prior : str, default="none"
        Hierarchical prior for per-dataset capture scaling
        (``mu_eta``).  Accepted values: ``"none"``, ``"gaussian"``,
        ``"horseshoe"``, ``"neg"``.

    Mixture and variational guide configuration
    --------------------------------------------
    n_components : int, optional
        Number of mixture components for cell type discovery.
        If ``None``, uses a single-component model.  When
        ``annotation_key`` is provided and ``n_components`` is omitted,
        the number of components is automatically inferred from the
        number of unique non-null annotation labels.

    mixture_params : str or List[str], default="all"
        Which parameters should vary across mixture components.
        Accepts a **semantic shorthand** string or an explicit list
        of internal parameter names.

        **Semantic shorthands** (resolved automatically based on the
        chosen ``parameterization`` and ``model``):

        - ``"all"`` (default) -- every parameter becomes
          component-specific.
        - ``"biological"`` -- only core NB parameters vary.
        - ``"mean"`` -- only the expression-level parameter varies.
        - ``"prob"`` -- only the probability/odds parameter varies.
        - ``"gate"`` -- only the zero-inflation gate varies (ZINB
          only).

        **Explicit list** (power-user): pass a list like
        ``["mu", "phi"]``.  Set to ``None`` to disable mixture
        behaviour entirely.

    guide_rank : int, optional
        Rank for low-rank variational guide on gene-specific
        parameters.  ``None`` uses a mean-field guide.

    joint_params : str or List[str], optional
        Gene-specific parameters to model jointly via a single
        low-rank covariance.  Requires ``guide_rank`` or
        ``guide_flow``.  Accepts the same shorthands as
        ``mixture_params``.

    dense_params : str or List[str], optional
        Subset of ``joint_params`` that receive full cross-gene
        low-rank coupling.

    guide_flow : str, optional
        Normalizing-flow type for the variational guide.  Mutually
        exclusive with ``guide_rank``.  Supported types:
        ``"spline_coupling"``, ``"affine_coupling"``, ``"maf"``,
        ``"iaf"``.

    guide_flow_num_layers : int, default=4
        Number of flow layers in the normalizing-flow guide.

    guide_flow_hidden_dims : List[int], optional
        Hidden dimensions for the conditioner network.  Default is
        ``[64, 64]``.

    guide_flow_activation : str, default="relu"
        Activation function for flow conditioner MLPs.

    guide_flow_n_bins : int, default=8
        Number of spline bins (only for ``"spline_coupling"``).

    guide_flow_mixture_strategy : str, default="independent"
        Strategy for handling mixture components in flow guides.
        ``"independent"`` creates a separate FlowChain per component;
        ``"shared"`` uses a single FlowChain conditioned on a one-hot
        component index.

    guide_flow_zero_init : bool, default=True
        Zero-initialize the conditioner output layer so the flow
        starts as an identity transform.

    guide_flow_layer_norm : bool, default=True
        Apply LayerNorm after each hidden Dense layer in the
        conditioner MLP.

    guide_flow_residual : bool, default=True
        Add residual connections between hidden layers of the same
        width.

    guide_flow_soft_clamp : bool, default=True
        Use smooth asymmetric ``arctan``-based clamp on the affine
        coupling log-scale instead of hard ``jnp.clip``.

    guide_flow_loft : bool, default=True
        Apply a LOFT layer and a trainable final affine after all
        coupling layers.

    guide_flow_log_det_f64 : bool, default=False
        Accumulate the log-determinant Jacobian in float64.  When
        True, ``enable_x64`` is automatically promoted to True.

    Prior overrides and VAE architecture
    -------------------------------------
    priors : Dict[str, Any], optional
        Dictionary of prior hyperparameters keyed by parameter name.
        Most entries are tuples of hyperparameters — e.g.
        ``{"p": (1.0, 1.0), "r": (0.0, 1.0)}``.  For ``"mixing"``,
        a single scalar is broadcast to all ``n_components``.

        Two entries have *dict-shaped* values rather than tuples:

        - ``"capture_efficiency"`` (alias for ``"eta_capture"``): the
          biology-informed capture-anchor prior for PLN/NBLN/LNMVCP.
        - ``"loadings"`` (alias for the W matrix): the **shrinkage
          prior on the low-rank loadings matrix** for PLN/NBLN
          Laplace fits.  Value is a strategy spec — e.g.
          ``{"type": "horseshoe_columnwise", "tau_scale": 1.0}``.
          See ``src/scribe/laplace/_w_priors.py`` for the four
          registered strategies (``none``, ``gaussian``,
          ``horseshoe_columnwise``, ``neg_columnwise``).  When passed,
          this entry is extracted from the dict before downstream
          stages see it, so the rest of the priors dict stays tuple-
          shaped as usual.

    latent_dim : int, optional
        Rank of the low-rank loadings matrix ``W ∈ R^{G × k}`` in the
        generative model — the ``k`` in PLN / NBLN / LNM's
        ``Σ = W W^T + diag(d)``.  Used by all inference methods that
        fit PLN-family models (Laplace, SVI, MCMC, VAE).  For VAE, this
        is also the encoder/decoder latent dimensionality.  ``None``
        (default) falls back to ``vae_latent_dim`` (or ``10`` if neither
        is set).  Passing both ``latent_dim`` and ``vae_latent_dim``
        raises ``ValueError``.

        This is the **preferred** kwarg for setting the latent factor
        dimension.  ``vae_latent_dim`` is retained as a legacy alias
        because the kwarg was originally introduced for VAE inference;
        it works identically but is misnamed for non-VAE methods.

    vae_latent_dim : int, optional
        Legacy alias for ``latent_dim``.  ``None`` (default) means "not
        explicitly set"; resolves to ``latent_dim`` if supplied, else
        ``10``.  Prefer ``latent_dim`` in new code.

    vae_encoder_hidden_dims : List[int], optional
        Hidden layer widths for the VAE encoder network.

    vae_decoder_hidden_dims : List[int], optional
        Hidden layer widths for the VAE decoder network.

    vae_activation : str, optional
        Activation function used in VAE encoder/decoder MLPs.

    vae_input_transform : str, default="log1p"
        Input transform applied to counts before entering the VAE
        encoder.  Supported: ``"log1p"``, ``"log"``, ``"sqrt"``,
        ``"identity"``, ``"log1p_prop"``, ``"clr"``, ``"log1p_norm"``.

    vae_standardize : bool or None, default=None
        Whether to standardize transformed VAE inputs to zero mean
        and unit variance per gene.  ``None`` auto-picks based on
        model (``True`` for LNM, ``False`` otherwise).

    vae_decoder_transforms : Dict[str, str], optional
        Mapping from decoder output names to transform names.

    vae_flow_type : str, default="none"
        Normalizing-flow prior family for VAE latent variables.
        Supported: ``"none"``, ``"affine_coupling"``,
        ``"spline_coupling"``, ``"maf"``, ``"iaf"``.

    vae_flow_num_layers : int, default=4
        Number of flow layers for the VAE latent flow.

    vae_flow_hidden_dims : List[int], optional
        Hidden layer widths used by VAE flow conditioners.

    Capture amortization and inference controls
    --------------------------------------------
    amortize_capture : bool, default=False
        Whether to use amortized inference for capture probability.
        When True, a neural network predicts variational parameters
        from total UMI count, reducing parameters from O(n_cells)
        to O(1).  Only applies to VCP models.

    capture_hidden_dims : List[int], optional
        Hidden layer dimensions for the capture amortizer MLP.
        Default is ``[64, 32]``.

    capture_activation : str, default="leaky_relu"
        Activation function for the capture amortizer MLP.

    capture_output_transform : str, default="softplus"
        Transform for positive output parameters in constrained mode.

    capture_clamp_min : float or None, default=0.1
        Minimum clamp for amortizer positive outputs.

    capture_clamp_max : float or None, default=50.0
        Maximum clamp for amortizer positive outputs.

    capture_amortization : AmortizationConfig or dict, optional
        Single config object for capture amortization.  When provided,
        overrides the individual ``capture_*`` parameters above.

    inference_method : str, default="svi"
        Inference method to use:

            - ``"svi"``: Stochastic Variational Inference (fast,
              scalable)
            - ``"mcmc"``: Markov Chain Monte Carlo (exact, slower)
            - ``"vae"``: Variational Autoencoder (representation
              learning)
            - ``"laplace"``: Laplace approximation

    n_steps : int, default=50_000
        Number of optimization steps for SVI/VAE inference.

    batch_size : int, optional
        Mini-batch size for SVI/VAE.  ``None`` uses full-batch.
        Automatically coerced to ``None`` if larger than dataset size.

    optimizer_config : Dict[str, Any], optional
        Serializable optimizer specification for SVI/VAE.  Minimal
        form: ``{"name": "adam", "step_size": 1e-3}``.  Supported
        names: ``"adam"``, ``"clipped_adam"``, ``"adagrad"``,
        ``"rmsprop"``, ``"sgd"``, ``"momentum"``.

    stable_update : bool, default=True
        Use numerically stable parameter updates in SVI.

    log_progress_lines : bool, default=False
        Emit periodic plain-text progress lines during SVI/VAE.

    n_samples : int, default=2_000
        Number of MCMC samples (only for ``inference_method="mcmc"``).

    n_warmup : int, default=1_000
        Number of MCMC warmup samples.

    n_chains : int, default=1
        Number of MCMC chains to run in parallel.

    early_stopping : EarlyStoppingConfig or dict, optional
        Early stopping configuration for SVI/VAE.  ``None`` (default)
        disables early stopping.

    restore_best : bool, default=False
        Track and restore the best variational parameters during
        training.

    kl_annealing : KLAnnealingConfig, dict, or bool, optional
        KL annealing configuration.  Automatically defaulted ON for
        VAE-mode fits.  Pass ``KLAnnealingConfig(enabled=False)`` to
        disable.

    kl_annealing_warmup : int, optional
        Shortcut: sets ``kl_annealing=KLAnnealingConfig(warmup=N)``.

    laplace_config : LaplaceConfig or dict, optional
        Laplace-specific overrides (e.g. ``{"n_newton_steps": 15}``).

    SVI-to-Laplace cascade (NBLN-Laplace only)
    -------------------------------------------
    informative_priors_from : ScribeSVIResults, optional
        Source SVI results (typically a converged NBVCP-SVI fit on the
        same dataset).  Empirical Gaussian priors are derived from this
        source's posterior samples and injected into the NBLN-Laplace
        loss for ``r_g``, ``mu_g``, and per-cell ``eta_c``.  Restricted
        to ``model="nbln"`` with ``inference_method="laplace"``; any
        other combination raises ``ValueError`` early.  See
        ``paper/_nb_lognormal.qmd`` (``sec-nbln-svi-cascade``).

    informative_priors_tau : float, default=1.0
        Scale-inflation factor for the cascade priors' standard
        deviations.  Values ``> 1`` loosen the prior; ``2``-``3`` is a
        reasonable starting point for noisy/sparse data.

    informative_priors_n_samples : int, default=1000
        Number of posterior samples drawn from the cascade source for
        moment-matching the empirical Gaussian priors.

    informative_priors_verbose : bool, default=True
        Emit informational messages describing the cascade contract
        (capture-mode detection, gene-identity check outcome, freeze
        choice).

    informative_priors_freeze : tuple of str, default=("r", "eta")
        Subset of ``{"r", "mu", "eta"}`` whose values are *pinned* at
        the cascade-source MAP and excluded from the NBLN optimizer
        dict.  Default ``("r", "eta")`` eliminates the rigid-translation
        gauge degeneracy by structurally fixing per-cell ``eta`` while
        the NBLN M-step refines ``mu``, ``W``, and ``d``.  Descriptive
        aliases are accepted: ``"dispersion" -> "r"``,
        ``"mean_expression"`` (or ``"expression"``) ``-> "mu"``,
        ``"capture_efficiency" -> "eta"`` — see
        ``FREEZE_KEY_ALIASES`` in
        ``scribe.models.config.parameter_mapping``.  Pass ``()`` for
        plain (no-freeze) cascade behaviour.  See
        ``paper/_diffexp_nbln_robustness.qmd`` and
        ``sec-nbln-cascade-freeze``.

    w_prior : dict, optional
        **Deprecated.** Legacy alias for the W-loadings shrinkage
        prior.  Pass it as ``priors={"loadings": {...}}`` instead (see
        the ``priors`` entry above).  Both routes resolve to the same
        internal plumbing; passing both raises ``ValueError``.  Emits
        ``DeprecationWarning`` when used.

    Data access, annotations, and initialization
    ---------------------------------------------
    cells_axis : int, default=0
        Axis for cells in count matrix.  0 means cells are rows.

    layer : str, optional
        Layer in AnnData to use for counts.  ``None`` uses ``.X``.

    gene_coverage : float, optional
        Pre-fit gene coverage filter threshold.  Genes with coverage
        below this fraction are pooled into a trailing "other"
        pseudo-gene column.

    seed : int, default=42
        Random seed for reproducibility.

    annotation_key : str or list of str, optional
        Column name(s) in ``adata.obs`` containing categorical
        cell-type annotations.  When provided, the annotations are
        used as soft priors on per-cell mixture component assignments.

        If ``n_components`` is **not** specified, it is automatically
        inferred from the number of unique non-null annotation labels.

        When a **list** of column names is given, composite labels are
        formed by joining the per-column values with ``"__"``.

    annotation_confidence : float, default=3.0
        Strength of the annotation prior (kappa).

        * ``0`` -- annotations are ignored.
        * ``3`` (default) -- annotated component gets ~20x prior boost.
        * Large values -- approaches hard assignment.

    annotation_component_order : list of str, optional
        Explicit mapping from annotation labels to component indices.
        The *i*-th element is assigned to component *i*.  ``None``
        sorts labels alphabetically.

    annotation_min_cells : int, optional
        Minimum number of cells required for an annotation label to be
        used.  Labels with fewer cells are treated as unlabeled.

    svi_init : ScribeSVIResults, optional
        SVI results to use for initializing MCMC chains.  MAP
        estimates are extracted and injected as ``init_to_value``
        into the NUTS kernel.  Only valid when
        ``inference_method="mcmc"``.  Cross-parameterization
        initialization is fully supported.

    enable_x64 : bool, optional
        Whether to run inference in float64 precision.  When ``None``:

        - **MCMC** -> ``True`` (Hamiltonian dynamics benefit from
          double precision)
        - **SVI / VAE** -> ``False`` (float32 is sufficient and
          faster)

    Power-user config overrides
    ---------------------------
    model_config : ModelConfig, optional
        Fully configured model configuration object.  If provided,
        overrides model, parameterization, unconstrained,
        n_components, mixture_params, guide_rank, and priors.

    inference_config : InferenceConfig, optional
        Fully configured inference configuration object.  If provided,
        overrides inference_method, n_steps, batch_size,
        optimizer_config, stable_update, log_progress_lines,
        n_samples, n_warmup, and n_chains.

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
        If model, parameterization, or inference_method is not
        recognized, or if configuration is invalid.

    Examples
    --------
    Basic usage (default model is NBVCP with variable capture):

    >>> results = scribe.fit(adata)

    Zero-inflated model with mixture components:

    >>> results = scribe.fit(
    ...     adata,
    ...     zero_inflation=True,
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

    Initialize MCMC from SVI results (cross-parameterization):

    >>> svi_results = scribe.fit(
    ...     adata, model="nbdm", parameterization="linked"
    ... )
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
    # -- Resolve the latent-factor dimension --------------------------------
    # ``latent_dim`` is the new primary kwarg; ``vae_latent_dim`` is the
    # legacy alias retained for back-compat.  They are aliases — passing
    # both is an error.  Default falls back to ``10`` when neither is
    # supplied (preserves the previous default).
    if latent_dim is not None and vae_latent_dim is not None:
        raise ValueError(
            "Pass either `latent_dim` (preferred) or `vae_latent_dim` "
            "(legacy alias), not both — they refer to the same quantity "
            "(rank of the low-rank loadings matrix W).  Use `latent_dim` "
            "in new code; `vae_latent_dim` is retained for backward "
            "compatibility."
        )
    _effective_latent_dim = (
        latent_dim if latent_dim is not None
        else (vae_latent_dim if vae_latent_dim is not None else 10)
    )

    # -- Extract the W-prior strategy spec from ``priors`` ------------------
    # Preferred API: ``priors={"loadings": {"type": "horseshoe_columnwise",
    # ...}}`` lives alongside other parameter priors instead of as a
    # top-level kwarg.  Legacy ``w_prior=`` is still accepted but
    # deprecated.  Both forms point to the same downstream plumbing.
    #
    # The W-prior entry is *popped* from the priors dict before the
    # FitContext sees it — downstream stages (build_model_config,
    # normalize_prior_keys, etc.) operate on tuple-shaped entries only.
    _priors_for_ctx = priors
    _w_prior_from_priors = None
    if priors is not None:
        # Resolve both the descriptive alias and the internal key.
        if "loadings" in priors and "W" in priors:
            raise ValueError(
                "priors dict contains both 'loadings' and 'W' "
                "(internal key for the same parameter).  Use only "
                "one — 'loadings' is the preferred descriptive form."
            )
        for _key in ("loadings", "W"):
            if _key in priors:
                _w_prior_from_priors = priors[_key]
                # Build a copy with the entry removed.
                _priors_for_ctx = {
                    k: v for k, v in priors.items() if k != _key
                }
                # Empty dicts pass through as None to keep the
                # downstream "no priors" path clean.
                if not _priors_for_ctx:
                    _priors_for_ctx = None
                break

    # Reconcile with the deprecated top-level kwarg.  Two values =
    # ambiguous; one of either = use it.  ``w_prior`` is the legacy
    # path and emits a DeprecationWarning when used.
    if _w_prior_from_priors is not None and w_prior is not None:
        raise ValueError(
            "W-shrinkage prior was specified both via "
            "`priors={'loadings': ...}` (preferred) and the legacy "
            "`w_prior=` kwarg.  Pass it only via the priors dict; "
            "`w_prior=` is retained for backward compatibility but "
            "is deprecated."
        )
    if w_prior is not None:
        import warnings as _warnings
        _warnings.warn(
            "`w_prior=` kwarg is deprecated; pass the W-shrinkage "
            "prior inside the priors dict as "
            "`priors={'loadings': {'type': '...', ...}}` instead. "
            "Both routes resolve to the same internal plumbing.",
            DeprecationWarning, stacklevel=2,
        )
        _effective_w_prior = w_prior
    else:
        _effective_w_prior = _w_prior_from_priors

    # -- Pack all named arguments into FitContext for stage consumption --------
    ctx = FitContext(
        counts=counts,
        model=model,
        priors=_priors_for_ctx,
        n_components=n_components,
        kwargs=dict(
            variable_capture=variable_capture,
            zero_inflation=zero_inflation,
            parameterization=parameterization,
            unconstrained=unconstrained,
            expression_prior=expression_prior,
            prob_prior=prob_prior,
            zero_inflation_prior=zero_inflation_prior,
            n_datasets=n_datasets,
            dataset_key=dataset_key,
            dataset_params=dataset_params,
            dataset_mixing=dataset_mixing,
            expression_dataset_prior=expression_dataset_prior,
            prob_dataset_prior=prob_dataset_prior,
            prob_dataset_mode=prob_dataset_mode,
            zero_inflation_dataset_prior=zero_inflation_dataset_prior,
            overdispersion_dataset_prior=overdispersion_dataset_prior,
            auto_downgrade_single_dataset_hierarchy=auto_downgrade_single_dataset_hierarchy,
            horseshoe_tau0=horseshoe_tau0,
            horseshoe_slab_df=horseshoe_slab_df,
            horseshoe_slab_scale=horseshoe_slab_scale,
            neg_u=neg_u,
            neg_a=neg_a,
            neg_tau=neg_tau,
            capture_scaling_prior=capture_scaling_prior,
            expression_anchor=expression_anchor,
            expression_anchor_sigma=expression_anchor_sigma,
            overdispersion=overdispersion,
            overdispersion_prior=overdispersion_prior,
            d_mode=d_mode,
            alr_reference_idx=alr_reference_idx,
            mixture_params=mixture_params,
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
            guide_flow_log_det_f64=guide_flow_log_det_f64,
            priors=priors,
            vae_latent_dim=_effective_latent_dim,
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
            capture_amortization=capture_amortization,
            inference_method=inference_method,
            n_steps=n_steps,
            batch_size=batch_size,
            optimizer_config=optimizer_config,
            stable_update=stable_update,
            log_progress_lines=log_progress_lines,
            n_samples=n_samples,
            n_warmup=n_warmup,
            n_chains=n_chains,
            early_stopping=early_stopping,
            restore_best=restore_best,
            kl_annealing=kl_annealing,
            kl_annealing_warmup=kl_annealing_warmup,
            laplace_config=laplace_config,
            cells_axis=cells_axis,
            layer=layer,
            gene_coverage=gene_coverage,
            seed=seed,
            annotation_key=annotation_key,
            annotation_confidence=annotation_confidence,
            annotation_component_order=annotation_component_order,
            annotation_min_cells=annotation_min_cells,
            svi_init=svi_init,
            informative_priors_from=informative_priors_from,
            informative_priors_tau=informative_priors_tau,
            informative_priors_n_samples=informative_priors_n_samples,
            informative_priors_verbose=informative_priors_verbose,
            informative_priors_freeze=informative_priors_freeze,
            w_prior=_effective_w_prior,
            enable_x64=enable_x64,
            model_config=model_config,
            inference_config=inference_config,
        ),
    )

    # -- Pipeline stages (each reads/writes ctx) ------------------------------
    resolve_model_flags(ctx)
    validate_inputs(ctx)
    process_data_and_datasets(ctx)
    apply_gene_coverage_and_alr(ctx)
    if ctx.model.lower() in ("lnm", "lnmvcp"):
        resolve_lnm_auto_priors(ctx)
    build_annotation_priors(ctx)
    build_model_config(ctx)
    build_inference_config(ctx)
    dispatch_inference(ctx)
    postprocess_results(ctx)

    return ctx.results
