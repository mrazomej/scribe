"""
Parameter group definitions for model configuration using Pydantic for type
safety and validation.

These configuration groups define logically related sets of parameters (such as
priors, guides, or VAE-specific settings) that compose the overall SCRIBE model
configuration. By organizing parameters into self-contained modular groups, the
codebase enables:

    - Automatic validation and helpful error messages using Pydantic's
      declarative fields and validators.
    - Strict type safety for all user-supplied or programmatically constructed
      configurations.
    - Reuse and composability: each group can be combined, nested, or shared
      across multiple model types.
    - Easy documentation of available parameters and their intended constraints.
    - Immutable, robust config objects that prevent accidental mutation or use
      of unsupported parameters.

All parameter groups inherit from Pydantic's BaseModel, ensuring configs remain
valid, immutable, and explicit. The groups here are the foundational building
blocks used to create full model configurations in SCRIBE.
"""

from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from .enums import InferenceMethod

# Import GuideFamily for Pydantic's runtime type checking
# We use Any as the type hint since GuideFamily is a dataclass that Pydantic
# doesn't natively understand, and we have arbitrary_types_allowed=True
from ..components.guide_families import GuideFamily

# PriorConfig, GuideConfig, UnconstrainedPriorConfig, and
# UnconstrainedGuideConfig have been removed. Prior and guide hyperparameters
# are now stored directly in ParamSpec objects (see
# scribe.models.builders.parameter_specs).


# ==============================================================================
# Prior Overrides Configuration Group
# ==============================================================================


class PriorOverrides(BaseModel):
    """Prior parameter overrides with attribute-style access and model_copy.

    Stores prior hyperparameters (e.g., Beta, LogNormal) as tuples keyed by
    parameter name. Used for config.priors.p, config.priors.model_copy(), etc.

    Validation is performed by ModelConfigBuilder before construction.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    def __getattr__(self, name: str) -> Any:
        """Attribute access for prior params (e.g. config.priors.p)."""
        if name.startswith("_") or name in ("model_copy", "model_dump"):
            return object.__getattribute__(self, name)
        try:
            extra = object.__getattribute__(self, "__pydantic_extra__")
            if extra is not None and name in extra:
                return extra[name]
        except AttributeError:
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        ) from None


# ==============================================================================
# Guide Family Configuration Group
# ==============================================================================


class GuideFamilyConfig(BaseModel):
    """Per-parameter guide family configuration.

    This class specifies which variational family (MeanField, LowRank,
    Amortized) to use for each parameter in the model. Unlike `GuideConfig`
    which stores the hyperparameters (e.g., alpha/beta for a Beta distribution),
    this class determines the *structure* of the variational approximation.

    Parameters that are not specified default to `MeanFieldGuide()`.

    Parameters
    ----------
    p : GuideFamily, optional
        Guide family for the success probability parameter (canonical, linked,
        mean_prob) or LNM total-count NB probability (logistic_normal).
    r : GuideFamily, optional
        Guide family for the dispersion parameter.
    mu : GuideFamily, optional
        Guide family for the mean parameter (linked parameterization).
    phi : GuideFamily, optional
        Guide family for the odds ratio parameter.
    gate : GuideFamily, optional
        Guide family for the zero-inflation gate parameter.
    p_capture : GuideFamily, optional
        Guide family for the capture probability parameter.
    phi_capture : GuideFamily, optional
        Guide family for the capture odds ratio parameter.
    mixing : GuideFamily, optional
        Guide family for mixture weights.

    Examples
    --------
    >>> from scribe.models.config import GuideFamilyConfig
    >>> from scribe.models.components import MeanFieldGuide, LowRankGuide, AmortizedGuide
    >>>
    >>> # All mean-field (default)
    >>> config = GuideFamilyConfig()
    >>>
    >>> # Low-rank for r, amortized for p_capture
    >>> config = GuideFamilyConfig(
    ...     r=LowRankGuide(rank=10),
    ...     p_capture=AmortizedGuide(amortizer=my_amortizer),
    ... )

    See Also
    --------
    GuideConfig : Configuration for guide hyperparameters (alpha/beta values).
    scribe.models.components.guide_families : Guide family implementations.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    # All possible parameters - None means use default MeanFieldGuide
    p: Optional[GuideFamily] = Field(
        None,
        description=(
            "Guide family for success probability (canonical / linked / "
            "mean_prob) or LNM total-count NB probability (logistic_normal)"
        ),
    )
    r: Optional[GuideFamily] = Field(
        None, description="Guide family for dispersion"
    )
    mu: Optional[GuideFamily] = Field(
        None, description="Guide family for mean (linked parameterization)"
    )
    phi: Optional[GuideFamily] = Field(
        None, description="Guide family for odds ratio"
    )
    gate: Optional[GuideFamily] = Field(
        None, description="Guide family for zero-inflation gate"
    )
    p_capture: Optional[GuideFamily] = Field(
        None, description="Guide family for capture probability"
    )
    phi_capture: Optional[GuideFamily] = Field(
        None, description="Guide family for capture odds ratio"
    )
    mixing: Optional[GuideFamily] = Field(
        None, description="Guide family for mixture weights"
    )
    r_T: Optional[GuideFamily] = Field(
        None,
        description="Guide family for LNM total-count NB dispersion r_T",
    )
    y_alr: Optional[GuideFamily] = Field(
        None,
        description=(
            "Guide family for ALR compositional coordinates "
            "(LNM gene-level parameter, typically low-rank or joint)"
        ),
    )
    d_lnm: Optional[GuideFamily] = Field(
        None,
        description=(
            "Guide family for LNM learned diagonal ALR scales (d_mode=learned)"
        ),
    )

    # --------------------------------------------------------------------------
    # Amortization Configuration
    # --------------------------------------------------------------------------

    # Note: AmortizationConfig is defined below in this file. We use a forward
    # reference string to avoid circular dependency issues.
    capture_amortization: Optional["AmortizationConfig"] = Field(
        None,
        description=(
            "Configuration for amortized inference of capture probability "
            "(p_capture or phi_capture). When enabled, uses an MLP to predict "
            "variational parameters from total UMI count instead of learning "
            "separate parameters per cell."
        ),
    )

    # --------------------------------------------------------------------------
    # Accessor Method
    # --------------------------------------------------------------------------

    def get(self, name: str) -> GuideFamily:
        """Get the guide family for a parameter, defaulting to MeanFieldGuide.

        Parameters
        ----------
        name : str
            The parameter name (e.g., "r", "p_capture").

        Returns
        -------
        GuideFamily
            The configured guide family, or MeanFieldGuide() if not specified.

        Examples
        --------
        >>> config = GuideFamilyConfig(r=LowRankGuide(rank=10))
        >>> config.get("r")  # Returns LowRankGuide(rank=10)
        >>> config.get("p")  # Returns MeanFieldGuide()
        """
        from ..components.guide_families import MeanFieldGuide

        value = getattr(self, name, None)
        return value if value is not None else MeanFieldGuide()


# UnconstrainedPriorConfig and UnconstrainedGuideConfig have been removed.
# Prior and guide hyperparameters are now stored directly in ParamSpec objects
# with validation based on the distribution type.


# ==============================================================================
# Amortization Configuration Group
# ==============================================================================


class AmortizationConfig(BaseModel):
    """Configuration for amortized inference of cell-specific parameters.

    This class specifies the architecture of the neural network (MLP) that
    maps sufficient statistics (e.g., total UMI count) to variational
    parameters for cell-specific quantities like capture probability.

    The MLP architecture is:
        sufficient_statistic → [Linear → activation] × n_layers → output_heads

    Parameters
    ----------
    enabled : bool, default=False
        Whether to use amortized inference for this parameter.
    hidden_dims : List[int], default=[64, 32]
        Dimensions of hidden layers in the MLP. The number of layers is
        determined by the length of this list.
    activation : str, default="relu"
        Activation function for hidden layers. Supported: "relu", "gelu",
        "silu", "tanh", "sigmoid", etc.
    input_transformation : str, default="log1p"
        Transformation applied to input data before computing sufficient
        statistic. Options: "log1p", "log", "sqrt", "identity",
        "log1p_prop", "clr", and "log1p_norm".

    Examples
    --------
    >>> # Default configuration
    >>> config = AmortizationConfig(enabled=True)

    >>> # Custom architecture via YAML
    >>> # amortization:
    >>> #   capture:
    >>> #     enabled: true
    >>> #     hidden_dims: [128, 64, 32]
    >>> #     activation: gelu

    See Also
    --------
    scribe.models.components.amortizers.Amortizer : The MLP implementation.
    scribe.models.components.amortizers.TOTAL_COUNT : Sufficient statistic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        False, description="Whether to use amortized inference"
    )
    hidden_dims: List[int] = Field(
        default_factory=lambda: [64, 32],
        description="Hidden layer dimensions for the amortizer MLP",
    )
    activation: str = Field(
        "relu", description="Activation function for hidden layers"
    )
    input_transformation: str = Field(
        "log1p",
        description=(
            "Transformation for input data (log1p, log, sqrt, identity, "
            "log1p_prop, clr, log1p_norm)"
        ),
    )
    output_transform: str = Field(
        "softplus",
        description=(
            "Transform for positive output parameters in constrained mode. "
            "'softplus' (default): softplus(x) + 0.5, bounded away from zero "
            "and grows linearly for large inputs. "
            "'exp': exponential transform (original behavior, can produce "
            "extreme values)."
        ),
    )
    output_clamp_min: Optional[float] = Field(
        0.1,
        ge=0,
        description=(
            "Minimum clamp for positive output parameters (alpha, beta) in "
            "constrained mode. Prevents BetaPrime/Beta with extreme shape "
            "parameters. Set to None to disable. Default: 0.1."
        ),
    )
    output_clamp_max: Optional[float] = Field(
        50.0,
        gt=0,
        description=(
            "Maximum clamp for positive output parameters (alpha, beta) in "
            "constrained mode. Prevents extreme concentration. "
            "Set to None to disable. Default: 50.0."
        ),
    )

    # --------------------------------------------------------------------------

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        """Validate hidden dimensions are positive."""
        if not v:
            raise ValueError("hidden_dims must have at least one layer")
        if any(d <= 0 for d in v):
            raise ValueError(f"Hidden dimensions must be positive, got {v}")
        return v

    # --------------------------------------------------------------------------

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v: str) -> str:
        """Validate activation function is supported."""
        valid_activations = {
            "relu",
            "gelu",
            "silu",
            "tanh",
            "sigmoid",
            "elu",
            "leaky_relu",
            "softplus",
            "swish",
            "celu",
            "selu",
        }
        if v.lower() not in valid_activations:
            raise ValueError(
                f"Unknown activation '{v}'. Valid options: {valid_activations}"
            )
        return v.lower()

    # --------------------------------------------------------------------------

    @field_validator("input_transformation")
    @classmethod
    def validate_input_transformation(cls, v: str) -> str:
        """Validate input transformation is supported."""
        valid_transforms = {
            "log1p",
            "log",
            "sqrt",
            "identity",
            "log1p_prop",
            "clr",
            "log1p_norm",
        }
        if v.lower() not in valid_transforms:
            raise ValueError(
                f"Unknown transformation '{v}'. "
                f"Valid options: {valid_transforms}"
            )
        return v.lower()

    # --------------------------------------------------------------------------

    @field_validator("output_transform")
    @classmethod
    def validate_output_transform(cls, v: str) -> str:
        """Validate output transform is supported."""
        valid = {"exp", "softplus"}
        if v.lower() not in valid:
            raise ValueError(
                f"Unknown output_transform '{v}'. Valid options: {valid}"
            )
        return v.lower()

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_clamp_range(self) -> "AmortizationConfig":
        """Validate that clamp_min < clamp_max when both are set."""
        if (
            self.output_clamp_min is not None
            and self.output_clamp_max is not None
            and self.output_clamp_min >= self.output_clamp_max
        ):
            raise ValueError(
                f"output_clamp_min ({self.output_clamp_min}) must be less "
                f"than output_clamp_max ({self.output_clamp_max})"
            )
        return self

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AmortizationConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# VAE Configuration Group
# ==============================================================================


class VAEConfig(BaseModel):
    """VAE-specific configuration with composable architecture.

    Clean schema for encoder, decoder, flow prior, and preprocessing.
    All legacy fields have been removed.
    """

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    # Architecture
    latent_dim: int = Field(10, gt=0, description="Latent space dimensionality")
    encoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [128, 64],
        description="Encoder hidden layer sizes",
    )
    decoder_hidden_dims: List[int] = Field(
        default_factory=lambda: [64, 128],
        description="Decoder hidden layer sizes",
    )
    activation: str = Field("relu", description="Activation function")
    input_transform: str = Field(
        "log1p",
        description=(
            "Input transformation before encoder "
            "(log1p, log, sqrt, identity, log1p_prop, clr, log1p_norm)"
        ),
    )
    standardize: bool = Field(
        False, description="Whether to standardize input data"
    )

    # --------------------------------------------------------------------------
    # LNM-specific data-derived fields
    # --------------------------------------------------------------------------
    # The three optional fields below carry data-derived constants that
    # the Logistic-Normal Multinomial (LNM) factory consumes to anchor
    # encoder / decoder initialization to the dataset. They are filled
    # by the public API (``scribe.api.fit``) for ``model in {"lnm",
    # "lnmvcp"}`` and remain ``None`` for every other model, so the
    # default behavior of standard NBDM/ZINB pipelines is unchanged.
    #
    # Each field is typed as ``Optional[Any]`` because Pydantic does not
    # natively understand ``jnp.ndarray``; runtime enforcement of the
    # array shape is handled by the consuming factory code, which is
    # the source of truth for the contract anyway.

    empirical_alr_bias_init: Optional[Any] = Field(
        None,
        description=(
            "Optional ALR bias used to initialize the LNM linear-decoder "
            "``y_alr`` head. Shape ``(n_genes - 1,)``. When set, the "
            "decoder bias is anchored to the empirical ALR mean of the "
            "training counts so the very first forward pass already "
            "reproduces the dataset's marginal composition. None for "
            "non-LNM models."
        ),
    )
    standardize_mean: Optional[Any] = Field(
        None,
        description=(
            "Per-feature mean to subtract from the (transformed) "
            "encoder input when ``standardize=True``. Shape "
            "``(n_genes,)``. Computed in the same input-transform space "
            "the encoder uses (e.g. ``log1p_prop`` for LNM). None for "
            "models that do not request input standardization or for "
            "which standardization stats are not pre-computed."
        ),
    )
    standardize_std: Optional[Any] = Field(
        None,
        description=(
            "Per-feature standard deviation paired with "
            "``standardize_mean``. Shape ``(n_genes,)``. Floored to a "
            "small epsilon at consumption time to avoid division by "
            "zero on constant features."
        ),
    )

    # PLN-specific data-derived initialization fields.
    empirical_log_mean_bias_init: Optional[Any] = Field(
        None,
        description=(
            "Optional per-gene log-mean bias for PLN decoder "
            "initialization. Shape ``(n_genes,)``. When set, the "
            "decoder bias is anchored to ``log(mean(u_g) + c)`` so "
            "that the initial Poisson rates are at the right order of "
            "magnitude. None for non-PLN models."
        ),
    )
    pca_loadings_init: Optional[Any] = Field(
        None,
        description=(
            "Optional PCA-based initialization for the PLN decoder "
            "weight matrix W. Shape ``(n_genes, latent_dim)``. "
            "Derived from truncated SVD of the centered log-count "
            "matrix. None for non-PLN models or when PCA init is "
            "disabled."
        ),
    )

    # Decoder output transforms (optional per-param overrides)
    decoder_transforms: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Override default output transforms per decoder param. "
            "Keys are param names (e.g. 'r', 'mu', 'gate'), values are "
            "transform names: identity, exp, softplus, sigmoid, clamp_exp."
        ),
    )

    # Flow prior
    flow_type: str = Field(
        "none",
        description="Flow type for prior: none, coupling_affine, coupling_spline, maf, iaf",
    )
    flow_num_layers: int = Field(4, gt=0, description="Number of flow layers")
    flow_hidden_dims: List[int] = Field(
        default_factory=lambda: [64, 64],
        description="Flow conditioner hidden dimensions",
    )

    @field_validator("decoder_transforms")
    @classmethod
    def validate_decoder_transforms(cls, v):
        if v is not None:
            valid = {"identity", "exp", "softplus", "sigmoid", "clamp_exp"}
            for param, transform in v.items():
                if transform not in valid:
                    raise ValueError(
                        f"Invalid transform '{transform}' for param '{param}'. "
                        f"Must be one of {valid}"
                    )
        return v

    @field_validator("flow_type")
    @classmethod
    def validate_flow_type(cls, v):
        valid = {"none", "coupling_affine", "coupling_spline", "maf", "iaf"}
        if v not in valid:
            raise ValueError(f"Invalid flow_type: {v}. Must be one of {valid}")
        return v

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, v):
        valid = {"relu", "gelu", "silu", "tanh", "elu", "leaky_relu"}
        if v.lower() not in valid:
            raise ValueError(f"Invalid activation: {v}. Must be one of {valid}")
        return v.lower()

    @field_validator("input_transform")
    @classmethod
    def validate_input_transform(cls, v):
        valid = {
            "log1p",
            "log",
            "sqrt",
            "identity",
            "log1p_prop",
            "clr",
            "log1p_norm",
        }
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid input_transform: {v}. Must be one of {valid}"
            )
        return v.lower()

    @field_validator(
        "encoder_hidden_dims", "decoder_hidden_dims", "flow_hidden_dims"
    )
    @classmethod
    def validate_hidden_dims(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("hidden_dims must have at least one layer")
        if any(d <= 0 for d in v):
            raise ValueError(f"Hidden dimensions must be positive, got {v}")
        return v


# ==============================================================================
# Early Stopping Configuration Group
# ==============================================================================


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping during SVI optimization.

    Early stopping monitors the training loss and stops optimization when the
    loss stops improving, preventing overfitting and saving computation time.
    The implementation uses a smoothed loss (moving average) to reduce noise in
    the stopping decision.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to enable early stopping. If False, training runs for the full
        `n_steps` regardless of convergence.
    patience : int, default=500
        Number of steps without improvement before stopping. The counter resets
        each time an improvement is detected.
    min_delta : float, default=100.0
        Minimum absolute improvement in smoothed loss to qualify as progress.
        Used when `min_delta_pct` is not specified. The default of 100.0 works
        for typical ELBO values.
    min_delta_pct : float, optional
        Minimum relative improvement (as percentage) in smoothed loss to qualify
        as progress. Computed as `100 * (best_loss - smoothed_loss) /
        best_loss`. If specified, takes precedence over `min_delta`. For example,
        0.01 means 0.01% improvement is required. This scales automatically with
        loss magnitude.
    check_every : int, default=10
        How often (in steps) to check for convergence. Checking every step adds
        overhead; checking less frequently may miss the optimal stop point.
    warmup : int, default=5000
        Number of warmup steps before early stopping is activated. During
        warmup, loss is tracked but stopping criteria are not evaluated. This
        allows the model to stabilize before we start checking for convergence.
        Set to 0 to disable warmup.
    smoothing_window : int, default=50
        Window size for computing the smoothed (moving average) loss. Larger
        windows reduce noise but respond slower to changes.
    restore_best : bool, default=True
        Whether to restore parameters from the best checkpoint (lowest smoothed
        loss) when early stopping is triggered.
    checkpoint_dir : str, optional
        Directory for Orbax checkpoints. When set, saves best parameters to
        disk whenever loss improves, enabling resumable training. Set
        automatically by Hydra when using `infer.py`. For direct API use,
        can be set manually to enable checkpointing. Default is None (no
        checkpointing).
    resume : bool, default=True
        Whether to resume from an existing checkpoint if one exists.
        Default is True (auto-resume). Set to False to start fresh training
        while still saving checkpoints for future resumption.

    Examples
    --------
    >>> # Default configuration (no checkpointing)
    >>> config = EarlyStoppingConfig()

    >>> # Using percentage-based threshold
    >>> config = EarlyStoppingConfig(
    ...     patience=200,
    ...     min_delta_pct=0.01,  # 0.01% improvement required
    ...     check_every=5,
    ... )

    >>> # Using absolute threshold (default)
    >>> config = EarlyStoppingConfig(
    ...     patience=200,
    ...     min_delta=50.0,  # 50 absolute improvement required
    ... )

    >>> # Disable early stopping
    >>> config = EarlyStoppingConfig(enabled=False)

    >>> # With checkpointing (for direct API use)
    >>> config = EarlyStoppingConfig(
    ...     checkpoint_dir="./my_checkpoints",
    ...     resume=True,  # Will resume if checkpoint exists
    ... )

    Notes
    -----
    The smoothed loss at step `t` is computed as the mean of the last
    `smoothing_window` loss values. This helps distinguish true convergence from
    temporary fluctuations in the ELBO.

    The early stopping algorithm:
        1. Every `check_every` steps, compute the smoothed loss
        2. Compute relative improvement: `(best_loss - smoothed_loss) / best_loss`
        3. If relative improvement > `min_delta_pct`, update best_loss and
           reset patience counter
        4. Otherwise, increment patience counter by `check_every`
        5. If patience counter >= `patience`, stop training and optionally
           restore best parameters

    See Also
    --------
    SVIConfig : Parent configuration that includes early stopping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(True, description="Whether to enable early stopping")
    patience: int = Field(
        500,
        gt=0,
        description="Steps without improvement before stopping",
    )
    min_delta: float = Field(
        1.0,
        ge=0,
        description=(
            "Minimum absolute improvement in loss to qualify as progress. "
            "Used when min_delta_pct is not specified."
        ),
    )
    min_delta_pct: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Minimum relative improvement (as percentage) to qualify as progress. "
            "If specified, takes precedence over min_delta. "
            "E.g., 0.01 means 0.01% improvement required."
        ),
    )
    check_every: int = Field(
        100,
        gt=0,
        description="Check convergence every N steps",
    )
    warmup: int = Field(
        5000,
        ge=0,
        description=(
            "Number of warmup steps before early stopping is activated. "
            "During warmup, loss is tracked but stopping criteria are not evaluated. "
            "Set to 0 to disable warmup."
        ),
    )
    smoothing_window: int = Field(
        100,
        gt=0,
        description="Window size for loss smoothing (moving average)",
    )
    checkpoint_every: int = Field(
        2500,
        gt=0,
        description=(
            "How often to save checkpoints (in steps). Checkpoints are only saved "
            "when an improvement is detected AND at least checkpoint_every steps "
            "have passed since the last checkpoint. Set to 1 to save on every improvement."
        ),
    )
    restore_best: bool = Field(
        True,
        description="Restore best parameters when early stopping triggers",
    )

    # Checkpointing configuration
    checkpoint_dir: Optional[str] = Field(
        None,
        description=(
            "Directory for Orbax checkpoints. When set, saves best parameters "
            "to disk on improvement. Set automatically by Hydra in infer.py. "
            "For direct API use, can be set manually to enable checkpointing."
        ),
    )
    resume: bool = Field(
        True,
        description=(
            "Whether to resume from checkpoint if one exists. Default True. "
            "Set to False to start fresh while still saving checkpoints."
        ),
    )

    # --------------------------------------------------------------------------

    @field_validator("smoothing_window")
    @classmethod
    def validate_smoothing_window(cls, v: int, info) -> int:
        """Warn if smoothing window is larger than patience."""
        # Note: We can't access other fields in field_validator easily,
        # so we just validate that smoothing_window is reasonable
        if v > 1000:
            import warnings

            warnings.warn(
                f"Large smoothing_window ({v}) may cause slow response "
                "to convergence. Consider using a smaller value.",
                UserWarning,
            )
        return v

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "EarlyStoppingConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# KL Annealing Configuration Group
# ==============================================================================


class KLAnnealingConfig(BaseModel):
    """Schedule for the KL term in the ELBO during VAE-mode SVI training.

    KL annealing introduces a step-dependent weight ``beta(step)`` on the
    KL component of the ELBO so the encoder can fit the data
    reconstruction half before the prior pressure is fully applied. This
    suppresses two well-known failure modes for linear-decoder VAEs with
    high-dimensional latents:

    1. **Posterior collapse** (``q(z|u) -> N(0, I)`` regardless of input)
       — caused by full-strength KL pressure dominating gradient flow
       early in training.
    2. **Aggregate-posterior drift** (``mean_c q(z|u_c) != N(0, I)``)
       — the converse: the encoder fits per-cell reconstruction well
       but the aggregate posterior wanders far from the prior, and any
       convex decoder path (``exp(W·z)`` in PLN) amplifies the drift
       into prediction bias.

    The standard linear schedule ramps ``beta`` from ``beta_min`` (at
    ``step=0``) to ``beta_max`` (at ``step=warmup``) and clamps at
    ``beta_max`` for ``step > warmup``. ``beta_max=1.0`` recovers the
    standard ELBO; setting ``beta_max < 1.0`` implements a permanent
    β-VAE-style down-weighting.

    Annealing only affects training. Post-fit metrics (PPC, MAP,
    importance-sampled marginal log-likelihood) all use the full
    ``beta=1`` ELBO regardless of the schedule used during training.

    Parameters
    ----------
    enabled : bool, default=True
        Whether to apply KL annealing. When ``False`` the schedule is
        ignored and the standard ELBO is used (``beta=1`` always).
        The :class:`SVIConfig.kl_annealing` field can also be left
        ``None`` to disable annealing entirely without instantiating a
        config object.
    schedule : {"linear"}, default="linear"
        Shape of the annealing schedule. Only ``"linear"`` is
        implemented in v1; the field is a literal type so future
        schedules (``"cosine"``, ``"cyclic"``, ...) can be added without
        breaking existing configs.
    warmup : int, default=2_000
        Number of SVI steps over which to ramp ``beta`` linearly from
        ``beta_min`` to ``beta_max``. ``warmup=0`` is equivalent to
        ``enabled=False`` (returns ``beta_max`` immediately).
    beta_min : float, default=0.0
        Starting weight on the KL term (inclusive). ``0.0`` means the
        first step is pure reconstruction. Larger values
        (e.g. ``0.1``) keep a faint KL signal alive throughout warmup,
        which can prevent the encoder from drifting too far before the
        KL term comes online.
    beta_max : float, default=1.0
        Final weight on the KL term (post-warmup). ``1.0`` recovers the
        standard ELBO. ``< 1.0`` permanently down-weights KL
        (β-VAE-style) at the cost of looser aggregate-posterior
        regularisation.

    Examples
    --------
    >>> # Default linear ramp from 0 to 1 over 2000 steps.
    >>> KLAnnealingConfig()

    >>> # Faster warmup with a permanent β-VAE-style down-weight.
    >>> KLAnnealingConfig(warmup=500, beta_max=0.5)

    >>> # Disabled (equivalent to setting svi.kl_annealing=None)
    >>> KLAnnealingConfig(enabled=False)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        True,
        description=(
            "Whether to apply KL annealing. When False the schedule is "
            "ignored and beta=1 is used throughout training."
        ),
    )
    schedule: Literal["linear"] = Field(
        "linear",
        description=(
            "Shape of the annealing schedule. Only 'linear' is "
            "supported in v1."
        ),
    )
    warmup: int = Field(
        2_000,
        ge=0,
        description=(
            "Number of SVI steps from beta_min to beta_max. 0 disables "
            "annealing (returns beta_max immediately)."
        ),
    )
    beta_min: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Starting weight on the KL term (inclusive). 0.0 = pure "
            "reconstruction at step 0."
        ),
    )
    beta_max: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Final weight on the KL term (post-warmup). 1.0 recovers "
            "the standard ELBO."
        ),
    )

    @model_validator(mode="after")
    def _check_min_le_max(self) -> "KLAnnealingConfig":
        """Validate ``beta_min <= beta_max`` (a strictly increasing ramp)."""
        if self.beta_min > self.beta_max:
            raise ValueError(
                f"KLAnnealingConfig: beta_min ({self.beta_min}) must be "
                f"<= beta_max ({self.beta_max}). The schedule is a "
                "monotone ramp from beta_min to beta_max."
            )
        return self

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "KLAnnealingConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# SVI Configuration Group
# ==============================================================================


class SVIConfig(BaseModel):
    """Configuration for Stochastic Variational Inference.

    This class configures the SVI optimization process, including the optimizer,
    loss function, number of steps, mini-batching, and early stopping.

    Parameters
    ----------
    optimizer : Any, optional
        Prebuilt NumPyro optimizer object for variational inference.
        This is a power-user override and takes precedence over
        ``optimizer_config``.
    optimizer_config : OptimizerConfig, optional
        Serializable optimizer specification (name + kwargs) for API/Hydra
        usage. If neither ``optimizer`` nor ``optimizer_config`` is provided,
        the inference engine uses its default optimizer (Adam, 0.001).
    loss : Any, optional
        Loss function for variational inference. Defaults to TraceMeanField_ELBO
        if not specified.
    n_steps : int, default=100_000
        Maximum number of optimization steps. Training may stop earlier if early
        stopping is enabled and convergence is detected.
    batch_size : int, optional
        Mini-batch size for stochastic optimization. If None, uses the full
        dataset (batch gradient descent).
    stable_update : bool, default=True
        Use numerically stable parameter updates. When True, uses
        `svi.stable_update()` which handles NaN/Inf gracefully.
    log_progress_lines : bool, default=False
        Whether to emit plain-text progress log lines during SVI training.
        When enabled, the engine writes one newline log approximately every
        ``max(1, n_steps // 20)`` steps (about 20 updates per run), which is
        useful for non-interactive environments such as SLURM log files.
    early_stopping : EarlyStoppingConfig, optional
        Configuration for early stopping. If None, early stopping is disabled
        and training runs for the full `n_steps`.

    Examples
    --------
    >>> # Default configuration (no early stopping)
    >>> config = SVIConfig()

    >>> # With early stopping
    >>> config = SVIConfig(
    ...     n_steps=50_000,
    ...     early_stopping=EarlyStoppingConfig(patience=500),
    ... )

    >>> # Custom optimizer and loss
    >>> import numpyro
    >>> config = SVIConfig(
    ...     optimizer=numpyro.optim.Adam(step_size=0.01),
    ...     loss=numpyro.infer.Trace_ELBO(),
    ...     n_steps=20_000,
    ... )

    See Also
    --------
    EarlyStoppingConfig : Configuration for early stopping criteria.
    """

    class OptimizerConfig(BaseModel):
        """Serializable optimizer specification for SVI/VAE training.

        Parameters
        ----------
        name : str, default="adam"
            Optimizer name. Supported values are ``"adam"``,
            ``"clipped_adam"``, ``"adagrad"``, ``"rmsprop"``,
            ``"sgd"``, and ``"momentum"``.
        step_size : float, optional
            Optimizer learning rate.
        b1 : float, optional
            First-moment coefficient for Adam-like optimizers.
        b2 : float, optional
            Second-moment coefficient for Adam-like optimizers.
        eps : float, optional
            Numerical epsilon.
        momentum : float, optional
            Momentum coefficient for momentum-based optimizers.
        weight_decay : float, optional
            Weight decay coefficient for optimizers that support it.
        grad_clip_norm : float, optional
            Global clipping norm. For ``clipped_adam`` this maps to
            ``clip_norm``.

        Notes
        -----
        Extra fields are allowed and passed through as optimizer kwargs.
        This keeps the configuration forward-compatible with optimizer-specific
        keyword arguments not covered by the common fields above.
        """

        model_config = ConfigDict(frozen=True, extra="allow")

        name: str = Field(
            "adam",
            description=(
                "Optimizer name: adam, clipped_adam, adagrad, rmsprop, "
                "sgd, momentum"
            ),
        )
        step_size: Optional[float] = Field(
            None, gt=0, description="Optimizer learning rate"
        )
        b1: Optional[float] = Field(
            None, gt=0, lt=1, description="First-moment coefficient"
        )
        b2: Optional[float] = Field(
            None, gt=0, lt=1, description="Second-moment coefficient"
        )
        eps: Optional[float] = Field(
            None, gt=0, description="Numerical epsilon"
        )
        momentum: Optional[float] = Field(
            None,
            ge=0,
            lt=1,
            description="Momentum coefficient for momentum-based optimizers",
        )
        weight_decay: Optional[float] = Field(
            None, ge=0, description="Weight decay coefficient"
        )
        grad_clip_norm: Optional[float] = Field(
            None,
            gt=0,
            description=(
                "Global gradient clipping norm (mapped to clip_norm for "
                "clipped_adam)"
            ),
        )

        @field_validator("name")
        @classmethod
        def _normalize_name(cls, value: str) -> str:
            """Normalize and validate optimizer names."""
            normalized = value.strip().lower()
            valid_names = {
                "adam",
                "clipped_adam",
                "adagrad",
                "rmsprop",
                "sgd",
                "momentum",
            }
            if normalized not in valid_names:
                raise ValueError(
                    f"Unsupported optimizer name {value!r}. "
                    f"Choose one of {sorted(valid_names)}."
                )
            return normalized

        def extra_kwargs(self) -> Dict[str, Any]:
            """Return passthrough optimizer kwargs from extra config fields."""
            known = set(type(self).model_fields.keys())
            dumped = self.model_dump(exclude_none=True)
            return {
                key: value for key, value in dumped.items() if key not in known
            }

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    optimizer: Optional[Any] = Field(
        None,
        description="Optimizer for variational inference (defaults to Adam)",
    )
    optimizer_config: Optional[OptimizerConfig] = Field(
        None,
        description=(
            "Serializable optimizer specification used to build a NumPyro "
            "optimizer at runtime"
        ),
    )
    loss: Optional[Any] = Field(
        None, description="Loss function (defaults to TraceMeanField_ELBO)"
    )
    n_steps: int = Field(
        100_000, gt=0, description="Maximum number of optimization steps"
    )
    batch_size: Optional[int] = Field(
        None, gt=0, description="Mini-batch size. If None, uses full dataset"
    )
    stable_update: bool = Field(
        True, description="Use numerically stable parameter updates"
    )
    log_progress_lines: bool = Field(
        False,
        description=(
            "Emit plain-text progress lines every max(1, n_steps // 20) "
            "steps during SVI training"
        ),
    )
    early_stopping: Optional[EarlyStoppingConfig] = Field(
        None,
        description="Early stopping configuration. If None, disabled.",
    )
    restore_best: bool = Field(
        False,
        description=(
            "Track the best (lowest smoothed loss) variational parameters "
            "during training and restore them at the end.  Works "
            "independently of early stopping — when True and no "
            "early_stopping config is provided, a minimal internal "
            "config is created to enable the custom training loop with "
            "best-state tracking."
        ),
    )
    kl_annealing: Optional[KLAnnealingConfig] = Field(
        None,
        description=(
            "KL annealing schedule for the ELBO during training. "
            "Defaults are auto-resolved in the public ``scribe.fit`` "
            "API: ON for any VAE-mode fit (warmup=2000), OFF for plain "
            "SVI/MCMC, and force-OFF for Laplace mode. Pass an "
            "explicit ``KLAnnealingConfig(enabled=False)`` to disable, "
            "or a custom config to override the defaults. When None, "
            "the standard ``TraceMeanField_ELBO`` is used and the "
            "training loop is byte-identical to the pre-annealing path."
        ),
    )

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SVIConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# MCMC Configuration Group
# ==============================================================================


class MCMCConfig(BaseModel):
    """Configuration for Markov Chain Monte Carlo inference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_samples: int = Field(2_000, gt=0, description="Number of MCMC samples")
    n_warmup: int = Field(1_000, gt=0, description="Number of warmup samples")
    n_chains: int = Field(1, gt=0, description="Number of parallel chains")
    mcmc_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Additional keyword arguments for MCMC kernel"
    )

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "MCMCConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# Laplace Configuration Group
# ==============================================================================


class LaplaceConfig(BaseModel):
    """Configuration for Laplace-approximation inference (PLN-only).

    The Laplace inference path replaces the encoder with a per-cell
    Newton-iterated MAP on the latent log-rate ``x_c`` (and joint
    ``eta_c`` when the biology-informed capture anchor is active).
    The outer SVI loop updates global parameters (``mu``, ``W``,
    ``d``, capture-prior hyperparameters) via a Laplace-approximated
    ELBO that includes the ``-½ log det(-H_c)`` correction term.

    Mathematically equivalent to the variational-EM scheme used by
    ``PLNmodels`` in R: outer Adam on globals, inner Newton on
    per-cell latents. PLN is globally log-concave so Newton converges
    in 5-10 iterations from a warm start. Using Woodbury on
    ``Σ = W W' + diag(d)`` keeps each step ``O(G·k + k³)`` per cell.

    Structurally eliminates aggregate-posterior drift (no encoder, no
    parametric ``q(z|u)`` family to drift). The trade-off is that
    inference at new cells requires re-running Newton on demand
    rather than a single forward pass through an encoder. For
    research-scale analysis on the training cells this is fine;
    serving pipelines that need to score new cells at high throughput
    would need the encoder VAE path instead.

    Parameters
    ----------
    n_steps : int, default=50_000
        Number of outer SVI steps over the global parameters. Lower
        than the VAE path's typical 250k because the inner Newton
        gives the global gradient a clean signal at every step (no
        encoder-decoder warm-up).
    n_newton_steps : int, default=5
        Number of inner Newton iterations per outer SVI step. With
        warm-started latents, 5 is more than enough for convergence
        (quadratic in a log-concave problem). Higher values are
        wasteful but safe.
    newton_tolerance : float, default=1e-4
        Surfaced as part of ``ScribeLaplaceResults``: the engine
        warns if any cell's final L∞-gradient norm exceeds this
        threshold. Not used to early-stop Newton (we use a fixed
        iteration count for ``vmap`` compatibility).
    damping : float, default=1e-4
        Tikhonov damping added to the diagonal of ``-H_xx`` and to
        the η-block scalar to stabilise Newton when the Hessian is
        ill-conditioned (near-zero ``d``, sparse cells). The default
        is small enough not to bias the MAP for well-conditioned
        problems and large enough to avoid Cholesky failures.
    optimizer_config : SVIConfig.OptimizerConfig, optional
        Outer-loop optimizer specification (Adam by default, lr=1e-3).
    optimizer : Any, optional
        Pre-built NumPyro optimizer; takes precedence over
        ``optimizer_config``.
    batch_size : int, optional
        Mini-batch size for the outer SVI loop. ``None`` = full batch.
    early_stopping : EarlyStoppingConfig, optional
        Early stopping configuration for the outer loop. Same
        semantics as for SVI.
    restore_best : bool, default=False
        Track and restore the best-loss outer-loop parameters.
    convergence_action : {"warn", "raise", "ignore"}, default="warn"
        Action when any cell's final ``‖∇f‖_∞`` exceeds
        ``newton_tolerance``:

        - ``"warn"``: emit a warning at the end of training listing
          the offending cell indices.
        - ``"raise"``: raise ``RuntimeError`` after training.
        - ``"ignore"``: silent.
    fallback_to_encoder : bool, default=False
        Reserved for future work. Implementing this would require
        keeping an encoder around alongside Laplace, which contradicts
        the "no encoder in Laplace mode" design. Documented here for
        forward compatibility only.

    Examples
    --------
    >>> # Default Laplace config: 50k outer steps, 5 inner Newton steps.
    >>> LaplaceConfig()

    >>> # Tighter inner Newton + slower outer for hard problems:
    >>> LaplaceConfig(n_newton_steps=10, n_steps=100_000, damping=1e-3)

    See Also
    --------
    SVIConfig : Configuration for the standard SVI / VAE path.
    """

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    n_steps: int = Field(
        50_000,
        gt=0,
        description="Number of outer SVI steps over global parameters.",
    )
    n_newton_steps: int = Field(
        5,
        ge=1,
        le=200,
        description=(
            "Number of inner Newton iterations per outer SVI step. "
            "Quadratic convergence on log-concave PLN means 5 is "
            "usually plenty from a warm start."
        ),
    )
    newton_tolerance: float = Field(
        1e-4,
        gt=0,
        description=(
            "L∞ gradient-norm tolerance used for convergence "
            "diagnostics (not for early-stopping Newton)."
        ),
    )
    damping: float = Field(
        1e-2,
        ge=0,
        description=(
            "Tikhonov damping added to the diagonal of -H during the "
            "Newton step to stabilise ill-conditioned cells. Default "
            "1e-2 is conservative — the kernel itself converges with "
            "damping=0 in well-conditioned cases, but joint (x, η) "
            "Newton on real data can produce explosive first steps "
            "when the Schur complement on the η block is small. "
            "Power users can drop this below 1e-4 if they observe "
            "the MAP being biased away from the data. NOTE: this "
            "damping only affects the *Newton solver*. The Laplace "
            "correction ``-½ log det(-H)`` in the outer ELBO is "
            "computed at damping=0 against the true posterior "
            "Hessian (see ``laplace_log_det_neg_H`` in "
            "``scribe.laplace._newton_pln``) — so increasing this "
            "knob trades Newton-step stability for solve quality "
            "without biasing the reported ELBO."
        ),
    )
    optimizer: Optional[Any] = Field(
        None,
        description="Pre-built NumPyro optimizer for outer-loop SGD.",
    )
    optimizer_config: Optional["SVIConfig.OptimizerConfig"] = Field(
        None,
        description=(
            "Serializable optimizer specification. Built into a "
            "NumPyro optimizer at runtime when ``optimizer`` is not "
            "explicitly set."
        ),
    )
    batch_size: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Mini-batch size for the outer SVI loop. None uses the "
            "full dataset (batch gradient descent)."
        ),
    )
    early_stopping: Optional[EarlyStoppingConfig] = Field(
        None,
        description="Early stopping configuration for the outer loop.",
    )
    restore_best: bool = Field(
        False,
        description=(
            "Track and restore the best-loss outer-loop parameters."
        ),
    )
    convergence_action: Literal["warn", "raise", "ignore"] = Field(
        "warn",
        description=(
            "Action when Newton fails to converge on some cells: "
            "emit a warning, raise RuntimeError, or stay silent."
        ),
    )
    fallback_to_encoder: bool = Field(
        False,
        description=(
            "Reserved for future work; implementing this requires an "
            "encoder alongside Laplace, which contradicts the no-"
            "encoder design. Currently has no effect."
        ),
    )
    log_progress_lines: bool = Field(
        False,
        description=(
            "When True, the engine prints a plain-text progress line "
            "in addition to the interactive rich/tqdm progress bar at "
            "each periodic update. Useful for non-interactive runs "
            "captured to log files. Mirrors ``SVIConfig.log_progress_lines``."
        ),
    )

    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string."""
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "LaplaceConfig":
        """Deserialize config from YAML string."""
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)


# ==============================================================================
# Data Configuration Group
# ==============================================================================


class DataConfig(BaseModel):
    """Configuration for data processing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cells_axis: int = Field(
        0,
        ge=0,
        le=1,
        description="Axis for cells in count matrix (0=rows, 1=columns)",
    )
    layer: Optional[str] = Field(
        None, description="Layer in AnnData to use for counts. If None, uses .X"
    )


# ==============================================================================
# Unified Inference Configuration Group
# ==============================================================================


class InferenceConfig(BaseModel):
    """Unified inference configuration with method-specific validation.

    This class provides a single interface for all inference method
    configurations (SVI, MCMC, VAE), with automatic validation to ensure the
    correct config type is provided for each inference method.

    Parameters
    ----------
    method : InferenceMethod
        The inference method this configuration is for.
    svi : Optional[SVIConfig], default=None
        SVI-specific configuration. Required if method is SVI or VAE.
    mcmc : Optional[MCMCConfig], default=None
        MCMC-specific configuration. Required if method is MCMC.

    Raises
    ------
    ValueError
        If the wrong config type is provided for the specified inference method,
        or if the required config is missing.

    Examples
    --------
    Create from SVIConfig:

    >>> from scribe.models.config import InferenceConfig, SVIConfig
    >>> svi_config = SVIConfig(n_steps=50000, batch_size=256)
    >>> inference_config = InferenceConfig.from_svi(svi_config)

    Create from MCMCConfig:

    >>> from scribe.models.config import InferenceConfig, MCMCConfig
    >>> mcmc_config = MCMCConfig(n_samples=5000, n_chains=4)
    >>> inference_config = InferenceConfig.from_mcmc(mcmc_config)

    Direct construction (with validation):

    >>> inference_config = InferenceConfig(
    ...     method=InferenceMethod.SVI,
    ...     svi=SVIConfig(n_steps=10000),
    ...     mcmc=None
    ... )

    See Also
    --------
    SVIConfig : Configuration for SVI inference.
    MCMCConfig : Configuration for MCMC inference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Discriminator field - determines which config type should be used
    method: "InferenceMethod" = Field(
        ..., description="Inference method this configuration is for"
    )

    # Method-specific configs (only one should be set based on method)
    svi: Optional[SVIConfig] = Field(
        None, description="SVI configuration (required for SVI and VAE methods)"
    )
    mcmc: Optional[MCMCConfig] = Field(
        None, description="MCMC configuration (required for MCMC method)"
    )
    laplace: Optional[LaplaceConfig] = Field(
        None,
        description=(
            "Laplace configuration (required for LAPLACE method, "
            "currently PLN-only)."
        ),
    )

    # --------------------------------------------------------------------------
    # Factory Methods
    # --------------------------------------------------------------------------

    @classmethod
    def from_svi(cls, svi_config: SVIConfig) -> "InferenceConfig":
        """Create InferenceConfig from SVIConfig.

        Parameters
        ----------
        svi_config : SVIConfig
            SVI configuration object.

        Returns
        -------
        InferenceConfig
            InferenceConfig with method=SVI and the provided svi_config.

        Examples
        --------
        >>> from scribe.models.config import InferenceConfig, SVIConfig
        >>> svi_config = SVIConfig(n_steps=50000)
        >>> inference_config = InferenceConfig.from_svi(svi_config)
        """
        from .enums import InferenceMethod

        return cls(method=InferenceMethod.SVI, svi=svi_config, mcmc=None)

    # --------------------------------------------------------------------------

    @classmethod
    def from_mcmc(cls, mcmc_config: MCMCConfig) -> "InferenceConfig":
        """Create InferenceConfig from MCMCConfig.

        Parameters
        ----------
        mcmc_config : MCMCConfig
            MCMC configuration object.

        Returns
        -------
        InferenceConfig
            InferenceConfig with method=MCMC and the provided mcmc_config.

        Examples
        --------
        >>> from scribe.models.config import InferenceConfig, MCMCConfig
        >>> mcmc_config = MCMCConfig(n_samples=5000)
        >>> inference_config = InferenceConfig.from_mcmc(mcmc_config)
        """
        from .enums import InferenceMethod

        return cls(method=InferenceMethod.MCMC, svi=None, mcmc=mcmc_config)

    # --------------------------------------------------------------------------

    @classmethod
    def from_vae(cls, svi_config: SVIConfig) -> "InferenceConfig":
        """Create InferenceConfig for VAE inference from SVIConfig.

        VAE inference uses SVI configuration since it's essentially SVI with
        neural network components.

        Parameters
        ----------
        svi_config : SVIConfig
            SVI configuration object (used for VAE inference).

        Returns
        -------
        InferenceConfig
            InferenceConfig with method=VAE and the provided svi_config.

        Examples
        --------
        >>> from scribe.models.config import InferenceConfig, SVIConfig
        >>> svi_config = SVIConfig(n_steps=100000)
        >>> inference_config = InferenceConfig.from_vae(svi_config)
        """
        from .enums import InferenceMethod

        return cls(method=InferenceMethod.VAE, svi=svi_config, mcmc=None)

    # --------------------------------------------------------------------------

    @classmethod
    def from_laplace(
        cls, laplace_config: "LaplaceConfig"
    ) -> "InferenceConfig":
        """Create InferenceConfig for Laplace inference (PLN-only).

        Parameters
        ----------
        laplace_config : LaplaceConfig
            Laplace-specific configuration object.

        Returns
        -------
        InferenceConfig
            InferenceConfig with method=LAPLACE and the provided
            laplace_config. ``svi`` and ``mcmc`` are left as ``None``.

        Examples
        --------
        >>> from scribe.models.config import InferenceConfig, LaplaceConfig
        >>> laplace_config = LaplaceConfig(n_steps=50_000, n_newton_steps=8)
        >>> inference_config = InferenceConfig.from_laplace(laplace_config)
        """
        from .enums import InferenceMethod

        return cls(
            method=InferenceMethod.LAPLACE,
            svi=None,
            mcmc=None,
            laplace=laplace_config,
        )

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, v):
        """Convert string to InferenceMethod enum if needed."""
        from .enums import InferenceMethod

        if isinstance(v, str):
            return InferenceMethod(v)
        return v

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------

    def model_post_init(self, __context):
        """Validate that the correct config type is provided for the method."""
        if self.method == InferenceMethod.SVI:
            if self.svi is None:
                raise ValueError("SVIConfig required for SVI inference")
            if self.mcmc is not None:
                raise ValueError("MCMCConfig not allowed for SVI inference")
            if self.laplace is not None:
                raise ValueError(
                    "LaplaceConfig not allowed for SVI inference"
                )
        elif self.method == InferenceMethod.MCMC:
            if self.mcmc is None:
                raise ValueError("MCMCConfig required for MCMC inference")
            if self.svi is not None:
                raise ValueError("SVIConfig not allowed for MCMC inference")
            if self.laplace is not None:
                raise ValueError(
                    "LaplaceConfig not allowed for MCMC inference"
                )
        elif self.method == InferenceMethod.VAE:
            # VAE uses SVI config
            if self.svi is None:
                raise ValueError("SVIConfig required for VAE inference")
            if self.mcmc is not None:
                raise ValueError("MCMCConfig not allowed for VAE inference")
            if self.laplace is not None:
                raise ValueError(
                    "LaplaceConfig not allowed for VAE inference"
                )
        elif self.method == InferenceMethod.LAPLACE:
            if self.laplace is None:
                raise ValueError(
                    "LaplaceConfig required for Laplace inference"
                )
            if self.svi is not None:
                raise ValueError(
                    "SVIConfig not allowed for Laplace inference"
                )
            if self.mcmc is not None:
                raise ValueError(
                    "MCMCConfig not allowed for Laplace inference"
                )
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    # --------------------------------------------------------------------------
    # Accessor Methods
    # --------------------------------------------------------------------------

    def get_config(
        self,
    ) -> Union[SVIConfig, MCMCConfig, "LaplaceConfig"]:
        """Get the appropriate config for the inference method.

        Returns
        -------
        Union[SVIConfig, MCMCConfig, LaplaceConfig]
            The configuration object for the active inference method.

        Raises
        ------
        ValueError
            If the inference method is not recognized.

        Examples
        --------
        >>> inference_config = InferenceConfig.from_svi(SVIConfig())
        >>> svi_config = inference_config.get_config()  # Returns SVIConfig
        """
        from .enums import InferenceMethod

        if (
            self.method == InferenceMethod.SVI
            or self.method == InferenceMethod.VAE
        ):
            return self.svi
        elif self.method == InferenceMethod.MCMC:
            return self.mcmc
        elif self.method == InferenceMethod.LAPLACE:
            return self.laplace
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    # --------------------------------------------------------------------------
    # Serialization Methods
    # --------------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize config to YAML string.

        Returns
        -------
        str
            YAML representation of the config.

        Examples
        --------
        >>> config = InferenceConfig.from_svi(SVIConfig())
        >>> yaml_str = config.to_yaml()
        >>> print(yaml_str)
        """
        import yaml

        data = self.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "InferenceConfig":
        """Deserialize config from YAML string.

        Parameters
        ----------
        yaml_str : str
            YAML string representation of the config.

        Returns
        -------
        InferenceConfig
            Deserialized config object.

        Examples
        --------
        >>> yaml_str = '''
        ... method: svi
        ... svi:
        ...   n_steps: 50000
        ... '''
        >>> config = InferenceConfig.from_yaml(yaml_str)
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)

    def to_yaml_file(self, path: str) -> None:
        """Save config to YAML file.

        Parameters
        ----------
        path : str
            Path to save the YAML file.

        Examples
        --------
        >>> config.to_yaml_file("inference_config.yaml")
        """
        with open(path, "w") as f:
            f.write(self.to_yaml())

    @classmethod
    def from_yaml_file(cls, path: str) -> "InferenceConfig":
        """Load config from YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        InferenceConfig
            Loaded config object.

        Examples
        --------
        >>> config = InferenceConfig.from_yaml_file("inference_config.yaml")
        """
        with open(path) as f:
            return cls.from_yaml(f.read())
