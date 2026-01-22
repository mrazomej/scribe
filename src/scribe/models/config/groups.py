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

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import jax.numpy as jnp
from .enums import VAEPriorType, VAEMaskType, VAEActivation, InferenceMethod

# Import GuideFamily for Pydantic's runtime type checking
# We use Any as the type hint since GuideFamily is a dataclass that Pydantic
# doesn't natively understand, and we have arbitrary_types_allowed=True
from ..components.guide_families import GuideFamily

# PriorConfig, GuideConfig, UnconstrainedPriorConfig, and
# UnconstrainedGuideConfig have been removed. Prior and guide hyperparameters
# are now stored directly in ParamSpec objects (see
# scribe.models.builders.parameter_specs).


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
        Guide family for the success probability parameter.
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
        None, description="Guide family for success probability"
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
        statistic. Options: "log1p", "log", "sqrt", "identity".

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
        description="Transformation for input data (log1p, log, sqrt, identity)",
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
        valid_transforms = {"log1p", "log", "sqrt", "identity"}
        if v.lower() not in valid_transforms:
            raise ValueError(
                f"Unknown transformation '{v}'. "
                f"Valid options: {valid_transforms}"
            )
        return v.lower()

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
    """VAE-specific configuration with validation."""

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    # Architecture
    latent_dim: int = Field(3, gt=0, description="Latent space dimensionality")
    hidden_dims: Optional[List[int]] = Field(
        None, description="Encoder/decoder hidden layer sizes"
    )
    activation: Optional[VAEActivation] = Field(
        None, description="Activation function"
    )
    input_transformation: Optional[str] = None

    # VCP encoder (for variable capture models)
    vcp_hidden_dims: Optional[List[int]] = None
    vcp_activation: Optional[VAEActivation] = None

    # Prior configuration
    prior_type: VAEPriorType = Field(
        VAEPriorType.STANDARD, description="Prior type"
    )
    prior_num_layers: int = Field(
        2, gt=0, description="Number of layers for decoupled prior"
    )
    prior_hidden_dims: Optional[List[int]] = None
    prior_activation: Optional[VAEActivation] = None
    prior_mask_type: VAEMaskType = Field(
        VAEMaskType.ALTERNATING, description="Mask type for decoupled prior"
    )

    # Preprocessing
    standardize: bool = Field(
        False, description="Whether to standardize input data"
    )
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None

    @field_validator("hidden_dims", "vcp_hidden_dims", "prior_hidden_dims")
    @classmethod
    def validate_hidden_dims(
        cls, v: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Validate hidden dimensions are positive."""
        if v is not None and any(d <= 0 for d in v):
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
    min_delta_pct : float, default=0.01
        Minimum relative improvement (as percentage) in smoothed loss to qualify
        as progress. Computed as `100 * (best_loss - smoothed_loss) /
        best_loss`. The default of 0.01 means 0.01% improvement is required.
        This works across different dataset sizes since it scales with loss
        magnitude. For example, if the loss is 1,000,000, an improvement of at
        least 100 is required to exceed the 0.01% threshold.
    check_every : int, default=10
        How often (in steps) to check for convergence. Checking every step adds
        overhead; checking less frequently may miss the optimal stop point.
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

    >>> # More aggressive early stopping
    >>> config = EarlyStoppingConfig(
    ...     patience=200,
    ...     min_delta_pct=0.1,  # 0.1% improvement required
    ...     check_every=5,
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
    min_delta_pct: float = Field(
        0.01,
        ge=0,
        description=(
            "Minimum relative improvement (as percentage) to qualify as progress. "
            "Default 0.01 means 0.01% improvement required. "
            "Scales automatically with loss magnitude."
        ),
    )
    check_every: int = Field(
        10,
        gt=0,
        description="Check convergence every N steps",
    )
    smoothing_window: int = Field(
        50,
        gt=0,
        description="Window size for loss smoothing (moving average)",
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
# SVI Configuration Group
# ==============================================================================


class SVIConfig(BaseModel):
    """Configuration for Stochastic Variational Inference.

    This class configures the SVI optimization process, including the optimizer,
    loss function, number of steps, mini-batching, and early stopping.

    Parameters
    ----------
    optimizer : Any, optional
        Optimizer for variational inference. Defaults to Adam with
        step_size=0.001 if not specified.
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

    model_config = ConfigDict(
        frozen=True, arbitrary_types_allowed=True, extra="forbid"
    )

    optimizer: Optional[Any] = Field(
        None,
        description="Optimizer for variational inference (defaults to Adam)",
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
    early_stopping: Optional[EarlyStoppingConfig] = Field(
        None,
        description="Early stopping configuration. If None, disabled.",
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
        elif self.method == InferenceMethod.MCMC:
            if self.mcmc is None:
                raise ValueError("MCMCConfig required for MCMC inference")
            if self.svi is not None:
                raise ValueError("SVIConfig not allowed for MCMC inference")
        elif self.method == InferenceMethod.VAE:
            # VAE uses SVI config
            if self.svi is None:
                raise ValueError("SVIConfig required for VAE inference")
            if self.mcmc is not None:
                raise ValueError("MCMCConfig not allowed for VAE inference")
        else:
            raise ValueError(f"Unknown inference method: {self.method}")

    # --------------------------------------------------------------------------
    # Accessor Methods
    # --------------------------------------------------------------------------

    def get_config(self) -> Union[SVIConfig, MCMCConfig]:
        """Get the appropriate config for the inference method.

        Returns
        -------
        Union[SVIConfig, MCMCConfig]
            The SVI or MCMC configuration, depending on the inference method.

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
