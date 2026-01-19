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
# SVI Configuration Group
# ==============================================================================


class SVIConfig(BaseModel):
    """Configuration for Stochastic Variational Inference."""

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
        100_000, gt=0, description="Number of optimization steps"
    )
    batch_size: Optional[int] = Field(
        None, gt=0, description="Mini-batch size. If None, uses full dataset"
    )
    stable_update: bool = Field(
        True, description="Use numerically stable parameter updates"
    )


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
