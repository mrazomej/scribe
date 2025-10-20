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

from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import jax.numpy as jnp
from .enums import VAEPriorType, VAEMaskType, VAEActivation

# ==============================================================================
# Prior Configuration Group
# ==============================================================================


class PriorConfig(BaseModel):
    """Prior parameters with automatic validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p: Optional[Tuple[float, float]] = Field(
        None, description="Success probability prior (Beta)"
    )
    r: Optional[Tuple[float, float]] = Field(
        None, description="Dispersion prior (LogNormal)"
    )
    mu: Optional[Tuple[float, float]] = Field(
        None, description="Mean prior (LogNormal)"
    )
    phi: Optional[Tuple[float, float]] = Field(
        None, description="Odds ratio prior (BetaPrime)"
    )
    gate: Optional[Tuple[float, float]] = Field(
        None, description="Zero-inflation gate prior"
    )
    p_capture: Optional[Tuple[float, float]] = Field(
        None, description="Capture probability prior"
    )
    phi_capture: Optional[Tuple[float, float]] = Field(
        None, description="Capture phi prior"
    )
    mixing: Optional[Tuple[float, ...]] = Field(
        None, description="Mixture weights prior (Dirichlet)"
    )

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("p", "phi", "gate", "p_capture", "phi_capture")
    @classmethod
    def validate_positive_params(
        cls, v: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate that parameters are positive."""
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"Prior must be a 2-tuple, got {len(v)}")
            if any(x <= 0 for x in v):
                raise ValueError(f"Prior parameters must be positive, got {v}")
        return v

    @field_validator("r", "mu")
    @classmethod
    def validate_lognormal_params(
        cls, v: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Validate LogNormal parameters (location can be zero/negative, scale must
        be positive).
        """
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"Prior must be a 2-tuple, got {len(v)}")
            if v[1] <= 0:  # Scale parameter must be positive
                raise ValueError(
                    f"LogNormal scale parameter must be positive, got {v}"
                )
        return v

    # --------------------------------------------------------------------------

    @field_validator("mixing")
    @classmethod
    def validate_mixing(
        cls, v: Optional[Tuple[float, ...]]
    ) -> Optional[Tuple[float, ...]]:
        """Validate mixing parameters."""
        if v is not None and any(x <= 0 for x in v):
            raise ValueError(f"Mixing parameters must be positive, got {v}")
        return v


# ==============================================================================
# Guide Configuration Group
# ==============================================================================


class GuideConfig(BaseModel):
    """Guide parameters with automatic validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p: Optional[Tuple[float, float]] = None
    r: Optional[Tuple[float, float]] = None
    mu: Optional[Tuple[float, float]] = None
    phi: Optional[Tuple[float, float]] = None
    gate: Optional[Tuple[float, float]] = None
    p_capture: Optional[Tuple[float, float]] = None
    phi_capture: Optional[Tuple[float, float]] = None
    mixing: Optional[Tuple[float, ...]] = None

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("p", "r", "mu", "phi", "gate", "p_capture", "phi_capture")
    @classmethod
    def validate_positive_params(
        cls, v: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate that parameters are positive."""
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"Guide must be a 2-tuple, got {len(v)}")
            if any(x <= 0 for x in v):
                raise ValueError(f"Guide parameters must be positive, got {v}")
        return v


# ==============================================================================
# Unconstrained Prior Configuration Group
# ==============================================================================


class UnconstrainedPriorConfig(BaseModel):
    """Unconstrained prior parameters (Normal distributions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p: Optional[Tuple[float, float]] = Field(
        None, description="p_unconstrained prior (Normal)"
    )
    r: Optional[Tuple[float, float]] = Field(
        None, description="r_unconstrained prior (Normal)"
    )
    mu: Optional[Tuple[float, float]] = Field(
        None, description="mu_unconstrained prior (Normal)"
    )
    phi: Optional[Tuple[float, float]] = Field(
        None, description="phi_unconstrained prior (Normal)"
    )
    gate: Optional[Tuple[float, float]] = Field(
        None, description="gate_unconstrained prior (Normal)"
    )
    p_capture: Optional[Tuple[float, float]] = Field(
        None, description="p_capture_unconstrained prior (Normal)"
    )
    phi_capture: Optional[Tuple[float, float]] = Field(
        None, description="phi_capture_unconstrained prior (Normal)"
    )
    mixing: Optional[Tuple[float, ...]] = Field(
        None, description="mixing_unconstrained prior (Normal)"
    )

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("p", "r", "mu", "phi", "gate", "p_capture", "phi_capture")
    @classmethod
    def validate_tuple_length(
        cls, v: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Validate that parameters are 2-tuples."""
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"Prior must be a 2-tuple, got {len(v)}")
        return v

    # --------------------------------------------------------------------------

    @field_validator("mixing")
    @classmethod
    def validate_mixing_tuple_length(
        cls, v: Optional[Tuple[float, ...]]
    ) -> Optional[Tuple[float, ...]]:
        """Validate mixing parameters are tuples."""
        if v is not None:
            if len(v) < 2:
                raise ValueError(
                    f"Mixing must have at least 2 elements, got {len(v)}"
                )
        return v


# ==============================================================================
# Unconstrained Guide Configuration Group
# ==============================================================================


class UnconstrainedGuideConfig(BaseModel):
    """Unconstrained guide parameters (Normal distributions)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p: Optional[Tuple[float, float]] = None
    r: Optional[Tuple[float, float]] = None
    mu: Optional[Tuple[float, float]] = None
    phi: Optional[Tuple[float, float]] = None
    gate: Optional[Tuple[float, float]] = None
    p_capture: Optional[Tuple[float, float]] = None
    phi_capture: Optional[Tuple[float, float]] = None
    mixing: Optional[Tuple[float, ...]] = None


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
