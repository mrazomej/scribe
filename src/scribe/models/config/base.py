"""Base model configuration classes using Pydantic."""

from typing import Optional, Set, Union
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    computed_field,
    ConfigDict,
)
from .enums import ModelType, Parameterization, InferenceMethod
from .groups import (
    PriorConfig,
    GuideConfig,
    UnconstrainedPriorConfig,
    UnconstrainedGuideConfig,
    VAEConfig,
)
from .parameter_mapping import get_active_parameters

# ==============================================================================
# Constrained Model Configuration Class
# ==============================================================================


class ConstrainedModelConfig(BaseModel):
    """
    Constrained (standard) model configuration.

    This class defines the complete set of configuration options for SCRIBE
    models using "constrained" (interpretable, strongly-typed) parameters and
    priors—for example, using distributions like Beta, LogNormal, or Dirichlet
    with semantic support. The class enforces correctness, immutability, and
    clarity of model setup.

    This is not constructed directly; users should assemble configurations via
    the ModelConfigBuilder, which ensures validity and best practices.

    Parameters
    ----------
    base_model : str
        The core model family (e.g., 'nbdm', 'zinb_mix', etc.).
    parameterization : Parameterization
        How parameters are represented internally (standard, linked, etc.).
    inference_method : InferenceMethod
        Inference engine type (SVI, MCMC, VAE, etc.).
    n_components : int, optional
        Number of mixture components, if mixture modeling is enabled.
    component_specific_params : bool
        If True, each mixture component has independent parameters.
    guide_rank : int, optional
        For low-rank guide inference, the latent rank.
    priors : PriorConfig
        Grouped prior distribution configurations.
    guides : GuideConfig
        Guide (variational family) specification for SVI/VAE inference.
    vae : VAEConfig, optional
        Nested configuration for Variational Autoencoders, if applicable.

    Notes
    -----
    - Configuration objects are immutable and validated automatically on
      creation.
    - Unrecognized parameters are forbidden and will raise validation errors.
    - Intended for safe, reproducible, and fully-specified SCRIBE model setups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core configuration
    base_model: str = Field(
        ..., description="Model type (e.g., 'nbdm', 'zinb_mix')"
    )
    parameterization: Parameterization = Field(
        Parameterization.STANDARD, description="Parameterization type"
    )
    inference_method: InferenceMethod = Field(
        InferenceMethod.SVI, description="Inference method"
    )

    # Mixture configuration
    n_components: Optional[int] = Field(
        None, gt=1, description="Number of mixture components"
    )
    component_specific_params: bool = Field(
        False, description="Component-specific parameters"
    )

    # Guide configuration
    guide_rank: Optional[int] = Field(
        None, gt=0, description="Low-rank guide rank"
    )

    # Parameter configurations
    priors: PriorConfig = Field(default_factory=PriorConfig)
    guides: GuideConfig = Field(default_factory=GuideConfig)
    vae: Optional[VAEConfig] = Field(
        None, description="VAE configuration (if using VAE inference)"
    )

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        """Validate base model type."""
        base = v.replace("_mix", "")
        valid_models = {m.value for m in ModelType}
        if base not in valid_models:
            raise ValueError(
                f"Invalid model type: {v}. Must be one of {valid_models}"
            )
        return v

    # --------------------------------------------------------------------------

    @field_validator("vae")
    @classmethod
    def validate_vae_inference(
        cls, v: Optional[VAEConfig], info
    ) -> Optional[VAEConfig]:
        """Validate VAE config is provided for VAE inference."""
        if (
            info.data.get("inference_method") == InferenceMethod.VAE
            and v is None
        ):
            # Provide default VAE config
            return VAEConfig()
        return v

    # --------------------------------------------------------------------------
    # Computed Fields
    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_mixture(self) -> bool:
        """Check if this is a mixture model."""
        return self.n_components is not None and self.n_components > 1

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_zero_inflated(self) -> bool:
        """Check if this is a zero-inflated model."""
        return "zinb" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def uses_variable_capture(self) -> bool:
        """Check if this model uses variable capture."""
        return "vcp" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def active_parameters(self) -> Set[str]:
        """Get the set of active parameters for this configuration."""
        return get_active_parameters(
            parameterization=self.parameterization,
            model_type=self.base_model,
            is_mixture=self.is_mixture,
            is_zero_inflated=self.is_zero_inflated,
            uses_variable_capture=self.uses_variable_capture,
        )

    # --------------------------------------------------------------------------

    def with_updated_priors(self, **priors) -> "ConstrainedModelConfig":
        """Create a new config with updated priors (immutable pattern)."""
        return self.model_copy(
            update={"priors": self.priors.model_copy(update=priors)}
        )

    # --------------------------------------------------------------------------

    def with_updated_vae(self, **vae_params) -> "ConstrainedModelConfig":
        """Create a new config with updated VAE parameters."""
        if self.vae is None:
            raise ValueError("Cannot update VAE config when VAE is None")
        return self.model_copy(
            update={"vae": self.vae.model_copy(update=vae_params)}
        )


# ==============================================================================
# Unconstrained Model Configuration Class
# ==============================================================================


class UnconstrainedModelConfig(BaseModel):
    """
    Unconstrained model configuration.

    This class defines the complete set of configuration options for SCRIBE
    models using "unconstrained" (unconstrained, non-interpretable) parameters
    and priors—for example, using Normal distributions without semantic support.
    The class enforces correctness, immutability, and clarity of model setup.

    This is not constructed directly; users should assemble configurations via

    Parameters
    ----------
    base_model : str
        The core model family (e.g., 'nbdm', 'zinb_mix', etc.).
    parameterization : Parameterization
        How parameters are represented internally (standard, linked, etc.).
    inference_method : InferenceMethod
        Inference engine type (SVI, MCMC, VAE, etc.).
    n_components : int, optional
        Number of mixture components, if mixture modeling is enabled.
    component_specific_params : bool
        If True, each mixture component has independent parameters.
    guide_rank : int, optional
        For low-rank guide inference, the latent rank.
    priors : UnconstrainedPriorConfig
        Grouped prior distribution configurations.
    guides : UnconstrainedGuideConfig
        Guide (variational family) specification for SVI/VAE inference.
    vae : VAEConfig, optional
        Nested configuration for Variational Autoencoders, if applicable.

    Notes
    -----
    - Configuration objects are immutable and validated automatically on
      creation.
    - Unrecognized parameters are forbidden and will raise validation errors.
    - Intended for safe, reproducible, and fully-specified SCRIBE model setups.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Core configuration
    base_model: str = Field(..., description="Model type")
    parameterization: Parameterization = Field(
        Parameterization.STANDARD, description="Parameterization type"
    )
    inference_method: InferenceMethod = Field(
        InferenceMethod.SVI, description="Inference method"
    )

    # Mixture configuration
    n_components: Optional[int] = Field(None, gt=1)
    component_specific_params: bool = False

    # Guide configuration
    guide_rank: Optional[int] = Field(None, gt=0)

    # Parameter configurations
    priors: UnconstrainedPriorConfig = Field(
        default_factory=UnconstrainedPriorConfig
    )
    guides: UnconstrainedGuideConfig = Field(
        default_factory=UnconstrainedGuideConfig
    )
    vae: Optional[VAEConfig] = Field(None)

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v: str) -> str:
        """Validate base model type."""
        base = v.replace("_mix", "")
        valid_models = {m.value for m in ModelType}
        if base not in valid_models:
            raise ValueError(f"Invalid model type: {v}")
        return v

    # --------------------------------------------------------------------------

    @field_validator("vae")
    @classmethod
    def validate_vae_inference(
        cls, v: Optional[VAEConfig], info
    ) -> Optional[VAEConfig]:
        """Validate VAE config for VAE inference."""
        if (
            info.data.get("inference_method") == InferenceMethod.VAE
            and v is None
        ):
            return VAEConfig()
        return v

    # --------------------------------------------------------------------------
    # Computed Fields
    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_mixture(self) -> bool:
        """Check if this is a mixture model."""
        return self.n_components is not None and self.n_components > 1

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def is_zero_inflated(self) -> bool:
        """Check if this is a zero-inflated model."""
        return "zinb" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def uses_variable_capture(self) -> bool:
        return "vcp" in self.base_model

    # --------------------------------------------------------------------------

    @computed_field
    @property
    def active_parameters(self) -> Set[str]:
        """Get active parameters (same logic as constrained)."""
        return get_active_parameters(
            parameterization=self.parameterization,
            model_type=self.base_model,
            is_mixture=self.is_mixture,
            is_zero_inflated=self.is_zero_inflated,
            uses_variable_capture=self.uses_variable_capture,
        )

    def with_updated_priors(self, **priors) -> "UnconstrainedModelConfig":
        """Create a new config with updated priors."""
        return self.model_copy(
            update={"priors": self.priors.model_copy(update=priors)}
        )


# ==============================================================================
# Model Configuration Type Alias
# ==============================================================================

# Type alias for both config types
ModelConfig = Union[ConstrainedModelConfig, UnconstrainedModelConfig]
