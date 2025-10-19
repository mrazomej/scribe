"""Builder pattern for constructing ModelConfig instances."""

from typing import Optional, Dict, Any, Union
from .enums import ModelType, Parameterization, InferenceMethod, VAEPriorType
from .groups import (
    PriorConfig,
    GuideConfig,
    UnconstrainedPriorConfig,
    UnconstrainedGuideConfig,
    VAEConfig,
)
from .base import ConstrainedModelConfig, UnconstrainedModelConfig, ModelConfig

# ==============================================================================
# Model Configuration Builder Class
# ==============================================================================


class ModelConfigBuilder:
    """
    ModelConfigBuilder provides a fluent builder pattern for constructing
    validated, immutable, and type-safe model configuration objects
    (ModelConfig) used in SCRIBE models.

    This class enables users to construct complex model configurations through a
    chainable API, specifying model type, parameterization, inference method,
    priors, guides, variational settings, and more, with each chosen option
    validated for compatibility. Configurations assembled through this builder
    are guaranteed to conform to the strict type and validity requirements
    enforced throughout the SCRIBE configuration system.

    The builder tracks all relevant components (such as model types,
    parameterizations, priors, variational parameters, and mixture behaviors)
    and then assembles a complete ModelConfig instance, ensuring robust
    correctness and a clear, user-friendly experience for interactive use,
    scripting, or integration with pipelines.

    Parameters
    ----------
    None at initialization. All configuration is done via method chaining.

    Notes
    -----
    - The builder enforces validation at each step, refusing to build invalid or
      inconsistent configurations.
    - All constructed config objects are immutable and type-checked.
    - Supports both constrained and unconstrained configs as well as advanced
      geometry (mixture, VAE) and inference options.

    Examples
    --------
    Simple SVI model with default settings:
        >>> config = (ModelConfigBuilder()
        ...     .for_model("nbdm")
        ...     .with_inference("svi")
        ...     .build())

    Unconstrained linked model with specified priors:
        >>> config = (ModelConfigBuilder()
        ...     .for_model("zinb")
        ...     .with_parameterization("linked")
        ...     .unconstrained()
        ...     .with_priors(p=(1.0, 1.0), mu=(0.0, 1.0))
        ...     .build())

    Variational Autoencoder (VAE) with custom latent size and activation:
        >>> config = (ModelConfigBuilder()
        ...     .for_model("nbdm")
        ...     .with_vae(latent_dim=5, hidden_dims=[256, 128], activation="gelu")
        ...     .build())

    Mixture model:
        >>> config = (ModelConfigBuilder()
        ...     .for_model("zinb")
        ...     .as_mixture(n_components=3)
        ...     .build())
    """

    def __init__(self):
        """Initialize builder with default values."""
        self._base_model: Optional[str] = None
        self._parameterization: Parameterization = Parameterization.STANDARD
        self._inference_method: InferenceMethod = InferenceMethod.SVI
        self._unconstrained: bool = False
        self._n_components: Optional[int] = None
        self._component_specific_params: bool = False
        self._guide_rank: Optional[int] = None
        self._priors: Dict[str, Any] = {}
        self._guides: Dict[str, Any] = {}
        self._vae_params: Dict[str, Any] = {}

    # --------------------------------------------------------------------------

    def for_model(
        self, model_type: Union[str, ModelType]
    ) -> "ModelConfigBuilder":
        """Set the base model type.

        Parameters
        ----------
        model_type : str or ModelType
            Model type (e.g., "nbdm", "zinb", "nbvcp", "zinbvcp")
        """
        if isinstance(model_type, ModelType):
            self._base_model = model_type.value
        else:
            self._base_model = model_type
        return self

    # --------------------------------------------------------------------------

    def with_parameterization(
        self, param: Union[str, Parameterization]
    ) -> "ModelConfigBuilder":
        """Set the parameterization type.

        Parameters
        ----------
        param : str or Parameterization
            Parameterization type ("standard", "linked", "odds_ratio")
        """
        if isinstance(param, str):
            self._parameterization = Parameterization(param)
        else:
            self._parameterization = param
        return self

    # --------------------------------------------------------------------------

    def with_inference(
        self, method: Union[str, InferenceMethod]
    ) -> "ModelConfigBuilder":
        """Set the inference method.

        Parameters
        ----------
        method : str or InferenceMethod
            Inference method ("svi", "mcmc", "vae")
        """
        if isinstance(method, str):
            self._inference_method = InferenceMethod(method)
        else:
            self._inference_method = method
        return self

    # --------------------------------------------------------------------------

    def unconstrained(self) -> "ModelConfigBuilder":
        """Use unconstrained parameterization."""
        self._unconstrained = True
        return self

    # --------------------------------------------------------------------------

    def as_mixture(
        self, n_components: int, component_specific: bool = False
    ) -> "ModelConfigBuilder":
        """Configure as mixture model.

        Parameters
        ----------
        n_components : int
            Number of mixture components (must be >= 2)
        component_specific : bool
            Whether to use component-specific parameters
        """
        if n_components < 2:
            raise ValueError("n_components must be >= 2 for mixture models")
        self._n_components = n_components
        self._component_specific_params = component_specific
        return self

    # --------------------------------------------------------------------------

    def with_low_rank_guide(self, rank: int) -> "ModelConfigBuilder":
        """Use low-rank guide.

        Parameters
        ----------
        rank : int
            Rank for low-rank guide (must be > 0)
        """
        if rank <= 0:
            raise ValueError("guide_rank must be positive")
        self._guide_rank = rank
        return self

    # --------------------------------------------------------------------------

    def with_priors(self, **priors) -> "ModelConfigBuilder":
        """Set prior parameters.

        Parameters
        ----------
        **priors
            Prior parameters (e.g., p=(1.0, 1.0), r=(2.0, 0.5))
        """
        self._priors.update(priors)
        return self

    # --------------------------------------------------------------------------

    def with_guides(self, **guides) -> "ModelConfigBuilder":
        """Set guide parameters.

        Parameters
        ----------
        **guides
            Guide parameters
        """
        self._guides.update(guides)
        return self

    # --------------------------------------------------------------------------

    def with_vae(
        self,
        latent_dim: int = 3,
        hidden_dims: Optional[list] = None,
        activation: Optional[str] = None,
        prior_type: Union[str, VAEPriorType] = VAEPriorType.STANDARD,
        **kwargs,
    ) -> "ModelConfigBuilder":
        """Configure VAE parameters.

        Parameters
        ----------
        latent_dim : int
            Latent space dimensionality
        hidden_dims : list, optional
            Hidden layer sizes
        activation : str, optional
            Activation function ("relu", "gelu", "tanh")
        prior_type : str or VAEPriorType
            Prior type ("standard" or "decoupled")
        **kwargs
            Additional VAE parameters
        """
        self._inference_method = InferenceMethod.VAE
        self._vae_params = {
            "latent_dim": latent_dim,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "prior_type": (
                prior_type
                if isinstance(prior_type, VAEPriorType)
                else VAEPriorType(prior_type)
            ),
            **kwargs,
        }
        return self

    # --------------------------------------------------------------------------

    def build(self) -> ModelConfig:
        """Build and validate the configuration.

        Returns
        -------
        ModelConfig
            Validated model configuration (either ConstrainedModelConfig or
            UnconstrainedModelConfig)

        Raises
        ------
        ValueError
            If base_model is not set or configuration is invalid
        """
        if self._base_model is None:
            raise ValueError(
                "base_model is required. Use .for_model() to set it."
            )

        # Append _mix if mixture model
        base_model = self._base_model
        if self._n_components is not None:
            base_model = f"{self._base_model}_mix"

        # Create VAE config if VAE inference
        vae_config = None
        if self._inference_method == InferenceMethod.VAE:
            vae_config = VAEConfig(**self._vae_params)

        # Build appropriate config type
        if self._unconstrained:
            return UnconstrainedModelConfig(
                base_model=base_model,
                parameterization=self._parameterization,
                inference_method=self._inference_method,
                n_components=self._n_components,
                component_specific_params=self._component_specific_params,
                guide_rank=self._guide_rank,
                priors=UnconstrainedPriorConfig(**self._priors),
                guides=UnconstrainedGuideConfig(**self._guides),
                vae=vae_config,
            )
        else:
            return ConstrainedModelConfig(
                base_model=base_model,
                parameterization=self._parameterization,
                inference_method=self._inference_method,
                n_components=self._n_components,
                component_specific_params=self._component_specific_params,
                guide_rank=self._guide_rank,
                priors=PriorConfig(**self._priors),
                guides=GuideConfig(**self._guides),
                vae=vae_config,
            )
