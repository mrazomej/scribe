"""Builder pattern for constructing ModelConfig instances."""

from typing import Optional, Dict, Any, Union, List
from .enums import ModelType, Parameterization, InferenceMethod, VAEPriorType
from .groups import (
    VAEConfig,
    GuideFamilyConfig,
)
from .base import ModelConfig
from .parameter_mapping import get_required_parameters
from ..builders.parameter_specs import ParamSpec

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
        self._mixture_params: Optional[List[str]] = None
        self._guide_families: Optional[GuideFamilyConfig] = None
        self._priors: Dict[str, Any] = (
            {}
        )  # For backward compatibility with with_priors()
        self._guides: Dict[str, Any] = (
            {}
        )  # For backward compatibility with with_guides()
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
        self,
        n_components: int,
        mixture_params: Optional[List[str]] = None,
    ) -> "ModelConfigBuilder":
        """Configure as mixture model.

        Parameters
        ----------
        n_components : int
            Number of mixture components (must be >= 2)
        mixture_params : List[str], optional
            List of parameter names that should be mixture-specific.
            If None, all gene-specific parameters will be mixture-specific
            by default.
        """
        if n_components < 2:
            raise ValueError("n_components must be >= 2 for mixture models")
        self._n_components = n_components
        self._mixture_params = mixture_params
        return self

    # --------------------------------------------------------------------------

    def with_guide_families(
        self, guide_families: GuideFamilyConfig
    ) -> "ModelConfigBuilder":
        """Set per-parameter guide family configuration.

        Parameters
        ----------
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration specifying which
            variational families to use for each parameter.
        """
        self._guide_families = guide_families
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

    def _get_default_priors(self) -> Dict[str, Any]:
        """Get sensible default priors based on model configuration."""
        defaults = {}

        # Get required parameters for this configuration
        required_params = get_required_parameters(
            self._parameterization,
            self._base_model,
            self._n_components is not None,
            "zinb" in self._base_model if self._base_model else False,
            "vcp" in self._base_model if self._base_model else False,
        )

        # Set defaults based on constrained vs unconstrained
        if self._unconstrained:
            # Unconstrained: Normal distributions
            for param in required_params:
                if param in [
                    "p",
                    "r",
                    "mu",
                    "gate",
                    "p_capture",
                    "phi",
                    "phi_capture",
                ]:
                    defaults[param] = (0.0, 1.0)  # Normal(0, 1)
                elif param == "mixing":
                    defaults[param] = tuple(
                        [0.0] * self._n_components
                    )  # Normal for each
        else:
            # Constrained: Appropriate constrained distributions
            for param in required_params:
                if param in ["p", "gate", "p_capture"]:
                    defaults[param] = (1.0, 1.0)  # Beta(1,1) = Uniform
                elif param in ["r", "mu"]:
                    defaults[param] = (0.0, 1.0)  # LogNormal(0,1)
                elif param in ["phi", "phi_capture"]:
                    defaults[param] = (1.0, 1.0)  # BetaPrime(1,1)
                elif param == "mixing":
                    defaults[param] = tuple(
                        [1.0] * self._n_components
                    )  # Dirichlet

        return defaults

    # --------------------------------------------------------------------------

    def _apply_defaults(self) -> None:
        """Apply default priors for any missing parameters."""
        defaults = self._get_default_priors()

        # Only set defaults for priors that user hasn't provided
        for param, default_value in defaults.items():
            if param not in self._priors:
                self._priors[param] = default_value

    # --------------------------------------------------------------------------

    def build(self) -> ModelConfig:
        """Build and validate the configuration.

        Returns
        -------
        ModelConfig
            Validated unified model configuration

        Raises
        ------
        ValueError
            If base_model is not set or configuration is invalid
        """
        if self._base_model is None:
            raise ValueError(
                "base_model is required. Use .for_model() to set it."
            )

        # Append _mix if mixture model (but not if already ends with _mix)
        base_model = self._base_model
        if self._n_components is not None and not self._base_model.endswith(
            "_mix"
        ):
            base_model = f"{self._base_model}_mix"

        # Create VAE config if VAE inference
        vae_config = None
        if self._inference_method == InferenceMethod.VAE:
            vae_config = VAEConfig(**self._vae_params)

        # param_specs will be created by preset factories
        # The builder doesn't create them directly - that's handled by the
        # preset factories which have access to the full model context
        # For now, pass empty list - preset factories will populate it
        param_specs: List[ParamSpec] = []

        # Build unified ModelConfig
        return ModelConfig(
            base_model=base_model,
            parameterization=self._parameterization,
            inference_method=self._inference_method,
            unconstrained=self._unconstrained,
            n_components=self._n_components,
            mixture_params=self._mixture_params,
            guide_families=self._guide_families,
            param_specs=param_specs,
            vae=vae_config,
        )
