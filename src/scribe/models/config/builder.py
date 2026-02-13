"""Builder pattern for constructing ModelConfig instances."""

from typing import Optional, Dict, Any, Union, List
from .enums import ModelType, Parameterization, InferenceMethod
from .groups import (
    VAEConfig,
    GuideFamilyConfig,
    PriorOverrides,
)
from .base import ModelConfig
from .parameter_mapping import get_required_parameters
from ..builders.parameter_specs import ParamSpec
from ..components.guide_families import LowRankGuide

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
            If None, all sampled core parameters for the selected
            parameterization will be mixture-specific by default.
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
        latent_dim: int = 10,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        input_transform: str = "log1p",
        standardize: bool = False,
        decoder_transforms: Optional[Dict[str, str]] = None,
        flow_type: str = "none",
        flow_num_layers: int = 4,
        flow_hidden_dims: Optional[List[int]] = None,
    ) -> "ModelConfigBuilder":
        """Configure VAE parameters with composable architecture.

        Parameters
        ----------
        latent_dim : int
            Latent space dimensionality.
        encoder_hidden_dims : List[int], optional
            Encoder hidden layer sizes. Default: [128, 64].
        decoder_hidden_dims : List[int], optional
            Decoder hidden layer sizes. Default: [64, 128].
        activation : str
            Activation function (relu, gelu, silu, tanh, elu, leaky_relu).
        input_transform : str
            Input transformation before encoder (log1p, log, sqrt, identity).
        standardize : bool
            Whether to standardize input data.
        decoder_transforms : Dict[str, str], optional
            Per-param override of decoder output transforms.
        flow_type : str
            Flow prior type: none, coupling_affine, coupling_spline, maf, iaf.
        flow_num_layers : int
            Number of flow layers.
        flow_hidden_dims : List[int], optional
            Flow conditioner hidden dimensions. Default: [64, 64].
        """
        self._inference_method = InferenceMethod.VAE
        vae_params = {
            "latent_dim": latent_dim,
            "activation": activation,
            "input_transform": input_transform,
            "standardize": standardize,
            "flow_type": flow_type,
            "flow_num_layers": flow_num_layers,
        }
        if encoder_hidden_dims is not None:
            vae_params["encoder_hidden_dims"] = encoder_hidden_dims
        if decoder_hidden_dims is not None:
            vae_params["decoder_hidden_dims"] = decoder_hidden_dims
        if decoder_transforms is not None:
            vae_params["decoder_transforms"] = decoder_transforms
        if flow_hidden_dims is not None:
            vae_params["flow_hidden_dims"] = flow_hidden_dims
        self._vae_params = vae_params
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

    def _prior_value_to_tuple(self, value: Any) -> tuple:
        """Convert prior value to tuple (handles JAX/NumPy arrays)."""
        if isinstance(value, (tuple, list)):
            return tuple(value)
        if hasattr(value, "tolist"):
            return tuple(value.tolist())
        if hasattr(value, "__iter__") and not isinstance(value, (str, dict)):
            return tuple(value)
        raise ValueError(
            f"Prior value must be a tuple, list, or array-like, got {type(value)}"
        )

    def _validate_priors(self) -> None:
        """Validate prior parameters before building ModelConfig.

        Raises
        ------
        ValueError
            If any prior has invalid values (negative, wrong length).
        """
        # Params that expect 2-tuples (Beta, LogNormal, BetaPrime)
        SCALAR_2_PARAMS = frozenset(
            {"p", "r", "mu", "gate", "p_capture", "phi", "phi_capture"}
        )
        # Beta-like: both must be > 0
        BETA_PARAMS = frozenset({"p", "gate", "p_capture"})
        # LogNormal: scale (second) must be > 0
        LOGNORMAL_PARAMS = frozenset({"r", "mu"})
        # BetaPrime: both must be > 0
        BETAPRIME_PARAMS = frozenset({"phi", "phi_capture"})

        for param, value in self._priors.items():
            try:
                vals = self._prior_value_to_tuple(value)
            except ValueError as e:
                raise ValueError(
                    f"Prior for '{param}' must be a tuple, list, or array-like"
                ) from e
            if param == "mixing":
                expected = self._n_components or 2
                if len(vals) != expected:
                    raise ValueError(
                        f"Prior for 'mixing' must have {expected} values "
                        f"(n_components), got {len(vals)}"
                    )
                if self._unconstrained:
                    if any(v < 0 for v in vals):
                        raise ValueError(
                            "Prior parameters must be non-negative "
                            "(unconstrained mixing)"
                        )
                else:
                    if any(v <= 0 for v in vals):
                        raise ValueError("Prior parameters must be positive")
            elif param in SCALAR_2_PARAMS:
                if len(vals) != 2:
                    raise ValueError(
                        f"Prior for '{param}' must have 2 values, got {len(vals)}"
                    )
                if param in BETA_PARAMS or param in BETAPRIME_PARAMS:
                    # Constrained: Beta/BetaPrime need both > 0
                    # Unconstrained: Normal(loc, scale), only scale must be > 0
                    if self._unconstrained:
                        if vals[1] <= 0:
                            raise ValueError(
                                "Prior parameters must be positive"
                            )
                    else:
                        if any(v <= 0 for v in vals):
                            raise ValueError(
                                "Prior parameters must be positive"
                            )
                elif param in LOGNORMAL_PARAMS:
                    if vals[1] <= 0:
                        raise ValueError("Prior parameters must be positive")

    def _validate_guide_families(self) -> None:
        """Validate guide family configuration.

        Raises
        ------
        ValueError
            If LowRankGuide has rank <= 0.
        """
        if self._guide_families is None:
            return
        for name in (
            "p",
            "r",
            "mu",
            "phi",
            "gate",
            "p_capture",
            "phi_capture",
            "mixing",
        ):
            gf = getattr(self._guide_families, name, None)
            if isinstance(gf, LowRankGuide) and gf.rank <= 0:
                raise ValueError("rank must be positive")

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

        # Apply defaults and validate priors
        self._apply_defaults()
        self._validate_priors()
        self._validate_guide_families()

        # base_model should always be the base type (e.g., "nbdm", not
        # "nbdm_mix")
        # Mixture is indicated by n_components, not by modifying base_model
        base_model = self._base_model

        # Create VAE config if VAE inference
        vae_config = None
        if self._inference_method == InferenceMethod.VAE:
            vae_config = VAEConfig(**self._vae_params)

        # param_specs will be created by preset factories
        # The builder doesn't create them directly - that's handled by the
        # preset factories which have access to the full model context
        # For now, pass empty list - preset factories will populate it
        param_specs: List[ParamSpec] = []

        # Build priors from merged user + defaults (convert arrays to tuples)
        priors_dict = {
            k: self._prior_value_to_tuple(v) for k, v in self._priors.items()
        }
        priors = PriorOverrides(**priors_dict)

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
            priors=priors,
            vae=vae_config,
        )
