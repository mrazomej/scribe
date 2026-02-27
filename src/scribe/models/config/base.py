"""Base model configuration classes using Pydantic."""

from typing import Optional, Set, Dict, Any, List
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
    ConfigDict,
)
from .enums import ModelType, Parameterization, InferenceMethod
from .groups import (
    VAEConfig,
    GuideFamilyConfig,
    PriorOverrides,
)
from .parameter_mapping import get_active_parameters
from ..builders.parameter_specs import ParamSpec

# ==============================================================================
# Unified Model Configuration Class
# ==============================================================================


class ModelConfig(BaseModel):
    """
    Unified model configuration for SCRIBE models.

    This class defines the complete set of configuration options for SCRIBE
    models, supporting both constrained (interpretable) and unconstrained
    (unconstrained) parameterizations. The class enforces correctness,
    immutability, and clarity of model setup.

    This is not constructed directly; users should assemble configurations via
    the ModelConfigBuilder, which ensures validity and best practices.

    The actual model and guide functions are created by the unified factory
    `create_model(model_config)` in `scribe.models.presets.factory`.

    Parameters
    ----------
    base_model : str
        The core model family (e.g., 'nbdm', 'zinb', 'nbvcp', 'zinbvcp').
    parameterization : Parameterization
        How parameters are represented internally (standard, linked, etc.).
    inference_method : InferenceMethod
        Inference engine type (SVI, MCMC, VAE, etc.).
    unconstrained : bool, default=False
        If True, use unconstrained parameterization (Normal distributions).
        If False, use constrained parameterization (Beta, LogNormal, etc.).
    n_components : int, optional
        Number of mixture components, if mixture modeling is enabled.
    mixture_params : List[str], optional
        List of parameter names that should be mixture-specific. If None and
        n_components is set, all sampled core parameters for the selected
        parameterization will be mixture-specific by default.
    guide_families : GuideFamilyConfig, optional
        Per-parameter guide family configuration. Allows specifying different
        variational families (MeanField, LowRank, Amortized) for each parameter.
    param_specs : List[ParamSpec], optional
        Optional list of parameter specifications for user-provided overrides.
        When provided, these can contain custom prior/guide hyperparameters
        that override the defaults. The unified factory uses these to customize
        the model. If empty (default), the factory uses default hyperparameters.
    vae : VAEConfig, optional
        Nested configuration for Variational Autoencoders, if applicable.

    Notes
    -----
    - Configuration objects are immutable and validated automatically on
      creation.
    - Unrecognized parameters are forbidden and will raise validation errors.
    - Intended for safe, reproducible, and fully-specified SCRIBE model setups.
    - The `param_specs` field is primarily for user overrides. The unified
      factory (`create_model`) constructs the complete parameter specifications
      based on the model type and parameterization, then applies any overrides
      from `param_specs`.

    See Also
    --------
    scribe.models.presets.factory.create_model : Creates model/guide from config.
    ModelConfigBuilder : Builder for creating ModelConfig objects.
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
    unconstrained: bool = Field(
        False, description="Use unconstrained parameterization"
    )

    # Hierarchical flags
    hierarchical_p: bool = Field(
        False,
        description=(
            "Gene-specific p/phi with hierarchical prior. "
            "Requires unconstrained=True."
        ),
    )
    hierarchical_gate: bool = Field(
        False,
        description=(
            "Gene-specific gate with hierarchical prior. "
            "Only valid for zero-inflated models. Requires unconstrained=True."
        ),
    )

    # Mixture configuration
    n_components: Optional[int] = Field(
        None, gt=1, description="Number of mixture components"
    )
    mixture_params: Optional[List[str]] = Field(
        None,
        description="List of parameter names that should be mixture-specific",
    )

    # Guide configuration
    guide_families: Optional[GuideFamilyConfig] = Field(
        None,
        description="Per-parameter guide family configuration",
    )

    # Parameter specifications (optional user overrides)
    param_specs: List[ParamSpec] = Field(
        default_factory=list,
        description=(
            "Optional list of parameter specifications for user-provided "
            "prior/guide hyperparameter overrides. The unified factory uses "
            "these to customize default parameters."
        ),
    )

    # Prior overrides (populated by ModelConfigBuilder with defaults)
    priors: PriorOverrides = Field(
        default_factory=PriorOverrides,
        description="Prior hyperparameters (e.g., Beta, LogNormal) per parameter.",
    )

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

    @model_validator(mode="after")
    def validate_hierarchical_flags(self) -> "ModelConfig":
        """Validate hierarchical flag consistency.

        - hierarchical_gate requires a zero-inflated model.
        - Both hierarchical flags require unconstrained=True.
        """
        if self.hierarchical_gate and not self.is_zero_inflated:
            raise ValueError(
                "hierarchical_gate=True requires a zero-inflated model "
                "(zinb or zinbvcp), but base_model="
                f"{self.base_model!r}."
            )
        if self.hierarchical_p and not self.unconstrained:
            raise ValueError(
                "hierarchical_p=True requires unconstrained=True."
            )
        if self.hierarchical_gate and not self.unconstrained:
            raise ValueError(
                "hierarchical_gate=True requires unconstrained=True."
            )
        return self

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_param_specs_consistency(self) -> "ModelConfig":
        """Validate that param_specs overrides are consistent with config.

        This validates:
        1. Any provided param_specs have consistent unconstrained flag
        2. Any provided param_specs are valid for this model type

        Note: param_specs is optional and may be empty. The unified factory
        constructs complete specs based on model type/parameterization.
        """
        if not self.param_specs:
            # No overrides provided, nothing to validate
            return self

        # Check unconstrained consistency for any provided specs
        for spec in self.param_specs:
            if spec.unconstrained != self.unconstrained:
                raise ValueError(
                    f"Parameter '{spec.name}': "
                    f"unconstrained flag ({spec.unconstrained}) "
                    f"must match ModelConfig.unconstrained "
                    f"({self.unconstrained})"
                )

        # Validate provided params are valid for this model
        is_mixture = self.n_components is not None
        is_zero_inflated = self.is_zero_inflated
        uses_variable_capture = self.uses_variable_capture

        provided_params = {spec.name for spec in self.param_specs}
        active_params = get_active_parameters(
            self.parameterization,
            self.base_model,
            is_mixture,
            is_zero_inflated,
            uses_variable_capture,
            hierarchical_p=self.hierarchical_p,
            hierarchical_gate=self.hierarchical_gate,
        )

        unexpected_params = provided_params - active_params
        if unexpected_params:
            raise ValueError(
                f"Unexpected parameters for {self.base_model} with "
                f"{self.parameterization.value} parameterization: "
                f"{', '.join(sorted(unexpected_params))}"
            )

        return self

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
    def is_hierarchical(self) -> bool:
        """Check if this model uses any hierarchical prior (p/phi or gate)."""
        return self.hierarchical_p or self.hierarchical_gate

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
            hierarchical_p=self.hierarchical_p,
            hierarchical_gate=self.hierarchical_gate,
        )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def get_prior_overrides(self) -> Dict[str, Any]:
        """Extract prior overrides from priors field or param_specs.

        Returns a dictionary mapping parameter names to their prior
        hyperparameters. Prefers the priors field when it has content;
        otherwise falls back to param_specs.

        Returns
        -------
        Dict[str, Any]
            Dictionary of prior overrides, e.g., {"p": (2.0, 2.0)}.
        """
        # Prefer priors from the priors field when it has content
        extra = getattr(self.priors, "__pydantic_extra__", None)
        if extra:
            return dict(extra)
        # Fall back to param_specs
        priors = {}
        for spec in self.param_specs:
            if hasattr(spec, "prior") and spec.prior is not None:
                priors[spec.name] = spec.prior
        return priors

    # --------------------------------------------------------------------------

    def get_guide_overrides(self) -> Dict[str, Any]:
        """Extract guide overrides from param_specs.

        Returns a dictionary mapping parameter names to their guide
        hyperparameters. Only includes parameters that have explicit
        guide values set.

        Returns
        -------
        Dict[str, Any]
            Dictionary of guide overrides.
        """
        guides = {}
        for spec in self.param_specs:
            if hasattr(spec, "guide") and spec.guide is not None:
                guides[spec.name] = spec.guide
        return guides

    # --------------------------------------------------------------------------

    def get_active_priors(self) -> Dict[str, Any]:
        """Get active prior parameters as a dictionary.

        .. deprecated::
            Use `get_prior_overrides()` instead.
        """
        priors_dict = {}
        for spec in self.param_specs:
            if spec.prior is not None:
                priors_dict[f"{spec.name}_prior"] = spec.prior
        return priors_dict

    # --------------------------------------------------------------------------

    def with_updated_priors(self, **priors) -> "ModelConfig":
        """Create a new config with updated priors (immutable pattern).

        Parameters
        ----------
        **priors
            Prior parameters keyed by parameter name.
            Example: with_updated_priors(p=(2.0, 2.0), r=(1.0, 0.5))

        Returns
        -------
        ModelConfig
            New config with updated priors.
        """
        # Use priors field (populated by builder)
        current = self.get_prior_overrides()
        updated = {**current, **priors}
        return self.model_copy(
            update={"priors": PriorOverrides(**updated)}
        )

    # --------------------------------------------------------------------------

    def with_updated_vae(self, **vae_params) -> "ModelConfig":
        """Create a new config with updated VAE parameters.

        Parameters
        ----------
        **vae_params
            VAE parameters to update.

        Returns
        -------
        ModelConfig
            New config with updated VAE settings.

        Raises
        ------
        ValueError
            If VAE config is None.
        """
        if self.vae is None:
            raise ValueError("Cannot update VAE config when VAE is None")
        return self.model_copy(
            update={"vae": self.vae.model_copy(update=vae_params)}
        )

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
        >>> config = ModelConfigBuilder().for_model("nbdm").build()
        >>> yaml_str = config.to_yaml()
        >>> print(yaml_str)
        """
        import yaml

        # Use model_dump with mode="json" to get serializable dict
        # Exclude computed properties that shouldn't be serialized
        data = self.model_dump(
            mode="json",
            exclude={
                "is_mixture",
                "is_zero_inflated",
                "uses_variable_capture",
                "is_hierarchical",
                "active_parameters",
            },
        )
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ModelConfig":
        """Deserialize config from YAML string.

        Parameters
        ----------
        yaml_str : str
            YAML string representation of the config.

        Returns
        -------
        ModelConfig
            Deserialized config object.

        Examples
        --------
        >>> yaml_str = '''
        ... base_model: nbdm
        ... parameterization: standard
        ... '''
        >>> config = ModelConfig.from_yaml(yaml_str)
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)

    # --------------------------------------------------------------------------

    def to_yaml_file(self, path: str) -> None:
        """Save config to YAML file.

        Parameters
        ----------
        path : str
            Path to save the YAML file.

        Examples
        --------
        >>> config.to_yaml_file("model_config.yaml")
        """
        with open(path, "w") as f:
            f.write(self.to_yaml())

    # --------------------------------------------------------------------------

    @classmethod
    def from_yaml_file(cls, path: str) -> "ModelConfig":
        """Load config from YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.

        Returns
        -------
        ModelConfig
            Loaded config object.

        Examples
        --------
        >>> config = ModelConfig.from_yaml_file("model_config.yaml")
        """
        with open(path) as f:
            return cls.from_yaml(f.read())
