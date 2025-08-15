"""
Unified model configuration for SCRIBE.

This module provides a single ModelConfig class that handles all
parameterizations (standard, linked, odds_ratio, unconstrained) uniformly.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
from dataclasses import dataclass
import jax.numpy as jnp
from flax import nnx


@dataclass
class ModelConfig:
    """
    Unified configuration class for all SCRIBE model parameterizations.

    This class handles all parameterizations uniformly:
    - standard: Beta/LogNormal distributions for p/r parameters
    - linked: Beta/LogNormal for p/mu parameters
    - odds_ratio: BetaPrime/LogNormal for phi/mu parameters
    - unconstrained: Normal distributions on transformed parameters

    The parameterization determines which parameters are used and how they're
    interpreted by the model and guide functions.

    Parameters
    ----------
    base_model : str
        Name of the base model (e.g., "nbdm", "zinb", "nbvcp", "zinbvcp")
        Can include "_mix" suffix for mixture models
    parameterization : str
        Parameterization type:
            - "standard"
            - "linked"
            - "odds_ratio"
            - "unconstrained"
    n_components : Optional[int], default=None
        Number of mixture components for mixture models
    inference_method : str, default="svi"
        Inference method:
            - "svi"
            - "mcmc"
    """

    # Core configuration
    base_model: str
    parameterization: str = "standard"
    unconstrained: bool = False
    n_components: Optional[int] = None
    inference_method: str = "svi"
    component_specific_params: bool = False

    # Success probability parameter (p) - used in standard and linked
    p_param_prior: Optional[tuple] = None
    p_param_guide: Optional[tuple] = None

    # Dispersion parameter (r) - used in standard
    r_param_prior: Optional[tuple] = None
    r_param_guide: Optional[tuple] = None

    # Mean parameter (mu) - used in linked and odds_ratio
    mu_param_prior: Optional[tuple] = None
    mu_param_guide: Optional[tuple] = None

    # Phi parameter - used in odds_ratio
    phi_param_prior: Optional[tuple] = None
    phi_param_guide: Optional[tuple] = None

    # Zero-inflation gate - used in ZINB models
    gate_param_prior: Optional[tuple] = None
    gate_param_guide: Optional[tuple] = None

    # Capture probability - used in VCP models
    p_capture_param_prior: Optional[tuple] = None
    p_capture_param_guide: Optional[tuple] = None

    # Capture phi - used in VCP models with odds_ratio
    phi_capture_param_prior: Optional[tuple] = None
    phi_capture_param_guide: Optional[tuple] = None

    # Mixture weights - used in mixture models
    mixing_param_prior: Optional[tuple] = None
    mixing_param_guide: Optional[tuple] = None

    # Unconstrained parameters (used when unconstrained=True)
    # These are Normal distributions on transformed parameters

    # Standard parameterization unconstrained parameters
    p_unconstrained_prior: Optional[tuple] = None
    p_unconstrained_guide: Optional[tuple] = None

    r_unconstrained_prior: Optional[tuple] = None
    r_unconstrained_guide: Optional[tuple] = None

    # Linked parameterization unconstrained parameters
    mu_unconstrained_prior: Optional[tuple] = None
    mu_unconstrained_guide: Optional[tuple] = None

    # Odds ratio parameterization unconstrained parameters
    phi_unconstrained_prior: Optional[tuple] = None
    phi_unconstrained_guide: Optional[tuple] = None

    # Common unconstrained parameters
    gate_unconstrained_prior: Optional[tuple] = None
    gate_unconstrained_guide: Optional[tuple] = None

    p_capture_unconstrained_prior: Optional[tuple] = None
    p_capture_unconstrained_guide: Optional[tuple] = None

    phi_capture_unconstrained_prior: Optional[tuple] = None
    phi_capture_unconstrained_guide: Optional[tuple] = None

    mixing_logits_unconstrained_prior: Optional[tuple] = None
    mixing_logits_unconstrained_guide: Optional[tuple] = None

    # VAE parameters (used when inference_method="vae")
    vae_latent_dim: int = 3
    vae_hidden_dims: Optional[List[int]] = None
    vae_activation: Optional[str] = None
    vae_input_transformation: Optional[str] = None

    # VAE VCP encoder parameters (for variable capture models)
    vae_vcp_hidden_dims: Optional[List[int]] = None
    vae_vcp_activation: Optional[str] = None

    # VAE prior configuration
    vae_prior_type: str = "standard"  # "standard" or "decoupled"
    vae_prior_num_layers: int = 2  # For decoupled prior
    vae_prior_hidden_dims: Optional[List[int]] = None  # For decoupled prior
    vae_prior_activation: Optional[str] = None  # For decoupled prior
    vae_prior_mask_type: str = "alternating"  # For decoupled prior

    # VAE data preprocessing
    vae_standardize: bool = False  # Whether to standardize input data

    # VAE standardization parameters (computed from data)
    standardize_mean: Optional[jnp.ndarray] = None
    standardize_std: Optional[jnp.ndarray] = None

    def validate(self):
        """Validate configuration parameters."""
        self._validate_base_model()
        self._validate_parameterization()
        self._validate_inference_method()
        self._validate_mixture_components()
        self._set_default_priors()

    def _set_default_priors(self):
        """Set default priors for parameters that are None."""
        # Note: This method will be updated in Phase 2 to handle the
        # unconstrained flag For now, we keep the old logic for backward
        # compatibility

        if hasattr(self, "unconstrained") and self.unconstrained:
            # Set defaults for unconstrained parameterization
            if self.parameterization == "standard":
                if self.p_unconstrained_prior is None:
                    self.p_unconstrained_prior = (
                        0.0,
                        1.0,
                    )  # Normal on logit(p)
                if self.r_unconstrained_prior is None:
                    self.r_unconstrained_prior = (0.0, 1.0)  # Normal on log(r)
            elif self.parameterization == "linked":
                if self.p_unconstrained_prior is None:
                    self.p_unconstrained_prior = (
                        0.0,
                        1.0,
                    )  # Normal on logit(p)
                if self.mu_unconstrained_prior is None:
                    self.mu_unconstrained_prior = (
                        0.0,
                        1.0,
                    )  # Normal on log(mu)
            elif self.parameterization == "odds_ratio":
                if self.phi_unconstrained_prior is None:
                    self.phi_unconstrained_prior = (
                        0.0,
                        1.0,
                    )  # Normal on log(phi)
                if self.mu_unconstrained_prior is None:
                    self.mu_unconstrained_prior = (
                        0.0,
                        1.0,
                    )  # Normal on log(mu)

            # Common unconstrained parameters
            if (
                self.gate_unconstrained_prior is None
                and "zinb" in self.base_model
            ):
                self.gate_unconstrained_prior = (
                    0.0,
                    1.0,
                )  # Normal on logit(gate)
            if (
                self.p_capture_unconstrained_prior is None
                and "vcp" in self.base_model
            ):
                self.p_capture_unconstrained_prior = (
                    0.0,
                    1.0,
                )  # Normal on logit(p_capture)
            if (
                self.phi_capture_unconstrained_prior is None
                and "vcp" in self.base_model
                and self.parameterization == "odds_ratio"
            ):
                self.phi_capture_unconstrained_prior = (
                    0.0,
                    1.0,
                )  # Normal on log(phi_capture)
            if (
                self.mixing_logits_unconstrained_prior is None
                and self.is_mixture_model()
            ):
                self.mixing_logits_unconstrained_prior = (
                    0.0,
                    1.0,
                )  # Normal on logits
        else:
            # Set defaults for constrained parameterizations
            if self.parameterization == "standard":
                if self.p_param_prior is None:
                    self.p_param_prior = (1.0, 1.0)  # Beta
                if self.r_param_prior is None:
                    self.r_param_prior = (0.0, 1.0)  # LogNormal
            elif self.parameterization == "linked":
                if self.p_param_prior is None:
                    self.p_param_prior = (1.0, 1.0)  # Beta
                if self.mu_param_prior is None:
                    self.mu_param_prior = (0.0, 1.0)  # LogNormal
            elif self.parameterization == "odds_ratio":
                if self.phi_param_prior is None:
                    self.phi_param_prior = (1.0, 1.0)  # BetaPrime
                if self.mu_param_prior is None:
                    self.mu_param_prior = (0.0, 1.0)  # LogNormal

            # Model-specific defaults
            if self.gate_param_prior is None and "zinb" in self.base_model:
                self.gate_param_prior = (1.0, 1.0)  # Beta
            if self.p_capture_param_prior is None and "vcp" in self.base_model:
                if self.parameterization == "odds_ratio":
                    if self.phi_capture_param_prior is None:
                        self.phi_capture_param_prior = (1.0, 1.0)  # BetaPrime
                else:
                    self.p_capture_param_prior = (1.0, 1.0)  # Beta
            if self.mixing_param_prior is None and self.is_mixture_model():
                self.mixing_param_prior = jnp.ones(
                    self.n_components
                )  # Symmetric Dirichlet

    def _validate_base_model(self):
        """Validate base model specification."""
        valid_base_models = {
            "nbdm",
            "zinb",
            "nbvcp",
            "zinbvcp",
            "nbdm_mix",
            "zinb_mix",
            "nbvcp_mix",
            "zinbvcp_mix",
        }

        if self.base_model not in valid_base_models:
            raise ValueError(
                f"Invalid base_model: {self.base_model}. "
                f"Must be one of {valid_base_models}"
            )

    def _validate_parameterization(self):
        """Validate parameterization specification."""
        valid_parameterizations = {
            "standard",
            "linked",
            "odds_ratio",
        }

        if self.parameterization not in valid_parameterizations:
            raise ValueError(
                f"Invalid parameterization: {self.parameterization}. "
                f"Must be one of {valid_parameterizations}"
            )

    def _validate_inference_method(self):
        """Validate inference method specification."""
        valid_inference_methods = {
            "svi",
            "mcmc",
            "vae",
        }

        if self.inference_method not in valid_inference_methods:
            raise ValueError(
                f"Invalid inference_method: {self.inference_method}. "
                f"Must be one of {valid_inference_methods}"
            )

        # Validate VAE-specific parameters
        if self.inference_method == "vae":
            # Set default VAE parameters if not provided
            if self.vae_hidden_dims is None:
                self.vae_hidden_dims = [128, 128, 128]
            if self.vae_activation is None:
                self.vae_activation = "relu"

            # Set default VCP encoder parameters if using variable capture
            if self.uses_variable_capture():
                if self.vae_vcp_hidden_dims is None:
                    self.vae_vcp_hidden_dims = [
                        64,
                        32,
                    ]  # Smaller than main encoder since input is simpler
                if self.vae_vcp_activation is None:
                    self.vae_vcp_activation = "relu"  # Same as main encoder

            # Validate VAE prior configuration
            valid_prior_types = {"standard", "decoupled"}
            if self.vae_prior_type not in valid_prior_types:
                raise ValueError(
                    f"Invalid vae_prior_type: {self.vae_prior_type}. "
                    f"Must be one of {valid_prior_types}"
                )

            # Set default decoupled prior parameters if using decoupled prior
            if self.vae_prior_type == "decoupled":
                if self.vae_prior_hidden_dims is None:
                    self.vae_prior_hidden_dims = [
                        64,
                        64,
                    ]  # Default: 2 hidden layers of 64
                if self.vae_prior_activation is None:
                    self.vae_prior_activation = "relu"

    def _validate_mixture_components(self):
        """Validate mixture model configuration."""
        if self.is_mixture_model():
            if not self.base_model.endswith("_mix"):
                self.base_model = f"{self.base_model}_mix"

            if self.n_components is None or self.n_components < 2:
                raise ValueError("Mixture models require n_components >= 2")

        else:
            if self.n_components is not None:
                raise ValueError(
                    "Non-mixture models should not specify n_components"
                )

    # Utility methods
    def is_mixture_model(self) -> bool:
        """Check if this is a mixture model configuration."""
        return self.n_components is not None and self.n_components > 1

    def is_zero_inflated(self) -> bool:
        """Check if this is a zero-inflated model configuration."""
        return "zinb" in self.base_model

    def uses_variable_capture(self) -> bool:
        """Check if this model uses variable capture probability."""
        return "vcp" in self.base_model

    def check_parameterization(self) -> bool:
        """Check if this uses a constrained parameterization."""
        return self.parameterization in [
            "standard",
            "linked",
            "odds_ratio",
        ]

    def get_active_parameters(self) -> List[str]:
        """
        Get list of parameters that should be active for this configuration.
        """
        params = []

        if hasattr(self, "unconstrained") and self.unconstrained:
            # Unconstrained parameters based on parameterization
            if self.parameterization == "standard":
                params.extend(["p_unconstrained", "r_unconstrained"])
            elif self.parameterization == "linked":
                params.extend(["p_unconstrained", "mu_unconstrained"])
            elif self.parameterization == "odds_ratio":
                params.extend(["phi_unconstrained", "mu_unconstrained"])

            # Common unconstrained parameters
            if self.is_zero_inflated():
                params.append("gate_unconstrained")
            if self.uses_variable_capture():
                if self.parameterization == "odds_ratio":
                    params.append("phi_capture_unconstrained")
                else:
                    params.append("p_capture_unconstrained")
            if self.is_mixture_model():
                params.append("mixing_logits_unconstrained")
        else:
            # Constrained parameters based on parameterization
            if self.parameterization == "standard":
                params.extend(["p", "r"])
            elif self.parameterization == "linked":
                params.extend(["p", "mu"])
            elif self.parameterization == "odds_ratio":
                params.extend(["phi", "mu"])

            if self.is_zero_inflated():
                params.append("gate")
            if self.uses_variable_capture():
                if self.parameterization == "odds_ratio":
                    params.append("phi_capture")
                else:
                    params.append("p_capture")
            if self.is_mixture_model():
                params.append("mixing")

        return params

    def get_active_priors(self) -> Dict[str, Any]:
        """Get dictionary of active prior parameters."""
        active_priors = {}
        for param in self.get_active_parameters():
            if hasattr(self, "unconstrained") and self.unconstrained:
                # For unconstrained models, use unconstrained prior names
                if param.endswith("_unconstrained"):
                    prior_name = f"{param}_prior"
                else:
                    # Handle cases where param doesn't have _unconstrained suffix
                    prior_name = f"{param}_unconstrained_prior"
            else:
                # For constrained models, use regular prior names
                prior_name = f"{param}_param_prior"

            if (
                hasattr(self, prior_name)
                and getattr(self, prior_name) is not None
            ):
                active_priors[param] = getattr(self, prior_name)

        return active_priors

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get information about a specific parameter."""
        info = {
            "name": param_name,
            "active": param_name in self.get_active_parameters(),
            "parameterization": self.parameterization,
        }

        # Add distribution parameter information if available
        if hasattr(self, f"{param_name}_param_prior"):
            info["prior_params"] = getattr(self, f"{param_name}_param_prior")
            info["guide_params"] = getattr(self, f"{param_name}_param_guide")

        # Add unconstrained parameter information if applicable
        if getattr(self, "unconstrained", False):
            if hasattr(self, f"{param_name}_unconstrained_loc"):
                info["unconstrained_loc"] = getattr(
                    self, f"{param_name}_unconstrained_loc"
                )
                info["unconstrained_scale"] = getattr(
                    self, f"{param_name}_unconstrained_scale"
                )

        return info

    def summary(self) -> str:
        """Generate a summary string of the configuration."""
        lines = [
            f"ModelConfig Summary:",
            f"  Base Model: {self.base_model}",
            f"  Parameterization: {self.parameterization}",
            f"  Unconstrained: {getattr(self, 'unconstrained', False)}",
            f"  Inference Method: {self.inference_method}",
        ]

        if self.is_mixture_model():
            lines.append(f"  Mixture Components: {self.n_components}")

        lines.append(
            f"  Active Parameters: {', '.join(self.get_active_parameters())}"
        )

        # Add VAE-specific information
        if self.inference_method == "vae":
            lines.append(f"  VAE Latent Dim: {self.vae_latent_dim}")
            lines.append(f"  VAE Hidden Dims: {self.vae_hidden_dims}")
            lines.append(f"  VAE Prior Type: {self.vae_prior_type}")
            if self.uses_variable_capture():
                lines.append(
                    f"  VAE VCP Hidden Dims: {self.vae_vcp_hidden_dims}"
                )
                lines.append(f"  VAE VCP Activation: {self.vae_vcp_activation}")
            if self.is_decoupled_prior():
                lines.append(
                    f"  VAE Prior Hidden Dims: {self.vae_prior_hidden_dims}"
                )
                lines.append(
                    f"  VAE Prior Num Layers: {self.vae_prior_num_layers}"
                )
                lines.append(
                    f"  VAE Prior Activation: {self.vae_prior_activation}"
                )

        return "\n".join(lines)

    def get_vae_prior_config(self) -> Dict[str, Any]:
        """
        Get VAE prior configuration information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing VAE prior configuration
        """
        config = {
            "prior_type": self.vae_prior_type,
        }

        if self.vae_prior_type == "decoupled":
            config.update(
                {
                    "prior_hidden_dims": self.vae_prior_hidden_dims,
                    "prior_num_layers": self.vae_prior_num_layers,
                    "prior_activation": self.vae_prior_activation,
                    "prior_mask_type": self.vae_prior_mask_type,
                }
            )

        return config

    def is_decoupled_prior(self) -> bool:
        """
        Check if this configuration uses a decoupled prior.

        Returns
        -------
        bool
            True if using decoupled prior, False otherwise
        """
        return self.vae_prior_type == "decoupled"
