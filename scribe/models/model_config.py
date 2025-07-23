"""
Unified model configuration for SCRIBE.

This module provides a single ModelConfig class that handles all parameterizations
(standard, linked, odds_ratio, unconstrained) uniformly.
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

    # Unconstrained parameters (used when parameterization="unconstrained")
    # These are Normal distributions on transformed parameters
    p_unconstrained_prior: Optional[tuple] = None
    p_unconstrained_guide: Optional[tuple] = None

    r_unconstrained_prior: Optional[tuple] = None
    r_unconstrained_guide: Optional[tuple] = None

    gate_unconstrained_prior: Optional[tuple] = None
    gate_unconstrained_guide: Optional[tuple] = None

    p_capture_unconstrained_prior: Optional[tuple] = None
    p_capture_unconstrained_guide: Optional[tuple] = None

    mixing_logits_unconstrained_prior: Optional[tuple] = None
    mixing_logits_unconstrained_guide: Optional[tuple] = None

    # VAE parameters (used when inference_method="vae")
    vae_latent_dim: int = 3
    vae_hidden_dims: Optional[List[int]] = None
    vae_activation: Optional[Callable] = None
    vae_output_activation: Optional[Callable] = None

    def validate(self):
        """Validate configuration parameters."""
        self._validate_base_model()
        self._validate_parameterization()
        self._validate_inference_method()
        self._validate_mixture_components()
        self._set_default_priors()

    def _set_default_priors(self):
        """Set default priors for parameters that are None."""
        if self.parameterization == "unconstrained":
            # Set defaults for unconstrained parameterization
            if self.p_unconstrained_prior is None:
                self.p_unconstrained_prior = (0.0, 1.0)  # Normal on logit(p)
            if self.r_unconstrained_prior is None:
                self.r_unconstrained_prior = (0.0, 1.0)  # Normal on log(r)
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
            "unconstrained",
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
                self.vae_hidden_dims = [256, 256]  # Default: 2 hidden layers of 256
            if self.vae_activation is None:
                # Import here to avoid circular imports
                from flax import nnx
                self.vae_activation = nnx.gelu
            if self.vae_output_activation is None:
                # Import here to avoid circular imports
                import jax
                self.vae_output_activation = nnx.softplus

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
            "unconstrained",
        ]

    def get_active_parameters(self) -> List[str]:
        """
        Get list of parameters that should be active for this configuration.
        """
        params = []

        if self.parameterization == "unconstrained":
            params.extend(["p_unconstrained", "r_unconstrained"])
            if self.is_zero_inflated():
                params.append("gate_unconstrained")
            if self.uses_variable_capture():
                params.append("p_capture_unconstrained")
            if self.is_mixture_model():
                params.append("mixing_logits_unconstrained")
        else:
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
            prior_name = f"{param}_param_prior"
            unconstrained_prior_name = f"{param}_unconstrained_prior"
            if (
                hasattr(self, prior_name)
                and getattr(self, prior_name) is not None
            ):
                active_priors[param] = getattr(self, prior_name)
            elif (
                hasattr(self, unconstrained_prior_name)
                and getattr(self, unconstrained_prior_name) is not None
            ):
                active_priors[param] = getattr(self, unconstrained_prior_name)
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
        if self.parameterization == "unconstrained":
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
            f"  Inference Method: {self.inference_method}",
        ]

        if self.is_mixture_model():
            lines.append(f"  Mixture Components: {self.n_components}")

        lines.append(
            f"  Active Parameters: {', '.join(self.get_active_parameters())}"
        )

        return "\n".join(lines)
