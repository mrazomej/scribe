"""
Unified model configuration for SCRIBE.

This module provides a single ModelConfig class that handles all parameterizations
(standard, linked, odds_ratio, unconstrained) uniformly.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import jax.numpy as jnp
import numpyro.distributions as dist


@dataclass
class ModelConfig:
    """
    Unified configuration class for all SCRIBE model parameterizations.

    This class handles all parameterizations uniformly:
    - standard: Beta/Gamma distributions for p/r parameters
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

    Parameter Distributions:
    -----------------------
    Each parameter can have model and guide distributions specified.
    The interpretation depends on the parameterization:

    For constrained parameterizations (standard, linked, odds_ratio):
    - Distributions are in natural parameter space
    - Used directly in model/guide functions

    For unconstrained parameterization:
    - Distributions are Normal on transformed parameters
    - Parameters are transformed back to natural space in model
    """

    # Core configuration
    base_model: str
    parameterization: str = "standard"
    n_components: Optional[int] = None
    inference_method: str = "svi"

    # Success probability parameter (p) - used in standard and linked
    p_distribution_model: Optional[dist.Distribution] = None
    p_distribution_guide: Optional[dist.Distribution] = None
    p_param_prior: Optional[tuple] = None
    p_param_guide: Optional[tuple] = None

    # Dispersion parameter (r) - used in standard
    r_distribution_model: Optional[dist.Distribution] = None
    r_distribution_guide: Optional[dist.Distribution] = None
    r_param_prior: Optional[tuple] = None
    r_param_guide: Optional[tuple] = None

    # Mean parameter (mu) - used in linked and odds_ratio
    mu_distribution_model: Optional[dist.Distribution] = None
    mu_distribution_guide: Optional[dist.Distribution] = None
    mu_param_prior: Optional[tuple] = None
    mu_param_guide: Optional[tuple] = None

    # Phi parameter - used in odds_ratio
    phi_distribution_model: Optional[dist.Distribution] = None
    phi_distribution_guide: Optional[dist.Distribution] = None
    phi_param_prior: Optional[tuple] = None
    phi_param_guide: Optional[tuple] = None

    # Zero-inflation gate - used in ZINB models
    gate_distribution_model: Optional[dist.Distribution] = None
    gate_distribution_guide: Optional[dist.Distribution] = None
    gate_param_prior: Optional[tuple] = None
    gate_param_guide: Optional[tuple] = None

    # Capture probability - used in VCP models
    p_capture_distribution_model: Optional[dist.Distribution] = None
    p_capture_distribution_guide: Optional[dist.Distribution] = None
    p_capture_param_prior: Optional[tuple] = None
    p_capture_param_guide: Optional[tuple] = None

    # Capture phi - used in VCP models with odds_ratio
    phi_capture_distribution_model: Optional[dist.Distribution] = None
    phi_capture_distribution_guide: Optional[dist.Distribution] = None
    phi_capture_param_prior: Optional[tuple] = None
    phi_capture_param_guide: Optional[tuple] = None

    # Mixture weights - used in mixture models
    mixing_distribution_model: Optional[dist.Distribution] = None
    mixing_distribution_guide: Optional[dist.Distribution] = None
    mixing_param_prior: Optional[tuple] = None
    mixing_param_guide: Optional[tuple] = None

    # Unconstrained parameters (used when parameterization="unconstrained")
    # These are Normal distributions on transformed parameters
    p_unconstrained_loc: Optional[float] = None
    p_unconstrained_scale: Optional[float] = None
    r_unconstrained_loc: Optional[float] = None
    r_unconstrained_scale: Optional[float] = None
    gate_unconstrained_loc: Optional[float] = None
    gate_unconstrained_scale: Optional[float] = None
    p_capture_unconstrained_loc: Optional[float] = None
    p_capture_unconstrained_scale: Optional[float] = None
    mixing_logits_unconstrained_loc: Optional[float] = None
    mixing_logits_unconstrained_scale: Optional[float] = None

    def validate(self):
        """Validate configuration parameters."""
        self._validate_base_model()
        self._validate_parameterization()
        self._validate_mixture_components()
        self._validate_model_specific_parameters()
        self._validate_parameterization_specific_parameters()

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

    def _validate_mixture_components(self):
        """Validate mixture model configuration."""
        if self.is_mixture_model():
            if not self.base_model.endswith("_mix"):
                self.base_model = f"{self.base_model}_mix"

            if self.n_components is None or self.n_components < 2:
                raise ValueError("Mixture models require n_components >= 2")

            # Check for mixing parameters based on parameterization
            if self.parameterization == "unconstrained":
                if (
                    self.mixing_logits_unconstrained_loc is None
                    or self.mixing_logits_unconstrained_scale is None
                ):
                    # Set defaults
                    self.mixing_logits_unconstrained_loc = 0.0
                    self.mixing_logits_unconstrained_scale = 1.0
            else:
                # For constrained parameterizations, check based on inference method
                if self.inference_method == "svi":
                    if (
                        self.mixing_distribution_model is None
                        or self.mixing_distribution_guide is None
                    ):
                        raise ValueError(
                            "Mixture models with SVI require mixing distributions"
                        )
                else:  # MCMC
                    if self.mixing_distribution_model is None:
                        raise ValueError(
                            "Mixture models with MCMC require mixing distribution model"
                        )
        else:
            if self.n_components is not None:
                raise ValueError(
                    "Non-mixture models should not specify n_components"
                )

    def _validate_model_specific_parameters(self):
        """Validate parameters specific to model variants."""
        # Zero-inflation validation
        if self.is_zero_inflated():
            if self.parameterization == "unconstrained":
                if (
                    self.gate_unconstrained_loc is None
                    or self.gate_unconstrained_scale is None
                ):
                    # Set defaults
                    self.gate_unconstrained_loc = 0.0
                    self.gate_unconstrained_scale = 1.0
            else:
                # For constrained parameterizations, check based on inference method
                if self.inference_method == "svi":
                    if (
                        self.gate_distribution_model is None
                        or self.gate_distribution_guide is None
                    ):
                        raise ValueError(
                            "ZINB models with SVI require gate distributions"
                        )
                else:  # MCMC
                    if self.gate_distribution_model is None:
                        raise ValueError(
                            "ZINB models with MCMC require gate distribution model"
                        )

        # Variable capture validation
        if self.uses_variable_capture():
            if self.parameterization == "unconstrained":
                if (
                    self.p_capture_unconstrained_loc is None
                    or self.p_capture_unconstrained_scale is None
                ):
                    # Set defaults
                    self.p_capture_unconstrained_loc = 0.0
                    self.p_capture_unconstrained_scale = 1.0
            elif self.parameterization == "odds_ratio":
                # For odds_ratio parameterization, check based on inference method
                if self.inference_method == "svi":
                    if (
                        self.phi_capture_distribution_model is None
                        or self.phi_capture_distribution_guide is None
                    ):
                        raise ValueError(
                            "VCP models with odds_ratio and SVI require "
                            "phi_capture distributions"
                        )
                else:  # MCMC
                    if self.phi_capture_distribution_model is None:
                        raise ValueError(
                            "VCP models with odds_ratio and MCMC require "
                            "phi_capture distribution model"
                        )
            else:
                # For other constrained parameterizations, check based on inference method
                if self.inference_method == "svi":
                    if (
                        self.p_capture_distribution_model is None
                        or self.p_capture_distribution_guide is None
                    ):
                        raise ValueError(
                            "VCP models with SVI require capture probability distributions"
                        )
                else:  # MCMC
                    if self.p_capture_distribution_model is None:
                        raise ValueError(
                            "VCP models with MCMC require capture probability distribution model"
                        )

    def _validate_parameterization_specific_parameters(self):
        """Validate parameters specific to each parameterization."""
        if self.parameterization == "standard":
            self._validate_standard_parameters()
        elif self.parameterization == "linked":
            self._validate_linked_parameters()
        elif self.parameterization == "odds_ratio":
            self._validate_odds_ratio_parameters()
        elif self.parameterization == "unconstrained":
            self._validate_unconstrained_parameters()

    def _validate_standard_parameters(self):
        """Validate standard parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "p_distribution_model",
                "p_distribution_guide",
                "r_distribution_model",
                "r_distribution_guide",
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"standard parameterization requires {param}"
                    )

    def _validate_linked_parameters(self):
        """Validate linked parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "p_distribution_model",
                "p_distribution_guide",
                "mu_distribution_model",
                "mu_distribution_guide",
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"linked parameterization requires {param}"
                    )

    def _validate_odds_ratio_parameters(self):
        """Validate odds_ratio parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "phi_distribution_model",
                "phi_distribution_guide",
                "mu_distribution_model",
                "mu_distribution_guide",
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"odds_ratio parameterization requires {param}"
                    )

    def _validate_unconstrained_parameters(self):
        """Validate unconstrained parameterization parameters."""
        # Set defaults for unconstrained parameters if not specified
        if self.p_unconstrained_loc is None:
            self.p_unconstrained_loc = 0.0
        if self.p_unconstrained_scale is None:
            self.p_unconstrained_scale = 1.0
        if self.r_unconstrained_loc is None:
            self.r_unconstrained_loc = 0.0
        if self.r_unconstrained_scale is None:
            self.r_unconstrained_scale = 1.0

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

    def is_constrained_parameterization(self) -> bool:
        """Check if this uses a constrained parameterization."""
        return self.parameterization in ["standard", "linked", "odds_ratio"]

    def is_unconstrained_parameterization(self) -> bool:
        """Check if this uses unconstrained parameterization."""
        return self.parameterization == "unconstrained"

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

    def get_parameter_info(self, param_name: str) -> Dict[str, Any]:
        """Get information about a specific parameter."""
        info = {
            "name": param_name,
            "active": param_name in self.get_active_parameters(),
            "parameterization": self.parameterization,
        }

        # Add distribution information if available
        if hasattr(self, f"{param_name}_distribution_model"):
            info["model_distribution"] = getattr(
                self, f"{param_name}_distribution_model"
            )
            info["guide_distribution"] = getattr(
                self, f"{param_name}_distribution_guide"
            )
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
