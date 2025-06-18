"""
Unified model configuration for SCRIBE.

This module provides a single ModelConfig class that handles all parameterizations
(mean_field, mean_variance, beta_prime, unconstrained) uniformly.
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
    - mean_field: Beta/Gamma distributions for p/r parameters
    - mean_variance: Beta/LogNormal for p/mu parameters  
    - beta_prime: BetaPrime/LogNormal for phi/mu parameters
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
            - "mean_field"
            - "mean_variance"
            - "beta_prime"
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
    
    For constrained parameterizations (mean_field, mean_variance, beta_prime):
    - Distributions are in natural parameter space
    - Used directly in model/guide functions
    
    For unconstrained parameterization:
    - Distributions are Normal on transformed parameters
    - Parameters are transformed back to natural space in model
    """
    
    # Core configuration
    base_model: str
    parameterization: str = "mean_field"
    n_components: Optional[int] = None
    inference_method: str = "svi"
    
    # Success probability parameter (p) - used in mean_field and mean_variance
    p_distribution_model: Optional[dist.Distribution] = None
    p_distribution_guide: Optional[dist.Distribution] = None
    p_param_prior: Optional[tuple] = None
    p_param_guide: Optional[tuple] = None
    
    # Dispersion parameter (r) - used in mean_field
    r_distribution_model: Optional[dist.Distribution] = None
    r_distribution_guide: Optional[dist.Distribution] = None
    r_param_prior: Optional[tuple] = None
    r_param_guide: Optional[tuple] = None
    
    # Mean parameter (mu) - used in mean_variance and beta_prime
    mu_distribution_model: Optional[dist.Distribution] = None
    mu_distribution_guide: Optional[dist.Distribution] = None
    mu_param_prior: Optional[tuple] = None
    mu_param_guide: Optional[tuple] = None
    
    # Phi parameter - used in beta_prime
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
    
    # Capture phi - used in VCP models with beta_prime
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
            "nbdm", "zinb", "nbvcp", "zinbvcp",
            "nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"
        }
        
        if self.base_model not in valid_base_models:
            raise ValueError(f"Invalid base_model: {self.base_model}. "
                           f"Must be one of {valid_base_models}")
    
    def _validate_parameterization(self):
        """Validate parameterization specification."""
        valid_parameterizations = {
            "mean_field", "mean_variance", "beta_prime", "unconstrained"
        }
        
        if self.parameterization not in valid_parameterizations:
            raise ValueError(
                f"Invalid parameterization: {self.parameterization}. "
                f"Must be one of {valid_parameterizations}"
            )
    
    def _validate_mixture_components(self):
        """Validate mixture model configuration."""
        if self.is_mixture_model():
            if not self.base_model.endswith('_mix'):
                self.base_model = f"{self.base_model}_mix"
            
            if self.n_components is None or self.n_components < 2:
                raise ValueError("Mixture models require n_components >= 2")
            
            # Check for mixing parameters based on parameterization
            if self.parameterization == "unconstrained":
                if (self.mixing_logits_unconstrained_loc is None or 
                    self.mixing_logits_unconstrained_scale is None):
                    # Set defaults
                    self.mixing_logits_unconstrained_loc = 0.0
                    self.mixing_logits_unconstrained_scale = 1.0
            else:
                if (self.mixing_distribution_model is None or 
                    self.mixing_distribution_guide is None):
                    raise ValueError(
                        "Mixture models require mixing distributions"
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
                if (self.gate_unconstrained_loc is None or 
                    self.gate_unconstrained_scale is None):
                    # Set defaults
                    self.gate_unconstrained_loc = 0.0
                    self.gate_unconstrained_scale = 1.0
            else:
                if (self.gate_distribution_model is None or 
                    self.gate_distribution_guide is None):
                    raise ValueError("ZINB models require gate distributions")
        
        # Variable capture validation
        if self.uses_variable_capture():
            if self.parameterization == "unconstrained":
                if (self.p_capture_unconstrained_loc is None or 
                    self.p_capture_unconstrained_scale is None):
                    # Set defaults
                    self.p_capture_unconstrained_loc = 0.0
                    self.p_capture_unconstrained_scale = 1.0
            elif self.parameterization == "beta_prime":
                if (self.phi_capture_distribution_model is None or 
                    self.phi_capture_distribution_guide is None):
                    raise ValueError(
                        "VCP models with beta_prime require "
                        "phi_capture distributions"
                    )
            else:
                if (self.p_capture_distribution_model is None or 
                    self.p_capture_distribution_guide is None):
                    raise ValueError(
                        "VCP models require capture probability distributions"
                    )
    
    def _validate_parameterization_specific_parameters(self):
        """Validate parameters specific to each parameterization."""
        if self.parameterization == "mean_field":
            self._validate_mean_field_parameters()
        elif self.parameterization == "mean_variance":
            self._validate_mean_variance_parameters()
        elif self.parameterization == "beta_prime":
            self._validate_beta_prime_parameters()
        elif self.parameterization == "unconstrained":
            self._validate_unconstrained_parameters()
    
    def _validate_mean_field_parameters(self):
        """Validate mean_field parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "p_distribution_model", 
                "p_distribution_guide",
                "r_distribution_model", 
                "r_distribution_guide"
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"mean_field parameterization requires {param}"
                    )
    
    def _validate_mean_variance_parameters(self):
        """Validate mean_variance parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "p_distribution_model", 
                "p_distribution_guide",
                "mu_distribution_model", 
                "mu_distribution_guide"
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"mean_variance parameterization requires {param}"
                    )
    
    def _validate_beta_prime_parameters(self):
        """Validate beta_prime parameterization parameters."""
        if self.inference_method == "svi":
            required_params = [
                "phi_distribution_model", 
                "phi_distribution_guide",
                "mu_distribution_model", 
                "mu_distribution_guide"
            ]
            for param in required_params:
                if getattr(self, param) is None:
                    raise ValueError(
                        f"beta_prime parameterization requires {param}"
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
        return self.parameterization in [
            "mean_field", "mean_variance", "beta_prime"
        ]
    
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
            if self.parameterization == "mean_field":
                params.extend(["p", "r"])
            elif self.parameterization == "mean_variance":
                params.extend(["p", "mu"])
            elif self.parameterization == "beta_prime":
                params.extend(["phi", "mu"])
            
            if self.is_zero_inflated():
                params.append("gate")
            if self.uses_variable_capture():
                if self.parameterization == "beta_prime":
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
                self, f"{param_name}_distribution_model")
            info["guide_distribution"] = getattr(
                self, f"{param_name}_distribution_guide")
            info["prior_params"] = getattr(self, f"{param_name}_param_prior")
            info["guide_params"] = getattr(self, f"{param_name}_param_guide")
        
        # Add unconstrained parameter information if applicable
        if self.parameterization == "unconstrained":
            if hasattr(self, f"{param_name}_unconstrained_loc"):
                info["unconstrained_loc"] = getattr(
                    self, f"{param_name}_unconstrained_loc")
                info["unconstrained_scale"] = getattr(
                    self, f"{param_name}_unconstrained_scale")
        
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
        
        lines.append(f"  Active Parameters: {', '.join(self.get_active_parameters())}")
        
        return "\n".join(lines) 