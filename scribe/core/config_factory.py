"""
Unified model configuration factory for SCRIBE inference.

This module handles the creation of unified ModelConfig objects for all
parameterizations and inference methods.
"""

from typing import Dict, Optional, Any
import jax.numpy as jnp
import numpyro.distributions as dist
from ..model_config import ModelConfig
from ..stats import BetaPrime


class ModelConfigFactory:
    """Factory for creating unified model configuration objects."""
    
    @staticmethod
    def create_config(
        model_type: str,
        parameterization: str,
        inference_method: str,
        priors: Dict[str, Any],
        n_components: Optional[int] = None
    ) -> ModelConfig:
        """
        Create a unified ModelConfig for any parameterization and inference
        method.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        parameterization : str
            Parameterization type ("mean_field", "mean_variance", "beta_prime",
            "unconstrained")
        inference_method : str
            Inference method ("svi" or "mcmc")
        priors : Dict[str, Any]
            Dictionary of prior parameters
        n_components : Optional[int]
            Number of mixture components
            
        Returns
        -------
        ModelConfig
            Unified configuration object
        """
        # Create base config
        config = ModelConfig(
            base_model=model_type,
            parameterization=parameterization,
            inference_method=inference_method,
            n_components=n_components
        )
        
        # Fill in parameters based on parameterization
        if parameterization == "unconstrained":
            ModelConfigFactory._configure_unconstrained(config, priors)
        else:
            ModelConfigFactory._configure_constrained(
                config, parameterization, priors, inference_method)
        
        # Validate and return
        config.validate()
        return config
    
    @staticmethod
    def _configure_unconstrained(config: ModelConfig, priors: Dict[str, Any]):
        """Configure unconstrained parameterization."""
        # Core parameters
        config.p_unconstrained_loc = priors.get("p_prior", (0, 1))[0]
        config.p_unconstrained_scale = priors.get("p_prior", (0, 1))[1]
        config.r_unconstrained_loc = priors.get("r_prior", (0, 1))[0]
        config.r_unconstrained_scale = priors.get("r_prior", (0, 1))[1]
        
        # Optional parameters for zero-inflated models
        if config.is_zero_inflated():
            gate_prior = priors.get("gate_prior", (0, 1))
            config.gate_unconstrained_loc = gate_prior[0]
            config.gate_unconstrained_scale = gate_prior[1]
        
        # Optional parameters for variable capture models
        if config.uses_variable_capture():
            p_capture_prior = priors.get("p_capture_prior", (0, 1))
            config.p_capture_unconstrained_loc = p_capture_prior[0]
            config.p_capture_unconstrained_scale = p_capture_prior[1]
        
        # Optional parameters for mixture models
        if config.is_mixture_model():
            mixing_prior = priors.get("mixing_prior", (0, 1))
            if isinstance(mixing_prior, tuple) and len(mixing_prior) == 2:
                config.mixing_logits_unconstrained_loc = mixing_prior[0]
                config.mixing_logits_unconstrained_scale = mixing_prior[1]
    
    @staticmethod
    def _configure_constrained(
        config: ModelConfig, 
        parameterization: str, 
        priors: Dict[str, Any], 
        inference_method: str
    ):
        """Configure constrained parameterizations."""
        if parameterization == "mean_field":
            ModelConfigFactory._configure_mean_field(
                config, priors, inference_method)
        elif parameterization == "mean_variance":
            ModelConfigFactory._configure_mean_variance(
                config, priors, inference_method)
        elif parameterization == "beta_prime":
            ModelConfigFactory._configure_beta_prime(
                config, priors, inference_method)
    
    @staticmethod
    def _configure_mean_field(
        config: ModelConfig, priors: Dict[str, Any], inference_method: str):
        """Configure mean_field parameterization."""
        # Success probability
        p_prior = priors.get("p_prior", (1, 1))
        config.p_distribution_model = dist.Beta(p_prior[0], p_prior[1])
        config.p_param_prior = p_prior
        if inference_method == "svi":
            config.p_distribution_guide = dist.Beta(p_prior[0], p_prior[1])
            config.p_param_guide = p_prior
        
        # Dispersion parameter
        r_prior = priors.get("r_prior", (1, 1))
        if len(r_prior) == 2:
            # Could be Gamma or LogNormal - default to LogNormal for mean_field
            config.r_distribution_model = dist.LogNormal(r_prior[0], r_prior[1])
            config.r_param_prior = r_prior
            if inference_method == "svi":
                config.r_distribution_guide = dist.LogNormal(
                    r_prior[0], r_prior[1])
                config.r_param_guide = r_prior
        
        ModelConfigFactory._configure_optional_parameters(
            config, priors, inference_method)
    
    @staticmethod
    def _configure_mean_variance(
        config: ModelConfig, priors: Dict[str, Any], inference_method: str):
        """Configure mean_variance parameterization."""
        # Success probability
        p_prior = priors.get("p_prior", (1, 1))
        config.p_distribution_model = dist.Beta(p_prior[0], p_prior[1])
        config.p_param_prior = p_prior
        if inference_method == "svi":
            config.p_distribution_guide = dist.Beta(p_prior[0], p_prior[1])
            config.p_param_guide = p_prior
        
        # Mean parameter
        mu_prior = priors.get("mu_prior", (1, 1))
        config.mu_distribution_model = dist.LogNormal(mu_prior[0], mu_prior[1])
        config.mu_param_prior = mu_prior
        if inference_method == "svi":
            config.mu_distribution_guide = dist.LogNormal(
                mu_prior[0], mu_prior[1])
            config.mu_param_guide = mu_prior
        
        ModelConfigFactory._configure_optional_parameters(
            config, priors, inference_method)
    
    @staticmethod
    def _configure_beta_prime(
        config: ModelConfig, priors: Dict[str, Any], inference_method: str):
        """Configure beta_prime parameterization."""
        # Phi parameter
        phi_prior = priors.get("phi_prior", (1, 1))
        config.phi_distribution_model = BetaPrime(phi_prior[0], phi_prior[1])
        config.phi_param_prior = phi_prior
        if inference_method == "svi":
            config.phi_distribution_guide = BetaPrime(
                phi_prior[0], phi_prior[1])
            config.phi_param_guide = phi_prior
        
        # Mean parameter
        mu_prior = priors.get("mu_prior", (1, 1))
        config.mu_distribution_model = dist.LogNormal(mu_prior[0], mu_prior[1])
        config.mu_param_prior = mu_prior
        if inference_method == "svi":
            config.mu_distribution_guide = dist.LogNormal(
                mu_prior[0], mu_prior[1])
            config.mu_param_guide = mu_prior
        
        # Special handling for capture probability in beta_prime
        if config.uses_variable_capture():
            phi_capture_prior = priors.get("phi_capture_prior", (1, 1))
            config.phi_capture_distribution_model = BetaPrime(
                phi_capture_prior[0], phi_capture_prior[1])
            config.phi_capture_param_prior = phi_capture_prior
            if inference_method == "svi":
                config.phi_capture_distribution_guide = BetaPrime(
                    phi_capture_prior[0], phi_capture_prior[1])
                config.phi_capture_param_guide = phi_capture_prior
        
        # Configure other optional parameters (except p_capture which is handled above)
        if config.is_zero_inflated():
            gate_prior = priors.get("gate_prior", (1, 1))
            config.gate_distribution_model = dist.Beta(
                gate_prior[0], gate_prior[1])
            config.gate_param_prior = gate_prior
            if inference_method == "svi":
                config.gate_distribution_guide = dist.Beta(
                    gate_prior[0], gate_prior[1])
                config.gate_param_guide = gate_prior
        
        if config.is_mixture_model():
            ModelConfigFactory._configure_mixing_parameters(
                config, priors, inference_method)
    
    @staticmethod
    def _configure_optional_parameters(
        config: ModelConfig, priors: Dict[str, Any], inference_method: str):
        """Configure optional parameters (gate, p_capture, mixing)."""
        # Gate parameter for ZINB models
        if config.is_zero_inflated():
            gate_prior = priors.get("gate_prior", (1, 1))
            config.gate_distribution_model = dist.Beta(
                gate_prior[0], gate_prior[1])
            config.gate_param_prior = gate_prior
            if inference_method == "svi":
                config.gate_distribution_guide = dist.Beta(
                    gate_prior[0], gate_prior[1])
                config.gate_param_guide = gate_prior
        
        # Capture probability for VCP models
        if config.uses_variable_capture():
            p_capture_prior = priors.get("p_capture_prior", (1, 1))
            config.p_capture_distribution_model = dist.Beta(
                p_capture_prior[0], p_capture_prior[1])
            config.p_capture_param_prior = p_capture_prior
            if inference_method == "svi":
                config.p_capture_distribution_guide = dist.Beta(
                    p_capture_prior[0], p_capture_prior[1])
                config.p_capture_param_guide = p_capture_prior
        
        # Mixing weights for mixture models
        if config.is_mixture_model():
            ModelConfigFactory._configure_mixing_parameters(
                config, priors, inference_method)
    
    @staticmethod
    def _configure_mixing_parameters(
        config: ModelConfig, priors: Dict[str, Any], inference_method: str):
        """Configure mixing parameters for mixture models."""
        mixing_prior = priors.get("mixing_prior", jnp.ones(config.n_components))
        
        if isinstance(mixing_prior, (tuple, list)):
            if len(mixing_prior) == 1:
                # Symmetric Dirichlet
                concentration = jnp.ones(config.n_components) * mixing_prior[0]
            elif len(mixing_prior) == config.n_components:
                # Asymmetric Dirichlet
                concentration = jnp.array(mixing_prior)
            else:
                # Default to symmetric
                concentration = jnp.ones(config.n_components)
        else:
            # Assume it's already an array-like
            concentration = jnp.array(mixing_prior)
        
        config.mixing_distribution_model = dist.Dirichlet(concentration)
        config.mixing_param_prior = mixing_prior
        if inference_method == "svi":
            config.mixing_distribution_guide = dist.Dirichlet(concentration)
            config.mixing_param_guide = mixing_prior 