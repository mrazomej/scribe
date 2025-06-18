"""
Prior configuration factory for SCRIBE inference.

This module handles the creation of default prior parameters and validation
for both SVI and MCMC inference methods.
"""

from typing import Dict, Optional, Tuple, Any
import jax.numpy as jnp


class PriorConfigFactory:
    """Factory for creating and validating prior configurations."""
    
    @staticmethod
    def create_default_priors(
        model_type: str,
        inference_method: str,
        parameterization: Optional[str] = None,
        unconstrained_model: Optional[bool] = None,
        r_dist: Optional[str] = None,
        n_components: Optional[int] = None
    ) -> Dict[str, Tuple]:
        """
        Create default prior parameters based on inference method and model configuration.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        inference_method : str
            Inference method ("svi" or "mcmc")
        parameterization : Optional[str]
            For SVI: "mean_field", "mean_variance", or "beta_prime"
        unconstrained_model : Optional[bool]
            For MCMC: whether to use unconstrained parameterization
        r_dist : Optional[str]
            Distribution for r parameter ("gamma" or "lognormal")
        n_components : Optional[int]
            Number of mixture components
            
        Returns
        -------
        Dict[str, Tuple]
            Dictionary of default prior parameters
        """
        defaults = {}
        
        if inference_method == "svi":
            if parameterization is None:
                raise ValueError("parameterization must be specified for SVI inference")
            defaults.update(PriorConfigFactory._get_svi_defaults(model_type, parameterization, r_dist, n_components))
        elif inference_method == "mcmc":
            if unconstrained_model is None:
                raise ValueError("unconstrained_model must be specified for MCMC inference")
            defaults.update(PriorConfigFactory._get_mcmc_defaults(model_type, unconstrained_model, r_dist, n_components))
        else:
            raise ValueError(f"Unknown inference method: {inference_method}")
            
        return defaults
    
    @staticmethod
    def _get_svi_defaults(
        model_type: str, 
        parameterization: str, 
        r_dist: Optional[str],
        n_components: Optional[int]
    ) -> Dict[str, Tuple]:
        """Get default priors for SVI inference."""
        defaults = {}
        
        # Default priors based on parameterization
        if parameterization == "mean_field":
            defaults["r_prior"] = (1, 1)  # Default for r distribution
            defaults["p_prior"] = (1, 1)  # Beta distribution
        elif parameterization == "mean_variance":
            defaults["p_prior"] = (1, 1)  # Beta distribution  
            defaults["mu_prior"] = (1, 1)  # LogNormal distribution
        elif parameterization == "beta_prime":
            defaults["phi_prior"] = (1, 1)  # BetaPrime distribution
            defaults["mu_prior"] = (1, 1)  # LogNormal distribution
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Model-specific priors
        if "zinb" in model_type:
            defaults["gate_prior"] = (1, 1)  # Beta distribution
            
        if "vcp" in model_type:
            if parameterization in ["mean_field", "mean_variance"]:
                defaults["p_capture_prior"] = (1, 1)  # Beta distribution
            elif parameterization == "beta_prime":
                defaults["phi_capture_prior"] = (1, 1)  # BetaPrime distribution
                
        if "mix" in model_type and n_components is not None:
            defaults["mixing_prior"] = jnp.ones(n_components)  # Symmetric Dirichlet
            
        return defaults
    
    @staticmethod
    def _get_mcmc_defaults(
        model_type: str, 
        unconstrained_model: bool, 
        r_dist: Optional[str],
        n_components: Optional[int]
    ) -> Dict[str, Tuple]:
        """Get default priors for MCMC inference."""
        defaults = {}
        
        if unconstrained_model:
            # Defaults for unconstrained models (normal priors on transformed parameters)
            defaults["r_prior"] = (0, 1)    # Normal on log(r)
            defaults["p_prior"] = (0, 1)    # Normal on logit(p)
            
            if "zinb" in model_type:
                defaults["gate_prior"] = (0, 1)  # Normal on logit(gate)
            if "vcp" in model_type:
                defaults["p_capture_prior"] = (0, 1)  # Normal on logit(p_capture)
            if "mix" in model_type:
                defaults["mixing_prior"] = (0, 1)  # Normal on logits
        else:
            # Defaults for constrained models (natural parameter distributions)
            if r_dist == "gamma":
                defaults["r_prior"] = (2, 0.1)  # Gamma distribution
            else:  # lognormal
                defaults["r_prior"] = (1, 1)   # LogNormal distribution
                
            defaults["p_prior"] = (1, 1)       # Beta distribution
            
            if "zinb" in model_type:
                defaults["gate_prior"] = (1, 1)  # Beta distribution
            if "vcp" in model_type:
                defaults["p_capture_prior"] = (1, 1)  # Beta distribution
            if "mix" in model_type:
                defaults["mixing_prior"] = (1.0,)  # Symmetric Dirichlet
                
        return defaults
    
    @staticmethod
    def validate_priors(
        model_type: str,
        inference_method: str, 
        parameterization: Optional[str],
        prior_dict: Dict[str, Any]
    ) -> None:
        """
        Validate that provided priors are compatible with model and inference method.
        
        Parameters
        ----------
        model_type : str
            Type of model
        inference_method : str
            Inference method ("svi" or "mcmc")
        parameterization : Optional[str]
            Parameterization type (for SVI)
        prior_dict : Dict[str, Any]
            Dictionary of prior parameters to validate
            
        Raises
        ------
        ValueError
            If priors are incompatible with the configuration
        """
        # Check for required priors based on model type and parameterization
        if inference_method == "svi":
            if parameterization is None:
                raise ValueError("parameterization must be specified for SVI inference")
            PriorConfigFactory._validate_svi_priors(model_type, parameterization, prior_dict)
        elif inference_method == "mcmc":
            PriorConfigFactory._validate_mcmc_priors(model_type, prior_dict)
        else:
            raise ValueError(f"Unknown inference method: {inference_method}")
    
    @staticmethod
    def _validate_svi_priors(model_type: str, parameterization: str, prior_dict: Dict[str, Any]) -> None:
        """Validate SVI-specific priors."""
        # Check parameterization-specific requirements
        if parameterization == "mean_field":
            required_priors = ["p_prior", "r_prior"]
        elif parameterization == "mean_variance":
            required_priors = ["p_prior", "mu_prior"]
        elif parameterization == "beta_prime":
            required_priors = ["phi_prior", "mu_prior"]
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Check model-specific requirements
        if "zinb" in model_type:
            required_priors.append("gate_prior")
        if "vcp" in model_type:
            if parameterization in ["mean_field", "mean_variance"]:
                required_priors.append("p_capture_prior")
            elif parameterization == "beta_prime":
                required_priors.append("phi_capture_prior")
        if "mix" in model_type:
            required_priors.append("mixing_prior")
        
        # Validate that all required priors are present and valid
        for prior_name in required_priors:
            if prior_name not in prior_dict:
                raise ValueError(f"Missing required prior: {prior_name}")
            
            prior_value = prior_dict[prior_name]
            if prior_name == "mixing_prior":
                # Special validation for mixing priors
                if not isinstance(prior_value, (tuple, list, jnp.ndarray)):
                    raise ValueError(f"mixing_prior must be a sequence, got {type(prior_value)}")
            else:
                # Standard tuple validation
                if not isinstance(prior_value, tuple) or len(prior_value) != 2:
                    raise ValueError(f"{prior_name} must be a tuple of length 2, got {prior_value}")
    
    @staticmethod
    def _validate_mcmc_priors(model_type: str, prior_dict: Dict[str, Any]) -> None:
        """Validate MCMC-specific priors."""
        required_priors = ["p_prior", "r_prior"]
        
        # Check model-specific requirements
        if "zinb" in model_type:
            required_priors.append("gate_prior")
        if "vcp" in model_type:
            required_priors.append("p_capture_prior")
        if "mix" in model_type:
            required_priors.append("mixing_prior")
        
        # Validate that all required priors are present and valid
        for prior_name in required_priors:
            if prior_name not in prior_dict:
                raise ValueError(f"Missing required prior: {prior_name}")
            
            prior_value = prior_dict[prior_name]
            if prior_name == "mixing_prior":
                # Special validation for mixing priors
                if not isinstance(prior_value, (tuple, list, jnp.ndarray)):
                    raise ValueError(f"mixing_prior must be a sequence, got {type(prior_value)}")
            else:
                # Standard tuple validation
                if not isinstance(prior_value, tuple) or len(prior_value) != 2:
                    raise ValueError(f"{prior_name} must be a tuple of length 2, got {prior_value}") 