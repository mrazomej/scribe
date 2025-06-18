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
        parameterization: str,
        r_distribution: Optional[Any] = None,
        mu_distribution: Optional[Any] = None,
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
        parameterization : str
            Parameterization type ("mean_field", "mean_variance", "beta_prime", "unconstrained")
        r_distribution : Optional[Any]
            NumPyro distribution class for r parameter
        mu_distribution : Optional[Any]
            NumPyro distribution class for mu parameter
        n_components : Optional[int]
            Number of mixture components
            
        Returns
        -------
        Dict[str, Tuple]
            Dictionary of default prior parameters
        """
        if inference_method not in ["svi", "mcmc"]:
            raise ValueError(f"Unknown inference method: {inference_method}")
            
        return PriorConfigFactory._get_unified_defaults(
            model_type, 
            parameterization, 
            r_distribution, 
            mu_distribution, 
            n_components
        )
    
    @staticmethod
    def _get_unified_defaults(
        model_type: str,
        parameterization: str,
        r_distribution: Optional[Any] = None,
        mu_distribution: Optional[Any] = None,
        n_components: Optional[int] = None
    ) -> Dict[str, Tuple]:
        """Get default priors for unified parameterization approach."""
        defaults = {}
        
        # Default priors based on parameterization
        if parameterization == "mean_field":
            # Choose r_prior based on distribution type
            if r_distribution is not None:
                # Try to infer appropriate defaults from distribution name
                dist_name = getattr(r_distribution, '__name__', str(r_distribution))
                if 'Gamma' in dist_name:
                    defaults["r_prior"] = (2, 0.1)  # Gamma (concentration, rate)
                elif 'LogNormal' in dist_name:
                    defaults["r_prior"] = (1, 1)    # LogNormal (mu, sigma)
                else:
                    defaults["r_prior"] = (1, 1)    # Default to LogNormal-style
            else:
                defaults["r_prior"] = (1, 1)        # Default LogNormal
            defaults["p_prior"] = (1, 1)            # Beta distribution
            
        elif parameterization == "mean_variance":
            defaults["p_prior"] = (1, 1)            # Beta distribution  
            defaults["mu_prior"] = (1, 1)           # LogNormal distribution
            
        elif parameterization == "beta_prime":
            defaults["phi_prior"] = (1, 1)          # BetaPrime distribution
            defaults["mu_prior"] = (1, 1)           # LogNormal distribution
            
        elif parameterization == "unconstrained":
            # Normal priors on transformed parameters
            defaults["r_prior"] = (0, 1)            # Normal on log(r)
            defaults["p_prior"] = (0, 1)            # Normal on logit(p)
            
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Model-specific priors
        if "zinb" in model_type:
            if parameterization == "unconstrained":
                defaults["gate_prior"] = (0, 1)     # Normal on logit(gate)
            else:
                defaults["gate_prior"] = (1, 1)     # Beta distribution
            
        if "vcp" in model_type:
            if parameterization == "unconstrained":
                defaults["p_capture_prior"] = (0, 1)  # Normal on logit(p_capture)
            elif parameterization in ["mean_field", "mean_variance"]:
                defaults["p_capture_prior"] = (1, 1)  # Beta distribution
            elif parameterization == "beta_prime":
                defaults["phi_capture_prior"] = (1, 1)  # BetaPrime distribution
                
        if "mix" in model_type and n_components is not None:
            if parameterization == "unconstrained":
                defaults["mixing_prior"] = (0, 1)    # Normal on logits (simplified)
            else:
                defaults["mixing_prior"] = jnp.ones(n_components)  # Symmetric Dirichlet
            
        return defaults
    
    @staticmethod
    def validate_priors(
        model_type: str,
        inference_method: str, 
        parameterization: str,
        prior_dict: Dict[str, Any]
    ) -> None:
        """
        Validate that provided priors are compatible with model and inference
        method.
        
        Parameters
        ----------
        model_type : str
            Type of model
        inference_method : str
            Inference method ("svi" or "mcmc")
        parameterization : str
            Parameterization type
        prior_dict : Dict[str, Any]
            Dictionary of prior parameters to validate
            
        Raises
        ------
        ValueError
            If priors are incompatible with the configuration
        """
        PriorConfigFactory._validate_unified_priors(model_type, parameterization, prior_dict)
    
    @staticmethod
    def _validate_unified_priors(model_type: str, parameterization: str, prior_dict: Dict[str, Any]) -> None:
        """Validate priors for unified parameterization approach."""
        # Check parameterization-specific requirements
        if parameterization == "mean_field":
            required_priors = ["p_prior", "r_prior"]
        elif parameterization == "mean_variance":
            required_priors = ["p_prior", "mu_prior"]
        elif parameterization == "beta_prime":
            required_priors = ["phi_prior", "mu_prior"]
        elif parameterization == "unconstrained":
            required_priors = ["p_prior", "r_prior"]
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Check model-specific requirements
        if "zinb" in model_type:
            required_priors.append("gate_prior")
        if "vcp" in model_type:
            if parameterization == "unconstrained":
                required_priors.append("p_capture_prior")
            elif parameterization in ["mean_field", "mean_variance"]:
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