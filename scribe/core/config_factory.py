"""
Model configuration factory for SCRIBE inference.

This module handles the creation of model configuration objects for both SVI
and MCMC inference methods.
"""

from typing import Dict, Optional, Any
import jax.numpy as jnp
import numpyro.distributions as dist
from ..model_config import ConstrainedModelConfig, UnconstrainedModelConfig
from ..stats import BetaPrime


class ModelConfigFactory:
    """Factory for creating model configuration objects."""
    
    @staticmethod
    def create_svi_config(
        model_type: str,
        parameterization: str, 
        distributions: Dict[str, Any],
        n_components: Optional[int] = None
    ) -> ConstrainedModelConfig:
        """
        Create ConstrainedModelConfig for SVI inference.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        parameterization : str
            Parameterization type ("mean_field", "mean_variance", "beta_prime")
        distributions : Dict[str, Any]
            Dictionary containing model and guide distributions
        n_components : Optional[int]
            Number of mixture components
            
        Returns
        -------
        ConstrainedModelConfig
            Configuration object for SVI inference
        """
        return ConstrainedModelConfig(
            base_model=model_type,
            n_components=n_components,
            # r distribution
            r_distribution_model=distributions.get("r_distribution_model"),
            r_distribution_guide=distributions.get("r_distribution_guide"),
            r_param_prior=distributions.get("r_param_prior"),
            r_param_guide=distributions.get("r_param_guide"),
            # p distribution
            p_distribution_model=distributions.get("p_distribution_model"),
            p_distribution_guide=distributions.get("p_distribution_guide"),
            p_param_prior=distributions.get("p_param_prior"),
            p_param_guide=distributions.get("p_param_guide"),
            # gate distribution
            gate_distribution_model=distributions.get("gate_distribution_model"),
            gate_distribution_guide=distributions.get("gate_distribution_guide"),
            gate_param_prior=distributions.get("gate_param_prior"),
            gate_param_guide=distributions.get("gate_param_guide"),
            # p_capture distribution
            p_capture_distribution_model=distributions.get("p_capture_distribution_model"),
            p_capture_distribution_guide=distributions.get("p_capture_distribution_guide"),
            p_capture_param_prior=distributions.get("p_capture_param_prior"),
            p_capture_param_guide=distributions.get("p_capture_param_guide"),
            # mixing distribution
            mixing_distribution_model=distributions.get("mixing_distribution_model"),
            mixing_distribution_guide=distributions.get("mixing_distribution_guide"),
            mixing_param_prior=distributions.get("mixing_param_prior"),
            mixing_param_guide=distributions.get("mixing_param_guide"),
            # mu distribution
            mu_distribution_model=distributions.get("mu_distribution_model"),
            mu_param_prior=distributions.get("mu_param_prior"),
            mu_distribution_guide=distributions.get("mu_distribution_guide"),
            mu_param_guide=distributions.get("mu_param_guide"),
            # phi distribution
            phi_distribution_model=distributions.get("phi_distribution_model"),
            phi_param_prior=distributions.get("phi_param_prior"),
            phi_distribution_guide=distributions.get("phi_distribution_guide"),
            phi_param_guide=distributions.get("phi_param_guide"),
            # phi_capture distribution
            phi_capture_distribution_model=distributions.get("phi_capture_distribution_model"),
            phi_capture_distribution_guide=distributions.get("phi_capture_distribution_guide"),
            phi_capture_param_prior=distributions.get("phi_capture_param_prior"),
            phi_capture_param_guide=distributions.get("phi_capture_param_guide"),
            # Parameterization
            parameterization=parameterization
        )
    
    @staticmethod
    def create_mcmc_config(
        model_type: str,
        unconstrained_model: bool,
        priors: Dict[str, Any],
        n_components: Optional[int] = None
    ) -> Any:
        """
        Create model config for MCMC inference.
        
        Parameters
        ----------
        model_type : str
            Type of model
        unconstrained_model : bool
            Whether to use unconstrained parameterization
        priors : Dict[str, Any]
            Dictionary of prior parameters
        n_components : Optional[int]
            Number of mixture components
            
        Returns
        -------
        Union[UnconstrainedModelConfig, ConstrainedModelConfig]
            Configuration object for MCMC inference
        """
        if unconstrained_model:
            return ModelConfigFactory._create_unconstrained_config(model_type, priors, n_components)
        else:
            return ModelConfigFactory._create_constrained_mcmc_config(model_type, priors, n_components)
    
    @staticmethod
    def _create_unconstrained_config(
        model_type: str,
        priors: Dict[str, Any],
        n_components: Optional[int]
    ) -> UnconstrainedModelConfig:
        """Create UnconstrainedModelConfig for MCMC."""
        # Extract mixing priors if applicable
        mixing_logits_loc = None
        mixing_logits_scale = None
        if "mix" in model_type and "mixing_prior" in priors:
            mixing_prior = priors["mixing_prior"]
            if isinstance(mixing_prior, tuple) and len(mixing_prior) == 2:
                mixing_logits_loc = mixing_prior[0]
                mixing_logits_scale = mixing_prior[1]
        
        config = UnconstrainedModelConfig(
            base_model=model_type,
            n_components=n_components,
            # Unconstrained parameterization - all use normal priors on transformed parameters
            p_unconstrained_loc=priors["p_prior"][0],
            p_unconstrained_scale=priors["p_prior"][1],
            r_unconstrained_loc=priors["r_prior"][0],
            r_unconstrained_scale=priors["r_prior"][1],
            gate_unconstrained_loc=priors.get("gate_prior", (None, None))[0],
            gate_unconstrained_scale=priors.get("gate_prior", (None, None))[1],
            p_capture_unconstrained_loc=priors.get("p_capture_prior", (None, None))[0],
            p_capture_unconstrained_scale=priors.get("p_capture_prior", (None, None))[1],
            mixing_logits_unconstrained_loc=mixing_logits_loc,
            mixing_logits_unconstrained_scale=mixing_logits_scale,
        )
        
        # Validate and set defaults
        config.validate()
        return config
    
    @staticmethod
    def _create_constrained_mcmc_config(
        model_type: str,
        priors: Dict[str, Any],
        n_components: Optional[int]
    ) -> ConstrainedModelConfig:
        """Create ConstrainedModelConfig for MCMC."""
        # Create distribution objects
        
        # Success probability distribution
        p_prior = priors["p_prior"]
        p_dist_model = dist.Beta(p_prior[0], p_prior[1])
        p_dist_guide = dist.Beta(p_prior[0], p_prior[1])
        
        # Dispersion parameter distribution - determine from r_prior length or default to gamma
        r_prior = priors["r_prior"]
        if len(r_prior) == 2:
            # Assume Gamma if 2 parameters, could also be LogNormal
            # We'll default to Gamma but this could be made configurable
            r_dist_model = dist.Gamma(r_prior[0], r_prior[1])
            r_dist_guide = dist.Gamma(r_prior[0], r_prior[1])
        else:
            raise ValueError(f"Invalid r_prior length: {len(r_prior)}")
        
        # Optional distributions
        gate_dist_model = gate_dist_guide = None
        if "zinb" in model_type and "gate_prior" in priors:
            gate_prior = priors["gate_prior"]
            gate_dist_model = dist.Beta(gate_prior[0], gate_prior[1])
            gate_dist_guide = dist.Beta(gate_prior[0], gate_prior[1])
        
        p_capture_dist_model = p_capture_dist_guide = None
        if "vcp" in model_type and "p_capture_prior" in priors:
            p_capture_prior = priors["p_capture_prior"]
            p_capture_dist_model = dist.Beta(p_capture_prior[0], p_capture_prior[1])
            p_capture_dist_guide = dist.Beta(p_capture_prior[0], p_capture_prior[1])
        
        mixing_dist_model = mixing_dist_guide = None
        if "mix" in model_type and "mixing_prior" in priors:
            mixing_prior = priors["mixing_prior"]
            if isinstance(mixing_prior, tuple) and len(mixing_prior) == 1:
                # Symmetric Dirichlet
                concentration = jnp.ones(n_components) * mixing_prior[0]
            elif isinstance(mixing_prior, (tuple, list)) and len(mixing_prior) == n_components:
                # Asymmetric Dirichlet
                concentration = jnp.array(mixing_prior)
            else:
                # Default to symmetric Dirichlet
                concentration = jnp.ones(n_components)
            mixing_dist_model = dist.Dirichlet(concentration)
            mixing_dist_guide = dist.Dirichlet(concentration)
        
        config = ConstrainedModelConfig(
            base_model=model_type,
            n_components=n_components,
            # Distribution objects
            r_distribution_model=r_dist_model,
            r_distribution_guide=r_dist_guide,
            r_param_prior=r_prior,
            r_param_guide=r_prior,
            p_distribution_model=p_dist_model,
            p_distribution_guide=p_dist_guide,
            p_param_prior=p_prior,
            p_param_guide=p_prior,
            gate_distribution_model=gate_dist_model,
            gate_distribution_guide=gate_dist_guide,
            gate_param_prior=priors.get("gate_prior"),
            gate_param_guide=priors.get("gate_prior"),
            p_capture_distribution_model=p_capture_dist_model,
            p_capture_distribution_guide=p_capture_dist_guide,
            p_capture_param_prior=priors.get("p_capture_prior"),
            p_capture_param_guide=priors.get("p_capture_prior"),
            mixing_distribution_model=mixing_dist_model,
            mixing_distribution_guide=mixing_dist_guide,
            mixing_param_prior=priors.get("mixing_prior"),
            mixing_param_guide=priors.get("mixing_prior"),
        )
        
        # Validate configuration
        config.validate()
        return config 