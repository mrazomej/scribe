"""
Distribution builder for SVI inference.

This module creates the appropriate distributions for SVI models and guides
based on the parameterization and prior specifications.
"""

from typing import Dict, Optional, Tuple, Any
import jax.numpy as jnp
import numpyro.distributions as dist
from ..stats import BetaPrime


class SVIDistributionBuilder:
    """Builds distributions for SVI inference."""
    
    @staticmethod
    def build_distributions(
        model_type: str,
        parameterization: str,
        priors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build all distributions needed for SVI inference.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        parameterization : str
            Parameterization type ("mean_field", "mean_variance", "beta_prime")
        priors : Dict[str, Any]
            Dictionary of prior parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all model and guide distributions
        """
        distributions = {}
        
        # Build core distributions based on parameterization
        if parameterization == "mean_field":
            distributions.update(SVIDistributionBuilder._build_mean_field_distributions(priors))
        elif parameterization == "mean_variance":
            distributions.update(SVIDistributionBuilder._build_mean_variance_distributions(priors))
        elif parameterization == "beta_prime":
            distributions.update(SVIDistributionBuilder._build_beta_prime_distributions(priors))
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Build model-specific distributions
        if "zinb" in model_type:
            distributions.update(SVIDistributionBuilder._build_gate_distributions(priors))
        
        if "vcp" in model_type:
            distributions.update(SVIDistributionBuilder._build_capture_distributions(
                parameterization, priors
            ))
        
        if "mix" in model_type:
            distributions.update(SVIDistributionBuilder._build_mixing_distributions(priors))
        
        return distributions
    
    @staticmethod
    def _build_mean_field_distributions(priors: Dict[str, Any]) -> Dict[str, Any]:
        """Build distributions for mean-field parameterization."""
        distributions = {}
        
        # p distribution
        p_prior = priors["p_prior"]
        distributions["p_distribution_model"] = dist.Beta(p_prior[0], p_prior[1])
        distributions["p_distribution_guide"] = dist.Beta(p_prior[0], p_prior[1])
        distributions["p_param_prior"] = p_prior
        distributions["p_param_guide"] = p_prior
        
        # r distribution
        r_prior = priors["r_prior"]
        distributions["r_distribution_model"] = dist.LogNormal(r_prior[0], r_prior[1])
        distributions["r_distribution_guide"] = dist.LogNormal(r_prior[0], r_prior[1])
        distributions["r_param_prior"] = r_prior
        distributions["r_param_guide"] = r_prior
        
        return distributions
    
    @staticmethod
    def _build_mean_variance_distributions(priors: Dict[str, Any]) -> Dict[str, Any]:
        """Build distributions for mean-variance parameterization."""
        distributions = {}
        
        # p distribution
        p_prior = priors["p_prior"]
        distributions["p_distribution_model"] = dist.Beta(p_prior[0], p_prior[1])
        distributions["p_distribution_guide"] = dist.Beta(p_prior[0], p_prior[1])
        distributions["p_param_prior"] = p_prior
        distributions["p_param_guide"] = p_prior
        
        # mu distribution
        mu_prior = priors["mu_prior"]
        distributions["mu_distribution_model"] = dist.LogNormal(mu_prior[0], mu_prior[1])
        distributions["mu_distribution_guide"] = dist.LogNormal(mu_prior[0], mu_prior[1])
        distributions["mu_param_prior"] = mu_prior
        distributions["mu_param_guide"] = mu_prior
        
        return distributions
    
    @staticmethod
    def _build_beta_prime_distributions(priors: Dict[str, Any]) -> Dict[str, Any]:
        """Build distributions for beta-prime parameterization."""
        distributions = {}
        
        # phi distribution
        phi_prior = priors["phi_prior"]
        distributions["phi_distribution_model"] = BetaPrime(phi_prior[0], phi_prior[1])
        distributions["phi_distribution_guide"] = BetaPrime(phi_prior[0], phi_prior[1])
        distributions["phi_param_prior"] = phi_prior
        distributions["phi_param_guide"] = phi_prior
        
        # mu distribution
        mu_prior = priors["mu_prior"]
        distributions["mu_distribution_model"] = dist.LogNormal(mu_prior[0], mu_prior[1])
        distributions["mu_distribution_guide"] = dist.LogNormal(mu_prior[0], mu_prior[1])
        distributions["mu_param_prior"] = mu_prior
        distributions["mu_param_guide"] = mu_prior
        
        return distributions
    
    @staticmethod
    def _build_gate_distributions(priors: Dict[str, Any]) -> Dict[str, Any]:
        """Build gate distributions for ZINB models."""
        distributions = {}
        
        gate_prior = priors["gate_prior"]
        distributions["gate_distribution_model"] = dist.Beta(gate_prior[0], gate_prior[1])
        distributions["gate_distribution_guide"] = dist.Beta(gate_prior[0], gate_prior[1])
        distributions["gate_param_prior"] = gate_prior
        distributions["gate_param_guide"] = gate_prior
        
        return distributions
    
    @staticmethod
    def _build_capture_distributions(
        parameterization: str, 
        priors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build capture probability distributions for VCP models."""
        distributions = {}
        
        if parameterization in ["mean_field", "mean_variance"]:
            p_capture_prior = priors["p_capture_prior"]
            distributions["p_capture_distribution_model"] = dist.Beta(
                p_capture_prior[0], p_capture_prior[1]
            )
            distributions["p_capture_distribution_guide"] = dist.Beta(
                p_capture_prior[0], p_capture_prior[1]
            )
            distributions["p_capture_param_prior"] = p_capture_prior
            distributions["p_capture_param_guide"] = p_capture_prior
        elif parameterization == "beta_prime":
            phi_capture_prior = priors["phi_capture_prior"]
            distributions["phi_capture_distribution_model"] = BetaPrime(
                phi_capture_prior[0], phi_capture_prior[1]
            )
            distributions["phi_capture_distribution_guide"] = BetaPrime(
                phi_capture_prior[0], phi_capture_prior[1]
            )
            distributions["phi_capture_param_prior"] = phi_capture_prior
            distributions["phi_capture_param_guide"] = phi_capture_prior
        
        return distributions
    
    @staticmethod
    def _build_mixing_distributions(priors: Dict[str, Any]) -> Dict[str, Any]:
        """Build mixing distributions for mixture models."""
        distributions = {}
        
        mixing_prior = priors["mixing_prior"]
        if isinstance(mixing_prior, (list, tuple, jnp.ndarray)):
            concentration = jnp.array(mixing_prior)
        else:
            raise ValueError(f"Invalid mixing_prior type: {type(mixing_prior)}")
            
        distributions["mixing_distribution_model"] = dist.Dirichlet(concentration)
        distributions["mixing_distribution_guide"] = dist.Dirichlet(concentration)
        distributions["mixing_param_prior"] = mixing_prior
        distributions["mixing_param_guide"] = mixing_prior
        
        return distributions 