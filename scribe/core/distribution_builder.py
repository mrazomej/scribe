"""
Unified distribution builder for both SVI and MCMC inference.

This module creates the appropriate distributions for models based on the 
parameterization and prior specifications. It accepts actual NumPyro distribution
classes for maximum flexibility.
"""

from typing import Dict, Optional, Tuple, Any, Type, Union
import jax.numpy as jnp
import numpyro.distributions as dist
from ..stats import BetaPrime


class DistributionBuilder:
    """Builds distributions for both SVI and MCMC inference."""
    
    @staticmethod
    def build_distributions(
        model_type: str,
        parameterization: str,
        inference_method: str,
        priors: Dict[str, Any],
        r_distribution: Optional[Type[dist.Distribution]] = None,
        mu_distribution: Optional[Type[dist.Distribution]] = None
    ) -> Dict[str, Any]:
        """
        Build all distributions needed for inference.
        
        Parameters
        ----------
        model_type : str
            Type of model (e.g., "nbdm", "zinb_mix", etc.)
        parameterization : str
            Parameterization type ("standard", "linked", "odds_ratio",
            "unconstrained")
        inference_method : str
            Inference method ("svi" or "mcmc")
        priors : Dict[str, Any]
            Dictionary of prior parameters
        r_distribution : Optional[Type[dist.Distribution]]
            NumPyro distribution class for r parameter (e.g., dist.Gamma,
            dist.LogNormal) If None, defaults to LogNormal for constrained,
            Normal for unconstrained
        mu_distribution : Optional[Type[dist.Distribution]]
            NumPyro distribution class for mu parameter (e.g., dist.LogNormal,
            dist.Gamma) If None, defaults to LogNormal
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all model and guide distributions
        """
        distributions = {}
        
        # Set default distributions if not provided
        if r_distribution is None:
            if parameterization == "unconstrained":
                r_distribution = dist.Normal
            else:
                r_distribution = dist.LogNormal
        if mu_distribution is None:
            mu_distribution = dist.LogNormal
        
        # Build core distributions based on parameterization
        if parameterization == "standard":
            distributions.update(
                DistributionBuilder._build_standard_distributions(
                    priors, r_distribution, inference_method
                ))
        elif parameterization == "linked":
            distributions.update(
                DistributionBuilder._build_linked_distributions(
                    priors, mu_distribution, inference_method
                ))
        elif parameterization == "odds_ratio":
            distributions.update(
                DistributionBuilder._build_odds_ratio_distributions(
                    priors, mu_distribution, inference_method
                ))
        elif parameterization == "unconstrained":
            distributions.update(
                DistributionBuilder._build_unconstrained_distributions(
                    priors, r_distribution, inference_method
                ))
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        # Build model-specific distributions
        if "zinb" in model_type:
            distributions.update(
                DistributionBuilder._build_gate_distributions(
                    priors, parameterization, inference_method
                ))
        
        if "vcp" in model_type:
            distributions.update(
                DistributionBuilder._build_capture_distributions(
                    parameterization, priors, inference_method
                ))
        
        if "mix" in model_type:
            distributions.update(
                DistributionBuilder._build_mixing_distributions(
                    priors, parameterization, inference_method
                ))
        
        return distributions
    
    @staticmethod
    def _build_standard_distributions(
        priors: Dict[str, Any], 
        r_distribution: Type[dist.Distribution],
        inference_method: str
    ) -> Dict[str, Any]:
        """Build distributions for mean-field parameterization."""
        distributions = {}
        
        # p distribution (always Beta for constrained)
        p_prior = priors["p_prior"]
        distributions["p_distribution_model"] = dist.Beta(
            p_prior[0], p_prior[1])
        if inference_method == "svi":
            distributions["p_distribution_guide"] = dist.Beta(
                p_prior[0], p_prior[1])
            distributions["p_param_guide"] = p_prior
        distributions["p_param_prior"] = p_prior
        
        # r distribution (configurable)
        r_prior = priors["r_prior"]
        distributions["r_distribution_model"] = r_distribution(*r_prior)
        if inference_method == "svi":
            distributions["r_distribution_guide"] = r_distribution(*r_prior)
            distributions["r_param_guide"] = r_prior
        distributions["r_param_prior"] = r_prior
        
        return distributions
    
    @staticmethod
    def _build_linked_distributions(
        priors: Dict[str, Any], 
        mu_distribution: Type[dist.Distribution],
        inference_method: str
    ) -> Dict[str, Any]:
        """Build distributions for mean-variance parameterization."""
        distributions = {}
        
        # p distribution
        p_prior = priors["p_prior"]
        distributions["p_distribution_model"] = dist.Beta(
            p_prior[0], p_prior[1])
        if inference_method == "svi":
            distributions["p_distribution_guide"] = dist.Beta(
                p_prior[0], p_prior[1])
            distributions["p_param_guide"] = p_prior
        distributions["p_param_prior"] = p_prior
        
        # mu distribution (configurable)
        mu_prior = priors["mu_prior"]
        distributions["mu_distribution_model"] = mu_distribution(*mu_prior)
        if inference_method == "svi":
            distributions["mu_distribution_guide"] = mu_distribution(*mu_prior)
            distributions["mu_param_guide"] = mu_prior
        distributions["mu_param_prior"] = mu_prior
        
        return distributions
    
    @staticmethod
    def _build_odds_ratio_distributions(
        priors: Dict[str, Any], 
        mu_distribution: Type[dist.Distribution],
        inference_method: str
    ) -> Dict[str, Any]:
        """Build distributions for beta-prime parameterization."""
        distributions = {}
        
        # phi distribution (always BetaPrime)
        phi_prior = priors["phi_prior"]
        distributions["phi_distribution_model"] = BetaPrime(
            phi_prior[0], phi_prior[1])
        if inference_method == "svi":
            distributions["phi_distribution_guide"] = BetaPrime(
                phi_prior[0], phi_prior[1])
            distributions["phi_param_guide"] = phi_prior
        distributions["phi_param_prior"] = phi_prior
        
        # mu distribution (configurable)
        mu_prior = priors["mu_prior"]
        distributions["mu_distribution_model"] = mu_distribution(*mu_prior)
        if inference_method == "svi":
            distributions["mu_distribution_guide"] = mu_distribution(*mu_prior)
            distributions["mu_param_guide"] = mu_prior
        distributions["mu_param_prior"] = mu_prior
        
        return distributions
    
    @staticmethod
    def _build_unconstrained_distributions(
        priors: Dict[str, Any], 
        r_distribution: Type[dist.Distribution],
        inference_method: str
    ) -> Dict[str, Any]:
        """Build distributions for unconstrained parameterization."""
        distributions = {}
        
        # For unconstrained, we typically use Normal distributions on
        # transformed parameters
        # p_logit ~ Normal (logit-transformed p)
        p_prior = priors["p_prior"]
        distributions["p_distribution_model"] = dist.Normal(p_prior[0], p_prior[1])
        distributions["p_param_prior"] = p_prior
        
        # r_log ~ configurable distribution (typically Normal for
        # log-transformed r)
        r_prior = priors["r_prior"]
        distributions["r_distribution_model"] = r_distribution(*r_prior)
        distributions["r_param_prior"] = r_prior
        
        return distributions
    
    @staticmethod
    def _build_gate_distributions(
        priors: Dict[str, Any], 
        parameterization: str,
        inference_method: str
    ) -> Dict[str, Any]:
        """Build gate distributions for ZINB models."""
        distributions = {}
        
        if parameterization == "unconstrained":
            # gate_logit ~ Normal for unconstrained
            gate_prior = priors["gate_prior"]
            distributions["gate_distribution_model"] = dist.Normal(
                gate_prior[0], gate_prior[1])
            distributions["gate_param_prior"] = gate_prior
        else:
            # gate ~ Beta for constrained
            gate_prior = priors["gate_prior"]
            distributions["gate_distribution_model"] = dist.Beta(
                gate_prior[0], gate_prior[1])
            if inference_method == "svi":
                distributions["gate_distribution_guide"] = dist.Beta(
                    gate_prior[0], gate_prior[1])
                distributions["gate_param_guide"] = gate_prior
            distributions["gate_param_prior"] = gate_prior
        
        return distributions
    
    @staticmethod
    def _build_capture_distributions(
        parameterization: str, 
        priors: Dict[str, Any],
        inference_method: str
    ) -> Dict[str, Any]:
        """Build capture probability distributions for VCP models."""
        distributions = {}
        
        if parameterization == "unconstrained":
            # p_capture_logit ~ Normal for unconstrained
            p_capture_prior = priors["p_capture_prior"]
            distributions["p_capture_distribution_model"] = dist.Normal(
                p_capture_prior[0], p_capture_prior[1]
            )
            distributions["p_capture_param_prior"] = p_capture_prior
        elif parameterization in ["standard", "linked"]:
            # p_capture ~ Beta for constrained
            p_capture_prior = priors["p_capture_prior"]
            distributions["p_capture_distribution_model"] = dist.Beta(
                p_capture_prior[0], p_capture_prior[1]
            )
            if inference_method == "svi":
                distributions["p_capture_distribution_guide"] = dist.Beta(
                    p_capture_prior[0], p_capture_prior[1]
                )
                distributions["p_capture_param_guide"] = p_capture_prior
            distributions["p_capture_param_prior"] = p_capture_prior
        elif parameterization == "odds_ratio":
            # phi_capture ~ BetaPrime for beta-prime parameterization
            phi_capture_prior = priors["phi_capture_prior"]
            distributions["phi_capture_distribution_model"] = BetaPrime(
                phi_capture_prior[0], phi_capture_prior[1]
            )
            if inference_method == "svi":
                distributions["phi_capture_distribution_guide"] = BetaPrime(
                    phi_capture_prior[0], phi_capture_prior[1]
                )
                distributions["phi_capture_param_guide"] = phi_capture_prior
            distributions["phi_capture_param_prior"] = phi_capture_prior
        
        return distributions
    
    @staticmethod
    def _build_mixing_distributions(
        priors: Dict[str, Any], 
        parameterization: str,
        inference_method: str
    ) -> Dict[str, Any]:
        """Build mixing distributions for mixture models."""
        distributions = {}
        
        mixing_prior = priors["mixing_prior"]
        
        if parameterization == "unconstrained":
            # For unconstrained, we might use Normal on logits of mixing weights
            # This is more complex for Dirichlet, so for now we'll use the same
            # approach
            if isinstance(mixing_prior, (list, tuple, jnp.ndarray)):
                concentration = jnp.array(mixing_prior)
            else:
                raise ValueError(
                    f"Invalid mixing_prior type: {type(mixing_prior)}")
            distributions["mixing_distribution_model"] = dist.Dirichlet(concentration)
            distributions["mixing_param_prior"] = mixing_prior
        else:
            # Standard Dirichlet for constrained
            if isinstance(mixing_prior, (list, tuple, jnp.ndarray)):
                concentration = jnp.array(mixing_prior)
            else:
                raise ValueError(f"Invalid mixing_prior type: {type(mixing_prior)}")
                
            distributions["mixing_distribution_model"] = dist.Dirichlet(concentration)
            if inference_method == "svi":
                distributions["mixing_distribution_guide"] = dist.Dirichlet(concentration)
                distributions["mixing_param_guide"] = mixing_prior
            distributions["mixing_param_prior"] = mixing_prior
        
        return distributions


def validate_distribution_priors(
    distribution_class: Type[dist.Distribution], 
    prior_params: Tuple
) -> bool:
    """
    Validate that prior parameters match the expected parameters for a
    distribution.
    
    Parameters
    ----------
    distribution_class : Type[dist.Distribution]
        The NumPyro distribution class
    prior_params : Tuple
        The prior parameters to validate
        
    Returns
    -------
    bool
        True if parameters are valid for the distribution
        
    Raises
    ------
    ValueError
        If parameters don't match the distribution's expected parameters
    """
    try:
        # Try to instantiate the distribution with the given parameters
        test_dist = distribution_class(*prior_params)
        return True
    except Exception as e:
        raise ValueError(
            f"Invalid prior parameters {prior_params} for distribution "
            f"{distribution_class.__name__}: {str(e)}"
        )


def get_supported_distributions() -> Dict[str, Dict[str, Type[dist.Distribution]]]:
    """
    Get dictionary of supported distributions for different parameters.
    
    Returns
    -------
    Dict[str, Dict[str, Type[dist.Distribution]]]
        Dictionary mapping parameter types to supported distributions
    """
    return {
        "r_parameter": {
            "gamma": dist.Gamma,
            "lognormal": dist.LogNormal,
            "inverse_gamma": dist.InverseGamma,
            "exponential": dist.Exponential,
            "normal": dist.Normal  # For unconstrained (log-transformed)
        },
        "mu_parameter": {
            "lognormal": dist.LogNormal,
            "gamma": dist.Gamma,
            "inverse_gamma": dist.InverseGamma,
            "exponential": dist.Exponential,
            "normal": dist.Normal  # For unconstrained (log-transformed)
        },
        "probability_parameters": {
            "beta": dist.Beta,
            "normal": dist.Normal  # For unconstrained (logit-transformed)
        },
        "phi_parameter": {
            "odds_ratio": BetaPrime,
            "gamma": dist.Gamma,
            "lognormal": dist.LogNormal
        }
    } 