"""
Distribution configurations for SCRIBE.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions.distribution import DistributionMeta
from numpyro.distributions import constraints
from dataclasses import dataclass
import scipy.stats as stats

# ------------------------------------------------------------------------------
# Model configurations
# ------------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Configuration class for SCRIBE model parameters and distributions.
    
    This class handles configuration of model types and their required
    distributions. It supports single and mixture model variants of NBDM, ZINB,
    NBVCP, and ZINBVCP models.
    
    Parameters
    ----------
    base_model : str
        Base model type. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp", or
        their mixture variants ending in "_mix"
    r_distribution_model : numpyro.distributions.Distribution
        Distribution object for the dispersion (r) parameter in the model
    r_distribution_guide : numpyro.distributions.Distribution  
        Distribution object for the dispersion (r) parameter in the guide
    p_distribution_model : numpyro.distributions.Distribution
        Distribution object for the success probability (p) parameter in the
        model
    p_distribution_guide : numpyro.distributions.Distribution
        Distribution object for the success probability (p) parameter in the
        guide
    gate_distribution_model : Optional[numpyro.distributions.Distribution]
        Distribution object for the dropout gate parameter in the model.
        Required for ZINB models.
    gate_distribution_guide : Optional[numpyro.distributions.Distribution]  
        Distribution object for the dropout gate parameter in the guide.
        Required for ZINB models.
    p_capture_distribution_model : Optional[numpyro.distributions.Distribution]
        Distribution object for the capture probability parameter in the model.
        Required for VCP models.
    p_capture_distribution_guide : Optional[numpyro.distributions.Distribution]
        Distribution object for the capture probability parameter in the guide.
        Required for VCP models.
    n_components : Optional[int]
        Number of mixture components. If specified, model type will be converted
        to mixture variant.
    """
    # Base model type (e.g. "nbdm", "zinb")
    base_model: str
       
    # Distribution config for success probability parameter  
    p_distribution_model: dist.Distribution
    p_distribution_guide: dist.Distribution
    p_param_prior: tuple
    p_param_guide: tuple

    # Distribution config for dispersion parameter
    r_distribution_model: dist.Distribution
    r_distribution_guide: dist.Distribution
    r_param_prior: tuple
    r_param_guide: tuple
    
    # Optional distribution config for dropout gate parameter
    gate_distribution_model: Optional[dist.Distribution] = None
    gate_distribution_guide: Optional[dist.Distribution] = None
    gate_param_prior: Optional[tuple] = None
    gate_param_guide: Optional[tuple] = None

    # Optional distribution config for capture probability parameter
    p_capture_distribution_model: Optional[dist.Distribution] = None
    p_capture_distribution_guide: Optional[dist.Distribution] = None
    p_capture_param_prior: Optional[tuple] = None
    p_capture_param_guide: Optional[tuple] = None

    # Optional number of mixture components
    n_components: Optional[int] = None
    
    def __post_init__(self):
        # Define set of valid model types
        valid_models = {"nbdm", "zinb", "nbvcp", "zinbvcp", 
                       "nbdm_mix", "zinb_mix", "nbvcp_mix", "zinbvcp_mix"}
        
        # Validate that provided model type is supported
        if self.base_model not in valid_models:
            raise ValueError(f"Invalid model type: {self.base_model}")
            
        # If n_components is specified, convert model type to mixture variant
        if self.n_components is not None and not self.base_model.endswith("_mix"):
            self.base_model = f"{self.base_model}_mix"
            
        # Ensure ZINB models have gate distribution specified
        if "zinb" in self.base_model and \
            (self.gate_distribution_model is None or 
             self.gate_distribution_guide is None):
            raise ValueError("ZINB models require gate_distribution_model and gate_distribution_guide")

        # Ensure non ZINB models do not have gate distribution specified
        if "zinb" not in self.base_model and \
            (self.gate_distribution_model is not None or 
             self.gate_distribution_guide is not None):
            raise ValueError("Non-ZINB models do not require gate_distribution")
            
        # Ensure VCP models have capture probability distribution specified
        if "vcp" in self.base_model and \
            (self.p_capture_distribution_model is None or 
             self.p_capture_distribution_guide is None):
            raise ValueError("VCP models require p_capture_distribution_model and p_capture_distribution_guide")

        # Ensure non VCP models do not have capture probability distribution
        # specified
        if "vcp" not in self.base_model and \
            (self.p_capture_distribution_model is not None or 
             self.p_capture_distribution_guide is not None):
            raise ValueError("Non-VCP models do not require p_capture_distribution")

# ------------------------------------------------------------------------------
# Model Configurations
# ------------------------------------------------------------------------------

def get_model_config(
    base_model: str,
    p_distribution_model: str = "beta",
    p_distribution_guide: str = "beta",
    p_param_prior: tuple = (1, 1),
    p_param_guide: tuple = (1, 1),
    r_distribution_model: str = "lognormal",
    r_distribution_guide: str = "lognormal", 
    r_param_prior: tuple = (0, 1),
    r_param_guide: tuple = (0, 1),
    gate_distribution_model: Optional[str] = None,
    gate_distribution_guide: Optional[str] = None,
    gate_param_prior: Optional[tuple] = None,
    gate_param_guide: Optional[tuple] = None,
    p_capture_distribution_model: Optional[str] = None,
    p_capture_distribution_guide: Optional[str] = None,
    p_capture_param_prior: Optional[tuple] = None,
    p_capture_param_guide: Optional[tuple] = None,
    n_components: Optional[int] = None
) -> ModelConfig:
    """
    Create a ModelConfig with specified distribution configurations and priors.

    Parameters
    ----------
    base_model : str
        Type of model to use. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp"
        or their mixture variants with "_mix" suffix
    r_distribution_model : str, default="lognormal"
        Distribution type for dispersion parameter in model. Must be "gamma" or
        "lognormal"
    r_distribution_guide : str, default="lognormal"
        Distribution type for dispersion parameter in guide. Must be "gamma" or
        "lognormal"
    r_prior : tuple, default=(0, 1)
        Prior parameters for dispersion distribution. For gamma: (shape, rate),
        for lognormal: (mu, sigma)
    p_distribution_model : str, default="beta"
        Distribution type for success probability in model. Currently only
        "beta" supported
    p_distribution_guide : str, default="beta"
        Distribution type for success probability in guide. Currently only
        "beta" supported
    p_prior : tuple, default=(1, 1)
        Prior parameters (alpha, beta) for success probability Beta distribution
    gate_distribution_model : str, optional
        Distribution type for dropout gate in model. Required for ZINB models.
        Currently only "beta" supported
    gate_distribution_guide : str, optional
        Distribution type for dropout gate in guide. Required for ZINB models.
        Currently only "beta" supported
    gate_prior : tuple, optional
        Prior parameters (alpha, beta) for dropout gate Beta distribution.
        Required for ZINB models
    p_capture_distribution_model : str, optional
        Distribution type for capture probability in model. Required for VCP
        models. Currently only "beta" supported
    p_capture_distribution_guide : str, optional
        Distribution type for capture probability in guide. Required for VCP
        models. Currently only "beta" supported
    p_capture_prior : tuple, optional
        Prior parameters (alpha, beta) for capture probability Beta
        distribution. Required for VCP models
    n_components : int, optional
        Number of mixture components. Required for mixture model variants

    Returns
    -------
    ModelConfig
        Configuration object containing all model and guide distribution
        settings

    Raises
    ------
    ValueError
        If invalid distribution types are provided or required distributions are
        missing for specific model types
    """
    # Set dispersion distribution based on r_dist parameter
    if r_distribution_model == "gamma":
        r_model = dist.Gamma(*r_param_prior)
    elif r_distribution_model == "lognormal":
        r_model = dist.LogNormal(*r_param_prior)
    else:
        raise ValueError(f"Invalid dispersion distribution: {r_distribution_model}")

    if r_distribution_guide == "gamma":
        r_guide = dist.Gamma(*r_param_guide)
    elif r_distribution_guide == "lognormal":
        r_guide = dist.LogNormal(*r_param_guide)
    else:
        raise ValueError(f"Invalid dispersion distribution: {r_distribution_guide}")

    # Set success probability distribution
    if p_distribution_model == "beta":
        p_model = dist.Beta(*p_param_prior)
    else:
        raise ValueError(f"Invalid success probability distribution: {p_distribution_model}")

    if p_distribution_guide == "beta":
        p_guide = dist.Beta(*p_param_guide)
    else:
        raise ValueError(f"Invalid success probability distribution: {p_distribution_guide}")

    # Check if only one gate distribution is provided
    if bool(gate_distribution_model is not None) != bool(gate_distribution_guide is not None):
        raise ValueError("gate_distribution_model and gate_distribution_guide must both be provided or not provided")

    # Set gate distribution model
    if gate_distribution_model is not None:
        if gate_distribution_model == "beta":
            gate_model = dist.Beta(*gate_param_prior)
        else:
            raise ValueError(f"Invalid gate distribution: {gate_distribution_model}")
    else:
        gate_model = None

    # Set gate distribution guide
    if gate_distribution_guide is not None:
        if gate_distribution_guide == "beta":
            gate_guide = dist.Beta(*gate_param_guide)
        else:
            raise ValueError(f"Invalid gate distribution: {gate_distribution_guide}")
    else:
        gate_guide = None

    # Check if only one capture probability distribution is provided
    if bool(p_capture_distribution_model is not None) != bool(p_capture_distribution_guide is not None):
        raise ValueError("p_capture_distribution_model and p_capture_distribution_guide must both be provided or not provided")

    # Set capture probability distribution model
    if p_capture_distribution_model is not None:
        if p_capture_distribution_model == "beta":
            p_capture_model = dist.Beta(*p_capture_param_prior)
        else:
            raise ValueError(f"Invalid capture probability distribution: {p_capture_distribution_model}")
    else:
        p_capture_model = None

    # Set capture probability distribution guide
    if p_capture_distribution_guide is not None:
        if p_capture_distribution_guide == "beta":
            p_capture_guide = dist.Beta(*p_capture_param_guide)
        else:
            raise ValueError(f"Invalid capture probability distribution: {p_capture_distribution_guide}")
    else:
        p_capture_guide = None

    # Return model configuration
    return ModelConfig(
        base_model=base_model,
        p_distribution_model=p_model,
        p_distribution_guide=p_guide,
        p_param_prior=p_param_prior,
        p_param_guide=p_param_guide,
        r_distribution_model=r_model,
        r_distribution_guide=r_guide,
        r_param_prior=r_param_prior,
        r_param_guide=r_param_guide,
        gate_distribution_model=gate_model,
        gate_distribution_guide=gate_guide,
        gate_param_prior=gate_param_prior,
        gate_param_guide=gate_param_guide,
        p_capture_distribution_model=p_capture_model,
        p_capture_distribution_guide=p_capture_guide,
        p_capture_param_prior=p_capture_param_prior,
        p_capture_param_guide=p_capture_param_guide,
        n_components=n_components
    )

# ------------------------------------------------------------------------------
# Default Distributions
# ------------------------------------------------------------------------------

# Define default distributions
GAMMA_DEFAULT = (dist.Gamma(2, 0.1), (2, 0.1))
LOGNORMAL_DEFAULT = (dist.LogNormal(1, 1), (1, 1))
BETA_DEFAULT = (dist.Beta(1, 1), (1, 1))

# ------------------------------------------------------------------------------
# Default Model Configurations and Pre-configured Models
# ------------------------------------------------------------------------------

def get_default_model_config(
        base_model: str, 
        n_components: Optional[int] = None
    ) -> ModelConfig:
    """
    Get default model configuration for a given model type.
    
    This function returns a ModelConfig object with default distribution
    configurations for the specified model type. It supports NBDM, ZINB, NBVCP
    and ZINBVCP models with either gamma or lognormal distributions for the
    dispersion parameter.
    
    Parameters
    ----------
    base_model : str
        The base model type. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp"
    r_dist : str, default="gamma"
        Distribution type for dispersion parameter. Must be "gamma" or
        "lognormal"
    n_components : Optional[int], default=None
        Number of mixture components. If specified, creates mixture model
        variant
        
    Returns
    -------
    ModelConfig
        Configuration object with default distribution settings for specified
        model
        
    Raises
    ------
    ValueError
        If an unknown model type is provided
    """
    # Return config for Negative Binomial-Dirichlet Multinomial model
    if base_model == "nbdm":
        return ModelConfig(
            base_model=base_model,
            # Success probability distribution
            p_distribution_model=BETA_DEFAULT[0],  
            p_distribution_guide=BETA_DEFAULT[0],  
            p_param_prior=BETA_DEFAULT[1],  
            p_param_guide=BETA_DEFAULT[1],  
            # Dispersion parameter distribution
            r_distribution_model=GAMMA_DEFAULT[0],  
            r_distribution_guide=GAMMA_DEFAULT[0],  
            r_param_prior=GAMMA_DEFAULT[1], 
            r_param_guide=GAMMA_DEFAULT[1], 
            # Optional number of mixture components
            n_components=n_components
        )
    
    # Return config for Zero-Inflated Negative Binomial model
    elif base_model == "zinb":
        return ModelConfig(
            base_model=base_model,
            # Success probability distribution
            p_distribution_model=BETA_DEFAULT[0],  
            p_distribution_guide=BETA_DEFAULT[0],  
            p_param_prior=BETA_DEFAULT[1],  
            p_param_guide=BETA_DEFAULT[1],  
            # Dispersion parameter distribution
            r_distribution_model=GAMMA_DEFAULT[0],  
            r_distribution_guide=GAMMA_DEFAULT[0],  
            r_param_prior=GAMMA_DEFAULT[1], 
            r_param_guide=GAMMA_DEFAULT[1], 
            # Dropout gate distribution
            gate_distribution_model=BETA_DEFAULT[0],  
            gate_distribution_guide=BETA_DEFAULT[0],  
            gate_param_prior=BETA_DEFAULT[1],  
            gate_param_guide=BETA_DEFAULT[1],  
            # Optional number of mixture components
            n_components=n_components
        )
    
    # Return config for Negative Binomial with Variable Capture Probability
    # model
    elif base_model == "nbvcp":
        return ModelConfig(
            base_model=base_model,
            # Dispersion parameter distribution
            r_distribution_model=GAMMA_DEFAULT[0],  
            r_distribution_guide=GAMMA_DEFAULT[0],  
            r_param_prior=GAMMA_DEFAULT[1],  
            r_param_guide=GAMMA_DEFAULT[1],  
            # Success probability distribution
            p_distribution_model=BETA_DEFAULT[0],  
            p_distribution_guide=BETA_DEFAULT[0],  
            p_param_prior=BETA_DEFAULT[1],  
            p_param_guide=BETA_DEFAULT[1],  
            # Capture probability distribution
            p_capture_distribution_model=BETA_DEFAULT[0],  
            p_capture_distribution_guide=BETA_DEFAULT[0],  
            p_capture_param_prior=BETA_DEFAULT[1],  
            p_capture_param_guide=BETA_DEFAULT[1],  
            # Optional number of mixture components
            n_components=n_components
        )
    
    # Return config for Zero-Inflated Negative Binomial with Variable Capture
    # Probability model
    elif base_model == "zinbvcp":
        return ModelConfig(
            base_model=base_model,
            # Dispersion parameter distribution
            r_distribution_model=GAMMA_DEFAULT[0],  
            r_distribution_guide=GAMMA_DEFAULT[0],  
            r_param_prior=GAMMA_DEFAULT[1],  
            r_param_guide=GAMMA_DEFAULT[1],  
            # Success probability distribution
            p_distribution_model=BETA_DEFAULT[0],  
            p_distribution_guide=BETA_DEFAULT[0],  
            p_param_prior=BETA_DEFAULT[1],  
            p_param_guide=BETA_DEFAULT[1],  
            # Dropout gate distribution
            gate_distribution_model=BETA_DEFAULT[0],  
            gate_distribution_guide=BETA_DEFAULT[0],  
            gate_param_prior=BETA_DEFAULT[1],  
            gate_param_guide=BETA_DEFAULT[1],  
            # Capture probability distribution
            p_capture_distribution_model=BETA_DEFAULT[0],  
            p_capture_distribution_guide=BETA_DEFAULT[0],  
            p_capture_param_prior=BETA_DEFAULT[1],  
            p_capture_param_guide=BETA_DEFAULT[1],  
            # Optional number of mixture components
            n_components=n_components
        )
    
    # Raise error for unknown model types
    else:
        raise ValueError(f"Unknown model type: {base_model}")