"""
Distribution configurations for SCRIBE.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import jax.numpy as jnp
import numpyro.distributions as dist

# -----------------------------------------------------------------------------
# Abstract Base Class
# -----------------------------------------------------------------------------

@dataclass
class AbstractModelConfig(ABC):
    """
    Abstract base class for SCRIBE model configurations.
    
    This class defines the common interface and shared attributes for both
    constrained and unconstrained model configurations.
    """
    # Common attributes for all model types
    base_model: str
    
    @abstractmethod
    def validate(self):
        """Validate configuration parameters."""
        pass
    
    def is_mixture_model(self) -> bool:
        """Check if this is a mixture model configuration."""
        return self.n_components is not None and self.n_components > 1
    
    def is_zero_inflated(self) -> bool:
        """Check if this is a zero-inflated model configuration."""
        return "zinb" in self.base_model
    
    def uses_variable_capture(self) -> bool:
        """Check if this model uses variable capture probability."""
        return "vcp" in self.base_model

# -----------------------------------------------------------------------------
# Constrained Configuration (primarily for SVI)
# -----------------------------------------------------------------------------

@dataclass
class ConstrainedModelConfig(AbstractModelConfig):
    """
    Configuration class for constrained parameterization used in Stochastic
    Variational Inference (SVI).
    
    This class defines the distributions and their parameters for the
    constrained parameterization of SCRIBE models. The constrained
    parameterization uses distributions in their natural parameter space:
        - Beta distributions for probabilities (p, gate, p_capture) in [0,1]
        - Gamma or LogNormal distributions for dispersion (r) in (0,∞)
        - Dirichlet distributions for mixture weights on the simplex
    
    Attributes
    ----------
    r_distribution_model : dist.Distribution
        Distribution object for dispersion parameter r in model (Gamma or
        LogNormal)
    r_distribution_guide : dist.Distribution  
        Distribution object for dispersion parameter r in guide (Gamma or
        LogNormal)
    r_param_prior : tuple
        Parameters for r distribution in model (shape,rate) for Gamma or
        (mu,sigma) for LogNormal
    r_param_guide : tuple
        Parameters for r distribution in guide (shape,rate) for Gamma or
        (mu,sigma) for LogNormal
        
    p_distribution_model : dist.Distribution
        Beta distribution object for success probability p in model
    p_distribution_guide : dist.Distribution
        Beta distribution object for success probability p in guide  
    p_param_prior : tuple
        Parameters (alpha,beta) for p Beta distribution in model
    p_param_guide : tuple
        Parameters (alpha,beta) for p Beta distribution in guide
        
    gate_distribution_model : Optional[dist.Distribution]
        Beta distribution object for dropout gate in zero-inflated models
    gate_distribution_guide : Optional[dist.Distribution]
        Beta distribution object for dropout gate in guide
    gate_param_prior : Optional[tuple]
        Parameters (alpha,beta) for gate Beta distribution in model
    gate_param_guide : Optional[tuple]
        Parameters (alpha,beta) for gate Beta distribution in guide
        
    p_capture_distribution_model : Optional[dist.Distribution]
        Beta distribution object for capture probability in variable capture
        models
    p_capture_distribution_guide : Optional[dist.Distribution]
        Beta distribution object for capture probability in guide
    p_capture_param_prior : Optional[tuple]
        Parameters (alpha,beta) for capture probability Beta distribution in
        model
    p_capture_param_guide : Optional[tuple]
        Parameters (alpha,beta) for capture probability Beta distribution in
        guide
        
    mixing_distribution_model : Optional[dist.Distribution]
        Dirichlet distribution object for mixture weights in mixture models
    mixing_distribution_guide : Optional[dist.Distribution]
        Dirichlet distribution object for mixture weights in guide
    mixing_param_prior : Optional[tuple]
        Concentration parameters for mixing Dirichlet distribution in model
    mixing_param_guide : Optional[tuple]
        Concentration parameters for mixing Dirichlet distribution in guide
    """
    # Dispersion parameter distributions
    r_distribution_model: dist.Distribution
    r_distribution_guide: dist.Distribution
    r_param_prior: tuple
    r_param_guide: tuple
    
    # Success probability distributions
    p_distribution_model: dist.Distribution
    p_distribution_guide: dist.Distribution
    p_param_prior: tuple
    p_param_guide: tuple
    
    # Optional: Zero-inflation gate distributions
    gate_distribution_model: Optional[dist.Distribution] = None
    gate_distribution_guide: Optional[dist.Distribution] = None
    gate_param_prior: Optional[tuple] = None
    gate_param_guide: Optional[tuple] = None
    
    # Optional: Capture probability distributions
    p_capture_distribution_model: Optional[dist.Distribution] = None
    p_capture_distribution_guide: Optional[dist.Distribution] = None
    p_capture_param_prior: Optional[tuple] = None
    p_capture_param_guide: Optional[tuple] = None
    
    # Optional: Mixture model distributions
    mixing_distribution_model: Optional[dist.Distribution] = None
    mixing_distribution_guide: Optional[dist.Distribution] = None
    mixing_param_prior: Optional[tuple] = None
    mixing_param_guide: Optional[tuple] = None

    # Number of mixture components
    n_components: Optional[int] = None
    
    def validate(self):
        """Validate constrained configuration parameters."""
        # Validate mixture components
        if self.is_mixture_model():
            if not self.base_model.endswith('_mix'):
                self.base_model = f"{self.base_model}_mix"
                
            if (self.mixing_distribution_model is None or 
                self.mixing_distribution_guide is None):
                raise ValueError("Mixture models require mixing distributions")
        
        # Validate zero-inflation parameters
        if self.is_zero_inflated():
            if (self.gate_distribution_model is None or 
                self.gate_distribution_guide is None):
                raise ValueError("ZINB models require gate distributions")
        elif (self.gate_distribution_model is not None or 
              self.gate_distribution_guide is not None):
            raise ValueError("Non-ZINB models should not have gate distributions")
        
        # Validate variable capture parameters
        if self.uses_variable_capture():
            if (self.p_capture_distribution_model is None or 
                self.p_capture_distribution_guide is None):
                raise ValueError("VCP models require capture probability distributions")
        elif (self.p_capture_distribution_model is not None or 
              self.p_capture_distribution_guide is not None):
            raise ValueError("Non-VCP models should not have capture probability distributions")

# -----------------------------------------------------------------------------
# Unconstrained Configuration (for MCMC)
# -----------------------------------------------------------------------------

@dataclass
class UnconstrainedModelConfig(AbstractModelConfig):
    """
    Configuration class for unconstrained parameterization used in MCMC.
    
    This class handles model parameters represented in unconstrained space (real
    line) suitable for efficient MCMC sampling. Parameters are transformed from
    their natural constrained space using:
        - logit transform for probabilities (p, gate, p_capture) mapping [0,1]
          to (-∞,∞)
        - log transform for positive reals (r) mapping (0,∞) to (-∞,∞)
        - softmax for mixture weights mapping simplex to (-∞,∞)^K

    Attributes
    ----------
    p_unconstrained_loc : float, default=0.0
        Location parameter for logit-normal prior on success probability
    p_unconstrained_scale : float, default=1.0
        Scale parameter for logit-normal prior on success probability

    r_unconstrained_loc : float, default=0.0
        Location parameter for log-normal prior on dispersion
    r_unconstrained_scale : float, default=1.0
        Scale parameter for log-normal prior on dispersion

    gate_unconstrained_loc : Optional[float], default=None
        Location parameter for logit-normal prior on dropout gate
    gate_unconstrained_scale : Optional[float], default=None
        Scale parameter for logit-normal prior on dropout gate

    p_capture_unconstrained_loc : Optional[float], default=None
        Location parameter for logit-normal prior on capture probability
    p_capture_unconstrained_scale : Optional[float], default=None
        Scale parameter for logit-normal prior on capture probability

    mixing_unconstrained_loc : Optional[float], default=None
        Location parameter for normal prior on mixture weights (pre-softmax)
    mixing_unconstrained_scale : Optional[float], default=None
        Scale parameter for normal prior on mixture weights (pre-softmax)
    """
    # Unconstrained p parameters (logit-normal)
    p_unconstrained_loc: float = 0.0
    p_unconstrained_scale: float = 1.0
    
    # Unconstrained r parameters (log-normal)
    r_unconstrained_loc: float = 0.0
    r_unconstrained_scale: float = 1.0
    
    # Optional: Unconstrained gate parameters (logit-normal)
    gate_unconstrained_loc: Optional[float] = None
    gate_unconstrained_scale: Optional[float] = None
    
    # Optional: Unconstrained p_capture parameters (logit-normal)
    p_capture_unconstrained_loc: Optional[float] = None
    p_capture_unconstrained_scale: Optional[float] = None
    
    # Optional: Unconstrained mixing parameters (normal with softmax)
    mixing_unconstrained_loc: Optional[float] = None
    mixing_unconstrained_scale: Optional[float] = None

    # Number of mixture components
    n_components: Optional[int] = None
    
    def validate(self):
        """Validate unconstrained configuration parameters."""
        # Validate mixture components
        if self.is_mixture_model():
            if not self.base_model.endswith('_mix'):
                self.base_model = f"{self.base_model}_mix"
                
            if (self.mixing_unconstrained_loc is None or 
                self.mixing_unconstrained_scale is None):
                self.mixing_unconstrained_loc = 0.0
                self.mixing_unconstrained_scale = 1.0
        
        # Validate zero-inflation parameters
        if self.is_zero_inflated():
            if (self.gate_unconstrained_loc is None or 
                self.gate_unconstrained_scale is None):
                self.gate_unconstrained_loc = 0.0
                self.gate_unconstrained_scale = 1.0
        
        # Validate variable capture parameters
        if self.uses_variable_capture():
            if (self.p_capture_unconstrained_loc is None or 
                self.p_capture_unconstrained_scale is None):
                self.p_capture_unconstrained_loc = 0.0
                self.p_capture_unconstrained_scale = 1.0

# # ------------------------------------------------------------------------------
# # Model Configurations
# # ------------------------------------------------------------------------------

# # Add these as static methods to AbstractModelConfig

# @staticmethod
# def create_constrained(
#     base_model: str,
#     r_distribution_model: dist.Distribution,
#     p_distribution_model: dist.Distribution,
#     r_param_prior: tuple,
#     p_param_prior: tuple,
#     n_components: Optional[int] = None,
#     **kwargs
# ) -> 'ConstrainedModelConfig':
#     """Create a constrained model configuration."""
#     config = ConstrainedModelConfig(
#         base_model=base_model,
#         r_distribution_model=r_distribution_model,
#         r_distribution_guide=r_distribution_model,  # Default: same as model
#         p_distribution_model=p_distribution_model,
#         p_distribution_guide=p_distribution_model,  # Default: same as model
#         r_param_prior=r_param_prior,
#         r_param_guide=r_param_prior,  # Default: same as prior
#         p_param_prior=p_param_prior,
#         p_param_guide=p_param_prior,  # Default: same as prior
#         n_components=n_components,
#         **kwargs
#     )
#     config.validate()
#     return config

# # ------------------------------------------------------------------------------

# @staticmethod
# def create_unconstrained(
#     base_model: str,
#     n_components: Optional[int] = None,
#     p_unconstrained_loc: float = 0.0,
#     p_unconstrained_scale: float = 1.0,
#     r_unconstrained_loc: float = 0.0,
#     r_unconstrained_scale: float = 1.0,
#     **kwargs
# ) -> 'UnconstrainedModelConfig':
#     """Create an unconstrained model configuration."""
#     config = UnconstrainedModelConfig(
#         base_model=base_model,
#         n_components=n_components,
#         p_unconstrained_loc=p_unconstrained_loc,
#         p_unconstrained_scale=p_unconstrained_scale,
#         r_unconstrained_loc=r_unconstrained_loc,
#         r_unconstrained_scale=r_unconstrained_scale,
#         **kwargs
#     )
#     config.validate()
#     return config

# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------
# # Model Configurations
# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------

# def get_model_config(
#     base_model: str,
#     p_distribution_model: str = "beta",
#     p_distribution_guide: str = "beta",
#     p_param_prior: tuple = (1, 1),
#     p_param_guide: tuple = (1, 1),
#     r_distribution_model: str = "lognormal",
#     r_distribution_guide: str = "lognormal", 
#     r_param_prior: tuple = (0, 1),
#     r_param_guide: tuple = (0, 1),
#     gate_distribution_model: Optional[str] = None,
#     gate_distribution_guide: Optional[str] = None,
#     gate_param_prior: Optional[tuple] = None,
#     gate_param_guide: Optional[tuple] = None,
#     p_capture_distribution_model: Optional[str] = None,
#     p_capture_distribution_guide: Optional[str] = None,
#     p_capture_param_prior: Optional[tuple] = None,
#     p_capture_param_guide: Optional[tuple] = None,
#     n_components: Optional[int] = None,
#     mixing_distribution_model: Optional[str] = None,
#     mixing_distribution_guide: Optional[str] = None,
#     mixing_param_prior: Optional[tuple] = None,
#     mixing_param_guide: Optional[tuple] = None
# ) -> ModelConfig:
#     """
#     Create a ModelConfig with specified distribution configurations and priors.

#     Parameters
#     ----------
#     base_model : str
#         Type of model to use. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp"
#         or their mixture variants with "_mix" suffix
#     r_distribution_model : str, default="lognormal"
#         Distribution type for dispersion parameter in model. Must be "gamma" or
#         "lognormal"
#     r_distribution_guide : str, default="lognormal"
#         Distribution type for dispersion parameter in guide. Must be "gamma" or
#         "lognormal"
#     r_prior : tuple, default=(0, 1)
#         Prior parameters for dispersion distribution. For gamma: (shape, rate),
#         for lognormal: (mu, sigma)
#     p_distribution_model : str, default="beta"
#         Distribution type for success probability in model. Currently only
#         "beta" supported
#     p_distribution_guide : str, default="beta"
#         Distribution type for success probability in guide. Currently only
#         "beta" supported
#     p_prior : tuple, default=(1, 1)
#         Prior parameters (alpha, beta) for success probability Beta distribution
#     gate_distribution_model : str, optional
#         Distribution type for dropout gate in model. Required for ZINB models.
#         Currently only "beta" supported
#     gate_distribution_guide : str, optional
#         Distribution type for dropout gate in guide. Required for ZINB models.
#         Currently only "beta" supported
#     gate_prior : tuple, optional
#         Prior parameters (alpha, beta) for dropout gate Beta distribution.
#         Required for ZINB models
#     p_capture_distribution_model : str, optional
#         Distribution type for capture probability in model. Required for VCP
#         models. Currently only "beta" supported
#     p_capture_distribution_guide : str, optional
#         Distribution type for capture probability in guide. Required for VCP
#         models. Currently only "beta" supported
#     p_capture_prior : tuple, optional
#         Prior parameters (alpha, beta) for capture probability Beta
#         distribution. Required for VCP models
#     n_components : int, optional
#         Number of mixture components. Required for mixture model variants

#     Returns
#     -------
#     ModelConfig
#         Configuration object containing all model and guide distribution
#         settings

#     Raises
#     ------
#     ValueError
#         If invalid distribution types are provided or required distributions are
#         missing for specific model types
#     """
#     # Set dispersion distribution based on r_dist parameter
#     if r_distribution_model == "gamma":
#         r_model = dist.Gamma(*r_param_prior)
#     elif r_distribution_model == "lognormal":
#         r_model = dist.LogNormal(*r_param_prior)
#     else:
#         raise ValueError(f"Invalid dispersion distribution: {r_distribution_model}")

#     if r_distribution_guide == "gamma":
#         r_guide = dist.Gamma(*r_param_guide)
#     elif r_distribution_guide == "lognormal":
#         r_guide = dist.LogNormal(*r_param_guide)
#     else:
#         raise ValueError(f"Invalid dispersion distribution: {r_distribution_guide}")

#     # Set success probability distribution
#     if p_distribution_model == "beta":
#         p_model = dist.Beta(*p_param_prior)
#     else:
#         raise ValueError(f"Invalid success probability distribution: {p_distribution_model}")

#     if p_distribution_guide == "beta":
#         p_guide = dist.Beta(*p_param_guide)
#     else:
#         raise ValueError(f"Invalid success probability distribution: {p_distribution_guide}")

#     # Check if only one gate distribution is provided
#     if bool(gate_distribution_model is not None) != bool(gate_distribution_guide is not None):
#         raise ValueError("gate_distribution_model and gate_distribution_guide must both be provided or not provided")

#     # Set gate distribution model
#     if gate_distribution_model is not None:
#         if gate_distribution_model == "beta":
#             gate_model = dist.Beta(*gate_param_prior)
#         else:
#             raise ValueError(f"Invalid gate distribution: {gate_distribution_model}")
#     else:
#         gate_model = None

#     # Set gate distribution guide
#     if gate_distribution_guide is not None:
#         if gate_distribution_guide == "beta":
#             gate_guide = dist.Beta(*gate_param_guide)
#         else:
#             raise ValueError(f"Invalid gate distribution: {gate_distribution_guide}")
#     else:
#         gate_guide = None

#     # Check if only one capture probability distribution is provided
#     if bool(p_capture_distribution_model is not None) != bool(p_capture_distribution_guide is not None):
#         raise ValueError("p_capture_distribution_model and p_capture_distribution_guide must both be provided or not provided")

#     # Set capture probability distribution model
#     if p_capture_distribution_model is not None:
#         if p_capture_distribution_model == "beta":
#             p_capture_model = dist.Beta(*p_capture_param_prior)
#         else:
#             raise ValueError(f"Invalid capture probability distribution: {p_capture_distribution_model}")
#     else:
#         p_capture_model = None

#     # Set capture probability distribution guide
#     if p_capture_distribution_guide is not None:
#         if p_capture_distribution_guide == "beta":
#             p_capture_guide = dist.Beta(*p_capture_param_guide)
#         else:
#             raise ValueError(f"Invalid capture probability distribution: {p_capture_distribution_guide}")
#     else:
#         p_capture_guide = None

#     # Set mixing distribution model
#     if mixing_distribution_model is not None:
#         if mixing_distribution_model == "dirichlet":
#             mixing_model = dist.Dirichlet(*mixing_param_prior)
#         else:
#             raise ValueError(f"Invalid mixing distribution: {mixing_distribution_model}")
#     else:
#         mixing_model = None

#     # Set mixing distribution guide
#     if mixing_distribution_guide is not None:
#         if mixing_distribution_guide == "dirichlet":
#             mixing_guide = dist.Dirichlet(*mixing_param_guide)
#         else:
#             raise ValueError(f"Invalid mixing distribution: {mixing_distribution_guide}")
#     else:
#         mixing_guide = None

#     # Return model configuration
#     return ModelConfig(
#         base_model=base_model,
#         p_distribution_model=p_model,
#         p_distribution_guide=p_guide,
#         p_param_prior=p_param_prior,
#         p_param_guide=p_param_guide,
#         r_distribution_model=r_model,
#         r_distribution_guide=r_guide,
#         r_param_prior=r_param_prior,
#         r_param_guide=r_param_guide,
#         gate_distribution_model=gate_model,
#         gate_distribution_guide=gate_guide,
#         gate_param_prior=gate_param_prior,
#         gate_param_guide=gate_param_guide,
#         p_capture_distribution_model=p_capture_model,
#         p_capture_distribution_guide=p_capture_guide,
#         p_capture_param_prior=p_capture_param_prior,
#         p_capture_param_guide=p_capture_param_guide,
#         n_components=n_components,
#         mixing_distribution_model=mixing_model,
#         mixing_distribution_guide=mixing_guide,
#         mixing_param_prior=mixing_param_prior,
#         mixing_param_guide=mixing_param_guide
#     )

# # ------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------

# # In model_config.py

# def get_model_config_with_unconstrained(
#     base_model: str,
#     # Constrained parameters
#     p_distribution_model: str = "beta",
#     p_distribution_guide: str = "beta",
#     p_param_prior: tuple = (1, 1),
#     p_param_guide: tuple = (1, 1),
#     r_distribution_model: str = "gamma",
#     r_distribution_guide: str = "gamma", 
#     r_param_prior: tuple = (2, 0.1),
#     r_param_guide: tuple = (2, 0.1),
#     gate_distribution_model: Optional[str] = None,
#     gate_distribution_guide: Optional[str] = None,
#     gate_param_prior: Optional[tuple] = None,
#     gate_param_guide: Optional[tuple] = None,
#     p_capture_distribution_model: Optional[str] = None,
#     p_capture_distribution_guide: Optional[str] = None,
#     p_capture_param_prior: Optional[tuple] = None,
#     p_capture_param_guide: Optional[tuple] = None,
#     n_components: Optional[int] = None,
#     mixing_distribution_model: Optional[str] = None,
#     mixing_distribution_guide: Optional[str] = None,
#     mixing_param_prior: Optional[tuple] = None,
#     mixing_param_guide: Optional[tuple] = None,
#     # Unconstrained parameters
#     p_unconstrained_loc: float = 0.0,
#     p_unconstrained_scale: float = 1.0,
#     r_unconstrained_loc: float = 0.0,
#     r_unconstrained_scale: float = 1.0,
#     gate_unconstrained_loc: Optional[float] = None,
#     gate_unconstrained_scale: Optional[float] = None,
#     p_capture_unconstrained_loc: Optional[float] = None,
#     p_capture_unconstrained_scale: Optional[float] = None,
#     mixing_unconstrained_loc: Optional[float] = None,
#     mixing_unconstrained_scale: Optional[float] = None,
# ) -> ModelConfig:
#     """
#     Create a ModelConfig with both constrained and unconstrained
#     parameterizations.
    
#     This function extends get_model_config to also include unconstrained
#     parameters for MCMC sampling.
    
#     Parameters
#     ----------
#     base_model : str
#         Type of model to use. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp"
#         or their mixture variants with "_mix" suffix
#     p_distribution_model : str, default="beta"
#         Distribution type for success probability in model. Currently only
#         "beta" supported
#     p_distribution_guide : str, default="beta"
#         Distribution type for success probability in guide. Currently only
#         "beta" supported
#     p_param_prior : tuple, default=(1, 1)
#         Prior parameters (alpha, beta) for success probability Beta distribution
#     p_param_guide : tuple, default=(1, 1)
#         Prior parameters (alpha, beta) for success probability Beta distribution
#     r_distribution_model : str, default="gamma"
#         Distribution type for dispersion parameter in model. Must be "gamma" or
#         "lognormal"
#     r_distribution_guide : str, default="gamma"
#         Distribution type for dispersion parameter in guide. Must be "gamma" or
#         "lognormal"
#     r_param_prior : tuple, default=(2, 0.1)
#         Prior parameters for dispersion distribution. For gamma: (shape, rate),
#         for lognormal: (mu, sigma)
#     r_param_guide : tuple, default=(2, 0.1)
#         Prior parameters for dispersion distribution. For gamma: (shape, rate),
#         for lognormal: (mu, sigma)
#     gate_distribution_model : str, optional
#         Distribution type for dropout gate in model. Required for ZINB models.
#         Currently only "beta" supported
#     gate_distribution_guide : str, optional
#         Distribution type for dropout gate in guide. Required for ZINB models.
#         Currently only "beta" supported
#     gate_param_prior : tuple, optional
#         Prior parameters (alpha, beta) for dropout gate Beta distribution.
#         Required for ZINB models
#     gate_param_guide : tuple, optional
#         Prior parameters (alpha, beta) for dropout gate Beta distribution.
#         Required for ZINB models
#     p_capture_distribution_model : str, optional
#         Distribution type for capture probability in model. Required for VCP
#         models. Currently only "beta" supported
#     p_capture_distribution_guide : str, optional
#         Distribution type for capture probability in guide. Required for VCP
#         models. Currently only "beta" supported
#     p_capture_param_prior : tuple, optional
#         Prior parameters (alpha, beta) for capture probability Beta
#         distribution. Required for VCP models
#     p_capture_param_guide : tuple, optional
#         Prior parameters (alpha, beta) for capture probability Beta
#         distribution. Required for VCP models
#     n_components : int, optional
#         Number of mixture components. Required for mixture model variants
#     mixing_distribution_model : str, optional
#         Distribution type for mixing weights in model. Required for mixture
#         models. Currently only "dirichlet" supported
#     mixing_distribution_guide : str, optional
#         Distribution type for mixing weights in guide. Required for mixture
#         models. Currently only "dirichlet" supported
#     mixing_param_prior : tuple, optional
#         Prior parameters for mixing weights distribution. Required for mixture
#         models.
#     mixing_param_guide : tuple, optional
#         Prior parameters for mixing weights distribution. Required for mixture
#         models.
#     p_unconstrained_loc : float, default=0.0
#         Location parameter for logit-normal prior on p in unconstrained space
#     p_unconstrained_scale : float, default=1.0
#         Scale parameter for logit-normal prior on p in unconstrained space
#     r_unconstrained_loc : float, default=0.0
#         Location parameter for log-normal prior on r in unconstrained space
#     r_unconstrained_scale : float, default=1.0
#         Scale parameter for log-normal prior on r in unconstrained space
#     gate_unconstrained_loc : float, optional
#         Location parameter for logit-normal prior on gate in unconstrained
#         space. Required for ZINB models.
#     gate_unconstrained_scale : float, optional
#         Scale parameter for logit-normal prior on gate in unconstrained space.
#         Required for ZINB models.
#     p_capture_unconstrained_loc : float, optional
#         Location parameter for logit-normal prior on p_capture in unconstrained
#         space. Required for VCP models.
#     p_capture_unconstrained_scale : float, optional
#         Scale parameter for logit-normal prior on p_capture in unconstrained
#         space. Required for VCP models.
#     mixing_unconstrained_loc : float, optional
#         Location parameter for normal prior on mixing weights in unconstrained
#         space. Required for mixture models.
#     mixing_unconstrained_scale : float, optional
#         Scale parameter for normal prior on mixing weights in unconstrained
#         space. Required for mixture models.

#     Returns
#     -------
#     ModelConfig
#         Configuration object containing all model and guide distribution
#         settings, including both constrained and unconstrained parameterizations

#     Raises
#     ------
#     ValueError
#         If invalid distribution types are provided or required distributions are
#         missing for specific model types
#     """
#     # Create the basic ModelConfig with constrained parameterization
#     config = get_model_config(
#         base_model=base_model,
#         p_distribution_model=p_distribution_model,
#         p_distribution_guide=p_distribution_guide,
#         p_param_prior=p_param_prior,
#         p_param_guide=p_param_guide,
#         r_distribution_model=r_distribution_model,
#         r_distribution_guide=r_distribution_guide,
#         r_param_prior=r_param_prior,
#         r_param_guide=r_param_guide,
#         gate_distribution_model=gate_distribution_model,
#         gate_distribution_guide=gate_distribution_guide,
#         gate_param_prior=gate_param_prior,
#         gate_param_guide=gate_param_guide,
#         p_capture_distribution_model=p_capture_distribution_model,
#         p_capture_distribution_guide=p_capture_distribution_guide,
#         p_capture_param_prior=p_capture_param_prior,
#         p_capture_param_guide=p_capture_param_guide,
#         n_components=n_components,
#         mixing_distribution_model=mixing_distribution_model,
#         mixing_distribution_guide=mixing_distribution_guide,
#         mixing_param_prior=mixing_param_prior,
#         mixing_param_guide=mixing_param_guide
#     )
    
#     # Add unconstrained parameters
#     config.p_unconstrained_loc = p_unconstrained_loc
#     config.p_unconstrained_scale = p_unconstrained_scale
#     config.r_unconstrained_loc = r_unconstrained_loc
#     config.r_unconstrained_scale = r_unconstrained_scale
    
#     # Set conditional unconstrained parameters based on model type
#     if "zinb" in base_model:
#         config.gate_unconstrained_loc = gate_unconstrained_loc or 0.0
#         config.gate_unconstrained_scale = gate_unconstrained_scale or 1.0
    
#     if "vcp" in base_model:
#         config.p_capture_unconstrained_loc = p_capture_unconstrained_loc or 0.0
#         config.p_capture_unconstrained_scale = p_capture_unconstrained_scale or 1.0
    
#     if n_components is not None and n_components > 1:
#         config.mixing_unconstrained_loc = mixing_unconstrained_loc or 0.0
#         config.mixing_unconstrained_scale = mixing_unconstrained_scale or 1.0
    
#     return config


# # ------------------------------------------------------------------------------
# # Default Distributions
# # ------------------------------------------------------------------------------

# # Define default distributions
# GAMMA_DEFAULT = (dist.Gamma(2, 0.1), (2, 0.1))
# LOGNORMAL_DEFAULT = (dist.LogNormal(1, 1), (1, 1))
# BETA_DEFAULT = (dist.Beta(1, 1), (1, 1))

# # ------------------------------------------------------------------------------
# # Default Model Configurations and Pre-configured Models
# # ------------------------------------------------------------------------------

# def get_default_model_config(
#         base_model: str, 
#         n_components: Optional[int] = None
#     ) -> ModelConfig:
#     """
#     Get default model configuration for a given model type.
    
#     This function returns a ModelConfig object with default distribution
#     configurations for the specified model type. It supports NBDM, ZINB, NBVCP
#     and ZINBVCP models with either gamma or lognormal distributions for the
#     dispersion parameter.
    
#     Parameters
#     ----------
#     base_model : str
#         The base model type. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp"
#     r_dist : str, default="gamma"
#         Distribution type for dispersion parameter. Must be "gamma" or
#         "lognormal"
#     n_components : Optional[int], default=None
#         Number of mixture components. If specified, creates mixture model
#         variant
        
#     Returns
#     -------
#     ModelConfig
#         Configuration object with default distribution settings for specified
#         model
        
#     Raises
#     ------
#     ValueError
#         If an unknown model type is provided
#     """
#     # Return config for Negative Binomial-Dirichlet Multinomial model
#     if base_model == "nbdm":
#         return ModelConfig(
#             base_model=base_model,
#             # Success probability distribution
#             p_distribution_model=BETA_DEFAULT[0],  
#             p_distribution_guide=BETA_DEFAULT[0],  
#             p_param_prior=BETA_DEFAULT[1],  
#             p_param_guide=BETA_DEFAULT[1],  
#             # Dispersion parameter distribution
#             r_distribution_model=GAMMA_DEFAULT[0],  
#             r_distribution_guide=GAMMA_DEFAULT[0],  
#             r_param_prior=GAMMA_DEFAULT[1], 
#             r_param_guide=GAMMA_DEFAULT[1], 
#             # Optional number of mixture components
#             n_components=n_components
#         )
    
#     # Return config for Zero-Inflated Negative Binomial model
#     elif base_model == "zinb":
#         return ModelConfig(
#             base_model=base_model,
#             # Success probability distribution
#             p_distribution_model=BETA_DEFAULT[0],  
#             p_distribution_guide=BETA_DEFAULT[0],  
#             p_param_prior=BETA_DEFAULT[1],  
#             p_param_guide=BETA_DEFAULT[1],  
#             # Dispersion parameter distribution
#             r_distribution_model=GAMMA_DEFAULT[0],  
#             r_distribution_guide=GAMMA_DEFAULT[0],  
#             r_param_prior=GAMMA_DEFAULT[1], 
#             r_param_guide=GAMMA_DEFAULT[1], 
#             # Dropout gate distribution
#             gate_distribution_model=BETA_DEFAULT[0],  
#             gate_distribution_guide=BETA_DEFAULT[0],  
#             gate_param_prior=BETA_DEFAULT[1],  
#             gate_param_guide=BETA_DEFAULT[1],  
#             # Optional number of mixture components
#             n_components=n_components
#         )
    
#     # Return config for Negative Binomial with Variable Capture Probability
#     # model
#     elif base_model == "nbvcp":
#         return ModelConfig(
#             base_model=base_model,
#             # Dispersion parameter distribution
#             r_distribution_model=GAMMA_DEFAULT[0],  
#             r_distribution_guide=GAMMA_DEFAULT[0],  
#             r_param_prior=GAMMA_DEFAULT[1],  
#             r_param_guide=GAMMA_DEFAULT[1],  
#             # Success probability distribution
#             p_distribution_model=BETA_DEFAULT[0],  
#             p_distribution_guide=BETA_DEFAULT[0],  
#             p_param_prior=BETA_DEFAULT[1],  
#             p_param_guide=BETA_DEFAULT[1],  
#             # Capture probability distribution
#             p_capture_distribution_model=BETA_DEFAULT[0],  
#             p_capture_distribution_guide=BETA_DEFAULT[0],  
#             p_capture_param_prior=BETA_DEFAULT[1],  
#             p_capture_param_guide=BETA_DEFAULT[1],  
#             # Optional number of mixture components
#             n_components=n_components
#         )
    
#     # Return config for Zero-Inflated Negative Binomial with Variable Capture
#     # Probability model
#     elif base_model == "zinbvcp":
#         return ModelConfig(
#             base_model=base_model,
#             # Dispersion parameter distribution
#             r_distribution_model=GAMMA_DEFAULT[0],  
#             r_distribution_guide=GAMMA_DEFAULT[0],  
#             r_param_prior=GAMMA_DEFAULT[1],  
#             r_param_guide=GAMMA_DEFAULT[1],  
#             # Success probability distribution
#             p_distribution_model=BETA_DEFAULT[0],  
#             p_distribution_guide=BETA_DEFAULT[0],  
#             p_param_prior=BETA_DEFAULT[1],  
#             p_param_guide=BETA_DEFAULT[1],  
#             # Dropout gate distribution
#             gate_distribution_model=BETA_DEFAULT[0],  
#             gate_distribution_guide=BETA_DEFAULT[0],  
#             gate_param_prior=BETA_DEFAULT[1],  
#             gate_param_guide=BETA_DEFAULT[1],  
#             # Capture probability distribution
#             p_capture_distribution_model=BETA_DEFAULT[0],  
#             p_capture_distribution_guide=BETA_DEFAULT[0],  
#             p_capture_param_prior=BETA_DEFAULT[1],  
#             p_capture_param_guide=BETA_DEFAULT[1],  
#             # Optional number of mixture components
#             n_components=n_components
#         )
    
#     # Raise error for unknown model types
#     else:
#         raise ValueError(f"Unknown model type: {base_model}")