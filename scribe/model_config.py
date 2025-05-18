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

    mixing_logits_unconstrained_loc : Optional[float], default=None
        Location parameter for normal prior on mixture weights (pre-softmax)
    mixing_logits_unconstrained_scale : Optional[float], default=None
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
    mixing_logits_unconstrained_loc: Optional[float] = None
    mixing_logits_unconstrained_scale: Optional[float] = None

    # Number of mixture components
    n_components: Optional[int] = None
    
    def validate(self):
        """Validate unconstrained configuration parameters."""
        # Validate mixture components
        if self.is_mixture_model():
            if not self.base_model.endswith('_mix'):
                self.base_model = f"{self.base_model}_mix"
                
            if (self.mixing_logits_unconstrained_loc is None or 
                self.mixing_logits_unconstrained_scale is None):
                self.mixing_logits_unconstrained_loc = 0.0
                self.mixing_logits_unconstrained_scale = 1.0
        
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
