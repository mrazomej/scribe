"""
Distribution configurations for SCRIBE.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints
from dataclasses import dataclass
import scipy.stats as stats

@dataclass
class DistributionConfig:
    """
    Configuration class for probability distributions used in SCRIBE models.
    
    This class provides a unified interface for working with different
    probability distributions across multiple backends (NumPyro and SciPy). It
    handles parameter initialization, distribution instantiation, and mode
    calculation.
    
    Parameters
    ----------
    dist_type : str
        The type of distribution (e.g. "gamma", "lognormal", "beta")
    param_names : List[str] 
        Names of the distribution parameters (e.g. ["alpha", "beta"])
    param_constraints : Dict[str, str]
        Constraints on parameter values (e.g. {"alpha": "positive"})
    init_values : Dict[str, float]
        Initial values for each parameter
    """
    # The type of distribution (gamma, lognormal, etc)
    dist_type: str
    
    # List of parameter names for this distribution
    param_names: List[str]
    
    # Dictionary mapping parameter names to their constraints
    param_constraints: Dict[str, constraints.Constraint]
    
    # Dictionary of default initialization values for parameters  
    init_values: Dict[str, float]
    
    # --------------------------------------------------------------------------

    def get_numpyro_dist(self, params: Dict[str, Any]) -> dist.Distribution:
        """
        Create a NumPyro distribution with the given parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter values
            
        Returns
        -------
        dist.Distribution
            The instantiated NumPyro distribution
        """
        # For gamma distribution, return Gamma with alpha and beta params
        if self.dist_type == "gamma":
            return dist.Gamma(
                params[self.param_names[0]], 
                params[self.param_names[1]]
            )
        # For lognormal, return LogNormal with mu and sigma params    
        elif self.dist_type == "lognormal":
            return dist.LogNormal(
                loc=params[self.param_names[0]], 
                scale=jnp.exp(params[self.param_names[1]])
            )
        # For beta, return Beta with alpha and beta params
        elif self.dist_type == "beta":
            return dist.Beta(
                concentration1=params[self.param_names[0]], 
                concentration0=params[self.param_names[1]]
            )
        # Raise error if distribution type is not recognized
        else:
            raise ValueError(f"Unregistered distribution type: {self.dist_type}")

    # --------------------------------------------------------------------------
            
    def get_scipy_dist(self, params: Dict[str, Any]) -> stats.rv_continuous:
        """
        Create a SciPy distribution with the given parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter values
            
        Returns
        -------
        stats.rv_continuous
            The instantiated SciPy distribution
        """
        # For gamma, return gamma with shape=alpha, scale=1/beta
        if self.dist_type == "gamma":
            return stats.gamma(
                params[self.param_names[0]], 
                loc=0, 
                scale=1/params[self.param_names[1]]
            )
        # For lognormal, return lognorm with s=sigma, scale=exp(mu)
        elif self.dist_type == "lognormal":
            return stats.lognorm(
                params[self.param_names[1]],
                loc=0,
                scale=jnp.exp(params[self.param_names[0]])
            )
        # For beta, return beta with a=alpha, b=beta
        elif self.dist_type == "beta":
            return stats.beta(
                params[self.param_names[0]], 
                params[self.param_names[1]]
            )
        # Raise error if distribution type is not recognized
        else:
            raise ValueError(f"Unregistered distribution type: {self.dist_type}")

    # --------------------------------------------------------------------------
            
    def get_mode(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Calculate the mode of the distribution with given parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of parameter values
            
        Returns
        -------
        jnp.ndarray
            The mode of the distribution
        """
        # Import mode calculation functions
        from .stats import beta_mode, gamma_mode, lognorm_mode
        
        # For gamma, calculate mode using gamma_mode function
        if self.dist_type == "gamma":
            return gamma_mode(
                params[self.param_names[0]], 
                params[self.param_names[1]]
            )
        # For lognormal, calculate mode using lognorm_mode function
        elif self.dist_type == "lognormal":
            return lognorm_mode(
                params[self.param_names[0]], 
                params[self.param_names[1]]
            )
        # For beta, calculate mode using beta_mode function
        elif self.dist_type == "beta":
            return beta_mode(
                params[self.param_names[0]], 
                params[self.param_names[1]]
            )
        # Raise error if distribution type is not recognized
        else:
            raise ValueError(f"Unregistered distribution type: {self.dist_type}")

    # --------------------------------------------------------------------------

    def get_param_dict(
            self, 
            var_name: str, 
            shape: Tuple[int, ...] = ()
        ) -> Dict[str, jnp.ndarray]:
        """
        Create a dictionary of initialized parameters for this distribution.
        
        Parameters
        ----------
        var_name : str
            Name of the variable this distribution represents
        shape : Tuple[int, ...], optional
            Shape of parameter arrays, defaults to ()
            
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to initialized arrays
        """
        # Create dictionary with parameter arrays initialized to default values
        return {
            f"{param_name}_{var_name}": jnp.ones(shape) * self.init_values[param_name]
            for param_name in self.param_names
        }

# ------------------------------------------------------------------------------
# Common distribution configurations
# ------------------------------------------------------------------------------

GAMMA_CONFIG = DistributionConfig(
    dist_type="gamma",
    param_names=["alpha", "beta"],
    param_constraints={
        "alpha": constraints.positive, 
        "beta": constraints.positive
    },
    init_values={"alpha": 2.0, "beta": 0.1}
)

# ------------------------------------------------------------------------------

LOGNORMAL_CONFIG = DistributionConfig(
    dist_type="lognormal", 
    param_names=["mu", "sigma"],
    param_constraints={
        "mu": constraints.real, 
        "sigma": constraints.positive
    },
    init_values={"mu": 0.0, "sigma": 1.0}
)

# ------------------------------------------------------------------------------

BETA_CONFIG = DistributionConfig(
    dist_type="beta",
    param_names=["alpha", "beta"],
    param_constraints={
        "alpha": constraints.positive, 
        "beta": constraints.positive
    },
    init_values={"alpha": 1.0, "beta": 1.0}
)

# ------------------------------------------------------------------------------
# Model configurations
# ------------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Configuration class for SCRIBE model parameters and distributions.
    
    This class handles configuration of model types, their required
    distributions, and validation of model parameters. It supports single and
    mixture model variants of NBDM, ZINB, NBVCP, and ZINBVCP models.
    
    Parameters
    ----------
    base_model : str
        Base model type. Must be one of: "nbdm", "zinb", "nbvcp", "zinbvcp", or
        their mixture variants ending in "_mix"
    r_distribution : DistributionConfig
        Configuration for the dispersion (r) parameter distribution
    p_distribution : DistributionConfig  
        Configuration for the success probability (p) parameter distribution
    gate_distribution : Optional[DistributionConfig]
        Configuration for the dropout gate parameter distribution. Required for
        ZINB models.
    p_capture_distribution : Optional[DistributionConfig]
        Configuration for the capture probability parameter distribution.
        Required for VCP models.
    n_components : Optional[int]
        Number of mixture components. If specified, model type will be converted
        to mixture variant.
    """
    # Base model type (e.g. "nbdm", "zinb")
    base_model: str
    
    # Distribution config for dispersion parameter
    r_distribution: DistributionConfig
    
    # Distribution config for success probability parameter  
    p_distribution: DistributionConfig
    
    # Optional distribution config for dropout gate parameter
    gate_distribution: Optional[DistributionConfig] = None
    
    # Optional distribution config for capture probability parameter
    p_capture_distribution: Optional[DistributionConfig] = None
    
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
        if "zinb" in self.base_model and self.gate_distribution is None:
            raise ValueError("ZINB models require gate_distribution")
            
        # Ensure VCP models have capture probability distribution specified
        if "vcp" in self.base_model and self.p_capture_distribution is None:
            raise ValueError("VCP models require p_capture_distribution")

# ------------------------------------------------------------------------------
# Default Model Configurations and Pre-configured Models
# ------------------------------------------------------------------------------

def get_default_model_config(
        base_model: str, 
        r_dist: str = "gamma", 
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
    # Set dispersion distribution based on r_dist parameter
    if r_dist == "gamma":
        r_config = GAMMA_CONFIG
    else:
        r_config = LOGNORMAL_CONFIG
    
    # Return config for Negative Binomial-Dirichlet Multinomial model
    if base_model == "nbdm":
        return ModelConfig(
            base_model=base_model,
            r_distribution=r_config,  # Dispersion parameter distribution
            p_distribution=BETA_CONFIG,  # Success probability distribution
            n_components=n_components  # Optional mixture components
        )
    
    # Return config for Zero-Inflated Negative Binomial model
    elif base_model == "zinb":
        return ModelConfig(
            base_model=base_model,
            r_distribution=r_config,  # Dispersion parameter distribution
            p_distribution=BETA_CONFIG,  # Success probability distribution
            gate_distribution=BETA_CONFIG,  # Dropout gate distribution
            n_components=n_components  # Optional mixture components
        )
    
    # Return config for Negative Binomial with Variable Capture Probability
    # model
    elif base_model == "nbvcp":
        return ModelConfig(
            base_model=base_model,
            r_distribution=r_config,  # Dispersion parameter distribution
            p_distribution=BETA_CONFIG,  # Success probability distribution
            p_capture_distribution=BETA_CONFIG,  # Capture probability distribution
            n_components=n_components  # Optional mixture components
        )
    
    # Return config for Zero-Inflated Negative Binomial with Variable Capture
    # Probability model
    elif base_model == "zinbvcp":
        return ModelConfig(
            base_model=base_model,
            r_distribution=r_config,  # Dispersion parameter distribution
            p_distribution=BETA_CONFIG,  # Success probability distribution
            gate_distribution=BETA_CONFIG,  # Dropout gate distribution
            p_capture_distribution=BETA_CONFIG,  # Capture probability distribution
            n_components=n_components  # Optional mixture components
        )
    
    # Raise error for unknown model types
    else:
        raise ValueError(f"Unknown model type: {base_model}")

# ------------------------------------------------------------------------------
# Guide configurations
# ------------------------------------------------------------------------------

@dataclass
class GuideConfig:
    """
    Configuration for guide (variational) distributions in SCRIBE models.
    
    This class defines the structure of variational distributions used to
    approximate the true posterior distributions of model parameters. It ensures
    the guide distributions match the structure of the generative model.
    
    Parameters
    ----------
    r_distribution : DistributionConfig
        Configuration for the dispersion parameter's variational distribution
    p_distribution : DistributionConfig  
        Configuration for the success probability's variational distribution
    gate_distribution : Optional[DistributionConfig]
        Configuration for the dropout gate's variational distribution (for
        zero-inflated models)
    p_capture_distribution : Optional[DistributionConfig]
        Configuration for the capture probability's variational distribution
        (for VCP models)
    """
    # Required distribution for dispersion parameter r
    r_distribution: DistributionConfig
    
    # Required distribution for success probability p 
    p_distribution: DistributionConfig
    
    # Optional distribution for dropout gate (zero-inflated models only)
    gate_distribution: Optional[DistributionConfig] = None
    
    # Optional distribution for capture probability (VCP models only)
    p_capture_distribution: Optional[DistributionConfig] = None
    
    def validate_with_model(self, model_config: ModelConfig):
        """
        Validate that guide configuration matches model configuration.
        
        Checks that the guide has appropriate distributions defined for all
        parameters in the model. Raises ValueError if guide is missing required
        distributions.
        
        Parameters
        ----------
        model_config : ModelConfig
            The model configuration to validate against
        
        Raises
        ------
        ValueError
            If guide is missing distributions required by the model
        """
        # Check that guide has gate distribution if model does
        if model_config.gate_distribution is not None and self.gate_distribution is None:
            raise ValueError("Model has gate distribution but guide does not")
            
        # Check that guide has capture probability distribution if model does
        if model_config.p_capture_distribution is not None and self.p_capture_distribution is None:
            raise ValueError("Model has p_capture distribution but guide does not")

# ------------------------------------------------------------------------------

def get_default_guide_config(
        model_config: ModelConfig, 
        r_dist: str = "gamma"
    ) -> GuideConfig:
    """
    Get default guide configuration matching model structure.
    
    Creates a GuideConfig with appropriate variational distributions for all
    parameters in the model. Uses standard configurations for each distribution
    type.
    
    Parameters
    ----------
    model_config : ModelConfig
        Model configuration to match guide structure to
    r_dist : str, default="gamma"
        Distribution type to use for dispersion parameter ("gamma" or
        "lognormal")
        
    Returns
    -------
    GuideConfig
        Default guide configuration matching model structure
    """
    # Select distribution config for r based on specified type
    r_config = GAMMA_CONFIG if r_dist == "gamma" else LOGNORMAL_CONFIG
    
    # Create guide config matching model structure
    return GuideConfig(
        r_distribution=r_config,
        p_distribution=BETA_CONFIG,
        gate_distribution=model_config.gate_distribution,
        p_capture_distribution=model_config.p_capture_distribution
    )

# ------------------------------------------------------------------------------
# Pre-configured common model configurations
# ------------------------------------------------------------------------------

# NBDM model with gamma distribution for dispersion
NBDM_GAMMA = (
    get_default_model_config("nbdm", "gamma"),
    get_default_guide_config(get_default_model_config("nbdm", "gamma"))
)
# NBDM model with lognormal distribution for dispersion
NBDM_LOGNORMAL = (
    get_default_model_config("nbdm", "lognormal"),
    get_default_guide_config(get_default_model_config("nbdm", "lognormal"))
)

# ZINB model with gamma distribution for dispersion
ZINB_GAMMA = (
    get_default_model_config("zinb", "gamma"),
    get_default_guide_config(get_default_model_config("zinb", "gamma"))
)

# ZINB model with lognormal distribution for dispersion
ZINB_LOGNORMAL = (
    get_default_model_config("zinb", "lognormal"),
    get_default_guide_config(get_default_model_config("zinb", "lognormal"))
)

# NBVCP model with gamma distribution for dispersion
NBVCP_GAMMA = (
    get_default_model_config("nbvcp", "gamma"),
    get_default_guide_config(get_default_model_config("nbvcp", "gamma"))
)

# NBVCP model with lognormal distribution for dispersion
NBVCP_LOGNORMAL = (
    get_default_model_config("nbvcp", "lognormal"),
    get_default_guide_config(get_default_model_config("nbvcp", "lognormal"))
)

# ZINBVCP model with gamma distribution for dispersion
ZINBVCP_GAMMA = (
    get_default_model_config("zinbvcp", "gamma"),
    get_default_guide_config(get_default_model_config("zinbvcp", "gamma"))
)

# ZINBVCP model with lognormal distribution for dispersion
ZINBVCP_LOGNORMAL = (
    get_default_model_config("zinbvcp", "lognormal"),
    get_default_guide_config(get_default_model_config("zinbvcp", "lognormal"))
)

