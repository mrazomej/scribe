"""
SCRIBE: Single-cell Bayesian Inference Ensemble

A Bayesian method for identifying cell-type specific differences in gene expression
from single-cell RNA-sequencing data.
"""

# Import unified inference interface
from .inference import run_scribe

# Import legacy interfaces for backward compatibility
from .svi import run_scribe as run_scribe_svi
from .mcmc import run_scribe as run_scribe_mcmc

# Import result classes
from .results_svi import ScribeSVIResults
from .results_mcmc import ScribeMCMCResults

# Import core components for advanced usage
from .core import InputProcessor, PriorConfigFactory, ModelConfigFactory

# Import configuration classes
from .model_config import ConstrainedModelConfig, UnconstrainedModelConfig

# Import models and utilities
from . import models
from . import models_mix
from . import utils
from . import viz
from . import stats
from . import sampling

__version__ = "0.1.0"

__all__ = [
    # Main unified interface
    "run_scribe",
    
    # Legacy interfaces
    "run_scribe_svi", 
    "run_scribe_mcmc",
    
    # Results classes
    "ScribeSVIResults",
    "ScribeMCMCResults",
    
    # Core components
    "InputProcessor",
    "PriorConfigFactory",
    "ModelConfigFactory",
    
    # Configuration classes
    "ConstrainedModelConfig",
    "UnconstrainedModelConfig",
    
    # Modules
    "models",
    "models_mix", 
    "utils",
    "viz",
    "stats",
    "sampling",
]