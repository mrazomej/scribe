"""
SCRIBE: Single-cell Bayesian Inference Ensemble

A Bayesian method for identifying cell-type specific differences in gene expression
from single-cell RNA-sequencing data.
"""

# Import unified inference interface
from .inference import run_scribe

# Import result classes
from .svi import ScribeSVIResults
from .mcmc import ScribeMCMCResults

# Import core components for advanced usage
from .core import InputProcessor, PriorConfigFactory, ModelConfigFactory

# Import configuration classes
from .models import ModelConfig

# Import models and utilities
from . import models
from . import utils
from . import viz
from . import stats
from . import sampling

__version__ = "0.1.0"

__all__ = [
    # Main unified interface
    "run_scribe",
    # Results classes
    "ScribeSVIResults",
    "ScribeMCMCResults",
    # Core components
    "InputProcessor",
    "PriorConfigFactory",
    "ModelConfigFactory",
    # Configuration classes
    "ModelConfig",
    # Modules
    "models",
    "utils",
    "viz",
    "stats",
    "sampling",
]
