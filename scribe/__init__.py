"""
SCRIBE: Single-cell Bayesian Inference Ensemble

A Bayesian method for identifying cell-type specific differences in gene expression
from single-cell RNA-sequencing data.
"""

# Import core components for advanced usage
from .core import InputProcessor
from .models.model_config import ModelConfig

from . import viz
from . import utils
from . import stats

# Import configuration classes
# Import main inference function
from .inference import run_scribe

# Import results classes
from .mcmc import ScribeMCMCResults
from .svi import ScribeSVIResults

__version__ = "0.1.0"

__all__ = [
    # Core components
    "InputProcessor",
    # Configuration classes
    "ModelConfig",
    # Main inference function
    "run_scribe",
    # Results classes
    "ScribeSVIResults",
    "ScribeMCMCResults",
    # Other modules
    "viz",
    "utils",
    "stats",
]
