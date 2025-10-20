"""
SCRIBE: Single-cell Bayesian Inference Ensemble

A Bayesian method for identifying cell-type specific differences in gene expression
from single-cell RNA-sequencing data.
"""

# Suppress known warnings from dependencies BEFORE any imports
import warnings

# Suppress FutureWarnings from scanpy/anndata about deprecated __version__ usage
# Use a more general approach to catch all instances of this warning
warnings.filterwarnings(
    "ignore",
    message=".*__version__ is deprecated.*",
    category=FutureWarning,
)

# Also suppress the specific warning from anndata
warnings.filterwarnings(
    "ignore",
    message=".*importlib.metadata.version.*",
    category=FutureWarning,
)

# Import core components for advanced usage
from .core import InputProcessor
from .models.config import ModelConfig, SVIConfig, MCMCConfig, DataConfig

from . import viz
from . import utils
from . import stats
from . import data_loader
from .catalog import ExperimentCatalog

# ------------------------------------------------------------------------------
# Register KL with NumPyro on import (idempotent)

try:
    from .stats import _kl_betaprime  # side effect: decorator runs
except Exception as _e:
    import logging

    logging.warning(
        "SCRIBE: Failed to register BetaPrime KL divergence with NumPyro. "
        "Analytic KL for BetaPrime distributions will not be available. "
        f"Error: {_e}"
    )
    warnings.warn(
        "SCRIBE: Could not register BetaPrime KL divergence with NumPyro. "
        f"Error: {_e}",
        RuntimeWarning,
    )

try:
    from .stats import _kl_lognormal  # side effect: decorator runs
except Exception as _e:
    import logging

    logging.warning(
        "SCRIBE: Failed to register LogNormal KL divergence with NumPyro. "
        "Analytic KL for LogNormal distributions will not be available. "
        f"Error: {_e}"
    )
    warnings.warn(
        "SCRIBE: Could not register LogNormal KL divergence with NumPyro. "
        f"Error: {_e}",
        RuntimeWarning,
    )

# ------------------------------------------------------------------------------

# Apply distribution mode monkey patches
try:
    from .stats import apply_distribution_mode_patches

    apply_distribution_mode_patches()
except Exception as _e:
    import logging

    logging.warning(
        "SCRIBE: Failed to apply distribution mode patches. "
        "Distribution mode properties will not be available. "
        f"Error: {_e}"
    )
    warnings.warn(
        "SCRIBE: Could not apply distribution mode patches. " f"Error: {_e}",
        RuntimeWarning,
    )

# ------------------------------------------------------------------------------

# Import configuration classes
# Import main inference function
from .inference import run_scribe

# Import results classes
from .mcmc import ScribeMCMCResults
from .svi import ScribeSVIResults
from .vae import ScribeVAEResults

__version__ = "0.1.0"

__all__ = [
    # Core components
    "InputProcessor",
    # Configuration classes
    "ModelConfig",
    "SVIConfig",
    "MCMCConfig",
    "DataConfig",
    # Main inference function
    "run_scribe",
    # Results classes
    "ScribeSVIResults",
    "ScribeMCMCResults",
    "ScribeVAEResults",
    # Experiment management
    "ExperimentCatalog",
    # Other modules
    "viz",
    "utils",
    "stats",
    # Utility functions
    "apply_distribution_mode_patches",
    "data_loader",
]
