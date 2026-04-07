"""
SCRIBE: Single-cell Bayesian Inference Ensemble

A Bayesian method for identifying cell-type specific differences in gene expression
from single-cell RNA-sequencing data.
"""

# Suppress known warnings from dependencies BEFORE any imports
import importlib
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
from .models.config import (
    ModelConfig,
    SVIConfig,
    MCMCConfig,
    DataConfig,
    InferenceConfig,
)

from . import viz
from . import utils
from . import stats

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

# Keep optional-heavy surfaces lazy so base `import scribe` remains usable
# without CLI/Hydra extras.
_LAZY_EXPORTS = {
    "data_loader": ("scribe.data_loader", None),
    "catalog": ("scribe.catalog", None),
    "ExperimentCatalog": ("scribe.catalog", "ExperimentCatalog"),
}


def __getattr__(name: str):
    """Resolve optional exports lazily on first access."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'scribe' has no attribute '{name}'")


# Import configuration classes
# Import main inference function
from .inference import run_scribe

# Import simplified API
from .api import fit

# Import results classes
from .mcmc import ScribeMCMCResults
from .svi import ScribeSVIResults
from .vae import ScribeVAEResults
from .de import ScribeDEResults, compare, compare_datasets
from .mc import ScribeModelComparisonResults, compare_models
from . import mc

__version__ = "0.1.0"

__all__ = [
    # Core components
    "InputProcessor",
    # Configuration classes
    "ModelConfig",
    "SVIConfig",
    "MCMCConfig",
    "DataConfig",
    "InferenceConfig",
    # Main inference functions
    "fit",  # Simplified API (recommended)
    "run_scribe",  # Lower-level API
    # Results classes
    "ScribeSVIResults",
    "ScribeMCMCResults",
    "ScribeVAEResults",
    "ScribeDEResults",
    "compare",
    # Model comparison
    "ScribeModelComparisonResults",
    "compare_models",
    "mc",
    # Experiment management
    "ExperimentCatalog",
    "catalog",
    # Other modules
    "viz",
    "utils",
    "stats",
    # Utility functions
    "apply_distribution_mode_patches",
    "data_loader",
]
