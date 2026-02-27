"""
Enums and constants for model configuration.

Enums (enumerations) are symbolic names bound to unique, constant values. They
provide an expressive, type-safe way to represent a group of related, fixed
choices—such as all allowed model types or inference strategies.

Using enums in our codebase helps to:
    - Avoid invalid configuration values by restricting possible options.
    - Improve documentation and code clarity by giving descriptive names to
      valid choices.
    - Enable static analysis and autocompletion in IDEs, making development
      safer and faster.
    - Facilitate validation of user input and configuration files.

All model configuration enums are collected and documented here to unify and
safeguard the configurable aspects of model setup.
"""

from enum import Enum

# ==============================================================================
# Enums for model configuration
# ==============================================================================


class ModelType(str, Enum):
    """Supported model types."""

    NBDM = "nbdm"
    ZINB = "zinb"
    NBVCP = "nbvcp"
    ZINBVCP = "zinbvcp"

    def with_mixture(self) -> str:
        """Get the mixture version of this model."""
        return f"{self.value}_mix"


# ------------------------------------------------------------------------------


class Parameterization(str, Enum):
    """Supported parameterization types.

    Parameterizations define which parameters are sampled directly vs derived:

    - CANONICAL (STANDARD): Samples p (probability) and r (dispersion) directly
    - MEAN_PROB (LINKED): Samples p (probability) and mu (mean), derives r
    - MEAN_ODDS (ODDS_RATIO): Samples phi (odds ratio) and mu (mean), derives p
      and r

    Hierarchical gene-specific priors (on p/phi and gate) are controlled via
    boolean flags (``hierarchical_p``, ``hierarchical_gate``) in ModelConfig
    rather than via separate enum values.

    The old names (STANDARD, LINKED, ODDS_RATIO) are kept for backward
    compatibility.
    """

    # Standard parameterizations
    CANONICAL = "canonical"
    MEAN_PROB = "mean_prob"
    MEAN_ODDS = "mean_odds"
    # Backward compatibility
    STANDARD = "standard"
    LINKED = "linked"
    ODDS_RATIO = "odds_ratio"


# ------------------------------------------------------------------------------


class InferenceMethod(str, Enum):
    """Supported inference methods."""

    SVI = "svi"
    MCMC = "mcmc"
    VAE = "vae"


# ------------------------------------------------------------------------------


class VAEPriorType(str, Enum):
    """VAE prior types."""

    STANDARD = "standard"
    DECOUPLED = "decoupled"


# ------------------------------------------------------------------------------


class VAEMaskType(str, Enum):
    """VAE mask types for decoupled prior."""

    ALTERNATING = "alternating"
    SEQUENTIAL = "sequential"


# ------------------------------------------------------------------------------


class VAEActivation(str, Enum):
    """VAE activation functions."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
