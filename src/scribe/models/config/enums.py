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
    ``HierarchicalPriorType`` enum fields (``p_prior``, ``gate_prior``) in
    ModelConfig rather than via separate enum values.

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

    # ------------------------------------------------------------------
    # Backward-compatible deserialization of removed hierarchical values.
    # Old pickles stored Parameterization("hierarchical_mean_odds") etc.
    # _missing_ lets pickle.load() resolve them to the base enum value
    # instead of raising ValueError.
    # ------------------------------------------------------------------

    @classmethod
    def _missing_(cls, value: object) -> "Parameterization | None":
        _legacy = {
            "hierarchical_canonical": "canonical",
            "hierarchical_mean_prob": "mean_prob",
            "hierarchical_mean_odds": "mean_odds",
        }
        if isinstance(value, str):
            base = _legacy.get(value)
            if base is not None:
                return cls(base)
        return None


# ------------------------------------------------------------------------------


class InferenceMethod(str, Enum):
    """Supported inference methods."""

    SVI = "svi"
    MCMC = "mcmc"
    VAE = "vae"


# ------------------------------------------------------------------------------


class HierarchicalPriorType(str, Enum):
    """Type of hierarchical shrinkage prior for a model parameter.

    Each gene-level or dataset-level parameter slot can independently
    select one of these prior types.  ``NONE`` means the parameter has
    no hierarchy (shared / flat prior); any other value activates a
    hierarchical prior with per-gene or per-dataset local scales.

    Attributes
    ----------
    NONE : str
        No hierarchy — parameter is shared or uses a flat prior.
    GAUSSIAN : str
        Normal (Gaussian) hierarchy with a shared location and scale.
    HORSESHOE : str
        Regularized (Finnish) horseshoe with Half-Cauchy global/local
        scales and an Inverse-Gamma slab.
    NEG : str
        Normal-Exponential-Gamma — a member of the TPBN family using a
        Gamma-Gamma hierarchy that is friendlier to SVI than the
        horseshoe.
    """

    NONE = "none"
    GAUSSIAN = "gaussian"
    HORSESHOE = "horseshoe"
    NEG = "neg"


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
