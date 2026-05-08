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
    LNM = "lnm"
    LNMVCP = "lnmvcp"
    PLN = "pln"
    NBLN = "nbln"
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
    ``HierarchicalPriorType`` enum fields (``prob_prior``,
    ``zero_inflation_prior``) in ModelConfig rather than via separate enum
    values.

    The old names (STANDARD, LINKED, ODDS_RATIO) are kept for backward
    compatibility.
    """

    # Standard (DM-family) parameterizations
    CANONICAL = "canonical"
    MEAN_PROB = "mean_prob"
    MEAN_ODDS = "mean_odds"
    # LNM-family parameterizations. The three variants mirror the
    # DM-family trio: ``canonical`` keeps ``(r_T, p)`` as sampled
    # globals, ``mean_prob`` keeps ``(p, mu_T)``, and ``mean_odds``
    # keeps ``(phi_T, mu_T)`` and derives both ``r_T`` and ``p``.
    # The user-facing API maps ``parameterization="canonical"`` etc.
    # to one of these LNM-family enum values when the model is
    # ``"lnm"`` / ``"lnmvcp"``.
    LOGISTIC_NORMAL_CANONICAL = "logistic_normal_canonical"
    LOGISTIC_NORMAL_MEAN_PROB = "logistic_normal_mean_prob"
    LOGISTIC_NORMAL_MEAN_ODDS = "logistic_normal_mean_odds"
    # Count-LogNormal parameterization.  Covers every base model with
    # a low-rank Gaussian latent on log-rates (``μ + W·u + √d⊙ε``)
    # and a count observation channel mixed against the implied
    # LogNormal rate density:
    #
    # - ``base_model="pln"``  →  Poisson observation channel
    # - ``base_model="nbln"`` →  Negative Binomial observation channel
    #
    # The structural form (Gaussian-on-log-rates with low-rank
    # factorisation) is shared; observation channels are distinguished
    # by ``base_model``, mirroring how CANONICAL covers ``nbdm``,
    # ``nbvcp``, ``zinb``, ``zinbvcp``.  Per-model differences in
    # site names (``d_pln`` vs ``d_nbln``) and extra globals (``r_g``
    # for NBLN) are handled at the registry / factory layer via
    # dispatch on ``base_model``.
    COUNT_LOGNORMAL = "count_lognormal"
    # Backward-compatible alias retained so existing imports
    # (``Parameterization.POISSON_LOGNORMAL``) continue to resolve to
    # the same enum member.  New code should prefer ``COUNT_LOGNORMAL``.
    POISSON_LOGNORMAL = "count_lognormal"
    # Internal alias of LOGISTIC_NORMAL_CANONICAL kept so that internal
    # code still comparing against ``Parameterization.LOGISTIC_NORMAL``
    # resolves to canonical (the historical default). No longer
    # accepted from the user-facing API — that path raises a clear
    # error directing users to the three variants.
    LOGISTIC_NORMAL = "logistic_normal_canonical"
    # Backward compatibility aliases for the DM family
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
        # Legacy values from older pickles / configs. The
        # ``hierarchical_*`` variants were removed long ago; the
        # ``logistic_normal`` value was retired in favor of three
        # LNM-family variants (canonical / mean_prob / mean_odds) and
        # we map the legacy single-key form to ``canonical`` so old
        # pickles still load. Users hitting this path via the API get
        # an explicit error from
        # ``resolve_user_parameterization_for_model``.
        _legacy = {
            "hierarchical_canonical": "canonical",
            "hierarchical_mean_prob": "mean_prob",
            "hierarchical_mean_odds": "mean_odds",
            "logistic_normal": "logistic_normal_canonical",
            # Renamed in 2026-05: ``poisson_lognormal`` was the original
            # name (when only the PLN base model existed).  After NBLN
            # was added, the parameterization was renamed to
            # ``count_lognormal`` to reflect that it covers any count
            # observation channel mixed against a LogNormal rate.  Old
            # pickles persisting the legacy value still load via this
            # mapping.
            "poisson_lognormal": "count_lognormal",
        }
        if isinstance(value, str):
            base = _legacy.get(value)
            if base is not None:
                return cls(base)
        return None


# ------------------------------------------------------------------------------


class InferenceMethod(str, Enum):
    """Supported inference methods.

    - ``SVI``: Stochastic Variational Inference with mean-field guides.
    - ``MCMC``: Hamiltonian Monte Carlo (NUTS) for exact-ish posterior.
    - ``VAE``: SVI with an amortized encoder (the standard
      VAE pattern). All LNM and PLN-with-encoder fits use this method.
    - ``LAPLACE``: Per-cell Newton-iterated MAP + Gaussian Laplace
      approximation around the mode. PLN-only. Replaces the encoder
      with deterministic optimization on each cell's posterior; the
      outer loop is still SVI on the global parameters but with the
      Laplace-approximated marginal likelihood as its objective.
      See ``svi/_laplace_newton.py`` for the inner kernel.
    """

    SVI = "svi"
    MCMC = "mcmc"
    VAE = "vae"
    LAPLACE = "laplace"


# ------------------------------------------------------------------------------


class OverdispersionType(str, Enum):
    """Type of gene-specific overdispersion beyond the NB family.

    Controls whether the count distribution uses the standard Negative
    Binomial or a heavier-tailed generalization.  The overdispersion
    model is orthogonal to the model type (NBDM / ZINB / NBVCP /
    ZINBVCP) and composes with zero-inflation and variable capture.

    Attributes
    ----------
    NONE : str
        Standard Negative Binomial — no extra overdispersion.
    BNB : str
        Beta Negative Binomial — a Beta-distributed cell-to-cell
        variability in the NB success probability, yielding power-law
        tails and a closed-form PMF.
    """

    NONE = "none"
    BNB = "bnb"


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
