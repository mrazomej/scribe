"""
Parameterization strategy classes for model construction.

This module provides a strategy pattern for handling different parameterization
schemes. Each parameterization encapsulates:

- Core parameters it requires (e.g., p+r, p+mu, phi+mu)
- How to build parameter specs (constrained vs unconstrained)
- Derived parameter computations
- Model-specific parameter transformations (e.g., p_capture → phi_capture)

This eliminates nested conditionals in preset factories and makes it easy to
add new parameterizations.

Classes
-------
Parameterization
    Abstract base class for parameterization schemes.
CanonicalParameterization
    Directly samples p (probability) and r (dispersion).
MeanProbParameterization
    Samples p (probability) and mu (mean), derives r.
MeanOddsParameterization
    Samples phi (odds ratio) and mu (mean), derives p and r.
LogisticNormalParameterization
    Population total-count parameters (``r_T``, ``p``) plus VAE-decoded
    ALR coordinates ``y_alr`` for compositional multinomial structure.

Examples
--------
>>> from scribe.models.parameterizations import PARAMETERIZATIONS
>>> param_strategy = PARAMETERIZATIONS["canonical"]
>>> param_specs = param_strategy.build_param_specs(
...     unconstrained=False,
...     guide_families=GuideFamilyConfig(),
... )
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple

import jax.numpy as jnp

from ..builders.parameter_specs import (
    BetaSpec,
    BetaPrimeSpec,
    DerivedParam,
    PositiveNormalSpec,
    LogNormalSpec,
    ParamSpec,
    SigmoidNormalSpec,
)
from ..config import GuideFamilyConfig

# ==============================================================================
# Abstract Base Class
# ==============================================================================


class Parameterization(ABC):
    """Abstract base class for parameterization schemes.

    Each parameterization defines:
        - What core parameters to sample
        - How to build parameter specs (constrained vs unconstrained)
        - What derived parameters to compute
        - How to transform model-specific parameter names

    Subclasses must implement all abstract methods and properties.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this parameterization."""
        pass

    @property
    @abstractmethod
    def core_parameters(self) -> List[str]:
        """List of core parameter names this parameterization requires."""
        pass

    @property
    @abstractmethod
    def gene_param_name(self) -> str:
        """Name of the gene-specific parameter (e.g., 'r' or 'mu')."""
        pass

    @abstractmethod
    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for this parameterization.

        Parameters
        ----------
        unconstrained : bool
            If True, use Normal+transform specs. Otherwise, use constrained
            distribution specs (Beta, LogNormal, etc.).
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration.
        n_components : Optional[int], default=None
            Number of mixture components. If provided, parameters in
            mixture_params will be marked as mixture-specific.
        mixture_params : Optional[List[str]], default=None
            List of parameter names to mark as mixture-specific.
            If None and n_components is set, defaults to all sampled core
            parameters for the selected parameterization.

        Returns
        -------
        List[ParamSpec]
            Parameter specifications for core parameters.
        """
        pass

    @abstractmethod
    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameter computations.

        Returns
        -------
        List[DerivedParam]
            Derived parameter specifications.
        """
        pass

    def decoder_output_spec(self, base_model: str) -> List[Tuple[str, str]]:
        """Return (param_name, transform) for each decoder output head.

        The decoder produces the gene-specific core parameter for this
        parameterization. Additional heads (e.g., gate) are added based
        on the base_model type.

        Parameters
        ----------
        base_model : str
            Base model type: "nbdm", "zinb", "nbvcp", "zinbvcp".

        Returns
        -------
        List of (param_name, transform) tuples.
        """
        specs = [(self.gene_param_name, "softplus")]
        if base_model in ("zinb", "zinbvcp"):
            specs.append(("gate", "sigmoid"))
        return specs

    def transform_model_param(self, param_name: str) -> str:
        """Transform model-specific parameter names.

        Some parameterizations transform model-specific parameters.
        For example, mean_odds transforms p_capture → phi_capture.

        Parameters
        ----------
        param_name : str
            Original parameter name (e.g., "p_capture").

        Returns
        -------
        str
            Transformed parameter name (e.g., "phi_capture" for mean_odds,
            unchanged for others).
        """
        # Default: no transformation
        return param_name

    @property
    def requires_vae(self) -> bool:
        """Whether this parameterization is only supported with VAE inference.

        Returns
        -------
        bool
            False for standard parameterizations; subclasses may override.
        """
        return False


# ==============================================================================
# Concrete Parameterization Classes
# ==============================================================================

# ------------------------------------------------------------------------------
# Canonical Parameterization
# ------------------------------------------------------------------------------


class CanonicalParameterization(Parameterization):
    """Canonical parameterization: directly samples p and r.

    This is the standard parameterization where both the success probability
    p and dispersion r are sampled directly from their priors.

    Core parameters:
    - p: Success probability (Beta or SigmoidNormal)
    - r: Dispersion parameter (LogNormal or ExpNormal)

    Derived parameters:
    - mu: Mean expression ``mu = r * p / (1 - p)``.  Declared so that
      axis membership (dataset, mixture) propagates correctly from the
      core parameters to ``mu`` during layout inference.
    """

    @property
    def name(self) -> str:
        return "canonical"

    @property
    def core_parameters(self) -> List[str]:
        return ["p", "r"]

    @property
    def gene_param_name(self) -> str:
        return "r"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for canonical parameterization."""
        p_family = guide_families.get("p")
        r_family = guide_families.get("r")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["p", "r"]
            is_p_mixture = "p" in mixture_params
            is_r_mixture = "r" in mixture_params
        else:
            is_p_mixture = False
            is_r_mixture = False

        if unconstrained:
            return [
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                PositiveNormalSpec(
                    name="r",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
            ]
        else:
            return [
                BetaSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                LogNormalSpec(
                    name="r",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Derived parameter: mu = r * p / (1 - p).

        Although ``r`` and ``p`` are the *sampled* parameters in canonical
        mode, ``mu`` is derived from them during MAP / posterior extraction
        (see ``_compute_canonical_parameters``).  Declaring this dependency
        lets ``expand_membership_from_derived`` propagate dataset (and
        mixture) axis membership from ``r`` / ``p`` to ``mu``, ensuring
        that ``get_dataset()`` slices ``mu`` along the correct axis.
        """
        return [DerivedParam("mu", _compute_mu_from_r_p, ["r", "p"])]


# ------------------------------------------------------------------------------
# Mean Probability Parameterization
# ------------------------------------------------------------------------------


class MeanProbParameterization(Parameterization):
    """Mean-probability parameterization: samples p and mu, derives r.

    This parameterization samples the success probability p and mean expression
    mu, then derives the dispersion r as:

        r = mu * (1 - p) / p

    This links the dispersion to the mean, which can help capture correlations
    between p and r in the variational posterior.

    Core parameters:
    - p: Success probability (Beta or SigmoidNormal)
    - mu: Mean expression (LogNormal or ExpNormal)

    Derived parameters:
    - r: Dispersion (computed from p and mu)
    """

    @property
    def name(self) -> str:
        return "mean_prob"

    @property
    def core_parameters(self) -> List[str]:
        return ["p", "mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for mean-probability parameterization."""
        p_family = guide_families.get("p")
        mu_family = guide_families.get("mu")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["p", "mu"]
            is_p_mixture = "p" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_p_mixture = False
            is_mu_mixture = False

        if unconstrained:
            return [
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                PositiveNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]
        else:
            return [
                BetaSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameter: r = mu * (1 - p) / p."""
        return [DerivedParam("r", _compute_r_from_mu_p, ["p", "mu"])]


# ------------------------------------------------------------------------------
# Mean Odds Parameterization
# ------------------------------------------------------------------------------


class MeanOddsParameterization(Parameterization):
    """Mean-odds parameterization: samples phi and mu, derives p and r.

    This parameterization samples the odds ratio phi and mean expression mu,
    then derives both the success probability p and dispersion r:

        p = 1 / (1 + phi)
        r = mu * phi

    This is numerically more stable than mean_prob when p is close to 0 or 1,
    as it avoids division by p in the computation of r.

    Core parameters:
    - phi: Odds ratio (BetaPrime or ExpNormal)
    - mu: Mean expression (LogNormal or ExpNormal)

    Derived parameters:
    - p: Success probability (computed from phi)
    - r: Dispersion (computed from phi and mu)

    Model-specific transformations:
    - p_capture → phi_capture (for VCP models)
    """

    @property
    def name(self) -> str:
        return "mean_odds"

    @property
    def core_parameters(self) -> List[str]:
        return ["phi", "mu"]

    @property
    def gene_param_name(self) -> str:
        return "mu"

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build parameter specs for mean-odds parameterization."""
        phi_family = guide_families.get("phi")
        mu_family = guide_families.get("mu")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                # Default: make all sampled core params mixture-specific
                mixture_params = ["phi", "mu"]
            is_phi_mixture = "phi" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_phi_mixture = False
            is_mu_mixture = False

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="phi",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
                PositiveNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]
        else:
            return [
                BetaPrimeSpec(
                    name="phi",
                    shape_dims=(),
                    default_params=(1.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
                LogNormalSpec(
                    name="mu",
                    shape_dims=("n_genes",),
                    default_params=(0.0, 1.0),
                    is_gene_specific=True,
                    guide_family=mu_family,
                    is_mixture=is_mu_mixture,
                ),
            ]

    # --------------------------------------------------------------------------

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameters: p = 1/(1+phi), r = mu*phi."""
        return [
            DerivedParam("r", _compute_r_from_mu_phi, ["phi", "mu"]),
            DerivedParam("p", lambda phi: 1.0 / (1.0 + phi), ["phi"]),
        ]

    # --------------------------------------------------------------------------

    def transform_model_param(self, param_name: str) -> str:
        """Transform p_capture to phi_capture for mean_odds parameterization."""
        if param_name == "p_capture":
            return "phi_capture"
        return param_name


# ------------------------------------------------------------------------------
# Logistic-Normal Multinomial Parameterization
# ------------------------------------------------------------------------------


class LogisticNormalParameterization(Parameterization):
    """Logistic-normal multinomial parameterization for VAE-only models.

    Population-level total UMI counts follow a Negative Binomial with
    dispersion ``r_T`` and base success probability ``p`` (NumPyro ``probs``,
    before any VCP modulation on totals).  Per-cell gene compositions use a
    multinomial on the simplex after an ALR map from decoder outputs ``y_alr``
    (length ``G-1``).  Optional learned diagonal ALR noise is controlled at the
    model level via ``ModelConfig.d_mode``; the extra parameter ``d_lnm`` is
    wired in the VAE factory when ``d_mode=\"learned\"``.

    Core parameters (global, not gene-specific):

    - ``r_T``: Total-count NB dispersion (LogNormal or PositiveNormal).
    - ``p``: Total-count NB success probability (Beta or SigmoidNormal).

    The ``d_lnm`` vector is **not** included in :meth:`core_parameters`; it
    is appended by the preset factory when ``d_mode=\"learned\"``.
    """

    def __init__(self, d_mode: Literal["low_rank", "learned"] = "low_rank"):
        """Store diagonal mode for introspection (factory uses ``ModelConfig``).

        Parameters
        ----------
        d_mode : {\"low_rank\", \"learned\"}, default=\"low_rank\"
            Mirrors ``ModelConfig.d_mode``.  The singleton registry instance
            uses the default; the preset factory reads runtime config.
        """
        self._init_d_mode: Literal["low_rank", "learned"] = d_mode

    @property
    def name(self) -> str:
        return "logistic_normal"

    @property
    def core_parameters(self) -> List[str]:
        return ["r_T", "p"]

    @property
    def gene_param_name(self) -> str:
        return "y_alr"

    @property
    def requires_vae(self) -> bool:
        return True

    def build_param_specs(
        self,
        unconstrained: bool,
        guide_families: GuideFamilyConfig,
        n_components: Optional[int] = None,
        mixture_params: Optional[List[str]] = None,
    ) -> List[ParamSpec]:
        """Build specs for ``r_T`` (LogNormal or PositiveNormal) and ``p``."""
        r_family = guide_families.get("r_T")
        p_family = guide_families.get("p")

        # Mixture-aware flags mirror canonical p/r handling.
        if n_components is not None:
            if mixture_params is None:
                mixture_params = ["r_T", "p"]
            is_r_mixture = "r_T" in mixture_params
            is_p_mixture = "p" in mixture_params
        else:
            is_r_mixture = False
            is_p_mixture = False

        if unconstrained:
            return [
                PositiveNormalSpec(
                    name="r_T",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=r_family,
                    is_mixture=is_r_mixture,
                ),
                SigmoidNormalSpec(
                    name="p",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=p_family,
                    is_mixture=is_p_mixture,
                ),
            ]
        return [
            LogNormalSpec(
                name="r_T",
                shape_dims=(),
                default_params=(0.0, 1.0),
                guide_family=r_family,
                is_mixture=is_r_mixture,
            ),
            BetaSpec(
                name="p",
                shape_dims=(),
                default_params=(1.0, 1.0),
                guide_family=p_family,
                is_mixture=is_p_mixture,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        """No closed-form derived parameters for this parameterization."""
        return []

    def decoder_output_spec(self, base_model: str) -> List[Tuple[str, str]]:
        """Return a single identity head for ALR coordinates ``y_alr``.

        Parameters
        ----------
        base_model : str
            Ignored; included for API compatibility with other
            parameterizations.

        Returns
        -------
        list of tuple
            One identity head: ``("y_alr", "identity")``, i.e. a linear map from
            latent ``z`` to ALR coordinates without an extra output nonlinearity.
        """
        del base_model
        return [("y_alr", "identity")]


# ==============================================================================
# NOTE: Hierarchical parameterization classes have been removed.
# Hierarchical priors on p/phi and gate are now controlled via boolean flags
# (hierarchical_p, hierarchical_gate) in ModelConfig. The factory dynamically
# replaces flat specs with hierarchical triplets when these flags are set.
# ==============================================================================

# ==============================================================================
# Derived Parameter Compute Functions
# ==============================================================================
#
# These functions contain *only* pure math.  Shape alignment across
# different axis layouts (e.g. mu with a dataset axis vs. phi without)
# is handled by the call site in ``ModelBuilder`` using
# ``merge_layouts`` / ``align_to_layout`` before invoking these.


def _compute_r_from_mu_phi(phi: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the dispersion parameter ``r = mu * phi``.

    Inputs are expected to be pre-aligned by the caller (i.e. already
    broadcast-compatible).  The model builder uses ``AxisLayout`` metadata
    to insert singleton dimensions where axes differ before calling this.

    Parameters
    ----------
    phi : jnp.ndarray
        Odds ratio parameter (pre-aligned).
    mu : jnp.ndarray
        Mean parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Dispersion parameter ``r = mu * phi``.
    """
    return mu * phi


# ------------------------------------------------------------------------------


def _compute_mu_from_r_p(r: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Compute the mean parameter ``mu = r * p / (1 - p)``.

    Used by the canonical parameterization's ``DerivedParam`` declaration
    so that ``expand_membership_from_derived`` can propagate axis
    membership from ``r`` and ``p`` to ``mu``.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter (pre-aligned).
    p : jnp.ndarray
        Success probability parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Mean parameter ``mu = r * p / (1 - p)``.
    """
    return r * p / (1 - p)


# ------------------------------------------------------------------------------


def _compute_r_from_mu_p(p: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute the dispersion parameter ``r = mu * (1 - p) / p``.

    Inputs are expected to be pre-aligned by the caller (i.e. already
    broadcast-compatible).  The model builder uses ``AxisLayout`` metadata
    to insert singleton dimensions where axes differ before calling this.

    Parameters
    ----------
    p : jnp.ndarray
        Success probability parameter (pre-aligned).
    mu : jnp.ndarray
        Mean parameter (pre-aligned).

    Returns
    -------
    jnp.ndarray
        Dispersion parameter ``r = mu * (1 - p) / p``.
    """
    return mu * (1 - p) / p


# ==============================================================================
# Parameterization Registry
# ==============================================================================

# Create singleton instances
_canonical = CanonicalParameterization()
_mean_prob = MeanProbParameterization()
_mean_odds = MeanOddsParameterization()
_logistic_normal = LogisticNormalParameterization()

# Registry mapping names to parameterization instances
PARAMETERIZATIONS = {
    # Standard parameterizations
    "canonical": _canonical,
    "mean_prob": _mean_prob,
    "mean_odds": _mean_odds,
    "logistic_normal": _logistic_normal,
    # Backward compatibility
    "standard": _canonical,
    "linked": _mean_prob,
    "odds_ratio": _mean_odds,
}

__all__ = [
    "Parameterization",
    # Standard
    "CanonicalParameterization",
    "MeanProbParameterization",
    "MeanOddsParameterization",
    "LogisticNormalParameterization",
    "PARAMETERIZATIONS",
    # Derived parameter compute functions (pure math, alignment-free)
    "_compute_mu_from_r_p",
    "_compute_r_from_mu_phi",
    "_compute_r_from_mu_p",
]
