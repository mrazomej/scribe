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
from typing import List, Optional, Tuple

import jax.numpy as jnp

from ..builders.parameter_specs import (
    BetaSpec,
    BetaPrimeSpec,
    DerivedParam,
    ExpNormalSpec,
    HierarchicalExpNormalSpec,
    HierarchicalSigmoidNormalSpec,
    LogNormalSpec,
    NormalWithTransformSpec,
    ParamSpec,
    SigmoidNormalSpec,
    SoftplusNormalSpec,
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

    No derived parameters are computed.
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
                ExpNormalSpec(
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
        """No derived parameters for canonical parameterization."""
        return []


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
                ExpNormalSpec(
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
                ExpNormalSpec(
                    name="phi",
                    shape_dims=(),
                    default_params=(0.0, 1.0),
                    guide_family=phi_family,
                    is_mixture=is_phi_mixture,
                ),
                ExpNormalSpec(
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


# ==============================================================================
# Hierarchical Parameterization Classes
# ==============================================================================
#
# These parameterizations relax the shared-p assumption of the standard
# Dirichlet-Multinomial factorization by placing a hierarchical prior on the
# success probability (or odds ratio).  The generative model becomes:
#
#     hyper_loc   ~ Normal(0, 1)           (population-level location)
#     hyper_scale ~ Softplus(Normal(0, 1)) (population-level spread)
#     param_g     = transform(Normal(hyper_loc, hyper_scale))  per gene
#
# This provides adaptive shrinkage: the data determine how much gene-specific
# variation to allow.  When hyper_scale -> 0, genes collapse to a shared value
# (recovering the standard model).  When hyper_scale is large, genes have
# diverse values.
# ==============================================================================

# ------------------------------------------------------------------------------
# Hierarchical Canonical Parameterization
# ------------------------------------------------------------------------------


class HierarchicalCanonicalParameterization(Parameterization):
    """Hierarchical canonical: gene-specific p_g with hyperprior, direct r.

    Relaxes the shared-p assumption by placing a hierarchical Normal prior
    on the logit-space success probability.  Each gene draws its own
    ``p_g`` from the learned population distribution.

    Core parameters:
    - logit_p_loc: Population mean of logit(p) (Normal, scalar)
    - logit_p_scale: Population spread of logit(p) (Softplus, scalar)
    - p: Gene-specific success probability (hierarchical sigmoid)
    - r: Dispersion parameter (ExpNormal or LogNormal)

    No derived parameters are computed.
    """

    @property
    def name(self) -> str:
        return "hierarchical_canonical"

    @property
    def core_parameters(self) -> List[str]:
        return ["logit_p_loc", "logit_p_scale", "p", "r"]

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
        """Build parameter specs for hierarchical canonical parameterization.

        Parameters
        ----------
        unconstrained : bool
            Ignored (always unconstrained for hierarchical).
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration.
        n_components : Optional[int]
            Number of mixture components.
        mixture_params : Optional[List[str]]
            Parameters to mark as mixture-specific.

        Returns
        -------
        List[ParamSpec]
            Parameter specifications including hyperparameters and
            hierarchical gene-specific p.
        """
        p_family = guide_families.get("p")
        r_family = guide_families.get("r")

        # Determine which parameters are mixture-specific
        if n_components is not None:
            if mixture_params is None:
                mixture_params = ["p", "r"]
            is_p_mixture = "p" in mixture_params
            is_r_mixture = "r" in mixture_params
        else:
            is_p_mixture = False
            is_r_mixture = False

        return [
            # --- Hyperparameters (global scalars) ---
            # logit_p_loc: unconstrained location in logit space
            NormalWithTransformSpec(
                name="logit_p_loc",
                shape_dims=(),
                default_params=(0.0, 1.0),
            ),
            # logit_p_scale: positive scale in logit space
            SoftplusNormalSpec(
                name="logit_p_scale",
                shape_dims=(),
                default_params=(0.0, 0.5),
            ),
            # --- Hierarchical gene-specific p ---
            HierarchicalSigmoidNormalSpec(
                name="p",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="logit_p_loc",
                hyper_scale_name="logit_p_scale",
                is_gene_specific=True,
                guide_family=p_family,
                is_mixture=is_p_mixture,
            ),
            # --- Standard gene-specific r ---
            ExpNormalSpec(
                name="r",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=r_family,
                is_mixture=is_r_mixture,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        """No derived parameters for hierarchical canonical."""
        return []


# ------------------------------------------------------------------------------
# Hierarchical Mean Probability Parameterization
# ------------------------------------------------------------------------------


class HierarchicalMeanProbParameterization(Parameterization):
    """Hierarchical mean-prob: gene-specific p_g with hyperprior, mu, derives r.

    Relaxes the shared-p assumption while keeping the mean-probability
    derived parameterization.  Gene-specific ``p_g`` is drawn from a
    hierarchical Normal prior in logit space, and ``r`` is derived as
    ``r = mu * (1 - p) / p``.

    Core parameters:
    - logit_p_loc: Population mean of logit(p)
    - logit_p_scale: Population spread of logit(p)
    - p: Gene-specific success probability (hierarchical sigmoid)
    - mu: Gene-specific mean expression (ExpNormal or LogNormal)

    Derived parameters:
    - r = mu * (1 - p) / p
    """

    @property
    def name(self) -> str:
        return "hierarchical_mean_prob"

    @property
    def core_parameters(self) -> List[str]:
        return ["logit_p_loc", "logit_p_scale", "p", "mu"]

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
        """Build parameter specs for hierarchical mean-prob parameterization.

        Parameters
        ----------
        unconstrained : bool
            Ignored (always unconstrained for hierarchical).
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration.
        n_components : Optional[int]
            Number of mixture components.
        mixture_params : Optional[List[str]]
            Parameters to mark as mixture-specific.

        Returns
        -------
        List[ParamSpec]
            Parameter specifications including hyperparameters,
            hierarchical p, and standard mu.
        """
        p_family = guide_families.get("p")
        mu_family = guide_families.get("mu")

        if n_components is not None:
            if mixture_params is None:
                mixture_params = ["p", "mu"]
            is_p_mixture = "p" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_p_mixture = False
            is_mu_mixture = False

        return [
            # --- Hyperparameters (global scalars) ---
            NormalWithTransformSpec(
                name="logit_p_loc",
                shape_dims=(),
                default_params=(0.0, 1.0),
            ),
            SoftplusNormalSpec(
                name="logit_p_scale",
                shape_dims=(),
                default_params=(0.0, 0.5),
            ),
            # --- Hierarchical gene-specific p ---
            HierarchicalSigmoidNormalSpec(
                name="p",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="logit_p_loc",
                hyper_scale_name="logit_p_scale",
                is_gene_specific=True,
                guide_family=p_family,
                is_mixture=is_p_mixture,
            ),
            # --- Standard gene-specific mu ---
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=is_mu_mixture,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameter: r = mu * (1 - p) / p.

        Returns
        -------
        List[DerivedParam]
            Single derived parameter for dispersion r.

        Notes
        -----
        Both p and mu are now gene-specific, so no special broadcasting
        is needed — they have the same shape.
        """
        return [DerivedParam("r", _compute_r_from_mu_p, ["p", "mu"])]


# ------------------------------------------------------------------------------
# Hierarchical Mean Odds Parameterization
# ------------------------------------------------------------------------------


class HierarchicalMeanOddsParameterization(Parameterization):
    """Hierarchical mean-odds: gene-specific phi_g with hyperprior, mu, derives p & r.

    Relaxes the shared-phi assumption while keeping the numerically stable
    mean-odds derived parameterization.  Gene-specific ``phi_g`` is drawn
    from a hierarchical Normal prior in log space, and both ``p`` and ``r``
    are derived:

        p = 1 / (1 + phi)
        r = mu * phi

    Core parameters:
    - log_phi_loc: Population mean of log(phi)
    - log_phi_scale: Population spread of log(phi)
    - phi: Gene-specific odds ratio (hierarchical exp)
    - mu: Gene-specific mean expression (ExpNormal or LogNormal)

    Derived parameters:
    - p = 1 / (1 + phi)
    - r = mu * phi
    """

    @property
    def name(self) -> str:
        return "hierarchical_mean_odds"

    @property
    def core_parameters(self) -> List[str]:
        return ["log_phi_loc", "log_phi_scale", "phi", "mu"]

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
        """Build parameter specs for hierarchical mean-odds parameterization.

        Parameters
        ----------
        unconstrained : bool
            Ignored (always unconstrained for hierarchical).
        guide_families : GuideFamilyConfig
            Per-parameter guide family configuration.
        n_components : Optional[int]
            Number of mixture components.
        mixture_params : Optional[List[str]]
            Parameters to mark as mixture-specific.

        Returns
        -------
        List[ParamSpec]
            Parameter specifications including hyperparameters,
            hierarchical phi, and standard mu.
        """
        phi_family = guide_families.get("phi")
        mu_family = guide_families.get("mu")

        if n_components is not None:
            if mixture_params is None:
                mixture_params = ["phi", "mu"]
            is_phi_mixture = "phi" in mixture_params
            is_mu_mixture = "mu" in mixture_params
        else:
            is_phi_mixture = False
            is_mu_mixture = False

        return [
            # --- Hyperparameters (global scalars) ---
            NormalWithTransformSpec(
                name="log_phi_loc",
                shape_dims=(),
                default_params=(0.0, 1.0),
            ),
            SoftplusNormalSpec(
                name="log_phi_scale",
                shape_dims=(),
                default_params=(0.0, 0.5),
            ),
            # --- Hierarchical gene-specific phi ---
            HierarchicalExpNormalSpec(
                name="phi",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                hyper_loc_name="log_phi_loc",
                hyper_scale_name="log_phi_scale",
                is_gene_specific=True,
                guide_family=phi_family,
                is_mixture=is_phi_mixture,
            ),
            # --- Standard gene-specific mu ---
            ExpNormalSpec(
                name="mu",
                shape_dims=("n_genes",),
                default_params=(0.0, 1.0),
                is_gene_specific=True,
                guide_family=mu_family,
                is_mixture=is_mu_mixture,
            ),
        ]

    def build_derived_params(self) -> List[DerivedParam]:
        """Build derived parameters: p = 1/(1+phi), r = mu*phi.

        Returns
        -------
        List[DerivedParam]
            Derived parameters for dispersion r and probability p.

        Notes
        -----
        Both phi and mu are gene-specific, so element-wise operations
        work directly without special broadcasting.
        """
        return [
            DerivedParam("r", _compute_r_from_mu_phi, ["phi", "mu"]),
            DerivedParam("p", lambda phi: 1.0 / (1.0 + phi), ["phi"]),
        ]

    def transform_model_param(self, param_name: str) -> str:
        """Transform p_capture to phi_capture for mean_odds parameterization."""
        if param_name == "p_capture":
            return "phi_capture"
        return param_name


# ==============================================================================
# Helper Functions for Derived Parameters
# ==============================================================================


def _broadcast_scalar_for_mixture(
    scalar_param: jnp.ndarray, gene_param: jnp.ndarray
) -> jnp.ndarray:
    """
    Expand a scalar mixture parameter for broadcasting with gene-specific
    params.

    When a parameter is mixture-specific but not gene-specific (shape:
    n_components,) and needs to broadcast with a parameter that is both
    mixture-specific and gene-specific (shape: n_components, n_genes), we need
    to expand the scalar parameter to (n_components, 1) for proper broadcasting.

    Parameters
    ----------
    scalar_param : jnp.ndarray
        Parameter that may need expansion. Shape can be:
        - () for scalar (non-mixture)
        - (n_components,) for mixture-specific, non-gene-specific
    gene_param : jnp.ndarray
        Gene-specific parameter. Shape can be:
        - (n_genes,) for gene-specific (non-mixture)
        - (n_components, n_genes) for mixture-specific and gene-specific

    Returns
    -------
    jnp.ndarray
        The scalar_param, possibly expanded to (n_components, 1) for broadcasting.
    """
    if (
        scalar_param.ndim == 1
        and gene_param.ndim == 2
        and scalar_param.shape[0] == gene_param.shape[0]
    ):
        # Expand from (n_components,) to (n_components, 1) for broadcasting
        return scalar_param[:, None]
    return scalar_param


# ------------------------------------------------------------------------------


def _compute_r_from_mu_phi(phi: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute r = mu * phi with proper broadcasting for mixture models.

    Parameters
    ----------
    phi : jnp.ndarray
        Odds ratio parameter.
    mu : jnp.ndarray
        Mean parameter.

    Returns
    -------
    jnp.ndarray
        Dispersion parameter r = mu * phi.
    """
    phi = _broadcast_scalar_for_mixture(phi, mu)
    return mu * phi


# ------------------------------------------------------------------------------


def _compute_r_from_mu_p(p: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Compute r = mu * (1 - p) / p with proper broadcasting for mixture models.

    Parameters
    ----------
    p : jnp.ndarray
        Success probability parameter.
    mu : jnp.ndarray
        Mean parameter.

    Returns
    -------
    jnp.ndarray
        Dispersion parameter r = mu * (1 - p) / p.
    """
    p = _broadcast_scalar_for_mixture(p, mu)
    return mu * (1 - p) / p


# ==============================================================================
# Parameterization Registry
# ==============================================================================

# Create singleton instances
_canonical = CanonicalParameterization()
_mean_prob = MeanProbParameterization()
_mean_odds = MeanOddsParameterization()
_hier_canonical = HierarchicalCanonicalParameterization()
_hier_mean_prob = HierarchicalMeanProbParameterization()
_hier_mean_odds = HierarchicalMeanOddsParameterization()

# Registry mapping names to parameterization instances
PARAMETERIZATIONS = {
    # Standard parameterizations
    "canonical": _canonical,
    "mean_prob": _mean_prob,
    "mean_odds": _mean_odds,
    # Hierarchical parameterizations (gene-specific p/phi with hyperprior)
    "hierarchical_canonical": _hier_canonical,
    "hierarchical_mean_prob": _hier_mean_prob,
    "hierarchical_mean_odds": _hier_mean_odds,
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
    # Hierarchical
    "HierarchicalCanonicalParameterization",
    "HierarchicalMeanProbParameterization",
    "HierarchicalMeanOddsParameterization",
    "PARAMETERIZATIONS",
    # Helper functions for derived parameter broadcasting
    "_broadcast_scalar_for_mixture",
    "_compute_r_from_mu_phi",
    "_compute_r_from_mu_p",
]
