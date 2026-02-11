"""
Parameter specifications for composable model building.

This module defines typed parameter specifications that encapsulate:

- Distribution type (Beta, LogNormal, transformed Normal, etc.)
- Shape dimensions (scalar, gene-specific, cell-specific)
- Constraint information (derived from distribution/transform)
- Guide family assignment (per-parameter variational family)

The specs use multiple dispatch (via multipledispatch) to route
sampling operations to the correct implementation without if-elif chains.

Classes
-------
ParamSpec
    Base class for all parameter specifications.
BetaSpec
    Beta-distributed parameter (support: (0, 1)).
LogNormalSpec
    Log-normal distributed parameter (support: (0, ∞)).
BetaPrimeSpec
    Beta-prime distributed parameter (support: (0, ∞)).
DirichletSpec
    Dirichlet-distributed parameter (support: simplex).
NormalWithTransformSpec
    Base class for transformed Normal parameters.
SigmoidNormalSpec
    Normal + sigmoid transform (support: (0, 1)).
ExpNormalSpec
    Normal + exp transform (support: (0, ∞)).
SoftplusNormalSpec
    Normal + softplus transform (support: (0, ∞)).
LatentSpec
    Base class for latent variable specs (VAE z); params → guide distribution.
GaussianLatentSpec
    Diagonal-Gaussian latent (loc, log_scale from encoder).

Functions
---------
sample_prior
    Sample a parameter from its prior distribution (dispatched).
resolve_shape
    Resolve symbolic shape dimensions to concrete shapes.

Examples
--------
>>> # Define parameters for standard NBDM
>>> params = [
...     BetaSpec("p", (), (1.0, 1.0)),
...     LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True),
... ]

See Also
--------
scribe.models.builders.model_builder : Uses specs to build models.
scribe.models.components.guide_families : Guide dispatch implementations.
"""

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Type,
)

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform
from pydantic import BaseModel, Field, model_validator, ConfigDict

from scribe.stats.distributions import BetaPrime

# Import GuideFamily at runtime (safe because guide_families doesn't import from
# builders)
from ..components.guide_families import GuideFamily

if TYPE_CHECKING:
    from ..config import ModelConfig


# ==============================================================================
# Derived Parameters (deterministic transformations of sampled parameters)
# ==============================================================================


@dataclass
class DerivedParam:
    """A parameter computed deterministically from other sampled parameters.

    Derived parameters are computed using `numpyro.deterministic` after
    the base parameters are sampled. They are useful for:

    - Computing r from p and mu in linked parameterization
    - Computing p from phi in odds_ratio parameterization
    - Any deterministic transformation of sampled values

    Parameters
    ----------
    name : str
        Name for the derived parameter. Used as the deterministic site name.
    compute : Callable[..., jnp.ndarray]
        Function that computes the derived value from dependencies.
        Should accept keyword arguments matching the dependency names.
    deps : List[str]
        Names of parameters this derived param depends on.
        These must be sampled before the derived param is computed.

    Examples
    --------
    >>> # Linked parameterization: r = mu * (1-p) / p
    >>> DerivedParam("r", lambda p, mu: mu * (1-p) / p, ["p", "mu"])
    >>>
    >>> # Odds ratio: p = 1 / (1 + phi)
    >>> DerivedParam("p", lambda phi: 1.0 / (1.0 + phi), ["phi"])
    """

    name: str
    compute: Callable[..., jnp.ndarray]
    deps: list


# ==============================================================================
# Shape Resolution
# ==============================================================================


def resolve_shape(
    shape_dims: Tuple[str, ...],
    dims: Dict[str, int],
    is_mixture: bool = False,
) -> Tuple[int, ...]:
    """
    Resolve symbolic shape dimensions to concrete integer shapes.

    Parameters
    ----------
    shape_dims : Tuple[str, ...]
        Symbolic dimensions like ("n_genes",) or ("n_cells",).
        Empty tuple () indicates a scalar parameter.
    dims : Dict[str, int]
        Mapping from dimension names to sizes.
        E.g., {"n_cells": 10000, "n_genes": 2000, "n_components": 3}.
    is_mixture : bool, default=False
        If True, prepend n_components dimension to the resolved shape.
        Requires "n_components" to be present in dims.

    Returns
    -------
    Tuple[int, ...]
        Concrete shape. Empty tuple for scalars.
        If is_mixture=True, shape is (n_components, ...).

    Examples
    --------
    >>> dims = {"n_cells": 100, "n_genes": 50}
    >>> resolve_shape(("n_genes",), dims)
    (50,)
    >>> resolve_shape((), dims)
    ()
    >>> dims_mix = {"n_cells": 100, "n_genes": 50, "n_components": 3}
    >>> resolve_shape(("n_genes",), dims_mix, is_mixture=True)
    (3, 50)
    >>> resolve_shape((), dims_mix, is_mixture=True)
    (3,)

    Raises
    ------
    KeyError
        If a dimension name is not found in dims.
        If is_mixture=True and "n_components" not in dims.
    """
    if not shape_dims:
        base_shape = ()
    else:
        base_shape = tuple(dims[dim] for dim in shape_dims)

    if is_mixture:
        if "n_components" not in dims:
            raise KeyError("n_components must be in dims when is_mixture=True")
        return (dims["n_components"],) + base_shape

    return base_shape


# ==============================================================================
# Base ParamSpec - shared structure for all parameter types
# ==============================================================================


class ParamSpec(BaseModel):
    """
    Base class for parameter specifications.

    A ParamSpec defines everything needed to sample a parameter in both
    the model (prior) and guide (variational posterior). It encapsulates:

    - Name and shape information
    - Distribution type and default parameters
    - Prior and guide hyperparameters (with validation)
    - Whether the parameter is gene-specific or cell-specific
    - Whether the parameter is mixture-specific (per-component)
    - Which guide family to use for variational inference

    Parameters
    ----------
    name : str
        Name of the parameter (e.g., "p", "r", "mu", "phi", "gate",
        "p_capture"). This is used as the sample site name in NumPyro.
    shape_dims : Tuple[str, ...]
        Symbolic shape dimensions. Options:
        - () : scalar parameter
        - ("n_genes",) : gene-specific parameter
        - ("n_cells",) : cell-specific parameter
    default_params : Tuple[float, ...]
        Default distribution parameters (distribution-specific).
        E.g., (1.0, 1.0) for Beta(alpha, beta).
    prior : Tuple[float, float], optional
        Prior hyperparameters for this parameter. Validated based on
        distribution type. If None, uses default_params.
    guide : Tuple[float, float], optional
        Guide hyperparameters for this parameter. Validated based on
        distribution type.
        If None, uses default_params.
    unconstrained : bool, default=False
        Whether this uses unconstrained parameterization (Normal + transform).
    is_gene_specific : bool, default=False
        If True, parameter has shape (n_genes,). Used for plate handling.
    is_cell_specific : bool, default=False
        If True, parameter has shape (n_cells,). Sampled inside cell plate.
    is_mixture : bool, default=False
        If True, parameter is mixture-specific (one value per component).
        When True, shape expands to include n_components dimension:
        - Scalar: () → (n_components,)
        - Gene-specific: (n_genes,) → (n_components, n_genes)
        Cannot be True if is_cell_specific=True (cell-specific params are
        already per-cell and cannot be per-component).
    guide_family : GuideFamily, optional
        Which variational family to use for this parameter.
        If None, defaults to MeanFieldGuide.

    Attributes
    ----------
    support : Constraint
        The constraint on the parameter's support (valid sampled values).
        Matches NumPyro's Distribution.support attribute.
    arg_constraints : Dict[str, Constraint]
        Constraints on the distribution's parameters.
        Matches NumPyro's Distribution.arg_constraints attribute.

    Raises
    ------
    ValueError
        If is_mixture=True and is_cell_specific=True (incompatible).
        If prior/guide hyperparameters are invalid for the distribution type.

    See Also
    --------
    BetaSpec : Beta-distributed parameters.
    LogNormalSpec : Log-normal distributed parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str
    shape_dims: Tuple[str, ...]
    default_params: Tuple[float, ...]
    prior: Optional[Tuple[float, float]] = Field(
        None,
        description="Prior hyperparameters (validated based on distribution)",
    )
    guide: Optional[Tuple[float, float]] = Field(
        None,
        description="Guide hyperparameters (validated based on distribution)",
    )
    unconstrained: bool = Field(
        False, description="Whether this uses unconstrained parameterization"
    )
    is_gene_specific: bool = Field(
        False, description="If True, parameter has shape (n_genes,)"
    )
    is_cell_specific: bool = Field(
        False, description="If True, parameter has shape (n_cells,)"
    )
    is_mixture: bool = Field(
        False, description="If True, parameter is mixture-specific"
    )
    guide_family: Optional[GuideFamily] = Field(
        None, description="Variational family for this parameter"
    )

    # --------------------------------------------------------------------------
    # Validation Methods
    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_spec(self) -> "ParamSpec":
        """Validate parameter specification."""
        if self.is_mixture and self.is_cell_specific:
            raise ValueError(
                f"Parameter '{self.name}': is_mixture and is_cell_specific "
                "cannot both be True. Cell-specific parameters are already "
                "per-cell and cannot be per-component."
            )
        return self

    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_hyperparameters(self) -> "ParamSpec":
        """
        Validate prior and guide hyperparameters based on distribution type.
        """
        # Get distribution type from subclass
        dist_type = self._get_distribution_type()

        if self.prior is not None:
            self._validate_hyperparameter(self.prior, "prior", dist_type)

        if self.guide is not None:
            self._validate_hyperparameter(self.guide, "guide", dist_type)

        return self

    # --------------------------------------------------------------------------

    def _get_distribution_type(self) -> Type:
        """Get the distribution type for this parameter spec."""
        # This will be overridden by subclasses
        raise NotImplementedError(
            "Subclasses must implement _get_distribution_type"
        )

    # --------------------------------------------------------------------------

    def _validate_hyperparameter(
        self, value: Tuple[float, float], name: str, dist_type: Type
    ) -> None:
        """Validate hyperparameter tuple based on distribution type."""
        if len(value) != 2:
            raise ValueError(
                f"Parameter '{self.name}': {name} "
                f"must be a 2-tuple, got {len(value)}"
            )

        # Beta and BetaPrime: both values must be positive
        if dist_type in (dist.Beta, BetaPrime):
            if any(x <= 0 for x in value):
                raise ValueError(
                    f"Parameter '{self.name}': {name} values must be positive "
                    f"for {dist_type.__name__}, got {value}"
                )

        # LogNormal and Normal: scale (second value) must be positive, location
        # can be any float
        elif dist_type in (dist.LogNormal, dist.Normal):
            if value[1] <= 0:
                raise ValueError(
                    f"Parameter '{self.name}': {name} scale (second value) must be "
                    f"positive for {dist_type.__name__}, got {value}"
                )

        # Dirichlet: handled separately (variable length tuple)
        # No validation needed here as it's handled in DirichletSpec

    # --------------------------------------------------------------------------

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the constraint on this parameter's support (valid sampled
        values).

        Matches NumPyro's Distribution.support attribute.

        Returns
        -------
        Constraint
            NumPyro constraint object defining valid parameter values.

        Raises
        ------
        NotImplementedError
            If called on base class.
        """
        raise NotImplementedError("Subclasses must implement support property")

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Return constraints on the distribution's parameters.

        Matches NumPyro's Distribution.arg_constraints attribute.

        Returns
        -------
        Dict[str, Constraint]
            Dictionary mapping parameter names to their constraints.

        Raises
        ------
        NotImplementedError
            If called on base class.
        """
        raise NotImplementedError(
            "Subclasses must implement arg_constraints property"
        )

    # --------------------------------------------------------------------------
    # Amortized guide (optional; only specs that support amortization implement)
    # --------------------------------------------------------------------------

    def make_amortized_guide_dist(self, var_params: Dict[str, jnp.ndarray]):
        """
        Build guide distribution from amortizer output (AmortizedOutput.params).

        Only BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, and ExpNormalSpec
        support amortization. Others raise NotImplementedError.

        Parameters
        ----------
        var_params : Dict[str, jnp.ndarray]
            Amortizer output params (see AmortizedOutput contract). Constrained:
            keys "alpha", "beta". Unconstrained: "loc", "log_scale".

        Returns
        -------
        Distribution
            NumPyro distribution to sample from.

        Raises
        ------
        NotImplementedError
            If this spec does not support amortized guides.
        """
        raise NotImplementedError(
            f"Amortized guides are not supported for spec type {type(self).__name__}. "
            "Supported specs: BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, ExpNormalSpec."
        )

    @property
    def amortized_guide_sample_site(self) -> str:
        """
        Sample site name for amortized guide (e.g. spec.name or
        constrained_name).
        """
        raise NotImplementedError(
            f"Amortized guides are not supported for spec type "
            f"{type(self).__name__}."
        )


# ==============================================================================
# Constrained Parameter Types (direct sampling from constrained distributions)
# ==============================================================================

# ------------------------------------------------------------------------------
# Beta Distribution Parameter Specification
# ------------------------------------------------------------------------------


class BetaSpec(ParamSpec):
    """Parameter with Beta(alpha, beta) distribution.

    Support: (0, 1)

    The Beta distribution is commonly used for:
        - Dropout probability p
        - Zero-inflation gate probability
        - Capture probability p_capture

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (alpha, beta) parameters. E.g., (1.0, 1.0) for uniform.
    prior : Tuple[float, float], optional
        Prior hyperparameters (alpha, beta). Both must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (alpha, beta). Both must be positive.

    Examples
    --------
    >>> # Scalar dropout probability with uniform prior
    >>> BetaSpec("p", (), (1.0, 1.0))
    >>> # Gene-specific gate with informative prior
    >>> BetaSpec("gate", ("n_genes",), (2.0, 8.0), is_gene_specific=True)
    """

    def _get_distribution_type(self) -> Type:
        """Return Beta distribution type."""
        return dist.Beta

    @property
    def support(self) -> constraints.Constraint:
        """Return unit_interval constraint for sampled values."""
        return dist.Beta.support  # constraints.unit_interval

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Return constraints on alpha and beta parameters (must be positive).
        """
        # {"concentration1": positive, "concentration0": positive}
        return dist.Beta.arg_constraints

    def make_amortized_guide_dist(self, var_params: Dict[str, jnp.ndarray]):
        """
        Build Beta guide from amortizer output (alpha, beta in constrained
        space).
        """
        return dist.Beta(var_params["alpha"], var_params["beta"])

    @property
    def amortized_guide_sample_site(self) -> str:
        """Sample site name for amortized guide."""
        return self.name


# ------------------------------------------------------------------------------
# LogNormal Distribution Parameter Specification
# ------------------------------------------------------------------------------


class LogNormalSpec(ParamSpec):
    """Parameter with LogNormal(loc, scale) distribution.

    Support: (0, ∞)

    The LogNormal distribution is commonly used for:
        - Dispersion parameter r
        - Mean expression mu

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (loc, scale) parameters in log-space.
        E.g., (0.0, 1.0) for median=1, spread of ~1 order of magnitude.
    prior : Tuple[float, float], optional
        Prior hyperparameters (loc, scale). Scale must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (loc, scale). Scale must be positive.

    Examples
    --------
    >>> # Gene-specific dispersion
    >>> LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)
    """

    def _get_distribution_type(self) -> Type:
        """Return LogNormal distribution type."""
        return dist.LogNormal

    @property
    def support(self) -> constraints.Constraint:
        """Return positive constraint for sampled values."""
        return dist.LogNormal.support  # constraints.positive

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Return constraints on loc (real) and scale (positive) parameters."""
        return (
            dist.LogNormal.arg_constraints
        )  # {"loc": real, "scale": positive}


# ------------------------------------------------------------------------------
# BetaPrime Distribution Parameter Specification
# ------------------------------------------------------------------------------


class BetaPrimeSpec(ParamSpec):
    """Parameter with BetaPrime(alpha, beta) distribution.

    Support: (0, ∞)

    The BetaPrime distribution is used for:
        - Odds ratio phi in the odds_ratio parameterization

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (alpha, beta) parameters.
    prior : Tuple[float, float], optional
        Prior hyperparameters (alpha, beta). Both must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (alpha, beta). Both must be positive.

    Examples
    --------
    >>> # Scalar odds ratio
    >>> BetaPrimeSpec("phi", (), (1.0, 1.0))

    Notes
    -----
    BetaPrime uses the odds-of-Beta convention. If p ~ Beta(α, β),
    then φ = (1-p)/p ~ BetaPrime(α, β).
    """

    def _get_distribution_type(self) -> Type:
        """Return BetaPrime distribution type."""
        return BetaPrime

    @property
    def support(self) -> constraints.Constraint:
        """Return positive constraint for sampled values."""
        return constraints.positive

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Return constraints on alpha and beta parameters (must be positive).
        """
        return {
            "concentration1": constraints.positive,
            "concentration0": constraints.positive,
        }

    def make_amortized_guide_dist(self, var_params: Dict[str, jnp.ndarray]):
        """
        Build BetaPrime guide from amortizer output (alpha, beta in constrained
        space).
        """
        return BetaPrime(var_params["alpha"], var_params["beta"])

    @property
    def amortized_guide_sample_site(self) -> str:
        """Sample site name for amortized guide."""
        return self.name


# ------------------------------------------------------------------------------
# Dirichlet Distribution Parameter Specification
# ------------------------------------------------------------------------------


class DirichletSpec(ParamSpec):
    """Parameter with Dirichlet(concentration) distribution.

    Support: simplex (values sum to 1)

    The Dirichlet distribution is used for:
        - Mixture weights in mixture models
        - Compositional parameters

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions. Should include component dimension.
    default_params : Tuple[float, ...]
        Default concentration parameters.
    prior : Tuple[float, ...], optional
        Prior hyperparameters (concentration). All values must be positive.
        Variable length tuple.
    guide : Tuple[float, ...], optional
        Guide hyperparameters (concentration). All values must be positive.
        Variable length tuple.

    Examples
    --------
    >>> # Mixture weights for 5 components
    >>> DirichletSpec("weights", ("n_components",), (1.0,))
    """

    # Override prior/guide to allow variable length tuples for Dirichlet
    prior: Optional[Tuple[float, ...]] = Field(
        None,
        description="Prior hyperparameters (concentration, variable length)",
    )
    guide: Optional[Tuple[float, ...]] = Field(
        None,
        description="Guide hyperparameters (concentration, variable length)",
    )

    # --------------------------------------------------------------------------

    def _get_distribution_type(self) -> Type:
        """Return Dirichlet distribution type."""
        return dist.Dirichlet

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_dirichlet_spec(self) -> "DirichletSpec":
        """Validate Dirichlet-specific constraints."""
        if self.is_mixture:
            raise ValueError(
                f"Parameter '{self.name}': is_mixture cannot be True. "
                "DirichletSpec is not mixture-specific. "
                "It defines the mixture weights."
            )
        return self

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_dirichlet_hyperparameters(self) -> "DirichletSpec":
        """
        Validate Dirichlet hyperparameters (variable length, all positive).
        """
        if self.prior is not None:
            if len(self.prior) < 2:
                raise ValueError(
                    f"Parameter '{self.name}': "
                    "prior must have at least 2 elements "
                    f"for Dirichlet, got {len(self.prior)}"
                )
            if any(x <= 0 for x in self.prior):
                raise ValueError(
                    f"Parameter '{self.name}': "
                    "prior values must be positive "
                    f"for Dirichlet, got {self.prior}"
                )

        if self.guide is not None:
            if len(self.guide) < 2:
                raise ValueError(
                    f"Parameter '{self.name}': "
                    "guide must have at least 2 elements "
                    f"for Dirichlet, got {len(self.guide)}"
                )
            if any(x <= 0 for x in self.guide):
                raise ValueError(
                    f"Parameter '{self.name}': "
                    "guide values must be positive "
                    f"for Dirichlet, got {self.guide}"
                )

        return self

    @property
    def support(self) -> constraints.Constraint:
        """Return simplex constraint for sampled values."""
        return dist.Dirichlet.support  # constraints.simplex

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Return constraints on concentration parameter (must be positive)."""
        return (
            dist.Dirichlet.arg_constraints
        )  # {"concentration": independent(positive, 1)}


# ==============================================================================
# Unconstrained Parameter Types (Normal + NumPyro Transform)
# ==============================================================================


class NormalWithTransformSpec(ParamSpec):
    """Parameter sampled from Normal, then transformed to constrained space.

    This is the base class for unconstrained parameterizations. Instead of
    sampling directly from a constrained distribution (e.g., Beta), we:

    1. Sample from Normal(loc, scale)
    2. Apply a transform (e.g., sigmoid, exp) to get constrained values

    Using NumPyro Transform objects provides:
    - Automatic constraint derivation via transform.codomain
    - Proper Jacobian computation for ELBO
    - Native TransformedDistribution integration

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (loc, scale) for the base Normal.
    prior : Tuple[float, float], optional
        Prior hyperparameters (loc, scale). Scale must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (loc, scale). Scale must be positive.
    transform : Transform
        NumPyro transform object (e.g., SigmoidTransform()).
    constrained_name : str, optional
        Name for the constrained (transformed) parameter.
        Defaults to the base name.

    See Also
    --------
    SigmoidNormalSpec : Normal + sigmoid.
    ExpNormalSpec : Normal + exp.
    SoftplusNormalSpec : Normal + softplus.

    References
    ----------
    NumPyro transforms:
    https://num.pyro.ai/en/stable/distributions.html#transforms
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.IdentityTransform()
    )
    constrained_name: Optional[str] = None

    # --------------------------------------------------------------------------

    def _get_distribution_type(self) -> Type:
        """
        Return Normal distribution type (base distribution before transform).
        """
        return dist.Normal

    # --------------------------------------------------------------------------

    @model_validator(mode="after")
    def set_constrained_name(self) -> "NormalWithTransformSpec":
        """Set constrained_name to name if not provided."""
        if self.constrained_name is None:
            object.__setattr__(self, "constrained_name", self.name)
        return self

    # --------------------------------------------------------------------------

    @property
    def support(self) -> constraints.Constraint:
        """
        Derive support directly from transform's codomain.

        The support of a transformed distribution is determined by the
        output space (codomain) of the transform applied to the base Normal.

        Returns
        -------
        Constraint
            The output constraint of the transform (e.g., unit_interval for
            sigmoid).
        """
        return self.transform.codomain

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Return constraints on the base Normal's parameters.

        For transformed Normal distributions, the underlying distribution
        is Normal(loc, scale), so:
            - loc can be any real number
            - scale must be positive

        Returns
        -------
        Dict[str, Constraint]
            Constraints matching Normal distribution parameters.
        """
        return dist.Normal.arg_constraints  # {"loc": real, "scale": positive}

    def make_amortized_guide_dist(self, var_params: Dict[str, jnp.ndarray]):
        """
        Build Normal+transform guide from amortizer output (loc, log_scale in
        log-space).
        """
        loc = var_params["loc"]
        scale = jnp.exp(var_params["log_scale"])
        base_dist = dist.Normal(loc, scale)
        return dist.TransformedDistribution(base_dist, self.transform)

    @property
    def amortized_guide_sample_site(self) -> str:
        """Sample site name for amortized guide (constrained parameter name)."""
        return self.constrained_name


# ------------------------------------------------------------------------------
# Normal with Sigmoid Transform Parameter Specification
# ------------------------------------------------------------------------------


class SigmoidNormalSpec(NormalWithTransformSpec):
    """Normal + SigmoidTransform. For parameters in (0, 1).

    Samples from Normal, then applies sigmoid to constrain to (0, 1). Useful
    when you want the unconstrained parameterization but need values in the unit
    interval.

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (loc, scale) for the base Normal.
        Note: These are in unconstrained space.
    prior : Tuple[float, float], optional
        Prior hyperparameters (loc, scale). Scale must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (loc, scale). Scale must be positive.

    Examples
    --------
    >>> # Unconstrained dropout probability
    >>> SigmoidNormalSpec("p", (), (0.0, 1.0))

    Notes
    -----
    Default (0.0, 1.0) gives sigmoid(Normal(0, 1)) which has median ~0.5.
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )


# ------------------------------------------------------------------------------
# Normal with Exp Transform Parameter Specification
# ------------------------------------------------------------------------------


class ExpNormalSpec(NormalWithTransformSpec):
    """Normal + ExpTransform. For parameters in (0, ∞).

    Samples from Normal, then applies exp to constrain to positive values.
    Equivalent to LogNormal in the unconstrained parameterization.

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (loc, scale) for the base Normal (log-space params).
    prior : Tuple[float, float], optional
        Prior hyperparameters (loc, scale). Scale must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (loc, scale). Scale must be positive.

    Examples
    --------
    >>> # Unconstrained dispersion
    >>> ExpNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )


# ------------------------------------------------------------------------------
# Normal with Softplus Transform Parameter Specification
# ------------------------------------------------------------------------------


class SoftplusNormalSpec(NormalWithTransformSpec):
    """Normal + SoftplusTransform. For parameters in (0, ∞).

    Samples from Normal, then applies softplus to constrain to positive.
    Softplus is smoother than exp near zero: softplus(x) = log(1 + exp(x)).

    Parameters
    ----------
    name : str
        Parameter name.
    shape_dims : Tuple[str, ...]
        Shape dimensions.
    default_params : Tuple[float, float]
        Default (loc, scale) for the base Normal.
    prior : Tuple[float, float], optional
        Prior hyperparameters (loc, scale). Scale must be positive.
    guide : Tuple[float, float], optional
        Guide hyperparameters (loc, scale). Scale must be positive.

    Examples
    --------
    >>> # Softplus-constrained dispersion (smoother near zero)
    >>> SoftplusNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)

    Notes
    -----
    Softplus is often preferred over exp when values near zero are expected,
    as it has a more stable gradient.
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SoftplusTransform()
    )


# ==============================================================================
# LatentSpec — guide distribution for VAE latent z
# ==============================================================================


class LatentSpec(BaseModel):
    """Base class for latent variable specifications (VAE z).

    Encapsulates the mapping from encoder output (params dict) to the NumPyro
    guide distribution for the latent z, mirroring
    ParamSpec.make_amortized_guide_dist. Subclasses implement make_guide_dist
    for concrete latent families (e.g. Gaussian).

    Parameters
    ----------
    sample_site : str
        NumPyro sample site name for the latent (e.g. "z").
    flow : object, optional
        Optional normalizing flow (e.g. ``FlowChain``) to use as a learned
        prior on z.  When ``None`` (default) the prior is a standard
        distribution (e.g. ``Normal(0, I)``).  When set, the model builder
        wraps the flow in a ``FlowDistribution`` so that
        ``z ~ FlowDistribution(flow, base=Normal(0, I))``.  The flow
        parameters are learned jointly during SVI via NumPyro's param store.

    See Also
    --------
    GaussianLatentSpec : Diagonal-Gaussian latent (loc, log_scale from encoder).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    sample_site: str = Field(
        default="z", description="NumPyro sample site name for z"
    )
    flow: Optional[Any] = Field(
        default=None,
        description=(
            "Optional prior flow (e.g. FlowChain). When set, the model "
            "samples z from FlowDistribution(flow, base=Normal(0,I)) "
            "instead of the standard prior."
        ),
    )

    def make_guide_dist(
        self, var_params: Dict[str, jnp.ndarray]
    ) -> dist.Distribution:
        """Build guide distribution from encoder output (var_params).

        Subclasses must implement this. var_params is a dict assembled by the
        guide builder from encoder output (e.g. {"loc": ..., "log_scale": ...}).

        Parameters
        ----------
        var_params : Dict[str, jnp.ndarray]
            Encoder output as a dict; keys depend on the latent family.

        Returns
        -------
        dist.Distribution
            NumPyro distribution to use for numpyro.sample(sample_site, dist).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement make_guide_dist()"
        )

    def make_prior_dist(self) -> dist.Distribution:
        """Build the prior distribution for z in the model.

        Returns
        -------
        dist.Distribution
            NumPyro distribution for ``numpyro.sample(sample_site, dist)``
            in the model (e.g. standard Normal for a Gaussian latent).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement make_prior_dist()"
        )


# ------------------------------------------------------------------------------
# Gaussian Latent Specification
# ------------------------------------------------------------------------------


class GaussianLatentSpec(LatentSpec):
    """Diagonal-Gaussian latent posterior (encoder outputs loc, log_scale).

    Encoder is assumed to output log-variance (same as legacy VAE), so
    scale = exp(0.5 * log_scale) for the Normal distribution.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    sample_site : str
        NumPyro sample site name (default "z").
    flow : object, optional
        Optional normalizing flow for a learned prior on z.  Inherited
        from ``LatentSpec``.  When set, the model builder uses
        ``FlowDistribution(flow, base=Normal(0, I))`` instead of the
        standard ``Normal(0, I)`` prior.  The flow's ``features``
        attribute must equal ``latent_dim``.
    """

    latent_dim: int = Field(..., description="Dimensionality of the latent z")
    sample_site: str = Field(
        default="z", description="NumPyro sample site name for z"
    )

    def make_guide_dist(
        self, var_params: Dict[str, jnp.ndarray]
    ) -> dist.Distribution:
        """Build Normal(loc, scale).to_event(1) from encoder output.

        var_params must have keys "loc" and "log_scale". log_scale is
        interpreted as log-variance (legacy convention): scale = exp(0.5 *
        log_scale).
        """
        loc = var_params["loc"]
        log_scale = var_params["log_scale"]
        scale = jnp.exp(0.5 * log_scale)
        return dist.Normal(loc, scale).to_event(1)

    def make_prior_dist(self) -> dist.Distribution:
        """Standard Normal prior: z ~ N(0, I).to_event(1)."""
        return dist.Normal(
            jnp.zeros(self.latent_dim),
            jnp.ones(self.latent_dim),
        ).to_event(1)


# ==============================================================================
# Multiple Dispatch for Prior Sampling
# ==============================================================================


# ------------------------------------------------------------------------------
# Beta Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(BetaSpec, dict, object)
def sample_prior(
    spec: BetaSpec, dims: Dict[str, int], model_config: "ModelConfig"
) -> jnp.ndarray:
    """Sample from Beta prior.

    Parameters
    ----------
    spec : BetaSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with prior hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value.
    """
    # Get prior params from spec or use defaults
    params = spec.prior if spec.prior is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    if shape == ():
        # Scalar parameter
        return numpyro.sample(spec.name, dist.Beta(*params))
    else:
        # Array parameter (e.g., gene-specific, or mixture-specific)
        return numpyro.sample(
            spec.name, dist.Beta(*params).expand(shape).to_event(len(shape))
        )


# ------------------------------------------------------------------------------
# LogNormal Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(LogNormalSpec, dict, object)
def sample_prior(
    spec: LogNormalSpec, dims: Dict[str, int], model_config: "ModelConfig"
) -> jnp.ndarray:
    """Sample from LogNormal prior.

    Parameters
    ----------
    spec : LogNormalSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with prior hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value.
    """
    params = spec.prior if spec.prior is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    return numpyro.sample(
        spec.name, dist.LogNormal(*params).expand(shape).to_event(len(shape))
    )


# ------------------------------------------------------------------------------
# BetaPrime Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(BetaPrimeSpec, dict, object)
def sample_prior(
    spec: BetaPrimeSpec, dims: Dict[str, int], model_config: "ModelConfig"
) -> jnp.ndarray:
    """Sample from BetaPrime prior.

    Parameters
    ----------
    spec : BetaPrimeSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with prior hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value.
    """
    params = spec.prior if spec.prior is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    if shape == ():
        return numpyro.sample(spec.name, BetaPrime(*params))
    else:
        return numpyro.sample(
            spec.name, BetaPrime(*params).expand(shape).to_event(len(shape))
        )


# ------------------------------------------------------------------------------
# Dirichlet Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(DirichletSpec, dict, object)
def sample_prior(
    spec: DirichletSpec, dims: Dict[str, int], model_config: "ModelConfig"
) -> jnp.ndarray:
    """Sample from Dirichlet prior.

    Parameters
    ----------
    spec : DirichletSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with prior hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value on the simplex.
    """
    params = spec.prior if spec.prior is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # For Dirichlet, concentration is a vector
    concentration = (
        jnp.full(shape, params[0]) if len(params) == 1 else jnp.array(params)
    )

    return numpyro.sample(spec.name, dist.Dirichlet(concentration))


# ------------------------------------------------------------------------------
# Normal with Transform Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(NormalWithTransformSpec, dict, object)
def sample_prior(
    spec: NormalWithTransformSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample from TransformedDistribution(Normal, transform).

    Using TransformedDistribution gives us:
    - Automatic Jacobian adjustment in the log_prob
    - Clean integration with NumPyro's inference
    - The sample is already in constrained space

    This works for both mean-field and low-rank guides since both now use
    TransformedDistribution in the guide, ensuring consistent behavior.

    Parameters
    ----------
    spec : NormalWithTransformSpec
        The parameter specification (or subclass like SigmoidNormalSpec).
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration with prior hyperparameters.

    Returns
    -------
    jnp.ndarray
        Sampled parameter value in constrained space (transform applied via
        TransformedDistribution).
    """
    params = spec.prior if spec.prior is not None else spec.default_params
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # Create base Normal distribution
    if shape == ():
        base_dist = dist.Normal(*params)
    else:
        base_dist = dist.Normal(*params).expand(shape).to_event(len(shape))

    # Wrap with transform - handles Jacobian automatically
    # This works for both mean-field and low-rank guides since both now use
    # TransformedDistribution in the guide
    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)

    # Sample directly in constrained space (transform applied internally)
    return numpyro.sample(spec.constrained_name, transformed_dist)
