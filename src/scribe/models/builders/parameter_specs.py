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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from multipledispatch import dispatch
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform

from scribe.stats.distributions import BetaPrime

if TYPE_CHECKING:
    from ..components.guide_families import GuideFamily
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


@dataclass
class ParamSpec:
    """
    Base class for parameter specifications.

    A ParamSpec defines everything needed to sample a parameter in both
    the model (prior) and guide (variational posterior). It encapsulates:

    - Name and shape information
    - Default distribution parameters
    - Whether the parameter is gene-specific or cell-specific
    - Whether the parameter is mixture-specific (per-component)
    - Which guide family to use for variational inference

    Parameters
    ----------
    name : str
        Name of the parameter (e.g., "p", "r", "mu", "phi", "gate", "p_capture").
        This is used as the sample site name in NumPyro.
    shape_dims : Tuple[str, ...]
        Symbolic shape dimensions. Options:
        - () : scalar parameter
        - ("n_genes",) : gene-specific parameter
        - ("n_cells",) : cell-specific parameter
    default_params : Tuple[float, ...]
        Default distribution parameters (distribution-specific).
        E.g., (1.0, 1.0) for Beta(alpha, beta).
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

    See Also
    --------
    BetaSpec : Beta-distributed parameters.
    LogNormalSpec : Log-normal distributed parameters.
    """

    name: str
    shape_dims: Tuple[str, ...]
    default_params: Tuple[float, ...]
    is_gene_specific: bool = False
    is_cell_specific: bool = False
    is_mixture: bool = False
    guide_family: Optional["GuideFamily"] = None

    def __post_init__(self):
        """Validate parameter specification."""
        if self.is_mixture and self.is_cell_specific:
            raise ValueError(
                f"Parameter '{self.name}': is_mixture and is_cell_specific "
                "cannot both be True. Cell-specific parameters are already "
                "per-cell and cannot be per-component."
            )

    @property
    def support(self) -> constraints.Constraint:
        """
        Return the constraint on this parameter's support (valid sampled values).

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


# ==============================================================================
# Constrained Parameter Types (direct sampling from constrained distributions)
# ==============================================================================

# ------------------------------------------------------------------------------
# Beta Distribution Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Scalar dropout probability with uniform prior
    >>> BetaSpec("p", (), (1.0, 1.0))
    >>> # Gene-specific gate with informative prior
    >>> BetaSpec("gate", ("n_genes",), (2.0, 8.0), is_gene_specific=True)
    """

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


# ------------------------------------------------------------------------------
# LogNormal Distribution Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Gene-specific dispersion
    >>> LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)
    """

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


@dataclass
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

    Examples
    --------
    >>> # Scalar odds ratio
    >>> BetaPrimeSpec("phi", (), (1.0, 1.0))

    Notes
    -----
    BetaPrime uses the odds-of-Beta convention. If p ~ Beta(α, β),
    then φ = (1-p)/p ~ BetaPrime(α, β).
    """

    @property
    def support(self) -> constraints.Constraint:
        """Return positive constraint for sampled values."""
        return constraints.positive

    # --------------------------------------------------------------------------

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Return constraints on alpha and beta parameters (must be positive)."""
        return {
            "concentration1": constraints.positive,
            "concentration0": constraints.positive,
        }


# ------------------------------------------------------------------------------
# Dirichlet Distribution Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Mixture weights for 5 components
    >>> DirichletSpec("weights", ("n_components",), (1.0,))
    """

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

    # --------------------------------------------------------------------------

    # post_init methd that checks DirichletSpec is not mixture-specific
    def __post_init__(self):
        """Validate parameter specification."""
        if self.is_mixture:
            raise ValueError(
                f"Parameter '{self.name}': is_mixture cannot be True. "
                "DirichletSpec is not mixture-specific. It defines the mixture weights."
            )


# ==============================================================================
# Unconstrained Parameter Types (Normal + NumPyro Transform)
# ==============================================================================


@dataclass
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

    transform: Transform = field(
        default_factory=lambda: dist.transforms.IdentityTransform()
    )
    constrained_name: Optional[str] = None

    def __post_init__(self):
        """Set constrained_name to name if not provided."""
        if self.constrained_name is None:
            object.__setattr__(self, "constrained_name", self.name)

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


# ------------------------------------------------------------------------------
# Normal with Sigmoid Transform Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Unconstrained dropout probability
    >>> SigmoidNormalSpec("p", (), (0.0, 1.0))

    Notes
    -----
    Default (0.0, 1.0) gives sigmoid(Normal(0, 1)) which has median ~0.5.
    """

    transform: Transform = field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )


# ------------------------------------------------------------------------------
# Normal with Exp Transform Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Unconstrained dispersion
    >>> ExpNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)
    """

    transform: Transform = field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )


# ------------------------------------------------------------------------------
# Normal with Softplus Transform Parameter Specification
# ------------------------------------------------------------------------------


@dataclass
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

    Examples
    --------
    >>> # Softplus-constrained dispersion (smoother near zero)
    >>> SoftplusNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)

    Notes
    -----
    Softplus is often preferred over exp when values near zero are expected,
    as it has a more stable gradient.
    """

    transform: Transform = field(
        default_factory=lambda: dist.transforms.SoftplusTransform()
    )


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
    # Get prior params from config or use defaults
    params = (
        getattr(model_config.priors, spec.name, None) or spec.default_params
    )
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
    params = (
        getattr(model_config.priors, spec.name, None) or spec.default_params
    )
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
    params = (
        getattr(model_config.priors, spec.name, None) or spec.default_params
    )
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
    params = (
        getattr(model_config.priors, spec.name, None) or spec.default_params
    )
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
        Sampled parameter value in constrained space.
    """
    params = (
        getattr(model_config.priors, spec.name, None) or spec.default_params
    )
    shape = resolve_shape(spec.shape_dims, dims, is_mixture=spec.is_mixture)

    # Create base Normal distribution
    if shape == ():
        base_dist = dist.Normal(*params)
    else:
        base_dist = dist.Normal(*params).expand(shape).to_event(len(shape))

    # Wrap with transform - handles Jacobian automatically
    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)

    # Sample directly in constrained space (transform applied internally)
    return numpyro.sample(spec.constrained_name, transformed_dist)
