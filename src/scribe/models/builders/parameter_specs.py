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
PositiveNormalSpec
    Normal + configurable positive transform (support: (0, ∞)).
SoftplusNormalSpec
    Normal + softplus transform (support: (0, ∞)).
GammaSpec
    Gamma-distributed parameter (support: (0, ∞)); NEG prior psi/zeta layers.
HalfCauchySpec
    Half-Cauchy distributed parameter; horseshoe shrinkage scales.
InverseGammaSpec
    Inverse-Gamma distributed parameter; horseshoe slab.
NEGHierarchicalSigmoidNormalSpec
    Gene-level NEG prior with Sigmoid transform (p, gate).
NEGHierarchicalPositiveNormalSpec
    Gene-level NEG prior with configurable positive transform (phi).
NEGDatasetPositiveNormalSpec
    Dataset-level NEG prior with configurable positive transform (mu).
NEGDatasetSigmoidNormalSpec
    Dataset-level NEG prior with Sigmoid transform (p, gate).
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
    is_dataset: bool = False,
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
    is_dataset : bool, default=False
        If True, prepend n_datasets dimension to the resolved shape.
        Requires "n_datasets" to be present in dims.  The dataset
        dimension is prepended **after** the mixture dimension when
        both are True: ``(n_components, n_datasets, ...)``.

    Returns
    -------
    Tuple[int, ...]
        Concrete shape. Empty tuple for scalars.
        If is_mixture=True, shape is (n_components, ...).
        If is_dataset=True, shape is (n_datasets, ...).

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
    >>> dims_ds = {"n_cells": 100, "n_genes": 50, "n_datasets": 2}
    >>> resolve_shape(("n_genes",), dims_ds, is_dataset=True)
    (2, 50)
    >>> resolve_shape((), dims_ds, is_dataset=True)
    (2,)

    Raises
    ------
    KeyError
        If a dimension name is not found in dims.
        If is_mixture=True and "n_components" not in dims.
        If is_dataset=True and "n_datasets" not in dims.
    """
    if not shape_dims:
        base_shape = ()
    else:
        base_shape = tuple(dims[dim] for dim in shape_dims)

    # Prepend dataset dimension first (innermost batch axis)
    if is_dataset:
        if "n_datasets" not in dims:
            raise KeyError("n_datasets must be in dims when is_dataset=True")
        base_shape = (dims["n_datasets"],) + base_shape

    # Prepend mixture dimension (outermost batch axis)
    if is_mixture:
        if "n_components" not in dims:
            raise KeyError("n_components must be in dims when is_mixture=True")
        base_shape = (dims["n_components"],) + base_shape

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
    is_dataset: bool = Field(
        False,
        description=(
            "If True, parameter is per-dataset in a multi-dataset model. "
            "Shape expands to include n_datasets dimension: "
            "scalar () -> (n_datasets,), gene-specific (n_genes,) -> "
            "(n_datasets, n_genes)."
        ),
    )
    guide_family: Optional[GuideFamily] = Field(
        None, description="Variational family for this parameter"
    )

    # --------------------------------------------------------------------------
    # Alias support
    # --------------------------------------------------------------------------

    @property
    def alias_names(self) -> list:
        """Additional name prefixes used to match variational-parameter keys.

        Subclasses override this when the guide creates parameters under a
        different name than ``self.name``.  The matching helpers in
        ``_dataset.py`` check both ``self.name`` and every alias returned
        here so that cell-specific / dataset-specific subsetting works
        correctly for reparameterised latent variables.

        Returns
        -------
        list of str
            Empty by default; subclasses may return e.g. ``["eta_capture"]``.
        """
        return []

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
        if self.is_dataset and self.is_cell_specific:
            raise ValueError(
                f"Parameter '{self.name}': is_dataset and is_cell_specific "
                "cannot both be True. Cell-specific parameters are sampled "
                "per-cell and are indexed into per-dataset params at the "
                "likelihood level."
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
        # Dirichlet uses variable-length concentration vectors;
        # its own validator (validate_dirichlet_hyperparameters) handles it.
        if dist_type is dist.Dirichlet:
            return
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

        Only BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, and PositiveNormalSpec
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
            "Supported specs: BetaSpec, BetaPrimeSpec, SigmoidNormalSpec, PositiveNormalSpec."
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
    PositiveNormalSpec : Normal + configurable positive transform (exp or
    softplus).
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
# Normal with Configurable Positive Transform Parameter Specification
# ------------------------------------------------------------------------------


class PositiveNormalSpec(NormalWithTransformSpec):
    """Normal + configurable positive transform. For parameters in (0, ∞).

    Samples from Normal, then applies a positive transform (default:
    ExpTransform, overridable by the factory to SoftplusTransform for
    numerical stability) to constrain to positive values.
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
    >>> PositiveNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True)
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
# Data-Informed Mean Anchoring Specification
# ==============================================================================


class AnchoredNormalSpec(NormalWithTransformSpec):
    """NormalWithTransformSpec with per-gene anchor centers from data.

    Used for the data-informed mean anchoring prior where each gene's
    prior center is derived from the observed sample mean:

        log(mu_g) ~ N(log(u_bar_g / nu_bar), sigma^2)

    Per-gene centers are stored as a numpy/JAX array for efficiency;
    Python tuples are also accepted but converted on first use.

    Parameters
    ----------
    name : str
        Parameter name (e.g. "log_mu_loc", "log_mu_dataset_loc").
    shape_dims : Tuple[str, ...]
        Shape dimensions (typically ("n_genes",)).
    default_params : Tuple[float, float]
        Fallback (loc, scale) used only if anchor_centers is empty.
    anchor_centers : array-like
        Per-gene log-centers: log(u_bar_g / nu_bar + epsilon).
        Accepts numpy arrays, JAX arrays, or tuples of floats.
    anchor_sigma : float
        Log-scale standard deviation for the anchoring prior.
    """

    anchor_centers: Any = Field(
        default_factory=tuple,
        description="Per-gene log-space anchor centers (array-like).",
    )
    anchor_sigma: float = Field(
        0.3,
        gt=0.0,
        description="Log-scale sigma for the anchoring prior.",
    )


# ==============================================================================
# Biology-Informed Capture Probability Specification
# ==============================================================================


class BiologyInformedCaptureSpec(ParamSpec):
    """Capture prior anchored to library size via total mRNA per cell.

    Instead of an uninformative Beta(1,1) or BetaPrime(1,1) prior, this spec
    defines the capture probability through the latent variable

        eta_c = log(M_c / L_c)

    with a Normal prior whose mean depends on the observed library size:

        eta_c ~ N(log M_0 - log L_c, sigma_M^2)

    The capture parameter is then computed via exact (no-approximation)
    transformations:

        phi_capture_c = exp(eta_c) - 1    (mean_odds parameterization)
        p_capture_c   = exp(-eta_c)       (canonical / mean_prob)

    For the data-driven variant (``mu_eta_prior`` is not None), log M_0
    is replaced by a learned per-dataset latent variable with hierarchical
    shrinkage toward a shared population mean:

        mu_eta_pop  ~ N(log M_0, sigma_mu^2)
        mu_eta^{(d)} ~ Shrinkage(mu_eta_pop, ...)   [shape (D,)]
        eta_c       ~ TruncatedNormal(mu_eta^{(d_c)} - log L_c, sigma_M^2, low=0)

    The shrinkage type (Gaussian, Horseshoe, NEG) is selected via
    ``mu_eta_prior``.

    Parameters
    ----------
    name : str
        Capture parameter name ("phi_capture" or "p_capture").
    log_M0 : float
        log(M_0) where M_0 is the expected total mRNA per cell.
    sigma_M : float
        Log-scale std-dev of cell-to-cell mRNA variation.
    mu_eta_prior : str or None
        Hierarchical prior type for per-dataset mu_eta.  One of
        ``"gaussian"``, ``"horseshoe"``, ``"neg"``, or ``None`` (fixed M_0).
    sigma_mu : float
        Prior std-dev on the population-level mu_eta_pop parameter.
    use_phi_capture : bool
        If True, output phi_capture = exp(eta) - 1.
        If False, output p_capture = exp(-eta).

    See Also
    --------
    paper/_capture_prior.qmd : Full derivation and biological motivation.
    """

    log_M0: float = Field(
        ..., description="log(M_0): log of expected total mRNA per cell."
    )
    sigma_M: float = Field(
        ..., gt=0, description="Log-scale cell-to-cell mRNA variation."
    )
    mu_eta_prior: Optional[str] = Field(
        None,
        description=(
            "Hierarchical prior type for per-dataset mu_eta.  "
            "None = fixed M_0; 'gaussian', 'horseshoe', 'neg' = "
            "learn per-dataset mu_eta with shrinkage."
        ),
    )
    sigma_mu: float = Field(
        1.0,
        gt=0,
        description="Prior std-dev on population-level mu_eta_pop.",
    )
    use_phi_capture: bool = Field(
        ...,
        description=(
            "True → phi_capture = exp(eta)-1; " "False → p_capture = exp(-eta)."
        ),
    )

    @property
    def data_driven(self) -> bool:
        """Whether mu_eta is learned (True) or fixed (False)."""
        return self.mu_eta_prior is not None

    @property
    def alias_names(self) -> list:
        """Latent variable aliases sampled by the guide.

        The spec name is ``phi_capture`` or ``p_capture``, but the guide
        creates variational parameters keyed on ``eta_capture`` (legacy
        truncated-normal: ``eta_capture_loc`` / ``eta_capture_scale``)
        or ``eta_capture_raw`` (softplus-normal: ``eta_capture_raw_loc``
        / ``eta_capture_raw_scale``).  Exposing both aliases ensures
        cell-specific subsetting picks up whichever set of keys is
        present when splitting by dataset.
        """
        return ["eta_capture", "eta_capture_raw"]

    def _get_distribution_type(self) -> Type:
        """Return Normal as the base distribution type for eta."""
        return dist.Normal

    @property
    def support(self) -> constraints.Constraint:
        """Capture parameter is positive (phi) or unit-interval (p)."""
        if self.use_phi_capture:
            return constraints.positive
        return constraints.unit_interval

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Normal base distribution constraints for eta."""
        return dist.Normal.arg_constraints


# ==============================================================================
# Hierarchical Normal + Transform Specifications
# ==============================================================================
#
# These specs define gene-specific parameters whose prior is determined by
# hyperparameters that are themselves sampled.  The generative process is:
#
#     hyper_loc   ~ prior (sampled as a global parameter)
#     hyper_scale ~ prior (sampled as a global parameter, positive)
#     unconstrained_g ~ Normal(hyper_loc, hyper_scale)   for each gene g
#     param_g = transform(unconstrained_g)
#
# When all genes share the same hyperparameters, the gene-specific parameters
# are exchangeable draws from the same population distribution — a classical
# Bayesian hierarchical structure.  This relaxes the shared-p assumption of
# the Dirichlet-Multinomial factorization while retaining soft shrinkage
# toward a common value.
# ==============================================================================


class HierarchicalNormalWithTransformSpec(NormalWithTransformSpec):
    """
    Hierarchical Normal + Transform: gene-specific parameter with learned prior.

    Instead of using fixed prior hyperparameters, this spec draws its
    Normal(loc, scale) parameters from other sampled hyperparameters.
    The generative model is::

        hyper_loc   ~ some prior  (global parameter)
        hyper_scale ~ some prior  (global parameter, positive)
        param_g     = transform(Normal(hyper_loc, hyper_scale))  per gene

    This enables gene-specific parameters (e.g., p_g, phi_g) whose prior
    is learned from data, providing adaptive shrinkage.

    Parameters
    ----------
    name : str
        Parameter name (e.g., ``"p"``).  Used as the NumPyro sample site.
    shape_dims : Tuple[str, ...]
        Shape dimensions.  Typically ``("n_genes",)`` for gene-specific
        hierarchical parameters.
    default_params : Tuple[float, float]
        Fallback (loc, scale) for the base Normal when hyperparameters are
        not yet available.  Not used during hierarchical sampling.
    hyper_loc_name : str
        Name of the sampled hyperparameter providing the prior location
        (e.g., ``"logit_p_loc"``).
    hyper_scale_name : str
        Name of the sampled hyperparameter providing the prior scale
        (e.g., ``"logit_p_scale"``).  Must be positive at sampling time.
    is_gene_specific : bool
        Should be ``True`` for hierarchical gene-specific parameters.
    is_mixture : bool
        If ``True``, parameter is per-component per-gene.
    transform : Transform
        NumPyro transform applied to the base Normal sample
        (e.g., ``SigmoidTransform`` for (0,1), ``ExpTransform`` or
        ``SoftplusTransform`` for (0,∞)).

    Examples
    --------
    >>> # Hierarchical p with sigmoid transform: p_g = sigmoid(Normal(loc, scale))
    >>> HierarchicalSigmoidNormalSpec(
    ...     name="p",
    ...     shape_dims=("n_genes",),
    ...     default_params=(0.0, 1.0),
    ...     hyper_loc_name="logit_p_loc",
    ...     hyper_scale_name="logit_p_scale",
    ...     is_gene_specific=True,
    ... )

    See Also
    --------
    HierarchicalSigmoidNormalSpec : Sigmoid-transformed hierarchical spec.
    HierarchicalPositiveNormalSpec : Positive-transform hierarchical spec.
    """

    hyper_loc_name: str = Field(
        ...,
        description="Name of the hyperparameter for the Normal location",
    )
    hyper_scale_name: str = Field(
        ...,
        description="Name of the hyperparameter for the Normal scale",
    )

    # --------------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample from the hierarchical prior using learned hyperparameters.

        Constructs a ``TransformedDistribution(Normal(loc, scale), transform)``
        where ``loc`` and ``scale`` are drawn from the population-level
        hyperprior (already in ``param_values``).

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes (must contain ``"n_genes"`` and optionally
            ``"n_components"``).
        param_values : Dict[str, jnp.ndarray]
            Dictionary of already-sampled parameter values.  Must contain
            ``self.hyper_loc_name`` and ``self.hyper_scale_name``.

        Returns
        -------
        jnp.ndarray
            Sampled gene-specific parameter in constrained space (after
            applying the transform).  Shape is ``(n_genes,)`` or
            ``(n_components, n_genes)`` if ``is_mixture=True``.

        Raises
        ------
        KeyError
            If the required hyperparameters are not in ``param_values``.
        """
        loc = param_values[self.hyper_loc_name]
        scale = param_values[self.hyper_scale_name]
        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # Build hierarchical prior: Normal(learned_loc, learned_scale)
        if shape == ():
            base_dist = dist.Normal(loc, scale)
        else:
            base_dist = (
                dist.Normal(loc, scale).expand(shape).to_event(len(shape))
            )

        transformed_dist = dist.TransformedDistribution(
            base_dist, self.transform
        )
        return numpyro.sample(self.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Hierarchical Sigmoid Normal Specification
# ------------------------------------------------------------------------------


class HierarchicalSigmoidNormalSpec(HierarchicalNormalWithTransformSpec):
    """Hierarchical Normal + Sigmoid for gene-specific parameters in (0, 1).

    Generative model::

        logit_p_loc   ~ Normal(0, 1)           [global hyperparameter]
        logit_p_scale ~ Softplus(Normal(0, 1))  [global hyperparameter]
        p_g = sigmoid(Normal(logit_p_loc, logit_p_scale))  [per gene]

    This is the hierarchical extension of ``SigmoidNormalSpec``, used for
    gene-specific success probability ``p_g`` in the hierarchical
    canonical and mean_prob parameterizations.

    Parameters
    ----------
    name : str
        Parameter name (typically ``"p"``).
    shape_dims : Tuple[str, ...]
        Shape dimensions (typically ``("n_genes",)``).
    default_params : Tuple[float, float]
        Fallback (loc, scale).
    hyper_loc_name : str
        Hyperparameter name for location (e.g., ``"logit_p_loc"``).
    hyper_scale_name : str
        Hyperparameter name for scale (e.g., ``"logit_p_scale"``).

    Examples
    --------
    >>> spec = HierarchicalSigmoidNormalSpec(
    ...     name="p",
    ...     shape_dims=("n_genes",),
    ...     default_params=(0.0, 1.0),
    ...     hyper_loc_name="logit_p_loc",
    ...     hyper_scale_name="logit_p_scale",
    ...     is_gene_specific=True,
    ... )

    Notes
    -----
    When ``logit_p_scale`` is small, genes are tightly clustered around
    ``sigmoid(logit_p_loc)`` (strong shrinkage).  When it is large,
    genes have diverse p values (weak shrinkage).  The data determine
    the appropriate degree of shrinkage via the hyperprior.
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )


# ------------------------------------------------------------------------------
# Hierarchical Positive Normal Specification
# ------------------------------------------------------------------------------


class HierarchicalPositiveNormalSpec(HierarchicalNormalWithTransformSpec):
    """
    Hierarchical Normal + positive transform for gene-specific parameters in (0,
    ∞).

    Generative model::

        log_phi_loc   ~ Normal(0, 1)            [global hyperparameter]
        log_phi_scale ~ Softplus(Normal(0, 1))   [global hyperparameter]
        phi_g = transform(Normal(log_phi_loc, log_phi_scale))  [per gene; transform = exp or softplus]

    This is the hierarchical extension of ``PositiveNormalSpec``, used for
    gene-specific odds ratio ``phi_g`` in the hierarchical mean_odds
    parameterization.

    Parameters
    ----------
    name : str
        Parameter name (typically ``"phi"``).
    shape_dims : Tuple[str, ...]
        Shape dimensions (typically ``("n_genes",)``).
    default_params : Tuple[float, float]
        Fallback (loc, scale).
    hyper_loc_name : str
        Hyperparameter name for location (e.g., ``"log_phi_loc"``).
    hyper_scale_name : str
        Hyperparameter name for scale (e.g., ``"log_phi_scale"``).

    Examples
    --------
    >>> spec = HierarchicalPositiveNormalSpec(
    ...     name="phi",
    ...     shape_dims=("n_genes",),
    ...     default_params=(0.0, 1.0),
    ...     hyper_loc_name="log_phi_loc",
    ...     hyper_scale_name="log_phi_scale",
    ...     is_gene_specific=True,
    ... )

    Notes
    -----
    When ``log_phi_scale`` is small, genes have similar odds ratios
    (strong shrinkage toward the transformed location).  When large,
    genes can have widely varying phi values.
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )


# ==============================================================================
# Horseshoe Gene-Level Hierarchical Specifications
# ==============================================================================


class HorseshoeHierarchicalSigmoidNormalSpec(
    HierarchicalNormalWithTransformSpec
):
    """Gene-level horseshoe prior with Sigmoid transform and NCP.

    Replaces the single global scale of ``HierarchicalSigmoidNormalSpec``
    with a regularized horseshoe structure (per-gene local scales).  Uses
    non-centered parameterization: samples ``z_g ~ Normal(0, 1)`` and
    computes the constrained parameter deterministically.

    Generative model (NCP form)::

        tau       ~ HalfCauchy(tau_0)                  [global shrinkage]
        lambda_g  ~ HalfCauchy(1)                      [per-gene local scale]
        c^2       ~ InvGamma(slab_df/2, slab_df*s^2/2) [slab]
        lt_g      = c * lambda_g / sqrt(c^2 + tau^2 * lambda_g^2)
        z_g       ~ Normal(0, 1)                       [NCP raw]
        p_g       = sigmoid(hyper_loc + tau * lt_g * z_g) [deterministic]

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"p"`` or ``"gate"``).
    tau_name : str
        Name of the global shrinkage site (``HalfCauchySpec``).
    lambda_name : str
        Name of the per-gene local scale site (``HalfCauchySpec``).
    c_sq_name : str
        Name of the slab site (``InverseGammaSpec``).
    raw_name : str
        Sample-site name for the NCP ``z`` variable (e.g. ``"p_raw"``).
    uses_ncp : bool
        Always ``True`` for this spec.

    See Also
    --------
    HierarchicalSigmoidNormalSpec : Normal-hierarchy version without horseshoe.
    HorseshoeDatasetSigmoidNormalSpec : Dataset-level variant.
    """

    tau_name: str = Field(
        ..., description="Name of the global shrinkage HalfCauchy site"
    )
    lambda_name: str = Field(
        ..., description="Name of the per-gene local scale HalfCauchy site"
    )
    c_sq_name: str = Field(
        ..., description="Name of the slab InverseGamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )

    # ------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic transform.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain ``hyper_loc_name``, ``tau_name``,
            ``lambda_name``, ``c_sq_name``.

        Returns
        -------
        jnp.ndarray
            Constrained parameter in (0, 1).
        """
        loc = param_values[self.hyper_loc_name]
        tau = param_values[self.tau_name]
        lam = param_values[self.lambda_name]
        c_sq = param_values[self.c_sq_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # Regularized local scale: c * lambda / sqrt(c^2 + tau^2 * lambda^2)
        c = jnp.sqrt(c_sq)
        eff_scale = tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)

        # NCP: sample z ~ Normal(0, 1)
        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        # Deterministic transform: constrained = transform(loc + eff_scale * z)
        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


class HorseshoeHierarchicalPositiveNormalSpec(
    HierarchicalNormalWithTransformSpec
):
    """Gene-level horseshoe prior with positive transform and NCP.

    Identical to ``HorseshoeHierarchicalSigmoidNormalSpec`` but applies
    a positive transform (exp or softplus) so the constrained parameter
    lives in ``(0, inf)``.  Used for ``phi`` (odds ratio) under
    ``mean_odds`` parameterization.

    Generative model (NCP form)::

        tau       ~ HalfCauchy(tau_0)
        lambda_g  ~ HalfCauchy(1)
        c^2       ~ InvGamma(slab_df/2, slab_df*s^2/2)
        lt_g      = c * lambda_g / sqrt(c^2 + tau^2 * lambda_g^2)
        z_g       ~ Normal(0, 1)
        phi_g     = transform(hyper_loc + tau * lt_g * z_g)  [transform = exp or softplus]

    See Also
    --------
    HorseshoeHierarchicalSigmoidNormalSpec : Sigmoid variant for p / gate.
    HorseshoeDatasetPositiveNormalSpec : Dataset-level variant.
    """

    tau_name: str = Field(
        ..., description="Name of the global shrinkage HalfCauchy site"
    )
    lambda_name: str = Field(
        ..., description="Name of the per-gene local scale HalfCauchy site"
    )
    c_sq_name: str = Field(
        ..., description="Name of the slab InverseGamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic positive transform.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain ``hyper_loc_name``, ``tau_name``,
            ``lambda_name``, ``c_sq_name``.

        Returns
        -------
        jnp.ndarray
            Constrained parameter in (0, inf).
        """
        loc = param_values[self.hyper_loc_name]
        tau = param_values[self.tau_name]
        lam = param_values[self.lambda_name]
        c_sq = param_values[self.c_sq_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        c = jnp.sqrt(c_sq)
        eff_scale = tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


# ==============================================================================
# Dataset-Level Hierarchical Specification
# ==============================================================================


class DatasetHierarchicalNormalWithTransformSpec(NormalWithTransformSpec):
    """
    Dataset-level hierarchical Normal + Transform.

    Extends the gene-level hierarchical spec to the multi-dataset setting.
    Each dataset gets its own value for this parameter, drawn from a shared
    (population-level) Normal distribution whose loc and scale are themselves
    sampled hyperparameters.

    The generative model is::

        hyper_loc   ~ some prior        (population hyperparameter)
        hyper_scale ~ some prior > 0    (population hyperparameter)
        param^(d)   = transform(Normal(hyper_loc, hyper_scale))  per dataset

    For gene-specific dataset parameters the shape is ``(n_datasets, n_genes)``
    and the hierarchy is over the leading dataset axis.

    Parameters
    ----------
    name : str
        Parameter name (e.g., ``"mu"``).
    shape_dims : Tuple[str, ...]
        Base shape dims **excluding** the dataset axis.  The dataset
        dimension is prepended automatically because ``is_dataset=True``.
    default_params : Tuple[float, float]
        Fallback (loc, scale) for the base Normal.
    hyper_loc_name : str
        Name of the sampled population-level location hyperparameter.
    hyper_scale_name : str
        Name of the sampled population-level scale hyperparameter.
    is_dataset : bool
        Must be ``True``.
    transform : Transform
        NumPyro transform applied to the base Normal sample.

    See Also
    --------
    DatasetHierarchicalPositiveNormalSpec : Positive-transform variant for (0, inf).
    DatasetHierarchicalSigmoidNormalSpec : Sigmoid-transformed variant for (0, 1).
    """

    hyper_loc_name: str = Field(
        ...,
        description="Name of the population-level location hyperparameter",
    )
    hyper_scale_name: str = Field(
        ...,
        description="Name of the population-level scale hyperparameter",
    )

    # Component indices shared across 2+ datasets.  When set together
    # with is_mixture=True and is_dataset=True, the hierarchical scale
    # is clamped to near-zero for *non-shared* components so they have
    # negligible dataset variation (Approach A masking).
    #
    # Future (Approach B): non-shared components could be given shape
    # (G,) instead of (D, G), eliminating wasted parameters at the cost
    # of splitting/concatenating tensors in the likelihood and guide.
    shared_component_indices: Optional[Tuple[int, ...]] = Field(
        None,
        description=(
            "Component indices shared across 2+ datasets. Non-shared "
            "components get their hierarchical scale clamped to suppress "
            "meaningless dataset variation."
        ),
    )

    # --------------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample per-dataset parameters from the population prior.

        Constructs ``TransformedDistribution(Normal(loc, scale), transform)``
        where loc and scale come from already-sampled population-level
        hyperparameters.  The resulting sample has a leading ``n_datasets``
        dimension.

        When the parameter is both mixture- and dataset-aware (shape
        ``(K, D, *shape_dims)``), the hyperprior loc may have shape
        ``(K, *shape_dims)`` (per-component but not per-dataset).  In
        that case a singleton dataset dimension is inserted at axis 1 so
        that broadcasting to ``(K, D, *shape_dims)`` works correctly.

        If ``shared_component_indices`` is set, the hierarchical scale
        is clamped to a near-zero value for non-shared components,
        suppressing inter-dataset variation for components that only
        exist in one dataset.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes (must contain ``"n_datasets"`` and optionally
            ``"n_genes"``, ``"n_components"``).
        param_values : Dict[str, jnp.ndarray]
            Already-sampled parameter values.  Must contain
            ``self.hyper_loc_name`` and ``self.hyper_scale_name``.

        Returns
        -------
        jnp.ndarray
            Sampled parameter in constrained space with shape
            ``(n_datasets, ...)`` (e.g., ``(D,)`` for scalar-per-dataset
            or ``(D, G)`` for gene-specific-per-dataset).
        """
        loc = param_values[self.hyper_loc_name]
        scale = param_values[self.hyper_scale_name]
        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # When both is_mixture and is_dataset are True the full shape is
        # (K, D, *shape_dims) but the hyperprior loc has (K, *shape_dims)
        # — no D dimension.  Insert a singleton D at axis 1 so that
        # Normal(loc, scale).expand(shape) broadcasts correctly.
        if self.is_mixture and self.is_dataset:
            if loc.ndim >= 1:
                loc = jnp.expand_dims(loc, axis=1)
            if scale.ndim >= 1:
                scale = jnp.expand_dims(scale, axis=1)

        # Scale masking: clamp the hierarchical scale to near-zero for
        # non-shared components so they have negligible dataset variation.
        if (
            self.shared_component_indices is not None
            and self.is_mixture
            and self.is_dataset
            and len(shape) > 0
        ):
            K = dims["n_components"]
            mask = jnp.zeros(K, dtype=bool)
            mask = mask.at[jnp.array(self.shared_component_indices)].set(True)
            # Reshape mask for broadcasting: (K,) → (K, 1, ..., 1)
            n_trailing = len(shape) - 1
            for _ in range(n_trailing):
                mask = mask[..., None]
            scale = jnp.where(mask, scale, 1e-6)

        if shape == ():
            base_dist = dist.Normal(loc, scale)
        else:
            base_dist = (
                dist.Normal(loc, scale).expand(shape).to_event(len(shape))
            )

        transformed_dist = dist.TransformedDistribution(
            base_dist, self.transform
        )
        return numpyro.sample(self.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Dataset Hierarchical Positive Normal (for mu, r, phi — positive params)
# ------------------------------------------------------------------------------


class DatasetHierarchicalPositiveNormalSpec(
    DatasetHierarchicalNormalWithTransformSpec
):
    """
    Dataset-level hierarchical Normal + positive transform for per-dataset
    parameters in (0, inf).

    Generative model::

        log_mu_loc   ~ Normal(0, 1)             [population hyperparameter]
        log_mu_scale ~ Softplus(Normal(0, 0.5))  [population hyperparameter]
        mu^(d) = transform(Normal(log_mu_loc, log_mu_scale))  [per dataset; transform = exp or softplus]

    For gene-specific variants, the shape is ``(n_datasets, n_genes)``.

    Examples
    --------
    >>> spec = DatasetHierarchicalPositiveNormalSpec(
    ...     name="mu",
    ...     shape_dims=("n_genes",),
    ...     default_params=(0.0, 1.0),
    ...     hyper_loc_name="log_mu_dataset_loc",
    ...     hyper_scale_name="log_mu_dataset_scale",
    ...     is_gene_specific=True,
    ...     is_dataset=True,
    ... )
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )


# ------------------------------------------------------------------------------
# Dataset Hierarchical Sigmoid Normal (for p, gate — (0,1) params)
# ------------------------------------------------------------------------------


class DatasetHierarchicalSigmoidNormalSpec(
    DatasetHierarchicalNormalWithTransformSpec
):
    """
    Dataset-level hierarchical Normal + Sigmoid for per-dataset parameters in
    (0, 1).

    Generative model::

        logit_p_loc   ~ Normal(0, 1)             [population hyperparameter]
        logit_p_scale ~ Softplus(Normal(0, 0.5))  [population hyperparameter]
        p^(d) = sigmoid(Normal(logit_p_loc, logit_p_scale))  [per dataset]

    Examples
    --------
    >>> spec = DatasetHierarchicalSigmoidNormalSpec(
    ...     name="p",
    ...     shape_dims=(),
    ...     default_params=(0.0, 1.0),
    ...     hyper_loc_name="logit_p_dataset_loc",
    ...     hyper_scale_name="logit_p_dataset_scale",
    ...     is_dataset=True,
    ... )
    """

    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )


# ==============================================================================
# Horseshoe Dataset-Level Hierarchical Specifications
# ==============================================================================


class HorseshoeDatasetPositiveNormalSpec(
    DatasetHierarchicalNormalWithTransformSpec
):
    """Dataset-level horseshoe prior with positive transform and NCP.

    Replaces the single global scale of ``DatasetHierarchicalPositiveNormalSpec``
    with a regularized horseshoe structure.  Uses NCP: samples
    ``z_{g,d} ~ Normal(0, 1)`` and computes ``mu_g^{(d)}`` deterministically.

    Generative model (NCP form)::

        tau        ~ HalfCauchy(tau_0)                   [global shrinkage]
        lambda_g   ~ HalfCauchy(1)                       [per-gene local scale]
        c^2        ~ InvGamma(slab_df/2, slab_df*s^2/2)  [slab]
        lt_g       = c * lambda_g / sqrt(c^2 + tau^2 * lambda_g^2)
        z_{g,d}    ~ Normal(0, 1)                        [NCP raw, (D, G)]
        mu_g^{(d)} = transform(hyper_loc_g + tau * lt_g * z_{g,d})  [deterministic; transform = exp or softplus]

    Parameters
    ----------
    tau_name : str
        Name of the global shrinkage site.
    lambda_name : str
        Name of the per-gene local scale site.
    c_sq_name : str
        Name of the slab site.
    raw_name : str
        Sample-site name for the NCP ``z`` variable (e.g. ``"mu_raw"``).
    uses_ncp : bool
        Always ``True``.

    See Also
    --------
    DatasetHierarchicalPositiveNormalSpec : Normal-hierarchy version.
    HorseshoeHierarchicalSigmoidNormalSpec : Gene-level variant.
    """

    tau_name: str = Field(
        ..., description="Name of the global shrinkage HalfCauchy site"
    )
    lambda_name: str = Field(
        ..., description="Name of the per-gene local scale HalfCauchy site"
    )
    c_sq_name: str = Field(
        ..., description="Name of the slab InverseGamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )

    # ------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic transform.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain ``hyper_loc_name``, ``tau_name``,
            ``lambda_name``, ``c_sq_name``.

        Returns
        -------
        jnp.ndarray
            Constrained parameter in (0, inf) with shape
            ``(n_datasets, n_genes)``.
        """
        loc = param_values[self.hyper_loc_name]
        tau = param_values[self.tau_name]
        lam = param_values[self.lambda_name]
        c_sq = param_values[self.c_sq_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # When both mixture and dataset, loc has (K, *shape) but we need
        # (K, D, *shape).  Insert singleton D at axis 1 for broadcasting.
        if self.is_mixture and self.is_dataset and loc.ndim >= 1:
            loc = jnp.expand_dims(loc, axis=1)

        # Regularized local scale: c * lambda / sqrt(c^2 + tau^2 * lambda^2)
        c = jnp.sqrt(c_sq)
        eff_scale = tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)

        # NCP: sample z ~ Normal(0, 1)
        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        # Deterministic transform: constrained = exp(loc + eff_scale * z)
        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


class HorseshoeDatasetSigmoidNormalSpec(
    DatasetHierarchicalNormalWithTransformSpec
):
    """Dataset-level horseshoe prior with Sigmoid transform and NCP.

    Replaces the single global scale of ``DatasetHierarchicalSigmoidNormalSpec``
    with a regularized horseshoe structure.  Used for dataset-level ``p``
    and dataset-level ``gate``.

    Generative model (NCP form)::

        tau        ~ HalfCauchy(tau_0)                   [global shrinkage]
        lambda_g   ~ HalfCauchy(1)                       [per-gene local scale]
        c^2        ~ InvGamma(slab_df/2, slab_df*s^2/2)  [slab]
        lt_g       = c * lambda_g / sqrt(c^2 + tau^2 * lambda_g^2)
        z_{g,d}    ~ Normal(0, 1)                        [NCP raw]
        p_g^{(d)}  = sigmoid(hyper_loc_g + tau * lt_g * z_{g,d})

    Parameters
    ----------
    tau_name : str
        Name of the global shrinkage site.
    lambda_name : str
        Name of the per-gene local scale site.
    c_sq_name : str
        Name of the slab site.
    raw_name : str
        Sample-site name for the NCP ``z`` variable.
    uses_ncp : bool
        Always ``True``.

    See Also
    --------
    DatasetHierarchicalSigmoidNormalSpec : Normal-hierarchy version.
    """

    tau_name: str = Field(
        ..., description="Name of the global shrinkage HalfCauchy site"
    )
    lambda_name: str = Field(
        ..., description="Name of the per-gene local scale HalfCauchy site"
    )
    c_sq_name: str = Field(
        ..., description="Name of the slab InverseGamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )

    # ------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic transform.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain ``hyper_loc_name``, ``tau_name``,
            ``lambda_name``, ``c_sq_name``.

        Returns
        -------
        jnp.ndarray
            Constrained parameter in (0, 1).
        """
        loc = param_values[self.hyper_loc_name]
        tau = param_values[self.tau_name]
        lam = param_values[self.lambda_name]
        c_sq = param_values[self.c_sq_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # When both mixture and dataset, loc has (K, *shape) but we need
        # (K, D, *shape).  Insert singleton D at axis 1 for broadcasting.
        if self.is_mixture and self.is_dataset and loc.ndim >= 1:
            loc = jnp.expand_dims(loc, axis=1)

        # Regularized local scale
        c = jnp.sqrt(c_sq)
        eff_scale = tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)

        # NCP: sample z ~ Normal(0, 1)
        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        # Deterministic transform
        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


# ==============================================================================
# Horseshoe Hyperparameter Specifications
# ==============================================================================


class HalfCauchySpec(ParamSpec):
    """Half-Cauchy distributed parameter for horseshoe shrinkage scales.

    Used for the global shrinkage ``tau`` (scalar) and per-gene local
    scales ``lambda_g`` (gene-specific) in the regularized horseshoe.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"tau_p"``, ``"lambda_p"``).
    shape_dims : Tuple[str, ...]
        Shape dimensions. Scalar ``()`` for tau, ``("n_genes",)`` for lambda.
    scale : float
        Scale parameter of the Half-Cauchy distribution (default 1.0).
        For tau, this is ``tau_0`` controlling expected sparsity.
    default_params : Tuple[float, ...]
        Not used for HalfCauchy (scale is set via ``scale`` field).
        Defaults to ``(1.0,)`` to satisfy ParamSpec.
    """

    scale: float = Field(1.0, description="Scale of the HalfCauchy prior")
    default_params: Tuple[float, ...] = Field(
        default=(1.0,), description="Unused; scale is set via 'scale' field."
    )

    def _get_distribution_type(self):
        return dist.HalfCauchy

    @property
    def constrained_name(self) -> str:
        """Half-Cauchy is already positive; constrained name equals name."""
        return self.name

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class InverseGammaSpec(ParamSpec):
    """Inverse-Gamma distributed parameter for the horseshoe slab.

    Used for the slab ``c^2`` that bounds the maximum effective scale
    in the regularized horseshoe.

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"c_sq_p"``).
    shape_dims : Tuple[str, ...]
        Shape dimensions (typically scalar ``()``).
    concentration : float
        Shape parameter ``alpha = slab_df / 2``.
    rate : float
        Rate parameter ``beta = slab_df * slab_scale^2 / 2``.
    default_params : Tuple[float, ...]
        Not used for InverseGamma. Defaults to ``(2.0, 8.0)`` (typical values).
    """

    concentration: float = Field(
        ..., description="Shape parameter (slab_df / 2)"
    )
    rate: float = Field(
        ..., description="Rate parameter (slab_df * slab_scale^2 / 2)"
    )
    default_params: Tuple[float, ...] = Field(
        default=(2.0, 8.0),
        description="Unused; concentration and rate set directly.",
    )

    def _get_distribution_type(self):
        return dist.InverseGamma

    @property
    def constrained_name(self) -> str:
        """InverseGamma is already positive; constrained name equals name."""
        return self.name

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# ------------------------------------------------------------------------------
# Gamma Distribution Parameter Specification (NEG prior hierarchy)
# ------------------------------------------------------------------------------


class GammaSpec(ParamSpec):
    """Gamma-distributed parameter for NEG prior hierarchy layers.

    Used for both the per-gene variance ``psi_g`` (inner layer) and the
    per-gene rate ``zeta_g`` (outer layer) in the Normal-Exponential-Gamma
    (NEG) prior.

    The NEG hierarchy is:
        zeta_g ~ Gamma(a, tau)        [per-gene rate]
        psi_g  ~ Gamma(u, zeta_g)     [per-gene variance; u=1 => Exponential]
        beta_g ~ Normal(0, sqrt(psi)) [coefficient]

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"psi_p"``, ``"zeta_p"``).
    shape_dims : Tuple[str, ...]
        Shape dimensions.  ``("n_genes",)`` for per-gene sites.
    concentration : float
        Shape parameter of the Gamma distribution (``u`` for psi, ``a`` for zeta).
    rate_name : str or None
        When set, the rate is read from ``param_values[rate_name]`` at
        sample time (for psi, whose rate is zeta_g).  When ``None``, a
        fixed ``rate`` is used (for zeta, whose rate is the global tau).
    rate : float
        Fixed rate parameter.  Only used when ``rate_name is None``.
    default_params : Tuple[float, ...]
        Defaults to ``(1.0, 1.0)``.
    """

    concentration: float = Field(
        ..., description="Shape parameter of the Gamma"
    )
    rate_name: Optional[str] = Field(
        None,
        description=(
            "Site name from which to read the rate at sample time.  "
            "None means use the fixed 'rate' field."
        ),
    )
    rate: float = Field(
        1.0, description="Fixed rate (used when rate_name is None)"
    )
    default_params: Tuple[float, ...] = Field(
        default=(1.0, 1.0),
        description="Unused; concentration and rate set directly.",
    )

    def _get_distribution_type(self):
        return dist.Gamma

    @property
    def constrained_name(self) -> str:
        """Gamma is already positive; constrained name equals name."""
        return self.name

    @property
    def support(self) -> constraints.Constraint:
        """Return positive constraint for sampled values."""
        return dist.Gamma.support

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """Return constraints on concentration and rate (both positive)."""
        return dist.Gamma.arg_constraints

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


# ==============================================================================
# NEG Gene-Level Hierarchical Specifications
# ==============================================================================


class NEGHierarchicalSigmoidNormalSpec(HierarchicalNormalWithTransformSpec):
    """Gene-level NEG prior with Sigmoid transform and NCP.

    The Normal-Exponential-Gamma replaces the horseshoe's Half-Cauchy local
    scales with a Gamma-Gamma hierarchy that is friendlier to SVI.

    Generative model (NCP form)::

        zeta_g  ~ Gamma(a, tau)          [per-gene rate]
        psi_g   ~ Gamma(u, zeta_g)       [per-gene variance; u=1 => Exponential]
        z_g     ~ Normal(0, 1)           [NCP raw]
        p_g     = sigmoid(hyper_loc + sqrt(psi_g) * z_g)  [deterministic]

    Parameters
    ----------
    name : str
        Parameter name (e.g. ``"p"`` or ``"gate"``).
    psi_name : str
        Name of the per-gene variance site (``GammaSpec``).
    zeta_name : str
        Name of the per-gene rate site (``GammaSpec``).
    raw_name : str
        Sample-site name for the NCP ``z`` variable (e.g. ``"p_raw"``).
    uses_ncp : bool
        Always ``True`` for this spec.
    """

    psi_name: str = Field(
        ..., description="Name of the per-gene variance Gamma site"
    )
    zeta_name: str = Field(
        ..., description="Name of the per-gene rate Gamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )

    # ------------------------------------------------------------------

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic transform.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain ``hyper_loc_name`` and ``psi_name``.

        Returns
        -------
        jnp.ndarray
            Constrained parameter in (0, 1).
        """
        loc = param_values[self.hyper_loc_name]
        psi = param_values[self.psi_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # Effective scale from the NEG hierarchy
        eff_scale = jnp.sqrt(psi)

        # NCP: sample z ~ Normal(0, 1)
        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        # Deterministic transform
        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


class NEGHierarchicalPositiveNormalSpec(HierarchicalNormalWithTransformSpec):
    """Gene-level NEG prior with positive transform and NCP.

    Identical to ``NEGHierarchicalSigmoidNormalSpec`` but applies a
    positive transform (exp or softplus) so the constrained parameter
    lives in ``(0, inf)``.  Used for ``phi`` (odds ratio) under
    ``mean_odds`` parameterization.
    """

    psi_name: str = Field(
        ..., description="Name of the per-gene variance Gamma site"
    )
    zeta_name: str = Field(
        ..., description="Name of the per-gene rate Gamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic positive transform."""
        loc = param_values[self.hyper_loc_name]
        psi = param_values[self.psi_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        eff_scale = jnp.sqrt(psi)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


# ==============================================================================
# NEG Dataset-Level Hierarchical Specifications
# ==============================================================================


class NEGDatasetPositiveNormalSpec(DatasetHierarchicalNormalWithTransformSpec):
    """Dataset-level NEG prior with positive transform (for mu/r datasets).

    Replaces the horseshoe's Half-Cauchy scales with the NEG Gamma-Gamma
    hierarchy at the dataset level.
    """

    psi_name: str = Field(
        ..., description="Name of the per-gene variance Gamma site"
    )
    zeta_name: str = Field(
        ..., description="Name of the per-gene rate Gamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.ExpTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic positive transform."""
        loc = param_values[self.hyper_loc_name]
        psi = param_values[self.psi_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # When both mixture and dataset, loc has (K, *shape) but we need
        # (K, D, *shape).  Insert singleton D at axis 1 for broadcasting.
        if self.is_mixture and self.is_dataset and loc.ndim >= 1:
            loc = jnp.expand_dims(loc, axis=1)

        eff_scale = jnp.sqrt(psi)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


class NEGDatasetSigmoidNormalSpec(DatasetHierarchicalNormalWithTransformSpec):
    """Dataset-level NEG prior with Sigmoid transform (for p/gate datasets).

    Replaces the horseshoe's Half-Cauchy scales with the NEG Gamma-Gamma
    hierarchy at the dataset level.
    """

    psi_name: str = Field(
        ..., description="Name of the per-gene variance Gamma site"
    )
    zeta_name: str = Field(
        ..., description="Name of the per-gene rate Gamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SigmoidTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample via NCP: z ~ N(0,1), then deterministic sigmoid transform."""
        loc = param_values[self.hyper_loc_name]
        psi = param_values[self.psi_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        # When both mixture and dataset, loc has (K, *shape) but we need
        # (K, D, *shape).  Insert singleton D at axis 1 for broadcasting.
        if self.is_mixture and self.is_dataset and loc.ndim >= 1:
            loc = jnp.expand_dims(loc, axis=1)

        eff_scale = jnp.sqrt(psi)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


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
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

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
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

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
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

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
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

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
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

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


# ------------------------------------------------------------------------------
# Anchored Normal Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(AnchoredNormalSpec, dict, object)
def sample_prior(
    spec: AnchoredNormalSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample from Normal prior with per-gene data-informed centers.

    Each gene gets its own prior center derived from the observed
    sample mean, while the log-scale sigma is shared across genes.

    Parameters
    ----------
    spec : AnchoredNormalSpec
        The anchored parameter specification with per-gene centers.
    dims : Dict[str, int]
        Dimension sizes (must include "n_genes").
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled parameter values in constrained space.
    """
    # jnp.asarray is a no-op when the input is already a JAX array,
    # avoiding repeated Python-to-JAX conversion during JIT tracing.
    centers = jnp.asarray(spec.anchor_centers)
    sigma = spec.anchor_sigma

    # Build distribution with per-gene centers and shared sigma.
    # centers has shape (n_genes,); sigma is a scalar.
    base_dist = dist.Normal(centers, sigma).to_event(1)

    # Apply transform (identity for log_mu_loc, but kept for generality)
    transformed_dist = dist.TransformedDistribution(base_dist, spec.transform)

    return numpyro.sample(spec.constrained_name, transformed_dist)


# ------------------------------------------------------------------------------
# Half-Cauchy Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(HalfCauchySpec, dict, object)
def sample_prior(
    spec: HalfCauchySpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample from Half-Cauchy prior (horseshoe shrinkage scales).

    Parameters
    ----------
    spec : HalfCauchySpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    if shape == ():
        return numpyro.sample(spec.name, dist.HalfCauchy(spec.scale))
    else:
        return numpyro.sample(
            spec.name,
            dist.HalfCauchy(spec.scale).expand(shape).to_event(len(shape)),
        )


# ------------------------------------------------------------------------------
# Inverse-Gamma Distribution Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(InverseGammaSpec, dict, object)
def sample_prior(
    spec: InverseGammaSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample from Inverse-Gamma prior (horseshoe slab).

    Parameters
    ----------
    spec : InverseGammaSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    if shape == ():
        return numpyro.sample(
            spec.name,
            dist.InverseGamma(spec.concentration, spec.rate),
        )
    else:
        return numpyro.sample(
            spec.name,
            dist.InverseGamma(spec.concentration, spec.rate)
            .expand(shape)
            .to_event(len(shape)),
        )


# ------------------------------------------------------------------------------
# Gamma Distribution Prior Sampling (NEG prior hierarchy)
# ------------------------------------------------------------------------------


@dispatch(GammaSpec, dict, object)
def sample_prior(
    spec: GammaSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample from Gamma prior (NEG psi/zeta layers).

    When ``rate_name`` is set, the rate must be read from param_values;
    use the 4-arg overload with param_values in that case.

    Parameters
    ----------
    spec : GammaSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    if spec.rate_name is not None:
        raise ValueError(
            f"GammaSpec '{spec.name}' has rate_name='{spec.rate_name}'; "
            "use sample_prior(spec, dims, model_config, param_values) "
            "to provide the rate from the dependent site."
        )
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    if shape == ():
        return numpyro.sample(
            spec.name,
            dist.Gamma(spec.concentration, spec.rate),
        )
    else:
        return numpyro.sample(
            spec.name,
            dist.Gamma(spec.concentration, spec.rate)
            .expand(shape)
            .to_event(len(shape)),
        )


@dispatch(GammaSpec, dict, object, dict)
def sample_prior(
    spec: GammaSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    param_values: Dict[str, jnp.ndarray],
) -> jnp.ndarray:
    """Sample from Gamma prior when rate is read from another site.

    Used for psi_g ~ Gamma(u, zeta_g) where zeta_g is already in param_values.

    Parameters
    ----------
    spec : GammaSpec
        The parameter specification.
    dims : Dict[str, int]
        Dimension sizes.
    model_config : ModelConfig
        Model configuration.
    param_values : Dict[str, jnp.ndarray]
        Must contain spec.rate_name when rate_name is set.

    Returns
    -------
    jnp.ndarray
        Sampled positive parameter value.
    """
    rate = (
        param_values[spec.rate_name]
        if spec.rate_name is not None
        else spec.rate
    )
    shape = resolve_shape(
        spec.shape_dims,
        dims,
        is_mixture=spec.is_mixture,
        is_dataset=spec.is_dataset,
    )

    if shape == ():
        return numpyro.sample(
            spec.name,
            dist.Gamma(spec.concentration, rate),
        )
    else:
        return numpyro.sample(
            spec.name,
            dist.Gamma(spec.concentration, rate)
            .expand(shape)
            .to_event(len(shape)),
        )


# ------------------------------------------------------------------------------
# Biology-Informed Capture Probability Prior Sampling
# ------------------------------------------------------------------------------


@dispatch(BiologyInformedCaptureSpec, dict, object)
def sample_prior(
    spec: BiologyInformedCaptureSpec,
    dims: Dict[str, int],
    model_config: "ModelConfig",
) -> jnp.ndarray:
    """Sample capture parameter from biology-informed prior.

    The actual per-cell sampling of eta_c happens inside the VCP likelihood
    (which has access to counts and library sizes). This dispatch handles
    the shared mu_eta parameter for data-driven mode, and delegates
    per-cell sampling to the likelihood.

    For data-driven mode, samples the shared total-mRNA parameter:
        mu_eta ~ N(log_M0, sigma_mu^2)

    For biology-informed mode (fixed M_0), this is a no-op — the VCP
    likelihood reads log_M0 from the spec directly.

    Parameters
    ----------
    spec : BiologyInformedCaptureSpec
        The capture parameter specification.
    dims : Dict[str, int]
        Dimension sizes (includes n_cells).
    model_config : ModelConfig
        Model configuration.

    Returns
    -------
    jnp.ndarray or None
        The shared mu_eta sample (data-driven) or None (biology-informed).
    """
    if spec.data_driven:
        # Sample the shared log-total-mRNA parameter (outside cell plate)
        mu_eta = numpyro.sample(
            "mu_eta",
            dist.Normal(spec.log_M0, spec.sigma_mu),
        )
        return mu_eta
    # Biology-informed with fixed M_0: no-op here; VCP likelihood handles
    # per-cell eta_c sampling
    return None


# ==============================================================================
# BNB (Beta Negative Binomial) Overdispersion Specs
# ==============================================================================


class HorseshoeBNBConcentrationSpec(HierarchicalNormalWithTransformSpec):
    """Horseshoe prior on the BNB excess-dispersion fraction omega_g.

    The BNB extension adds a per-gene parameter kappa_g > 2 that
    controls overdispersion beyond the NB.  We reparameterize as
    omega_g = softplus(loc + scale * z), which is non-negative, and
    recover kappa_g = 2 + (r_g + 1) / omega_g in the likelihood.

    The horseshoe encourages omega_g ~ 0 (NB behaviour) for most
    genes, with a data-driven subset escaping to finite omega_g.

    Generative model (NCP form)::

        tau         ~ HalfCauchy(tau_0)
        lambda_g    ~ HalfCauchy(1)
        c^2         ~ InvGamma(slab_df/2, slab_df*s^2/2)
        lt_g        = c * lambda_g / sqrt(c^2 + tau^2 * lambda_g^2)
        z_g         ~ Normal(0, 1)
        omega_g     = softplus(hyper_loc + tau * lt_g * z_g)

    Parameters
    ----------
    name : str
        Constrained parameter name (``"bnb_concentration"``).
    tau_name : str
        Global shrinkage HalfCauchy site.
    lambda_name : str
        Per-gene local scale HalfCauchy site.
    c_sq_name : str
        Slab InverseGamma site.
    raw_name : str
        NCP z site (``"bnb_concentration_raw"``).
    """

    tau_name: str = Field(
        ..., description="Name of the global shrinkage HalfCauchy site"
    )
    lambda_name: str = Field(
        ..., description="Name of the per-gene local scale HalfCauchy site"
    )
    c_sq_name: str = Field(
        ..., description="Name of the slab InverseGamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SoftplusTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample omega_g via NCP with horseshoe shrinkage.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain hyper_loc, tau, lambda, c_sq sites.

        Returns
        -------
        jnp.ndarray
            omega_g in (0, inf), shape ``(n_genes,)``.
        """
        loc = param_values[self.hyper_loc_name]
        tau = param_values[self.tau_name]
        lam = param_values[self.lambda_name]
        c_sq = param_values[self.c_sq_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        c = jnp.sqrt(c_sq)
        eff_scale = tau * c * lam / jnp.sqrt(c_sq + tau**2 * lam**2)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


class NEGBNBConcentrationSpec(HierarchicalNormalWithTransformSpec):
    """NEG prior on the BNB excess-dispersion fraction omega_g.

    Same role as ``HorseshoeBNBConcentrationSpec`` but uses the
    Normal-Exponential-Gamma hierarchy, which is friendlier to SVI.

    Generative model (NCP form)::

        zeta_g  ~ Gamma(a, tau_NEG)
        psi_g   ~ Gamma(u, zeta_g)
        z_g     ~ Normal(0, 1)
        omega_g = softplus(hyper_loc + sqrt(psi_g) * z_g)

    Parameters
    ----------
    name : str
        Constrained parameter name (``"bnb_concentration"``).
    psi_name : str
        Per-gene variance Gamma site.
    zeta_name : str
        Per-gene rate Gamma site.
    raw_name : str
        NCP z site (``"bnb_concentration_raw"``).
    """

    psi_name: str = Field(
        ..., description="Name of the per-gene variance Gamma site"
    )
    zeta_name: str = Field(
        ..., description="Name of the per-gene rate Gamma site"
    )
    raw_name: str = Field(
        ..., description="Sample-site name for the NCP z variable"
    )
    uses_ncp: bool = Field(
        True, description="Flag indicating NCP parameterization"
    )
    transform: Transform = Field(
        default_factory=lambda: dist.transforms.SoftplusTransform()
    )

    def sample_hierarchical(
        self,
        dims: Dict[str, int],
        param_values: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Sample omega_g via NCP with NEG shrinkage.

        Parameters
        ----------
        dims : Dict[str, int]
            Dimension sizes.
        param_values : Dict[str, jnp.ndarray]
            Must contain hyper_loc, psi, zeta sites.

        Returns
        -------
        jnp.ndarray
            omega_g in (0, inf), shape ``(n_genes,)``.
        """
        loc = param_values[self.hyper_loc_name]
        psi = param_values[self.psi_name]

        shape = resolve_shape(
            self.shape_dims,
            dims,
            is_mixture=self.is_mixture,
            is_dataset=self.is_dataset,
        )

        eff_scale = jnp.sqrt(psi)

        if shape == ():
            z_dist = dist.Normal(0.0, 1.0)
        else:
            z_dist = dist.Normal(0.0, 1.0).expand(shape).to_event(len(shape))
        z = numpyro.sample(self.raw_name, z_dist)

        unconstrained = loc + eff_scale * z
        constrained = self.transform(unconstrained)
        numpyro.deterministic(self.constrained_name, constrained)
        return constrained


# ---------------------------------------------------------------------------
# Backward-compatible aliases for pickle deserialization of old models.
# Old checkpoints reference the *Exp* names; these aliases ensure
# unpickling resolves to the renamed *Positive* classes.
# ---------------------------------------------------------------------------
ExpNormalSpec = PositiveNormalSpec
HierarchicalExpNormalSpec = HierarchicalPositiveNormalSpec
DatasetHierarchicalExpNormalSpec = DatasetHierarchicalPositiveNormalSpec
HorseshoeHierarchicalExpNormalSpec = HorseshoeHierarchicalPositiveNormalSpec
HorseshoeDatasetExpNormalSpec = HorseshoeDatasetPositiveNormalSpec
NEGHierarchicalExpNormalSpec = NEGHierarchicalPositiveNormalSpec
NEGDatasetExpNormalSpec = NEGDatasetPositiveNormalSpec
