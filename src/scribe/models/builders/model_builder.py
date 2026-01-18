"""Model builder for composing NumPyro probabilistic models.

This module provides a builder pattern for constructing NumPyro model functions
from reusable parameter specifications. It handles the complexity of plate
management, batch sampling, and derived parameter computation.

Classes
-------
ModelBuilder
    Builder for constructing NumPyro model functions.

Examples
--------
>>> from scribe.models.builders import ModelBuilder, BetaSpec, LogNormalSpec, DerivedParam
>>> from scribe.models.components import NegativeBinomialLikelihood
>>>
>>> model = (ModelBuilder()
...     .add_param(BetaSpec("p", (), (1.0, 1.0)))
...     .add_param(LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
...     .with_likelihood(NegativeBinomialLikelihood())
...     .build())

See Also
--------
scribe.models.builders.parameter_specs : Parameter specification classes.
scribe.models.components.likelihoods : Likelihood components.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpyro

from .parameter_specs import (
    DerivedParam,
    DirichletSpec,
    ParamSpec,
    sample_prior,
)

if TYPE_CHECKING:
    from ..components.likelihoods import Likelihood
    from ..config import ModelConfig

# ==============================================================================
# ModelBuilder Class
# ==============================================================================


class ModelBuilder:
    """Builder for constructing NumPyro model functions from parameter specs.

    This builder handles three distinct sampling modes for cell plates:

    1. **Full sampling** (counts provided, no batch_size):
       Sample all cells at once. Used for small datasets or MCMC.

    2. **Batch sampling** (counts provided, batch_size specified):
       Sample a mini-batch of cells via subsampling. Used for SVI on
       large datasets. The plate returns batch indices for indexing.

    3. **Prior predictive** (counts=None):
       Sample from the prior without conditioning on data. Used for
       prior predictive checks and synthetic data generation.

    Attributes
    ----------
    param_specs : List[ParamSpec]
        Parameter specifications to sample in the model.
    derived_params : List[DerivedParam]
        Parameters computed from sampled values.
    likelihood : Likelihood
        The likelihood component for sampling observations.

    Examples
    --------
    >>> # Build a simple NBDM model
    >>> model = (ModelBuilder()
    ...     .add_param(BetaSpec("p", (), (1.0, 1.0)))
    ...     .add_param(LogNormalSpec("r", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
    ...     .with_likelihood(NegativeBinomialLikelihood())
    ...     .build())
    >>>
    >>> # Build a linked parameterization model
    >>> model = (ModelBuilder()
    ...     .add_param(BetaSpec("p", (), (1.0, 1.0)))
    ...     .add_param(LogNormalSpec("mu", ("n_genes",), (0.0, 1.0), is_gene_specific=True))
    ...     .add_derived("r", lambda p, mu: mu * (1-p) / p, ["p", "mu"])
    ...     .with_likelihood(NegativeBinomialLikelihood())
    ...     .build())
    """

    def __init__(self):
        """Initialize an empty ModelBuilder."""
        self.param_specs: List[ParamSpec] = []
        self.derived_params: List[DerivedParam] = []
        self.likelihood: Optional["Likelihood"] = None

    # --------------------------------------------------------------------------

    @property
    def is_mixture(self) -> bool:
        """Check if any parameter is mixture-specific.

        Returns
        -------
        bool
            True if any parameter has is_mixture=True.
        """
        return any(spec.is_mixture for spec in self.param_specs)

    # --------------------------------------------------------------------------

    def add_param(self, spec: ParamSpec) -> "ModelBuilder":
        """Add a parameter specification to the model.

        Parameters
        ----------
        spec : ParamSpec
            The parameter specification (BetaSpec, LogNormalSpec, etc.)

        Returns
        -------
        ModelBuilder
            Self, for method chaining.

        Examples
        --------
        >>> builder = ModelBuilder()
        >>> builder.add_param(BetaSpec("p", (), (1.0, 1.0)))
        >>> builder.add_param(LogNormalSpec("r", ("n_genes",), (0.0, 1.0)))
        """
        self.param_specs.append(spec)
        return self

    # --------------------------------------------------------------------------

    def add_derived(
        self, name: str, compute: Callable, deps: List[str]
    ) -> "ModelBuilder":
        """Add a derived parameter computed from other parameters.

        Parameters
        ----------
        name : str
            Name for the deterministic site.
        compute : Callable
            Function computing derived value. Keyword args match dep names.
        deps : List[str]
            Names of parameters this depends on.

        Returns
        -------
        ModelBuilder
            Self, for method chaining.

        Examples
        --------
        >>> # Linked parameterization: r = mu * (1-p) / p
        >>> builder.add_derived("r", lambda p, mu: mu * (1-p) / p, ["p", "mu"])
        """
        self.derived_params.append(DerivedParam(name, compute, deps))
        return self

    # --------------------------------------------------------------------------

    def with_likelihood(self, likelihood: "Likelihood") -> "ModelBuilder":
        """Set the likelihood component for the model.

        Parameters
        ----------
        likelihood : Likelihood
            Likelihood object (NegativeBinomialLikelihood, ZINBLikelihood, etc.)

        Returns
        -------
        ModelBuilder
            Self, for method chaining.

        Examples
        --------
        >>> from scribe.models.components import NegativeBinomialLikelihood
        >>> builder.with_likelihood(NegativeBinomialLikelihood())
        """
        self.likelihood = likelihood
        return self

    # --------------------------------------------------------------------------

    def build(self) -> Callable:
        """Build and return the NumPyro model function.

        Returns
        -------
        Callable
            A NumPyro model function with signature:
            model(n_cells, n_genes, model_config, counts=None, batch_size=None)

        Raises
        ------
        ValueError
            If no likelihood has been set.

        Examples
        --------
        >>> model = builder.build()
        >>> # Use for SVI
        >>> svi = SVI(model, guide, ...)
        """
        if self.likelihood is None:
            raise ValueError(
                "Likelihood must be set before building. Use with_likelihood()."
            )

        # Capture builder state in closure
        specs = self.param_specs
        derived = self.derived_params
        likelihood = self.likelihood

        def model(
            n_cells: int,
            n_genes: int,
            model_config: "ModelConfig",
            counts: Optional[jnp.ndarray] = None,
            batch_size: Optional[int] = None,
        ):
            """NumPyro model function.

            Parameters
            ----------
            n_cells : int
                Total number of cells in the dataset.
            n_genes : int
                Number of genes.
            model_config : ModelConfig
                Configuration containing prior/guide hyperparameters.
            counts : Optional[jnp.ndarray], shape (n_cells, n_genes)
                Observed count matrix. If None, samples from prior (prior
                predictive).
            batch_size : Optional[int]
                Mini-batch size for stochastic VI. If None, uses all cells.
            """
            # ================================================================
            # Setup dimensions dict for shape resolution
            # We need this dict to resolve symbolic shape_dims like ("n_genes",)
            # into concrete shapes like (2000,)
            # ================================================================
            dims = {"n_cells": n_cells, "n_genes": n_genes}
            if (
                hasattr(model_config, "n_components")
                and model_config.n_components
            ):
                dims["n_components"] = model_config.n_components

            param_values: Dict[str, jnp.ndarray] = {}

            # ================================================================
            # 0. Sample MIXING WEIGHTS if this is a mixture model
            #    Mixing weights are sampled before other parameters.
            #    They define the component assignment probabilities.
            # ================================================================
            is_mixture = any(s.is_mixture for s in specs)
            if is_mixture:
                if "n_components" not in dims:
                    raise ValueError(
                        "n_components must be set in model_config when "
                        "using mixture parameters"
                    )
                n_components = dims["n_components"]

                # Check if mixing_weights spec already exists
                mixing_spec = next(
                    (s for s in specs if s.name == "mixing_weights"), None
                )
                if mixing_spec is None:
                    # Create default Dirichlet spec for mixing weights
                    mixing_prior_params = getattr(
                        model_config.priors, "mixing", None
                    ) or tuple([1.0] * n_components)
                    mixing_spec = DirichletSpec(
                        "mixing_weights",
                        (),
                        mixing_prior_params,
                        is_mixture=False,  # Mixing weights are not mixture-specific
                    )

                param_values["mixing_weights"] = sample_prior(
                    mixing_spec, dims, model_config
                )

            # ================================================================
            # 1. Sample GLOBAL parameters (neither gene-specific nor
            #    cell-specific)
            #    These are scalar parameters shared across all cells/genes.
            #    Examples: p (dropout), phi (odds ratio)
            #    Sampled OUTSIDE any plate since they're not indexed.
            # ================================================================
            global_specs = [
                s
                for s in specs
                if not s.is_gene_specific
                and not s.is_cell_specific
                and s.name != "mixing_weights"  # Already sampled above
            ]
            for spec in global_specs:
                param_values[spec.name] = sample_prior(spec, dims, model_config)

            # ================================================================
            # 2. Sample GENE-SPECIFIC parameters
            #    Shape: (n_genes,) - one value per gene, shared across cells
            #    Examples: r (dispersion), mu (mean), gate (zero-inflation prob)
            #    Using to_event(1) to mark the gene dimension as non-independent
            # ================================================================
            gene_specs = [s for s in specs if s.is_gene_specific]
            for spec in gene_specs:
                param_values[spec.name] = sample_prior(spec, dims, model_config)

            # ================================================================
            # 3. Compute DERIVED parameters from sampled values
            #    Examples: r = mu * (1-p) / p (linked parameterization)
            #    These are deterministic transformations of sampled parameters
            # ================================================================
            for d in derived:
                dep_values = {k: param_values[k] for k in d.deps}
                param_values[d.name] = numpyro.deterministic(
                    d.name, d.compute(**dep_values)
                )

            # ================================================================
            # 4. Handle CELL-SPECIFIC params and LIKELIHOOD inside cell plate
            #    Three modes based on counts and batch_size:
            #    - Prior predictive: counts=None
            #    - Full sampling: counts provided, batch_size=None
            #    - Batch sampling: counts provided, batch_size specified
            # ================================================================
            cell_specs = [s for s in specs if s.is_cell_specific]
            likelihood.sample(
                param_values=param_values,
                cell_specs=cell_specs,
                counts=counts,
                dims=dims,
                batch_size=batch_size,
                model_config=model_config,
            )

        return model
