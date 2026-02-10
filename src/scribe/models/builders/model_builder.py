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
from numpyro.contrib.module import flax_module

from .parameter_specs import (
    DerivedParam,
    DirichletSpec,
    ParamSpec,
    sample_prior,
)
from ..components.guide_families import VAELatentGuide

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

        # Validate: VAE and mixture are mutually exclusive.
        # The VAE's continuous latent space replaces discrete mixture
        # components, so combining them is not supported.
        has_vae = any(
            isinstance(s.guide_family, VAELatentGuide)
            and getattr(s.guide_family, "decoder", None) is not None
            for s in self.param_specs
        )
        has_mixture = any(s.is_mixture for s in self.param_specs)
        if has_vae and has_mixture:
            raise ValueError(
                "VAE and mixture models cannot be combined. The VAE's "
                "continuous latent space replaces discrete mixture components."
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
                    # Use uniform prior (all concentrations = 1)
                    mixing_prior_params = tuple([1.0] * n_components)
                    mixing_spec = DirichletSpec(
                        name="mixing_weights",
                        shape_dims=(),
                        default_params=mixing_prior_params,
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

            # ============================================================
            # 4a. Detect if any cell-specific parameter uses a VAE guide.
            #     If present, build a vae_cell_fn closure to run VAE logic
            #     inside the cell plate in the likelihood.
            #     The closure does the following for each cell:
            #       1. Samples z from the prior distribution using the
            #       latent_spec.
            #       2. Runs the decoder network on z to generate per-cell
            #       parameters.
            #       3. Registers each decoded output as a deterministic site.
            # ============================================================
            vae_cell_fn = None
            for s in cell_specs:
                gf = s.guide_family
                # Check if this spec uses a VAELatentGuide with a decoder and
                # latent_spec defined
                if (
                    isinstance(gf, VAELatentGuide)
                    and gf.decoder is not None
                    and gf.latent_spec is not None
                ):
                    latent_spec = gf.latent_spec
                    decoder = gf.decoder

                    # Define the closure; called inside the likelihood cell
                    # plate before observation sampling
                    def vae_cell_fn(_batch_idx):
                        # Sample the latent variable z from its prior
                        z = numpyro.sample(
                            latent_spec.sample_site,
                            latent_spec.make_prior_dist(),
                        )
                        # Get a Flax-wrapped decoder module-ready for JAX shape
                        # and parameter management
                        decoder_net = flax_module(
                            "vae_decoder",
                            decoder,
                            input_shape=(latent_spec.latent_dim,),
                        )
                        # Run decoder on z (can optionally support covariates in
                        # the future)
                        decoder_out = decoder_net(z)
                        # Register outputs as deterministic sites so they're
                        # available to the rest of the model
                        for name, value in decoder_out.items():
                            numpyro.deterministic(name, value)
                        return decoder_out

                    # Only the first detected VAE cell spec is handled; exit
                    # after building the closure
                    break

            # Filter out VAE marker specs — their latent is sampled
            # inside vae_cell_fn, so they must NOT be sample_prior'd
            # again by the likelihood.  Non-VAE cell specs (e.g.
            # p_capture for VCP) are kept and sampled normally.
            if vae_cell_fn is not None:
                non_vae_cell_specs = [
                    s
                    for s in cell_specs
                    if not isinstance(s.guide_family, VAELatentGuide)
                ]
            else:
                non_vae_cell_specs = cell_specs

            likelihood.sample(
                param_values=param_values,
                cell_specs=non_vae_cell_specs,
                counts=counts,
                dims=dims,
                batch_size=batch_size,
                model_config=model_config,
                vae_cell_fn=vae_cell_fn,
            )

        return model
