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
import numpyro.distributions as dist
from numpyro.contrib.module import flax_module

from scribe.flows import FlowDistribution
from .parameter_specs import (
    DatasetHierarchicalNormalWithTransformSpec,
    DerivedParam,
    DirichletSpec,
    GammaSpec,
    HierarchicalNormalWithTransformSpec,
    ParamSpec,
    sample_prior,
)
from ..components.guide_families import VAELatentGuide
from ...core.axis_layout import (
    AxisLayout,
    align_to_layout,
    layout_from_param_spec,
    merge_layouts,
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
        self.vae_in_plate_derived: List[DerivedParam] = []
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

    def set_vae_in_plate_derived(
        self, derived_params: List[DerivedParam]
    ) -> "ModelBuilder":
        """
        Set derived params that depend on decoder outputs (computed in
        vae_cell_fn).

        These are derived params whose deps include at least one decoder output.
        They must be computed inside the cell plate after the decoder runs.

        Parameters
        ----------
        derived_params : List[DerivedParam]
            Derived params that depend on decoder outputs (e.g., r = mu * phi
            when mu comes from decoder).

        Returns
        -------
        ModelBuilder
            Self, for method chaining.
        """
        self.vae_in_plate_derived = list(derived_params)
        return self

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

        # Validate: if a flow prior is set, its features must match
        # the latent dimensionality.
        for s in self.param_specs:
            gf = s.guide_family
            if (
                isinstance(gf, VAELatentGuide)
                and gf.latent_spec is not None
                and getattr(gf.latent_spec, "flow", None) is not None
            ):
                latent_spec = gf.latent_spec
                flow_features = getattr(latent_spec.flow, "features", None)
                if (
                    flow_features is not None
                    and flow_features != latent_spec.latent_dim
                ):
                    raise ValueError(
                        f"Flow features ({flow_features}) must equal "
                        f"latent_dim ({latent_spec.latent_dim}). "
                        "The prior flow must operate on the same "
                        "dimensionality as the latent space."
                    )

        # Capture builder state in closure
        specs = self.param_specs
        derived = self.derived_params
        in_plate_derived = self.vae_in_plate_derived
        likelihood = self.likelihood

        def model(
            n_cells: int,
            n_genes: int,
            model_config: "ModelConfig",
            counts: Optional[jnp.ndarray] = None,
            batch_size: Optional[int] = None,
            annotation_prior_logits: Optional[jnp.ndarray] = None,
            dataset_indices: Optional[jnp.ndarray] = None,
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
            annotation_prior_logits : Optional[jnp.ndarray], shape (n_cells,
            n_components)
                Per-cell logit offsets for mixture component assignment priors.
                When provided for a mixture model, global mixing weights are
                combined with these logits to produce cell-specific mixing
                probabilities.  If None (default), the global mixing weights are
                used for all cells.
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
            if (
                hasattr(model_config, "n_datasets")
                and model_config.n_datasets
            ):
                dims["n_datasets"] = model_config.n_datasets

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
                    # Create default Dirichlet spec for mixing weights.
                    # For multi-dataset models, dataset_mixing_enabled adds a
                    # dataset axis so each dataset gets its own simplex.
                    mixing_prior_params = tuple([1.0] * n_components)
                    use_dataset_mixing = bool(
                        getattr(model_config, "dataset_mixing_enabled", False)
                    )
                    shape_dims = (
                        ("n_components",) if use_dataset_mixing else ()
                    )
                    mixing_spec = DirichletSpec(
                        name="mixing_weights",
                        shape_dims=shape_dims,
                        default_params=mixing_prior_params,
                        is_mixture=False,  # Mixing weights are not mixture-specific
                        is_dataset=use_dataset_mixing,
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
                if isinstance(
                    spec, DatasetHierarchicalNormalWithTransformSpec
                ):
                    # Dataset-level hierarchical param (e.g. scalar p
                    # per dataset) uses learned hyperparameters
                    param_values[spec.name] = spec.sample_hierarchical(
                        dims, param_values
                    )
                elif (
                    isinstance(spec, GammaSpec)
                    and spec.rate_name is not None
                ):
                    # Gamma with rate from another site (e.g. psi ~ Gamma(u, zeta))
                    param_values[spec.name] = sample_prior(
                        spec, dims, model_config, param_values
                    )
                else:
                    param_values[spec.name] = sample_prior(
                        spec, dims, model_config
                    )

            # ================================================================
            # 2. Sample GENE-SPECIFIC parameters
            #    Shape: (n_genes,) - one value per gene, shared across cells
            #    Examples: r (dispersion), mu (mean), gate (zero-inflation prob)
            #    Using to_event(1) to mark the gene dimension as non-independent
            #
            #    Hierarchical specs (HierarchicalNormalWithTransformSpec) are
            #    gene-specific parameters whose prior is defined by already-
            #    sampled hyperparameters (from step 1). These use a dedicated
            #    sampling method that reads hyperparameter values from
            #    param_values instead of using fixed prior hyperparameters.
            # ================================================================
            gene_specs = [s for s in specs if s.is_gene_specific]
            for spec in gene_specs:
                if isinstance(
                    spec,
                    (
                        HierarchicalNormalWithTransformSpec,
                        DatasetHierarchicalNormalWithTransformSpec,
                    ),
                ):
                    # Hierarchical prior: loc and scale come from
                    # hyperparameters sampled in step 1 (gene-level or
                    # dataset-level hierarchy)
                    param_values[spec.name] = spec.sample_hierarchical(
                        dims, param_values
                    )
                elif (
                    isinstance(spec, GammaSpec)
                    and spec.rate_name is not None
                ):
                    # Gamma with rate from another site (e.g. psi ~ Gamma(u, zeta))
                    param_values[spec.name] = sample_prior(
                        spec, dims, model_config, param_values
                    )
                else:
                    param_values[spec.name] = sample_prior(
                        spec, dims, model_config
                    )

            # ================================================================
            # 3. Compute DERIVED parameters from sampled values
            #    Examples: r = mu * (1-p) / p (linked parameterization)
            #    These are deterministic transformations of sampled parameters.
            #
            #    Dependencies may have different axis layouts (e.g. mu is
            #    (K, D, G) from a dataset hierarchy while phi is (K, G)
            #    from gene-level only).  We use AxisLayout metadata to
            #    align every dependency to a common target layout *before*
            #    calling the compute function, so compute functions can
            #    assume pre-aligned inputs and contain only pure math.
            # ================================================================

            # Build semantic layouts for all sampled parameters so that
            # derived-param alignment can look up each dependency's axes.
            param_layouts: Dict[str, AxisLayout] = {
                spec.name: layout_from_param_spec(spec) for spec in specs
            }

            for d in derived:
                # Determine the target layout: the union of all dependency
                # layouts (e.g. merging ("components", "genes") with
                # ("components", "datasets", "genes") yields the latter).
                dep_layouts = [
                    param_layouts.get(k, AxisLayout(())) for k in d.deps
                ]
                target = (
                    merge_layouts(*dep_layouts)
                    if dep_layouts
                    else AxisLayout(())
                )

                # Align each dependency tensor to the target layout by
                # inserting singleton dimensions where axes are missing.
                aligned = {
                    k: align_to_layout(
                        param_values[k],
                        param_layouts.get(k, AxisLayout(())),
                        target,
                    )
                    for k in d.deps
                }

                param_values[d.name] = numpyro.deterministic(
                    d.name, d.compute(**aligned)
                )
                # Record the derived param's layout so downstream derived
                # params (or the likelihood) can reference it.
                param_layouts[d.name] = target

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
                        # Build the prior distribution for z.
                        # If a prior flow is set on the latent spec, register
                        # it with flax_module and wrap in FlowDistribution;
                        # otherwise use the standard prior (Normal(0, I)).
                        if latent_spec.flow is not None:
                            # Register the flow as a Flax module so its
                            # parameters are tracked by NumPyro's param store
                            flow_fn = flax_module(
                                "vae_prior_flow",
                                latent_spec.flow,
                                input_shape=(latent_spec.latent_dim,),
                            )
                            # Base distribution: standard Normal matching the
                            # latent dimensionality
                            base = dist.Normal(
                                jnp.zeros(latent_spec.latent_dim),
                                jnp.ones(latent_spec.latent_dim),
                            ).to_event(1)
                            # Wrap into FlowDistribution — a proper NumPyro
                            # distribution with sample() and log_prob()
                            prior = FlowDistribution(flow_fn, base)
                        else:
                            # Standard Gaussian prior: z ~ N(0, I)
                            prior = latent_spec.make_prior_dist()

                        # Sample the latent variable z from the prior
                        z = numpyro.sample(latent_spec.sample_site, prior)
                        # Get a Flax-wrapped decoder module ready for JAX
                        # shape and parameter management
                        decoder_net = flax_module(
                            "vae_decoder",
                            decoder,
                            input_shape=(latent_spec.latent_dim,),
                        )
                        # Run decoder on z to produce per-cell parameters
                        decoder_out = dict(decoder_net(z))
                        # Register outputs as deterministic sites so they're
                        # available to the rest of the model
                        for name, value in decoder_out.items():
                            numpyro.deterministic(name, value)

                        # Compute in-plate derived params (deps include
                        # decoder outputs).  Align dependencies using
                        # AxisLayout metadata, same as the pre-plate path.
                        # Decoder outputs are cell-specific tensors that
                        # typically have no component/dataset axes, so
                        # their layout defaults to AxisLayout(()).
                        for d in in_plate_derived:
                            deps = {}
                            dep_layouts = []
                            for dep in d.deps:
                                if dep in decoder_out:
                                    deps[dep] = decoder_out[dep]
                                else:
                                    deps[dep] = param_values[dep]
                                dep_layouts.append(
                                    param_layouts.get(
                                        dep, AxisLayout(())
                                    )
                                )

                            target = (
                                merge_layouts(*dep_layouts)
                                if dep_layouts
                                else AxisLayout(())
                            )
                            aligned = {
                                k: align_to_layout(
                                    v,
                                    param_layouts.get(
                                        k, AxisLayout(())
                                    ),
                                    target,
                                )
                                for k, v in deps.items()
                            }

                            result = d.compute(**aligned)
                            numpyro.deterministic(d.name, result)
                            decoder_out[d.name] = result
                            param_layouts[d.name] = target

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

            # Pass param_layouts so the likelihood can use semantic
            # AxisLayout metadata for broadcasting in mixture models,
            # even when model_config.param_specs is empty.
            likelihood.sample(
                param_values=param_values,
                cell_specs=non_vae_cell_specs,
                counts=counts,
                dims=dims,
                batch_size=batch_size,
                model_config=model_config,
                vae_cell_fn=vae_cell_fn,
                annotation_prior_logits=annotation_prior_logits,
                dataset_indices=dataset_indices,
                param_layouts=param_layouts,
            )

        return model
