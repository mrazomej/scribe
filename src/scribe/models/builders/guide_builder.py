"""Guide builder orchestration for NumPyro variational guides.

This module keeps the public `GuideBuilder` API while delegating dispatch
implementations to internal `_guide_*_mixin.py` modules.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import jax.numpy as jnp
import numpyro

from .parameter_specs import (
    BiologyInformedCaptureSpec,
    DirichletSpec,
    ParamSpec,
)
from ..components.guide_families import (
    JointLowRankGuide,
    MeanFieldGuide,
    VAELatentGuide,
)

# Import dispatch modules eagerly so multipledispatch registrations are loaded.
from . import _guide_horseshoe_mixin as _guide_horseshoe_mixin
from . import _guide_neg_mixin as _guide_neg_mixin
from . import _guide_lowrank_mixin as _guide_lowrank_mixin
from . import _guide_meanfield_mixin as _guide_meanfield_mixin

# Import concrete symbols used directly by GuideBuilder.
from ._guide_amortized_mixin import _setup_grouped_amortized_latent
from ._guide_cell_specific_mixin import (
    guide_mu_eta_hierarchy,
    setup_cell_specific_guide,
)
from ._guide_joint_mixin import _woodbury_conditional_params, setup_joint_guide
from ._guide_structured_joint_mixin import setup_structured_joint_guide
from ._guide_meanfield_mixin import setup_guide

if TYPE_CHECKING:
    from ..config import ModelConfig


class GuideBuilder:
    """Builder for constructing NumPyro guide functions from parameter specs.

    The GuideBuilder takes parameter specifications and constructs a guide
    function that samples from the variational posterior. Each parameter can
    have its own guide family (MeanField, LowRank, Amortized).

    Attributes
    ----------
    param_specs : List[ParamSpec]
        Parameter specifications for the guide.

    Examples
    --------
    >>> from scribe.models.builders import GuideBuilder, BetaSpec, LogNormalSpec
    >>> from scribe.models.components import MeanFieldGuide, LowRankGuide
    >>>
    >>> specs = [
    ...     BetaSpec("p", (), (1.0, 1.0), guide_family=MeanFieldGuide()),
    ...     LogNormalSpec("r", ("n_genes",), (0.0, 1.0), guide_family=LowRankGuide(rank=10)),
    ... ]
    >>> guide = GuideBuilder().from_specs(specs).build()
    """

    def __init__(self):
        """Initialize an empty GuideBuilder."""
        self.param_specs: List[ParamSpec] = []

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

    def from_specs(self, specs: List[ParamSpec]) -> "GuideBuilder":
        """Set parameter specifications from a list.

        Parameters
        ----------
        specs : List[ParamSpec]
            List of parameter specifications.

        Returns
        -------
        GuideBuilder
            Self, for method chaining.
        """
        self.param_specs = specs
        return self

    # --------------------------------------------------------------------------

    def add_param(self, spec: ParamSpec) -> "GuideBuilder":
        """Add a single parameter specification.

        Parameters
        ----------
        spec : ParamSpec
            Parameter specification to add.

        Returns
        -------
        GuideBuilder
            Self, for method chaining.
        """
        self.param_specs.append(spec)
        return self

    # --------------------------------------------------------------------------

    def build(self) -> Callable:
        """Build and return the NumPyro guide function.

        Returns
        -------
        Callable
            A NumPyro guide function with signature:
            guide(n_cells, n_genes, model_config, counts=None, batch_size=None)
        """
        specs = self.param_specs

        def guide(
            n_cells: int,
            n_genes: int,
            model_config: "ModelConfig",
            counts: Optional[jnp.ndarray] = None,
            batch_size: Optional[int] = None,
            annotation_prior_logits: Optional[jnp.ndarray] = None,
            dataset_indices: Optional[jnp.ndarray] = None,
        ):
            """NumPyro guide function.

            Parameters
            ----------
            n_cells : int
                Total number of cells in the dataset.
            n_genes : int
                Number of genes.
            model_config : ModelConfig
                Configuration containing guide hyperparameters.
            counts : Optional[jnp.ndarray], shape (n_cells, n_genes)
                Observed count matrix (needed for amortized guides).
            batch_size : Optional[int]
                Mini-batch size for stochastic VI.
            annotation_prior_logits : Optional[jnp.ndarray]
                Accepted for API compatibility with the model function
                (NumPyro passes the same kwargs to both model and guide).
                Ignored by the guide — annotation priors are observed data,
                not latent variables requiring variational approximation.
            """
            # ================================================================
            # Setup dimensions dict
            # ================================================================
            dims = {"n_cells": n_cells, "n_genes": n_genes}
            if (
                hasattr(model_config, "n_components")
                and model_config.n_components
            ):
                dims["n_components"] = model_config.n_components
            if hasattr(model_config, "n_datasets") and model_config.n_datasets:
                dims["n_datasets"] = model_config.n_datasets

            # ================================================================
            # 0. Setup guide for MIXING WEIGHTS if this is a mixture model
            # ================================================================
            is_mixture = any(s.is_mixture for s in specs)
            if is_mixture:
                if "n_components" not in dims:
                    raise ValueError(
                        "n_components must be set in model_config when "
                        "using mixture parameters"
                    )
                # Check if mixing_weights spec exists
                mixing_spec = next(
                    (s for s in specs if s.name == "mixing_weights"), None
                )
                if mixing_spec is None:
                    # Create default Dirichlet spec for mixing weights
                    n_components = dims["n_components"]
                    # Get mixing prior from param_specs if available
                    mixing_prior_params = None
                    if model_config.param_specs:
                        for spec in model_config.param_specs:
                            if spec.name == "mixing" and spec.prior is not None:
                                mixing_prior_params = spec.prior
                                break
                    if mixing_prior_params is None:
                        mixing_prior_params = tuple([1.0] * n_components)
                    from .parameter_specs import DirichletSpec

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
                        prior=mixing_prior_params,
                        is_mixture=False,  # Mixing weights are not mixture-specific
                        is_dataset=use_dataset_mixing,
                    )
                guide_family = mixing_spec.guide_family or MeanFieldGuide()
                setup_guide(mixing_spec, guide_family, dims, model_config)

            # ================================================================
            # 1. Collect JointLowRankGuide groups across ALL spec types
            #    (global, gene-specific).  Joint groups are processed
            #    first so that scalar and gene-specific params sharing a
            #    JointLowRankGuide are handled together.
            # ================================================================
            global_specs = [
                s
                for s in specs
                if not s.is_gene_specific
                and not s.is_cell_specific
                and s.name != "mixing_weights"  # Already handled above
            ]
            gene_specs = [s for s in specs if s.is_gene_specific]

            # Scan both global and gene-specific specs for joint groups
            joint_groups: Dict[str, List] = {}
            joint_handled = set()
            for spec in (*global_specs, *gene_specs):
                gf = spec.guide_family
                if isinstance(gf, JointLowRankGuide):
                    grp = gf.group
                    if grp not in joint_groups:
                        joint_groups[grp] = []
                    joint_groups[grp].append(spec)
                    joint_handled.add(spec.name)

            # Process joint groups first (may contain a mix of scalar
            # and gene-specific specs with heterogeneous shapes).
            # When dense_params is set on the guide marker, use the
            # structured path (dense low-rank + nondense gene-local).
            for grp_name, grp_specs in joint_groups.items():
                guide_family = grp_specs[0].guide_family
                if guide_family.dense_params is not None:
                    setup_structured_joint_guide(
                        grp_specs, guide_family, dims, model_config
                    )
                else:
                    setup_joint_guide(
                        grp_specs, guide_family, dims, model_config
                    )

            # ================================================================
            # 2. Setup guides for remaining GLOBAL parameters
            # ================================================================
            for spec in global_specs:
                if spec.name in joint_handled:
                    continue
                guide_family = spec.guide_family or MeanFieldGuide()
                setup_guide(spec, guide_family, dims, model_config)

            # ================================================================
            # 3. Setup guides for remaining GENE-SPECIFIC parameters
            # ================================================================
            for spec in gene_specs:
                if spec.name in joint_handled:
                    continue
                guide_family = spec.guide_family or MeanFieldGuide()
                setup_guide(spec, guide_family, dims, model_config)

            # ================================================================
            # 2.5. Pre-plate: data-driven biology-informed capture mu_eta
            # ================================================================
            cell_specs = [s for s in specs if s.is_cell_specific]
            _n_ds = getattr(model_config, "n_datasets", None) or 0
            for spec in cell_specs:
                if (
                    isinstance(spec, BiologyInformedCaptureSpec)
                    and spec.data_driven
                ):
                    guide_mu_eta_hierarchy(spec, _n_ds)

            # ================================================================
            # 3. Setup guides for CELL-SPECIFIC parameters (inside cell plate)
            #    Handle batch indexing for non-amortized guides
            #    Register amortizer modules ONCE before the plate loop
            #    If any spec uses VAELatentGuide with encoder+latent_spec,
            #    run the VAE latent block once (encoder -> latent_spec -> sample z)
            # ================================================================
            grouped_guide = None
            for s in cell_specs:
                gf = s.guide_family
                if (
                    isinstance(gf, VAELatentGuide)
                    and gf.encoder is not None
                    and gf.latent_spec is not None
                ):
                    grouped_guide = gf
                    break

            if cell_specs:
                # Linen modules are registered automatically when flax_module is called
                # No need to pre-register - flax_module handles parameter registration
                # and is JIT-safe (no retracing issues)

                if batch_size is None:
                    # Full sampling
                    with numpyro.plate("cells", n_cells):
                        if grouped_guide is not None:
                            _setup_grouped_amortized_latent(
                                grouped_guide,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=None,
                            )
                        for spec in cell_specs:
                            guide_family = spec.guide_family or MeanFieldGuide()
                            if spec.guide_family is grouped_guide:
                                continue  # z already sampled; decoder/params later
                            setup_cell_specific_guide(
                                spec,
                                guide_family,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=None,
                            )
                else:
                    # Batch sampling
                    with numpyro.plate(
                        "cells", n_cells, subsample_size=batch_size
                    ) as idx:
                        if grouped_guide is not None:
                            _setup_grouped_amortized_latent(
                                grouped_guide,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=idx,
                            )
                        for spec in cell_specs:
                            guide_family = spec.guide_family or MeanFieldGuide()
                            if spec.guide_family is grouped_guide:
                                continue  # z already sampled; decoder/params later
                            setup_cell_specific_guide(
                                spec,
                                guide_family,
                                dims,
                                model_config,
                                counts=counts,
                                batch_idx=idx,
                            )

        return guide
