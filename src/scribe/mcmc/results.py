"""
Results classes for SCRIBE MCMC inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, field, replace
import warnings

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
from jax import random, jit, vmap
import pandas as pd
from numpyro.infer import MCMC
import numpy as np

from ..sampling import (
    generate_predictive_samples,
    sample_biological_nb,
    denoise_counts as _denoise_counts_util,
)
from ..core.component_indexing import (
    normalize_component_indices,
    renormalize_mixing_logits,
    renormalize_mixing_weights,
)
from ..models.config import ModelConfig
from ..core.normalization import normalize_counts_from_posterior
from ..svi._gene_subsetting import build_gene_axis_by_key

# ------------------------------------------------------------------------------
# MCMC Subset class (moved to module level)
# ------------------------------------------------------------------------------


@dataclass
class ScribeMCMCSubset:
    """Lightweight container for subset of MCMC results."""

    samples: Dict
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    n_components: Optional[int] = None
    predictive_samples: Optional[jnp.ndarray] = None

    # Add posterior_samples property for compatibility
    @property
    def posterior_samples(self):
        """Get posterior samples in canonical form."""
        return self.get_posterior_samples(canonical=True)

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """Support indexing by genes and components."""
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "Tuple indexing must be (gene_indexer, component_indexer)."
                )
            gene_indexer, component_indexer = index
            gene_subset = self[gene_indexer]
            return gene_subset.get_component(component_indexer)

        # If index is a boolean mask, use it directly
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            bool_index = index
        # Handle integer indexing
        elif isinstance(index, int):
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given index
            bool_index = bool_index.at[index].set(True)
        # Handle slice indexing
        elif isinstance(index, slice):
            # Get indices from slice
            indices = jnp.arange(self.n_genes)[index]
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(
            index[-1], (bool, jnp.bool_)
        ):
            # Get indices from list/array
            indices = jnp.array(index)
            # Initialize boolean index
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            # Already a boolean index
            bool_index = index

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Get subset of samples
        new_samples = self._subset_posterior_samples(self.samples, bool_index)

        # Create new instance with subset data
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=int(
                bool_index.sum()
                if hasattr(bool_index, "sum")
                else len(bool_index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            n_components=self.n_components,
        )

    # --------------------------------------------------------------------------

    def get_component(
        self, component_index, renormalize: bool = True
    ) -> "ScribeMCMCSubset":
        """
        Return a component-restricted view of this subset.

        Parameters
        ----------
        component_index : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCSubset
            Subset restricted to requested components.
        """
        selected = normalize_component_indices(
            component_index, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))
        return self.get_components(selected, renormalize=renormalize)

    # --------------------------------------------------------------------------

    def get_components(
        self, component_indices, renormalize: bool = True
    ) -> "ScribeMCMCSubset":
        """
        Select multiple components while preserving mixture semantics.

        Parameters
        ----------
        component_indices : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCSubset
            Mixture-aware subset with reduced ``n_components``.
        """
        selected = normalize_component_indices(
            component_indices, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))

        new_samples = self._subset_samples_by_components(
            self.samples,
            selected,
            renormalize=renormalize,
            squeeze_single=False,
        )
        new_n_components = int(selected.shape[0])
        new_model_config = self.model_config.model_copy(
            update={"n_components": new_n_components}
        )
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=self.model_type,
            model_config=new_model_config,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=new_n_components,
        )

    # --------------------------------------------------------------------------

    def _create_single_component_subset(
        self, component_index: int
    ) -> "ScribeMCMCSubset":
        """Create legacy non-mixture subset for one component."""
        new_samples = self._subset_samples_by_components(
            self.samples,
            jnp.asarray([component_index], dtype=jnp.int32),
            renormalize=True,
            squeeze_single=True,
        )
        base_model = self.model_type.replace("_mix", "")
        new_model_config = self.model_config.model_copy(
            update={"base_model": base_model, "n_components": None}
        )
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,
            model_config=new_model_config,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=None,
        )

    # --------------------------------------------------------------------------

    def _subset_samples_by_components(
        self,
        samples: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
        squeeze_single: bool = False,
    ) -> Dict:
        """
        Subset sample dictionary along component axis for mixture parameters.

        Parameters
        ----------
        samples : Dict
            Raw sample dictionary.
        component_indices : jnp.ndarray
            One-dimensional selected component indices.
        renormalize : bool, default=True
            Whether to renormalize selected mixing fractions.
        squeeze_single : bool, default=False
            If True and one component is selected, remove component axis.

        Returns
        -------
        Dict
            Subset sample dictionary.
        """
        if samples is None:
            return None

        specs_by_name = {}
        if getattr(self.model_config, "param_specs", None):
            specs_by_name = {s.name: s for s in self.model_config.param_specs}

        n_comp = self.n_components
        use_single = squeeze_single and int(component_indices.shape[0]) == 1
        single_index = int(component_indices[0]) if use_single else None
        new_samples = {}

        for key, values in samples.items():
            spec = specs_by_name.get(key)
            has_mixture_axis = hasattr(values, "ndim") and values.ndim > 1
            is_named_mixture_weight = key in {
                "mixing_weights",
                "mixing_logits_unconstrained",
            }
            is_mixture = (
                has_mixture_axis
                and (
                    (spec is not None and spec.is_mixture)
                    or is_named_mixture_weight
                )
            )
            if not is_mixture:
                new_samples[key] = values
                continue

            # Component axis is typically 1, but some sample tensors include
            # extra singleton dimensions where the component axis appears later.
            component_axis = 1
            if values.shape[component_axis] != n_comp:
                candidates = [
                    ax
                    for ax in range(1, values.ndim)
                    if values.shape[ax] == n_comp
                ]
                if not candidates:
                    new_samples[key] = values
                    continue
                component_axis = candidates[0]

            slicer = [slice(None)] * values.ndim
            slicer[component_axis] = (
                single_index if use_single else component_indices
            )
            selected = values[tuple(slicer)]

            if renormalize and key == "mixing_weights" and not use_single:
                selected = renormalize_mixing_weights(selected, axis=-1)
            elif (
                renormalize
                and key == "mixing_logits_unconstrained"
                and not use_single
            ):
                selected = renormalize_mixing_logits(selected, axis=-1)

            new_samples[key] = selected

        return new_samples

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        When model_config.param_specs is set, use metadata-based subsetting;
        otherwise use last-axis heuristic as fallback.
        """
        if samples is None:
            return None

        new_samples = {}
        original_n_genes = self.n_genes
        gene_axis_by_key = None
        if getattr(self.model_config, "param_specs", None):
            gene_axis_by_key = build_gene_axis_by_key(
                self.model_config.param_specs, samples, original_n_genes
            )

        for key, value in samples.items():
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue
            if gene_axis_by_key is not None and key in gene_axis_by_key:
                gene_axis = gene_axis_by_key[key]
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_samples[key] = value[tuple(slicer)]
                continue
            if value.ndim > 0 and value.shape[-1] == original_n_genes:
                new_samples[key] = value[..., index]
            else:
                new_samples[key] = value
        return new_samples

    # --------------------------------------------------------------------------

    def _is_component_specific_param(self, param_name: str) -> bool:
        """Check if a parameter is component-specific (mixture-specific).

        In the new system, each parameter's `is_mixture` field in `param_specs`
        indicates if it's component-specific. This replaces the deprecated
        `component_specific_params` boolean.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g., "p", "phi")

        Returns
        -------
        bool
            True if the parameter has is_mixture=True in param_specs
        """
        if not self.model_config.param_specs:
            # Fallback: check mixture_params if param_specs not populated
            mixture_params = self.model_config.mixture_params or []
            return param_name in mixture_params

        # Check param_specs for is_mixture=True
        for spec in self.model_config.param_specs:
            if spec.name == param_name:
                return spec.is_mixture
        return False

    # --------------------------------------------------------------------------

    def get_posterior_samples(self, canonical=True):
        """Get posterior samples in canonical form."""
        if canonical:
            # Convert to canonical form on-the-fly
            return self._convert_to_canonical(self.samples)
        else:
            return self.samples

    # --------------------------------------------------------------------------

    def _convert_to_canonical(self, samples):
        """
        Convert samples to canonical (p, r) form based on present
        parameters.

        Parameters
        ----------
        samples : Dict
            Raw samples dictionary

        Returns
        -------
        Dict
            Canonical samples dictionary
        """
        # If no samples, return empty dict
        if not samples:
            return {}

        # Create a copy of samples to avoid modifying the original
        canonical_samples = samples.copy()

        # Convert based on what parameters are present in the samples

        # Handle odds_ratio parameterization (phi, mu -> p, r)
        if "phi" in canonical_samples and "mu" in canonical_samples:
            # Extract phi and mu
            phi = canonical_samples["phi"]
            mu = canonical_samples["mu"]
            # Compute p from phi
            canonical_samples["p"] = 1.0 / (1.0 + phi)

            # Reshape phi to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                if self._is_component_specific_param("phi"):
                    # Component-specific phi:
                    # shape (n_samples, n_components) -> reshape
                    phi_reshaped = phi[:, :, None]
                else:
                    # Shared phi: shape (n_samples,) -> reshape
                    phi_reshaped = phi[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                phi_reshaped = phi[:, None]
            # Compute r from mu and phi
            canonical_samples["r"] = mu * phi_reshaped

        # Handle linked parameterization (mu, p -> r)
        elif (
            "mu" in canonical_samples
            and "p" in canonical_samples
            and "r" not in canonical_samples
        ):
            # Extract p and mu
            p = canonical_samples["p"]
            mu = canonical_samples["mu"]
            # Reshape p to broadcast with mu based on mixture model
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                if self._is_component_specific_param("p"):
                    # Component-specific p:
                    # shape (n_samples, n_components) -> reshape
                    p_reshaped = p[:, :, None]
                else:
                    # Shared p: shape (n_samples,) -> reshape
                    p_reshaped = p[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                p_reshaped = p[:, None]
            # Compute r from mu and p
            canonical_samples["r"] = mu * (1.0 - p_reshaped) / p_reshaped

        # Handle unconstrained parameterization (r_unconstrained, p_unconstrained, etc.)
        if (
            "r_unconstrained" in canonical_samples
            and "r" not in canonical_samples
        ):
            # compute r from r_unconstrained
            canonical_samples["r"] = jnp.exp(
                canonical_samples["r_unconstrained"]
            )

        if (
            "p_unconstrained" in canonical_samples
            and "p" not in canonical_samples
        ):
            # Compute p from p_unconstrained
            canonical_samples["p"] = sigmoid(
                canonical_samples["p_unconstrained"]
            )

        if (
            "gate_unconstrained" in canonical_samples
            and "gate" not in canonical_samples
        ):
            # Compute gate from gate_unconstrained
            canonical_samples["gate"] = sigmoid(
                canonical_samples["gate_unconstrained"]
            )

        # Handle VCP capture probability conversions
        if (
            "phi_capture" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Extract phi_capture
            phi_capture = canonical_samples["phi_capture"]
            # Compute p_capture from phi_capture
            canonical_samples["p_capture"] = 1.0 / (1.0 + phi_capture)

        if (
            "p_capture_unconstrained" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Compute p_capture from p_capture_unconstrained
            canonical_samples["p_capture"] = sigmoid(
                canonical_samples["p_capture_unconstrained"]
            )

        # Handle mixing weights computation for mixture models
        if (
            "mixing_logits_unconstrained" in canonical_samples
            and "mixing_weights" not in canonical_samples
        ):
            # Compute mixing weights from mixing_logits_unconstrained using
            # softmax
            canonical_samples["mixing_weights"] = softmax(
                canonical_samples["mixing_logits_unconstrained"],
                axis=-1,
            )

        return canonical_samples

    # --------------------------------------------------------------------------

    def get_posterior_quantiles(self, param, quantiles=(0.025, 0.5, 0.975)):
        """Get quantiles for a specific parameter from samples."""
        return _get_posterior_quantiles(
            self.get_posterior_samples(), param, quantiles
        )

    # --------------------------------------------------------------------------

    def get_map(self):
        """Get MAP estimates for parameters."""
        return _get_map_estimate(self.get_posterior_samples())

    # --------------------------------------------------------------------------

    def log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        sample_chunk_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """Compute log likelihood of data under posterior samples."""
        # Get canonical samples
        samples = self.get_posterior_samples(canonical=True)

        return _compute_log_likelihood(
            samples,
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
            sample_chunk_size=sample_chunk_size,
            return_by=return_by,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
            dtype=dtype,
        )

    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        predictive_samples = _generate_ppc_samples(
            self.get_posterior_samples(canonical=True),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate prior predictive samples using the model."""
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        # Generate prior predictive samples
        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples

    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples_dirichlet: int = 1,
        fit_distribution: bool = False,
        store_samples: bool = True,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """Normalize counts using posterior samples of the r parameter."""
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        # Get canonical samples
        posterior_samples = self.get_posterior_samples(canonical=True)

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            fit_distribution=fit_distribution,
            store_samples=store_samples,
            sample_axis=sample_axis,
            return_concentrations=return_concentrations,
            backend=backend,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------

    def cell_type_probabilities(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Compute probabilistic cell type assignments."""
        from ..core.cell_type_assignment import compute_cell_type_probabilities

        return compute_cell_type_probabilities(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )


# ------------------------------------------------------------------------------
# MCMC results class
# ------------------------------------------------------------------------------


class ScribeMCMCResults(MCMC):
    """
    SCRIBE MCMC results class that extends numpyro.infer.MCMC.

    This class inherits all functionality from MCMC while adding SCRIBE-specific
    attributes and methods for analyzing single-cell RNA sequencing data.

    Attributes
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Type of model used for inference
    model_config : ModelConfig
        Configuration object specifying model architecture and priors
    prior_params : Dict[str, Any]
        Dictionary of prior parameter values used during inference
    obs : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    var : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    n_obs : Optional[int]
        Number of observations (cells), if provided
    n_vars : Optional[int]
        Number of variables (genes), if provided
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """

    def __init__(
        self,
        mcmc,  # MCMC instance
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict] = None,
        n_obs: Optional[int] = None,
        n_vars: Optional[int] = None,
        predictive_samples: Optional[Dict] = None,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize ScribeMCMCResults from an existing MCMC instance.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance containing samples and diagnostics
        n_cells : int
            Number of cells in the dataset
        n_genes : int
            Number of genes in the dataset
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        obs : Optional[pd.DataFrame]
            Cell metadata
        var : Optional[pd.DataFrame]
            Gene metadata
        uns : Optional[Dict]
            Unstructured metadata
        n_obs : Optional[int]
            Number of observations (cells)
        n_vars : Optional[int]
            Number of variables (genes)
        predictive_samples : Optional[Dict]
            Predictive samples from the model
        n_components : Optional[int]
            Number of mixture components
        """
        # Initialize the MCMC parent class attributes
        # We don't call __init__ because we want to take attributes from the
        # existing mcmc instance Instead, we copy the relevant attributes from
        # the mcmc instance
        self.__dict__.update(mcmc.__dict__)

        # Set SCRIBE-specific attributes
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.model_type = model_type
        self.model_config = model_config
        self.prior_params = prior_params

        # AnnData-related attributes
        self.obs = obs
        self.var = var
        self.uns = uns
        self.n_obs = n_obs
        self.n_vars = n_vars

        # Optional attributes
        self.predictive_samples = predictive_samples
        self.prior_predictive_samples = None
        self.denoised_counts = None
        self.n_components = (
            n_components
            if n_components is not None
            else model_config.n_components
        )

        # Validate configuration
        self._validate_model_config()

    @classmethod
    def from_mcmc(
        cls,
        mcmc: MCMC,
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ):
        """
        Create ScribeMCMCResults from an existing MCMC instance.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance to extend
        n_cells : int
            Number of cells in the dataset
        n_genes : int
            Number of genes in the dataset
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for the model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        **kwargs
            Additional arguments to pass to ScribeMCMCResults constructor

        Returns
        -------
        ScribeMCMCResults
            Extended MCMC instance with SCRIBE functionality
        """
        # Create ScribeMCMCResults instance
        return cls(
            mcmc=mcmc,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            **kwargs,
        )

    # --------------------------------------------------------------------------
    # From AnnData
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        mcmc: MCMC,
        adata: "AnnData",
        model_type: str,
        model_config: ModelConfig,
        prior_params: Dict[str, Any],
        **kwargs,
    ):
        """
        Create ScribeMCMCResults from MCMC instance and AnnData object.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC instance to extend
        adata : AnnData
            AnnData object containing the data
        model_type : str
            Type of model used for inference
        model_config : ModelConfig
            Configuration object for the model
        prior_params : Dict[str, Any]
            Dictionary of prior parameter values
        **kwargs
            Additional arguments to pass to ScribeMCMCResults constructor

        Returns
        -------
        ScribeMCMCResults
            Extended MCMC instance with SCRIBE functionality
        """
        # Create ScribeMCMCResults instance
        return cls(
            mcmc=mcmc,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs,
        )

    # --------------------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------------------

    def _validate_model_config(self):
        """Validate model configuration matches model type."""
        # Validate base model
        # For mixture models, model_type is "nbdm_mix" but base_model is "nbdm"
        # Check if model_type matches base_model or base_model + "_mix"
        expected_base = (
            self.model_type[:-4]
            if self.model_type.endswith("_mix")
            else self.model_type
        )
        if self.model_config.base_model != expected_base:
            raise ValueError(
                f"Model type '{self.model_type}' does not match config "
                f"base model '{self.model_config.base_model}'"
            )

        # Validate n_components consistency
        if self.n_components is not None:
            if not self.model_type.endswith("_mix"):
                raise ValueError(
                    f"Model type '{self.model_type}' is not a mixture model "
                    f"but n_components={self.n_components} was specified"
                )
            if self.model_config.n_components != self.n_components:
                raise ValueError(
                    f"n_components mismatch: {self.n_components} vs "
                    f"{self.model_config.n_components} in model_config"
                )

    # --------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------

    def _is_component_specific_param(self, param_name: str) -> bool:
        """Check if a parameter is component-specific (mixture-specific).

        In the new system, each parameter's `is_mixture` field in `param_specs`
        indicates if it's component-specific. This replaces the deprecated
        `component_specific_params` boolean.

        Parameters
        ----------
        param_name : str
            Parameter name (e.g., "p", "phi")

        Returns
        -------
        bool
            True if the parameter has is_mixture=True in param_specs
        """
        if not self.model_config.param_specs:
            # Fallback: check mixture_params if param_specs not populated
            mixture_params = self.model_config.mixture_params or []
            return param_name in mixture_params

        # Check param_specs for is_mixture=True
        for spec in self.model_config.param_specs:
            if spec.name == param_name:
                return spec.is_mixture
        return False

    # --------------------------------------------------------------------------
    # Override get_samples to provide canonical samples by default
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Parameter conversion method
    # --------------------------------------------------------------------------

    def _convert_to_canonical(self, samples):
        """
        Convert samples to canonical (p, r) form based on present parameters.

        Parameters
        ----------
        samples : Dict
            Raw samples dictionary

        Returns
        -------
        Dict
            Canonical samples dictionary
        """
        # If no samples, return empty dict
        if not samples:
            return {}

        # Create a copy of samples to avoid modifying the original
        canonical_samples = samples.copy()

        # Convert based on what parameters are present in the samples

        # Handle odds_ratio parameterization (phi, mu -> p, r)
        if "phi" in canonical_samples and "mu" in canonical_samples:
            # Extract phi and mu
            phi = canonical_samples["phi"]
            mu = canonical_samples["mu"]
            # Compute p from phi
            canonical_samples["p"] = 1.0 / (1.0 + phi)

            # Reshape phi to broadcast with mu based on mixture model and
            # component specificity
            if self.n_components is not None:
                # Mixture model: mu has shape (n_samples, n_components, n_genes)
                if self._is_component_specific_param("phi"):
                    # Component-specific phi: shape (n_samples, n_components)
                    phi_reshaped = phi[:, :, None]
                else:
                    # Shared phi: shape (n_samples,) ->
                    # broadcast to (n_samples, 1, 1)
                    phi_reshaped = phi[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                phi_reshaped = phi[:, None]
            # Compute r from mu and phi
            canonical_samples["r"] = mu * phi_reshaped

        # Handle linked parameterization (mu, p -> r)
        elif (
            "mu" in canonical_samples
            and "p" in canonical_samples
            and "r" not in canonical_samples
        ):
            # Extract p and mu
            p = canonical_samples["p"]
            mu = canonical_samples["mu"]
            # Reshape p to broadcast with mu based on mixture model and
            # component specificity
            if self.n_components is not None:
                if self._is_component_specific_param("p"):
                    # Component-specific p: shape (n_samples, n_components)
                    p_reshaped = p[:, :, None]
                else:
                    # Shared p: shape (n_samples,) ->
                    # broadcast to (n_samples, 1, 1)
                    p_reshaped = p[:, None, None]
            else:
                # Non-mixture model: mu has shape (n_samples, n_genes)
                p_reshaped = p[:, None]
            # Compute r from mu and p
            canonical_samples["r"] = mu * (1.0 - p_reshaped) / p_reshaped

        # Handle unconstrained parameterization (r_unconstrained,
        # p_unconstrained, etc.)
        if (
            "r_unconstrained" in canonical_samples
            and "r" not in canonical_samples
        ):
            # compute r from r_unconstrained
            canonical_samples["r"] = jnp.exp(
                canonical_samples["r_unconstrained"]
            )

        if (
            "p_unconstrained" in canonical_samples
            and "p" not in canonical_samples
        ):
            # Compute p from p_unconstrained
            canonical_samples["p"] = sigmoid(
                canonical_samples["p_unconstrained"]
            )

        if (
            "gate_unconstrained" in canonical_samples
            and "gate" not in canonical_samples
        ):
            # Compute gate from gate_unconstrained
            canonical_samples["gate"] = sigmoid(
                canonical_samples["gate_unconstrained"]
            )

        # Handle VCP capture probability conversions
        if (
            "phi_capture" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Extract phi_capture
            phi_capture = canonical_samples["phi_capture"]
            # Compute p_capture from phi_capture
            canonical_samples["p_capture"] = 1.0 / (1.0 + phi_capture)

        if (
            "p_capture_unconstrained" in canonical_samples
            and "p_capture" not in canonical_samples
        ):
            # Compute p_capture from p_capture_unconstrained
            canonical_samples["p_capture"] = sigmoid(
                canonical_samples["p_capture_unconstrained"]
            )

        # Handle mixing weights computation for mixture models
        if (
            "mixing_logits_unconstrained" in canonical_samples
            and "mixing_weights" not in canonical_samples
        ):
            # Compute mixing weights from mixing_logits_unconstrained using
            # softmax
            canonical_samples["mixing_weights"] = softmax(
                canonical_samples["mixing_logits_unconstrained"], axis=-1
            )

        return canonical_samples

    def get_samples(self, group_by_chain=False, canonical=False):
        """
        Get samples from the MCMC run, with option to return canonical form.

        Parameters
        ----------
        group_by_chain : bool, default=False
            Whether to preserve the chain dimension. If True, all samples will
            have num_chains as the size of their leading dimension.
        canonical : bool, default=True
            Whether to return samples in canonical (p, r) form. If False,
            returns raw samples as they were collected during MCMC.

        Returns
        -------
        Dict
            Dictionary of parameter samples. If canonical=True, parameters are
            converted to canonical form. If canonical=False, returns raw
            samples.
        """
        # Get raw samples from parent class
        raw_samples = super().get_samples(group_by_chain=group_by_chain)

        if not canonical:
            return raw_samples

        # Convert to canonical form on-the-fly
        return self._convert_to_canonical(raw_samples)

    # --------------------------------------------------------------------------
    # SCRIBE-specific methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(self, canonical=True):
        """
        Get posterior samples from the MCMC run.

        This is a convenience method to match the ScribeResults interface.
        Returns canonical samples by default.

        Returns
        -------
        Dict
            Dictionary of parameter samples in canonical form
        """
        return self.get_samples(canonical=canonical)

    # --------------------------------------------------------------------------

    def get_posterior_quantiles(
        self, param, quantiles=(0.025, 0.5, 0.975), canonical=True
    ):
        """
        Get quantiles for a specific parameter from MCMC samples.

        Parameters
        ----------
        param : str
            Parameter name
        quantiles : tuple, default=(0.025, 0.5, 0.975)
            Quantiles to compute

        Returns
        -------
        dict
            Dictionary mapping quantiles to values
        """
        return _get_posterior_quantiles(
            self.get_samples(canonical=canonical), param, quantiles
        )

    # --------------------------------------------------------------------------

    def get_map(self, canonical=True):
        """
        Get the maximum a posteriori (MAP) estimate from MCMC samples.

        For each parameter, this returns the value with the highest
        posterior density.

        Returns
        -------
        dict
            Dictionary of MAP estimates for each parameter
        """
        samples = self.get_samples(canonical=canonical)

        # Get extra fields to compute joint log density
        try:
            potential_energy = self.get_extra_fields()["potential_energy"]
            return _get_map_estimate(samples, potential_energy)
        except:
            # Fallback: Use general function without potential energy
            return _get_map_estimate(samples)

    # --------------------------------------------------------------------------
    # Get model function
    # --------------------------------------------------------------------------

    def _model(self) -> Callable:
        """Get the model function for this model type."""
        model = _get_model_fn(self.model_config)
        return model

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def _log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        return _get_log_likelihood_fn(self.model_type)

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Generate posterior predictive check (PPC) samples using posterior
        parameter samples.

        This method uses the posterior parameter samples to generate new data
        from the model. These samples can be used to assess model fit and
        perform posterior predictive checks.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key. Defaults to random.PRNGKey(42) if None
        batch_size : int, optional
            Batch size for generating samples. If None, uses full dataset.
        store_samples : bool, optional
            Whether to store the generated samples in self.predictive_samples
            (default: True)

        Returns
        -------
        jnp.ndarray
            Array of posterior predictive samples
        """
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        # Generate predictive samples
        predictive_samples = _generate_ppc_samples(
            self.get_samples(canonical=True),
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples

        return predictive_samples

    # --------------------------------------------------------------------------
    # Biological (denoised) posterior predictive checks
    # --------------------------------------------------------------------------

    def get_ppc_samples_biological(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        cell_batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """Generate biological posterior predictive check samples.

        Samples from the base Negative Binomial NB(r, p) only, stripping
        technical noise parameters such as cell-specific capture
        probability (``p_capture`` / ``phi_capture``) and zero-inflation
        gate.  For NBDM models the result is identical to
        :meth:`get_ppc_samples`.

        Uses the full posterior MCMC samples of ``r`` and ``p`` (and
        ``mixing_weights`` for mixture models).  For each posterior draw
        a count matrix of shape ``(n_cells, n_genes)`` is generated by
        sampling from NB(r, p) directly, yielding a denoised view of the
        data that reflects the underlying biology.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        batch_size : int or None, optional
            Not used directly (kept for API symmetry with
            :meth:`get_ppc_samples`).  Use ``cell_batch_size`` to
            control memory usage.
        store_samples : bool, optional
            If ``True``, stores samples in
            ``self.predictive_samples_biological``.  Default: ``True``.
        cell_batch_size : int or None, optional
            If set, cells are processed in batches of this size to limit
            peak memory.  ``None`` processes all cells at once.

        Returns
        -------
        jnp.ndarray
            Biological count samples with shape
            ``(n_posterior_samples, n_cells, n_genes)``.

        See Also
        --------
        get_ppc_samples : Standard PPC including technical noise.
        scribe.sampling.sample_biological_nb : Core sampling utility.

        Notes
        -----
        The biological PPC is motivated by the Dirichlet-Multinomial
        model derivation: the composition of NB with a Binomial capture
        step yields NB with an effective :math:`\\hat{p}`.  By sampling
        from NB(r, p) directly we recover the latent biological
        distribution.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Get canonical posterior samples (contains r, p, and optionally
        # mixing_weights, gate, p_capture, etc.)
        canonical_samples = self.get_samples(canonical=True)

        r = canonical_samples["r"]
        p = canonical_samples["p"]
        mixing_weights = canonical_samples.get("mixing_weights", None)

        # Generate biological (denoised) count samples
        bio_samples = sample_biological_nb(
            r=r,
            p=p,
            n_cells=self.n_cells,
            rng_key=rng_key,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if store_samples:
            self.predictive_samples_biological = bio_samples

        return bio_samples

    # --------------------------------------------------------------------------
    # Bayesian denoising of observed counts
    # --------------------------------------------------------------------------

    def denoise_counts(
        self,
        counts: jnp.ndarray,
        method: str = "mean",
        rng_key: Optional[random.PRNGKey] = None,
        return_variance: bool = False,
        cell_batch_size: Optional[int] = None,
        store_result: bool = True,
        verbose: bool = True,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Denoise observed counts using MCMC posterior samples.

        Propagates parameter uncertainty by computing the denoised
        posterior for each MCMC draw.  The result has a leading
        ``n_posterior_samples`` dimension that can be summarised
        (e.g. ``.mean(axis=0)`` for a Bayesian point estimate).

        For VCP models this accounts for per-cell capture probability.
        For ZINB variants it corrects zero observations for technical
        dropout.  For NBDM the result is the identity.

        Parameters
        ----------
        counts : jnp.ndarray
            Observed UMI count matrix ``(n_cells, n_genes)``.
        method : {'mean', 'mode', 'sample'}, optional
            Summary of the per-sample denoised posterior.

            * ``'mean'``: closed-form posterior mean (shrinkage estimator).
            * ``'mode'``: posterior mode (MAP denoised count).
            * ``'sample'``: one stochastic draw per cell/gene per sample.

            Default: ``'mean'``.
        rng_key : random.PRNGKey or None, optional
            JAX PRNG key.  Defaults to ``random.PRNGKey(42)``.
        return_variance : bool, optional
            If ``True``, return dict with ``'denoised_counts'`` and
            ``'variance'``.  Default: ``False``.
        cell_batch_size : int or None, optional
            Cell batching inside each posterior draw.
        store_result : bool, optional
            Store result in ``self.denoised_counts``.  Default: ``True``.
        verbose : bool, optional
            Print progress messages.  Default: ``True``.

        Returns
        -------
        jnp.ndarray or Dict[str, jnp.ndarray]
            Denoised counts with shape
            ``(n_posterior_samples, n_cells, n_genes)`` (or dict with
            variance when ``return_variance=True``).

        See Also
        --------
        get_ppc_samples_biological : Biological PPC (not conditioned on
            observed counts).
        scribe.sampling.denoise_counts : Core denoising utility.
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        canonical_samples = self.get_samples(canonical=True)

        r = canonical_samples["r"]
        p = canonical_samples["p"]
        p_capture = canonical_samples.get("p_capture")
        gate = canonical_samples.get("gate")
        mixing_weights = canonical_samples.get("mixing_weights")

        if verbose:
            extras = []
            if p_capture is not None:
                extras.append("VCP")
            if gate is not None:
                extras.append("gate")
            extra_str = f" [{', '.join(extras)}]" if extras else ""
            n_post = r.shape[0]
            print(
                f"Denoising with {n_post} MCMC samples"
                f" ({self.model_type}){extra_str}, method='{method}'..."
            )

        result = _denoise_counts_util(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=rng_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            cell_batch_size=cell_batch_size,
        )

        if verbose:
            shape = (
                result["denoised_counts"].shape
                if return_variance
                else result.shape
            )
            print(f"Denoised counts shape: {shape}")

        if store_result:
            self.denoised_counts = (
                result["denoised_counts"] if return_variance else result
            )

        return result

    # --------------------------------------------------------------------------
    # Count normalization methods
    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples_dirichlet: int = 1000,
        fit_distribution: bool = True,
        store_samples: bool = False,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        batch_size: int = 2048,
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Normalize counts using posterior samples of the r parameter.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key. Defaults to random.PRNGKey(42).
        n_samples_dirichlet : int, default=1000
            Number of samples to draw from each Dirichlet distribution.
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the generated samples.
        store_samples : bool, default=False
            If True, includes the raw Dirichlet samples in the output.
        sample_axis : int, default=0
            Axis for Dirichlet fitting.
        return_concentrations : bool, default=False
            If True, returns the original r parameter samples.
        backend : str, default="numpyro"
            ``"numpyro"`` or ``"scipy"`` for distribution objects.
        batch_size : int, default=2048
            Number of posterior samples per batched Dirichlet sampling call.
            Larger values use more GPU memory but fewer dispatches.
        verbose : bool, default=True
            If True, prints progress messages.

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Normalized expression results.

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet.
        """
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)

        # Get canonical samples using the new method
        posterior_samples = self.get_samples(canonical=True)

        # Use the shared normalization function
        return normalize_counts_from_posterior(
            posterior_samples=posterior_samples,
            n_components=self.n_components,
            rng_key=rng_key,
            n_samples_dirichlet=n_samples_dirichlet,
            fit_distribution=fit_distribution,
            store_samples=store_samples,
            sample_axis=sample_axis,
            return_concentrations=return_concentrations,
            backend=backend,
            batch_size=batch_size,
            verbose=verbose,
        )

    # --------------------------------------------------------------------------
    # Prior predictive sampling methods
    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: Optional[random.PRNGKey] = None,
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """
        Generate prior predictive samples using the model.

        This method generates samples from the prior predictive distribution by
        first sampling parameters from the prior distributions and then
        generating data from the model using these parameters. These samples can
        be used to assess model behavior before seeing any data.

        Parameters
        ----------
        rng_key : random.PRNGKey, optional
            JAX random number generator key (default: PRNGKey(42))
        n_samples : int, optional
            Number of prior predictive samples to generate (default: 100)
        batch_size : int, optional
            Batch size for generating samples. If None, uses full dataset.
        store_samples : bool, optional
            Whether to store the generated samples in
            self.prior_predictive_samples (default: True)

        Returns
        -------
        jnp.ndarray
            Array of prior predictive samples
        """
        # Create default RNG key if not provided (lazy initialization)
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        # Generate prior predictive samples
        prior_predictive_samples = _generate_prior_predictive_samples(
            self.model_type,
            self.n_cells,
            self.n_genes,
            self.model_config,
            rng_key=rng_key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        # Store samples if requested
        if store_samples:
            self.prior_predictive_samples = prior_predictive_samples

        return prior_predictive_samples

    # --------------------------------------------------------------------------
    # Compute log likelihood methods
    # --------------------------------------------------------------------------

    def log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        sample_chunk_size: Optional[int] = None,
        return_by: str = "cell",
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data under posterior samples.

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate likelihood on
        batch_size : Optional[int], default=None
            Size of mini-batches used for likelihood computation
        return_by : str, default='cell'
            Specifies how to return the log probabilities. Must be one of:
                - 'cell': returns log probabilities summed over genes
                - 'gene': returns log probabilities summed over cells
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        split_components : bool, default=False
            If True, returns log likelihoods for each mixture component
            separately. Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations

        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters. For standard models:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=False:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            For mixture models with split_components=True:
                - 'cell': shape (n_samples, n_cells, n_components)
                - 'gene': shape (n_samples, n_genes, n_components)

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Get canonical samples using the new method
        samples = self.get_samples(canonical=True)

        return _compute_log_likelihood(
            samples,
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
            sample_chunk_size=sample_chunk_size,
            return_by=return_by,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=split_components,
            weights=weights,
            weight_type=weight_type,
            dtype=dtype,
        )

    # --------------------------------------------------------------------------
    # Indexing by genes
    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        When model_config.param_specs is set, use metadata-based subsetting;
        otherwise use last-axis heuristic as fallback.
        """
        if samples is None:
            return None

        new_samples = {}
        original_n_genes = self.n_genes
        gene_axis_by_key = None
        if getattr(self.model_config, "param_specs", None):
            gene_axis_by_key = build_gene_axis_by_key(
                self.model_config.param_specs, samples, original_n_genes
            )

        for key, value in samples.items():
            if not hasattr(value, "ndim"):
                new_samples[key] = value
                continue
            if gene_axis_by_key is not None and key in gene_axis_by_key:
                gene_axis = gene_axis_by_key[key]
                slicer = [slice(None)] * value.ndim
                slicer[gene_axis] = index
                new_samples[key] = value[tuple(slicer)]
                continue
            if value.ndim > 0 and value.shape[-1] == original_n_genes:
                new_samples[key] = value[..., index]
            else:
                new_samples[key] = value
        return new_samples

    def __getitem__(self, index):
        """
        Enable indexing of ``ScribeMCMCResults`` by genes and components.

        This allows selecting a subset of genes for analysis, e.g.:
        results[0:10]  # Get first 10 genes
        results[gene_indices]  # Get genes by index

        Parameters
        ----------
        index : int, slice, or array-like
            Indices of genes to select

        Returns
        -------
        ScribeMCMCSubset
            A new ScribeMCMCResults object containing only the selected genes
        """
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(
                    "Tuple indexing must be (gene_indexer, component_indexer)."
                )
            gene_indexer, component_indexer = index
            gene_subset = self[gene_indexer]
            return gene_subset.get_component(component_indexer)

        # If index is a boolean mask, use it directly
        if isinstance(index, (jnp.ndarray, np.ndarray)) and index.dtype == bool:
            bool_index = index
        # Handle integer indexing
        elif isinstance(index, int):
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given index
            bool_index = bool_index.at[index].set(True)
        # Handle slice indexing
        elif isinstance(index, slice):
            # Get indices from slice
            indices = jnp.arange(self.n_genes)[index]
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(
            index[-1], (bool, jnp.bool_)
        ):
            # Get indices from list/array
            indices = jnp.array(index)
            # Initialize boolean index
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
        else:
            # Already a boolean index
            bool_index = index

        # Create new metadata if available
        new_var = self.var.iloc[bool_index] if self.var is not None else None

        # Get subset of samples
        samples = self.get_samples(canonical=False)
        new_samples = self._subset_posterior_samples(samples, bool_index)

        # Create and return the subset using the module-level class.
        return ScribeMCMCSubset(
            samples=new_samples,
            n_cells=self.n_cells,
            n_genes=int(
                bool_index.sum()
                if hasattr(bool_index, "sum")
                else len(bool_index)
            ),
            model_type=self.model_type,
            model_config=self.model_config,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            n_components=self.n_components,
        )

    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index, renormalize: bool = True):
        """
        Return a component-restricted view of mixture model results.

        Parameters
        ----------
        component_index : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCSubset
            Subset containing requested components.
        """
        selected = normalize_component_indices(
            component_index, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))
        return self.get_components(selected, renormalize=renormalize)

    # --------------------------------------------------------------------------

    def get_components(
        self, component_indices, renormalize: bool = True
    ) -> ScribeMCMCSubset:
        """
        Select multiple components while preserving mixture semantics.

        Parameters
        ----------
        component_indices : int, slice, array-like, or bool mask
            Selector over mixture components.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.

        Returns
        -------
        ScribeMCMCSubset
            Mixture subset with reduced ``n_components``.
        """
        selected = normalize_component_indices(
            component_indices, self.n_components
        )
        if int(selected.shape[0]) == 1:
            return self._create_single_component_subset(int(selected[0]))

        samples = self.get_samples(canonical=False)
        component_samples = self._subset_samples_by_components(
            samples,
            selected,
            renormalize=renormalize,
            squeeze_single=False,
        )
        new_n_components = int(selected.shape[0])
        modified_model_config = self.model_config.model_copy(
            update={"n_components": new_n_components}
        )
        return ScribeMCMCSubset(
            samples=component_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=self.model_type,
            model_config=modified_model_config,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=new_n_components,
        )

    # --------------------------------------------------------------------------

    def _create_single_component_subset(
        self, component_index: int
    ) -> ScribeMCMCSubset:
        """Create legacy non-mixture subset for one selected component."""
        samples = self.get_samples(canonical=False)
        component_samples = self._subset_samples_by_components(
            samples,
            jnp.asarray([component_index], dtype=jnp.int32),
            renormalize=True,
            squeeze_single=True,
        )
        base_model = self.model_type.replace("_mix", "")
        modified_model_config = self.model_config.model_copy(
            update={"base_model": base_model, "n_components": None}
        )
        return ScribeMCMCSubset(
            samples=component_samples,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,
            model_config=modified_model_config,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            n_components=None,
        )

    # --------------------------------------------------------------------------

    def _subset_samples_by_components(
        self,
        samples: Dict,
        component_indices: jnp.ndarray,
        renormalize: bool = True,
        squeeze_single: bool = False,
    ) -> Dict:
        """
        Subset sample dictionary along the mixture-component axis.

        Parameters
        ----------
        samples : Dict
            Raw sample dictionary.
        component_indices : jnp.ndarray
            One-dimensional selected component indices.
        renormalize : bool, default=True
            Whether to renormalize selected mixture fractions.
        squeeze_single : bool, default=False
            If True and one component is selected, remove the component axis.

        Returns
        -------
        Dict
            Sample dictionary restricted to selected components.
        """
        if samples is None:
            return None

        specs_by_name = {}
        if getattr(self.model_config, "param_specs", None):
            specs_by_name = {s.name: s for s in self.model_config.param_specs}

        n_comp = self.n_components
        use_single = squeeze_single and int(component_indices.shape[0]) == 1
        single_index = int(component_indices[0]) if use_single else None
        new_samples: Dict = {}

        for key, values in samples.items():
            spec = specs_by_name.get(key)
            has_mixture_axis = hasattr(values, "ndim") and values.ndim > 1
            is_named_mixture_weight = key in {
                "mixing_weights",
                "mixing_logits_unconstrained",
            }
            is_mixture = (
                has_mixture_axis
                and (
                    (spec is not None and spec.is_mixture)
                    or is_named_mixture_weight
                )
            )
            if not is_mixture:
                new_samples[key] = values
                continue

            component_axis = 1
            if values.shape[component_axis] != n_comp:
                candidates = [
                    ax
                    for ax in range(1, values.ndim)
                    if values.shape[ax] == n_comp
                ]
                if not candidates:
                    new_samples[key] = values
                    continue
                component_axis = candidates[0]

            slicer = [slice(None)] * values.ndim
            slicer[component_axis] = (
                single_index if use_single else component_indices
            )
            selected = values[tuple(slicer)]

            if renormalize and key == "mixing_weights" and not use_single:
                selected = renormalize_mixing_weights(selected, axis=-1)
            elif (
                renormalize
                and key == "mixing_logits_unconstrained"
                and not use_single
            ):
                selected = renormalize_mixing_logits(selected, axis=-1)

            new_samples[key] = selected

        return new_samples

    # --------------------------------------------------------------------------
    # Cell type assignment method for mixture models
    # --------------------------------------------------------------------------

    def cell_type_probabilities(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        temperature: Optional[float] = None,
        weights: Optional[jnp.ndarray] = None,
        weight_type: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments and fit Dirichlet
        distributions to characterize assignment uncertainty.

        For each cell, this method:
            1. Computes component-specific log-likelihoods using posterior
               samples
            2. Converts these to probability distributions over cell types
            3. Fits a Dirichlet distribution to characterize the uncertainty in
               these assignments

        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        ignore_nans : bool, default=False
            If True, removes any samples that contain NaNs.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the assignment
            probabilities
        temperature : Optional[float], default=None
            If provided, apply temperature scaling to log probabilities
        weights : Optional[jnp.ndarray], default=None
            Array used to weight genes when computing log likelihoods
        weight_type : Optional[str], default=None
            How to apply weights. Must be one of:
                - 'multiplicative': multiply log probabilities by weights
                - 'additive': add weights to log probabilities
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'concentration': Dirichlet concentration parameters for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'mean_probabilities': Mean assignment probabilities for each
                  cell. Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'sample_probabilities': Assignment probabilities for each
                  posterior sample. Shape: (n_samples, n_cells, n_components)

        Raises
        ------
        ValueError
            - If the model is not a mixture model
            - If posterior samples have not been generated yet

        Note
        ----
        Most of the log-likelihood value differences between cell types are
        extremely large. Thus, the computation usually returns either 0 or 1.
        This computation is therefore not very useful, but it is included for
        completeness.
        """
        from ..core.cell_type_assignment import compute_cell_type_probabilities

        return compute_cell_type_probabilities(
            results=self,
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )


# ------------------------------------------------------------------------------
# Shared helper functions for both ScribeMCMCResults and ScribeMCMCSubset
# ------------------------------------------------------------------------------


def _get_model_fn(model_config) -> Callable:
    """Get the model function for the given model configuration.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration containing all necessary parameters

    Returns
    -------
    Callable
        The model function
    """
    from ..models.model_registry import get_model_and_guide

    return get_model_and_guide(model_config, guide_families=None)[0]


# ------------------------------------------------------------------------------


def _get_log_likelihood_fn(model_type: str) -> Callable:
    """Get the log likelihood function for this model type."""
    from scribe.models.model_registry import get_log_likelihood_fn

    return get_log_likelihood_fn(model_type)


# ------------------------------------------------------------------------------


def _compute_log_likelihood(
    samples: Dict,
    counts: jnp.ndarray,
    model_type: str,
    n_components: Optional[int] = None,
    batch_size: Optional[int] = None,
    sample_chunk_size: Optional[int] = None,
    return_by: str = "cell",
    cells_axis: int = 0,
    ignore_nans: bool = False,
    split_components: bool = False,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Compute log likelihood of data under posterior samples."""
    # Get number of samples from first parameter
    n_samples = samples[next(iter(samples))].shape[0]

    # Get likelihood function
    likelihood_fn = _get_log_likelihood_fn(model_type)

    # Determine if this is a mixture model
    is_mixture = n_components is not None and n_components > 1

    # Define function to compute likelihood for a single sample
    @jit
    def compute_sample_lik(i):
        # Extract parameters for this sample
        params_i = {k: v[i] for k, v in samples.items()}
        # For mixture models we need to pass split_components and weights
        if is_mixture:
            return likelihood_fn(
                counts,
                params_i,
                batch_size=batch_size,
                cells_axis=cells_axis,
                return_by=return_by,
                split_components=split_components,
                weights=weights,
                weight_type=weight_type,
                dtype=dtype,
            )
        else:
            return likelihood_fn(
                counts,
                params_i,
                batch_size=batch_size,
                cells_axis=cells_axis,
                return_by=return_by,
                dtype=dtype,
            )

    # Use chunked vmap to reduce peak memory when requested.
    if (
        sample_chunk_size is None
        or sample_chunk_size <= 0
        or sample_chunk_size >= n_samples
    ):
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
    else:
        chunks = []
        for start in range(0, n_samples, sample_chunk_size):
            end = min(start + sample_chunk_size, n_samples)
            chunks.append(vmap(compute_sample_lik)(jnp.arange(start, end)))
        log_liks = jnp.concatenate(chunks, axis=0)

    # Handle NaNs if requested
    if ignore_nans:
        # Check for NaNs appropriately based on dimensions
        if is_mixture and split_components:
            # Handle case with component dimension
            valid_samples = ~jnp.any(
                jnp.any(jnp.isnan(log_liks), axis=-1), axis=-1
            )
        else:
            # Standard case
            valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=-1)

        # Filter out samples with NaNs
        if jnp.any(~valid_samples):
            print(
                f"    - Fraction of samples removed: "
                f"{1 - jnp.mean(valid_samples)}"
            )
            return log_liks[valid_samples]

    return log_liks


# ------------------------------------------------------------------------------


def _get_posterior_quantiles(
    samples: Dict, param: str, quantiles=(0.025, 0.5, 0.975)
):
    """Get quantiles for a specific parameter from MCMC samples."""
    param_samples = samples[param]
    return {q: jnp.quantile(param_samples, q) for q in quantiles}


# ------------------------------------------------------------------------------


def _get_map_estimate(
    samples: Dict, potential_energy: Optional[jnp.ndarray] = None
):
    """Get the maximum a posteriori (MAP) estimate from samples."""
    if potential_energy is not None:
        # Get index of minimum potential energy (maximum log density)
        map_idx = int(jnp.argmin(potential_energy))
        # Extract parameters at this index
        map_estimate = {
            param: values[map_idx] for param, values in samples.items()
        }
        return map_estimate
    else:
        # Fallback: Return posterior mean as a robust estimator
        map_estimate = {}
        for param, values in samples.items():
            # Using mean as a more robust estimator than mode
            map_estimate[param] = jnp.mean(values, axis=0)
        return map_estimate


# ------------------------------------------------------------------------------


def _generate_ppc_samples(
    samples: Dict,
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples."""
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Get the model function
    model = _get_model_fn(model_config)

    # Prepare base model arguments
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    # Generate predictive samples
    from scribe.sampling import generate_predictive_samples

    return generate_predictive_samples(
        model,
        samples,
        model_args,
        rng_key=rng_key,
        batch_size=batch_size,
    )


# ------------------------------------------------------------------------------


def _generate_prior_predictive_samples(
    model_type: str,
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate prior predictive samples using the model."""
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)
    
    # Get the model function
    model = _get_model_fn(model_config)

    # Prepare base model arguments
    model_args = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "model_config": model_config,
    }

    # Generate prior predictive samples
    from scribe.sampling import generate_prior_predictive_samples

    return generate_prior_predictive_samples(
        model,
        model_args,
        rng_key=rng_key,
        n_samples=n_samples,
        batch_size=batch_size,
    )
