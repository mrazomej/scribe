"""
Results classes for SCRIBE MCMC inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, field
import warnings

import jax.numpy as jnp
from jax import random, jit, vmap
import pandas as pd
from numpyro.infer import MCMC
import numpy as np

from ..sampling import generate_predictive_samples
from ..models.model_config import ModelConfig
from ..core.normalization import normalize_counts_from_posterior

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
        # Extract posterior means as point estimates
        samples = mcmc.get_samples()

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
        # Extract posterior means as point estimates
        samples = mcmc.get_samples()
        params = {
            param: jnp.mean(values, axis=0) for param, values in samples.items()
        }

        # Create ScribeMCMCResults instance
        return cls(
            mcmc=mcmc,
            params=params,
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
        if self.model_config.base_model != self.model_type:
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
    # SCRIBE-specific methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(self):
        """
        Get posterior samples from the MCMC run.

        This is a convenience method to match the ScribeResults interface.

        Returns
        -------
        Dict
            Dictionary of parameter samples
        """
        return self.get_samples()

    # --------------------------------------------------------------------------

    def get_posterior_quantiles(self, param, quantiles=(0.025, 0.5, 0.975)):
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
        return _get_posterior_quantiles(self.get_samples(), param, quantiles)

    # --------------------------------------------------------------------------

    def get_map(self):
        """
        Get the maximum a posteriori (MAP) estimate from MCMC samples.

        For each parameter, this returns the value with the highest
        posterior density.

        Returns
        -------
        dict
            Dictionary of MAP estimates for each parameter
        """
        samples = self.get_samples()

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
        model = _get_model_fn(self.model_type, self.model_config)
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
        rng_key: random.PRNGKey = random.PRNGKey(42),
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
            JAX random number generator key (default: PRNGKey(42))
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
        # Generate predictive samples
        predictive_samples = _generate_ppc_samples(
            self.get_samples(),
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
    # Count normalization methods
    # --------------------------------------------------------------------------

    def normalize_counts(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples_dirichlet: int = 1000,
        fit_distribution: bool = True,
        store_samples: bool = False,
        sample_axis: int = 0,
        return_concentrations: bool = False,
        backend: str = "numpyro",
        verbose: bool = True,
    ) -> Dict[str, Union[jnp.ndarray, object]]:
        """
        Normalize counts using posterior samples of the r parameter.

        This method takes posterior samples of the dispersion parameter (r) and
        uses them as concentration parameters for Dirichlet distributions to generate
        normalized expression profiles. For mixture models, normalization is performed
        per component, resulting in an extra dimension in the output.

        Based on the insights from the Dirichlet-multinomial model derivation, the
        r parameters represent the concentration parameters of a Dirichlet distribution
        that can be used to generate normalized expression profiles.

        The method generates Dirichlet samples using all posterior samples of r, then
        fits a single Dirichlet distribution to all these samples (or one per component
        for mixture models).

        Parameters
        ----------
        rng_key : random.PRNGKey, default=random.PRNGKey(42)
            JAX random number generator key
        n_samples_dirichlet : int, default=1000
            Number of samples to draw from each Dirichlet distribution
        fit_distribution : bool, default=True
            If True, fits a Dirichlet distribution to the generated samples using
            fit_dirichlet_minka from stats.py
        store_samples : bool, default=False
            If True, includes the raw Dirichlet samples in the output
        sample_axis : int, default=0
            Axis containing samples in the Dirichlet fitting (passed to fit_dirichlet_minka)
        return_concentrations : bool, default=False
            If True, returns the original r parameter samples used as concentrations
        backend : str, default="numpyro"
            Statistical package to use for distributions when fit_distribution=True.
            Must be one of:
            - "numpyro": Returns numpyro.distributions.Dirichlet objects
            - "scipy": Returns scipy.stats distributions via numpyro_to_scipy conversion
        verbose : bool, default=True
            If True, prints progress messages

        Returns
        -------
        Dict[str, Union[jnp.ndarray, object]]
            Dictionary containing normalized expression profiles. Keys depend on
            input arguments:
            - 'samples': Raw Dirichlet samples (if store_samples=True)
            - 'concentrations': Fitted concentration parameters (if fit_distribution=True)
            - 'mean_probabilities': Mean probabilities from fitted distribution (if fit_distribution=True)
            - 'distributions': Dirichlet distribution objects (if fit_distribution=True)
            - 'original_concentrations': Original r parameter samples (if return_concentrations=True)

            For non-mixture models:
            - samples: shape (n_posterior_samples, n_genes, n_samples_dirichlet) or
                      (n_posterior_samples, n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_genes,) - single fitted distribution
            - mean_probabilities: shape (n_genes,) - single fitted distribution
            - distributions: single Dirichlet distribution object

            For mixture models:
            - samples: shape (n_posterior_samples, n_components, n_genes, n_samples_dirichlet) or
                      (n_posterior_samples, n_components, n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_components, n_genes) - one fitted distribution per component
            - mean_probabilities: shape (n_components, n_genes) - one fitted distribution per component
            - distributions: list of n_components Dirichlet distribution objects

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet, or if 'r' parameter
            is not found in posterior samples

        Examples
        --------
        >>> # For a non-mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_genes,)
        >>> print(type(normalized['distributions']))  # Single Dirichlet distribution

        >>> # For a mixture model
        >>> normalized = results.normalize_counts(
        ...     n_samples_dirichlet=100,
        ...     fit_distribution=True
        ... )
        >>> print(normalized['mean_probabilities'].shape)  # (n_components, n_genes)
        >>> print(len(normalized['distributions']))  # n_components
        """
        # Get posterior samples from MCMC
        posterior_samples = self.get_posterior_samples()

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
    # Prior predictive sampling methods
    # --------------------------------------------------------------------------

    def get_prior_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
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
        return _compute_log_likelihood(
            self.get_samples(),
            counts,
            self.model_type,
            n_components=self.n_components,
            batch_size=batch_size,
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
        """
        if samples is None:
            return None

        new_posterior_samples = {}

        # Handle gene-specific parameters
        for param_name, values in samples.items():
            if param_name in [
                "r",
                "r_unconstrained",
                "r_unconstrained__decentered",
                "gate",
                "gate_unconstrained",
                "gate_unconstrained__decentered",
            ]:
                if self.n_components is not None:
                    # Shape: (n_samples, n_components, n_genes)
                    new_posterior_samples[param_name] = values[..., index]
                else:
                    # Shape: (n_samples, n_genes)
                    new_posterior_samples[param_name] = values[..., index]
            else:
                # Copy non-gene-specific parameters as is
                new_posterior_samples[param_name] = values

        return new_posterior_samples

    def __getitem__(self, index):
        """
        Enable indexing of ScribeMCMCResults object by genes.

        This allows selecting a subset of genes for analysis, e.g.:
        results[0:10]  # Get first 10 genes
        results[gene_indices]  # Get genes by index

        Parameters
        ----------
        index : int, slice, or array-like
            Indices of genes to select

        Returns
        -------
        ScribeMCMCResults
            A new ScribeMCMCResults object containing only the selected genes
        """
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
        samples = self.get_samples()
        new_samples = self._subset_posterior_samples(samples, bool_index)

        # Create new instance with subset data
        # We can't use inheritance for the subset since we need to detach from
        # the mcmc instance Instead, create a lightweight version that stores
        # the subset data
        from dataclasses import dataclass

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

            # ------------------------------------------------------------------

            def __getitem__(self, index):
                """Support further indexing of subset."""
                # Convert subset indexing to original indexing
                if self.var is not None:
                    if isinstance(index, int):
                        return self.var.index[index]
                    elif isinstance(index, slice):
                        return self.var.index[index]
                    else:
                        return self.var.index[index]
                else:
                    # If no var metadata, just return the index
                    return index

            # ------------------------------------------------------------------

            def get_posterior_samples(self):
                """Get posterior samples."""
                return self.samples

            # ------------------------------------------------------------------

            def get_posterior_quantiles(
                self, param, quantiles=(0.025, 0.5, 0.975)
            ):
                """Get quantiles for a specific parameter from samples."""
                return _get_posterior_quantiles(self.samples, param, quantiles)

            # ------------------------------------------------------------------

            def get_map(self):
                """Get MAP estimates for parameters."""
                return _get_map_estimate(self.samples)

            # ------------------------------------------------------------------

            def log_likelihood(
                self,
                counts: jnp.ndarray,
                batch_size: Optional[int] = None,
                return_by: str = "cell",
                cells_axis: int = 0,
                ignore_nans: bool = False,
                split_components: bool = False,
                weights: Optional[jnp.ndarray] = None,
                weight_type: Optional[str] = None,
                dtype: jnp.dtype = jnp.float32,
            ) -> jnp.ndarray:
                """Compute log likelihood of data under posterior samples."""
                return _compute_log_likelihood(
                    self.samples,
                    counts,
                    self.model_type,
                    n_components=self.n_components,
                    batch_size=batch_size,
                    return_by=return_by,
                    cells_axis=cells_axis,
                    ignore_nans=ignore_nans,
                    split_components=split_components,
                    weights=weights,
                    weight_type=weight_type,
                    dtype=dtype,
                )

            # ------------------------------------------------------------------

            def get_ppc_samples(
                self,
                rng_key: random.PRNGKey = random.PRNGKey(42),
                batch_size: Optional[int] = None,
                store_samples: bool = True,
            ) -> jnp.ndarray:
                """Generate predictive samples using posterior parameter samples."""
                predictive_samples = _generate_ppc_samples(
                    self.samples,
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

            # ------------------------------------------------------------------

            def get_prior_predictive_samples(
                self,
                rng_key: random.PRNGKey = random.PRNGKey(42),
                n_samples: int = 100,
                batch_size: Optional[int] = None,
                store_samples: bool = True,
            ) -> jnp.ndarray:
                """Generate prior predictive samples using the model."""
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

            # ------------------------------------------------------------------

            def normalize_counts(
                self,
                rng_key: random.PRNGKey = random.PRNGKey(42),
                n_samples_dirichlet: int = 1,
                fit_distribution: bool = False,
                store_samples: bool = True,
                sample_axis: int = 0,
                return_concentrations: bool = False,
                backend: str = "numpyro",
                verbose: bool = True,
            ) -> Dict[str, Union[jnp.ndarray, object]]:
                """Normalize counts using posterior samples of the r parameter."""
                # Use the shared normalization function
                return normalize_counts_from_posterior(
                    posterior_samples=self.samples,
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

        # ----------------------------------------------------------------------

        # Create and return the subset
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


# ------------------------------------------------------------------------------
# Shared helper functions for both ScribeMCMCResults and ScribeMCMCSubset
# ------------------------------------------------------------------------------


def _get_model_fn(model_type: str, model_config) -> Callable:
    """Get the model function for this model type and parameterization."""
    from ..models.model_registry import get_model_and_guide

    parameterization = model_config.parameterization or "standard"
    return get_model_and_guide(model_type, parameterization)[0]


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

    # Use vmap for parallel computation (more memory intensive)
    log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))

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
                f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}"
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
    rng_key: random.PRNGKey = random.PRNGKey(42),
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples."""
    # Get the model function
    model = _get_model_fn(model_type, model_config)

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
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Generate prior predictive samples using the model."""
    # Get the model function
    model = _get_model_fn(model_type, model_config)

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
