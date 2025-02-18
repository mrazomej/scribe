"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import numpyro.distributions as dist
from jax import random, jit, vmap

import scipy.stats as stats

from .sampling import (
    sample_variational_posterior, 
    generate_predictive_samples, 
    generate_ppc_samples
)
from .stats import (
    fit_dirichlet_minka, 
    beta_mode, 
    gamma_mode, 
    dirichlet_mode, 
    lognorm_mode
)
from .distributions import ModelConfig, GuideConfig

# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------

@dataclass
class BaseScribeResults(ABC):
    """
    Base class for SCRIBE inference results.
    
    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, loss history, and dataset dimensions. It can
    optionally store metadata from an AnnData object and posterior samples.

    Attributes
    ----------
    params : Dict
        Dictionary of inferred model parameters from SCRIBE
    loss_history : jnp.ndarray
        Array containing the ELBO loss values during training
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Type of model used for inference
    param_spec: Dict[str, Dict]
        Specification for parameters. Each parameter has a type ('global',
        'gene-specific', or 'cell-specific') and optionally 'component_specific'
        for mixture models.
    cell_metadata : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    gene_metadata : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    posterior_samples : Optional[Dict]
        Samples from the posterior distribution, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """
    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    guide_config: GuideConfig
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData object
    cell_metadata: Optional[pd.DataFrame] = None
    gene_metadata: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    
    # Optional results
    posterior_samples: Optional[Dict] = None
    n_components: Optional[int] = None

    # --------------------------------------------------------------------------

    def __post_init__(self):
        """Validate model configuration and parameters."""
        # Set n_components from model_config if not explicitly provided
        if self.n_components is None and self.model_config.n_components is not None:
            self.n_components = self.model_config.n_components
            
        self._validate_model_config()
        self._validate_guide_config()
        self._validate_parameters()

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
            if not self.model_type.endswith('_mix'):
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
    
    def _validate_guide_config(self):
        """Validate guide configuration is compatible with model config."""
        self.guide_config.validate_with_model(self.model_config)

    # --------------------------------------------------------------------------

    def _validate_parameters(self):
        """Validate parameters match model configuration."""
        # Get expected shapes for different parameter types
        gene_shape = (self.n_genes,)
        cell_shape = (self.n_cells,)
        global_shape = ()
        
        # Adjust shapes for mixture models
        if hasattr(self, 'n_components') and self.n_components is not None:
            gene_shape = (self.n_components, self.n_genes)
            global_shape = (self.n_components,)
        
        # Check r parameters
        r_params = {
            f"{param}_{var}": self.params[f"{param}_{var}"] 
            for param in self.model_config.r_distribution.param_names
            for var in ['r']
        }
        for param, value in r_params.items():
            if value.shape != gene_shape:
                raise ValueError(
                    f"Parameter {param} has shape {value.shape}, "
                    f"expected {gene_shape}"
                )
        
        # Check p parameters
        p_params = {
            f"{param}_{var}": self.params[f"{param}_{var}"] 
            for param in self.model_config.p_distribution.param_names
            for var in ['p']
        }
        for param, value in p_params.items():
            if value.shape != global_shape:
                raise ValueError(
                    f"Parameter {param} has shape {value.shape}, "
                    f"expected {global_shape}"
                )
        
        # Check gate parameters for zero-inflated models
        if self.model_config.gate_distribution is not None:
            gate_params = {
                f"{param}_{var}": self.params[f"{param}_{var}"] 
                for param in self.model_config.gate_distribution.param_names
                for var in ['gate']
            }
            for param, value in gate_params.items():
                if value.shape != gene_shape:
                    raise ValueError(
                        f"Parameter {param} has shape {value.shape}, "
                        f"expected {gene_shape}"
                    )
        
        # Check capture probability parameters for VCP models
        if self.model_config.p_capture_distribution is not None:
            p_capture_params = {
                f"{param}_{var}": self.params[f"{param}_{var}"] 
                for param in self.model_config.p_capture_distribution.param_names
                for var in ['p_capture']
            }
            for param, value in p_capture_params.items():
                if value.shape != cell_shape:
                    raise ValueError(
                        f"Parameter {param} has shape {value.shape}, "
                        f"expected {cell_shape}"
                    )
        
        # Check mixing weights for mixture models
        if hasattr(self, 'n_components') and self.n_components is not None:
            if 'alpha_mixing' not in self.params:
                raise ValueError("Mixture model requires 'alpha_mixing' parameter")
            if self.params['alpha_mixing'].shape != (self.n_components,):
                raise ValueError(
                    f"Parameter alpha_mixing has shape {self.params['alpha_mixing'].shape}, "
                    f"expected ({self.n_components},)"
                )
    # --------------------------------------------------------------------------
    
    

    # --------------------------------------------------------------------------
    # Create BaseScribeResults from AnnData object
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: "AnnData",
        params: Dict,
        loss_history: jnp.ndarray,
        **kwargs
    ):
        """Create ScribeResults from AnnData object."""
        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            cell_metadata=adata.obs.copy(),
            gene_metadata=adata.var.copy(),
            uns=adata.uns.copy(),
            **kwargs
        )

    # --------------------------------------------------------------------------
    # Abstract methods for model-specific results
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters.
        
        Parameters
        ----------
        backend : str, default="scipy"
            Statistical package to use for distributions. Must be one of:
            - "scipy": Returns scipy.stats distributions
            - "numpyro": Returns numpyro.distributions
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions
        """
        pass

    # --------------------------------------------------------------------------

    @abstractmethod
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational
        posterior.
        """
        pass

    # --------------------------------------------------------------------------

    @abstractmethod
    def get_model_args(self) -> Dict:
        """Get the model and guide arguments for this model type."""
        pass

    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        """
        new_params = dict(params)
        
        # Subset only gene-specific parameters
        for param_name, spec in self.param_spec.items():
            if spec['type'] == 'gene-specific':
                if spec.get('component_specific', False):
                    # Keep component dimension but subset gene dimension
                    new_params[param_name] = params[param_name][..., index]
                else:
                    # Just subset gene dimension
                    new_params[param_name] = params[param_name][index]
        
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # Use param_spec to determine how to subset each parameter
            for param_name, spec in self.param_spec.items():
                if param_name in param_samples:
                    if spec['type'] == 'gene-specific':
                        if spec.get('component_specific', False):
                            # Keep samples and component dimensions, subset gene dimension
                            new_param_samples[param_name] = param_samples[param_name][..., index]
                        else:
                            # Keep samples dimension, subset gene dimension
                            new_param_samples[param_name] = param_samples[param_name][..., index]
                    else:
                        # Keep non-gene-specific parameters as is
                        new_param_samples[param_name] = param_samples[param_name]
            
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
        
        return new_posterior_samples

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of BaseScribeResults object.
        """
        # Handle integer indexing
        if isinstance(index, int):
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given index
            bool_index = bool_index.at[index].set(True)
            # Set index to boolean index
            index = bool_index
        
        # Handle slice indexing
        elif isinstance(index, slice):
            # Get indices from slice
            indices = jnp.arange(self.n_genes)[index]
            # Initialize boolean index
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
            # Set index to boolean index
            index = bool_index
        
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(index[-1], (bool, jnp.bool_)):
            # Get indices from list/array
            indices = jnp.array(index)
            # Initialize boolean index
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
            # Set index to boolean index
            index = bool_index

        # Create new params dict with subset of parameters
        new_params = self._subset_params(self.params, index)
        
        # Create new metadata if available
        new_gene_metadata = self.gene_metadata.iloc[index] if self.gene_metadata is not None else None

        # Create new posterior samples if available
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_samples(self.posterior_samples, index)
            
        # Let subclass create its own instance with additional attributes
        return self._create_subset(
            index=index,
            new_params=new_params,
            new_gene_metadata=new_gene_metadata,
            new_posterior_samples=new_posterior_samples
        )

    # --------------------------------------------------------------------------

    @abstractmethod
    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'BaseScribeResults':
        """Create a new instance with a subset of genes."""
        pass

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the model and guide functions for this model type."""
        pass

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        pass

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
        custom_args: Optional[Dict] = None,
    ) -> Dict:
        """Sample parameters from the variational posterior distribution."""
        # Get the guide function for this model type
        _, guide = self.get_model_and_guide()
        
        # Get the model arguments
        model_args = self.get_model_args()

        # Add custom arguments to model_args if provided
        if custom_args is not None:
            model_args.update(custom_args)
        
        # Sample from posterior
        posterior_samples = sample_variational_posterior(
            guide,
            self.params,
            model_args,
            rng_key=rng_key,
            n_samples=n_samples
        )
        
        # Store samples if requested
        if store_samples:
            if self.posterior_samples is not None:
                self.posterior_samples['parameter_samples'] = posterior_samples
            else:
                self.posterior_samples = {'parameter_samples': posterior_samples}
            
        return posterior_samples
    
    # --------------------------------------------------------------------------

    def get_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        custom_args: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Get the model function
        model, _ = self.get_model_and_guide()
        
        # Get the model arguments
        model_args = self.get_model_args()

        # Add custom arguments to model_args if provided
        if custom_args is not None:
            model_args.update(custom_args)
        
        # Generate samples
        return generate_predictive_samples(
            model,
            self.posterior_samples["parameter_samples"],
            model_args,
            rng_key=rng_key,
            batch_size=batch_size
        )

    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        resample_parameters: bool = False,
        custom_args: Optional[Dict] = None,
    ) -> Dict:
        """Generate posterior predictive check samples."""
        # Get the model and guide functions
        model, guide = self.get_model_and_guide()
        
        # Get the model arguments
        model_args = self.get_model_args()

        # Add custom arguments to model_args if provided
        if custom_args is not None:
            model_args.update(custom_args)
        
        # Check if we need to resample parameters
        need_params = (
            resample_parameters or 
            self.posterior_samples is None or 
            'parameter_samples' not in self.posterior_samples
        )
        
        if need_params:
            # Generate both parameter and predictive samples
            samples = generate_ppc_samples(
                model,
                guide,
                self.params,
                model_args,
                rng_key=rng_key,
                n_samples=n_samples,
                batch_size=batch_size
            )
        else:
            # Split RNG key for predictive sampling only
            _, key_pred = random.split(rng_key)
            # Use existing parameter samples and generate new predictive samples
            samples = {
                'parameter_samples': self.posterior_samples['parameter_samples'],
                'predictive_samples': generate_predictive_samples(
                    model,
                    self.posterior_samples['parameter_samples'],
                    model_args,
                    rng_key=key_pred,
                    batch_size=batch_size
                )
            }
        
        if store_samples:
            self.posterior_samples = samples
            
        return samples

# ------------------------------------------------------------------------------
# Base classes for standard models
# ------------------------------------------------------------------------------

@dataclass
class StandardResults(BaseScribeResults):
    """
    Base class for standard (non-mixture) models.

    This class extends BaseScribeResults to handle single-component models that
    don't use mixture distributions. It provides implementations for getting
    model arguments and creating subsets of the results.

    Methods
    -------
    get_model_args() -> Dict
        Returns a dictionary with the default model arguments (n_cells, n_genes)
    
    get_param_spec() -> Dict
        Abstract method that must be implemented by subclasses to specify
        parameter types and shapes for validation
    
    _create_subset(index, new_params, new_gene_metadata, new_posterior_samples)
    -> StandardResults
        Creates a new StandardResults instance containing only a subset of genes
    """
    def get_model_args(self) -> Dict:
        """Standard models just need cells and genes."""
        return {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
        }

    # --------------------------------------------------------------------------

    @abstractmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        pass
    
    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'StandardResults':
        """
        Create a new StandardResults instance containing only a subset of genes.

        Parameters
        ----------
        index : Union[np.ndarray, List[int]]
            Boolean mask or list of indices indicating which genes to keep
        new_params : Dict
            Dictionary of model parameters for the subset of genes
        new_gene_metadata : Optional[pd.DataFrame]
            Gene metadata for the subset of genes
        new_posterior_samples : Optional[Dict]
            Posterior samples for the subset of genes

        Returns
        -------
        StandardResults
            A new StandardResults instance containing only the specified subset
            of genes, with all cell-level information preserved
        """
        return type(self)(
            model_type=self.model_type,
            params=new_params,
            prior_params=self.prior_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=None,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples,
            param_spec=self.param_spec
        )

    # --------------------------------------------------------------------------

    def compute_log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        return_by: str = 'cell',
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32
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
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
            
        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on return_by:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
                
        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Check if posterior samples exist
        if (self.posterior_samples is None or 
            'parameter_samples' not in self.posterior_samples):
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )
        
        # Get parameter samples
        parameter_samples = self.posterior_samples['parameter_samples']
        
        # Get number of samples from first parameter
        n_samples = next(iter(parameter_samples.values())).shape[0]
        
        # Get likelihood function
        likelihood_fn = self.get_log_likelihood_fn()
        
        # Define function to compute likelihood for a single sample
        @jit
        def compute_sample_lik(i):
            # Extract parameters for this sample
            params_i = {k: v[i] for k, v in parameter_samples.items()}
            # Return likelihood
            return likelihood_fn(
                counts, 
                params_i, 
                batch_size=batch_size,
                cells_axis=cells_axis,
                return_by=return_by,
                dtype=dtype
            )
        
        # Compute log likelihoods for all samples using vmap
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
        
        # Handle NaNs if requested
        if ignore_nans:
            valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=1)
            print(f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}")
            return log_liks[valid_samples]
        
        return log_liks

# ------------------------------------------------------------------------------
# Base classes for mixture models
# ------------------------------------------------------------------------------

@dataclass
class MixtureResults(BaseScribeResults):
    """
    Base class for mixture model results.
    
    This class extends BaseScribeResults to handle mixture models which have
    multiple components. It provides functionality for managing
    component-specific parameters and validation.

    The class enforces validation of component-specific parameters through the
    param_spec attribute, ensuring parameters have the correct shape with
    respect to the number of components.

    Methods
    -------
    get_model_args() -> Dict
        Returns a dictionary of model arguments including n_components
    
    get_param_spec() -> Dict
        Abstract method that must be implemented by subclasses to specify
        parameter types and shapes for validation
    
    _create_subset(index, new_params, new_gene_metadata, new_posterior_samples)
    -> MixtureResults
        Creates a new MixtureResults instance containing only a subset of genes
    """
    def get_model_args(self) -> Dict:
        """
        Get arguments needed for mixture model initialization.

        Returns
        -------
        Dict
            Dictionary containing n_cells, n_genes, and n_components
        """
        return {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'n_components': self.n_components,
        }
    
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        pass

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'MixtureResults':
        """
        Create a new instance with a subset of genes.

        Parameters
        ----------
        index : array-like
            Boolean mask or integer indices for selecting genes
        new_params : Dict
            Parameter dictionary for the subset
        new_gene_metadata : Optional[pd.DataFrame]
            Gene metadata for the subset
        new_posterior_samples : Optional[Dict]
            Posterior samples for the subset

        Returns
        -------
        MixtureResults
            New instance containing only the selected genes
        """
        return type(self)(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples,
            param_spec=self.param_spec,
            prior_params=self.prior_params
        )
    
    # --------------------------------------------------------------------------

    def compute_log_likelihood(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        return_by: str = 'cell',
        cells_axis: int = 0,
        ignore_nans: bool = False,
        split_components: bool = False,
        weights: Optional[jnp.ndarray] = None,
        dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """
        Compute log likelihood of data under posterior samples for mixture
        model.
        
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
            separately. If False, returns marginalized log likelihoods.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods either by cell (when
            return_by='gene') or by gene (when return_by='cell').
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
            
        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on return_by and
            split_components: If split_components=False:
                - 'cell': shape (n_samples, n_cells)
                - 'gene': shape (n_samples, n_genes)
            If split_components=True:
                - 'cell': shape (n_samples, n_components, n_cells)
                - 'gene': shape (n_samples, n_components, n_genes)
                
        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Check if posterior samples exist
        if (self.posterior_samples is None or 
            'parameter_samples' not in self.posterior_samples):
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )
        
        # Get parameter samples
        parameter_samples = self.posterior_samples['parameter_samples']
        
        # Get number of samples from first parameter
        n_samples = parameter_samples[next(iter(parameter_samples))].shape[0]
        
        # Get likelihood function
        likelihood_fn = self.get_log_likelihood_fn()
        
        # Define function to compute likelihood for a single sample
        @jit
        def compute_sample_lik(i):
            # Extract parameters for this sample
            params_i = {k: v[i] for k, v in parameter_samples.items()}
            # Return likelihood
            return likelihood_fn(
                counts, 
                params_i, 
                batch_size=batch_size,
                cells_axis=cells_axis,
                return_by=return_by,
                split_components=split_components,
                weights=weights,
                dtype=dtype
            )
        
        # Compute log likelihoods for all samples using vmap
        log_liks = vmap(compute_sample_lik)(jnp.arange(n_samples))
        
        # Handle NaNs if requested
        if ignore_nans:
            # Adjust nan checking for component dimension if present
            if split_components:
                valid_samples = ~jnp.any(
                    jnp.any(jnp.isnan(log_liks), axis=2), 
                    axis=1
                )
            else:
                valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=1)
            print(f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}")
            return log_liks[valid_samples]
        
        return log_liks

    # --------------------------------------------------------------------------

    def compute_cell_type_assignments(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        ignore_nans: bool = False,
        dtype: jnp.dtype = jnp.float32,
        fit_distribution: bool = True,
        weights: Optional[jnp.ndarray] = None,
        verbose: bool = True
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
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods either by cell (when
            return_by='gene') or by gene (when return_by='cell').
        verbose : bool, default=True
            If True, prints progress

        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'concentration': Dirichlet concentration parameters for each
                  cell Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'mean_probabilities': Mean assignment probabilities for each
                  cell Shape: (n_cells, n_components). Only returned if
                  fit_distribution is True.
                - 'sample_probabilities': Assignment probabilities for each
                  posterior sample. Shape: (n_samples, n_cells, n_components)

        Raises
        ------
        ValueError
            If posterior samples have not been generated yet
        """
        # Check if posterior samples exist
        if (self.posterior_samples is None or 
            'parameter_samples' not in self.posterior_samples):
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )

        if verbose:
            print("- Computing component-specific log-likelihoods...")

        # Compute component-specific log-likelihoods
        # Shape: (n_samples, n_cells, n_components)
        log_liks = self.compute_log_likelihood(
            counts,
            batch_size=batch_size,
            return_by='cell',
            cells_axis=cells_axis,
            ignore_nans=ignore_nans,
            split_components=True,
            weights=weights,
            dtype=dtype
        )

        if verbose:
            print("- Converting log-likelihoods to probabilities...")

        # Convert log-likelihoods to probabilities using log-sum-exp for
        # stability First compute log(sum(exp(x))) along component axis
        log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)
        # Then subtract and exponentiate to get probabilities
        probabilities = jnp.exp(log_liks - log_sum_exp)

        # Get shapes
        n_samples, n_cells, n_components = probabilities.shape

        if fit_distribution:
            if verbose:
                print("- Fitting Dirichlet distribution...")

            # Initialize array for Dirichlet concentration parameters
            concentrations = jnp.zeros((n_cells, n_components), dtype=dtype)

            # Fit Dirichlet distribution for each cell
            for cell in range(n_cells):
                if cell % 1_000 == 0:
                    print(f"    - Fitting Dirichlet distribution for cell {cell} of {n_cells}")
                # Get probability vectors for this cell across all samples
                cell_probs = probabilities[:, cell, :]
                # Fit Dirichlet using Minka's fixed-point method
                concentrations = concentrations.at[cell].set(
                    fit_dirichlet_minka(cell_probs)
                )

            # Compute mean probabilities (Dirichlet mean)
            concentration_sums = jnp.sum(concentrations, axis=1, keepdims=True)
            mean_probabilities = concentrations / concentration_sums

            return {
                'concentration': concentrations,
                'mean_probabilities': mean_probabilities,
                'sample_probabilities': probabilities
            }
        else:
            return {
                'sample_probabilities': probabilities
            }

    # --------------------------------------------------------------------------

    def compute_cell_type_assignments_map(
        self,
        counts: jnp.ndarray,
        batch_size: Optional[int] = None,
        cells_axis: int = 0,
        dtype: jnp.dtype = jnp.float32,
        weights: Optional[jnp.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """
        Compute probabilistic cell type assignments using MAP estimates of parameters.
        
        For each cell, this method:
            1. Computes component-specific log-likelihoods using MAP parameter estimates
            2. Converts these to probability distributions over cell types
        
        Parameters
        ----------
        counts : jnp.ndarray
            Count data to evaluate assignments for
        batch_size : Optional[int], default=None
            Size of mini-batches for likelihood computation
        cells_axis : int, default=0
            Axis along which cells are arranged. 0 means cells are rows.
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods by gene
        verbose : bool, default=True
            If True, prints progress
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'probabilities': Assignment probabilities for each cell
                Shape: (n_cells, n_components)
        """
        if verbose:
            print("- Computing component-specific log-likelihoods...")

        # Get the log likelihood function
        likelihood_fn = self.get_log_likelihood_fn()

        # Get the MAP estimates
        map_estimates = self.get_map()
        
        # Compute component-specific log-likelihoods using MAP estimates
        # Shape: (n_cells, n_components)
        log_liks = likelihood_fn(
            counts,
            map_estimates,
            batch_size=batch_size,
            cells_axis=cells_axis,
            return_by='cell',
            split_components=True,
            weights=weights,
            dtype=dtype
        )

        if verbose:
            print("- Converting log-likelihoods to probabilities...")

        # Convert log-likelihoods to probabilities using log-sum-exp for stability
        # First compute log(sum(exp(x))) along component axis
        log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)
        # Then subtract and exponentiate to get probabilities
        probabilities = jnp.exp(log_liks - log_sum_exp)

        return {
            'probabilities': probabilities
        }

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Standard Models
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial model
# ------------------------------------------------------------------------------

@dataclass
class NBDMResults(StandardResults):
    """
    Results for Negative Binomial-Dirichlet Multinomial model.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific'},
            'beta_r': {'type': 'gene-specific'}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "nbdm", f"Invalid model type: {self.model_type}"
        # Set n_components to None since this is not a mixture model
        self.n_components = None
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for NBDM parameters."""
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'], 
                    loc=0, 
                    scale=1/self.params['beta_r']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r'])
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        For Beta(α,β) distribution, the mode is:
            (α-1)/(α+β-2) when α,β > 1
            undefined when α,β ≤ 1
            
        For Gamma(α,β) distribution, the mode is:
            (α-1)/β when α > 1
            0 when α ≤ 1
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        
        Notes
        -----
        Returns NaN for cases where mode is undefined (e.g., Beta with α,β ≤ 1)
        """
        map_dict = {}
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBDM model and guide functions."""
        from .models import nbdm_model, nbdm_guide
        return nbdm_model, nbdm_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBDM model."""
        from .models import nbdm_log_likelihood
        return nbdm_log_likelihood

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial model
# ------------------------------------------------------------------------------

@dataclass
class ZINBResults(StandardResults):
    """
    Results for Zero-Inflated Negative Binomial model.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific'},
            'beta_r': {'type': 'gene-specific'},
            'alpha_gate': {'type': 'gene-specific'},
            'beta_gate': {'type': 'gene-specific'}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "zinb", f"Invalid model type: {self.model_type}"
        # Set n_components to None since this is not a mixture model
        self.n_components = None
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for ZINB parameters."""
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'], 
                    loc=0, 
                    scale=1/self.params['beta_r']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'gate': dist.Beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        For Beta(α,β) distribution, the mode is:
            (α-1)/(α+β-2) when α,β > 1
            undefined when α,β ≤ 1
            
        For Gamma(α,β) distribution, the mode is:
            (α-1)/β when α > 1
            0 when α ≤ 1
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        
        Notes
        -----
        Returns NaN for cases where mode is undefined (e.g., Beta with α,β ≤ 1)
        """
        map_dict = {}
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Hangle Beta distribution for gate parameter
        map_dict['gate'] = beta_mode(
            self.params['alpha_gate'],
            self.params['beta_gate']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINB model and guide functions."""
        from .models import zinb_model, zinb_guide
        return zinb_model, zinb_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINB model."""
        from .models import zinb_log_likelihood
        return zinb_log_likelihood

# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability model
# ------------------------------------------------------------------------------

@dataclass
class NBVCPResults(StandardResults):
    """
    Results for Negative Binomial model with variable capture probability.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific'},
            'beta_r': {'type': 'gene-specific'},
            'alpha_p_capture': {'type': 'cell-specific'},
            'beta_p_capture': {'type': 'cell-specific'}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "nbvcp", f"Invalid model type: {self.model_type}"
        # Set n_components to None since this is not a mixture model
        self.n_components = None
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for NBVCP parameters."""
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'], 
                    loc=0, 
                    scale=1/self.params['beta_r']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        For Beta(α,β) distribution, the mode is:
            (α-1)/(α+β-2) when α,β > 1
            undefined when α,β ≤ 1
            
        For Gamma(α,β) distribution, the mode is:
            (α-1)/β when α > 1
            0 when α ≤ 1
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        
        Notes
        -----
        Returns NaN for cases where mode is undefined (e.g., Beta with α,β ≤ 1)
        """
        map_dict = {}
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Handle Beta distribution for p_capture parameter
        map_dict['p_capture'] = beta_mode(
            self.params['alpha_p_capture'],
            self.params['beta_p_capture']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBVCP model and guide functions."""
        from .models import nbvcp_model, nbvcp_guide
        return nbvcp_model, nbvcp_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBVCP model."""
        from .models import nbvcp_log_likelihood
        return nbvcp_log_likelihood

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

@dataclass
class ZINBVCPResults(StandardResults):
    """
    Results for Zero-Inflated Negative Binomial model with variable capture
    probability.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific'},
            'beta_r': {'type': 'gene-specific'},
            'alpha_gate': {'type': 'gene-specific'},
            'beta_gate': {'type': 'gene-specific'},
            'alpha_p_capture': {'type': 'cell-specific'},
            'beta_p_capture': {'type': 'cell-specific'}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "zinbvcp", f"Invalid model type: {self.model_type}"
        # Set n_components to 1 since this is not a mixture model
        self.n_components = 1
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for ZINBVCP parameters."""
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'], 
                    loc=0, 
                    scale=1/self.params['beta_r']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'gate': dist.Beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                ),
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        For Beta(α,β) distribution, the mode is:
            (α-1)/(α+β-2) when α,β > 1
            undefined when α,β ≤ 1
            
        For Gamma(α,β) distribution, the mode is:
            (α-1)/β when α > 1
            0 when α ≤ 1
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        
        Notes
        -----
        Returns NaN for cases where mode is undefined (e.g., Beta with α,β ≤ 1)
        """
        map_dict = {}
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Handle Beta distribution for gate parameter
        map_dict['gate'] = beta_mode(
            self.params['alpha_gate'],
            self.params['beta_gate']
        )
        
        # Handle Beta distribution for p_capture parameter
        map_dict['p_capture'] = beta_mode(
            self.params['alpha_p_capture'],
            self.params['beta_p_capture']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINBVCP model and guide functions."""
        from .models import zinbvcp_model, zinbvcp_guide
        return zinbvcp_model, zinbvcp_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINBVCP model."""
        from .models import zinbvcp_log_likelihood
        return zinbvcp_log_likelihood

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Mixture Models
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class NBDMMixtureResults(MixtureResults):
    """
    Results for Negative Binomial-Dirichlet Multinomial mixture model.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_mixing': {'type': 'global', 'component_specific': True},
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific', 'component_specific': True},
            'beta_r': {'type': 'gene-specific', 'component_specific': True}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "nbdm_mix", f"Invalid model type: {self.model_type}"
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for mixture model parameters."""
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1/self.params['beta_r']
                )
            }
        elif backend == "numpyro":
            return {
                'mixing_weights': dist.Dirichlet(self.params['alpha_mixing']),
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r'])
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        """
        map_dict = {}
        
        # Handle Dirichlet distribution for mixing weights
        map_dict['mixing_weights'] = dirichlet_mode(
            self.params['alpha_mixing']
        )
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBDMMixture model and guide functions."""
        from .models_mix import nbdm_mixture_model, nbdm_mixture_guide
        return nbdm_mixture_model, nbdm_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBDMMixture model."""
        from .models_mix import nbdm_mixture_log_likelihood
        return nbdm_mixture_log_likelihood

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class ZINBMixtureResults(MixtureResults):
    """
    Results for Zero-Inflated Negative Binomial mixture model.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_mixing': {'type': 'global', 'component_specific': True},
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific', 'component_specific': True},
            'beta_r': {'type': 'gene-specific', 'component_specific': True},
            'alpha_gate': {'type': 'gene-specific', 'component_specific': True},
            'beta_gate': {'type': 'gene-specific', 'component_specific': True}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "zinb_mix", f"Invalid model type: {self.model_type}"
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for mixture model parameters."""
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1/self.params['beta_r']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                )
            }
        elif backend == "numpyro":
            return {
                'mixing_weights': dist.Dirichlet(self.params['alpha_mixing']),
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'gate': dist.Beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        """
        map_dict = {}
        
        # Handle Dirichlet distribution for mixing weights
        map_dict['mixing_weights'] = dirichlet_mode(
            self.params['alpha_mixing']
        )
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Handle Beta distribution for gate parameter
        map_dict['gate'] = beta_mode(
            self.params['alpha_gate'],
            self.params['beta_gate']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINB mixture model and guide functions."""
        from .models_mix import zinb_mixture_model, zinb_mixture_guide
        return zinb_mixture_model, zinb_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINB mixture model."""
        from .models_mix import zinb_mixture_log_likelihood
        return zinb_mixture_log_likelihood

# ------------------------------------------------------------------------------
# Negative Binomial-Variable Capture Probability Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class NBVCPMixtureResults(MixtureResults):
    """
    Results for Negative Binomial mixture model with variable capture
    probability.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_mixing': {'type': 'global', 'component_specific': True},
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific', 'component_specific': True},
            'beta_r': {'type': 'gene-specific', 'component_specific': True},
            'alpha_p_capture': {'type': 'cell-specific'},
            'beta_p_capture': {'type': 'cell-specific'}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "nbvcp_mix", f"Invalid model type: {self.model_type}"
        # Call superclass post-init
        super().__post_init__()

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for mixture model parameters."""
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1/self.params['beta_r']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        elif backend == "numpyro":
            return {
                'mixing_weights': dist.Dirichlet(self.params['alpha_mixing']),
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        """
        map_dict = {}
        
        # Handle Dirichlet distribution for mixing weights
        map_dict['mixing_weights'] = dirichlet_mode(
            self.params['alpha_mixing']
        )
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Handle Beta distribution for p_capture parameter
        map_dict['p_capture'] = beta_mode(
            self.params['alpha_p_capture'],
            self.params['beta_p_capture']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBVCP mixture model and guide functions."""
        from .models_mix import nbvcp_mixture_model, nbvcp_mixture_guide
        return nbvcp_mixture_model, nbvcp_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBVCP mixture model."""
        from .models_mix import nbvcp_mixture_log_likelihood
        return nbvcp_mixture_log_likelihood

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial-Variable Capture Probability Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class ZINBVCPMixtureResults(MixtureResults):
    """
    Results for Zero-Inflated Negative Binomial mixture model with variable
    capture probability.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_mixing': {'type': 'global', 'component_specific': True},
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'alpha_r': {'type': 'gene-specific', 'component_specific': True},
            'beta_r': {'type': 'gene-specific', 'component_specific': True},
            'alpha_p_capture': {'type': 'cell-specific'},
            'beta_p_capture': {'type': 'cell-specific'},
            'alpha_gate': {'type': 'gene-specific', 'component_specific': True},
            'beta_gate': {'type': 'gene-specific', 'component_specific': True}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "zinbvcp_mix", f"Invalid model type: {self.model_type}"
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for mixture model parameters."""
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1/self.params['beta_r']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        elif backend == "numpyro":
            return {
                'mixing_weights': dist.Dirichlet(self.params['alpha_mixing']),
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'gate': dist.Beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                ),
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'],
                    self.params['beta_p_capture']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        """
        map_dict = {}
        
        # Handle Dirichlet distribution for mixing weights
        map_dict['mixing_weights'] = dirichlet_mode(
            self.params['alpha_mixing']
        )
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = gamma_mode(
            self.params['alpha_r'],
            self.params['beta_r']
        )
        
        # Handle Beta distribution for gate parameter
        map_dict['gate'] = beta_mode(
            self.params['alpha_gate'],
            self.params['beta_gate']
        )
        
        # Handle Beta distribution for p_capture parameter
        map_dict['p_capture'] = beta_mode(
            self.params['alpha_p_capture'],
            self.params['beta_p_capture']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINBVCP mixture model and guide functions."""
        from .models_mix import zinbvcp_mixture_model, zinbvcp_mixture_guide
        return zinbvcp_mixture_model, zinbvcp_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINBVCP mixture model."""
        from .models_mix import zinbvcp_mixture_log_likelihood
        return zinbvcp_mixture_log_likelihood

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Log-Normal Mixture Models
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class NBDMMixtureLogResults(MixtureResults):
    """
    Results for Negative Binomial-Dirichlet Multinomial mixture model with
    log-normal prior on r parameters.
    """
    @classmethod
    def get_param_spec(self) -> Dict:
        """Get the parameter specification for this model type."""
        return {
            'alpha_mixing': {'type': 'global', 'component_specific': True},
            'alpha_p': {'type': 'global'},
            'beta_p': {'type': 'global'},
            'mu_r': {'type': 'gene-specific', 'component_specific': True},
            'sigma_r': {'type': 'gene-specific', 'component_specific': True}
        }
    
    # --------------------------------------------------------------------------

    def __post_init__(self):
        # Set parameter spec
        self.param_spec = self.get_param_spec()
        # Verify model type and validate parameters
        assert self.model_type == "nbdm_log_mix", f"Invalid model type: {self.model_type}"
        # Call superclass post-init
        super().__post_init__()

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get the variational distributions for mixture model parameters."""
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.lognorm(
                    self.params['sigma_r'],
                    loc=0,
                    scale=jnp.exp(self.params['mu_r'])
                )
            }
        elif backend == "numpyro":
            return {
                'mixing_weights': dist.Dirichlet(self.params['alpha_mixing']),
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.LogNormal(self.params['mu_r'], self.params['sigma_r'])
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary mapping parameter names to their MAP estimates
        """
        map_dict = {}
        
        # Handle Dirichlet distribution for mixing weights
        map_dict['mixing_weights'] = dirichlet_mode(
            self.params['alpha_mixing']
        )
        
        # Handle Beta distribution for p parameter
        map_dict['p'] = beta_mode(
            self.params['alpha_p'],
            self.params['beta_p']
        )
        
        # Handle Gamma distribution for r parameter
        map_dict['r'] = lognorm_mode(
            self.params['mu_r'],
            self.params['sigma_r']
        )
        
        return map_dict

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBDMMixture model and guide functions."""
        from .models_mix_log import nbdm_log_mixture_model, nbdm_log_mixture_guide
        return nbdm_log_mixture_model, nbdm_log_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBDMMixture model."""
        from .models_mix import nbdm_mixture_log_likelihood
        return nbdm_mixture_log_likelihood


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Custom Model Results
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@dataclass
class CustomResults(BaseScribeResults):
    """
    Results class for custom user-defined models.
    
    This class provides a flexible framework for handling custom models while
    maintaining compatibility with SCRIBE's parameter handling and results
    infrastructure.

    Parameters
    ----------
    params : Dict
        Dictionary of inferred model parameters
    loss_history : jnp.ndarray
        Array containing the ELBO loss values during training
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Identifier for the custom model
    param_spec : Dict[str, Dict]
        Specification of parameter types. Each parameter name should match
        exactly as it appears in params, with entries specifying:
            - 'type': one of ['global', 'gene-specific', 'cell-specific']
            - 'component_specific': True/False (for mixture models)
    custom_model : Callable
        The user's custom model function
    custom_guide : Callable
        The user's custom guide function
    custom_log_likelihood_fn : Optional[Callable]
        Optional function to compute log likelihood for the custom model
    get_distributions_fn : Optional[Callable]
        Optional function to implement get_distributions behavior
    get_map_fn : Optional[Callable]
        Optional function to implement get_map behavior
    get_model_args_fn : Optional[Callable]
        Optional function to customize model arguments
    n_components : Optional[int]
        Number of mixture components (if mixture model)
    cell_metadata : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs
    gene_metadata : Optional[pd.DataFrame]
        Gene-level metadata from adata.var
    uns : Optional[Dict]
        Unstructured metadata from adata.uns
    posterior_samples : Optional[Dict]
        Samples from the posterior distribution
    """
    # Additional attributes for custom models
    custom_model: Optional[Callable] = None
    custom_guide: Optional[Callable] = None
    custom_log_likelihood_fn: Optional[Callable] = None
    get_distributions_fn: Optional[Callable] = None
    get_map_fn: Optional[Callable] = None
    get_model_args_fn: Optional[Callable] = None

    
    def __post_init__(self):
        """Set model type to 'custom' and validate parameters."""
        # Set model type to 'custom' if not already set
        if not hasattr(self, 'model_type') or self.model_type is None:
            self.model_type = "custom"
        # Call superclass post-init for parameter validation
        super().__post_init__()
    
    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """Get distributions using provided get_distributions_fn if available."""
        if self.get_distributions_fn is not None:
            return self.get_distributions_fn(self.params, backend)
        return {}

    # --------------------------------------------------------------------------
    
    def get_map(self) -> Dict[str, jnp.ndarray]:
        """Get the maximum a posteriori (MAP) estimates from the variational posterior."""
        if self.get_map_fn is not None:
            return self.get_map_fn(self.params)
        return {}

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the custom model and guide functions."""
        if self.custom_model is None or self.custom_guide is None:
            raise ValueError("Custom model and guide functions not provided")
        return self.custom_model, self.custom_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function if provided."""
        if self.custom_log_likelihood_fn is not None:
            return self.custom_log_likelihood_fn
        raise ValueError("No custom log likelihood function provided")

    # --------------------------------------------------------------------------

    def get_model_args(self) -> Dict:
        """
        Get model arguments using provided get_model_args_fn if available,
        otherwise return standard arguments based on presence of n_components.
        """
        if self.get_model_args_fn is not None:
            return self.get_model_args_fn(self)
        
        # Default implementation based on whether it's a mixture model
        args = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes
        }
        if self.n_components is not None:
            args['n_components'] = self.n_components
        return args

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'CustomResults':
        """Create a new CustomResults instance with a subset of genes."""
        return CustomResults(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            param_spec=self.param_spec,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples,
            custom_model=self.custom_model,
            custom_guide=self.custom_guide,
            custom_log_likelihood_fn=self.custom_log_likelihood_fn,
            get_distributions_fn=self.get_distributions_fn,
            get_map_fn=self.get_map_fn,
            get_model_args_fn=self.get_model_args_fn,
        )