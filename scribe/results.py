"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, replace

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
    get_distribution_mode
)
from .model_config import ModelConfig
from .utils import numpyro_to_scipy

# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------

@dataclass
class ScribeResults:
    """
    Base class for SCRIBE inference results.
    
    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, loss history, dataset dimensions, and model
    configuration. It can optionally store metadata from an AnnData object and
    posterior/predictive samples.

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
    posterior_samples : Optional[Dict]
        Samples of parameters from the posterior distribution, if generated
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
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
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData object
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    
    # Optional results
    posterior_samples: Optional[Dict] = None
    predictive_samples: Optional[Dict] = None
    n_components: Optional[int] = None

    # --------------------------------------------------------------------------

    def __post_init__(self):
        """Validate model configuration and parameters."""
        # Set n_components from model_config if not explicitly provided
        if self.n_components is None and self.model_config.n_components is not None:
            self.n_components = self.model_config.n_components
            
        self._validate_model_config()

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
                
        # Validate required distributions based on model type
        if "zinb" in self.model_type:
            if (self.model_config.gate_distribution_model is None or 
                self.model_config.gate_distribution_guide is None):
                raise ValueError("ZINB models require gate distributions")
        else:
            if (self.model_config.gate_distribution_model is not None or 
                self.model_config.gate_distribution_guide is not None):
                raise ValueError("Non-ZINB models should not have gate distributions")
                
        if "vcp" in self.model_type:
            if (self.model_config.p_capture_distribution_model is None or
                self.model_config.p_capture_distribution_guide is None):
                raise ValueError("VCP models require capture probability distributions")
        else:
            if (self.model_config.p_capture_distribution_model is not None or
                self.model_config.p_capture_distribution_guide is not None):
                raise ValueError(
                    "Non-VCP models should not have capture probability distributions"
                )

    # --------------------------------------------------------------------------
    # Create ScribeResults from AnnData object
    # --------------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        adata: "AnnData",
        params: Dict,
        loss_history: jnp.ndarray,
        model_config: ModelConfig,
        **kwargs
    ):
        """Create ScribeResults from AnnData object."""
        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_config=model_config,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs
        )
    
    # --------------------------------------------------------------------------
    # Get distributions using configs
    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters using model config.
        
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
        if backend not in ["scipy", "numpyro"]:
            raise ValueError(f"Invalid backend: {backend}")
            
        distributions = {}
        
        # Handle r distribution
        r_params = {}
        for param_name in self.model_config.r_distribution_guide.arg_constraints:
            r_params[param_name] = self.params[f"r_{param_name}"]
        
        if backend == "scipy":
            distributions['r'] = numpyro_to_scipy(
                self.model_config.r_distribution_guide.__class__(**r_params)
            )
        else:  # numpyro
            distributions['r'] = self.model_config.r_distribution_guide.__class__(**r_params)
        
        # Handle p distribution
        p_params = {}
        for param_name in self.model_config.p_distribution_guide.arg_constraints:
            p_params[param_name] = self.params[f"p_{param_name}"]
        
        if backend == "scipy":
            distributions['p'] = numpyro_to_scipy(
                self.model_config.p_distribution_guide.__class__(**p_params)
            )
        else:  # numpyro
            distributions['p'] = self.model_config.p_distribution_guide.__class__(**p_params)
        
        # Add gate distribution if present
        if self.model_config.gate_distribution_guide is not None:
            gate_params = {}
            for param_name in self.model_config.gate_distribution_guide.arg_constraints:
                gate_params[param_name] = self.params[f"gate_{param_name}"]
                
            if backend == "scipy":
                distributions['gate'] = numpyro_to_scipy(
                    self.model_config.gate_distribution_guide.__class__(**gate_params)
                )
            else:  # numpyro
                distributions['gate'] = self.model_config.gate_distribution_guide.__class__(**gate_params)
            
        # Add p_capture distribution if present
        if self.model_config.p_capture_distribution_guide is not None:
            p_capture_params = {}
            for param_name in self.model_config.p_capture_distribution_guide.arg_constraints:
                p_capture_params[param_name] = self.params[f"p_capture_{param_name}"]
                
            if backend == "scipy":
                distributions['p_capture'] = numpyro_to_scipy(
                    self.model_config.p_capture_distribution_guide.__class__(**p_capture_params)
                )
            else:  # numpyro
                distributions['p_capture'] = self.model_config.p_capture_distribution_guide.__class__(**p_capture_params)
            
        # Add mixing weights if mixture model
        if self.model_config.n_components is not None:
            # Extract mixing weights
            mixing_params = self.params[f"mixing_concentration"]
            
            if backend == "scipy":
                distributions['mixing_weights'] = numpyro_to_scipy(
                    self.model_config.mixing_distribution_guide.__class__(
                        concentration=mixing_params
                    )
                )
            else:
                distributions['mixing_weights'] = self.model_config.mixing_distribution_guide.__class__(
                    concentration=mixing_params
                )
        
        return distributions

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        """
        distributions = self.get_distributions(backend="numpyro")
        return {
            param: get_distribution_mode(dist) 
            for param, dist in distributions.items()
        }

    # --------------------------------------------------------------------------
    # Indexing by genes
    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        """
        new_params = dict(params)
        
        # Handle r parameters (always gene-specific)
        r_param_names = list(self.model_config.r_distribution_guide.arg_constraints.keys())
        for param_name in r_param_names:
            param_key = f"r_{param_name}"
            if param_key in params:
                if self.n_components is not None:
                    # Keep component dimension but subset gene dimension
                    new_params[param_key] = params[param_key][..., index]
                else:
                    # Just subset gene dimension
                    new_params[param_key] = params[param_key][index]
        
        # Handle gate parameters if present (gene-specific)
        if self.model_config.gate_distribution_guide is not None:
            gate_param_names = list(self.model_config.gate_distribution_guide.arg_constraints.keys())
            for param_name in gate_param_names:
                param_key = f"gate_{param_name}"
                if param_key in params:
                    if self.n_components is not None:
                        # Keep component dimension but subset gene dimension
                        new_params[param_key] = params[param_key][..., index]
                    else:
                        # Just subset gene dimension
                        new_params[param_key] = params[param_key][index]
        
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        
        Parameters
        ----------
        samples : Dict
            Dictionary of samples from the posterior distribution, where each
            key represents a parameter type ('r', 'p', 'gate', etc.) and values
            are arrays of samples
        index : array-like
            Boolean or integer index specifying which genes to keep
            
        Returns
        -------
        Dict
            New dictionary containing posterior samples for the selected genes
        """
        if samples is None:
            return None
            
        new_posterior_samples = {}
        
        # Handle gene-specific parameters
        
        # r samples (always gene-specific)
        if 'r' in samples:
            if self.n_components is not None:
                # Shape: (n_samples, n_components, n_genes)
                new_posterior_samples['r'] = samples['r'][..., index]
            else:
                # Shape: (n_samples, n_genes)
                new_posterior_samples['r'] = samples['r'][..., index]
        
        # gate samples (gene-specific if present)
        if 'gate' in samples:
            if self.n_components is not None:
                # Shape: (n_samples, n_components, n_genes)
                new_posterior_samples['gate'] = samples['gate'][..., index]
            else:
                # Shape: (n_samples, n_genes)
                new_posterior_samples['gate'] = samples['gate'][..., index]
        
        # Copy non-gene-specific parameters as is
        
        # p samples (global)
        if 'p' in samples:
            # Shape: (n_samples,) or (n_samples, n_components)
            new_posterior_samples['p'] = samples['p']
        
        # p_capture samples (cell-specific)
        if 'p_capture' in samples:
            # Shape: (n_samples, n_cells)
            new_posterior_samples['p_capture'] = samples['p_capture']
        
        # mixing weights if present
        if 'mixing_weights' in samples:
            # Shape: (n_samples, n_components)
            new_posterior_samples['mixing_weights'] = samples['mixing_weights']

        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _subset_predictive_samples(self, samples: jnp.ndarray, index) -> jnp.ndarray:
        """Create a new predictive samples array for the given index."""
        if samples is None:
            return None
            
        # For predictive samples, subset the gene dimension (last dimension)
        return samples[..., index]

    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of ScribeResults object.
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
        new_var = self.var.iloc[index] if self.var is not None else None

        # Create new posterior samples if available
        new_posterior_samples = self._subset_posterior_samples(
            self.posterior_samples, index
        ) if self.posterior_samples is not None else None
            
        # Create new predictive samples if available
        new_predictive_samples = self._subset_predictive_samples(
            self.predictive_samples, index
        ) if self.predictive_samples is not None else None
            
        # Create new instance with subset data
        return self._create_subset(
            index=index,
            new_params=new_params,
            new_var=new_var,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples
        )

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_var: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray]
    ) -> 'ScribeResults':
        """Create a new instance with a subset of genes."""
        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            model_type=self.model_type,
            model_config=self.model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=self.n_components
        )
    
    # --------------------------------------------------------------------------
    # Indexing by component
    # --------------------------------------------------------------------------

    def get_component(self, component_index):
        """
        Create a view of the results selecting a specific mixture component.
        
        This method returns a new ScribeResults object that contains parameter
        values for the specified component, allowing for further gene-based
        indexing. Only applicable to mixture models.
        
        Parameters
        ----------
        component_index : int
            Index of the component to select
        
        Returns
        -------
        ScribeResults
            A new ScribeResults object with parameters for the selected component
            
        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Component view only applies to mixture models with multiple components"
            )
            
        # Check if component_index is valid
        if component_index < 0 or component_index >= self.n_components:
            raise ValueError(
                f"Component index {component_index} out of range [0, {self.n_components-1}]"
            )
        
        # Create new params dict with component subset
        new_params = dict(self.params)
        
        # Handle r parameters (always gene-specific)
        r_param_names = list(
            self.model_config.r_distribution_guide.arg_constraints.keys()
        )
        # Loop through r parameters
        for param_name in r_param_names:
            # Create parameter key
            param_key = f"r_{param_name}"
            # Check if parameter is present
            if param_key in self.params:
                # Select component dimension
                new_params[param_key] = self.params[param_key][component_index]
        
        # Handle gate parameters if present (gene-specific)
        if self.model_config.gate_distribution_guide is not None:
            # Get gate parameter names
            gate_param_names = list(
                self.model_config.gate_distribution_guide.arg_constraints.keys()
            )
            # Loop through gate parameters
            for param_name in gate_param_names:
                # Create parameter key
                param_key = f"gate_{param_name}"
                # Check if parameter is present
                if param_key in self.params:
                    # Select component dimension
                    new_params[param_key] = self.params[param_key][component_index]
        
        # Create new posterior samples if available
        new_posterior_samples = None
        if self.posterior_samples is not None:
            new_posterior_samples = self._subset_posterior_samples_component(
                self.posterior_samples, component_index
            )
        
        # Create new predictive samples if available - this is more complex
        # as we would need to condition on the component
        new_predictive_samples = None
        
        # Create new instance with component subset
        return self._create_component_subset(
            component_index=component_index,
            new_params=new_params,
            new_posterior_samples=new_posterior_samples,
            new_predictive_samples=new_predictive_samples
        )

    # --------------------------------------------------------------------------

    def _subset_posterior_samples_component(self, samples: Dict, component_index) -> Dict:
        """
        Create a new posterior samples dictionary for the given component index.
        """
        if samples is None:
            return None
            
        new_posterior_samples = {}
        
        # Check if we have parameter_samples sub-dictionary
        if 'parameter_samples' in samples:
            param_samples = samples['parameter_samples']
            new_param_samples = {}
            
            # Handle r parameters (component and gene-specific)
            if 'r' in param_samples:
                # Shape is typically (n_samples, n_components, n_genes)
                # Select component dimension to get (n_samples, n_genes)
                new_param_samples['r'] = param_samples['r'][:, component_index]
            
            # Handle gate parameters if present (component and gene-specific)
            if 'gate' in param_samples:
                # Shape is typically (n_samples, n_components, n_genes)
                # Select component dimension to get (n_samples, n_genes)
                new_param_samples['gate'] = param_samples['gate'][:, component_index]
            
            # Copy global parameters as is (p, mixing_weights)
            if 'p' in param_samples:
                new_param_samples['p'] = param_samples['p']
            
            # Handle p_capture parameters (cell-specific)
            if 'p_capture' in param_samples:
                new_param_samples['p_capture'] = param_samples['p_capture']
            
            new_posterior_samples['parameter_samples'] = new_param_samples
            return new_posterior_samples
        
        # Process each parameter type individually if not using parameter_samples dict
        for param_key, param_value in samples.items():
            if param_key.startswith('r_') or param_key.startswith('gate_'):
                # Shape is typically (n_samples, n_components, n_genes) or (n_components, n_genes)
                if len(param_value.shape) == 3:  # With samples dimension
                    new_posterior_samples[param_key] = param_value[:, component_index]
                elif len(param_value.shape) == 2:  # Without samples dimension
                    new_posterior_samples[param_key] = param_value[component_index]
            else:
                # Copy other parameters as is
                new_posterior_samples[param_key] = param_value
        
        return new_posterior_samples

    # --------------------------------------------------------------------------

    def _create_component_subset(
        self,
        component_index,
        new_params: Dict,
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray]
    ) -> 'ScribeResults':
        """Create a new instance for a specific component."""
        # Create a non-mixture model type
        base_model = self.model_type.replace('_mix', '')

        # Create a modified model config with n_components=None to indicate
        # this is now a non-mixture result after component selection
        new_model_config = replace(
            self.model_config,
            base_model=base_model,
            n_components=None,
            mixing_distribution_model=None,
            mixing_distribution_guide=None
        )
               
        return type(self)(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=self.n_genes,
            model_type=base_model,  # Remove _mix suffix
            model_config=new_model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=self.var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=self.n_vars,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=None  # No longer a mixture model
        )

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the model and guide functions based on model type."""
        from .model_registry import get_model_and_guide
        return get_model_and_guide(self.model_type)

    # --------------------------------------------------------------------------
    # Get log likelihood function
    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for this model type."""
        from .model_registry import get_log_likelihood_fn
        return get_log_likelihood_fn(self.model_type)

    # --------------------------------------------------------------------------
    # Posterior sampling methods
    # --------------------------------------------------------------------------

    def get_posterior_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
    ) -> Dict:
        """Sample parameters from the variational posterior distribution."""
        # Get the guide function 
        _, guide = self.get_model_and_guide()
        
        # Prepare base model arguments
        model_args = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'model_config': self.model_config
        }
        
        # Add specialized arguments based on model type
        if self.model_type == "nbdm":
            model_args['total_counts'] = None  # Will be filled during sampling
            
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
            self.posterior_samples = posterior_samples
            
        return posterior_samples

    # --------------------------------------------------------------------------

    def get_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Get the model and guide functions
        model, _ = self.get_model_and_guide()
        
        # Prepare base model arguments
        model_args = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'model_config': self.model_config,
        }
        
        # Add specialized arguments based on model type
        if self.model_type == "nbdm":
            model_args['total_counts'] = None  # Will be filled during sampling
            
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )
        
        # Generate predictive samples
        predictive_samples = generate_predictive_samples(
            model,
            self.posterior_samples,
            model_args,
            rng_key=rng_key,
            batch_size=batch_size
        )
        
        # Store samples if requested
        if store_samples:
            self.predictive_samples = predictive_samples
            
        return predictive_samples

    # --------------------------------------------------------------------------

    def get_ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        resample_parameters: bool = False,
    ) -> Dict:
        """Generate posterior predictive check samples."""
        # Check if we need to resample parameters
        need_params = (
            resample_parameters or 
            self.posterior_samples is None
        )

        # Generate posterior samples if needed
        if need_params:
            # Sample parameters and generate predictive samples
            self.get_posterior_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                store_samples=store_samples,
            )

        # Generate predictive samples using existing parameters
        _, key_pred = random.split(rng_key)

        self.get_predictive_samples(
            rng_key=key_pred,
            batch_size=batch_size,
            store_samples=store_samples,
        )
            
        return {
            'parameter_samples': self.posterior_samples,
            'predictive_samples': self.predictive_samples
        }
    # --------------------------------------------------------------------------
    # Compute log likelihood methods 
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
            If True, returns log likelihoods for each mixture component separately.
            Only applicable for mixture models.
        weights : Optional[jnp.ndarray], default=None
            Array used to weight the log likelihoods (for mixture models).
        dtype : jnp.dtype, default=jnp.float32
            Data type for numerical precision in computations
            
        Returns
        -------
        jnp.ndarray
            Array of log likelihoods. Shape depends on model type, return_by and
            split_components parameters.
            For standard models:
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
        # Check if posterior samples exist
        if self.posterior_samples is None:
            raise ValueError(
                "No posterior samples found. Call get_posterior_samples() first."
            )
        
        # Get parameter samples
        parameter_samples = self.posterior_samples
        
        # Get number of samples from first parameter
        n_samples = parameter_samples[next(iter(parameter_samples))].shape[0]
        
        # Get likelihood function
        likelihood_fn = self.get_log_likelihood_fn()
        
        # Determine if this is a mixture model
        is_mixture = self.n_components is not None and self.n_components > 1
        
        # Define function to compute likelihood for a single sample
        @jit
        def compute_sample_lik(i):
            # Extract parameters for this sample
            params_i = {k: v[i] for k, v in parameter_samples.items()}
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
                    dtype=dtype
                )
            else:
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
            # Check for NaNs appropriately based on dimensions
            if is_mixture and split_components:
                # Handle case with component dimension
                valid_samples = ~jnp.any(
                    jnp.any(jnp.isnan(log_liks), axis=-1), 
                    axis=-1
                )
            else:
                # Standard case
                valid_samples = ~jnp.any(jnp.isnan(log_liks), axis=-1)
                
            # Filter out samples with NaNs
            if jnp.any(~valid_samples):
                print(f"    - Fraction of samples removed: {1 - jnp.mean(valid_samples)}")
                return log_liks[valid_samples]
        
        return log_liks 
    # --------------------------------------------------------------------------
    # Cell type assignment method for mixture models
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
            1. Computes component-specific log-likelihoods using posterior samples
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
            Array used to weight genes when computing log likelihoods
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
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Cell type assignment only applies to mixture models with "
                "multiple components"
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
        # stability. First compute log(sum(exp(x))) along component axis.
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
                if verbose and cell % 1000 == 0 and cell > 0:
                    print(f"    - Fitting Dirichlet distributions for "
                          f"cells {cell}-{min(cell+1000, n_cells)} out of "
                          f"{n_cells} cells")
                    
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
        Compute probabilistic cell type assignments using MAP estimates of
        parameters.
        
        For each cell, this method:
            1. Computes component-specific log-likelihoods using MAP parameter
               estimates
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
            Array used to weight genes when computing log likelihoods
        verbose : bool, default=True
            If True, prints progress messages
        
        Returns
        -------
        Dict[str, jnp.ndarray]
            Dictionary containing:
                - 'probabilities': Assignment probabilities for each cell.
                  Shape: (n_cells, n_components)
                  
        Raises
        ------
        ValueError
            If the model is not a mixture model
        """
        # Check if this is a mixture model
        if self.n_components is None or self.n_components <= 1:
            raise ValueError(
                "Cell type assignment only applies to mixture models with "
                "multiple components"
            )

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

        # Convert log-likelihoods to probabilities using log-sum-exp for
        # stability. First compute log(sum(exp(x))) along component axis
        log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)
        # Then subtract and exponentiate to get probabilities
        probabilities = jnp.exp(log_liks - log_sum_exp)

        return {
            'probabilities': probabilities
        }