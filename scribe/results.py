"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass

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
    guide_config : GuideConfig
        Configuration object specifying variational family and initialization
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
    guide_config: GuideConfig
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
        """Validate parameters match guide configuration."""
        # Get expected parameter shape based on n_components
        if self.n_components is not None:
            r_expected_shape = (self.n_components, self.n_genes)
        else:
            r_expected_shape = (self.n_genes,)
            
        # Loop through r parameters from guide config
        for param_name in self.guide_config.r_distribution.param_names:
            # Get parameter key
            param_key = f"{param_name}_r"
            # Check if parameter exists and has correct shape
            if param_key not in self.params:
                raise ValueError(f"Missing parameter {param_key}")
            if self.params[param_key].shape != r_expected_shape:
                raise ValueError(
                    f"Parameter {param_key} has shape {self.params[param_key].shape}, "
                    f"expected {r_expected_shape}"
                )
                
        # Validate p parameters (typically scalar/global)
        param_p_shape = ()
        # Loop through p parameters from guide config
        for param_name in self.guide_config.p_distribution.param_names:
            # Get parameter key
            param_key = f"{param_name}_p"
            # Check if parameter exists and has correct shape
            if param_key not in self.params:
                raise ValueError(f"Missing parameter {param_key}")
            if self.params[param_key].shape != param_p_shape:
                raise ValueError(
                    f"Parameter {param_key} has shape {self.params[param_key].shape}, "
                    f"expected {param_p_shape}"
                )
        
        # Validate gate parameters if present
        if self.guide_config.gate_distribution is not None:
            # Loop through gate parameters
            for param_name in self.guide_config.gate_distribution.param_names:
                # Get parameter key
                param_key = f"{param_name}_gate"
                # Check if parameter exists and has correct shape
                if param_key not in self.params:
                    raise ValueError(f"Missing parameter {param_key}")
                if self.params[param_key].shape != r_expected_shape:
                    raise ValueError(
                        f"Parameter {param_key} has shape {self.params[param_key].shape}, "
                        f"expected {r_expected_shape}"
                    )
        
        # Validate p_capture parameters if present
        if self.guide_config.p_capture_distribution is not None:
            # Get expected shape for cell-specific parameter
            expected_shape = (self.n_cells,)
            # Loop through p_capture parameters 
            for param_name in self.guide_config.p_capture_distribution.param_names:
                # Get parameter key
                param_key = f"{param_name}_p_capture"
                # Check if parameter exists and has correct shape
                if param_key not in self.params:
                    raise ValueError(f"Missing parameter {param_key}")
                if self.params[param_key].shape != expected_shape:
                    raise ValueError(
                        f"Parameter {param_key} has shape {self.params[param_key].shape}, "
                        f"expected {expected_shape}"
                    )
                    
        # If mixture model, validate mixing parameters
        if self.n_components is not None:
            # Check if mixing parameter exists and has correct shape
            if "alpha_mixing" not in self.params:
                raise ValueError("Missing mixing weight parameter")
            if self.params["alpha_mixing"].shape != (self.n_components,):
                raise ValueError(
                    f"Mixing parameter has shape {self.params['alpha_mixing'].shape}, "
                    f"expected ({self.n_components},)"
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
        guide_config: GuideConfig,
        **kwargs
    ):
        """Create ScribeResults from AnnData object."""
        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=adata.n_obs,
            n_genes=adata.n_vars,
            model_config=model_config,
            guide_config=guide_config,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            n_obs=adata.n_obs,
            n_vars=adata.n_vars,
            **kwargs
        )
    
    # --------------------------------------------------------------------------
    # Methods for model-specific results
    # --------------------------------------------------------------------------

    def get_model_args(self) -> Dict:
        """
        Get model arguments based on configuration.
        
        Returns
        -------
        Dict
            Dictionary containing arguments needed for model initialization
        """
        args = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
        }
        
        # Add n_components if needed
        if self.n_components is not None:
            args['n_components'] = self.n_components
        
        # Extract distribution names
        r_dist_name = self.model_config.r_distribution.dist_type
        # Add r distribution configuration
        if r_dist_name == 'gamma':
            args['r_prior'] = (
                self.model_config.r_distribution.init_values['alpha'],
                self.model_config.r_distribution.init_values['beta']
            )
        elif r_dist_name == 'lognormal':
            args['r_prior'] = (
                self.model_config.r_distribution.init_values['mu'],
                self.model_config.r_distribution.init_values['sigma']
            )
        
        # Add p distribution configuration
        args['p_prior'] = (
            self.model_config.p_distribution.init_values['alpha'],
            self.model_config.p_distribution.init_values['beta']
        )
        
        # Add gate configuration if present
        if self.model_config.gate_distribution is not None:
            args['gate_prior'] = (
                self.model_config.gate_distribution.init_values['alpha'],
                self.model_config.gate_distribution.init_values['beta']
            )
        
        # Add p_capture configuration if present
        if self.model_config.p_capture_distribution is not None:
            args['p_capture_prior'] = (
                self.model_config.p_capture_distribution.init_values['alpha'],
                self.model_config.p_capture_distribution.init_values['beta']
            )
        
        # Add mixing configuration if mixture model
        if self.n_components is not None:
            args['mixing_prior'] = 1.0  # Default concentration
        
        return args

    # --------------------------------------------------------------------------
    # Get distributions using configs
    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters using guide config.
        
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
            
        dist_method = "get_scipy_dist" if backend == "scipy" else "get_numpyro_dist"
        
        # Extract parameter prefixes and actual values
        r_params = {
            param_name: self.params[f"{param_name}_r"] 
            for param_name in self.guide_config.r_distribution.param_names
        }
        p_params = {
            param_name: self.params[f"{param_name}_p"]
            for param_name in self.guide_config.p_distribution.param_names  
        }
        
        distributions = {
            'r': getattr(self.guide_config.r_distribution, dist_method)(r_params),
            'p': getattr(self.guide_config.p_distribution, dist_method)(p_params)
        }
        
        # Add gate distribution if present
        if self.guide_config.gate_distribution is not None:
            gate_params = {
                param_name: self.params[f"{param_name}_gate"]
                for param_name in self.guide_config.gate_distribution.param_names
            }
            distributions['gate'] = getattr(
                self.guide_config.gate_distribution, dist_method
            )(gate_params)
            
        # Add p_capture distribution if present
        if self.guide_config.p_capture_distribution is not None:
            p_capture_params = {
                param_name: self.params[f"{param_name}_p_capture"]
                for param_name in self.guide_config.p_capture_distribution.param_names
            }
            distributions['p_capture'] = getattr(
                self.guide_config.p_capture_distribution, dist_method
            )(p_capture_params)
            
        # Add mixing weights if mixture model
        if self.n_components is not None:
            if backend == "scipy":
                distributions['mixing_weights'] = stats.dirichlet(
                    self.params['alpha_mixing'])
            else:
                distributions['mixing_weights'] = dist.Dirichlet(
                    self.params['alpha_mixing'])
        
        return distributions

    # --------------------------------------------------------------------------

    def get_map(self) -> Dict[str, jnp.ndarray]:
        """
        Get the maximum a posteriori (MAP) estimates from the variational posterior.
        """
        # Extract parameter prefixes and actual values
        r_params = {
            param_name: self.params[f"{param_name}_r"] 
            for param_name in self.guide_config.r_distribution.param_names
        }
        p_params = {
            param_name: self.params[f"{param_name}_p"]
            for param_name in self.guide_config.p_distribution.param_names  
        }
        
        map_estimates = {
            'r': self.guide_config.r_distribution.get_mode(r_params),
            'p': self.guide_config.p_distribution.get_mode(p_params)
        }
        
        # Add gate MAP if present
        if self.guide_config.gate_distribution is not None:
            gate_params = {
                param_name: self.params[f"{param_name}_gate"]
                for param_name in self.guide_config.gate_distribution.param_names
            }
            map_estimates['gate'] = self.guide_config.gate_distribution.get_mode(gate_params)
            
        # Add p_capture MAP if present
        if self.guide_config.p_capture_distribution is not None:
            p_capture_params = {
                param_name: self.params[f"{param_name}_p_capture"] 
                for param_name in self.guide_config.p_capture_distribution.param_names
            }
            map_estimates['p_capture'] = self.guide_config.p_capture_distribution.get_mode(
                p_capture_params
            )
            
        # Add mixing weights MAP if mixture model
        if self.n_components is not None:
            from .stats import dirichlet_mode
            map_estimates['mixing_weights'] = dirichlet_mode(self.params['alpha_mixing'])
        
        return map_estimates

    # --------------------------------------------------------------------------
    # Indexing
    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        """
        new_params = dict(params)
        
        # Identify gene-specific parameters based on guide configuration
        
        # Handle r parameters, which are always gene-specific
        for param_name in self.guide_config.r_distribution.param_names:
            param_key = f"{param_name}_r"
            if param_key in params:
                if self.n_components is not None:
                    # Keep component dimension but subset gene dimension
                    new_params[param_key] = params[param_key][..., index]
                else:
                    # Just subset gene dimension
                    new_params[param_key] = params[param_key][index]
        
        # Handle gate parameters if present (gene-specific)
        if self.guide_config.gate_distribution is not None:
            for param_name in self.guide_config.gate_distribution.param_names:
                param_key = f"{param_name}_gate"
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
        """
        if samples is None:
            return None
            
        new_posterior_samples = {}
        
        # Process r parameters (always gene-specific)
        for param_name in self.guide_config.r_distribution.param_names:
            param_key = f"{param_name}_r"
            if param_key in samples:
                if self.n_components is not None:
                    # Keep samples and component dimensions, subset gene dimension
                    new_posterior_samples[param_key] = samples[param_key][..., index]
                else:
                    # Keep samples dimension, subset gene dimension
                    new_posterior_samples[param_key] = samples[param_key][..., index]
        
        # Process gate parameters if present (gene-specific)
        if self.guide_config.gate_distribution is not None:
            for param_name in self.guide_config.gate_distribution.param_names:
                param_key = f"{param_name}_gate"
                if param_key in samples:
                    if self.n_components is not None:
                        # Keep samples and component dimensions, subset gene dimension
                        new_posterior_samples[param_key] = samples[param_key][..., index]
                    else:
                        # Keep samples dimension, subset gene dimension
                        new_posterior_samples[param_key] = samples[param_key][..., index]
        
        # Copy non-gene-specific parameters as is
        
        # p parameters (global)
        for param_name in self.guide_config.p_distribution.param_names:
            param_key = f"{param_name}_p"
            if param_key in samples:
                new_posterior_samples[param_key] = samples[param_key]
        
        # p_capture parameters (cell-specific)
        if self.guide_config.p_capture_distribution is not None:
            for param_name in self.guide_config.p_capture_distribution.param_names:
                param_key = f"{param_name}_p_capture"
                if param_key in samples:
                    new_posterior_samples[param_key] = samples[param_key]
        
        # Mixing weights if present
        if self.n_components is not None and "alpha_mixing" in samples:
            new_posterior_samples["alpha_mixing"] = samples["alpha_mixing"]
        
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
            guide_config=self.guide_config,
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
        # Standard models
        if self.model_type == "nbdm":
            from .models import nbdm_log_likelihood
            return nbdm_log_likelihood
        elif self.model_type == "zinb":
            from .models import zinb_log_likelihood
            return zinb_log_likelihood
        elif self.model_type == "nbvcp":
            from .models import nbvcp_log_likelihood
            return nbvcp_log_likelihood
        elif self.model_type == "zinbvcp":
            from .models import zinbvcp_log_likelihood
            return zinbvcp_log_likelihood
        
        # Mixture models
        elif self.model_type == "nbdm_mix":
            from .models_mix import nbdm_mixture_log_likelihood
            return nbdm_mixture_log_likelihood
        elif self.model_type == "zinb_mix":
            from .models_mix import zinb_mixture_log_likelihood
            return zinb_mixture_log_likelihood
        elif self.model_type == "nbvcp_mix":
            from .models_mix import nbvcp_mixture_log_likelihood
            return nbvcp_mixture_log_likelihood
        elif self.model_type == "zinbvcp_mix":
            from .models_mix import zinbvcp_mixture_log_likelihood
            return zinbvcp_mixture_log_likelihood
        elif self.model_type == "nbdm_log_mix":
            from .models_mix import nbdm_mixture_log_likelihood
            return nbdm_mixture_log_likelihood
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

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
        # Get the guide function 
        _, guide = self.get_model_and_guide()
        
        # Get the model arguments (already updated to use configs)
        model_args = self.get_model_args()

        # Add custom arguments if provided
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
            self.posterior_samples = posterior_samples
            
        return posterior_samples

    # --------------------------------------------------------------------------

    def get_predictive_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        custom_args: Optional[Dict] = None,
    ) -> jnp.ndarray:
        """Generate predictive samples using posterior parameter samples."""
        # Get the model and guide functions
        model, _ = self.get_model_and_guide()
        # Get the model arguments
        model_args = self.get_model_args()

        # Add custom arguments if provided
        if custom_args is not None:
            model_args.update(custom_args)
        
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
        custom_args: Optional[Dict] = None,
    ) -> Dict:
        """Generate posterior predictive check samples."""
        # Get the model and guide functions
        model, guide = self.get_model_and_guide()
        # Get the model arguments
        model_args = self.get_model_args()

        # Add custom arguments if provided
        if custom_args is not None:
            model_args.update(custom_args)
        
        # Check if we need to resample parameters
        need_params = (
            resample_parameters or 
            self.posterior_samples is None
        )

        # Generate PPC samples
        if need_params:
            # Sample parameters and generate predictive samples
            posterior_samples = self.get_posterior_samples(
                rng_key=rng_key,
                n_samples=n_samples,
                store_samples=store_samples,
                custom_args=custom_args
            )
            _, key_pred = random.split(rng_key)
            predictive_samples = self.get_predictive_samples(
                rng_key=key_pred,
                batch_size=batch_size,
                store_samples=store_samples,
                custom_args=custom_args
            )
        else:
            # Just generate predictive samples using existing parameters
            _, key_pred = random.split(rng_key)
            predictive_samples = self.get_predictive_samples(
                rng_key=key_pred,
                batch_size=batch_size,
                store_samples=store_samples,
                custom_args=custom_args
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
                    print(f"    - Fitted distributions for {cell}/{n_cells} cells")
                    
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