"""
Results classes for SCRIBE MCMC inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass, field
import warnings

import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
import numpyro.distributions as dist
from jax import random, jit, vmap
from numpyro.infer import MCMC

import numpy as np
import scipy.stats as stats

from .model_config import UnconstrainedModelConfig

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
    model_config : UnconstrainedModelConfig
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
        model_config: UnconstrainedModelConfig,
        prior_params: Dict[str, Any],
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        uns: Optional[Dict] = None,
        n_obs: Optional[int] = None,
        n_vars: Optional[int] = None,
        predictive_samples: Optional[Dict] = None,
        n_components: Optional[int] = None,
        **kwargs
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
        model_config : UnconstrainedModelConfig
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
        self.n_components = (n_components if n_components is not None else
                             model_config.n_components)
        
        # Validate configuration
        self._validate_model_config()
    
    @classmethod
    def from_mcmc(
        cls,
        mcmc: MCMC,
        n_cells: int,
        n_genes: int,
        model_type: str,
        model_config: UnconstrainedModelConfig,
        prior_params: Dict[str, Any],
        **kwargs
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
        model_config : UnconstrainedModelConfig
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
            **kwargs
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
        model_config: UnconstrainedModelConfig,
        prior_params: Dict[str, Any],
        **kwargs
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
        model_config : UnconstrainedModelConfig
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
        params = {param: jnp.mean(values, axis=0) for param, values in samples.items()}
        
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
            **kwargs
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
        samples = self.get_samples()[param]
        return {q: jnp.quantile(samples, q) for q in quantiles}
    
    # --------------------------------------------------------------------------

    def get_map_estimate(self):
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
            potential_energy = self.get_extra_fields()['potential_energy']
            # Get index of minimum potential energy (maximum log density)
            map_idx = int(jnp.argmin(potential_energy))
            # Extract parameters at this index
            map_estimate = {param: values[map_idx] for param, values in samples.items()}
            return map_estimate
        except:
            # Fallback: Return posterior mean as a robust estimator
            # for multimodal distributions or sparse sampling
            map_estimate = {}
            
            for param, values in samples.items():
                # Using mean as a more robust estimator than mode
                map_estimate[param] = jnp.mean(values, axis=0)
            
            return map_estimate
    
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
            if param_name in ['r', 'r_unconstrained', 'gate', 'gate_unconstrained']:
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
        # Handle integer indexing
        if isinstance(index, int):
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = bool_index.at[index].set(True)
            index = bool_index
        
        # Handle slice indexing
        elif isinstance(index, slice):
            indices = jnp.arange(self.n_genes)[index]
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
            index = bool_index
        
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(index[-1], (bool, jnp.bool_)):
            indices = jnp.array(index)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
            index = bool_index

        # Create new metadata if available
        new_var = self.var.iloc[index] if self.var is not None else None

        # Get subset of samples
        samples = self.get_samples()
        new_samples = self._subset_posterior_samples(samples, index)
            
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
            model_config: UnconstrainedModelConfig
            prior_params: Dict
            obs: Optional[pd.DataFrame] = None
            var: Optional[pd.DataFrame] = None
            uns: Optional[Dict] = None
            n_obs: Optional[int] = None
            n_vars: Optional[int] = None
            n_components: Optional[int] = None
            
            def __getitem__(self, index):
                """Support further indexing of subset."""
                # Convert subset indexing to original indexing
                if isinstance(index, int):
                    return self.var.index[index]
                elif isinstance(index, slice):
                    return self.var.index[index]
                else:
                    return self.var.index[index]
        
        # Create and return the subset
        return ScribeMCMCSubset(
            samples=new_samples,
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
            n_components=self.n_components
        )