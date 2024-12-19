"""
Results classes for SCRIBE inference.
"""

# Imports for class definitions
from typing import Dict, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Imports for inference results
import jax.numpy as jnp
import pandas as pd
import numpyro.distributions as dist
from numpyro.infer import Predictive
from jax import random

# Imports for model-specific results
from .models import get_model_and_guide

# Imports for sampling
from .sampling import sample_variational_posterior, generate_predictive_samples, generate_ppc_samples


# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------

@dataclass
class BaseScribeResults(ABC):
    """
    Base class for SCRIBE inference results.
    
    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, training history, and dataset dimensions. It can
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
    cell_metadata : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    gene_metadata : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    posterior_samples : Optional[Dict]
        Samples from the posterior distribution, if generated
    """
    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    
    # Standard metadata from AnnData object
    cell_metadata: Optional[pd.DataFrame] = None
    gene_metadata: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    
    # Optional results
    posterior_samples: Optional[Dict] = None

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
    def get_distributions(self) -> Dict[str, dist.Distribution]:
        """
        Get the variational distributions for all parameters.
        
        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their Numpyro distributions
        """
        pass

    # --------------------------------------------------------------------------

    @abstractmethod
    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        
        This method must be implemented by subclasses to handle model-specific
        parameter subsetting.
        """
        pass

    # --------------------------------------------------------------------------

    @abstractmethod
    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        
        This method must be implemented by subclasses to handle model-specific
        posterior sample subsetting.
        """
        pass

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
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set True for the given indices
            bool_index = jnp.where(jnp.isin(jnp.arange(self.n_genes), indices), True, bool_index)
            # Set index to boolean index
            index = bool_index

        # Create new params dict with subset of parameters
        new_params = self._subset_params(self.params, index)
        
        # Create new metadata if available
        new_gene_metadata = self.gene_metadata.iloc[index] if self.gene_metadata is not None else None

        # Create new posterior samples if available
        new_posterior_samples = None
        # Check if posterior samples are available
        if self.posterior_samples is not None:
            # Create new posterior samples for the given index
            new_posterior_samples = self._subset_posterior_samples(self.posterior_samples, index)
            
        # Create new instance with the subset of data
        return self.__class__(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples
        )

    # --------------------------------------------------------------------------
    # Posterior predictive check samples
    # --------------------------------------------------------------------------

    def sample_posterior(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        store_samples: bool = True,
    ) -> Dict:
        """
        Sample parameters from the variational posterior distribution.
        """
        # Get the guide function for this model type
        _, guide = get_model_and_guide(self.model_type)
        
        # Use the sampling utility function
        posterior_samples = sample_variational_posterior(
            guide,
            self.params,
            self.n_cells,
            self.n_genes,
            rng_key,
            n_samples
        )
        
        # Store samples if requested
        if store_samples:
            if self.posterior_samples is not None:
                self.posterior_samples['parameter_samples'] = posterior_samples
            else:
                self.posterior_samples = {'parameter_samples': posterior_samples}
            
        return posterior_samples

    # --------------------------------------------------------------------------

    def generate_predictive_samples(
        self,
        rng_key: random.PRNGKey,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Generate predictive samples using posterior parameter samples.
        """
        # Get the model function for this model type
        model, _ = get_model_and_guide(self.model_type)
        
        # Use the sampling utility function
        return generate_predictive_samples(
            model,
            self.posterior_samples["parameter_samples"],
            self.n_cells,
            self.n_genes,
            rng_key,
            batch_size
        )

    # --------------------------------------------------------------------------

    def ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(42),
        n_samples: int = 100,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
        resample_parameters: bool = False,
    ) -> Dict:
        """
        Generate posterior predictive check samples.
        """
        # Get the model and guide functions
        model, guide = get_model_and_guide(self.model_type)
        
        # Check if we need to resample parameters
        need_params = (
            resample_parameters or 
            self.posterior_samples is None or 
            'parameter_samples' not in self.posterior_samples
        )
        
        if need_params:
            # Use generate_ppc_samples utility function
            samples = generate_ppc_samples(
                model,
                guide,
                self.params,
                self.n_cells,
                self.n_genes,
                rng_key,
                n_samples,
                batch_size
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
                    self.n_cells,
                    self.n_genes,
                    key_pred,
                    batch_size
                )
            }
        
        if store_samples:
            self.posterior_samples = samples
            
        return samples


# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial model
# ------------------------------------------------------------------------------

@dataclass
class NBDMResults(BaseScribeResults):
    """
    Results for Negative Binomial-Dirichlet Multinomial model.
    """
    def __post_init__(self):
        # Change this to only verify the model type instead of setting it
        assert self.model_type == "nbdm", f"Invalid model type: {self.model_type}"

    def get_distributions(self) -> Dict[str, dist.Distribution]:
        """
        Get the variational distributions for NBDM parameters.
        """
        return {
            'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
            'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r'])
        }

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for NBDM model.
        """
        new_params = dict(params)
        # Only r parameters are gene-specific
        new_params['alpha_r'] = params['alpha_r'][index]
        new_params['beta_r'] = params['beta_r'][index]
        return new_params

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for NBDM model.
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # p is shared across genes
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            # r is gene-specific
            if "r" in param_samples:
                new_param_samples["r"] = param_samples["r"][:, index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][:, :, index]
            
        return new_posterior_samples


# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial model
# ------------------------------------------------------------------------------

@dataclass
class ZINBResults(BaseScribeResults):
    """
    Results for Zero-Inflated Negative Binomial model.
    """
    def __post_init__(self):
        # Change this to only verify the model type instead of setting it
        assert self.model_type == "zinb", f"Invalid model type: {self.model_type}"

    def get_distributions(self) -> Dict[str, dist.Distribution]:
        """
        Get the variational distributions for ZINB parameters.
        """
        return {
            'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
            'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
            'gate': dist.Beta(self.params['alpha_gate'], self.params['beta_gate'])
        }

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for ZINB model.
        """
        new_params = dict(params)
        # Both r and gate parameters are gene-specific
        new_params['alpha_r'] = params['alpha_r'][index]
        new_params['beta_r'] = params['beta_r'][index]
        new_params['alpha_gate'] = params['alpha_gate'][index]
        new_params['beta_gate'] = params['beta_gate'][index]
        return new_params

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for ZINB model.
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # p is shared across genes
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            # r and gate are gene-specific
            if "r" in param_samples:
                new_param_samples["r"] = param_samples["r"][:, index]
            if "gate" in param_samples:
                new_param_samples["gate"] = param_samples["gate"][:, index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][:, :, index]
            
        return new_posterior_samples