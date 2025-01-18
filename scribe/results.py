"""
Results classes for SCRIBE inference.
"""

# Imports for class definitions
from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Imports for inference results
import jax.numpy as jnp
import pandas as pd
import numpyro.distributions as dist
from numpyro.infer import Predictive
from jax import random

# Import scipy.stats for distributions
import scipy.stats as stats

# Imports for model-specific results
from .models import get_model_and_guide, get_log_likelihood_fn

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
    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions
        """
        pass

    # --------------------------------------------------------------------------

    @abstractmethod
    def get_model_args(self) -> Dict:
        """
        Get the model and guide arguments for this model type.

        For standard models, this is just the number of cells and genes.
        For mixture models, this is the number of cells, genes, and components.
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
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
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
    # Get model and guide functions
    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """
        Get the model and guide functions for this model type.
        """
        return get_model_and_guide(self.model_type)

    # --------------------------------------------------------------------------
    # Get likelihood function
    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """
        Get the log likelihood function for this model type.
        """
        return get_log_likelihood_fn(self.model_type)

    # --------------------------------------------------------------------------
    # Posterior predictive check samples
    # --------------------------------------------------------------------------

    def get_posterior_samples(
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
        
        # Get the model arguments for this model type
        model_args = self.get_model_args()
        
        # Use the sampling utility function
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
        rng_key: random.PRNGKey,
        batch_size: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Generate predictive samples using posterior parameter samples.
        """
        # Get the model function for this model type
        model, _ = get_model_and_guide(self.model_type)
        
        # Get the model arguments for this model type
        model_args = self.get_model_args()
        
        # Use the sampling utility function
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
    ) -> Dict:
        """
        Generate posterior predictive check samples.
        """
        # Get the model and guide functions
        model, guide = get_model_and_guide(self.model_type)
        
        # Get the model arguments for this model type
        model_args = self.get_model_args()
        
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
# Base class for standard (non-mixture) models
# ------------------------------------------------------------------------------

@dataclass
class StandardResults(BaseScribeResults):
    """
    Abstract base class for standard (non-mixture) models.
    """
    def get_model_args(self) -> Dict:
        """
        Standard models just need cells and genes.
        """
        return {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
        }

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial model
# ------------------------------------------------------------------------------

@dataclass
class NBDMResults(StandardResults):
    """
    Results for Negative Binomial-Dirichlet Multinomial model.
    """
    def __post_init__(self):
        # Change this to only verify the model type instead of setting it
        assert self.model_type == "nbdm", f"Invalid model type: {self.model_type}"

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for NBDM parameters.
        """
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
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

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for NBDM model.
        """
        new_params = dict(params)
        # Only r parameters are gene-specific
        new_params['alpha_r'] = params['alpha_r'][index]
        new_params['beta_r'] = params['beta_r'][index]
        return new_params

    # --------------------------------------------------------------------------

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
class ZINBResults(StandardResults):
    """
    Results for Zero-Inflated Negative Binomial model.
    """
    def __post_init__(self):
        # Change this to only verify the model type instead of setting it
        assert self.model_type == "zinb", f"Invalid model type: {self.model_type}"

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for ZINB parameters.
        """
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'], self.params['beta_gate']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'gate': dist.Beta(
                    self.params['alpha_gate'], self.params['beta_gate']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------
# Negative Binomial with variable capture probability model
# ------------------------------------------------------------------------------

@dataclass
class NBVCPResults(StandardResults):
    """
    Results for Negative Binomial with variable capture probability model.
    """
    def __post_init__(self):
        assert self.model_type == "nbvcp", f"Invalid model type: {self.model_type}"

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for NBVCP parameters.
        """
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
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

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for NBVCP model.
        """
        new_params = dict(params)
        # Only r parameters are gene-specific
        new_params['alpha_r'] = params['alpha_r'][index]
        new_params['beta_r'] = params['beta_r'][index]
        # p and p_capture parameters are shared across genes
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for NBVCP model.
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
            # p_capture is cell-specific
            if "p_capture" in param_samples:
                new_param_samples["p_capture"] = param_samples["p_capture"]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][:, :, index]
            
        return new_posterior_samples

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with variable capture probability
# ------------------------------------------------------------------------------

@dataclass
class ZINBVCPResults(StandardResults):
    """
    Results for Zero-Inflated Negative Binomial model with variable capture probability.
    
    This class extends BaseScribeResults to handle the specific parameters and
    structure of the ZINBVCP model, which includes:
    - A shared success probability (p)
    - Gene-specific dispersion parameters (r)
    - Cell-specific capture probabilities (p_capture)
    - Gene-specific dropout probabilities (gate)
    """
    def __post_init__(self):
        assert self.model_type == "zinbvcp", f"Invalid model type: {self.model_type}"

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for ZINBVCP parameters.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions:
            - 'p': Beta distribution for success probability
            - 'r': Gamma distribution for gene-specific dispersion
            - 'p_capture': Beta distribution for cell-specific capture
            probabilities
            - 'gate': Beta distribution for gene-specific dropout probabilities
        """
        if backend == "scipy":
            return {
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'], 
                    self.params['beta_p_capture']
                ),
                'gate': stats.beta(
                    self.params['alpha_gate'], self.params['beta_gate']
                )
            }
        elif backend == "numpyro":
            return {
                'p': dist.Beta(self.params['alpha_p'], self.params['beta_p']),
                'r': dist.Gamma(self.params['alpha_r'], self.params['beta_r']),
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'], 
                    self.params['beta_p_capture']
                ),
                'gate': dist.Beta(
                    self.params['alpha_gate'], self.params['beta_gate']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for ZINBVCP model.
        
        Parameters
        ----------
        params : Dict
            Original parameter dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New parameter dictionary with subset of gene-specific parameters
        """
        new_params = dict(params)
        # r and gate parameters are gene-specific
        new_params['alpha_r'] = params['alpha_r'][index]
        new_params['beta_r'] = params['beta_r'][index]
        new_params['alpha_gate'] = params['alpha_gate'][index]
        new_params['beta_gate'] = params['beta_gate'][index]
        # p and p_capture parameters are shared or cell-specific, so keep as is
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for ZINBVCP model.
        
        Parameters
        ----------
        samples : Dict
            Original samples dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New samples dictionary with subset of gene-specific samples
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # p is shared across genes
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            # p_capture is cell-specific
            if "p_capture" in param_samples:
                new_param_samples["p_capture"] = param_samples["p_capture"]
            # r and gate are gene-specific
            if "r" in param_samples:
                new_param_samples["r"] = param_samples["r"][:, index]
            if "gate" in param_samples:
                new_param_samples["gate"] = param_samples["gate"][:, index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][:, :, index]
            
        return new_posterior_samples

# ------------------------------------------------------------------------------
# Base class for mixture models
# ------------------------------------------------------------------------------

@dataclass
class MixtureResults(BaseScribeResults):
    """
    Abstract base class for mixture models.
    """
    n_components: int = 2
    
    def get_model_args(self) -> Dict:
        """Mixture models need components too."""
        return {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes,
            'n_components': self.n_components,
        }

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class NBDMMixtureResults(MixtureResults):
    """
    Results for Negative Binomial mixture model.

    This class extends BaseScribeResults to handle Negative Binomial-Dirichlet
    Multinomial mixture model specifics with the following parameters:
        - mixing_weights: Dirichlet distribution for mixture weights
        - p: Beta distributions for success probabilities (one per component)
        - r: Gamma distributions for dispersion parameters (one per component
          per gene)
    """
    # Number of mixture components
    n_components: int = 2

    def __post_init__(self):
        assert self.model_type == "nbdm_mix", f"Invalid model type: {self.model_type}"

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for mixture model parameters.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their Numpyro distributions:
            - 'mixing_weights': Dirichlet distribution for mixture weights
            - 'p': Beta distributions for success probabilities (one per component)
            - 'r': Gamma distributions for dispersion parameters (one per component per gene)
        """
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
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

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for subset of genes.
        
        Parameters
        ----------
        params : Dict
            Original parameter dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New parameter dictionary with subset of gene-specific parameters
        """
        new_params = dict(params)
        # Only r parameters are gene-specific
        # Keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for subset of genes.
        
        Parameters
        ----------
        samples : Dict
            Original samples dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New samples dictionary with subset of gene-specific samples
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # mixing_weights are component-specific
            if "mixing_weights" in param_samples:
                new_param_samples["mixing_weights"] = param_samples["mixing_weights"]
            
            # p is component-specific
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            
            # r is both component and gene-specific
            if "r" in param_samples:
                # Keep samples and component dimensions, subset gene dimension
                new_param_samples["r"] = param_samples["r"][..., index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
            
        return new_posterior_samples

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class ZINBMixtureResults(MixtureResults):
    """
    Results for Zero-Inflated Negative Binomial mixture model.

    This class extends MixtureResults to handle ZINB mixture model specifics
    with the following parameters:
        - mixing_weights: Dirichlet distribution for mixture weights
        - p: Beta distributions for success probabilities (one per component)
        - r: Gamma distributions for dispersion parameters (one per component
          per gene)
        - gate: Beta distributions for dropout probabilities (one per component
          per gene)
    
    Attributes
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
        Type of model used for inference (must be "zinb_mix")
    n_components : int
        Number of mixture components
    cell_metadata : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    gene_metadata : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    posterior_samples : Optional[Dict]
        Samples from the posterior distribution, if generated
    """
    # Number of mixture components
    n_components: int = 2

    def __post_init__(self):
        assert self.model_type == "zinb_mix", f"Invalid model type: {self.model_type}"

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for mixture model parameters.

        Parameters
        ----------
        backend : str
            Statistical package to use for distributions. Must be one of:
            - "scipy": Returns scipy.stats distributions
            - "numpyro": Returns numpyro.distributions

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their distributions:
            - 'mixing_weights': Dirichlet distribution for mixture weights
            - 'p': Beta distributions for success probabilities (one per component)
            - 'r': Gamma distributions for dispersion parameters (one per component
              per gene)
            - 'gate': Beta distributions for dropout probabilities (one per
              component per gene)
        """
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
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

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for subset of genes.
        
        Parameters
        ----------
        params : Dict
            Original parameter dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New parameter dictionary with subset of gene-specific parameters
        """
        new_params = dict(params)
        # Both r and gate parameters are gene-specific
        # Keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        new_params['alpha_gate'] = params['alpha_gate'][:, index]
        new_params['beta_gate'] = params['beta_gate'][:, index]
        return new_params

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for subset of genes.
        
        Parameters
        ----------
        samples : Dict
            Original samples dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New samples dictionary with subset of gene-specific samples
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # mixing_weights are component-specific
            if "mixing_weights" in param_samples:
                new_param_samples["mixing_weights"] = param_samples["mixing_weights"]
            
            # p is component-specific
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            
            # r and gate are both component and gene-specific
            if "r" in param_samples:
                # Keep samples and component dimensions, subset gene dimension
                new_param_samples["r"] = param_samples["r"][..., index]
            if "gate" in param_samples:
                new_param_samples["gate"] = param_samples["gate"][..., index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
            
        return new_posterior_samples

# ------------------------------------------------------------------------------
# Negative Binomial-Variable Capture Probability Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class NBVCPMixtureResults(MixtureResults):
    """
    Results for Negative Binomial mixture model with variable capture
    probability.

    This class extends MixtureResults to handle the specific parameters and
    structure of the NBVCP mixture model, which includes:
        - Mixing weights for the mixture components
        - A shared success probability p
        - Component-specific dispersion parameters r
        - Cell-specific capture probabilities p_capture
    """
    def __post_init__(self):
        assert self.model_type == "nbvcp_mix", f"Invalid model type: {self.model_type}"

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for mixture model parameters.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their distributions:
            - 'mixing_weights': Dirichlet distribution for mixture weights
            - 'p': Beta distribution for success probability
            - 'r': Gamma distributions for dispersion parameters (one per component per gene)
            - 'p_capture': Beta distributions for capture probabilities (one per cell)
        """
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
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

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for subset of genes.
        
        Parameters
        ----------
        params : Dict
            Original parameter dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New parameter dictionary with subset of gene-specific parameters
        """
        new_params = dict(params)
        # Only r parameters are gene-specific
        # Keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        return new_params

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for subset of genes.
        
        Parameters
        ----------
        samples : Dict
            Original samples dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New samples dictionary with subset of gene-specific samples
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # mixing_weights are component-specific
            if "mixing_weights" in param_samples:
                new_param_samples["mixing_weights"] = param_samples["mixing_weights"]
            
            # p is shared across genes
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            
            # p_capture is cell-specific
            if "p_capture" in param_samples:
                new_param_samples["p_capture"] = param_samples["p_capture"]
                
            # r is both component and gene-specific
            if "r" in param_samples:
                # Keep samples and component dimensions, subset gene dimension
                new_param_samples["r"] = param_samples["r"][..., index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
            
        return new_posterior_samples


@dataclass
class ZINBVCPMixtureResults(MixtureResults):
    """
    Results for Zero-Inflated Negative Binomial mixture model with variable capture probability.

    This class extends MixtureResults to handle the ZINBVCP mixture model specifics with:
        - Mixing weights for the mixture components
        - A shared success probability p
        - Component-specific dispersion parameters r
        - Cell-specific capture probabilities p_capture
        - Gene-specific dropout probabilities (gate)
    """
    def __post_init__(self):
        assert self.model_type == "zinbvcp_mix", f"Invalid model type: {self.model_type}"

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for mixture model parameters.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their distributions:
            - 'mixing_weights': Dirichlet distribution for mixture weights
            - 'p': Beta distribution for success probability
            - 'r': Gamma distributions for dispersion parameters (one per component per gene)
            - 'p_capture': Beta distributions for capture probabilities (one per cell)
            - 'gate': Beta distributions for dropout probabilities (one per gene)
        """
        if backend == "scipy":
            return {
                'mixing_weights': stats.dirichlet(self.params['alpha_mixing']),
                'p': stats.beta(self.params['alpha_p'], self.params['beta_p']),
                'r': stats.gamma(
                    self.params['alpha_r'],
                    loc=0,
                    scale=1 / self.params['beta_r']
                ),
                'p_capture': stats.beta(
                    self.params['alpha_p_capture'], 
                    self.params['beta_p_capture']
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
                'p_capture': dist.Beta(
                    self.params['alpha_p_capture'], 
                    self.params['beta_p_capture']
                ),
                'gate': dist.Beta(
                    self.params['alpha_gate'],
                    self.params['beta_gate']
                )
            }
        else:
            raise ValueError(f"Invalid backend: {backend}")

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create new parameter dictionary for subset of genes.
        
        Parameters
        ----------
        params : Dict
            Original parameter dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New parameter dictionary with subset of gene-specific parameters
        """
        new_params = dict(params)
        # Both r and gate parameters are gene-specific
        # For r: keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        # For gate: subset gene dimension
        new_params['alpha_gate'] = params['alpha_gate'][:, index]
        new_params['beta_gate'] = params['beta_gate'][:, index]
        return new_params

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create new posterior samples dictionary for subset of genes.
        
        Parameters
        ----------
        samples : Dict
            Original samples dictionary
        index : array-like
            Boolean or integer index for selecting genes
            
        Returns
        -------
        Dict
            New samples dictionary with subset of gene-specific samples
        """
        new_posterior_samples = {}
        
        if "parameter_samples" in samples:
            param_samples = samples["parameter_samples"]
            new_param_samples = {}
            
            # mixing_weights are component-specific
            if "mixing_weights" in param_samples:
                new_param_samples["mixing_weights"] = param_samples["mixing_weights"]
            
            # p is shared across genes
            if "p" in param_samples:
                new_param_samples["p"] = param_samples["p"]
            
            # p_capture is cell-specific
            if "p_capture" in param_samples:
                new_param_samples["p_capture"] = param_samples["p_capture"]
                
            # r is both component and gene-specific
            if "r" in param_samples:
                # Keep samples and component dimensions, subset gene dimension
                new_param_samples["r"] = param_samples["r"][..., index]
            
            # gate is gene-specific
            if "gate" in param_samples:
                new_param_samples["gate"] = param_samples["gate"][..., index]
                
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
            
        return new_posterior_samples