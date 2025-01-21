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
        """
        Create a new instance with a subset of genes.
        
        This method must be implemented by subclasses to handle class-specific
        attributes when creating a subset.
        
        Parameters
        ----------
        index : array-like
            Boolean or integer index for selecting genes
        new_params : Dict
            Subsetted parameters dictionary
        new_gene_metadata : Optional[pd.DataFrame]
            Subsetted gene metadata
        new_posterior_samples : Optional[Dict]
            Subsetted posterior samples
            
        Returns
        -------
        BaseScribeResults
            New instance with subset of genes
        """
        pass

    # --------------------------------------------------------------------------
    # Get model and guide functions
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """
        Get the model and guide functions for this model type.
        """
        pass

    # --------------------------------------------------------------------------
    # Get likelihood function
    # --------------------------------------------------------------------------

    @abstractmethod
    def get_log_likelihood_fn(self) -> Callable:
        """
        Get the log likelihood function for this model type.
        """
        pass

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
        _, guide = self.get_model_and_guide()
        
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
        model, _ = self.get_model_and_guide()
        
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
        model, guide = self.get_model_and_guide()
        
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

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBDM model and guide functions."""
        from .models import nbdm_model, nbdm_guide
        return nbdm_model, nbdm_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBDM model."""
        from .models import nbdm_log_likelihood
        return nbdm_log_likelihood

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'NBDMResults':
        """Create a new NBDMResults instance with a subset of genes."""
        return NBDMResults(
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

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINB model and guide functions."""
        from .models import zinb_model, zinb_guide
        return zinb_model, zinb_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINB model."""
        from .models import zinb_log_likelihood
        return zinb_log_likelihood

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'ZINBResults':
        """Create a new ZINBResults instance with a subset of genes."""
        return ZINBResults(
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

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBVCP model and guide functions."""
        from .models import nbvcp_model, nbvcp_guide
        return nbvcp_model, nbvcp_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBVCP model."""
        from .models import nbvcp_log_likelihood
        return nbvcp_log_likelihood

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'NBVCPResults':
        """Create a new NBVCPResults instance with a subset of genes."""
        return NBVCPResults(
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

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINBVCP model and guide functions."""
        from .models import zinbvcp_model, zinbvcp_guide
        return zinbvcp_model, zinbvcp_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINBVCP model."""
        from .models import zinbvcp_log_likelihood
        return zinbvcp_log_likelihood

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'ZINBVCPResults':
        """Create a new ZINBVCPResults instance with a subset of genes."""
        return ZINBVCPResults(
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

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBDMMixture model and guide functions."""
        from .models_mix import nbdm_mixture_model, nbdm_mixture_guide
        return nbdm_mixture_model, nbdm_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBDMMixture model."""
        pass

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'NBDMMixtureResults':
        """Create a new NBDMMixtureResults instance with a subset of genes."""
        return NBDMMixtureResults(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples
        )

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

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINB model and guide functions."""
        from .models_mix import zinb_mixture_model, zinb_mixture_guide
        return zinb_mixture_model, zinb_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINB model."""
        pass

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
        # Both r and gate parameters are gene-specific
        # Keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        new_params['alpha_gate'] = params['alpha_gate'][:, index]
        new_params['beta_gate'] = params['beta_gate'][:, index]
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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'ZINBMixtureResults':
        """Create a new ZINBMixtureResults instance with a subset of genes."""
        return ZINBMixtureResults(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples
        )

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

    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the NBVCP model and guide functions."""
        from .models_mix import nbvcp_mixture_model, nbvcp_mixture_guide
        return nbvcp_mixture_model, nbvcp_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for NBVCP model."""
        pass

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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'NBVCPMixtureResults':
        """Create a new NBVCPMixtureResults instance with a subset of genes."""
        return NBVCPMixtureResults(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples
        )

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial-Variable Capture Probability Mixture Model
# ------------------------------------------------------------------------------

@dataclass
class ZINBVCPMixtureResults(MixtureResults):
    """
    Results for Zero-Inflated Negative Binomial mixture model with variable
    capture probability.

    This class extends MixtureResults to handle the ZINBVCP mixture model
    specifics with:
        - Mixing weights for the mixture components
        - A shared success probability p
        - Component-specific dispersion parameters r
        - Cell-specific capture probabilities p_capture
        - Gene-specific dropout probabilities (gate)
    """
    def __post_init__(self):
        assert self.model_type == "zinbvcp_mix", f"Invalid model type: {self.model_type}"

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for mixture model parameters.

        Returns
        -------
        Dict[str, dist.Distribution]
            Dictionary mapping parameter names to their distributions:
                - 'mixing_weights': Dirichlet distribution for mixture weights
                - 'p': Beta distribution for success probability
                - 'r': Gamma distributions for dispersion parameters (one per
                  component per gene)
                - 'p_capture': Beta distributions for capture probabilities (one
                  per cell)
                - 'gate': Beta distributions for dropout probabilities (one per
                  gene)
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

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """Get the ZINBVCP model and guide functions."""
        from .models_mix import zinbvcp_mixture_model, zinbvcp_mixture_guide
        return zinbvcp_mixture_model, zinbvcp_mixture_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for ZINBVCP model."""
        pass

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
        # Both r and gate parameters are gene-specific
        # For r: keep component dimension, subset gene dimension
        new_params['alpha_r'] = params['alpha_r'][:, index]
        new_params['beta_r'] = params['beta_r'][:, index]
        # For gate: subset gene dimension
        new_params['alpha_gate'] = params['alpha_gate'][:, index]
        new_params['beta_gate'] = params['beta_gate'][:, index]
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

    # --------------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_gene_metadata: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict]
    ) -> 'ZINBVCPMixtureResults':
        """Create a new ZINBVCPMixtureResults instance with a subset of genes."""
        return ZINBVCPMixtureResults(
            model_type=self.model_type,
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=int(index.sum() if hasattr(index, 'sum') else len(index)),
            n_components=self.n_components,
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples
        )

# ------------------------------------------------------------------------------
# Custom Model Results
# ------------------------------------------------------------------------------

@dataclass
class CustomResults(BaseScribeResults):
    """
    Results class for custom user-defined models.
    
    This class extends BaseScribeResults to handle user-defined models with
    flexible parameter specifications. It supports both standard and mixture
    model types and maintains compatibility with SCRIBE's inference and sampling
    infrastructure.

    Parameters
    ----------
    params : Dict
        Dictionary of inferred variational model parameters
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
        exactly as it appears in params. For example:
            {
                'alpha_p': {'type': 'global'}, 'beta_p': {'type': 'global'},
                'alpha_r': {'type': 'gene-specific'}, 'beta_r': {'type':
                'gene-specific'}
            }
        Where type must be one of: ['global', 'gene-specific', 'cell-specific']
    custom_model : Callable
        The user's custom model function
    custom_guide : Callable
        The user's custom guide function
    custom_log_likelihood_fn : Optional[Callable]
        Optional function to compute log likelihood for the custom model
    get_distributions_fn : Optional[Callable]
        Optional function to implement get_distributions behavior for the
        custom model
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
    param_spec: Optional[Dict[str, Dict]] = None
    custom_model: Optional[Callable] = None
    custom_guide: Optional[Callable] = None
    custom_log_likelihood_fn: Optional[Callable] = None
    get_distributions_fn: Optional[Callable] = None
    get_model_args_fn: Optional[Callable] = None
    n_components: Optional[int] = None

    def __post_init__(self):
        """Validate parameter specifications against actual parameters."""
        self._validate_param_spec()
    
    # --------------------------------------------------------------------------

    def _validate_param_spec(self):
        """
        Validate that parameter specifications match actual parameters.
        
        This method checks that all parameters in params have specifications and
        that all specified parameters exist in params.
        """
        # Check that all parameters in params have specifications
        for param_name in self.params.keys():
            if param_name not in self.param_spec:
                raise ValueError(
                    f"Parameter {param_name} found in params but not in "
                    f"param_spec"
                )
        
        # Check that all specified parameters exist in params
        for param_name in self.param_spec.keys():
            if param_name not in self.params:
                raise ValueError(
                    f"Parameter {param_name} specified but not found in params"
                )
            
            # Validate parameter type
            spec = self.param_spec[param_name]
            if 'type' not in spec:
                raise ValueError(
                    f"No type specified for parameter {param_name}"
                )
            
            # Check that parameter type is valid
            if spec['type'] not in ['global', 'gene-specific', 'cell-specific']:
                raise ValueError(
                    f"Invalid parameter type {spec['type']} for {param_name}"
                )
            
            # Validate parameter shapes based on type
            param_shape = self.params[param_name].shape
            
            # Check that gene-specific parameters have n_genes as last dimension
            if spec['type'] == 'gene-specific':
                if param_shape[-1] != self.n_genes:
                    raise ValueError(
                        f"Gene-specific parameter {param_name} should have "
                        f"n_genes ({self.n_genes}) as last dimension, "
                        f"got shape {param_shape}"
                    )
            
            # Check that cell-specific parameters have n_cells as last dimension
            elif spec['type'] == 'cell-specific':
                if param_shape[-1] != self.n_cells:
                    raise ValueError(
                        f"Cell-specific parameter {param_name} should have "
                        f"n_cells ({self.n_cells}) as last dimension, "
                        f"got shape {param_shape}"
                    )

    # --------------------------------------------------------------------------

    def get_model_and_guide(self) -> Tuple[Callable, Callable]:
        """
        Get the custom model and guide functions.
        
        Returns
        -------
        Tuple[Callable, Callable]
            Custom model and guide functions
        """
        return self.custom_model, self.custom_guide

    # --------------------------------------------------------------------------

    def get_log_likelihood_fn(self) -> Callable:
        """Get the log likelihood function for custom model."""
        if self.custom_log_likelihood_fn is not None:
            return self.custom_log_likelihood_fn
        else:
            raise ValueError("No custom log likelihood function provided")

    # --------------------------------------------------------------------------

    def get_model_args(self) -> Dict:
        """
        Get the model and guide arguments. If get_model_args_fn was provided,
        uses that function. Otherwise returns default arguments.

        Returns
        -------
        Dict
            Dictionary of arguments to pass to model/guide functions
        """
        if self.get_model_args_fn is not None:
            return self.get_model_args_fn(self)
        
        # Default implementation
        args = {
            'n_cells': self.n_cells,
            'n_genes': self.n_genes
        }
        if self.n_components is not None:
            args['n_components'] = self.n_components
        return args

    # --------------------------------------------------------------------------

    def get_distributions(self, backend: str = "scipy") -> Dict[str, Any]:
        """
        Get the variational distributions for all parameters if a custom
        get_distributions_fn was provided.
        
        Parameters
        ----------
        backend : str
            Statistical package to use for distributions
            
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their distributions, or
            empty dict if no get_distributions_fn provided
        """
        if self.get_distributions_fn is not None:
            return self.get_distributions_fn(self.params, backend)
        return {}

    # --------------------------------------------------------------------------

    def _subset_params(self, params: Dict, index) -> Dict:
        """
        Create a new parameter dictionary for the given index.
        
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
        
        # Subset only gene-specific parameters
        for param_name, spec in self.param_spec.items():
            if spec['type'] == 'gene-specific':
                new_params[param_name] = params[param_name][..., index]
        
        return new_params

    # --------------------------------------------------------------------------

    def _subset_posterior_samples(self, samples: Dict, index) -> Dict:
        """
        Create a new posterior samples dictionary for the given index.
        
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
            
            # Handle each parameter according to its specification
            for param_name, spec in self.param_spec.items():
                if param_name in param_samples:
                    if spec['type'] == 'gene-specific':
                        new_param_samples[param_name] = param_samples[param_name][..., index]
                    else:
                        new_param_samples[param_name] = param_samples[param_name]
            
            new_posterior_samples["parameter_samples"] = new_param_samples
        
        if "predictive_samples" in samples:
            # Keep samples and cells dimensions, subset gene dimension
            new_posterior_samples["predictive_samples"] = samples["predictive_samples"][..., index]
        
        return new_posterior_samples

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
            cell_metadata=self.cell_metadata,
            gene_metadata=new_gene_metadata,
            uns=self.uns,
            posterior_samples=new_posterior_samples,
            param_spec=self.param_spec,
            custom_model=self.custom_model,
            custom_guide=self.custom_guide,
            custom_log_likelihood_fn=self.custom_log_likelihood_fn,
            get_distributions_fn=self.get_distributions_fn,
            get_model_args_fn=self.get_model_args_fn,
            n_components=self.n_components
        )