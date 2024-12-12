"""
Stochastic Variational Inference (SVI) module for single-cell RNA sequencing
data analysis.

This module implements a Dirichlet-Multinomial model for scRNA-seq count data
using Numpyro for variational inference.
"""

from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import Predictive, SVI, TraceMeanField_ELBO
from dataclasses import dataclass
from typing import NamedTuple, Dict, Optional, Union
import pandas as pd
import scipy.sparse

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def model(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Numpyro model for Dirichlet-Multinomial single-cell RNA sequencing data.

    This model assumes a hierarchical structure where:
    1. Each cell has a total count drawn from a Negative Binomial distribution
    2. The counts for individual genes are drawn from a Dirichlet-Multinomial
       distribution conditioned on the total count.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p parameter.
        Default is (1, 1) for a uniform prior.
    r_prior : tuple of float
        Parameters (shape, rate) for the Gamma prior on r parameters.
        Default is (2, 2).
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
        If None, generates samples from the prior.
    total_counts : array-like, optional
        Total counts per cell of shape (n_cells,).
        Required if counts is provided.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference.
        If None, uses full dataset.
    """
    # Define the prior on the p parameter
    p = numpyro.sample("p", dist.Beta(p_prior[0], p_prior[1]))

    # Define the prior on the r parameters - one for each category (gene)
    r = numpyro.sample("r", dist.Gamma(
        r_prior[0],
        r_prior[1]
    ).expand([n_genes])
    )

    # Sum of r parameters
    r_total = numpyro.deterministic("r_total", jnp.sum(r))

    # If we have observed data, condition on it
    if counts is not None:
        # If batch size is not provided, use the entire dataset
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells, dim=-1):
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(r, total_count=total_counts),
                    obs=counts
                )
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size
            ) as idx:
                # Likelihood for the total counts - one for each cell
                numpyro.sample(
                    "total_counts",
                    dist.NegativeBinomialProbs(r_total, p),
                    obs=total_counts[idx]
                )

            # Define plate for cells individual counts
            with numpyro.plate("cells", n_cells, dim=-1) as idx:
                # Likelihood for the individual counts - one for each cell
                numpyro.sample(
                    "counts",
                    dist.DirichletMultinomial(
                        r, total_count=total_counts[idx]),
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make a NegativeBinomial distribution that returns a vector of
            # length n_genes
            dist_nb = dist.NegativeBinomialProbs(r, p).to_event(1)
            counts = numpyro.sample("counts", dist_nb)


# ------------------------------------------------------------------------------
# Beta-Gamma Variational Posterior
# ------------------------------------------------------------------------------

def guide(
    n_cells: int,
    n_genes: int,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 2),
    counts=None,
    total_counts=None,
    batch_size=None,
):
    """
    Define the variational distribution for stochastic variational inference.
    
    This guide function specifies the form of the variational distribution that
    will be optimized to approximate the true posterior. It defines a mean-field
    variational family where:
    - The success probability p follows a Beta distribution
    - Each gene's overdispersion parameter r follows an independent Gamma
    distribution
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    p_prior : tuple of float
        Parameters (alpha, beta) for the Beta prior on p (default: (1,1))
    r_prior : tuple of float
        Parameters (alpha, beta) for the Gamma prior on r (default: (2,2))
    counts : array_like, optional
        Observed counts matrix of shape (n_cells, n_genes)
    total_counts : array_like, optional
        Total counts per cell of shape (n_cells,)
    batch_size : int, optional
        Mini-batch size for stochastic optimization
    """
    # register alpha_p and beta_p parameters for the Beta distribution in the
    # variational posterior
    alpha_p = numpyro.param(
        "alpha_p",
        jnp.array(p_prior[0]),
        constraint=constraints.positive
    )
    beta_p = numpyro.param(
        "beta_p",
        jnp.array(p_prior[1]),
        constraint=constraints.positive
    )

    # register one alpha_r and one beta_r parameters for the Gamma distributions
    # for each of the n_genes categories
    alpha_r = numpyro.param(
        "alpha_r",
        jnp.ones(n_genes) * r_prior[0],
        constraint=constraints.positive
    )
    beta_r = numpyro.param(
        "beta_r",
        jnp.ones(n_genes) * r_prior[1],
        constraint=constraints.positive
    )

    # Sample from the variational posterior parameters
    numpyro.sample("p", dist.Beta(alpha_p, beta_p))
    numpyro.sample("r", dist.Gamma(alpha_r, beta_r))

# ------------------------------------------------------------------------------
# Stochastic Variational Inference with Numpyro
# ------------------------------------------------------------------------------

def create_svi_instance(
    n_cells: int,
    n_genes: int,
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
):
    """
    Create an SVI instance with the defined model and guide.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    step_size : float, optional
        Learning rate for the Adam optimizer. Default is 0.001.
    loss : numpyro.infer.elbo, optional
        Loss function to use for the SVI. Default is TraceMeanField_ELBO.

    Returns
    -------
    numpyro.infer.SVI
        Configured SVI instance ready for training
    """
    return SVI(
        model,
        guide,
        optimizer,
        loss=loss
    )

# ------------------------------------------------------------------------------

def run_inference(
    svi_instance: SVI,
    rng_key: random.PRNGKey,
    counts: jnp.ndarray,
    n_steps: int = 100_000,
    batch_size: int = 512,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    cells_axis: int = 0,
) -> numpyro.infer.svi.SVIRunResult:
    """
    Run stochastic variational inference on the provided count data.

    Parameters
    ----------
    svi_instance : numpyro.infer.SVI
        Configured SVI instance for running inference
    rng_key : jax.random.PRNGKey
        Random number generator key
    counts : jax.numpy.ndarray
        Count matrix. If cells_axis=0 (default), shape is (n_cells, n_genes).
        If cells_axis=1, shape is (n_genes, n_cells).
    n_steps : int, optional
        Number of optimization steps. Default is 100,000
    batch_size : int, optional
        Mini-batch size for stochastic optimization. Default is 512
    p_prior : tuple of float, optional
        Parameters (alpha, beta) for Beta prior on p. Default is (1, 1)
    r_prior : tuple of float, optional
        Parameters (shape, rate) for Gamma prior on r. Default is (2, 2)
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default),
        1 means cells are columns.

    Returns
    -------
    numpyro.infer.svi.SVIRunResult
        Results from the SVI run containing optimized parameters and loss
        history
    """
    # Extract dimensions and compute total counts based on cells axis
    if cells_axis == 0:
        n_cells, n_genes = counts.shape
        total_counts = counts.sum(axis=1)
    else:
        n_genes, n_cells = counts.shape
        total_counts = counts.sum(axis=0)
        counts = counts.T  # Transpose to make cells rows for model

    # Run the inference algorithm
    return svi_instance.run(
        rng_key,
        n_steps,
        n_cells,
        n_genes,
        p_prior=p_prior,
        r_prior=r_prior,
        counts=counts,
        total_counts=total_counts,
        batch_size=batch_size
    )

# ------------------------------------------------------------------------------
# Posterior predictive samples
# ------------------------------------------------------------------------------

def generate_ppc_samples(
    params: Dict,
    n_cells: int,
    n_genes: int,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> Dict:
    """
    Generate posterior predictive check samples.
    
    Parameters
    ----------
    params : Dict
        Dictionary containing optimized variational parameters
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.
        
    Returns
    -------
    Dict
        Dictionary containing:
        - 'parameter_samples': Samples from the variational posterior
        - 'predictive_samples': Samples from the predictive distribution
    """
    # Split RNG key for parameter sampling and predictive sampling
    key_params, key_pred = random.split(rng_key)
    
    # Create predictive object for posterior parameter samples
    predictive_param = Predictive(
        guide,
        params=params,
        num_samples=n_samples
    )
    
    # Sample parameters from the variational posterior
    posterior_param_samples = predictive_param(
        key_params,
        n_cells,
        n_genes,
        counts=None,
        total_counts=None
    )
    
    # Create predictive object for generating new data
    predictive = Predictive(
        model,
        posterior_param_samples,
        num_samples=n_samples
    )
    
    # Generate predictive samples
    predictive_samples = predictive(
        key_pred,
        n_cells,
        n_genes,
        counts=None,
        total_counts=None,
        batch_size=batch_size
    )
    
    return {
        'parameter_samples': posterior_param_samples,
        'predictive_samples': predictive_samples["counts"]
    }

# ------------------------------------------------------------------------------
# ScribeResults class
# ------------------------------------------------------------------------------

@dataclass
class ScribeResults:
    """
    Container for SCRIBE inference results and associated metadata.

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
    cell_metadata : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    gene_metadata : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    posterior_samples : Optional[Dict]
        Samples from the posterior distribution, if generated

    Methods
    -------
    from_anndata(adata, params, loss_history, **kwargs)
        Class method to create ScribeResults from an AnnData object
    ppc_samples(rng_key, n_samples=98, batch_size=None)
        Generate posterior predictive check samples
    """
    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    
    # Standard metadata from AnnData object
    cell_metadata: Optional[pd.DataFrame] = None    # from adata.obs
    gene_metadata: Optional[pd.DataFrame] = None    # from adata.var
    uns: Optional[Dict] = None                      # from adata.uns
    
    # Optional results
    posterior_samples: Optional[Dict] = None
    
    # --------------------------------------------------------------------------
    # Create ScribeResults from AnnData object
    # --------------------------------------------------------------------------
    
    @classmethod
    def from_anndata(
        cls,
        adata: "AnnData",
        params: Dict,
        loss_history: jnp.ndarray,
        **kwargs
    ):
        """
        Create ScribeResults from AnnData object.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing single-cell data
        params : Dict
            Dictionary of model parameters from SCRIBE inference
        loss_history : jnp.ndarray
            Array containing the loss values during training
        **kwargs : dict
            Additional keyword arguments to pass to ScribeResults constructor

        Returns
        -------
        ScribeResults
            New ScribeResults instance containing inference results and metadata

        Example
        -------
        >>> # Assume `adata` is an AnnData object with your single-cell data
        >>> params = {...}  # Your model parameters  
        >>> loss_history = jnp.array([...])  # Your loss history
        >>> results = ScribeResults.from_anndata(adata, params, loss_history)
        """
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
    # Posterior predictive check samples
    # --------------------------------------------------------------------------
    
    def ppc_samples(
        self,
        rng_key: random.PRNGKey = random.PRNGKey(41),
        n_samples: int = 99,
        batch_size: Optional[int] = None,
        store_samples: bool = True,
    ) -> Dict:
        """
        Generate posterior predictive check samples.
        
        This is a method wrapper around the generate_ppc_samples function. See
        generate_ppc_samples for full documentation.

        Parameters
        ----------
        store_samples : bool, optional
            If True, stores the generated samples in the posterior_samples
            attribute. Default is True.
        """
        from .svi import generate_ppc_samples

        samples = generate_ppc_samples(
            self.params,
            self.n_cells,
            self.n_genes,
            rng_key,
            n_samples=n_samples,
            batch_size=batch_size
        )
        
        if store_samples:
            self.posterior_samples = samples
            
        return samples

    # --------------------------------------------------------------------------
    # Enable indexing of ScribeResults object
    # --------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Enable indexing of ScribeResults object.
        
        Parameters
        ----------
        index : str, bool array, int array, int, or slice
            Index specification. Can be:
            - A boolean mask
            - A list of integer positions
            - A string that matches a column in cell_metadata or gene_metadata
            - An integer for single gene selection
            - A slice object for range selection
            
        Returns
        -------
        ScribeResults
            A new ScribeResults object with the subset of data
        """
        # Handle integer indexing
        if isinstance(index, int):
            # Initialize boolean index for all genes
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            # Set the gene at the specified index to True
            bool_index = bool_index.at[index].set(True)
            # Use the boolean index for subsetting
            index = bool_index
        
        # Handle slice indexing
        elif isinstance(index, slice):
            # Create array of indices
            indices = jnp.arange(self.n_genes)[index]
            # Create boolean mask
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = jnp.isin(jnp.arange(self.n_genes), indices)
            index = bool_index
        
        # Handle list/array indexing
        elif not isinstance(index, (bool, jnp.bool_)) and not isinstance(index[-1], (bool, jnp.bool_)):
            indices = jnp.array(index)
            bool_index = jnp.zeros(self.n_genes, dtype=bool)
            bool_index = jnp.where(jnp.isin(jnp.arange(self.n_genes), indices), True, bool_index)
            index = bool_index
        
        # Create new params dict with subset of r parameters
        new_params = dict(self.params)
        new_params['alpha_r'] = self.params['alpha_r'][index]
        new_params['beta_r'] = self.params['beta_r'][index]
        
        # Create new metadata if available
        new_gene_metadata = self.gene_metadata.iloc[index] if self.gene_metadata is not None else None

        # Create new posterior samples if available
        if self.posterior_samples is not None:
            # Extract r parameter samples from posterior samples
            new_r_param_samples = self.posterior_samples["parameter_samples"]["r"][:, index]
            # create new posterior samples with the new r parameter samples
            new_param_samples = {
                "parameter_samples": {
                    "p": self.posterior_samples["parameter_samples"]["p"],
                    "r": new_r_param_samples
                }
            }
            # Extract predictive samples from posterior samples
            new_predictive_samples = self.posterior_samples["predictive_samples"][:, :, index]
            # create new posterior samples with the new predictive samples
            new_posterior_samples = {
                "parameter_samples": new_param_samples,
                "predictive_samples": new_predictive_samples
            }
        else:
            new_posterior_samples = None
            
        
        return ScribeResults(
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
# SCRIBE inference pipeline
# ------------------------------------------------------------------------------

def run_scribe(
    counts: Union[jnp.ndarray, "AnnData"],
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_steps: int = 100_000,
    batch_size: int = 512,
    p_prior: tuple = (1, 1),
    r_prior: tuple = (2, 0.1),
    loss: numpyro.infer.elbo = TraceMeanField_ELBO(),
    optimizer: numpyro.optim.optimizers = numpyro.optim.Adam(step_size=0.001),
    cells_axis: int = 0,
    layer: Optional[str] = None,
) -> ScribeResults:
    """
    Run the complete SCRIBE inference pipeline and return a ScribeResults
    object.

    Parameters
    ----------
    counts : Union[jax.numpy.ndarray, AnnData]
        Either a count matrix or an AnnData object: - If ndarray and
        cells_axis=0 (default), shape is (n_cells, n_genes) - If ndarray and
        cells_axis=1, shape is (n_genes, n_cells) - If AnnData, will use .X or
        specified layer
    rng_key : jax.random.PRNGKey, optional
        Random number generator key. Default is PRNGKey(42)
    n_steps : int, optional
        Number of optimization steps. Default is 100,000
    batch_size : int, optional
        Mini-batch size for stochastic optimization. Default is 512
    p_prior : tuple of float, optional
        Parameters (alpha, beta) for Beta prior on p. Default is (1, 1)
    r_prior : tuple of float, optional
        Parameters (shape, rate) for Gamma prior on r. Default is (2, 0.1)
    loss : numpyro.infer.elbo, optional
        Loss function to use for the SVI. Default is TraceMeanField_ELBO
    optimizer : numpyro.optim.optimizers, optional
        Optimizer to use for SVI. Default is Adam with step_size=0.001
    cells_axis : int, optional
        Axis along which cells are arranged. 0 means cells are rows (default), 1
        means cells are columns.
    layer : str, optional
        If counts is AnnData, specifies which layer to use. If None, uses .X

    Returns
    -------
    ScribeResults
        Results container with inference results and optional metadata
    """
    # Check if input is AnnData
    is_anndata = False
    try:
        import anndata
        is_anndata = isinstance(counts, anndata.AnnData)
    except ImportError:
        pass

    # Handle AnnData input
    if is_anndata:
        adata = counts
        # Get counts from specified layer or .X
        count_data = adata.layers[layer] if layer else adata.X
        # Convert to dense array if sparse
        if scipy.sparse.issparse(count_data):
            count_data = count_data.toarray()
        count_data = jnp.array(count_data)
    else:
        count_data = counts
        adata = None

    # Extract dimensions based on cells axis
    if cells_axis == 0:
        n_cells, n_genes = count_data.shape
    else:
        n_genes, n_cells = count_data.shape
        count_data = count_data.T  # Transpose to make cells rows

    # Create SVI instance
    svi = create_svi_instance(
        n_cells,
        n_genes,
        optimizer=optimizer,
        loss=loss
    )

    # Run inference
    svi_results = run_inference(
        svi,
        rng_key,
        count_data,
        n_steps=n_steps,
        batch_size=batch_size,
        p_prior=p_prior,
        r_prior=r_prior,
        cells_axis=0  # Already transposed if needed
    )

    # Create ScribeResults object
    if is_anndata:
        results = ScribeResults.from_anndata(
            adata=adata,
            params=svi_results.params,
            loss_history=svi_results.losses
        )
    else:
        results = ScribeResults(
            params=svi_results.params,
            loss_history=svi_results.losses,
            n_cells=n_cells,
            n_genes=n_genes
        )

    return results