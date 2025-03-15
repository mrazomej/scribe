"""
Models for single-cell RNA sequencing data with unconstrained parameterization.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp
# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints

# Import typing
from typing import Callable, Dict, Tuple, Optional

# Import model config
from .model_config import UnconstrainedModelConfig

# ------------------------------------------------------------------------------
# Negative Binomial-Dirichlet Multinomial Model
# ------------------------------------------------------------------------------

def nbdm_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: UnconstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial single-cell RNA sequencing data with
    unconstrained parameterization.
    
    This model assumes that for each cell:
        1. The success probability p is sampled in logit space and transformed
           back
        2. The gene-specific dispersion parameters r are sampled in log space
           and transformed back
        3. The counts follow a Negative Binomial distribution with parameters r
           per gene and shared success probability p
           
    Note: This model is equivalent to the Negative Binomial-Dirichlet
    Multinomial model where:
        1. The total UMI count follows a Negative Binomial distribution with
           parameters r_total and p
        2. Given the total count, the individual gene counts follow a
           Dirichlet-Multinomial distribution with concentration parameters r
    The equivalence comes from the fact that a Dirichlet-Multinomial with a
    Negative Binomial total count is equivalent to independent Negative
    Binomials for each gene.
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : UnconstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters: 
            - p_unconstrained_loc: Location for p_unconstrained distribution
            - p_unconstrained_scale: Scale for p_unconstrained distribution
            - r_unconstrained_loc: Location for r_unconstrained distribution
            - r_unconstrained_scale: Scale for r_unconstrained distribution
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - p_unconstrained ~ Normal(p_unconstrained_loc, p_unconstrained_scale) [shape=(1,)]
        - r_unconstrained ~ Normal(r_unconstrained_loc, r_unconstrained_scale) [shape=(n_genes,)]
        - p = sigmoid(p_unconstrained) [shape=(1,)]
        - r = exp(r_unconstrained) [shape=(n_genes,)]

    Likelihood:
        - counts ~ NegativeBinomialLogits(r, p_unconstrained) [shape=(n_cells, n_genes)]
    """
    # Sample unconstrained p parameter
    p_unconstrained = numpyro.sample(
        "p_unconstrained", 
        dist.Normal(
            model_config.p_unconstrained_loc, 
            model_config.p_unconstrained_scale
        )
    )

    # Sample unconstrained r parameter
    r_unconstrained = numpyro.sample(
        "r_unconstrained", 
        dist.Normal(
            model_config.r_unconstrained_loc, 
            model_config.r_unconstrained_scale
        ).expand([n_genes])
    )

    # Convert to probability space for model
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    # Convert to original space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    
    # Define base negative binomial distribution
    base_dist = dist.NegativeBinomialLogits(r, p_unconstrained).to_event(1)
    
    # Model likelihood
    if counts is not None:
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the counts
                numpyro.sample("counts", base_dist, obs=counts)
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the counts
                numpyro.sample("counts", base_dist, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make the distribution return a vector of length n_genes
            counts = numpyro.sample("counts", base_dist)

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Model
# ------------------------------------------------------------------------------

def zinb_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: UnconstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial single-cell RNA sequencing
    data with unconstrained parameterization.
    
    This model assumes that for each cell:
        1. The success probability p is sampled in logit space and transformed
           back
        2. The gene-specific dispersion parameters r are sampled in log space
           and transformed back
        3. The gene-specific dropout rates gate are sampled in logit space and
           transformed back
        4. The counts follow a Zero-Inflated Negative Binomial distribution with
           parameters r per gene, shared success probability p, and
           gene-specific dropout rates gate
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : UnconstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters: 
            - p_unconstrained_loc: Location for p_unconstrained distribution
            - p_unconstrained_scale: Scale for p_unconstrained distribution
            - r_unconstrained_loc: Location for r_unconstrained distribution
            - r_unconstrained_scale: Scale for r_unconstrained distribution
            - gate_unconstrained_loc: Location for gate_unconstrained
              distribution
            - gate_unconstrained_scale: Scale for gate_unconstrained
              distribution
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - p_unconstrained ~ Normal(p_unconstrained_loc, p_unconstrained_scale)
          [shape=(1,)]
        - r_unconstrained ~ Normal(r_unconstrained_loc, r_unconstrained_scale)
          [shape=(n_genes,)]
        - gate_unconstrained ~ Normal(gate_unconstrained_loc,
          gate_unconstrained_scale) [shape=(n_genes,)]
        - p = sigmoid(p_unconstrained) [shape=(1,)]
        - r = exp(r_unconstrained) [shape=(n_genes,)]
        - gate = sigmoid(gate_unconstrained) [shape=(n_genes,)]

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomialLogits(r, p_unconstrained,
          gate_unconstrained) [shape=(n_cells, n_genes)]
    """
    # Sample unconstrained p parameter
    p_unconstrained = numpyro.sample(
        "p_unconstrained", 
        dist.Normal(
            model_config.p_unconstrained_loc, 
            model_config.p_unconstrained_scale
        )
    )
    
    # Sample unconstrained r parameter
    r_unconstrained = numpyro.sample(
        "r_unconstrained", 
        dist.Normal(
            model_config.r_unconstrained_loc, 
            model_config.r_unconstrained_scale
        ).expand([n_genes])
    )
    
    # Sample unconstrained gate parameter
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", 
        dist.Normal(
            model_config.gate_unconstrained_loc, 
            model_config.gate_unconstrained_scale
        ).expand([n_genes])
    )
    
    # Convert to probability space for model
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    # Convert to original space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    # Convert to probability space for gate
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))
    
    # Define base negative binomial distribution
    base_dist = dist.NegativeBinomialLogits(r, p_unconstrained)
    
    # Create zero-inflated distribution
    zinb = dist.ZeroInflatedDistribution(base_dist, gate=gate).to_event(1)
    
    # Model likelihood
    if counts is not None:
        if batch_size is None:
            # Define plate for cells total counts
            with numpyro.plate("cells", n_cells):
                # Likelihood for the counts
                numpyro.sample("counts", zinb, obs=counts)
        else:
            # Define plate for cells total counts
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Likelihood for the counts
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Make the distribution return a vector of length n_genes
            counts = numpyro.sample("counts", zinb)

# ------------------------------------------------------------------------------
# Negative Binomial with Variable Capture Probability (NBVCP) Model
# ------------------------------------------------------------------------------

def nbvcp_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: UnconstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Negative Binomial with Variable Capture Probability
    (NBVCP) using unconstrained parameterization.
    
    This model assumes that for each cell:
        1. The base success probability p is sampled in logit space and
           transformed back
        2. The gene-specific dispersion parameters r are sampled in log space
           and transformed back
        3. Each cell has a cell-specific capture probability p_capture sampled
           in logit space
        4. The effective success probability for each gene in each cell is
           computed from p and p_capture using the composition formula
        5. The counts follow a Negative Binomial distribution with parameters r
           per gene and cell-specific adjusted success probability p_hat
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : UnconstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters:
            - p_unconstrained_loc: Location for p_unconstrained distribution
            - p_unconstrained_scale: Scale for p_unconstrained distribution
            - r_unconstrained_loc: Location for r_unconstrained distribution
            - r_unconstrained_scale: Scale for r_unconstrained distribution
            - p_capture_unconstrained_loc: Location for p_capture_unconstrained
              distribution
            - p_capture_unconstrained_scale: Scale for p_capture_unconstrained
              distribution
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - p_unconstrained ~ Normal(p_unconstrained_loc, p_unconstrained_scale)
          [shape=(1,)]
        - r_unconstrained ~ Normal(r_unconstrained_loc, r_unconstrained_scale)
          [shape=(n_genes,)]
        - p = sigmoid(p_unconstrained) [shape=(1,)]
        - r = exp(r_unconstrained) [shape=(n_genes,)]
        
    Local Parameters:
        - p_capture_unconstrained ~ Normal(p_capture_unconstrained_loc,
          p_capture_unconstrained_scale) [shape=(n_cells,)]
        - p_capture = sigmoid(p_capture_unconstrained) [shape=(n_cells,)]
        - p_hat = p * p_capture / (1 - p * (1 - p_capture)) [shape=(n_cells,
          n_genes)]

    Likelihood:
        - counts ~ NegativeBinomialLogits(r, logit(p_hat)) [shape=(n_cells,
          n_genes)]
    """
    # Sample unconstrained global p parameter
    p_unconstrained = numpyro.sample(
        "p_unconstrained", 
        dist.Normal(
            model_config.p_unconstrained_loc, 
            model_config.p_unconstrained_scale
        )
    )
    
    # Sample unconstrained r parameter
    r_unconstrained = numpyro.sample(
        "r_unconstrained", 
        dist.Normal(
            model_config.r_unconstrained_loc, 
            model_config.r_unconstrained_scale
        ).expand([n_genes])
    )
    
    # Convert to probability space for model
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    # Convert to original space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    
    # Model likelihood
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample unconstrained cell-specific capture probability
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        model_config.p_capture_unconstrained_loc,
                        model_config.p_capture_unconstrained_scale
                    )
                )
                
                # Convert to probability space
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    jsp.special.expit(p_capture_unconstrained)
                )
                
                # Reshape p_capture for broadcasting and compute effective
                # probability
                p_capture_reshaped = p_capture[:, None]
                
                # Compute effective probability (p_hat)
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                
                # Likelihood for the counts
                numpyro.sample(
                    "counts", 
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1), 
                    obs=counts
                )
        else:
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Sample unconstrained cell-specific capture probability
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        model_config.p_capture_unconstrained_loc,
                        model_config.p_capture_unconstrained_scale
                    )
                )
                
                # Convert to probability space
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    jsp.special.expit(p_capture_unconstrained)
                )
                
                # Reshape p_capture for broadcasting and compute effective
                # probability
                p_capture_reshaped = p_capture[:, None]
                
                # Compute effective probability (p_hat)
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                
                # Likelihood for the counts
                numpyro.sample(
                    "counts", 
                    dist.NegativeBinomialProbs(r, p_hat).to_event(1), 
                    obs=counts[idx]
                )
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample unconstrained cell-specific capture probability
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(
                    model_config.p_capture_unconstrained_loc,
                    model_config.p_capture_unconstrained_scale
                )
            )
            
            # Convert to probability space
            p_capture = numpyro.deterministic(
                "p_capture", 
                jsp.special.expit(p_capture_unconstrained)
            )
            
            # Reshape p_capture for broadcasting and compute effective probability
            p_capture_reshaped = p_capture[:, None]
            
            # Compute effective probability (p_hat)
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            )
            
            # Sample counts
            numpyro.sample(
                "counts", 
                dist.NegativeBinomialProbs(r, p_hat).to_event(1)
            )

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial with Variable Capture Probability (ZINBVCP) Model
# ------------------------------------------------------------------------------

def zinbvcp_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: UnconstrainedModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro model for Zero-Inflated Negative Binomial with Variable Capture
    Probability (ZINBVCP) using unconstrained parameterization.
    
    This model assumes that for each cell:
        1. The base success probability p is sampled in logit space and
           transformed back
        2. The gene-specific dispersion parameters r are sampled in log space
           and transformed back
        3. The gene-specific dropout rates gate are sampled in logit space and
           transformed back
        4. Each cell has a cell-specific capture probability p_capture sampled
           in logit space
        5. The effective success probability for each gene in each cell is
           computed from p and p_capture using the composition formula
        6. The counts follow a Zero-Inflated Negative Binomial distribution with
           parameters r per gene, cell-specific adjusted success probability
           p_hat, and gene-specific dropout rates gate
    
    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : UnconstrainedModelConfig
        Configuration object containing prior distributions for model
        parameters:
            - p_unconstrained_loc: Location for p_unconstrained distribution
            - p_unconstrained_scale: Scale for p_unconstrained distribution
            - r_unconstrained_loc: Location for r_unconstrained distribution
            - r_unconstrained_scale: Scale for r_unconstrained distribution
            - gate_unconstrained_loc: Location for gate_unconstrained
              distribution
            - gate_unconstrained_scale: Scale for gate_unconstrained
              distribution
            - p_capture_unconstrained_loc: Location for p_capture_unconstrained
              distribution
            - p_capture_unconstrained_scale: Scale for p_capture_unconstrained
              distribution
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes). If None, generates
        samples from the prior.
    batch_size : int, optional
        Mini-batch size for stochastic variational inference. If None, uses full
        dataset.

    Model Structure
    --------------
    Global Parameters:
        - p_unconstrained ~ Normal(p_unconstrained_loc, p_unconstrained_scale)
          [shape=(1,)]
        - r_unconstrained ~ Normal(r_unconstrained_loc, r_unconstrained_scale)
          [shape=(n_genes,)]
        - gate_unconstrained ~ Normal(gate_unconstrained_loc,
          gate_unconstrained_scale) [shape=(n_genes,)]
        - p = sigmoid(p_unconstrained) [shape=(1,)]
        - r = exp(r_unconstrained) [shape=(n_genes,)]
        - gate = sigmoid(gate_unconstrained) [shape=(n_genes,)]
        
    Local Parameters:
        - p_capture_unconstrained ~ Normal(p_capture_unconstrained_loc,
          p_capture_unconstrained_scale) [shape=(n_cells,)]
        - p_capture = sigmoid(p_capture_unconstrained) [shape=(n_cells,)]
        - p_hat = p * p_capture / (1 - p * (1 - p_capture)) [shape=(n_cells,
          n_genes)]

    Likelihood:
        - counts ~ ZeroInflatedNegativeBinomialLogits(r, logit(p_hat),
          gate_unconstrained) [shape=(n_cells, n_genes)]
    """
    # Sample unconstrained global p parameter
    p_unconstrained = numpyro.sample(
        "p_unconstrained", 
        dist.Normal(
            model_config.p_unconstrained_loc, 
            model_config.p_unconstrained_scale
        )
    )
    
    # Sample unconstrained r parameter
    r_unconstrained = numpyro.sample(
        "r_unconstrained", 
        dist.Normal(
            model_config.r_unconstrained_loc, 
            model_config.r_unconstrained_scale
        ).expand([n_genes])
    )
    
    # Sample unconstrained gate parameter
    gate_unconstrained = numpyro.sample(
        "gate_unconstrained", 
        dist.Normal(
            model_config.gate_unconstrained_loc, 
            model_config.gate_unconstrained_scale
        ).expand([n_genes])
    )
    
    # Convert to probability space for model
    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained))
    # Convert to original space
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained))
    # Convert to probability space for gate
    gate = numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained))
    
    # Model likelihood
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                # Sample unconstrained cell-specific capture probability
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        model_config.p_capture_unconstrained_loc,
                        model_config.p_capture_unconstrained_scale
                    )
                )
                
                # Convert to probability space
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    jsp.special.expit(p_capture_unconstrained)
                )
                
                # Reshape p_capture for broadcasting and compute effective
                # probability
                p_capture_reshaped = p_capture[:, None]
                
                # Compute effective probability (p_hat)
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                
                # Create base negative binomial distribution with adjusted
                # probabilities
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Likelihood for the counts
                numpyro.sample("counts", zinb, obs=counts)
        else:
            with numpyro.plate(
                "cells",
                n_cells,
                subsample_size=batch_size,
            ) as idx:
                # Sample unconstrained cell-specific capture probability
                p_capture_unconstrained = numpyro.sample(
                    "p_capture_unconstrained",
                    dist.Normal(
                        model_config.p_capture_unconstrained_loc,
                        model_config.p_capture_unconstrained_scale
                    )
                )
                
                # Convert to probability space
                p_capture = numpyro.deterministic(
                    "p_capture", 
                    jsp.special.expit(p_capture_unconstrained)
                )
                
                # Reshape p_capture for broadcasting and compute effective
                # probability
                p_capture_reshaped = p_capture[:, None]
                
                # Compute effective probability (p_hat)
                p_hat = numpyro.deterministic(
                    "p_hat",
                    p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
                )
                
                # Create base negative binomial distribution with adjusted
                # probabilities
                base_dist = dist.NegativeBinomialProbs(r, p_hat)
                
                # Create zero-inflated distribution
                zinb = dist.ZeroInflatedDistribution(
                    base_dist, gate=gate).to_event(1)
                
                # Likelihood for the counts
                numpyro.sample("counts", zinb, obs=counts[idx])
    else:
        # Predictive model (no obs)
        with numpyro.plate("cells", n_cells):
            # Sample unconstrained cell-specific capture probability
            p_capture_unconstrained = numpyro.sample(
                "p_capture_unconstrained",
                dist.Normal(
                    model_config.p_capture_unconstrained_loc,
                    model_config.p_capture_unconstrained_scale
                )
            )
            
            # Convert to probability space
            p_capture = numpyro.deterministic(
                "p_capture", 
                jsp.special.expit(p_capture_unconstrained)
            )
            
            # Reshape p_capture for broadcasting and compute effective
            # probability
            p_capture_reshaped = p_capture[:, None]
            
            # Compute effective probability (p_hat)
            p_hat = numpyro.deterministic(
                "p_hat",
                p * p_capture_reshaped / (1 - p * (1 - p_capture_reshaped))
            )
            
            # Create base negative binomial distribution with adjusted
            # probabilities
            base_dist = dist.NegativeBinomialProbs(r, p_hat)
            
            # Create zero-inflated distribution
            zinb = dist.ZeroInflatedDistribution(
                base_dist, gate=gate).to_event(1)
            
            # Sample counts
            numpyro.sample("counts", zinb)


