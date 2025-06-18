"""
Models for single-cell RNA sequencing data with unconstrained parameterization
for mixture models, suitable for MCMC.
"""

# Import JAX-related libraries
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import softmax
# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints # Though not strictly used for unconstrained params, good to have if any part needs it.

# Import typing
from typing import Callable, Dict, Tuple, Optional

# Import model config
from .model_config import ModelConfig

# ------------------------------------------------------------------------------
# Negative Binomial Mixture Model - Unconstrained
# ------------------------------------------------------------------------------

def nbdm_mixture_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Negative Binomial single-cell RNA
    sequencing data. Parameters are sampled in unconstrained space.

    Parameters
    ----------
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_config : ModelConfig
        Configuration object containing prior distributions for unconstrained
        model parameters. Expected attributes include:
            - n_components: Number of mixture components.
            - mixing_logits_unconstrained_loc, mixing_logits_unconstrained_scale
            - p_unconstrained_loc, p_unconstrained_scale
            - r_unconstrained_loc, r_unconstrained_scale
    counts : array-like, optional
        Observed counts matrix of shape (n_cells, n_genes).
    batch_size : int, optional
        Mini-batch size. If None, uses full dataset.

    Model Structure (Unconstrained parameters first)
    --------------
    Global Parameters:
        - mixing_logits_unconstrained ~ Normal(...) [shape=(n_components,)]
        - p_unconstrained_comp ~ Normal(...) [shape=(n_components,)]
        - r_unconstrained_comp ~ Normal(...) [shape=(n_components, n_genes)]
    
    Deterministic Transformations:
        - p = sigmoid(p_unconstrained_comp)
        - r = exp(r_unconstrained_comp)

    Likelihood:
        counts ~ MixtureSameFamily(
            Categorical(logits=mixing_logits_unconstrained),
            NegativeBinomialLogits(total_count=r, logits=p_unconstrained_comp)
        )
    """
    n_components = model_config.n_components

    # Sample unconstrained mixing logits
    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(
            model_config.mixing_logits_unconstrained_loc,
            model_config.mixing_logits_unconstrained_scale
        ).expand([n_components])
    )
    # Add deterministic site for mixing_weights
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    # Sample unconstrained p for each component
    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(
            model_config.p_unconstrained_loc,
            model_config.p_unconstrained_scale
        ).expand([n_components])
    )

    # Sample unconstrained r for each component and gene
    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(
            model_config.r_unconstrained_loc,
            model_config.r_unconstrained_scale
        ).expand([n_components, n_genes])
    )

    # Deterministic transformations to constrained space (mainly for interpretation/logging)
    numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))

    # Define mixing distribution
    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Define component distribution using unconstrained parameters
    # Logits for p is p_unconstrained_comp. total_count for NB is r.
    # NegativeBinomialLogits expects total_count (r) and logits (p_unconstrained)
    # r is (K, G), p_unconstrained_comp is (K,). We need p_unconstrained_comp[:, None] for broadcasting -> (K,1)
    # This makes the NB distribution have batch_shape (K,G).
    # .to_event(1) makes gene dimension part of event_shape. Batch shape becomes (K,). Event shape (G,).
    component_dist = dist.NegativeBinomialLogits(
        total_count=r, # This is already exp(r_unconstrained_comp)
        logits=p_unconstrained_comp[:, None] 
    ).to_event(1)
    
    # Create mixture distribution
    mixture_model = dist.MixtureSameFamily(mixing_dist, component_dist)

    # Model likelihood
    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture_model, obs=counts)
        else:
            with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
                numpyro.sample("counts", mixture_model, obs=counts[idx])
    else:
        # Predictive
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture_model)

# ------------------------------------------------------------------------------
# Zero-Inflated Negative Binomial Mixture Model - Unconstrained
# ------------------------------------------------------------------------------

def zinb_mixture_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Zero-Inflated Negative Binomial
    single-cell RNA sequencing data.

    Parameters
    ----------
    model_config : ModelConfig
        Expected attributes include those for nbdm_mixture_model_unconstrained plus:
            - gate_unconstrained_loc, gate_unconstrained_scale

    Model Structure (Unconstrained parameters first)
    --------------
    Global Parameters: (Same as NBDM-Mix)
        + gate_unconstrained_comp ~ Normal(...) [shape=(n_components, n_genes)]
    
    Deterministic Transformations: (Same as NBDM-Mix)
        + gate = sigmoid(gate_unconstrained_comp)

    Likelihood:
        counts ~ MixtureSameFamily(
            Categorical(logits=mixing_logits_unconstrained),
            ZeroInflatedDistribution(
                NegativeBinomialLogits(total_count=r, logits=p_unconstrained_comp),
                gate_logits=gate_unconstrained_comp
            )
        )
    """
    n_components = model_config.n_components

    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(
            model_config.mixing_logits_unconstrained_loc,
            model_config.mixing_logits_unconstrained_scale
        ).expand([n_components])
    )
    # Add deterministic site for mixing_weights
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(
            model_config.p_unconstrained_loc,
            model_config.p_unconstrained_scale
        ).expand([n_components])
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(
            model_config.r_unconstrained_loc,
            model_config.r_unconstrained_scale
        ).expand([n_components, n_genes])
    )
    
    gate_unconstrained_comp = numpyro.sample(
        "gate_unconstrained_comp",
        dist.Normal(
            model_config.gate_unconstrained_loc,
            model_config.gate_unconstrained_scale
        ).expand([n_components, n_genes])
    )

    numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp))
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp))
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained_comp))

    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained)

    # Base NB distribution: batch_shape (K,G), event_shape ()
    base_nb_dist = dist.NegativeBinomialLogits(
        total_count=r, 
        logits=p_unconstrained_comp[:, None]
    )
    
    # Zero-inflated distribution: gate_logits is (K,G)
    # ZID batch_shape (K,G), event_shape ()
    zinb_base_comp_dist = dist.ZeroInflatedDistribution(
        base_nb_dist, 
        gate_logits=gate_unconstrained_comp
    )

    # .to_event(1) makes gene dimension part of event_shape. Batch shape (K,). Event shape (G,).
    component_dist = zinb_base_comp_dist.to_event(1)
    
    mixture_model = dist.MixtureSameFamily(mixing_dist, component_dist)

    if counts is not None:
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample("counts", mixture_model, obs=counts)
        else:
            with numpyro.plate("cells", n_cells, subsample_size=batch_size) as idx:
                numpyro.sample("counts", mixture_model, obs=counts[idx])
    else:
        with numpyro.plate("cells", n_cells):
            numpyro.sample("counts", mixture_model)

# ------------------------------------------------------------------------------
# Negative Binomial Mixture with Variable Capture Probability - Unconstrained
# ------------------------------------------------------------------------------

def nbvcp_mixture_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Negative Binomial with variable
    capture probability (NBVCP).

    Parameters
    ----------
    model_config : ModelConfig
        Expected attributes include those for nbdm_mixture_model_unconstrained plus:
            - p_capture_unconstrained_loc, p_capture_unconstrained_scale

    Model Structure (Unconstrained parameters first)
    --------------
    Global Parameters: (Same as NBDM-Mix)
    Local (Cell-specific) Parameters:
        - p_capture_unconstrained ~ Normal(...) [shape=(n_cells or batch_size,)]
    
    Deterministic Transformations:
        Global: p, r (Same as NBDM-Mix)
        Local:
            - p_capture = sigmoid(p_capture_unconstrained)
            - p_hat = p * p_capture / (1 - p * (1 - p_capture)) 
                      [shape=(batch_size, n_components)]
    Likelihood (Mixture defined inside cell plate):
        counts ~ MixtureSameFamily(
            Categorical(logits=mixing_logits_unconstrained),
            NegativeBinomialProbs(total_count=r, probs=p_hat) 
        )
    """
    n_components = model_config.n_components

    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(
            model_config.mixing_logits_unconstrained_loc,
            model_config.mixing_logits_unconstrained_scale
        ).expand([n_components])
    )
    # Add deterministic site for mixing_weights
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(
            model_config.p_unconstrained_loc,
            model_config.p_unconstrained_scale
        ).expand([n_components])
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(
            model_config.r_unconstrained_loc,
            model_config.r_unconstrained_scale
        ).expand([n_components, n_genes])
    )

    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp)) # Shape (K,)
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp)) # Shape (K, G)

    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained) # Global

    # Plate for cells
    plate_context = numpyro.plate("cells", n_cells, subsample_size=batch_size if counts is not None else None)
    
    with plate_context as idx:
        current_counts = counts[idx] if counts is not None else None
        
        # Sample unconstrained cell-specific capture probability
        p_capture_unconstrained_batch = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(
                model_config.p_capture_unconstrained_loc,
                model_config.p_capture_unconstrained_scale
            )
            # Implicitly batched by the plate to shape (B,)
        )
        
        p_capture_batch = numpyro.deterministic(
            "p_capture", 
            jsp.special.expit(p_capture_unconstrained_batch)
        ) # Shape (B,)

        # Calculate p_hat:
        # p is (K,), p_capture_batch is (B,)
        # p_hat_batch should be (B, K) for B cells, K components
        p_hat_batch = numpyro.deterministic(
            "p_hat",
            p[None, :] * p_capture_batch[:, None] / \
            (1 - p[None, :] * (1 - p_capture_batch[:, None]))
        ) # Shape (B, K)

        # Revised component_dist_batch for correct MixtureSameFamily shape requirements
        # mixing_dist is global, event_shape (K,). component_dist batch_shape[-1] must be K.
        # p_hat_batch is (B,K). r is (K,G).
        # Target component_dist batch_shape: (B,K), event_shape: (G,)
        component_dist_batch = dist.NegativeBinomialProbs(
            total_count=r[None, :, :],  # Shape (1,K,G) -> broadcasts to (B,K,G)
            probs=p_hat_batch[:, :, None] # Shape (B,K,1) -> broadcasts to (B,K,G)
        ).to_event(1) # batch_shape=(B,K), event_shape=(G,)

        mixture_model_batch = dist.MixtureSameFamily(mixing_dist, component_dist_batch)
        # mixture_model_batch has batch_shape (B,), event_shape (G,)
        
        numpyro.sample("counts", mixture_model_batch, obs=current_counts)


# ------------------------------------------------------------------------------
# Zero-Inflated NB Mixture with Variable Capture Prob - Unconstrained
# ------------------------------------------------------------------------------

def zinbvcp_mixture_model_unconstrained(
    n_cells: int,
    n_genes: int,
    model_config: ModelConfig,
    counts=None,
    batch_size=None,
):
    """
    Numpyro unconstrained mixture model for Zero-Inflated Negative Binomial
    with variable capture probability (ZINBVCP).

    Parameters
    ----------
    model_config : ModelConfig
        Expected attributes include those for nbvcp_mixture_model_unconstrained plus:
            - gate_unconstrained_loc, gate_unconstrained_scale

    Model Structure (Unconstrained parameters first)
    --------------
    Global Parameters: (Same as NBDM-Mix)
        + gate_unconstrained_comp ~ Normal(...) [shape=(n_components, n_genes)]
    Local (Cell-specific) Parameters: (Same as NBVCP-Mix)
    
    Deterministic Transformations:
        Global: p, r
            + gate = sigmoid(gate_unconstrained_comp)
        Local: p_capture, p_hat (Same as NBVCP-Mix)

    Likelihood (Mixture defined inside cell plate):
        counts ~ MixtureSameFamily(
            Categorical(logits=mixing_logits_unconstrained),
            ZeroInflatedDistribution(
                NegativeBinomialProbs(total_count=r, probs=p_hat),
                gate_logits=gate_unconstrained_comp
            )
        )
    """
    n_components = model_config.n_components

    mixing_logits_unconstrained = numpyro.sample(
        "mixing_logits_unconstrained",
        dist.Normal(
            model_config.mixing_logits_unconstrained_loc,
            model_config.mixing_logits_unconstrained_scale
        ).expand([n_components])
    )
    # Add deterministic site for mixing_weights
    numpyro.deterministic(
        "mixing_weights", softmax(mixing_logits_unconstrained, axis=-1)
    )

    p_unconstrained_comp = numpyro.sample(
        "p_unconstrained_comp",
        dist.Normal(
            model_config.p_unconstrained_loc,
            model_config.p_unconstrained_scale
        ).expand([n_components])
    )

    r_unconstrained_comp = numpyro.sample(
        "r_unconstrained_comp",
        dist.Normal(
            model_config.r_unconstrained_loc,
            model_config.r_unconstrained_scale
        ).expand([n_components, n_genes])
    )
    
    gate_unconstrained_comp = numpyro.sample(
        "gate_unconstrained_comp", # Shape (K,G)
        dist.Normal(
            model_config.gate_unconstrained_loc,
            model_config.gate_unconstrained_scale
        ).expand([n_components, n_genes])
    )

    p = numpyro.deterministic("p", jsp.special.expit(p_unconstrained_comp)) # Shape (K,)
    r = numpyro.deterministic("r", jnp.exp(r_unconstrained_comp)) # Shape (K, G)
    # gate is not strictly needed if using gate_logits, but good for interpretation
    numpyro.deterministic("gate", jsp.special.expit(gate_unconstrained_comp))

    mixing_dist = dist.Categorical(logits=mixing_logits_unconstrained) # Global

    plate_context = numpyro.plate("cells", n_cells, subsample_size=batch_size if counts is not None else None)

    with plate_context as idx:
        current_counts = counts[idx] if counts is not None else None

        p_capture_unconstrained_batch = numpyro.sample(
            "p_capture_unconstrained",
            dist.Normal(
                model_config.p_capture_unconstrained_loc,
                model_config.p_capture_unconstrained_scale
            )
        ) # Shape (B,)
        
        p_capture_batch = numpyro.deterministic(
            "p_capture", 
            jsp.special.expit(p_capture_unconstrained_batch)
        ) # Shape (B,)

        p_hat_batch = numpyro.deterministic(
            "p_hat",
            p[None, :] * p_capture_batch[:, None] / \
            (1 - p[None, :] * (1 - p_capture_batch[:, None]))
        ) # Shape (B, K)

        # Revised component_dist_batch for correct MixtureSameFamily shape requirements
        # p_hat_batch is (B,K). r is (K,G). gate_unconstrained_comp is (K,G).
        # Target component_dist batch_shape: (B,K), event_shape: (G,)

        # Base NB distribution for ZINB
        # total_count: r[None,:,:] is (1,K,G)
        # probs: p_hat_batch[:,:,None] is (B,K,1)
        # base_nb_dist_for_zinb has batch_shape (B,K,G)
        base_nb_dist_for_zinb = dist.NegativeBinomialProbs(
            total_count=r[None, :, :],
            probs=p_hat_batch[:, :, None]
        )

        # ZeroInflatedDistribution for ZINB
        # gate_logits: gate_unconstrained_comp[None,:,:] is (1,K,G)
        # zinb_base_comp_dist has batch_shape (B,K,G)
        zinb_base_comp_dist = dist.ZeroInflatedDistribution(
            base_nb_dist_for_zinb,
            gate_logits=gate_unconstrained_comp[None, :, :] 
        )
        
        component_dist_batch = zinb_base_comp_dist.to_event(1) # batch_shape=(B,K), event_shape=(G,)
        
        mixture_model_batch = dist.MixtureSameFamily(mixing_dist, component_dist_batch)
        # mixture_model_batch has batch_shape (B,), event_shape (G,)

        numpyro.sample("counts", mixture_model_batch, obs=current_counts) 