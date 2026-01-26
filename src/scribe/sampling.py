"""
Sampling utilities for SCRIBE.
"""

from jax import random
import jax.numpy as jnp
from numpyro.infer import Predictive
from typing import Dict, Optional, Union, Callable, List
from numpyro.infer import SVI
from numpyro.handlers import block

# ------------------------------------------------------------------------------
# Posterior predictive samples
# ------------------------------------------------------------------------------


def sample_variational_posterior(
    guide: Callable,
    params: Dict,
    model: Callable,
    model_args: Dict,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples: int = 100,
    return_sites: Optional[Union[str, List[str]]] = None,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
    """
    Sample parameters from the variational posterior distribution.

    Parameters
    ----------
    guide : Callable
        Guide function
    params : Dict
        Dictionary containing optimized variational parameters
    model : Callable
        Model function
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
    return_sites : Optional[Union[str, List[str]]], optional
        Sites to return from the model. If None, returns all sites.
    counts : Optional[jnp.ndarray], optional
        Observed count matrix of shape (n_cells, n_genes). Required when using
        amortized capture probability (e.g., with
        amortization.capture.enabled=true). For non-amortized models, this can
        be None. Default: None.

    Returns
    -------
    Dict
        Dictionary containing samples from the variational posterior
    """
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Add counts to model_args if provided (needed for amortized guides)
    if counts is not None:
        model_args = {**model_args, "counts": counts}

    # Create predictive object for posterior parameter samples
    predictive_param = Predictive(guide, params=params, num_samples=n_samples)

    # Sample parameters from the variational posterior
    posterior_samples = predictive_param(rng_key, **model_args)

    # Also run the model to get deterministic sites.
    # We block the 'counts' site to prevent Predictive from sampling it,
    # which avoids a potentially huge memory allocation.
    blocked_model = block(model, hide=["counts"])
    predictive_model = Predictive(
        blocked_model, posterior_samples=posterior_samples
    )
    model_samples = predictive_model(rng_key, **model_args)

    # Combine samples from guide and model
    posterior_samples.update(model_samples)
    return posterior_samples


# ------------------------------------------------------------------------------


def generate_predictive_samples(
    model: Callable,
    posterior_samples: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """
    Generate predictive samples using posterior parameter samples.

    Parameters
    ----------
    model : Callable
        Model function
    posterior_samples : Dict
        Dictionary containing samples from the variational posterior
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.

    Returns
    -------
    jnp.ndarray
        Array of predictive samples
    """
    # Find the first array value to get num_samples
    # Skip nested dicts from flax_module parameters (e.g., "amortizer$params")
    num_samples = None
    for value in posterior_samples.values():
        if hasattr(value, "shape"):
            num_samples = value.shape[0]
            break

    if num_samples is None:
        raise ValueError(
            "Could not determine num_samples from posterior_samples. "
            "No array values found (all values are nested dicts?)."
        )

    # Create predictive object for generating new data
    predictive = Predictive(
        model,
        posterior_samples,
        num_samples=num_samples,
        # Include deterministic parameters in the predictive distribution
        exclude_deterministic=False,
    )

    # Generate predictive samples
    predictive_samples = predictive(
        rng_key, **model_args, batch_size=batch_size
    )

    return predictive_samples["counts"]


# ------------------------------------------------------------------------------


def generate_ppc_samples(
    model: Callable,
    guide: Callable,
    params: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
    """
    Generate posterior predictive check samples.

    Parameters
    ----------
    model : Callable
        Model function
    guide : Callable
        Guide function
    params : Dict
        Dictionary containing optimized variational parameters
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of posterior samples to generate (default: 100)
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.
    counts : Optional[jnp.ndarray], optional
        Observed count matrix of shape (n_cells, n_genes). Required when using
        amortized capture probability (e.g., with amortization.capture.enabled=true).
        For non-amortized models, this can be None. Default: None.

    Returns
    -------
    Dict
        Dictionary containing: - 'parameter_samples': Samples from the
        variational posterior - 'predictive_samples': Samples from the
        predictive distribution
    """
    # Split RNG key for parameter sampling and predictive sampling
    key_params, key_pred = random.split(rng_key)

    # Sample from variational posterior
    posterior_param_samples = sample_variational_posterior(
        guide, params, model, model_args, key_params, n_samples, counts=counts
    )

    # Generate predictive samples
    predictive_samples = generate_predictive_samples(
        model,
        posterior_param_samples,
        model_args,
        key_pred,
        batch_size,
    )

    return {
        "parameter_samples": posterior_param_samples,
        "predictive_samples": predictive_samples,
    }


# ------------------------------------------------------------------------------


def generate_prior_predictive_samples(
    model: Callable,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """
    Generate prior predictive samples using the model.

    Parameters
    ----------
    model : Callable
        Model function
    model_args : Dict
        Dictionary containing model arguments. For standard models, this is
        just the number of cells and genes. For mixture models, this is the
        number of cells, genes, and components.
    rng_key : random.PRNGKey
        JAX random number generator key
    n_samples : int, optional
        Number of prior predictive samples to generate (default: 100)
    batch_size : int, optional
        Batch size for generating samples. If None, uses full dataset.

    Returns
    -------
    jnp.ndarray
        Array of prior predictive samples
    """
    # Create predictive object for generating new data from the prior
    predictive = Predictive(model, num_samples=n_samples)

    # Generate prior predictive samples
    prior_predictive_samples = predictive(
        rng_key, **model_args, batch_size=batch_size
    )

    return prior_predictive_samples["counts"]
