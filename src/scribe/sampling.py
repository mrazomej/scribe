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
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples: int = 100,
    return_sites: Optional[Union[str, List[str]]] = None,
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

    Returns
    -------
    Dict
        Dictionary containing samples from the variational posterior
    """
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
    # Create predictive object for generating new data
    predictive = Predictive(
        model,
        posterior_samples,
        # Take the number of samples from the first parameter
        num_samples=next(iter(posterior_samples.values())).shape[0],
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
        guide, params, model, model_args, key_params, n_samples
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
