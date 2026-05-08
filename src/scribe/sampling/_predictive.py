"""Variational and prior predictive sampling via NumPyro ``Predictive``."""

from typing import Any, Callable, Dict, List, Optional, Union

from jax import random
import jax.numpy as jnp
from numpyro.infer import Predictive
from numpyro.handlers import block, condition


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

    # Separate model-only kwargs that the guide does not accept (e.g.
    # total_count_max used by the Multinomial likelihood).
    guide_args = {k: v for k, v in model_args.items()
                  if k != "total_count_max"}

    # Create predictive object for posterior parameter samples
    predictive_param = Predictive(guide, params=params, num_samples=n_samples)

    # Sample parameters from the variational posterior
    posterior_samples = predictive_param(rng_key, **guide_args)

    # Also run the model to get deterministic sites.
    # We block the 'counts' site to prevent Predictive from sampling it,
    # which avoids a potentially huge memory allocation.
    blocked_model = block(model, hide=["counts"])
    predictive_model = Predictive(
        blocked_model, posterior_samples=posterior_samples, params=params
    )
    model_samples = predictive_model(rng_key, **model_args)

    # Combine samples from guide and model
    posterior_samples.update(model_samples)
    return posterior_samples


def generate_predictive_samples(
    model: Callable,
    posterior_samples: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
    params: Optional[Dict] = None,
    condition_data: Optional[Dict[str, Any]] = None,
) -> jnp.ndarray:
    """Generate predictive samples using posterior parameter samples.

    NumPyro's ``Predictive`` vectorises over the posterior-sample
    dimension automatically (controlled by its ``parallel`` flag).
    Cell-level batching is intentionally absent here: posterior samples
    are drawn at full-cell resolution, and the predictive model must
    replay with the same cell dimension.

    Parameters
    ----------
    model : Callable
        Model function.
    posterior_samples : Dict
        Dictionary containing samples from the variational posterior.
    model_args : Dict
        Dictionary containing model arguments (n_cells, n_genes,
        model_config, etc.).  Passed as keyword arguments to the model
        via ``Predictive``.
    rng_key : random.PRNGKey
        JAX random number generator key.
    params : Dict, optional
        Optional trained parameter dictionary used to substitute model-side
        ``numpyro.param`` / ``flax_module`` weights during replay. This is
        required for VAE models so decoder parameters are not re-initialized.
    condition_data : Dict[str, Any], optional
        Per-site values to condition on during the predictive replay,
        passed through ``numpyro.handlers.condition``. Use this to fix
        certain latent sites at observed values *while still sampling
        downstream sites freshly*. The canonical example is the
        **conditional PPC** for LNM-family models: pass
        ``{"u_T": observed_totals}`` so the predictive replay uses the
        per-cell observed library sizes rather than re-drawing them
        from the global NB. Sites listed here are not present in
        ``posterior_samples`` (they were ``obs=``-tagged during
        training and so are not stored in the variational posterior);
        ``condition`` injects them into the replay's trace before
        ``Predictive`` substitutes the posterior-sampled latents.
        ``None`` (the default) reproduces the original behaviour:
        every latent gets sampled from the model.

    Returns
    -------
    jnp.ndarray
        Array of predictive count samples.
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

    # When the caller supplies ``condition_data``, wrap the model so
    # those sites are fixed before NumPyro's ``Predictive`` runs. The
    # wrapped model is what ``Predictive`` traces. The order matters:
    # ``condition`` substitutes its sites at sample time; ``Predictive``
    # then substitutes the posterior-sampled latents for the remaining
    # sites; the still-free sites (e.g. ``counts``) are sampled fresh
    # from their distributions, conditioned on whatever upstream values
    # the conditioning + posterior-sample chain has fixed.
    model_to_use = model
    if condition_data is not None and len(condition_data) > 0:
        model_to_use = condition(model, data=condition_data)

    # Create predictive object for generating new data
    predictive = Predictive(
        model_to_use,
        posterior_samples,
        num_samples=num_samples,
        exclude_deterministic=False,
        params=params if params is not None else {},
    )

    # NumPyro's Predictive.__call__ passes **kwargs directly to the
    # model.  We must NOT add extra kwargs (like batch_size) here --
    # they would leak into the model as cell-plate subsample_size and
    # create a shape mismatch with the full-cell posterior samples.
    predictive_samples = predictive(rng_key, **model_args)

    return predictive_samples["counts"]


def generate_ppc_samples(
    model: Callable,
    guide: Callable,
    params: Dict,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
    """Generate posterior predictive check samples.

    Parameters
    ----------
    model : Callable
        Model function.
    guide : Callable
        Guide function.
    params : Dict
        Dictionary containing optimized variational parameters.
    model_args : Dict
        Dictionary containing model arguments (n_cells, n_genes,
        model_config, etc.).
    rng_key : random.PRNGKey
        JAX random number generator key.
    n_samples : int, optional
        Number of posterior samples to generate (default: 100).
    counts : Optional[jnp.ndarray], optional
        Observed count matrix of shape (n_cells, n_genes).  Required
        when using amortized capture probability.  Default: None.

    Returns
    -------
    Dict
        Dictionary with keys ``parameter_samples`` and
        ``predictive_samples``.
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
        params=params,
    )

    return {
        "parameter_samples": posterior_param_samples,
        "predictive_samples": predictive_samples,
    }


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
