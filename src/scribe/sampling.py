"""
Sampling utilities for SCRIBE.

This module provides functions for posterior sampling, predictive sampling,
and posterior predictive checks (PPCs). It also provides a "biological" PPC
utility that strips technical noise parameters (capture probability, zero-
inflation gate) and samples from the base Negative Binomial distribution
only, reflecting the underlying biology without experimental artifacts.
"""

from jax import random, vmap
import jax.numpy as jnp
import numpyro.distributions as dist
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


# ------------------------------------------------------------------------------
# Biological (denoised) PPC sampling
# ------------------------------------------------------------------------------


def sample_biological_nb(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    n_samples: int = 1,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Sample from the base Negative Binomial, stripping technical noise.

    Generates count samples from the biological NB(r, p) distribution only,
    ignoring technical parameters such as capture probability (``p_capture``
    / ``phi_capture``) and zero-inflation gate. This reflects the "true"
    underlying gene expression as modeled by the Negative Binomial portion
    of the generative process (see the Dirichlet-Multinomial derivation in
    the paper supplement).

    For NBDM models this is equivalent to a standard PPC.  For VCP and ZINB
    variants it yields a *denoised* view of the data.

    The function supports both point estimates (MAP) and full posterior
    samples.  When ``r`` has a leading sample dimension the function uses
    ``jax.vmap`` to vectorise over samples efficiently.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.

        * Standard model, MAP: shape ``(n_genes,)``.
        * Standard model, posterior: shape ``(n_samples, n_genes)``.
        * Mixture model, MAP: shape ``(n_components, n_genes)``.
        * Mixture model, posterior: shape ``(n_samples, n_components,
          n_genes)``.
    p : jnp.ndarray
        Success probability of the Negative Binomial.

        * MAP: scalar or shape ``(n_components,)`` for component-specific p.
        * Posterior: shape ``(n_samples,)`` or ``(n_samples, n_components)``.
    n_cells : int
        Number of cells to generate counts for.
    rng_key : random.PRNGKey
        JAX PRNG key for reproducible sampling.
    n_samples : int, optional
        Number of posterior samples.  When ``r`` already has a leading sample
        dimension this is inferred automatically and this argument is
        ignored.  Default: 1.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.

        * MAP: shape ``(n_components,)``.
        * Posterior: shape ``(n_samples, n_components)``.

        When ``None`` the model is treated as a standard (non-mixture) model.
    cell_batch_size : int or None, optional
        If set, cells are processed in batches of this size to limit peak
        memory usage.  When ``None`` all cells are sampled at once.

    Returns
    -------
    jnp.ndarray
        Sampled counts with shape ``(n_samples, n_cells, n_genes)``.

    Notes
    -----
    The mathematical justification is that the VCP model composes a base
    NB(r, p) with a Binomial capture step:

    .. math::
        \\hat{p} = \\frac{p \\cdot \\nu}{1 - p(1 - \\nu)}

    By sampling from NB(r, p) directly we bypass the capture distortion and
    any zero-inflation, recovering the latent biological distribution.

    Examples
    --------
    >>> # MAP-based biological PPC (standard model)
    >>> samples = sample_biological_nb(
    ...     r=map_estimates["r"],  # (n_genes,)
    ...     p=map_estimates["p"],  # scalar
    ...     n_cells=1000,
    ...     rng_key=jax.random.PRNGKey(0),
    ...     n_samples=5,
    ... )
    >>> samples.shape
    (5, 1000, 5)

    >>> # Full posterior biological PPC (mixture model)
    >>> samples = sample_biological_nb(
    ...     r=posterior["r"],                   # (100, 3, n_genes)
    ...     p=posterior["p"],                   # (100,)
    ...     n_cells=500,
    ...     rng_key=jax.random.PRNGKey(1),
    ...     mixing_weights=posterior["mixing_weights"],  # (100, 3)
    ... )
    >>> samples.shape
    (100, 500, n_genes)
    """
    is_mixture = mixing_weights is not None

    # ------------------------------------------------------------------
    # Determine whether r has a leading sample dimension.
    # Standard model MAP: r.ndim == 1  (n_genes,)
    # Standard model posterior: r.ndim == 2  (n_samples, n_genes)
    # Mixture model MAP: r.ndim == 2  (n_components, n_genes)
    # Mixture model posterior: r.ndim == 3  (n_samples, n_components, n_genes)
    # ------------------------------------------------------------------
    has_sample_dim = (is_mixture and r.ndim == 3) or (
        not is_mixture and r.ndim == 2
    )

    if has_sample_dim:
        # Infer n_samples from the leading dimension of r
        actual_n_samples = r.shape[0]
        # Generate one PRNG key per posterior sample
        keys = random.split(rng_key, actual_n_samples)

        # Define a single-sample helper that we will vmap over
        def _sample_one(key_i, r_i, p_i, mw_i):
            return _sample_biological_nb_single(
                r=r_i,
                p=p_i,
                n_cells=n_cells,
                rng_key=key_i,
                mixing_weights=mw_i,
                cell_batch_size=cell_batch_size,
            )

        # Prepare mixing_weights for vmap (None → dummy zeros that are
        # ignored inside the helper)
        if is_mixture:
            return vmap(_sample_one)(keys, r, p, mixing_weights)
        else:
            # vmap requires concrete arrays – pass a dummy for mw
            dummy_mw = jnp.zeros(actual_n_samples)
            return vmap(
                lambda k, ri, pi, _mw: _sample_biological_nb_single(
                    r=ri,
                    p=pi,
                    n_cells=n_cells,
                    rng_key=k,
                    mixing_weights=None,
                    cell_batch_size=cell_batch_size,
                )
            )(keys, r, p, dummy_mw)
    else:
        # MAP path: no leading sample dimension, so we loop n_samples times
        keys = random.split(rng_key, n_samples)
        all_samples = []
        for i in range(n_samples):
            sample_i = _sample_biological_nb_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=keys[i],
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
            )
            all_samples.append(sample_i)
        # Stack along a new leading sample axis → (n_samples, n_cells, n_genes)
        return jnp.stack(all_samples, axis=0)


# ------------------------------------------------------------------------------


def _sample_biological_nb_single(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Sample one realisation of biological NB counts for all cells.

    This is the inner workhorse called once per posterior sample (or once
    per MAP draw).  It handles both standard and mixture models and
    supports optional cell batching to bound memory usage.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion parameter.

        * Standard: shape ``(n_genes,)``.
        * Mixture: shape ``(n_components, n_genes)``.
    p : jnp.ndarray
        Success probability (scalar or ``(n_components,)``).
    n_cells : int
        Number of cells.
    rng_key : random.PRNGKey
        PRNG key.
    mixing_weights : jnp.ndarray or None
        Component weights ``(n_components,)`` for mixture models.
    cell_batch_size : int or None
        Optional cell-level batching.

    Returns
    -------
    jnp.ndarray
        Counts array of shape ``(n_cells, n_genes)``.
    """
    is_mixture = mixing_weights is not None

    if cell_batch_size is None:
        cell_batch_size = n_cells

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    batch_results = []

    for batch_idx in range(n_batches):
        start = batch_idx * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        batch_n = end - start

        rng_key, batch_key = random.split(rng_key)

        if is_mixture:
            # ----------------------------------------------------------
            # Mixture model: sample component per cell, then draw NB
            # from that component's parameters.
            # ----------------------------------------------------------
            comp_key, sample_key = random.split(batch_key)

            # Draw component assignments: (batch_n,)
            components = dist.Categorical(probs=mixing_weights).sample(
                comp_key, (batch_n,)
            )

            # Gather per-cell r values: (batch_n, n_genes)
            r_batch = r[components]

            # Gather per-cell p values
            p_is_component_specific = p.ndim >= 1 and p.shape[0] == r.shape[0]
            if p_is_component_specific:
                # p: (n_components,) → (batch_n,) → (batch_n, 1)
                p_batch = p[components][:, None]
            else:
                p_batch = p

            nb = dist.NegativeBinomialProbs(r_batch, p_batch)
            batch_counts = nb.sample(sample_key)  # (batch_n, n_genes)
        else:
            # ----------------------------------------------------------
            # Standard model: p is shared across all cells.
            # ----------------------------------------------------------
            nb = dist.NegativeBinomialProbs(r, p)
            batch_counts = nb.sample(batch_key, (batch_n,))  # (batch_n, n_genes)

        batch_results.append(batch_counts)

    return jnp.concatenate(batch_results, axis=0)
