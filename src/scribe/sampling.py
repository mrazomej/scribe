"""
Sampling utilities for SCRIBE.

This module provides functions for posterior sampling, predictive sampling,
and posterior predictive checks (PPCs). It also provides:

* A **biological PPC** utility that strips technical noise parameters
  (capture probability, zero-inflation gate) and samples from the base
  Negative Binomial distribution only, reflecting the underlying biology
  without experimental artifacts.

* A **Bayesian denoising** utility that takes *observed* count matrices and
  posterior parameter estimates to compute the closed-form posterior of the
  true (pre-capture, pre-dropout) transcript counts.  See
  ``paper/_denoising.qmd`` for the full mathematical derivation.

Parameterization Convention
---------------------------
Throughout this module the canonical ``p`` follows the numpyro convention:
it is the ``probs`` argument of ``NegativeBinomialProbs``, i.e. the
probability of each Bernoulli trial producing a count.  The NB mean is
therefore ``r * p / (1 - p)``.  This is the *complement* of the paper's
:math:`p` (which appears as :math:`p^r` in the PMF).
"""

from jax import random, vmap
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import Predictive
from typing import Dict, Optional, Tuple, Union, Callable, List
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
                p_batch = p[components]
                # When p is (n_components,), indexing yields (batch_n,) and
                # we need (batch_n, 1) to broadcast with r_batch
                # (batch_n, n_genes).  When p is (n_components, n_genes)
                # (hierarchical), indexing already yields (batch_n, n_genes).
                if p_batch.ndim == 1:
                    p_batch = p_batch[:, None]
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


# ------------------------------------------------------------------------------
# Full-model posterior PPC sampling (NB / ZINB / VCP / mixtures)
# ------------------------------------------------------------------------------


def sample_posterior_ppc(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    n_samples: int = 1,
    gate: Optional[jnp.ndarray] = None,
    p_capture: Optional[jnp.ndarray] = None,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Sample from the full generative model using posterior parameters.

    Generates posterior predictive count samples that include **all** model
    components (NB base, zero-inflation gate, capture probability, mixture
    assignments).  Unlike :func:`sample_biological_nb`, this produces
    replicate data comparable to the *observed* counts and is appropriate
    for PPC-based goodness-of-fit evaluation.

    The function supports both MAP point estimates and full posterior
    parameter arrays.  When ``r`` has a leading sample dimension the
    function uses ``jax.vmap`` to vectorise over posterior draws.

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

        * MAP: scalar or ``(n_components,)`` for component-specific p.
        * Posterior: ``(n_samples,)`` or ``(n_samples, n_components)``.
    n_cells : int
        Number of cells to generate counts for.
    rng_key : random.PRNGKey
        JAX PRNG key for reproducible sampling.
    n_samples : int, optional
        Number of draws when ``r`` has no leading sample dimension (MAP
        path).  Ignored when the sample dimension is inferred from ``r``.
        Default: 1.
    gate : jnp.ndarray or None, optional
        Zero-inflation gate probability.

        * MAP standard: ``(n_genes,)``.
        * Posterior standard: ``(n_samples, n_genes)``.
        * MAP mixture: ``(n_components, n_genes)``.
        * Posterior mixture: ``(n_samples, n_components, n_genes)``.

        ``None`` for non-ZINB models.
    p_capture : jnp.ndarray or None, optional
        Per-cell capture probability.

        * MAP: ``(n_cells,)``.
        * Posterior: ``(n_samples, n_cells)``.

        ``None`` for non-VCP models.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.

        * MAP: ``(n_components,)``.
        * Posterior: ``(n_samples, n_components)``.

        ``None`` for non-mixture models.
    cell_batch_size : int or None, optional
        If set, cells are processed in batches of this size to limit peak
        memory.  Particularly useful for VCP models.  ``None`` processes
        all cells at once.

    Returns
    -------
    jnp.ndarray
        Sampled counts with shape ``(n_samples, n_cells, n_genes)``.

    See Also
    --------
    sample_biological_nb : Biological-only (denoised) PPC sampling.

    Examples
    --------
    >>> # Full posterior PPC for a ZINB-VCP model
    >>> samples = sample_posterior_ppc(
    ...     r=posterior["r"],          # (S, n_genes)
    ...     p=posterior["p"],          # (S,)
    ...     n_cells=5000,
    ...     rng_key=jax.random.PRNGKey(0),
    ...     gate=posterior["gate"],    # (S, n_genes)
    ...     p_capture=posterior["p_capture"],  # (S, n_cells)
    ... )
    >>> samples.shape
    (S, 5000, n_genes)
    """
    is_mixture = mixing_weights is not None

    # ------------------------------------------------------------------
    # Detect leading sample dimension using the same heuristic as
    # sample_biological_nb: posterior arrays have one extra leading axis.
    # Standard MAP: r.ndim == 1 ; posterior: r.ndim == 2
    # Mixture  MAP: r.ndim == 2 ; posterior: r.ndim == 3
    # ------------------------------------------------------------------
    has_sample_dim = (is_mixture and r.ndim == 3) or (
        not is_mixture and r.ndim == 2
    )

    if has_sample_dim:
        actual_n_samples = r.shape[0]
        keys = random.split(rng_key, actual_n_samples)

        # Build per-sample slices, using dummy arrays for None optionals
        # so vmap sees concrete array inputs.
        gate_arr = gate if gate is not None else jnp.zeros(actual_n_samples)
        p_cap_arr = (
            p_capture
            if p_capture is not None
            else jnp.zeros(actual_n_samples)
        )
        mw_arr = (
            mixing_weights
            if mixing_weights is not None
            else jnp.zeros(actual_n_samples)
        )

        # Flags must be static for the vmap-ed function
        _has_gate = gate is not None
        _has_p_capture = p_capture is not None
        _is_mixture = is_mixture

        def _sample_one(key_i, r_i, p_i, gate_i, p_cap_i, mw_i):
            return _sample_posterior_ppc_single(
                r=r_i,
                p=p_i,
                n_cells=n_cells,
                rng_key=key_i,
                gate=gate_i if _has_gate else None,
                p_capture=p_cap_i if _has_p_capture else None,
                mixing_weights=mw_i if _is_mixture else None,
                cell_batch_size=cell_batch_size,
            )

        return vmap(_sample_one)(
            keys, r, p, gate_arr, p_cap_arr, mw_arr
        )
    else:
        # MAP path: loop n_samples times
        keys = random.split(rng_key, n_samples)
        all_samples = []
        for i in range(n_samples):
            sample_i = _sample_posterior_ppc_single(
                r=r,
                p=p,
                n_cells=n_cells,
                rng_key=keys[i],
                gate=gate,
                p_capture=p_capture,
                mixing_weights=mixing_weights,
                cell_batch_size=cell_batch_size,
            )
            all_samples.append(sample_i)
        return jnp.stack(all_samples, axis=0)


def _sample_posterior_ppc_single(
    r: jnp.ndarray,
    p: jnp.ndarray,
    n_cells: int,
    rng_key: random.PRNGKey,
    gate: Optional[jnp.ndarray] = None,
    p_capture: Optional[jnp.ndarray] = None,
    mixing_weights: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
) -> jnp.ndarray:
    """Sample one PPC realisation from the full generative model.

    Inner workhorse called once per posterior draw (or once per MAP draw).
    Handles standard, ZINB, VCP, and mixture models with optional cell
    batching.

    Parameters
    ----------
    r : jnp.ndarray
        Dispersion.  ``(n_genes,)`` for standard, ``(n_components,
        n_genes)`` for mixture.
    p : jnp.ndarray
        Success probability.  Scalar or ``(n_components,)`` for mixture.
    n_cells : int
        Number of cells.
    rng_key : random.PRNGKey
        PRNG key.
    gate : jnp.ndarray or None
        Zero-inflation gate.  ``(n_genes,)`` or ``(n_components,
        n_genes)`` for per-component gates.
    p_capture : jnp.ndarray or None
        Per-cell capture probability ``(n_cells,)``.
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
    has_vcp = p_capture is not None
    has_gate = gate is not None

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
            # -------------------------------------------------------
            # Mixture model: sample component per cell, gather params
            # -------------------------------------------------------
            n_components = r.shape[0]
            n_genes = r.shape[1]
            comp_key, sample_key = random.split(batch_key)

            # Component assignments: (batch_n,)
            components = dist.Categorical(probs=mixing_weights).sample(
                comp_key, (batch_n,)
            )

            # Gather per-cell r: (batch_n, n_genes)
            r_batch = r[components]

            # Gather per-cell p
            p_is_component_specific = (
                p.ndim >= 1 and p.shape[0] == n_components
            )
            if p_is_component_specific:
                p_batch = p[components]
                if p_batch.ndim == 1:
                    p_batch = p_batch[:, None]
            else:
                p_batch = p

            # Gather per-cell gate
            if has_gate:
                if gate.ndim == 2 and gate.shape[0] == n_components:
                    gate_batch = gate[components]
                else:
                    gate_batch = gate
            else:
                gate_batch = None

            # VCP: compute p_effective
            if has_vcp:
                p_cap = p_capture[start:end]  # (batch_n,)
                p_cap_exp = p_cap[:, None]    # (batch_n, 1)
                p_effective = (
                    p_batch * p_cap_exp / (1 - p_batch * (1 - p_cap_exp))
                )
            else:
                p_effective = p_batch

            # NB distribution
            nb = dist.NegativeBinomialProbs(r_batch, p_effective)

            # Apply zero-inflation if present
            if gate_batch is not None:
                sample_dist = dist.ZeroInflatedDistribution(
                    nb, gate=gate_batch
                )
            else:
                sample_dist = nb

            batch_counts = sample_dist.sample(sample_key)

        else:
            # -------------------------------------------------------
            # Standard (non-mixture) model
            # -------------------------------------------------------
            # VCP: compute effective p per cell in this batch
            if has_vcp:
                p_cap = p_capture[start:end]       # (batch_n,)
                p_cap_reshaped = p_cap[:, None]    # (batch_n, 1)
                p_effective = (
                    p * p_cap_reshaped
                    / (1 - p * (1 - p_cap_reshaped))
                )
            else:
                p_effective = p

            nb = dist.NegativeBinomialProbs(r, p_effective)

            if has_gate:
                sample_dist = dist.ZeroInflatedDistribution(
                    nb, gate=gate
                )
            else:
                sample_dist = nb

            # Shape depends on whether VCP gives the distribution a
            # batch dimension.
            if has_vcp:
                batch_counts = sample_dist.sample(batch_key)
            else:
                batch_counts = sample_dist.sample(
                    batch_key, (batch_n,)
                )

        batch_results.append(batch_counts)

    return jnp.concatenate(batch_results, axis=0)


# ------------------------------------------------------------------------------
# Bayesian denoising of observed counts
# ------------------------------------------------------------------------------

# Allowed values for individual method elements
_VALID_DENOISE_METHODS = {"mean", "mode", "sample"}


def _validate_denoise_method(method: Union[str, Tuple[str, str]]) -> None:
    """Validate the ``method`` argument for denoising functions.

    Accepts a single string or a tuple of two strings, each of which
    must be one of ``'mean'``, ``'mode'``, or ``'sample'``.

    Parameters
    ----------
    method : str or tuple of (str, str)
        The method specification to validate.

    Raises
    ------
    ValueError
        If the method is not a valid string or 2-tuple of valid strings.
    """
    if isinstance(method, str):
        if method not in _VALID_DENOISE_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_DENOISE_METHODS} or a "
                f"2-tuple thereof, got '{method}'"
            )
    elif isinstance(method, tuple):
        if len(method) != 2:
            raise ValueError(
                f"method tuple must have exactly 2 elements, "
                f"got {len(method)}"
            )
        for i, m in enumerate(method):
            if m not in _VALID_DENOISE_METHODS:
                raise ValueError(
                    f"method[{i}] must be one of {_VALID_DENOISE_METHODS}, "
                    f"got '{m}'"
                )
    else:
        raise ValueError(
            f"method must be a string or a 2-tuple of strings, "
            f"got {type(method).__name__}"
        )


def _method_needs_rng(method: Union[str, Tuple[str, str]]) -> bool:
    """Return True if any element of ``method`` requires an RNG key."""
    if isinstance(method, str):
        return method == "sample"
    return "sample" in method


def denoise_counts(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray] = None,
    gate: Optional[jnp.ndarray] = None,
    method: Union[str, Tuple[str, str]] = "mean",
    rng_key: Optional[random.PRNGKey] = None,
    return_variance: bool = False,
    mixing_weights: Optional[jnp.ndarray] = None,
    component_assignment: Optional[jnp.ndarray] = None,
    cell_batch_size: Optional[int] = None,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise observed counts using the Bayesian posterior of true transcripts.

    Given observed UMI counts and posterior parameter estimates, computes
    the posterior distribution of the true (pre-capture, pre-dropout)
    transcript counts for each cell and gene.  The derivation exploits
    Poisson-Gamma conjugacy and the Poisson thinning property; see
    ``paper/_denoising.qmd`` for the full mathematics.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed UMI count matrix of shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion (total_count) parameter in canonical form.

        * Standard model, single param set: ``(n_genes,)``.
        * Standard model, multi-sample: ``(n_samples, n_genes)``.
        * Mixture model, single: ``(n_components, n_genes)``.
        * Mixture model, multi-sample: ``(n_samples, n_components, n_genes)``.
    p : jnp.ndarray
        Success probability (numpyro probs convention, *not* the paper's p).

        * Single param set: scalar or ``(n_components,)``.
        * Multi-sample: ``(n_samples,)`` or ``(n_samples, n_components)``.
    p_capture : jnp.ndarray or None, optional
        Per-cell capture probability :math:`\\nu_c`.  Shape ``(n_cells,)``
        for a single param set or ``(n_samples, n_cells)`` for multi-sample.
        ``None`` for models without variable capture probability (nbdm, zinb),
        which is equivalent to :math:`\\nu_c = 1` (perfect capture).
    gate : jnp.ndarray or None, optional
        Zero-inflation gate probability.  Shape ``(n_genes,)`` or
        ``(n_components, n_genes)`` for a single param set; with a leading
        ``n_samples`` dimension for multi-sample.  ``None`` for models
        without zero-inflation.
    method : str or tuple of (str, str), optional
        Summary statistic to return.  Accepts either a single string
        applied uniformly to all positions, or a tuple
        ``(general_method, zi_zero_method)`` for independent control:

        * ``general_method``: used for non-zero positions and for all
          positions in non-ZINB models (no gate).
        * ``zi_zero_method``: used exclusively for zero positions in
          ZINB models (the gate/NB mixture posterior).

        Valid values for each element:

        * ``'mean'``: closed-form posterior mean (shrinkage estimator).
        * ``'mode'``: posterior mode (MAP denoised count).
        * ``'sample'``: one stochastic draw from the denoised posterior.

        A single string ``s`` is equivalent to ``(s, s)``.
        Default: ``'mean'``.
    rng_key : random.PRNGKey or None, optional
        JAX PRNG key.  Required when any element of ``method`` is
        ``'sample'``.
    return_variance : bool, optional
        If ``True``, return a dictionary with keys ``'denoised_counts'``
        and ``'variance'`` instead of a plain array.  Default: ``False``.
    mixing_weights : jnp.ndarray or None, optional
        Component mixing weights for mixture models.  Shape
        ``(n_components,)`` or ``(n_samples, n_components)``.
    component_assignment : jnp.ndarray or None, optional
        Pre-computed per-cell component assignments of shape
        ``(n_cells,)`` (integer indices).  When provided, each cell uses
        its assigned component's parameters instead of marginalising
        over components.  Ignored for non-mixture models.
    cell_batch_size : int or None, optional
        Process cells in batches of this size to limit peak memory.
        ``None`` processes all cells at once.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        If ``return_variance`` is ``False`` (default): denoised count matrix
        with shape ``(n_cells, n_genes)`` (single param set) or
        ``(n_samples, n_cells, n_genes)`` (multi-sample).

        If ``return_variance`` is ``True``: dictionary with keys
        ``'denoised_counts'`` and ``'variance'``, each with the shape above.

    Notes
    -----
    The denoising posterior for uncaptured transcripts is:

    .. math::

        d_g \\mid u_g \\sim \\text{NB}\\!\\left(r_g + u_g,\\;
        \\nu_c + (1-\\nu_c)(1-p)\\right)

    where :math:`p` is the paper's success probability
    (= ``1 - canonical_p``).  The posterior mean simplifies to:

    .. math::

        \\mathbb{E}[m_g \\mid u_g] =
        \\frac{u_g + r_g \\, p_{\\text{can}} (1-\\nu_c)}
        {1 - p_{\\text{can}} (1-\\nu_c)}

    For ZINB models, zero observations use a mixture posterior
    weighted by the probability that the zero came from the gate.

    See Also
    --------
    sample_biological_nb : Biological PPC (samples from NB prior, not
        conditioned on observed counts).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from scribe.sampling import denoise_counts
    >>> counts = jnp.array([[5, 0, 3], [0, 2, 0]])
    >>> r = jnp.array([2.0, 1.5, 3.0])
    >>> p = jnp.float32(0.6)
    >>> nu = jnp.array([0.5, 0.7])
    >>> denoised = denoise_counts(counts, r, p, p_capture=nu)
    >>> denoised.shape
    (2, 3)

    Tuple method for independent control of ZINB zeros:

    >>> denoised = denoise_counts(
    ...     counts, r, p, p_capture=nu,
    ...     gate=jnp.array([0.2, 0.3, 0.1]),
    ...     method=("mean", "sample"),
    ... )
    """
    _validate_denoise_method(method)
    if _method_needs_rng(method) and rng_key is None:
        rng_key = random.PRNGKey(42)

    is_mixture = mixing_weights is not None

    # Detect leading sample dimension
    has_sample_dim = (is_mixture and r.ndim == 3) or (
        not is_mixture and r.ndim == 2
    )

    if not has_sample_dim:
        return _denoise_single(
            counts=counts,
            r=r,
            p=p,
            p_capture=p_capture,
            gate=gate,
            method=method,
            rng_key=rng_key,
            return_variance=return_variance,
            mixing_weights=mixing_weights,
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
        )

    # Multi-sample path: iterate over posterior draws
    n_samples = r.shape[0]
    keys = (
        random.split(rng_key, n_samples)
        if _method_needs_rng(method)
        else [None] * n_samples
    )

    result_list: List[jnp.ndarray] = []
    var_list: List[jnp.ndarray] = []

    for s in range(n_samples):
        r_s = r[s]
        p_s = p[s] if p.ndim >= 1 and p.shape[0] == n_samples else p
        pc_s = (
            p_capture[s]
            if p_capture is not None and p_capture.ndim == 2
            else p_capture
        )
        g_s = (
            gate[s]
            if gate is not None and gate.ndim > (1 if not is_mixture else 2)
            else gate
        )
        mw_s = (
            mixing_weights[s]
            if mixing_weights is not None and mixing_weights.ndim == 2
            else mixing_weights
        )

        out = _denoise_single(
            counts=counts,
            r=r_s,
            p=p_s,
            p_capture=pc_s,
            gate=g_s,
            method=method,
            rng_key=keys[s],
            return_variance=return_variance,
            mixing_weights=mw_s,
            component_assignment=component_assignment,
            cell_batch_size=cell_batch_size,
        )

        if return_variance:
            result_list.append(out["denoised_counts"])
            var_list.append(out["variance"])
        else:
            result_list.append(out)

    stacked = jnp.stack(result_list, axis=0)
    if return_variance:
        return {
            "denoised_counts": stacked,
            "variance": jnp.stack(var_list, axis=0),
        }
    return stacked


# ------------------------------------------------------------------------------


def _denoise_single(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: Optional[jnp.ndarray],
    component_assignment: Optional[jnp.ndarray],
    cell_batch_size: Optional[int],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Dispatch denoising for a single set of parameters.

    Handles both standard and mixture models.  For mixture models with
    ``component_assignment`` provided, gathers per-cell parameters and
    delegates to the standard path.  Otherwise marginalises over components.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts, shape ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion.  ``(n_genes,)`` for standard, ``(n_components, n_genes)``
        for mixture.
    p : jnp.ndarray
        Success probability.  Scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability per cell, ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability.  ``(n_genes,)`` or ``(n_components, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string or ``(general, zi_zeros)``
        tuple; see :func:`denoise_counts`.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).
    return_variance : bool
        Whether to include variance in the output.
    mixing_weights : jnp.ndarray or None
        Component weights ``(n_components,)`` for mixture models.
    component_assignment : jnp.ndarray or None
        Per-cell component indices ``(n_cells,)`` for mixture models.
    cell_batch_size : int or None
        Batch cells to limit memory.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    is_mixture = mixing_weights is not None

    if is_mixture and component_assignment is not None:
        # Gather per-cell parameters from assigned components
        r_cell = r[component_assignment]  # (n_cells, n_genes)
        p_is_comp = p.ndim >= 1 and p.shape[0] == r.shape[0]
        p_cell = p[component_assignment] if p_is_comp else p
        # When p was (n_components,), gathering yields (n_cells,).
        # Reshape to (n_cells, 1) to disambiguate from gene-specific
        # p of shape (n_genes,) in downstream broadcasting.
        if p_cell.ndim == 1 and p_cell.shape[0] == counts.shape[0]:
            p_cell = p_cell[:, None]
        g_cell = (
            gate[component_assignment]
            if gate is not None and gate.ndim == 2
            else gate
        )
        return _denoise_standard(
            counts, r_cell, p_cell, p_capture, g_cell,
            method, rng_key, return_variance, cell_batch_size,
        )

    if is_mixture and component_assignment is None:
        return _denoise_mixture_marginal(
            counts, r, p, p_capture, gate, method, rng_key,
            return_variance, mixing_weights, cell_batch_size,
        )

    # Standard (non-mixture) model
    return _denoise_standard(
        counts, r, p, p_capture, gate,
        method, rng_key, return_variance, cell_batch_size,
    )


# ------------------------------------------------------------------------------


def _denoise_standard(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    cell_batch_size: Optional[int],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise counts for a standard (non-mixture) model, single param set.

    Implements the core denoising formulas from ``paper/_denoising.qmd``.

    The key quantity is ``probs_post`` = canonical_p * (1 - nu_c), the
    numpyro-convention success probability of the posterior NB for the
    uncaptured transcripts d_g.  When ``p_capture`` is ``None`` (no VCP),
    ``probs_post = 0`` and denoising reduces to identity (plus any gate
    correction at zeros).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_genes,)`` or ``(n_cells, n_genes)`` when gathered
        from mixture component assignments.
    p : jnp.ndarray
        Success probability (scalar or broadcastable).
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability ``(n_genes,)`` or ``(n_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string or ``(general, zi_zeros)``
        tuple; see :func:`denoise_counts`.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).
    return_variance : bool
        Whether to return variance alongside denoised counts.
    cell_batch_size : int or None
        Optional cell batching.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    n_cells, n_genes = counts.shape

    if cell_batch_size is None:
        cell_batch_size = n_cells

    needs_rng = _method_needs_rng(method)

    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    denoised_parts: List[jnp.ndarray] = []
    variance_parts: List[jnp.ndarray] = []

    for b in range(n_batches):
        start = b * cell_batch_size
        end = min(start + cell_batch_size, n_cells)
        counts_b = counts[start:end]

        # Slice cell-specific params
        pc_b = (
            p_capture[start:end] if p_capture is not None else None
        )

        # r may be (n_genes,) or (n_cells, n_genes) [component-gathered]
        r_b = r[start:end] if r.ndim == 2 else r

        # gate may be (n_genes,) or (n_cells, n_genes)
        gate_b = (
            gate[start:end] if gate is not None and gate.ndim == 2 else gate
        )

        # After the reshape in _denoise_single, per-cell p is always 2D:
        # (n_cells, 1) or (n_cells, n_genes).  Gene-specific p is 1D
        # (n_genes,) and scalar p is 0D — neither should be sliced.
        p_b = (
            p[start:end]
            if p.ndim >= 2 and p.shape[0] == n_cells
            else p
        )

        if needs_rng:
            rng_key, batch_key = random.split(rng_key)
        else:
            batch_key = None

        d, v = _denoise_batch(
            counts_b, r_b, p_b, pc_b, gate_b, method, batch_key,
        )
        denoised_parts.append(d)
        if return_variance:
            variance_parts.append(v)

    denoised = jnp.concatenate(denoised_parts, axis=0)
    if return_variance:
        variance = jnp.concatenate(variance_parts, axis=0)
        return {"denoised_counts": denoised, "variance": variance}
    return denoised


# ------------------------------------------------------------------------------


def _denoise_batch(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
) -> tuple:
    """Denoise a single batch of cells (no further splitting).

    Returns ``(denoised, variance)`` where ``variance`` is always computed
    (the caller decides whether to keep it).

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts for this batch, ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Success probability: scalar ``()``, gene-specific ``(n_genes,)``,
        or per-cell ``(batch_cells, 1)`` / ``(batch_cells, n_genes)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(batch_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)`` or
        ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string applies uniformly; a tuple
        ``(general_method, zi_zero_method)`` allows the ZINB zero
        correction to use a different method from the rest.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).

    Returns
    -------
    denoised : jnp.ndarray
        Denoised counts ``(batch_cells, n_genes)``.
    variance : jnp.ndarray
        Posterior variance ``(batch_cells, n_genes)``.
    """
    # Normalize method to (general_method, zi_zero_method)
    if isinstance(method, str):
        general_method, zi_zero_method = method, method
    else:
        general_method, zi_zero_method = method

    # Per-cell p arrives as (batch_cells, 1) from the gathering step in
    # _denoise_single / _denoise_mixture_marginal, gene-specific p as
    # (n_genes,), and scalar p as ().  All broadcast correctly with
    # (batch_cells, n_genes) tensors without further reshaping.
    p_eff = p

    # probs_post is the numpyro probs for the posterior NB of uncaptured
    # transcripts d_g.  probs_post = canonical_p * (1 - nu_c).
    # When no VCP (nu_c = 1): probs_post = 0 → d_g = 0 → identity.
    if p_capture is not None:
        nu = p_capture[:, None]  # (batch_cells, 1)
        probs_post = p_eff * (1.0 - nu)
    else:
        probs_post = jnp.zeros(())

    # Complement: 1 - probs_post = p'_paper (the posterior "success" prob
    # in the paper's convention).  Used as denominator in most formulas.
    one_minus_pp = 1.0 - probs_post

    # ------------------------------------------------------------------
    # NB denoising applied to every position (uses general_method)
    # ------------------------------------------------------------------
    if general_method == "mean":
        denoised_nb = (counts + r * probs_post) / one_minus_pp
    elif general_method == "mode":
        alpha = r + counts
        # Mode of NB(alpha, p'): floor((alpha-1) * probs / (1-probs))
        # when alpha > 1, else 0.
        d_mode = jnp.floor(
            jnp.maximum(alpha - 1.0, 0.0) * probs_post / one_minus_pp
        )
        denoised_nb = counts + d_mode
    else:
        # general_method == "sample"
        alpha = r + counts
        key_nb, rng_key = random.split(rng_key)
        d_sample = dist.NegativeBinomialProbs(
            total_count=alpha, probs=probs_post
        ).sample(key_nb)
        denoised_nb = counts + d_sample

    # Variance of d_g | u_g is NB variance: alpha * probs / (1-probs)^2
    var_nb = (r + counts) * probs_post / one_minus_pp**2

    # ------------------------------------------------------------------
    # ZINB zero correction: when gate is present and u_g = 0, the
    # denoised posterior is a mixture of gate and NB pathways.
    # Uses zi_zero_method for the zero positions.
    # ------------------------------------------------------------------
    if gate is not None:
        is_zero = counts == 0

        # Gate weight w = P(gate fired | u=0)
        w = _compute_gate_weight(gate, r, p_eff, one_minus_pp)

        # Gate pathway: the cell was expressing normally but dropout
        # prevented observation.  Denoised count follows the prior NB(r, p).
        if zi_zero_method == "mean":
            gate_val = r * p_eff / (1.0 - p_eff)
        elif zi_zero_method == "mode":
            gate_val = jnp.floor(
                jnp.maximum(r - 1.0, 0.0) * p_eff / (1.0 - p_eff)
            )
        else:
            key_gate, rng_key = random.split(rng_key)
            gate_val = dist.NegativeBinomialProbs(
                total_count=r, probs=p_eff
            ).sample(key_gate)

        # NB pathway value at u=0: the posterior for unobserved mRNA
        # given that the NB component produced the zero.  For VCP models
        # this is positive (capture loss hides real expression); without
        # VCP, probs_post=0 so nb_zero_val collapses to 0.
        if zi_zero_method == general_method:
            nb_zero_val = denoised_nb
        elif zi_zero_method == "mean":
            nb_zero_val = (counts + r * probs_post) / one_minus_pp
        elif zi_zero_method == "mode":
            alpha_z = r + counts
            d_mode_z = jnp.floor(
                jnp.maximum(alpha_z - 1.0, 0.0)
                * probs_post / one_minus_pp
            )
            nb_zero_val = counts + d_mode_z
        else:
            # Sample from the NB posterior for unobserved transcripts
            # at u=0.  For VCP, d ~ NB(r, probs_post) gives the mRNA
            # lost to capture.  Without VCP probs_post=0 → d=0.
            alpha_z = r + counts
            key_nb_z, rng_key = random.split(rng_key)
            d_sample_z = dist.NegativeBinomialProbs(
                total_count=alpha_z, probs=probs_post
            ).sample(key_nb_z)
            nb_zero_val = counts + d_sample_z

        # Combine gate and NB pathways at zero positions
        if zi_zero_method == "mean":
            zinb_zero = w * gate_val + (1.0 - w) * nb_zero_val
        elif zi_zero_method == "mode":
            zinb_zero = jnp.where(w > 0.5, gate_val, nb_zero_val)
        else:
            # Sample: use w to decide whether the zero was from dropout.
            # If gate fired (dropout), sample a replacement from the
            # biological prior NB(r, p).  If genuine NB zero, use the
            # NB posterior (accounts for mRNA lost to capture in VCP;
            # collapses to 0 without VCP since probs_post=0).
            key_bern, rng_key = random.split(rng_key)
            chose_gate = (
                dist.Bernoulli(probs=w).sample(key_bern).astype(bool)
            )
            zinb_zero = jnp.where(chose_gate, gate_val, nb_zero_val)

        denoised = jnp.where(is_zero, zinb_zero, denoised_nb)

        # Variance at zero positions: law of total variance for the mixture
        var_gate = r * p_eff / (1.0 - p_eff) ** 2
        var_nb_zero = var_nb  # already correct at u=0 positions
        mean_gate = r * p_eff / (1.0 - p_eff)
        mean_nb_zero = (r * probs_post) / one_minus_pp
        mixture_var = (
            w * var_gate
            + (1.0 - w) * var_nb_zero
            + w * (1.0 - w) * (mean_gate - mean_nb_zero) ** 2
        )
        variance = jnp.where(is_zero, mixture_var, var_nb)
    else:
        denoised = denoised_nb
        variance = var_nb

    return denoised, variance


# ------------------------------------------------------------------------------


def _compute_gate_weight(
    gate: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    one_minus_probs_post: jnp.ndarray,
) -> jnp.ndarray:
    """Posterior probability that a zero observation came from the gate.

    Implements Bayes' rule for the zero-inflation mixture:

    .. math::

        w = \\frac{g}{g + (1-g)\\,\\hat{p}_{\\text{paper}}^{\\,r_g}}

    where :math:`\\hat{p}_{\\text{paper}} = (1 - p_{\\text{can}}) /
    (1 - p_{\\text{can}}(1-\\nu_c))`.

    In the numpyro probs convention, the NB probability of observing zero
    is :math:`(1 - \\text{probs})^r`.  For the *observation* model the
    relevant probs is ``p_hat_numpyro``, and its complement is exactly
    ``(1 - canonical_p) / one_minus_probs_post``.

    Parameters
    ----------
    gate : jnp.ndarray
        Gate probability, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion, ``(n_genes,)`` or ``(batch_cells, n_genes)``.
    p : jnp.ndarray
        Canonical success probability (scalar or broadcastable).
    one_minus_probs_post : jnp.ndarray
        ``1 - probs_post``, the paper's :math:`p'`.  Shape ``()`` or
        ``(batch_cells, 1)``.

    Returns
    -------
    jnp.ndarray
        Gate weight *w* with the same shape as ``gate`` (broadcast).
    """
    # p_hat_paper = (1 - canonical_p) / one_minus_probs_post
    # P_NB(u=0) = p_hat_paper^r  in the paper convention
    p_hat_paper = (1.0 - p) / one_minus_probs_post
    nb_zero_prob = p_hat_paper ** r

    w = gate / (gate + (1.0 - gate) * nb_zero_prob)
    return w


# ------------------------------------------------------------------------------


def _denoise_mixture_marginal(
    counts: jnp.ndarray,
    r: jnp.ndarray,
    p: jnp.ndarray,
    p_capture: Optional[jnp.ndarray],
    gate: Optional[jnp.ndarray],
    method: Union[str, Tuple[str, str]],
    rng_key: Optional[random.PRNGKey],
    return_variance: bool,
    mixing_weights: jnp.ndarray,
    cell_batch_size: Optional[int],
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Denoise by marginalising over mixture components.

    For ``method='mean'`` (or general_method ``'mean'``):

    .. math::

        \\mathbb{E}[m_g \\mid u_g] = \\sum_k w_k \\,
        \\mathbb{E}[m_g \\mid u_g, \\text{comp}=k]

    For ``method='sample'`` (or general_method ``'sample'``): sample a
    component per cell from ``mixing_weights``, then sample from that
    component's denoised posterior.

    Parameters
    ----------
    counts : jnp.ndarray
        Observed counts ``(n_cells, n_genes)``.
    r : jnp.ndarray
        Dispersion ``(n_components, n_genes)``.
    p : jnp.ndarray
        Success probability, scalar or ``(n_components,)``.
    p_capture : jnp.ndarray or None
        Capture probability ``(n_cells,)`` or ``None``.
    gate : jnp.ndarray or None
        Gate ``(n_genes,)`` or ``(n_components, n_genes)`` or ``None``.
    method : str or tuple of (str, str)
        Denoising method.  A single string or ``(general, zi_zeros)``
        tuple; see :func:`denoise_counts`.  The general_method controls
        whether components are sampled or marginalised.
    rng_key : random.PRNGKey or None
        PRNG key (needed when any element of ``method`` is ``'sample'``).
    return_variance : bool
        Whether to return variance.
    mixing_weights : jnp.ndarray
        Component weights ``(n_components,)``.
    cell_batch_size : int or None
        Cell batching.

    Returns
    -------
    jnp.ndarray or Dict[str, jnp.ndarray]
        Denoised counts (and optionally variance).
    """
    # Extract general_method to decide marginalisation vs sampling path
    general_method = method[0] if isinstance(method, tuple) else method

    n_components = r.shape[0]
    p_is_comp = p.ndim >= 1 and p.shape[0] == n_components

    if general_method == "sample":
        # Sample component per cell, gather per-cell params, then use
        # the standard (non-mixture) path.
        key_comp, key_rest = random.split(rng_key)
        comp = dist.Categorical(probs=mixing_weights).sample(
            key_comp, (counts.shape[0],)
        )
        r_cell = r[comp]  # (n_cells, n_genes)
        p_cell = p[comp] if p_is_comp else p
        # Disambiguate per-cell p from gene-specific p (see _denoise_single)
        if p_cell.ndim == 1 and p_cell.shape[0] == counts.shape[0]:
            p_cell = p_cell[:, None]
        g_cell = (
            gate[comp] if gate is not None and gate.ndim == 2 else gate
        )
        return _denoise_standard(
            counts, r_cell, p_cell, p_capture, g_cell,
            method, key_rest, return_variance, cell_batch_size,
        )

    # Marginalise over components (mean or mode for the general path).
    # An rng_key may still be needed if zi_zero_method is "sample".
    needs_rng = _method_needs_rng(method)
    n_cells, n_genes = counts.shape
    denoised_acc = jnp.zeros((n_cells, n_genes))
    variance_acc = jnp.zeros((n_cells, n_genes))

    for k in range(n_components):
        r_k = r[k]
        p_k = p[k] if p_is_comp else p
        g_k = (
            gate[k] if gate is not None and gate.ndim == 2 else gate
        )

        # Split rng_key per component if the zi_zero path needs sampling
        if needs_rng:
            rng_key, comp_key = random.split(rng_key)
        else:
            comp_key = None

        out_k = _denoise_standard(
            counts, r_k, p_k, p_capture, g_k,
            method, comp_key, True, cell_batch_size,
        )

        d_k = out_k["denoised_counts"]
        v_k = out_k["variance"]
        w_k = mixing_weights[k]

        denoised_acc = denoised_acc + w_k * d_k
        # Law of total variance: Var = E[Var_k] + Var[E_k]
        variance_acc = variance_acc + w_k * (v_k + d_k**2)

    variance = variance_acc - denoised_acc**2

    if return_variance:
        return {"denoised_counts": denoised_acc, "variance": variance}
    return denoised_acc
