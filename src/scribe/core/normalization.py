"""
Shared normalization utilities for SCRIBE inference results.

This module provides count normalization functionality that can be used by both
SVI and MCMC inference results classes.

Performance
-----------
Dirichlet sampling is performed in batches of ``batch_size`` posterior samples
(default 256) to balance GPU throughput against memory usage.  See
``core.normalization_logistic`` for the batched sampling helpers.
"""

from typing import Dict, Optional, Union
import jax.numpy as jnp
from jax import random
import warnings

import numpyro.distributions as dist
from ..stats import sample_dirichlet_from_parameters, fit_dirichlet_minka
from ..utils import numpyro_to_scipy

# Re-use the same batched sampling helpers from the logistic-normal module
from .normalization_logistic import (
    _batched_dirichlet_sample,
    _batched_dirichlet_sample_raw,
    _DEFAULT_BATCH_SIZE,
)


def normalize_counts_from_posterior(
    posterior_samples: Dict[str, jnp.ndarray],
    n_components: Optional[int] = None,
    rng_key: Optional[random.PRNGKey] = None,
    n_samples_dirichlet: int = 1,
    fit_distribution: bool = True,
    store_samples: bool = False,
    sample_axis: int = 0,
    return_concentrations: bool = False,
    backend: str = "numpyro",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Normalize counts using posterior samples of the r parameter.

    This function takes posterior samples of the dispersion parameter (r) and
    uses them as concentration parameters for Dirichlet distributions to
    generate normalized expression profiles.  For mixture models,
    normalization is performed per component.

    Parameters
    ----------
    posterior_samples : Dict[str, jnp.ndarray]
        Dictionary containing posterior samples, must include 'r' parameter.
    n_components : Optional[int], default=None
        Number of mixture components. If None, assumes non-mixture model.
    rng_key : random.PRNGKey, optional
        JAX random number generator key. Defaults to random.PRNGKey(42) if
        None.
    n_samples_dirichlet : int, default=1
        Number of samples to draw from each Dirichlet distribution.
    fit_distribution : bool, default=True
        If True, fits a Dirichlet distribution to the generated samples using
        ``fit_dirichlet_minka``.
    store_samples : bool, default=False
        If True, includes the raw Dirichlet samples in the output.
    sample_axis : int, default=0
        Axis containing samples in the Dirichlet fitting (passed to
        ``fit_dirichlet_minka``).
    return_concentrations : bool, default=False
        If True, returns the original r parameter samples used as
        concentrations.
    backend : str, default="numpyro"
        Statistical package to use for distributions when
        ``fit_distribution=True``.  Must be ``"numpyro"`` or ``"scipy"``.
    batch_size : int, default=256
        Number of posterior samples to process in each batched Dirichlet
        sampling call.  Larger values use more GPU memory but require fewer
        Python-to-JAX dispatches.
    verbose : bool, default=True
        If True, prints progress messages.

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Dictionary containing normalized expression profiles.

    Raises
    ------
    ValueError
        If 'r' parameter is not found in posterior_samples.
    """
    # Create default RNG key if not provided (lazy initialization)
    if rng_key is None:
        rng_key = random.PRNGKey(42)

    # Validate inputs
    if "r" not in posterior_samples:
        raise ValueError(
            "'r' parameter not found in posterior_samples. "
            "This method requires posterior samples of the dispersion "
            "parameter. Please run get_posterior_samples() first."
        )

    if backend not in ["scipy", "numpyro"]:
        raise ValueError(
            f"Invalid backend: {backend}. Must be 'scipy' or 'numpyro'"
        )

    # Get r parameter samples
    r_samples = posterior_samples["r"]

    if verbose:
        print(f"Using r parameter samples with shape: {r_samples.shape}")

    # Determine if this is a mixture model
    is_mixture = n_components is not None and n_components > 1

    # Process mixture model
    if is_mixture:
        if verbose:
            print(f"Processing mixture model with {n_components} components")
        assert n_components is not None  # Type assertion for linter
        return _normalize_mixture_model(
            r_samples,
            n_components,
            rng_key,
            n_samples_dirichlet,
            fit_distribution,
            store_samples,
            sample_axis,
            return_concentrations,
            backend,
            batch_size,
            verbose,
        )
    # Process non-mixture model
    else:
        if verbose:
            print("Processing non-mixture model")
        return _normalize_non_mixture_model(
            r_samples,
            rng_key,
            n_samples_dirichlet,
            fit_distribution,
            store_samples,
            sample_axis,
            return_concentrations,
            backend,
            batch_size,
            verbose,
        )


# ------------------------------------------------------------------------------
# Normalization functions
# ------------------------------------------------------------------------------


def _normalize_non_mixture_model(
    r_samples: jnp.ndarray,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    fit_distribution: bool,
    store_samples: bool,
    sample_axis: int,
    return_concentrations: bool,
    backend: str,
    batch_size: int,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Handle normalization for non-mixture models.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (n_posterior_samples, n_genes)
        Posterior samples of concentration parameters.
    rng_key : random.PRNGKey
        JAX PRNG key.
    n_samples_dirichlet : int
        Dirichlet draws per posterior sample.
    fit_distribution : bool
        Whether to fit a summary Dirichlet.
    store_samples : bool
        Whether to return the raw Dirichlet samples.
    sample_axis : int
        Axis for ``fit_dirichlet_minka``.
    return_concentrations : bool
        Whether to return original r samples.
    backend : str
        ``"numpyro"`` or ``"scipy"``.
    batch_size : int
        Posterior samples per batched JAX call.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Normalized expression results.
    """
    n_posterior_samples, n_genes = r_samples.shape

    if verbose:
        n_batches = (n_posterior_samples + batch_size - 1) // batch_size
        print(
            f"Generating {n_samples_dirichlet} Dirichlet sample(s) for each "
            f"of {n_posterior_samples} posterior samples "
            f"(batch_size={batch_size}, {n_batches} batches)"
        )

    # ------------------------------------------------------------------
    # Batched Dirichlet sampling  (replaces per-sample Python loop)
    # ------------------------------------------------------------------
    # _batched_dirichlet_sample_raw preserves the original per-posterior-sample
    # shape so that store_samples=True returns the expected output layout.
    dirichlet_samples = _batched_dirichlet_sample_raw(
        r_samples,
        n_samples_dirichlet=n_samples_dirichlet,
        rng_key=rng_key,
        batch_size=batch_size,
        verbose=verbose,
    )
    # dirichlet_samples shape:
    #   n_samples_dirichlet == 1  →  (N, D)
    #   n_samples_dirichlet  > 1  →  (N, D, S)

    # Initialize results dictionary
    results = {}

    # Store samples if requested
    if store_samples:
        results["samples"] = dirichlet_samples

    # Fit distribution if requested
    if fit_distribution:
        if verbose:
            print("Fitting Dirichlet distributions to samples")

        if n_samples_dirichlet == 1:
            # Already (N, D) — use directly for fitting
            all_samples = dirichlet_samples
        else:
            # (N, D, S) → (N, S, D) → (N*S, D)
            all_samples = dirichlet_samples.transpose(0, 2, 1).reshape(
                -1, n_genes
            )

        # Fit single Dirichlet distribution to all pooled samples
        fitted_concentrations = fit_dirichlet_minka(
            all_samples, sample_axis=sample_axis
        )

        # Store concentrations
        results["concentrations"] = fitted_concentrations

        # Compute mean probabilities (Dirichlet mean)
        concentration_sum = jnp.sum(fitted_concentrations)
        results["mean_probabilities"] = (
            fitted_concentrations / concentration_sum
        )

        # Create distribution object
        if verbose:
            print(
                f"Creating Dirichlet distribution object with {backend} "
                "backend"
            )
        dirichlet_dist = dist.Dirichlet(fitted_concentrations)
        if backend == "scipy":
            dirichlet_dist = numpyro_to_scipy(dirichlet_dist)
        results["distributions"] = dirichlet_dist

    # Return original concentrations if requested
    if return_concentrations:
        results["original_concentrations"] = r_samples

    return results


# ------------------------------------------------------------------------------


def _normalize_mixture_model(
    r_samples: jnp.ndarray,
    n_components: int,
    rng_key: random.PRNGKey,
    n_samples_dirichlet: int,
    fit_distribution: bool,
    store_samples: bool,
    sample_axis: int,
    return_concentrations: bool,
    backend: str,
    batch_size: int,
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Handle normalization for mixture models.

    Parameters
    ----------
    r_samples : jnp.ndarray, shape (n_posterior_samples, n_components, n_genes)
        Posterior samples of concentration parameters per component.
    n_components : int
        Number of mixture components.
    rng_key : random.PRNGKey
        JAX PRNG key.
    n_samples_dirichlet : int
        Dirichlet draws per posterior sample.
    fit_distribution : bool
        Whether to fit a summary Dirichlet per component.
    store_samples : bool
        Whether to return raw Dirichlet samples.
    sample_axis : int
        Axis for ``fit_dirichlet_minka``.
    return_concentrations : bool
        Whether to return original r samples.
    backend : str
        ``"numpyro"`` or ``"scipy"``.
    batch_size : int
        Posterior samples per batched JAX call.
    verbose : bool
        Print progress messages.

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Normalized expression results per component.
    """
    n_posterior_samples, n_components_check, n_genes = r_samples.shape

    if n_components_check != n_components:
        raise ValueError(
            f"Mismatch between n_components ({n_components}) and "
            f"r_samples shape ({r_samples.shape})"
        )

    if verbose:
        n_batches = (n_posterior_samples + batch_size - 1) // batch_size
        print(
            f"Generating {n_samples_dirichlet} Dirichlet sample(s) for each "
            f"of {n_posterior_samples} posterior samples and {n_components} "
            f"components (batch_size={batch_size}, {n_batches} batches/comp)"
        )

    # ------------------------------------------------------------------
    # Batched Dirichlet sampling per component
    # ------------------------------------------------------------------
    # Build per-component sample arrays, then stack.
    component_samples_list = []
    for c in range(n_components):
        # Slice concentrations for component c: (n_posterior, n_genes)
        r_component = r_samples[:, c, :]

        # Per-component sub-key for reproducibility
        key_c = random.fold_in(rng_key, c)

        # Batched sampling preserving per-posterior-sample structure
        comp_samples = _batched_dirichlet_sample_raw(
            r_component,
            n_samples_dirichlet=n_samples_dirichlet,
            rng_key=key_c,
            batch_size=batch_size,
            verbose=verbose and c == 0,  # progress bar for first component
        )
        # comp_samples shape:
        #   n_samples_dirichlet == 1  →  (N, D)
        #   n_samples_dirichlet  > 1  →  (N, D, S)
        component_samples_list.append(comp_samples)

    # Stack into (N, K, D) or (N, K, D, S)
    if n_samples_dirichlet == 1:
        dirichlet_samples = jnp.stack(component_samples_list, axis=1)
    else:
        dirichlet_samples = jnp.stack(component_samples_list, axis=1)

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------
    results = {}

    # Store samples if requested
    if store_samples:
        results["samples"] = dirichlet_samples

    # Fit distribution if requested
    if fit_distribution:
        if verbose:
            print("Fitting Dirichlet distributions to samples")

        # Initialise concentrations for each component
        concentrations = jnp.zeros((n_components, n_genes))

        for c in range(n_components):
            if verbose:
                print(f"    Fitting single Dirichlet for component {c}")

            if n_samples_dirichlet == 1:
                # (N, D) slice for this component
                component_samples = dirichlet_samples[:, c, :]
            else:
                # (N, D, S) → (N, S, D) → (N*S, D)
                component_samples = (
                    dirichlet_samples[:, c, :, :]
                    .transpose(0, 2, 1)
                    .reshape(-1, n_genes)
                )

            fitted_concentrations = fit_dirichlet_minka(
                component_samples, sample_axis=sample_axis
            )
            concentrations = concentrations.at[c].set(fitted_concentrations)

        # Store concentrations
        results["concentrations"] = concentrations

        # Compute mean probabilities (Dirichlet mean)
        concentration_sums = jnp.sum(concentrations, axis=1, keepdims=True)
        results["mean_probabilities"] = concentrations / concentration_sums

        # Create distribution objects per component
        if verbose:
            print(
                f"Creating Dirichlet distribution objects per component "
                f"with {backend} backend"
            )

        distributions = []
        for c in range(n_components):
            dirichlet_dist = dist.Dirichlet(concentrations[c])
            if backend == "scipy":
                dirichlet_dist = numpyro_to_scipy(dirichlet_dist)
            distributions.append(dirichlet_dist)
        results["distributions"] = distributions

    # Return original concentrations if requested
    if return_concentrations:
        results["original_concentrations"] = r_samples

    return results
