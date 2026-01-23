"""
Shared normalization utilities for SCRIBE inference results.

This module provides count normalization functionality that can be used by both
SVI and MCMC inference results classes.
"""

from typing import Dict, Optional, Union
import jax.numpy as jnp
from jax import random
import warnings

import numpyro.distributions as dist
from ..stats import sample_dirichlet_from_parameters, fit_dirichlet_minka
from ..utils import numpyro_to_scipy


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
    verbose: bool = True,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """
    Normalize counts using posterior samples of the r parameter.

    This function takes posterior samples of the dispersion parameter (r) and
    uses them as concentration parameters for Dirichlet distributions to
    generate normalized expression profiles. For mixture models, normalization
    is performed per component.

    Parameters
    ----------
    posterior_samples : Dict[str, jnp.ndarray]
        Dictionary containing posterior samples, must include 'r' parameter
    n_components : Optional[int], default=None
        Number of mixture components. If None, assumes non-mixture model.
    rng_key : random.PRNGKey, optional
        JAX random number generator key. Defaults to random.PRNGKey(42) if None
    n_samples_dirichlet : int, default=1000
        Number of samples to draw from each Dirichlet distribution
    fit_distribution : bool, default=True
        If True, fits a Dirichlet distribution to the generated samples using
        fit_dirichlet_minka
    store_samples : bool, default=False
        If True, includes the raw Dirichlet samples in the output
    sample_axis : int, default=0
        Axis containing samples in the Dirichlet fitting (passed to
        fit_dirichlet_minka)
    return_concentrations : bool, default=False
        If True, returns the original r parameter samples used as concentrations
    backend : str, default="numpyro"
        Statistical package to use for distributions when fit_distribution=True.
        Must be one of: - "numpyro": Returns numpyro.distributions.Dirichlet
        objects - "scipy": Returns scipy.stats distributions via
        numpyro_to_scipy conversion
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    Dict[str, Union[jnp.ndarray, object]]
        Dictionary containing normalized expression profiles. Keys depend on
        input arguments:
            - 'samples': Raw Dirichlet samples (if store_samples=True)
            - 'concentrations': Fitted concentration parameters (if
              fit_distribution=True)
            - 'mean_probabilities': Mean probabilities from fitted distribution
              (if fit_distribution=True)
            - 'distributions': Dirichlet distribution objects (if
              fit_distribution=True)
            - 'original_concentrations': Original r parameter samples (if
              return_concentrations=True)

        For non-mixture models:
            - samples: shape (n_posterior_samples, n_genes, n_samples_dirichlet)
              or (n_posterior_samples, n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_genes,) - single fitted distribution
            - mean_probabilities: shape (n_genes,) - single fitted distribution
            - distributions: single Dirichlet distribution object

        For mixture models:
            - samples: shape (n_posterior_samples, n_components, n_genes,
              n_samples_dirichlet) or (n_posterior_samples, n_components,
              n_genes) if n_samples_dirichlet=1
            - concentrations: shape (n_components, n_genes) - one fitted
              distribution per component
            - mean_probabilities: shape (n_components, n_genes) - one fitted
              distribution per component
            - distributions: list of n_components Dirichlet distribution objects

    Raises
    ------
    ValueError
        If 'r' parameter is not found in posterior_samples
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
        # n_components is guaranteed to be not None here due to the is_mixture
        # condition
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
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """Handle normalization for non-mixture models."""
    # r_samples shape: (n_posterior_samples, n_genes)
    n_posterior_samples, n_genes = r_samples.shape

    if verbose:
        print(
            f"Generating {n_samples_dirichlet} Dirichlet samples for each of "
            f"{n_posterior_samples} posterior samples"
        )

    # Initialize results dictionary
    results = {}

    # Generate Dirichlet samples for each posterior sample
    if n_samples_dirichlet == 1:
        # Single sample case - more efficient
        dirichlet_samples = jnp.zeros((n_posterior_samples, n_genes))
        for i in range(n_posterior_samples):
            if verbose and i % 100 == 0:
                print(
                    f"    Processing posterior sample {i}/{n_posterior_samples}"
                )

            # Use r values as concentration parameters
            key_i = random.fold_in(rng_key, i)
            sample_i = sample_dirichlet_from_parameters(
                r_samples[i : i + 1], n_samples_dirichlet=1, rng_key=key_i
            )
            dirichlet_samples = dirichlet_samples.at[i].set(sample_i[0])
    else:
        # Multiple samples case
        dirichlet_samples = jnp.zeros(
            (n_posterior_samples, n_genes, n_samples_dirichlet)
        )
        for i in range(n_posterior_samples):
            if verbose and i % 100 == 0:
                print(
                    f"    Processing posterior sample {i}/{n_posterior_samples}"
                )

            # Use r values as concentration parameters
            key_i = random.fold_in(rng_key, i)
            sample_i = sample_dirichlet_from_parameters(
                r_samples[i : i + 1],
                n_samples_dirichlet=n_samples_dirichlet,
                rng_key=key_i,
            )
            dirichlet_samples = dirichlet_samples.at[i].set(sample_i[0])

    # Store samples if requested
    if store_samples:
        results["samples"] = dirichlet_samples

    # Fit distribution if requested
    if fit_distribution:
        if verbose:
            print("Fitting Dirichlet distributions to samples")

        if n_samples_dirichlet == 1:
            all_samples = dirichlet_samples
        else:
            # Reshape samples: (n_posterior_samples, n_genes,
            # n_samples_dirichlet) -> (n_posterior_samples *
            # n_samples_dirichlet, n_genes)
            # First transpose to (n_posterior_samples, n_samples_dirichlet,
            # n_genes) so that each row after reshape represents a valid
            # Dirichlet sample
            all_samples = dirichlet_samples.transpose(0, 2, 1).reshape(
                -1, n_genes
            )

        # Fit single Dirichlet distribution to all samples
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

        # Create single distribution object
        if verbose:
            print(
                f"Creating Dirichlet distribution object with {backend} backend"
            )
        # Create numpyro distribution object
        dirichlet_dist = dist.Dirichlet(fitted_concentrations)
        # Check if backend is scipy and convert to scipy distribution
        if backend == "scipy":
            # Convert to scipy distribution
            dirichlet_dist = numpyro_to_scipy(dirichlet_dist)
        # Store distribution object
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
    verbose: bool,
) -> Dict[str, Union[jnp.ndarray, object]]:
    """Handle normalization for mixture models."""
    # r_samples shape: (n_posterior_samples, n_components, n_genes)
    n_posterior_samples, n_components_check, n_genes = r_samples.shape

    if n_components_check != n_components:
        raise ValueError(
            f"Mismatch between n_components ({n_components}) and "
            f"r_samples shape ({r_samples.shape})"
        )

    if verbose:
        print(
            f"Generating {n_samples_dirichlet} Dirichlet samples for each of "
            f"{n_posterior_samples} posterior samples and {n_components} components"
        )

    # Initialize results dictionary
    results = {}

    # Generate Dirichlet samples for each posterior sample and component
    if n_samples_dirichlet == 1:
        # Single sample case - more efficient
        dirichlet_samples = jnp.zeros(
            (n_posterior_samples, n_components, n_genes)
        )
        for i in range(n_posterior_samples):
            if verbose and i % 100 == 0:
                print(
                    f"    Processing posterior sample {i}/{n_posterior_samples}"
                )

            for c in range(n_components):
                # Use r values as concentration parameters for this component
                key_i_c = random.fold_in(rng_key, i * n_components + c)
                sample_i_c = sample_dirichlet_from_parameters(
                    r_samples[i, c : c + 1],
                    n_samples_dirichlet=1,
                    rng_key=key_i_c,
                )
                dirichlet_samples = dirichlet_samples.at[i, c].set(
                    sample_i_c[0]
                )
    else:
        # Multiple samples case
        dirichlet_samples = jnp.zeros(
            (n_posterior_samples, n_components, n_genes, n_samples_dirichlet)
        )
        for i in range(n_posterior_samples):
            if verbose and i % 100 == 0:
                print(
                    f"    Processing posterior sample {i}/{n_posterior_samples}"
                )

            for c in range(n_components):
                # Use r values as concentration parameters for this component
                key_i_c = random.fold_in(rng_key, i * n_components + c)
                sample_i_c = sample_dirichlet_from_parameters(
                    r_samples[i, c : c + 1],
                    n_samples_dirichlet=n_samples_dirichlet,
                    rng_key=key_i_c,
                )
                dirichlet_samples = dirichlet_samples.at[i, c].set(
                    sample_i_c[0]
                )

    # Store samples if requested
    if store_samples:
        results["samples"] = dirichlet_samples

    # Fit distribution if requested
    if fit_distribution:
        if verbose:
            print("Fitting Dirichlet distributions to samples")

        if n_samples_dirichlet == 1:
            # Even with single samples, fit single Dirichlet distribution per
            # component using all posterior samples
            if verbose:
                print(
                    f"    Collecting all {n_posterior_samples} samples per "
                    f"component to fit single Dirichlet per component"
                )

            # Initialize concentrations for each component
            concentrations = jnp.zeros((n_components, n_genes))

            # Fit single Dirichlet distribution per component
            for c in range(n_components):
                if verbose:
                    print(f"    Fitting single Dirichlet for component {c}")

                # Collect all samples for this component from all posterior
                # samples
                # Shape: (n_posterior_samples, n_genes)
                component_samples = dirichlet_samples[:, c, :]

                # Fit single Dirichlet distribution to all samples for this
                # component
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
                    f"Creating single Dirichlet distribution object per "
                    f"component with {backend} backend"
                )

            distributions = []
            for c in range(n_components):
                # Create Dirichlet distribution for this component
                dirichlet_dist = dist.Dirichlet(concentrations[c])
                if backend == "scipy":
                    dirichlet_dist = numpyro_to_scipy(dirichlet_dist)
                distributions.append(dirichlet_dist)
            results["distributions"] = distributions
        else:
            # Fit single Dirichlet distribution per component using all
            # posterior samples
            if verbose:
                print(
                    f"    Collecting all "
                    f"{n_posterior_samples * n_samples_dirichlet} samples per "
                    f"component to fit single Dirichlet per component"
                )

            # Initialize concentrations for each component
            concentrations = jnp.zeros((n_components, n_genes))

            # Fit single Dirichlet distribution per component
            for c in range(n_components):
                if verbose:
                    print(f"    Fitting single Dirichlet for component {c}")

                # Collect all samples for this component from all posterior
                # samples
                # Shape: (n_posterior_samples, n_genes, n_samples_dirichlet)
                # -> (n_posterior_samples * n_samples_dirichlet, n_genes)
                # First transpose to (n_posterior_samples, n_samples_dirichlet,
                # n_genes) so that each row after reshape represents a valid
                # Dirichlet sample
                component_samples = (
                    dirichlet_samples[:, c, :, :]
                    .transpose(0, 2, 1)
                    .reshape(-1, n_genes)
                )

                # Fit single Dirichlet distribution to all samples for this
                # component
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
                    f"Creating single Dirichlet distribution object per "
                    f"component with {backend} backend"
                )

            distributions = []
            for c in range(n_components):
                # Create Dirichlet distribution for this component
                dirichlet_dist = dist.Dirichlet(concentrations[c])
                if backend == "scipy":
                    dirichlet_dist = numpyro_to_scipy(dirichlet_dist)
                distributions.append(dirichlet_dist)
            results["distributions"] = distributions

    # Return original concentrations if requested
    if return_concentrations:
        results["original_concentrations"] = r_samples

    return results
