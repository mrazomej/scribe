"""
Shared normalization utilities for SCRIBE inference results.

This module provides count normalization functionality that can be used by both
SVI and MCMC inference results classes.
"""

from typing import Dict, Optional, Union
import jax.numpy as jnp
from jax import random
import warnings

from ..stats import sample_dirichlet_from_parameters, fit_dirichlet_minka


def normalize_counts_from_posterior(
    posterior_samples: Dict[str, jnp.ndarray],
    n_components: Optional[int] = None,
    rng_key: random.PRNGKey = random.PRNGKey(42),
    n_samples_dirichlet: int = 1,
    fit_distribution: bool = True,
    store_samples: bool = False,
    sample_axis: int = 0,
    return_concentrations: bool = False,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:
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
    rng_key : random.PRNGKey, default=random.PRNGKey(42)
        JAX random number generator key
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
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing normalized expression profiles. Keys depend on
        input arguments: - 'samples': Raw Dirichlet samples (if
        store_samples=True) - 'concentrations': Fitted concentration parameters
        (if fit_distribution=True) - 'mean_probabilities': Mean probabilities
        from fitted distribution (if fit_distribution=True) -
        'original_concentrations': Original r parameter samples (if
        return_concentrations=True)

        For non-mixture models: - samples: shape (n_posterior_samples, n_genes,
        n_samples_dirichlet) or
                  (n_posterior_samples, n_genes) if n_samples_dirichlet=1
        - concentrations: shape (n_posterior_samples, n_genes)
        - mean_probabilities: shape (n_posterior_samples, n_genes)

        For mixture models: - samples: shape (n_posterior_samples, n_components,
        n_genes, n_samples_dirichlet) or
                  (n_posterior_samples, n_components, n_genes) if
                  n_samples_dirichlet=1
        - concentrations: shape (n_posterior_samples, n_components, n_genes)
        - mean_probabilities: shape (n_posterior_samples, n_components, n_genes)

    Raises
    ------
    ValueError
        If 'r' parameter is not found in posterior_samples
    """
    # Validate that r parameter is available
    if "r" not in posterior_samples:
        raise ValueError(
            "'r' parameter not found in posterior_samples. "
            "This method requires posterior samples of the dispersion "
            "parameter. Please run get_posterior_samples() first."
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
    verbose: bool,
) -> Dict[str, jnp.ndarray]:
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
            # For single samples, we can't fit a distribution
            # Instead, return the samples as both concentrations and mean probabilities
            if verbose:
                print(
                    "    Using single samples as both concentrations and mean probabilities"
                )
            results["concentrations"] = dirichlet_samples
            results["mean_probabilities"] = dirichlet_samples
        else:
            # Fit Dirichlet distribution for each posterior sample
            concentrations = jnp.zeros((n_posterior_samples, n_genes))

            for i in range(n_posterior_samples):
                if verbose and i % 100 == 0:
                    print(
                        f"    Fitting Dirichlet for posterior sample {i}/{n_posterior_samples}"
                    )

                # Fit using samples for this posterior sample
                sample_data = dirichlet_samples[
                    i
                ].T  # Shape: (n_samples_dirichlet, n_genes)
                fitted_concentrations = fit_dirichlet_minka(
                    sample_data, sample_axis=sample_axis
                )
                concentrations = concentrations.at[i].set(fitted_concentrations)

            results["concentrations"] = concentrations

            # Compute mean probabilities (Dirichlet mean)
            concentration_sums = jnp.sum(concentrations, axis=1, keepdims=True)
            results["mean_probabilities"] = concentrations / concentration_sums

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
    verbose: bool,
) -> Dict[str, jnp.ndarray]:
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
            # For single samples, we can't fit a distribution
            # Instead, return the samples as both concentrations and mean probabilities
            if verbose:
                print(
                    "    Using single samples as both concentrations and mean probabilities"
                )
            results["concentrations"] = dirichlet_samples
            results["mean_probabilities"] = dirichlet_samples
        else:
            # Fit Dirichlet distribution for each posterior sample and component
            concentrations = jnp.zeros(
                (n_posterior_samples, n_components, n_genes)
            )

            for i in range(n_posterior_samples):
                if verbose and i % 100 == 0:
                    print(
                        f"    Fitting Dirichlet for posterior sample {i}/{n_posterior_samples}"
                    )

                for c in range(n_components):
                    # Fit using samples for this posterior sample and component
                    sample_data = dirichlet_samples[
                        i, c
                    ].T  # Shape: (n_samples_dirichlet, n_genes)
                    fitted_concentrations = fit_dirichlet_minka(
                        sample_data, sample_axis=sample_axis
                    )
                    concentrations = concentrations.at[i, c].set(
                        fitted_concentrations
                    )

            results["concentrations"] = concentrations

            # Compute mean probabilities (Dirichlet mean)
            concentration_sums = jnp.sum(concentrations, axis=2, keepdims=True)
            results["mean_probabilities"] = concentrations / concentration_sums

    # Return original concentrations if requested
    if return_concentrations:
        results["original_concentrations"] = r_samples

    return results
