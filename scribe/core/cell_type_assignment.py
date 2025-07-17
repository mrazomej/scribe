"""
Cell type assignment utilities for SCRIBE.

This module provides unified functions for computing cell type probabilities
that work with both SVI and MCMC results objects.
"""

import jax.numpy as jnp
from jax.nn import softmax
from typing import Dict, Optional, Union, Any

from ..stats import fit_dirichlet_minka, hellinger_gamma, hellinger_lognormal


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------


def temperature_scaling(
    log_probs: jnp.ndarray, temperature: float, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Apply temperature scaling to log probabilities before normalization.

    Temperature scaling modifies the sharpness of probability distributions by
    dividing log probabilities by a temperature parameter T:

        p_i = softmax(log_p_i / T)

    When T < 1, this sharpens the distribution by amplifying differences between
    probabilities. When T > 1, this smooths the distribution by reducing
    differences.

    Mathematically, if p_i are the original probabilities:

        p_i = exp(log_p_i) / sum_j(exp(log_p_j))

    Then temperature scaling gives:

        p_i(T) = exp(log_p_i/T) / sum_j(exp(log_p_j/T))
                = p_i^(1/T) / sum_j(p_j^(1/T))

    As T → 0, this approaches a one-hot distribution.
    As T → ∞, this approaches a uniform distribution.

    Parameters
    ----------
    log_probs : jnp.ndarray
        Array of log probabilities to scale
    temperature : float
        Temperature parameter (lower values increase contrast)
    dtype : jnp.dtype, default=jnp.float32
        Data type for computations

    Returns
    -------
    jnp.ndarray
        Temperature-scaled log probabilities
    """
    return log_probs / jnp.array(temperature, dtype=dtype)


def hellinger_distance_weights(
    params: Dict,
    n_components: int,
    n_genes: int,
    dtype: jnp.dtype = jnp.float32,
    min_distance: float = 1e-5,
    power: float = 2.0,
    normalize: bool = True,
    r_dist: Optional[str] = None,
) -> jnp.ndarray:
    """
    Compute weights based on pairwise Hellinger distances between component
    posterior distributions for each gene.

    Parameters
    ----------
    params : Dict
        Dictionary of posterior model parameters including variational parameters:
        - For Gamma distribution: 'r_concentration' and 'r_rate'
        - For LogNormal distribution: 'r_loc' and 'r_scale'
    n_components : int
        Number of components in the mixture model
    n_genes : int
        Number of genes in the dataset
    dtype : jnp.dtype, default=jnp.float32
        Data type for computations
    min_distance : float, default=1e-5
        Minimum distance value to use (avoids numerical issues)
    power : float, default=2.0
        Power to raise distances to (higher values increase contrast)
    normalize : bool, default=True
        Whether to normalize weights to sum to 1
    r_dist : Optional[str], default=None
        Type of distribution used for r parameter ('gamma', 'lognormal').
        If None, will try to infer from parameter names.

    Returns
    -------
    jnp.ndarray
        Array of shape (n_genes,) containing weights for each gene
    """
    # Assert that n_components is greater than 1
    if n_components <= 1:
        raise ValueError(
            "This function only works for mixture models with more than "
            "one component."
        )

    # Determine distribution type if not provided
    if r_dist is None:
        if "r_concentration" in params and "r_rate" in params:
            r_dist = "gamma"
        elif "r_loc" in params and "r_scale" in params:
            r_dist = "lognormal"
        else:
            raise ValueError(
                "Could not determine distribution type from parameters. "
                "Please specify r_dist='gamma' or 'lognormal'."
            )

    # Extract parameters based on distribution type
    if r_dist == "gamma":
        # Check if required parameters are present
        if "r_concentration" not in params or "r_rate" not in params:
            raise ValueError(
                "Gamma distribution requires 'r_concentration' and "
                "'r_rate' parameters."
            )
        # Extract parameters
        r_param1 = jnp.array(params["r_concentration"], dtype=dtype)
        r_param2 = jnp.array(params["r_rate"], dtype=dtype)
    elif r_dist == "lognormal":
        # Check if required parameters are present
        if "r_loc" not in params or "r_scale" not in params:
            raise ValueError(
                "LogNormal distribution requires 'r_loc' and "
                "'r_scale' parameters."
            )
        # Extract parameters
        r_param1 = jnp.array(params["r_loc"], dtype=dtype)
        r_param2 = jnp.array(params["r_scale"], dtype=dtype)
    else:
        raise ValueError(
            f"Unsupported distribution type: {r_dist}. "
            "Must be 'gamma' or 'lognormal'."
        )

    # Get dimensions - parameters should be of shape (n_components, n_genes)
    n_components_r, n_genes_r = r_param1.shape

    # Assert that n_components_r and n_genes_r are equal
    if n_components_r != n_components or n_genes_r != n_genes:
        raise ValueError(
            "Parameter dimensions do not match. "
            "Please check your model configuration."
        )

    # Initialize weights
    weights = jnp.zeros(n_genes, dtype=dtype)

    # Create component pairs for vectorized computation
    component_pairs = [
        (i, j) for i in range(n_components) for j in range(i + 1, n_components)
    ]

    # Compute distances for all component pairs
    for i, j in component_pairs:
        # Compute r parameter distances
        if r_dist == "gamma":
            gene_distances = hellinger_gamma(
                r_param1[i], r_param2[i], r_param1[j], r_param2[j]
            )
        else:  # lognormal
            gene_distances = hellinger_lognormal(
                r_param1[i], r_param2[i], r_param1[j], r_param2[j]
            )

        # Ensure minimum distance
        gene_distances = jnp.maximum(gene_distances, min_distance)

        # Add to weights
        weights = weights + gene_distances

    # Apply power to emphasize differences
    if power != 1.0:
        weights = jnp.power(weights, power)

    # Normalize if requested
    if normalize:
        weights = weights / jnp.sum(weights)

    return weights


def differential_expression_weights(
    params: Dict,
    dtype: jnp.dtype = jnp.float32,
    power: float = 2.0,
    normalize: bool = True,
) -> jnp.ndarray:
    """
    Compute weights based on how differentially expressed each gene is between
    components, using the ratio of dispersion parameters.

    Parameters
    ----------
    params : Dict
        Dictionary of posterior model parameters including:
        - 'r': shape (n_components, n_genes) - dispersion parameters
    dtype : jnp.dtype, default=jnp.float32
        Data type for computations
    power : float, default=2.0
        Power to raise fold changes to (higher values increase contrast)
    normalize : bool, default=True
        Whether to normalize weights to sum to 1

    Returns
    -------
    jnp.ndarray
        Array of shape (n_genes,) containing weights for each gene
    """
    # Extract parameters and ensure correct type
    r = jnp.array(params.get("r"), dtype=dtype)

    # Get dimensions
    n_components, n_genes = r.shape

    # Special case for single component - equal weights
    if n_components <= 1:
        return jnp.ones(n_genes, dtype=dtype) / n_genes

    # Initialize weights
    weights = jnp.zeros(n_genes, dtype=dtype)

    # Compute pairwise fold changes for each gene
    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Get dispersion parameters for both components
            r_i, r_j = r[i], r[j]

            # Compute fold change (take log to make symmetric)
            # Add small epsilon to avoid log(0)
            epsilon = jnp.finfo(dtype).eps
            fold_change = jnp.abs(jnp.log((r_i + epsilon) / (r_j + epsilon)))

            # Add to weights (sum over all component pairs)
            weights += fold_change

    # Apply power to emphasize differences
    if power != 1.0:
        weights = jnp.power(weights, power)

    # Normalize if requested
    if normalize:
        weights = weights / jnp.sum(weights)

    return weights


def top_genes_mask(
    weights: jnp.ndarray, n_top: int, dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Create a binary mask that selects the top N genes based on weights.

    Parameters
    ----------
    weights : jnp.ndarray
        Array of shape (n_genes,) containing weights for each gene
    n_top : int
        Number of top genes to select
    dtype : jnp.dtype, default=jnp.float32
        Data type for mask

    Returns
    -------
    jnp.ndarray
        Binary mask of shape (n_genes,) with 1.0 for top genes and 0.0 for
        others
    """
    # Get the indices of the top N genes
    n_genes = weights.shape[0]
    n_top = min(n_top, n_genes)  # Ensure n_top is not larger than n_genes

    # Get indices of top n_top weights
    top_indices = jnp.argsort(weights)[-n_top:]

    # Create binary mask
    mask = jnp.zeros_like(weights, dtype=dtype)
    mask = mask.at[top_indices].set(1.0)

    return mask


# ------------------------------------------------------------------------------
# Main cell type assignment functions
# ------------------------------------------------------------------------------


def compute_cell_type_probabilities(
    results,
    counts: jnp.ndarray,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    ignore_nans: bool = False,
    dtype: jnp.dtype = jnp.float32,
    fit_distribution: bool = True,
    temperature: Optional[float] = None,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Compute probabilistic cell type assignments and fit Dirichlet
    distributions to characterize assignment uncertainty.

    This function works with both ScribeSVIResults and ScribeMCMCResults objects.

    For each cell, this method:
        1. Computes component-specific log-likelihoods using posterior
           samples
        2. Converts these to probability distributions over cell types
        3. Fits a Dirichlet distribution to characterize the uncertainty in
           these assignments

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Results object containing fitted model parameters and samples
    counts : jnp.ndarray
        Count data to evaluate assignments for
    batch_size : Optional[int], default=None
        Size of mini-batches for likelihood computation
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows.
    ignore_nans : bool, default=False
        If True, removes any samples that contain NaNs.
    dtype : jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    fit_distribution : bool, default=True
        If True, fits a Dirichlet distribution to the assignment
        probabilities
    temperature : Optional[float], default=None
        If provided, apply temperature scaling to log probabilities
    weights : Optional[jnp.ndarray], default=None
        Array used to weight genes when computing log likelihoods
    weight_type : Optional[str], default=None
        How to apply weights. Must be one of:
            - 'multiplicative': multiply log probabilities by weights
            - 'additive': add weights to log probabilities
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing:
            - 'concentration': Dirichlet concentration parameters for each
              cell. Shape: (n_cells, n_components). Only returned if
              fit_distribution is True.
            - 'mean_probabilities': Mean assignment probabilities for each
              cell. Shape: (n_cells, n_components). Only returned if
              fit_distribution is True.
            - 'sample_probabilities': Assignment probabilities for each
              posterior sample. Shape: (n_samples, n_cells, n_components)

    Raises
    ------
    ValueError
        - If the model is not a mixture model
        - If posterior samples have not been generated yet

    Note
    ----
    Most of the log-likelihood value differences between cell types are
    extremely large. Thus, the computation usually returns either 0 or 1.
    This computation is therefore not very useful, but it is included for
    completeness.
    """
    # Check if this is a mixture model
    if (
        not hasattr(results, "n_components")
        or results.n_components is None
        or results.n_components <= 1
    ):
        raise ValueError(
            "Cell type assignment only applies to mixture models with "
            "multiple components"
        )

    if verbose:
        print("- Computing component-specific log-likelihoods...")

    # Compute component-specific log-likelihoods
    # Shape: (n_samples, n_cells, n_components)
    log_liks = results.log_likelihood(
        counts,
        batch_size=batch_size,
        return_by="cell",
        cells_axis=cells_axis,
        ignore_nans=ignore_nans,
        split_components=True,
        weights=weights,
        weight_type=weight_type,
        dtype=dtype,
    )

    if verbose:
        print("- Converting log-likelihoods to probabilities...")

    # Apply temperature scaling if requested
    if temperature is not None:
        log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

    # Convert log-likelihoods to probabilities using optimized softmax
    probabilities = softmax(log_liks, axis=-1)

    # Get shapes
    n_samples, n_cells, n_components = probabilities.shape

    if fit_distribution:
        if verbose:
            print("- Fitting Dirichlet distribution...")

        # Initialize array for Dirichlet concentration parameters
        concentrations = jnp.zeros((n_cells, n_components), dtype=dtype)

        # Fit Dirichlet distribution for each cell
        for cell in range(n_cells):
            if verbose and cell % 1000 == 0:
                print(
                    f"    - Fitting Dirichlet distributions for "
                    f"cells {cell}-{min(cell+1000, n_cells)} out of "
                    f"{n_cells} cells"
                )

            # Get probability vectors for this cell across all samples
            cell_probs = probabilities[:, cell, :]
            # Fit Dirichlet using Minka's fixed-point method
            concentrations = concentrations.at[cell].set(
                fit_dirichlet_minka(cell_probs)
            )

        # Compute mean probabilities (Dirichlet mean)
        concentration_sums = jnp.sum(concentrations, axis=1, keepdims=True)
        mean_probabilities = concentrations / concentration_sums

        return {
            "concentration": concentrations,
            "mean_probabilities": mean_probabilities,
            "sample_probabilities": probabilities,
        }
    else:
        return {"sample_probabilities": probabilities}

# ------------------------------------------------------------------------------

def compute_cell_type_probabilities_map(
    results,
    counts: jnp.ndarray,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    dtype: jnp.dtype = jnp.float32,
    temperature: Optional[float] = None,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    use_mean: bool = False,
    verbose: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Compute probabilistic cell type assignments using MAP estimates of
    parameters.

    This function works with both ScribeSVIResults and ScribeMCMCResults
    objects.

    For each cell, this method:
        1. Computes component-specific log-likelihoods using MAP parameter
        estimates
        2. Converts these to probability distributions over cell types

    Parameters
    ----------
    results : ScribeSVIResults or ScribeMCMCResults
        Results object containing fitted model parameters
    counts : jnp.ndarray
        Count data to evaluate assignments for
    batch_size : Optional[int], default=None
        Size of mini-batches for likelihood computation
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows.
    dtype : jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    temperature : Optional[float], default=None
        If provided, apply temperature scaling to log probabilities
    weights : Optional[jnp.ndarray], default=None
        Array used to weight genes when computing log likelihoods
    weight_type : Optional[str], default=None
        How to apply weights. Must be one of:
            - 'multiplicative': multiply log probabilities by weights
            - 'additive': add weights to log probabilities
    use_mean : bool, default=False
        If True, replaces undefined MAP values (NaN) with posterior means
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing:
            - 'probabilities': Assignment probabilities for each cell.
            Shape: (n_cells, n_components)

    Raises
    ------
    ValueError
        If the model is not a mixture model
    """
    # Check if this is a mixture model
    if (
        not hasattr(results, "n_components")
        or results.n_components is None
        or results.n_components <= 1
    ):
        raise ValueError(
            "Cell type assignment only applies to mixture models with "
            "multiple components"
        )

    if verbose:
        print("- Computing component-specific log-likelihoods...")

    # Compute component-specific log-likelihoods using MAP estimates
    # Shape: (n_cells, n_components)
    log_liks = results.log_likelihood_map(
        counts,
        batch_size=batch_size,
        cells_axis=cells_axis,
        return_by="cell",
        split_components=True,
        weights=weights,
        weight_type=weight_type,
        use_mean=use_mean,
        verbose=verbose,
        dtype=dtype,
    )

    # Apply temperature scaling if requested
    if temperature is not None:
        log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

    if verbose:
        print("- Converting log-likelihoods to probabilities...")

    # Convert log-likelihoods to probabilities using optimized softmax
    probabilities = softmax(log_liks, axis=-1)

    return {"probabilities": probabilities}
