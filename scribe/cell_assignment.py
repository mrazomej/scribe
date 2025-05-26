"""
Cell type assignment utilities for SCRIBE.

This module provides functions for assigning cells to components in mixture
models, with various strategies for enhancing the discriminative power of the
model.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from functools import partial
from typing import Dict, Optional, List, Tuple, Union, Callable, Any

# Import Hellinger distance functions
from .stats import hellinger_gamma, hellinger_lognormal, hellinger_beta

# ------------------------------------------------------------------------------
# Utility functions for computing discriminative weights
# ------------------------------------------------------------------------------

def hellinger_distance_weights(
    params: Dict,
    n_components: int,
    n_genes: int,
    dtype: jnp.dtype = jnp.float32,
    min_distance: float = 1e-5,
    power: float = 2.0,
    normalize: bool = True,
    r_dist: Optional[str] = None
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
        - For p parameter: 'p_concentration1' and 'p_concentration0' (Beta)
        - For gate parameter: 'gate_logits' or 'gate_concentration1'/'gate_concentration0'
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
        if 'r_concentration' in params and 'r_rate' in params:
            r_dist = 'gamma'
        elif 'r_loc' in params and 'r_scale' in params:
            r_dist = 'lognormal'
        else:
            raise ValueError(
                "Could not determine distribution type from parameters. "
                "Please specify r_dist='gamma' or 'lognormal'."
            )
    
    # Extract parameters based on distribution type
    if r_dist == 'gamma':
        # Check if required parameters are present
        if 'r_concentration' not in params or 'r_rate' not in params:
            raise ValueError(
                "Gamma distribution requires 'r_concentration' and "
                "'r_rate' parameters."
            )
        # Extract parameters
        r_param1 = jnp.array(params['r_concentration'], dtype=dtype)
        r_param2 = jnp.array(params['r_rate'], dtype=dtype)
    elif r_dist == 'lognormal':
        # Check if required parameters are present
        if 'r_loc' not in params or 'r_scale' not in params:
            raise ValueError(
                "LogNormal distribution requires 'r_loc' and "
                "'r_scale' parameters."
            )
        # Extract parameters
        r_param1 = jnp.array(params['r_loc'], dtype=dtype)
        r_param2 = jnp.array(params['r_scale'], dtype=dtype)
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
        (i, j) for i in range(n_components) for j in range(i+1, n_components)
    ]
    
    # Compute distances for all component pairs
    for i, j in component_pairs:
        # Compute r parameter distances
        if r_dist == 'gamma':
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

# ------------------------------------------------------------------------------

def differential_expression_weights(
    params: Dict,
    dtype: jnp.dtype = jnp.float32,
    power: float = 2.0,
    normalize: bool = True
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
    r = jnp.array(params.get('r'), dtype=dtype)
    
    # Get dimensions
    n_components, n_genes = r.shape
    
    # Special case for single component - equal weights
    if n_components <= 1:
        return jnp.ones(n_genes, dtype=dtype) / n_genes
    
    # Initialize weights
    weights = jnp.zeros(n_genes, dtype=dtype)
    
    # Compute pairwise fold changes for each gene
    for i in range(n_components):
        for j in range(i+1, n_components):
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

# ------------------------------------------------------------------------------

def assignment_entropy_weights(
    entropies: jnp.ndarray,
    dtype: jnp.dtype = jnp.float32,
    power: float = 2.0,
    normalize: bool = True
) -> jnp.ndarray:
    """
    Compute weights based on the entropy of component assignments across cells,
    with the assumption that genes producing more decisive assignments are more
    discriminative.
    
    Parameters
    ----------
    probabilities : jnp.ndarray
        Array of shape (n_samples, n_cells, n_components) containing component
        assignment probabilities for each cell across posterior samples
    dtype : jnp.dtype, default=jnp.float32
        Data type for computations
    power : float, default=2.0
        Power to raise the inversed entropy to (higher values increase contrast)
    normalize : bool, default=True
        Whether to normalize weights to sum to 1
        
    Returns
    -------
    jnp.ndarray
        Array of shape (n_genes,) containing weights for each gene
    """
    # Average entropy across samples
    # Shape: (n_cells,)
    mean_entropy = jnp.mean(entropies, axis=0)
    
    # Convert entropy to weights (lower entropy = higher weight)
    weights = 1.0 - (mean_entropy / jnp.log(entropies.shape[1]))
    
    # Ensure non-negative weights
    weights = jnp.maximum(weights, 0.0)
    
    # Apply power to emphasize differences
    if power != 1.0:
        weights = jnp.power(weights, power)
    
    # Normalize if requested
    if normalize:
        weights = weights / jnp.sum(weights)
    
    return weights

# ------------------------------------------------------------------------------

def top_genes_mask(
    weights: jnp.ndarray,
    n_top: int,
    dtype: jnp.dtype = jnp.float32
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
# Temperature scaling functions
# ------------------------------------------------------------------------------

def temperature_scaling(
    log_probs: jnp.ndarray,
    temperature: float,
    dtype: jnp.dtype = jnp.float32
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

# ------------------------------------------------------------------------------
# Cell type assignment functions
# ------------------------------------------------------------------------------

def compute_cell_type_assignments_weighted(
    results,
    counts: jnp.ndarray,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    ignore_nans: bool = False,
    dtype: jnp.dtype = jnp.float32,
    fit_distribution: bool = True,
    weights_method: str = "hellinger",
    n_top_genes: Optional[int] = None,
    temperature: Optional[float] = None,
    weight_type: str = "multiplicative",
    power: float = 2.0,
    verbose: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Enhanced cell type assignment that uses gene weights to improve
    discrimination.

    This method improves on the base cell type assignment by:
        1. Applying weights to emphasize differentially expressed genes
        2. Optionally using only the top N most discriminative genes
        3. Applying temperature scaling to sharpen probability distributions

    Parameters
    ----------
    results : ScribeSVIResults
        The results object containing the fitted model
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
        If True, fits a Dirichlet distribution to the assignment probabilities
    weights_method : str, default="hellinger"
        Method to compute gene weights, one of: - "hellinger": Use Hellinger
        distance between component distributions - "differential": Use fold
        change between component parameters - "uniform": Use uniform weights -
        "precomputed": Use provided weights parameter
    n_top_genes : Optional[int], default=None
        If provided, only use the top N genes for assignment
    temperature : Optional[float], default=None
        If provided, apply temperature scaling to log probabilities
    weight_type : str, default="multiplicative"
        How to apply weights, either "multiplicative" or "additive"
    power : float, default=2.0
        Power to apply to weights to increase contrast
    verbose : bool, default=True
        If True, prints progress messages

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing:
            - 'concentration': Dirichlet concentration parameters for each cell
            - 'mean_probabilities': Mean assignment probabilities for each cell
            - 'sample_probabilities': Assignment probabilities for each
              posterior sample
            - 'weights': The weights used for each gene
    """
    # Validate model type
    if results.n_components is None or results.n_components <= 1:
        raise ValueError(
            "Cell type assignment only applies to mixture models with "
            "multiple components"
        )
        
    # Compute or use provided weights
    if weights_method == "hellinger":
        if verbose:
            print("- Computing weights using Hellinger distance...")
        weights = compute_posterior_hellinger_weights(
            results.params, dtype=dtype, power=power
        )
    elif weights_method == "differential":
        if verbose:
            print("- Computing weights using differential expression...")
        weights = compute_differential_expression_weights(
            results.params, dtype=dtype, power=power
        )
    elif weights_method == "uniform":
        if verbose:
            print("- Using uniform weights...")
        weights = jnp.ones(results.n_genes, dtype=dtype) / results.n_genes
    else:
        raise ValueError(
            f"Unknown weights_method: {weights_method}. Must be one of "
            "'hellinger', 'differential', or 'uniform'."
        )
    
    # Apply top N genes masking if requested
    if n_top_genes is not None:
        if verbose:
            print(f"- Using only top {n_top_genes} genes...")
        # Create mask for top genes
        mask = compute_top_n_mask(weights, n_top_genes, dtype=dtype)
        # Apply mask (keeps weights proportional among selected genes)
        weights = weights * mask
        # Re-normalize
        weights = weights / jnp.sum(weights)

    if verbose:
        print("- Computing component-specific log-likelihoods...")

    # Compute component-specific log-likelihoods
    # Shape: (n_samples, n_cells, n_components)
    log_liks = results.compute_log_likelihood(
        counts,
        batch_size=batch_size,
        return_by='cell',
        cells_axis=cells_axis,
        ignore_nans=ignore_nans,
        split_components=True,
        weights=weights,
        weight_type=weight_type,
        dtype=dtype
    )

    if verbose:
        print("- Converting log-likelihoods to probabilities...")

    # Apply temperature scaling if requested
    if temperature is not None:
        if verbose:
            print(f"- Applying temperature scaling with T={temperature}...")
        log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

    # Convert log-likelihoods to probabilities using log-sum-exp for stability
    log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)
    # Then subtract and exponentiate to get probabilities
    probabilities = jnp.exp(log_liks - log_sum_exp)

    # Get shapes
    n_samples, n_cells, n_components = probabilities.shape

    if fit_distribution:
        if verbose:
            print("- Fitting Dirichlet distribution...")

        # Initialize array for Dirichlet concentration parameters
        concentrations = jnp.zeros((n_cells, n_components), dtype=dtype)

        # Fit Dirichlet distribution for each cell
        from scribe.stats import fit_dirichlet_minka
        for cell in range(n_cells):
            if verbose and cell % 1000 == 0 and cell > 0:
                print(f"    - Fitting Dirichlet distributions for "
                      f"cells {cell}-{min(cell+1000, n_cells)} out of "
                      f"{n_cells} cells")
                
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
            'concentration': concentrations,
            'mean_probabilities': mean_probabilities,
            'sample_probabilities': probabilities,
            'weights': weights
        }
    else:
        return {
            'sample_probabilities': probabilities,
            'weights': weights
        }

# ------------------------------------------------------------------------------

def compute_cell_type_assignments_map_weighted(
    results,
    counts: jnp.ndarray,
    batch_size: Optional[int] = None,
    cells_axis: int = 0,
    dtype: jnp.dtype = jnp.float32,
    weights_method: str = "hellinger",
    n_top_genes: Optional[int] = None,
    temperature: Optional[float] = None,
    weight_type: str = "multiplicative",
    power: float = 2.0,
    use_mean: bool = False,
    verbose: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Compute weighted cell type assignments using MAP estimates of parameters.
    
    Parameters
    ----------
    results : ScribeSVIResults
        The results object containing the fitted model
    counts : jnp.ndarray
        Count data to evaluate assignments for
    batch_size : Optional[int], default=None
        Size of mini-batches for likelihood computation
    cells_axis : int, default=0
        Axis along which cells are arranged. 0 means cells are rows.
    dtype : jnp.dtype, default=jnp.float32
        Data type for numerical precision in computations
    weights_method : str, default="hellinger"
        Method to compute gene weights
    n_top_genes : Optional[int], default=None
        If provided, only use the top N genes for assignment
    temperature : Optional[float], default=None
        If provided, apply temperature scaling to log probabilities
    weight_type : str, default="multiplicative"
        How to apply weights, either "multiplicative" or "additive"
    power : float, default=2.0
        Power to apply to weights to increase contrast
    use_mean : bool, default=False
        If True, replaces undefined MAP values (NaN) with posterior means
    verbose : bool, default=True
        If True, prints progress messages
    
    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing:
            - 'probabilities': Assignment probabilities for each cell
            - 'weights': The weights used for each gene
    """
    # Validate model type
    if results.n_components is None or results.n_components <= 1:
        raise ValueError(
            "Cell type assignment only applies to mixture models with "
            "multiple components"
        )

    # Compute weights using the same methods as the full posterior version
    if weights_method == "hellinger":
        if verbose:
            print("- Computing weights using Hellinger distance...")
        weights = compute_posterior_hellinger_weights(
            results.params, dtype=dtype, power=power
        )
    elif weights_method == "differential":
        if verbose:
            print("- Computing weights using differential expression...")
        weights = compute_differential_expression_weights(
            results.params, dtype=dtype, power=power
        )
    elif weights_method == "uniform":
        if verbose:
            print("- Using uniform weights...")
        weights = jnp.ones(results.n_genes, dtype=dtype) / results.n_genes
    else:
        raise ValueError(
            f"Unknown weights_method: {weights_method}. Must be one of "
            "'hellinger', 'differential', or 'uniform'."
        )
    
    # Apply top N genes masking if requested
    if n_top_genes is not None:
        if verbose:
            print(f"- Using only top {n_top_genes} genes...")
        # Create mask for top genes
        mask = compute_top_n_mask(weights, n_top_genes, dtype=dtype)
        # Apply mask (keeps weights proportional among selected genes)
        weights = weights * mask
        # Re-normalize
        weights = weights / jnp.sum(weights)

    if verbose:
        print("- Computing component-specific log-likelihoods...")

    # Get the log likelihood function
    likelihood_fn = results.get_log_likelihood_fn()

    # Get the MAP estimates
    map_estimates = results.get_map()
    
    # Replace NaN values with means if requested
    if use_mean:
        # Get distributions to compute means
        distributions = results.get_distributions(backend="numpyro")
        
        # Check each parameter for NaNs and replace with means
        any_replaced = False
        for param, value in map_estimates.items():
            # Check if any values are NaN
            if jnp.any(jnp.isnan(value)):
                # Update flag
                any_replaced = True
                # Get mean value
                mean_value = distributions[param].mean
                # Replace NaN values with means
                map_estimates[param] = jnp.where(
                    jnp.isnan(value),
                    mean_value,
                    value
                )
        
        if any_replaced and verbose:
            print("    - Replaced undefined MAP values with posterior means")
    
    # Compute component-specific log-likelihoods using MAP estimates
    # Shape: (n_cells, n_components)
    log_liks = likelihood_fn(
        counts,
        map_estimates,
        batch_size=batch_size,
        cells_axis=cells_axis,
        return_by='cell',
        split_components=True,
        weights=weights,
        weight_type=weight_type,
        dtype=dtype
    )

    # Apply temperature scaling if requested
    if temperature is not None:
        if verbose:
            print(f"- Applying temperature scaling with T={temperature}...")
        log_liks = temperature_scaling(log_liks, temperature, dtype=dtype)

    if verbose:
        print("- Converting log-likelihoods to probabilities...")

    # Convert log-likelihoods to probabilities using log-sum-exp for stability
    log_sum_exp = jsp.special.logsumexp(log_liks, axis=-1, keepdims=True)
    # Then subtract and exponentiate to get probabilities
    probabilities = jnp.exp(log_liks - log_sum_exp)

    return {
        'probabilities': probabilities,
        'weights': weights
    }


# ------------------------------------------------------------------------------
# Helper functions for visualization and diagnostics
# ------------------------------------------------------------------------------

def get_top_discriminative_genes(
    results,
    n_genes: int = 50,
    weights_method: str = "hellinger",
    power: float = 2.0,
    return_weights: bool = False
) -> Union[List[str], Tuple[List[str], jnp.ndarray]]:
    """
    Get the top discriminative genes based on model parameters.
    
    Parameters
    ----------
    results : ScribeSVIResults
        The results object containing the fitted model
    n_genes : int, default=50
        Number of top genes to return
    weights_method : str, default="hellinger"
        Method to compute gene weights
    power : float, default=2.0
        Power to apply to weights to increase contrast
    return_weights : bool, default=False
        If True, also return the weights for each gene
        
    Returns
    -------
    Union[List[str], Tuple[List[str], jnp.ndarray]]
        List of gene names (if var is available) or indices,
        optionally with corresponding weights
    """
    # Validate model type
    if results.n_components is None or results.n_components <= 1:
        raise ValueError(
            "Discriminative genes are only meaningful for mixture models "
            "with multiple components"
        )
    
    # Compute weights
    if weights_method == "hellinger":
        weights = compute_posterior_hellinger_weights(
            results.params, power=power
        )
    elif weights_method == "differential":
        weights = compute_differential_expression_weights(
            results.params, power=power
        )
    else:
        raise ValueError(
            f"Unknown weights_method: {weights_method}. Must be one of "
            "'hellinger' or 'differential'."
        )
    
    # Limit to number of available genes
    n_genes = min(n_genes, results.n_genes)
    
    # Get indices of top genes
    top_indices = jnp.argsort(weights)[-n_genes:][::-1]  # Reverse to get descending order
    
    # Convert indices to gene names if var is available
    if results.var is not None:
        try:
            # Try to get gene names from index
            gene_names = results.var.index[top_indices].tolist()
        except:
            # Fall back to gene indices if names not available
            gene_names = top_indices.tolist()
    else:
        # Use gene indices if var not available
        gene_names = top_indices.tolist()
    
    # Return with weights if requested
    if return_weights:
        return gene_names, weights[top_indices]
    else:
        return gene_names


def visualize_gene_weights(
    results,
    weights_method: str = "hellinger",
    n_top: int = 50,
    power: float = 2.0,
    figsize: Tuple[int, int] = (10, 6),
    return_fig: bool = False
):
    """
    Visualize the weights assigned to genes for component discrimination.
    
    Parameters
    ----------
    results : ScribeSVIResults
        The results object containing the fitted model
    weights_method : str, default="hellinger"
        Method to compute gene weights
    n_top : int, default=50
        Number of top genes to highlight in the visualization
    power : float, default=2.0
        Power to apply to weights to increase contrast
    figsize : Tuple[int, int], default=(10, 6)
        Figure size (width, height) in inches
    return_fig : bool, default=False
        If True, return the figure object
        
    Returns
    -------
    Optional[Figure]
        Matplotlib figure object if return_fig=True
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get top genes and weights
    top_genes, top_weights = get_top_discriminative_genes(
        results, n_genes=n_top, weights_method=weights_method, 
        power=power, return_weights=True
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all weights in ascending order
    if weights_method == "hellinger":
        weights = compute_posterior_hellinger_weights(
            results.params, power=power
        )
    else:
        weights = compute_differential_expression_weights(
            results.params, power=power
        )
    
    # Sort weights
    sorted_weights = np.sort(weights)
    
    # Plot all weights
    ax.plot(sorted_weights, label='All genes', color='gray', alpha=0.5)
    
    # Plot top weights
    top_indices = np.argsort(weights)[-n_top:]
    ax.scatter(
        np.arange(len(weights) - n_top, len(weights)),
        sorted_weights[-n_top:],
        color='red',
        label=f'Top {n_top} genes',
        zorder=3
    )
    
    # Set labels and title
    ax.set_xlabel('Gene rank')
    ax.set_ylabel('Weight')
    ax.set_title(f'Gene weights using {weights_method} method (power={power})')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Return figure if requested
    if return_fig:
        return fig
    

def visualize_component_distances(
    results,
    counts=None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_gene_names: bool = True,
    n_top_genes: int = 20,
    weights_method: str = "hellinger",
    power: float = 2.0,
    return_fig: bool = False
):
    """
    Visualize the distances between components for top discriminative genes.
    
    Parameters
    ----------
    results : ScribeSVIResults
        The results object containing the fitted model
    counts : Optional[jnp.ndarray], default=None
        Count data to evaluate component distances. If provided, uses empirical
        distributions instead of model parameters.
    figsize : Tuple[int, int], default=(10, 8)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap to use for heatmap
    show_gene_names : bool, default=True
        If True, show gene names on y-axis
    n_top_genes : int, default=20
        Number of top genes to show
    weights_method : str, default="hellinger"
        Method to compute gene weights
    power : float, default=2.0
        Power to apply to weights to increase contrast
    return_fig : bool, default=False
        If True, return the figure object
        
    Returns
    -------
    Optional[Figure]
        Matplotlib figure object if return_fig=True
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get top discriminative genes
    top_genes = get_top_discriminative_genes(
        results, n_genes=n_top_genes, weights_method=weights_method, 
        power=power, return_weights=False
    )
    
    # Extract parameters
    r = np.array(results.params['r'])
    p = np.array(results.params['p'])
    n_components = results.n_components
    
    # Create figure
    fig, axes = plt.subplots(
        1, n_components*(n_components-1)//2, 
        figsize=figsize,
        sharey=True
    )
    
    # If only one pair, wrap in list
    if n_components == 2:
        axes = [axes]
    
    # Compute pairwise distances
    from scribe.stats import hellinger_gamma
    
    # Counter for subplot index
    subplot_idx = 0
    
    # For each pair of components
    for i in range(n_components):
        for j in range(i+1, n_components):
            # Get current axis
            ax = axes[subplot_idx]
            
            # Extract parameters for both components
            r_i, r_j = r[i], r[j]
            p_i = p if np.isscalar(p) else p[i]
            p_j = p if np.isscalar(p) else p[j]
            
            # Compute Hellinger distances for all genes
            distances = hellinger_gamma(r_i, 1.0/p_i, r_j, 1.0/p_j)
            
            # Get indices of top genes
            if isinstance(top_genes[0], str) and results.var is not None:
                top_indices = [np.where(results.var.index == gene)[0][0] 
                              for gene in top_genes]
            else:
                top_indices = top_genes
            
            # Get distances for top genes
            top_distances = distances[top_indices]
            
            # Plot distances as horizontal bars
            y_pos = np.arange(len(top_genes))
            ax.barh(y_pos, top_distances, color='skyblue')
            
            # Set labels for this subplot
            ax.set_title(f'Component {i+1} vs {j+1}')
            ax.set_xlabel('Hellinger Distance')
            
            # Only show y-labels for first subplot
            if subplot_idx == 0 and show_gene_names:
                if isinstance(top_genes[0], str):
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(top_genes)
                else:
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels([f'Gene {idx}' for idx in top_genes])
            else:
                ax.set_yticks([])
            
            # Move to next subplot
            subplot_idx += 1
    
    # Adjust layout
    plt.tight_layout()
    
    # Return figure if requested
    if return_fig:
        return fig