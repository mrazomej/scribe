"""
Cell type assignment for mixture models in SCRIBE.

This module provides basic functionality for assigning cells to mixture
components based on posterior probabilities computed from fitted model
parameters. It works with both MCMC and SVI results objects.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from typing import Dict, Optional, Union, Tuple, Any
import warnings


def assign_cell_types(
    results,
    counts: jnp.ndarray,
    method: str = "posterior_samples",
    cells_axis: int = 0,
    batch_size: Optional[int] = None,
    return_probabilities: bool = False,
    return_sample_probabilities: bool = False,
    fit_distribution: bool = True,
    temperature: Optional[float] = None,
    weights: Optional[jnp.ndarray] = None,
    weight_type: Optional[str] = None,
    dtype: jnp.dtype = jnp.float32,
    verbose: bool = True,
) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Basic cell type assignment for mixture models using the existing API.

    This function is a wrapper around the existing cell type assignment methods
    in the results objects, providing a unified interface that works with both
    MCMC and SVI results.

    Parameters
    ----------
    results : ScribeMCMCResults or ScribeSVIResults
        Fitted model results object containing parameters and model information
    counts : jnp.ndarray
        Count matrix of shape (n_cells, n_genes) or (n_genes, n_cells)
    method : str, default="posterior_samples"
        Method for assignment: - "posterior_samples": Use posterior samples
        (requires get_posterior_samples) - "map": Use MAP (maximum a posteriori)
        estimates
    cells_axis : int, default=0
        Axis along which cells are arranged (0=rows, 1=columns)
    batch_size : Optional[int], default=None
        Process cells in batches to manage memory usage
    return_probabilities : bool, default=False
        If True, return assignment probabilities (MAP method only)
    return_sample_probabilities : bool, default=False
        If True, return probabilities for each posterior sample
    fit_distribution : bool, default=True
        If True, fits Dirichlet distribution to assignment probabilities
    temperature : Optional[float], default=None
        Temperature scaling parameter for probabilities
    weights : Optional[jnp.ndarray], default=None
        Gene weights for likelihood computation
    weight_type : Optional[str], default=None
        How to apply weights ('multiplicative' or 'additive')
    dtype : jnp.dtype, default=jnp.float32
        Data type for computations
    verbose : bool, default=True
        Whether to print progress messages

    Returns
    -------
    Union[jnp.ndarray, Dict[str, jnp.ndarray]]
        If method="map" and return_probabilities=False:
            jnp.ndarray: Cell assignments of shape (n_cells,)
        Otherwise:
            Dict containing assignment results with keys like:
                - 'assignments': Most likely component for each cell
                - 'probabilities': Assignment probabilities
                - 'concentration': Dirichlet concentration parameters (if
                  fit_distribution=True)
                - 'sample_probabilities': Probabilities for each sample (if
                  available)

    Raises
    ------
    ValueError
        If the model is not a mixture model or method is invalid
    """
    # Validate that this is a mixture model
    if not hasattr(results, "n_components") or results.n_components is None:
        raise ValueError("Cell type assignment requires a mixture model")

    if results.n_components <= 1:
        raise ValueError("Cell type assignment requires multiple components")

    if method == "posterior_samples":
        # Use the existing cell_type_assignments method
        if not hasattr(results, "cell_type_assignments"):
            raise ValueError(
                "Results object does not support posterior sample assignments"
            )

        assignment_results = results.cell_type_assignments(
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            dtype=dtype,
            fit_distribution=fit_distribution,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )

        # Add hard assignments
        if "mean_probabilities" in assignment_results:
            assignments = jnp.argmax(
                assignment_results["mean_probabilities"], axis=1
            )
        elif "sample_probabilities" in assignment_results:
            # Use mean across samples for assignment
            mean_probs = jnp.mean(
                assignment_results["sample_probabilities"], axis=0
            )
            assignments = jnp.argmax(mean_probs, axis=1)
        else:
            raise ValueError(
                "No probability information found in assignment results"
            )

        assignment_results["assignments"] = assignments

        # Remove sample probabilities if not requested
        if (
            not return_sample_probabilities
            and "sample_probabilities" in assignment_results
        ):
            del assignment_results["sample_probabilities"]

        return assignment_results

    elif method == "map":
        # Use the existing cell_type_assignments_map method
        if not hasattr(results, "cell_type_assignments_map"):
            raise ValueError("Results object does not support MAP assignments")

        assignment_results = results.cell_type_assignments_map(
            counts=counts,
            batch_size=batch_size,
            cells_axis=cells_axis,
            dtype=dtype,
            temperature=temperature,
            weights=weights,
            weight_type=weight_type,
            verbose=verbose,
        )

        # Extract assignments
        probabilities = assignment_results["probabilities"]
        assignments = jnp.argmax(probabilities, axis=1)

        if return_probabilities:
            return {"assignments": assignments, "probabilities": probabilities}
        else:
            return assignments

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be 'posterior_samples' or 'map'."
        )


def get_assignment_summary(
    assignment_results: Dict[str, Any], confidence_threshold: float = 0.8
) -> Dict[str, Any]:
    """
    Generate a summary of cell type assignments.

    Parameters
    ----------
    assignment_results : Dict[str, Any]
        Results from assign_cell_types() function
    confidence_threshold : float, default=0.8
        Threshold for high-confidence assignments

    Returns
    -------
    Dict[str, Any]
        Summary statistics including counts per component, confidence metrics
    """
    # Extract assignments and probabilities
    assignments = assignment_results["assignments"]
    probabilities = assignment_results.get(
        "probabilities"
    ) or assignment_results.get("mean_probabilities")

    n_cells = len(assignments)
    n_components = int(jnp.max(assignments) + 1)

    # Count assignments per component
    component_counts = {}
    for comp in range(n_components):
        component_counts[f"component_{comp}"] = int(
            jnp.sum(assignments == comp)
        )

    summary = {
        "n_cells": n_cells,
        "n_components": n_components,
        "component_counts": component_counts,
        "component_proportions": {
            k: v / n_cells for k, v in component_counts.items()
        },
    }

    if probabilities is not None:
        # Compute confidence metrics
        max_probs = jnp.max(probabilities, axis=1)

        summary.update(
            {
                "mean_max_probability": float(jnp.mean(max_probs)),
                "median_max_probability": float(jnp.median(max_probs)),
                "high_confidence_assignments": int(
                    jnp.sum(max_probs > confidence_threshold)
                ),
                "high_confidence_fraction": float(
                    jnp.mean(max_probs > confidence_threshold)
                ),
            }
        )

        # Per-component confidence
        component_confidence = {}
        for comp in range(n_components):
            comp_mask = assignments == comp
            if jnp.sum(comp_mask) > 0:
                comp_probs = max_probs[comp_mask]
                component_confidence[f"component_{comp}"] = {
                    "mean_confidence": float(jnp.mean(comp_probs)),
                    "high_confidence_count": int(
                        jnp.sum(comp_probs > confidence_threshold)
                    ),
                }

        summary["component_confidence"] = component_confidence

    return summary


def plot_assignment_results(
    assignment_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 4)
):
    """
    Create basic plots of assignment results.

    Parameters
    ----------
    assignment_results : Dict[str, Any]
        Results from assign_cell_types() function
    figsize : Tuple[int, int], default=(12, 4)
        Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib is required for plotting")

    # Extract assignments and probabilities
    assignments = assignment_results["assignments"]
    probabilities = assignment_results.get(
        "probabilities"
    ) or assignment_results.get("mean_probabilities")

    n_components = int(jnp.max(assignments) + 1)

    if probabilities is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes = [axes[0], axes[1], None]

    # Plot 1: Assignment counts
    component_counts = [
        int(jnp.sum(assignments == i)) for i in range(n_components)
    ]
    axes[0].bar(range(n_components), component_counts)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Number of cells")
    axes[0].set_title("Cells per component")
    axes[0].set_xticks(range(n_components))

    # Plot 2: Assignment proportions
    proportions = np.array(component_counts) / len(assignments)
    axes[1].pie(
        proportions,
        labels=[f"Comp {i}" for i in range(n_components)],
        autopct="%1.1f%%",
    )
    axes[1].set_title("Component proportions")

    # Plot 3: Confidence distribution (if probabilities available)
    if probabilities is not None and axes[2] is not None:
        max_probs = jnp.max(probabilities, axis=1)
        axes[2].hist(np.array(max_probs), bins=50, alpha=0.7)
        axes[2].axvline(
            0.8, color="red", linestyle="--", label="High confidence"
        )
        axes[2].set_xlabel("Maximum posterior probability")
        axes[2].set_ylabel("Number of cells")
        axes[2].set_title("Assignment confidence")
        axes[2].legend()

    plt.tight_layout()
    return fig


def compute_assignment_entropy(
    assignment_results: Dict[str, Any],
) -> jnp.ndarray:
    """
    Compute entropy of assignment probabilities for each cell.

    Parameters
    ----------
    assignment_results : Dict[str, Any]
        Results from assign_cell_types() function containing probabilities

    Returns
    -------
    jnp.ndarray
        Entropy values for each cell (lower = more confident)
    """
    # Get probabilities
    if "sample_probabilities" in assignment_results:
        # Use mean across samples
        probabilities = jnp.mean(
            assignment_results["sample_probabilities"], axis=0
        )
    elif "probabilities" in assignment_results:
        probabilities = assignment_results["probabilities"]
    elif "mean_probabilities" in assignment_results:
        probabilities = assignment_results["mean_probabilities"]
    else:
        raise ValueError(
            "No probability information found in assignment results"
        )

    # Compute entropy: -∑(p_i * log(p_i))
    eps = jnp.finfo(jnp.float32).eps
    entropy = -jnp.sum(probabilities * jnp.log(probabilities + eps), axis=1)

    return entropy
