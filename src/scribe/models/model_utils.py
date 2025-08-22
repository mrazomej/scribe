"""
Utility functions for model guide parameter setup.

This module provides shared helper functions for setting up variational guide
parameters across all SCRIBE model types (standard, mixture, unconstrained).
"""

# Import JAX-related libraries
import jax.numpy as jnp

# Import Pyro-related libraries
import numpyro
import numpyro.distributions as dist

# Import typing
from typing import Dict, Tuple, Optional, List

# Import model config
from .model_config import ModelConfig

# ------------------------------------------------------------------------------


def setup_scalar_parameter(
    param_name: str,
    distribution: dist.Distribution,
    shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Extract parameters from a distribution and create numpyro.param entries.

    Parameters
    ----------
    param_name : str
        Base name for the parameter (e.g., "p", "r", "gate")
    distribution : dist.Distribution
        The distribution to extract parameters from
    shape : Optional[Tuple[int, ...]]
        Shape to broadcast initial values to. If None, uses scalar values.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary of parameter values ready for distribution instantiation

    Examples
    --------
    >>> # Scalar parameter
    >>> params = setup_scalar_parameter("p", dist.Beta(1.0, 1.0))
    >>>
    >>> # Gene-specific parameter
    >>> params = setup_scalar_parameter("r", dist.Gamma(2.0, 0.1), (n_genes,))
    >>>
    >>> # Component-gene specific parameter (for mixture models)
    >>> params = setup_scalar_parameter(
    ...     "r", dist.Gamma(2.0, 0.1), (n_components, n_genes)
    ... )
    """
    # Extract the distribution's parameter values and constraints
    values = distribution.get_args()
    # Get the constraint objects for each parameter
    constraints = distribution.arg_constraints
    # Initialize empty dictionary to store parameter values
    params = {}

    # Iterate through each parameter and its constraint
    for arg_name, constraint in constraints.items():
        # Get the initial value for this parameter
        initial_value = values[arg_name]
        # If shape is provided, broadcast the initial value to that shape
        if shape is not None:
            initial_value = jnp.ones(shape) * initial_value

        # Create a numpyro parameter with the given name, initial value, and
        # constraint
        params[arg_name] = numpyro.param(
            f"{param_name}_{arg_name}", initial_value, constraint=constraint
        )

    # Return the dictionary of parameter values
    return params


# ------------------------------------------------------------------------------


def setup_cell_specific_scalar_parameter(
    param_name: str,
    distribution: dist.Distribution,
    n_cells: int,
) -> Dict[str, jnp.ndarray]:
    """
    Set up cell-specific parameters (without sampling).

    This creates numpyro.param entries for parameters that vary per cell.

    Parameters
    ----------
    param_name : str
        Base name for the parameter
    distribution : dist.Distribution
        The distribution to extract parameters from
    n_cells : int
        Total number of cells

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary of parameter values ready for distribution instantiation

    Examples
    --------
    >>> # For VCP models
    >>> params = setup_cell_specific_scalar_parameter(
    ...     "p_capture",
    ...     model_config.p_capture_distribution_guide,
    ...     n_cells
    ... )
    """
    # Extract distribution parameters
    values = distribution.get_args()
    constraints = distribution.arg_constraints
    params = {}

    # Create parameters with cell dimensions
    for arg_name, constraint in constraints.items():
        initial_value = jnp.ones(n_cells) * values[arg_name]
        params[arg_name] = numpyro.param(
            f"{param_name}_{arg_name}",
            initial_value,
            constraint=constraint,
        )

    return params


# ------------------------------------------------------------------------------


def setup_and_sample_parameter(
    param_name: str,
    distribution: dist.Distribution,
    shape: Optional[Tuple[int, ...]] = None,
    cell_specific: bool = False,
    n_cells: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    """
    Set up and sample from a parameter distribution.

    Parameters
    ----------
    param_name : str
        Base name for the parameter
    distribution : dist.Distribution
        The distribution to sample from
    shape : Optional[Tuple[int, ...]]
        Shape to broadcast initial values to. If None, uses scalar values.
    cell_specific : bool, default=False
        Whether this parameter varies per cell (e.g., p_capture in VCP models)
    n_cells : Optional[int]
        Number of cells. Required if cell_specific=True.
    batch_size : Optional[int]
        Mini-batch size for cell-specific parameters. If None, processes all
        cells at once.

    Examples
    --------
    >>> # Standard models
    >>> setup_and_sample_parameter("p", model_config.p_distribution_guide)
    >>> setup_and_sample_parameter(
    ...     "r", model_config.r_distribution_guide, shape=(n_genes,)
    ... )
    >>>
    >>> # Mixture models
    >>> setup_and_sample_parameter(
    ...     "r",
    ...     model_config.r_distribution_guide,
    ...     shape=(n_components, n_genes)
    ... )
    >>>
    >>> # Cell-specific parameters (VCP models)
    >>> setup_and_sample_parameter(
    ...     "p_capture",
    ...     model_config.p_capture_distribution_guide,
    ...     cell_specific=True,
    ...     n_cells=n_cells,
    ...     batch_size=batch_size
    ... )
    """
    if cell_specific:
        if n_cells is None:
            raise ValueError("n_cells must be provided when cell_specific=True")

        # Use the cell-specific helper function to create parameters
        params = setup_cell_specific_scalar_parameter(
            param_name, distribution, n_cells
        )

        # Handle sampling with appropriate plate and batching
        if batch_size is None:
            with numpyro.plate("cells", n_cells):
                numpyro.sample(param_name, distribution.__class__(**params))
        else:
            with numpyro.plate(
                "cells", n_cells, subsample_size=batch_size
            ) as idx:
                # Index the parameters for the batch
                batch_params = {
                    name: param[idx] for name, param in params.items()
                }
                numpyro.sample(
                    param_name, distribution.__class__(**batch_params)
                )
    else:
        # Use the standard scalar parameter helper function
        params = setup_scalar_parameter(param_name, distribution, shape)
        numpyro.sample(param_name, distribution.__class__(**params))


# ------------------------------------------------------------------------------
# Parameter Configuration Registry (Advanced Usage)
# ------------------------------------------------------------------------------

# Standard parameter configurations for different model types
GUIDE_PARAMETER_CONFIGS = {
    # Standard models
    "nbdm_standard": [
        ("p", "p_distribution_guide", None),
        ("r", "r_distribution_guide", "n_genes"),
    ],
    "zinb_standard": [
        ("p", "p_distribution_guide", None),
        ("r", "r_distribution_guide", "n_genes"),
        ("gate", "gate_distribution_guide", "n_genes"),
    ],
    "nbvcp_standard": [
        ("p", "p_distribution_guide", None),
        ("r", "r_distribution_guide", "n_genes"),
        ("p_capture", "p_capture_distribution_guide", "n_cells"),
    ],
    "zinbvcp_standard": [
        ("p", "p_distribution_guide", None),
        ("r", "r_distribution_guide", "n_genes"),
        ("gate", "gate_distribution_guide", "n_genes"),
        ("p_capture", "p_capture_distribution_guide", "n_cells"),
    ],
    # Linked parameterizations
    "nbdm_linked": [
        ("p", "p_distribution_guide", None),
        ("mu", "mu_distribution_guide", "n_genes"),
    ],
    "zinb_linked": [
        ("p", "p_distribution_guide", None),
        ("mu", "mu_distribution_guide", "n_genes"),
        ("gate", "gate_distribution_guide", "n_genes"),
    ],
    # Odds ratio parameterizations
    "nbdm_odds_ratio": [
        ("phi", "phi_distribution_guide", None),
        ("mu", "mu_distribution_guide", "n_genes"),
    ],
    "zinb_odds_ratio": [
        ("phi", "phi_distribution_guide", None),
        ("mu", "mu_distribution_guide", "n_genes"),
        ("gate", "gate_distribution_guide", "n_genes"),
    ],
    # Mixture models (add "_mix" suffix and components dimension)
}

# ------------------------------------------------------------------------------


def setup_guide_parameters_from_config(
    guide_type: str,
    model_config: ModelConfig,
    n_cells: int,
    n_genes: int,
    batch_size: Optional[int] = None,
) -> None:
    """
    Generic parameter setup using configuration registry.

    This is an advanced helper that can automatically set up parameters
    for most standard guide functions based on their type.

    Parameters
    ----------
    guide_type : str
        The type of guide (e.g., "nbdm_standard", "zinb_linked")
    model_config : ModelConfig
        The model configuration
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    batch_size : Optional[int]
        Mini-batch size for cell-specific parameters

    Examples
    --------
    >>> # Instead of manual parameter setup
    >>> setup_guide_parameters_from_config(
    ...     "zinb_standard", model_config, n_cells, n_genes
    ... )
    """
    # Check if the guide type is valid
    if guide_type not in GUIDE_PARAMETER_CONFIGS:
        raise ValueError(f"Unknown guide type: {guide_type}")

    # Get the configuration for the guide type
    config = GUIDE_PARAMETER_CONFIGS[guide_type]

    # Set up each parameter
    for param_name, config_attr, shape_spec in config:
        distribution = getattr(model_config, config_attr)

        # Determine shape
        shape = None
        if shape_spec == "n_genes":
            shape = (n_genes,)
        elif shape_spec == "n_cells":
            # Handle cell-specific parameters specially
            setup_and_sample_parameter(
                param_name,
                distribution,
                cell_specific=True,
                n_cells=n_cells,
                batch_size=batch_size,
            )
            continue
        elif shape_spec and hasattr(shape_spec, "__iter__"):
            shape = shape_spec

        setup_and_sample_parameter(param_name, distribution, shape)
