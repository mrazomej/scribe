"""
Input processing and validation utilities for SCRIBE inference.

This module handles data preprocessing, input validation, and model type
determination that is shared between SVI and MCMC inference methods.
"""

from typing import Union, Tuple, Optional
import jax.numpy as jnp
import scipy.sparse


class InputProcessor:
    """Handles input processing and validation for SCRIBE inference."""

    @staticmethod
    def process_counts_data(
        counts: Union[jnp.ndarray, "AnnData"],
        cells_axis: int = 0,
        layer: Optional[str] = None,
    ) -> Tuple[jnp.ndarray, Optional["AnnData"], int, int]:
        """
        Process count data from various input formats.

        Parameters
        ----------
        counts : Union[jnp.ndarray, AnnData]
            Count matrix or AnnData object containing counts
        cells_axis : int, default=0
            Axis for cells in count matrix (0=rows, 1=columns)
        layer : Optional[str], default=None
            Layer in AnnData to use for counts. If None, uses .X

        Returns
        -------
        Tuple[jnp.ndarray, Optional[AnnData], int, int]
            Processed count data, original AnnData (if provided), n_cells, n_genes
        """
        # Handle AnnData input
        if hasattr(counts, "obs"):
            adata = counts
            count_data = adata.layers[layer] if layer else adata.X
            if scipy.sparse.issparse(count_data):
                count_data = count_data.toarray()
            count_data = jnp.array(count_data)
        else:
            count_data = jnp.array(counts)
            adata = None

        # Extract dimensions and standardize to cells-as-rows
        if cells_axis == 0:
            n_cells, n_genes = count_data.shape
        else:
            n_genes, n_cells = count_data.shape
            count_data = count_data.T  # Transpose to make cells rows

        return count_data, adata, n_cells, n_genes

    @staticmethod
    def validate_model_configuration(
        zero_inflated: bool,
        variable_capture: bool,
        mixture_model: bool,
        n_components: Optional[int],
    ) -> None:
        """
        Validate model configuration flags.

        Parameters
        ----------
        zero_inflated : bool
            Whether to use zero-inflated model
        variable_capture : bool
            Whether to use variable capture probability
        mixture_model : bool
            Whether to use mixture model
        n_components : Optional[int]
            Number of mixture components

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        # Validate mixture model configuration
        if n_components is not None and not mixture_model:
            raise ValueError(
                "n_components must be None when mixture_model=False"
            )

        if mixture_model:
            if n_components is None or n_components < 2:
                raise ValueError(
                    "n_components must be specified and greater than 1 "
                    "when mixture_model=True"
                )

    @staticmethod
    def determine_model_type(
        zero_inflated: bool, variable_capture: bool, mixture_model: bool
    ) -> str:
        """
        Convert boolean flags to model type string.

        Parameters
        ----------
        zero_inflated : bool
            Whether to use zero-inflated model
        variable_capture : bool
            Whether to use variable capture probability
        mixture_model : bool
            Whether to use mixture model

        Returns
        -------
        str
            Model type string (e.g., "nbdm", "zinb_mix", "nbvcp", etc.)
        """
        # Determine base model
        base_model = "nbdm"
        if zero_inflated:
            base_model = "zinb"
        if variable_capture:
            if zero_inflated:
                base_model = "zinbvcp"
            else:
                base_model = "nbvcp"

        # Add mixture suffix if needed
        if mixture_model:
            base_model = f"{base_model}_mix"

        return base_model
