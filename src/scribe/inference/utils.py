"""
Shared utilities for inference module.

This module provides common utilities for data processing, validation, and
type conversion used across all inference methods.
"""

from typing import Union, Optional, Tuple, TYPE_CHECKING
import jax.numpy as jnp

if TYPE_CHECKING:
    from anndata import AnnData
    from ..models.config import ModelConfig, InferenceConfig

from ..core import InputProcessor
from ..models.config import DataConfig


def process_counts_data(
    counts: Union[jnp.ndarray, "AnnData"],
    data_config: DataConfig,
) -> Tuple[jnp.ndarray, Optional["AnnData"], int, int]:
    """Process count data from various input formats.

    This is a convenience wrapper around InputProcessor.process_counts_data
    that uses a DataConfig object.

    Parameters
    ----------
    counts : Union[jnp.ndarray, AnnData]
        Count matrix or AnnData object containing single-cell RNA-seq counts.
    data_config : DataConfig
        Data processing configuration specifying cells_axis and layer.

    Returns
    -------
    Tuple[jnp.ndarray, Optional[AnnData], int, int]
        A tuple containing:
            - count_data: Processed count matrix (cells as rows)
            - adata: Original AnnData object if provided, None otherwise
            - n_cells: Number of cells
            - n_genes: Number of genes

    Examples
    --------
    >>> from scribe.models.config import DataConfig
    >>> from scribe.inference.utils import process_counts_data
    >>>
    >>> data_config = DataConfig(cells_axis=0, layer="counts")
    >>> count_data, adata, n_cells, n_genes = process_counts_data(
    ...     counts=adata,
    ...     data_config=data_config
    ... )
    """
    return InputProcessor.process_counts_data(
        counts, data_config.cells_axis, data_config.layer
    )


# ------------------------------------------------------------------------------


def validate_model_inference_match(
    model_config: "ModelConfig",  # type: ignore
    inference_method: str,
) -> None:
    """Validate that model config inference method matches specified method.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration object.
    inference_method : str
        Inference method string ("svi", "mcmc", or "vae").

    Raises
    ------
    ValueError
        If the model config's inference method doesn't match the specified
        inference method.

    Examples
    --------
    >>> from scribe.models.config import ModelConfigBuilder
    >>> from scribe.inference.utils import validate_model_inference_match
    >>>
    >>> model_config = (ModelConfigBuilder()
    ...     .for_model("nbdm")
    ...     .with_inference("svi")
    ...     .build())
    >>>
    >>> validate_model_inference_match(model_config, "svi")  # OK
    >>> validate_model_inference_match(model_config, "mcmc")  # Raises ValueError
    """
    from ..models.config.enums import InferenceMethod

    model_method = model_config.inference_method.value
    if model_method != inference_method:
        raise ValueError(
            f"Inference method mismatch: model_config specifies "
            f"'{model_method}' but inference_method='{inference_method}'"
        )


# ------------------------------------------------------------------------------


def validate_inference_config_match(
    model_config: "ModelConfig",  # type: ignore
    inference_config: "InferenceConfig",  # type: ignore
) -> None:
    """Validate that model config and inference config methods match.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration object.
    inference_config : InferenceConfig
        Inference configuration object.

    Raises
    ------
    ValueError
        If the model config's inference method doesn't match the inference
        config's method.

    Examples
    --------
    >>> from scribe.models.config import ModelConfigBuilder, InferenceConfig, SVIConfig
    >>> from scribe.inference.utils import validate_inference_config_match
    >>>
    >>> model_config = (ModelConfigBuilder()
    ...     .for_model("nbdm")
    ...     .with_inference("svi")
    ...     .build())
    >>>
    >>> inference_config = InferenceConfig.from_svi(SVIConfig())
    >>> validate_inference_config_match(model_config, inference_config)  # OK
    """
    if model_config.inference_method != inference_config.method:
        raise ValueError(
            f"Inference method mismatch: model_config specifies "
            f"'{model_config.inference_method.value}' but inference_config "
            f"specifies '{inference_config.method.value}'"
        )
