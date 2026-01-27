"""
Results classes for SCRIBE inference.
"""

from typing import Dict, Optional, Union, Callable, Tuple, Any
from dataclasses import dataclass

import jax.numpy as jnp
import pandas as pd

from ..models.config import ModelConfig

try:
    from anndata import AnnData
except ImportError:
    AnnData = None

# Import mixins
from ._core import CoreResultsMixin
from ._parameter_extraction import ParameterExtractionMixin
from ._gene_subsetting import GeneSubsettingMixin
from ._component import ComponentMixin
from ._model_helpers import ModelHelpersMixin
from ._sampling import SamplingMixin
from ._likelihood import LikelihoodMixin
from ._mixture_analysis import MixtureAnalysisMixin
from ._normalization import NormalizationMixin

# ------------------------------------------------------------------------------
# Base class for inference results
# ------------------------------------------------------------------------------


@dataclass
class ScribeSVIResults(
    CoreResultsMixin,
    ParameterExtractionMixin,
    GeneSubsettingMixin,
    ComponentMixin,
    ModelHelpersMixin,
    SamplingMixin,
    LikelihoodMixin,
    MixtureAnalysisMixin,
    NormalizationMixin,
):
    """
    Base class for SCRIBE variational inference results.

    This class stores the results from SCRIBE's variational inference procedure,
    including model parameters, loss history, dataset dimensions, and model
    configuration. It can optionally store metadata from an AnnData object and
    posterior/predictive samples.

    Attributes
    ----------
    params : Dict
        Dictionary of inferred model parameters from SCRIBE
    loss_history : jnp.ndarray
        Array containing the ELBO loss values during training
    n_cells : int
        Number of cells in the dataset
    n_genes : int
        Number of genes in the dataset
    model_type : str
        Type of model used for inference
    model_config : ModelConfig
        Configuration object specifying model architecture and priors
    prior_params : Dict[str, Any]
        Dictionary of prior parameter values used during inference
    obs : Optional[pd.DataFrame]
        Cell-level metadata from adata.obs, if provided
    var : Optional[pd.DataFrame]
        Gene-level metadata from adata.var, if provided
    uns : Optional[Dict]
        Unstructured metadata from adata.uns, if provided
    n_obs : Optional[int]
        Number of observations (cells), if provided
    n_vars : Optional[int]
        Number of variables (genes), if provided
    posterior_samples : Optional[Dict]
        Samples of parameters from the posterior distribution, if generated
    predictive_samples : Optional[Dict]
        Predictive samples generated from the model, if generated
    n_components : Optional[int]
        Number of mixture components, if using a mixture model
    """

    # Core inference results
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData object
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # Optional results
    posterior_samples: Optional[Dict] = None
    predictive_samples: Optional[Dict] = None
    n_components: Optional[int] = None

    # Internal: tracks original gene count before subsetting (for amortizer validation)
    # When using amortized capture probability, counts must have shape
    # (n_cells, _original_n_genes) because the amortizer computes sufficient
    # statistics (e.g., total UMI count) by summing across ALL genes.
    _original_n_genes: Optional[int] = None
