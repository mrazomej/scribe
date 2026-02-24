"""
VAE results class for the composable builder architecture.

Extends ``ScribeSVIResults`` with latent-space operations (encode, decode,
sample, embed) using Linen modules and trained parameters from NumPyro's param
store.  No NNX, no registry lookup, no parameterization-specific imports.

Classes
-------
ScribeVAEResults
    Dataclass storing VAE inference results with latent-space methods.

See Also
--------
scribe.svi.results : ScribeSVIResults base class.
scribe.svi._latent_space : LatentSpaceMixin providing latent operations.
scribe.svi._latent_dispatch : Dispatched encoder-dependent operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

from ..models.config import ModelConfig
from ..core.serialization import make_model_config_pickle_safe
from ._core import CoreResultsMixin
from ._gene_subsetting import GeneSubsettingMixin
from ._latent_space import (
    LatentSpaceMixin,
    _DECODER_KEY,
    _ENCODER_KEY,
    _FLOW_KEY,
)
from ._likelihood import LikelihoodMixin
from ._mixture_analysis import MixtureAnalysisMixin
from ._model_helpers import ModelHelpersMixin
from ._normalization import NormalizationMixin
from ._parameter_extraction import ParameterExtractionMixin
from ._sampling import SamplingMixin

# ==============================================================================
# ScribeVAEResults
# ==============================================================================


@dataclass
class ScribeVAEResults(
    LatentSpaceMixin,
    CoreResultsMixin,
    ParameterExtractionMixin,
    GeneSubsettingMixin,
    ModelHelpersMixin,
    SamplingMixin,
    LikelihoodMixin,
    MixtureAnalysisMixin,
    NormalizationMixin,
):
    """Results from VAE-based variational inference.

    Extends the standard ``ScribeSVIResults`` fields with VAE-specific
    attributes (encoder, decoder, latent spec) and latent-space methods provided
    by ``LatentSpaceMixin``. VAE and mixture models are mutually exclusive (the
    continuous latent space replaces discrete mixture components), so
    ``ComponentMixin`` is not included.

    The class stores the *original un-initialized Linen module instances*
    alongside the trained parameters.  Inference-time reconstruction uses
    ``module.apply({"params": subtree}, x)`` — the standard Flax Linen
    pattern — avoiding any NNX ``split``/``merge`` machinery.

    Attributes
    ----------
    params : Dict
        Full NumPyro parameter dict.  Contains keys like
        ``vae_encoder$params``, ``vae_decoder$params``, and optionally
        ``vae_prior_flow$params``.
    loss_history : jnp.ndarray
        ELBO loss values during training.
    n_cells : int
        Number of cells.
    n_genes : int
        Number of genes.
    model_type : str
        Model type identifier.
    model_config : ModelConfig
        Model architecture and prior configuration.
    prior_params : Dict[str, Any]
        Prior hyperparameters used during inference.
    _encoder : Any
        Un-initialized encoder Linen module (e.g. ``GaussianEncoder``).
    _decoder : Any
        Un-initialized decoder Linen module (e.g. ``MultiHeadDecoder``).
    _latent_spec : Any
        Latent specification (e.g. ``GaussianLatentSpec``), with optional
        ``.flow`` attribute for flow-based priors.
    latent_samples : Optional[jnp.ndarray]
        Cached latent samples (set by ``get_latent_samples`` or
        ``get_latent_samples_conditioned_on_data``).
    cell_embeddings : Optional[jnp.ndarray]
        Cached cell embeddings (set by ``get_latent_embeddings``).

    Examples
    --------
    >>> # Create from SVI training artifacts
    >>> results = ScribeVAEResults(
    ...     params=svi.get_params(svi_state),
    ...     loss_history=losses,
    ...     n_cells=1000, n_genes=2000,
    ...     model_type="vae_nb",
    ...     model_config=config,
    ...     prior_params={},
    ...     _encoder=encoder,
    ...     _decoder=decoder,
    ...     _latent_spec=latent_spec,
    ... )
    >>> embeddings = results.get_latent_embeddings(counts)
    >>> z_samples = results.get_latent_samples_conditioned_on_data(counts)
    """

    # --- Inherited from ScribeSVIResults ---
    params: Dict
    loss_history: jnp.ndarray
    n_cells: int
    n_genes: int
    model_type: str
    model_config: ModelConfig
    prior_params: Dict[str, Any]

    # Standard metadata from AnnData
    obs: Optional[pd.DataFrame] = None
    var: Optional[pd.DataFrame] = None
    uns: Optional[Dict] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None

    # Optional results
    posterior_samples: Optional[Dict] = None
    predictive_samples: Optional[Dict] = None
    n_components: Optional[int] = None

    # Internal (from GeneSubsettingMixin)
    _original_n_genes: Optional[int] = None
    _gene_axis_by_key: Optional[Dict[str, int]] = None

    # --- VAE-specific fields ---
    _encoder: Any = None
    _decoder: Any = None
    _latent_spec: Any = None

    # Cached latent results
    latent_samples: Optional[jnp.ndarray] = None
    cell_embeddings: Optional[jnp.ndarray] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self):
        """Validate VAE fields and delegate to parent.

        Ensures that the encoder, decoder, and latent spec are set, and
        that the params dict contains the expected keys.
        """
        # Delegate to CoreResultsMixin for n_components setup
        super().__post_init__()

        # Validate that VAE-specific fields are set
        if self._encoder is None:
            raise ValueError(
                "ScribeVAEResults requires _encoder "
                "(un-initialized Linen module)."
            )
        if self._decoder is None:
            raise ValueError(
                "ScribeVAEResults requires _decoder "
                "(un-initialized Linen module)."
            )
        if self._latent_spec is None:
            raise ValueError(
                "ScribeVAEResults requires _latent_spec "
                "(e.g. GaussianLatentSpec)."
            )

        # Validate that params contain the expected keys
        if _ENCODER_KEY not in self.params:
            raise ValueError(
                f"params dict missing '{_ENCODER_KEY}'. "
                "Did the SVI training include a VAE encoder?"
            )
        if _DECODER_KEY not in self.params:
            raise ValueError(
                f"params dict missing '{_DECODER_KEY}'. "
                "Did the SVI training include a VAE decoder?"
            )

        # If flow is set, its params must be present
        if self._latent_spec.flow is not None and _FLOW_KEY not in self.params:
            raise ValueError(
                f"Latent spec has a flow prior but params dict "
                f"missing '{_FLOW_KEY}'."
            )

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle-safe state for composable VAE results."""
        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Gene subsetting override
    # ------------------------------------------------------------------

    def _create_subset(
        self,
        index,
        new_params: Dict,
        new_var: Optional[pd.DataFrame],
        new_posterior_samples: Optional[Dict],
        new_predictive_samples: Optional[jnp.ndarray],
    ) -> "ScribeVAEResults":
        """Create a gene-subset copy with updated decoder heads and params.

        Overrides the parent ``_create_subset`` to additionally:

        1. Subset each decoder output head's ``output_dim`` to the
           number of selected genes.
        2. Subset the ``head_<name>`` kernel and bias in
           ``vae_decoder$params`` along the output dimension.
        3. Build a new ``MultiHeadDecoder`` with updated output heads.

        Parameters
        ----------
        index : jnp.ndarray
            Boolean mask for selected genes.
        new_params : Dict
            Already-subset params dict (from parent).
        new_var : Optional[pd.DataFrame]
            Gene metadata for the subset.
        new_posterior_samples : Optional[Dict]
            Subset posterior samples, or None.
        new_predictive_samples : Optional[jnp.ndarray]
            Subset predictive samples, or None.

        Returns
        -------
        ScribeVAEResults
            A new results object with subset decoder.
        """
        from dataclasses import replace as dc_replace
        from ..models.components.vae_components import (
            DecoderOutputHead,
            MultiHeadDecoder,
        )

        new_n_genes = int(index.sum() if hasattr(index, "sum") else len(index))

        # Track original gene count for amortizer validation
        original_n_genes = (
            getattr(self, "_original_n_genes", None) or self.n_genes
        )

        # Subset decoder output heads and params
        decoder_params = new_params.get(_DECODER_KEY, {})
        new_heads = []
        new_dec_params = dict(decoder_params)

        for head in self._decoder.output_heads:
            # Update the output_dim to the new gene count
            new_head = DecoderOutputHead(
                param_name=head.param_name,
                output_dim=new_n_genes,
                transform=head.transform,
            )
            new_heads.append(new_head)

            # Subset head kernel and bias in decoder params
            head_key = f"head_{head.param_name}"
            if head_key in decoder_params:
                head_params = dict(decoder_params[head_key])
                if "kernel" in head_params:
                    # kernel shape: (hidden_dim, output_dim) -> subset axis 1
                    head_params["kernel"] = head_params["kernel"][:, index]
                if "bias" in head_params:
                    # bias shape: (output_dim,) -> subset axis 0
                    head_params["bias"] = head_params["bias"][index]
                new_dec_params[head_key] = head_params

        # Update decoder params in the full params dict
        new_params = dict(new_params)
        new_params[_DECODER_KEY] = new_dec_params

        # Build a new decoder with updated heads
        new_decoder = MultiHeadDecoder(
            output_dim=0,
            latent_dim=self._decoder.latent_dim,
            hidden_dims=self._decoder.hidden_dims,
            output_heads=tuple(new_heads),
        )

        return ScribeVAEResults(
            params=new_params,
            loss_history=self.loss_history,
            n_cells=self.n_cells,
            n_genes=new_n_genes,
            model_type=self.model_type,
            model_config=self.model_config,
            prior_params=self.prior_params,
            obs=self.obs,
            var=new_var,
            uns=self.uns,
            n_obs=self.n_obs,
            n_vars=new_var.shape[0] if new_var is not None else None,
            posterior_samples=new_posterior_samples,
            predictive_samples=new_predictive_samples,
            n_components=self.n_components,
            _original_n_genes=original_n_genes,
            _gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
            _encoder=self._encoder,
            _decoder=new_decoder,
            _latent_spec=self._latent_spec,
        )

    # ------------------------------------------------------------------
    # Classmethod factory
    # ------------------------------------------------------------------

    @classmethod
    def from_training(
        cls,
        params: Dict,
        loss_history: jnp.ndarray,
        n_cells: int,
        n_genes: int,
        model_config: ModelConfig,
        vae_guide_family,
        model_type: str = "vae",
        prior_params: Optional[Dict] = None,
        **kwargs,
    ) -> "ScribeVAEResults":
        """Create VAE results from SVI training artifacts.

        Convenience factory that extracts the encoder, decoder, and
        latent spec from a ``VAELatentGuide`` instance.

        Parameters
        ----------
        params : Dict
            Trained params from ``svi.get_params(svi_state)``.
        loss_history : jnp.ndarray
            ELBO loss values from training.
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        model_config : ModelConfig
            Model configuration.
        vae_guide_family : VAELatentGuide
            The guide family used during training, containing encoder,
            decoder, and latent_spec.
        model_type : str
            Model type identifier (default ``"vae"``).
        prior_params : Dict, optional
            Prior hyperparameters.  Default: empty dict.
        **kwargs
            Additional keyword arguments passed to the constructor
            (e.g. ``obs``, ``var``, ``uns``).

        Returns
        -------
        ScribeVAEResults
            A new results object.
        """
        if prior_params is None:
            prior_params = {}

        return cls(
            params=params,
            loss_history=loss_history,
            n_cells=n_cells,
            n_genes=n_genes,
            model_type=model_type,
            model_config=model_config,
            prior_params=prior_params,
            _encoder=vae_guide_family.encoder,
            _decoder=vae_guide_family.decoder,
            _latent_spec=vae_guide_family.latent_spec,
            **kwargs,
        )
