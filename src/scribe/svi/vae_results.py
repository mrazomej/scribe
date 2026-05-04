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

from dataclasses import dataclass, field, replace as dc_replace
from typing import TYPE_CHECKING, Any, Dict, Optional
import pickle

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..models.config import ModelConfig

if TYPE_CHECKING:
    from ..core.axis_layout import AxisLayout
from ..core.serialization import make_model_config_pickle_safe
from ._core import CoreResultsMixin
from ._gene_subsetting import GeneSubsettingMixin, _has_flow_params
from ._latent_space import (
    LatentSpaceMixin,
    _DECODER_KEY,
    _ENCODER_KEY,
    _FLOW_KEY,
)
from ._likelihood import LikelihoodMixin
from ._lnm_extraction import LNMExtractionMixin
from ._pln_extraction import PLNExtractionMixin
from ._mixture_analysis import MixtureAnalysisMixin
from ._model_helpers import ModelHelpersMixin
from ._normalization import NormalizationMixin
from ._parameter_extraction import ParameterExtractionMixin
from ._sampling import SamplingMixin
from .variational_results_base import ScribeVariationalResults


def _is_pickle_serializable(value: Any) -> bool:
    """Return ``True`` when ``value`` can be serialized by ``pickle``.

    Parameters
    ----------
    value : Any
        Arbitrary Python object to probe for stdlib ``pickle`` support.

    Returns
    -------
    bool
        ``True`` when ``pickle.dumps(value)`` succeeds, ``False`` otherwise.
    """
    try:
        pickle.dumps(value)
        return True
    except Exception:
        return False


def _sanitize_decoder_for_pickle(decoder: Any) -> Any:
    """Return a pickle-safe decoder copy when head initializers are closures.

    Parameters
    ----------
    decoder : Any
        Decoder module carried by ``ScribeVAEResults._decoder``.

    Returns
    -------
    Any
        Original decoder when no sanitization is needed, otherwise a shallow
        copy with non-picklable ``bias_init`` callables replaced by ``None``.

    Notes
    -----
    Flax initializers such as ``nn.initializers.constant`` are runtime-local
    closures (e.g. ``constant.<locals>.init``). Those closures are only needed
    when layers are initialized, not for inference on already-trained params.
    During serialization we therefore drop only unpicklable head initializers,
    preserving every other decoder attribute.
    """
    # Preserve behavior for non-decoder or legacy objects that do not expose
    # ``output_heads``.
    output_heads = getattr(decoder, "output_heads", None)
    if output_heads is None:
        return decoder

    sanitized_heads = []
    changed = False

    # Rewrite only the problematic per-head initializer closures so that
    # stdlib pickle can serialize the VAE results object.
    for head in output_heads:
        bias_init = getattr(head, "bias_init", None)
        if bias_init is not None and not _is_pickle_serializable(bias_init):
            sanitized_heads.append(dc_replace(head, bias_init=None))
            changed = True
        else:
            sanitized_heads.append(head)

    if not changed:
        return decoder

    sanitized_heads_tuple = tuple(sanitized_heads)
    if hasattr(decoder, "replace"):
        return decoder.replace(output_heads=sanitized_heads_tuple)
    return dc_replace(decoder, output_heads=sanitized_heads_tuple)


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
    LNMExtractionMixin,
    PLNExtractionMixin,
    ScribeVariationalResults,
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
    param_layouts : Optional[Dict[str, AxisLayout]]
        Semantic axis metadata for each parameter key.  Built at inference
        time or lazily reconstructed from tensor shapes via the ``layouts``
        property.
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
    _gene_coverage: Optional[float] = None
    _gene_coverage_mask: Optional[np.ndarray] = None
    _excluded_gene_names: Optional[list[str]] = None
    _total_count_max: Optional[int] = None
    _gene_axis_by_key: Optional[Dict[str, int]] = None
    _subset_gene_index: Optional[np.ndarray] = None

    # Semantic axis metadata for each parameter key.  Built from
    # ``param_specs`` at inference time or reconstructed lazily from
    # tensor shapes.  ``None`` on old pickles or manual construction.
    param_layouts: Optional[Dict[str, "AxisLayout"]] = None

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
        """Return pickle-safe state for composable VAE results.

        Returns
        -------
        Dict[str, Any]
            Instance state with a serialization-safe ``model_config`` and a
            decoder copy whose non-picklable ``bias_init`` closures have been
            removed from output heads.
        """
        state = dict(self.__dict__)
        state["model_config"] = make_model_config_pickle_safe(
            state.get("model_config")
        )
        state["_decoder"] = _sanitize_decoder_for_pickle(state.get("_decoder"))
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Semantic axis layouts (AxisLayout integration)
    # ------------------------------------------------------------------

    @property
    def layouts(self) -> Dict[str, "AxisLayout"]:
        """Semantic axis layouts for every parameter key.

        Returns pre-built ``param_layouts`` when available. Otherwise
        reconstructs lazily from tensor shapes using
        ``derive_axis_membership`` and ``reconstruct_param_layouts``,
        mirroring the ``ScribeSVIResults.layouts`` property.

        Returns
        -------
        dict of str to AxisLayout
        """
        if self.param_layouts is not None:
            return self.param_layouts

        from ..core.axis_layout import (
            reconstruct_param_layouts,
            derive_axis_membership,
        )

        mc = self.model_config
        _mp, _dp = derive_axis_membership(mc)

        _layouts = reconstruct_param_layouts(
            self.params,
            n_genes=self.n_genes,
            n_cells=self.n_cells,
            n_components=getattr(mc, "n_components", None),
            n_datasets=getattr(mc, "n_datasets", None),
            mixture_params=_mp,
            dataset_params=_dp,
            gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
            has_sample_dim=False,
        )
        object.__setattr__(self, "param_layouts", _layouts)
        return _layouts

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
        from dataclasses import fields as dc_fields
        from ..models.components.vae_components import (
            DecoderOutputHead,
            MultiHeadDecoder,
        )

        # Boolean masks: count True entries via .sum().
        # Integer arrays: count elements via len() (summing the values would
        # give the wrong answer, e.g. [0,1,2,3,4].sum() == 10, not 5).
        if hasattr(index, "dtype") and np.dtype(index.dtype) == np.bool_:
            new_n_genes = int(index.sum())
        else:
            new_n_genes = len(index)

        # Track original gene count for amortizer validation
        original_n_genes = (
            getattr(self, "_original_n_genes", None) or self.n_genes
        )

        # Compose gene indices when re-subsetting so the final index is
        # always relative to the original (full) gene list.
        prev_gene_idx = getattr(self, "_subset_gene_index", None)
        if prev_gene_idx is not None:
            gene_index_abs = prev_gene_idx[index]
        else:
            gene_index_abs = np.asarray(index)

        # Subset decoder output heads and params.
        # Keep this logic explicit instead of relying on generic dict slicing:
        # each decoder head is a separate Flax Dense module keyed as
        # "head_<param_name>" with shape semantics
        #   kernel: (hidden_dim, n_genes)
        #   bias:   (n_genes,)
        # and must remain synchronized with DecoderOutputHead.output_dim.
        decoder_params = new_params.get(_DECODER_KEY, {})
        if not isinstance(decoder_params, dict):
            raise ValueError(
                "Expected 'vae_decoder$params' to be a dict when "
                "subsetting VAE results."
            )
        new_heads = []
        new_dec_params = dict(decoder_params)
        head_field_names = {f.name for f in dc_fields(DecoderOutputHead)}

        for head in self._decoder.output_heads:
            head_key = f"head_{head.param_name}"
            if head_key not in decoder_params:
                raise ValueError(
                    "Decoder params missing required head key "
                    f"'{head_key}' during VAE subset creation."
                )
            head_params = dict(decoder_params[head_key])

            # Validate expected tensor ranks before slicing to surface corrupt
            # states early with actionable errors.
            kernel = head_params.get("kernel")
            if kernel is not None and (
                not hasattr(kernel, "ndim") or int(kernel.ndim) != 2
            ):
                raise ValueError(
                    f"Expected '{head_key}.kernel' to be rank-2; "
                    f"got shape {getattr(kernel, 'shape', None)}."
                )
            bias = head_params.get("bias")
            if bias is not None and (
                not hasattr(bias, "ndim") or int(bias.ndim) != 1
            ):
                raise ValueError(
                    f"Expected '{head_key}.bias' to be rank-1; "
                    f"got shape {getattr(bias, 'shape', None)}."
                )

            # Build kwargs dynamically so this logic remains compatible with
            # DecoderOutputHead extensions (e.g., optional bias_init vectors).
            new_head_kwargs = dict(
                param_name=head.param_name,
                output_dim=new_n_genes,
                transform=head.transform,
            )
            if "bias_init" in head_field_names and hasattr(head, "bias_init"):
                bias_init = getattr(head, "bias_init")
                if hasattr(bias_init, "shape") and len(bias_init.shape) == 1:
                    new_head_kwargs["bias_init"] = bias_init[index]
                else:
                    new_head_kwargs["bias_init"] = bias_init
            new_heads.append(DecoderOutputHead(**new_head_kwargs))

            # Slice the decoder head parameters along the gene output axis.
            if "kernel" in head_params:
                head_params["kernel"] = head_params["kernel"][:, index]
            if "bias" in head_params:
                head_params["bias"] = head_params["bias"][index]

            # Validate the post-slice widths so model/guide reconstruction can
            # safely reuse this subsetted params dict.
            if "kernel" in head_params and int(
                head_params["kernel"].shape[1]
            ) != int(new_n_genes):
                raise ValueError(
                    f"Subsetted '{head_key}.kernel' width "
                    f"{int(head_params['kernel'].shape[1])} does not match "
                    f"expected n_genes={int(new_n_genes)}."
                )
            if "bias" in head_params and int(
                head_params["bias"].shape[0]
            ) != int(new_n_genes):
                raise ValueError(
                    f"Subsetted '{head_key}.bias' width "
                    f"{int(head_params['bias'].shape[0])} does not match "
                    f"expected n_genes={int(new_n_genes)}."
                )
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

        subset = ScribeVAEResults(
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
            _gene_coverage=getattr(self, "_gene_coverage", None),
            _gene_coverage_mask=getattr(self, "_gene_coverage_mask", None),
            _excluded_gene_names=getattr(self, "_excluded_gene_names", None),
            _total_count_max=getattr(self, "_total_count_max", None),
            _gene_axis_by_key=getattr(self, "_gene_axis_by_key", None),
            _subset_gene_index=gene_index_abs,
            # Keep only layouts that are still present in the subsetted params
            # dict to avoid dangling metadata entries from removed keys.
            param_layouts={
                key: value
                for key, value in self.layouts.items()
                if key in new_params
            },
            _encoder=self._encoder,
            _decoder=new_decoder,
            _latent_spec=self._latent_spec,
        )

        # Preserve full-dimension params for flow-based guides.
        # Re-use the existing cached copy when re-subsetting.
        original_params = getattr(self, "_original_params", None)
        if original_params is None and _has_flow_params(self.params):
            original_params = self.params
        if original_params is not None:
            subset._original_params = original_params

        return subset

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
